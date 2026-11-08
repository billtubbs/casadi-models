"""Tests of recipes (cas_models.recipes): models record how they were
made, and save_model() / to_yaml() write that (the class or connection
function and its arguments, to any depth) so that from_yaml() and
load_model() rebuild the same model, with symbols shared as in the
original and, optionally, values in place of some parameters."""

import pickle

import casadi as cas
import numpy as np
import pytest
import yaml
from test_yaml_utils import (
    FUNCTION_CASES,
    assert_same_connected,
    assert_same_functions,
    dt_connected_arx,
    dt_custom,
)

from cas_models.continuous_time.models import (
    SSModelCTLinearFOSISO,
    SSModelCTLinearO2SISO,
    StateSpaceModelCT,
    StateSpaceModelCTFromABCD,
)
from cas_models.continuous_time.regulators import SSModelCTPIInt
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTARXSISO,
)
from cas_models.recipes import recipe
from cas_models.transformations import (
    connect_systems_in_series,
    set_param_values,
    vstack_systems,
)
from cas_models.yaml_utils import _decode, _encode, from_yaml, load_model, to_yaml

# Recording


def test_class_recipe():
    """The class and the arguments given (not the defaults)"""
    model = StateSpaceModelDTARXSISO(na=2, nb=1, dt=2.0)
    assert recipe(model) == {
        "class": StateSpaceModelDTARXSISO,
        "args": {"na": 2, "nb": 1, "dt": 2.0},
    }


def test_class_recipe_outermost_init():
    """A subclass calling its base's __init__ records its own arguments"""
    model = SSModelCTLinearFOSISO(K=2.0)  # its __init__ calls SSModelCTFromABCDSISO's
    assert recipe(model) == {"class": SSModelCTLinearFOSISO, "args": {"K": 2.0}}
    assert recipe(SSModelCTPIInt(name="c")) == {
        "class": SSModelCTPIInt,
        "args": {"name": "c"},
    }


def test_base_class_has_no_recipe():
    """A model made from functions is saved from its fields"""
    assert recipe(dt_custom()) is None


def test_function_recipe():
    a, b = SSModelCTLinearFOSISO(name="a"), SSModelCTLinearFOSISO(name="b")
    model = connect_systems_in_series([a, b], StateSpaceModelCT, keys=["a", "b"])
    r = recipe(model)
    assert r["function"] == "connect_systems_in_series"
    assert r["args"]["systems"][0] is a and r["args"]["systems"][1] is b
    assert r["args"]["model_class"] is StateSpaceModelCT  # positional, named
    assert r["args"]["keys"] == ["a", "b"]


def test_operators_recipe():
    """*, + and - go through the connection functions"""
    a, b = SSModelCTLinearFOSISO(name="a"), SSModelCTLinearFOSISO(K=2, name="b")
    assert recipe(a * b)["function"] == "connect_systems_in_series"
    assert recipe(a + b)["function"] == "connect_systems_in_parallel"
    assert recipe(-a)["function"] == "negate_system"
    diff = recipe(a - b)
    assert diff["function"] == "connect_systems_in_parallel"
    assert recipe(diff["args"]["systems"][1])["function"] == "negate_system"


def test_recipe_args_copied():
    """Changing a list given afterwards doesn't change the recipe"""
    systems = [SSModelCTLinearFOSISO(name="a"), SSModelCTLinearFOSISO(name="b")]
    model = vstack_systems(systems)
    systems.append(SSModelCTLinearFOSISO(name="c"))
    assert len(recipe(model)["args"]["systems"]) == 2


# Writing and reading back


@pytest.mark.parametrize("name", FUNCTION_CASES)
def test_round_trip_functions(name, tmp_path):
    """Every connection function: the model made in code, written and
    read back, is the same model; writing it again gives the same text"""
    case_text, make_model = FUNCTION_CASES[name]
    model = make_model()
    text = to_yaml(model)
    function = case_text.split("\n", 1)[0]  # function: <name>
    assert text.split("\n")[1] == function
    loaded = from_yaml(text)
    assert_same_connected(loaded, model)
    assert to_yaml(loaded) == text
    model.save(tmp_path / "model")
    assert_same_connected(load_model(tmp_path / "model"), model)


def test_round_trip_nested():
    """Recipes nested to any depth, operators included"""
    plant = SSModelCTLinearFOSISO(K=1, T1=2, name="plant")
    ctrl = SSModelCTPIInt(Kc=0.5, Ti=30.0, name="ctrl")
    model = (plant * ctrl) + SSModelCTLinearO2SISO(name="d") - plant
    text = to_yaml(model)
    data = yaml.safe_load(text)
    assert data["function"] == "connect_systems_in_parallel"
    loaded = from_yaml(text)
    assert_same_connected(loaded, model)
    assert to_yaml(loaded) == text


def test_pickle_keeps_recipe():
    model = dt_connected_arx()
    loaded = pickle.loads(pickle.dumps(model))
    assert recipe(loaded)["function"] == "connect_systems"
    assert to_yaml(loaded) == to_yaml(model)


class CallableModel(StateSpaceModelCTFromABCD):
    """A subclass with an argument that can't be written (a function)"""

    def __init__(self, gain_of):
        super().__init__([[-1.0]], [[1.0]], [[gain_of(2.0)]], [[0.0]])


def test_unwritable_args_fall_back_to_base():
    model = CallableModel(lambda x: 3 * x)
    data = yaml.safe_load(to_yaml(model))
    assert data["class"] == "StateSpaceModelCT"
    loaded = from_yaml(to_yaml(model))
    assert type(loaded) is StateSpaceModelCT
    assert loaded.h.serialize() == model.h.serialize()
    # Inside a recipe, too
    data = yaml.safe_load(to_yaml(model * SSModelCTLinearFOSISO()))
    assert data["args"]["systems"][1]["class"] == "StateSpaceModelCT"


def test_tuple_and_complex_values():
    value = (1, (2.0, "a"), 0.5 + 0.25j)
    node = _encode(value, None, "x")
    assert node == {"tuple": [1, {"tuple": [2.0, "a"]}, {"complex": [0.5, 0.25]}]}
    assert _decode(node, None) == value
    model = StateSpaceModelDTARXSISO(poles=[0.8 + 0.3j, 0.8 - 0.3j], nb=1)
    data = yaml.safe_load(to_yaml(model))
    assert data["class"] == "StateSpaceModelDTARXSISO"
    assert_same_functions(from_yaml(to_yaml(model)), model)


# Symbols


def test_symbols_written():
    """Symbols given as arguments are written as symbol nodes; one
    symbol in two models stays one parameter"""
    K, T1 = cas.SX.sym("K"), cas.SX.sym("T1")
    model = vstack_systems(
        [
            SSModelCTLinearO2SISO(K=K, T1=T1, T2=5.0, name="a"),
            SSModelCTLinearFOSISO(K=K, T1=T1, name="b"),
        ]
    )
    text = to_yaml(model)
    data = yaml.safe_load(text)
    assert data["args"]["systems"][0]["args"]["K"] == {"symbol": "K"}
    assert data["args"]["systems"][1]["args"]["T1"] == {"symbol": "T1"}
    loaded = from_yaml(text)
    assert list(loaded.params) == ["K", "T1"]
    assert_same_connected(loaded, model)


def test_symbols_vector_and_matrix():
    model = StateSpaceModelDTARXSISO(poles=cas.SX.sym("p", 2), nb=1)
    data = yaml.safe_load(to_yaml(model))
    assert data["args"]["poles"] == {"symbol": "p", "shape": [2]}
    assert_same_functions(from_yaml(to_yaml(model)), model)
    model = StateSpaceModelCTFromABCD(
        A=cas.SX.sym("A", 2, 2), B=[[1.0], [0.0]], C=[[0.0, 1.0]], D=0.0
    )
    data = yaml.safe_load(to_yaml(model))
    assert data["args"]["A"] == {"symbol": "A", "shape": [2, 2]}
    assert_same_functions(from_yaml(to_yaml(model)), model)


def test_symbols_same_name_different_symbols():
    """They would load as one parameter"""
    model = vstack_systems(
        [
            SSModelCTLinearFOSISO(K=cas.SX.sym("K"), name="a"),
            SSModelCTLinearFOSISO(K=cas.SX.sym("K"), name="b"),
        ]
    )
    with pytest.raises(ValueError, match="two different symbols are named K"):
        to_yaml(model)


def test_symbols_element_of_vector_clash():
    """A scalar symbol named like an element of a vector symbol"""
    model = vstack_systems(
        [
            StateSpaceModelDTARXSISO(poles=cas.SX.sym("p", 2), nb=1, name="a"),
            StateSpaceModelDTARXSISO(poles=[cas.SX.sym("p_0")], nb=1, name="b"),
        ]
    )
    with pytest.raises(ValueError, match="an element of a vector symbol"):
        to_yaml(model)


def test_symbolic_expression_falls_back_to_base():
    """An argument that is an expression (2 * K) can't be written: the
    model is saved from its fields (its functions), still exactly"""
    K = cas.SX.sym("K")
    model = SSModelCTLinearFOSISO(K=2 * K)
    data = yaml.safe_load(to_yaml(model))
    assert data["class"] == "StateSpaceModelCT"
    loaded = from_yaml(to_yaml(model))
    assert list(loaded.params) == ["K", "T1"]
    assert loaded.f.serialize() == model.f.serialize()
    with pytest.raises(NotImplementedError, match="symbolic expression"):
        _encode(2 * K, None, "K")


# Parameter values


def eval_F(model, values=()):
    """F at a fixed state and input, with the parameter values given"""
    rng = np.random.default_rng(0)
    x, u = rng.standard_normal(model.n), rng.standard_normal(model.nu)
    return np.asarray(model.F(0.0, x, u, *values)).ravel()


ARX_VALUES = {
    "CRAC1_a_0": -1.5,
    "CRAC1_a_1": 0.56,
    "CRAC1_b_0": 0.1,
    "CRAC1_b_1": 0.05,
    "CRAC2_a": -0.8,
    "CRAC2_b": 0.2,
}


def test_params_values():
    """Values in place of the symbols: the model with those values"""
    model = dt_connected_arx()
    assert list(model.params) == list(ARX_VALUES)
    text = to_yaml(model, params=ARX_VALUES)
    loaded = from_yaml(text)
    assert loaded.params == {}
    np.testing.assert_allclose(
        eval_F(loaded), eval_F(model, ARX_VALUES.values()), rtol=1e-14
    )
    assert loaded.state_names == model.state_names


def test_params_some_values():
    """Parameters without values stay symbolic, with the same names"""
    model = dt_connected_arx()
    some = {k: v for k, v in ARX_VALUES.items() if k.startswith("CRAC1")}
    loaded = from_yaml(to_yaml(model, params=some))
    assert list(loaded.params) == ["CRAC2_a", "CRAC2_b"]
    np.testing.assert_allclose(
        eval_F(loaded, [-0.8, 0.2]),
        eval_F(model, ARX_VALUES.values()),
        rtol=1e-14,
    )


def test_params_vector_elements():
    """Values for some elements of a vector symbol"""
    model = StateSpaceModelDTARXSISO(poles=cas.SX.sym("p", 2), nb=1)
    data = yaml.safe_load(to_yaml(model, params={"p_0": 0.9}))
    assert data["args"]["poles"] == [0.9, {"symbol": "p_1"}]
    loaded = from_yaml(to_yaml(model, params={"p_0": 0.9}))
    assert list(loaded.params) == ["p_1", "b"]
    np.testing.assert_allclose(
        eval_F(loaded, [0.5, 0.2]), eval_F(model, [0.9, 0.5, 0.2]), rtol=1e-14
    )


def test_params_save(tmp_path):
    model = dt_connected_arx()
    model.save(tmp_path / "fitted", params=ARX_VALUES)
    loaded = load_model(tmp_path / "fitted")
    assert loaded.params == {}
    np.testing.assert_allclose(
        eval_F(loaded), eval_F(model, ARX_VALUES.values()), rtol=1e-14
    )


def test_params_errors():
    model = dt_connected_arx()
    with pytest.raises(ValueError, match="params: K isn't a parameter"):
        to_yaml(model, params={"K": 1.0})
    with pytest.raises(ValueError, match="params: CRAC2_a is 1 x 1"):
        to_yaml(model, params={"CRAC2_a": [1.0, 2.0]})


def test_dt_model_class_name():
    """model_class is written by name and found on reading"""
    data = yaml.safe_load(to_yaml(dt_connected_arx()))
    assert data["args"]["model_class"] == "StateSpaceModelDT"
    assert isinstance(from_yaml(to_yaml(dt_connected_arx())), StateSpaceModelDT)


def test_params_internal_symbols_as_node_values():
    """Parameters a class makes in its __init__ (na=2: a_0, a_1) aren't
    in its arguments: their values go in values: beside them"""
    data = yaml.safe_load(to_yaml(dt_connected_arx(), params=ARX_VALUES))
    path = data["args"]["systems"][0]
    assert path["args"] == {"na": 2, "nb": 2, "input_name": "u", "output_name": "y"}
    assert path["values"] == {"a_0": -1.5, "a_1": 0.56, "b_0": 0.1, "b_1": 0.05}
    assert data["args"]["systems"][1]["values"] == {"a": -0.8, "b": 0.2}


def test_params_base_form(tmp_path):
    """A model made from functions takes values too"""
    model = dt_custom()
    values = {name: 0.5 for name in model.params}
    loaded = from_yaml(to_yaml(model, params=values))
    assert loaded.params == {}
    np.testing.assert_allclose(
        eval_F(loaded), eval_F(model, [0.5] * len(values)), rtol=1e-14
    )


def test_set_param_values():
    model = dt_connected_arx()
    fixed = set_param_values(model, {"CRAC2_a": -0.8, "CRAC2_b": 0.2})
    assert type(fixed) is StateSpaceModelDT
    assert list(fixed.params) == ["CRAC1_a_0", "CRAC1_a_1", "CRAC1_b_0", "CRAC1_b_1"]
    assert fixed.params["CRAC1_a_0"] is model.params["CRAC1_a_0"]  # same symbols
    np.testing.assert_allclose(
        eval_F(fixed, [-1.5, 0.56, 0.1, 0.05]),
        eval_F(model, ARX_VALUES.values()),
        rtol=1e-14,
    )
    # Its recipe: the model's, with the values
    text = to_yaml(fixed)
    data = yaml.safe_load(text)
    assert data["function"] == "connect_systems"
    assert data["values"] == {"CRAC2_a": -0.8, "CRAC2_b": 0.2}
    loaded = from_yaml(text)
    assert list(loaded.params) == list(fixed.params)
    np.testing.assert_allclose(
        eval_F(loaded, [-1.5, 0.56, 0.1, 0.05]), eval_F(fixed, [-1.5, 0.56, 0.1, 0.05])
    )
    assert to_yaml(loaded) == text
    # Then all of them: the values add up
    rest = {k: v for k, v in ARX_VALUES.items() if k.startswith("CRAC1")}
    data = yaml.safe_load(to_yaml(set_param_values(fixed, rest)))
    assert data["values"] == ARX_VALUES | {"CRAC2_a": -0.8, "CRAC2_b": 0.2}


def test_set_param_values_errors():
    model = dt_connected_arx()
    with pytest.raises(ValueError, match="K: not parameters"):
        set_param_values(model, {"K": 1.0})
    with pytest.raises(ValueError, match="CRAC2_a is 1 x 1, the value 2 x 1"):
        set_param_values(model, {"CRAC2_a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="values must be a mapping"):
        from_yaml(
            "cas_models_format: 1\nclass: SSModelCTLinearFOSISO\nvalues: [1]\n"
        )
