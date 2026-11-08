"""Tests of recipes (cas_models.recipes): models record how they were
made, and save_model() / to_yaml() write that (the class or connection
function and its arguments, to any depth) so that from_yaml() and
load_model() rebuild the same model, with symbols shared as in the
original. A model with values set by set_param_values() is written
as that call: its structure from scratch and the values."""

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
from cas_models.yaml_utils import (
    _decode,
    _encode,
    from_yaml,
    load_model,
    to_yaml,
)

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
    model = SSModelCTLinearFOSISO(
        K=2.0
    )  # its __init__ calls SSModelCTFromABCDSISO's
    assert recipe(model) == {
        "class": SSModelCTLinearFOSISO,
        "args": {"K": 2.0},
    }
    assert recipe(SSModelCTPIInt(name="c")) == {
        "class": SSModelCTPIInt,
        "args": {"name": "c"},
    }


def test_base_class_has_no_recipe():
    """A model made from functions is saved from its fields"""
    assert recipe(dt_custom()) is None


def test_function_recipe():
    a, b = SSModelCTLinearFOSISO(name="a"), SSModelCTLinearFOSISO(name="b")
    model = connect_systems_in_series(
        [a, b], StateSpaceModelCT, keys=["a", "b"]
    )
    r = recipe(model)
    assert r["function"] == "connect_systems_in_series"
    assert r["args"]["systems"][0] is a and r["args"]["systems"][1] is b
    assert r["args"]["model_class"] is StateSpaceModelCT  # positional, named
    assert r["args"]["keys"] == ["a", "b"]


def test_operators_recipe():
    """*, + and - go through the connection functions"""
    a, b = (
        SSModelCTLinearFOSISO(name="a"),
        SSModelCTLinearFOSISO(K=2, name="b"),
    )
    assert recipe(a * b)["function"] == "connect_systems_in_series"
    assert recipe(a + b)["function"] == "connect_systems_in_parallel"
    assert recipe(-a)["function"] == "negate_system"
    diff = recipe(a - b)
    assert diff["function"] == "connect_systems_in_parallel"
    assert recipe(diff["args"]["systems"][1])["function"] == "negate_system"


def test_recipe_args_copied():
    """Changing a list given afterwards doesn't change the recipe"""
    systems = [
        SSModelCTLinearFOSISO(name="a"),
        SSModelCTLinearFOSISO(name="b"),
    ]
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
    assert node == {
        "tuple": [1, {"tuple": [2.0, "a"]}, {"complex": [0.5, 0.25]}]
    }
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
            StateSpaceModelDTARXSISO(
                poles=[cas.SX.sym("p_0")], nb=1, name="b"
            ),
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


def test_dt_model_class_name():
    """model_class is written by name and found on reading"""
    data = yaml.safe_load(to_yaml(dt_connected_arx()))
    assert data["args"]["model_class"] == "StateSpaceModelDT"
    assert isinstance(
        from_yaml(to_yaml(dt_connected_arx())), StateSpaceModelDT
    )


def test_set_param_values():
    """Values in place of some parameters, after building the model"""
    model = dt_connected_arx()
    assert list(model.params) == list(ARX_VALUES)
    fixed = set_param_values(model, {"CRAC2_a": -0.8, "CRAC2_b": 0.2})
    assert type(fixed) is StateSpaceModelDT
    assert list(fixed.params) == [
        "CRAC1_a_0",
        "CRAC1_a_1",
        "CRAC1_b_0",
        "CRAC1_b_1",
    ]
    assert (
        fixed.params["CRAC1_a_0"] is model.params["CRAC1_a_0"]
    )  # same symbols
    assert fixed.state_names == model.state_names
    np.testing.assert_allclose(
        eval_F(fixed, [-1.5, 0.56, 0.1, 0.05]),
        eval_F(model, ARX_VALUES.values()),
        rtol=1e-14,
    )
    # Then the rest: no parameters left
    rest = {k: v for k, v in ARX_VALUES.items() if k.startswith("CRAC1")}
    fitted = set_param_values(fixed, rest)
    assert fitted.params == {}
    np.testing.assert_allclose(
        eval_F(fitted), eval_F(model, ARX_VALUES.values()), rtol=1e-14
    )


def test_set_param_values_from_yaml():
    """The YAML is the symbolic model; values are set after loading it"""
    loaded = from_yaml(to_yaml(dt_connected_arx()))
    fitted = set_param_values(loaded, ARX_VALUES)
    np.testing.assert_allclose(
        eval_F(fitted),
        eval_F(dt_connected_arx(), ARX_VALUES.values()),
        rtol=1e-14,
    )


def test_set_param_values_recipe(tmp_path):
    """A fitted model is saved as the set_param_values call: the
    structure from scratch and the values; it loads as the same model"""
    model = dt_connected_arx()
    fitted = set_param_values(model, ARX_VALUES)
    r = recipe(fitted)
    assert r["function"] == "set_param_values"
    assert r["args"]["sys"] is model
    text = to_yaml(fitted)
    data = yaml.safe_load(text)
    assert data["function"] == "set_param_values"
    assert data["args"]["sys"]["function"] == "connect_systems"
    assert data["args"]["values"] == ARX_VALUES
    loaded = from_yaml(text)
    assert loaded.params == {}
    np.testing.assert_allclose(eval_F(loaded), eval_F(fitted), rtol=1e-14)
    assert to_yaml(loaded) == text
    folder = fitted.save(tmp_path / "fitted")
    np.testing.assert_allclose(eval_F(load_model(folder)), eval_F(fitted), rtol=1e-14)


def test_set_param_values_some_and_nested():
    """Some values (the rest stay symbolic, with the same names); values
    set twice; a fitted model inside a connection"""
    model = dt_connected_arx()
    some = {k: v for k, v in ARX_VALUES.items() if k.startswith("CRAC1")}
    partly = set_param_values(model, some)
    loaded = from_yaml(to_yaml(partly))
    assert list(loaded.params) == ["CRAC2_a", "CRAC2_b"]
    np.testing.assert_allclose(
        eval_F(loaded, [-0.8, 0.2]), eval_F(model, ARX_VALUES.values()), rtol=1e-14
    )
    twice = set_param_values(partly, {"CRAC2_a": -0.8, "CRAC2_b": 0.2})
    data = yaml.safe_load(to_yaml(twice))
    assert data["args"]["sys"]["function"] == "set_param_values"
    assert from_yaml(to_yaml(twice)).params == {}
    plant = set_param_values(SSModelCTLinearFOSISO(name="plant"), {"K": 2.0, "T1": 5.0})
    loop = SSModelCTPIInt(name="ctrl") * plant
    data = yaml.safe_load(to_yaml(loop))
    assert data["args"]["systems"][0]["function"] == "set_param_values"
    loaded = from_yaml(to_yaml(loop))
    assert list(loaded.params) == ["Kc", "Ti"]
    assert_same_connected(loaded, loop)


def test_set_param_values_numpy_values():
    """Values as estimate_params() may give them (numpy numbers, arrays)"""
    arx = StateSpaceModelDTARXSISO(poles=cas.SX.sym("p", 2), nb=1)
    fitted = set_param_values(arx, {"p_0": np.float64(0.9), "b": np.array([0.2])})
    loaded = from_yaml(to_yaml(fitted))
    assert list(loaded.params) == ["p_1"]
    np.testing.assert_allclose(eval_F(loaded, [0.5]), eval_F(fitted, [0.5]), rtol=1e-14)


def test_set_param_values_ct_and_vector():
    """Continuous-time models, and vector symbols one element at a time"""
    model = SSModelCTLinearO2SISO(name="G")
    fixed = set_param_values(model, {"K": 2.0, "T2": 5.0})
    assert type(fixed) is StateSpaceModelCT
    assert list(fixed.params) == ["T1"]
    x, u = np.array([0.3, -0.2]), np.array([1.0])
    np.testing.assert_allclose(
        np.asarray(fixed.f(0.0, x, u, 10.0)),
        np.asarray(model.f(0.0, x, u, 2.0, 10.0, 5.0)),
    )
    arx = StateSpaceModelDTARXSISO(poles=cas.SX.sym("p", 2), nb=1)
    fixed = set_param_values(arx, {"p_0": 0.9})
    assert list(fixed.params) == ["p_1", "b"]
    np.testing.assert_allclose(
        eval_F(fixed, [0.5, 0.2]), eval_F(arx, [0.9, 0.5, 0.2])
    )


def test_set_param_values_errors():
    model = dt_connected_arx()
    with pytest.raises(ValueError, match="K: not parameters"):
        set_param_values(model, {"K": 1.0})
    with pytest.raises(ValueError, match="CRAC2_a is 1 x 1, the value 2 x 1"):
        set_param_values(model, {"CRAC2_a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="only class and args, not values"):
        from_yaml(
            "cas_models_format: 1\nclass: SSModelCTLinearFOSISO\nvalues: {K: 1}\n"
        )


def test_symbols_and_numbers_in_a_matrix():
    """A matrix of numbers and single symbols is written element by
    element (rows), and loads as the same model of the same class"""
    from cas_models.discrete_time.models import StateSpaceModelDTFromABCD

    A = cas.vertcat(cas.SX.sym("A", 1, 2), cas.SX([[0, 1]]))
    model = StateSpaceModelDTFromABCD(
        A=A, B=cas.DM([[0], [1]]), C=cas.DM([[1, 0]]), D=cas.DM([[0]])
    )
    data = yaml.safe_load(to_yaml(model))
    assert data["class"] == "StateSpaceModelDTFromABCD"
    assert data["args"]["A"] == [
        [{"symbol": "A_0"}, {"symbol": "A_1"}],
        [0.0, 1.0],
    ]
    loaded = from_yaml(to_yaml(model))
    assert type(loaded) is StateSpaceModelDTFromABCD
    assert list(loaded.params) == ["A_0", "A_1"]
    assert loaded.F.serialize() == model.F.serialize()
    assert to_yaml(loaded) == to_yaml(model)


def test_symbols_and_numbers_in_a_vector():
    """A column vector is written as a list"""
    g = cas.SX.sym("g")
    B = cas.vertcat(g, 0.5)
    model = StateSpaceModelCTFromABCD(
        [[-1.0, 0.0], [0.0, -2.0]], B, [[1.0, 1.0]], 0.0
    )
    data = yaml.safe_load(to_yaml(model))
    assert data["args"]["B"] == [{"symbol": "g"}, 0.5]
    assert_same_functions(from_yaml(to_yaml(model)), model)


def test_expression_in_a_matrix_falls_back_to_base():
    K = cas.SX.sym("K")
    model = StateSpaceModelCTFromABCD([[-1.0]], [[2 * K]], [[1.0]], 0.0)
    data = yaml.safe_load(to_yaml(model))
    assert data["class"] == "StateSpaceModelCT"
    assert list(from_yaml(to_yaml(model)).params) == ["K"]
