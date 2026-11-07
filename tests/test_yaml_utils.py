"""Unit tests for saving and loading models (src/cas_models/yaml_utils.py
and the save and load methods of StateSpaceModelCT and StateSpaceModelDT)

Models don't compare equal after loading (the dataclass __eq__ compares
the CasADi Functions by identity), so assert_same_model() compares them
field by field: Functions by their signature and serialization and by
evaluating them at random arguments, the parameters by name and shape.
"""

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import casadi as cas
import numpy as np
import pytest
import yaml

from cas_models.continuous_time.models import (
    SSModelCTDirectTransmission,
    SSModelCTLinearFOSISO,
    SSModelCTLinearIntegratorSISO,
    SSModelCTLinearO2SISO,
    SSModelCTLinearO2UnderdampedSISO,
    StateSpaceModelCT,
    StateSpaceModelCTFromABCD,
)
from cas_models.continuous_time.regulators import (
    SSModelCTPIDInt,
    SSModelCTPIInt,
)
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTARXSISO,
    StateSpaceModelDTDelay,
    StateSpaceModelDTFromCT,
    StateSpaceModelDTFromCTRK4,
    StateSpaceModelDTTFSISO,
)
from cas_models.transformations import (
    block_diag_systems,
    connect_feedback_system,
    connect_systems,
    connect_systems_in_parallel,
    connect_systems_in_series,
    hstack_systems,
    negate_system,
    vstack_systems,
)
from cas_models.yaml_utils import (
    _decode,
    _encode,
    _file_refs,
    _without_defaults,
    from_yaml,
    load_model,
    read_tsv,
    to_yaml,
    write_tsv,
)


def assert_same_model(model, loaded, base, exact=True):
    """Check a loaded model has the same base-class fields as the
    original. Functions are compared by their serialization only if
    exact."""
    assert type(loaded) is base
    for field in fields(base):
        value, loaded_value = (
            getattr(model, field.name),
            getattr(loaded, field.name),
        )
        if isinstance(value, cas.Function):
            assert str(loaded_value) == str(value), field.name
            if exact:
                assert loaded_value.serialize() == value.serialize()
        elif field.name == "params":
            # New symbols, with the same names and shapes
            assert list(loaded_value) == list(value)
            for name, p in loaded_value.items():
                assert isinstance(p, cas.SX) and p.is_symbolic()
                assert p.shape == value[name].shape
        else:
            assert loaded_value == value, field.name

    # The state and output functions give the same results
    rng = np.random.default_rng(0)
    for func in ("state_func", "output_func"):
        f = getattr(model, base._attr_names[func])
        f_loaded = getattr(loaded, base._attr_names[func])
        args = [rng.uniform(0.5, 1.5, f.size_in(i)) for i in range(f.n_in())]
        for out, out_loaded in zip(f.call(args), f_loaded.call(args)):
            np.testing.assert_array_equal(out_loaded.full(), out.full())


def read_model_yaml(folder):
    with open(folder / "model.yaml") as f:
        return yaml.safe_load(f)


# Models


def ct_custom():
    """Nonlinear, with a vector parameter p"""
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 2)
    u = cas.SX.sym("u")
    p = cas.SX.sym("p", 2)
    rhs = cas.vertcat(-p[0] * x[0] ** 2 + u, x[0] - p[1] * x[1])
    f = cas.Function("f", [t, x, u, p], [rhs], ["t", "x", "u", "p"], ["rhs"])
    h = cas.Function("h", [t, x, u, p], [x[1]], ["t", "x", "u", "p"], ["y"])
    return StateSpaceModelCT(
        f, h, n=2, params={"p": p}, name="custom", state_names=["x", "z"]
    )


def dt_custom():
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", 2)
    uk = cas.SX.sym("uk")
    a = cas.SX.sym("a")
    xkp1 = cas.vertcat(a * xk[0] + uk, cas.sin(xk[0]) + 0.5 * xk[1])
    F = cas.Function(
        "F", [t, xk, uk, a], [xkp1], ["t", "xk", "uk", "a"], ["xkp1"]
    )
    H = cas.Function(
        "H", [t, xk, uk, a], [xk[1]], ["t", "xk", "uk", "a"], ["yk"]
    )
    return StateSpaceModelDT(F, H, n=2, dt=0.5, params={"a": a})


def dt_connected_arx():
    """Two ARX paths summed, as in ffdtools.models.ARXPaths"""
    inputs = ["CRAC1", "CRAC2"]
    paths = [
        StateSpaceModelDTARXSISO(na=2, nb=2, input_name="u", output_name="y"),
        StateSpaceModelDTARXSISO(na=1, nb=1, input_name="u", output_name="y"),
    ]
    connections = {f"{name}_u": name for name in inputs}
    connections["R11"] = [f"{name}_y" for name in inputs]
    return connect_systems(
        paths,
        connections,
        StateSpaceModelDT,
        input_names=inputs,
        output_names=["R11"],
        keys=inputs,
        verbose_names=True,
        name="R11",
    )


def ct_pi_feedback():
    plant = SSModelCTLinearFOSISO(K=1, T1=2, name="plant")
    ctrl = SSModelCTPIInt(name="ctrl")
    return connect_feedback_system(plant * ctrl, model_class=StateSpaceModelCT)


# (base class, make model, Functions compared exactly)
MODELS = {
    "CT custom": (StateSpaceModelCT, ct_custom, True),
    "CT PI feedback": (StateSpaceModelCT, ct_pi_feedback, True),
    "CT sum": (
        StateSpaceModelCT,
        lambda: SSModelCTLinearFOSISO() + SSModelCTLinearFOSISO(K=2),
        True,
    ),
    "DT custom": (StateSpaceModelDT, dt_custom, True),
    "DT connected ARX": (StateSpaceModelDT, dt_connected_arx, True),
    # Subclasses without their own saved arguments: saved as their base
    "CT FO (as base)": (StateSpaceModelCT, SSModelCTLinearFOSISO, True),
    "DT ARX (as base)": (
        StateSpaceModelDT,
        lambda: StateSpaceModelDTARXSISO(na=2, nb=1),
        True,
    ),
    "DT ARX symbolic poles and zeros (as base)": (
        StateSpaceModelDT,
        lambda: StateSpaceModelDTARXSISO(
            poles=[cas.SX.sym("p_0"), cas.SX.sym("p_1")],
            zeros=[cas.SX.sym("z_0")],
        ),
        True,
    ),
    "DT ARX with dt (as base)": (
        StateSpaceModelDT,
        lambda: StateSpaceModelDTARXSISO(na=2, nb=1, dt=2.0),
        True,
    ),
    "DT ARX numeric poles and zeros (as base)": (
        StateSpaceModelDT,
        lambda: StateSpaceModelDTARXSISO(
            poles=[0.8 + 0.3j, 0.8 - 0.3j], zeros=[-0.4], gain=0.2, nk=2
        ),
        True,
    ),
    # CasADi changes the serialization of a cvodes integrator in each
    # serialize-deserialize round trip (the results are the same)
    "DT from CT cvodes (as base)": (
        StateSpaceModelDT,
        lambda: StateSpaceModelDTFromCT(SSModelCTLinearFOSISO(), 1.0),
        False,
    ),
}


@pytest.mark.parametrize("name", MODELS)
def test_save_load(name, tmp_path):
    base, make_model, exact = MODELS[name]
    model = make_model()
    folder = tmp_path / "model"
    model.save(folder)
    assert_same_model(model, base.load(folder), base, exact)
    assert_same_model(model, load_model(folder), base, exact)


def test_saved_files(tmp_path):
    """The folder holds model.yaml and the functions' files"""
    model = dt_connected_arx()
    folder = tmp_path / "R11"
    model.save(folder)
    assert sorted(p.name for p in folder.iterdir()) == [
        "F.casadi",
        "H.casadi",
        "model.yaml",
    ]
    data = read_model_yaml(folder)
    assert data == {
        "cas_models_format": 1,
        "class": "StateSpaceModelDT",
        "args": {
            "F": {"file": "F.casadi"},
            "H": {"file": "H.casadi"},
            "n": 6,
            "nu": 2,
            "params": {
                "CRAC1_a_0": [1, 1],
                "CRAC1_a_1": [1, 1],
                "CRAC1_b_0": [1, 1],
                "CRAC1_b_1": [1, 1],
                "CRAC2_a": [1, 1],
                "CRAC2_b": [1, 1],
            },
            "name": "R11",
            "input_names": ["CRAC1", "CRAC2"],
            "state_names": [
                "CRAC1_x1",
                "CRAC1_x2",
                "CRAC1_x3",
                "CRAC1_x4",
                "CRAC2_x1",
                "CRAC2_x2",
            ],
            "output_names": ["R11"],
        },
    }


def test_saved_files_ct(tmp_path):
    model = ct_custom()
    folder = tmp_path / "custom"
    model.save(folder)
    data = read_model_yaml(folder)
    assert data["class"] == "StateSpaceModelCT"
    assert data["args"]["f"] == {"file": "f.casadi"}
    assert data["args"]["h"] == {"file": "h.casadi"}
    assert data["args"]["params"] == {"p": [2, 1]}
    assert "dt" not in data["args"]


def test_defaults_left_out(tmp_path):
    """Arguments equal to their defaults aren't saved (here ny=1, dt,
    params when there are none, and the names)"""
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk")
    uk = cas.SX.sym("uk")
    F = cas.Function(
        "F", [t, xk, uk], [0.5 * xk + uk], ["t", "xk", "uk"], ["xkp1"]
    )
    H = cas.Function("H", [t, xk, uk], [xk], ["t", "xk", "uk"], ["yk"])
    model = StateSpaceModelDT(F, H, n=1)
    folder = tmp_path / "model"
    model.save(folder)
    assert read_model_yaml(folder)["args"] == {
        "F": {"file": "F.casadi"},
        "H": {"file": "H.casadi"},
        "n": 1,
        "input_names": ["u"],
        "state_names": ["x"],
        "output_names": ["y"],
    }
    assert_same_model(model, StateSpaceModelDT.load(folder), StateSpaceModelDT)


class Example:
    def __init__(self, a, b=1, c=None, d="cvodes", e=2.0, f=True, g=(1, 2)):
        pass


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"a": 5, "b": 1, "c": None, "d": "cvodes"}, {"a": 5}),
        ({"a": None}, {"a": None}),  # no default
        ({"b": 2, "c": 0.0, "d": "rk"}, {"b": 2, "c": 0.0, "d": "rk"}),
        ({"b": 1.0}, {"b": 1.0}),  # a float isn't the int default
        ({"b": True, "f": 1}, {"b": True, "f": 1}),  # bools aren't ints
        ({"e": 2.0, "f": True}, {}),
        ({"g": (1, 2)}, {"g": (1, 2)}),  # only plain defaults left out
        ({"c": cas.DM(0)}, {"c": cas.DM(0)}),
    ],
)
def test_without_defaults(args, expected):
    result = _without_defaults(Example, args)
    assert list(result) == list(expected)
    for k in expected:
        assert result[k] is args[k]


def write_model_yaml(folder, text):
    folder.mkdir()
    (folder / "model.yaml").write_text("cas_models_format: 1\n" + text)


def assert_same_functions(model, expected):
    """The same class, and the same state and output functions"""
    assert type(model) is type(expected)
    for func in ("state_func", "output_func"):
        name = expected._attr_names[func]
        f, f_expected = getattr(model, name), getattr(expected, name)
        assert f.serialize() == f_expected.serialize(), name
    assert list(model.params) == list(expected.params)


def test_load_without_args(tmp_path):
    """A model.yaml without args loads the class with no arguments"""
    write_model_yaml(tmp_path / "model", "class: SSModelCTLinearFOSISO\n")
    model = StateSpaceModelCT.load(tmp_path / "model")
    assert_same_functions(model, SSModelCTLinearFOSISO())
    assert list(model.params) == ["K", "T1"]


# Subclasses from the arguments of their __init__ (as written by hand,
# until they save their own)

LOAD_CASES = {
    "O2 mixed": (
        "class: SSModelCTLinearO2SISO\nargs: {K: 2.0, name: G}\n",
        lambda: SSModelCTLinearO2SISO(K=2.0, name="G"),
    ),
    "PID": (
        "class: SSModelCTPIDInt\nargs: {Kc: 2.0, Ti: 5.0, Td: 0.8, Tf: 0.1}\n",
        lambda: SSModelCTPIDInt(Kc=2.0, Ti=5.0, Td=0.8, Tf=0.1),
    ),
    "ABCD, DM": (
        "class: StateSpaceModelCTFromABCD\n"
        "args:\n"
        "  A:\n"
        "    DM:\n"
        "    - [-1.0, 0.0]\n"
        "    - [0.5, -2.0]\n"
        "  B: {DM: [1.0, 0.0]}\n"
        "  C: {DM: [[0.0, 1.0]]}\n"
        "  D: {DM: [0.0]}\n",
        lambda: StateSpaceModelCTFromABCD(
            cas.DM([[-1.0, 0.0], [0.5, -2.0]]),
            cas.DM([1.0, 0.0]),
            cas.DM([[0.0, 1.0]]),
            cas.DM(0.0),
        ),
    ),
    "ABCD, plain lists": (
        "class: StateSpaceModelCTFromABCD\n"
        "args:\n"
        "  A:\n"
        "  - [-1.0, 0.0]\n"
        "  - [0.5, -2.0]\n"
        "  B: [1.0, 0.0]\n"
        "  C: [[0.0, 1.0]]\n"
        "  D: 0.0\n",
        lambda: StateSpaceModelCTFromABCD(
            cas.DM([[-1.0, 0.0], [0.5, -2.0]]),
            cas.DM([1.0, 0.0]),
            cas.DM([[0.0, 1.0]]),
            cas.DM(0.0),
        ),
    ),
    "ARX symbolic": (
        "class: StateSpaceModelDTARXSISO\nargs: {na: 2, nb: 1}\n",
        lambda: StateSpaceModelDTARXSISO(na=2, nb=1),
    ),
    "ARX fixed": (
        "class: StateSpaceModelDTARXSISO\n"
        "args:\n"
        "  A: {DM: [-1.62, 0.65]}\n"
        "  B: {DM: [0.012, 0.009]}\n",
        lambda: StateSpaceModelDTARXSISO(
            A=cas.DM([-1.62, 0.65]), B=cas.DM([0.012, 0.009])
        ),
    ),
    "ARX poles and zeros": (
        "class: StateSpaceModelDTARXSISO\n"
        "args: {poles: [0.9, 0.5], zeros: [-0.4], gain: 0.2, nk: 2}\n",
        lambda: StateSpaceModelDTARXSISO(
            poles=[0.9, 0.5], zeros=[-0.4], gain=0.2, nk=2
        ),
    ),
    "ARX poles and zeros, DM": (
        "class: StateSpaceModelDTARXSISO\n"
        "args:\n"
        "  poles: {DM: [0.9, 0.5]}\n"
        "  zeros: {DM: [-0.4]}\n"
        "  gain: 0.2\n",
        lambda: StateSpaceModelDTARXSISO(
            poles=[0.9, 0.5], zeros=[-0.4], gain=0.2
        ),
    ),
    "ARX with dt": (
        "class: StateSpaceModelDTARXSISO\nargs: {na: 1, nb: 1, dt: 2.0}\n",
        lambda: StateSpaceModelDTARXSISO(na=1, nb=1, dt=2.0),
    ),
    "ARX poles, symbolic B": (
        "class: StateSpaceModelDTARXSISO\nargs: {poles: [0.9], nb: 2}\n",
        lambda: StateSpaceModelDTARXSISO(poles=[0.9], nb=2),
    ),
    "TF": (
        "class: StateSpaceModelDTTFSISO\n"
        "args: {num: [0.2, 0.1], den: [1, -1.4, 0.49]}\n",
        lambda: StateSpaceModelDTTFSISO(num=[0.2, 0.1], den=[1, -1.4, 0.49]),
    ),
    "delay": (
        "class: StateSpaceModelDTDelay\n"
        "args:\n"
        "  nk: 3\n"
        "  nu: 2\n"
        "  G:\n"
        "    DM:\n"
        "    - [1.0, 0.5]\n"
        "    - [0.0, 2.0]\n"
        "  dt: 1.0\n",
        lambda: StateSpaceModelDTDelay(
            nk=3, nu=2, G=cas.DM([[1.0, 0.5], [0.0, 2.0]]), dt=1.0
        ),
    ),
    "nested model": (
        "class: StateSpaceModelDTFromCTRK4\n"
        "args:\n"
        "  model_ct:\n"
        "    class: SSModelCTLinearO2SISO\n"
        "    args: {K: 2.0}\n"
        "  dt: 1.0\n",
        lambda: StateSpaceModelDTFromCTRK4(SSModelCTLinearO2SISO(K=2.0), 1.0),
    ),
    "nested model without args": (
        "class: StateSpaceModelDTFromCTRK4\n"
        "args:\n"
        "  model_ct: {class: SSModelCTLinearFOSISO}\n"
        "  dt: 0.5\n",
        lambda: StateSpaceModelDTFromCTRK4(SSModelCTLinearFOSISO(), 0.5),
    ),
}


@pytest.mark.parametrize("name", LOAD_CASES)
def test_load_subclass_args(name, tmp_path):
    text, make_model = LOAD_CASES[name]
    write_model_yaml(tmp_path / "model", text)
    expected = make_model()
    assert_same_functions(load_model(tmp_path / "model"), expected)
    # Loaded by its own class or a base class
    assert_same_functions(type(expected).load(tmp_path / "model"), expected)


def test_load_subclass_wrong_class(tmp_path):
    write_model_yaml(
        tmp_path / "model", "class: SSModelCTLinearO2SISO\nargs: {K: 2.0}\n"
    )
    with pytest.raises(ValueError, match="isn't StateSpaceModelDTARXSISO"):
        StateSpaceModelDTARXSISO.load(tmp_path / "model")
    with pytest.raises(ValueError, match="isn't StateSpaceModelDT"):
        StateSpaceModelDT.load(tmp_path / "model")


def test_save_creates_parent_folders(tmp_path):
    folder = tmp_path / "a" / "b" / "model"
    dt_custom().save(folder)
    assert (folder / "model.yaml").is_file()


def test_save_overwrite(tmp_path):
    """Saving in a folder that isn't empty needs overwrite=True, which
    deletes the old model's files and leaves the others"""
    folder = tmp_path / "model"
    ct_custom().save(folder)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        dt_custom().save(folder)
    (folder / "notes.txt").write_text("keep me")

    # The old model's files are f.casadi and h.casadi
    model = dt_custom()
    model.save(folder, overwrite=True)
    assert sorted(p.name for p in folder.iterdir()) == [
        "F.casadi",
        "H.casadi",
        "model.yaml",
        "notes.txt",
    ]
    assert_same_model(model, StateSpaceModelDT.load(folder), StateSpaceModelDT)


def test_save_in_folder_of_other_files(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        dt_custom().save(folder)
    dt_custom().save(folder, overwrite=True)
    assert (folder / "notes.txt").read_text() == "keep me"


def test_load_wrong_class(tmp_path):
    folder = tmp_path / "model"
    dt_custom().save(folder)
    with pytest.raises(ValueError, match="isn't StateSpaceModelCT"):
        StateSpaceModelCT.load(folder)
    # A subclass saved in its base form loads only as the base
    StateSpaceModelDTARXSISO(na=1, nb=1).save(tmp_path / "arx")
    with pytest.raises(ValueError, match="isn't StateSpaceModelDTARXSISO"):
        StateSpaceModelDTARXSISO.load(tmp_path / "arx")


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no model.yaml"):
        load_model(tmp_path)

    folder = tmp_path / "model"
    dt_custom().save(folder)
    data = read_model_yaml(folder)

    data["cas_models_format"] = 2
    with open(folder / "model.yaml", "w") as f:
        yaml.safe_dump(data, f)
    with pytest.raises(ValueError, match="cas_models_format 2, not 1"):
        load_model(folder)

    data["cas_models_format"] = 1
    data["class"] = "NoSuchModel"
    with open(folder / "model.yaml", "w") as f:
        yaml.safe_dump(data, f)
    with pytest.raises(ValueError, match="'NoSuchModel' isn't"):
        load_model(folder)


def test_save_params_not_symbols(tmp_path):
    """Parameters must be SX symbols (their names and shapes are saved)"""
    model = dt_custom()
    model.params = {"a": 2 * model.params["a"]}
    with pytest.raises(NotImplementedError, match="isn't an SX symbol"):
        model.save(tmp_path / "model")


# Tab-separated text files


@pytest.mark.parametrize(
    "value,shape",
    [
        (cas.DM([[0.1, 1 / 3], [np.pi, -2e-300]]), [2, 2]),
        (cas.DM([1.0, 2.5, -3.0]), [3, 1]),  # column vector
        (cas.DM([[1.0, 2.5, -3.0]]), [1, 3]),  # row vector
        (np.array([1.0, 2.0]), [2, 1]),  # 1-D: a column vector
        (np.array([[1, 2, 3], [4, 5, 6]]), [2, 3]),
        (cas.DM(7.0), [1, 1]),
        (cas.DM(0, 3), [0, 3]),
        (cas.DM(2, 0), [2, 0]),
    ],
)
def test_tsv_round_trip(value, shape, tmp_path):
    path = tmp_path / "m.tsv"
    assert write_tsv(path, value) == shape
    loaded = read_tsv(path, shape)
    assert isinstance(loaded, cas.DM)
    assert loaded.shape == tuple(shape)
    expected = value.full() if isinstance(value, cas.DM) else value
    np.testing.assert_array_equal(loaded.full(), np.reshape(expected, shape))


def test_tsv_text(tmp_path):
    """One row per line, tab-separated, the shortest exact values"""
    path = tmp_path / "m.tsv"
    write_tsv(path, cas.DM([[0.1, 1 / 3], [np.pi, -2e-300]]))
    assert path.read_text() == (
        "0.1\t0.3333333333333333\n3.141592653589793\t-2e-300\n"
    )
    write_tsv(path, cas.DM([1.0, 2.5]))
    assert path.read_text() == "1.0\n2.5\n"


def test_tsv_sparse(tmp_path):
    """A sparse matrix is written dense, its structural zeros as 0.0"""
    path = tmp_path / "m.tsv"
    value = cas.sparsify(cas.DM([[1.0, 0.0], [0.0, 4.0]]))
    assert value.nnz() == 2
    write_tsv(path, value)
    assert path.read_text() == "1.0\t0.0\n0.0\t4.0\n"
    np.testing.assert_array_equal(
        read_tsv(path, [2, 2]).full(), [[1.0, 0.0], [0.0, 4.0]]
    )


def test_tsv_wrong_shape(tmp_path):
    path = tmp_path / "m.tsv"
    write_tsv(path, cas.DM([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="3 x 1 values, not 1 x 3"):
        read_tsv(path, [1, 3])


# Vectors and small matrices in model.yaml, larger ones in data files.
# No base-class argument is a matrix, so these use the private helpers
# that save_model() and load_model() use for each argument.


@pytest.mark.parametrize(
    "value,expected",
    [
        (cas.DM([1.0, 2.5]), {"DM": [1.0, 2.5]}),  # column vector
        (cas.DM([[1.0, 2.5, -3.0]]), {"DM": [[1.0, 2.5, -3.0]]}),  # row
        (cas.DM([[1, 2], [3, 4]]), {"DM": [[1.0, 2.0], [3.0, 4.0]]}),
        (cas.DM(7.0), {"DM": [7.0]}),
        (np.array([1, 2]), {"DM": [1.0, 2.0]}),  # 1-D: a column vector
        (cas.DM(np.arange(8.0)), {"DM": list(np.arange(8.0))}),  # 8: inline
        (cas.DM(np.arange(8.0)).T, {"DM": [list(np.arange(8.0))]}),
        (
            cas.DM(np.arange(25.0).reshape(5, 5)),  # 5 x 5: inline
            {"DM": np.arange(25.0).reshape(5, 5).tolist()},
        ),
        (cas.DM(0, 3), {"DM": [], "shape": [0, 3]}),
        (cas.DM(2, 0), {"DM": [], "shape": [2, 0]}),
        # Sparse: dense, its structural zeros as 0.0
        (
            cas.sparsify(cas.DM([[1.0, 0.0], [0.0, 4.0]])),
            {"DM": [[1.0, 0.0], [0.0, 4.0]]},
        ),
        # Constant SX: as DM
        (cas.SX([0.2, 0.1]), {"DM": [0.2, 0.1]}),
    ],
)
def test_inline_matrix(value, expected, tmp_path):
    encoded = _encode(value, tmp_path, "A")
    assert encoded == expected
    assert list(tmp_path.iterdir()) == []  # no data file
    # Through YAML text and back
    loaded = _decode(yaml.safe_load(yaml.safe_dump(encoded)), tmp_path)
    assert isinstance(loaded, cas.DM)
    value = cas.DM(value)  # (a 1-D array as a column vector)
    assert loaded.shape == value.shape
    np.testing.assert_array_equal(loaded.full(), value.full())


def test_inline_matrix_yaml_text(tmp_path):
    """Readable in model.yaml, values in full precision"""
    args = {
        "A": _encode(cas.DM([-1.62, 1 / 3]), tmp_path, "A"),
        "D": _encode(cas.DM([[1.0, 0.0], [0.5, 2.0]]), tmp_path, "D"),
    }
    text = yaml.safe_dump(args, sort_keys=False, default_flow_style=None)
    assert text == (
        "A:\n"
        "  DM: [-1.62, 0.3333333333333333]\n"
        "D:\n"
        "  DM:\n"
        "  - [1.0, 0.0]\n"
        "  - [0.5, 2.0]\n"
    )


@pytest.mark.parametrize(
    "shape",
    [
        (9, 1),  # vectors of more than MAX_INLINE_VECTOR elements
        (1, 9),
        (6, 5),  # matrices larger than MAX_INLINE_MATRIX
        (5, 6),
        (6, 2),
    ],
)
def test_matrix_to_file(shape, tmp_path):
    """Larger vectors and matrices: a data file"""
    value = cas.DM(np.arange(float(np.prod(shape))).reshape(shape))
    encoded = _encode(value, tmp_path, "A")
    assert encoded == {"file": "A.tsv", "shape": list(shape)}
    loaded = _decode(encoded, tmp_path)
    assert loaded.shape == shape
    np.testing.assert_array_equal(loaded.full(), value.full())


def test_matrix_file_text(tmp_path):
    value = cas.DM(np.arange(12.0).reshape(6, 2))
    _encode(value, tmp_path, "A")
    assert (tmp_path / "A.tsv").read_text() == (
        "0.0\t1.0\n2.0\t3.0\n4.0\t5.0\n6.0\t7.0\n8.0\t9.0\n10.0\t11.0\n"
    )


def test_matrices_in_dict_and_list(tmp_path):
    """Nested arguments: files named <argument>_<key> or _<index>"""
    big = cas.DM(np.arange(10.0))
    value = {"G": big, "small": cas.DM([1.0, 2.0]), "items": [big, 3]}
    encoded = _encode(value, tmp_path, "opts")
    assert encoded == {
        "G": {"file": "opts_G.tsv", "shape": [10, 1]},
        "small": {"DM": [1.0, 2.0]},
        "items": [{"file": "opts_items_0.tsv", "shape": [10, 1]}, 3],
    }
    loaded = _decode(encoded, tmp_path)
    np.testing.assert_array_equal(loaded["G"].full(), big.full())
    np.testing.assert_array_equal(loaded["items"][0].full(), big.full())
    assert loaded["items"][1] == 3


def test_symbolic_expression_not_saved(tmp_path):
    a = cas.SX.sym("a")
    with pytest.raises(NotImplementedError, match="symbolic expression"):
        _encode(cas.vertcat(1, a), tmp_path, "den")


def test_load_imports_model_modules(tmp_path):
    """Classes in modules the caller hasn't imported (here the
    regulators) are found: run in a new Python process, as this module
    imports them"""
    write_model_yaml(tmp_path / "model", "class: SSModelCTPIDInt\n")
    code = (
        "import sys\n"
        "from cas_models.yaml_utils import load_model\n"
        "assert 'cas_models.continuous_time.regulators' not in sys.modules\n"
        f"model = load_model({str(tmp_path / 'model')!r})\n"
        "print(type(model).__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SSModelCTPIDInt"


# Saved model files in tests/data/models: each loads and is the same as
# the model built directly. They also check that files saved earlier
# still load.

MODELS_DIR = Path(__file__).parent / "data" / "models"


def dt_connected_arx_flow():
    """As dt_connected_arx(), with the names of the four_racks_dr case"""
    inputs = ["CRAC1_flow", "CRAC2_flow"]
    paths = [
        StateSpaceModelDTARXSISO(na=2, nb=2, input_name="u", output_name="y"),
        StateSpaceModelDTARXSISO(na=1, nb=1, input_name="u", output_name="y"),
    ]
    connections = {f"{name}_u": name for name in inputs}
    connections["R11_intake"] = [f"{name}_y" for name in inputs]
    return connect_systems(
        paths,
        connections,
        StateSpaceModelDT,
        input_names=inputs,
        output_names=["R11_intake"],
        keys=inputs,
        verbose_names=True,
        name="R11_intake",
    )


def ct_abcd_6_states():
    rng = np.random.default_rng(1)
    A = np.round(-np.eye(6) + 0.1 * rng.standard_normal((6, 6)), 3)
    return StateSpaceModelCTFromABCD(
        cas.DM(A), cas.DM(np.eye(6, 1)), cas.DM(np.ones((1, 6)) / 6), 0.0
    )


MODEL_FILES = {
    "ct_fo_symbolic": lambda: SSModelCTLinearFOSISO(),
    "ct_o2_fixed": lambda: SSModelCTLinearO2SISO(
        K=2.0, T1=3.0, T2=5.0, name="G", input_name="Q", output_name="T"
    ),
    "ct_o2_mixed": lambda: SSModelCTLinearO2SISO(K=2.0),
    "ct_integrator": lambda: SSModelCTLinearIntegratorSISO(K=0.5),
    "ct_o2_underdamped": lambda: SSModelCTLinearO2UnderdampedSISO(
        K=1.0, zeta=0.3, omega_n=2.0
    ),
    "ct_pi": lambda: SSModelCTPIInt(Kc=0.5, Ti=30.0, name="ctrl"),
    "ct_pid": lambda: SSModelCTPIDInt(),
    "ct_abcd": lambda: StateSpaceModelCTFromABCD(
        [[-1.0, 0.0, 0.0], [0.5, -2.0, 0.0], [0.0, 0.0, -3.0]],
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        input_names=["a", "b"],
    ),
    "ct_abcd_6_states": ct_abcd_6_states,
    "ct_direct": lambda: SSModelCTDirectTransmission(
        D=[[1.0, 0.0], [0.5, 2.0]]
    ),
    "ct_pi_feedback": ct_pi_feedback,
    "dt_arx_symbolic": lambda: StateSpaceModelDTARXSISO(na=2, nb=2),
    "dt_arx_fixed": lambda: StateSpaceModelDTARXSISO(
        A=cas.DM([-1.62, 0.65]), B=cas.DM([0.012, 0.009])
    ),
    "dt_tf": lambda: StateSpaceModelDTTFSISO(
        num=[0.2, 0.1], den=[1, -1.4, 0.49]
    ),
    "dt_delay": lambda: StateSpaceModelDTDelay(
        nk=3, nu=2, G=[[1.0, 0.5], [0.0, 2.0]], dt=1.0
    ),
    "dt_from_ct_rk4": lambda: StateSpaceModelDTFromCTRK4(
        SSModelCTLinearO2SISO(K=2.0), 1.0
    ),
    "dt_from_ct_cvodes": lambda: StateSpaceModelDTFromCT(
        SSModelCTLinearFOSISO(), 1.0, integrator_opts={"abstol": 1e-10}
    ),
    "dt_connected_arx": dt_connected_arx_flow,
}

# Saved by save() (the base-class form); the others hold the arguments
# of their class's __init__
SAVED_AS_BASE = {
    "ct_pi_feedback": StateSpaceModelCT,
    "dt_connected_arx": StateSpaceModelDT,
}


def test_model_files_all_tested():
    folders = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
    assert folders == sorted(MODEL_FILES)


@pytest.mark.parametrize("name", MODEL_FILES)
def test_load_model_file(name):
    expected = MODEL_FILES[name]()
    model = load_model(MODELS_DIR / name)
    if name in SAVED_AS_BASE:
        assert_same_model(expected, model, SAVED_AS_BASE[name])
    else:
        assert_same_functions(model, expected)


@pytest.mark.parametrize("name", SAVED_AS_BASE)
def test_save_model_file(name, tmp_path):
    """save() writes the same model.yaml (the .casadi files depend on
    the CasADi version: they are checked by loading them)"""
    folder = tmp_path / name
    MODEL_FILES[name]().save(folder)
    assert (folder / "model.yaml").read_text() == (
        MODELS_DIR / name / "model.yaml"
    ).read_text()
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        p.name for p in (MODELS_DIR / name).iterdir()
    )


# YAML text: to_yaml() and from_yaml()


@pytest.mark.parametrize("name", MODELS)
def test_to_yaml_from_yaml(name):
    """The text alone rebuilds the model, by the base class's from_yaml()
    or from_yaml()"""
    base, make_model, exact = MODELS[name]
    model = make_model()
    text = model.to_yaml()
    assert text == to_yaml(model)
    assert _file_refs(yaml.safe_load(text)) == []
    assert_same_model(model, base.from_yaml(text), base, exact)
    assert_same_model(model, from_yaml(text), base, exact)


def test_to_yaml_same_as_saved(tmp_path):
    """to_yaml() is model.yaml with the functions inline"""
    model = dt_connected_arx()
    model.save(tmp_path / "model")
    saved = read_model_yaml(tmp_path / "model")
    data = yaml.safe_load(model.to_yaml())
    for key in ("F", "H"):
        assert saved["args"][key] == {"file": f"{key}.casadi"}
        assert data["args"][key] == {"Function": getattr(model, key).serialize()}
        del saved["args"][key], data["args"][key]
    assert data == saved


def test_to_yaml_text():
    """Functions as {Function: <serialization>}, a model of no
    arguments as just its class"""
    text = SSModelCTLinearFOSISO().to_yaml()
    data = yaml.safe_load(text)
    assert data["class"] == "StateSpaceModelCT"
    assert set(data["args"]["f"]) == {"Function"}
    text = StateSpaceModelDTARXSISO(na=2, nb=1).to_yaml()
    assert text.startswith("cas_models_format: 1\nclass: StateSpaceModelDT\n")


@pytest.mark.parametrize("shape", [(9, 1), (1, 9), (6, 5), (20, 20)])
def test_to_yaml_large_matrix_inline(shape):
    """Without a folder, matrices of any size are inline"""
    value = cas.DM(np.arange(float(np.prod(shape))).reshape(shape))
    encoded = _encode(value, None, "A")
    assert set(encoded) == {"DM"}
    loaded = _decode(encoded, None)
    assert loaded.shape == shape
    np.testing.assert_array_equal(loaded.full(), value.full())


@pytest.mark.parametrize("name", LOAD_CASES)
def test_from_yaml_subclass_args(name):
    """Written by hand, as for load_model()"""
    text, make_model = LOAD_CASES[name]
    expected = make_model()
    text = "cas_models_format: 1\n" + text
    assert_same_functions(from_yaml(text), expected)
    assert_same_functions(type(expected).from_yaml(text), expected)


@pytest.mark.parametrize("name", MODEL_FILES)
def test_from_yaml_model_file_text(name):
    """The text of a saved model.yaml, with the folder of its files"""
    folder = MODELS_DIR / name
    text = (folder / "model.yaml").read_text()
    expected = MODEL_FILES[name]()
    model = from_yaml(text, folder=folder)
    if name in SAVED_AS_BASE:
        assert_same_model(expected, model, SAVED_AS_BASE[name])
    else:
        assert_same_functions(model, expected)
    if _file_refs(yaml.safe_load(text)):
        with pytest.raises(ValueError, match="give the folder it is in"):
            from_yaml(text)


def test_from_yaml_errors():
    with pytest.raises(ValueError, match="cas_models_format None, not 1"):
        from_yaml("class: SSModelCTLinearFOSISO\n")
    with pytest.raises(ValueError, match="cas_models_format 2, not 1"):
        from_yaml("cas_models_format: 2\nclass: SSModelCTLinearFOSISO\n")
    text = SSModelCTLinearFOSISO().to_yaml()
    with pytest.raises(ValueError, match="isn't StateSpaceModelDT"):
        StateSpaceModelDT.from_yaml(text)


# Models built by connection functions (function: ..., args: ...)


def arx(na, nb, name, **kwargs):
    return StateSpaceModelDTARXSISO(na=na, nb=nb, name=name, **kwargs)


ARX_NODES = (
    "  - class: StateSpaceModelDTARXSISO\n"
    "    args: {na: 2, nb: 2, dt: 2.0, name: CRAC1}\n"
    "  - class: StateSpaceModelDTARXSISO\n"
    "    args: {na: 1, nb: 1, dt: 2.0, name: CRAC2}\n"
)


def arx_pair():
    return [arx(2, 2, "CRAC1", dt=2.0), arx(1, 1, "CRAC2", dt=2.0)]


# (YAML text without the cas_models_format line, the model built in code)
FUNCTION_CASES = {
    "block_diag_systems": (
        "function: block_diag_systems\nargs:\n  systems:\n" + ARX_NODES,
        lambda: block_diag_systems(arx_pair(), StateSpaceModelDT),
    ),
    "hstack_systems": (
        "function: hstack_systems\nargs:\n  name: R11\n  systems:\n"
        + ARX_NODES,
        lambda: hstack_systems(arx_pair(), StateSpaceModelDT, name="R11"),
    ),
    "vstack_systems": (
        "function: vstack_systems\n"
        "args:\n"
        "  systems:\n"
        "  - {class: SSModelCTLinearFOSISO, args: {K: 2.0, name: a}}\n"
        "  - {class: SSModelCTLinearO2SISO, args: {name: b}}\n",
        lambda: vstack_systems(
            [SSModelCTLinearFOSISO(K=2.0, name="a"), SSModelCTLinearO2SISO(name="b")],
            StateSpaceModelCT,
        ),
    ),
    "connect_systems_in_parallel": (
        "function: connect_systems_in_parallel\n"
        "args:\n"
        "  systems:\n"
        "  - {class: SSModelCTLinearFOSISO, args: {name: a}}\n"
        "  - {class: SSModelCTLinearFOSISO, args: {name: b}}\n"
        "  keys: [a, b]\n"
        "  verbose_names: true\n",
        lambda: connect_systems_in_parallel(
            [SSModelCTLinearFOSISO(name="a"), SSModelCTLinearFOSISO(name="b")],
            StateSpaceModelCT,
            keys=["a", "b"],
            verbose_names=True,
        ),
    ),
    "connect_systems_in_series": (
        "function: connect_systems_in_series\n"
        "args:\n"
        "  systems:\n"
        "  - {class: SSModelCTPIInt, args: {Kc: 0.5, Ti: 30.0, name: ctrl}}\n"
        "  - {class: SSModelCTLinearFOSISO, args: {K: 1, T1: 2, name: plant}}\n",
        lambda: connect_systems_in_series(
            [
                SSModelCTPIInt(Kc=0.5, Ti=30.0, name="ctrl"),
                SSModelCTLinearFOSISO(K=1, T1=2, name="plant"),
            ],
            StateSpaceModelCT,
        ),
    ),
    "connect_systems": (
        "function: connect_systems\n"
        "args:\n"
        "  systems:\n"
        "  - class: StateSpaceModelDTARXSISO\n"
        "    args: {na: 2, nb: 2, input_name: u, output_name: y}\n"
        "  - class: StateSpaceModelDTARXSISO\n"
        "    args: {na: 1, nb: 1, input_name: u, output_name: y}\n"
        "  connections:\n"
        "    CRAC1_u: CRAC1\n"
        "    CRAC2_u: CRAC2\n"
        "    R11: [CRAC1_y, CRAC2_y]\n"
        "  input_names: [CRAC1, CRAC2]\n"
        "  output_names: [R11]\n"
        "  keys: [CRAC1, CRAC2]\n"
        "  verbose_names: true\n"
        "  name: R11\n",
        dt_connected_arx,
    ),
    "connect_feedback_system": (
        "function: connect_feedback_system\n"
        "args:\n"
        "  sys1:\n"
        "    function: connect_systems_in_series\n"
        "    args:\n"
        "      systems:\n"
        "      - {class: SSModelCTPIInt, args: {name: ctrl}}\n"
        "      - {class: SSModelCTLinearFOSISO, args: {K: 1, T1: 2, name: plant}}\n",
        ct_pi_feedback,
    ),
    "negate_system": (
        "function: negate_system\n"
        "args:\n"
        "  sys: {class: SSModelCTLinearFOSISO, args: {K: 2.0, name: a}}\n",
        lambda: -SSModelCTLinearFOSISO(K=2.0, name="a"),
    ),
    "nested": (
        "function: hstack_systems\n"
        "args:\n"
        "  systems:\n"
        "  - function: vstack_systems\n"
        "    args:\n"
        "      systems:\n"
        "      - {class: SSModelCTLinearFOSISO, args: {name: a}}\n"
        "      - {class: SSModelCTLinearFOSISO, args: {name: b}}\n"
        "  - function: block_diag_systems\n"
        "    args:\n"
        "      systems:\n"
        "      - {class: SSModelCTLinearFOSISO, args: {name: c}}\n"
        "      - {function: negate_system, args: {sys: {class: SSModelCTLinearFOSISO, args: {name: d}}}}\n",
        lambda: hstack_systems(
            [
                vstack_systems(
                    [SSModelCTLinearFOSISO(name="a"), SSModelCTLinearFOSISO(name="b")],
                    StateSpaceModelCT,
                ),
                block_diag_systems(
                    [
                        SSModelCTLinearFOSISO(name="c"),
                        -SSModelCTLinearFOSISO(name="d"),
                    ],
                    StateSpaceModelCT,
                ),
            ],
            StateSpaceModelCT,
        ),
    ),
    "model_class given": (
        "function: hstack_systems\n"
        "args:\n  model_class: StateSpaceModelDT\n  systems:\n" + ARX_NODES,
        lambda: hstack_systems(arx_pair(), StateSpaceModelDT),
    ),
    "connection of a class node with a file-free nested model": (
        "function: connect_systems_in_parallel\n"
        "args:\n"
        "  systems:\n"
        "  - class: StateSpaceModelDTFromCTRK4\n"
        "    args:\n"
        "      model_ct: {class: SSModelCTLinearFOSISO, args: {K: 2.0}}\n"
        "      dt: 1.0\n"
        "  - class: StateSpaceModelDTTFSISO\n"
        "    args: {num: [0.2], den: [1, -0.8], dt: 1.0}\n",
        lambda: connect_systems_in_parallel(
            [
                StateSpaceModelDTFromCTRK4(SSModelCTLinearFOSISO(K=2.0), 1.0),
                StateSpaceModelDTTFSISO(num=[0.2], den=[1, -0.8], dt=1.0),
            ],
            StateSpaceModelDT,
        ),
    ),
}


def test_function_cases_cover_all_functions():
    from cas_models.yaml_utils import CONNECTION_FUNCTIONS

    assert set(CONNECTION_FUNCTIONS) <= set(FUNCTION_CASES)


def assert_same_connected(model, expected):
    """The same class, functions (exactly), parameters, dimensions,
    names and time step"""
    assert_same_functions(model, expected)
    for attr in (
        "n",
        "nu",
        "ny",
        "name",
        "input_names",
        "state_names",
        "output_names",
    ):
        assert getattr(model, attr) == getattr(expected, attr), attr
    assert getattr(model, "dt", None) == getattr(expected, "dt", None)


@pytest.mark.parametrize("name", FUNCTION_CASES)
def test_from_yaml_function(name, tmp_path):
    """A function node builds the model the function builds, from
    from_yaml(), the base class's from_yaml() and load_model()"""
    text, make_model = FUNCTION_CASES[name]
    expected = make_model()
    base = (
        StateSpaceModelCT
        if isinstance(expected, StateSpaceModelCT)
        else StateSpaceModelDT
    )
    text = "cas_models_format: 1\n" + text
    assert_same_connected(from_yaml(text), expected)
    assert_same_connected(base.from_yaml(text), expected)
    write_model_yaml(tmp_path / "model", text.split("\n", 1)[1])
    assert_same_connected(load_model(tmp_path / "model"), expected)
    assert_same_connected(base.load(tmp_path / "model"), expected)


def test_from_yaml_function_with_data_file():
    """Nested class nodes can refer to data files (given the folder)"""
    folder = MODELS_DIR / "ct_abcd_6_states"
    text = (
        "cas_models_format: 1\n"
        "function: connect_systems_in_parallel\n"
        "args:\n"
        "  systems:\n"
        "  - " + (folder / "model.yaml").read_text().split("\n", 1)[1]
        .replace("\n", "\n    ").rstrip()
        + "\n  - class: StateSpaceModelCTFromABCD\n"
        "    args: {A: {file: A.tsv, shape: [6, 6]}, B: {DM: [1, 0, 0, 0, 0, 0]},"
        " C: {DM: [[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]]}, D: 0.0}\n"
    )
    model = from_yaml(text, folder=folder)
    assert (model.n, model.nu, model.ny) == (12, 1, 1)
    with pytest.raises(ValueError, match="give the folder it is in"):
        from_yaml(text)


FUNCTION_ERRORS = {
    "unknown function": (
        "function: no_such_function\nargs: {systems: []}\n",
        "'no_such_function' isn't one of block_diag_systems",
    ),
    "format in a nested model": (
        "function: hstack_systems\n"
        "args:\n"
        "  systems:\n"
        "  - cas_models_format: 1\n"
        "    class: SSModelCTLinearFOSISO\n",
        "a model has only class and args, not cas_models_format",
    ),
    "class and function": (
        "function: hstack_systems\nclass: SSModelCTLinearFOSISO\n",
        "a class or a function, not both",
    ),
    "positional args": (
        "function: hstack_systems\nargs: [1, 2]\n",
        "positional arguments aren't supported",
    ),
    "no model": ("args: {K: 1.0}\n", "no model"),
    "no models to infer model_class": (
        "function: block_diag_systems\nargs: {systems: []}\n",
        "Cannot validate empty system list",
    ),
    "unknown model_class": (
        "function: hstack_systems\n"
        "args:\n"
        "  model_class: NoSuchClass\n"
        "  systems: [{class: SSModelCTLinearFOSISO}]\n",
        "'NoSuchClass' isn't",
    ),
}


@pytest.mark.parametrize("name", FUNCTION_ERRORS)
def test_from_yaml_function_errors(name):
    text, match = FUNCTION_ERRORS[name]
    with pytest.raises(ValueError, match=match):
        from_yaml("cas_models_format: 1\n" + text)


def test_from_yaml_function_wrong_base():
    """With a base class, the function's model must be one"""
    text = (
        "cas_models_format: 1\nfunction: hstack_systems\n"
        "args:\n  systems:\n" + ARX_NODES
    )
    with pytest.raises(ValueError, match="makes a StateSpaceModelDT, which isn't"):
        StateSpaceModelCT.from_yaml(text)
    with pytest.raises(ValueError, match="isn't StateSpaceModelDTARXSISO"):
        StateSpaceModelDTARXSISO.from_yaml(text)


def test_from_yaml_function_mixed_ct_dt():
    """The function's own checks apply (CT and DT can't be combined)"""
    text = (
        "cas_models_format: 1\nfunction: hstack_systems\nargs:\n"
        "  systems:\n"
        "  - {class: SSModelCTLinearFOSISO}\n"
        "  - {class: StateSpaceModelDTARXSISO, args: {na: 1, nb: 1}}\n"
    )
    with pytest.raises(ValueError, match="Cannot combine systems with different types"):
        from_yaml(text)


# Symbols: {symbol: name} and {symbol: name, shape: [...]}

H = "cas_models_format: 1\n"

SHARED = (
    "function: vstack_systems\n"
    "args:\n"
    "  systems:\n"
    "  - class: SSModelCTLinearO2SISO\n"
    "    args: {K: {symbol: K}, T1: {symbol: T1}, T2: 5.0, name: a}\n"
    "  - class: SSModelCTLinearFOSISO\n"
    "    args: {K: {symbol: K}, T1: {symbol: T1}, name: b}\n"
)


def test_symbol_shared_parameters(tmp_path):
    """The same name is the same symbol: one parameter for both models,
    as when the models are built with the same symbols in code"""
    model = from_yaml(H + SHARED)
    assert list(model.params) == ["K", "T1"]
    K, T1 = cas.SX.sym("K"), cas.SX.sym("T1")
    expected = vstack_systems(
        [
            SSModelCTLinearO2SISO(K=K, T1=T1, T2=5.0, name="a"),
            SSModelCTLinearFOSISO(K=K, T1=T1, name="b"),
        ]
    )
    assert_same_connected(model, expected)
    write_model_yaml(tmp_path / "model", SHARED)
    assert_same_connected(load_model(tmp_path / "model"), expected)


def test_symbol_shared_verbose_names():
    """With verbose_names, a shared parameter names all its systems"""
    text = SHARED.replace("args:\n  systems:", "args:\n  verbose_names: true\n  systems:")
    model = from_yaml(H + text)
    assert list(model.params) == ["a_b_K", "a_b_T1"]


def test_symbol_shared_across_nested_functions():
    text = (
        "function: connect_systems_in_parallel\n"
        "args:\n"
        "  systems:\n"
        "  - function: connect_systems_in_series\n"
        "    args:\n"
        "      systems:\n"
        "      - {class: SSModelCTLinearFOSISO, args: {K: {symbol: K}, T1: 1.0}}\n"
        "      - {class: SSModelCTLinearFOSISO, args: {K: 2.0, T1: {symbol: T}}}\n"
        "  - {class: SSModelCTLinearFOSISO, args: {K: {symbol: K}, T1: {symbol: T}}}\n"
    )
    model = from_yaml(H + text)
    assert sorted(model.params) == ["K", "T"]


def test_symbol_separate_models_get_separate_symbols():
    """Without symbols (or across documents), parameters aren't shared"""
    text = SHARED.replace("{symbol: K}", "null").replace("{symbol: T1}", "null")
    model = from_yaml(H + text)
    assert len(model.params) == 4  # renamed apart: a_K, b_K, ...
    m1, m2 = from_yaml(H + SHARED), from_yaml(H + SHARED)
    assert m1.params["K"] is not m2.params["K"]


@pytest.mark.parametrize(
    "args, params",
    [
        # A scalar (the default shape)
        ("{poles: [{symbol: p}], nb: 1}", ["p", "b"]),
        # A vector: [n] is a column
        ("{poles: {symbol: p, shape: [2]}, nb: 1}", ["p_0", "p_1", "b"]),
        ("{poles: {symbol: p, shape: [2, 1]}, nb: 1}", ["p_0", "p_1", "b"]),
        # In a list, with numbers
        ("{poles: [{symbol: p_0}, 0.5], B: [{symbol: g}, 0.1]}", ["p_0", "g"]),
        # The same symbol twice in one model
        ("{poles: [{symbol: p}, {symbol: p}], nb: 1}", ["p", "b"]),
    ],
)
def test_symbol_arx(args, params):
    model = from_yaml(H + f"class: StateSpaceModelDTARXSISO\nargs: {args}\n")
    assert list(model.params) == params


def test_symbol_vector_matches_code():
    model = from_yaml(
        H + "class: StateSpaceModelDTARXSISO\n"
        "args: {poles: {symbol: p, shape: [2]}, zeros: [{symbol: z}], gain: 0.5}\n"
    )
    expected = StateSpaceModelDTARXSISO(
        poles=cas.SX.sym("p", 2), zeros=[cas.SX.sym("z")], gain=0.5
    )
    assert_same_functions(model, expected)


def test_symbol_matrices():
    model = from_yaml(
        H + "class: StateSpaceModelCTFromABCD\n"
        "args:\n"
        "  A: {symbol: A, shape: [2, 2]}\n"
        "  B: [[1.0], [0.0]]\n"
        "  C: [[0.0, 1.0]]\n"
        "  D: 0.0\n"
    )
    assert sorted(model.params) == ["A_0", "A_1", "A_2", "A_3"]
    model = from_yaml(
        H + "class: StateSpaceModelDTDelay\n"
        "args: {nk: 2, nu: 2, G: [[1.0, 0.0], [0.0, {symbol: g}]]}\n"
    )
    assert list(model.params) == ["g"]


SYMBOL_ERRORS = {
    "zero size": ("{symbol: p, shape: [0]}", "positive integers"),
    "negative": ("{symbol: p, shape: [-1]}", "positive integers"),
    "three dimensions": ("{symbol: p, shape: [2, 2, 2]}", "one or two"),
    "not a list": ("{symbol: p, shape: 2}", "a list of one or two"),
    "empty": ("{symbol: p, shape: []}", "a list of one or two"),
    "bool": ("{symbol: p, shape: [true]}", "positive integers"),
    "extra key": ("{symbol: p, value: 2}", "only symbol and shape, not value"),
    "not an identifier": ("{symbol: 'p 1'}", "valid identifier"),
}


@pytest.mark.parametrize("name", SYMBOL_ERRORS)
def test_symbol_errors(name):
    node, match = SYMBOL_ERRORS[name]
    with pytest.raises(ValueError, match=match):
        from_yaml(H + f"class: StateSpaceModelDTARXSISO\nargs: {{poles: {node}, nb: 1}}\n")


def test_symbol_shape_must_agree():
    text = (
        "class: StateSpaceModelDTARXSISO\n"
        "args: {poles: {symbol: p, shape: [2]}, B: [{symbol: p}]}\n"
    )
    with pytest.raises(ValueError, match=r"symbol p: shape \[1, 1\] here, \[2, 1\]"):
        from_yaml(H + text)


def test_symbol_vector_for_a_scalar_argument():
    """The class's own checks apply"""
    text = "class: SSModelCTLinearFOSISO\nargs: {K: {symbol: K, shape: [2]}}\n"
    with pytest.raises(ValueError, match="K: give a scalar, not 2 values"):
        from_yaml(H + text)
