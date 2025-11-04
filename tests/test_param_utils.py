import casadi as cas
from cas_models.param_utils import (
    make_list_of_enumerated_names,
    concatenate_lists_of_names,
    merge_param_dicts,
    make_symbolic_vars_from_kwargs,
    extract_symbolic_params,
)
from casadi import SX


def test_make_list_of_enumerated_names():
    names = make_list_of_enumerated_names("x", 3)
    assert names == ["x1", "x2", "x3"]

    names = make_list_of_enumerated_names("sys", 2, sep="_")
    assert names == ["sys_1", "sys_2"]


def test_concatenate_lists_of_names():
    # Without custom system names
    sys1_param_names = ["K", "T1", "T2"]
    sys2_param_names = ["K", "T1", "theta"]
    names = concatenate_lists_of_names([sys1_param_names, sys2_param_names])
    assert names == [
        "sys1_K",
        "sys1_T1",
        "sys1_T2",
        "sys2_K",
        "sys2_T1",
        "sys2_theta",
    ]

    # With custom system names
    names = concatenate_lists_of_names(
        [sys1_param_names, sys2_param_names], keys=["G1", "G2"]
    )
    assert names == ["G1_K", "G1_T1", "G1_T2", "G2_K", "G2_T1", "G2_theta"]


def test_merge_param_dicts():
    K = SX.sym("K")
    T1_1 = SX.sym("T1_1")
    T1_2 = SX.sym("T1_2")
    T2_2 = SX.sym("T2_2")
    params1 = {"K": K, "T1": T1_1}
    params2 = {"K": K, "T1": T1_2, "T2": T2_2}
    result = merge_param_dicts([params1, params2], keys=["sys1", "sys2"])
    assert result == {
        "K": K,
        "sys1_T1": T1_1,
        "sys2_T1": T1_2,
        "T2": T2_2,
    }

    result = merge_param_dicts(
        [params1, params2], keys=["sys1", "sys2"], verbose_names=True
    )
    assert result == {
        "sys1_sys2_K": K,
        "sys1_T1": T1_1,
        "sys2_T1": T1_2,
        "sys2_T2": T2_2,
    }


def test_make_symbolic_vars_from_kwargs():
    params = make_symbolic_vars_from_kwargs(K=2.0, T1=0.8)
    assert params == {"K": 2.0, "T1": 0.8}

    params = make_symbolic_vars_from_kwargs(K=2.0, T1=None, T2=None)
    assert params["K"] == 2.0
    assert isinstance(params["T1"], cas.SX)
    assert params["T1"].name() == "T1"
    assert isinstance(params["T2"], cas.SX)
    assert params["T2"].name() == "T2"

    K = cas.SX.sym("K")
    T1 = cas.SX.sym("T1")
    params = make_symbolic_vars_from_kwargs(K=K, T1=T1)
    assert params == {"K": K, "T1": T1}


def test_extract_symbolic_params():
    a = cas.SX.sym("a")
    b = cas.SX.sym("b")
    x = cas.MX.sym("x", 2)

    # Contains no symbolic params
    assert extract_symbolic_params({"K": 1.0, "T1": 2}) == {}

    params = {"a": a}
    symbolic_params = extract_symbolic_params(params)
    assert cas.is_equal(symbolic_params["a"], a)

    params = {"T1": 1.0, "a": a, "b": b, "expression": a * b + b**2}
    symbolic_params = extract_symbolic_params(params)
    assert list(symbolic_params.keys()) == ["a", "b"]
    assert cas.is_equal(symbolic_params["a"], a)
    assert cas.is_equal(symbolic_params["b"], b)
