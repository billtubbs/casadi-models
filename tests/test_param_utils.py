import casadi as cas
from cas_models.param_utils import (
    concatenate_lists_of_names,
    merge_param_dicts,
)
from casadi import SX


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
    breakpoint()
    assert result == {
        "sys1_sys2_K": K,
        "sys1_T1": T1_1,
        "sys2_T1": T1_2,
        "sys2_T2": T2_2,
    }
