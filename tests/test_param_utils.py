"""Unit tests for src/cas_models/param_utils.py module"""

import pytest
import casadi as cas
from casadi import SX
from cas_models.param_utils import (
    make_list_of_enumerated_names,
    concatenate_lists_of_names,
    make_list_of_unique_names,
    merge_param_dicts,
    make_symbolic_vars_from_kwargs,
)


def test_make_list_of_enumerated_names():
    names = make_list_of_enumerated_names("x", 3)
    assert names == ["x1", "x2", "x3"]

    names = make_list_of_enumerated_names("sys", 2, sep="_")
    assert names == ["sys_1", "sys_2"]


def test_make_list_of_unique_names():
    keys = [None, None, None]
    names = make_list_of_unique_names(keys)
    assert names == ["sys1", "sys2", "sys3"]

    keys = [None, None, "out"]
    names = make_list_of_unique_names(keys)
    assert names == ["sys1", "sys2", "out"]

    keys = make_list_of_enumerated_names("sys", 3, sep="_")
    names = make_list_of_unique_names(keys)
    assert names == ["sys_1", "sys_2", "sys_3"]

    keys = [None, None, "sys1"]
    names = make_list_of_unique_names(keys)
    assert names == ["sys2", "sys3", "sys1"]

    # Duplicate strings are numbered
    keys = ["motor", "motor"]
    names = make_list_of_unique_names(keys)
    assert names == ["motor1", "motor2"]

    # Only the duplicated name gets numbered; unique names are unchanged
    keys = ["plant", "motor", "motor"]
    names = make_list_of_unique_names(keys)
    assert names == ["plant", "motor1", "motor2"]

    # Mixed: duplicate strings alongside None values
    keys = ["motor", None, "motor"]
    names = make_list_of_unique_names(keys)
    assert names == ["motor1", "sys1", "motor2"]


def test_concatenate_lists_of_names():
    # Test 1: Default behavior (verbose_names=False) with conflicts
    sys1_param_names = ["K", "T1", "T2"]
    sys2_param_names = ["K", "T1", "theta"]
    names = concatenate_lists_of_names([sys1_param_names, sys2_param_names])
    # K and T1 conflict, so both get prefixed. T2 and theta are unique.
    assert names == ["sys1_K", "sys1_T1", "T2", "sys2_K", "sys2_T1", "theta"]

    # Test 2: With custom system names (verbose_names=False)
    names = concatenate_lists_of_names(
        [sys1_param_names, sys2_param_names], keys=["G1", "G2"]
    )
    assert names == ["G1_K", "G1_T1", "T2", "G2_K", "G2_T1", "theta"]

    # Test 3: With verbose_names=True (always prepend keys)
    names = concatenate_lists_of_names(
        [sys1_param_names, sys2_param_names], verbose_names=True
    )
    assert names == [
        "sys1_K",
        "sys1_T1",
        "sys1_T2",
        "sys2_K",
        "sys2_T1",
        "sys2_theta",
    ]

    # Test 4: With custom keys and verbose_names=True
    names = concatenate_lists_of_names(
        [sys1_param_names, sys2_param_names],
        keys=["G1", "G2"],
        verbose_names=True,
    )
    assert names == ["G1_K", "G1_T1", "G1_T2", "G2_K", "G2_T1", "G2_theta"]

    # Test 5: No conflicts - all names should be kept as-is (verbose_names=False)
    sys1_names = ["a", "b", "c"]
    sys2_names = ["d", "e", "f"]
    names = concatenate_lists_of_names(
        [sys1_names, sys2_names], keys=["x", "y"]
    )
    assert names == ["a", "b", "c", "d", "e", "f"]

    # Test 6: Same test with verbose_names=True - all get prefixed
    names = concatenate_lists_of_names(
        [sys1_names, sys2_names], keys=["x", "y"], verbose_names=True
    )
    assert names == ["x_a", "x_b", "x_c", "y_d", "y_e", "y_f"]

    # Test 7: Partial conflicts - only conflicting names get prefixed
    sys1_names = ["u1", "x", "y1"]
    sys2_names = ["u2", "x", "y2"]
    names = concatenate_lists_of_names(
        [sys1_names, sys2_names], keys=["s1", "s2"]
    )
    # Only 'x' conflicts
    assert names == ["u1", "s1_x", "y1", "u2", "s2_x", "y2"]

    # Test 8: Three systems with various conflicts
    sys1_names = ["a", "b"]
    sys2_names = ["b", "c"]
    sys3_names = ["c", "d"]
    names = concatenate_lists_of_names(
        [sys1_names, sys2_names, sys3_names], keys=["x", "y", "z"]
    )
    # 'a' is unique, 'b' conflicts (x,y), 'c' conflicts (y,z), 'd' is unique
    assert names == ["a", "x_b", "y_b", "y_c", "z_c", "d"]

    # Test 9: Error - duplicate keys
    with pytest.raises(ValueError, match="not enough unique keys"):
        concatenate_lists_of_names([sys1_names, sys2_names], keys=["x", "x"])

    # Test 10: Error - not enough keys
    with pytest.raises(ValueError, match="not enough unique keys"):
        concatenate_lists_of_names(
            [sys1_names, sys2_names, sys3_names], keys=["x", "y"]
        )


def test_merge_param_dicts():
    # Example 1 - with minimal param names
    K = SX.sym("K")
    T1_1 = SX.sym("T1_1")
    T1_2 = SX.sym("T1_2")
    T2_2 = SX.sym("T2_2")
    params1 = {"K": K, "T1": T1_1}
    params2 = {"K": K, "T1": T1_2, "T2": T2_2}
    result = merge_param_dicts([params1, params2], ["sys1", "sys2"])
    assert result == {
        "K": K,
        "sys1_T1": T1_1,
        "sys2_T1": T1_2,
        "T2": T2_2,
    }

    # With verbose param names
    result = merge_param_dicts(
        [params1, params2], ["sys1", "sys2"], verbose_names=True
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
    assert params["T1"].shape == (1, 1)
    assert isinstance(params["T2"], cas.SX)
    assert params["T2"].name() == "T2"
    assert params["T2"].shape == (1, 1)

    K = cas.SX.sym("K")
    T1 = cas.SX.sym("T1")
    params = make_symbolic_vars_from_kwargs(K=K, T1=T1)
    assert params == {"K": K, "T1": T1}

    params = make_symbolic_vars_from_kwargs(A=(3, 2), B=(3, 1))
    assert isinstance(params["A"], cas.SX)
    assert params["A"].shape == (3, 2)
    assert isinstance(params["B"], cas.SX)
    assert params["B"].shape == (3, 1)
