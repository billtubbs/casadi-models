"""Test that ARX model matches Octave/Matlab implementation"""

import pytest
import numpy as np
import casadi as cas
from cas_models.discrete_time.models import (
    StateSpaceModelDTARXSISO,
    tf_to_ss_oct_np,
    tf_to_ss_oct_cas,
)


@pytest.fixture
def tf_test_case_1():
    """Second order system - simple observable canonical form.

    Input: num=[0.2, 0.1], den=[1, -1.4, 0.49]
    """
    return {
        "num": np.array([0.2, 0.1]),
        "den": np.array([1, -1.4, 0.49]),
        "A": np.array([[0.0, 0.49], [1.0, -1.4]]),
        "B": np.array([[0.1], [0.2]]),
        "C": np.array([[0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_2():
    """Third order system - simple observable canonical form.

    Input: num=[0.05, 0.1, 0.05], den=[1, -1.5, 0.7, -0.1]
    """
    return {
        "num": np.array([0.05, 0.1, 0.05]),
        "den": np.array([1, -1.5, 0.7, -0.1]),
        "A": np.array([[0.0, 0.0, -0.1], [1.0, 0.0, 0.7], [0.0, 1.0, -1.5]]),
        "B": np.array([[0.05], [0.1], [0.05]]),
        "C": np.array([[0.0, 0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_3():
    """Fifth order system - simple observable canonical form.

    Input: num=[0.01, 0.03, 0.03, 0.02, 0.01],
           den=[1, -2.3, 2.6, -1.8, 0.72, -0.12]
    """
    return {
        "num": np.array([0.01, 0.03, 0.03, 0.02, 0.01]),
        "den": np.array([1, -2.3, 2.6, -1.8, 0.72, -0.12]),
        "A": np.array([
            [0.0, 0.0, 0.0, 0.0, -0.12],
            [1.0, 0.0, 0.0, 0.0, 0.72],
            [0.0, 1.0, 0.0, 0.0, -1.8],
            [0.0, 0.0, 1.0, 0.0, 2.6],
            [0.0, 0.0, 0.0, 1.0, -2.3]
        ]),
        "B": np.array([[0.01], [0.02], [0.03], [0.03], [0.01]]),
        "C": np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_4():
    """First order system - simple observable canonical form.

    Input: num=[1], den=[1, -1]
    """
    return {
        "num": np.array([1]),
        "den": np.array([1, -1]),
        "A": np.array([[-1.0]]),
        "B": np.array([[1.0]]),
        "C": np.array([[1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_5():
    """Sixth order system - simple observable canonical form.

    Input: num=[0, 0, 0, 0, 0, 0, 0.1],
           den=[1.0, -4.514214, 8.884062, -9.749747, 6.204163, -2.124264, 0.3]
    """
    return {
        "num": np.array([0, 0, 0, 0, 0, 0, 0.1000]),
        "den": np.array([1.000000, -4.514214, 8.884062, -9.749747,
                        6.204163, -2.124264, 0.300000]),
        "A": np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, 0.0, 0.0, -2.124264],
            [0.0, 1.0, 0.0, 0.0, 0.0, 6.204163],
            [0.0, 0.0, 1.0, 0.0, 0.0, -9.749747],
            [0.0, 0.0, 0.0, 1.0, 0.0, 8.884062],
            [0.0, 0.0, 0.0, 0.0, 1.0, -4.514214]
        ]),
        "B": np.array([[0.1], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        "C": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.mark.parametrize(
    "test_case_fixture",
    ["tf_test_case_1", "tf_test_case_2", "tf_test_case_3",
     "tf_test_case_4", "tf_test_case_5"]
)
def test_tf_to_ss_oct_np(test_case_fixture, request):
    """Test tf_to_ss_oct_np against Octave's tf2ss output (NumPy version).

    This tests the observable canonical form implementation using NumPy arrays
    against expected state-space matrices from Octave's tf2ss function.
    """
    test_case = request.getfixturevalue(test_case_fixture)

    # Run conversion
    A, B, C, D = tf_to_ss_oct_np(test_case["num"], test_case["den"])

    # Compare matrices with tolerance
    atol = 1e-4
    assert np.allclose(A, test_case["A"], atol=atol)
    assert np.allclose(B, test_case["B"], atol=atol)
    assert np.allclose(C, test_case["C"], atol=atol)
    assert np.allclose(D, test_case["D"], atol=atol)


@pytest.mark.parametrize(
    "test_case_fixture",
    ["tf_test_case_1", "tf_test_case_2", "tf_test_case_3",
     "tf_test_case_4", "tf_test_case_5"]
)
def test_tf_to_ss_oct_cas(test_case_fixture, request):
    """Test tf_to_ss_oct_cas against Octave's tf2ss output (CasADi version).

    This tests the observable canonical form implementation using CasADi symbolic
    arrays against expected state-space matrices from Octave's tf2ss function.
    """
    test_case = request.getfixturevalue(test_case_fixture)

    # Convert to CasADi vectors
    num_cas = cas.DM(test_case["num"])
    den_cas = cas.DM(test_case["den"])

    # Run conversion
    A, B, C, D = tf_to_ss_oct_cas(num_cas, den_cas)

    # Convert to numpy for comparison
    A_np = np.array(cas.DM(A))
    B_np = np.array(cas.DM(B))
    C_np = np.array(cas.DM(C))
    D_np = np.array(cas.DM(D))

    # Compare matrices with tolerance
    atol = 1e-4
    assert np.allclose(A_np, test_case["A"], atol=atol)
    assert np.allclose(B_np, test_case["B"], atol=atol)
    assert np.allclose(C_np, test_case["C"], atol=atol)
    assert np.allclose(D_np, test_case["D"], atol=atol)
