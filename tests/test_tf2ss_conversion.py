"""Test tf2ss conversion functions match Octave/Matlab implementation"""

import pytest
import numpy as np
import casadi as cas
from cas_models.discrete_time.models import (
    tf_to_ss_oct_np,
    tf_to_ss_oct_cas,
)


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
