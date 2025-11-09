"""Test that ARX model matches Octave/Matlab implementation"""

import pytest
import numpy as np
import casadi as cas
from cas_models.discrete_time.models import StateSpaceModelDTARXSISO


def test_ARX221_observable_canonical_form():
    """Test that ARX(2,2,1) model matches Octave output

    From Octave arx() function with na=2, nb=2, nk=1:

    Transfer function:
           0.1867 z^-2 + 0.1654 z^-3
    y1 = ----------------------------
          1 - 1.404 z^-1 + 0.6977 z^-2

    State-space (observable canonical form):
    A = [0       1       0     ]
        [0       0      -1     ]
        [0    0.6977   1.404  ]

    B = [0.1654]
        [0.1867]
        [0     ]

    C = [0  0  -1]
    D = [0]
    """
    # ARX coefficients from Octave output (high precision)
    # A(q^-1) = 1 - 1.40429502*q^-1 + 0.69767633*q^-2
    # B(q^-1) = 0.18669536 + 0.16536220*q^-1 (before delay)
    A_coeffs = cas.DM([-1.40429502, 0.69767633])
    B_coeffs = cas.DM([0.18669536, 0.16536220])
    na = 2
    nb = 2
    nk = 1

    # Create model
    model = StateSpaceModelDTARXSISO(A=A_coeffs, B=B_coeffs, nk=nk)

    # Expected dimensions (observable canonical form)
    expected_n = max(na, nb + nk)  # max(2, 3) = 3
    assert model.n == expected_n
    assert model.na == na
    assert model.nb == nb
    assert model.nk == nk

    # Expected matrices from Octave (correctly oriented based on labels)
    # From sys_ss.a with row/column labels:
    #        x1   x2      x3
    #   x1   0    0       0
    #   x2   1    0       0.6977
    #   x3   0   -1       1.404
    A_octave = np.array([
        [0.0,         0.0,         0.0],
        [1.0,         0.0,  0.69767633],
        [0.0,        -1.0,  1.40429502]
    ])

    B_octave = np.array([
        [0.16536220],
        [0.18669536],
        [0.0]
    ])

    C_octave = np.array([[0.0, 0.0, -1.0]])
    D_octave = np.array([[0.0]])

    # Extract matrices from model by evaluating state transition
    t = cas.DM(0)
    xk = cas.SX.sym("xk", 3)
    uk = cas.SX.sym("uk")

    # Get F function result
    xkp1_expr = model.F(t, xk, uk)

    # Extract A matrix: xkp1 when uk=0
    A_extracted = cas.jacobian(xkp1_expr, xk)
    A_numerical = cas.DM(cas.substitute(A_extracted, uk, 0))

    # Extract B matrix: d(xkp1)/d(uk)
    B_extracted = cas.jacobian(xkp1_expr, uk)
    B_numerical = cas.DM(B_extracted)

    # Get H function result
    yk_expr = model.H(t, xk, uk)

    # Extract C matrix: y when uk=0
    C_extracted = cas.jacobian(yk_expr, xk)
    C_numerical = cas.DM(cas.substitute(C_extracted, uk, 0))

    # Extract D matrix: dy/d(uk)
    D_extracted = cas.jacobian(yk_expr, uk)
    D_numerical = cas.DM(D_extracted)

    # Compare matrices
    print("\nA matrix comparison:")
    print("Model:\n", A_numerical)
    print("Octave:\n", A_octave)
    assert np.allclose(A_numerical, A_octave, atol=1e-4)

    print("\nB matrix comparison:")
    print("Model:\n", B_numerical)
    print("Octave:\n", B_octave)
    assert np.allclose(B_numerical, B_octave, atol=1e-4)

    print("\nC matrix comparison:")
    print("Model:\n", C_numerical)
    print("Octave:\n", C_octave)
    assert np.allclose(C_numerical, C_octave, atol=1e-4)

    print("\nD matrix comparison:")
    print("Model:\n", D_numerical)
    print("Octave:\n", D_octave)
    assert np.allclose(D_numerical, D_octave, atol=1e-4)


if __name__ == "__main__":
    test_ARX221_observable_canonical_form()
    print("\nAll tests passed!")
