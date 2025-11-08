"""Unit tests for src/cas_models/continuous_time/models.py module"""

import pytest
import numpy as np
import casadi as cas
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
)


@pytest.fixture
def symbolic_FO_SISO():
    """Example SISO system 1: 1st order"""
    # Dimensions
    n = 1
    nu = 1
    ny = 1

    # Parameters (symbolic)
    a1 = cas.SX.sym("a1")
    b0 = cas.SX.sym("b0")
    params = {"a1": a1, "b0": b0}

    # State space model matrices
    A = -a1
    B = b0
    C = 1
    D = 0

    # Construct ODE right-hand side
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", n)
    uk = cas.SX.sym("uk")
    xkp1 = -a1 * xk + uk

    F = cas.Function(
        "F",
        [t, xk, uk, *params.values()],
        [xkp1],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )

    y = xk
    H = cas.Function(
        "H",
        [t, xk, uk, *params.values()],
        [y],
        ["t", "xk", "uk", *params.keys()],
        ["yk"],
    )

    return n, nu, ny, A, B, C, D, t, F, H, params


@pytest.fixture
def symbolic_AR211_SISO():
    """Example SISO system 2: AR(2,1,1) model

    y(k) + a1*y(k-1) + a2*y(k-2) = b1*u(k-1)

    """
    # Dimensions
    na = 2
    nb = 1
    nk = 1

    # Parameters (symbolic)
    Aq = cas.SX.sym("Aq", na)
    Bq = cas.SX.sym("Bq", nb + 1)
    params = {"Aq": Aq, "Bq": Bq}

    # State space model matrices (controllable canonical form)
    A = cas.sparsify(
        cas.vertcat(
            cas.horzcat(-Aq[0], -Aq[1], Bq[0], Bq[1]),
            cas.horzcat(1, 0, 0, 0),
            cas.horzcat(0, 1, 0, 0),
            cas.horzcat(0, 0, 1, 0),
        )
    )
    B = cas.sparsify(cas.vertcat(0, 0, 0, 1))
    C = cas.sparsify(cas.horzcat(1, 0, 0, 0))
    D = cas.sparsify(0)

    n = A.shape[0]
    assert A.shape[1] == n
    nu = B.shape[1]
    ny = C.shape[0]
    assert D.shape == (ny, nu)

    # Construct ODE right-hand side
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", n)
    uk = cas.SX.sym("uk")
    xkp1 = A @ xk + B @ uk

    F = cas.Function(
        "F",
        [t, xk, uk, *params.values()],
        [xkp1],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )

    yk = C @ xk + D @ uk
    H = cas.Function(
        "H",
        [t, xk, uk, *params.values()],
        [yk],
        ["t", "xk", "uk", *params.keys()],
        ["yk"],
    )

    return n, nu, ny, A, B, C, D, t, F, H, params


def test_StateSpaceModelCT_FO_SISO(symbolic_FO_SISO):
    n, nu, ny, A, B, C, D, t, F, H, params = symbolic_FO_SISO

    # nu and ny should be 1 by default - TODO: Use SISO version
    model = StateSpaceModelDT(F, H, n, nu, ny, params=params)

    assert str(model) == (
        "StateSpaceModelDT("
        "F=Function(F:(t,xk,uk,a1,b0)->(xkp1) SXFunction), "
        "H=Function(H:(t,xk,uk,a1,b0)->(yk) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'a1': SX(a1), 'b0': SX(b0)}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    assert float(model.F(0.0, 0.0, 0.0, 0.2, 0.8)) == 0.0
    assert float(model.H(0.0, 0.0, 0.0, 0.2, 0.8)) == 0.0
    assert float(model.F(0.0, 1.0, 0.0, 0.2, 0.8)) == -0.2
    assert float(model.H(0.0, 1.0, 0.0, 0.2, 0.8)) == 1.0


def test_symbolic_AR211_SISO(symbolic_AR211_SISO):
    n, nu, ny, A, B, C, D, t, F, H, params = symbolic_AR211_SISO

    # nu and ny should be 1 by default - TODO: Use SISO version
    model = StateSpaceModelDT(F, H, n, nu, ny, params=params)

    assert str(model) == (
        "StateSpaceModelDT(F=Function("
        "F:(t,xk[4],uk,Aq[2],Bq[2])->(xkp1[4]) SXFunction), "
        "H=Function(H:(t,xk[4],uk,Aq[2],Bq[2])->(yk) SXFunction), "
        "n=4, nu=1, ny=1, "
        "params={'Aq': SX([Aq_0, Aq_1]), 'Bq': SX([Bq_0, Bq_1])}, "
        "input_names=['u'], state_names=['x1', 'x2', 'x3', 'x4'], output_names=['y']"
        ")"
    )

    # Test function calls
    t = cas.DM(0)
    xk = cas.DM([0.0, 0.0, 0.0, 0.0])
    uk = cas.DM([-0.5])
    Aq = cas.DM([0.6, 0.1])
    Bq = cas.DM([0.8, 0.2])
    assert np.allclose(model.F(t, xk, uk, Aq, Bq), cas.DM([0, 0, 0, -0.5]))
    assert np.allclose(model.H(t, xk, uk, Aq, Bq), cas.DM([0]))


# def test_StateSpaceModelCTFromABCD_FO_SISO(symbolic_FO_SISO):
#     _, _, _, A, B, C, D, _, _, _, _ = symbolic_FO_SISO

#     model = StateSpaceModelDTFromABCD(A, B, C, D)

#     assert str(model) == (
#         "StateSpaceModelDTFromABCD("
#         "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
#         "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
#         "n=1, nu=1, ny=1, "
#         "params={'K': SX(K), 'T1': SX(T1)}, "
#         "input_names=['u'], state_names=['x'], output_names=['y']"
#         ")"
#     )

#     # Test function calls - with scalars
#     assert np.array_equal(model.F(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
#     assert np.array_equal(model.H(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
#     assert np.array_equal(model.F(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[-0.5]]))
#     assert np.array_equal(model.H(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[0.5]]))


# def test_StateSpaceModelCTFromABCD_O2_SISO(symbolic_O2_SISO):
#     _, _, _, A, B, C, D, _, _, _, _ = symbolic_O2_SISO

#     model = StateSpaceModelCTFromABCD(A, B, C, D)

#     assert str(model) == (
#         "StateSpaceModelCTFromABCD("
#         "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
#         "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
#         "n=2, nu=1, ny=1, "
#         "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, "
#         "input_names=['u'], state_names=['x1', 'x2'], output_names=['y'])"
#     )

#     # Test function calls
#     t = cas.DM(0)
#     x = cas.DM([0.0, 1.0])
#     u = cas.DM([-0.5])
#     K = cas.DM(0.8)
#     T1 = cas.DM(0.6)
#     T2 = cas.DM(3.0)
#     assert np.allclose(model.F(t, x, u, K, T1, T2), cas.DM([1, -2.5]))
#     assert np.allclose(model.H(t, x, u, K, T1, T2), cas.DM(0))

#     T1 = T2
#     x = cas.DM([3.0, 1.0])
#     assert np.allclose(model.F(t, x, u, K, T1, T2), cas.DM([1, -1.5]))
#     assert np.allclose(
#         model.H(t, x, u, K, T1, T2), cas.DM(0.26666666666666666)
#     )
