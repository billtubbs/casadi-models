import casadi as cas
from cas_models.continuous_time.models import (
    StateSpaceModelCT,
    StateSpaceModelCTFromABCD,
)


def test_StateSpaceModelCT():
    # Example 1: SISO 1st order system
    n = 1
    K = cas.SX.sym("K")
    T1 = cas.SX.sym("T1")
    params = {"K": K, "T1": T1}

    # Construct ODE right-hand side
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u")
    rhs = -x / T1 + u

    f = cas.Function(
        "f",
        [t, x, u, *params.values()],
        [rhs],
        ["t", "x", "u", *params.keys()],
        ["rhs"],
    )

    y = K * x / T1
    h = cas.Function(
        "h",
        [t, x, u, *params.values()],
        [y],
        ["t", "x", "u", *params.keys()],
        ["rhs"],
    )

    model = StateSpaceModelCT(f, h, n)  # nu and ny should be 1 by default

    assert str(model) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,K,T1)->(rhs) SXFunction), "
        "n=1, nu=1, ny=1, params={}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    assert float(model.f(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.h(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.f(0.0, 1.0, 0.0, 1.0, 2.0)) == -0.5
    assert float(model.h(0.0, 1.0, 0.0, 1.0, 2.0)) == 0.5


def test_StateSpaceModelCTFromABCD():
    # Example 1: SISO 1st order system
    K = cas.SX.sym("K")
    T1 = cas.SX.sym("T1")
    params = {"K": K, "T1": T1}

    # State space model matrices
    A = -1 / T1
    B = 1
    C = K / T1
    D = 0

    model = StateSpaceModelCTFromABCD(A, B, C, D, params=params)

    assert str(model) == (
        "StateSpaceModelCTFromABCD("
        "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1)}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    assert float(model.f(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.h(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.f(0.0, 1.0, 0.0, 1.0, 2.0)) == -0.5
    assert float(model.h(0.0, 1.0, 0.0, 1.0, 2.0)) == 0.5