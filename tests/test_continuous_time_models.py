import numpy as np
import casadi as cas
from cas_models.continuous_time.models import (
    StateSpaceModelCT,
    StateSpaceModelCTFromABCD,
    SSModelCTDirectTransmission,
    SSModelCTLinearFONoGainSISO
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

    assert cas.is_equal(model.A, A)
    assert cas.is_equal(model.B, B)
    assert cas.is_equal(model.C, C)
    assert cas.is_equal(model.D, D)

    assert str(model) == (
        "StateSpaceModelCTFromABCD("
        "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1)}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    # Test function calls - with scalars
    assert np.array_equal(model.f(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
    assert np.array_equal(model.h(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
    assert np.array_equal(model.f(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[-0.5]]))
    assert np.array_equal(model.h(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[0.5]]))


def test_SSModelCTDirectTransmission():

    # Example 1: SISO static gain = 1
    model = SSModelCTDirectTransmission(nu=1)

    # assert cas.is_equal(model.A, A)
    # assert cas.is_equal(model.B, B)
    # assert cas.is_equal(model.C, C)
    assert cas.is_equal(model.D, cas.DM(1))

    assert str(model) == (
        "SSModelCTDirectTransmission("
        "f=Function(f:(t,x[0],u)->(rhs[0]) SXFunction), "
        "h=Function(h:(t,x[0],u)->(y) SXFunction), "
        "n=0, nu=1, ny=1, params={}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )
    assert np.array_equal(model.f(0.0, cas.DM.zeros(0), 1.1), np.empty((0, 1)))
    assert np.array_equal(
        model.h(0.0, cas.DM.zeros(0), 1.1), np.array([[1.1]])
    )

    # Example 2: 2x2 static gains
    D = np.array([[1.0, -0.5], [-0.25, 2.0]])
    model = SSModelCTDirectTransmission(D=D)

    # assert cas.is_equal(model.A, A)
    # assert cas.is_equal(model.B, B)
    # assert cas.is_equal(model.C, C)
    assert cas.is_equal(model.D, D)

    assert str(model) == (
        "SSModelCTDirectTransmission("
        "f=Function(f:(t,x[0],u[2])->(rhs[0]) SXFunction), "
        "h=Function(h:(t,x[0],u[2])->(y[2]) SXFunction), "
        "n=0, nu=1, ny=1, params={}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0, 2.0]).reshape((-1, 1))
    x = np.empty((0, 1))
    assert np.array_equal(model.f(0.0, x, u), np.empty((0, 1)))
    assert np.array_equal(model.h(0.0, x, u), np.array([[-2.0], [4.25]]))


def test_SSModelCTLinearFONoGainSISO():

    # Example 1: Symbolic time constant
    model = SSModelCTLinearFONoGainSISO(T1=None)

    assert cas.is_equal(cas.simplify(model.A - (-1 / model.params['T1'])), 0)
    assert cas.is_equal(model.B, 1)
    assert cas.is_equal(cas.simplify(model.C - 1 / model.params['T1']), 0)
    assert cas.is_equal(model.D, 0)

    assert str(model) == (
        "SSModelCTLinearFONoGainSISO("
        "f=Function(f:(t,x,u,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, params={'T1': SX(T1)}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    # Example 2: Fixed time constant
    T1 = 0.5
    model = SSModelCTLinearFONoGainSISO(T1=T1)

    assert np.allclose(model.A, -1 / T1)
    assert cas.is_equal(model.B, 1)
    assert np.allclose(model.C, 1 / T1)
    assert cas.is_equal(model.D, 0)

    assert str(model) == (
        "SSModelCTLinearFONoGainSISO("
        "f=Function(f:(t,x,u)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u)->(y) SXFunction), "
        "n=1, nu=1, ny=1, params={}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )
