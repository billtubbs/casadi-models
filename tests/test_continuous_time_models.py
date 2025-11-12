"""Unit tests for src/cas_models/continuous_time/models.py module"""

import pytest
import numpy as np
import casadi as cas
from cas_models.continuous_time.models import (
    StateSpaceModelCT,
    StateSpaceModelCTStaticNonlinearity,
    StateSpaceModelCTFromABCD,
    SSModelCTFromABCDSISO,
    SSModelCTDirectTransmission,
    SSModelCTLinearFONoGainSISO,
    SSModelCTLinearFOSISO,
    SSModelCTLinearO2SISO,
    SSModelCTLinearO2NoGainSISO,
    SSModelCTLinearO2UnderdampedSISO,
    block_diag,
    connect_nonlinear_systems_in_parallel,
    connect_nonlinear_systems_in_series,
)


@pytest.fixture
def symbolic_FO_SISO():
    """Example SISO system 1: 1st order"""
    # Dimensions
    n = 1
    nu = 1
    ny = 1

    # Parameters (symbolic)
    K = cas.SX.sym("K")
    T1 = cas.SX.sym("T1")
    params = {"K": K, "T1": T1}

    # State space model matrices
    A = -1 / T1
    B = 1
    C = K / T1
    D = 0

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
        ["y"],
    )

    return n, nu, ny, A, B, C, D, t, f, h, params


@pytest.fixture
def symbolic_O2_SISO():
    """Example SISO system 2: 1st order"""
    # Dimensions
    n = 2
    nu = 1
    ny = 1

    # Parameters (symbolic)
    K = cas.SX.sym("K")
    T1 = cas.SX.sym("T1")
    T2 = cas.SX.sym("T2")
    params = {"K": K, "T1": T1, "T2": T2}

    # State space model matrices
    A = cas.sparsify(
        cas.blockcat([[0, 1], [-1 / (T1 * T2), (-T1 - T2) / (T1 * T2)]])
    )
    B = cas.sparsify(cas.blockcat([[0], [1]]))
    C = cas.sparsify(cas.blockcat([[K / (T1 * T2), 0]]))
    D = cas.sparsify(cas.DM(0))

    # Construct ODE right-hand side
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u")
    rhs = A @ x + B @ u

    f = cas.Function(
        "f",
        [t, x, u, *params.values()],
        [rhs],
        ["t", "x", "u", *params.keys()],
        ["rhs"],
    )

    y = C @ x + D @ u
    h = cas.Function(
        "h",
        [t, x, u, *params.values()],
        [y],
        ["t", "x", "u", *params.keys()],
        ["y"],
    )

    return n, nu, ny, A, B, C, D, t, f, h, params


def test_StateSpaceModelCT_FO_SISO(symbolic_FO_SISO):
    n, _, _, _, _, _, _, _, f, h, params = symbolic_FO_SISO

    # nu and ny should be 1 by default
    model = StateSpaceModelCT(f, h, n, params=params)

    assert str(model) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1)}, "
        "input_names=['u'], state_names=['x'], output_names=['y'])"
    )

    assert float(model.f(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.h(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.f(0.0, 1.0, 0.0, 1.0, 2.0)) == -0.5
    assert float(model.h(0.0, 1.0, 0.0, 1.0, 2.0)) == 0.5


def test_StateSpaceModelCTStaticNonlinearity():
    nu = 3
    ny = 1

    # Construct a nonlinear output function
    # 4th order - polynomial
    n = 0
    nu = 1
    ny = 1
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)
    p = cas.SX.sym("p", 4)
    y = p[3] * u**3 + p[2] * u**2 - p[1] * u + p[0]

    # TODO: How to pass params as vectors
    # params = {"p": p}

    symbolic_params = {}
    for pi in cas.symvar(cas.SX(p)):
        symbolic_params[pi.name()] = pi

    h = cas.Function(
        "f",
        [t, x, u, *symbolic_params.values()],
        [y],
        ["t", "x", "u", *symbolic_params.keys()],
        ["y"],
    )

    model = StateSpaceModelCTStaticNonlinearity(
        h, nu=nu, ny=ny, params=symbolic_params
    )

    assert str(model) == (
        "StateSpaceModelCTStaticNonlinearity("
        "f=Function(f:(t,x[0],u,p_0,p_1,p_2,p_3)->(rhs[0]) SXFunction), "
        "h=Function(f:(t,x[0],u,p_0,p_1,p_2,p_3)->(y) SXFunction), "
        "n=0, nu=1, ny=1, "
        "params={'p_0': SX(p_0), 'p_1': SX(p_1), 'p_2': SX(p_2), 'p_3': SX(p_3)}, "
        "input_names=['u'], state_names=['x'], output_names=['y'])"
    )

    # Test function calculations
    t = 0.0
    x = []
    u = -1.5
    p_0, p_1, p_2, p_3 = [1, -7, 1, 2]
    rhs = model.f(t, x, u, p_0, p_1, p_2, p_3)
    assert rhs.shape == (0, 1)  # empty
    y = model.h(t, x, u, p_0, p_1, p_2, p_3)
    assert y == p_3 * u**3 + p_2 * u**2 - p_1 * u + p_0


def test_StateSpaceModelCT_O2_SISO(symbolic_O2_SISO):
    n, _, _, _, _, _, _, _, f, h, params = symbolic_O2_SISO

    # nu and ny should be 1 by default
    model = StateSpaceModelCT(f, h, n, params=params)

    assert str(model) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y'])"
    )

    # Test function calls
    t = cas.DM(0)
    x = cas.DM([0.0, 1.0])
    u = cas.DM([-0.5])
    K = cas.DM(0.8)
    T1 = cas.DM(0.6)
    T2 = cas.DM(3.0)
    assert np.allclose(model.f(t, x, u, K, T1, T2), cas.DM([1, -2.5]))
    assert np.allclose(model.h(t, x, u, K, T1, T2), cas.DM(0))

    T1 = T2
    x = cas.DM([3.0, 1.0])
    assert np.allclose(model.f(t, x, u, K, T1, T2), cas.DM([1, -1.5]))
    assert np.allclose(
        model.h(t, x, u, K, T1, T2), cas.DM(0.26666666666666666)
    )


def test_StateSpaceModelCTFromABCD_FO_SISO(symbolic_FO_SISO):
    _, _, _, A, B, C, D, _, _, _, _ = symbolic_FO_SISO

    model = StateSpaceModelCTFromABCD(A, B, C, D)

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


def test_StateSpaceModelCTFromABCD_O2_SISO(symbolic_O2_SISO):
    _, _, _, A, B, C, D, _, _, _, _ = symbolic_O2_SISO

    model = StateSpaceModelCTFromABCD(A, B, C, D)

    assert str(model) == (
        "StateSpaceModelCTFromABCD("
        "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y'])"
    )

    # Test function calls
    t = cas.DM(0)
    x = cas.DM([0.0, 1.0])
    u = cas.DM([-0.5])
    K = cas.DM(0.8)
    T1 = cas.DM(0.6)
    T2 = cas.DM(3.0)
    assert np.allclose(model.f(t, x, u, K, T1, T2), cas.DM([1, -2.5]))
    assert np.allclose(model.h(t, x, u, K, T1, T2), cas.DM(0))

    T1 = T2
    x = cas.DM([3.0, 1.0])
    assert np.allclose(model.f(t, x, u, K, T1, T2), cas.DM([1, -1.5]))
    assert np.allclose(
        model.h(t, x, u, K, T1, T2), cas.DM(0.26666666666666666)
    )


def test_SSModelCTFromABCDSISO(symbolic_FO_SISO):
    _, _, _, A, B, C, D, _, _, _, _ = symbolic_FO_SISO

    model = SSModelCTFromABCDSISO(A, B, C, D)

    assert str(model) == (
        "SSModelCTFromABCDSISO("
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

    assert str(model) == (
        "SSModelCTDirectTransmission("
        "f=Function(f:(t,x[0],u[2])->(rhs[0]) SXFunction), "
        "h=Function(h:(t,x[0],u[2])->(y[2]) SXFunction), "
        "n=0, nu=2, ny=2, params={}, "
        "input_names=['u1', 'u2'], state_names=['x'], "
        "output_names=['y1', 'y2']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0, 2.0]).reshape((-1, 1))
    x = np.empty((0, 1))
    assert np.array_equal(model.f(0.0, x, u), np.empty((0, 1)))
    assert np.array_equal(model.h(0.0, x, u), np.array([[-2.0], [4.25]]))


def test_SSModelCTLinearFONoGainSISO():
    # Example 1: Symbolic time constant
    model = SSModelCTLinearFONoGainSISO()

    assert str(model) == (
        "SSModelCTLinearFONoGainSISO("
        "f=Function(f:(t,x,u,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'T1': SX(T1)}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0]).reshape((-1, 1))
    assert np.array_equal(model.f(0.0, x, u, 0.5), [[-5]])
    assert np.array_equal(model.h(0.0, x, u, 0.5), [[4]])

    # Example 2: Fixed time constant
    T1 = 0.5
    model = SSModelCTLinearFONoGainSISO(T1=T1)

    assert str(model) == (
        "SSModelCTLinearFONoGainSISO("
        "f=Function(f:(t,x,u)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={}, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0]).reshape((-1, 1))
    assert np.array_equal(model.f(0.0, x, u), [[-5]])
    assert np.array_equal(model.h(0.0, x, u), [[4]])


def test_SSModelCTLinearO2SISO():
    # Example 1: Symbolic time constant
    model = SSModelCTLinearO2SISO()

    assert str(model) == (
        "SSModelCTLinearO2SISO("
        "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0, -1.0]).reshape((-1, 1))
    assert np.allclose(model.f(0.0, x, u, 2.0, 0.5, 2.5), [[-1], [-0.2]])
    assert np.allclose(model.h(0.0, x, u, 2.0, 0.5, 2.5), [[3.2]])

    # Example 2: Fixed parameters
    K = 2.0
    T1 = 0.5
    model = SSModelCTLinearO2SISO(K=K, T1=T1)
    assert str(model) == (
        "SSModelCTLinearO2SISO("
        "f=Function(f:(t,x[2],u,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'T2': SX(T2)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0, -1.0]).reshape((-1, 1))
    assert np.allclose(model.f(0.0, x, u, 2.5), [[-1], [-0.2]])
    assert np.allclose(model.h(0.0, x, u, 2.5), [[3.2]])


def test_SSModelCTLinearO2NoGainSISO():
    # Example 1: Symbolic time constant
    model = SSModelCTLinearO2NoGainSISO()

    assert str(model) == (
        "SSModelCTLinearO2NoGainSISO("
        "f=Function(f:(t,x[2],u,T1,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,T1,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'T1': SX(T1), 'T2': SX(T2)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0, -1.0]).reshape((-1, 1))
    assert np.allclose(model.f(0.0, x, u, 0.5, 2.5), [[-1], [-0.2]])
    assert np.allclose(model.h(0.0, x, u, 0.5, 2.5), [[1.6]])

    # Example 2: Fixed parameters
    T1 = 0.5
    model = SSModelCTLinearO2NoGainSISO(T1=T1)
    assert str(model) == (
        "SSModelCTLinearO2NoGainSISO("
        "f=Function(f:(t,x[2],u,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'T2': SX(T2)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0, -1.0]).reshape((-1, 1))
    assert np.allclose(model.f(0.0, x, u, 2.5), [[-1], [-0.2]])
    assert np.allclose(model.h(0.0, x, u, 2.5), [[1.6]])


def test_SSModelCTLinearO2UnderdampedSISO():
    # Example 1: Symbolic time constant
    model = SSModelCTLinearO2UnderdampedSISO()

    assert str(model) == (
        "SSModelCTLinearO2UnderdampedSISO("
        "f=Function(f:(t,x[2],u,K,omega_n,zeta)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,omega_n,zeta)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'omega_n': SX(omega_n), 'zeta': SX(zeta)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0, -1.0]).reshape((-1, 1))
    assert np.allclose(model.f(0.0, x, u, 2.0, 1.0, 0.5), [[-1], [-2]])
    assert np.allclose(model.h(0.0, x, u, 2.0, 1.0, 0.5), [[4.0]])

    # Example 2: Fixed parameters
    zeta = 0.5
    omega_n = 1.0
    model = SSModelCTLinearO2UnderdampedSISO(zeta=zeta, omega_n=omega_n)
    assert str(model) == (
        "SSModelCTLinearO2UnderdampedSISO("
        "f=Function(f:(t,x[2],u,K)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K)}, "
        "input_names=['u'], state_names=['x1', 'x2'], output_names=['y'])"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0]).reshape((-1, 1))
    x = np.array([2.0, -1.0]).reshape((-1, 1))
    assert np.allclose(model.f(0.0, x, u, 2.0), [[-1], [-2]])
    assert np.allclose(model.h(0.0, x, u, 2.0), [[4.0]])


def test_block_diag():
    matrices = [
        cas.DM.ones(2, 2),
        2 * cas.DM.ones(1, 1),
        cas.DM.zeros(0, 0),  # empty matrix
        3 * cas.DM.ones(3, 2),
    ]
    result = block_diag(matrices)
    assert np.array_equal(
        cas.DM(result),
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 3.0, 3.0],
            ]
        ),
    )


def test_connect_nonlinear_systems_in_parallel():
    sys1 = SSModelCTLinearFOSISO()
    sys2 = SSModelCTLinearFONoGainSISO()

    # With defaults
    sys_combined = connect_nonlinear_systems_in_parallel([sys1, sys2])

    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(y[2]) SXFunction), "
        "n=2, nu=2, ny=2, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, "
        "input_names=['sys1_u', 'sys2_u'], state_names=['sys1_x', 'sys2_x'], "
        "output_names=['sys1_y', 'sys2_y'])"
    )

    # With custom keys
    sys3 = SSModelCTLinearFONoGainSISO()
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2, sys3], keys=["a", "b", "c"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[3],u[3],K,a_T1,b_T1,c_T1)->(rhs[3]) SXFunction), "
        "h=Function(h:(t,x[3],u[3],K,a_T1,b_T1,c_T1)->(y[3]) SXFunction), "
        "n=3, nu=3, ny=3, "
        "params={'K': SX(K), 'a_T1': SX(T1), 'b_T1': SX(T1), 'c_T1': SX(T1)}, "
        "input_names=['a_u', 'b_u', 'c_u'], "
        "state_names=['a_x', 'b_x', 'c_x'], "
        "output_names=['a_y', 'b_y', 'c_y'])"
    )

    # With one constant and a shared parameter
    sys1 = SSModelCTLinearFOSISO(K=2)
    sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys3 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2, sys3], keys=["a", "b", "c"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[3],u[3],T1)->(rhs[3]) SXFunction), "
        "h=Function(h:(t,x[3],u[3],T1)->(y[3]) SXFunction), "
        "n=3, nu=3, ny=3, params={'T1': SX(T1)}, "
        "input_names=['a_u', 'b_u', 'c_u'], "
        "state_names=['a_x', 'b_x', 'c_x'], "
        "output_names=['a_y', 'b_y', 'c_y'])"
    )


def test_connect_nonlinear_systems_in_series():
    sys1 = SSModelCTLinearFOSISO()
    sys2 = SSModelCTLinearFONoGainSISO()

    # With defaults
    sys_combined = connect_nonlinear_systems_in_series([sys1, sys2])
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,sys1_T1,sys2_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, "
        "input_names=['u'], state_names=['sys2_x', 'sys1_x'], "
        "output_names=['y'])"
    )

    # With custom keys
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], keys=["in", "out"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,in_T1,out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,in_T1,out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'in_T1': SX(T1), 'out_T1': SX(T1)}, "
        "input_names=['u'], state_names=['out_x', 'in_x'], "
        "output_names=['y'])"
    )

    # With verbose names
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], keys=["in", "out"], verbose_names=True
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,in_K,in_T1,out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,in_K,in_T1,out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'in_K': SX(K), 'in_T1': SX(T1), 'out_T1': SX(T1)}, "
        "input_names=['u'], state_names=['out_x', 'in_x'], "
        "output_names=['y'])"
    )

    # With one constant and a shared parameter
    sys1 = SSModelCTLinearFOSISO(K=2)
    sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], keys=["in", "out"], verbose_names=True
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,in_out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,in_out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'in_out_T1': SX(T1)}, "
        "input_names=['u'], state_names=['out_x', 'in_x'], "
        "output_names=['y'])"
    )


def test_tf_models():
    sys1 = SSModelCTLinearFONoGainSISO(T1=1)
    sys2 = SSModelCTLinearFOSISO(K=2, T1=2.5)

    # TODO: Test discrete time integration and simulation
