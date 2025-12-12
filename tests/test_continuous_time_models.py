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
    is_ss_ct,
    ATTR_NAMES,
)
from cas_models.discrete_time.models import is_ss_dt
from cas_models.transformations import (
    block_diag,
    connect_nonlinear_systems_in_parallel,
    connect_nonlinear_systems_in_series,
    connect_nonlinear_systems,
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
        "params={'K': SX(K), 'T1': SX(T1)}, name=None, "
        "input_names=['u'], state_names=['x'], output_names=['y'])"
    )

    assert float(model.f(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.h(0.0, 0.0, 0.0, 1.0, 2.0)) == 0.0
    assert float(model.f(0.0, 1.0, 0.0, 1.0, 2.0)) == -0.5
    assert float(model.h(0.0, 1.0, 0.0, 1.0, 2.0)) == 0.5

    # Test model type identification
    assert is_ss_ct(model) is True
    assert is_ss_dt(model) is False


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
        "params={'p_0': SX(p_0), 'p_1': SX(p_1), 'p_2': SX(p_2), 'p_3': SX(p_3)}, name=None, "
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

    # Test model type identification
    assert is_ss_ct(model) is True
    assert is_ss_dt(model) is False


def test_StateSpaceModelCT_O2_SISO(symbolic_O2_SISO):
    n, _, _, _, _, _, _, _, f, h, params = symbolic_O2_SISO

    # nu and ny should be 1 by default
    model = StateSpaceModelCT(f, h, n, params=params)

    assert str(model) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, name=None, "
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

    # Test model type identification
    assert is_ss_ct(model) is True
    assert is_ss_dt(model) is False


def test_StateSpaceModelCTFromABCD_FO_SISO(symbolic_FO_SISO):
    _, _, _, A, B, C, D, _, _, _, _ = symbolic_FO_SISO

    model = StateSpaceModelCTFromABCD(A, B, C, D)

    assert str(model) == (
        "StateSpaceModelCTFromABCD("
        "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1)}, name=None, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    # Test function calls - with scalars
    assert np.array_equal(model.f(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
    assert np.array_equal(model.h(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
    assert np.array_equal(model.f(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[-0.5]]))
    assert np.array_equal(model.h(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[0.5]]))

    # Test model type identification
    assert is_ss_ct(model) is True
    assert is_ss_dt(model) is False


def test_StateSpaceModelCTFromABCD_O2_SISO(symbolic_O2_SISO):
    _, _, _, A, B, C, D, _, _, _, _ = symbolic_O2_SISO

    model = StateSpaceModelCTFromABCD(A, B, C, D)

    assert str(model) == (
        "StateSpaceModelCTFromABCD("
        "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, name=None, "
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

    # Test model type identification
    assert is_ss_ct(model) is True
    assert is_ss_dt(model) is False


def test_SSModelCTFromABCDSISO(symbolic_FO_SISO):
    _, _, _, A, B, C, D, _, _, _, _ = symbolic_FO_SISO

    model = SSModelCTFromABCDSISO(A, B, C, D)

    assert str(model) == (
        "SSModelCTFromABCDSISO("
        "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'K': SX(K), 'T1': SX(T1)}, name=None, "
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
        "n=0, nu=1, ny=1, params={}, name=None, "
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
        "n=0, nu=2, ny=2, params={}, name=None, "
        "input_names=['u1', 'u2'], state_names=['x'], "
        "output_names=['y1', 'y2']"
        ")"
    )

    # Test function calls - with numpy arrays
    u = np.array([-1.0, 2.0]).reshape((-1, 1))
    x = np.empty((0, 1))
    assert np.array_equal(model.f(0.0, x, u), np.empty((0, 1)))
    assert np.array_equal(model.h(0.0, x, u), np.array([[-2.0], [4.25]]))

    # Test model type identification
    assert is_ss_ct(model) is True
    assert is_ss_dt(model) is False


def test_SSModelCTLinearFONoGainSISO():
    # Example 1: Symbolic time constant
    model = SSModelCTLinearFONoGainSISO()

    assert str(model) == (
        "SSModelCTLinearFONoGainSISO("
        "f=Function(f:(t,x,u,T1)->(rhs) SXFunction), "
        "h=Function(h:(t,x,u,T1)->(y) SXFunction), "
        "n=1, nu=1, ny=1, "
        "params={'T1': SX(T1)}, name=None, "
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
        "params={}, name=None, "
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
        "params={'K': SX(K), 'T1': SX(T1), 'T2': SX(T2)}, name=None, "
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
        "params={'T2': SX(T2)}, name=None, "
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
        "params={'T1': SX(T1), 'T2': SX(T2)}, name=None, "
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
        "params={'T2': SX(T2)}, name=None, "
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
        "params={'K': SX(K), 'omega_n': SX(omega_n), 'zeta': SX(zeta)}, name=None, "
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
        "params={'K': SX(K)}, name=None, "
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

    # With defaults - using new generalized function
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2], ATTR_NAMES, StateSpaceModelCT
    )

    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(y[2]) SXFunction), "
        "n=2, nu=2, ny=2, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, name='sys1_sys2', "
        "input_names=['sys1_u', 'sys2_u'], state_names=['sys1_x', 'sys2_x'], "
        "output_names=['sys1_y', 'sys2_y'])"
    )

    # With custom keys
    sys3 = SSModelCTLinearFONoGainSISO()
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2, sys3], ATTR_NAMES, StateSpaceModelCT, keys=["a", "b", "c"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[3],u[3],K,a_T1,b_T1,c_T1)->(rhs[3]) SXFunction), "
        "h=Function(h:(t,x[3],u[3],K,a_T1,b_T1,c_T1)->(y[3]) SXFunction), "
        "n=3, nu=3, ny=3, "
        "params={'K': SX(K), 'a_T1': SX(T1), 'b_T1': SX(T1), 'c_T1': SX(T1)}, "
        "name='sys1_sys2_sys3', "
        "input_names=['a_u', 'b_u', 'c_u'], "
        "state_names=['a_x', 'b_x', 'c_x'], "
        "output_names=['a_y', 'b_y', 'c_y'])"
    )

    # With one constant and a shared parameter
    sys1 = SSModelCTLinearFOSISO(K=2)
    sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys3 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2, sys3], ATTR_NAMES, StateSpaceModelCT, keys=["a", "b", "c"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[3],u[3],T1)->(rhs[3]) SXFunction), "
        "h=Function(h:(t,x[3],u[3],T1)->(y[3]) SXFunction), "
        "n=3, nu=3, ny=3, params={'T1': SX(T1)}, name='sys1_sys2_sys3', "
        "input_names=['a_u', 'b_u', 'c_u'], "
        "state_names=['a_x', 'b_x', 'c_x'], "
        "output_names=['a_y', 'b_y', 'c_y'])"
    )


def test_connect_nonlinear_systems_in_series():
    sys1 = SSModelCTLinearFOSISO()
    sys2 = SSModelCTLinearFONoGainSISO()

    # With defaults - using new generalized function
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], ATTR_NAMES, StateSpaceModelCT
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,sys1_T1,sys2_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, name='sys1_sys2', "
        "input_names=['u'], state_names=['sys2_x', 'sys1_x'], "
        "output_names=['y'])"
    )

    # With custom keys
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], ATTR_NAMES, StateSpaceModelCT, keys=["in", "out"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,in_T1,out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,in_T1,out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'in_T1': SX(T1), 'out_T1': SX(T1)}, name='sys1_sys2', "
        "input_names=['u'], state_names=['out_x', 'in_x'], "
        "output_names=['y'])"
    )

    # With verbose names
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], ATTR_NAMES, StateSpaceModelCT,
        keys=["in", "out"], verbose_names=True
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,in_K,in_T1,out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,in_K,in_T1,out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'in_K': SX(K), 'in_T1': SX(T1), 'out_T1': SX(T1)}, name='sys1_sys2', "
        "input_names=['u'], state_names=['out_x', 'in_x'], "
        "output_names=['y'])"
    )

    # With one constant and a shared parameter
    sys1 = SSModelCTLinearFOSISO(K=2)
    sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], ATTR_NAMES, StateSpaceModelCT,
        keys=["in", "out"], verbose_names=True
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,in_out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,in_out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'in_out_T1': SX(T1)}, name='sys1_sys2', "
        "input_names=['u'], state_names=['out_x', 'in_x'], "
        "output_names=['y'])"
    )


def test_mul_operator_series_connection():
    """Test the * operator for connecting systems in series"""
    sys1 = SSModelCTLinearFOSISO()
    sys2 = SSModelCTLinearFONoGainSISO()

    # Use * operator to connect in series
    sys_combined = sys1 * sys2

    # Should produce the same result as connect_nonlinear_systems_in_series
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,sys1_T1,sys2_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, name='sys1_sys2', "
        "input_names=['u'], state_names=['sys2_x', 'sys1_x'], "
        "output_names=['y'])"
    )

    # Test chaining multiple systems: sys1 * sys2 * sys3
    sys3 = SSModelCTLinearO2NoGainSISO()
    sys_combined_3 = sys1 * sys2 * sys3

    assert str(sys_combined_3) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[4],u,K,sys1_T1,sys2_T1,T1,T2)->(rhs[4]) SXFunction), "
        "h=Function(h:(t,x[4],u,K,sys1_T1,sys2_T1,T1,T2)->(y) SXFunction), "
        "n=4, nu=1, ny=1, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1), "
        "'T1': SX(T1), 'T2': SX(T2)}, name='sys1_sys2_sys1', "
        "input_names=['u'], "
        "state_names=['sys1_x1', 'sys1_x2', 'sys1_sys2_sys2_x', 'sys1_sys2_sys1_x'], "
        "output_names=['y'])"
    )

    assert sys_combined_3.n == 4  # Total of 3 states
    assert sys_combined_3.nu == 1  # Input dimension
    assert sys_combined_3.ny == 1  # Output dimension
    assert is_ss_ct(sys_combined_3) is True
    assert is_ss_dt(sys_combined_3) is False


def test_tf_models():
    sys1 = SSModelCTLinearFONoGainSISO(T1=1)
    sys2 = SSModelCTLinearFOSISO(K=2, T1=2.5)

    # TODO: Test discrete time integration and simulation


def test_connect_simple_feedback():
    """Test simple feedback connection between two systems."""
    # Create two simple systems
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Connect in feedback: sys2 output -> sys1 input, sys1 output -> sys2 input
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections=[('sys2_y', 'sys1_u'), ('sys1_y', 'sys2_u')],
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    # Verify dimensions
    assert connected.n == 2  # Total states from both systems
    assert connected.nu == 0  # No external inputs (all connected)
    assert connected.ny == 2  # All outputs exposed
    assert connected.input_names == []  # No external inputs
    assert 'sys1_y' in connected.output_names
    assert 'sys2_y' in connected.output_names

    # Verify it's a valid CT system
    assert is_ss_ct(connected) is True


def test_connect_list_format():
    """Test list of tuples connection format."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # List format connections
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections=[('sys2_y', 'sys1_u')],
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    assert connected.n == 2
    assert connected.nu == 1  # sys2_u is external
    assert connected.ny == 2
    assert 'sys2_u' in connected.input_names
    assert len(connected.input_names) == 1


def test_connect_dict_format():
    """Test dictionary connection format."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Dict format connections
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections={'sys1_u': 'sys2_y'},
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    assert connected.n == 2
    assert connected.nu == 1  # sys2_u is external
    assert connected.ny == 2
    assert 'sys2_u' in connected.input_names


def test_connect_summing_junction():
    """Test summing junction with multiple outputs to one input."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)

    # Summing junction: sys1_u receives weighted sum of sys2_y and sys3_y
    connected = connect_nonlinear_systems(
        [sys1, sys2, sys3],
        connections={
            'sys1_u': {'sys2_y': 1.0, 'sys3_y': -0.5},  # Weighted sum
        },
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    assert connected.n == 3
    assert connected.nu == 2  # sys2_u and sys3_u are external
    assert connected.ny == 3
    assert 'sys2_u' in connected.input_names
    assert 'sys3_u' in connected.input_names
    assert 'sys1_u' not in connected.input_names  # Connected input


def test_connect_trimming():
    """Test input/output trimming."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Connect with explicit input/output selection
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections={'sys1_u': 'sys2_y'},
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
        input_names=['sys2_u'],  # Only expose sys2_u
        output_names=['sys1_y'],  # Only expose sys1_y
    )

    assert connected.nu == 1
    assert connected.ny == 1
    assert connected.input_names == ['sys2_u']
    assert connected.output_names == ['sys1_y']


def test_connect_no_connections():
    """Test with empty connections (should equal parallel)."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # No connections
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections=[],
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    assert connected.n == 2
    assert connected.nu == 2  # All inputs external
    assert connected.ny == 2  # All outputs exposed


def test_connect_all_inputs_connected():
    """Test closed-loop system (all inputs connected)."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # All inputs connected (closed-loop)
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections={
            'sys1_u': 'sys2_y',
            'sys2_u': 'sys1_y',
        },
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    assert connected.n == 2
    assert connected.nu == 0  # No external inputs
    assert connected.ny == 2
    assert connected.input_names == []


def test_connect_invalid_input_name():
    """Test error on non-existent input."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    with pytest.raises(ValueError, match="Connection target input 'sys3_u' not found"):
        connect_nonlinear_systems(
            [sys1, sys2],
            connections={'sys3_u': 'sys1_y'},  # sys3 doesn't exist
            attr_names=ATTR_NAMES,
            model_class=StateSpaceModelCT,
        )


def test_connect_invalid_output_name():
    """Test error on non-existent output."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    with pytest.raises(ValueError, match="Connection source output 'sys3_y'"):
        connect_nonlinear_systems(
            [sys1, sys2],
            connections={'sys1_u': 'sys3_y'},  # sys3 doesn't exist
            attr_names=ATTR_NAMES,
            model_class=StateSpaceModelCT,
        )


def test_connect_duplicate_list_connections():
    """Test error on many-to-one connections in list format."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    with pytest.raises(ValueError, match="Duplicate connection target 'sys1_u'"):
        connect_nonlinear_systems(
            [sys1, sys2],
            connections=[
                ('sys2_y', 'sys1_u'),
                ('sys1_y', 'sys1_u'),  # Duplicate target
            ],
            attr_names=ATTR_NAMES,
            model_class=StateSpaceModelCT,
        )


def test_connect_vs_series():
    """Test that connect gives same result as series for simple cascade."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Connect in series using connect_nonlinear_systems
    connected_via_connect = connect_nonlinear_systems(
        [sys1, sys2],
        connections={'sys2_u': 'sys1_y'},  # sys1 output to sys2 input
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
        input_names=['sys1_u'],
        output_names=['sys2_y'],
    )

    # Connect using series function
    connected_via_series = connect_nonlinear_systems_in_series(
        [sys1, sys2],
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    # Should have same dimensions
    assert connected_via_connect.n == connected_via_series.n
    assert connected_via_connect.nu == connected_via_series.nu
    assert connected_via_connect.ny == connected_via_series.ny

    # Test with same inputs
    t_val = 0.0
    x_val = cas.DM.zeros(2, 1)
    u_val = cas.DM([1.0])

    y_connect = connected_via_connect.h(t_val, x_val, u_val)
    y_series = connected_via_series.h(t_val, x_val, u_val)

    assert np.allclose(np.array(y_connect), np.array(y_series))


def test_connect_three_systems():
    """Complex example with three systems."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)

    # Complex connections
    connected = connect_nonlinear_systems(
        [sys1, sys2, sys3],
        connections={
            'sys1_u': {'sys2_y': 1.0, 'sys3_y': -0.5},  # Summing junction
            'sys2_u': 'sys1_y',  # Feedback from sys1
        },
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
        input_names=['sys3_u'],  # Only sys3 input is external
        output_names=['sys1_y', 'sys2_y'],  # Expose sys1 and sys2 outputs
    )

    assert connected.n == 3  # All three state variables
    assert connected.nu == 1  # Only sys3_u is external
    assert connected.ny == 2  # sys1_y and sys2_y
    assert connected.input_names == ['sys3_u']
    assert connected.output_names == ['sys1_y', 'sys2_y']


def test_connect_list_with_unit_gains():
    """Test dictionary format with list of outputs (unit gains)."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)

    # Using list format for summing with unit gains
    connected = connect_nonlinear_systems(
        [sys1, sys2, sys3],
        connections={
            'sys1_u': ['sys2_y', 'sys3_y'],  # Sum with unit gains
        },
        attr_names=ATTR_NAMES,
        model_class=StateSpaceModelCT,
    )

    assert connected.n == 3
    assert connected.nu == 2  # sys2_u and sys3_u
    assert 'sys1_u' not in connected.input_names
