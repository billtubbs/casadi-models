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
from cas_models.transformations import block_diag


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
        "input_names=['u'], state_names=[], output_names=['y'])"
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
        "input_names=['u'], state_names=[], output_names=['y']"
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
        "input_names=['u1', 'u2'], state_names=[], "
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
        "input_names=['u'], state_names=['x1', 'x2', 'sys2_x', 'sys1_x'], "
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
