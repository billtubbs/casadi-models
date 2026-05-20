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
)
from cas_models.validation import is_ss_ct, is_ss_dt
from cas_models.transformations import block_diag, sum_systems


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
    """Test the * operator for connecting systems in series.

    G1 * G2 follows the matrix-multiplication convention: signal flows through
    G2 first, then G1 (i.e. u -> G2 -> G1 -> y).
    """
    sys1 = SSModelCTLinearFOSISO()
    sys2 = SSModelCTLinearFONoGainSISO()

    # sys1 * sys2: signal flows u -> sys2 -> sys1 -> y
    sys_combined = sys1 * sys2

    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,sys1_T1,sys2_T1,K)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,sys1_T1,sys2_T1,K)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'sys1_T1': SX(T1), 'sys2_T1': SX(T1), 'K': SX(K)}, name='sys1_sys2', "
        "input_names=['u'], state_names=['sys2_x', 'sys1_x'], "
        "output_names=['y'])"
    )

    # Test chaining: sys1 * sys2 * sys3 means u -> sys3 -> sys2 -> sys1 -> y
    sys3 = SSModelCTLinearO2NoGainSISO()
    sys_combined_3 = sys1 * sys2 * sys3

    assert str(sys_combined_3) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[4],u,T1,T2,sys1_T1,sys2_T1,K)->(rhs[4]) SXFunction), "
        "h=Function(h:(t,x[4],u,T1,T2,sys1_T1,sys2_T1,K)->(y) SXFunction), "
        "n=4, nu=1, ny=1, "
        "params={'T1': SX(T1), 'T2': SX(T2), 'sys1_T1': SX(T1), "
        "'sys2_T1': SX(T1), 'K': SX(K)}, name='sys1_sys1_sys2', "
        "input_names=['u'], state_names=['sys2_x', 'sys1_x', 'x1', 'x2'], "
        "output_names=['y'])"
    )

    assert sys_combined_3.n == 4  # Total of 3 states
    assert sys_combined_3.nu == 1  # Input dimension
    assert sys_combined_3.ny == 1  # Output dimension
    assert is_ss_ct(sys_combined_3) is True
    assert is_ss_dt(sys_combined_3) is False


def test_mul_operator_signal_flow_order():
    """Verify that * follows matrix-multiplication signal-flow order.

    For linear SISO systems G1 * G2 = G2 * G1 in the Laplace domain, so
    ordering cannot be detected numerically.  Inserting a squaring
    nonlinearity NL (y = u**2) breaks commutativity and lets us confirm:

        NL * G1 * G2  →  u → G2 → G1 → NL → y,  output = (2·x_G1)²
        G1 * NL * G2  →  u → G2 → NL → G1 → y,  output =  2·x_G1
        G2 * G1 * NL  →  u → NL → G1 → G2 → y,  output =  3·x_G2

    All three outputs are numerically distinct, so this test fails if
    the order argument to connect_systems_in_series is swapped.
    """
    t_sym = cas.SX.sym("t")

    # Squaring nonlinearity: y = u**2 (no state, no params)
    x_nl = cas.SX.sym("x", 0)
    u_nl = cas.SX.sym("u")
    h_nl = cas.Function(
        "h", [t_sym, x_nl, u_nl], [u_nl**2], ["t", "x", "u"], ["y"]
    )
    NL = StateSpaceModelCTStaticNonlinearity(h_nl)

    # G1: dx/dt = -x + u,      y = 2*x  (gain=2, no direct feedthrough)
    x1 = cas.SX.sym("x")
    u1 = cas.SX.sym("u")
    G1 = StateSpaceModelCT(
        cas.Function(
            "f", [t_sym, x1, u1], [-x1 + u1], ["t", "x", "u"], ["rhs"]
        ),
        cas.Function("h", [t_sym, x1, u1], [2 * x1], ["t", "x", "u"], ["y"]),
        n=1,
    )

    # G2: dx/dt = -0.5*x + u,  y = 3*x  (gain=3, no direct feedthrough)
    x2 = cas.SX.sym("x")
    u2 = cas.SX.sym("u")
    G2 = StateSpaceModelCT(
        cas.Function(
            "f", [t_sym, x2, u2], [-0.5 * x2 + u2], ["t", "x", "u"], ["rhs"]
        ),
        cas.Function("h", [t_sym, x2, u2], [3 * x2], ["t", "x", "u"], ["y"]),
        n=1,
    )

    x_G1 = cas.DM([1.5])
    x_G2 = cas.DM([0.8])
    u = cas.DM([2.0])
    t = 0.0

    # NL * G1 * G2: u → G2 → G1 → NL → y
    # Output stage is NL, so y = (G1.y)² = (2·x_G1)²
    # State vector: [x_G1, x_G2]
    sys_NL_G1_G2 = NL * G1 * G2
    assert sys_NL_G1_G2.n == 2
    y_NL_G1_G2 = float(sys_NL_G1_G2.h(t, cas.vertcat(x_G1, x_G2), u))
    assert np.isclose(y_NL_G1_G2, (2 * float(x_G1)) ** 2)  # 9.0

    # G1 * NL * G2: u → G2 → NL → G1 → y
    # Output stage is G1 (D=0), so y = G1.y = 2·x_G1
    # State vector: [x_G1, x_G2]
    sys_G1_NL_G2 = G1 * NL * G2
    assert sys_G1_NL_G2.n == 2
    y_G1_NL_G2 = float(sys_G1_NL_G2.h(t, cas.vertcat(x_G1, x_G2), u))
    assert np.isclose(y_G1_NL_G2, 2 * float(x_G1))  # 3.0

    # G2 * G1 * NL: u → NL → G1 → G2 → y
    # Output stage is G2 (D=0), so y = G2.y = 3·x_G2
    # State vector: [x_G2, x_G1]
    sys_G2_G1_NL = G2 * G1 * NL
    assert sys_G2_G1_NL.n == 2
    y_G2_G1_NL = float(sys_G2_G1_NL.h(t, cas.vertcat(x_G2, x_G1), u))
    assert np.isclose(y_G2_G1_NL, 3 * float(x_G2))  # 2.4

    # All three outputs are distinct, confirming order sensitivity
    assert y_NL_G1_G2 != y_G1_NL_G2
    assert y_G1_NL_G2 != y_G2_G1_NL
    assert y_NL_G1_G2 != y_G2_G1_NL


def test_add_operator_parallel_connection():
    """Test the + operator connects continuous-time systems with shared input and summed outputs."""
    sys1 = SSModelCTLinearFOSISO(K=2.0, T1=1.0)
    sys2 = SSModelCTLinearFONoGainSISO(T1=0.5)

    sys_combined = sys1 + sys2

    # States stacked, nu and ny unchanged
    assert sys_combined.n == 2
    assert sys_combined.nu == 1
    assert sys_combined.ny == 1
    assert sys_combined.input_names == sys1.input_names
    assert sys_combined.output_names == sys1.output_names
    assert is_ss_ct(sys_combined) is True
    assert is_ss_dt(sys_combined) is False

    # Output must equal sum of individual outputs
    # K and T1 are baked in as numeric so h(t, x, u) takes no extra args
    t_val = cas.DM(0.0)
    x1 = cas.DM([1.5])
    x2 = cas.DM([0.8])
    x = cas.vertcat(x1, x2)
    u = cas.DM([1.0])
    y1 = float(sys1.h(t_val, x1, u))
    y2 = float(sys2.h(t_val, x2, u))
    y_combined = float(sys_combined.h(t_val, x, u))
    assert np.isclose(y_combined, y1 + y2)

    # + is equivalent to sum_systems
    sys_via_func = sum_systems([sys1, sys2], model_class=StateSpaceModelCT)
    assert str(sys_combined) == str(sys_via_func)


def test_describe(capsys):
    """Test describe() prints a human-readable summary of a CT model."""
    sys = SSModelCTLinearFOSISO(K=2.0, T1=1.0, name="plant")
    sys.describe()
    out = capsys.readouterr().out
    assert "SSModelCTLinearFOSISO" in out
    assert "Name: plant" in out
    assert "States (n=1)" in out
    assert "Inputs (nu=1)" in out
    assert "Outputs (ny=1)" in out
    assert "Parameters" in out
    assert "dt" not in out


def test_tf_models():
    sys1 = SSModelCTLinearFONoGainSISO(T1=1)
    sys2 = SSModelCTLinearFOSISO(K=2, T1=2.5)

    # TODO: Test discrete time integration and simulation
