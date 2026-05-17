"""Unit tests for src/cas_models/continuous_time/models.py module"""

import pytest
import numpy as np
import pandas as pd
import casadi as cas
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTARXSISO,
    StateSpaceModelDTDelay,
    StateSpaceModelDTTFSISO,
    StateSpaceModelDTFromCTRK4,
    StateSpaceModelDTFromCT,
)
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.validation import is_ss_ct, is_ss_dt
from cas_models.transformations import sum_systems
from pathlib import Path


DATA_DIR = Path("tests/data")


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


@pytest.fixture
def data_TP04_Q1a():
    data = pd.read_csv(DATA_DIR / "TP04_Q1a.csv")
    return data


@pytest.fixture
def data_TP04_Q1a_ss():
    data = pd.read_csv(DATA_DIR / "TP04_Q1a_ss.csv")
    return data


@pytest.fixture
def data_tf2ss_simulation():
    """Load Octave simulation results for tf2ss test cases"""
    data = pd.read_csv(DATA_DIR / "results_tf2ss.csv")
    return data


def test_StateSpaceModelDT_FO_SISO(symbolic_FO_SISO):
    n, nu, ny, A, B, C, D, t, F, H, params = symbolic_FO_SISO

    # nu and ny should be 1 by default - TODO: Use SISO version
    model = StateSpaceModelDT(F, H, n, nu, ny, params=params)

    assert str(model) == (
        "StateSpaceModelDT("
        "F=Function(F:(t,xk,uk,a1,b0)->(xkp1) SXFunction), "
        "H=Function(H:(t,xk,uk,a1,b0)->(yk) SXFunction), "
        "n=1, nu=1, ny=1, dt=None, "
        "params={'a1': SX(a1), 'b0': SX(b0)}, name=None, "
        "input_names=['u'], state_names=['x'], output_names=['y']"
        ")"
    )

    assert float(model.F(0.0, 0.0, 0.0, 0.2, 0.8)) == 0.0
    assert float(model.H(0.0, 0.0, 0.0, 0.2, 0.8)) == 0.0
    assert float(model.F(0.0, 1.0, 0.0, 0.2, 0.8)) == -0.2
    assert float(model.H(0.0, 1.0, 0.0, 0.2, 0.8)) == 1.0

    # Test model type identification
    assert is_ss_ct(model) is False
    assert is_ss_dt(model) is True


def test_symbolic_AR211_SISO(symbolic_AR211_SISO):
    n, nu, ny, A, B, C, D, t, F, H, params = symbolic_AR211_SISO

    # nu and ny should be 1 by default - TODO: Use SISO version
    model = StateSpaceModelDT(F, H, n, nu, ny, params=params)

    assert str(model) == (
        "StateSpaceModelDT(F=Function("
        "F:(t,xk[4],uk,Aq[2],Bq[2])->(xkp1[4]) SXFunction), "
        "H=Function(H:(t,xk[4],uk,Aq[2],Bq[2])->(yk) SXFunction), "
        "n=4, nu=1, ny=1, dt=None, "
        "params={'Aq': SX([Aq_0, Aq_1]), 'Bq': SX([Bq_0, Bq_1])}, "
        "name=None, "
        "input_names=['u'], state_names=['x1', 'x2', 'x3', 'x4'], "
        "output_names=['y']"
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

    # Test model type identification
    assert is_ss_ct(model) is False
    assert is_ss_dt(model) is True


def test_StateSpaceModelDTTFSISO_input_types(tf_test_case_2):
    """Test StateSpaceModelDTTFSISO accepts different input types.

    Tests that the model can be instantiated with CasADi DM, numpy arrays,
    and Python lists, and produces consistent results.
    """
    # Expected string representation (should be same for all input types)
    str_rep_expected = (
        "StateSpaceModelDTTFSISO("
        "F=Function(F:(t,xk[3],uk)->(xkp1[3]) SXFunction), "
        "H=Function(H:(t,xk[3],uk)->(yk) SXFunction), "
        "n=3, nu=1, ny=1, dt=None, "
        "params={}, name=None, "
        "input_names=['u'], state_names=['x1', 'x2', 'x3'], "
        "output_names=['y']"
        ")"
    )

    # Test with CasADi DM
    num = cas.DM(tf_test_case_2["num"])
    den = cas.DM(tf_test_case_2["den"])
    model_dm = StateSpaceModelDTTFSISO(num=num, den=den)
    assert str(model_dm) == str_rep_expected
    assert model_dm.n == 3

    # Test with numpy arrays
    num = tf_test_case_2["num"]
    den = tf_test_case_2["den"]
    model_np = StateSpaceModelDTTFSISO(num=num, den=den)
    assert str(model_np) == str_rep_expected
    assert model_np.n == 3

    # Test with Python lists
    num = tf_test_case_2["num"].tolist()
    den = tf_test_case_2["den"].tolist()
    model_list = StateSpaceModelDTTFSISO(num=num, den=den)
    assert str(model_list) == str_rep_expected
    assert model_list.n == 3

    # Test with CasADi symbolic variables
    num = cas.SX.sym("b", len(num))
    den = cas.SX.sym("a", len(den))
    model_sx = StateSpaceModelDTTFSISO(num=num, den=den)
    assert str(model_sx) == (
        "StateSpaceModelDTTFSISO("
        "F=Function(F:(t,xk[3],uk,b_0,b_1,b_2,a_0,a_1,a_2,a_3)->(xkp1[3]) SXFunction), "
        "H=Function(H:(t,xk[3],uk,b_0,b_1,b_2,a_0,a_1,a_2,a_3)->(yk) SXFunction), "
        "n=3, nu=1, ny=1, dt=None, "
        "params={'b_0': SX(b_0), 'b_1': SX(b_1), 'b_2': SX(b_2), "
        "'a_0': SX(a_0), 'a_1': SX(a_1), 'a_2': SX(a_2), 'a_3': SX(a_3)}, "
        "name=None, "
        "input_names=['u'], state_names=['x1', 'x2', 'x3'], "
        "output_names=['y'])"
    )


@pytest.mark.parametrize(
    "sys_num,test_case_fixture",
    [
        (1, "tf_test_case_1"),
        (2, "tf_test_case_2"),
        (3, "tf_test_case_3"),
        (4, "tf_test_case_4"),
        (5, "tf_test_case_5"),
    ],
)
def test_StateSpaceModelDTTFSISO(
    sys_num, test_case_fixture, request, data_tf2ss_simulation
):
    """Test StateSpaceModelDTTFSISO simulation matches Octave output.

    This tests that StateSpaceModelDTTFSISO produces the same output
    as Octave's ss(tf(...)) when simulated with the same input signal.
    """
    test_case = request.getfixturevalue(test_case_fixture)

    # Convert to CasADi vectors
    num = cas.DM(test_case["num"])
    den = cas.DM(test_case["den"])

    # Create model instance
    model = StateSpaceModelDTTFSISO(num=num, den=den)

    # Model dimensions
    n = model.n
    assert model.nu == 1
    assert model.ny == 1

    # Test model type identification
    assert is_ss_ct(model) is False
    assert is_ss_dt(model) is True

    # Load test data
    t = cas.DM(data_tf2ss_simulation["t"].to_numpy())
    nT = t.shape[0] - 1
    u = cas.DM(data_tf2ss_simulation["u"].to_numpy())
    y_octave = data_tf2ss_simulation[f"sys{sys_num}_y"].to_numpy()

    # Simulate model from zero initial conditions
    xk = cas.DM.zeros(n)
    y = []
    for k in range(nT + 1):
        yk = model.H(t[k], xk, u[k])
        y.append(yk)
        xk = model.F(t[k], xk, u[k])

    y = cas.vcat(y)

    # Compare output with Octave simulation results
    # Don't compare states since Octave uses different state-space form
    # Use looser tolerance for higher-order systems with complex poles
    atol = 1e-2 if sys_num == 5 else 1e-6
    assert np.allclose(np.array(y).flatten(), y_octave, atol=atol), (
        f"System {sys_num} output doesn't match Octave output"
    )


def test_StateSpaceModelDTARXSISO(data_TP04_Q1a_ss):
    # ARX(2, 2, 1) model with fixed coefficients
    A = [-1.40429502, 0.69767633]
    B = [0.18669536, 0.16536220]
    model = StateSpaceModelDTARXSISO(A=A, B=B)

    assert str(model) == (
        "StateSpaceModelDTARXSISO("
        "F=Function(F:(t,xk[3],uk)->(xkp1[3]) SXFunction), "
        "H=Function(H:(t,xk[3],uk)->(yk) SXFunction), "
        "n=3, nu=1, ny=1, dt=None, "
        "params={}, name=None, "
        "input_names=['u'], state_names=['x1', 'x2', 'x3'], "
        "output_names=['y']"
        ")"
    )

    # Model dimensions
    n = model.n
    assert model.nu == 1
    assert model.ny == 1

    # Test model type identification
    assert is_ss_ct(model) is False
    assert is_ss_dt(model) is True

    # Load test data
    t = cas.DM(data_TP04_Q1a_ss["t"].to_numpy())
    nT = t.shape[0] - 1
    u = cas.DM(data_TP04_Q1a_ss["u_data"].to_numpy())
    y_m = cas.DM(data_TP04_Q1a_ss["y_data"].to_numpy())

    # Simulate model
    xk = cas.DM.zeros(n)
    X = []
    y = []
    for k in range(nT + 1):
        yk = model.H(t[k], xk, u[k])
        y.append(yk)
        X.append(xk.T)
        xk = model.F(t[k], xk, u[k])

    X = cas.vcat(X)
    y = cas.vcat(y)

    # Compare with Octave simulation results
    X_octave = data_TP04_Q1a_ss[["x1", "x2", "x3"]].to_numpy()
    y_octave = data_TP04_Q1a_ss["y"].to_numpy().reshape(-1, 1)

    # Verify states match
    assert np.allclose(np.array(X), X_octave, atol=1e-6), (
        "States don't match Octave output"
    )

    # Verify outputs match
    assert np.allclose(np.array(y), y_octave, atol=1e-6), (
        "Outputs don't match Octave output"
    )

    # ARX(2, 2, 1) model with symbolic coefficients
    na = 2
    nb = 2
    model = StateSpaceModelDTARXSISO(na=na, nb=nb)

    assert str(model) == (
        "StateSpaceModelDTARXSISO("
        "F=Function(F:(t,xk[3],uk,a_0,a_1,b_0,b_1)->(xkp1[3]) SXFunction), "
        "H=Function(H:(t,xk[3],uk,a_0,a_1,b_0,b_1)->(yk) SXFunction), "
        "n=3, nu=1, ny=1, dt=None, "
        "params={'a_0': SX(a_0), 'a_1': SX(a_1), 'b_0': SX(b_0), 'b_1': SX(b_1)}, "
        "name=None, "
        "input_names=['u'], state_names=['x1', 'x2', 'x3'], "
        "output_names=['y']"
        ")"
    )

    # Simulate model
    xk = cas.DM.zeros(n)
    X = []
    y = []
    for k in range(nT + 1):
        yk = model.H(t[k], xk, u[k], *A, *B)
        y.append(yk)
        X.append(xk.T)
        xk = model.F(t[k], xk, u[k], *A, *B)

    X = cas.vcat(X)
    y = cas.vcat(y)

    # Verify states match
    assert np.allclose(np.array(X), X_octave, atol=1e-6), (
        "States don't match Octave output"
    )

    # Verify outputs match
    assert np.allclose(np.array(y), y_octave, atol=1e-6), (
        "Outputs don't match Octave output"
    )

    # Simulate symbolically
    xk = cas.DM.zeros(n)
    X = []
    y = []
    for k in range(nT + 1):
        yk = model.H(t[k], xk, u[k], *model.params.values())
        y.append(yk)
        X.append(xk.T)
        xk = model.F(t[k], xk, u[k], *model.params.values())

    X = cas.vcat(X)
    y = cas.vcat(y)

    # Construct CasADi function to compute prediction error
    calculate_sumsq_error = cas.Function(
        "calculate_sumsq_error",
        model.params.values(),
        [cas.sumsqr(y - y_m) / (nT + 1)],
        model.params.keys(),
        ["sumsq_error"],
    )

    opti = cas.Opti()
    params = {}
    for name, param in model.params.items():
        params[name] = opti.variable(param.shape[0])

    prediction_error = calculate_sumsq_error(*params.values())
    opti.minimize(prediction_error)

    # Solve with nonlinear solver (suppress verbose output)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
    sol = opti.solve()

    assert sol.stats()["return_status"] == "Solve_Succeeded"
    assert sol.value(prediction_error) < 1e-10

    params_sol = [sol.value(v) for v in params.values()]
    assert np.allclose(
        params_sol, [-1.40429502, 0.69767633, 0.18669536, 0.16536220]
    )


# TODO: Introduce this class as an intermediate step to state space models.
# def test_StateSpaceModelDTFromABCD_FO_SISO(symbolic_FO_SISO):
#     _, _, _, A, B, C, D, _, _, _, _ = symbolic_FO_SISO

#     model = StateSpaceModelDTFromABCD(A, B, C, D)

#     assert str(model) == (
#         "StateSpaceModelDTFromABCD("
#         "f=Function(f:(t,x,u,K,T1)->(rhs) SXFunction), "
#         "h=Function(h:(t,x,u,K,T1)->(y) SXFunction), "
#         "n=1, nu=1, ny=1, dt=None, "
#         "params={'K': SX(K), 'T1': SX(T1)}, "
#         "input_names=['u'], state_names=['x'], output_names=['y']"
#         ")"
#     )

#     # Test function calls - with scalars
#     assert np.array_equal(model.F(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
#     assert np.array_equal(model.H(0.0, 0.0, 0.0, 1.0, 2.0), np.array([[0.0]]))
#     assert np.array_equal(model.F(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[-0.5]]))
#     assert np.array_equal(model.H(0.0, 1.0, 0.0, 1.0, 2.0), np.array([[0.5]]))


# def test_StateSpaceModelDTFromABCD_O2_SISO(symbolic_O2_SISO):
#     _, _, _, A, B, C, D, _, _, _, _ = symbolic_O2_SISO

#     model = StateSpaceModelDTFromABCD(A, B, C, D)

#     assert str(model) == (
#         "StateSpaceModelCTFromABCD("
#         "f=Function(f:(t,x[2],u,K,T1,T2)->(rhs[2]) SXFunction), "
#         "h=Function(h:(t,x[2],u,K,T1,T2)->(y) SXFunction), "
#         "n=2, nu=1, ny=1, dt=None, "
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


def test_StateSpaceModelDTFromCTRK4():
    """Test conversion from continuous-time to discrete-time using RK4."""
    # Create a simple continuous-time first-order system
    # dx/dt = -a*x + b*u
    # y = x
    n = 1
    nu = 1
    ny = 1
    a = cas.SX.sym("a")
    b = cas.SX.sym("b")
    params = {"a": a, "b": b}

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    rhs = -a * x + b * u
    y = x

    f = cas.Function(
        "f", [t, x, u, a, b], [rhs], ["t", "x", "u", "a", "b"], ["rhs"]
    )
    h = cas.Function(
        "h", [t, x, u, a, b], [y], ["t", "x", "u", "a", "b"], ["y"]
    )

    model_ct = StateSpaceModelCT(f, h, n, nu, ny, params=params)

    # Convert to discrete-time with dt=0.1
    dt = 0.1
    model_dt = StateSpaceModelDTFromCTRK4(model_ct, dt)

    # Verify the discrete-time model
    assert model_dt.n == n
    assert model_dt.nu == nu
    assert model_dt.ny == ny
    assert model_dt.dt == dt
    assert len(model_dt.params) == len(params)

    # Test with numeric parameter values
    a_val = 2.0
    b_val = 1.5
    x0 = np.array([1.0])
    u_val = 0.5
    t_val = 0.0

    # Compute next state
    x1 = model_dt.F(t_val, x0, u_val, a_val, b_val)
    y0 = model_dt.H(t_val, x0, u_val, a_val, b_val)

    # Output should equal state
    assert np.allclose(y0, x0)

    # State should have evolved (not equal to x0 due to dynamics)
    assert not np.allclose(x1, x0)

    # For a = 2.0, b = 1.5, u = 0.5, x = 1.0
    # dx/dt = -2*1 + 1.5*0.5 = -2 + 0.75 = -1.25
    # After dt=0.1 with RK4, x should decrease (roughly by 1.25*0.1 = 0.125)
    # but RK4 is more accurate than Euler
    assert x1 < x0  # State should decrease


def test_StateSpaceModelDTFromCT():
    """Test conversion from continuous-time to discrete-time using CasADi
    integrator.
    """
    # Create a simple continuous-time first-order system
    # dx/dt = -a*x + b*u
    # y = x
    n = 1
    nu = 1
    ny = 1
    a = cas.SX.sym("a")
    b = cas.SX.sym("b")
    params = {"a": a, "b": b}

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    rhs = -a * x + b * u
    y = x

    f = cas.Function(
        "f", [t, x, u, a, b], [rhs], ["t", "x", "u", "a", "b"], ["rhs"]
    )
    h = cas.Function(
        "h", [t, x, u, a, b], [y], ["t", "x", "u", "a", "b"], ["y"]
    )

    model_ct = StateSpaceModelCT(f, h, n, nu, ny, params=params)

    # Convert to discrete-time with dt=0.1 using default cvodes solver
    dt = 0.1
    model_dt = StateSpaceModelDTFromCT(model_ct, dt)

    # Verify the discrete-time model
    assert model_dt.n == n
    assert model_dt.nu == nu
    assert model_dt.ny == ny
    assert model_dt.dt == dt
    assert len(model_dt.params) == len(params)

    # Test with numeric parameter values
    a_val = 2.0
    b_val = 1.5
    x0 = np.array([1.0])
    u_val = 0.5
    t_val = 0.0

    # Compute next state
    x1 = model_dt.F(t_val, x0, u_val, a_val, b_val)
    y0 = model_dt.H(t_val, x0, u_val, a_val, b_val)

    # Output should equal state
    assert np.allclose(y0, x0)

    # State should have evolved
    assert not np.allclose(x1, x0)
    assert x1 < x0  # State should decrease


def test_StateSpaceModelDTFromCT_compare_solvers():
    """Compare RK4 and cvodes integrators for CT to DT conversion."""
    # Create a simple continuous-time first-order system
    n = 1
    nu = 1
    ny = 1
    a = cas.SX.sym("a")
    b = cas.SX.sym("b")
    params = {"a": a, "b": b}

    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    rhs = -a * x + b * u
    y = x

    f = cas.Function(
        "f", [t, x, u, a, b], [rhs], ["t", "x", "u", "a", "b"], ["rhs"]
    )
    h = cas.Function(
        "h", [t, x, u, a, b], [y], ["t", "x", "u", "a", "b"], ["y"]
    )

    model_ct = StateSpaceModelCT(f, h, n, nu, ny, params=params)

    # Convert using both methods
    dt = 0.1
    model_rk4 = StateSpaceModelDTFromCTRK4(model_ct, dt)
    model_cvodes = StateSpaceModelDTFromCT(
        model_ct,
        dt,
        solver="cvodes",
        integrator_opts={"abstol": 1e-10, "reltol": 1e-10},
    )

    # Test with numeric values
    a_val = 2.0
    b_val = 1.5
    x0 = np.array([1.0])
    u_val = 0.5
    t_val = 0.0

    # Compute next state with both methods
    x1_rk4 = model_rk4.F(t_val, x0, u_val, a_val, b_val)
    x1_cvodes = model_cvodes.F(t_val, x0, u_val, a_val, b_val)

    # Results should be very close (both are high-accuracy integrators)
    assert np.allclose(x1_rk4, x1_cvodes, rtol=1e-5, atol=1e-7)

    # Test with RK integrator as well
    model_rk = StateSpaceModelDTFromCT(
        model_ct,
        dt,
        solver="rk",
        integrator_opts={"number_of_finite_elements": 4},
    )
    x1_rk = model_rk.F(t_val, x0, u_val, a_val, b_val)

    # RK and RK4 should also be close
    assert np.allclose(x1_rk4, x1_rk, rtol=1e-4, atol=1e-6)


def test_StateSpaceModelDTDelay_SISO():
    """Test SISO delay model."""
    nk = 5
    model = StateSpaceModelDTDelay(nk)

    # Verify model dimensions
    assert model.n == nk
    assert model.nu == 1
    assert model.ny == 1
    assert model.nk == nk
    assert model.G.shape == (1, 1)
    assert float(model.G) == 1.0

    # Test step response - output should be delayed by nk steps
    x = cas.DM.zeros(model.n, 1)
    u_val = 1.0
    t_val = 0.0

    for k in range(15):
        y = model.H(t_val, x, u_val)

        # Output should be 0 before delay, 1 after delay
        if k < nk:
            assert float(y) == 0.0, (
                f"At k={k}, expected y=0 but got {float(y)}"
            )
        else:
            assert float(y) == 1.0, (
                f"At k={k}, expected y=1 but got {float(y)}"
            )

        x = model.F(t_val, x, u_val)
        t_val += 1


def test_StateSpaceModelDTDelay_MIMO_identity():
    """Test MIMO delay model with identity gain matrix."""
    nu = 2
    nk = 3
    model = StateSpaceModelDTDelay(nk, nu)

    # Verify model dimensions
    assert model.n == nk * nu
    assert model.nu == nu
    assert model.ny == nu  # ny defaults to nu when G is None
    assert model.nk == nk
    assert model.G.shape == (nu, nu)
    assert np.allclose(np.array(model.G), np.eye(nu))

    # Test step response with different values on each input
    x = cas.DM.zeros(model.n, 1)
    u_val = cas.DM([[1.0], [2.0]])
    t_val = 0.0

    for k in range(10):
        y = np.array(model.H(t_val, x, u_val)).flatten()

        # Outputs should be delayed versions of inputs
        if k < nk:
            assert np.allclose(y, [0.0, 0.0]), (
                f"At k={k}, expected y=[0, 0] but got {y}"
            )
        else:
            assert np.allclose(y, [1.0, 2.0]), (
                f"At k={k}, expected y=[1, 2] but got {y}"
            )

        x = model.F(t_val, x, u_val)
        t_val += 1


def test_StateSpaceModelDTDelay_MIMO_custom_gain():
    """Test MIMO delay model with custom gain matrix."""
    nu = 2
    nk = 3
    G = cas.DM([[1.0, 0.5], [0.0, 2.0]])
    model = StateSpaceModelDTDelay(nk, nu, G=G)

    # Verify model dimensions
    ny = G.shape[0]  # ny is inferred from G
    assert model.n == nk * nu
    assert model.nu == nu
    assert model.ny == ny
    assert model.nk == nk
    assert model.G.shape == (ny, nu)
    assert np.allclose(np.array(model.G), np.array(G))

    # Test step response
    x = cas.DM.zeros(model.n, 1)
    u_val = cas.DM([[1.0], [2.0]])
    t_val = 0.0

    # Expected output after delay: y = G * u
    # y[0] = 1.0*1.0 + 0.5*2.0 = 2.0
    # y[1] = 0.0*1.0 + 2.0*2.0 = 4.0
    y_expected = np.array([2.0, 4.0])

    for k in range(10):
        y = np.array(model.H(t_val, x, u_val)).flatten()

        # Outputs should be zero before delay, G*u after delay
        if k < nk:
            assert np.allclose(y, [0.0, 0.0]), (
                f"At k={k}, expected y=[0, 0] but got {y}"
            )
        else:
            assert np.allclose(y, y_expected), (
                f"At k={k}, expected y={y_expected} but got {y}"
            )

        x = model.F(t_val, x, u_val)
        t_val += 1


def test_StateSpaceModelDTDelay_varying_input():
    """Test delay model with time-varying input."""
    nk = 4
    model = StateSpaceModelDTDelay(nk)

    # Simulate with varying input signal
    x = cas.DM.zeros(model.n, 1)
    t_val = 0.0

    # Input sequence: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Expected output (delayed by 4): [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    for k in range(10):
        u_val = float(k)
        y = float(model.H(t_val, x, u_val))

        # Output should be input from nk steps ago
        if k < nk:
            y_expected = 0.0
        else:
            y_expected = float(k - nk)

        assert np.isclose(y, y_expected), (
            f"At k={k}, expected y={y_expected} but got {y}"
        )

        x = model.F(t_val, x, u_val)
        t_val += 1


def test_StateSpaceModelDTDelay_edge_cases():
    """Test edge cases for delay model."""
    # Test nk=1 (minimum delay)
    model = StateSpaceModelDTDelay(1)
    assert model.n == 1

    x = cas.DM.zeros(1, 1)
    y = float(model.H(0.0, x, 1.0))
    assert y == 0.0  # Initially zero

    x = model.F(0.0, x, 1.0)
    y = float(model.H(1.0, x, 1.0))
    assert y == 1.0  # After 1 step, output equals previous input

    # Test non-square MIMO system
    G = cas.DM([[1.0, 0.5, 0.2], [0.3, 0.0, 1.5]])
    model = StateSpaceModelDTDelay(2, nu=3, G=G)
    assert model.n == 2 * 3  # nk * nu
    assert model.nu == 3
    assert model.ny == 2
    assert model.G.shape == (2, 3)
    assert np.allclose(np.array(model.G), np.array(G))


def test_StateSpaceModelDTDelay_compare_with_TF():
    """Compare StateSpaceModelDTDelay with StateSpaceModelDTTFSISO for pure delay."""
    nk = 5

    # Create delay model
    model_delay = StateSpaceModelDTDelay(nk)

    # Create equivalent TF model: G(z) = z^(-nk)
    num = cas.DM([1])
    den = cas.DM([1] + [0] * nk)
    model_tf = StateSpaceModelDTTFSISO(num=num, den=den)

    # Both should have same number of states
    assert model_delay.n == model_tf.n == nk

    # Simulate both with same input
    x_delay = cas.DM.zeros(model_delay.n, 1)
    x_tf = cas.DM.zeros(model_tf.n, 1)
    u_val = 1.0
    t_val = 0.0

    for k in range(12):
        y_delay = float(model_delay.H(t_val, x_delay, u_val))
        y_tf = float(model_tf.H(t_val, x_tf, u_val))

        # Outputs should match
        assert np.isclose(y_delay, y_tf), (
            f"At k={k}, delay model y={y_delay} but TF model y={y_tf}"
        )

        x_delay = model_delay.F(t_val, x_delay, u_val)
        x_tf = model_tf.F(t_val, x_tf, u_val)
        t_val += 1


def test_mul_operator_series_connection():
    """Test the * operator for connecting discrete-time systems in series"""
    sys1 = StateSpaceModelDTTFSISO(num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5]))
    sys2 = StateSpaceModelDTTFSISO(num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3]))

    # Use * operator to connect in series
    sys_combined = sys1 * sys2

    # Verify it creates a combined system
    assert sys_combined.n == 2  # Combined states from both systems
    assert sys_combined.nu == 1  # Single input (from sys1)
    assert sys_combined.ny == 1  # Single output (from sys2)
    assert sys_combined.input_names == ["u"]  # No prefix (no conflict)
    assert sys_combined.output_names == ["y"]  # No prefix (no conflict)
    assert "x" in sys_combined.state_names[0]
    assert "x" in sys_combined.state_names[1]

    # Test chaining multiple systems: sys1 * sys2 * sys3
    sys3 = StateSpaceModelDTTFSISO(num=cas.DM([0, 0.5]), den=cas.DM([1, -0.2]))
    sys_combined_3 = sys1 * sys2 * sys3

    # Verify three-system chain
    assert sys_combined_3.n == 3  # All three states
    assert sys_combined_3.nu == 1  # Single input
    assert sys_combined_3.ny == 1  # Single output
    assert sys_combined_3.input_names == ["u"]  # No prefix (no conflict)
    assert sys_combined_3.output_names == ["y"]  # No prefix (no conflict)

    # Verify functionality: output of combined system equals
    # output of sys3 when fed sys2's output, fed sys1's output
    t_val = 0.0
    x1 = cas.DM([0.0])
    x2 = cas.DM([0.0])
    x3 = cas.DM([0.0])
    x_combined = cas.DM.zeros(3, 1)
    u_val = cas.DM([1.0])

    # Single step through individual systems
    y1 = sys1.H(t_val, x1, u_val)
    y2 = sys2.H(t_val, x2, y1)  # sys2 input is sys1 output
    y3 = sys3.H(t_val, x3, y2)  # sys3 input is sys2 output

    # Single step through combined system
    y_combined = sys_combined_3.H(t_val, x_combined, u_val)

    # Outputs should match
    assert np.isclose(float(y3), float(y_combined))


def test_add_operator_parallel_connection():
    """Test the + operator connects discrete-time systems with shared input and summed outputs."""
    sys1 = StateSpaceModelDTTFSISO(num=cas.DM([0, 2.0]), den=cas.DM([1, -0.5]))
    sys2 = StateSpaceModelDTTFSISO(num=cas.DM([0, 1.0]), den=cas.DM([1, -0.3]))

    sys_combined = sys1 + sys2

    # States stacked, nu and ny unchanged
    assert sys_combined.n == 2
    assert sys_combined.nu == 1
    assert sys_combined.ny == 1
    assert sys_combined.input_names == sys1.input_names
    assert sys_combined.output_names == sys1.output_names
    assert is_ss_ct(sys_combined) is False
    assert is_ss_dt(sys_combined) is True

    # Output must equal sum of individual outputs
    t_val = cas.DM(0.0)
    x1 = cas.DM([1.5])
    x2 = cas.DM([0.8])
    x = cas.vertcat(x1, x2)
    u = cas.DM([1.0])
    y1 = float(sys1.H(t_val, x1, u))
    y2 = float(sys2.H(t_val, x2, u))
    y_combined = float(sys_combined.H(t_val, x, u))
    assert np.isclose(y_combined, y1 + y2)

    # + is equivalent to sum_systems
    sys_via_func = sum_systems([sys1, sys2], model_class=StateSpaceModelDT)
    assert str(sys_combined) == str(sys_via_func)
