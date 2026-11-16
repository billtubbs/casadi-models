"""Unit tests for src/cas_models/discrete_time/kalman.py module"""

from pathlib import Path

import casadi as cas
import control
import numpy as np
import pandas as pd
import pytest
import scipy

from cas_models.discrete_time.kalman import (
    EKFF,
    EKFP,
    KFF,
    KFFSS,
    KFP,
    KFPSS,
    ekf_update,
    kalman_predict_f,
    kalman_update_f,
    kf_update,
    linear_model_output,
    linear_model_state_transition,
    make_steady_state_KFF_with_fixedABCDQR,
    make_steady_state_KFP_with_fixedABCDQR,
    steady_state_kf_gain,
)
from cas_models.continuous_time.models import SSModelCTLinearFOSISO
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTFromABCD,
    StateSpaceModelDTFromCTRK4,
)
from cas_models.discrete_time.simulate import (
    make_n_step_simulation_function_from_model,
)
from cas_models.transformations import connect_systems

# Test system used in the tests below: 2 states, 1 input, 2 outputs
A_test = np.array([[0.7, 1.0], [0.0, 1.0]])
B_test = np.array([[1.0], [0.0]])
C_test = np.array([[0.3, 0.0], [0.0, 1.0]])
D_test = np.zeros((2, 1))
Q_test = np.diag([0.01, 0.01])
R_test = np.diag([0.1, 0.1])


def _test_matrices():
    return tuple(map(cas.DM, (A_test, B_test, C_test, D_test, Q_test, R_test)))


def np_kalman_update(A, B, C, D, xk_pred, Pk_pred, Q, R, yk, uk):
    """NumPy version of the original JAX code (1-D vectors),
    used as a reference to test the CasADi functions.
    """
    yk_pred = C @ xk_pred + D @ uk
    Sk_pred = C @ Pk_pred @ C.T + R
    Kf = np.linalg.lstsq(Sk_pred.T, (Pk_pred @ C.T).T, rcond=-1)[0].T
    xk_est = xk_pred + Kf @ (yk - yk_pred)
    z = np.eye(A.shape[0]) - Kf @ C
    Pk = z @ Pk_pred @ z.T + Kf @ R @ Kf.T
    Pk = (Pk + Pk.T) / 2
    xkp1_est = A @ xk_est + B @ uk
    Pkp1 = A @ Pk @ A.T + Q
    return xk_est, Pk, xkp1_est, Pkp1, yk_pred, Kf


def test_linear_model_functions():
    A, B, C, D = map(cas.DM, (A_test, B_test, C_test, D_test))
    xk = cas.DM([1.0, 2.0])
    uk = cas.DM([0.5])

    xkp1 = linear_model_state_transition(A, B, xk, uk)
    assert isinstance(xkp1, cas.DM)
    assert xkp1.shape == (2, 1)
    np.testing.assert_allclose(xkp1, [[0.7 + 2.0 + 0.5], [2.0]])

    yk = linear_model_output(C, D, xk, uk)
    assert yk.shape == (2, 1)
    np.testing.assert_allclose(yk, [[0.3], [2.0]])

    # Non-zero D
    D = cas.DM([[1.0], [-1.0]])
    yk = linear_model_output(C, D, xk, uk)
    np.testing.assert_allclose(yk, [[0.3 + 0.5], [2.0 - 0.5]])

    # No inputs (nu = 0)
    xkp1 = linear_model_state_transition(
        A, cas.DM.zeros(2, 0), xk, cas.DM.zeros(0)
    )
    np.testing.assert_allclose(xkp1, A_test @ [[1.0], [2.0]])

    # Symbolic arguments
    for T in (cas.SX, cas.MX):
        xs, us = T.sym("x", 2), T.sym("u", 1)
        f = cas.Function(
            "f", [xs, us], [linear_model_state_transition(A, B, xs, us)]
        )
        np.testing.assert_allclose(f(xk, uk), [[3.2], [2.0]])


def test_kalman_update_f():
    # Scalar system with known solution:
    # S = 1 * 1 * 1 + 1 = 2, Kf = 1 / 2,
    # x(k|k) = 0 + 0.5 * (4 - 0) = 2,
    # P(k|k) = 0.5^2 * 1 + 0.5^2 * 1 = 0.5
    one = cas.DM(1.0)
    xk_est, Pk, Kf, Sk = kalman_update_f(
        one, one, cas.DM(0.0), one, cas.DM(0.0), cas.DM(4.0)
    )
    for value in (xk_est, Pk, Kf, Sk):
        assert isinstance(value, cas.DM)
        assert value.shape == (1, 1)
    np.testing.assert_allclose(Sk, 2.0)
    np.testing.assert_allclose(Kf, 0.5)
    np.testing.assert_allclose(xk_est, 2.0)
    np.testing.assert_allclose(Pk, 0.5)

    # 2-state system compared to the standard form of the equations
    C, R = cas.DM(C_test), cas.DM(R_test)
    xk_pred = cas.DM([1.0, -1.0])
    Pk_pred = cas.DM([[2.0, 0.5], [0.5, 1.0]])
    yk_pred = C @ xk_pred
    yk = cas.DM([0.5, 0.0])
    xk_est, Pk, Kf, Sk = kalman_update_f(C, R, xk_pred, Pk_pred, yk_pred, yk)
    P, Cn = np.array(Pk_pred), C_test
    S = Cn @ P @ Cn.T + R_test
    K = P @ Cn.T @ np.linalg.inv(S)
    assert Kf.shape == (2, 2)
    np.testing.assert_allclose(Sk, S)
    np.testing.assert_allclose(Kf, K)
    np.testing.assert_allclose(
        xk_est, np.array(xk_pred) + K @ np.array(yk - yk_pred)
    )
    np.testing.assert_allclose(Pk, P - K @ S @ K.T)
    np.testing.assert_allclose(Pk, Pk.T)

    # Symbolic arguments
    for T in (cas.SX, cas.MX):
        x, P = T.sym("x", 2), T.sym("P", 2, 2)
        y = T.sym("y", 2)
        f = cas.Function("f", [x, P, y], kalman_update_f(C, R, x, P, C @ x, y))
        for a, b in zip(f(xk_pred, Pk_pred, yk), (xk_est, Pk, Kf, Sk)):
            np.testing.assert_allclose(a, b)


def test_kalman_predict_f():
    A, B, Q = map(cas.DM, (A_test, B_test, Q_test))
    xk_est = cas.DM([1.0, 2.0])
    Pk = cas.DM([[2.0, 0.5], [0.5, 1.0]])
    uk = cas.DM([0.5])

    xkp1_est, Pkp1 = kalman_predict_f(A, B, Q, xk_est, Pk, uk)
    assert xkp1_est.shape == (2, 1)
    assert Pkp1.shape == (2, 2)
    np.testing.assert_allclose(xkp1_est, [[3.2], [2.0]])
    np.testing.assert_allclose(Pkp1, A_test @ np.array(Pk) @ A_test.T + Q_test)

    # Symbolic arguments
    for T in (cas.SX, cas.MX):
        x, P, u = T.sym("x", 2), T.sym("P", 2, 2), T.sym("u", 1)
        f = cas.Function("f", [x, P, u], kalman_predict_f(A, B, Q, x, P, u))
        for a, b in zip(f(xk_est, Pk, uk), (xkp1_est, Pkp1)):
            np.testing.assert_allclose(a, b)


def test_kf_update():
    A, B, C, D, Q, R = _test_matrices()

    # Compare to NumPy reference over 50 time steps with random data
    rng = np.random.default_rng(0)
    xkp1_est, Pkp1 = cas.DM.zeros(2), cas.DM.eye(2)
    xkp1_np, Pkp1_np = np.zeros(2), np.eye(2)
    for k in range(50):
        yk, uk = rng.normal(size=2), rng.normal(size=1)
        results = kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, yk, uk)
        results_np = np_kalman_update(
            A_test,
            B_test,
            C_test,
            D_test,
            xkp1_np,
            Pkp1_np,
            Q_test,
            R_test,
            yk,
            uk,
        )
        for a, b in zip(results, results_np):
            assert isinstance(a, cas.DM)
            np.testing.assert_allclose(
                np.array(a).reshape(b.shape), b, atol=1e-12
            )
        xkp1_est, Pkp1 = results[2], results[3]
        xkp1_np, Pkp1_np = results_np[2], results_np[3]

    # After many steps the gain should converge to the steady-state
    # Kalman filter gain from the discrete algebraic Riccati equation
    P_ss = scipy.linalg.solve_discrete_are(A_test.T, C_test.T, Q_test, R_test)
    Kf_ss = P_ss @ C_test.T @ np.linalg.inv(C_test @ P_ss @ C_test.T + R_test)
    np.testing.assert_allclose(Pkp1, P_ss, atol=1e-8)
    np.testing.assert_allclose(results[5], Kf_ss, atol=1e-8)

    # Lists and floats are accepted as yk and uk
    results = kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, [1.0, 2.0], 0.5)
    assert results[0].shape == (2, 1)

    # No inputs (nu = 0), uk omitted
    results = kf_update(
        A,
        cas.DM.zeros(2, 0),
        C,
        cas.DM.zeros(2, 0),
        cas.DM.zeros(2),
        cas.DM.eye(2),
        Q,
        R,
        [1.0, 2.0],
    )
    results_np = np_kalman_update(
        A_test,
        np.zeros((2, 0)),
        C_test,
        np.zeros((2, 0)),
        np.zeros(2),
        np.eye(2),
        Q_test,
        R_test,
        np.array([1.0, 2.0]),
        np.zeros(0),
    )
    for a, b in zip(results, results_np):
        np.testing.assert_allclose(np.array(a).reshape(b.shape), b)

    # Wrong-sized measurement raises an error
    with pytest.raises(AssertionError, match="yk.shape"):
        kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, [1.0], 0.5)

    # Symbolic arguments: build a cas.Function of one time step
    for T in (cas.SX, cas.MX):
        x, P = T.sym("x", 2), T.sym("P", 2, 2)
        y, u = T.sym("y", 2), T.sym("u", 1)
        f = cas.Function(
            "kf_update",
            [x, P, y, u],
            list(kf_update(A, B, C, D, x, P, Q, R, y, u)),
        )
        expected = kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, yk, uk)
        for a, b in zip(f(xkp1_est, Pkp1, yk, uk), expected):
            np.testing.assert_allclose(a, b)


def _kf_test_point():
    """Filter state x_est(k|k-1), P(k|k-1) and inputs u(k), y(k) for
    one-step tests, and the expected kf_update results."""
    A, B, C, D, Q, R = _test_matrices()
    xkp1_est = cas.DM([0.1, -0.2])
    Pkp1 = cas.DM([[0.25, 0.05], [0.05, 0.3]])
    xk = cas.vertcat(xkp1_est, cas.vec(Pkp1))
    uk, yk = cas.DM([0.7]), cas.DM([0.4, -0.3])
    expected = kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, yk, uk)
    return xk, cas.vertcat(uk, yk), expected


def _check_KF_system(kf, est_name):
    """Checks common to both forms of the filter."""
    assert kf.name is None
    assert kf.dt == 0.5
    assert (kf.n, kf.nu, kf.ny) == (6, 3, 8)
    assert kf.input_names == ["uk", "yk_1", "yk_2"]
    assert kf.state_names == [
        "xkp1_est_1",
        "xkp1_est_2",
        "Pkp1_1",
        "Pkp1_2",
        "Pkp1_3",
        "Pkp1_4",
    ]
    x, y = f"x{est_name}", f"y{est_name}"
    assert kf.output_names == [
        f"{x}_1",
        f"{x}_2",
        f"{y}_1",
        f"{y}_2",
        f"{x}_var_1",
        f"{x}_var_2",
        f"{y}_var_1",
        f"{y}_var_2",
    ]

    # State transition: one step of kf_update, with inputs ordered as
    # named (u(k), y(k))
    xk, uk, expected = _kf_test_point()
    xkp1 = kf.F(0.0, xk, uk)
    np.testing.assert_allclose(xkp1[:2], expected[2])
    np.testing.assert_allclose(xkp1[2:], cas.vec(expected[3]))

    # The variances don't depend on the inputs (only on P(k|k-1))
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(np.array(cas.DM(sp, 1))[4:], 0)


def test_KFP():
    A, B, C, D, Q, R = _test_matrices()
    all_outputs = dict(
        include_output_estimates=True,
        include_state_variance=True,
        include_output_variance=True,
    )
    kf = KFP(A, B, C, D, Q, R, dt=0.5, **all_outputs)
    _check_KF_system(kf, "kp1_est")

    # Outputs: x_est(k+1|k), C x_est(k+1|k), diag(P(k+1|k)),
    # diag(C P(k+1|k) C')
    xk, uk, expected = _kf_test_point()
    xkp1_est, Pkp1 = expected[2], expected[3]
    y = kf.H(0.0, xk, uk)
    np.testing.assert_allclose(y[0:2], xkp1_est)
    np.testing.assert_allclose(y[2:4], C @ xkp1_est)
    np.testing.assert_allclose(y[4:6], cas.diag(Pkp1))
    np.testing.assert_allclose(y[6:8], cas.diag(C @ Pkp1 @ C.T))

    # The estimates depend directly on the inputs (feedthrough):
    # on u(k) (except x2 and y2, since B[1] = 0) and y(k)
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(
        np.array(cas.DM(sp, 1))[:4],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]],
    )

    # Default and selected outputs
    kf = KFP(A, B, C, D, Q, R)
    assert kf.output_names == ["xkp1_est_1", "xkp1_est_2"]
    kf = KFP(
        A,
        B,
        C,
        D,
        Q,
        R,
        include_state_estimates=False,
        include_state_variance=True,
    )
    assert kf.output_names == ["xkp1_est_var_1", "xkp1_est_var_2"]

    # y_est(k+1|k) needs D = 0
    D1 = cas.DM([[0.0], [1.0]])
    with pytest.raises(ValueError, match="unless D = 0"):
        KFP(A, B, C, D1, Q, R, include_output_estimates=True)
    kf = KFP(A, B, C, D1, Q, R, include_output_variance=True)
    assert kf.ny == 4

    # NumPy arrays are accepted
    kf_np = KFP(A_test, B_test, C_test, D_test, Q_test, R_test)
    np.testing.assert_allclose(kf_np.H(0.0, xk, uk), xkp1_est)


def test_KFF():
    A, B, C, D, Q, R = _test_matrices()
    all_outputs = dict(
        include_output_estimates=True,
        include_state_variance=True,
        include_output_variance=True,
    )
    kf = KFF(A, B, C, D, Q, R, dt=0.5, **all_outputs)
    _check_KF_system(kf, "k_est")

    # Outputs: x_est(k|k), y_est(k|k), diag(P(k|k)), diag(C P(k|k) C')
    xk, uk, expected = _kf_test_point()
    xk_est, Pk = expected[0], expected[1]
    yk_est = C @ xk_est + D @ uk[:1]
    y = kf.H(0.0, xk, uk)
    np.testing.assert_allclose(y[0:2], xk_est)
    np.testing.assert_allclose(y[2:4], yk_est)
    np.testing.assert_allclose(y[4:6], cas.diag(Pk))
    np.testing.assert_allclose(y[6:8], cas.diag(C @ Pk @ C.T))

    # The estimates depend directly on y(k) but not on u(k) (D = 0)
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(np.array(cas.DM(sp, 1))[:4, 0], 0)
    np.testing.assert_array_equal(np.array(cas.DM(sp, 1))[:4, 1:], 1)

    # With D non-zero, they depend on u(k) too
    D1 = cas.DM([[0.0], [1.0]])
    kf = KFF(A, B, C, D1, Q, R, include_output_estimates=True)
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(np.array(cas.DM(sp, 1))[:, 0], 1)

    # Default and selected outputs
    kf = KFF(A, B, C, D, Q, R)
    assert kf.output_names == ["xk_est_1", "xk_est_2"]
    kf = KFF(
        A,
        B,
        C,
        D,
        Q,
        R,
        include_state_estimates=False,
        include_output_variance=True,
    )
    assert kf.output_names == ["yk_est_var_1", "yk_est_var_2"]


KF_FORMS = {
    # form: (constructor, index of the state estimates in kf_update's
    # results, and of the plant states they estimate, relative to k)
    "prediction": (KFP, 2, 1),
    "filtering": (KFF, 0, 0),
}


@pytest.mark.parametrize("form", KF_FORMS)
def test_make_KF_simulation(form):
    """Multi-step simulation compared to the NumPy reference."""
    make_KF, i_est, _ = KF_FORMS[form]
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF(A, B, C, D, Q, R)
    nT = 20
    sim = make_n_step_simulation_function_from_model(kf, nT)
    rng = np.random.default_rng(1)
    U = rng.normal(size=(1, nT))
    Y = rng.normal(size=(2, nT))
    x0 = np.concatenate([np.zeros(2), np.eye(2).flatten(order="F")])
    t = np.arange(nT + 1) * 1.0
    X_kf, Y_kf = sim(t, np.vstack([U, Y]).T, x0)
    X_kf, Y_kf = np.array(X_kf), np.array(Y_kf)

    xkp1_np, Pkp1_np = np.zeros(2), np.eye(2)
    for k in range(nT):
        results_np = np_kalman_update(
            A_test,
            B_test,
            C_test,
            D_test,
            xkp1_np,
            Pkp1_np,
            Q_test,
            R_test,
            Y[:, k],
            U[:, k],
        )
        xkp1_np, Pkp1_np = results_np[2], results_np[3]
        np.testing.assert_allclose(Y_kf[k], results_np[i_est], atol=1e-12)
        np.testing.assert_allclose(X_kf[k + 1, :2], xkp1_np, atol=1e-12)
        np.testing.assert_allclose(
            X_kf[k + 1, 2:], Pkp1_np.flatten(order="F"), atol=1e-12
        )


def _simulate_plant_and_KF(
    form, Q, R, P0, x0_plant, x0_est, U, noise_sd, seed=0
):
    """Simulate the test system as the plant, add measurement noise to
    its outputs, and run the Kalman filter (same model as the plant)
    on the plant inputs and measurements.

    Returns X_est (nT, 2), the filter's state estimates at times
    0, ..., nT - 1, and X (nT, 2), the plant states they estimate:
    x(k+1) for the prediction form, x(k) for the filtering form.
    """
    make_KF, _, shift = KF_FORMS[form]
    nT = U.shape[0]
    t = np.arange(nT + 1) * 1.0
    plant = StateSpaceModelDTFromABCD(A_test, B_test, C_test, D_test)
    X, Y = make_n_step_simulation_function_from_model(plant, nT)(
        t, U, x0_plant
    )
    rng = np.random.default_rng(seed)
    Y_meas = np.array(Y)[:nT] + noise_sd * rng.normal(size=(nT, 2))
    kf = make_KF(A_test, B_test, C_test, D_test, Q, R)
    x0_kf = np.concatenate([x0_est, np.array(P0).flatten(order="F")])
    _, X_est = make_n_step_simulation_function_from_model(kf, nT)(
        t, np.hstack([U, Y_meas]), x0_kf
    )
    return np.array(X)[shift : nT + shift], np.array(X_est)[:nT]


@pytest.mark.parametrize("form", KF_FORMS)
def test_make_KF_tracks_plant_with_high_R_low_Q(form):
    """With very high measurement noise variances (R) and low model
    error variances (Q), the filter ignores the noisy measurements and
    reproduces the plant states (when it starts from the correct
    initial state)."""
    nT = 50
    U = np.random.default_rng(1).normal(size=(nT, 1))
    x0 = np.array([1.0, -1.0])
    Q, R = 1e-8 * np.eye(2), 1e6 * np.eye(2)
    X, X_est = _simulate_plant_and_KF(form, Q, R, Q, x0, x0, U, noise_sd=0.5)
    np.testing.assert_allclose(X_est, X, atol=1e-9)

    # For comparison, a filter that trusts the measurements (low R)
    # and not the model (high Q) follows the measurement noise
    Q, R = np.eye(2), 0.01 * np.eye(2)
    X, X_est = _simulate_plant_and_KF(form, Q, R, Q, x0, x0, U, noise_sd=0.5)
    assert np.abs(X_est - X).max() > 0.1


@pytest.mark.parametrize("form", KF_FORMS)
def test_make_KF_converges_with_low_R_high_Q(form):
    """With low measurement noise variances (R) and high model error
    variances (Q), the filter converges to the correct state estimates
    from an incorrect initial state when the plant is at steady
    state."""
    nT = 50
    u = 0.5
    x2 = -1.0
    x0 = np.array([(x2 + u) / 0.3, x2])  # steady state for u
    U = np.full((nT, 1), u)
    x0_est = np.zeros(2)
    Q, R = np.eye(2), 0.01 * np.eye(2)
    X, X_est = _simulate_plant_and_KF(
        form, Q, R, np.eye(2), x0, x0_est, U, noise_sd=0.0
    )
    np.testing.assert_allclose(X, np.tile(x0, (nT, 1)), atol=1e-12)
    assert np.abs(X_est[0] - X[0]).max() > 0.1
    np.testing.assert_allclose(X_est[10:], X[10:], atol=1e-9)

    # For comparison, a filter that trusts the model (low Q) and not
    # the measurements (high R) does not correct its initial error
    Q, R = 1e-8 * np.eye(2), 1e6 * np.eye(2)
    X, X_est = _simulate_plant_and_KF(
        form, Q, R, np.eye(2), x0, x0_est, U, noise_sd=0.0
    )
    assert np.abs(X_est[-1] - X[-1]).max() > 1.0


def test_make_KF_steady_state_variances():
    """After many time steps, the variance outputs converge to those of
    the steady-state Kalman filter: diag(P) where P(k+1|k) is the
    solution to the discrete algebraic Riccati equation, and
    P(k|k) = P - P C' (C P C' + R)^-1 C P."""
    P_pred = scipy.linalg.solve_discrete_are(
        A_test.T, C_test.T, Q_test, R_test
    )
    S = C_test @ P_pred @ C_test.T + R_test
    P_filt = P_pred - P_pred @ C_test.T @ np.linalg.solve(S, C_test @ P_pred)
    nT = 100
    t = np.arange(nT + 1) * 1.0
    UY = np.random.default_rng(2).normal(size=(nT, 3))
    x0 = np.concatenate([np.zeros(2), np.eye(2).flatten(order="F")])
    for make_KF, P in [
        (KFP, P_pred),
        (KFF, P_filt),
    ]:
        kf = make_KF(
            A_test,
            B_test,
            C_test,
            D_test,
            Q_test,
            R_test,
            include_state_estimates=False,
            include_state_variance=True,
            include_output_variance=True,
        )
        sim = make_n_step_simulation_function_from_model(kf, nT)
        _, Y = sim(t, UY, x0)
        Y = np.array(Y)
        np.testing.assert_allclose(Y[nT - 1, :2], np.diag(P), atol=1e-8)
        np.testing.assert_allclose(
            Y[nT - 1, 2:], np.diag(C_test @ P @ C_test.T), atol=1e-8
        )


def _connect_plant_KF_controller(make_KF, est_name, controller):
    """Connect the test system (plant), a Kalman filter, and a
    state-feedback controller in closed loop. The controller's inputs
    are the state estimates (xh1, xh2) and its output is uc."""
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF(A, B, C, D, Q, R)
    plant = StateSpaceModelDTFromABCD(
        A, B, C, D, input_names=["u"], output_names=["y1", "y2"]
    )
    connections = {
        "u": "uc",
        "uk": "uc",
        "yk_1": "y1",
        "yk_2": "y2",
        "xh1": f"{est_name}_1",
        "xh2": f"{est_name}_2",
    }
    return connect_systems([plant, kf, controller], connections)


def _controller_with_direct_transmission():
    """u(k) = -K x_est, no delay (e.g. a PI regulator)."""
    return StateSpaceModelDTFromABCD(
        cas.DM.zeros(0, 0),
        cas.DM.zeros(0, 2),
        cas.DM.zeros(1, 0),
        cas.DM([[-0.3, -0.5]]),
        input_names=["xh1", "xh2"],
        output_names=["uc"],
    )


def _controller_with_delay():
    """u(k+1) = -K x_est(k), a one-step delay (e.g. MPC)."""
    return StateSpaceModelDTFromABCD(
        cas.DM.zeros(1, 1),
        cas.DM([[-0.3, -0.5]]),
        cas.DM(1.0),
        cas.DM.zeros(1, 2),
        input_names=["xh1", "xh2"],
        output_names=["uc"],
    )


def test_make_KFP_closed_loop():
    """The prediction form's outputs x_est(k+1|k) depend directly on
    u(k), so a controller with direct transmission from its inputs to
    its output u(k) forms an algebraic loop, but a controller with a
    one-step delay does not."""
    make_KF = KFP
    with pytest.raises(ValueError, match="Algebraic loop"):
        _connect_plant_KF_controller(
            make_KF, "xkp1_est", _controller_with_direct_transmission()
        )
    system = _connect_plant_KF_controller(
        make_KF, "xkp1_est", _controller_with_delay()
    )
    assert (system.n, system.nu) == (2 + 6 + 1, 0)


def test_make_KFF_closed_loop():
    """The filtering form's outputs x_est(k|k) don't depend directly on
    u(k) when D = 0, so a controller with direct transmission doesn't
    form an algebraic loop."""
    make_KF = KFF
    system = _connect_plant_KF_controller(
        make_KF, "xk_est", _controller_with_direct_transmission()
    )
    assert (system.n, system.nu) == (2 + 6, 0)
    system = _connect_plant_KF_controller(
        make_KF, "xk_est", _controller_with_delay()
    )
    assert (system.n, system.nu) == (2 + 6 + 1, 0)


def test_steady_state_kf_gain():
    """The gain and covariances are those the dynamic filter converges
    to, and P(k+1|k) solves the discrete algebraic Riccati equation."""
    Kf, Pkp1, Pk, Sk_pred = steady_state_kf_gain(
        A_test, C_test, Q_test, R_test
    )
    for value, shape in zip((Kf, Pkp1, Pk, Sk_pred), [(2, 2)] * 4):
        assert isinstance(value, np.ndarray)
        assert value.shape == shape

    A, B, C, D, Q, R = _test_matrices()
    xkp1_est, Pkp1_dyn = cas.DM.zeros(2), cas.DM.eye(2)
    for k in range(500):
        results = kf_update(
            A, B, C, D, xkp1_est, Pkp1_dyn, Q, R, [0.0, 0.0], [0.0]
        )
        Pkp1_prev = Pkp1_dyn
        xkp1_est, Pkp1_dyn = results[2], results[3]
    np.testing.assert_allclose(Kf, results[5], atol=1e-10)
    np.testing.assert_allclose(Pkp1, Pkp1_dyn, atol=1e-10)
    np.testing.assert_allclose(Pk, results[1], atol=1e-10)
    np.testing.assert_allclose(Sk_pred, C @ Pkp1_prev @ C.T + R, atol=1e-10)

    P_ss = scipy.linalg.solve_discrete_are(A_test.T, C_test.T, Q_test, R_test)
    np.testing.assert_allclose(Pkp1, P_ss)

    # python-control's dlqe returns the correction gain in prediction
    # form, Kp = A Kf (MATLAB's dlqe returns Kf)
    Kp, P_dlqe, _ = control.dlqe(A_test, np.eye(2), C_test, Q_test, R_test)
    np.testing.assert_allclose(Kp, A_test @ Kf)
    np.testing.assert_allclose(P_dlqe, Pkp1)

    # CasADi DM arguments are accepted, symbolic ones aren't
    Kf_dm = steady_state_kf_gain(A, C, Q, R)[0]
    np.testing.assert_allclose(Kf_dm, Kf)
    with pytest.raises(TypeError, match="A: give a numerical matrix"):
        steady_state_kf_gain(cas.SX.sym("A", 2, 2), C, Q, R)


# Reference values from a MATLAB unit test of the steady-state gain
# (computed with MATLAB's dlqe, whose gain output is Kf), rounded to
# 4 decimals: (A, C, Q, R, P(k+1|k), Kf, Kp = A Kf)
MATLAB_KF_SS_EXAMPLES = {
    "course notes exercise 12.3": (
        [[0.8]],
        [[0.5]],
        [[0.3]],
        [[1.1]],
        [[0.6741]],
        None,
        [[0.2126]],
    ),
    "adapted from MATLAB dlqe documentation": (
        [[-0.9, 0.7], [-0.3, 0.1]],
        [[1.0, 1.0]],
        [[1.0, 0.0], [0.0, 3.0]],
        [[0.1]],
        [[4.7687, 0.9438], [0.9438, 3.2369]],
        [[0.5716], [0.4184]],
        [[-0.2216], [-0.1297]],
    ),
    "Mathworks answer": (
        [[0.7, 1.0], [0.0, 1.0]],
        [[0.3, 0.0]],
        [[0.0001, 0.0], [0.0, 0.01]],
        [[0.01]],
        [[0.0857, 0.0444], [0.0444, 0.0368]],
        [[1.4515], [0.7514]],
        None,
    ),
}


@pytest.mark.parametrize("example", MATLAB_KF_SS_EXAMPLES)
def test_steady_state_kf_gain_matlab_examples(example):
    A, C, Q, R, P_ref, Kf_ref, Kp_ref = MATLAB_KF_SS_EXAMPLES[example]
    A = np.array(A)
    Kf, Pkp1, _, _ = steady_state_kf_gain(A, C, Q, R)
    np.testing.assert_allclose(Pkp1, P_ref, atol=5e-5)
    if Kf_ref is not None:
        np.testing.assert_allclose(Kf, Kf_ref, atol=5e-5)
    if Kp_ref is not None:
        np.testing.assert_allclose(A @ Kf, Kp_ref, atol=5e-5)


STEADY_STATE_KF_FORMS = {
    # form: (steady-state KF constructor, dynamic KF class,
    # output names prefix)
    "prediction": (
        make_steady_state_KFP_with_fixedABCDQR,
        KFP,
        "kp1_est",
    ),
    "filtering": (
        make_steady_state_KFF_with_fixedABCDQR,
        KFF,
        "k_est",
    ),
}


@pytest.mark.parametrize("form", STEADY_STATE_KF_FORMS)
def test_make_steady_state_KF(form):
    make_KF_ss, _, est = STEADY_STATE_KF_FORMS[form]
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF_ss(A, B, C, D, Q, R, dt=0.5, include_output_estimates=True)
    assert isinstance(kf, StateSpaceModelDTFromABCD)
    assert kf.name is None
    assert kf.dt == 0.5
    assert (kf.n, kf.nu, kf.ny) == (2, 3, 4)
    assert kf.input_names == ["uk", "yk_1", "yk_2"]
    assert kf.state_names == ["xkp1_est_1", "xkp1_est_2"]
    assert kf.output_names == [
        f"x{est}_1",
        f"x{est}_2",
        f"y{est}_1",
        f"y{est}_2",
    ]

    # One time step compared to kf_update with the steady-state P
    Kf, Pkp1, _, _ = steady_state_kf_gain(A, C, Q, R)
    xkp1_est = cas.DM([0.1, -0.2])
    uk, yk = cas.DM([0.7]), cas.DM([0.4, -0.3])
    expected = kf_update(A, B, C, D, xkp1_est, cas.DM(Pkp1), Q, R, yk, uk)
    np.testing.assert_allclose(
        kf.F(0.0, xkp1_est, cas.vertcat(uk, yk)), expected[2]
    )
    y = kf.H(0.0, xkp1_est, cas.vertcat(uk, yk))
    if form == "prediction":
        np.testing.assert_allclose(y[:2], expected[2])
        np.testing.assert_allclose(y[2:], C @ expected[2])
    else:
        np.testing.assert_allclose(y[:2], expected[0])
        np.testing.assert_allclose(y[2:], C @ expected[0] + D @ uk)

    # Default outputs
    kf = make_KF_ss(A, B, C, D, Q, R)
    assert kf.output_names == [f"x{est}_1", f"x{est}_2"]

    # NumPy arrays are accepted
    kf_np = make_KF_ss(A_test, B_test, C_test, D_test, Q_test, R_test)
    np.testing.assert_allclose(
        kf_np.H(0.0, xkp1_est, cas.vertcat(uk, yk)), y[:2]
    )


@pytest.mark.parametrize("form", STEADY_STATE_KF_FORMS)
def test_make_steady_state_KF_with_D(form):
    """With D non-zero, the filtering form's outputs depend on u(k) and
    agree with kf_update, and the prediction form can't output
    y_est(k+1|k)."""
    make_KF_ss, _, _ = STEADY_STATE_KF_FORMS[form]
    A, B, C, _, Q, R = _test_matrices()
    D = cas.DM([[0.0], [1.0]])
    if form == "prediction":
        with pytest.raises(ValueError, match="unless D = 0"):
            make_KF_ss(A, B, C, D, Q, R, include_output_estimates=True)
        kf = make_KF_ss(A, B, C, D, Q, R)
        i_est = 2
    else:
        kf = make_KF_ss(A, B, C, D, Q, R, include_output_estimates=True)
        i_est = 0
    _, Pkp1, _, _ = steady_state_kf_gain(A, C, Q, R)
    xkp1_est = cas.DM([0.1, -0.2])
    uk, yk = cas.DM([0.7]), cas.DM([0.4, -0.3])
    expected = kf_update(A, B, C, D, xkp1_est, cas.DM(Pkp1), Q, R, yk, uk)
    np.testing.assert_allclose(
        kf.F(0.0, xkp1_est, cas.vertcat(uk, yk)), expected[2]
    )
    y = kf.H(0.0, xkp1_est, cas.vertcat(uk, yk))
    np.testing.assert_allclose(y[:2], expected[i_est])
    if form == "filtering":
        np.testing.assert_allclose(y[2:], C @ expected[0] + D @ uk)


@pytest.mark.parametrize("form", STEADY_STATE_KF_FORMS)
def test_make_steady_state_KF_matches_dynamic_KF(form):
    """Over many time steps, the steady-state filter gives the same
    estimates as the dynamic filter with the initial error covariance
    P(0|-1) set to the steady-state P(k+1|k)."""
    make_KF_ss, make_KF, _ = STEADY_STATE_KF_FORMS[form]
    A, B, C, D, Q, R = _test_matrices()
    _, Pkp1, _, _ = steady_state_kf_gain(A, C, Q, R)
    nT = 50
    t = np.arange(nT + 1) * 1.0
    UY = np.random.default_rng(3).normal(size=(nT, 3))
    x0_est = np.array([0.5, -0.5])

    kf_ss = make_KF_ss(A, B, C, D, Q, R, include_output_estimates=True)
    X_ss, Y_ss = make_n_step_simulation_function_from_model(kf_ss, nT)(
        t, UY, x0_est
    )
    kf = make_KF(A, B, C, D, Q, R, include_output_estimates=True)
    x0 = np.concatenate([x0_est, Pkp1.flatten(order="F")])
    X, Y = make_n_step_simulation_function_from_model(kf, nT)(t, UY, x0)
    np.testing.assert_allclose(X_ss, np.array(X)[:, :2], atol=1e-10)
    np.testing.assert_allclose(Y_ss, Y, atol=1e-10)


def test_make_steady_state_KF_closed_loop():
    """Feedthrough as in the dynamic filters: the prediction form forms
    an algebraic loop with a controller with direct transmission, the
    filtering form doesn't."""
    make_KF = make_steady_state_KFP_with_fixedABCDQR
    with pytest.raises(ValueError, match="Algebraic loop"):
        _connect_plant_KF_controller(
            make_KF, "xkp1_est", _controller_with_direct_transmission()
        )
    system = _connect_plant_KF_controller(
        make_KF, "xkp1_est", _controller_with_delay()
    )
    assert (system.n, system.nu) == (2 + 2 + 1, 0)

    make_KF = make_steady_state_KFF_with_fixedABCDQR
    system = _connect_plant_KF_controller(
        make_KF, "xk_est", _controller_with_direct_transmission()
    )
    assert (system.n, system.nu) == (2 + 2, 0)


# Classes: KFP, KFF, KFPSS, KFFSS


def _simulate(model, U, x0=None, params=()):
    """Outputs of model simulated over the inputs U (rows), as an array
    (rows: time steps, one more than U)."""
    x = np.zeros(model.n) if x0 is None else np.asarray(x0, dtype=float)
    Y = []
    for u in U:
        Y.append(np.asarray(model.H(0, x, u, *params)).ravel())
        x = np.asarray(model.F(0, x, u, *params)).ravel()
    return np.array(Y)


@pytest.mark.parametrize(
    "make, cls",
    [
        (make_steady_state_KFP_with_fixedABCDQR, KFPSS),
        (make_steady_state_KFF_with_fixedABCDQR, KFFSS),
    ],
)
def test_constructors_make_classes(make, cls):
    """Each constructor function makes its class, unnamed unless given a
    name (naming is left to the user)"""
    kf = make(*_test_matrices(), dt=0.5)
    assert type(kf) is cls
    assert kf.name is None and kf.dt == 0.5
    assert make(*_test_matrices(), name="observer").name == "observer"


def test_KFPSS_symbolic_gain():
    """A symbolic K becomes the filter's parameters; with values it is
    the filter with that gain (here the optimal one)"""
    A, B, C, D, Q, R = _test_matrices()
    Kf, _, _, _ = steady_state_kf_gain(A, C, Q, R)
    Kp = np.asarray(A) @ Kf
    K = cas.SX.sym("K", 2, 2)
    kf_sym = KFPSS(A, B, C, D, K, include_output_predictions=True)
    # (a matrix symbol's elements, numbered column by column)
    assert list(kf_sym.params) == ["K_0", "K_1", "K_2", "K_3"]
    kf_num = make_steady_state_KFP_with_fixedABCDQR(
        A, B, C, D, Q, R, include_output_predictions=True
    )
    rng = np.random.default_rng(0)
    U = rng.normal(size=(20, 3))  # [u; y]
    np.testing.assert_allclose(
        _simulate(kf_sym, U, params=list(Kp.ravel(order="F"))),
        _simulate(kf_num, U),
        atol=1e-12,
    )


@pytest.mark.parametrize("cls", [KFPSS, KFFSS])
def test_KFSS_output_predictions(cls):
    """y_est(k|k-1) = C x_est(k|k-1) (D = 0): with the prediction form,
    the previous step's y_est(k|k-1)'s successor y_est(k+1|k)"""
    A, B, C, D, Q, R = _test_matrices()
    K = np.array([[0.3, 0.1], [0.05, 0.4]])
    kf = cls(A, B, C, D, K, include_output_predictions=True)
    assert kf.output_names[-2:] == ["yk_pred_1", "yk_pred_2"]
    rng = np.random.default_rng(1)
    U = rng.normal(size=(15, 3))
    x = np.zeros(2)
    for u in U:
        y = np.asarray(kf.H(0, x, u)).ravel()
        np.testing.assert_allclose(y[-2:], np.asarray(C) @ x, atol=1e-14)
        x = np.asarray(kf.F(0, x, u)).ravel()
    if cls is KFPSS:
        kf = KFPSS(
            A,
            B,
            C,
            D,
            K,
            include_state_estimates=False,
            include_output_estimates=True,
            include_output_predictions=True,
        )
        Y = _simulate(kf, U)
        np.testing.assert_allclose(Y[1:, 2:], Y[:-1, :2], atol=1e-14)


def test_KFPSS_innovations():
    """Data from the innovations form x(k+1) = A x + B u + K e,
    y = C x + e: the filter with that K, from the true initial state,
    predicts y(k) with errors e(k), exactly"""
    A, B, C, D, _, _ = _test_matrices()
    A, B, C = (np.asarray(M) for M in (A, B, C))
    K = np.array([[0.3, 0.1], [0.05, 0.4]])
    rng = np.random.default_rng(2)
    nT = 50
    u, e = rng.normal(size=(nT, 1)), 0.1 * rng.normal(size=(nT, 2))
    x, y = np.zeros(2), []
    for k in range(nT):
        y.append(C @ x + e[k])
        x = A @ x + B @ u[k] + K @ e[k]
    y = np.array(y)
    kf = KFPSS(
        A,
        B,
        C,
        D,
        K,
        include_state_estimates=False,
        include_output_predictions=True,
    )
    y_pred = _simulate(kf, np.hstack([u, y]))
    np.testing.assert_allclose(y - y_pred, e, atol=1e-12)


def test_KFPSS_prediction_error_fit():
    """Prediction-error identification: the filter with a symbolic gain
    fitted to the data (estimate_params on its output predictions)
    recovers the gain of the innovations form that made them"""
    from cas_models.discrete_time.estimate import estimate_params

    # (a coupling strong enough for the output to inform both gains)
    A = np.array([[0.8, 0.5], [0.0, 0.6]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.zeros((1, 1))
    K_true = np.array([[0.5], [0.2]])
    rng = np.random.default_rng(3)
    nT = 2000
    u, e = rng.normal(size=(nT, 1)), 0.1 * rng.normal(size=(nT, 1))
    x, y = np.zeros(2), []
    for k in range(nT):
        y.append(C @ x + e[k])
        x = A @ x + B @ u[k] + K_true @ e[k]
    y = np.array(y)
    K = cas.vertcat(cas.SX.sym("k_1"), cas.SX.sym("k_2"))
    kf = KFPSS(
        A,
        B,
        C,
        D,
        K,
        dt=1.0,
        include_state_estimates=False,
        include_output_predictions=True,
    )
    t = np.arange(nT)
    params_est, _, _ = estimate_params(
        kf,
        t,
        np.hstack([u, y])[:-1],
        y,
        params_init={"k_1": 0.0, "k_2": 0.0},
        solver_options={"print_level": 0},
    )
    K_est = [params_est["k_1"], params_est["k_2"]]
    np.testing.assert_allclose(K_est, K_true.ravel(), atol=0.05)


def test_KFP_symbolic_QR():
    """Symbolic Q and R become the dynamic filter's parameters; with
    values it is the filter with those covariances"""
    A, B, C, D, Q, R = _test_matrices()
    q, r = cas.SX.sym("q"), cas.SX.sym("r")
    kf_sym = KFP(A, B, C, D, q * cas.DM.eye(2), r * cas.DM.eye(2))
    assert list(kf_sym.params) == ["q", "r"]
    kf_num = KFP(A, B, C, D, Q, R)
    rng = np.random.default_rng(4)
    U = rng.normal(size=(10, 3))
    x0 = np.r_[0.0, 0.0, np.eye(2).ravel()]
    np.testing.assert_allclose(
        _simulate(kf_sym, U, x0, params=[0.01, 0.1]),
        _simulate(kf_num, U, x0),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: KFPSS(A_test, B_test, C_test, D_test, np.ones((2, 2)) * 0.1),
        lambda: KFFSS(A_test, B_test, C_test, D_test, np.ones((2, 2)) * 0.1),
        lambda: KFP(A_test, B_test, C_test, D_test, Q_test, R_test),
        lambda: EKFF(
            StateSpaceModelDTFromCTRK4(
                SSModelCTLinearFOSISO(K=2.0, T1=3.0), 0.5
            ),
            [[0.01]],
            [[0.1]],
            include_output_estimates=True,
        ),
        lambda: make_steady_state_KFP_with_fixedABCDQR(
            A_test,
            B_test,
            C_test,
            D_test,
            Q_test,
            R_test,
            dt=0.5,
            include_output_predictions=True,
        ),
    ],
)
def test_save_load(make, tmp_path):
    """The filters save as their class and arguments and load as the same
    filter"""
    from cas_models.yaml_utils import from_yaml, load_model

    kf = make()
    for loaded in (
        from_yaml(kf.to_yaml()),
        load_model(kf.save(tmp_path / "kf")),
    ):
        assert type(loaded) is type(kf)
        assert (
            loaded.name == kf.name and loaded.output_names == kf.output_names
        )
        rng = np.random.default_rng(5)
        U = rng.normal(size=(5, kf.nu))
        x0 = np.r_[0.0, 0.0, np.eye(2).ravel()][: kf.n]
        np.testing.assert_allclose(
            _simulate(loaded, U, x0), _simulate(kf, U, x0), atol=1e-14
        )


def test_KFSS_gain_shape():
    with pytest.raises(ValueError, match="K is"):
        KFPSS(A_test, B_test, C_test, D_test, np.ones((2, 1)))
    with pytest.raises(ValueError, match="K is"):
        KFFSS(A_test, B_test, C_test, D_test, np.ones((3, 2)))


# Extended Kalman filters: EKFP, EKFF

DT_NL = 0.1


def _nonlinear_model(T=cas.SX, h_depends_on_t=False, h_depends_on_u=False):
    """A damped pendulum (damping b, a parameter), discretized by Euler,
    with a nonlinear output function with an offset:

        x1(k+1) = x1 + dt x2
        x2(k+1) = x2 + dt (-sin(x1) - b x2 + u)
          y1(k) = sin(x1) + 1 (+ 0.1 t x1, + 0.5 u)
          y2(k) = x2

    T: the functions' symbol type (cas.SX or cas.MX)."""
    t, xk, uk, b = T.sym("t"), T.sym("xk", 2), T.sym("uk"), T.sym("b")
    xkp1 = cas.vertcat(
        xk[0] + DT_NL * xk[1],
        xk[1] + DT_NL * (-cas.sin(xk[0]) - b * xk[1] + uk),
    )
    y1 = cas.sin(xk[0]) + 1.0
    if h_depends_on_t:
        y1 += 0.1 * t * xk[0]
    if h_depends_on_u:
        y1 += 0.5 * uk
    yk = cas.vertcat(y1, xk[1])
    args, names = [t, xk, uk, b], ["t", "xk", "uk", "b"]
    F = cas.Function("F", args, [xkp1], names, ["xkp1"])
    H = cas.Function("H", args, [yk], names, ["yk"])
    b_param = b if T is cas.SX else cas.MX.sym("b")
    return StateSpaceModelDT(
        F, H, 2, nu=1, ny=2, dt=DT_NL, params={"b": b_param}
    )


B_NL = 0.5
Q_NL = np.diag([1e-4, 1e-3])
R_NL = np.diag([0.01, 0.02])


def _np_nonlinear_model(x, u, t=0.0, h_depends_on_t=False):
    """f, h and their Jacobians A, C of _nonlinear_model (NumPy)."""
    f = np.array(
        [
            x[0] + DT_NL * x[1],
            x[1] + DT_NL * (-np.sin(x[0]) - B_NL * x[1] + u[0]),
        ]
    )
    A = np.array([[1.0, DT_NL], [-DT_NL * np.cos(x[0]), 1.0 - DT_NL * B_NL]])
    h = np.array([np.sin(x[0]) + 1.0, x[1]])
    C = np.array([[np.cos(x[0]), 0.0], [0.0, 1.0]])
    if h_depends_on_t:
        h[0] += 0.1 * t * x[0]
        C[0, 0] += 0.1 * t
    return f, A, h, C


def np_ekf_update(xk_pred, Pk_pred, yk, uk, t=0.0, h_depends_on_t=False):
    """NumPy reference of one EKF step of _nonlinear_model."""
    _, _, h, C = _np_nonlinear_model(xk_pred, uk, t, h_depends_on_t)
    S = C @ Pk_pred @ C.T + R_NL
    Kf = Pk_pred @ C.T @ np.linalg.inv(S)
    xk_est = xk_pred + Kf @ (yk - h)
    z = np.eye(2) - Kf @ C
    Pk = z @ Pk_pred @ z.T + Kf @ R_NL @ Kf.T
    Pk = (Pk + Pk.T) / 2
    f, A, _, _ = _np_nonlinear_model(xk_est, uk, t, h_depends_on_t)
    Pkp1 = A @ Pk @ A.T + Q_NL
    return xk_est, Pk, f, Pkp1, h, Kf


def test_ekf_update_linear_model():
    """With a linear model, ekf_update is kf_update, numerically and
    with symbolic arguments."""
    A, B, C, D, Q, R = _test_matrices()
    model = StateSpaceModelDTFromABCD(A, B, C, D)
    rng = np.random.default_rng(6)
    xk_pred, Pk_pred = cas.DM([0.1, -0.2]), cas.DM.eye(2)
    for k in range(20):
        yk, uk = rng.normal(size=2), rng.normal(size=1)
        results = ekf_update(
            model.F, model.H, xk_pred, Pk_pred, Q, R, yk, uk, t=k
        )
        expected = kf_update(A, B, C, D, xk_pred, Pk_pred, Q, R, yk, uk)
        for a, b in zip(results, expected):
            assert isinstance(a, cas.DM)
            np.testing.assert_allclose(a, b, atol=1e-12)
        xk_pred, Pk_pred = results[2], results[3]

    expected = kf_update(A, B, C, D, xk_pred, Pk_pred, Q, R, yk, uk)
    for T in (cas.SX, cas.MX):
        x, P = T.sym("x", 2), T.sym("P", 2, 2)
        y, u = T.sym("y", 2), T.sym("u", 1)
        f = cas.Function(
            "ekf_update",
            [x, P, y, u],
            list(ekf_update(model.F, model.H, x, P, Q, R, y, u)),
        )
        for a, b in zip(f(xk_pred, Pk_pred, yk, uk), expected):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_ekf_update_nonlinear_model():
    """One step at a time, compared to the NumPy reference with the
    Jacobians written by hand, including the output offset (the
    correction uses H(x_est(k|k-1)), not C x_est(k|k-1))."""
    model = _nonlinear_model()
    rng = np.random.default_rng(7)
    xk_pred, Pk_pred = np.array([0.3, -0.1]), np.eye(2) * 0.1
    xk_dm, Pk_dm = cas.DM(xk_pred), cas.DM(Pk_pred)
    for k in range(20):
        yk = np.array([1.2, 0.0]) + rng.normal(size=2) * 0.1
        uk = rng.normal(size=1)
        results = ekf_update(
            model.F,
            model.H,
            xk_dm,
            Pk_dm,
            Q_NL,
            R_NL,
            yk,
            uk,
            params=[B_NL],
        )
        expected = np_ekf_update(xk_pred, Pk_pred, yk, uk)
        for a, b in zip(results, expected):
            np.testing.assert_allclose(
                np.array(a).reshape(b.shape), b, atol=1e-12
            )
        xk_dm, Pk_dm = results[2], results[3]
        xk_pred, Pk_pred = expected[2], expected[3]


EKF_FORMS = {
    # form: (class, outputs from np_ekf_update's results)
    "prediction": (EKFP, "kp1_est"),
    "filtering": (EKFF, "k_est"),
}
ALL_OUTPUTS = dict(
    include_output_estimates=True,
    include_state_variance=True,
    include_output_variance=True,
    include_output_predictions=True,
)


def _np_ekf_outputs(form, results, uk, t=0.0, h_depends_on_t=False):
    """The filter's outputs (all included) from np_ekf_update's
    results."""
    xk_est, Pk, xkp1_est, Pkp1, yk_pred, _ = results
    if form == "prediction":
        x, P, t_y = xkp1_est, Pkp1, t + DT_NL
    else:
        x, P, t_y = xk_est, Pk, t
    _, _, h, C = _np_nonlinear_model(x, uk, t_y, h_depends_on_t)
    return np.concatenate([x, h, np.diag(P), np.diag(C @ P @ C.T), yk_pred])


@pytest.mark.parametrize("T", [cas.SX, cas.MX])
@pytest.mark.parametrize("form", EKF_FORMS)
def test_EKF_nonlinear_model(form, T):
    """Simulation over 30 steps compared to the NumPy reference: all
    outputs, and the states; the model's parameter b becomes the
    filter's. A model with MX functions gives the same filter."""
    cls, est = EKF_FORMS[form]
    ekf = cls(_nonlinear_model(T), Q_NL, R_NL, **ALL_OUTPUTS)
    assert ekf.dt == DT_NL
    assert list(ekf.params) == ["b"]
    assert ekf.input_names == ["uk", "yk_1", "yk_2"]
    assert ekf.output_names == [
        f"x{est}_1",
        f"x{est}_2",
        f"y{est}_1",
        f"y{est}_2",
        f"x{est}_var_1",
        f"x{est}_var_2",
        f"y{est}_var_1",
        f"y{est}_var_2",
        "yk_pred_1",
        "yk_pred_2",
    ]
    assert ekf.F.is_a("SXFunction") and ekf.H.is_a("SXFunction")

    rng = np.random.default_rng(8)
    xk_pred, Pk_pred = np.array([0.3, -0.1]), np.eye(2) * 0.1
    x = np.r_[xk_pred, Pk_pred.ravel(order="F")]
    for k in range(30):
        uk = rng.normal(size=1)
        yk = np.array([1.2, 0.0]) + rng.normal(size=2) * 0.1
        u = np.r_[uk, yk]
        y = np.asarray(ekf.H(k * DT_NL, x, u, B_NL)).ravel()
        x = np.asarray(ekf.F(k * DT_NL, x, u, B_NL)).ravel()
        results = np_ekf_update(xk_pred, Pk_pred, yk, uk)
        np.testing.assert_allclose(
            y, _np_ekf_outputs(form, results, uk), atol=1e-12
        )
        xk_pred, Pk_pred = results[2], results[3]
        np.testing.assert_allclose(
            x, np.r_[xk_pred, Pk_pred.ravel(order="F")], atol=1e-12
        )


@pytest.mark.parametrize("form", EKF_FORMS)
def test_EKF_linear_model_is_KF(form):
    """With a linear model, EKFP and EKFF are KFP and KFF."""
    cls, _ = EKF_FORMS[form]
    kf_cls = {"prediction": KFP, "filtering": KFF}[form]
    A, B, C, D, Q, R = _test_matrices()
    ekf = cls(
        StateSpaceModelDTFromABCD(A, B, C, D, dt=0.5), Q, R, **ALL_OUTPUTS
    )
    kf = kf_cls(A, B, C, D, Q, R, dt=0.5, **ALL_OUTPUTS)
    assert ekf.dt == 0.5
    assert ekf.output_names == kf.output_names
    assert ekf.state_names == kf.state_names
    xk, uk, _ = _kf_test_point()
    np.testing.assert_allclose(ekf.F(0.0, xk, uk), kf.F(0.0, xk, uk))
    np.testing.assert_allclose(ekf.H(0.0, xk, uk), kf.H(0.0, xk, uk))


def test_EKF_time_varying_output():
    """The model's functions get the time t; y_est(k+1|k) uses H at
    t + dt, so it needs the sample time."""
    model = _nonlinear_model(h_depends_on_t=True)
    rng = np.random.default_rng(9)
    for form, (cls, _) in EKF_FORMS.items():
        ekf = cls(model, Q_NL, R_NL, **ALL_OUTPUTS)
        xk_pred, Pk_pred = np.array([0.3, -0.1]), np.eye(2) * 0.1
        x = np.r_[xk_pred, Pk_pred.ravel(order="F")]
        t = 2.0
        uk = rng.normal(size=1)
        yk = np.array([1.2, 0.0]) + rng.normal(size=2) * 0.1
        y = np.asarray(ekf.H(t, x, np.r_[uk, yk], B_NL)).ravel()
        results = np_ekf_update(
            xk_pred, Pk_pred, yk, uk, t=t, h_depends_on_t=True
        )
        expected = _np_ekf_outputs(form, results, uk, t, h_depends_on_t=True)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    model.dt = None
    with pytest.raises(ValueError, match="sample time dt is needed"):
        EKFP(model, Q_NL, R_NL, include_output_estimates=True)
    ekf = EKFP(model, Q_NL, R_NL, dt=DT_NL, include_output_estimates=True)
    assert ekf.dt == DT_NL


def test_EKF_output_depends_on_u():
    """y_est(k+1|k) needs an output function independent of u; the
    filtering form's outputs then depend on u(k)."""
    model = _nonlinear_model(h_depends_on_u=True)
    with pytest.raises(ValueError, match="unless D = 0"):
        EKFP(model, Q_NL, R_NL, include_output_estimates=True)
    EKFP(model, Q_NL, R_NL, include_output_variance=True)
    ekf = EKFF(model, Q_NL, R_NL)
    sp = np.array(cas.DM(ekf.H.sparsity_jac("uk", "yk"), 1))
    assert sp[:, 0].any()  # u(k)


def test_EKF_errors():
    with pytest.raises(TypeError, match="discrete-time"):
        EKFP(SSModelCTLinearFOSISO(K=1.0, T1=2.0), [[0.1]], [[0.1]])
    model = _nonlinear_model()
    with pytest.raises(ValueError, match="Q is"):
        EKFP(model, np.eye(3), R_NL)
    with pytest.raises(ValueError, match="R is"):
        EKFF(model, Q_NL, np.eye(1))
    # A symbol in Q named as one of the model's parameters
    with pytest.raises(ValueError, match="'b'"):
        EKFP(model, cas.SX.sym("b") * np.eye(2), R_NL)


def test_EKF_continuous_time_model_params():
    """A continuous-time model discretized by RK4, with its parameters
    and symbols in Q and R as the filter's parameters (sorted by name),
    is the filter of the discretized model with those values."""
    model = StateSpaceModelDTFromCTRK4(SSModelCTLinearFOSISO(), 0.5)
    q, r = cas.SX.sym("q"), cas.SX.sym("r")
    ekf = EKFF(model, q, r)
    assert list(ekf.params) == ["K", "T1", "q", "r"]
    model_num = StateSpaceModelDTFromCTRK4(
        SSModelCTLinearFOSISO(K=2.0, T1=3.0), 0.5
    )
    ekf_num = EKFF(model_num, [[0.01]], [[0.1]])
    U = np.random.default_rng(10).normal(size=(10, 2))
    x0 = [0.0, 1.0]
    np.testing.assert_allclose(
        _simulate(ekf, U, x0, params=[2.0, 3.0, 0.01, 0.1]),
        _simulate(ekf_num, U, x0),
        atol=1e-12,
    )


def test_EKF_tracks_nonlinear_plant():
    """The filtering form's estimates of a noisy pendulum, from noisy
    measurements: much closer to the true states than the measurements
    are (x2 is measured directly)."""
    model = _nonlinear_model()
    ekf = EKFF(model, Q_NL, R_NL)
    rng = np.random.default_rng(11)
    nT = 300
    x_true = np.array([1.0, 0.0])
    x = np.r_[0.0, 0.0, np.eye(2).ravel()]  # poor initial estimate
    errors, meas_errors = [], []
    for k in range(nT):
        uk = np.array([0.5 * np.sin(0.05 * k)])
        _, _, h, _ = _np_nonlinear_model(x_true, uk)
        yk = h + rng.normal(size=2) * np.sqrt(np.diag(R_NL))
        u = np.r_[uk, yk]
        xk_est = np.asarray(ekf.H(0.0, x, u, B_NL)).ravel()
        x = np.asarray(ekf.F(0.0, x, u, B_NL)).ravel()
        if k >= 50:
            errors.append(xk_est - x_true)
            meas_errors.append(yk[1] - x_true[1])
        f, _, _, _ = _np_nonlinear_model(x_true, uk)
        x_true = f + rng.normal(size=2) * np.sqrt(np.diag(Q_NL))
    rmse = np.sqrt(np.mean(np.array(errors) ** 2, axis=0))
    assert rmse[1] < 0.5 * np.sqrt(np.mean(np.array(meas_errors) ** 2))
    assert rmse[0] < 0.1


@pytest.mark.parametrize("form", EKF_FORMS)
def test_EKF_closed_loop(form):
    """Feedthrough as in KFP and KFF: the prediction form forms an
    algebraic loop with a controller with direct transmission, the
    filtering form doesn't."""
    cls, est = EKF_FORMS[form]

    def make_KF(A, B, C, D, Q, R):
        return cls(StateSpaceModelDTFromABCD(A, B, C, D), Q, R)

    controller = _controller_with_direct_transmission()
    if form == "prediction":
        with pytest.raises(ValueError, match="Algebraic loop"):
            _connect_plant_KF_controller(make_KF, f"x{est}", controller)
    else:
        system = _connect_plant_KF_controller(make_KF, f"x{est}", controller)
        assert (system.n, system.nu) == (2 + 6, 0)


def test_EKFP_pendulum_benchmark():
    """The EKF of a pendulum with an input disturbance state, compared
    to the results of a homework solution (course GEL-7029, as in
    process-observers' test_EKFObserver.m): the estimates x_est(k+1|k)
    agree to rounding, except from t = 5.2 to 9.2 s, where the
    homework's calculation differs slightly (as in MATLAB: estimates
    within 1e-3, trace of P(k+1|k) within 1)."""
    data = pd.read_csv(
        Path(__file__).parent / "data" / "pend_sim_benchmark.csv"
    )
    K, m, L, g, dt = 1.2, 0.3, 0.4, 9.8, 0.1

    # States: angle, angular velocity, input disturbance; the pendulum
    # as process-models' pend/pend_xkp1.m (one Euler step) and
    # pend_StateFcn.m (the disturbance added to the input torque)
    t, xk, uk = cas.SX.sym("t"), cas.SX.sym("xk", 3), cas.SX.sym("uk")
    torque = uk + xk[2]
    xkp1 = cas.vertcat(
        xk[0] + dt * xk[1],
        xk[1]
        + dt * (-g / L * cas.sin(xk[0]) - K / m * xk[1] + torque / (m * L**2)),
        xk[2],
    )
    F = cas.Function("F", [t, xk, uk], [xkp1], ["t", "xk", "uk"], ["xkp1"])
    H = cas.Function("H", [t, xk, uk], [xk[0]], ["t", "xk", "uk"], ["yk"])
    model = StateSpaceModelDT(F, H, 3, nu=1, ny=1, dt=dt)
    ekf = EKFP(model, 10 * np.eye(3), [[1.0]], include_state_variance=True)

    x = np.r_[np.zeros(3), np.eye(3).ravel()]  # x_est(0|-1), P(0|-1)
    X_est, trP = [], []
    for _, row in data.iterrows():
        u = [row["u(k)"], row["y(k)"]]
        y = np.asarray(ekf.H(row["t"], x, u)).ravel()
        x = np.asarray(ekf.F(row["t"], x, u)).ravel()
        X_est.append(y[:3])
        trP.append(y[3:].sum())
    errors = np.abs(
        np.array(X_est) - data[["xef1(k+1)", "xef2(k+1)", "xef3(k+1)"]]
    ).max(axis=1)
    trP_errors = np.abs(np.array(trP) - data["trP(k)"])
    assert errors.max() < 1e-3
    assert trP_errors.max() < 1
    outside = (data["t"] < 5.15) | (data["t"] > 9.25)
    assert errors[outside].max() < 1e-10
    assert trP_errors[outside].max() < 1e-8


def test_EKFF_van_der_pol_matlab():
    """Compared to MATLAB's extendedKalmanFilter (System Identification
    Toolbox example, the van der Pol oscillator, with its measurements
    vdp_data.mat, saved as vdp_data.npz; as in process-observers'
    test_EKFObserver.m): the first correction and prediction, and the
    estimates after 101 measurements, to the 6 decimals of the MATLAB
    results."""
    y = np.load(Path(__file__).parent / "data" / "vdp_data.npz")["y"].ravel()
    dt = 0.05  # MATLAB's vdpStateFcn: one Euler step, mu = 1
    t, xk, uk = cas.SX.sym("t"), cas.SX.sym("xk", 2), cas.SX.sym("uk", 0)
    xkp1 = xk + dt * cas.vertcat(xk[1], (1 - xk[0] ** 2) * xk[1] - xk[0])
    F = cas.Function("F", [t, xk, uk], [xkp1], ["t", "xk", "uk"], ["xkp1"])
    H = cas.Function("H", [t, xk, uk], [xk[0]], ["t", "xk", "uk"], ["yk"])
    model = StateSpaceModelDT(F, H, 2, nu=0, ny=1)
    ekf = EKFF(
        model, np.diag([0.01, 0.01]), [[0.16]], include_state_variance=True
    )

    x = np.r_[1.0, 0.0, np.eye(2).ravel()]  # x_est(0|-1), P(0|-1)
    for k, yk in enumerate(y):
        out = np.asarray(ekf.H(0.0, x, [yk])).ravel()
        x = np.asarray(ekf.F(0.0, x, [yk])).ravel()
        if k == 0:
            # Correction: x_est(0|0), diag(P(0|0))
            np.testing.assert_array_equal(
                out.round(6), [1.831447, 0, 0.137931, 1]
            )
            # Prediction: x_est(1|0), P(1|0)
            np.testing.assert_array_equal(
                x[:2].round(6), [1.831447, -0.091572]
            )
            np.testing.assert_array_equal(
                x[2:].round(6), [0.150431, 0.037218, 0.037218, 0.788781]
            )
    np.testing.assert_array_equal(out[:2].round(6), [-0.837418, 1.304110])
    np.testing.assert_array_equal(x[:2].round(6), [-0.772213, 1.365460])
