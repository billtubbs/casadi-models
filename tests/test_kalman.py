"""Unit tests for src/cas_models/discrete_time/kalman.py module"""

import casadi as cas
import numpy as np
import pytest
import scipy

from cas_models.discrete_time.kalman import (
    kalman_predict_f,
    kalman_update_f,
    kf_update,
    linear_model_output,
    linear_model_state_transition,
    make_KF_for_prediction_with_fixedABCDQR,
)
from cas_models.discrete_time.models import StateSpaceModelDTFromABCD
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
    return tuple(
        map(cas.DM, (A_test, B_test, C_test, D_test, Q_test, R_test))
    )


def np_kalman_update(A, B, C, D, xkp1_est, Pkp1, Q, R, yk, uk):
    """NumPy version of the original JAX code (1-D vectors),
    used as a reference to test the CasADi functions.
    """
    yk_pred = C @ xkp1_est + D @ uk
    Sk = C @ Pkp1 @ C.T + R
    Kf = np.linalg.lstsq(Sk.T, (Pkp1 @ C.T).T, rcond=-1)[0].T
    xk_est = xkp1_est + Kf @ (yk - yk_pred)
    z = np.eye(A.shape[0]) - Kf @ C
    Pk = z @ Pkp1 @ z.T + Kf @ R @ Kf.T
    Pk = (Pk + Pk.T) / 2
    yk_est = C @ xk_est + D @ uk
    xkp1_est = A @ xk_est + B @ uk
    Pkp1 = A @ Pk @ A.T + Q
    return xk_est, Pk, yk_est, Sk, xkp1_est, Pkp1, yk_pred, Kf


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
    xk_est, Pk, Kf, Sk = kalman_update_f(
        C, R, xk_pred, Pk_pred, yk_pred, yk
    )
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
        f = cas.Function(
            "f", [x, P, y], kalman_update_f(C, R, x, P, C @ x, y)
        )
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
    np.testing.assert_allclose(
        Pkp1, A_test @ np.array(Pk) @ A_test.T + Q_test
    )

    # Symbolic arguments
    for T in (cas.SX, cas.MX):
        x, P, u = T.sym("x", 2), T.sym("P", 2, 2), T.sym("u", 1)
        f = cas.Function(
            "f", [x, P, u], kalman_predict_f(A, B, Q, x, P, u)
        )
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
        xkp1_est, Pkp1 = results[4], results[5]
        xkp1_np, Pkp1_np = results_np[4], results_np[5]

    # After many steps the gain should converge to the steady-state
    # Kalman filter gain from the discrete algebraic Riccati equation
    P_ss = scipy.linalg.solve_discrete_are(
        A_test.T, C_test.T, Q_test, R_test
    )
    Kf_ss = (
        P_ss @ C_test.T @ np.linalg.inv(C_test @ P_ss @ C_test.T + R_test)
    )
    np.testing.assert_allclose(Pkp1, P_ss, atol=1e-8)
    np.testing.assert_allclose(results[7], Kf_ss, atol=1e-8)

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


def test_make_KF_for_prediction_with_fixedABCDQR():
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF_for_prediction_with_fixedABCDQR(
        A, B, C, D, Q, R, dt=0.5, include_output_estimates=True
    )
    assert kf.name == "KF"
    assert kf.dt == 0.5
    assert (kf.n, kf.nu, kf.ny) == (6, 3, 4)
    assert kf.input_names == ["uk", "yk_1", "yk_2"]
    assert kf.state_names == [
        "xkp1_est_1",
        "xkp1_est_2",
        "Pkp1_1",
        "Pkp1_2",
        "Pkp1_3",
        "Pkp1_4",
    ]
    assert kf.output_names == [
        "xkp1_est_1",
        "xkp1_est_2",
        "yk_est_1",
        "yk_est_2",
    ]

    # One time step compared to kf_update, with inputs ordered
    # as named (u(k), y(k))
    xkp1_est = cas.DM([0.1, -0.2])
    Pkp1 = cas.DM([[0.25, 0.05], [0.05, 0.3]])
    xk = cas.vertcat(xkp1_est, cas.vec(Pkp1))
    uk, yk = cas.DM([0.7]), cas.DM([0.4, -0.3])
    expected = kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, yk, uk)
    xkp1 = kf.F(0.0, xk, cas.vertcat(uk, yk))
    np.testing.assert_allclose(xkp1[:2], expected[4])
    np.testing.assert_allclose(xkp1[2:], cas.vec(expected[5]))
    y = kf.H(0.0, xk, cas.vertcat(uk, yk))
    np.testing.assert_allclose(y[:2], expected[4])
    np.testing.assert_allclose(y[2:], expected[2])

    # The outputs depend directly on the inputs (feedthrough):
    # x_est(k+1|k) on u(k) (except x2, since B[1] = 0) and y(k),
    # y_est(k|k) on y(k) only (since D = 0)
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(
        np.array(cas.DM(sp, 1)),
        [[1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
    )

    # Output selection
    kf = make_KF_for_prediction_with_fixedABCDQR(A, B, C, D, Q, R)
    assert kf.output_names == ["xkp1_est_1", "xkp1_est_2"]
    kf = make_KF_for_prediction_with_fixedABCDQR(
        A,
        B,
        C,
        D,
        Q,
        R,
        include_state_estimates=False,
        include_output_estimates=True,
    )
    assert kf.output_names == ["yk_est_1", "yk_est_2"]

    # NumPy arrays are accepted
    kf_np = make_KF_for_prediction_with_fixedABCDQR(
        A_test, B_test, C_test, D_test, Q_test, R_test
    )
    np.testing.assert_allclose(
        kf_np.F(0.0, xk, cas.vertcat(uk, yk)), xkp1
    )


def test_make_KF_simulation():
    """Multi-step simulation compared to the NumPy reference."""
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF_for_prediction_with_fixedABCDQR(A, B, C, D, Q, R)
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
        xkp1_np, Pkp1_np = results_np[4], results_np[5]
        np.testing.assert_allclose(Y_kf[k], xkp1_np, atol=1e-12)
        np.testing.assert_allclose(X_kf[k + 1, :2], xkp1_np, atol=1e-12)
        np.testing.assert_allclose(
            X_kf[k + 1, 2:], Pkp1_np.flatten(order="F"), atol=1e-12
        )


def _simulate_plant_and_KF(Q, R, P0, x0_plant, x0_est, U, noise_sd, seed=0):
    """Simulate the test system as the plant, add measurement noise to
    its outputs, and run the Kalman filter (same model as the plant)
    on the plant inputs and measurements.

    Returns the plant states X (nT + 1, 2), and the filter's states
    X_est (nT + 1, 6) and outputs Y_est (nT + 1, 2). X_est[k, :2] is
    x_est(k|k-1) and Y_est[k] is x_est(k+1|k), the estimate of X[k+1].
    """
    nT = U.shape[0]
    t = np.arange(nT + 1) * 1.0
    plant = StateSpaceModelDTFromABCD(A_test, B_test, C_test, D_test)
    X, Y = make_n_step_simulation_function_from_model(plant, nT)(
        t, U, x0_plant
    )
    rng = np.random.default_rng(seed)
    Y_meas = np.array(Y)[:nT] + noise_sd * rng.normal(size=(nT, 2))
    kf = make_KF_for_prediction_with_fixedABCDQR(
        A_test, B_test, C_test, D_test, Q, R
    )
    x0_kf = np.concatenate([x0_est, np.array(P0).flatten(order="F")])
    X_est, Y_est = make_n_step_simulation_function_from_model(kf, nT)(
        t, np.hstack([U, Y_meas]), x0_kf
    )
    return np.array(X), np.array(X_est), np.array(Y_est)


def test_make_KF_tracks_plant_with_high_R_low_Q():
    """With very high measurement noise variances (R) and low model
    error variances (Q), the filter ignores the noisy measurements and
    reproduces the plant states (when it starts from the correct
    initial state)."""
    nT = 50
    U = np.random.default_rng(1).normal(size=(nT, 1))
    x0 = np.array([1.0, -1.0])
    Q, R = 1e-8 * np.eye(2), 1e6 * np.eye(2)
    X, X_est, Y_est = _simulate_plant_and_KF(
        Q, R, Q, x0, x0, U, noise_sd=0.5
    )
    np.testing.assert_allclose(X_est[:, :2], X, atol=1e-9)
    np.testing.assert_allclose(Y_est[:nT], X[1:], atol=1e-9)

    # For comparison, a filter that trusts the measurements (low R)
    # and not the model (high Q) follows the measurement noise
    Q, R = np.eye(2), 0.01 * np.eye(2)
    X, X_est, Y_est = _simulate_plant_and_KF(
        Q, R, Q, x0, x0, U, noise_sd=0.5
    )
    assert np.abs(Y_est[:nT] - X[1:]).max() > 0.1


def test_make_KF_converges_with_low_R_high_Q():
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
    X, X_est, Y_est = _simulate_plant_and_KF(
        Q, R, np.eye(2), x0, x0_est, U, noise_sd=0.0
    )
    np.testing.assert_allclose(X, np.tile(x0, (nT + 1, 1)), atol=1e-12)
    assert np.abs(Y_est[0] - X[1]).max() > 0.1
    np.testing.assert_allclose(Y_est[10:nT], X[11:], atol=1e-9)

    # For comparison, a filter that trusts the model (low Q) and not
    # the measurements (high R) does not correct its initial error
    Q, R = 1e-8 * np.eye(2), 1e6 * np.eye(2)
    X, X_est, Y_est = _simulate_plant_and_KF(
        Q, R, np.eye(2), x0, x0_est, U, noise_sd=0.0
    )
    assert np.abs(Y_est[nT - 1] - X[nT]).max() > 1.0


def _connect_plant_KF_controller(controller):
    """Connect the test system (plant), a Kalman filter, and a
    state-feedback controller in closed loop. The controller's inputs
    are the state estimates (xh1, xh2) and its output is uc."""
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF_for_prediction_with_fixedABCDQR(A, B, C, D, Q, R)
    plant = StateSpaceModelDTFromABCD(
        A, B, C, D, input_names=["u"], output_names=["y1", "y2"]
    )
    connections = {
        "u": "uc",
        "uk": "uc",
        "yk_1": "y1",
        "yk_2": "y2",
        "xh1": "xkp1_est_1",
        "xh2": "xkp1_est_2",
    }
    return connect_systems([plant, kf, controller], connections)


def test_make_KF_closed_loop_controller_with_direct_transmission():
    """The filter's outputs x_est(k+1|k) depend directly on u(k), so
    a controller with direct transmission from its inputs to its
    output u(k) (no delay, e.g. a PI regulator) forms an algebraic
    loop."""
    K = cas.DM([[0.3, 0.5]])
    controller = StateSpaceModelDTFromABCD(
        cas.DM.zeros(0, 0),
        cas.DM.zeros(0, 2),
        cas.DM.zeros(1, 0),
        -K,
        input_names=["xh1", "xh2"],
        output_names=["uc"],
    )
    with pytest.raises(ValueError, match="Algebraic loop"):
        _connect_plant_KF_controller(controller)


def test_make_KF_closed_loop_controller_with_delay():
    """A controller with a one-step delay between its inputs and its
    output u(k) (e.g. MPC) does not form an algebraic loop."""
    K = cas.DM([[0.3, 0.5]])
    controller = StateSpaceModelDTFromABCD(
        cas.DM.zeros(1, 1),
        -K,
        cas.DM(1.0),
        cas.DM.zeros(1, 2),
        input_names=["xh1", "xh2"],
        output_names=["uc"],
    )
    system = _connect_plant_KF_controller(controller)
    assert (system.n, system.nu) == (2 + 6 + 1, 0)
