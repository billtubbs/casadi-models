"""Unit tests for src/cas_models/discrete_time/kalman.py module"""

import casadi as cas
import control
import numpy as np
import pytest
import scipy

from cas_models.discrete_time.kalman import (
    kalman_predict_f,
    kalman_update_f,
    kf_update,
    linear_model_output,
    linear_model_state_transition,
    make_KF_for_filtering_with_fixedABCDQR,
    make_KF_for_prediction_with_fixedABCDQR,
    make_steady_state_KF_for_filtering_with_fixedABCDQR,
    make_steady_state_KF_for_prediction_with_fixedABCDQR,
    steady_state_kf_gain,
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
    return tuple(map(cas.DM, (A_test, B_test, C_test, D_test, Q_test, R_test)))


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
        xkp1_est, Pkp1 = results[4], results[5]
        xkp1_np, Pkp1_np = results_np[4], results_np[5]

    # After many steps the gain should converge to the steady-state
    # Kalman filter gain from the discrete algebraic Riccati equation
    P_ss = scipy.linalg.solve_discrete_are(A_test.T, C_test.T, Q_test, R_test)
    Kf_ss = P_ss @ C_test.T @ np.linalg.inv(C_test @ P_ss @ C_test.T + R_test)
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
    assert kf.name == "KF"
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
    np.testing.assert_allclose(xkp1[:2], expected[4])
    np.testing.assert_allclose(xkp1[2:], cas.vec(expected[5]))

    # The variances don't depend on the inputs (only on P(k|k-1))
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(np.array(cas.DM(sp, 1))[4:], 0)


def test_make_KF_for_prediction_with_fixedABCDQR():
    A, B, C, D, Q, R = _test_matrices()
    all_outputs = dict(
        include_output_estimates=True,
        include_state_variance=True,
        include_output_variance=True,
    )
    kf = make_KF_for_prediction_with_fixedABCDQR(
        A, B, C, D, Q, R, dt=0.5, **all_outputs
    )
    _check_KF_system(kf, "kp1_est")

    # Outputs: x_est(k+1|k), C x_est(k+1|k), diag(P(k+1|k)),
    # diag(C P(k+1|k) C')
    xk, uk, expected = _kf_test_point()
    xkp1_est, Pkp1 = expected[4], expected[5]
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
        include_state_variance=True,
    )
    assert kf.output_names == ["xkp1_est_var_1", "xkp1_est_var_2"]

    # y_est(k+1|k) needs D = 0
    D1 = cas.DM([[0.0], [1.0]])
    with pytest.raises(ValueError, match="unless D = 0"):
        make_KF_for_prediction_with_fixedABCDQR(
            A, B, C, D1, Q, R, include_output_estimates=True
        )
    kf = make_KF_for_prediction_with_fixedABCDQR(
        A, B, C, D1, Q, R, include_output_variance=True
    )
    assert kf.ny == 4

    # NumPy arrays are accepted
    kf_np = make_KF_for_prediction_with_fixedABCDQR(
        A_test, B_test, C_test, D_test, Q_test, R_test
    )
    np.testing.assert_allclose(kf_np.H(0.0, xk, uk), xkp1_est)


def test_make_KF_for_filtering_with_fixedABCDQR():
    A, B, C, D, Q, R = _test_matrices()
    all_outputs = dict(
        include_output_estimates=True,
        include_state_variance=True,
        include_output_variance=True,
    )
    kf = make_KF_for_filtering_with_fixedABCDQR(
        A, B, C, D, Q, R, dt=0.5, **all_outputs
    )
    _check_KF_system(kf, "k_est")

    # Outputs: x_est(k|k), y_est(k|k), diag(P(k|k)), diag(C P(k|k) C')
    xk, uk, expected = _kf_test_point()
    xk_est, Pk, yk_est = expected[0], expected[1], expected[2]
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
    kf = make_KF_for_filtering_with_fixedABCDQR(
        A, B, C, D1, Q, R, include_output_estimates=True
    )
    sp = kf.H.sparsity_jac("uk", "yk")
    np.testing.assert_array_equal(np.array(cas.DM(sp, 1))[:, 0], 1)

    # Default and selected outputs
    kf = make_KF_for_filtering_with_fixedABCDQR(A, B, C, D, Q, R)
    assert kf.output_names == ["xk_est_1", "xk_est_2"]
    kf = make_KF_for_filtering_with_fixedABCDQR(
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
    "prediction": (make_KF_for_prediction_with_fixedABCDQR, 4, 1),
    "filtering": (make_KF_for_filtering_with_fixedABCDQR, 0, 0),
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
        xkp1_np, Pkp1_np = results_np[4], results_np[5]
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
        (make_KF_for_prediction_with_fixedABCDQR, P_pred),
        (make_KF_for_filtering_with_fixedABCDQR, P_filt),
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


def test_make_KF_for_prediction_closed_loop():
    """The prediction form's outputs x_est(k+1|k) depend directly on
    u(k), so a controller with direct transmission from its inputs to
    its output u(k) forms an algebraic loop, but a controller with a
    one-step delay does not."""
    make_KF = make_KF_for_prediction_with_fixedABCDQR
    with pytest.raises(ValueError, match="Algebraic loop"):
        _connect_plant_KF_controller(
            make_KF, "xkp1_est", _controller_with_direct_transmission()
        )
    system = _connect_plant_KF_controller(
        make_KF, "xkp1_est", _controller_with_delay()
    )
    assert (system.n, system.nu) == (2 + 6 + 1, 0)


def test_make_KF_for_filtering_closed_loop():
    """The filtering form's outputs x_est(k|k) don't depend directly on
    u(k) when D = 0, so a controller with direct transmission doesn't
    form an algebraic loop."""
    make_KF = make_KF_for_filtering_with_fixedABCDQR
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
    Kf, Pkp1, Pk, Sk = steady_state_kf_gain(A_test, C_test, Q_test, R_test)
    for value, shape in zip((Kf, Pkp1, Pk, Sk), [(2, 2)] * 4):
        assert isinstance(value, np.ndarray)
        assert value.shape == shape

    A, B, C, D, Q, R = _test_matrices()
    xkp1_est, Pkp1_dyn = cas.DM.zeros(2), cas.DM.eye(2)
    for k in range(500):
        results = kf_update(
            A, B, C, D, xkp1_est, Pkp1_dyn, Q, R, [0.0, 0.0], [0.0]
        )
        xkp1_est, Pkp1_dyn = results[4], results[5]
    np.testing.assert_allclose(Kf, results[7], atol=1e-10)
    np.testing.assert_allclose(Pkp1, Pkp1_dyn, atol=1e-10)
    np.testing.assert_allclose(Pk, results[1], atol=1e-10)
    np.testing.assert_allclose(Sk, results[3], atol=1e-10)

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
    # form: (steady-state KF constructor, dynamic KF constructor,
    # output names prefix)
    "prediction": (
        make_steady_state_KF_for_prediction_with_fixedABCDQR,
        make_KF_for_prediction_with_fixedABCDQR,
        "kp1_est",
    ),
    "filtering": (
        make_steady_state_KF_for_filtering_with_fixedABCDQR,
        make_KF_for_filtering_with_fixedABCDQR,
        "k_est",
    ),
}


@pytest.mark.parametrize("form", STEADY_STATE_KF_FORMS)
def test_make_steady_state_KF(form):
    make_KF_ss, _, est = STEADY_STATE_KF_FORMS[form]
    A, B, C, D, Q, R = _test_matrices()
    kf = make_KF_ss(A, B, C, D, Q, R, dt=0.5, include_output_estimates=True)
    assert isinstance(kf, StateSpaceModelDTFromABCD)
    assert kf.name == "KF"
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
        kf.F(0.0, xkp1_est, cas.vertcat(uk, yk)), expected[4]
    )
    y = kf.H(0.0, xkp1_est, cas.vertcat(uk, yk))
    if form == "prediction":
        np.testing.assert_allclose(y[:2], expected[4])
        np.testing.assert_allclose(y[2:], C @ expected[4])
    else:
        np.testing.assert_allclose(y[:2], expected[0])
        np.testing.assert_allclose(y[2:], expected[2])

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
        i_est = 4
    else:
        kf = make_KF_ss(A, B, C, D, Q, R, include_output_estimates=True)
        i_est = 0
    _, Pkp1, _, _ = steady_state_kf_gain(A, C, Q, R)
    xkp1_est = cas.DM([0.1, -0.2])
    uk, yk = cas.DM([0.7]), cas.DM([0.4, -0.3])
    expected = kf_update(A, B, C, D, xkp1_est, cas.DM(Pkp1), Q, R, yk, uk)
    np.testing.assert_allclose(
        kf.F(0.0, xkp1_est, cas.vertcat(uk, yk)), expected[4]
    )
    y = kf.H(0.0, xkp1_est, cas.vertcat(uk, yk))
    np.testing.assert_allclose(y[:2], expected[i_est])
    if form == "filtering":
        np.testing.assert_allclose(y[2:], expected[2])


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
    make_KF = make_steady_state_KF_for_prediction_with_fixedABCDQR
    with pytest.raises(ValueError, match="Algebraic loop"):
        _connect_plant_KF_controller(
            make_KF, "xkp1_est", _controller_with_direct_transmission()
        )
    system = _connect_plant_KF_controller(
        make_KF, "xkp1_est", _controller_with_delay()
    )
    assert (system.n, system.nu) == (2 + 2 + 1, 0)

    make_KF = make_steady_state_KF_for_filtering_with_fixedABCDQR
    system = _connect_plant_KF_controller(
        make_KF, "xk_est", _controller_with_direct_transmission()
    )
    assert (system.n, system.nu) == (2 + 2, 0)
