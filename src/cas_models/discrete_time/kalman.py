"""Linear Kalman filter functions using CasADi.

The functions accept numerical (cas.DM, or array-like) and symbolic
(cas.SX or cas.MX) arguments, so they can be used to build
cas.Function objects and StateSpaceModelDT systems.
"""

import casadi as cas

from cas_models.discrete_time.models import StateSpaceModelDT
from cas_models.param_utils import make_list_of_enumerated_names
from cas_models.validation import (
    as_casadi_matrix,
    check_system_dimensions,
)


def linear_model_state_transition(A, B, xk, uk):
    """State at next time instant, x(k+1) = A x(k) + B u(k)."""
    xkp1 = A @ xk + B @ uk
    return xkp1


def linear_model_output(C, D, xk, uk):
    """Output at current time instant, y(k) = C x(k) + D u(k)."""
    yk = C @ xk + D @ uk
    return yk


def kalman_update_f(C, R, xk_pred, Pk_pred, yk_pred, yk):
    """Compute the updated correction gain, state estimate, state error
    covariance, output estimate, and output error covariance in the
    current time instant of a Kalman filter (filtering form) given
    a new measurement.

    y_est(k|k-1) = C(k) * x_est(k|k-1) + D(k) * u(k)
    S(k) = C(k) * P(k|k-1) * C'(k) + R(k)
    Kf(k) = P(k|k-1) * C'(k) * S(k)^-1
    x_est(k|k) = x_est(k|k-1) + Kf * (y(k) - y_est(k|k-1))
    P(k|k) = P(k|k-1) - Kf(k) * S(k) * Kf'(k)
    y_est(k|k) = C(k) * x_est(k|k) + D(k) * u(k)

    Arguments:
    C : (ny, n) matrix
        Measurement matrix.
    R : (ny, ny) matrix
        Measurement error covariance.
    xk_pred : (n, 1) vector
        Prior estimate of the states, x_est(k|k-1).
    Pk_pred : (n, n) matrix
        Prior estimate of the state error covariance, P(k|k-1).
    yk_pred : (ny, 1) vector
        Prior estimate of the system output, y_est(k|k-1).
    yk : (ny, 1) vector
        System output measurement at time k.

    Returns:
    xk_est : (n, 1) vector
        Posterior estimate of the states, x_est(k|k).
    Pk : (n, n) matrix
        Covariance of state estimation error, P(k|k).
    Kf : (n, ny) matrix
        Correction gain matrix.
    Sk : (ny, ny) matrix
        Covariance of output prediction error, S(k).
    """

    # Error covariance of output prediction error
    Sk = C @ Pk_pred @ C.T + R

    # Correction gain
    # Kf = Pk_pred * C.T / Sk  # in MATLAB
    # Numpy equivalent
    # Kf = np.linalg.lstsq(Sk.T, (Pk_pred @ C.T).T, rcond=-1)[0].T
    Kf = cas.mrdivide(Pk_pred @ C.T, Sk)

    # Update state predictions (prior estimates) using
    # measurements from current time to produce 'a posteriori'
    # state estimates
    xk_est = xk_pred + Kf @ (yk - yk_pred)

    # Update error covariance of state estimates
    # Pk = Pk_pred - Kf * Sk * Kf';  # inaccurate
    # Pk = Pk_pred - Kf * C * Pk_pred

    # The following is the 'Joseph Stabilized' version of
    # the update equation which guarantees that Pk is positive
    # semi-definite in the presence of roundoff error.
    # (See p73 of Lewis etal. Book Optimal and Robust Estimation).
    z = cas.DM.eye(Pk_pred.shape[0]) - Kf @ C
    Pk = z @ Pk_pred @ z.T + Kf @ R @ Kf.T

    # Trick to ensure that Pk is always symmetric, in case of
    # rounding errors
    Pk = (Pk + Pk.T) / 2

    return xk_est, Pk, Kf, Sk


def kalman_predict_f(A, B, Q, xk_est, Pk, uk):
    """compute the predicted states, outputs, and state error
    covariance of a Kalman filter (filtering form) in the next
    time instant given the model parameters, the current error
    covariance, and known input.

    x_est(k+1|k) = A(k) * x_est(k|k) + B(k) * u(k)
    P(k+1|k) = A(k) * P(k|k) * A'(k) + Q(k)

    Arguments:
    A, B : matrices
        Discrete-time system model matrices (assumes D = 0).
    Q : matrix, size (n, n)
        State error covariance.
    xk_est : (n, 1) vector
        Posterior estimate of the states, x_est(k|k).
    Pk : (n, n) matrix
        State error covariance, P(k|k).
    uk : (nu, 1) vector
        Known system inputs at time k.

    Returns:
    xkp1_est : (n, 1) vector
        Predicted states at next time instant, x_est(k+1|k).
    Pk : (n, n) matrix
        Predicted state error covariance at next time instant,
        P(k+1|k).
    """

    # Make predictions of states and outputs in next time step
    # x_est(k+1|k) and y_est(k+1|k) i.e. based on the data up
    # to time k. These will be used as priors in the next time
    # step.
    xkp1_est = linear_model_state_transition(A, B, xk_est, uk)

    # Error covariance of predicted state estimates
    Pkp1 = A @ Pk @ A.T + Q

    return xkp1_est, Pkp1


def kf_update(A, B, C, D, xkp1_est, Pkp1, Q, R, yk, uk=None):
    """Carry out one complete time step of a Kalman filter
    (filtering form):
    1. Calculate the correction gain, Kf(k) and
        output estimation error covariance, S(k).
    2. Correct the estimates of the states and outputs at
        the current time using the current measurement, y(k):
            x_est(k|k-1) -> x_est(k|k)
            y_est(k|k-1) -> y_est(k|k)
    3. Calculate the predicted states and state error
        covariance at the next time instant:
            x_est(k+1|k)
            P(k+1|k)
        given the current input, u(k), and the system model.

    To do steps 1 and 2 only use: kalman_update_f.
    To do step 3 only use: kalman_predict_f.

    Numerical arguments yk and uk are converted to cas.DM. Symbolic
    arguments (cas.SX or cas.MX) are passed through, so this
    function can be used to build a cas.Function.

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    xkp1_est : (n, 1) vector
        Prior estimate of the states, x_est(k|k-1).
    Pkp1 : (n, n) matrix
        Prior estimate of the state error covariance, P(k|k-1).
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    yk : (ny, 1) vector
        System output measurements at current time k.
    uk : (nu, 1) vector, optional
        System inputs at current time k. Only optional if
        nu = 0.

    Returns:
    xk_est : (n, 1) vector
        Posterior estimate of the states, x_est(k|k).
    Pk : (n, n) matrix
        Covariance of state estimation error, P(k|k).
    yk_est : (ny, 1) vector
        Posterior estimate of the system output, y_est(k|k).
    Sk : (ny, ny) matrix
        Covariance of output prediction error, S(k).
    xkp1_est : (n, 1) vector
        Predicted states at next time instant, x_est(k+1|k).
    Pkp1 : (n, n) matrix
        Predicted state error covariance at next time instant,
        P(k+1|k).
    yk_pred : (ny, 1) vector
        Prior estimate of the system output, y_est(k|k-1).
    Kf : (n, ny) matrix
        Correction gain matrix.
    """
    if uk is None:
        uk = cas.DM.zeros(0)
    else:
        uk = as_casadi_matrix(uk, "uk")
    yk = as_casadi_matrix(yk, "yk")

    n, nu, ny = check_system_dimensions(A, B, C, D)

    # Check size of arguments passed
    assert uk.shape == (nu, 1), "uk.shape"
    assert yk.shape == (ny, 1), "yk.shape"

    # Update estimates based on current measurement
    # Predicted output at the current time
    yk_pred = linear_model_output(C, D, xkp1_est, uk)

    # Update states
    xk_est, Pk, Kf, Sk = kalman_update_f(
        C, R, xkp1_est, Pkp1, yk_pred, yk
    )

    # Calculate system output estimate
    yk_est = linear_model_output(C, D, xk_est, uk)

    # Predictions based on current inputs
    xkp1_est, Pkp1 = kalman_predict_f(A, B, Q, xk_est, Pk, uk)

    return xk_est, Pk, yk_est, Sk, xkp1_est, Pkp1, yk_pred, Kf


def make_KF_for_prediction_with_fixedABCDQR(
    A,
    B,
    C,
    D,
    Q,
    R,
    dt=None,
    name="KF",
    include_state_estimates=True,
    include_output_estimates=False,
):
    """Make a StateSpaceModelDT system that simulates a dynamic Kalman
    filter (time-varying gain and error covariance) in prediction form,
    for the linear system model

        x(k+1) = A x(k) + B u(k) + w(k),  w(k) ~ N(0, Q)
          y(k) = C x(k) + D u(k) + v(k),  v(k) ~ N(0, R)

    with fixed A, B, C, D, Q, and R.

    Filter system:
        inputs: [u(k); y(k)], names uk..., yk...
        states: [x_est(k|k-1); vec(P(k|k-1))], names xkp1_est...,
            Pkp1... (P is vectorized column-wise, n * n elements)
        state transition: one step of kf_update, giving
            [x_est(k+1|k); vec(P(k+1|k))]
        outputs (each optional):
            x_est(k+1|k) if include_state_estimates (names xkp1_est...)
            y_est(k|k) if include_output_estimates (names yk_est...)

    The filter's inputs u(k) and y(k) are the plant's inputs and
    outputs. The state estimate output x_est(k+1|k) depends directly
    on them (direct feedthrough). It is intended for controllers with
    a one-step delay between their inputs and outputs (e.g. MPC),
    which use x_est(k+1|k) at time k to compute the control action
    sent at time k + 1. Controllers with direct transmission and no
    delay (e.g. a PI regulator) would form an algebraic loop with
    this filter.

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    dt : float, optional
        Sample time.
    name : str, default "KF"
        Name of the system.
    include_state_estimates : bool, default True
        Include x_est(k+1|k) in the outputs.
    include_output_estimates : bool, default False
        Include y_est(k|k) in the outputs.

    Returns:
    StateSpaceModelDT
    """
    A, B, C, D, Q, R = (
        as_casadi_matrix(value, name)
        for value, name in zip((A, B, C, D, Q, R), "ABCDQR")
    )
    n, nu, ny = check_system_dimensions(A, B, C, D)
    uk = cas.SX.sym("uk", nu)
    yk = cas.SX.sym("yk", ny)
    xkp1_est_in = cas.SX.sym("xkp1_est", n)
    Pkp1_in = cas.SX.sym("Pkp1", n, n)
    xk_est, Pk, yk_est, Sk, xkp1_est, Pkp1, yk_pred, Kf = kf_update(
        A, B, C, D, xkp1_est_in, Pkp1_in, Q, R, yk, uk=uk
    )

    # Kalman filter system model - input signals
    inputs = cas.vertcat(uk, yk)
    input_names = make_list_of_enumerated_names(
        "uk", nu, sep="_"
    ) + make_list_of_enumerated_names("yk", ny, sep="_")

    # Internal states
    states = cas.vertcat(xkp1_est_in, cas.vec(Pkp1_in))
    state_names = make_list_of_enumerated_names(
        "xkp1_est", n, sep="_"
    ) + make_list_of_enumerated_names("Pkp1", n * n, sep="_")
    states_next = cas.vertcat(xkp1_est, cas.vec(Pkp1))

    # Output signals
    outputs = []
    output_names = []
    if include_state_estimates:
        outputs.append(xkp1_est)
        output_names += make_list_of_enumerated_names("xkp1_est", n, sep="_")
    if include_output_estimates:
        outputs.append(yk_est)
        output_names += make_list_of_enumerated_names("yk_est", ny, sep="_")
    outputs = cas.vcat(outputs)

    t = cas.SX.sym("t")

    # State transition function
    F = cas.Function(
        "F",
        [t, states, inputs],
        [states_next],
        ["t", "xk", "uk"],
        ["xkp1"],
    )

    # Output function
    H = cas.Function(
        "H",
        [t, states, inputs],
        [outputs],
        ["t", "xk", "uk"],
        ["yk"],
    )

    system = StateSpaceModelDT(
        F,
        H,
        states.shape[0],
        nu=inputs.shape[0],
        ny=outputs.shape[0],
        dt=dt,
        params=None,
        name=name,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
    )

    return system
