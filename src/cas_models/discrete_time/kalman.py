"""Linear Kalman filters using CasADi: classes and functions.

The functions accept numerical (cas.DM, or array-like) and symbolic
(cas.SX or cas.MX) arguments, so they can be used to build
cas.Function objects and StateSpaceModelDT systems.

Classes
-------
KFP
    Dynamic Kalman filter (time-varying gain) in prediction form,
    estimating x_est(k+1|k). Nonlinear in its states (the covariance
    recursion); Q and R may be symbolic.
KFF
    Dynamic Kalman filter in filtering form, estimating x_est(k|k).
KFPSS
    Steady-state (fixed-gain) Kalman filter in prediction form: the
    innovations form, a linear system; the gain K may be symbolic (for
    prediction-error identification).
KFFSS
    Steady-state Kalman filter in filtering form.

The classes have concise names; the functions that make them are named
descriptively.

Functions
---------
linear_model_state_transition
    State at the next time instant, x(k+1) = A x(k) + B u(k).
linear_model_output
    Output at the current time instant, y(k) = C x(k) + D u(k).
kalman_update_f
    Correction step: gain, x_est(k|k) and P(k|k) given a measurement y(k).
kalman_predict_f
    Prediction step: x_est(k+1|k) and P(k+1|k) given the input u(k).
kf_update
    One complete time step (correction and prediction) of a Kalman filter.
make_KFP_with_fixedABCDQR
    A KFP for fixed A, B, C, D, Q and R.
make_KFF_with_fixedABCDQR
    A KFF for fixed A, B, C, D, Q and R.
steady_state_kf_gain
    Steady-state correction gain Kf and error covariances (solves the DARE).
make_steady_state_KFP_with_fixedABCDQR
    A KFPSS with the optimal gain for fixed A, B, C, D, Q and R.
make_steady_state_KFF_with_fixedABCDQR
    A KFFSS with the optimal gain for fixed A, B, C, D, Q and R.
"""

import casadi as cas
import control
import numpy as np

from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTFromABCD,
)
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
    xk_est, Pk, Kf, Sk = kalman_update_f(C, R, xkp1_est, Pkp1, yk_pred, yk)

    # Calculate system output estimate
    yk_est = linear_model_output(C, D, xk_est, uk)

    # Predictions based on current inputs
    xkp1_est, Pkp1 = kalman_predict_f(A, B, Q, xk_est, Pk, uk)

    return xk_est, Pk, yk_est, Sk, xkp1_est, Pkp1, yk_pred, Kf


def _symbolic_params(*matrices):
    """{name: symbol} of the SX symbols in the matrices, sorted by name
    (as StateSpaceModelDTFromABCD's parameters)."""
    params = {}
    for M in matrices:
        if isinstance(M, cas.SX):
            for p in cas.symvar(M):
                params[p.name()] = p
    return {name: params[name] for name in sorted(params)}


def _output_list(flags):
    """The output signals whose flags are True, in order, from a list of
    (signal, flag) pairs."""
    return [signal for signal, included in flags if included]


def _signal_names(outputs, sizes):
    """Output names: each signal a vector named <signal>_1, ...
    (just <signal> if it has one element)."""
    names = []
    for signal in outputs:
        names += make_list_of_enumerated_names(signal, sizes[signal], sep="_")
    return names


class _KF(StateSpaceModelDT):
    """Base class of the dynamic Kalman filters KFP and KFF (see their
    docstrings), which differ only in their outputs.

    outputs is a list of the names of the output signals, each a vector
    named <name>_1, <name>_2, ...:
        "xk_est", "xk_est_var" : x_est(k|k), diag(P(k|k))
        "yk_est", "yk_est_var" : y_est(k|k), diag(C P(k|k) C')
        "xkp1_est", "xkp1_est_var" : x_est(k+1|k), diag(P(k+1|k))
        "ykp1_est", "ykp1_est_var" : y_est(k+1|k) = C x_est(k+1|k)
            (D = 0 only), diag(C P(k+1|k) C')
        "yk_pred" : y_est(k|k-1) = C x_est(k|k-1) + D u(k)
    """

    def __init__(self, A, B, C, D, Q, R, outputs, dt=None, name=None):
        A, B, C, D, Q, R = (
            as_casadi_matrix(value, name)
            for value, name in zip((A, B, C, D, Q, R), "ABCDQR")
        )
        n, nu, ny = check_system_dimensions(A, B, C, D)
        if "ykp1_est" in outputs and not D.is_zero():
            raise ValueError(
                "y_est(k+1|k) depends on u(k+1), which isn't known at time "
                "k, unless D = 0"
            )
        params = _symbolic_params(A, B, C, D, Q, R)
        uk = cas.SX.sym("uk", nu)
        yk = cas.SX.sym("yk", ny)
        xkp1_est_in = cas.SX.sym("xkp1_est", n)
        Pkp1_in = cas.SX.sym("Pkp1", n, n)
        xk_est, Pk, yk_est, Sk, xkp1_est, Pkp1, yk_pred, Kf = kf_update(
            A, B, C, D, xkp1_est_in, Pkp1_in, Q, R, yk, uk=uk
        )

        inputs = cas.vertcat(uk, yk)
        input_names = make_list_of_enumerated_names(
            "uk", nu, sep="_"
        ) + make_list_of_enumerated_names("yk", ny, sep="_")
        states = cas.vertcat(xkp1_est_in, cas.vec(Pkp1_in))
        state_names = make_list_of_enumerated_names(
            "xkp1_est", n, sep="_"
        ) + make_list_of_enumerated_names("Pkp1", n * n, sep="_")
        states_next = cas.vertcat(xkp1_est, cas.vec(Pkp1))

        # Each output's estimate and its error variance use the same
        # covariance: P(k|k) for y_est(k|k), P(k+1|k) for y_est(k+1|k).
        # (The SX graph keeps only the operations the diagonals need.)
        signals = {
            "xk_est": lambda: xk_est,
            "xk_est_var": lambda: cas.diag(Pk),
            "yk_est": lambda: yk_est,
            "yk_est_var": lambda: cas.diag(C @ Pk @ C.T),
            "xkp1_est": lambda: xkp1_est,
            "xkp1_est_var": lambda: cas.diag(Pkp1),
            "ykp1_est": lambda: C @ xkp1_est,
            "ykp1_est_var": lambda: cas.diag(C @ Pkp1 @ C.T),
            "yk_pred": lambda: yk_pred,
        }
        values = {signal: signals[signal]() for signal in outputs}
        output_names = _signal_names(
            outputs, {s: v.shape[0] for s, v in values.items()}
        )
        y = cas.vertcat(cas.SX(0, 1), *values.values())

        t = cas.SX.sym("t")
        arg_names = ["t", "xk", "uk", *params]
        F = cas.Function(
            "F",
            [t, states, inputs, *params.values()],
            [states_next],
            arg_names,
            ["xkp1"],
        )
        H = cas.Function(
            "H",
            [t, states, inputs, *params.values()],
            [y],
            arg_names,
            ["yk"],
        )
        super().__init__(
            F,
            H,
            states.shape[0],
            nu=inputs.shape[0],
            ny=y.shape[0],
            dt=dt,
            params=params,
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class KFP(_KF):
    """Dynamic Kalman filter (time-varying gain and error covariance) in
    prediction form, which estimates the states at the next time
    instant, x_est(k+1|k), given the data up to time k.

    The model of the plant is the linear system

        x(k+1) = A x(k) + B u(k) + w(k),  w(k) ~ N(0, Q)
          y(k) = C x(k) + D u(k) + v(k),  v(k) ~ N(0, R)

    A, B, C, D, Q and R are matrices of numbers or SX symbols (the
    symbols become the filter's parameters, e.g. to fit Q and R).

    Filter system (nonlinear in its states: the error covariance
    recursion):
        inputs: [u(k); y(k)], names uk..., yk...
        states: [x_est(k|k-1); vec(P(k|k-1))], names xkp1_est...,
            Pkp1... (P is vectorized column-wise, n * n elements)
        state transition: one step of kf_update, giving
            [x_est(k+1|k); vec(P(k+1|k))]
        outputs (each optional, in this order):
            x_est(k+1|k), names xkp1_est...
            y_est(k+1|k) = C x_est(k+1|k), names ykp1_est...
            diag(P(k+1|k)), names xkp1_est_var...
            diag(C P(k+1|k) C'), names ykp1_est_var...
            y_est(k|k-1) = C x_est(k|k-1) + D u(k), names yk_pred...

    The outputs x_est(k+1|k) and y_est(k+1|k) depend directly on u(k)
    and y(k) (direct feedthrough). This form is intended for
    controllers with a one-step delay between their inputs and outputs
    (e.g. MPC), which use x_est(k+1|k) at time k to compute the control
    action sent at time k + 1. Controllers with direct transmission and
    no delay (e.g. a PI regulator) would form an algebraic loop with
    this filter: use KFF for those.

    y_est(k|k-1), the one-step-ahead prediction of y(k) made before the
    correction with y(k), depends on y only through the states: its
    errors y(k) - y_est(k|k-1) are the innovations, which a
    prediction-error fit minimizes.

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    dt : float, optional
        Sample time.
    name : str, optional
        Name of the system.
    include_state_estimates : bool, default True
        Include x_est(k+1|k) in the outputs.
    include_output_estimates : bool, default False
        Include y_est(k+1|k) = C x_est(k+1|k) in the outputs. Requires
        D = 0, since y_est(k+1|k) would otherwise depend on u(k+1).
    include_state_variance : bool, default False
        Include the variances of the state estimation errors,
        diag(P(k+1|k)), in the outputs.
    include_output_variance : bool, default False
        Include the variances of the errors of the output estimates
        y_est(k+1|k), diag(C P(k+1|k) C'), in the outputs. (These
        exclude the measurement noise: add diag(R) for the variances of
        the errors in predicting the measurements y(k+1).)
    include_output_predictions : bool, default False
        Include y_est(k|k-1) in the outputs.

    Raises:
    ValueError: if include_output_estimates is True and D isn't zero.
    """

    def __init__(
        self,
        A,
        B,
        C,
        D,
        Q,
        R,
        dt=None,
        name=None,
        include_state_estimates=True,
        include_output_estimates=False,
        include_state_variance=False,
        include_output_variance=False,
        include_output_predictions=False,
    ):
        outputs = _output_list(
            [
                ("xkp1_est", include_state_estimates),
                ("ykp1_est", include_output_estimates),
                ("xkp1_est_var", include_state_variance),
                ("ykp1_est_var", include_output_variance),
                ("yk_pred", include_output_predictions),
            ]
        )
        super().__init__(A, B, C, D, Q, R, outputs, dt=dt, name=name)


class KFF(_KF):
    """Dynamic Kalman filter (time-varying gain and error covariance) in
    filtering form, which estimates the states at the current time
    instant, x_est(k|k), given the data up to time k.

    The model of the plant is the linear system

        x(k+1) = A x(k) + B u(k) + w(k),  w(k) ~ N(0, Q)
          y(k) = C x(k) + D u(k) + v(k),  v(k) ~ N(0, R)

    A, B, C, D, Q and R are matrices of numbers or SX symbols (the
    symbols become the filter's parameters).

    Filter system (nonlinear in its states: the error covariance
    recursion):
        inputs: [u(k); y(k)], names uk..., yk...
        states: [x_est(k|k-1); vec(P(k|k-1))], names xkp1_est...,
            Pkp1... (P is vectorized column-wise, n * n elements)
        state transition: one step of kf_update, giving
            [x_est(k+1|k); vec(P(k+1|k))]
        outputs (each optional, in this order):
            x_est(k|k), names xk_est...
            y_est(k|k) = C x_est(k|k) + D u(k), names yk_est...
            diag(P(k|k)), names xk_est_var...
            diag(C P(k|k) C'), names yk_est_var...
            y_est(k|k-1) = C x_est(k|k-1) + D u(k), names yk_pred...

    The outputs x_est(k|k) and y_est(k|k) depend directly on y(k), and
    on u(k) only if D isn't zero (direct feedthrough). With D = 0, the
    filter can be used with controllers with direct transmission and no
    delay (e.g. a PI regulator), which use x_est(k|k) to compute the
    control action u(k) at time k, without forming an algebraic loop.

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    dt : float, optional
        Sample time.
    name : str, optional
        Name of the system.
    include_state_estimates : bool, default True
        Include x_est(k|k) in the outputs.
    include_output_estimates : bool, default False
        Include y_est(k|k) in the outputs.
    include_state_variance : bool, default False
        Include the variances of the state estimation errors,
        diag(P(k|k)), in the outputs.
    include_output_variance : bool, default False
        Include the variances of the errors of the output estimates
        y_est(k|k), diag(C P(k|k) C'), in the outputs.
    include_output_predictions : bool, default False
        Include y_est(k|k-1) in the outputs (see KFP).
    """

    def __init__(
        self,
        A,
        B,
        C,
        D,
        Q,
        R,
        dt=None,
        name=None,
        include_state_estimates=True,
        include_output_estimates=False,
        include_state_variance=False,
        include_output_variance=False,
        include_output_predictions=False,
    ):
        outputs = _output_list(
            [
                ("xk_est", include_state_estimates),
                ("yk_est", include_output_estimates),
                ("xk_est_var", include_state_variance),
                ("yk_est_var", include_output_variance),
                ("yk_pred", include_output_predictions),
            ]
        )
        super().__init__(A, B, C, D, Q, R, outputs, dt=dt, name=name)


def make_KFP_with_fixedABCDQR(A, B, C, D, Q, R, dt=None, name=None, **kwargs):
    """A dynamic Kalman filter in prediction form for the linear system
    with fixed matrices A, B, C, D and covariances Q and R: a KFP (see
    its docstring for the arguments, kwargs being its output options).
    """
    return KFP(A, B, C, D, Q, R, dt=dt, name=name, **kwargs)


def make_KFF_with_fixedABCDQR(A, B, C, D, Q, R, dt=None, name=None, **kwargs):
    """A dynamic Kalman filter in filtering form for the linear system
    with fixed matrices A, B, C, D and covariances Q and R: a KFF (see
    its docstring for the arguments, kwargs being its output options).
    """
    return KFF(A, B, C, D, Q, R, dt=dt, name=name, **kwargs)


def _as_numeric_matrix(value, name):
    """A numerical matrix argument as a 2-D NumPy array."""
    value = as_casadi_matrix(value, name)
    if not isinstance(value, cas.DM):
        raise TypeError(f"{name}: give a numerical matrix, not a symbolic one")
    return value.full()


def steady_state_kf_gain(A, C, Q, R):
    """Compute the correction gain and error covariances of the
    steady-state Kalman filter, by solving the discrete algebraic
    Riccati equation with control.dare.

    S = C P C' + R
    Kf = P C' S^-1
    P(k|k) = P - Kf S Kf'

    where P = P(k+1|k) = P(k|k-1) is the solution of

    P = A P A' - A P C' (C P C' + R)^-1 C P A' + Q

    Note: control.dare(A', C', Q, R) returns X = P and G = Kp', where
    Kp = A Kf is the correction gain of the Kalman filter in prediction
    form, not Kf, the correction gain in filtering form. Likewise,
    control.dlqe(A, I, C, Q, R) returns Kp, whereas MATLAB's dlqe
    returns Kf.

    Arguments:
    A, C : matrices
        Discrete-time system model matrices (numerical).
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.

    Returns:
    Kf : (n, ny) array
        Correction gain of the Kalman filter in filtering form (the
        correction gain in prediction form is Kp = A Kf).
    Pkp1 : (n, n) array
        Covariance of state prediction error, P(k+1|k).
    Pk : (n, n) array
        Covariance of state estimation error, P(k|k).
    Sk : (ny, ny) array
        Covariance of output prediction error, S(k).
    """
    A, C, Q, R = (
        _as_numeric_matrix(value, name)
        for value, name in zip((A, C, Q, R), "ACQR")
    )
    Pkp1, _, _ = control.dare(A.T, C.T, Q, R)
    Sk = C @ Pkp1 @ C.T + R
    Kf = np.linalg.solve(Sk.T, (Pkp1 @ C.T).T).T
    Pk = Pkp1 - Kf @ Sk @ Kf.T
    Pk = (Pk + Pk.T) / 2
    return Kf, Pkp1, Pk, Sk


class _KFSS(StateSpaceModelDTFromABCD):
    """Base class of the steady-state (fixed-gain) Kalman filters KFPSS
    and KFFSS (see their docstrings), which differ in their outputs and
    in the gain they are given. Kp is the correction gain in prediction
    form, Kf the one in filtering form (needed only for x_est(k|k) and
    y_est(k|k)).

    outputs is a list of the names of the output signals, each a vector
    named <name>_1, <name>_2, ...:
        "xk_est" : x_est(k|k)
        "yk_est" : y_est(k|k) = C x_est(k|k) + D u(k)
        "xkp1_est" : x_est(k+1|k)
        "ykp1_est" : y_est(k+1|k) = C x_est(k+1|k) (D = 0 only)
        "yk_pred" : y_est(k|k-1) = C x_est(k|k-1) + D u(k)
    """

    def __init__(self, A, B, C, D, Kp, Kf, outputs, dt=None, name=None):
        A, B, C, D = (
            cas.SX(as_casadi_matrix(value, name))
            for value, name in zip((A, B, C, D), "ABCD")
        )
        n, nu, ny = check_system_dimensions(A, B, C, D)
        Kp = cas.SX(as_casadi_matrix(Kp, "K"))
        if Kp.shape != (n, ny):
            raise ValueError(f"K is {Kp.shape}, not (n, ny) = {(n, ny)}")
        if "ykp1_est" in outputs and not D.is_zero():
            raise ValueError(
                "y_est(k+1|k) depends on u(k+1), which isn't known at time "
                "k, unless D = 0"
            )

        # Filter states x_est(k|k-1), inputs [u(k); y(k)]:
        # x_est(k+1|k) = (A - Kp C) x_est(k|k-1) + [B - Kp D, Kp] [u(k); y(k)]
        A_kf = A - Kp @ C
        B_kf = cas.horzcat(B - Kp @ D, Kp)
        D_u = cas.horzcat(D, cas.SX(ny, ny))  # D u(k)
        signals = {
            "xkp1_est": lambda: (A_kf, B_kf),
            "ykp1_est": lambda: (C @ A_kf, C @ B_kf),
            "yk_pred": lambda: (C, D_u),
        }
        if Kf is not None:
            Kf = cas.SX(Kf)
            # x_est(k|k) = (I - Kf C) x_est(k|k-1) + [-Kf D, Kf] [u(k); y(k)]
            C_xk = cas.SX.eye(n) - Kf @ C
            D_xk = cas.horzcat(-Kf @ D, Kf)
            signals["xk_est"] = lambda: (C_xk, D_xk)
            signals["yk_est"] = lambda: (C @ C_xk, C @ D_xk + D_u)
        values = {signal: signals[signal]() for signal in outputs}
        C_kf = cas.vertcat(cas.SX(0, n), *(v[0] for v in values.values()))
        D_kf = cas.vertcat(
            cas.SX(0, nu + ny), *(v[1] for v in values.values())
        )
        output_names = _signal_names(
            outputs, {s: v[0].shape[0] for s, v in values.items()}
        )
        super().__init__(
            A_kf,
            B_kf,
            C_kf,
            D_kf,
            dt=dt,
            name=name,
            input_names=(
                make_list_of_enumerated_names("uk", nu, sep="_")
                + make_list_of_enumerated_names("yk", ny, sep="_")
            ),
            state_names=make_list_of_enumerated_names("xkp1_est", n, sep="_"),
            output_names=output_names,
        )


class KFPSS(_KFSS):
    """Steady-state Kalman filter (fixed correction gain) in prediction
    form, which estimates the states at the next time instant,
    x_est(k+1|k), given the data up to time k: the innovations form of
    the linear system

        x(k+1) = A x(k) + B u(k) + K e(k)
          y(k) = C x(k) + D u(k) + e(k)

    with e(k) the innovations, y(k) - y_est(k|k-1) (Ljung, System
    Identification, 1999). A, B, C, D and K are matrices of numbers or
    SX symbols (the symbols become the filter's parameters): with K
    symbolic, a prediction-error fit of the filter estimates the model
    and the gain together. For the optimal (Kalman) gain given noise
    covariances Q and R, see make_steady_state_KFP_with_fixedABCDQR.

    Filter system (a linear system):
        inputs: [u(k); y(k)], names uk..., yk...
        states: x_est(k|k-1), names xkp1_est...
        state transition:
            x_est(k+1|k) = (A - K C) x_est(k|k-1)
                           + (B - K D) u(k) + K y(k)
        outputs (each optional, in this order):
            x_est(k+1|k), names xkp1_est...
            y_est(k+1|k) = C x_est(k+1|k), names ykp1_est...
            y_est(k|k-1) = C x_est(k|k-1) + D u(k), names yk_pred...

    x_est(k+1|k) and y_est(k+1|k) depend directly on u(k) and y(k)
    (direct feedthrough), as in KFP (see its docstring for which
    controllers it can be connected to). y_est(k|k-1) depends on y only
    through the states: its errors y(k) - y_est(k|k-1) are the
    innovations, which a prediction-error fit minimizes.

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    K : (n, ny) matrix
        Correction gain in prediction form (K = A Kf, with Kf the gain in
        filtering form).
    dt : float, optional
        Sample time.
    name : str, optional
        Name of the system.
    include_state_estimates : bool, default True
        Include x_est(k+1|k) in the outputs.
    include_output_estimates : bool, default False
        Include y_est(k+1|k) = C x_est(k+1|k) in the outputs. Requires
        D = 0, since y_est(k+1|k) would otherwise depend on u(k+1).
    include_output_predictions : bool, default False
        Include y_est(k|k-1) in the outputs.

    Raises:
    ValueError: if include_output_estimates is True and D isn't zero.
    """

    def __init__(
        self,
        A,
        B,
        C,
        D,
        K,
        dt=None,
        name=None,
        include_state_estimates=True,
        include_output_estimates=False,
        include_output_predictions=False,
    ):
        outputs = _output_list(
            [
                ("xkp1_est", include_state_estimates),
                ("ykp1_est", include_output_estimates),
                ("yk_pred", include_output_predictions),
            ]
        )
        super().__init__(A, B, C, D, K, None, outputs, dt=dt, name=name)


class KFFSS(_KFSS):
    """Steady-state Kalman filter (fixed correction gain) in filtering
    form, which estimates the states at the current time instant,
    x_est(k|k), given the data up to time k.

    The model of the plant is the linear system

        x(k+1) = A x(k) + B u(k) + w(k)
          y(k) = C x(k) + D u(k) + v(k)

    A, B, C, D and K (the correction gain in filtering form, Kf) are
    matrices of numbers or SX symbols (the symbols become the filter's
    parameters). For the optimal (Kalman) gain given noise covariances
    Q and R, see make_steady_state_KFF_with_fixedABCDQR.

    Filter system (a linear system, with Kp = A K, the correction gain
    in prediction form):
        inputs: [u(k); y(k)], names uk..., yk...
        states: x_est(k|k-1), names xkp1_est...
        state transition:
            x_est(k+1|k) = (A - Kp C) x_est(k|k-1)
                           + (B - Kp D) u(k) + Kp y(k)
        outputs (each optional, in this order):
            x_est(k|k) = (I - K C) x_est(k|k-1) - K D u(k) + K y(k),
                names xk_est...
            y_est(k|k) = C x_est(k|k) + D u(k), names yk_est...
            y_est(k|k-1) = C x_est(k|k-1) + D u(k), names yk_pred...

    x_est(k|k) and y_est(k|k) depend directly on y(k), and on u(k) only
    if D isn't zero (direct feedthrough), as in KFF (see its docstring
    for which controllers it can be connected to).

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    K : (n, ny) matrix
        Correction gain in filtering form, Kf.
    dt : float, optional
        Sample time.
    name : str, optional
        Name of the system.
    include_state_estimates : bool, default True
        Include x_est(k|k) in the outputs.
    include_output_estimates : bool, default False
        Include y_est(k|k) in the outputs.
    include_output_predictions : bool, default False
        Include y_est(k|k-1) in the outputs.
    """

    def __init__(
        self,
        A,
        B,
        C,
        D,
        K,
        dt=None,
        name=None,
        include_state_estimates=True,
        include_output_estimates=False,
        include_output_predictions=False,
    ):
        outputs = _output_list(
            [
                ("xk_est", include_state_estimates),
                ("yk_est", include_output_estimates),
                ("yk_pred", include_output_predictions),
            ]
        )
        A_sx = cas.SX(as_casadi_matrix(A, "A"))
        Kf = cas.SX(as_casadi_matrix(K, "K"))
        if Kf.shape[0] != A_sx.shape[0]:
            raise ValueError(f"K is {Kf.shape}, not (n, ny)")
        # (the base class checks Kp = A Kf is (n, ny))
        super().__init__(A, B, C, D, A_sx @ Kf, Kf, outputs, dt=dt, name=name)


def make_steady_state_KFP_with_fixedABCDQR(
    A, B, C, D, Q, R, dt=None, name=None, **kwargs
):
    """A steady-state Kalman filter in prediction form for the linear
    system with fixed (numerical) matrices A, B, C, D and noise
    covariances Q and R: a KFPSS with the optimal gain K = A Kf, from
    the solution of the discrete algebraic Riccati equation (see
    steady_state_kf_gain). kwargs are KFPSS's output options.

    The model of the plant is the linear system

        x(k+1) = A x(k) + B u(k) + w(k),  w(k) ~ N(0, Q)
          y(k) = C x(k) + D u(k) + v(k),  v(k) ~ N(0, R)

    It gives the same estimates as KFP when the latter's initial error
    covariance P(0|-1) is the steady-state P(k+1|k).
    """
    A, B, C, D, Q, R = (
        _as_numeric_matrix(value, name)
        for value, name in zip((A, B, C, D, Q, R), "ABCDQR")
    )
    Kf, _, _, _ = steady_state_kf_gain(A, C, Q, R)
    return KFPSS(A, B, C, D, A @ Kf, dt=dt, name=name, **kwargs)


def make_steady_state_KFF_with_fixedABCDQR(
    A, B, C, D, Q, R, dt=None, name=None, **kwargs
):
    """A steady-state Kalman filter in filtering form for the linear
    system with fixed (numerical) matrices A, B, C, D and noise
    covariances Q and R: a KFFSS with the optimal gain Kf, from the
    solution of the discrete algebraic Riccati equation (see
    steady_state_kf_gain). kwargs are KFFSS's output options.

    It gives the same estimates as KFF when the latter's initial error
    covariance P(0|-1) is the steady-state P(k+1|k).
    """
    A, B, C, D, Q, R = (
        _as_numeric_matrix(value, name)
        for value, name in zip((A, B, C, D, Q, R), "ABCDQR")
    )
    Kf, _, _, _ = steady_state_kf_gain(A, C, Q, R)
    return KFFSS(A, B, C, D, Kf, dt=dt, name=name, **kwargs)
