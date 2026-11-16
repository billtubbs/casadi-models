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
EKFP
    Extended Kalman filter in prediction form, of a nonlinear
    discrete-time model (KFP is the case of a linear model).
EKFF
    Extended Kalman filter in filtering form.

The classes have concise names; the functions that make them in other
ways (e.g. the optimal gain from Q and R) are named descriptively.

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
ekf_update
    One complete time step of an extended Kalman filter, given the
    model's functions F and H.
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
    is_ss_dt,
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

    S(k|k-1) = C(k) * P(k|k-1) * C'(k) + R(k)
    Kf(k) = P(k|k-1) * C'(k) * S(k|k-1)^-1
    x_est(k|k) = x_est(k|k-1) + Kf * (y(k) - y_est(k|k-1))
    P(k|k) = P(k|k-1) - Kf(k) * S(k|k-1) * Kf'(k)

    where y_est(k|k-1) = C(k) * x_est(k|k-1) + D(k) * u(k) is given.

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
    Sk_pred : (ny, ny) matrix
        Covariance of the output prediction error (the innovations)
        y(k) - y_est(k|k-1), S(k|k-1).
    """

    # Error covariance of output prediction error
    Sk_pred = C @ Pk_pred @ C.T + R

    # Correction gain
    # Kf = Pk_pred * C.T / Sk_pred  # in MATLAB
    # Numpy equivalent
    # Kf = np.linalg.lstsq(Sk_pred.T, (Pk_pred @ C.T).T, rcond=-1)[0].T
    Kf = cas.mrdivide(Pk_pred @ C.T, Sk_pred)

    # Update state predictions (prior estimates) using
    # measurements from current time to produce 'a posteriori'
    # state estimates
    xk_est = xk_pred + Kf @ (yk - yk_pred)

    # Update error covariance of state estimates
    # Pk = Pk_pred - Kf * Sk_pred * Kf';  # inaccurate
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

    return xk_est, Pk, Kf, Sk_pred


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


def kf_update(A, B, C, D, xk_pred, Pk_pred, Q, R, yk, uk=None):
    """Carry out one complete time step of a Kalman filter:
    1. Calculate the correction gain, Kf(k), from the prior
        state error covariance, P(k|k-1).
    2. Correct the state estimates at the current time using
        the current measurement, y(k):
            x_est(k|k-1) -> x_est(k|k)
            P(k|k-1) -> P(k|k)
    3. Calculate the predicted states and state error
        covariance at the next time instant:
            x_est(k+1|k)
            P(k+1|k)
        given the current input, u(k), and the system model.

    The posterior output estimates and the error covariances of the
    output estimates are not calculated here: compute those needed
    from the results, e.g. y_est(k|k) = C x_est(k|k) + D u(k).

    To do steps 1 and 2 only use: kalman_update_f.
    To do step 3 only use: kalman_predict_f.

    Numerical arguments yk and uk are converted to cas.DM. Symbolic
    arguments (cas.SX or cas.MX) are passed through, so this
    function can be used to build a cas.Function.

    Arguments:
    A, B, C, D : matrices
        Discrete-time system model matrices.
    xk_pred : (n, 1) vector
        Prior estimate of the states, x_est(k|k-1) (the previous
        time step's x_est(k+1|k)).
    Pk_pred : (n, n) matrix
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
    xkp1_est : (n, 1) vector
        Predicted states at next time instant, x_est(k+1|k).
    Pkp1 : (n, n) matrix
        Predicted state error covariance at next time instant,
        P(k+1|k).
    yk_pred : (ny, 1) vector
        Prior estimate of the system output, y_est(k|k-1) (needed for
        the correction, returned since it's already calculated).
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

    # Prior estimate of the output at the current time, y_est(k|k-1)
    yk_pred = linear_model_output(C, D, xk_pred, uk)

    # Update states based on current measurement
    xk_est, Pk, Kf, _ = kalman_update_f(C, R, xk_pred, Pk_pred, yk_pred, yk)

    # Predictions based on current inputs
    xkp1_est, Pkp1 = kalman_predict_f(A, B, Q, xk_est, Pk, uk)

    return xk_est, Pk, xkp1_est, Pkp1, yk_pred, Kf


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


def _sx_function(func):
    """A model's function as an SX function (expanded if it's an MX
    function), so the filter's graph can be built in SX."""
    return func if func.is_a("SXFunction") else func.expand()


def _model_params(model, F):
    """{name: SX symbol} of the model's parameters: its own symbols if
    they are SX, else new SX symbols of the same names and shapes."""
    params = {}
    for i, (name, p) in enumerate(model.params.items()):
        if isinstance(p, cas.SX):
            params[name] = p
        else:
            params[name] = cas.SX.sym(name, *F.size_in(3 + i))
    return params


def _merge_params(model_params, other_params):
    """The model's parameters and the symbols in Q and R, sorted by
    name (a symbol in both counts once; different symbols with the same
    name raise ValueError)."""
    params = dict(model_params)
    for name, p in other_params.items():
        if name in params and not cas.is_equal(params[name], p):
            raise ValueError(
                f"two different parameters are named {name!r} (in the model "
                "and in Q or R)"
            )
        params[name] = p
    return {name: params[name] for name in sorted(params)}


def ekf_update(F, H, xk_pred, Pk_pred, Q, R, yk, uk=None, t=0.0, params=()):
    """Carry out one complete time step of an extended Kalman filter,
    with the model's state transition and output functions F and H
    (cas.Function, (t, xk, uk, *params) -> xkp1 and yk, as those of a
    StateSpaceModelDT):
    1. Linearize the output function about the prior estimate,
        C(k) = dH/dx at (x_est(k|k-1), u(k)), predict the output
        y_est(k|k-1) = H(x_est(k|k-1), u(k)) and correct the state
        estimates with the measurement y(k) (kalman_update_f):
            x_est(k|k-1) -> x_est(k|k)
            P(k|k-1) -> P(k|k)
    2. Linearize the state transition function about the posterior
        estimate, A(k) = dF/dx at (x_est(k|k), u(k)), and predict:
            x_est(k+1|k) = F(x_est(k|k), u(k))
            P(k+1|k) = A(k) P(k|k) A'(k) + Q
    The Jacobians are computed by automatic differentiation. With a
    linear model, this is kf_update.

    Numerical arguments yk and uk are converted to cas.DM. Symbolic
    arguments (cas.SX, or cas.MX if F and H are MX functions or
    numerical) are passed through, so this function can be used to
    build a cas.Function.

    Arguments:
    F, H : cas.Function
        State transition and output functions of the model, with inputs
        named t, xk, uk, ... and outputs xkp1 and yk.
    xk_pred : (n, 1) vector
        Prior estimate of the states, x_est(k|k-1).
    Pk_pred : (n, n) matrix
        Prior estimate of the state error covariance, P(k|k-1).
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    yk : (ny, 1) vector
        System output measurements at current time k.
    uk : (nu, 1) vector, optional
        System inputs at current time k. Only optional if nu = 0.
    t : float or symbol, default 0.0
        Time at k.
    params : sequence, optional
        Values of the model's parameters, in the order of F and H's
        arguments after uk.

    Returns:
    xk_est, Pk, xkp1_est, Pkp1, yk_pred, Kf : as kf_update.
    """
    if uk is None:
        uk = cas.DM.zeros(0)
    else:
        uk = as_casadi_matrix(uk, "uk")
    yk = as_casadi_matrix(yk, "yk")
    assert uk.shape == (F.size1_in(2), 1), "uk.shape"
    assert yk.shape == (H.size1_out(0), 1), "yk.shape"
    params = list(params)
    F_x = F.factory("F_x", F.name_in(), ["jac:xkp1:xk"])
    H_x = H.factory("H_x", H.name_in(), ["jac:yk:xk"])

    # Update states based on current measurement
    Ck = H_x(t, xk_pred, uk, *params)
    yk_pred = H(t, xk_pred, uk, *params)
    xk_est, Pk, Kf, _ = kalman_update_f(Ck, R, xk_pred, Pk_pred, yk_pred, yk)

    # Predictions based on current inputs
    Ak = F_x(t, xk_est, uk, *params)
    xkp1_est = F(t, xk_est, uk, *params)
    Pkp1 = Ak @ Pk @ Ak.T + Q

    return xk_est, Pk, xkp1_est, Pkp1, yk_pred, Kf


class _KF(StateSpaceModelDT):
    """Base class of the dynamic Kalman filters KFP and KFF and the
    extended Kalman filters EKFP and EKFF (see their docstrings): a
    filter of a discrete-time model, linear or not (one step is
    ekf_update; for a linear model, the Jacobians are its A and C and
    this is the Kalman filter). The prediction and filtering forms
    differ only in their outputs.

    outputs is a list of the names of the output signals, each a vector
    named <name>_1, <name>_2, ...:
        "xk_est", "xk_est_var" : x_est(k|k), diag(P(k|k))
        "yk_est", "yk_est_var" : y_est(k|k) = H(x_est(k|k), u(k)),
            diag(C P(k|k) C')
        "xkp1_est", "xkp1_est_var" : x_est(k+1|k), diag(P(k+1|k))
        "ykp1_est", "ykp1_est_var" : y_est(k+1|k) = H(x_est(k+1|k))
            (H independent of u only), diag(C P(k+1|k) C')
        "yk_pred" : y_est(k|k-1) = H(x_est(k|k-1), u(k))
    with C the Jacobian dH/dx at the estimate. The output variances are
    those of the errors of the estimates of the noise-free outputs
    (without R).

    dt : the filter's sample time (default: the model's).
    """

    def __init__(self, model, Q, R, outputs, dt=None, name=None):
        if not is_ss_dt(model):
            raise TypeError(
                "model must be a discrete-time state-space model (convert a "
                "continuous-time model first, e.g. with "
                "StateSpaceModelDTFromCTRK4)"
            )
        n, nu, ny = model.n, model.nu, model.ny
        if dt is None:
            dt = model.dt
        Q, R = as_casadi_matrix(Q, "Q"), as_casadi_matrix(R, "R")
        if Q.shape != (n, n):
            raise ValueError(f"Q is {Q.shape}, not (n, n) = {(n, n)}")
        if R.shape != (ny, ny):
            raise ValueError(f"R is {R.shape}, not (ny, ny) = {(ny, ny)}")
        F_model = _sx_function(model.F)
        H_model = _sx_function(model.H)
        model_params = _model_params(model, F_model)
        params = _merge_params(model_params, _symbolic_params(Q, R))
        p = list(model_params.values())

        t = cas.SX.sym("t")
        uk = cas.SX.sym("uk", nu)
        yk = cas.SX.sym("yk", ny)
        # The filter's states at time k: x_est(k|k-1) and P(k|k-1)
        xk_pred = cas.SX.sym("xk_pred", n)
        Pk_pred = cas.SX.sym("Pk_pred", n, n)
        xk_est, Pk, xkp1_est, Pkp1, yk_pred, _ = ekf_update(
            F_model, H_model, xk_pred, Pk_pred, Q, R, yk, uk=uk, t=t, params=p
        )

        if "ykp1_est" in outputs or "ykp1_est_var" in outputs:
            y_next = H_model(t, xk_pred, uk, *p)
            if "ykp1_est" in outputs and cas.depends_on(y_next, uk):
                raise ValueError(
                    "y_est(k+1|k) depends on u(k+1), which isn't known at "
                    "time k, unless D = 0 (the output function doesn't "
                    "depend on u)"
                )
            if cas.depends_on(y_next, t) and dt is None:
                raise ValueError(
                    "y_est(k+1|k): the output function depends on t, so "
                    "the sample time dt is needed"
                )
        # Time at k + 1 (only used if H depends on t)
        t_next = t if dt is None else t + dt
        H_x = H_model.factory("H_x", H_model.name_in(), ["jac:yk:xk"])

        inputs = cas.vertcat(uk, yk)
        input_names = make_list_of_enumerated_names(
            "uk", nu, sep="_"
        ) + make_list_of_enumerated_names("yk", ny, sep="_")
        states = cas.vertcat(xk_pred, cas.vec(Pk_pred))
        state_names = make_list_of_enumerated_names(
            "xkp1_est", n, sep="_"
        ) + make_list_of_enumerated_names("Pkp1", n * n, sep="_")
        states_next = cas.vertcat(xkp1_est, cas.vec(Pkp1))

        def output_variance(Ck, P):
            return cas.diag(Ck @ P @ Ck.T)

        # The possible output signals, as functions so that only those
        # included in outputs are computed (and added to the graph).
        # Each output's estimate and its error variance use the same
        # covariance: P(k|k) for y_est(k|k), P(k+1|k) for y_est(k+1|k).
        # (The SX graph keeps only the operations the diagonals need.)
        # y_est(k+1|k) doesn't depend on u (checked above): u(k) is
        # passed in place of u(k+1).
        signals = {
            "xk_est": lambda: xk_est,
            "xk_est_var": lambda: cas.diag(Pk),
            "yk_est": lambda: H_model(t, xk_est, uk, *p),
            "yk_est_var": lambda: output_variance(H_x(t, xk_est, uk, *p), Pk),
            "xkp1_est": lambda: xkp1_est,
            "xkp1_est_var": lambda: cas.diag(Pkp1),
            "ykp1_est": lambda: H_model(t_next, xkp1_est, uk, *p),
            "ykp1_est_var": lambda: output_variance(
                H_x(t_next, xkp1_est, uk, *p), Pkp1
            ),
            "yk_pred": lambda: yk_pred,
        }
        values = {signal: signals[signal]() for signal in outputs}
        output_names = _signal_names(
            outputs, {s: v.shape[0] for s, v in values.items()}
        )
        y = cas.vertcat(cas.SX(0, 1), *values.values())

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


def _prediction_outputs(
    state_estimates,
    output_estimates,
    state_variance,
    output_variance,
    output_predictions,
):
    return _output_list(
        [
            ("xkp1_est", state_estimates),
            ("ykp1_est", output_estimates),
            ("xkp1_est_var", state_variance),
            ("ykp1_est_var", output_variance),
            ("yk_pred", output_predictions),
        ]
    )


def _filtering_outputs(
    state_estimates,
    output_estimates,
    state_variance,
    output_variance,
    output_predictions,
):
    return _output_list(
        [
            ("xk_est", state_estimates),
            ("yk_est", output_estimates),
            ("xk_est_var", state_variance),
            ("yk_est_var", output_variance),
            ("yk_pred", output_predictions),
        ]
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
        outputs = _prediction_outputs(
            include_state_estimates,
            include_output_estimates,
            include_state_variance,
            include_output_variance,
            include_output_predictions,
        )
        model = StateSpaceModelDTFromABCD(A, B, C, D, dt=dt)
        super().__init__(model, Q, R, outputs, dt=dt, name=name)


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
        outputs = _filtering_outputs(
            include_state_estimates,
            include_output_estimates,
            include_state_variance,
            include_output_variance,
            include_output_predictions,
        )
        model = StateSpaceModelDTFromABCD(A, B, C, D, dt=dt)
        super().__init__(model, Q, R, outputs, dt=dt, name=name)


class EKFP(_KF):
    """Extended Kalman filter in prediction form, which estimates the
    states at the next time instant, x_est(k+1|k), given the data up to
    time k.

    The model of the plant is a discrete-time model, linear or not
    (a StateSpaceModelDT: convert a continuous-time model first, e.g.
    with StateSpaceModelDTFromCTRK4), with additive noises:

        x(k+1) = F(t, x(k), u(k)) + w(k),  w(k) ~ N(0, Q)
          y(k) = H(t, x(k), u(k)) + v(k),  v(k) ~ N(0, R)

    Each time step, F and H are linearized about the latest estimates
    (Jacobians by automatic differentiation, inside the filter's graph):
    A(k) = dF/dx at (x_est(k|k), u(k)) and C(k) = dH/dx at
    (x_est(k|k-1), u(k)) (see ekf_update). With a linear model, the
    filter is KFP's. The model's parameters, and any SX symbols in Q and
    R, become the filter's parameters (e.g. to fit them).

    Filter system (inputs, states and outputs as KFP's):
        inputs: [u(k); y(k)], names uk..., yk...
        states: [x_est(k|k-1); vec(P(k|k-1))], names xkp1_est...,
            Pkp1... (P is vectorized column-wise, n * n elements)
        outputs (each optional, in this order):
            x_est(k+1|k), names xkp1_est...
            y_est(k+1|k) = H(x_est(k+1|k)), names ykp1_est...
            diag(P(k+1|k)), names xkp1_est_var...
            diag(C P(k+1|k) C'), C = dH/dx at x_est(k+1|k), names
                ykp1_est_var...
            y_est(k|k-1) = H(x_est(k|k-1), u(k)), names yk_pred...

    As with KFP, the outputs depend directly on u(k) and y(k): see KFP's
    docstring for which controllers it can be connected to.

    Arguments:
    model : StateSpaceModelDT
        Discrete-time model of the plant.
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    dt : float, optional
        Sample time (default: the model's). Needed for y_est(k+1|k) if
        H depends on t (evaluated at t + dt).
    name : str, optional
        Name of the system.
    include_state_estimates, include_output_estimates,
    include_state_variance, include_output_variance,
    include_output_predictions : bool
        The outputs, as KFP's. y_est(k+1|k) requires an output function
        that doesn't depend on u (D = 0).

    Raises:
    ValueError: if include_output_estimates is True and H depends on u.
    """

    def __init__(
        self,
        model,
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
        outputs = _prediction_outputs(
            include_state_estimates,
            include_output_estimates,
            include_state_variance,
            include_output_variance,
            include_output_predictions,
        )
        super().__init__(model, Q, R, outputs, dt=dt, name=name)


class EKFF(_KF):
    """Extended Kalman filter in filtering form, which estimates the
    states at the current time instant, x_est(k|k), given the data up
    to time k.

    The model of the plant, its linearization and the filter's
    parameters are as EKFP's (see its docstring); with a linear model,
    the filter is KFF's.

    Filter system (inputs, states and outputs as KFF's):
        inputs: [u(k); y(k)], names uk..., yk...
        states: [x_est(k|k-1); vec(P(k|k-1))], names xkp1_est...,
            Pkp1... (P is vectorized column-wise, n * n elements)
        outputs (each optional, in this order):
            x_est(k|k), names xk_est...
            y_est(k|k) = H(x_est(k|k), u(k)), names yk_est...
            diag(P(k|k)), names xk_est_var...
            diag(C P(k|k) C'), C = dH/dx at x_est(k|k), names
                yk_est_var...
            y_est(k|k-1) = H(x_est(k|k-1), u(k)), names yk_pred...

    As with KFF, the outputs depend directly on y(k), and on u(k) only
    if H does: see KFF's docstring for which controllers it can be
    connected to.

    Arguments:
    model : StateSpaceModelDT
        Discrete-time model of the plant.
    Q : (n, n) matrix
        State error covariance.
    R : (ny, ny) matrix
        Measurement error covariance.
    dt : float, optional
        Sample time (default: the model's).
    name : str, optional
        Name of the system.
    include_state_estimates, include_output_estimates,
    include_state_variance, include_output_variance,
    include_output_predictions : bool
        The outputs, as KFF's.
    """

    def __init__(
        self,
        model,
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
        outputs = _filtering_outputs(
            include_state_estimates,
            include_output_estimates,
            include_state_variance,
            include_output_variance,
            include_output_predictions,
        )
        super().__init__(model, Q, R, outputs, dt=dt, name=name)


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

    S(k|k-1) = C P C' + R
    Kf = P C' S(k|k-1)^-1
    P(k|k) = P - Kf S(k|k-1) Kf'

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
    Sk_pred : (ny, ny) array
        Covariance of the output prediction error (the innovations)
        y(k) - y_est(k|k-1), S(k|k-1).
    """
    A, C, Q, R = (
        _as_numeric_matrix(value, name)
        for value, name in zip((A, C, Q, R), "ACQR")
    )
    Pkp1, _, _ = control.dare(A.T, C.T, Q, R)
    Sk_pred = C @ Pkp1 @ C.T + R
    Kf = np.linalg.solve(Sk_pred.T, (Pkp1 @ C.T).T).T
    Pk = Pkp1 - Kf @ Sk_pred @ Kf.T
    Pk = (Pk + Pk.T) / 2
    return Kf, Pkp1, Pk, Sk_pred


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
        # Each possible output signal's rows of C and D, as functions so
        # that only those included in outputs are computed
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
