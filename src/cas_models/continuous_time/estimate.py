"""Parameter estimation for continuous-time models.

Output-error (simulation error) estimation, as in
cas_models.discrete_time.estimate, but for continuous-time models
simulated with explicit RK4 (see make_n_step_simulation_function_RK4).
The sample times can be irregular: each time step's length is taken from
the time vector t. The inputs are held constant between samples
(zero-order hold).

These functions use the same core as the discrete-time versions, with
one extra argument, n_sub, the number of RK4 substeps per time step.
CasADi's integrators (cvodes, rk) aren't used because they are very slow
to differentiate in these problems.

Functions
---------
prediction_errors
    Output errors of a CT model over a data set (single shooting).
prediction_errors_ms
    Output errors of a CT model over a data set by multiple shooting, for
    parameter estimation with cas.Opti.
estimate_params
    Estimate the parameters of a CT model from input-output data.
"""

import numpy as np

from cas_models.continuous_time.simulate import (
    make_n_step_simulation_function_RK4,
)
from cas_models.discrete_time.estimate import (
    _estimate_params,
    _prediction_errors,
    _prediction_errors_ms,
)


def prediction_errors(model, t, U, Y, x0, params, n_sub=1):
    """Output errors of a continuous-time model simulated over a data set.

    Simulates the model over the whole data set from x0 (single
    shooting) with RK4 and returns the differences between the simulated
    and measured outputs. See the discrete-time prediction_errors().

    Args:
        model: A continuous-time state-space model with f, h, n, nu, ny,
            and params attributes.
        t (array): Sample times, length nT+1, strictly increasing (not
            necessarily evenly spaced).
        U (array): Inputs, shape (nT, nu). U[k] is applied from t[k] to
            t[k+1].
        Y (array): Measured outputs, shape (nT+1, ny).
        x0: Initial state, length n (numeric, or a decision variable to
            estimate it).
        params (dict or list): Parameter values or decision variables, in
            the order of model.params.
        n_sub (int, optional): Number of RK4 substeps per time step.

    Returns:
        E: Errors, simulated minus measured outputs, shape (nT+1, ny).
    """
    _check_time(t)
    return _prediction_errors(
        _rk4_sim(model, n_sub), model, t, U, Y, x0, params
    )


def prediction_errors_ms(
    opti, model, t, U, Y, x0, params, n_steps, X_init=None, n_sub=1
):
    """Output errors of a continuous-time model by multiple shooting.

    See the discrete-time prediction_errors_ms() for the method. The
    segments are n_steps time steps long, whatever the lengths of the
    time steps.

    Args:
        opti (cas.Opti): Optimization problem to add the segment initial
            states and the continuity constraints to.
        model: A continuous-time state-space model with f, h, n, nu, ny,
            and params attributes.
        t (array): Sample times, length nT+1, strictly increasing.
        U (array): Inputs, shape (nT, nu). U[k] is applied from t[k] to
            t[k+1].
        Y (array): Measured outputs, shape (nT+1, ny).
        x0: Initial state of the first segment, length n (numeric, or a
            decision variable to estimate it).
        params (dict or list): Parameter values or decision variables, in
            the order of model.params.
        n_steps (int): Number of time steps per segment.
        X_init (array, optional): Initial guesses of the states, shape
            (nT+1, n). If None, the segment initial states start from
            zero.
        n_sub (int, optional): Number of RK4 substeps per time step.

    Returns:
        E: Errors, simulated minus measured outputs, shape (nT+1, ny).
        X0 (list): Initial states of the segments, x0 then the decision
            variables of the others (n x 1 each).
    """
    _check_time(t)
    return _prediction_errors_ms(
        _rk4_sim(model, n_sub),
        opti,
        model,
        t,
        U,
        Y,
        x0,
        params,
        n_steps,
        X_init,
    )


def estimate_params(
    model,
    t,
    U,
    Y,
    x0=None,
    estimate_x0=False,
    n_steps=None,
    params_init=None,
    X_init=None,
    param_bounds=None,
    solver_options=None,
    n_sub=1,
):
    """Estimate the parameters of a continuous-time model from input-
    output data by minimizing the sum of squared output errors.

    Same as the discrete-time estimate_params(), with the model
    simulated by RK4 over time steps taken from t (which can be
    irregular).

    Example:
        >>> params_est, _, sol = estimate_params(
        ...     model,
        ...     t,
        ...     U,
        ...     Y,
        ...     n_sub=2,
        ...     params_init={"K": 1.0, "T1": 2.0},
        ...     param_bounds={"T1": (0.1, None)},
        ...     solver_options={"print_level": 0},
        ... )

    Args:
        model: A continuous-time state-space model with symbolic
            parameters (f, h, n, nu, ny and params attributes).
        t (array): Sample times, length nT+1, strictly increasing (not
            necessarily evenly spaced).
        U (array): Inputs, shape (nT, nu). U[k] is applied from t[k] to
            t[k+1].
        Y (array): Measured outputs, shape (nT+1, ny).
        x0 (array, optional): Initial state, or its initial guess if
            estimate_x0 is True. If None, zeros.
        estimate_x0 (bool): If True, the initial state is also estimated.
        n_steps (int, optional): Number of time steps per multiple-
            shooting segment. If None, single shooting.
        params_init (dict, optional): Initial guesses of the parameters,
            by name. Parameters not given start at 0, so give at least
            the time constants (a zero time constant gives a division by
            zero in most models).
        X_init (array, optional): Initial guesses of the states for
            multiple shooting, shape (nT+1, n). If None and params_init is
            given, simulated with params_init.
        param_bounds (dict, optional): (lower, upper) bounds of
            parameters, by name. None for no bound. Lower bounds on time
            constants keep them positive and RK4 stable.
        solver_options (dict, optional): IPOPT options, e.g.
            {"print_level": 0} or
            {"hessian_approximation": "limited-memory"}.
        n_sub (int, optional): Number of RK4 substeps per time step.
            Increase it until the estimates stop changing.

    Returns:
        params_est (dict): Estimated parameter values, by name, in the
            order of model.params.
        x0_est (np.ndarray or None): Estimated initial state if
            estimate_x0, else None.
        sol (cas.OptiSol): The solution (see sol.stats()).

    Raises:
        ValueError: If t isn't strictly increasing, or params_init or
            param_bounds contain names that aren't parameters of the
            model.
        RuntimeError: If IPOPT fails.
    """
    _check_time(t)
    return _estimate_params(
        _rk4_sim(model, n_sub),
        model,
        t,
        U,
        Y,
        x0=x0,
        estimate_x0=estimate_x0,
        n_steps=n_steps,
        params_init=params_init,
        X_init=X_init,
        param_bounds=param_bounds,
        solver_options=solver_options,
    )


def _rk4_sim(model, n_sub):
    """Return make_sim(L) for a continuous-time model (RK4)."""

    def make_sim(L):
        return make_n_step_simulation_function_RK4(model, L, n_sub=n_sub)

    return make_sim


def _check_time(t):
    """Check that the sample times are strictly increasing."""
    if np.any(np.diff(np.asarray(t, dtype=float)) <= 0):
        raise ValueError("t must be strictly increasing")
