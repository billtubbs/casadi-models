"""Parameter estimation for discrete-time models.

Output-error (simulation error) estimation: the model is simulated over
a data set from a known (or estimated) initial state, and the sum of the
squared differences between the simulated and measured outputs is
minimized with cas.Opti and IPOPT.

Functions
---------
prediction_errors
    Output errors of a DT model over a data set (single shooting).
prediction_errors_ms
    Output errors of a DT model over a data set by multiple shooting, for
    parameter estimation with cas.Opti.
estimate_params
    Estimate the parameters of a DT model from input-output data.

The work is done by private core functions that take a function
make_sim(L) returning an L-step simulation function with the signature
(t_eval, U, x0, *params) -> (X, Y). The continuous-time versions in
cas_models.continuous_time.estimate use the same core with an RK4
simulation function.
"""

import casadi as cas
import numpy as np

from cas_models.discrete_time.simulate import (
    make_n_step_simulation_function_from_model,
)


def prediction_errors(model, t, U, Y, x0, params):
    """Output errors of a discrete-time model simulated over a data set.

    Simulates the model over the whole data set from x0 (single shooting)
    and returns the differences between the simulated and measured
    outputs. Use it to set up an output-error parameter estimation
    problem, e.g. with cas.Opti:

        opti.minimize(cas.sumsqr(prediction_errors(model, t, U, Y, x0, p)))

    Args:
        model: A discrete-time state-space model with F, H, n, nu, ny,
            and params attributes.
        t (array): Time vector of length nT+1.
        U (array): Inputs, shape (nT, nu). U[k] is applied from t[k] to
            t[k+1].
        Y (array): Measured outputs, shape (nT+1, ny).
        x0: Initial state, length n (numeric, or a decision variable to
            estimate it).
        params (dict or list): Parameter values or decision variables, in
            the order of model.params.

    Returns:
        E: Errors, simulated minus measured outputs, shape (nT+1, ny).
    """
    return _prediction_errors(_dt_sim(model), model, t, U, Y, x0, params)


def prediction_errors_ms(
    opti, model, t, U, Y, x0, params, n_steps, X_init=None
):
    """Output errors of a discrete-time model by multiple shooting.

    Splits the data set into segments of n_steps time steps (the last
    one is shorter if n_steps doesn't divide the number of time steps,
    nT) and simulates each segment from its own initial state. The
    initial states of the segments after the first are new decision
    variables of opti, and constraints added to opti make each segment
    start where the previous one ended. At a feasible point the errors
    are the same as those of prediction_errors() (single shooting), so
    both give the same estimates, but the multiple-shooting problem is
    less nonlinear in the parameters and less sensitive to poor initial
    guesses, especially for unstable or poorly damped models.

    Example:
        >>> opti = cas.Opti()
        >>> p = {name: opti.variable() for name in model.params}
        >>> E, X0 = prediction_errors_ms(
        ...     opti, model, t, U, Y, x0, p, n_steps=10, X_init=X_guess
        ... )
        >>> opti.minimize(cas.sumsqr(E))
        >>> opti.solver("ipopt")
        >>> sol = opti.solve()

    Args:
        opti (cas.Opti): Optimization problem to add the segment initial
            states and the continuity constraints to.
        model: A discrete-time state-space model with F, H, n, nu, ny,
            and params attributes.
        t (array): Time vector of length nT+1.
        U (array): Inputs, shape (nT, nu). U[k] is applied from t[k] to
            t[k+1].
        Y (array): Measured outputs, shape (nT+1, ny).
        x0: Initial state of the first segment, length n (numeric, or a
            decision variable to estimate it).
        params (dict or list): Parameter values or decision variables, in
            the order of model.params.
        n_steps (int): Number of time steps per segment. n_steps >= nT
            gives one segment (single shooting).
        X_init (array, optional): Initial guesses of the states, shape
            (nT+1, n), e.g. simulated with the initial parameter guesses.
            The segment initial states start from its rows at the segment
            starts. If None, they start from zero.

    Returns:
        E: Errors, simulated minus measured outputs, shape (nT+1, ny).
        X0 (list): Initial states of the segments, x0 then the decision
            variables of the others (n x 1 each).
    """
    return _prediction_errors_ms(
        _dt_sim(model), opti, model, t, U, Y, x0, params, n_steps, X_init
    )


# Core functions shared with cas_models.continuous_time.estimate. make_sim(L)
# returns an L-step simulation function (t_eval, U, x0, *params) -> (X, Y).


def _dt_sim(model):
    """Return make_sim(L) for a discrete-time model."""

    def make_sim(L):
        return make_n_step_simulation_function_from_model(model, L)

    return make_sim


def _prediction_errors(make_sim, model, t, U, Y, x0, params):
    """Core of prediction_errors()."""
    U, Y, nT = _check_data(model, t, U, Y)
    sim = make_sim(nT)
    _, Y_pred = sim(cas.DM(t), cas.DM(U), x0, *_param_list(model, params))
    return Y_pred - cas.DM(Y)


def _prediction_errors_ms(
    make_sim, opti, model, t, U, Y, x0, params, n_steps, X_init=None
):
    """Core of prediction_errors_ms()."""
    U, Y, nT = _check_data(model, t, U, Y)
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if X_init is not None:
        X_init = np.asarray(X_init, dtype=float).reshape(nT + 1, model.n)
    param_values = _param_list(model, params)

    # Segment start time steps: 0, n_steps, 2 * n_steps, ... (< nT)
    starts = list(range(0, nT, n_steps))
    sims = {}  # simulation functions by segment length
    X0 = [x0]
    errors = []
    for i, k in enumerate(starts):
        last = i == len(starts) - 1
        L = nT - k if last else n_steps
        if L not in sims:
            sims[L] = make_sim(L)
        X_seg, Y_seg = sims[L](
            cas.DM(t[k : k + L + 1]),
            cas.DM(U[k : k + L]),
            X0[i],
            *param_values,
        )
        if last:
            # The last segment also gives the output at the final time
            errors.append(Y_seg - cas.DM(Y[k:]))
        else:
            # The output at the segment end belongs to the next segment
            errors.append(Y_seg[:L, :] - cas.DM(Y[k : k + L]))
            x_next = opti.variable(model.n)
            if X_init is not None:
                opti.set_initial(x_next, X_init[k + L])
            opti.subject_to(X_seg[L, :].T == x_next)
            X0.append(x_next)

    return cas.vcat(errors), X0


def _estimate_params(
    make_sim,
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
):
    """Core of estimate_params()."""
    for name in {**(params_init or {}), **(param_bounds or {})}:
        if name not in model.params:
            raise ValueError(
                f"unknown parameter {name!r}; model parameters: "
                f"{list(model.params)}"
            )

    opti = cas.Opti()

    # Decision variables: one for each model parameter
    params = {
        name: opti.variable(*sym.shape) for name, sym in model.params.items()
    }
    if params_init is not None:
        for name, value in params_init.items():
            opti.set_initial(params[name], value)

    # Initial state: given (default zero) or estimated
    x0_value = cas.DM.zeros(model.n, 1) if x0 is None else cas.DM(x0)
    x0 = x0_value
    if estimate_x0:
        x0 = opti.variable(model.n)
        opti.set_initial(x0, x0_value)

    # Output errors
    if n_steps is None:
        errors = _prediction_errors(make_sim, model, t, U, Y, x0, params)
    else:
        # Initial guesses of the states at the segment starts, from a
        # simulation with the initial parameter guesses
        if X_init is None and params_init is not None:
            sim = make_sim(len(t) - 1)
            X_init, _ = sim(
                cas.DM(t),
                cas.DM(np.asarray(U, dtype=float).reshape(-1, model.nu)),
                x0_value,
                *[params_init.get(name, 0.0) for name in model.params],
            )
            X_init = X_init.full()
        errors, _ = _prediction_errors_ms(
            make_sim, opti, model, t, U, Y, x0, params, n_steps, X_init
        )

    # Objective: sum of squared errors between the data and the simulated
    # outputs
    opti.minimize(cas.sumsqr(errors))

    # Bounds on parameter values
    if param_bounds is not None:
        for name, (lb, ub) in param_bounds.items():
            if lb is not None:
                opti.subject_to(params[name] >= lb)
            if ub is not None:
                opti.subject_to(params[name] <= ub)

    opti.solver("ipopt", {"print_time": False}, solver_options or {})
    sol = opti.solve()

    params_est = {name: _value(sol.value(var)) for name, var in params.items()}
    x0_est = np.atleast_1d(sol.value(x0)) if estimate_x0 else None
    return params_est, x0_est, sol


def _check_data(model, t, U, Y):
    """Check the shapes of the data and return U, Y as 2-D arrays and nT."""
    nT = len(t) - 1
    U = np.asarray(U, dtype=float).reshape(-1, model.nu)
    Y = np.asarray(Y, dtype=float).reshape(-1, model.ny)
    if U.shape[0] != nT:
        raise ValueError(
            f"U has {U.shape[0]} rows, expected {nT} (one fewer than t; "
            "U[k] is applied from t[k] to t[k+1])"
        )
    if Y.shape[0] != nT + 1:
        raise ValueError(f"Y has {Y.shape[0]} rows, expected {nT + 1}")
    return U, Y, nT


def _param_list(model, params):
    """Return parameter values as a list in the order of model.params."""
    if isinstance(params, dict):
        missing = set(model.params) - set(params)
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        return [params[name] for name in model.params]
    params = list(params)
    if len(params) != len(model.params):
        raise ValueError(
            f"expected {len(model.params)} parameters, got {len(params)}"
        )
    return params


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
):
    """Estimate the parameters of a discrete-time model from input-output
    data by minimizing the sum of squared output errors.

    The model is simulated over the data set from x0 (output-error
    method), by single shooting or, if n_steps is given, by multiple
    shooting (see prediction_errors_ms), and the problem is solved with
    cas.Opti and IPOPT.

    Example:
        >>> params_est, _, sol = estimate_params(
        ...     model,
        ...     t,
        ...     U,
        ...     Y,
        ...     n_steps=20,
        ...     params_init={"K": 1.0, "T1": 2.0},
        ...     solver_options={"print_level": 0},
        ... )

    Args:
        model: A discrete-time state-space model with symbolic
            parameters (F, H, n, nu, ny and params attributes).
        t (array): Time vector of length nT+1.
        U (array): Inputs, shape (nT, nu). U[k] is applied from t[k] to
            t[k+1].
        Y (array): Measured outputs, shape (nT+1, ny).
        x0 (array, optional): Initial state, or its initial guess if
            estimate_x0 is True. If None, zeros.
        estimate_x0 (bool): If True, the initial state is also estimated.
        n_steps (int, optional): Number of time steps per multiple-
            shooting segment. If None, single shooting.
        params_init (dict, optional): Initial guesses of the parameters,
            by name. Parameters not given start at 0.
        X_init (array, optional): Initial guesses of the states for
            multiple shooting, shape (nT+1, n). If None and params_init is
            given, simulated with params_init.
        param_bounds (dict, optional): (lower, upper) bounds of
            parameters, by name. None for no bound.
        solver_options (dict, optional): IPOPT options, e.g.
            {"print_level": 0} or
            {"hessian_approximation": "limited-memory"}. The default is
            the exact Hessian, which is usually faster and more robust
            for these problems.

    Returns:
        params_est (dict): Estimated parameter values, by name, in the
            order of model.params.
        x0_est (np.ndarray or None): Estimated initial state if
            estimate_x0, else None.
        sol (cas.OptiSol): The solution (see sol.stats()).

    Raises:
        ValueError: If params_init or param_bounds contain names that
            aren't parameters of the model.
        RuntimeError: If IPOPT fails.
    """
    return _estimate_params(
        _dt_sim(model),
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


def _value(v):
    """Return a scalar solution value as a float, others as arrays."""
    v = np.asarray(v, dtype=float)
    return float(v) if v.size == 1 else v
