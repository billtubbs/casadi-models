"""Simulation functions for discrete-time models.

Functions
---------
make_n_step_simulation_function
    Build a CasADi n-step simulation function from state and output functions.
make_n_step_simulation_function_from_model
    Build an n-step simulation function directly from a DT model object.
prediction_errors
    Output errors of a DT model over a data set (single shooting).
prediction_errors_ms
    Output errors of a DT model over a data set by multiple shooting, for
    parameter estimation with cas.Opti.
"""

import casadi as cas
import numpy as np


def make_n_step_simulation_function(
    F, H, n, nu, ny, nT, params=None, name=None
):
    """Create a multi-step simulation function for discrete-time systems.

    This function creates a CasADi Function that simulates a discrete-time
    system for nT time steps, taking a sequence of inputs and initial state,
    and returning the complete state and output trajectories.

    Args:
        F (cas.Function): State transition function with signature
            (t, x, u, *params) -> xf where xf is the next state.
        H (cas.Function): Output function with signature
            (t, x, u, *params) -> y where y is the output.
        n (int): Number of states (dimension of x).
        nu (int): Number of inputs (dimension of u).
        ny (int): Number of outputs (dimension of y).
        nT (int): Number of time steps to simulate.
        params (dict, optional): Dictionary of symbolic parameters used by
            F and H. If None, defaults to empty dict.
        name (str, optional): Name for the returned function. If None,
            defaults to "{F.name()}_sim_{nT}_steps".

    Returns:
        cas.Function: A CasADi Function with signature
            (t_eval, U, x0, *params) -> (X, Y) where:
            - t_eval: Time vector of length nT+1
            - U: Input matrix of shape (nT, nu)
            - x0: Initial state vector of length n
            - X: State trajectory matrix of shape (nT+1, n)
            - Y: Output trajectory matrix of shape (nT+1, ny)

    Example:
        >>> # Create state transition and output functions
        >>> F = make_sim_step_function_RK4(f, n, nu, params=params)
        >>> # Wrap with fixed dt
        >>> t_sym = cas.SX.sym("t")
        >>> xk_sym = cas.SX.sym("xk", n)
        >>> uk_sym = cas.SX.sym("uk", nu)
        >>> dt = 0.1
        >>> xkp1 = F(t_sym, xk_sym, uk_sym, dt, *params.values())
        >>> F_fixed = cas.Function(
        ...     "F",
        ...     [t_sym, xk_sym, uk_sym, *params.values()],
        ...     [xkp1],
        ...     ["t", "xk", "uk", *params.keys()],
        ...     ["xkp1"],
        ... )
        >>> # Create multi-step simulation
        >>> sim = make_n_step_simulation_function(
        ...     F_fixed, h, n, nu, ny, nT=10, params=params
        ... )
        >>> # Run simulation
        >>> t_eval = np.linspace(0, 1.0, 11)
        >>> U = np.zeros((10, nu))
        >>> x0 = np.array([1.0, 0.0])
        >>> X, Y = sim(t_eval, U, x0, *param_values.values())
    """
    if params is None:
        params = {}
    if name is None:
        name = f"{F.name()}_sim_{nT}_steps"
    t_eval = cas.SX.sym("t_eval", nT + 1)
    U = cas.SX.sym("U", nT, nu)
    x0 = cas.SX.sym("x0", n)
    X = [x0.T]
    Y = []
    xk = x0
    tk = t_eval[0]
    for k in range(nT):
        tkp1 = t_eval[k + 1]
        uk = U[k, :].T
        xkp1 = F(tk, xk, uk, *params.values())
        yk = H(tk, xk, uk, *params.values())
        X.append(xkp1.T)
        Y.append(yk.T)
        tk = tkp1
        xk = xkp1

    yk = H(tk, xk, uk, *params.values())
    Y.append(yk.T)
    X = cas.vcat(X)
    Y = cas.vcat(Y)

    return cas.Function(
        name,
        [t_eval, U, x0, *params.values()],
        [X, Y],
        ["t_eval", "U", "x0", *params.keys()],
        ["X", "Y"],
    )


def make_n_step_simulation_function_from_model(model, nT, name=None):
    """Create a multi-step simulation function from a discrete-time model.

    This is a convenience wrapper around make_n_step_simulation_function
    that extracts the necessary information from a StateSpaceModelDT object.

    Args:
        model: A discrete-time state-space model with F, H, n, nu, ny,
            and params attributes.
        nT (int): Number of time steps to simulate.
        name (str, optional): Name for the returned function. If None,
            defaults to "{model.F.name()}_sim_{nT}_steps".

    Returns:
        cas.Function: A CasADi Function with signature
            (t_eval, U, x0, *params) -> (X, Y).

    Example:
        >>> model = StateSpaceModelDT(F, H, n=2, nu=1, ny=1, params=params)
        >>> sim = make_n_step_simulation_function_from_model(model, nT=10)
        >>> X, Y = sim(t_eval, U, x0, *param_values.values())
    """
    if name is None:
        name = f"{model.F.name()}_sim_{nT}_steps"
    return make_n_step_simulation_function(
        model.F,
        model.H,
        model.n,
        model.nu,
        model.ny,
        nT,
        params=model.params,
        name=name,
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
    U, Y, nT = _check_data(model, t, U, Y)
    sim = make_n_step_simulation_function_from_model(model, nT)
    _, Y_pred = sim(cas.DM(t), cas.DM(U), x0, *_param_list(model, params))
    return Y_pred - cas.DM(Y)


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
            sims[L] = make_n_step_simulation_function_from_model(model, L)
        X_seg, Y_seg = sims[L](
            cas.DM(t[k:k + L + 1]), cas.DM(U[k:k + L]), X0[i], *param_values
        )
        if last:
            # The last segment also gives the output at the final time
            errors.append(Y_seg - cas.DM(Y[k:]))
        else:
            # The output at the segment end belongs to the next segment
            errors.append(Y_seg[:L, :] - cas.DM(Y[k:k + L]))
            x_next = opti.variable(model.n)
            if X_init is not None:
                opti.set_initial(x_next, X_init[k + L])
            opti.subject_to(X_seg[L, :].T == x_next)
            X0.append(x_next)

    return cas.vcat(errors), X0


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
