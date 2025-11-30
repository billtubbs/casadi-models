"""Simulation functions for discrete-time models."""

import casadi as cas


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
