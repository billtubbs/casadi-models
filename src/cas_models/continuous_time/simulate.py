import casadi as cas


def make_step_function(mag=1.0, t_step=0.0):
    u0 = cas.DM(0.0)
    u_step = cas.DM(mag)
    t_step = cas.DM(t_step)
    t = cas.SX.sym("t")
    before_step = cas.Function("before_step", [], [u0])
    after_step = cas.Function("after_step", [], [u_step])
    f_cond = cas.Function.if_else("f_cond", after_step, before_step)
    y = f_cond(t >= t_step)
    return cas.Function("step", [t], [y], ["t"], ["y"])


def make_sim_step_function_RK4(f, n, nu, params=None, name="F"):
    if params is None:
        params = {}

    # Symbolic variables
    t = cas.SX.sym("t")
    dt = cas.SX.sym("dt")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    # RK4 approximation
    k1 = f(t, x, u, *params.values())
    k2 = f(t, x + dt / 2 * k1, u, *params.values())
    k3 = f(t, x + dt / 2 * k2, u, *params.values())
    k4 = f(t, x + dt * k3, u, *params.values())
    xf = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return cas.Function(
        name,
        [t, x, u, dt, *params.values()],
        [xf],
        ["t", "x", "u", "dt", *params.keys()],
        ["xf"],
    )


def make_sim_step_function_integrator(
    f, n, nu, params=None, name="F", solver='cvodes', integrator_opts=None
):
    """Create a simulation step function using CasADi's integrator
    framework.

    This function has the same signature as make_sim_step_function_RK4 but
    uses CasADi's built-in integration methods (e.g., cvodes, rk, idas)
    instead of manual RK4 implementation.

    The integrator is created with time scaling to allow dt to be a
    symbolic variable. Specifically, to integrate dx/dt = f(t, x, u) from
    t to t+dt, we use a change of variables s = (τ - t)/dt, giving
    dx/ds = dt * f(t + s*dt, x, u) integrated from s=0 to s=1.

    Args:
        f (cas.Function): CasADi Function for ODE right-hand side with
            signature (t, x, u, *params) -> rhs where rhs has shape (n, 1).
        n (int): Number of states (dimension of x).
        nu (int): Number of inputs (dimension of u).
        params (dict, optional): Dictionary of symbolic parameters used
            by f. If None, defaults to empty dict.
        name (str, optional): Name for the returned function.
            Default: "F".
        solver (str, optional): Integration method to use. Options:
            - 'cvodes': Variable-step implicit (good for stiff systems)
            - 'rk': Fixed-step Runge-Kutta method
            - 'idas': For DAE systems
            Default: 'cvodes'.
        integrator_opts (dict, optional): Options dict for the integrator.
            Common options for cvodes/idas:
            - 'abstol': Absolute tolerance (default: 1e-8)
            - 'reltol': Relative tolerance (default: 1e-6)
            - 'max_num_steps': Maximum number of steps (default: 10000)
            Common options for rk:
            - 'number_of_finite_elements': Number of integration steps
              (default: 20)

    Returns:
        cas.Function: A CasADi Function with signature
            (t, x, u, dt, *params) -> xf, identical to the signature
            returned by make_sim_step_function_RK4.

    Example:
        >>> t = cas.SX.sym("t")
        >>> x = cas.SX.sym("x", 2)
        >>> u = cas.SX.sym("u")
        >>> a = cas.SX.sym("a")
        >>> rhs = cas.vertcat(-a * x[0], x[1])
        >>> f = cas.Function("f", [t, x, u, a], [rhs],
        ...     ["t", "x", "u", "a"], ["rhs"])
        >>> # Using CVodes integrator
        >>> F_cvodes = make_sim_step_function_integrator(
        ...     f, n=2, nu=1, params={'a': a}, solver='cvodes')
        >>> # Using built-in RK integrator
        >>> F_rk = make_sim_step_function_integrator(
        ...     f, n=2, nu=1, params={'a': a}, solver='rk',
        ...     integrator_opts={'number_of_finite_elements': 4})

    Note:
        The returned function signature matches make_sim_step_function_RK4
        exactly, making it a drop-in replacement. The choice of solver
        affects accuracy, computational cost, and suitability for stiff
        systems.
    """
    if params is None:
        params = {}
    if integrator_opts is None:
        integrator_opts = {}

    # Symbolic variables for the wrapper function signature (matches RK4)
    t = cas.SX.sym("t")
    dt = cas.SX.sym("dt")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    # Create time-scaled DAE for the integrator
    # To integrate from t to t+dt with dx/dt = f(t, x, u, params),
    # we use scaled time s ∈ [0, 1] where
    # dx/ds = dt * f(t + s*dt, x, u, params)
    s = cas.SX.sym("s")  # scaled time from 0 to 1
    x_dae = cas.SX.sym("x_dae", n)
    dt_param = cas.SX.sym("dt_param")
    t_param = cas.SX.sym("t_param")
    u_param = cas.SX.sym("u_param", nu)
    params_dae = [cas.SX.sym(f"{key}_param") for key in params.keys()]

    # Current time in original coordinates
    t_current = t_param + s * dt_param

    # Time-scaled ODE right-hand side
    rhs_scaled = dt_param * f(t_current, x_dae, u_param, *params_dae)

    # Combine all parameters (dt, t, u, and model params) into single vector
    p_combined = cas.vertcat(dt_param, t_param, u_param, *params_dae)

    # Create DAE dictionary for the integrator
    dae = {
        't': s,
        'x': x_dae,
        'p': p_combined,
        'ode': rhs_scaled
    }

    # Create integrator from s=0 to s=1
    integrator = cas.integrator(
        name,
        solver,
        dae,
        0,  # t0 = 0
        1,  # tf = 1
        integrator_opts
    )

    # Call the integrator with the appropriate parameter values
    p_values = cas.vertcat(dt, t, u, *params.values())
    result = integrator(x0=x, p=p_values)
    xf = result['xf']

    # Return a CasADi Function with the same signature as RK4 version
    return cas.Function(
        name,
        [t, x, u, dt, *params.values()],
        [xf],
        ["t", "x", "u", "dt", *params.keys()],
        ["xf"],
    )


def make_n_step_simulation_function(
    F, H, n, nu, ny, nT, params=None, name=None
):
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
