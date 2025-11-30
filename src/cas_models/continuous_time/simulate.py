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


def _rk4_step_symbolic(f, t, x, u, dt, params):
    """Compute RK4 integration step symbolically.

    Args:
        f: CasADi Function for ODE right-hand side
        t: Symbolic time variable
        x: Symbolic state variable
        u: Symbolic input variable
        dt: Symbolic or numeric time step
        params: Dictionary of symbolic parameters

    Returns:
        Symbolic expression for next state xf
    """
    k1 = f(t, x, u, *params.values())
    k2 = f(t, x + dt / 2 * k1, u, *params.values())
    k3 = f(t, x + dt / 2 * k2, u, *params.values())
    k4 = f(t, x + dt * k3, u, *params.values())
    xf = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xf


def make_sim_step_function_RK4(f, n, nu, params=None, name="F"):
    """Create a simulation step function using RK4 with variable dt.

    Args:
        f (cas.Function): CasADi Function for ODE right-hand side
        n (int): Number of states
        nu (int): Number of inputs
        params (dict, optional): Dictionary of symbolic parameters
        name (str, optional): Name for the returned function

    Returns:
        cas.Function: Function with signature (t, x, u, dt, *params) -> xf
    """
    if params is None:
        params = {}

    # Symbolic variables
    t = cas.SX.sym("t")
    dt = cas.SX.sym("dt")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    # Compute RK4 step
    xf = _rk4_step_symbolic(f, t, x, u, dt, params)

    return cas.Function(
        name,
        [t, x, u, dt, *params.values()],
        [xf],
        ["t", "x", "u", "dt", *params.keys()],
        ["xf"],
    )


def make_sim_step_function_RK4_fixed_dt(f, n, nu, dt, params=None, name="F"):
    """Create a simulation step function using RK4 with fixed dt.

    This version uses discrete-time naming conventions (xk, uk, xkp1)
    for compatibility with discrete-time models.

    Args:
        f (cas.Function): CasADi Function for ODE right-hand side
        n (int): Number of states
        nu (int): Number of inputs
        dt (float): Fixed time step
        params (dict, optional): Dictionary of symbolic parameters
        name (str, optional): Name for the returned function

    Returns:
        cas.Function: Function with signature (t, xk, uk, *params) -> xkp1
    """
    if params is None:
        params = {}

    # Symbolic variables (no dt - it's fixed)
    # Use discrete-time naming: xk, uk, xkp1
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", n)
    uk = cas.SX.sym("uk", nu)

    # Compute RK4 step with fixed dt
    xkp1 = _rk4_step_symbolic(f, t, xk, uk, dt, params)

    return cas.Function(
        name,
        [t, xk, uk, *params.values()],
        [xkp1],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )


def _integrator_step_symbolic(
    f, n, nu, t, x, u, dt, params, solver, integrator_opts
):
    """Compute integration step using CasADi's integrator framework.

    Uses time scaling to handle symbolic or numeric dt.

    Args:
        f: CasADi Function for ODE right-hand side
        n: Number of states
        nu: Number of inputs
        t: Symbolic or numeric time variable
        x: Symbolic state variable
        u: Symbolic input variable
        dt: Symbolic or numeric time step
        params: Dictionary of symbolic parameters
        solver: Integration method ('cvodes', 'rk', 'idas')
        integrator_opts: Options dict for the integrator

    Returns:
        Symbolic expression for next state xf
    """
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
    dae = {"t": s, "x": x_dae, "p": p_combined, "ode": rhs_scaled}

    # Create integrator from s=0 to s=1
    integrator = cas.integrator(
        "integrator",
        solver,
        dae,
        0,  # t0 = 0
        1,  # tf = 1
        integrator_opts,
    )

    # Call the integrator with the appropriate parameter values
    p_values = cas.vertcat(dt, t, u, *params.values())
    result = integrator(x0=x, p=p_values)
    xf = result["xf"]

    return xf


def make_sim_step_function_integrator(
    f, n, nu, params=None, name="F", solver="cvodes", integrator_opts=None
):
    """Create a simulation step function using CasADi's integrator
    framework with variable dt.

    Args:
        f (cas.Function): CasADi Function for ODE right-hand side
        n (int): Number of states
        nu (int): Number of inputs
        params (dict, optional): Dictionary of symbolic parameters
        name (str, optional): Name for the returned function
        solver (str, optional): Integration method ('cvodes', 'rk', 'idas')
        integrator_opts (dict, optional): Options dict for the integrator

    Returns:
        cas.Function: Function with signature (t, x, u, dt, *params) -> xf
    """
    if params is None:
        params = {}
    if integrator_opts is None:
        integrator_opts = {}

    # Symbolic variables for the wrapper function signature
    t = cas.SX.sym("t")
    dt = cas.SX.sym("dt")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    # Compute integration step
    xf = _integrator_step_symbolic(
        f, n, nu, t, x, u, dt, params, solver, integrator_opts
    )

    # Return a CasADi Function with variable dt
    return cas.Function(
        name,
        [t, x, u, dt, *params.values()],
        [xf],
        ["t", "x", "u", "dt", *params.keys()],
        ["xf"],
    )


def make_sim_step_function_integrator_fixed_dt(
    f, n, nu, dt, params=None, name="F", solver="cvodes", integrator_opts=None
):
    """Create a simulation step function using CasADi's integrator
    framework with fixed dt.

    This version uses discrete-time naming conventions (xk, uk, xkp1)
    for compatibility with discrete-time models.

    Args:
        f (cas.Function): CasADi Function for ODE right-hand side
        n (int): Number of states
        nu (int): Number of inputs
        dt (float): Fixed time step
        params (dict, optional): Dictionary of symbolic parameters
        name (str, optional): Name for the returned function
        solver (str, optional): Integration method ('cvodes', 'rk', 'idas')
        integrator_opts (dict, optional): Options dict for the integrator

    Returns:
        cas.Function: Function with signature (t, xk, uk, *params) -> xkp1
    """
    if params is None:
        params = {}
    if integrator_opts is None:
        integrator_opts = {}

    # Symbolic variables (no dt - it's fixed)
    # Use discrete-time naming: xk, uk, xkp1
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", n)
    uk = cas.SX.sym("uk", nu)

    # Compute integration step with fixed dt
    xkp1 = _integrator_step_symbolic(
        f, n, nu, t, xk, uk, dt, params, solver, integrator_opts,
    )

    # Return a CasADi Function without dt argument
    return cas.Function(
        name,
        [t, xk, uk, *params.values()],
        [xkp1],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )
