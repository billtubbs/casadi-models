import casadi as cas
import numpy as np

from cas_models.discrete_time.simulate import make_n_step_simulation_function


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
    params_dae = [
        cas.SX.sym(f"{key}_param", *v.shape) for key, v in params.items()
    ]

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
        f,
        n,
        nu,
        t,
        xk,
        uk,
        dt,
        params,
        solver,
        integrator_opts,
    )

    # Return a CasADi Function without dt argument
    return cas.Function(
        name,
        [t, xk, uk, *params.values()],
        [xkp1],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )


def make_n_step_simulation_function_from_model(
    model, dt, nT, solver="cvodes", integrator_opts=None, name=None
):
    """Create a multi-step simulation function from a continuous-time model.

    Builds a fixed-dt integration step using CasADi's integrator framework,
    then wraps it in an nT-step simulation function.  Analogous to
    ``make_n_step_simulation_function_from_model`` in
    ``cas_models.discrete_time.simulate``, but starts from a continuous-time
    model and requires ``dt`` and ``solver`` for the integration step.

    Args:
        model: A continuous-time state-space model with attributes f, h,
            n, nu, ny, and params.
        dt (float): Fixed integration step size.
        nT (int): Number of time steps to simulate.
        solver (str, optional): CasADi integrator solver ('cvodes', 'rk',
            'idas').  Default 'cvodes'.
        integrator_opts (dict, optional): Options passed to the CasADi
            integrator.
        name (str, optional): Name for the compiled CasADi function.  If
            None, defaults to "{model.f.name()}_sim_{nT}_steps".

    Returns:
        cas.Function: A CasADi Function with signature
            (t_eval, U, x0, *params) -> (X, Y) where:
            - t_eval: Time vector of length nT+1
            - U: Input matrix of shape (nT, nu)
            - x0: Initial state vector of length n
            - X: State trajectory matrix of shape (nT+1, n)
            - Y: Output trajectory matrix of shape (nT+1, ny)

    Example:
        >>> model = build_cola_lv_ct_model()
        >>> sim = make_n_step_simulation_function_from_model(
        ...     model, dt=1.0, nT=300
        ... )
        >>> X, Y = sim(t_eval, U, x0, *param_values.values())
    """
    if name is None:
        name = f"{model.f.name()}_sim_{nT}_steps"

    F_step = make_sim_step_function_integrator_fixed_dt(
        model.f,
        model.n,
        model.nu,
        dt,
        params=model.params,
        solver=solver,
        integrator_opts=integrator_opts or {},
        name="F",
    )

    return make_n_step_simulation_function(
        F_step,
        model.h,
        model.n,
        model.nu,
        model.ny,
        nT,
        params=model.params,
        name=name,
    )


def make_steady_state_solver(
    model, name="ss_solver", method="newton", opts=None, auto_reduce=False
):
    """Build a steady-state solver for a continuous-time state-space model.

    Compiles a CasADi rootfinder that solves ``f(0, x, u, *params) = 0``
    for ``x``.  The solver is compiled once; the returned callable may then
    be called repeatedly with different inputs, parameter values, or initial
    guesses.

    Parameters
    ----------
    model : StateSpaceModelCT
        Continuous-time model with attributes ``f``, ``h``, ``n``, ``nu``,
        ``ny``, and ``params``.
    name : str, optional
        Name for the compiled CasADi rootfinder function.
    method : str, optional
        CasADi rootfinder algorithm.  ``"newton"`` (default) is fast with
        a good initial guess; ``"fast_newton"`` trades robustness for speed;
        ``"nlpsol"`` (wraps an NLP solver) is the most robust option for
        difficult problems.
    opts : dict, optional
        Options forwarded verbatim to ``cas.rootfinder``.
    auto_reduce : bool, optional
        If True and the model has structurally free states (states that do
        not appear in ``f``), automatically build a reduced rootfinder that
        solves only for the constrained states.  The free states are returned
        unchanged from ``x0``.  Default False.

    Returns
    -------
    callable
        A function ``(x0, u, param_vals) -> (x_ss, y_ss)`` where:

        - ``x0`` : array-like, shape ``(n,)`` — initial state guess;
          free states (when ``auto_reduce=True``) are returned as-is
        - ``u``  : array-like, shape ``(nu,)`` — input vector
        - ``param_vals`` : dict — values keyed by ``model.params``
        - ``x_ss`` : ``np.ndarray``, shape ``(n,)`` — steady-state state
        - ``y_ss`` : ``np.ndarray``, shape ``(ny,)`` — steady-state output

    Raises
    ------
    ValueError
        If one or more states do not appear in ``f`` (structurally free
        states), making the Jacobian rank-deficient.  The error message
        names the offending state indices and explains how to fix the
        problem, including the option of passing ``auto_reduce=True``.

    Notes
    -----
    For sweeps over an input range, pass the previous solution as ``x0``
    at each step (warm-starting).  Converging from a nearby point typically
    takes only a handful of Newton iterations.

    A state ``x[j]`` is *structurally free* if its value does not appear in
    any component of ``f``.  This is common in translation-invariant systems
    (e.g. absolute cart position in a cart-pole model) where the steady-state
    position is arbitrary.  With ``auto_reduce=True`` the free states are
    pinned at their ``x0`` values and only the remaining states are solved
    for.
    """
    if opts is None:
        opts = {}

    # Build SX variables matching model.params shapes.
    x_sx = cas.SX.sym("x", model.n)
    u_sx = cas.SX.sym("u", model.nu)
    p_sxs = [cas.SX.sym(k, *v.shape) for k, v in model.params.items()]

    # Residual: dx/dt at t=0 (steady state is time-invariant).
    rhs = model.f(cas.SX(0), x_sx, u_sx, *p_sxs)

    # Detect structurally free states: states absent from every rhs component.
    colind = cas.jacobian(rhs, x_sx).sparsity().colind()
    free_indices = [j for j in range(model.n) if colind[j + 1] == colind[j]]

    if free_indices:
        constrained = [j for j in range(model.n) if j not in set(free_indices)]
        if not auto_reduce:
            raise ValueError(
                f"Cannot build steady-state solver: state(s) at indices "
                f"{free_indices} do not appear in f(t, x, u, *params) and "
                f"are therefore unconstrained at steady state — any value "
                f"satisfies f=0, making the Jacobian structurally rank-"
                f"deficient.\n\n"
                f"To resolve this:\n"
                f"  • Remove these states from your model if they are "
                f"genuinely free (e.g. absolute position in a translation-"
                f"invariant system).\n"
                f"  • Pass auto_reduce=True to pin x[{free_indices}] at "
                f"their x0 values and solve only for the constrained states "
                f"at indices {constrained}."
            )

        # auto_reduce: select len(constrained) independent rows of rhs via
        # greedy column matching on the sparsity of df/dx[constrained].
        x_c_sx = cas.vertcat(*[x_sx[j] for j in constrained])
        J_c = cas.jacobian(rhs, x_c_sx)
        nz_rows, nz_cols = J_c.sparsity().get_triplet()

        rows_by_col: dict[int, list[int]] = {}
        for r, c in zip(nz_rows, nz_cols):
            rows_by_col.setdefault(r, []).append(c)

        col_assigned: set[int] = set()
        selected_rows: list[int] = []
        for r in range(model.n):
            for c in rows_by_col.get(r, []):
                if c not in col_assigned:
                    col_assigned.add(c)
                    selected_rows.append(r)
                    break
            if len(selected_rows) == len(constrained):
                break

        if len(selected_rows) < len(constrained):
            raise ValueError(
                f"auto_reduce=True failed: could not find a structurally "
                f"non-singular {len(constrained)}x{len(constrained)} "
                f"sub-system after removing free states {free_indices}. "
                f"Consider restructuring the model."
            )

        rhs_c = cas.vertcat(*[rhs[r] for r in selected_rows])
        p_sx = cas.vertcat(u_sx, *p_sxs)
        g = cas.Function("g", [x_c_sx, p_sx], [rhs_c])
        rf = cas.rootfinder(name, method, g, opts)

        def solve(x0, u, param_vals):
            x0_arr = np.asarray(x0, dtype=float).flatten()
            p_val = cas.vertcat(
                cas.DM(u), *[param_vals[k] for k in model.params]
            )
            x_c_ss = np.array(
                rf(cas.DM([x0_arr[j] for j in constrained]), p_val)
            ).flatten()
            x_ss = x0_arr.copy()
            for i, j in enumerate(constrained):
                x_ss[j] = x_c_ss[i]
            y_ss = np.array(
                model.h(
                    cas.DM(0),
                    cas.DM(x_ss),
                    cas.DM(u),
                    *[param_vals[k] for k in model.params],
                )
            ).flatten()
            return x_ss, y_ss

        return solve

    # Full rootfinder — no free states.
    p_sx = cas.vertcat(u_sx, *p_sxs)
    g = cas.Function("g", [x_sx, p_sx], [rhs])
    rf = cas.rootfinder(name, method, g, opts)

    def solve(x0, u, param_vals):
        p_val = cas.vertcat(cas.DM(u), *[param_vals[k] for k in model.params])
        x_ss = np.array(rf(cas.DM(x0), p_val)).flatten()
        y_ss = np.array(
            model.h(
                cas.DM(0),
                cas.DM(x_ss),
                cas.DM(u),
                *[param_vals[k] for k in model.params],
            )
        ).flatten()
        return x_ss, y_ss

    return solve
