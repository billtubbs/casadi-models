from functools import reduce
from collections import OrderedDict

import casadi as cas
import sympy
from dataclasses import dataclass
from cas_models.param_utils import (
    make_list_of_enumerated_names,
    concatenate_lists_of_names,
    merge_param_dicts,
    make_symbolic_vars_from_kwargs,
)
from cas_models.validation import validate_casadi_function_dims


def validate_f_function(f: cas.Function, n: int, nu: int, params=None):
    """Use this to check a state transition function has the correct
    arguments (excluding any parameters) and return dimensions.
    """
    arg_shapes = OrderedDict({"t": (1, 1), "x": (n, 1), "u": (nu, 1)})
    if params is not None:
        param_shapes = {name: param.shape for name, param in params.items()}
        arg_shapes.update(param_shapes)
    return_shapes = {"rhs": (n, 1)}
    return validate_casadi_function_dims(
        f,
        arg_shapes=arg_shapes,
        return_shapes=return_shapes,
    )


def validate_h_function(
    h: cas.Function, n: int, nu: int, ny: int, params=None
):
    """Use this to check an output function has the corret arguments
    (excluding any parameters) and return dimensions.
    """
    arg_shapes = OrderedDict({"t": (1, 1), "x": (n, 1), "u": (nu, 1)})
    if params is not None:
        param_shapes = {name: param.shape for name, param in params.items()}
        arg_shapes.update(param_shapes)
    return_shapes = {"y": (ny, 1)}
    return validate_casadi_function_dims(
        h, arg_shapes=arg_shapes, return_shapes=return_shapes
    )


@dataclass
class StateSpaceModelCT:
    """A continuous-time state-space model of a dynamical system
    of the form:

        dx(t)/dt = f(t, x(t), u(t))
        y(t) = h(t, x(t), u(t))

    """

    f: cas.Function
    h: cas.Function
    n: int
    nu: int
    ny: int
    params: dict
    input_names: list[str] = None
    state_names: list[str] = None
    output_names: list[str] = None

    def __init__(
        self,
        f,
        h,
        n,
        nu=1,
        ny=1,
        params=None,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        """Initialize a continuous-time state-space model.

        dx/dt(t) = f(t, x(t), u(t), *params.values())
            y(t) = h(t, x(t), u(t), *params.values())

        Args:
            f (cas.Function): State transition function with signature
                (t, x, u, *params.values()) -> rhs where rhs has shape (n, 1).
                Must have named inputs ["t", "x", "u", ...] and output ["rhs"].
            h (cas.Function): Output function with signature
                (t, x, u, *params.values()) -> y where y has shape (ny, 1).
                Must have named inputs ["t", "x", "u", ...] and output ["y"].
            n (int): Number of states (dimension of x).
            nu (int, optional): Number of inputs (dimension of u). Default: 1.
            ny (int, optional): Number of outputs (dimension of y). Default: 1.
            params (dict, optional): Dictionary of symbolic parameters used by
                f and h functions. If None, defaults to empty dict.
            input_names (list[str], optional): Names for input variables.
                If None, defaults to ["u"] or ["u1", "u2", ...] for nu > 1.
            state_names (list[str], optional): Names for state variables.
                If None, defaults to ["x"] or ["x1", "x2", ...] for n > 1.
            output_names (list[str], optional): Names for output variables.
                If None, defaults to ["y"] or ["y1", "y2", ...] for ny > 1.

        Note:
            The functions f and h are validated during initialization to ensure
            they have the correct argument names and dimensional consistency.

        Example:
            >>> t = cas.SX.sym("t")
            >>> x = cas.SX.sym("x", 2)
            >>> u = cas.SX.sym("u")
            >>> a = cas.SX.sym("a")
            >>> rhs = cas.vertcat(-a * x[0], x[1])
            >>> f = cas.Function("f", [t, x, u, a], [rhs],
            ...     ["t", "x", "u", "a"], ["rhs"])
            >>> h = cas.Function("h", [t, x, u, a], [x[0]],
            ...     ["t", "x", "u", "a"], ["y"])
            >>> model = StateSpaceModelCT(f, h, n=2, nu=1, ny=1,
            ...     params={'a': a})
        """
        self.f = f
        self.h = h
        self.n = n
        self.nu = nu
        self.ny = ny
        if params is None:
            params = {}
        self.params = params
        validate_f_function(f, n, nu, params=params)
        validate_h_function(h, n, nu, ny, params=params)
        if input_names is None:
            input_names = make_list_of_enumerated_names("u", nu)
        self.input_names = input_names
        if state_names is None:
            state_names = make_list_of_enumerated_names("x", n)
        self.state_names = state_names
        if output_names is None:
            output_names = make_list_of_enumerated_names("y", ny)
        self.output_names = output_names


class StateSpaceModelCTSISO(StateSpaceModelCT):
    """A continuous-time state-space model of a single-input,
    single output (SISO) dynamical system of the form:

        dx(t)/dt = f(t, x(t), u(t))
        y(t) = h(t, x(t), u(t))

    """

    def __init__(
        self,
        f,
        h,
        n,
        params=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        input_names = None if input_name is None else [input_name]
        output_names = None if output_name is None else [output_name]
        super().__init__(
            f,
            h,
            n,
            nu=1,
            ny=1,
            params=params,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class StateSpaceModelCTStaticNonlinearity(StateSpaceModelCT):
    """A continous-time state-space model of a static non-linearity,
    i.e. with no dynamics:

        y(t) = h(t, x(t), u(t))

    where x(t) = [], dx(t)/dt = f(t, x(t), u(t)) = [].

    """

    def __init__(
        self,
        h,
        nu=1,
        ny=1,
        params=None,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        if params is None:
            params = {}
        symbolic_params = {}
        for param in params.values():
            for p in cas.symvar(cas.SX(param)):
                symbolic_params[p.name()] = p

        # Construct an empty ODE right-hand side
        n = 0
        t = cas.SX.sym("t")
        x = cas.SX.sym("x", n)
        u = cas.SX.sym("u", nu)
        rhs = cas.SX.zeros(n)
        f = cas.Function(
            "f",
            [t, x, u, *symbolic_params.values()],
            [rhs],
            ["t", "x", "u", *symbolic_params.keys()],
            ["rhs"],
        )

        super().__init__(
            f,
            h,
            n,
            nu=nu,
            ny=ny,
            params=symbolic_params,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class StateSpaceModelCTFromABCD(StateSpaceModelCT):
    def __init__(
        self,
        A,
        B,
        C,
        D,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        """Creates a continous-time linear state-space model of the following
        form.

        dx/dt(t) = Ax(t) + Bu(t)
            y(t) = Cx(t) + Du(t)

        """

        # Convert to CasADi sparse arrays (could be DM, SX, or MX)
        A = cas.sparsify(A)
        B = cas.sparsify(B)
        C = cas.sparsify(C)
        D = cas.sparsify(D)

        n = A.shape[0]
        assert A.shape[1] == n
        nu = B.shape[1]
        assert B.shape[0] == n
        ny = C.shape[0]
        assert C.shape[1] == n
        assert D.shape == (ny, nu)
        symbolic_params = {}
        for m in [A, B, C, D]:
            params = cas.symvar(cas.SX(m))
            for p in params:
                symbolic_params.update({p.name(): p})
        symbolic_params = {
            name: symbolic_params[name] for name in sorted(symbolic_params)
        }

        # Construct ODE right-hand side
        t = cas.SX.sym("t")
        x = cas.SX.sym("x", n)
        u = cas.SX.sym("u", nu)
        rhs = A @ x + B @ u
        f = cas.Function(
            "f",
            [t, x, u, *symbolic_params.values()],
            [rhs],
            ["t", "x", "u", *symbolic_params.keys()],
            ["rhs"],
        )

        # Construct output function
        y = C @ x + D @ u
        h = cas.Function(
            "h",
            [t, x, u, *symbolic_params.values()],
            [y],
            ["t", "x", "u", *symbolic_params.keys()],
            ["y"],
        )

        super().__init__(
            f,
            h,
            n,
            params=symbolic_params,
            nu=nu,
            ny=ny,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class SSModelCTDirectTransmission(StateSpaceModelCTFromABCD):
    def __init__(
        self,
        nu=None,
        D=None,
        input_names=None,
        output_names=None,
    ):
        """Parameters for a continuous time linear state-space model
        with no dynamics. Either D or nu must be specified. If the D
        matrix is not provided, it is set to an identity matrix of
        shape (nu, nu).
        """
        if D is None:
            D = cas.SX.eye(nu)
            ny = nu
        else:
            ny, nu = D.shape
        n = 0  # no dynamics
        A = cas.SX.zeros(n, n)
        B = cas.SX.zeros(n, nu)
        C = cas.SX.zeros(ny, n)

        super().__init__(
            A,
            B,
            C,
            D,
            input_names=input_names,
            state_names=None,
            output_names=output_names,
        )


class SSModelCTFromABCDSISO(StateSpaceModelCTFromABCD):
    def __init__(
        self,
        A,
        B,
        C,
        D,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        # Convert to CasADi sparse arrays (could be DM, SX, or MX)
        B = cas.sparsify(B)
        C = cas.sparsify(C)
        D = cas.sparsify(D)
        assert B.shape[1] == 1
        assert C.shape[0] == 1
        assert D.shape == (1, 1)
        input_names = None if input_name is None else [input_name]
        output_names = None if output_name is None else [output_name]
        super().__init__(
            A,
            B,
            C,
            D,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class SSModelCTLinearFONoGainSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        T1=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        """Parameters for a continuous time state-space model
        of a first order system with a static gain of 1 and time
        constant T1.

            G(s) = 1 / (T1 * s + 1)

        """
        params = make_symbolic_vars_from_kwargs(T1=T1)
        T1 = params["T1"]
        A = -1 / T1
        B = cas.SX(1)
        C = 1 / T1
        D = cas.sparsify(cas.SX(0))

        super().__init__(
            A,
            B,
            C,
            D,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearFOSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        K=None,
        T1=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        """Parameters for a continuous time state-space model
        of a first order system with gain K and time constant T1.

            G(s) = K / (T1 * s + 1)

        """
        params = make_symbolic_vars_from_kwargs(K=K, T1=T1)
        K = params["K"]
        T1 = params["T1"]
        A = -1 / T1
        B = cas.SX(1)
        C = K / T1
        D = cas.sparsify(cas.SX(0))
        super().__init__(
            A,
            B,
            C,
            D,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearO2SISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        K=None,
        T1=None,
        T2=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        """Parameters for a continuous time state-space model
        of a first order system with gain K and time constant T1.

            G(s) = K / ((T1 * s + 1) * (T2 * s + 1))

        """
        params = make_symbolic_vars_from_kwargs(K=K, T1=T1, T2=T2)
        K = params["K"]
        T1 = params["T1"]
        T2 = params["T2"]
        A = cas.sparsify(
            cas.blockcat([[0, 1], [-1 / (T1 * T2), (-T1 - T2) / (T1 * T2)]])
        )
        B = cas.sparsify(cas.blockcat([[0], [1]]))
        C = cas.sparsify(cas.blockcat([[K / (T1 * T2), 0]]))
        D = cas.sparsify(cas.DM(0))
        params = {"K": K, "T1": T1, "T2": T2}
        super().__init__(
            A,
            B,
            C,
            D,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearO2NoGainSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        T1=None,
        T2=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        """Parameters for a continuous time state-space model
        of a first order system with gain K and time constant T1.

            G(s) = 1 / ((T1 * s + 1) * (T2 * s + 1))

        """
        params = make_symbolic_vars_from_kwargs(T1=T1, T2=T2)
        T1 = params["T1"]
        T2 = params["T2"]
        A = cas.sparsify(
            cas.blockcat([[0, 1], [-1 / (T1 * T2), (-T1 - T2) / (T1 * T2)]])
        )
        B = cas.sparsify(cas.blockcat([[0], [1]]))
        C = cas.sparsify(cas.blockcat([[1 / (T1 * T2), 0]]))
        D = cas.sparsify(cas.DM(0))
        params = {"T1": T1, "T2": T2}
        super().__init__(
            A,
            B,
            C,
            D,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearO2UnderdampedSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        K=None,
        zeta=None,
        omega_n=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        """Parameters for a continuous time state-space model
        of a first order system with gain K and time constant T1.

            G(s) = K / (s**2 + 2 * zeta * omega_n * s + omega_n**2)

        where
            s : Laplace variable
            zeta : damping coefficient
            omega_n : natural frequency

        """
        params = make_symbolic_vars_from_kwargs(
            K=K, zeta=zeta, omega_n=omega_n
        )
        K = params["K"]
        zeta = params["zeta"]
        omega_n = params["omega_n"]
        A = cas.sparsify(
            cas.blockcat([[0, 1], [-(omega_n**2), -2 * omega_n * zeta]])
        )
        B = cas.sparsify(cas.blockcat([[0], [1]]))
        C = cas.sparsify(cas.blockcat([[K, 0]]))
        D = cas.sparsify(cas.DM(0))
        super().__init__(
            A,
            B,
            C,
            D,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


def block_diag(matrices, square=False):
    col_sizes = [m.shape[1] for m in matrices]
    rows = []
    for i, matrix in enumerate(matrices):
        row = []
        n_rows = matrix.shape[0]
        if square:
            assert n_rows == col_sizes[i], "matrices not square"
        for j in range(i):
            row.append(cas.SX.zeros(n_rows, col_sizes[j]))
        row.append(matrix)
        for j in range(i + 1, len(col_sizes)):
            row.append(cas.SX.zeros(n_rows, col_sizes[j]))
        rows.append(row)
    return cas.blockcat(rows)


def linear_systems_in_parallel(
    systems, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a collection of linear systems into one large parallel
    system.
    """
    A = cas.sparsify(
        block_diag(list(sys["A"] for sys in systems), square=True)
    )
    B = cas.sparsify(block_diag(list(sys["B"] for sys in systems)))
    C = cas.sparsify(block_diag(list(sys["C"] for sys in systems)))
    D = cas.sparsify(block_diag(list(sys["D"] for sys in systems)))
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
    )
    input_names = concatenate_lists_of_names(
        [sys.input_names for sys in systems], keys=keys, prefix=prefix
    )
    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems], keys=keys, prefix=prefix
    )
    output_names = concatenate_lists_of_names(
        [sys.output_names for sys in systems], keys=keys, prefix=prefix
    )
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "params": params,
        "input_names": input_names,
        "state_names": state_names,
        "output_names": output_names,
    }


def linear_systems_in_series(
    systems, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a sequence of linear systems into one system by connecting their
    outputs and inputs in series.
    """
    n_sys = len(systems)
    col_sizes = [sys["A"].shape[1] for sys in systems]
    A_rows = []
    B_rows = []
    C_row = []
    for i, sys in enumerate(systems):
        A = sys["A"]
        B = sys["B"]
        C = sys["C"]
        D = sys["D"]

        # Add rows to A matrix
        n_rows = A.shape[0]
        assert n_rows == col_sizes[i], "A matrix not square"
        A_row = []
        if i > 0:
            for j in range(i - 1):
                A_row.append(cas.SX.zeros(n_rows, col_sizes[j]))
            A_row.append(B @ systems[i - 1]["C"])
        A_row.append(A)
        for j in range(i + 1, n_sys):
            A_row.append(cas.SX.zeros(n_rows, col_sizes[j]))
        A_rows.append(A_row)

        # Add rows to B matrix
        if i > 0:
            B_rows.append(B @ systems[i - 1]["D"])
        else:
            B_rows.append(B)

        # Add columns to C matrix
        if i < n_sys - 1:
            C_row.append(systems[i + 1]["D"] @ C)
        else:
            C_row.append(C)

    A = cas.sparsify(cas.blockcat(A_rows))
    B = cas.sparsify(cas.vcat(B_rows))
    C = cas.sparsify(cas.hcat(C_row))
    D = reduce(cas.SX.__matmul__, (sys["D"] for sys in systems))
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
    )
    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems], keys=keys, prefix=prefix
    )
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "params": params,
        "state_names": state_names,
    }


def connect_nonlinear_systems_in_parallel(
    systems, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a collection of nonlinear systems into one large
    parallel system.
    """

    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
    )

    t = cas.SX.sym("t")

    u_signals = []
    x_states = []
    rhs_expressions = []
    y_signals = []
    for sys in systems:
        x = cas.SX.sym("x", sys.n)
        x_states.append(x)

        u = cas.SX.sym("u", sys.nu)
        u_signals.append(u)

        # f(t, x, u, *params.values())
        rhs = sys.f(t, x, u, *sys.params.values())
        rhs_expressions.append(rhs)

        # h(t, x, u, *params.values())
        y = sys.h(t, x, u, *sys.params.values())
        y_signals.append(y)

    x = cas.vcat(x_states)
    n = x.shape[0]

    u = cas.vcat(u_signals)
    nu = u.shape[0]

    rhs = cas.vcat(rhs_expressions)
    assert rhs.shape == x.shape
    f = cas.Function(
        "f",
        [t, x, u, *params.values()],
        [rhs],
        ["t", "x", "u", *params.keys()],
        ["rhs"],
    )

    y = cas.vcat(y_signals)
    ny = y.shape[0]
    h = cas.Function(
        "h",
        [t, x, u, *params.values()],
        [y],
        ["t", "x", "u", *params.keys()],
        ["y"],
    )

    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems],
        keys=keys,
        prefix=prefix,
    )
    assert len(state_names) == n
    input_names = concatenate_lists_of_names(
        [sys.input_names for sys in systems], keys=keys, prefix=prefix
    )
    assert len(input_names) == nu
    output_names = concatenate_lists_of_names(
        [sys.output_names for sys in systems], keys=keys, prefix=prefix
    )
    assert len(output_names) == ny

    combined_system = StateSpaceModelCT(
        f,
        h,
        n,
        nu,
        ny,
        params=params,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
    )

    return combined_system


def connect_nonlinear_systems_in_series(
    systems, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a series of non-linear systems by connecting
    their inputs and outputs in series.
    """

    if keys is None:
        keys = [f"{prefix}{i + 1}" for i in range(len(systems))]

    param_dicts = [sys.params for sys in systems]
    state_name_lists = [sys.state_names for sys in systems]

    t = cas.SX.sym("t")
    combined_system = systems[0]
    for i, sys2 in enumerate(systems[1:], start=1):
        sys1 = combined_system
        assert sys2.nu == sys1.ny, "incompatible dimensions"

        # System 1
        n1 = sys1.n
        nu1 = sys1.nu
        u1 = cas.SX.sym("u", nu1)
        x1 = cas.SX.sym("x", n1)
        f1 = sys1.f
        params1 = merge_param_dicts(
            param_dicts[:i], keys=keys[:i], verbose_names=verbose_names
        )
        rhs1 = f1(t, x1, u1, *params1.values())
        h1 = sys1.h
        y1 = h1(t, x1, u1, *params1.values())

        # System 2
        n2 = sys2.n
        ny2 = sys2.ny
        u2 = y1
        x2 = cas.SX.sym("x", n2)
        f2 = sys2.f
        params2 = sys2.params
        rhs2 = f2(t, x2, u2, *params2.values())
        h2 = sys2.h
        y2 = h2(t, x2, u2, *params2.values())

        # Variables of combined system
        x = cas.vcat([x2, x1])  # stack with sys2 states at top
        u = u1
        nu = nu1

        # Combined ODE rhs function
        rhs = cas.vcat([rhs2, rhs1])
        n = n1 + n2
        assert rhs.shape[0] == n
        params = merge_param_dicts(
            param_dicts[: i + 1],
            keys=keys[: i + 1],
            verbose_names=verbose_names,
        )
        f = cas.Function(
            "f",
            [t, x, u, *params.values()],
            [rhs],
            ["t", "x", "u", *params.keys()],
            ["rhs"],
        )

        # Combined output function
        y = y2
        ny = ny2
        assert y.shape[0] == ny
        h = cas.Function(
            "h",
            [t, x, u, *params.values()],
            [y],
            ["t", "x", "u", *params.keys()],
            ["y"],
        )
        combined_system = StateSpaceModelCT(f, h, n, nu, ny, params=params)

    combined_system.state_names = concatenate_lists_of_names(
        list(reversed(state_name_lists)), keys=list(reversed(keys))
    )

    return combined_system


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


def make_sim_step_function_RK4(f, h, n, nu, params=None, name="F"):
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


def make_n_step_simulation_function(F, H, n, nu, ny, nT, params=None, name=None):
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
        dt = tkp1 - tk
        uk = U[k, :].T
        xkp1 = F(tk, xk, uk, tkp1 - tk, *params.values())
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


def sympy2casadi(sympy_expr, sympy_vars, casadi_vars, sparsify=True):
    """Convert Sympy expression to CasADi symbolic expression.

    Warning: This function uses Python's exec function, and thus should not
    be used on unsanitized input.

    Also note there is a bug in Sympy version 1.13.0 so it is better to wait
    until this is fixed before relying on the StateSpace model.

    See:
      - https://github.com/sympy/sympy/issues/26827

    """

    mapping = {
        "ImmutableDenseMatrix": cas.blockcat,
        "MutableDenseMatrix": cas.blockcat,
        "Abs": cas.fabs,
    }
    f = sympy.lambdify(sympy_vars, sympy_expr, modules=[mapping, cas])

    result = f(*casadi_vars)
    if sparsify:
        return cas.sparsify(result)
    else:
        return result


def make_casadi_and_sympy_vars(var_names):
    """Makes two dictionaries containing matching symbolic variables with
    the names defined in var_names.
    """
    sympy_vars = {}
    casadi_vars = {}
    for k, shape in var_names.items():
        if shape == ():
            sympy_vars[k] = sympy.Symbol(k, *shape)
            casadi_vars[k] = cas.SX.sym(k)
        else:
            sympy_vars[k] = sympy.MatrixSymbol(k, *shape)
            casadi_vars[k] = cas.SX.sym(k, *shape)
    return sympy_vars, casadi_vars


def convert_sympy_state_space_to_casadi_SX(
    sys, sympy_vars, casadi_vars, sparsify=True
):
    """Warning: The sympy2casadi function uses Python's exec function,
    and is thus a potential security threat.
    """
    A, B, C, D = [
        sympy2casadi(item, sympy_vars, casadi_vars, sparsify=sparsify)
        for item in [
            sys.state_matrix,
            sys.input_matrix,
            sys.output_matrix,
            sys.feedforward_matrix,
        ]
    ]
    return A, B, C, D
