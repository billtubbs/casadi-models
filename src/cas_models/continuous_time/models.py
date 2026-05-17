"""Continuous-time state-space model classes.

Classes
-------
StateSpaceModelCT
    Base class for continuous-time state-space models.
StateSpaceModelCTSISO
    Base class for SISO continuous-time models.
StateSpaceModelCTStaticNonlinearity
    Model with a static nonlinearity applied to each output.
StateSpaceModelCTFromABCD
    Continuous-time model built from A, B, C, D matrices.
SSModelCTDirectTransmission
    Static gain (no dynamics): y = D * u.
SSModelCTFromABCDSISO
    SISO model built from A, B, C, D matrices.
SSModelCTLinearFONoGainSISO
    First-order SISO: G(s) = 1 / (T1*s + 1).
SSModelCTLinearIntegratorSISO
    Integrator: G(s) = 1 / s.
SSModelCTLinearFOSISO
    First-order SISO with gain: G(s) = K / (T1*s + 1).
SSModelCTLinearO2SISO
    Second-order SISO with gain: G(s) = K / ((T1*s+1)*(T2*s+1)).
SSModelCTLinearO2NoGainSISO
    Second-order SISO (gain=1): G(s) = 1 / ((T1*s+1)*(T2*s+1)).
SSModelCTLinearO2UnderdampedSISO
    Second-order underdamped SISO: G(s) = wn^2 / (s^2 + 2*z*wn*s + wn^2).
SSModelCTFromSympySS
    Continuous-time model built from a SymPy state-space representation.

Functions
---------
validate_f_function
    Check that a CasADi function is a valid continuous-time state function.
validate_h_function
    Check that a CasADi function is a valid continuous-time output function.
"""

from collections import OrderedDict
from dataclasses import dataclass

import casadi as cas

from cas_models.param_utils import (
    make_list_of_enumerated_names,
    make_symbolic_vars_from_kwargs,
)
from cas_models.sympy_conversion import (
    convert_sympy_state_space_to_casadi_SX,
    get_free_symbols_in_sympy_ss,
    make_casadi_vars_from_sympy_vars,
)
from cas_models.validation import validate_casadi_function_dims
from cas_models.transformations import connect_systems_in_series, sum_systems

# Attribute names for continuous-time state-space models

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

    # Class attribute defining the naming convention for CT systems
    _attr_names = {
        "state_func": "f",
        "output_func": "h",
        "state_var": "x",
        "input_var": "u",
        "state_output": "rhs",
        "output_var": "y",
    }

    f: cas.Function
    h: cas.Function
    n: int
    nu: int
    ny: int
    params: dict
    name: str = None
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
        name=None,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        """Initialize a continuous-time state-space model.

        dx/dt(t) = f(t, x(t), u(t), *params.values())
            y(t) = h(t, x(t), u(t), *params.values())

        Args:
            f (cas.Function): State transition function with signature
                (t, x, u, *params.values()) -> rhs where rhs has
                shape (n, 1). Must have named inputs
                ["t", "x", "u", ...] and output ["rhs"].
            h (cas.Function): Output function with signature
                (t, x, u, *params.values()) -> y where y has shape
                (ny, 1). Must have named inputs ["t", "x", "u", ...]
                and output ["y"].
            n (int): Number of states (dimension of x).
            nu (int, optional): Number of inputs (dimension of u).
                Default: 1.
            ny (int, optional): Number of outputs (dimension of y).
                Default: 1.
            params (dict, optional): Dictionary of symbolic parameters
                used by f and h functions. If None, defaults to empty dict.
            name (str, optional): Optional name for the model.
                Default: None.
            input_names (list[str], optional): Names for input variables.
                If None, defaults to ["u"] or ["u1", "u2", ...]
                for nu > 1.
            state_names (list[str], optional): Names for state variables.
                If None, defaults to ["x"] or ["x1", "x2", ...]
                for n > 1.
            output_names (list[str], optional): Names for output variables.
                If None, defaults to ["y"] or ["y1", "y2", ...]
                for ny > 1.

        Note:
            The functions f and h are validated during initialization to
            ensure they have the correct argument names and dimensional
            consistency.

        Example:
            >>> t = cas.SX.sym("t")
            >>> x = cas.SX.sym("x", 2)
            >>> u = cas.SX.sym("u")
            >>> a = cas.SX.sym("a")
            >>> rhs = cas.vertcat(-a * x[0], x[1])
            >>> f = cas.Function(
            ...     "f", [t, x, u, a], [rhs], ["t", "x", "u", "a"], ["rhs"]
            ... )
            >>> h = cas.Function(
            ...     "h", [t, x, u, a], [x[0]], ["t", "x", "u", "a"], ["y"]
            ... )
            >>> model = StateSpaceModelCT(
            ...     f, h, n=2, nu=1, ny=1, params={"a": a}
            ... )
        """
        self.f = f
        self.h = h
        self.n = n
        self.nu = nu
        self.ny = ny
        if params is None:
            params = {}
        self.params = params
        self.name = name
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

    def __mul__(self, other):
        """Connect two continuous-time systems in series using the * operator.

        This allows easy composition of systems where the output of self is
        connected to the input of other.

        Args:
            other: Another StateSpaceModelCT instance to connect in series.

        Returns:
            StateSpaceModelCT: Combined system where self -> other.

        Example:
            >>> sys1 = SSModelCTLinearFOSISO(K=2, T1=1)
            >>> sys2 = SSModelCTLinearFOSISO(K=3, T1=2)
            >>> sys_combined = sys1 * sys2  # Connect in series

        Note:
            The output dimension of self must match the input dimension of
            other (self.ny == other.nu).
        """
        return connect_systems_in_series(
            [self, other], model_class=StateSpaceModelCT
        )

    def __add__(self, other):
        """Connect two continuous-time systems in parallel using the + operator.

        Both systems receive the same input and their outputs are summed,
        giving a combined system with the same input and output dimensions.

        Args:
            other: Another StateSpaceModelCT instance to connect in parallel.

        Returns:
            StateSpaceModelCT: Combined system where y = y_self + y_other.

        Note:
            Both systems must have the same nu and ny.
        """
        return sum_systems([self, other], model_class=StateSpaceModelCT)

    def describe(self):
        """Print a human-readable summary of the system."""
        attr = self._attr_names
        print(f"{type(self).__name__}:")
        print(f"  Name: {self.name}")
        print(f"  {getattr(self, attr['state_func'])}")
        print(f"  {getattr(self, attr['output_func'])}")
        print(f"  States (n={self.n}): {self.state_names}")
        print(f"  Inputs (nu={self.nu}): {self.input_names}")
        print(f"  Outputs (ny={self.ny}): {self.output_names}")
        print(f"  Parameters: {list(self.params.keys())}")


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
        name=None,
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
            name=name,
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
        name=None,
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
            name=name,
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
        name=None,
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
            name_key: symbolic_params[name_key]
            for name_key in sorted(symbolic_params)
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
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class SSModelCTDirectTransmission(StateSpaceModelCTFromABCD):
    def __init__(
        self,
        nu=None,
        D=None,
        name=None,
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
            name=name,
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
        name=None,
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
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class SSModelCTLinearFONoGainSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        T1=None,
        name=None,
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
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearIntegratorSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        K=None,
        name=None,
        input_name=None,
        state_names=None,
        output_name=None,
    ):
        """Parameters for a continuous time state-space model
        of an integrator system with gain K.

            G(s) = K / s

        """
        params = make_symbolic_vars_from_kwargs(K=K)
        K = params["K"]
        A = cas.sparsify(cas.SX(0))
        B = cas.SX(1)
        C = cas.SX(K)
        D = cas.sparsify(cas.SX(0))
        super().__init__(
            A,
            B,
            C,
            D,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearFOSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        K=None,
        T1=None,
        name=None,
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
            name=name,
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
        name=None,
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
        super().__init__(
            A,
            B,
            C,
            D,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTLinearO2NoGainSISO(SSModelCTFromABCDSISO):
    def __init__(
        self,
        T1=None,
        T2=None,
        name=None,
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
        super().__init__(
            A,
            B,
            C,
            D,
            name=name,
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
        name=None,
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
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTFromSympySS(StateSpaceModelCTFromABCD):
    def __init__(
        self,
        sys,
        name=None,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        """Construct a continuous time linear state-space model from
        a Sympy StateSpace (symbolic) model.
        """

        # Identify symbolic variables in Sympy StateSpace model
        sympy_vars = get_free_symbols_in_sympy_ss(sys)
        casadi_vars = make_casadi_vars_from_sympy_vars(sympy_vars)

        # Create CasADi state space model matrices
        A, B, C, D = convert_sympy_state_space_to_casadi_SX(
            sys, sympy_vars, casadi_vars, sparsify=True
        )

        super().__init__(
            A,
            B,
            C,
            D,
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )
