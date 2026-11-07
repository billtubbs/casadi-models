"""Discrete-time state-space model classes and transfer-function utilities.

Classes
-------
StateSpaceModelDT
    Base class for discrete-time state-space models.
StateSpaceModelDTFromCTRK4
    Discrete-time model obtained by discretising a CT model with RK4.
StateSpaceModelDTFromCT
    Discrete-time model obtained by discretising a CT model (CasADi integrator).
StateSpaceModelDTSISO
    Base class for SISO discrete-time models.
StateSpaceModelDTTFSISO
    Discrete-time SISO model defined by a transfer function (num/den).
StateSpaceModelDTDelay
    Discrete-time model with an explicit integer input or output delay.
StateSpaceModelDTARXSISO
    ARX (auto-regressive with exogenous input) SISO model.

Functions
---------
validate_F_function
    Check that a CasADi function is a valid discrete-time state function.
validate_H_function
    Check that a CasADi function is a valid discrete-time output function.
tf_to_ss_obs_np
    Convert a transfer function (num/den) to observer-canonical SS form (NumPy).
tf_to_ss_con_cas
    Convert a transfer function (num/den) to controller-canonical SS form (CasADi).
tf_to_ss_obs_cas
    Convert a transfer function (num/den) to observer-canonical SS form (CasADi).
poly_coefficients
    Coefficients of a polynomial in q^-1 from its roots (numeric or symbolic).
"""

from collections import OrderedDict
from dataclasses import dataclass

import casadi as cas
import numpy as np

from cas_models.continuous_time.simulate import (
    make_sim_step_function_integrator_fixed_dt,
    make_sim_step_function_RK4_fixed_dt,
)
from cas_models.param_utils import (
    make_list_of_enumerated_names,
)
from cas_models.pickle_utils import getstate, setstate
from cas_models.transformations import (
    connect_systems_in_parallel,
    connect_systems_in_series,
    negate_system,
)
from cas_models.validation import (
    as_sx_matrix,
    as_sx_scalar,
    as_sx_vector,
    validate_casadi_function_dims,
)
from cas_models.yaml_utils import (
    base_from_saved_args,
    base_save_args,
    from_yaml,
    load_model,
    save_model,
    to_yaml,
)

# Attribute names for discrete-time state-space models


def validate_F_function(F: cas.Function, n: int, nu: int, params=None):
    """Use this to check a state transition function has the correct
    arguments (excluding any parameters) and return dimensions.
    """
    arg_shapes = OrderedDict({"t": (1, 1), "xk": (n, 1), "uk": (nu, 1)})
    if params is not None:
        param_shapes = {name: param.shape for name, param in params.items()}
        arg_shapes.update(param_shapes)
    return_shapes = {"xkp1": (n, 1)}
    return validate_casadi_function_dims(
        F,
        arg_shapes=arg_shapes,
        return_shapes=return_shapes,
    )


def validate_H_function(
    H: cas.Function, n: int, nu: int, ny: int, params=None
):
    """Use this to check an output function has the corret arguments
    (excluding any parameters) and return dimensions.
    """
    arg_shapes = OrderedDict({"t": (1, 1), "xk": (n, 1), "uk": (nu, 1)})
    if params is not None:
        param_shapes = {name: param.shape for name, param in params.items()}
        arg_shapes.update(param_shapes)
    return_shapes = {"yk": (ny, 1)}
    return validate_casadi_function_dims(
        H, arg_shapes=arg_shapes, return_shapes=return_shapes
    )


@dataclass
class StateSpaceModelDT:
    """A discrete-time state-space model of a dynamical system
    of the form:

          x(k+1) = F(t, x(k), u(k), *params.values())
            y(k) = H(t, x(k), u(k), *params.values())

    """

    # Class attribute defining the naming convention for DT systems
    _attr_names = {
        "state_func": "F",
        "output_func": "H",
        "state_var": "xk",
        "input_var": "uk",
        "state_output": "xkp1",
        "output_var": "yk",
    }

    F: cas.Function
    H: cas.Function
    n: int
    nu: int
    ny: int
    dt: float
    params: dict
    name: str = None
    input_names: list[str] = None
    state_names: list[str] = None
    output_names: list[str] = None

    def __init__(
        self,
        F,
        H,
        n,
        nu=1,
        ny=1,
        dt=None,
        params=None,
        name=None,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        """Initialize a discrete-time state-space model.

          x(k+1) = F(t, x(k), u(k), *params.values())
            y(k) = H(t, x(k), u(k), *params.values())

        Args:
            F (cas.Function): State transition function with signature
                (t, xk, uk, *params.values()) -> xkp1 where xkp1 has shape
                (n, 1). Must have named inputs ["t", "xk", "uk", ...] and
                output ["xkp1"].
            H (cas.Function): Output function with signature
                (t, xk, uk, *params.values()) -> yk where yk has shape
                (ny, 1). Must have named inputs ["t", "xk", "uk", ...] and
                output ["yk"].
            n (int): Number of states (dimension of xk).
            nu (int, optional): Number of inputs (dimension of u). Default: 1.
            ny (int, optional): Number of outputs (dimension of y). Default: 1.
            params (dict, optional): Dictionary of symbolic parameters used by
                f and h functions. If None, defaults to empty dict.
            name (str, optional): Optional name for the model. Default: None.
            input_names (list[str], optional): Names for input variables.
                If None, defaults to ["u"] or ["u1", "u2", ...] for nu > 1.
            state_names (list[str], optional): Names for state variables.
                If None, defaults to ["x"] or ["x1", "x2", ...] for n > 1.
            output_names (list[str], optional): Names for output variables.
                If None, defaults to ["y"] or ["y1", "y2", ...] for ny > 1.

        Note:
            The functions F and H are validated during initialization to ensure
            they have the correct argument names and dimensional consistency.

        Example:
            >>> t = cas.SX.sym("t")
            >>> xk = cas.SX.sym("xk", 2)
            >>> uk = cas.SX.sym("uk")
            >>> a = cas.SX.sym("a")
            >>> xkp1 = cas.vertcat(-a * xk[0], xk[1])
            >>> F = cas.Function(
            ...     "F",
            ...     [t, xk, uk, a],
            ...     [xkp1],
            ...     ["t", "xk", "uk", "a"],
            ...     ["xkp1"],
            ... )
            >>> H = cas.Function(
            ...     "H",
            ...     [t, xk, uk, a],
            ...     [xk[0]],
            ...     ["t", "xk", "uk", "a"],
            ...     ["yk"],
            ... )
            >>> model = StateSpaceModelDT(
            ...     F, H, n=2, nu=1, ny=1, params={"a": a}
            ... )
        """
        self.F = F
        self.H = H
        self.n = n
        self.nu = nu
        self.ny = ny
        self.dt = dt
        if params is None:
            params = {}
        self.params = params
        self.name = name
        validate_F_function(F, n, nu, params=params)
        validate_H_function(H, n, nu, ny, params=params)
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
        """Connect two discrete-time systems in series using the * operator.

        Follows the matrix multiplication convention: for ``G1 * G2``, the
        signal flows through G2 first, then G1:

            u --> [other] --> [self] --> y

        so the combined transfer function is ``self(other(u))``, i.e.
        ``y = G1 * G2 * u``.  This matches the python-control library and
        standard linear-algebra notation where the rightmost factor acts first.

        Args:
            other: Another StateSpaceModelDT instance to connect in series.
                ``other.ny`` must equal ``self.nu``.

        Returns:
            StateSpaceModelDT: Combined system where other feeds into self.

        Example:
            >>> G1 = StateSpaceModelDTTFSISO(
            ...     num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
            ... )
            >>> G2 = StateSpaceModelDTTFSISO(
            ...     num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
            ... )
            >>> sys_combined = G1 * G2  # signal: u -> G2 -> G1 -> y
        """

        return connect_systems_in_series(
            [other, self], model_class=StateSpaceModelDT
        )

    def __add__(self, other):
        """Connect two discrete-time systems in parallel using the + operator.

        Both systems receive the same input and their outputs are summed,
        giving a combined system with the same input and output dimensions.

        Args:
            other: Another StateSpaceModelDT instance to connect in parallel.

        Returns:
            StateSpaceModelDT: Combined system where y = y_self + y_other.

        Note:
            Both systems must have the same nu and ny.
        """
        return connect_systems_in_parallel(
            [self, other], model_class=StateSpaceModelDT
        )

    def __neg__(self):
        """The system with its outputs negated, -sys: y = -h(t, x, u)
        (see cas_models.transformations.negate_system).
        """
        return negate_system(self, model_class=StateSpaceModelDT)

    def __sub__(self, other):
        """Connect two discrete-time systems in parallel with the second's
        outputs negated, sys1 - sys2 = sys1 + (-sys2): both receive the
        same input, y = y_self - y_other.

        Note:
            Both systems must have the same nu and ny.
        """
        return self + (-other)

    def __getstate__(self):
        """Pickle support: pickle.dump(model, file) works without a
        CasADi pickle context, and the symbols shared within the model
        (e.g. params and the expressions that use them) stay shared. To
        keep symbols shared between models, pickle them together inside
        a cas.global_pickle_context() (see cas_models.pickle_utils).
        """
        return getstate(self)

    def __setstate__(self, state):
        setstate(self, state)

    def save(self, folder, overwrite=False):
        """Save the model to folder: model.yaml, the class and the
        arguments that rebuild the model, and its data files (matrices
        as tab-separated text, functions as .casadi files). If the
        folder isn't empty, overwrite=True replaces the model saved
        there. See cas_models.yaml_utils.
        """
        save_model(self, folder, overwrite=overwrite)

    @classmethod
    def load(cls, folder):
        """Load a model saved with save(). The class saved must be cls
        or one of its subclasses.
        """
        return load_model(folder, cls)

    def to_yaml(self):
        """The model as YAML text, as save() writes model.yaml but with
        everything inline (functions serialized, matrices of any size),
        so that from_yaml() rebuilds the model from the text alone. See
        cas_models.yaml_utils.
        """
        return to_yaml(self)

    @classmethod
    def from_yaml(cls, text, folder=None):
        """Load a model from YAML text (from to_yaml(), or the text of
        a saved model.yaml with the folder of its data files). The
        class must be cls or one of its subclasses.
        """
        return from_yaml(text, cls, folder=folder)

    def _save_args(self):
        """The class and the arguments that rebuild the model: here, the
        base class and its fields, with the parameters as {name: shape}.
        Subclasses that can be rebuilt from their own arguments override
        this and _from_saved_args.
        """
        return base_save_args(self, StateSpaceModelDT)

    @classmethod
    def _from_saved_args(cls, args):
        """The model from the arguments of _save_args (data files read)."""
        return base_from_saved_args(cls, args)

    def describe(self):
        """Print a human-readable summary of the system."""
        attr = self._attr_names
        print(f"{type(self).__name__}:")
        print(f"  Name: {self.name}")
        print(f"  dt: {self.dt}")
        print(f"  {getattr(self, attr['state_func'])}")
        print(f"  {getattr(self, attr['output_func'])}")
        print(f"  States (n={self.n}): {self.state_names}")
        print(f"  Inputs (nu={self.nu}): {self.input_names}")
        print(f"  Outputs (ny={self.ny}): {self.output_names}")
        print(f"  Parameters: {list(self.params.keys())}")


class StateSpaceModelDTFromCTRK4(StateSpaceModelDT):
    """A discrete-time state-space model of a dynamical system
    of the form:

          x(k+1) = F(t, x(k), u(k), *params.values())
            y(k) = H(t, x(k), u(k), *params.values())

    created from a continuous-time system model using the Runge-Kutta 4
    integration scheme.
    """

    def __init__(self, model_ct, dt):
        f = model_ct.f
        n = model_ct.n
        nu = model_ct.nu
        ny = model_ct.ny
        params = model_ct.params

        # State transition function - RK4 integration with fixed time-step
        F = make_sim_step_function_RK4_fixed_dt(
            f, n, nu, dt, params=params, name="F"
        )

        # Output function - same as in continuous-time
        t = cas.SX.sym("t")
        xk = cas.SX.sym("xk", n)
        uk = cas.SX.sym("uk", nu)
        yk = model_ct.h(t, xk, uk, *params.values())
        H = cas.Function(
            "H",
            [t, xk, uk, *params.values()],
            [yk],
            ["t", "xk", "uk", *params.keys()],
            ["yk"],
        )

        # Copy information from continuous-time model
        name = model_ct.name
        input_names = model_ct.input_names
        state_names = model_ct.state_names
        output_names = model_ct.output_names

        super().__init__(
            F,
            H,
            n,
            nu=nu,
            ny=ny,
            dt=dt,
            params=params,
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class StateSpaceModelDTFromCT(StateSpaceModelDT):
    """A discrete-time state-space model of a dynamical system
    of the form:

          x(k+1) = F(t, x(k), u(k), *params.values())
            y(k) = H(t, x(k), u(k), *params.values())

    created from a continuous-time system model using CasADi's
    integrator framework.
    """

    def __init__(self, model_ct, dt, solver="cvodes", integrator_opts=None):
        """Initialize a discrete-time model from a continuous-time model.

        Args:
            model_ct: A continuous-time state-space model with attributes
                f, h, n, nu, ny, params.
            dt (float): Fixed time step for discretization.
            solver (str, optional): Integration method ('cvodes', 'rk',
                'idas'). Default: 'cvodes'.
            integrator_opts (dict, optional): Options dict for the
                integrator. Default: None.
        """
        f = model_ct.f
        n = model_ct.n
        nu = model_ct.nu
        ny = model_ct.ny
        params = model_ct.params

        # State transition function - CasADi integrator with fixed time-step
        F = make_sim_step_function_integrator_fixed_dt(
            f,
            n,
            nu,
            dt,
            params=params,
            name="F",
            solver=solver,
            integrator_opts=integrator_opts,
        )

        # Output function - same as in continuous-time
        t = cas.SX.sym("t")
        xk = cas.SX.sym("xk", n)
        uk = cas.SX.sym("uk", nu)
        yk = model_ct.h(t, xk, uk, *params.values())
        H = cas.Function(
            "H",
            [t, xk, uk, *params.values()],
            [yk],
            ["t", "xk", "uk", *params.keys()],
            ["yk"],
        )

        # Copy information from continuous-time model
        name = model_ct.name
        input_names = model_ct.input_names
        state_names = model_ct.state_names
        output_names = model_ct.output_names

        super().__init__(
            F,
            H,
            n,
            nu=nu,
            ny=ny,
            dt=dt,
            params=params,
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


class StateSpaceModelDTSISO(StateSpaceModelDT):
    """A discrete-time state-space model of a single-input,
    single output (SISO) dynamical system of the form:

          x(k+1) = F(t, x(k), u(k), *params.values())
            y(k) = H(t, x(k), u(k), *params.values())

    """

    def __init__(
        self,
        F,
        H,
        n,
        params=None,
        name=None,
        input_name=None,
        state_names=None,
        output_name=None,
        dt=None,
    ):
        input_names = None if input_name is None else [input_name]
        output_names = None if output_name is None else [output_name]
        super().__init__(
            F,
            H,
            n,
            nu=1,
            ny=1,
            dt=dt,
            params=params,
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


def tf_to_ss_obs_np(num, den):
    """Convert transfer function to observable canonical form (NumPy version).

    This function constructs the observable canonical form state-space matrices
    using NumPy arrays. The observable canonical form has a companion matrix
    structure that makes the states directly related to output derivatives.

    State-space form:
        x(k+1) = A*x(k) + B*u(k)
        y(k) = C*x(k) + D*u(k)

    where A has the form:
        A = [0   0   ...  0   -a_n    ]
            [1   0   ...  0   -a_{n-1}]
            [0   1   ...  0   -a_{n-2}]
            [...            ...        ]
            [0   0   ...  1   -a_1    ]

    B = [b_n, b_{n-1}, ..., b_1]^T, C = [0, 0, ..., 0, 1], D = b_0

    Args:
        num: Numerator polynomial coefficients in descending degree order
             [b_0, b_1, ..., b_m] (numpy array)
        den: Denominator polynomial coefficients in descending degree order
             [a_0, a_1, ..., a_n] where a_0=1 (numpy array)

    Returns:
        tuple: (A, B, C, D) state-space matrices as numpy arrays
    """

    num = np.atleast_1d(num).flatten()
    den = np.atleast_1d(den).flatten()

    # Normalize denominator
    if den[0] != 1.0:
        num = num / den[0]
        den = den / den[0]

    # Get system order
    n = len(den) - 1

    # Pad numerator to match denominator length if needed
    if len(num) < len(den):
        num = np.concatenate([np.zeros(len(den) - len(num)), num])

    # Extract coefficients (skip leading coefficient which should be 1/b_0)
    a_coeffs = den[1:]  # [a_1, a_2, ..., a_n]
    b_coeffs = num[1:]  # [b_1, b_2, ..., b_n] (skip b_0 for proper systems)

    # Construct A matrix - observable canonical form
    A = np.zeros((n, n))
    # Subdiagonal of ones
    for i in range(n - 1):
        A[i + 1, i] = 1.0
    # Last column contains NEGATIVE denominator coefficients in reverse order
    A[:, n - 1] = -a_coeffs[::-1]

    # Construct B matrix - numerator coefficients in reverse order
    B = b_coeffs[::-1].reshape(n, 1)

    # Construct C matrix - output from last state
    C = np.zeros((1, n))
    C[0, n - 1] = 1.0

    # D matrix - direct feedthrough (zero for proper systems)
    D = np.array([[num[0] if len(num) == len(den) else 0.0]])

    return A, B, C, D


def tf_to_ss_con_cas(num, den):
    """Convert transfer function to state-space representation, compatible
    with CasADi symbolic variables.

    The resulting state space represenation is in controller canonical
    form, similar to scipy's tf2ss function.

    Args:
        num: Numerator polynomial coefficients in descending degree order
             [b_0, b_1, ..., b_m] for b_0*z^m + b_1*z^(m-1) + ... + b_m
             (CasADi SX/MX/DM column vector)
        den: Denominator polynomial coefficients in descending degree order
             [a_0, a_1, ..., a_n] for a_0*z^n + a_1*z^(n-1) + ... + a_n
             (CasADi SX/MX/DM column vector)

    Returns:
        tuple: (A, B, C, D) state-space matrices in controller canonical form

    Note:
        - Denominator must be normalized (a_0 = 1.0)
        - System order K = len(den) - 1
        - For SISO systems: A is (K,K), B is (K,1), C is (1,K), D is (1,1)
    """
    # Get dimensions
    M = num.shape[0]  # numerator length
    N = den.shape[0]  # denominator length

    # System order
    K = N - 1

    # Normalize denominator (divide by first coefficient)
    # In scipy this is done via the normalize() function
    den_normalized = den / den[0]
    num_normalized = num / den[0]

    # Pad numerator to match denominator length if needed
    if M < N:
        num_padded = cas.vertcat(cas.SX.zeros(N - M, 1), num_normalized)
    else:
        num_padded = num_normalized

    # Extract D matrix (feedthrough) from first element
    D = cas.reshape(num_padded[0], 1, 1)

    # Handle special case K=1 (first order system)
    if K == 1:
        A = cas.SX.zeros(1, 1)
        B = cas.SX.ones(1, 1)
        C = num_padded[1] - num_padded[0] * den_normalized[1]
        C = cas.reshape(C, 1, 1)
        return A, B, C, D

    # Build A matrix: controller canonical form
    # First row: -[a_1, a_2, ..., a_n]
    # Remaining rows: shifted identity matrix
    A = cas.SX.zeros(K, K)
    for j in range(K):
        A[0, j] = -den_normalized[j + 1]
    for i in range(1, K):
        A[i, i - 1] = 1.0

    # Build B matrix: [1, 0, 0, ..., 0]^T
    B = cas.SX.zeros(K, 1)
    B[0] = 1.0

    # Build C matrix: num[1:] - num[0] * den[1:]
    # For SISO: C is a row vector
    C = cas.SX.zeros(1, K)
    for j in range(K):
        C[0, j] = num_padded[j + 1] - num_padded[0] * den_normalized[j + 1]

    return A, B, C, D


def tf_to_ss_obs_cas(num, den):
    """Convert transfer function to state-space representation, compatible
    with CasADi symbolic variables.

    This function constructs the observable canonical form state-space
    matrices using CasADi symbolic arrays. The observable canonical form
    has a companion matrix structure that makes the states directly
    related to output derivatives.

    State-space form:
        x(k+1) = A*x(k) + B*u(k)
        y(k) = C*x(k) + D*u(k)

    where A has the form:
        A = [0   0   ...  0   -a_n    ]
            [1   0   ...  0   -a_{n-1}]
            [0   1   ...  0   -a_{n-2}]
            [...            ...        ]
            [0   0   ...  1   -a_1    ]

    B = [b_n, b_{n-1}, ..., b_1]^T, C = [0, 0, ..., 0, 1], D = b_0

    Args:
        num: Numerator polynomial coefficients in descending degree order
             [b_0, b_1, ..., b_m] (CasADi SX/MX/DM column vector)
        den: Denominator polynomial coefficients in descending degree order
             [a_0, a_1, ..., a_n] where a_0=1 (CasADi SX/MX/DM column vector)

    Returns:
        tuple: (A, B, C, D) state-space matrices as CasADi expressions
    """
    # Get dimensions
    M = num.shape[0]  # numerator length
    N = den.shape[0]  # denominator length

    # System order
    n = N - 1

    # Normalize denominator
    den_normalized = den / den[0]
    num_normalized = num / den[0]

    # Pad numerator to match denominator length if needed
    if M < N:
        num_padded = cas.vertcat(cas.SX.zeros(N - M, 1), num_normalized)
    else:
        num_padded = num_normalized

    # Extract coefficients (skip leading coefficient which should be 1)
    a_coeffs = den_normalized[1:]  # [a_1, a_2, ..., a_n]
    b_coeffs = num_padded[1:]  # [b_1, b_2, ..., b_n]

    # Construct A matrix - observable canonical form
    A = cas.SX.zeros(n, n)
    # Subdiagonal of ones
    for i in range(n - 1):
        A[i + 1, i] = 1.0
    # Last column contains NEGATIVE denominator coefficients in reverse order
    for i in range(n):
        A[i, n - 1] = -a_coeffs[n - 1 - i]

    # Construct B matrix - numerator coefficients in reverse order
    B = cas.SX.zeros(n, 1)
    for i in range(n):
        B[i] = b_coeffs[n - 1 - i]

    # Construct C matrix - output from last state
    C = cas.SX.zeros(1, n)
    C[0, n - 1] = 1.0

    # D matrix - direct feedthrough (zero for proper systems)
    D = cas.reshape(num_padded[0], 1, 1)

    return A, B, C, D


class StateSpaceModelDTTFSISO(StateSpaceModelDTSISO):
    """A discrete-time model of a dynamical system defined by
    a transfer function of the following form:

        y(k) = G(z) u(k)

    where

                b_0*z^m + b_1*z^(m-1) + ... + b_m
        G(z) = -------------------------------------
                a_0*z^n + a_1*z^(n-1) + ... + a_n

    and z is the z-transform operator.

    The model is internally converted to state-space form using the
    observable canonical form representation:

        x(k+1) = A*x(k) + B*u(k)
          y(k) = C*x(k) + D*u(k)

    where A, B, C, D matrices are computed using the tf_to_ss_obs_cas
    function.

    """

    def __init__(
        self,
        num,
        den,
        name=None,
        input_name=None,
        state_names=None,
        output_name=None,
        dt=None,
    ):
        """Initialize a discrete-time model from a transfer function:

                    b_0*z^m + b_1*z^(m-1) + ... + b_m
            G(z) = -------------------------------------
                    a_0*z^n + a_1*z^(n-1) + ... + a_n

        where y(k) = G(z) u(k) and z is the z-transform operator.

        The transfer function is internally converted to state-space form
        using the observable canonical form representation.

        Args:
            num: Numerator polynomial coefficients in descending degree order
                [b_0, b_1, ..., b_m]. A vector of numbers and SX
                symbols or expressions, mixed (see
                cas_models.validation.as_sx_vector(); not MX).
            den: Denominator polynomial coefficients in descending degree order
                [a_0, a_1, ..., a_n]. A vector of numbers and SX
                symbols or expressions, mixed (see
                cas_models.validation.as_sx_vector(); not MX).
            name (str, optional): Optional name for the model. Default: None.
            input_name (str, optional): Name for the input variable. If None,
                defaults to "u".
            state_names (list[str], optional): Names for state variables. If
                None, defaults to ["x1", "x2", ...].
            output_name (str, optional): Name for the output variable. If None,
                defaults to "y".
            dt (float, optional): The time step, if the model has one
                (default: None). Kept as the model's dt; the state and
                output functions don't depend on it.

        Note:
            The number of states n equals len(den) - 1 (the order of the
            denominator polynomial).

        Example:
            >>> # Create model with numeric coefficients
            >>> num = cas.DM([0.2, 0.1])
            >>> den = cas.DM([1, -1.4, 0.49])
            >>> model = StateSpaceModelDTTFSISO(num=num, den=den)
            >>>
            >>> # Create model with symbolic parameters
            >>> b0, b1 = cas.SX.sym("b0"), cas.SX.sym("b1")
            >>> a1, a2 = cas.SX.sym("a1"), cas.SX.sym("a2")
            >>> num = cas.vertcat(b0, b1)
            >>> den = cas.vertcat(1, a1, a2)
            >>> model = StateSpaceModelDTTFSISO(num=num, den=den)

        """
        num = as_sx_vector(num, "num")
        den = as_sx_vector(den, "den")

        # Ensure column vectors
        if num.shape[1] != 1:
            num = num.T
        if den.shape[1] != 1:
            den = den.T

        assert num.shape[1] == 1, "num must be a column vector"
        assert den.shape[1] == 1, "den must be a column vector"

        # Use the CasADi conversion function to get state-space matrices
        A_mat, B_mat, C_mat, D_mat = tf_to_ss_obs_cas(num, den)

        # Get the state dimension from the A matrix
        n = A_mat.shape[0]

        # Create symbolic variables for the state-space equations
        t = cas.SX.sym("t")
        xk = cas.SX.sym("xk", n)
        uk = cas.SX.sym("uk")

        # State-space equations:
        # x(k+1) = A_mat * x(k) + B_mat * u(k)
        # y(k) = C_mat * x(k) + D_mat * u(k)
        xkp1 = cas.mtimes(A_mat, xk) + cas.mtimes(B_mat, uk)
        yk = cas.mtimes(C_mat, xk) + cas.mtimes(D_mat, uk)

        # Extract any symbolic variables from num and den as params
        params = {}
        for v in cas.symvar(num):
            params[v.name()] = v
        for v in cas.symvar(den):
            params[v.name()] = v

        # Create CasADi Functions for F and H
        F = cas.Function(
            "F",
            [t, xk, uk, *params.values()],
            [xkp1],
            ["t", "xk", "uk", *params.keys()],
            ["xkp1"],
        )

        H = cas.Function(
            "H",
            [t, xk, uk, *params.values()],
            [yk],
            ["t", "xk", "uk", *params.keys()],
            ["yk"],
        )

        # Store the numerator and denominator, as DM if they have no
        # symbols (DM pickles without a CasADi context)
        self.num = cas.DM(num) if num.is_constant() else num
        self.den = cas.DM(den) if den.is_constant() else den

        super().__init__(
            F,
            H,
            n,
            params=params,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
            dt=dt,
        )


class StateSpaceModelDTDelay(StateSpaceModelDT):
    """A discrete-time state-space model implementing a pure time delay.

    For a MIMO system with nu inputs and ny outputs, this model implements:

        y(k) = G * u(k - nk)

    where:
        - nk is the delay in time steps (scalar, applies to all I/O paths)
        - G is the gain matrix of shape (ny, nu)

    The delay is implemented using a shift register for each input, creating
    a state vector of dimension n = nk * nu, with the following structure:

        x(k) = [x1(k), x2(k), ..., xnu(k)]

    where each xi(k) represents the shift register for input i:
        xi(k) = [ui(k-1), ui(k-2), ..., ui(k-nk)]

    State-space matrices:
        - A: Block diagonal matrix with nu shift register blocks
        - B: Each input feeds into its corresponding shift register
        - C: Extracts delayed inputs and applies gain matrix G
        - D: Zero (no direct feedthrough)

    """

    def __init__(
        self,
        nk,
        nu=1,
        G=None,
        dt=None,
        name=None,
        input_names=None,
        state_names=None,
        output_names=None,
    ):
        """Initialize a discrete-time delay model.

        Args:
            nk (int): Delay in time steps (must be >= 1).
            nu (int, optional): Number of inputs. Default: 1.
            G (array-like, optional): Gain matrix of shape (ny, nu) mapping
                delayed inputs to outputs: numbers and SX symbols, mixed
                (see cas_models.validation.as_sx_matrix()); its symbols
                are the model's parameters. If None, defaults to identity
                matrix (ny = nu). If provided, ny is inferred from G.shape[0].
            dt (float, optional): Sample time. Default: None.
            name (str, optional): Optional name for the model. Default: None.
            input_names (list[str], optional): Names for input variables.
                If None, defaults to ["u"] or ["u1", "u2", ...] for nu > 1.
            state_names (list[str], optional): Names for state variables.
                If None, defaults to ["x1", "x2", ...].
            output_names (list[str], optional): Names for output variables.
                If None, defaults to ["y"] or ["y1", "y2", ...] for ny > 1.

        Raises:
            AssertionError: If nk < 1 or if G has incorrect dimensions.

        Example:
            >>> # SISO delay of 3 steps
            >>> model = StateSpaceModelDTDelay(nk=3)
            >>>
            >>> # MIMO delay of 5 steps with 2 inputs
            >>> model = StateSpaceModelDTDelay(nk=5, nu=2)
            >>>
            >>> # MIMO delay with custom gain matrix
            >>> G = cas.DM([[1.0, 0.5], [0.0, 2.0]])
            >>> model = StateSpaceModelDTDelay(nk=5, nu=2, G=G)

        """
        assert nk >= 1, "Delay nk must be >= 1"

        # Infer ny from G or default to nu
        if G is None:
            ny = nu
            G = cas.SX.eye(ny)
        else:
            G = as_sx_matrix(G, "G")
            ny = G.shape[0]
            assert G.shape[1] == nu, (
                f"G.shape[1] must equal nu ({nu}), got {G.shape[1]}"
            )

        # Number of states: nk states per input
        n = nk * nu

        # Build state-space matrices using SX for symbolic computation
        # A matrix: Block diagonal with shift register blocks
        A = cas.SX.zeros(n, n)
        for i in range(nu):
            # Each block is a shift register of size (nk x nk)
            block_start = i * nk
            for j in range(nk - 1):
                A[block_start + j + 1, block_start + j] = 1.0

        # B matrix: Each input feeds into first position of its shift register
        B = cas.SX.zeros(n, nu)
        for i in range(nu):
            B[i * nk, i] = 1.0

        # C matrix: Extract last state of each shift register and apply gain G
        # Last state of shift register i is at index (i+1)*nk - 1
        C_extract = cas.SX.zeros(nu, n)
        for i in range(nu):
            C_extract[i, (i + 1) * nk - 1] = 1.0

        # Apply gain matrix: C = G * C_extract
        C = cas.mtimes(cas.SX(G), C_extract)

        # D matrix: No direct feedthrough
        D = cas.SX.zeros(ny, nu)

        # Sparsify matrices to take advantage of sparse structure
        A = cas.sparsify(A)
        B = cas.sparsify(B)
        C = cas.sparsify(C)
        D = cas.sparsify(D)

        # Create symbolic variables for the state-space equations
        t = cas.SX.sym("t")
        xk = cas.SX.sym("xk", n)
        uk = cas.SX.sym("uk", nu)

        # State-space equations:
        # x(k+1) = A*x(k) + B*u(k)
        # y(k) = C*x(k) + D*u(k)
        xkp1 = cas.mtimes(A, xk) + cas.mtimes(B, uk)
        yk = cas.mtimes(C, xk) + cas.mtimes(D, uk)

        # Create CasADi Functions for F and H (with any symbols in G as
        # parameters)
        params = {v.name(): v for v in cas.symvar(G)}
        F = cas.Function(
            "F",
            [t, xk, uk, *params.values()],
            [xkp1],
            ["t", "xk", "uk", *params.keys()],
            ["xkp1"],
        )
        H = cas.Function(
            "H",
            [t, xk, uk, *params.values()],
            [yk],
            ["t", "xk", "uk", *params.keys()],
            ["yk"],
        )

        # Store delay-specific attributes
        self.nk = nk
        self.G = cas.DM(G) if G.is_constant() else G

        super().__init__(
            F,
            H,
            n,
            nu=nu,
            ny=ny,
            dt=dt,
            params=params,
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


def poly_coefficients(roots):
    """The coefficients [c_1, ..., c_n] of

        (1 - r_1 q^-1) ... (1 - r_n q^-1) = 1 + c_1 q^-1 + ... + c_n q^-n

    from the roots r_1, ..., r_n (numbers or CasADi SX scalars, e.g.
    symbols; real only, CasADi has no complex numbers). An empty list
    gives []. StateSpaceModelDTARXSISO also takes complex numeric roots
    (in conjugate pairs)."""
    c = [1]
    for r in roots:
        # Multiply by (1 - r q^-1)
        c = [
            (c[k] if k < len(c) else 0) - (r * c[k - 1] if k > 0 else 0)
            for k in range(len(c) + 1)
        ]
    return c[1:]


def _is_complex(value):
    """Whether value is a complex number, a complex NumPy array or a
    list or tuple with a complex number in it."""
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        return np.iscomplexobj(value) or (
            value.dtype == object and any(map(_is_complex, value.ravel()))
        )
    if isinstance(value, (list, tuple)):
        return any(map(_is_complex, value))
    return False


def _roots_to_coefficients(roots, name):
    """poly_coefficients(roots) as a column vector: SX if any root is
    symbolic (all real), else DM (complex roots allowed, in conjugate
    pairs). roots is a vector argument (see as_sx_vector()), or numbers
    with complex ones (a list, tuple or NumPy array)."""
    if _is_complex(roots):
        # Numbers only: as_sx_vector() refuses complex values
        values = np.ravel(np.array(roots, dtype=object))
        if any(isinstance(r, (cas.SX, cas.DM, cas.MX)) for r in values):
            raise TypeError(
                f"{name}: complex roots must all be numbers (symbolic "
                "roots must all be real)"
            )
        values = values.astype(complex)
    else:
        r = as_sx_vector(roots, name)
        if not r.is_constant():
            c = poly_coefficients([r[i] for i in range(r.numel())])
            return cas.vertcat(*c) if c else cas.SX(0, 1)
        values = np.array(cas.DM(r)).ravel()
    c = np.atleast_1d(np.poly(values))[1:]
    if np.any(np.abs(c.imag) > 1e-12 * np.maximum(1.0, np.abs(c.real))):
        raise ValueError(
            f"{name}: complex roots must come in conjugate pairs ({roots})"
        )
    return cas.DM(c.real.reshape(-1, 1))


class StateSpaceModelDTARXSISO(StateSpaceModelDTSISO):
    """A discrete-time ARX model of a dynamical system:

        A(q^-1) y(k) = B(q^-1) q^{-nk} u(k) + e(k)

    where

        A(q^-1) = 1 + a_1 q^-1 + ... + a_na q^{-na}
        B(q^-1) = b_1  + b_2 q^-1 + ... + b_nb q^{-nb+1}
        and q^-1 is the backward-in-time shift operator

    This translates into the following difference equation:

        y(k) = -a_1 y(k-1) - a_2 y(k-2) - ... - a_na y(k-na)
               + b_1 u(k-nk) + b_2 u(k-nk-1) + ... + b_nb u(k-nk-nb+1)
               + e(k)

    This is the same convention as Matlab's arx() and filter() and
    SciPy's lfilter(): the noise-free output is

        y = filter([zeros(1, nk), B], [1, A], u)

    The model is simulated without the noise term, e(k) = 0, using the
    past outputs and inputs as the states:

        x(k) = [y(k-1), ..., y(k-na), u(k-1), ..., u(k-nk-nb+1)]

    so there are n = na + nk + nb - 1 states (not a minimal realization
    when nb + nk > 1, but the states are easy to interpret and to
    initialize from data). With nk = 0, u(k) acts on y(k) directly.

    Comparing with Matlab and Octave arx():
    - Matlab (System Identification Toolbox): same convention. nk is the
      number of leading zeros of B, so a model estimated with
      arx(data, [na nb nk]) is StateSpaceModelDTARXSISO(A, B, nk=nk).
    - Octave (control package, checked with version 4.1.1): nk is one
      less than here. Octave's arx() always delays the input one step
      and adds nk steps to that, so a model estimated with
      arx(dat, 'na', na, 'nb', nb, 'nk', nk) is
      StateSpaceModelDTARXSISO(A, B, nk=nk + 1). Its default, nk = 0, is
      the usual one-step delay, and it can't estimate a model with
      direct transmission (nk = 0 here). Its documentation doesn't say
      so.
    In both cases, A here is the estimated denominator without its
    leading 1, and B the numerator without its leading zeros.

    """

    na: int
    nb: int
    nk: int = 1

    def __init__(
        self,
        A=None,
        B=None,
        na=None,
        nb=None,
        nk=1,
        name=None,
        input_name=None,
        state_names=None,
        output_name=None,
        poles=None,
        zeros=None,
        gain=None,
        dt=None,
    ):
        """Initialize a discrete-time ARX model:

        A(q^-1) y(k) = B(q^-1) q^{-nk} u(k) + e(k)

        where

            A(q^-1) = 1 + a_1 q^-1 + ... + a_na q^{-na}
            B(q^-1) = b_1  + b_2 q^-1 + ... + b_nb q^{-nb+1}
            and q^-1 is the backward-in-time shift operator

        This is internally represented as a state-space model with state
        vector:
            x(k) = [y(k-1), ..., y(k-na), u(k-1), ..., u(k-nk-nb+1)]

        Args:
            A (cas.SX, optional): Symbolic vector of A polynomial coefficients
                [a_1, ..., a_na]. If None, symbolic parameters will be created
                using na. Either A or na must be provided, but not both.
            B (cas.SX, optional): Symbolic vector of B polynomial coefficients
                [b_1, ..., b_nb]. If None, symbolic parameters will be created
                using nb. Either B or nb must be provided, but not both.
            na (int, optional): Order of the A polynomial (number of past
                outputs). Required if A is None, must be None if A is provided.
            nb (int, optional): Order of the B polynomial (number of input
                coefficients). Required if B is None, must be None if B is
                provided.
            nk (int, optional): Input delay (number of time steps). Default: 1.
            input_name (str, optional): Name for the input variable. If None,
                defaults to "u".
            state_names (list[str], optional): Names for state variables. If
                None, defaults to ["x1", "x2", ...].
            output_name (str, optional): Name for the output variable. If None,
                defaults to "y".
            poles (list, optional): The roots p_1, ..., p_na of A, instead
                of A or na:

                    A(q^-1) = (1 - p_1 q^-1) ... (1 - p_na q^-1)

                Numbers (complex ones in conjugate pairs) or real SX
                symbols (e.g. [cas.SX.sym("p_0"), cas.SX.sym("p_1")]),
                which become the model's parameters. A list, NumPy array
                or CasADi vector; [] for na = 0.
            zeros (list, optional): The roots z_1, ..., z_{nb-1} of B,
                instead of B or nb, with gain:

                    B(q^-1) = gain (1 - z_1 q^-1) ... (1 - z_{nb-1} q^-1)

                Numbers or real SX symbols, as for poles; [] for nb = 1.
            gain (optional): The leading coefficient b_1 of B, used
                with zeros (as in scipy.signal.zpk2tf; not the static
                gain B(1) / A(1)). A number or an SX symbol; if None, a
                symbol named "gain". Must be None without zeros.
            dt (float, optional): The time step, if the model has one
                (default: None). Kept as the model's dt; the state and
                output functions don't depend on it.

        Note:
            The total number of states is n = na + nk + nb - 1.
            A model made from poles and zeros is the same model as the
            one made from the coefficients they give: only the
            parameters differ when they are symbolic. The arguments
            given are kept as attributes poles, zeros and gain (None if
            not given).

        Example:
            >>> # Create ARX model with symbolic parameters
            >>> model = StateSpaceModelDTARXSISO(na=2, nb=3, nk=1)
            >>> # Or with specific coefficient values
            >>> A = cas.DM([0.5, 0.3])
            >>> B = cas.DM([1.0, 0.5, 0.2])
            >>> model = StateSpaceModelDTARXSISO(A=A, B=B, nk=1)
            >>> # Or from poles and zeros (numeric or symbolic)
            >>> model = StateSpaceModelDTARXSISO(
            ...     poles=[0.9, 0.5], zeros=[-0.4], gain=0.1
            ... )
            >>> p = [cas.SX.sym("p_0"), cas.SX.sym("p_1")]
            >>> model = StateSpaceModelDTARXSISO(poles=p, nb=2)

        """
        if poles is not None:
            assert A is None and na is None, "provide one of A, na or poles"
            A = _roots_to_coefficients(poles, "poles")
        if zeros is not None:
            assert B is None and nb is None, "provide one of B, nb or zeros"
            if gain is None:
                gain = cas.SX.sym("gain")
            gain = as_sx_scalar(gain, "gain")
            if gain.is_constant():
                gain = float(cas.DM(gain))
            B = gain * cas.vertcat(1, _roots_to_coefficients(zeros, "zeros"))
        else:
            assert gain is None, "gain is only used with zeros"
        if A is None:
            A = cas.SX.sym("a", na)
        else:
            assert na is None, "provide A or na, not both"
            A = as_sx_vector(A, "A")
            na = A.shape[0]
        if B is None:
            B = cas.SX.sym("b", nb)
        else:
            assert nb is None, "provide B or nb, not both"
            B = as_sx_vector(B, "B")
            nb = B.shape[0]
        na, nb, nk = int(na), int(nb), int(nk)
        assert na >= 0 and nb >= 1 and nk >= 0

        # States: na past outputs, then m past inputs
        m = nk + nb - 1  # largest input lag
        n = na + m
        t = cas.SX.sym("t")
        xk = cas.SX.sym("xk", n)
        uk = cas.SX.sym("uk")
        # (reshaped: CasADi returns 1 x 0, not 0 x 1, for an empty
        # slice of a 1 x 1 vector, e.g. na = 0, n = 1)
        y_past = cas.reshape(xk[0:na], na, 1)  # y(k-1), ..., y(k-na)

        # Inputs u(k), u(k-1), ..., u(k-m) and the output y(k)
        u_lags = cas.vertcat(uk, xk[na:n])
        yk = -cas.dot(A, y_past) + cas.dot(B, u_lags[nk : nk + nb])

        # Shift the past outputs and inputs by one time step:
        # x(k+1) = [y(k), ..., y(k-na+1), u(k), ..., u(k-m+1)]
        y_next = cas.vertcat(yk, xk[0 : na - 1]) if na > 0 else cas.SX(0, 1)
        xkp1 = cas.vertcat(y_next, u_lags[0:m])

        # Add any symbolic variables to params dictionary
        params = {v.name(): v for v in cas.symvar(A)}
        params.update({v.name(): v for v in cas.symvar(B)})

        F = cas.Function(
            "F",
            [t, xk, uk, *params.values()],
            [xkp1],
            ["t", "xk", "uk", *params.keys()],
            ["xkp1"],
        )

        H = cas.Function(
            "H",
            [t, xk, uk, *params.values()],
            [yk],
            ["t", "xk", "uk", *params.keys()],
            ["yk"],
        )

        self.na = na
        self.nb = nb
        self.nk = nk
        self.poles = poles
        self.zeros = zeros
        self.gain = gain if zeros is not None else None

        super().__init__(
            F,
            H,
            n,
            params=params,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
            dt=dt,
        )
