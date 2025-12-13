from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import casadi as cas

from cas_models.continuous_time.simulate import (
    make_sim_step_function_integrator_fixed_dt,
    make_sim_step_function_RK4_fixed_dt,
)
from cas_models.param_utils import (
    make_list_of_enumerated_names,
)
from cas_models.validation import validate_casadi_function_dims

# Attribute names for discrete-time state-space models
ATTR_NAMES = {
    "state_func": "F",
    "output_func": "H",
    "state_var": "xk",
    "input_var": "uk",
    "state_output": "xkp1",
    "output_var": "yk",
}


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


def is_ss_dt(sys):
    """Check if a system is a discrete-time state-space model.

    A discrete-time model is identified by having 'F' and 'H'
    attributes (uppercase), which are the state transition function
    and output function respectively.

    Args:
        sys: A system object to check

    Returns:
        bool: True if the system has discrete-time attributes (F, H),
              False otherwise
    """
    return hasattr(sys, "F") and hasattr(sys, "H")


def validate_equal_dt(systems):
    """Check all hdiscrete-time systems have the same time interval."""
    dt_values = [sys.dt for sys in systems]
    if not all(dt == dt_values[0] for dt in dt_values):
        raise ValueError(
            f"All discrete-time systems must have the same dt. "
            f"Found dt values: {dt_values}"
        )

@dataclass
class StateSpaceModelDT:
    """A discrete-time state-space model of a dynamical system
    of the form:

          x(k+1) = F(t, x(k), u(k), *params.values())
            y(k) = H(t, x(k), u(k), *params.values())

    """

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
    ):
        input_names = None if input_name is None else [input_name]
        output_names = None if output_name is None else [output_name]
        super().__init__(
            F,
            H,
            n,
            nu=1,
            ny=1,
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
                [b_0, b_1, ..., b_m]. Can be a CasADi SX/MX/DM expression,
                numpy array, or Python list. May contain symbolic parameters.
            den: Denominator polynomial coefficients in descending degree order
                [a_0, a_1, ..., a_n]. Can be a CasADi SX/MX/DM expression,
                numpy array, or Python list. May contain symbolic parameters.
            name (str, optional): Optional name for the model. Default: None.
            input_name (str, optional): Name for the input variable. If None,
                defaults to "u".
            state_names (list[str], optional): Names for state variables. If
                None, defaults to ["x1", "x2", ...].
            output_name (str, optional): Name for the output variable. If None,
                defaults to "y".

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
        # Convert num and den to CasADi SX format if needed
        if not isinstance(num, (cas.SX, cas.MX, cas.DM)):
            num = cas.SX(num)
        else:
            num = cas.SX(num)

        if not isinstance(den, (cas.SX, cas.MX, cas.DM)):
            den = cas.SX(den)
        else:
            den = cas.SX(den)

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

        # Store the numerator and denominator
        self.num = num
        self.den = den

        super().__init__(
            F,
            H,
            n,
            params=params,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
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
                delayed inputs to outputs. If None, defaults to identity
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
            G = cas.DM.eye(ny)
        else:
            G = cas.DM(G)
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

        # Create CasADi Functions for F and H
        F = cas.Function("F", [t, xk, uk], [xkp1], ["t", "xk", "uk"], ["xkp1"])
        H = cas.Function("H", [t, xk, uk], [yk], ["t", "xk", "uk"], ["yk"])

        # Store delay-specific attributes
        self.nk = nk
        self.G = G

        super().__init__(
            F,
            H,
            n,
            nu=nu,
            ny=ny,
            dt=dt,
            params={},
            name=name,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
        )


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

    The model is implemented in observable canonical form to match
    Matlab/Octave's arx() function, using a minimal state representation
    with n = max(na, nb+nk) states.

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
    ):
        """Initialize a discrete-time ARX model:

        A(q^-1) y(k) = B(q^-1) q^{-nk} u(k) + e(k)

        where

            A(q^-1) = 1 + a_1 q^-1 + ... + a_na q^{-na}
            B(q^-1) = b_1  + b_2 q^-1 + ... + b_nb q^{-nb+1}
            and q^-1 is the backward-in-time shift operator

        This is internally represented as a state-space model with state
        vector:
            x(k) = [y(k-1), ..., y(k-na), u(k-1), ..., u(k-nk-nb)]

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

        Note:
            The total number of states is n = max(na, nb + nk).

        Example:
            >>> # Create ARX model with symbolic parameters
            >>> model = StateSpaceModelDTARXSISO(na=2, nb=3, nk=1)
            >>> # Or with specific coefficient values
            >>> A = cas.DM([0.5, 0.3])
            >>> B = cas.DM([1.0, 0.5, 0.2])
            >>> model = StateSpaceModelDTARXSISO(A=A, B=B, nk=1)

        """
        if A is None:
            A = cas.SX.sym("a", na)
        else:
            assert na is None, "provide A or na, not both"
            A = cas.SX(A)
            assert A.shape[1] == 1
            na = A.shape[0]
        if B is None:
            B = cas.SX.sym("b", nb)
        else:
            assert nb is None, "provide B or nb, not both"
            B = cas.SX(B)
            assert B.shape[1] == 1
            nb = B.shape[0]

        # Construct state-space model in observable canonical form
        # (matches Matlab/Octave arx() output)
        t = cas.SX.sym("t")
        uk = cas.SX.sym("uk")

        # Minimal state dimension
        n = int(max(na, nb + nk))
        xk = cas.SX.sym("xk", n)

        # Pad coefficients to length n if needed
        A_padded = cas.sparsify(cas.vertcat(A, cas.SX.zeros(n - na, 1)))
        B_padded = cas.sparsify(cas.vertcat(B, cas.SX.zeros(n - nb, 1)))

        # Observable canonical form matrices
        # A matrix: companion form with shifts on subdiagonal and AR
        # coefficients in last column
        A_mat = cas.SX.zeros(n, n)
        for i in range(n - 1):
            # Alternating signs on subdiagonal: 1, -1, 1, -1, ...
            A_mat[i + 1, i] = (-1) ** i
        # Last column contains AR coefficients with specific pattern
        for i in range(na):
            if i % 2 == 0:
                A_mat[n - 1 - i, n - 1] = -A_padded[i]
            else:
                A_mat[n - 1 - i, n - 1] = A_padded[i]

        # B matrix: reversed B coefficients padded with zeros
        B_mat = cas.SX.zeros(n, 1)
        for i in range(nb):
            B_mat[nb - 1 - i] = B_padded[i]

        # C matrix: extracts output from last state with negation
        C_mat = cas.SX.zeros(1, n)
        C_mat[0, n - 1] = -1

        # D matrix: direct feedthrough (zero for ARX with nk >= 1)
        D_mat = cas.SX.zeros(1, 1)

        # State-space equations:
        # x(k+1) = A_mat * x(k) + B_mat * u(k)
        # y(k) = C_mat * x(k) + D_mat * u(k)
        xkp1 = cas.mtimes(A_mat, xk) + cas.mtimes(B_mat, uk)
        yk = cas.mtimes(C_mat, xk) + cas.mtimes(D_mat, uk)

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

        super().__init__(
            F,
            H,
            n,
            params=params,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )
