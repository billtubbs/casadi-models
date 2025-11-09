import casadi as cas
import sympy
from collections import OrderedDict
from dataclasses import dataclass
from cas_models.param_utils import (
    make_list_of_enumerated_names,
)
from cas_models.validation import validate_casadi_function_dims


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

    F: cas.Function
    H: cas.Function
    n: int
    nu: int
    ny: int
    params: dict
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
        params=None,
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
            >>> F = cas.Function("F", [t, xk, uk, a], [xkp1],
            ...     ["t", "xk", "uk", "a"], ["xkp1"])
            >>> H = cas.Function("H", [t, xk, uk, a], [xk[0]],
            ...     ["t", "xk", "uk", "a"], ["yk"])
            >>> model = StateSpaceModelDT(F, H, n=2, nu=1, ny=1,
            ...     params={'a': a})
        """
        self.F = F
        self.H = H
        self.n = n
        self.nu = nu
        self.ny = ny
        if params is None:
            params = {}
        self.params = params
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

        This is internally represented as a state-space model with state vector:
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
            The total number of states is n = na + nb + nk.

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
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )
