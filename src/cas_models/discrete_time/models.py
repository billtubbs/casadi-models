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

          x(k+1) = F(t, x(k), u(K), *params.values())
            y(k) = H(t, x(k), u(K), *params.values())

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

          x(k+1) = F(t, x(k), u(K), *params.values())
            y(k) = H(t, x(k), u(K), *params.values())

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
