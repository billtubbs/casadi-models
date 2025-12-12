import sympy
import casadi as cas


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

    Example:
    >>> var_names = {"x": (2, 1), "u": ()}
    >>> sympy_vars, casadi_vars = make_casadi_and_sympy_vars(var_names)
    >>> sympy_vars, casadi_vars
    ({'x': x, 'u': u}, {'x': SX([x_0, x_1]), 'u': SX(u)})
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


def make_casadi_vars_from_sympy_vars(sympy_vars):
    """Make CasADi SX variables from a dictionary of Sympy variables."""
    casadi_vars = []
    for v in sympy_vars:
        if isinstance(v, sympy.MatrixSymbol):
            casadi_vars.append(cas.SX.sym(v.name, v.shape[0], v.shape[1]))
        else:
            casadi_vars.append(cas.SX.sym(v.name))
    return casadi_vars


def convert_sympy_state_space_to_casadi_SX(
    sys, sympy_vars, casadi_vars, sparsify=True
):
    """Create CasADi SX matrices A, B, C, D from a Sympy StateSpace model."""
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


def get_free_symbols_in_sympy_ss(sys):
    """Get list of free symbols in a Sympy StateSpace model."""
    free_symbols = set()
    for item in [
        sys.state_matrix,
        sys.input_matrix,
        sys.output_matrix,
        sys.feedforward_matrix,
    ]:
        free_symbols.update(item.free_symbols)
    return list(free_symbols)
