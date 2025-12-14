"""Unit tests for src/cas_models/validation.py module"""

import pytest
import casadi as cas
from cas_models.validation import validate_casadi_function_dims


def test_validate_casadi_function_dims():
    args = {
        "t": cas.SX.sym("t"),
        "x": cas.SX.sym("x", 2),
        "u": cas.SX.sym("u", 1),
    }
    arg_shapes = {name: var.shape for name, var in args.items()}
    rhs = args["x"]
    return_vars = {"rhs": rhs}
    return_shapes = {name: var.shape for name, var in return_vars.items()}
    f = cas.Function(
        "f",
        args.values(),
        return_vars.values(),
        args.keys(),
        return_vars.keys(),
    )

    validate_casadi_function_dims(f, arg_shapes, return_shapes)

    wrong_arg_shapes = arg_shapes.copy()
    wrong_arg_shapes["t"] = (2, 1)
    with pytest.raises(AssertionError):
        validate_casadi_function_dims(f, wrong_arg_shapes, return_shapes)

    wrong_return_shapes = return_shapes.copy()
    wrong_return_shapes["rhs"] = (1, 2)
    with pytest.raises(AssertionError):
        validate_casadi_function_dims(f, arg_shapes, wrong_return_shapes)

    with pytest.raises(KeyError):
        validate_casadi_function_dims(f, arg_shapes, {})

    with pytest.raises(RuntimeError):
        validate_casadi_function_dims(f, return_shapes, arg_shapes)
