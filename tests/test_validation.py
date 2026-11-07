"""Unit tests for src/cas_models/validation.py module"""

import casadi as cas
import numpy as np
import pytest
from casadi import SX

from cas_models.validation import (
    as_sx_matrix,
    as_sx_scalar,
    as_sx_vector,
    validate_casadi_function_dims,
)


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


a = SX.sym("a")


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.5, [0.5]),
        (3, [3.0]),
        (np.float32(1.5), [1.5]),
        ([0.2, 0.1], [0.2, 0.1]),
        ((0.2, 0.1), [0.2, 0.1]),
        ([cas.DM(2.0), np.float64(1.5)], [2.0, 1.5]),
        (np.array([0.2, 0.1]), [0.2, 0.1]),
        (np.array([[0.2], [0.1]]), [0.2, 0.1]),
        (np.array([[0.2, 0.1]]), [0.2, 0.1]),
        (cas.DM([0.2, 0.1]), [0.2, 0.1]),
        (cas.DM([[0.2, 0.1]]), [0.2, 0.1]),  # a row, made a column
        ([], []),
        (np.zeros(0), []),
        (cas.DM(0, 1), []),
    ],
)
def test_as_sx_vector_numeric(value, expected):
    v = as_sx_vector(value)
    assert isinstance(v, SX) and v.shape == (len(expected), 1)
    assert v.is_constant()
    assert np.array_equal(np.array(cas.DM(v)).ravel(), expected)


@pytest.mark.parametrize(
    "value",
    [
        [1.0, a],
        (1.0, a),
        np.array([1.0, a], dtype=object),
        cas.vertcat(1.0, a),
        cas.horzcat(1.0, a),
    ],
)
def test_as_sx_vector_symbolic(value):
    """Symbols and numbers mixed"""
    v = as_sx_vector(value)
    assert v.shape == (2, 1) and not v.is_constant()
    assert str(v) == "[1, a]"


def test_as_sx_vector_sx_vector():
    b = SX.sym("b", 3)
    v = as_sx_vector(b)
    assert v.shape == (3, 1)
    assert [str(x) for x in cas.symvar(v)] == ["b_0", "b_1", "b_2"]


@pytest.mark.parametrize(
    "value, error, match",
    [
        (cas.MX.sym("m", 2), TypeError, "not MX"),
        ([1.0, cas.MX.sym("m")], TypeError, "not MX"),
        ([1.0, 2j], TypeError, "complex"),
        (np.array([1j]), TypeError, "complex"),
        ("abc", TypeError, "not str"),
        (None, TypeError, "not NoneType"),
        ([True], TypeError, "not bool"),
        (cas.DM.eye(2), ValueError, "a vector, not a 2 x 2 matrix"),
        (np.eye(2), ValueError, "a vector"),
        ([[1.0, 2.0]], ValueError, "nested list"),
        ([SX.sym("c", 2)], ValueError, "must be a scalar"),
    ],
)
def test_as_sx_vector_errors(value, error, match):
    with pytest.raises(error, match=match):
        as_sx_vector(value, "x")


def test_as_sx_scalar():
    assert as_sx_scalar(a) is not None and str(as_sx_scalar(a)) == "a"
    for value in (2.0, 2, cas.DM(2.0), [2.0], np.float64(2.0)):
        s = as_sx_scalar(value)
        assert s.shape == (1, 1) and float(cas.DM(s)) == 2.0
    with pytest.raises(ValueError, match="K: give a scalar, not 2 values"):
        as_sx_scalar([1.0, 2.0], "K")
    with pytest.raises(TypeError, match="not MX"):
        as_sx_scalar(cas.MX.sym("K"), "K")



@pytest.mark.parametrize(
    "value, expected",
    [
        (2.0, [[2.0]]),
        ([[-1.0, 0.0], [0.5, -2.0]], [[-1.0, 0.0], [0.5, -2.0]]),
        (((1, 2), (3, 4)), [[1.0, 2.0], [3.0, 4.0]]),
        ([[0.0, 1.0]], [[0.0, 1.0]]),
        ([1.0, 0.0], [[1.0], [0.0]]),  # a list of scalars: a column
        ([np.array([1.0, 2.0]), [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [3.0, 4.0]]),
        (np.array([1.0, 2.0]), [[1.0], [2.0]]),
        (cas.DM([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_as_sx_matrix_numeric(value, expected):
    M = as_sx_matrix(value)
    expected = np.array(expected)
    assert isinstance(M, SX) and M.shape == expected.shape
    assert M.is_constant()
    assert np.array_equal(np.array(cas.DM(M)), expected)


def test_as_sx_matrix_symbolic():
    a, b = SX.sym("a"), SX.sym("b")
    M = as_sx_matrix([[a, 0.0], [1.0, b]])
    assert M.shape == (2, 2) and str(M) == "\n[[a, 0], \n [1, b]]"
    assert as_sx_matrix(np.array([[a, 1.0]], dtype=object)).shape == (1, 2)
    P = SX.sym("P", 2, 3)
    assert as_sx_matrix(P).shape == (2, 3)


def test_as_sx_matrix_keeps_sparsity():
    M = as_sx_matrix(cas.sparsify(cas.DM([[1.0, 0.0], [0.0, 2.0]])))
    assert M.nnz() == 2


@pytest.mark.parametrize(
    "value, shape",
    [([], (0, 1)), (np.zeros((0, 3)), (0, 3)), (cas.DM(0, 2), (0, 2))],
)
def test_as_sx_matrix_empty(value, shape):
    assert as_sx_matrix(value).shape == shape


@pytest.mark.parametrize(
    "value, error, match",
    [
        (cas.MX.sym("m", 2, 2), TypeError, "not MX"),
        ([[1.0, cas.MX.sym("m")]], TypeError, "not MX"),
        ([[1.0, 2j]], TypeError, "complex"),
        (np.eye(2) * 1j, TypeError, "complex"),
        ("abc", TypeError, "not str"),
        ([[1.0, 2.0], [3.0]], ValueError, "different lengths"),
        ([[1.0], 2.0], ValueError, "rows or of scalars, not both"),
        ([[[1.0]]], ValueError, "nested list of rows"),
        (np.zeros((2, 2, 2)), ValueError, "a matrix, not an array"),
        ([[SX.sym("c", 2)]], ValueError, "must be a scalar"),
    ],
)
def test_as_sx_matrix_errors(value, error, match):
    with pytest.raises(error, match=match):
        as_sx_matrix(value, "M")
