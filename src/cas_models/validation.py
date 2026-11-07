from numbers import Real

import casadi as cas
import numpy as np


def validate_casadi_function_dims(
    f: cas.Function,
    arg_shapes: dict = None,
    return_shapes: dict = None,
    ignore_remaining=False,
):
    """Use this to check a CasADi function has the expected
    arguments and return variable dimensions.
    """
    arg_names = f.name_in()
    return_names = f.name_out()
    msg = "function validation error"
    for i, (name, expected_shape) in enumerate(arg_shapes.items()):
        assert f.index_in(name) == i, f"{msg} {name}"
        assert f.size_in(name) == expected_shape, f"{msg} {name}"
    if not ignore_remaining:
        assert len(arg_shapes) == len(arg_names), f"{msg}, too many args"
    for i, name in enumerate(return_names):
        assert f.index_out(name) == i, f"{msg} {name}"
        expected_shape = return_shapes[name]
        assert f.size_out(name) == expected_shape, f"{msg} {name}"


def is_ss_ct(sys):
    """Check if a system is a continuous-time state-space model.

    A continuous-time model is identified by having 'f' and 'h'
    attributes (lowercase), which are the right-hand-side of the
    differential equation and output function respectively.
    """
    return hasattr(sys, "f") and hasattr(sys, "h")


def is_ss_dt(sys):
    """Check if a system is a discrete-time state-space model.

    A discrete-time model is identified by having 'F' and 'H'
    attributes (uppercase), which are the state transition function
    and output function respectively.
    """
    return hasattr(sys, "F") and hasattr(sys, "H")


def validate_equal_dt(systems):
    """Check all discrete-time systems have the same time interval."""
    dt_values = [sys.dt for sys in systems]
    if not all(dt == dt_values[0] for dt in dt_values):
        raise ValueError(
            f"All discrete-time systems must have the same dt. "
            f"Found dt values: {dt_values}"
        )


def as_sx_vector(value, name="value"):
    """A model argument that is a vector of parameters, as an SX column
    vector.

    The elements can be numbers, symbols or expressions of them, mixed:
    e.g. [1.0, cas.SX.sym("a")]. value can be:
    - a real number (a vector of one element),
    - a list or tuple of real numbers and CasADi scalars (SX, or 1 x 1
      DM),
    - a NumPy array of one dimension, or of two with one of them 1 (an
      object array of SX scalars too),
    - a CasADi SX or DM vector, a column or a row (made a column).
    An empty one gives a 0 x 1 vector.

    Raises:
        TypeError: for MX (models' functions are built from SX symbols,
            so an MX argument can't be used), complex values and other
            types (strings, None, ...).
        ValueError: if value isn't a vector (a matrix, a nested list) or
            an element isn't a scalar.
    """
    if isinstance(value, cas.MX):
        raise TypeError(f"{name}: give SX symbols or numbers, not MX")
    if isinstance(value, (cas.SX, cas.DM)):
        if value.numel() > 0 and min(value.shape) != 1:
            raise ValueError(
                f"{name}: give a vector, not a {value.shape[0]} x "
                f"{value.shape[1]} matrix"
            )
        return cas.reshape(cas.SX(value), value.numel(), 1)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            raise TypeError(f"{name}: complex values aren't allowed")
        if value.ndim > 2 or (value.ndim == 2 and min(value.shape) > 1):
            raise ValueError(
                f"{name}: give a vector, not an array of shape {value.shape}"
            )
        value = list(value.ravel())
    elif isinstance(value, (list, tuple)):
        value = list(value)
    else:
        value = [value]
    return cas.vertcat(
        cas.SX(0, 1), *[_as_sx_element(v, name) for v in value]
    )


def as_sx_scalar(value, name="value"):
    """A model argument that is one parameter, as a 1 x 1 SX: a real
    number, an SX scalar (a symbol or expression) or a 1 x 1 DM (or a
    vector of one element, see as_sx_vector()).

    Raises:
        TypeError, ValueError: as as_sx_vector(), and ValueError if
            value has more or fewer than one element.
    """
    v = as_sx_vector(value, name)
    if v.numel() != 1:
        raise ValueError(f"{name}: give a scalar, not {v.numel()} values")
    return v


def _as_sx_element(v, name):
    """One element of a vector argument as a 1 x 1 SX."""
    if isinstance(v, cas.MX):
        raise TypeError(f"{name}: give SX symbols or numbers, not MX")
    if isinstance(v, (cas.SX, cas.DM)):
        if v.numel() != 1:
            raise ValueError(
                f"{name}: each element must be a scalar, not "
                f"{v.shape[0]} x {v.shape[1]}"
            )
        return cas.SX(v)
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.number)):
        if isinstance(v, (complex, np.complexfloating)):
            raise TypeError(f"{name}: complex values aren't allowed ({v})")
        if isinstance(v, (list, tuple, np.ndarray)):
            raise ValueError(f"{name}: give a vector, not a nested list")
        raise TypeError(
            f"{name}: elements must be real numbers or CasADi SX scalars, "
            f"not {type(v).__name__} ({v!r})"
        )
    return cas.SX(float(v))


def as_sx_matrix(value, name="value"):
    """A model argument that is a matrix of parameters, as an SX matrix
    of the same shape.

    The elements can be numbers, symbols or expressions of them, mixed.
    value can be:
    - a real number (1 x 1),
    - a list or tuple of rows (lists or tuples of the same length), or
      of scalars (a column vector, as cas.DM([1, 2]) is),
    - a NumPy array of up to two dimensions (one dimension: a column;
      an object array of SX scalars too),
    - a CasADi SX or DM matrix (its sparsity kept).
    An empty list gives a 0 x 1 matrix.

    Raises:
        TypeError: for MX, complex values and other types, as
            as_sx_vector().
        ValueError: for rows of different lengths, more than two
            dimensions or an element that isn't a scalar.
    """
    if isinstance(value, cas.MX):
        raise TypeError(f"{name}: give SX symbols or numbers, not MX")
    if isinstance(value, (cas.SX, cas.DM)):
        return cas.SX(value)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            raise TypeError(f"{name}: complex values aren't allowed")
        if value.ndim > 2:
            raise ValueError(
                f"{name}: give a matrix, not an array of shape {value.shape}"
            )
        if value.ndim < 2:
            value = value.reshape(-1, 1)
        rows = [list(row) for row in value]
        ncol = value.shape[1]
    elif isinstance(value, (list, tuple)):
        if any(isinstance(v, (list, tuple, np.ndarray)) for v in value):
            rows = [_as_row(v, name) for v in value]
            ncol = len(rows[0])
            if any(len(row) != ncol for row in rows):
                raise ValueError(f"{name}: rows of different lengths")
        else:
            rows = [[v] for v in value]  # a column
            ncol = 1
    else:
        rows = [[value]]
        ncol = 1
    if not rows:
        return cas.SX(0, ncol)
    return cas.vertcat(
        *[
            cas.horzcat(cas.SX(1, 0), *[_as_sx_element(v, name) for v in row])
            for row in rows
        ]
    )


def _as_row(row, name):
    """A row of a matrix argument given as a list of rows."""
    if not isinstance(row, (list, tuple, np.ndarray)):
        raise ValueError(f"{name}: give a list of rows or of scalars, not both")
    row = list(np.ravel(row)) if isinstance(row, np.ndarray) else list(row)
    if any(isinstance(v, (list, tuple, np.ndarray)) for v in row):
        raise ValueError(f"{name}: give a matrix, not a nested list of rows")
    return row
