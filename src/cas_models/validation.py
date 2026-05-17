import casadi as cas


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

    Args:
        sys: A system object to check

    Returns:
        bool: True if the system has continuous-time attributes (f, h),
              False otherwise
    """
    return hasattr(sys, "f") and hasattr(sys, "h")


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
    """Check all discrete-time systems have the same time interval."""
    dt_values = [sys.dt for sys in systems]
    if not all(dt == dt_values[0] for dt in dt_values):
        raise ValueError(
            f"All discrete-time systems must have the same dt. "
            f"Found dt values: {dt_values}"
        )
