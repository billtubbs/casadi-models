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
        assert f.index_in(name) == i, msg
        assert f.size_in(name) == expected_shape, msg
    if not ignore_remaining:
        assert len(arg_shapes) == len(arg_names), msg
    for i, name in enumerate(return_names):
        assert f.index_out(name) == i, msg
        expected_shape = return_shapes[name]
        assert f.size_out(name) == expected_shape, msg
