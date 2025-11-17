"""Transformations for combining state-space models.

This module provides generalized functions for combining both continuous-time
and discrete-time state-space models.
"""

import casadi as cas
from cas_models.param_utils import (
    concatenate_lists_of_names,
    merge_param_dicts,
)


def connect_nonlinear_systems_in_parallel(
    systems, attr_names, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a collection of nonlinear systems into one large parallel system.

    This function works for both continuous-time and discrete-time systems by
    using an attr_names dictionary to specify the appropriate attribute and variable
    names.

    Args:
        systems (list): List of state-space model objects to combine in parallel.
        attr_names (dict): Dictionary specifying naming conventions with keys:
            - 'state_func': Attribute name for state function ('f' or 'F')
            - 'output_func': Attribute name for output function ('h' or 'H')
            - 'state_var': Variable name for state ('x' or 'xk')
            - 'input_var': Variable name for input ('u' or 'uk')
            - 'state_output': Output name for state function ('rhs' or 'xkp1')
            - 'output_var': Output name for output function ('y' or 'yk')
            - 'model_class': Class to instantiate (StateSpaceModelCT or StateSpaceModelDT)
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to [prefix + "1", prefix + "2", ...].
        verbose_names (bool, optional): If True, use verbose parameter naming.
            Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: Combined parallel system.

    Example:
        >>> # For continuous-time models:
        >>> attr_names_ct = {
        ...     'state_func': 'f',
        ...     'output_func': 'h',
        ...     'state_var': 'x',
        ...     'input_var': 'u',
        ...     'state_output': 'rhs',
        ...     'output_var': 'y',
        ...     'model_class': StateSpaceModelCT
        ... }
        >>> combined = connect_nonlinear_systems_in_parallel(
        ...     [sys1, sys2], attr_names_ct
        ... )
        >>>
        >>> # For discrete-time models:
        >>> attr_names_dt = {
        ...     'state_func': 'F',
        ...     'output_func': 'H',
        ...     'state_var': 'xk',
        ...     'input_var': 'uk',
        ...     'state_output': 'xkp1',
        ...     'output_var': 'yk',
        ...     'model_class': StateSpaceModelDT
        ... }
        >>> combined = connect_nonlinear_systems_in_parallel(
        ...     [sys1, sys2], attr_names_dt
        ... )
    """
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
    )

    t = cas.SX.sym("t")

    u_signals = []
    x_states = []
    state_outputs = []
    y_signals = []

    for sys in systems:
        x = cas.SX.sym(attr_names['state_var'], sys.n)
        x_states.append(x)

        u = cas.SX.sym(attr_names['input_var'], sys.nu)
        u_signals.append(u)

        # Get state function and call it: f(t, x, u, *params.values())
        state_func = getattr(sys, attr_names['state_func'])
        state_out = state_func(t, x, u, *sys.params.values())
        state_outputs.append(state_out)

        # Get output function and call it: h(t, x, u, *params.values())
        output_func = getattr(sys, attr_names['output_func'])
        y = output_func(t, x, u, *sys.params.values())
        y_signals.append(y)

    x = cas.vcat(x_states)
    n = x.shape[0]

    u = cas.vcat(u_signals)
    nu = u.shape[0]

    state_combined = cas.vcat(state_outputs)
    assert state_combined.shape == x.shape

    state_function = cas.Function(
        attr_names['state_func'],
        [t, x, u, *params.values()],
        [state_combined],
        ["t", attr_names['state_var'], attr_names['input_var'], *params.keys()],
        [attr_names['state_output']],
    )

    y = cas.vcat(y_signals)
    ny = y.shape[0]

    output_function = cas.Function(
        attr_names['output_func'],
        [t, x, u, *params.values()],
        [y],
        ["t", attr_names['state_var'], attr_names['input_var'], *params.keys()],
        [attr_names['output_var']],
    )

    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems],
        keys=keys,
        prefix=prefix,
    )
    assert len(state_names) == n

    input_names = concatenate_lists_of_names(
        [sys.input_names for sys in systems], keys=keys, prefix=prefix
    )
    assert len(input_names) == nu

    output_names = concatenate_lists_of_names(
        [sys.output_names for sys in systems], keys=keys, prefix=prefix
    )
    assert len(output_names) == ny

    combined_system = attr_names['model_class'](
        state_function,
        output_function,
        n,
        nu,
        ny,
        params=params,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
    )

    return combined_system


def connect_nonlinear_systems_in_series(
    systems, attr_names, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a series of non-linear systems by connecting their inputs and
    outputs in series.

    This function works for both continuous-time and discrete-time systems by
    using an attr_names dictionary to specify the appropriate attribute and variable
    names.

    Args:
        systems (list): List of state-space model objects to combine in series.
            The output of each system is connected to the input of the next.
        attr_names (dict): Dictionary specifying naming conventions with keys:
            - 'state_func': Attribute name for state function ('f' or 'F')
            - 'output_func': Attribute name for output function ('h' or 'H')
            - 'state_var': Variable name for state ('x' or 'xk')
            - 'input_var': Variable name for input ('u' or 'uk')
            - 'state_output': Output name for state function ('rhs' or 'xkp1')
            - 'output_var': Output name for output function ('y' or 'yk')
            - 'model_class': Class to instantiate (StateSpaceModelCT or StateSpaceModelDT)
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to [prefix + "1", prefix + "2", ...].
        verbose_names (bool, optional): If True, use verbose parameter naming.
            Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: Combined series system.

    Example:
        >>> # For continuous-time models:
        >>> attr_names_ct = {
        ...     'state_func': 'f',
        ...     'output_func': 'h',
        ...     'state_var': 'x',
        ...     'input_var': 'u',
        ...     'state_output': 'rhs',
        ...     'output_var': 'y',
        ...     'model_class': StateSpaceModelCT
        ... }
        >>> combined = connect_nonlinear_systems_in_series(
        ...     [sys1, sys2], attr_names_ct
        ... )
    """
    if keys is None:
        keys = [f"{prefix}{i + 1}" for i in range(len(systems))]

    param_dicts = [sys.params for sys in systems]
    state_name_lists = [sys.state_names for sys in systems]

    t = cas.SX.sym("t")
    combined_system = systems[0]
    for i, sys2 in enumerate(systems[1:], start=1):
        sys1 = combined_system
        assert sys2.nu == sys1.ny, "incompatible dimensions"

        # System 1
        n1 = sys1.n
        nu1 = sys1.nu
        u1 = cas.SX.sym(attr_names['input_var'], nu1)
        x1 = cas.SX.sym(attr_names['state_var'], n1)
        state_func1 = getattr(sys1, attr_names['state_func'])
        params1 = merge_param_dicts(
            param_dicts[:i], keys=keys[:i], verbose_names=verbose_names
        )
        state_out1 = state_func1(t, x1, u1, *params1.values())
        output_func1 = getattr(sys1, attr_names['output_func'])
        y1 = output_func1(t, x1, u1, *params1.values())

        # System 2
        n2 = sys2.n
        ny2 = sys2.ny
        u2 = y1
        x2 = cas.SX.sym(attr_names['state_var'], n2)
        state_func2 = getattr(sys2, attr_names['state_func'])
        params2 = sys2.params
        state_out2 = state_func2(t, x2, u2, *params2.values())
        output_func2 = getattr(sys2, attr_names['output_func'])
        y2 = output_func2(t, x2, u2, *params2.values())

        # Variables of combined system
        x = cas.vcat([x2, x1])  # stack with sys2 states at top
        u = u1
        nu = nu1

        # Combined state function
        state_combined = cas.vcat([state_out2, state_out1])
        n = n1 + n2
        assert state_combined.shape[0] == n
        params = merge_param_dicts(
            param_dicts[: i + 1],
            keys=keys[: i + 1],
            verbose_names=verbose_names,
        )
        state_function = cas.Function(
            attr_names['state_func'],
            [t, x, u, *params.values()],
            [state_combined],
            ["t", attr_names['state_var'], attr_names['input_var'], *params.keys()],
            [attr_names['state_output']],
        )

        # Combined output function
        y = y2
        ny = ny2
        assert y.shape[0] == ny
        output_function = cas.Function(
            attr_names['output_func'],
            [t, x, u, *params.values()],
            [y],
            ["t", attr_names['state_var'], attr_names['input_var'], *params.keys()],
            [attr_names['output_var']],
        )
        combined_system = attr_names['model_class'](
            state_function, output_function, n, nu, ny, params=params
        )

    combined_system.state_names = concatenate_lists_of_names(
        list(reversed(state_name_lists)), keys=list(reversed(keys))
    )

    return combined_system
