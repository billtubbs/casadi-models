"""Transformations for combining state-space models.

Provides functions for combining continuous-time and discrete-time
state-space models in series, parallel and feedback, and stacked
horizontally, vertically or block-diagonally (as transfer matrices).

Functions
---------
connect_systems
    Connect models with arbitrary named signal connections.
connect_systems_in_series
    Connect models end-to-end, feeding each output into the next input
    (sysN * ... * sys2 * sys1).
connect_systems_in_parallel
    Combine models with shared inputs and summed outputs (sys1 + sys2).
connect_feedback_system
    Close a feedback loop around a forward-path model.
block_diag_systems
    Combine models side by side, unconnected: inputs and outputs stacked
    (transfer matrix [G1 0; 0 G2]).
hstack_systems
    Combine models with separate inputs and summed outputs ([G1, G2]).
vstack_systems
    Combine models with shared inputs and stacked outputs ([G1; G2]).
negate_system
    A model with its outputs negated (-sys; sys1 - sys2 is sys1 + -sys2).
set_param_values
    A model with values in place of some of its parameters.
validate_systems_are_compatible
    Raise if a list of models are not all the same type (CT vs DT).
infer_model_class
    The base class (StateSpaceModelCT or StateSpaceModelDT) of a list of
    models: model_class when it isn't given.
block_diag
    Construct a block-diagonal CasADi SX matrix from a list of matrices.

Before 2026-11, connect_systems_in_parallel was the block-diagonal
combination (now block_diag_systems) and sum_systems the parallel one
(now connect_systems_in_parallel).
"""

from functools import reduce

import casadi as cas

from cas_models.param_utils import (
    concatenate_lists_of_names,
    make_list_of_unique_names,
    merge_param_dicts,
)
from cas_models.recipes import recipe, records
from cas_models.validation import validate_equal_dt


def block_diag(matrices, square=False):
    col_sizes = [m.shape[1] for m in matrices]
    rows = []
    for i, matrix in enumerate(matrices):
        row = []
        n_rows = matrix.shape[0]
        if square:
            assert n_rows == col_sizes[i], "matrices not square"
        for j in range(i):
            row.append(cas.SX.zeros(n_rows, col_sizes[j]))
        row.append(matrix)
        for j in range(i + 1, len(col_sizes)):
            row.append(cas.SX.zeros(n_rows, col_sizes[j]))
        rows.append(row)
    return cas.blockcat(rows)


def _dt_kwargs(systems):
    """{"dt": dt} of the systems' common time step (checked equal by
    validate_systems_are_compatible), or {} if they have none (e.g.
    continuous-time systems, whose classes take no dt)."""
    dt = getattr(systems[0], "dt", None)
    return {} if dt is None else {"dt": dt}


def infer_model_class(systems):
    """The base class, StateSpaceModelCT or StateSpaceModelDT, of the
    systems (a list of models): the model_class of the connection
    functions when it isn't given.

    Raises:
        ValueError: If there are no systems, one isn't a state-space
            model, or they aren't all continuous time or all discrete
            time.
    """
    # Imported here: the model modules import this one
    from cas_models.continuous_time.models import StateSpaceModelCT
    from cas_models.discrete_time.models import StateSpaceModelDT

    if not systems:
        raise ValueError("no systems to take model_class from")
    found = set()
    for i, sys in enumerate(systems):
        for base in (StateSpaceModelCT, StateSpaceModelDT):
            if isinstance(sys, base):
                found.add(base)
                break
        else:
            raise ValueError(
                f"system {i} isn't a StateSpaceModelCT or StateSpaceModelDT "
                f"({type(sys).__name__})"
            )
    if len(found) > 1:
        raise ValueError(
            "Cannot combine continuous-time and discrete-time systems "
            "(give them all one kind, or model_class)"
        )
    return found.pop()


def _check_equal_dims(systems, dims, func_name):
    """Raise ValueError unless the systems all have the same values of
    the attributes dims (e.g. ["nu", "ny"])."""
    for dim in dims:
        expected = getattr(systems[0], dim)
        for sys in systems[1:]:
            if getattr(sys, dim) != expected:
                raise ValueError(
                    f"All systems must have the same {dim} for {func_name}. "
                    f"Got {dim}={getattr(sys, dim)}, expected "
                    f"{dim}={expected}."
                )


def _summed_names(name_lists):
    """The names of summed outputs, one per output, from each system's
    output names: the names if all the systems have the same, else
    sum_<name1>_<name2>... of the systems' names for that output (e.g.
    sum_y1_y2), so that the sum isn't named after one of its terms."""
    if all(names == name_lists[0] for names in name_lists):
        return list(name_lists[0])
    return ["sum_" + "_".join(names) for names in zip(*name_lists)]


def _combine_systems(
    systems,
    model_class,
    shared_inputs,
    summed_outputs,
    keys,
    verbose_names,
    prefix,
    name,
    sep,
):
    """The core of block_diag_systems, hstack_systems, vstack_systems
    and connect_systems_in_parallel: the systems' states stacked, their
    inputs shared (one input vector for all) or stacked, and their
    outputs summed or stacked. Stacked signals keep their systems'
    names (concatenated); shared inputs have the first system's names;
    summed outputs have the systems' names if they are all the same,
    else sum_<name1>_<name2>... (see _summed_names())."""
    validate_systems_are_compatible(systems)
    if model_class is None:
        model_class = infer_model_class(systems)
    attr_names = model_class._attr_names
    if keys is None:
        keys = [sys.name for sys in systems]
    keys = make_list_of_unique_names(keys, prefix=prefix)
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys,
        verbose_names=verbose_names,
        sep=sep,
    )

    t = cas.SX.sym("t")
    if shared_inputs:
        u = cas.SX.sym(attr_names["input_var"], systems[0].nu)

    u_signals = []
    x_states = []
    state_outputs = []
    y_signals = []

    for sys in systems:
        x = cas.SX.sym(attr_names["state_var"], sys.n)
        x_states.append(x)

        if not shared_inputs:
            u = cas.SX.sym(attr_names["input_var"], sys.nu)
            u_signals.append(u)

        # Get state function and call it: f(t, x, u, *params.values())
        state_func = getattr(sys, attr_names["state_func"])
        state_outputs.append(state_func(t, x, u, *sys.params.values()))

        # Get output function and call it: h(t, x, u, *params.values())
        output_func = getattr(sys, attr_names["output_func"])
        y_signals.append(output_func(t, x, u, *sys.params.values()))

    x = cas.vcat(x_states)
    n = x.shape[0]
    if not shared_inputs:
        u = cas.vcat(u_signals)
    nu = u.shape[0]
    state_combined = cas.vcat(state_outputs)
    assert state_combined.shape == x.shape

    if summed_outputs:
        y = y_signals[0]
        for y_i in y_signals[1:]:
            y = y + y_i
    else:
        y = cas.vcat(y_signals)
    ny = y.shape[0]

    arg_names = [
        "t",
        attr_names["state_var"],
        attr_names["input_var"],
        *params.keys(),
    ]
    state_function = cas.Function(
        attr_names["state_func"],
        [t, x, u, *params.values()],
        [state_combined],
        arg_names,
        [attr_names["state_output"]],
    )
    output_function = cas.Function(
        attr_names["output_func"],
        [t, x, u, *params.values()],
        [y],
        arg_names,
        [attr_names["output_var"]],
    )

    def names(attr):
        return concatenate_lists_of_names(
            [getattr(sys, attr) for sys in systems],
            keys=keys,
            prefix=prefix,
            verbose_names=verbose_names,
        )

    state_names = names("state_names")
    assert len(state_names) == n
    input_names = systems[0].input_names if shared_inputs else names(
        "input_names"
    )
    assert len(input_names) == nu
    output_names = (
        _summed_names([sys.output_names for sys in systems])
        if summed_outputs
        else names("output_names")
    )
    assert len(output_names) == ny

    # Generate combined system name if not provided
    if name is None:
        system_names = make_list_of_unique_names(
            [sys.name for sys in systems], prefix=prefix
        )
        name = sep.join(system_names)

    return model_class(
        state_function,
        output_function,
        n,
        nu,
        ny,
        **_dt_kwargs(systems),
        params=params,
        name=name,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
    )


@records
def block_diag_systems(
    systems,
    model_class=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine systems side by side, unconnected: their inputs and
    outputs stacked, each system's outputs depending only on its own
    inputs. As transfer matrices, the block-diagonal matrix

        G = [G1  0 ]    u = [u1]    y = [y1]
            [ 0  G2]        [u2]        [y2]

    (Matlab append(), python-control append()). Before 2026-11, this
    was connect_systems_in_parallel, which now sums the outputs.

    This function works for both continuous-time and discrete-time
    systems. The attribute names are determined from the model_class.

    Args:
        systems (list): List of state-space model objects to combine.
        model_class (optional): Class to instantiate for the combined
            system (StateSpaceModelCT or StateSpaceModelDT). If None, the
            systems' base class (see infer_model_class()). The
            _attr_names class attribute determines the naming conventions.
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to the system names or [prefix + "1", prefix + "2",
            ...].
        verbose_names (bool, optional): If True, use verbose parameter
            naming. Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".
        name (str, optional): Name for the combined system. If None,
            auto-generates from the system names using sep as separator.
        sep (str, optional): Separator for joining system names when
            auto-generating the combined system name (and for the
            parameter names merge_param_dicts() makes unique).

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: The combined system, with
            nu and ny the sums of the systems' (default sep "_").

    Example:
        >>> combined = block_diag_systems([sys1, sys2], StateSpaceModelCT)
    """
    return _combine_systems(
        systems,
        model_class,
        shared_inputs=False,
        summed_outputs=False,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )


@records
def hstack_systems(
    systems,
    model_class=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine systems with separate inputs and summed outputs:
    y = y1 + y2 + ..., each yi from its own inputs ui. As transfer
    matrices, the horizontal concatenation

        G = [G1  G2]    u = [u1]    y = G1 u1 + G2 u2
                            [u2]

    (Matlab [G1, G2]). The output names are the systems' if they are
    all the same, else sum_<name1>_<name2>... (e.g. sum_y1_y2).

    This function works for both continuous-time and discrete-time
    systems. The attribute names are determined from the model_class.

    Args:
        systems (list): List of state-space model objects to combine.
        model_class (optional): Class to instantiate for the combined
            system (StateSpaceModelCT or StateSpaceModelDT). If None, the
            systems' base class (see infer_model_class()). The
            _attr_names class attribute determines the naming conventions.
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to the system names or [prefix + "1", prefix + "2",
            ...].
        verbose_names (bool, optional): If True, use verbose parameter
            naming. Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".
        name (str, optional): Name for the combined system. If None,
            auto-generates from the system names using sep as separator.
        sep (str, optional): Separator for joining system names when
            auto-generating the combined system name (and for the
            parameter names merge_param_dicts() makes unique).

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: The combined system, with
            nu the sum of the systems' and their common ny.

    Raises:
        ValueError: If the systems do not all have the same ny.

    Example:
        >>> # One output, the sum of a path model per input
        >>> combined = hstack_systems([path1, path2], StateSpaceModelDT)
    """
    _check_equal_dims(systems, ["ny"], "hstack_systems")
    return _combine_systems(
        systems,
        model_class,
        shared_inputs=False,
        summed_outputs=True,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )


@records
def vstack_systems(
    systems,
    model_class=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine systems with shared inputs and stacked outputs: every
    system gets the same inputs u. As transfer matrices, the vertical
    concatenation

        G = [G1]    y = [y1] = [G1 u]
            [G2]        [y2]   [G2 u]

    (Matlab [G1; G2]). The input names are the first system's.

    This function works for both continuous-time and discrete-time
    systems. The attribute names are determined from the model_class.

    Args:
        systems (list): List of state-space model objects to combine.
        model_class (optional): Class to instantiate for the combined
            system (StateSpaceModelCT or StateSpaceModelDT). If None, the
            systems' base class (see infer_model_class()). The
            _attr_names class attribute determines the naming conventions.
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to the system names or [prefix + "1", prefix + "2",
            ...].
        verbose_names (bool, optional): If True, use verbose parameter
            naming. Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".
        name (str, optional): Name for the combined system. If None,
            auto-generates from the system names using sep as separator.
        sep (str, optional): Separator for joining system names when
            auto-generating the combined system name (and for the
            parameter names merge_param_dicts() makes unique).

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: The combined system, with
            their common nu and ny the sum of the systems'.

    Raises:
        ValueError: If the systems do not all have the same nu.

    Example:
        >>> # The outputs of several models of the same inputs
        >>> combined = vstack_systems([model1, model2], StateSpaceModelDT)
    """
    _check_equal_dims(systems, ["nu"], "vstack_systems")
    return _combine_systems(
        systems,
        model_class,
        shared_inputs=True,
        summed_outputs=False,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )


@records
def connect_systems_in_parallel(
    systems,
    model_class=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="+",
):
    """Combine systems with shared inputs and summed outputs: the
    classical parallel connection, y = G1 u + G2 u + ... (Matlab
    parallel(), python-control parallel(), sys1 + sys2). The combined
    system has the same input and output dimensions as each, the first
    system's input names, and their output names if they are all the
    same, else sum_<name1>_<name2>... (e.g. sum_y1_y2).

    Before 2026-11, this was sum_systems, and connect_systems_in_parallel
    was what is now block_diag_systems.

    This function works for both continuous-time and discrete-time
    systems. The attribute names are determined from the model_class.

    Args:
        systems (list): List of state-space model objects to combine.
        model_class (optional): Class to instantiate for the combined
            system (StateSpaceModelCT or StateSpaceModelDT). If None, the
            systems' base class (see infer_model_class()). The
            _attr_names class attribute determines the naming conventions.
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to the system names or [prefix + "1", prefix + "2",
            ...].
        verbose_names (bool, optional): If True, use verbose parameter
            naming. Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".
        name (str, optional): Name for the combined system. If None,
            auto-generates from the system names using sep as separator.
        sep (str, optional): Separator for joining system names when
            auto-generating the combined system name (and for the
            parameter names merge_param_dicts() makes unique).

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: The combined system (the
            default sep is "+").

    Raises:
        ValueError: If the systems do not all have the same nu and ny.

    Example:
        >>> combined = connect_systems_in_parallel(
        ...     [sys1, sys2], StateSpaceModelCT
        ... )
        >>> # Equivalent to sys1 + sys2 via the __add__ operator.
    """
    _check_equal_dims(systems, ["nu", "ny"], "connect_systems_in_parallel")
    return _combine_systems(
        systems,
        model_class,
        shared_inputs=True,
        summed_outputs=True,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )


@records
def negate_system(sys, model_class=None):
    """The system with its outputs negated, y = -h(t, x, u) (the -sys
    operator; sys1 - sys2 is sys1 + (-sys2)). The states, parameters
    and names are the same; the name, if any, gets a leading "-".
    model_class is sys's base class unless given."""
    if model_class is None:
        model_class = infer_model_class([sys])
    attr_names = model_class._attr_names
    output_func = getattr(sys, attr_names["output_func"])
    t = cas.SX.sym("t")
    x = cas.SX.sym(attr_names["state_var"], sys.n)
    u = cas.SX.sym(attr_names["input_var"], sys.nu)
    params = sys.params
    arg_names = [
        "t",
        attr_names["state_var"],
        attr_names["input_var"],
        *params.keys(),
    ]
    output_function = cas.Function(
        attr_names["output_func"],
        [t, x, u, *params.values()],
        [-output_func(t, x, u, *params.values())],
        arg_names,
        [attr_names["output_var"]],
    )
    return model_class(
        getattr(sys, attr_names["state_func"]),
        output_function,
        sys.n,
        sys.nu,
        sys.ny,
        **_dt_kwargs([sys]),
        params=params,
        name=None if sys.name is None else f"-{sys.name}",
        input_names=sys.input_names,
        state_names=sys.state_names,
        output_names=sys.output_names,
    )


def set_param_values(sys, values, model_class=None):
    """sys with values ({name: value} of sys.params, e.g. fitted values)
    in place of those parameters: a model of sys's base class (or
    model_class), its functions taking only the other parameters (the
    same symbols, so sharing with other models is kept). The states,
    inputs, outputs and names are the same.

    Its recipe (cas_models.recipes) is sys's with the values added, so
    it is saved as sys's YAML with values: beside the arguments, and
    loads as sys built again with these values set. A sys without a
    recipe gives one without (saved from its fields, the functions with
    the values in them).
    """
    unknown = [name for name in values if name not in sys.params]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)}: not parameters of the model "
            f"({', '.join(sys.params) or 'none'})"
        )
    if model_class is None:
        model_class = infer_model_class([sys])
    attr_names = model_class._attr_names
    fixed = {}
    for name, value in values.items():
        value = cas.DM(value)
        shape = sys.params[name].shape
        if value.shape != shape:
            raise ValueError(
                f"{name} is {shape[0]} x {shape[1]}, the value "
                f"{value.shape[0]} x {value.shape[1]}"
            )
        fixed[name] = value
    params = {k: p for k, p in sys.params.items() if k not in fixed}
    args = [fixed.get(k, p) for k, p in sys.params.items()]
    t = cas.SX.sym("t")
    x = cas.SX.sym(attr_names["state_var"], sys.n)
    u = cas.SX.sym(attr_names["input_var"], sys.nu)
    arg_names = [
        "t",
        attr_names["state_var"],
        attr_names["input_var"],
        *params.keys(),
    ]
    functions = {}
    for func, out in (("state_func", "state_output"), ("output_func", "output_var")):
        f = getattr(sys, attr_names[func])
        functions[func] = cas.Function(
            attr_names[func],
            [t, x, u, *params.values()],
            [f(t, x, u, *args)],
            arg_names,
            [attr_names[out]],
        )
    model = model_class(
        functions["state_func"],
        functions["output_func"],
        sys.n,
        sys.nu,
        sys.ny,
        **_dt_kwargs([sys]),
        params=params,
        name=sys.name,
        input_names=sys.input_names,
        state_names=sys.state_names,
        output_names=sys.output_names,
    )
    r = recipe(sys)
    if r is not None:
        given = {k: float(v) if v.is_scalar() else v for k, v in fixed.items()}
        model.__dict__["_recipe"] = {
            **r,
            "values": {**r.get("values", {}), **given},
        }
    return model


@records
def connect_systems_in_series(
    systems,
    model_class=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine a series of non-linear systems by connecting their inputs and
    outputs in series.

    This function works for both continuous-time and discrete-time systems.
    The attribute names are automatically determined from the model_class.

    Args:
        systems (list): List of state-space model objects to combine in series.
            The output of each system is connected to the input of the next.
        model_class (optional): Class to instantiate for the combined
            system (StateSpaceModelCT or StateSpaceModelDT). If None, the
            systems' base class (see infer_model_class()). The
            _attr_names class attribute determines the naming conventions.
        keys (list, optional): Custom keys for naming subsystems. If None,
            defaults to [prefix + "1", prefix + "2", ...].
        verbose_names (bool, optional): If True, use verbose parameter naming.
            Default: False.
        prefix (str, optional): Prefix for auto-generated subsystem keys.
            Default: "sys".
        name (str, optional): Name for the combined system. If None,
            auto-generates from system names using sep as separator.
            Default: None.
        sep (str, optional): Separator for joining system names when
            auto-generating the combined system name. Default: "_".

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: Combined series system.

    Example:
        >>> # For continuous-time models:
        >>> combined = connect_systems_in_series(
        ...     [sys1, sys2], StateSpaceModelCT
        ... )
    """
    validate_systems_are_compatible(systems)
    if model_class is None:
        model_class = infer_model_class(systems)
    attr_names = model_class._attr_names
    if keys is None:
        keys = [sys.name for sys in systems]
    keys = make_list_of_unique_names(keys, prefix=prefix)
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
        u1 = cas.SX.sym(attr_names["input_var"], nu1)
        x1 = cas.SX.sym(attr_names["state_var"], n1)
        state_func1 = getattr(sys1, attr_names["state_func"])
        params1 = merge_param_dicts(
            param_dicts[:i], keys[:i], verbose_names=verbose_names, sep=sep
        )
        state_out1 = state_func1(t, x1, u1, *params1.values())
        output_func1 = getattr(sys1, attr_names["output_func"])
        y1 = output_func1(t, x1, u1, *params1.values())

        # System 2
        n2 = sys2.n
        ny2 = sys2.ny
        u2 = y1
        x2 = cas.SX.sym(attr_names["state_var"], n2)
        state_func2 = getattr(sys2, attr_names["state_func"])
        params2 = sys2.params
        state_out2 = state_func2(t, x2, u2, *params2.values())
        output_func2 = getattr(sys2, attr_names["output_func"])
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
            keys[: i + 1],
            verbose_names=verbose_names,
            sep=sep,
        )
        state_function = cas.Function(
            attr_names["state_func"],
            [t, x, u, *params.values()],
            [state_combined],
            [
                "t",
                attr_names["state_var"],
                attr_names["input_var"],
                *params.keys(),
            ],
            [attr_names["state_output"]],
        )

        # Combined output function
        y = y2
        ny = ny2
        assert y.shape[0] == ny
        output_function = cas.Function(
            attr_names["output_func"],
            [t, x, u, *params.values()],
            [y],
            [
                "t",
                attr_names["state_var"],
                attr_names["input_var"],
                *params.keys(),
            ],
            [attr_names["output_var"]],
        )
        if verbose_names:
            input_names = [keys[0] + "_" + name for name in sys1.input_names]
        else:
            input_names = sys1.input_names
        if verbose_names:
            output_names = [
                keys[-1] + "_" + name for name in sys2.output_names
            ]
        else:
            output_names = sys2.output_names

        combined_system = model_class(
            state_function,
            output_function,
            n,
            nu,
            ny,
            **_dt_kwargs(systems),
            params=params,
            input_names=input_names,
            output_names=output_names,
        )

    combined_system.state_names = concatenate_lists_of_names(
        list(reversed(state_name_lists)),
        keys=list(reversed(keys)),
        verbose_names=verbose_names,
    )

    # Set combined system name
    if name is None:
        # Use separator to join system names
        system_names = make_list_of_unique_names(
            [sys.name for sys in systems], prefix=prefix
        )
        name = sep.join(system_names)
    combined_system.name = name

    return combined_system


def _normalize_connections(connections):
    """Convert connections to normalized dict format with gains.

    Args:
        connections: Connection specification (list of (source, target)
            tuples or dict of {target: source_spec})

    Returns:
        dict: {target: {source: gain, ...}, ...}

    Raises:
        ValueError: If list format has duplicate targets
    """
    if isinstance(connections, list):
        # List of tuples format: [('source', 'target'), ...]
        connections_dict = {}
        for conn in connections:
            if len(conn) != 2:
                raise ValueError(
                    f"Each connection tuple must have exactly 2 elements "
                    f"(source, target), got {len(conn)}"
                )
            source, target = conn
            if target in connections_dict:
                raise ValueError(
                    f"Duplicate connection target '{target}' in list "
                    f"format. Use dictionary format with gains for summing "
                    "junctions: connections = "
                    f"{{'{target}': {{'out1': 1.0, 'out2': 1.0}}}}"
                )
            connections_dict[target] = source
        connections = connections_dict

    # Normalize dict format
    connections_norm = {}
    for target, source_spec in connections.items():
        if isinstance(source_spec, str):
            # Simple string: 'sys2_y' -> {'sys2_y': 1.0}
            connections_norm[target] = {source_spec: 1.0}
        elif isinstance(source_spec, list):
            # List of sources: ['sys2_y', 'sys3_y'] -> {'sys2_y': 1.0,
            # 'sys3_y': 1.0}
            connections_norm[target] = {name: 1.0 for name in source_spec}
        elif isinstance(source_spec, dict):
            # Already in correct format: {'sys2_y': 1.0, 'sys3_y': -0.5}
            connections_norm[target] = source_spec
        else:
            raise ValueError(
                f"Connection value for '{target}' must be a string, "
                f"list, or dict, got {type(source_spec)}"
            )

    return connections_norm


def validate_systems_are_compatible(systems):
    """Check all systems are compatible for connection.

    Systems must have:
    1. The same _attr_names (i.e., same system type)
    2. For discrete-time systems, equal time intervals (dt)
    """
    if not systems:
        raise ValueError("Cannot validate empty system list")

    # Check all systems have same attribute naming convention
    expected_attrs = systems[0]._attr_names
    for i, sys in enumerate(systems[1:], 1):
        if sys._attr_names != expected_attrs:
            raise ValueError(
                f"Cannot combine systems with different types. "
                f"System 0 has {expected_attrs}, "
                f"but system {i} has {sys._attr_names}"
            )

    # Check discrete-time systems have same time interval
    if hasattr(systems[0], "dt"):
        validate_equal_dt(systems)


def _validate_connections(
    connections_norm, parallel_sys, external_inputs=None, external_outputs=None
):
    """Validate connection specification.

    Args:
        connections_norm: Normalized connections dict
        parallel_sys: The parallel system
        external_inputs: List of external input names, including any new
            signals not already in the parallel system (e.g. a setpoint
            injected via input_names).
        external_outputs: List of external output names, including any
            new signals not already in the parallel system (e.g. a sum of
            subsystem outputs declared in output_names).

    Raises:
        ValueError: If connections are invalid
    """
    if external_inputs is None:
        external_inputs = []
    if external_outputs is None:
        external_outputs = []

    # Check all targets exist: subsystem inputs or new external outputs
    for target in connections_norm:
        if target in parallel_sys.input_names:
            continue
        if target in parallel_sys.output_names:
            raise ValueError(
                f"Connection target '{target}' is an output of a "
                "subsystem. Targets must be subsystem inputs or new "
                "external outputs."
            )
        if target not in external_outputs:
            raise ValueError(
                f"Connection target '{target}' not found in parallel "
                "system. Available inputs: "
                f"{parallel_sys.input_names}. To make it a new external "
                "output, add it to output_names."
            )

    # Check all source names exist (can be outputs, inputs, or new externals)
    for target, source_dict in connections_norm.items():
        for source in source_dict:
            is_output = source in parallel_sys.output_names
            is_input = source in parallel_sys.input_names
            is_new_external = source in external_inputs
            if not is_output and not is_input and not is_new_external:
                raise ValueError(
                    f"Connection source '{source}' (for target "
                    f"'{target}') not found in parallel system. "
                    f"Available outputs: {parallel_sys.output_names}, "
                    f"available inputs: {parallel_sys.input_names}. To "
                    "make it a new external input, add it to input_names."
                )


def _build_output_vector(
    y_parallel,
    u_internal,
    u_ext,
    connections_norm,
    external_inputs,
    external_outputs,
    parallel_sys,
):
    """Build the external output vector of the connected system.

    Each external output is either a subsystem output, taken from the
    parallel system output vector, or a new output defined by a
    connection, computed as the weighted sum of its sources. A source of
    a new output can be a subsystem output, an external input, or a
    subsystem input (including a connected one, so internal signals can
    be exposed).

    Args:
        y_parallel: Full output vector from parallel system
        u_internal: Full (resolved) input vector of the parallel system
        u_ext: External input vector (symbolic)
        connections_norm: Normalized connections dict
        external_inputs: List of external input names
        external_outputs: List of external output names
        parallel_sys: The parallel system

    Returns:
        cas.SX: External output vector
    """
    output_name_to_idx = {
        name: i for i, name in enumerate(parallel_sys.output_names)
    }
    input_name_to_idx = {
        name: i for i, name in enumerate(parallel_sys.input_names)
    }
    external_input_map = {name: i for i, name in enumerate(external_inputs)}

    outputs = []
    for name in external_outputs:
        if name not in connections_norm:
            outputs.append(y_parallel[output_name_to_idx[name]])
            continue
        sum_expr = cas.SX(0)
        for src, gain in connections_norm[name].items():
            if src in output_name_to_idx:
                sum_expr += gain * y_parallel[output_name_to_idx[src]]
            elif src in external_input_map:
                sum_expr += gain * u_ext[external_input_map[src]]
            else:
                sum_expr += gain * u_internal[input_name_to_idx[src]]
        outputs.append(sum_expr)
    return cas.vertcat(*outputs)


def _build_internal_input_vector(
    u_ext,
    x,
    t,
    connections_norm,
    external_inputs,
    parallel_sys,
    attr_names,
):
    """Build the full internal input vector for the parallel system.

    Iteratively resolves connected inputs by propagating external inputs and
    system outputs through the connection graph. Handles multi-stage
    connections and mixed-source connections (where an input depends on both
    external inputs and system outputs).

    Each iteration evaluates the parallel system outputs with the current
    internal input estimates, then recomputes every connected input whose
    sources (external inputs or system outputs) are available. A connection
    chain of depth d converges after d + 1 iterations; running
    len(input_names) + 1 iterations covers any acyclic chain.

    Note: algebraic loops are not detected. If connections create a cycle
    where every link has direct feedthrough, the iterative resolution does
    not converge and the result depends on the number of iterations. See
    test_connect_systems_algebraic_loop_silent_failure.

    Args:
        u_ext: External input vector (symbolic)
        x: State vector (symbolic)
        t: Time variable (symbolic)
        connections_norm: Normalized connections dict
        external_inputs: List of external input names
        parallel_sys: The parallel system
        attr_names: Attribute names dict (CT or DT)

    Returns:
        cas.SX: Full internal input vector for parallel system
    """
    external_input_map = {name: i for i, name in enumerate(external_inputs)}
    input_name_to_idx = {
        name: i for i, name in enumerate(parallel_sys.input_names)
    }
    output_name_to_idx = {
        name: i for i, name in enumerate(parallel_sys.output_names)
    }

    # Initialize u_working: external inputs that directly appear in the
    # parallel system get their u_ext values; connected inputs start at zero.
    u_working = cas.SX.zeros(len(parallel_sys.input_names), 1)
    for name, i in external_input_map.items():
        if name in input_name_to_idx:
            u_working[input_name_to_idx[name]] = u_ext[i]

    output_func_attr = getattr(parallel_sys, attr_names["output_func"])

    # Iteratively resolve connected inputs. Each iteration evaluates the
    # parallel system outputs with the current u_working, then recomputes
    # every connected input whose sources are now available (either as
    # external inputs or system outputs). A chain of depth d converges after
    # d + 1 iterations; n_inputs + 1 is a safe upper bound for acyclic graphs.
    max_iterations = len(parallel_sys.input_names) + 1
    for _ in range(max_iterations):
        y_parallel = output_func_attr(
            t, x, u_working, *parallel_sys.params.values()
        )
        for input_name in parallel_sys.input_names:
            if input_name not in connections_norm:
                continue
            sum_expr = cas.SX(0)
            all_available = True
            for src, gain in connections_norm[input_name].items():
                if src in output_name_to_idx:
                    sum_expr += gain * y_parallel[output_name_to_idx[src]]
                elif src in external_input_map:
                    sum_expr += gain * u_ext[external_input_map[src]]
                else:
                    all_available = False
                    break
            if all_available:
                u_working[input_name_to_idx[input_name]] = sum_expr

    # Build the final u_internal vector from u_working and external pass-throughs.
    u_internal_parts = []
    for input_name in parallel_sys.input_names:
        if input_name in external_input_map:
            u_internal_parts.append(u_ext[external_input_map[input_name]])
        elif input_name in connections_norm:
            u_internal_parts.append(u_working[input_name_to_idx[input_name]])
        else:
            raise ValueError(
                f"Input '{input_name}' is neither external nor connected"
            )

    return cas.vertcat(*u_internal_parts)


@records
def connect_systems(
    systems,
    connections,
    model_class=None,
    input_names=None,
    output_names=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Connect multiple nonlinear systems with arbitrary connections.

    This function combines multiple state-space models (continuous-time or
    discrete-time) into a single connected system by:
    1. First arranging all systems in parallel
    2. Creating connections between signals
    3. Exposing only specified external inputs and outputs

    Terminology: a signal is a named input or output of a subsystem or of
    the connected system. A connection sets the value of one signal, its
    target, to the weighted sum of one or more other signals, its sources.

    Args:
        systems (list): List of state-space model objects to connect.
        connections (list or dict): Connection specification:
            - List format: [(source, target), ...] for simple one-to-one
                connections.
            - Dict format: {target: source_spec, ...} where source_spec
                can be:
                - A string: 'sys2_y' (simple connection from one source)
                - A list: ['sys2_y', 'sys3_y'] (sum with unit gains)
                - A dict: {'sys2_y': 1.0, 'sys3_y': -0.5} (weighted sum)

            Targets can be:
                - Subsystem inputs
                - New external outputs (names that are not subsystem
                  outputs), which must be listed in output_names

            Sources can be:
                - Subsystem outputs
                - External inputs, including new ones (names that are not
                  subsystem inputs), which must be listed in input_names
                - For new external outputs only: connected subsystem
                  inputs (to expose internal signals)

            Note: If a subsystem input appears as a source for another
            subsystem input, it must be an external input (not
            connected). This prevents circular dependencies.
        model_class (optional): Target model class (StateSpaceModelCT or
            StateSpaceModelDT). If None, the systems' base class (see
            infer_model_class()). The _attr_names class attribute
            determines the naming conventions.
        input_names (list, optional): List of external input signal names to
            expose. If None, exposes all inputs that are not connected.
            Names that are not subsystem inputs are new external inputs.
        output_names (list, optional): List of external output signal names
            to expose. If None, exposes all subsystem outputs. Names that
            are not subsystem outputs are new external outputs and must be
            connection targets.
        keys (list, optional): Custom keys for naming subsystems.
        verbose_names (bool, optional): Use verbose parameter naming.
            Default: False.
        prefix (str, optional): Prefix for auto-generated keys. Default: "sys".
        name (str, optional): Name for the connected system. If None,
            auto-generates from system names using sep as separator.
            Default: None.
        sep (str, optional): Separator for joining system names when
            auto-generating the combined system name. Default: "_".

    Returns:
        StateSpaceModelCT or StateSpaceModelDT: Connected system with specified
            external inputs and outputs.

    Raises:
        ValueError: If connections are invalid (non-existent signals, many-to-
            one without gains, dt mismatch for discrete-time systems, etc.)

    Warning:
        This function does not detect algebraic loops. User must ensure that
        connections do not create direct feedthrough loops (where output y
        depends on input u, and that same u depends on y through
        connections).

    Examples:
        >>> # Example 1: Simple feedback connection
        >>> from cas_models.continuous_time.models import StateSpaceModelCT
        >>> sys1 = StateSpaceModelCT(...)  # Plant
        >>> sys2 = StateSpaceModelCT(...)  # Controller
        >>>
        >>> # Connect output of sys2 to input of sys1, and vice versa
        >>> connected = connect_systems(
        ...     [sys1, sys2],
        ...     connections=[("sys2_y", "sys1_u"), ("sys1_y", "sys2_u")],
        ...     model_class=StateSpaceModelCT,
        ... )
        >>>
        >>> # Example 2: Summing junction with feedback
        >>> # sys3 output feeds into sys1, with feedback from sys1 and sys2
        >>> connected = connect_systems(
        ...     [sys1, sys2, sys3],
        ...     connections={
        ...         "sys1_u": {"sys2_y": 1.0, "sys3_y": -0.5},  # Weighted sum
        ...         "sys2_u": "sys1_y",  # Simple connection
        ...     },
        ...     model_class=StateSpaceModelCT,
        ...     input_names=["sys3_u"],  # Only sys3 input is external
        ...     output_names=["sys1_y", "sys2_y"],  # Expose these outputs
        ... )
        >>>
        >>> # Example 3: Closed-loop system (no external inputs)
        >>> connected = connect_systems(
        ...     [sys1, sys2],
        ...     connections={"sys1_u": "sys2_y", "sys2_u": "sys1_y"},
        ...     model_class=StateSpaceModelCT,
        ...     input_names=[],  # No external inputs
        ...     output_names=["sys1_y"],
        ... )
        >>>
        >>> # Example 4: Input-to-input connections (e.g., tank network)
        >>> # Connect one external input to multiple system inputs
        >>> connected = connect_systems(
        ...     [sys1, sys2, sys3],
        ...     connections={
        ...         # sys1_u gets sum of sys2_u and sys3_u
        ...         "sys1_u": ["sys2_u", "sys3_u"],
        ...         # sys2 has another input from sys1 output
        ...         "sys2_v": "sys1_y",
        ...     },
        ...     model_class=StateSpaceModelCT,
        ...     input_names=["sys2_u", "sys3_u"],  # These must be external
        ...     output_names=["sys3_y"],
        ... )
        >>>
        >>> # Example 5: New external output (sum of subsystem outputs)
        >>> # Two SISO models with different inputs and a summed output
        >>> connected = connect_systems(
        ...     [sys1, sys2],
        ...     connections={"y": {"sys1_y": 1.0, "sys2_y": 1.0}},
        ...     model_class=StateSpaceModelCT,
        ...     input_names=["sys1_u", "sys2_u"],
        ...     output_names=["y"],  # New output, a connection target
        ... )

    Note:
        - All input/output names must be unique (enforced by
            concatenate_lists_of_names)
        - For discrete-time systems, all must have the same dt value
        - Connections must not create algebraic loops (no automatic detection)

    See Also:
        block_diag_systems: Combine systems without connections
        connect_systems_in_series: Connect systems in series
    """
    if model_class is None:
        model_class = infer_model_class(systems)
    attr_names = model_class._attr_names

    # Step 1: Create parallel system
    parallel_sys = block_diag_systems(
        systems,
        model_class,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )

    # Combined system name
    sys_name = parallel_sys.name

    # Handle empty connections
    if not connections:
        connections = {}

    # Step 2: Normalize connections: {target: {source: gain, ...}, ...}
    connections_norm = _normalize_connections(connections)

    # Step 3: Determine external inputs and outputs
    if input_names is None:
        # All inputs NOT in connections are external
        external_inputs = [
            name
            for name in parallel_sys.input_names
            if name not in connections_norm
        ]
    else:
        external_inputs = list(input_names)
        # Validate no overlap with connection targets
        overlap = set(external_inputs) & set(connections_norm)
        if overlap:
            raise ValueError(
                f"Inputs cannot be both external and connected. "
                f"Found in both: {overlap}"
            )
        # Names not in the parallel system are treated as new external signals
        # (e.g. a setpoint injected into a summing junction). They are valid
        # as connection sources but not as connection targets.

    if output_names is None:
        external_outputs = parallel_sys.output_names  # All outputs
    else:
        external_outputs = list(output_names)
        # Names not in the parallel system are new external outputs, which
        # must be connection targets (e.g. a sum of subsystem outputs)
        for name in external_outputs:
            if (
                name not in parallel_sys.output_names
                and name not in connections_norm
            ):
                raise ValueError(
                    f"External output '{name}' not found in parallel system "
                    "and not a connection target. Available outputs: "
                    f"{parallel_sys.output_names}"
                )

    _validate_connections(
        connections_norm, parallel_sys, external_inputs, external_outputs
    )

    # Validate that subsystem inputs used as sources for other subsystem
    # inputs are external. Sources of new external outputs may be connected
    # inputs (to expose internal signals). If a name exists in both
    # output_names and input_names, it is resolved as an output (the
    # computation code already does this), so skip the check.
    for target, source_dict in connections_norm.items():
        if target not in parallel_sys.input_names:
            continue
        for source in source_dict:
            is_also_output = source in parallel_sys.output_names
            if source in parallel_sys.input_names and not is_also_output:
                # Source is an input-only name - must be external
                if source not in external_inputs:
                    raise ValueError(
                        f"Connection source input '{source}' (for target "
                        f"'{target}') must be an external input (not "
                        f"connected). Currently it is connected."
                    )

    # Step 4: Build connected state function
    t = cas.SX.sym("t")
    x = cas.SX.sym(attr_names["state_var"], parallel_sys.n)
    u_ext = cas.SX.sym(attr_names["input_var"], len(external_inputs))

    # Build internal input vector (avoiding algebraic loops)
    u_internal = _build_internal_input_vector(
        u_ext,
        x,
        t,
        connections_norm,
        external_inputs,
        parallel_sys,
        attr_names,
    )

    # Call parallel system's state function
    state_func_attr = getattr(parallel_sys, attr_names["state_func"])
    rhs = state_func_attr(t, x, u_internal, *parallel_sys.params.values())

    # Create connected state function
    state_func = cas.Function(
        attr_names["state_func"],
        [t, x, u_ext, *parallel_sys.params.values()],
        [rhs],
        [
            "t",
            attr_names["state_var"],
            attr_names["input_var"],
            *parallel_sys.params.keys(),
        ],
        [attr_names["state_output"]],
    )

    # Step 5: Build connected output function
    # Note: This may include output feedthrough
    u_internal = _build_internal_input_vector(
        u_ext,
        x,
        t,
        connections_norm,
        external_inputs,
        parallel_sys,
        attr_names,
    )

    # Get full parallel system outputs
    output_func_attr = getattr(parallel_sys, attr_names["output_func"])
    y_parallel = output_func_attr(
        t, x, u_internal, *parallel_sys.params.values()
    )

    # Select the external outputs and compute any new ones
    y_ext = _build_output_vector(
        y_parallel,
        u_internal,
        u_ext,
        connections_norm,
        external_inputs,
        external_outputs,
        parallel_sys,
    )

    # Create connected output function
    output_func = cas.Function(
        attr_names["output_func"],
        [t, x, u_ext, *parallel_sys.params.values()],
        [y_ext],
        [
            "t",
            attr_names["state_var"],
            attr_names["input_var"],
            *parallel_sys.params.keys(),
        ],
        [attr_names["output_var"]],
    )

    # Step 6: Create result model
    # Check if this is a discrete-time system (has dt attribute)
    if hasattr(parallel_sys, "dt") and parallel_sys.dt is not None:
        connected_sys = model_class(
            state_func,
            output_func,
            parallel_sys.n,
            len(external_inputs),
            len(external_outputs),
            dt=parallel_sys.dt,
            params=parallel_sys.params,
            name=sys_name,
            input_names=external_inputs,
            state_names=parallel_sys.state_names,
            output_names=external_outputs,
        )
    else:
        # Continuous-time system (no dt parameter)
        connected_sys = model_class(
            state_func,
            output_func,
            parallel_sys.n,
            len(external_inputs),
            len(external_outputs),
            params=parallel_sys.params,
            name=sys_name,
            input_names=external_inputs,
            state_names=parallel_sys.state_names,
            output_names=external_outputs,
        )

    return connected_sys


@records
def connect_feedback_system(
    sys1,
    sys2=None,
    model_class=None,
    sign=-1,
    input_names=None,
    output_names=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine two systems in a feedback loop.

    Builds the closed-loop state and output functions directly from CasADi
    symbolic expressions.  The loop topology is:

                  +
        y_sp ---( )--- e ---> sys1 ---+---> y
                 | sign               |
                 |                    |
                 +------- sys2 <------+

    sys1 must have no direct feedthrough (output must not depend on its
    input), or an algebraic loop results and a ValueError is raised.

    Args:
        sys1: Forward-path state-space model.  Must satisfy the no-direct-
            feedthrough condition (i.e. output does not depend on input).
        sys2: Feedback-path element.  Three forms are accepted:
            - None (default): unity feedback (feedback gain = 1).
            - int or float: constant scalar gain in the feedback path.
            - state-space model: a dynamic feedback compensator.  Must have
              as many inputs as sys1 has outputs and vice-versa.
              sys2 itself may have direct feedthrough only when sys1 does
              not (guaranteed by the algebraic-loop check above).
        model_class (optional): Class used to instantiate the returned
            model (e.g. StateSpaceModelCT or StateSpaceModelDT). If None,
            the base class of sys1 (and sys2 if it is a model; see
            infer_model_class()).
        sign: Sign applied to the feedback signal at the summing junction.
            Use -1 (default) for negative feedback, +1 for positive feedback.
        input_names: Names of the closed-loop reference inputs.  Default:
            ``[f"{n}_sp" for n in sys1.output_names]``.
        output_names: Names of the closed-loop outputs.  Default: derived
            from sys1's output names, disambiguated with system keys when
            names conflict.
        keys: Two-element list of string keys used to prefix shared
            parameter/state names when disambiguation is needed.  Default:
            derived from system names (``[sys1.name, sys2.name]``).
        verbose_names: If True, always prepend system keys to parameter and
            state names.  If False (default), keys are prepended only when
            names conflict across the two systems.
        prefix: Prefix used to auto-generate keys when a system's name is
            None (default: ``"sys"``).
        name: Name of the returned closed-loop model.  Default:
            ``"fbk_{sys1_key}"`` for unity/scalar feedback, or
            ``"fbk_{sys1_key}_{sys2_key}"`` for a dynamic feedback system.
        sep: Separator character used when joining system names to form the
            combined system ``name`` (default ``"_"``).

    Returns:
        A new state-space model of type ``model_class`` representing the
        closed-loop system.

    Raises:
        ValueError: If sys1 and sys2 are of different kinds, if sys2
            dimension mismatches sys1, or if an algebraic loop is detected.
        NotImplementedError: If a scalar ``sys2`` is passed with a
            discrete-time ``model_class``.
    """
    if model_class is None:
        model_class = infer_model_class(
            [sys1] if sys2 is None or isinstance(sys2, (int, float))
            else [sys1, sys2]
        )

    unity_feedback = sys2 is None or (
        isinstance(sys2, (int, float)) and sys2 == 1
    )
    scalar_gain = isinstance(sys2, (int, float))

    if scalar_gain:
        if model_class._attr_names["state_func"] != "f":
            raise NotImplementedError(
                "Scalar sys2 is not yet supported for discrete-time systems"
            )
        sys2_gain = float(sys2)
        has_sys2_states = False
    elif sys2 is None:
        sys2_gain = 1.0
        has_sys2_states = False
    else:
        if sys1.ny != sys2.nu:
            raise ValueError(
                "sys2 must have same number of inputs as outputs from sys1"
            )
        if sys2.ny != sys1.nu:
            raise ValueError(
                "sys1 must have same number of inputs as outputs from sys2"
            )
        sys2_gain = None
        has_sys2_states = True
        validate_systems_are_compatible([sys1, sys2])

    sys2_name = "gain" if not has_sys2_states else sys2.name
    effective_keys = make_list_of_unique_names(
        [sys1.name, sys2_name] if keys is None else list(keys), prefix=prefix
    )
    attr_names = model_class._attr_names
    output_func1 = getattr(sys1, attr_names["output_func"])
    state_func1 = getattr(sys1, attr_names["state_func"])
    if has_sys2_states:
        output_func2 = getattr(sys2, attr_names["output_func"])
        state_func2 = getattr(sys2, attr_names["state_func"])

    # Detect algebraic loops: an algebraic loop exists when the cycle has
    # direct feedthrough all the way round.  For unity/scalar feedback that
    # means sys1 alone; for a proper sys2, both must have feedthrough.
    t_chk = cas.SX.sym("t")
    u1_chk = cas.SX.sym("u", sys1.nu)
    y1_chk = output_func1(
        t_chk, cas.SX.sym("x", sys1.n), u1_chk, *sys1.params.values()
    )
    sys1_feedthrough = cas.depends_on(y1_chk, u1_chk)
    if has_sys2_states:
        u2_chk = cas.SX.sym("u", sys2.nu)
        y2_chk = output_func2(
            t_chk, cas.SX.sym("x", sys2.n), u2_chk, *sys2.params.values()
        )
        algebraic_loop = sys1_feedthrough and cas.depends_on(y2_chk, u2_chk)
    else:
        algebraic_loop = sys1_feedthrough
    if algebraic_loop:
        who = "sys1 and sys2 both have" if has_sys2_states else "sys1 has"
        raise ValueError(
            f"Algebraic loop: {who} direct feedthrough (output depends on "
            f"input). The feedback loop cannot be resolved without solving an "
            f"implicit equation."
        )

    param_lists = (
        [sys1.params, sys2.params] if has_sys2_states else [sys1.params]
    )
    params = merge_param_dicts(
        param_lists,
        effective_keys[: len(param_lists)],
        verbose_names=verbose_names,
        sep=sep,
    )

    t = cas.SX.sym("t")
    n1, nu1, ny1 = sys1.n, sys1.nu, sys1.ny
    x1 = cas.SX.sym(attr_names["state_var"], n1)
    if has_sys2_states:
        x2 = cas.SX.sym(attr_names["state_var"], sys2.n)

    # sys1 has no feedthrough (confirmed above), so y1 is independent of its
    # input — evaluate with zeros to obtain the feedback signal y2 and error e.
    y1 = output_func1(t, x1, cas.SX.zeros(nu1), *sys1.params.values())
    y2 = (
        output_func2(t, x2, y1, *sys2.params.values())
        if has_sys2_states
        else sys2_gain * y1
    )
    y_sp = cas.SX.sym(attr_names["input_var"], nu1)
    e = y_sp + sign * y2

    dx1 = state_func1(t, x1, e, *sys1.params.values())
    if has_sys2_states:
        x, dx, n = (
            cas.vcat([x1, x2]),
            cas.vcat([dx1, state_func2(t, x2, y1, *sys2.params.values())]),
            n1 + sys2.n,
        )
    else:
        x, dx, n = x1, dx1, n1

    func_args = [
        "t",
        attr_names["state_var"],
        attr_names["input_var"],
        *params.keys(),
    ]
    func_inputs = [t, x, y_sp, *params.values()]
    state_function = cas.Function(
        attr_names["state_func"],
        func_inputs,
        [dx],
        func_args,
        [attr_names["state_output"]],
    )
    output_function = cas.Function(
        attr_names["output_func"],
        func_inputs,
        [y1],
        func_args,
        [attr_names["output_var"]],
    )

    if input_names is None:
        input_names = [f"{n}_sp" for n in sys1.output_names]
    if output_names is None:
        if verbose_names:
            output_names = [
                f"{effective_keys[0]}_{n}" for n in sys1.output_names
            ]
        else:
            output_names = sys1.output_names
    state_names = concatenate_lists_of_names(
        [sys1.state_names, sys2.state_names if has_sys2_states else []],
        keys=effective_keys[:2],
        verbose_names=verbose_names,
    )
    if name is None:
        name_keys = (
            [effective_keys[0]] if unity_feedback else effective_keys[:2]
        )
        name = sep.join(["fbk"] + name_keys)

    return model_class(
        state_function,
        output_function,
        n,
        nu1,
        ny1,
        params=params,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
        name=name,
    )
