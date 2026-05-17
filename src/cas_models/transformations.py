"""Transformations for combining state-space models.

Provides functions for combining continuous-time and discrete-time
state-space models in parallel, series, and feedback configurations.

Functions
---------
connect_systems_in_parallel
    Combine a list of models with independent inputs and outputs side by side.
connect_systems_in_series
    Connect models end-to-end, feeding each output into the next input.
connect_systems
    Connect models with arbitrary named signal connections.
connect_feedback_system
    Close a feedback loop around a forward-path model.
validate_systems_are_compatible
    Raise if a list of models are not all the same type (CT vs DT).
block_diag
    Construct a block-diagonal CasADi SX matrix from a list of matrices.
linear_systems_in_parallel
    Combine linear systems in A, B, C, D dict form side by side.
linear_systems_in_series
    Combine linear systems in A, B, C, D dict form end-to-end.
"""

from functools import reduce
import casadi as cas
from cas_models.param_utils import (
    make_list_of_unique_names,
    concatenate_lists_of_names,
    merge_param_dicts,
)
from cas_models.discrete_time.models import is_ss_dt, validate_equal_dt


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


# TODO: These linear functions are redundant since no linear models
# currently defined
def linear_systems_in_parallel(
    systems, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a collection of linear systems into one large parallel
    system.
    """
    A = cas.sparsify(
        block_diag(list(sys["A"] for sys in systems), square=True)
    )
    B = cas.sparsify(block_diag(list(sys["B"] for sys in systems)))
    C = cas.sparsify(block_diag(list(sys["C"] for sys in systems)))
    D = cas.sparsify(block_diag(list(sys["D"] for sys in systems)))
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
    )
    input_names = concatenate_lists_of_names(
        [sys.input_names for sys in systems], keys=keys, prefix=prefix
    )
    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems], keys=keys, prefix=prefix
    )
    output_names = concatenate_lists_of_names(
        [sys.output_names for sys in systems], keys=keys, prefix=prefix
    )
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "params": params,
        "input_names": input_names,
        "state_names": state_names,
        "output_names": output_names,
    }


def linear_systems_in_series(
    systems, keys=None, verbose_names=False, prefix="sys"
):
    """Combine a sequence of linear systems into one system by connecting their
    outputs and inputs in series.
    """
    n_sys = len(systems)
    col_sizes = [sys["A"].shape[1] for sys in systems]
    A_rows = []
    B_rows = []
    C_row = []
    for i, sys in enumerate(systems):
        A = sys["A"]
        B = sys["B"]
        C = sys["C"]
        D = sys["D"]

        # Add rows to A matrix
        n_rows = A.shape[0]
        assert n_rows == col_sizes[i], "A matrix not square"
        A_row = []
        if i > 0:
            for j in range(i - 1):
                A_row.append(cas.SX.zeros(n_rows, col_sizes[j]))
            A_row.append(B @ systems[i - 1]["C"])
        A_row.append(A)
        for j in range(i + 1, n_sys):
            A_row.append(cas.SX.zeros(n_rows, col_sizes[j]))
        A_rows.append(A_row)

        # Add rows to B matrix
        if i > 0:
            B_rows.append(B @ systems[i - 1]["D"])
        else:
            B_rows.append(B)

        # Add columns to C matrix
        if i < n_sys - 1:
            C_row.append(systems[i + 1]["D"] @ C)
        else:
            C_row.append(C)

    A = cas.sparsify(cas.blockcat(A_rows))
    B = cas.sparsify(cas.vcat(B_rows))
    C = cas.sparsify(cas.hcat(C_row))
    D = reduce(cas.SX.__matmul__, (sys["D"] for sys in systems))
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
    )
    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems], keys=keys, prefix=prefix
    )
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "params": params,
        "state_names": state_names,
    }


def connect_systems_in_parallel(
    systems,
    model_class,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine a collection of nonlinear systems into one large parallel
    system.

    This function works for both continuous-time and discrete-time systems.
    The attribute names are automatically determined from the model_class.

    Args:
        systems (list): List of state-space model objects to combine in
            parallel.
        model_class: Class to instantiate for the combined system
            (StateSpaceModelCT or StateSpaceModelDT). The _attr_names class
            attribute determines the naming conventions.
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
        StateSpaceModelCT or StateSpaceModelDT: Combined parallel system.

    Example:
        >>> # For continuous-time models:
        >>> combined = connect_nonlinear_systems_in_parallel(
        ...     [sys1, sys2], StateSpaceModelCT
        ... )
        >>>
        >>> # For discrete-time models:
        >>> combined = connect_nonlinear_systems_in_parallel(
        ...     [sys1, sys2], StateSpaceModelDT
        ... )
    """
    validate_systems_are_compatible(systems)
    attr_names = model_class._attr_names
    if keys is None:
        keys = [sys.name for sys in systems]
    keys = make_list_of_unique_names(keys, prefix=prefix)
    params = merge_param_dicts(
        [sys.params for sys in systems],
        keys,
        verbose_names=verbose_names,
    )

    t = cas.SX.sym("t")

    u_signals = []
    x_states = []
    state_outputs = []
    y_signals = []

    for sys in systems:
        x = cas.SX.sym(attr_names["state_var"], sys.n)
        x_states.append(x)

        u = cas.SX.sym(attr_names["input_var"], sys.nu)
        u_signals.append(u)

        # Get state function and call it: f(t, x, u, *params.values())
        state_func = getattr(sys, attr_names["state_func"])
        state_out = state_func(t, x, u, *sys.params.values())
        state_outputs.append(state_out)

        # Get output function and call it: h(t, x, u, *params.values())
        output_func = getattr(sys, attr_names["output_func"])
        y = output_func(t, x, u, *sys.params.values())
        y_signals.append(y)

    x = cas.vcat(x_states)
    n = x.shape[0]

    u = cas.vcat(u_signals)
    nu = u.shape[0]

    state_combined = cas.vcat(state_outputs)
    assert state_combined.shape == x.shape

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

    y = cas.vcat(y_signals)
    ny = y.shape[0]

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

    state_names = concatenate_lists_of_names(
        [sys.state_names for sys in systems],
        keys=keys,
        prefix=prefix,
        verbose_names=verbose_names,
    )
    assert len(state_names) == n

    input_names = concatenate_lists_of_names(
        [sys.input_names for sys in systems],
        keys=keys,
        prefix=prefix,
        verbose_names=verbose_names,
    )
    assert len(input_names) == nu

    output_names = concatenate_lists_of_names(
        [sys.output_names for sys in systems],
        keys=keys,
        prefix=prefix,
        verbose_names=verbose_names,
    )
    assert len(output_names) == ny

    # Generate combined system name if not provided
    if name is None:
        # Use separator to join system names
        system_names = make_list_of_unique_names(
            [sys.name for sys in systems], prefix=prefix
        )
        name = sep.join(system_names)

    combined_system = model_class(
        state_function,
        output_function,
        n,
        nu,
        ny,
        params=params,
        name=name,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
    )

    return combined_system


def connect_systems_in_series(
    systems,
    model_class,
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
        model_class: Class to instantiate for the combined system
            (StateSpaceModelCT or StateSpaceModelDT). The _attr_names class
            attribute determines the naming conventions.
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
        >>> combined = connect_nonlinear_systems_in_series(
        ...     [sys1, sys2], StateSpaceModelCT
        ... )
    """
    validate_systems_are_compatible(systems)
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
            param_dicts[:i], keys[:i], verbose_names=verbose_names
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
        connections: Connection specification (list of tuples or dict)

    Returns:
        dict: {input_name: {output_name: gain, ...}, ...}

    Raises:
        ValueError: If list format has duplicate input targets
    """
    if isinstance(connections, list):
        # List of tuples format: [('output_name', 'input_name'), ...]
        connections_dict = {}
        for conn in connections:
            if len(conn) != 2:
                raise ValueError(
                    f"Each connection tuple must have exactly 2 elements "
                    f"(output_name, input_name), got {len(conn)}"
                )
            output_name, input_name = conn
            if input_name in connections_dict:
                raise ValueError(
                    f"Duplicate connection target '{input_name}' in list "
                    f"format. Use dictionary format with gains for summing "
                    "junctions: connections = "
                    f"{{'{input_name}': {{'out1': 1.0, 'out2': 1.0}}}}"
                )
            connections_dict[input_name] = output_name
        connections = connections_dict

    # Normalize dict format
    connections_norm = {}
    for input_name, output_spec in connections.items():
        if isinstance(output_spec, str):
            # Simple string: 'sys2_y' -> {'sys2_y': 1.0}
            connections_norm[input_name] = {output_spec: 1.0}
        elif isinstance(output_spec, list):
            # List of outputs: ['sys2_y', 'sys3_y'] -> {'sys2_y': 1.0,
            # 'sys3_y': 1.0}
            connections_norm[input_name] = {name: 1.0 for name in output_spec}
        elif isinstance(output_spec, dict):
            # Already in correct format: {'sys2_y': 1.0, 'sys3_y': -0.5}
            connections_norm[input_name] = output_spec
        else:
            raise ValueError(
                f"Connection value for '{input_name}' must be a string, "
                f"list, or dict, got {type(output_spec)}"
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
    connections_norm, parallel_sys, external_inputs=None
):
    """Validate connection specification.

    Args:
        connections_norm: Normalized connections dict
        parallel_sys: The parallel system
        external_inputs: List of external input names, including any new
            signals not already in the parallel system (e.g. a setpoint
            injected via input_names).

    Raises:
        ValueError: If connections are invalid
    """
    if external_inputs is None:
        external_inputs = []

    # Check all input names exist
    for input_name in connections_norm.keys():
        if input_name not in parallel_sys.input_names:
            raise ValueError(
                f"Connection target input '{input_name}' not found in "
                "parallel system. Available inputs: "
                f"{parallel_sys.input_names}"
            )

    # Check all source names exist (can be outputs, inputs, or new externals)
    for input_name, source_dict in connections_norm.items():
        for source_name in source_dict.keys():
            is_output = source_name in parallel_sys.output_names
            is_input = source_name in parallel_sys.input_names
            is_new_external = source_name in external_inputs
            if not is_output and not is_input and not is_new_external:
                raise ValueError(
                    f"Connection source '{source_name}' (for input "
                    f"'{input_name}') not found in parallel system. "
                    f"Available outputs: {parallel_sys.output_names}, "
                    f"available inputs: {parallel_sys.input_names}"
                )


def _extract_outputs(y_parallel, output_names, parallel_sys):
    """Extract specified outputs from parallel system output vector.

    Args:
        y_parallel: Full output vector from parallel system
        output_names: List of output names to extract
        parallel_sys: The parallel system

    Returns:
        cas.SX: Selected output vector
    """
    indices = [parallel_sys.output_names.index(name) for name in output_names]
    return cas.vertcat(*[y_parallel[i] for i in indices])


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

    Iteratively fills connected inputs to handle multi-stage connections
    (e.g., external inputs → mixer → tank). Starts with zeros for connected
    inputs and propagates values layer by layer.

    Note: algebraic loops are not detected. If connections create a cycle
    where every link has direct feedthrough, the unresolvable inputs are
    left as zero and the resulting functions silently return incorrect values.
    See test_connect_systems_algebraic_loop_silent_failure.

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
    # Create index mappings for efficient lookups
    external_input_map = {name: i for i, name in enumerate(external_inputs)}
    input_name_to_idx = {
        name: i for i, name in enumerate(parallel_sys.input_names)
    }
    output_name_to_idx = {
        name: i for i, name in enumerate(parallel_sys.output_names)
    }

    # Build the internal input vector
    u_internal_parts = []

    for input_name in parallel_sys.input_names:
        if input_name in external_input_map:
            # This is an external input - pass through directly
            idx = external_input_map[input_name]
            u_internal_parts.append(u_ext[idx])
        elif input_name in connections_norm:
            # This is a connected input - compute from outputs/external inputs
            # Build temporary input vector for evaluating outputs
            u_temp = cas.SX.zeros(len(parallel_sys.input_names), 1)
            filled_inputs = set()  # Track which inputs have been filled

            # Fill external inputs that correspond to parallel system inputs.
            # New external signals (not in the parallel system) are only
            # used as sources in the final weighted sum, not as pass-through
            # inputs here.
            for name, idx in external_input_map.items():
                if name in input_name_to_idx:
                    temp_idx = input_name_to_idx[name]
                    u_temp[temp_idx] = u_ext[idx]
                filled_inputs.add(name)

            # Iteratively fill connected inputs (handles multi-stage
            # connections)
            max_iterations = len(parallel_sys.input_names)
            for _ in range(max_iterations):
                changed = False

                # Fill inputs that depend only on external inputs
                for temp_name in parallel_sys.input_names:
                    if (
                        temp_name in connections_norm
                        and temp_name not in filled_inputs
                    ):
                        sources = connections_norm[temp_name]
                        if all(src in external_inputs for src in sources):
                            temp_idx = input_name_to_idx[temp_name]
                            sum_expr = sum(
                                gain * u_ext[external_input_map[src]]
                                for src, gain in sources.items()
                            )
                            u_temp[temp_idx] = sum_expr
                            filled_inputs.add(temp_name)
                            changed = True

                # Evaluate outputs with current u_temp
                output_func_attr = getattr(
                    parallel_sys, attr_names["output_func"]
                )
                y_parallel = output_func_attr(
                    t, x, u_temp, *parallel_sys.params.values()
                )

                # Fill inputs that depend only on outputs
                for temp_name in parallel_sys.input_names:
                    if (
                        temp_name in connections_norm
                        and temp_name not in filled_inputs
                    ):
                        sources = connections_norm[temp_name]
                        if all(
                            src in parallel_sys.output_names for src in sources
                        ):
                            temp_idx = input_name_to_idx[temp_name]
                            sum_expr = sum(
                                gain * y_parallel[output_name_to_idx[src]]
                                for src, gain in sources.items()
                            )
                            u_temp[temp_idx] = sum_expr
                            filled_inputs.add(temp_name)
                            changed = True

                # If nothing changed, we're done (or stuck in a cycle)
                # TODO: check here whether all connected inputs are in
                # filled_inputs; if not, an algebraic loop is preventing
                # resolution and we should raise ValueError rather than
                # silently using zeros for the unresolvable inputs.
                if not changed:
                    break

            # Compute weighted sum for this connected input
            sum_expr = 0
            for source_name, gain in connections_norm[input_name].items():
                if source_name in output_name_to_idx:
                    # Source is an output
                    idx = output_name_to_idx[source_name]
                    sum_expr += gain * y_parallel[idx]
                elif source_name in external_input_map:
                    # Source is an external input
                    idx = external_input_map[source_name]
                    sum_expr += gain * u_ext[idx]
                else:
                    raise ValueError(
                        f"Connection source '{source_name}' is neither an "
                        f"output nor an external input"
                    )

            u_internal_parts.append(sum_expr)
        else:
            raise ValueError(
                f"Input '{input_name}' is neither external nor connected"
            )

    return cas.vertcat(*u_internal_parts)


def connect_systems(
    systems,
    connections,
    model_class,
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
    2. Creating internal connections between outputs and inputs
    3. Exposing only specified external inputs and outputs

    Args:
        systems (list): List of state-space model objects to connect.
        connections (list or dict): Connection specification:
            - List format: [(source_name, input_name), ...] for simple
                one-to-one connections where source_name is an output name
                or external input name.
            - Dict format: {input_name: source_spec, ...} where source_spec
                can be:
                - A string: 'sys2_y' (simple connection from output or input)
                - A list: ['sys2_y', 'sys3_y'] (sum with unit gains)
                - A dict: {'sys2_y': 1.0, 'sys3_y': -0.5} (weighted sum)

            Source names can be either:
                - Output names from any of the systems in the parallel
                  connection
                - External input names (inputs that are NOT themselves
                  connected)

            Note: If an input name appears as a source, it must be an
            external input (not connected). This prevents circular
            dependencies.
        model_class: Target model class (StateSpaceModelCT or
            StateSpaceModelDT). The _attr_names class attribute determines
            the naming conventions.
        input_names (list, optional): List of external input signal names to
            expose. If None, exposes all inputs that are not connected.
        output_names (list, optional): List of external output signal names
            to expose. If None, exposes all outputs.
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
        >>> from cas_models.continuous_time.models import (
        ...     StateSpaceModelCT,
        ...     ATTR_NAMES,
        ... )
        >>> sys1 = StateSpaceModelCT(...)  # Plant
        >>> sys2 = StateSpaceModelCT(...)  # Controller
        >>>
        >>> # Connect output of sys2 to input of sys1, and vice versa
        >>> connected = connect_nonlinear_systems(
        ...     [sys1, sys2],
        ...     connections=[("sys2_y", "sys1_u"), ("sys1_y", "sys2_u")],
        ...     attr_names=ATTR_NAMES,
        ...     model_class=StateSpaceModelCT,
        ... )
        >>>
        >>> # Example 2: Summing junction with feedback
        >>> # sys3 output feeds into sys1, with feedback from sys1 and sys2
        >>> connected = connect_nonlinear_systems(
        ...     [sys1, sys2, sys3],
        ...     connections={
        ...         "sys1_u": {"sys2_y": 1.0, "sys3_y": -0.5},  # Weighted sum
        ...         "sys2_u": "sys1_y",  # Simple connection
        ...     },
        ...     attr_names=ATTR_NAMES,
        ...     model_class=StateSpaceModelCT,
        ...     input_names=["sys3_u"],  # Only sys3 input is external
        ...     output_names=["sys1_y", "sys2_y"],  # Expose these outputs
        ... )
        >>>
        >>> # Example 3: Closed-loop system (no external inputs)
        >>> connected = connect_nonlinear_systems(
        ...     [sys1, sys2],
        ...     connections={"sys1_u": "sys2_y", "sys2_u": "sys1_y"},
        ...     attr_names=ATTR_NAMES,
        ...     model_class=StateSpaceModelCT,
        ...     input_names=[],  # No external inputs
        ...     output_names=["sys1_y"],
        ... )
        >>>
        >>> # Example 4: Input-to-input connections (e.g., tank network)
        >>> # Connect one external input to multiple system inputs
        >>> connected = connect_nonlinear_systems(
        ...     [sys1, sys2, sys3],
        ...     connections={
        ...         # sys1_u gets sum of sys2_u and sys3_u
        ...         "sys1_u": ["sys2_u", "sys3_u"],
        ...         # sys2 has another input from sys1 output
        ...         "sys2_v": "sys1_y",
        ...     },
        ...     attr_names=ATTR_NAMES,
        ...     model_class=StateSpaceModelCT,
        ...     input_names=["sys2_u", "sys3_u"],  # These must be external
        ...     output_names=["sys3_y"],
        ... )

    Note:
        - All input/output names must be unique (enforced by
            concatenate_lists_of_names)
        - For discrete-time systems, all must have the same dt value
        - Connections must not create algebraic loops (no automatic detection)

    See Also:
        connect_nonlinear_systems_in_parallel: Combine systems without
            connections
        connect_nonlinear_systems_in_series: Connect systems in series
    """
    attr_names = model_class._attr_names

    # Step 1: Create parallel system
    parallel_sys = connect_systems_in_parallel(
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

    # Step 2: Normalize connections and determine external inputs
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
        external_inputs = input_names
        # Validate no overlap with connected inputs
        overlap = set(external_inputs) & set(connections_norm.keys())
        if overlap:
            raise ValueError(
                f"Inputs cannot be both external and connected. "
                f"Found in both: {overlap}"
            )
        # Names not in the parallel system are treated as new external signals
        # (e.g. a setpoint injected into a summing junction). They are valid
        # as connection sources but not as connection targets.

    _validate_connections(connections_norm, parallel_sys, external_inputs)

    if output_names is None:
        external_outputs = parallel_sys.output_names  # All outputs
    else:
        external_outputs = output_names
        # Validate all exist
        for name in external_outputs:
            if name not in parallel_sys.output_names:
                raise ValueError(
                    f"External output '{name}' not found in parallel system. "
                    f"Available: {parallel_sys.output_names}"
                )

    # Validate that input names in connection sources are external
    for input_name, source_dict in connections_norm.items():
        for source_name in source_dict.keys():
            if source_name in parallel_sys.input_names:
                # Source is an input name - must be external
                if source_name not in external_inputs:
                    raise ValueError(
                        f"Connection source input '{source_name}' (for input "
                        f"'{input_name}') must be an external input (not "
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

    # Extract selected external outputs
    y_ext = _extract_outputs(y_parallel, external_outputs, parallel_sys)

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
        model_class: Class used to instantiate the returned model (e.g.
            StateSpaceModelCT or StateSpaceModelDT).  Required.
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
            ``"fbk {sys1_key}"`` for unity/scalar feedback, or
            ``"fbk {sys1_key} {sys2_key}"`` for a dynamic feedback system.
        sep: Separator character used when joining keys with names (default
            ``"_"``).  Reserved for future use; not yet applied everywhere.

    Returns:
        A new state-space model of type ``model_class`` representing the
        closed-loop system.

    Raises:
        ValueError: If ``model_class`` is None, if sys2 dimension mismatches
            sys1, or if an algebraic loop is detected.
        NotImplementedError: If a scalar ``sys2`` is passed with a
            discrete-time ``model_class``.
    """
    if model_class is None:
        raise ValueError("model_class must be specified")

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
        sys2_out_names = (
            sys2.output_names if has_sys2_states else sys1.output_names
        )
        output_names = concatenate_lists_of_names(
            [sys1.output_names, sys2_out_names],
            keys=effective_keys[:2],
            verbose_names=verbose_names,
        )[:ny1]
    state_names = concatenate_lists_of_names(
        [sys1.state_names, sys2.state_names if has_sys2_states else []],
        keys=effective_keys[:2],
        verbose_names=verbose_names,
    )
    if name is None:
        name_keys = (
            [effective_keys[0]] if unity_feedback else effective_keys[:2]
        )
        name = " ".join(["fbk"] + name_keys)

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
