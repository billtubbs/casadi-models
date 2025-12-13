"""Transformations for combining state-space models.

This module provides generalized functions for combining both
continuous-time and discrete-time state-space models.
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


def connect_nonlinear_systems_in_parallel(
    systems,
    attr_names,
    model_class,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine a collection of nonlinear systems into one large parallel
    system.

    This function works for both continuous-time and discrete-time systems by
    using an attr_names dictionary to specify the appropriate attribute and
    variable names.

    Args:
        systems (list): List of state-space model objects to combine in
            parallel.
        attr_names (dict): Dictionary specifying naming conventions with keys:
            - 'state_func': Attribute name for state function ('f' or 'F')
            - 'output_func': Attribute name for output function ('h' or 'H')
            - 'state_var': Variable name for state ('x' or 'xk')
            - 'input_var': Variable name for input ('u' or 'uk')
            - 'state_output': Output name for state function ('rhs' or 'xkp1')
            - 'output_var': Output name for output function ('y' or 'yk')
        model_class: Class to instantiate for the combined system
            (StateSpaceModelCT or StateSpaceModelDT)
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
        >>> attr_names_ct = {
        ...     "state_func": "f",
        ...     "output_func": "h",
        ...     "state_var": "x",
        ...     "input_var": "u",
        ...     "state_output": "rhs",
        ...     "output_var": "y",
        ... }
        >>> combined = connect_nonlinear_systems_in_parallel(
        ...     [sys1, sys2], attr_names_ct, StateSpaceModelCT
        ... )
        >>>
        >>> # For discrete-time models:
        >>> attr_names_dt = {
        ...     "state_func": "F",
        ...     "output_func": "H",
        ...     "state_var": "xk",
        ...     "input_var": "uk",
        ...     "state_output": "xkp1",
        ...     "output_var": "yk",
        ... }
        >>> combined = connect_nonlinear_systems_in_parallel(
        ...     [sys1, sys2], attr_names_dt, StateSpaceModelDT
        ... )
    """
    validate_systems_are_compatible(systems)

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


def connect_nonlinear_systems_in_series(
    systems,
    attr_names,
    model_class,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine a series of non-linear systems by connecting their inputs and
    outputs in series.

    This function works for both continuous-time and discrete-time systems by
    using an attr_names dictionary to specify the appropriate attribute and
    variable names.

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
        model_class: Class to instantiate for the combined system
            (StateSpaceModelCT or StateSpaceModelDT)
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
        >>> attr_names_ct = {
        ...     "state_func": "f",
        ...     "output_func": "h",
        ...     "state_var": "x",
        ...     "input_var": "u",
        ...     "state_output": "rhs",
        ...     "output_var": "y",
        ... }
        >>> combined = connect_nonlinear_systems_in_series(
        ...     [sys1, sys2], attr_names_ct, StateSpaceModelCT
        ... )
    """
    validate_systems_are_compatible(systems)
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
        combined_system = model_class(
            state_function, output_function, n, nu, ny, params=params,
            input_names=combined_system.input_names,
            output_names=sys2.output_names
        )

    combined_system.state_names = concatenate_lists_of_names(
        list(reversed(state_name_lists)), keys=list(reversed(keys))
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
    """Check all systems are compatible. For this, they must all be continuous-time
    systems or all discrete-time systems with equal time intervals, dt.
    """
    dt_systems = [is_ss_dt(sys) for sys in systems]
    all_same = all(dt_systems) or not any(dt_systems)
    if not all_same:
        raise ValueError("Cannot combine discrete time and continuous time systems")
    
    # Check discrete-time systems have same time interval
    if dt_systems[0]:
        validate_equal_dt(systems)


def _validate_connections(connections_norm, parallel_sys):
    """Validate connection specification.

    Args:
        connections_norm: Normalized connections dict
        parallel_sys: The parallel system

    Raises:
        ValueError: If connections are invalid
    """
    # Check all input names exist
    for input_name in connections_norm.keys():
        if input_name not in parallel_sys.input_names:
            raise ValueError(
                f"Connection target input '{input_name}' not found in "
                "parallel system. Available inputs: "
                f"{parallel_sys.input_names}"
            )

    # Check all output names exist
    for input_name, output_sources in connections_norm.items():
        for output_name in output_sources.keys():
            if output_name not in parallel_sys.output_names:
                raise ValueError(
                    f"Connection source output '{output_name}' (for "
                    f"input '{input_name}') not found in parallel system. "
                    f"Available outputs: {parallel_sys.output_names}"
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
    for_state_func,
):
    """Build the full internal input vector for the parallel system.

    Args:
        u_ext: External input vector (symbolic)
        x: State vector (symbolic)
        t: Time variable (symbolic)
        connections_norm: Normalized connections dict
        external_inputs: List of external input names
        parallel_sys: The parallel system
        attr_names: Attribute names dict (CT or DT)
        for_state_func: If True, avoid algebraic loops (for state function)

    Returns:
        cas.SX: Full internal input vector for parallel system
    """
    # Create mapping of external input names to their indices in u_ext
    external_input_map = {name: i for i, name in enumerate(external_inputs)}

    # Build the internal input vector in the order of parallel_sys.input_names
    u_internal_parts = []

    for input_name in parallel_sys.input_names:
        if input_name in external_input_map:
            # This is an external input
            idx = external_input_map[input_name]
            u_internal_parts.append(u_ext[idx])
        elif input_name in connections_norm:
            # This is a connected input - compute weighted sum of outputs
            # Need to evaluate parallel system output to get source signals
            # For state function, use a temporary zero input to avoid algebraic
            # loops. For output function, we can use the actual internal input
            # (may have feedthrough).
            if for_state_func:
                # Use zero for connected inputs when building state function
                # This avoids algebraic loops
                u_temp = cas.SX.zeros(len(parallel_sys.input_names), 1)
                # Fill in external inputs
                for temp_input_name in parallel_sys.input_names:
                    if temp_input_name in external_input_map:
                        temp_idx = parallel_sys.input_names.index(
                            temp_input_name
                        )
                        ext_idx = external_input_map[temp_input_name]
                        u_temp[temp_idx] = u_ext[ext_idx]

                # Evaluate parallel system output with this temporary input
                output_func_attr = getattr(
                    parallel_sys, attr_names["output_func"]
                )
                y_parallel = output_func_attr(
                    t, x, u_temp, *parallel_sys.params.values()
                )
            else:
                # For output function, this creates potential algebraic loop
                # We'll need to handle this carefully
                # Use a placeholder for now and substitute later
                u_temp = cas.SX.zeros(len(parallel_sys.input_names), 1)
                for temp_input_name in parallel_sys.input_names:
                    if temp_input_name in external_input_map:
                        temp_idx = parallel_sys.input_names.index(
                            temp_input_name
                        )
                        ext_idx = external_input_map[temp_input_name]
                        u_temp[temp_idx] = u_ext[ext_idx]

                output_func_attr = getattr(
                    parallel_sys, attr_names["output_func"]
                )
                y_parallel = output_func_attr(
                    t, x, u_temp, *parallel_sys.params.values()
                )

            # Compute weighted sum of connected outputs
            sum_expr = 0
            for output_name, gain in connections_norm[input_name].items():
                output_idx = parallel_sys.output_names.index(output_name)
                sum_expr += gain * y_parallel[output_idx]

            u_internal_parts.append(sum_expr)
        else:
            raise ValueError(
                f"Input '{input_name}' is neither external nor connected"
            )

    return cas.vertcat(*u_internal_parts)


def connect_nonlinear_systems(
    systems,
    connections,
    attr_names,
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
            - List format: [(output_name, input_name), ...] for simple
                one-to-one connections.
            - Dict format: {input_name: output_spec, ...} where output_spec
                can be:
                - A string: 'sys2_y' (simple connection)
                - A list: ['sys2_y', 'sys3_y'] (sum with unit gains)
                - A dict: {'sys2_y': 1.0, 'sys3_y': -0.5} (weighted sum)
        attr_names (dict): Naming conventions dict (CT or DT) with keys:
            - 'state_func', 'output_func', 'state_var', 'input_var',
            - 'state_output', 'output_var'
        model_class: Target model class (StateSpaceModelCT or
            StateSpaceModelDT).
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
    # Step 1: Create parallel system
    parallel_sys = connect_nonlinear_systems_in_parallel(
        systems,
        attr_names,
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

    # Step 2: Normalize and validate connections
    connections_norm = _normalize_connections(connections)
    _validate_connections(connections_norm, parallel_sys)

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
        # Validate all exist in parallel system
        for name in external_inputs:
            if name not in parallel_sys.input_names:
                raise ValueError(
                    f"External input '{name}' not found in parallel system. "
                    f"Available: {parallel_sys.input_names}"
                )

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

    # Step 4: Build connected state function
    t = cas.SX.sym("t")
    x = cas.SX.sym(attr_names["state_var"], parallel_sys.n)
    u_ext = cas.SX.sym(attr_names["input_var"], len(external_inputs))

    # Build internal input vector (avoiding algebraic loops for state function)
    u_internal = _build_internal_input_vector(
        u_ext,
        x,
        t,
        connections_norm,
        external_inputs,
        parallel_sys,
        attr_names,
        for_state_func=True,
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
        for_state_func=False,
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


def connect_feedback_systems(
    sys1,
    sys2,
    attr_names,
    model_class,
    sign=-1,
    input_names=None,
    output_names=None,
    keys=None,
    verbose_names=False,
    prefix="sys",
    name=None,
    sep="_",
):
    """Combine two systems with a feedback interconnection from the
    output of the second system to the input of the first.

    This function works for both continuous-time and discrete-time systems by
    using an attr_names dictionary to specify the appropriate attribute and
    variable names.
    """
    sys_comb_open_loop = connect_nonlinear_systems_in_series(
        [sys1, sys2],
        attr_names,
        model_class,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )
    if sys_comb_open_loop.nu != sys_comb_open_loop.ny:
        raise ValueError(
            "sys1 must have same number of inputs as outputs from sys2"
        )
    if input_names is None:
        input_names = [f"{name}_sp" for name in sys_comb_open_loop.output_names]
    else:
        if len(input_names) != sys_comb_open_loop.ny:
            raise ValueError(
                "Number of input names must match number of sys2 outputs"
            )
    if output_names is None:
        output_names = sys_comb_open_loop.output_names
    sys_name = sys_comb_open_loop.name
    connections = {}
    for sp_name, e_name, y_name in zip(
        input_names,
        sys_comb_open_loop.input_names,
        sys_comb_open_loop.output_names,
    ):
        connections[f"{sys_name}_{e_name}"] = {
            sp_name: 1.0,
            f"{sys_name}_{y_name}": sign,
        }
    return connect_nonlinear_systems(
        [sys_comb_open_loop],
        connections,
        attr_names,
        model_class,
        input_names=input_names,
        output_names=output_names,
        keys=keys,
        verbose_names=verbose_names,
        prefix=prefix,
        name=name,
        sep=sep,
    )
