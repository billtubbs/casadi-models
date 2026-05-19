import casadi as cas
from collections import defaultdict


def make_list_of_enumerated_names(prefix, n, sep=""):
    """Convenience function to create default names of vector elements.

    Example:
    >>> make_list_of_enumerated_names("x", 3)
    ['x1', 'x2', 'x3']

    """
    if n > 1:
        names = [f"{prefix}{sep}{i + 1}" for i in range(n)]
    elif n == 1:
        names = [prefix]
    else:
        names = []
    return names


def make_list_of_unique_names(keys, prefix="sys"):
    """Generate a list of unique names from keys, replacing None values and
    deduplicating repeated strings by appending a numeric suffix.

    For each key in the input list:
    - If the key is not None and unique, use it as-is
    - If the key appears more than once, append a sequential number to each
      occurrence (e.g., two "motor" keys become "motor1", "motor2")
    - If the key is None, generate a unique name using the prefix and a
      sequential number (e.g., "sys1", "sys2", etc.)

    Args:
        keys: List of keys (strings or None values)
        prefix: Prefix to use for auto-generated names (default: "sys")

    Returns:
        List of unique names (all strings, no None values)

    Examples:
        >>> make_list_of_unique_names(["plant", None, "controller"])
        ["plant", "sys1", "controller"]
        >>> make_list_of_unique_names(["motor", "motor"])
        ["motor1", "motor2"]
    """
    counts = {}
    for key in keys:
        if key is not None:
            counts[key] = counts.get(key, 0) + 1

    seen = {}
    new_names = []
    i = 0
    for key in keys:
        if key is None:
            while True:
                i += 1
                new_name = prefix + str(i)
                if new_name not in keys:
                    break
        elif counts[key] > 1:
            seen[key] = seen.get(key, 0) + 1
            new_name = key + str(seen[key])
        else:
            new_name = key
        new_names.append(new_name)
    return new_names


def concatenate_lists_of_names(
    lists_of_names, keys=None, prefix="sys", verbose_names=False
):
    """Concatenate multiple lists of names into a single unique list.

    Args:
        lists_of_names: List of lists of name strings
        keys: List of keys to use for prefixing (generated if None)
        prefix: Prefix for auto-generated keys (default: "sys")
        verbose_names: If True, always prepend keys to all names.
                      If False (default), only prepend keys to conflicting names.

    Returns:
        List of unique names

    Examples:
        >>> concatenate_lists_of_names(
        ...     [["a", "b"], ["c", "d"]], keys=["x", "y"]
        ... )
        ['a', 'b', 'c', 'd']  # verbose_names=False (default), no conflicts

        >>> concatenate_lists_of_names(
        ...     [["a", "b"], ["a", "c"]], keys=["x", "y"]
        ... )
        ['x_a', 'b', 'y_a', 'c']  # 'a' conflicts, so both get prefixed

        >>> concatenate_lists_of_names(
        ...     [["a", "b"], ["c", "d"]], keys=["x", "y"], verbose_names=True
        ... )
        ['x_a', 'x_b', 'y_c', 'y_d']  # Always prepend keys
    """
    if keys is None:
        keys = make_list_of_enumerated_names(prefix, len(lists_of_names))
    elif len(lists_of_names) > len(set(keys)):
        raise ValueError("not enough unique keys")

    if verbose_names:
        # Original behavior: always prepend keys
        names = [
            f"{key}_{name}"
            for key, names in zip(keys, lists_of_names)
            for name in names
        ]
    else:
        # New behavior: only prepend keys when there are conflicts
        # First, identify which names appear in multiple lists
        name_to_keys = defaultdict(list)
        for key, names_list in zip(keys, lists_of_names):
            for name in names_list:
                name_to_keys[name].append(key)

        # Build the result list, prepending keys only for conflicting names
        names = []
        for key, names_list in zip(keys, lists_of_names):
            for name in names_list:
                if len(name_to_keys[name]) > 1:
                    # Name appears in multiple lists - prepend key
                    names.append(f"{key}_{name}")
                else:
                    # Name is unique - keep as-is
                    names.append(name)

    if len(names) > len(set(names)):
        raise ValueError("non-unique names")
    return names


def merge_param_dicts(
    list_of_dicts, keys=None, verbose_names=False, prefix="sys"
):
    """Merges a list of parameter dictionaries into one dictionary of
    unique model variables. Note that the same symbolic variables
    (dictionary values) may be used multiple times for different
    model variables (dictionary keys). However, if different symbolic
    variables are found with the same key, these keys are renamed
    to make them unique.

    Example:
    >>> K = cas.SX.sym("K")
    >>> T1_1 = cas.SX.sym("T1_1")
    >>> T1_2 = cas.SX.sym("T1_2")
    >>> T2_2 = cas.SX.sym("T2_2")
    >>> p1 = {"K": K, "T1": T1_1}
    >>> p2 = {"K": K, "T1": T1_2, "T2": T2_2}
    >>> merge_param_dicts([p1, p2], keys=["sys1", "sys2"])
    {'K': SX(K), 'sys1_T1': SX(T1_1), 'sys2_T1': SX(T1_2), 'T2': SX(T2_2)}
    >>> merge_param_dicts([p1, p2], keys=["sys1", "sys2"], verbose_names=True)
    {'sys1_sys2_K': SX(K),
    'sys1_T1': SX(T1_1),
    'sys2_T1': SX(T1_2),
    'sys2_T2': SX(T2_2)}
    """
    if keys is None:
        keys = make_list_of_enumerated_names(prefix, len(list_of_dicts))
    if len(list_of_dicts) > len(set(keys)):
        raise ValueError("not enough unique keys")

    # Group all parameters by their original key name
    key_groups = defaultdict(list)
    for sys_key, d in zip(keys, list_of_dicts):
        for orig_key, param in d.items():
            key_groups[orig_key].append((sys_key, param))

    merged_params = {}
    for orig_key, sys_params in key_groups.items():
        # Check if all parameters with this key are identical
        unique_params = set(param for _, param in sys_params)

        if len(unique_params) == 1:
            # All parameters are the same - can use simple name
            param = unique_params.pop()
            if verbose_names:
                # Include all systems that use this parameter
                systems = "_".join(
                    sorted(sys_key for sys_key, _ in sys_params)
                )
                new_key = f"{systems}_{orig_key}"
            else:
                new_key = orig_key
            merged_params[new_key] = param
        else:
            # Different parameters share the same key - must disambiguate
            for sys_key, param in sys_params:
                new_key = f"{sys_key}_{orig_key}"
                merged_params[new_key] = param

    return merged_params


def make_symbolic_vars_from_kwargs(**kwargs):
    """Parses the arguments and replaces any None values or tuples with
    symbolic CasADi variables (SX).  If a value is None, this signifies
    a scalar symbolic variable to be substituted.  If a value is a
    tuple, the tuple signifies the shape of the symbolic array.

    This is useful when defining arbitrary models with parameters that
    the user may assign explicit values, e.g. T1=1.0, or leave as
    arbitrary values, e.g. T1=cas.SX.sym('T1').

    Examples:
    >>> make_symbolic_vars_from_kwargs(T1=None)
    {'T1': SX(T1)}
    >>> make_symbolic_vars_from_kwargs(K=None, T1=3)
    {'K': SX(K), 'T1': 3}
    >>> make_symbolic_vars_from_kwargs(p=(3, 1))
    {'p': SX([p_0, p_1, p_2])}

    """
    out_vars = {}
    for key, value in kwargs.items():
        if value is None:
            # Create a scalar symbolic variable
            out_vars[key] = cas.SX.sym(key)
        elif isinstance(value, tuple):
            # Value indicates shape of symbolic variable
            out_vars[key] = cas.SX.sym(key, value)
        else:
            out_vars[key] = value
    return out_vars
