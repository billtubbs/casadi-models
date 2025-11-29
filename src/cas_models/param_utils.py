import casadi as cas
from collections import defaultdict


def make_list_of_enumerated_names(prefix, n, sep=""):
    """Convenience function to create default names of vector elements.

    Example:
    >>> make_list_of_enumerated_names('x', 3)
    ['x1', 'x2', 'x3']

    """
    if n > 1:
        names = [f"{prefix}{sep}{i + 1}" for i in range(n)]
    else:
        names = [prefix]
    return names


def make_list_of_unique_names(keys, prefix="sys"):
    new_names = []
    i = 0
    for key in keys:
        if key is None:
            while True:
                i += 1
                new_name = prefix + str(i)
                if new_name not in keys:
                    break
        else:
            new_name = key
        new_names.append(new_name)
    return new_names


def concatenate_lists_of_names(lists_of_names, keys=None, prefix="sys"):
    if keys is None:
        keys = make_list_of_enumerated_names(prefix, len(lists_of_names))
    elif len(lists_of_names) > len(set(keys)):
        raise ValueError("not enough unique keys")
    names = [
        f"{key}_{name}"
        for key, names in zip(keys, lists_of_names)
        for name in names
    ]
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
    >>> K = cas.SX.sym('K')
    >>> T1_1 = cas.SX.sym('T1_1')
    >>> T1_2 = cas.SX.sym('T1_2')
    >>> T2_2 = cas.SX.sym('T2_2')
    >>> p1 = {'K': K, 'T1': T1_1}
    >>> p2 = {'K': K, 'T1': T1_2, 'T2': T2_2}
    >>> merge_param_dicts([p1, p2], keys=['sys1', 'sys2'])
    {'K': SX(K), 'sys1_T1': SX(T1_1), 'sys2_T1': SX(T1_2), 'T2': SX(T2_2)}
    >>> merge_param_dicts([p1, p2], keys=['sys1', 'sys2'], verbose_names=True)
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
