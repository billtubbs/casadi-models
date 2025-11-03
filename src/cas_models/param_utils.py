from collections import defaultdict
from itertools import chain


def concatenate_lists_of_names(lists_of_names, keys=None, prefix="sys"):
    if keys is None:
        keys = [f"{prefix}{i + 1}" for i in range(len(lists_of_names))]
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
    >>> params1 = {'K': K, 'T1': T1_1}
    >>> params2 = {'K': K, 'T1': T1_2, 'T2': T2_2}
    >>> merge_param_dicts([params1, params2], keys=['sys1', 'sys2'])
    {'K': SX(K), 'sys1_T1': SX(T1_1), 'sys2_T1': SX(T1_2), 'T2': SX(T2_2)}
    >>> merge_param_dicts([params1, params2], keys=['sys1', 'sys2'], verbose_names=True)
    {'sys1_sys2_K': SX(K),
     'sys1_T1': SX(T1_1),
     'sys2_T1': SX(T1_2),
     'sys2_T2': SX(T2_2)}
    """
    # TODO: This function is horrendous.  Surely there is a simpler way.

    if keys is None:
        keys = [f"{prefix}{i + 1}" for i in range(len(list_of_dicts))]
    elif len(list_of_dicts) > len(set(keys)):
        raise ValueError("not enough unique keys")

    # Identify each unique parameter and where it is used
    keys_for_each_param = defaultdict(list)
    for i, d in enumerate(list_of_dicts):
        for sub_key, param in d.items():
            keys_for_each_param[param].append((i, sub_key))

    # Check for keys which are used by more than one parameter
    params_for_each_key = defaultdict(set)
    for param, keys_by_group in keys_for_each_param.items():
        unique_keys = set(v for k, v in keys_by_group)
        if len(unique_keys) == 1:
            if verbose_names:
                # Create a composite parameter name
                comp_names = '_'.join(sorted(keys[i] for i, _ in keys_by_group))
                new_key = f"{comp_names}_{unique_keys.pop()}"
            else:
                # Create a generic parameter name
                new_key = unique_keys.pop()
        else:
            # Create a composite parameter name
            new_key = "_".join(
                chain.from_iterable(
                    sorted((keys[i], key) for i, key in keys_by_group)
                )
            )
        params_for_each_key[new_key].add(
            (tuple(keys[i] for i, _ in keys_by_group), param)
        )

    merged_params = {}
    for key, params in params_for_each_key.items():
        if len(params) > 1:
            # Create composite parameter names if needed
            # Note: params is a set of tuples so needs to be sorted
            for groups, param in sorted(params):
                new_key = f"{'_'.join(groups)}_{key}"
                merged_params[new_key] = param
        else:
            _, param = params.pop()
            merged_params[key] = param

    # TODO: Is this needed?
    assert len(merged_params) == len(keys_for_each_param)

    return merged_params


def make_symbolic_vars_from_kwargs(**kwargs):
    out_vars = {}
    for key, value in kwargs.items():
        if value is None:
            out_vars[key] = cas.SX.sym(key)
        elif isinstance(value, str):
            out_vars[key] = cas.SX.sym(value)
        else:
            out_vars[key] = value
    return out_vars
