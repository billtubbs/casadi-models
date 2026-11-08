"""Recording how models are made, so that cas_models.yaml_utils can save
a model as the class or connection function that made it and their
arguments, to any depth, instead of its CasADi functions.

Every model made by a model class's own __init__ (a subclass of
StateSpaceModelCT or StateSpaceModelDT that defines one, e.g.
StateSpaceModelDTARXSISO(na=2, nb=1)) or by a connection function of
cas_models.transformations (connect_systems, hstack_systems, ..., and
so the operators *, +, - that call them) has a recipe, model._recipe:

    {"class": StateSpaceModelDTARXSISO, "args": {"na": 2, "nb": 1}}
    {"function": "connect_systems", "args": {"systems": [m1, m2], ...}}

args are the arguments given (by name; defaults not given aren't
there), with lists, tuples and dicts copied so that changing them
afterwards doesn't change the recipe. Models among them are the models
themselves (references, not copies): models aren't meant to be changed
once made. A model made by a base class's __init__ (a model of
functions) has none and is saved from its fields.

Functions
---------
record_init
    Wrap a model class's __init__ to record the recipe.
records
    Decorate a connection function to record the recipe.
recipe
    The recipe of a model, or None.
"""

import functools
import inspect

RECIPE = "_recipe"


def _copy_containers(value):
    """value with its lists, tuples and dicts copied (recursively), the
    rest (models, CasADi objects, ...) as they are."""
    if isinstance(value, list):
        return [_copy_containers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(v) for v in value)
    if isinstance(value, dict):
        return {k: _copy_containers(v) for k, v in value.items()}
    return value


def _named_args(func, args, kwargs, skip_first=False):
    """The arguments of a call func(*args, **kwargs) by name (**kwargs
    merged in), or None if it has *args (positional arguments can't be
    named)."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    named = {}
    for i, (name, value) in enumerate(bound.arguments.items()):
        if skip_first and i == 0:
            continue
        kind = sig.parameters[name].kind
        if kind is inspect.Parameter.VAR_POSITIONAL:
            if value:
                return None
        elif kind is inspect.Parameter.VAR_KEYWORD:
            named.update(value)
        else:
            named[name] = value
    return _copy_containers(named)


def record_init(init):
    """init (a model class's __init__) recording the recipe of the
    model it makes: the class of the model and the arguments of the
    outermost __init__ (a subclass's __init__ calling its base's records
    its own arguments, not the base's)."""

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        outermost = RECIPE not in self.__dict__
        if outermost:
            self.__dict__[RECIPE] = None  # inner __init__s don't record
        init(self, *args, **kwargs)
        if outermost:
            named = _named_args(init, (self, *args), kwargs, skip_first=True)
            self.__dict__[RECIPE] = (
                None if named is None else {"class": type(self), "args": named}
            )

    wrapper._records_recipe = True
    return wrapper


def records(func):
    """func (a connection function returning a model) recording the
    model's recipe: the function's name and the arguments given."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        named = _named_args(func, args, kwargs)
        model.__dict__[RECIPE] = (
            None if named is None else {"function": func.__name__, "args": named}
        )
        return model

    return wrapper


def recipe(model):
    """model's recipe (see the module docstring), or None."""
    return model.__dict__.get(RECIPE)


def record_subclass_init(cls):
    """For a model base class's __init_subclass__: wrap cls's own
    __init__, if it defines one, with record_init()."""
    init = cls.__dict__.get("__init__")
    if init is not None and not getattr(init, "_records_recipe", False):
        cls.__init__ = record_init(init)
