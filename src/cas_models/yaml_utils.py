"""Saving models to folders of readable files and loading them.

model.save(folder) writes the folder with a YAML file, model.yaml, of
the model's class and the arguments that rebuild it, and
StateSpaceModelCT.load(folder) or StateSpaceModelDT.load(folder) (or
load_model(folder)) rebuilds it:

    cas_models_format: 1
    class: StateSpaceModelDT
    args:
      F: {file: F.casadi}
      H: {file: H.casadi}
      n: 2
      ...
      params: {a: [1, 1]}

Vectors with up to MAX_INLINE_VECTOR (8) elements and matrices of up to
MAX_INLINE_MATRIX x MAX_INLINE_MATRIX (5 x 5) are written in model.yaml,
a column vector as a list, a row vector as a list of one row and a
matrix as a list of rows:

      A: {DM: [-1.62, 0.65]}
      c: {DM: [[1.0, 2.5, -3.0]]}
      D: {DM: [[1.0, 0.0], [0.5, 2.0]]}

(an empty one with its shape, {DM: [], shape: [0, 3]}). Larger ones go to
tab-separated text files and CasADi Functions to .casadi files (CasADi's
serialization of the function's expression graph, not readable) in the
folder, named after the argument (<argument>.tsv or .casadi; an
argument inside another, e.g. a dict entry, is <argument>_<key>);
model.yaml refers to them by name. The text files hold one matrix row
per line (a vector is a column, one value per line), and model.yaml
gives the shape. Values are written in full precision (the shortest form
that reads back exactly) and dense: zeros in a sparse matrix are written
as values (the models' constructors sparsify their matrices). Matrices
load as CasADi DM.

model.to_yaml() (or to_yaml(model)) gives the same YAML as text, with
everything in it: matrices inline whatever their size, and functions
as their CasADi serialization (a long string), not files:

      F: {Function: jhpnnagiieahaaaadaaaa...}

StateSpaceModelCT.from_yaml(text) or StateSpaceModelDT.from_yaml(text)
(or from_yaml(text)) rebuilds the model. It also reads text that refers
to data files, e.g. a model.yaml read from a folder, given the folder
they are in.

A model can also be built by a connection function of
cas_models.transformations (block_diag_systems, hstack_systems,
vstack_systems, connect_systems_in_parallel, connect_systems_in_series,
connect_systems, connect_feedback_system, negate_system) from models
given in the same way, to any depth:

    cas_models_format: 1
    function: hstack_systems
    args:
      name: R11
      systems:
      - class: StateSpaceModelDTARXSISO
        args: {na: 2, nb: 2, dt: 2.0, name: CRAC1}
      - class: StateSpaceModelDTARXSISO
        args: {na: 1, nb: 1, dt: 2.0, name: CRAC2}

Every model in the document is a mapping of a class or a function name
and its arguments, args (named; positional arguments aren't supported);
cas_models_format is only at the top. model_class is the base class of
the models connected unless given (a class name, e.g.
StateSpaceModelDT). These are written by hand: save_model() and
to_yaml() write the model made (its class and arguments), not how it
was made.

A symbolic argument is a symbol node, {symbol: name}, a scalar SX
symbol, or {symbol: name, shape: [n]} (a column vector) or [n, m] (a
matrix). It can be used wherever a value can, e.g. in a list:
poles: [{symbol: p_0}, 0.5]. The same name in a document is the same
symbol, so models that use it share the parameter:

    cas_models_format: 1
    function: vstack_systems
    args:
      systems:
      - class: SSModelCTLinearO2SISO
        args: {K: {symbol: K}, T1: {symbol: T1}, T2: 5.0}
      - class: SSModelCTLinearFOSISO
        args: {K: {symbol: K}, T1: {symbol: T1}}

The parameter is named after the symbol (its elements name_0, ... for
a vector, as CasADi names them), not the argument.

Arguments equal to their defaults in the class's __init__ (None, or a
plain number or string) are left out, and so is args if none are left
(e.g. a model with symbolic parameters, SSModelCTLinearFOSISO(), is just
its class).

The base classes save their fields, the most general form of a model:
the functions, dimensions, names, and the parameters as {name: shape}
(the loaded model has new symbols with these names and shapes). A
subclass that doesn't define its own arguments (_save_args and
_from_saved_args) is saved in this form and loads as its base class.

Functions
---------
save_model
    Save a model to a folder.
load_model
    Load a model from a folder.
to_yaml
    A model as YAML text.
from_yaml
    A model from YAML text.
base_save_args
    The base-class form of a model, for _save_args.
base_from_saved_args
    Rebuild a base-class model, for _from_saved_args.
write_tsv
    Write a matrix to a tab-separated text file.
read_tsv
    Read a matrix from a tab-separated text file.
"""

import importlib
import inspect
from dataclasses import fields
from pathlib import Path

import casadi as cas
import numpy as np
import yaml

FORMAT = 1
MODEL_FILE = "model.yaml"
# The keys that name what a model node builds
NODE_KINDS = ("class", "function")
# The functions a model node can call (cas_models.transformations): the
# ones that combine models into one
CONNECTION_FUNCTIONS = (
    "block_diag_systems",
    "hstack_systems",
    "vstack_systems",
    "connect_systems_in_parallel",
    "connect_systems_in_series",
    "connect_systems",
    "connect_feedback_system",
    "negate_system",
)
# Largest vectors (elements) and matrices (rows and columns) written in
# model.yaml
MAX_INLINE_VECTOR = 8
MAX_INLINE_MATRIX = 5


def _as_matrix(value):
    """A 2-D float array from a CasADi DM or array-like (a 1-D array is
    a column vector)."""
    if isinstance(value, cas.DM):
        value = value.full()
    value = np.asarray(value, dtype=float)
    if value.ndim < 2:
        value = value.reshape(-1, 1)
    if value.ndim != 2:
        raise ValueError(f"not a matrix (shape {value.shape})")
    return value


def write_tsv(path, value):
    """Write a matrix (CasADi DM or array-like; a 1-D array is a column
    vector) to a tab-separated text file, one row per line, each value
    in the shortest form that reads back exactly. Returns its shape."""
    value = _as_matrix(value)
    with open(path, "w") as f:
        f.writelines(
            "\t".join(repr(float(v)) for v in row) + "\n" for row in value
        )
    return list(value.shape)


def read_tsv(path, shape):
    """Read a matrix written by write_tsv() as a CasADi DM of the given
    shape."""
    nrow, ncol = shape
    if nrow == 0 or ncol == 0:
        return cas.DM(nrow, ncol)
    value = np.loadtxt(path, delimiter="\t", ndmin=2)
    if value.shape != (nrow, ncol):
        raise ValueError(
            f"{path}: {value.shape[0]} x {value.shape[1]} values, "
            f"not {nrow} x {ncol}"
        )
    return cas.DM(value)


def _encode(value, folder, key):
    """A YAML value for an argument, writing data files to folder (if
    None, everything inline: functions serialized, matrices of any
    size)."""
    if isinstance(value, cas.Function):
        if folder is None:
            return {"Function": value.serialize()}
        file = folder / f"{key}.casadi"
        value.save(str(file))
        return {"file": file.name}
    if isinstance(value, (cas.SX, cas.MX)):
        if not value.is_constant():
            raise NotImplementedError(
                f"{key}: can't save a symbolic expression ({value})"
            )
        value = cas.DM(value)
    if isinstance(value, (cas.DM, np.ndarray)):
        matrix = _as_matrix(value)
        if matrix.size == 0:
            return {"DM": [], "shape": list(matrix.shape)}
        nrow, ncol = matrix.shape
        inline = folder is None
        if ncol == 1 and (inline or nrow <= MAX_INLINE_VECTOR):
            return {"DM": [float(v) for v in matrix[:, 0]]}
        if (
            inline
            or (nrow == 1 and ncol <= MAX_INLINE_VECTOR)
            or (nrow <= MAX_INLINE_MATRIX and ncol <= MAX_INLINE_MATRIX)
        ):
            return {"DM": [[float(v) for v in row] for row in matrix]}
        file = folder / f"{key}.tsv"
        shape = write_tsv(file, value)
        return {"file": file.name, "shape": shape}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _encode(v, folder, f"{key}_{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, folder, f"{key}_{i}") for i, v in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise NotImplementedError(
        f"{key}: can't save a {type(value).__name__} ({value!r})"
    )


def _is_file_ref(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("file"), str)
        and set(value) <= {"file", "shape"}
    )


def _is_inline_matrix(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("DM"), list)
        and set(value) <= {"DM", "shape"}
    )


def _file_refs(value):
    """The data file names a YAML value refers to."""
    if _is_file_ref(value):
        return [value["file"]]
    if isinstance(value, dict):
        return [f for v in value.values() for f in _file_refs(v)]
    if isinstance(value, list):
        return [f for v in value for f in _file_refs(v)]
    return []


def _is_inline_function(value):
    return (
        isinstance(value, dict)
        and set(value) == {"Function"}
        and isinstance(value["Function"], str)
    )


def _decode(value, folder, symbols=None):
    """The argument from a YAML value, reading data files from folder.
    symbols is the document's symbol table ({name: SX}, see
    _make_symbol()), a new one if None."""
    if symbols is None:
        symbols = {}
    if _is_symbol(value):
        return _make_symbol(value, symbols)
    if _is_inline_function(value):
        return cas.Function.deserialize(value["Function"])
    if _is_file_ref(value):
        if folder is None:
            raise ValueError(
                f"the YAML refers to a data file, {value['file']}: give "
                "the folder it is in"
            )
        file = folder / value["file"]
        if file.suffix == ".casadi":
            return cas.Function.load(str(file))
        return read_tsv(file, value["shape"])
    if _is_inline_matrix(value):
        if "shape" in value:
            return cas.DM(*value["shape"])
        return cas.DM(value["DM"])
    if _is_model(value):
        return _make_model(value, folder, _base_classes(), symbols)
    if isinstance(value, dict):
        return {k: _decode(v, folder, symbols) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, folder, symbols) for v in value]
    return value


def _is_symbol(value):
    """Whether a YAML value is a symbol node, {symbol: name} or {symbol:
    name, shape: [...]}. A mapping with a symbol name and other keys is
    an error."""
    if not (isinstance(value, dict) and isinstance(value.get("symbol"), str)):
        return False
    extra = set(value) - {"symbol", "shape"}
    if extra:
        raise ValueError(
            f"symbol {value['symbol']}: a symbol has only symbol and "
            f"shape, not {', '.join(sorted(extra))}"
        )
    return True


def _make_symbol(value, symbols):
    """The SX symbol of a symbol node: {symbol: name} a scalar, {symbol:
    name, shape: [n]} a column vector, {symbol: name, shape: [n, m]} a
    matrix. The same name in a document is the same symbol (from
    symbols, the document's table), so the models that use it share the
    parameter; its shape must be the same everywhere."""
    name = value["symbol"]
    if not name.isidentifier():
        raise ValueError(
            f"symbol {name!r}: a name must be a valid identifier"
        )
    shape = value.get("shape", [1, 1])
    if (
        not isinstance(shape, list)
        or not 1 <= len(shape) <= 2
        or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 1
            for d in shape
        )
    ):
        raise ValueError(
            f"symbol {name}: shape must be a list of one or two positive "
            f"integers ([n] or [n, m]), not {shape!r}"
        )
    shape = tuple(shape) if len(shape) == 2 else (shape[0], 1)
    if name in symbols:
        if symbols[name].shape != shape:
            raise ValueError(
                f"symbol {name}: shape {list(shape)} here, "
                f"{list(symbols[name].shape)} where it was first used"
            )
        return symbols[name]
    symbols[name] = cas.SX.sym(name, *shape)
    return symbols[name]


def _model_kind(value):
    """"class" or "function" if a YAML value is a model node ({class:
    ..., args: ...} or {function: ..., args: ...}, args optional), else
    None. A mapping with a class or function name and other keys is an
    error (e.g. cas_models_format, which is only at the top)."""
    if not isinstance(value, dict):
        return None
    kinds = [k for k in NODE_KINDS if isinstance(value.get(k), str)]
    if not kinds:
        return None
    if len(kinds) > 1:
        raise ValueError(
            f"a model has a class or a function, not both ({value})"
        )
    extra = set(value) - {kinds[0], "args"}
    if extra:
        raise ValueError(
            f"{kinds[0]} {value[kinds[0]]}: a model has only "
            f"{kinds[0]} and args, not {', '.join(sorted(extra))} "
            "(cas_models_format only at the top of the document)"
        )
    return kinds[0]


def _is_model(value):
    """Whether a YAML value is a model node, e.g. an argument that is a
    model (see _model_kind())."""
    return _model_kind(value) is not None


def _read_model_file(folder):
    path = folder / MODEL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder}: no {MODEL_FILE}")
    with open(path) as f:
        return _check_format(yaml.safe_load(f), path)


def _check_format(data, where):
    """The model node of data (a parsed model YAML document: its
    cas_models_format and a class or function with its args), if it has
    this format and is a model."""
    if not isinstance(data, dict) or data.get("cas_models_format") != FORMAT:
        found = (
            data.get("cas_models_format") if isinstance(data, dict) else None
        )
        raise ValueError(f"{where}: cas_models_format {found!r}, not {FORMAT}")
    node = {k: v for k, v in data.items() if k != "cas_models_format"}
    if not _is_model(node):
        raise ValueError(
            f"{where}: no model (a class or a function name, with its args)"
        )
    return node


def _model_data(model, folder):
    """The model's YAML document as a dict, writing its data files to
    folder (None: everything inline)."""
    cls, args = model._save_args()
    args = _without_defaults(cls, args)
    data = {"cas_models_format": FORMAT, "class": cls.__name__}
    if args:
        data["args"] = {k: _encode(v, folder, k) for k, v in args.items()}
    return data


def _dump(data, stream=None):
    return yaml.safe_dump(
        data, stream, sort_keys=False, default_flow_style=None
    )


def save_model(model, folder, overwrite=False):
    """Save model to folder (created if needed): model.yaml and its data
    files. If the folder isn't empty, overwrite=True is needed: it then
    deletes the files of the model saved there (model.yaml and the files
    it refers to) and leaves any others."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{folder} isn't empty (save with overwrite=True)"
            )
        if (folder / MODEL_FILE).is_file():
            old = _read_model_file(folder)
            for name in _file_refs(old.get("args", {})):
                (folder / name).unlink(missing_ok=True)
            (folder / MODEL_FILE).unlink()
    folder.mkdir(parents=True, exist_ok=True)
    data = _model_data(model, folder)
    with open(folder / MODEL_FILE, "w") as f:
        _dump(data, f)


def to_yaml(model):
    """The model as YAML text: what save_model() writes to model.yaml,
    with everything inline (functions as their CasADi serialization,
    matrices of any size) so that the text alone rebuilds it."""
    return _dump(_model_data(model, None))


def _is_default(value, default):
    """Whether value is the plain default (None, a number or a string;
    other defaults are never left out)."""
    if default is None:
        return value is None
    plain = (bool, int, float, str)
    return (
        isinstance(default, plain)
        and type(value) is type(default)
        and value == default
    )


def _without_defaults(cls, args):
    """args without the arguments equal to their defaults in
    cls.__init__."""
    parameters = inspect.signature(cls.__init__).parameters
    return {
        k: v
        for k, v in args.items()
        if not (
            k in parameters
            and parameters[k].default is not inspect.Parameter.empty
            and _is_default(v, parameters[k].default)
        )
    }


def _model_classes(base):
    """base and all its subclasses (of the modules imported)."""
    classes = [base]
    for sub in base.__subclasses__():
        classes.extend(_model_classes(sub))
    return classes


# The modules with model classes, imported before looking up a class by
# name (a class is found among its base's subclasses only once its
# module is imported)
MODEL_MODULES = [
    "cas_models.continuous_time.models",
    "cas_models.continuous_time.regulators",
    "cas_models.discrete_time.models",
]


def _find_class(name, bases):
    for module in MODEL_MODULES:
        importlib.import_module(module)
    for base in bases:
        for cls in _model_classes(base):
            if cls.__name__ == name:
                return cls
    raise ValueError(
        f"class {name!r} isn't {' or '.join(b.__name__ for b in bases)} "
        "or one of their subclasses (import the module that defines it)"
    )


def load_model(folder, base=None):
    """Load a model saved in folder (its model.yaml: a class and its
    arguments, or a connection function and its arguments, see the
    module docstring). With base, the model must be base or one of its
    subclasses; without, StateSpaceModelCT, StateSpaceModelDT or one of
    their subclasses."""
    folder = Path(folder)
    data = _read_model_file(folder)
    bases = _base_classes() if base is None else [base]
    return _make_model(data, folder, bases)


def from_yaml(text, base=None, folder=None):
    """A model from YAML text (as to_yaml() or save_model() write it, or
    written by hand: a class and its arguments, or a connection function
    and its arguments, see the module docstring). With base, the model
    must be base or one of its subclasses; without,
    StateSpaceModelCT, StateSpaceModelDT or one of their subclasses.
    folder is where the data files the text refers to are, if any (e.g.
    the text of a saved model.yaml)."""
    data = _check_format(yaml.safe_load(text), "YAML text")
    bases = _base_classes() if base is None else [base]
    return _make_model(data, None if folder is None else Path(folder), bases)


def _base_classes():
    from cas_models.continuous_time.models import StateSpaceModelCT
    from cas_models.discrete_time.models import StateSpaceModelDT

    return [StateSpaceModelCT, StateSpaceModelDT]


def _make_model(data, folder, bases, symbols=None):
    """The model of a model node, {class: ..., args: ...} or {function:
    ..., args: ...} (args optional), its class one of bases or their
    subclasses, reading data files from folder. symbols is the
    document's symbol table (a new one if None)."""
    if symbols is None:
        symbols = {}
    args = data.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError(
            f"args must be a mapping of argument names to values, not "
            f"{type(args).__name__} (positional arguments aren't supported)"
        )
    if "function" in data:
        return _call_function(data["function"], args, folder, bases, symbols)
    cls = _find_class(data["class"], bases)
    args = {k: _decode(v, folder, symbols) for k, v in args.items()}
    return cls._from_saved_args(args)


def _call_function(name, args, folder, bases, symbols):
    """The model a connection function (CONNECTION_FUNCTIONS) makes from
    args. model_class, if given, is a class name; if not, the function
    takes it from the models it connects (see
    cas_models.transformations.infer_model_class()). The result must be
    one of bases or their subclasses."""
    from cas_models import transformations

    if name not in CONNECTION_FUNCTIONS:
        raise ValueError(
            f"function {name!r} isn't one of {', '.join(CONNECTION_FUNCTIONS)}"
        )
    args = {k: _decode(v, folder, symbols) for k, v in args.items()}
    if args.get("model_class") is not None:
        args["model_class"] = _find_class(args["model_class"], _base_classes())
    model = getattr(transformations, name)(**args)
    if not isinstance(model, tuple(bases)):
        raise ValueError(
            f"function {name} makes a {type(model).__name__}, which isn't "
            f"{' or '.join(b.__name__ for b in bases)} or one of their "
            "subclasses"
        )
    return model


def base_save_args(model, base):
    """The base-class form of model: base and its fields as arguments,
    with the parameters as {name: shape}."""
    args = {f.name: getattr(model, f.name) for f in fields(base)}
    for name, p in model.params.items():
        if not (isinstance(p, cas.SX) and p.is_symbolic()):
            raise NotImplementedError(
                f"params: {name} isn't an SX symbol ({p!r})"
            )
    args["params"] = {name: list(p.shape) for name, p in model.params.items()}
    if not args["params"]:
        args["params"] = None  # the default
    return base, args


def base_from_saved_args(cls, args):
    """A cls model from its saved arguments: the arguments of its
    __init__, with new symbols for params if given ({name: shape}, as
    base_save_args() writes them)."""
    args = dict(args)
    if args.get("params"):
        args["params"] = {
            name: cas.SX.sym(name, *shape)
            for name, shape in args["params"].items()
        }
    return cls(**args)
