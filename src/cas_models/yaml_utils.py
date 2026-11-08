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
connect_systems, connect_feedback_system, negate_system, and
set_param_values) from models given in the same way, to any depth:

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
StateSpaceModelDT).

save_model() and to_yaml() write how a model was made, its recipe (see
cas_models.recipes): the class and the arguments given to it, or the
connection function (also through the operators *, + and -) and its
arguments, the models among them written in the same way, to any
depth. A model made from functions (a base class's __init__), or one
with an argument that can't be written (an expression such as 2 * K,
a SymPy object, a Python function), is written as its base class and
fields instead (still exact). Tuples are written as {tuple: [...]} and
complex numbers as {complex: [re, im]}.

Parameter values: a model with its parameters symbolic is written
with them as symbols. A model with fitted values, made by
cas_models.transformations.set_param_values(model, values), is
written as that call, a function node like the connection functions:
the model's structure (from scratch) and the values,

    function: set_param_values
    args:
      sys:
        function: hstack_systems
        args: {...}
      values: {sys2_a_0: 99.0, ...}

so it loads as the model built again with those values set (by name:
the structure gives the same names).

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
    Save a model to a folder (by default, one named after the model).
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
safe_folder_name
    A model's name made safe as a folder name.
write_tsv
    Write a matrix to a tab-separated text file.
read_tsv
    Read a matrix from a tab-separated text file.
"""

import importlib
import inspect
import re
from dataclasses import fields
from pathlib import Path

import casadi as cas
import numpy as np
import yaml

from cas_models.recipes import recipe

FORMAT = 1
MODEL_FILE = "model.yaml"
# The keys that name what a model node builds
NODE_KINDS = ("class", "function")
# The functions a model node can call (cas_models.transformations): the
# ones that combine models into one, and set_param_values (a model with
# values for some of its parameters)
CONNECTION_FUNCTIONS = (
    "block_diag_systems",
    "hstack_systems",
    "vstack_systems",
    "connect_systems_in_parallel",
    "connect_systems_in_series",
    "connect_systems",
    "connect_feedback_system",
    "negate_system",
    "set_param_values",
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


class _Writer:
    """The state of writing one YAML document: the folder for data
    files (None: everything inline) and the symbols written so far
    ({name: SX}, so that one name is one symbol)."""

    def __init__(self, folder):
        self.folder = folder
        self.symbols = {}
        self.elements = {}  # element names of vector symbols: {name: vector name}

    def copy(self, folder="same"):
        other = _Writer(self.folder if folder == "same" else folder)
        other.symbols = dict(self.symbols)
        other.elements = dict(self.elements)
        return other


def _symbol_name(value):
    """(name, shape) of an SX that is one symbol, as _make_symbol() makes
    it (scalar: SX.sym(name); n x 1: SX.sym(name, n), elements name_0,
    ...; n x m: SX.sym(name, n, m)), else None (an expression, or
    symbols put together)."""
    if not isinstance(value, cas.SX) or value.numel() == 0:
        return None
    if not all(value.nz[i].is_symbolic() for i in range(value.nnz())):
        return None
    if value.nnz() != value.numel():
        return None
    if value.is_scalar():
        return value.name(), [1, 1]
    names = [value.nz[i].name() for i in range(value.nnz())]
    base = names[0][: -len("_0")] if names[0].endswith("_0") else None
    if base is None or names != [f"{base}_{i}" for i in range(len(names))]:
        return None
    nrow, ncol = value.shape
    return base, [nrow] if ncol == 1 else [nrow, ncol]


def _encode_symbol(value, writer, key):
    """A symbol node for an SX symbol; one name is one symbol in the
    document. A matrix of numbers and single symbols that isn't one
    symbol is written element by element (see _encode_elements())."""
    found = _symbol_name(value)
    if found is None:
        if value.is_scalar():
            raise NotImplementedError(
                f"{key}: can't save a symbolic expression ({value}): only "
                "single symbols (cas.SX.sym(name[, n[, m]])), numbers, and "
                "matrices of them"
            )
        return _encode_elements(value, writer, key)
    name, shape = found
    node = (
        {"symbol": name}
        if shape == [1, 1]
        else {"symbol": name, "shape": shape}
    )
    if name in writer.symbols:
        if not (
            writer.symbols[name].shape == value.shape
            and cas.is_equal(writer.symbols[name], value)
        ):
            raise ValueError(
                f"{key}: two different symbols are named {name}; they "
                "would load as one (rename one of them)"
            )
        return node
    clash = (
        name in writer.elements
        or any(f"{name}_{i}" in writer.symbols for i in range(value.numel()))
        if shape != [1, 1]
        else name in writer.elements
    )
    if clash:
        raise ValueError(
            f"{key}: symbol {name} and an element of a vector symbol "
            "(name_0, ...) would load as different symbols"
        )
    writer.symbols[name] = value
    if shape != [1, 1]:
        for i in range(value.numel()):
            writer.elements[f"{name}_{i}"] = name
    return node


def _encode_elements(value, writer, key):
    """A matrix of numbers and single symbols (e.g. [[A_0, A_1], [0,
    1]]) element by element: a list (a column vector) or a list of rows
    of numbers and symbol nodes; an element that is an expression can't
    be written (NotImplementedError)."""
    nrow, ncol = value.shape

    def element(e, k):
        if e.is_constant():
            return float(cas.DM(e))
        return _encode(e, writer, f"{key}_{k}")

    nodes = [
        [element(value[i, j], j * nrow + i) for j in range(ncol)]
        for i in range(nrow)
    ]
    return [row[0] for row in nodes] if ncol == 1 else nodes


def _encode(value, writer, key):
    """A YAML value for an argument, writing data files to writer.folder
    (if None, everything inline: functions serialized, matrices of any
    size). Models are written as model nodes, symbols as symbol nodes
    (see _Writer). writer can also be just the folder (or None)."""
    if not isinstance(writer, _Writer):
        writer = _Writer(writer)
    folder = writer.folder
    if _is_model_object(value):
        return _model_node(value, writer, key)
    if isinstance(value, type):
        return value.__name__  # e.g. a connection function's model_class
    if isinstance(value, cas.Function):
        if folder is None:
            return {"Function": value.serialize()}
        file = folder / f"{key}.casadi"
        value.save(str(file))
        return {"file": file.name}
    if isinstance(value, cas.MX) and not value.is_constant():
        raise NotImplementedError(
            f"{key}: can't save an MX expression ({value})"
        )
    if isinstance(value, cas.SX) and not value.is_constant():
        return _encode_symbol(value, writer, key)
    if isinstance(value, (cas.SX, cas.MX)):
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
        value = value.item()
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, tuple):
        return {
            "tuple": [
                _encode(v, writer, f"{key}_{i}") for i, v in enumerate(value)
            ]
        }
    if isinstance(value, dict):
        return {k: _encode(v, writer, f"{key}_{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_encode(v, writer, f"{key}_{i}") for i, v in enumerate(value)]
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


def _is_complex(value):
    return (
        isinstance(value, dict)
        and set(value) == {"complex"}
        and isinstance(value["complex"], list)
        and len(value["complex"]) == 2
    )


def _is_tuple(value):
    return (
        isinstance(value, dict)
        and set(value) == {"tuple"}
        and isinstance(value["tuple"], list)
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
    if _is_complex(value):
        return complex(*value["complex"])
    if _is_tuple(value):
        return tuple(_decode(v, folder, symbols) for v in value["tuple"])
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
        raise ValueError(f"symbol {name!r}: a name must be a valid identifier")
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
    """ "class" or "function" if a YAML value is a model node ({class:
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


def _is_model_object(value):
    from cas_models.continuous_time.models import StateSpaceModelCT
    from cas_models.discrete_time.models import StateSpaceModelDT

    return isinstance(value, (StateSpaceModelCT, StateSpaceModelDT))


def _args_node(kind, name, args, writer, key):
    """A model node {kind: name, args: ...} (args left out if empty).
    Arguments are named key_<argument> in data files (just <argument>
    at the top)."""
    node = {kind: name}
    if args:
        node["args"] = {
            k: _encode(v, writer, f"{key}_{k}" if key else k)
            for k, v in args.items()
        }
    return node


def _model_node(model, writer, key=""):
    """The model node of model: from its recipe (the connection function
    or the class that made it, and their arguments, see
    cas_models.recipes) if it has one that can be written, else its base
    class and fields (_save_args())."""
    r = recipe(model)
    if r is not None and "function" in r:
        return _args_node("function", r["function"], r["args"], writer, key)
    if r is not None:
        cls = r["class"]
        args = _without_defaults(cls, r["args"])
        try:  # a dry run: an argument may not be writable (e.g. SymPy)
            _args_node(
                "class", cls.__name__, args, writer.copy(folder=None), key
            )
        except NotImplementedError:
            pass
        else:
            return _args_node("class", cls.__name__, args, writer, key)
    cls, args = model._save_args()
    args = _without_defaults(cls, args)
    return _args_node("class", cls.__name__, args, writer, key)


def _model_data(model, folder):
    """The model's YAML document as a dict, writing its data files to
    folder (None: everything inline)."""
    return {"cas_models_format": FORMAT, **_model_node(model, _Writer(folder))}


def _dump(data, stream=None):
    """YAML text of a document: collections of plain values in flow
    style ([1, 2], {na: 2}), the rest, and the top level, in block
    style."""
    if not any(isinstance(v, (dict, list)) for v in data.values()):
        return yaml.safe_dump(data, stream, sort_keys=False)
    return yaml.safe_dump(
        data, stream, sort_keys=False, default_flow_style=None
    )


def safe_folder_name(name):
    """name with the characters a file or folder name can't have on
    Windows (the most restrictive: < > : " / \\ | ? * and control
    characters, / and \\ also being path separators) replaced by _, and
    trailing dots and spaces (which Windows drops) removed. A name that
    passes is valid on Windows, macOS and Linux."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name.rstrip(". ")


def _folder_from_name(name):
    """The folder to save a model named name in when no folder is given:
    safe_folder_name(name), with a note if it differs from name."""
    if not name:
        raise ValueError(
            "the model has no name: give the folder to save it in"
        )
    folder = safe_folder_name(name)
    if folder in ("", ".", ".."):
        raise ValueError(
            f"the model's name {name!r} can't be a folder name: give the "
            "folder to save it in"
        )
    if folder != name:
        print(f"Saving model {name!r} in folder {folder!r}")
    return Path(folder)


def save_model(model, folder=None, overwrite=False):
    """Save model to folder (created if needed; None: a folder named
    model.name in the current folder): model.yaml and its data files.
    If the folder isn't empty, overwrite=True is needed: it then
    deletes the files of the model saved there (model.yaml and the files
    it refers to) and leaves any others.
    Returns the folder (a Path)."""
    if folder is None:
        folder = _folder_from_name(model.name)
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
    return folder


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
    if args.get("params") and all(
        isinstance(shape, list) for shape in args["params"].values()
    ):
        args["params"] = {
            name: cas.SX.sym(name, *shape)
            for name, shape in args["params"].items()
        }
    return cls(**args)
