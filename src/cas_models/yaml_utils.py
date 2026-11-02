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
base_save_args
    The base-class form of a model, for _save_args.
base_from_saved_args
    Rebuild a base-class model, for _from_saved_args.
write_tsv
    Write a matrix to a tab-separated text file.
read_tsv
    Read a matrix from a tab-separated text file.
"""

import inspect
from dataclasses import fields
from pathlib import Path

import casadi as cas
import numpy as np
import yaml

FORMAT = 1
MODEL_FILE = "model.yaml"
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
        for row in value:
            f.write("\t".join(repr(float(v)) for v in row) + "\n")
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
    """A YAML value for an argument, writing data files to folder."""
    if isinstance(value, cas.Function):
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
        if ncol == 1 and nrow <= MAX_INLINE_VECTOR:
            return {"DM": [float(v) for v in matrix[:, 0]]}
        if (nrow == 1 and ncol <= MAX_INLINE_VECTOR) or (
            nrow <= MAX_INLINE_MATRIX and ncol <= MAX_INLINE_MATRIX
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


def _decode(value, folder):
    """The argument from a YAML value, reading data files from folder."""
    if _is_file_ref(value):
        file = folder / value["file"]
        if file.suffix == ".casadi":
            return cas.Function.load(str(file))
        return read_tsv(file, value["shape"])
    if _is_inline_matrix(value):
        if "shape" in value:
            return cas.DM(*value["shape"])
        return cas.DM(value["DM"])
    if isinstance(value, dict):
        return {k: _decode(v, folder) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, folder) for v in value]
    return value


def _read_model_file(folder):
    path = folder / MODEL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder}: no {MODEL_FILE}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or data.get("cas_models_format") != FORMAT:
        found = (
            data.get("cas_models_format") if isinstance(data, dict) else None
        )
        raise ValueError(f"{path}: cas_models_format {found!r}, not {FORMAT}")
    return data


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
    cls, args = model._save_args()
    args = _without_defaults(cls, args)
    data = {"cas_models_format": FORMAT, "class": cls.__name__}
    if args:
        data["args"] = {k: _encode(v, folder, k) for k, v in args.items()}
    with open(folder / MODEL_FILE, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)


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


def _find_class(name, bases):
    for base in bases:
        for cls in _model_classes(base):
            if cls.__name__ == name:
                return cls
    raise ValueError(
        f"class {name!r} isn't {' or '.join(b.__name__ for b in bases)} "
        "or one of their subclasses (import the module that defines it)"
    )


def load_model(folder, base=None):
    """Load a model saved in folder. With base, the class saved must be
    base or one of its subclasses; without, StateSpaceModelCT,
    StateSpaceModelDT or one of their subclasses."""
    folder = Path(folder)
    data = _read_model_file(folder)
    if base is None:
        from cas_models.continuous_time.models import StateSpaceModelCT
        from cas_models.discrete_time.models import StateSpaceModelDT

        bases = [StateSpaceModelCT, StateSpaceModelDT]
    else:
        bases = [base]
    cls = _find_class(data["class"], bases)
    args = data.get("args") or {}
    args = {k: _decode(v, folder) for k, v in args.items()}
    return cls._from_saved_args(args)


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
    """A cls model from the arguments of base_save_args(), with new
    symbols for the parameters."""
    args = dict(args)
    args["params"] = {
        name: cas.SX.sym(name, *shape)
        for name, shape in (args.get("params") or {}).items()
    }
    return cls(**args)
