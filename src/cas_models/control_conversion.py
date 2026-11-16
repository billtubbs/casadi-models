"""Conversion between linear cas_models models and python-control
systems, for python-control's linear analysis (minreal, balred, poles,
zeros, bode, dcgain, lqr, ...):

    from cas_models.control_conversion import from_control, to_control

    sys_min = from_control(to_control(sys).minreal())

Plain functions: the models from_control() returns are saved by their
matrices (StateSpaceModelCTFromABCD, StateSpaceModelDTFromABCD), not as
a recipe that redoes the python-control calls when loaded (numerical
results such as a minimal realization's states can change with library
versions).

Names: to_control() gives the python-control system the model's name
and input, output and state names. python-control's own functions
mostly don't keep names (minreal, balred and feedback return sys[k],
u[i], y[i], x[i]); from_control() keeps the names a system has, except
those generic ones, which become cas_models' defaults, and takes names
given as arguments instead (e.g. the original model's input and output
names after minreal).

Functions
---------
to_control
    A linear model as a python-control StateSpace system.
from_control
    A python-control system as a StateSpaceModelCTFromABCD or
    StateSpaceModelDTFromABCD.
"""

import re

import control
import numpy as np

from cas_models.transformations import check_linear, linearize
from cas_models.validation import is_ss_ct

# python-control's default names: sys[k], u[i], y[i], x[i]
_GENERIC_NAME = re.compile(r"^sys\[\d*\]$")
_GENERIC_LABELS = {
    "inputs": re.compile(r"^u\[\d+\]$"),
    "outputs": re.compile(r"^y\[\d+\]$"),
    "states": re.compile(r"^x\[\d+\]$"),
}


def to_control(sys, params=None):
    """A linear model (continuous or discrete time) as a python-control
    StateSpace system: its matrices (linearize()) with its sample time
    (a discrete-time model's dt, or True if it has none; 0 for
    continuous time), name (python-control's default sys[k] if it has
    none) and input, output and state names.

    sys must be linear (checked: a nonlinear model raises ValueError,
    linearize it first). A model with parameters needs their values,
    params ({name: value}), as for linearize()."""
    A, B, C, D = linearize(sys, params=params)  # also checks sys, params
    check_linear(sys, A, B, C, D, params)
    if is_ss_ct(sys):
        dt = 0
    else:
        dt = True if sys.dt is None else sys.dt
    kwargs = {}
    if sys.name is not None:
        kwargs["name"] = sys.name
    return control.ss(
        A,
        B,
        C,
        D,
        dt,
        inputs=list(sys.input_names),
        outputs=list(sys.output_names),
        states=list(sys.state_names),
        **kwargs,
    )


def _names(labels, kind, given):
    """given if not None, else labels unless they are all
    python-control's generic ones (None: cas_models' defaults)."""
    if given is not None:
        return list(given)
    if all(_GENERIC_LABELS[kind].match(label) for label in labels):
        return None
    return list(labels)


def from_control(
    ctsys, name=None, input_names=None, state_names=None, output_names=None
):
    """A python-control system (a StateSpace, or anything control.ss()
    converts, e.g. a TransferFunction) as a StateSpaceModelCTFromABCD
    (dt 0) or StateSpaceModelDTFromABCD (dt True: dt None; else dt),
    with numeric matrices.

    name, input_names, state_names and output_names are the model's
    names; each not given is the system's, unless it's python-control's
    generic one (sys[k], u[i], y[i], x[i]: what minreal, balred, etc.
    return), which leaves cas_models' default."""
    # Imported here: the model modules import transformations
    from cas_models.continuous_time.models import StateSpaceModelCTFromABCD
    from cas_models.discrete_time.models import StateSpaceModelDTFromABCD

    if not isinstance(ctsys, control.StateSpace):
        ctsys = control.ss(ctsys)
    if ctsys.dt is None:
        raise ValueError(
            "the system's time base is unspecified (dt None): give it dt 0 "
            "(continuous time) or a sample time"
        )
    if name is None and not _GENERIC_NAME.match(ctsys.name or "sys[]"):
        name = ctsys.name
    kwargs = {
        "name": name,
        "input_names": _names(ctsys.input_labels, "inputs", input_names),
        "state_names": _names(ctsys.state_labels, "states", state_names),
        "output_names": _names(ctsys.output_labels, "outputs", output_names),
    }
    A, B, C, D = (
        np.asarray(M, dtype=float).reshape(M.shape)
        for M in (ctsys.A, ctsys.B, ctsys.C, ctsys.D)
    )
    if ctsys.dt == 0:
        return StateSpaceModelCTFromABCD(A, B, C, D, **kwargs)
    dt = None if ctsys.dt is True else ctsys.dt
    return StateSpaceModelDTFromABCD(A, B, C, D, dt=dt, **kwargs)
