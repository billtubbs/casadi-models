"""Pickle support for models that hold CasADi symbols (SX, MX).

CasADi pickles SX and MX objects only inside a cas.global_pickle_context()
and loads them only inside a cas.global_unpickle_context(): one
serializer for all of them, so the symbols they share stay shared.
getstate() and setstate() (the models' __getstate__ and __setstate__)
open one for the object when the caller hasn't, so pickle.dump(model,
file) works on its own. Inside a caller's context they use it, so the
symbols shared between the objects pickled in it stay shared too.

CasADi's contexts don't nest (entering one replaces the active one and
leaving it clears it), so this uses CasADi's private thread-local
context state (casadi.casadi._thread_local, checked with CasADi 3.8.2)
to find the active context and restore it afterwards.

Functions
---------
getstate
    The pickle state of an object, with its CasADi objects packed.
setstate
    Restore an object from a state made by getstate.
"""

# TODO: Reconsider this. It relies on CasADi's private context state, and
# it isn't clear what pickling models with symbolic parameters is for (a
# symbolic model is a template, cheap to rebuild from code or a recipe).
# Alternative: bar pickling models that hold SX or MX objects (a clear
# error in __getstate__) and drop this module. Fixed-parameter models
# pickle without a context already: their attributes are Functions, DM
# and plain Python values.

import pickle
from contextlib import contextmanager

import casadi as cas
from casadi import casadi as _casadi_py

_PICKLE_CTX = "casadi_pickle_ctx"
_UNPICKLE_CTX = "casadi_unpickle_ctx"


def _active_context(name):
    return getattr(_casadi_py._thread_local, name, None)


@contextmanager
def _context(name, ctx):
    """Make ctx CasADi's active (un)pickle context, restoring the
    previous one on exit."""
    previous = _active_context(name)
    setattr(_casadi_py._thread_local, name, ctx)
    try:
        yield ctx
    finally:
        setattr(_casadi_py._thread_local, name, previous)


def getstate(obj):
    """The pickle state of obj (its __dict__). Outside a CasADi pickle
    context, the __dict__ is pickled to bytes in a context of its own;
    inside one, it is returned as it is, for the caller's context to
    pack."""
    if _active_context(_PICKLE_CTX) is not None:
        return {"state": obj.__dict__}
    with _context(_PICKLE_CTX, cas.StringSerializer()):
        return {"casadi_state": pickle.dumps(obj.__dict__)}


def setstate(obj, state):
    """Restore obj's __dict__ from a state made by getstate()."""
    if "state" in state:
        # Pickled in the caller's context: its CasADi objects are
        # already loaded (in the caller's unpickle context)
        obj.__dict__.update(state["state"])
        return
    with _context(_UNPICKLE_CTX, cas.StringDeserializer("")):
        obj.__dict__.update(pickle.loads(state["casadi_state"]))
