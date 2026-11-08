"""Unit tests for pickling models (src/cas_models/pickle_utils.py and the
__getstate__ and __setstate__ methods of StateSpaceModelCT and
StateSpaceModelDT)

Models don't compare equal after loading (the dataclass __eq__ compares
the CasADi Functions by identity), so assert_same_model() compares them
attribute by attribute: Functions by their signature and serialization,
CasADi matrices by their text and sparsity, and F and H (or f and h) by
evaluating them at random arguments.
"""

import copy
import pickle

import casadi as cas
import numpy as np
import pytest
from sympy import Matrix, symbols
from sympy.physics.control.lti import StateSpace

from cas_models.continuous_time.models import (
    SSModelCTDirectTransmission,
    SSModelCTFromSympySS,
    SSModelCTLinearFONoGainSISO,
    SSModelCTLinearFOSISO,
    SSModelCTLinearIntegratorSISO,
    SSModelCTLinearO2SISO,
    SSModelCTLinearO2UnderdampedSISO,
    StateSpaceModelCT,
    StateSpaceModelCTFromABCD,
    StateSpaceModelCTStaticNonlinearity,
)
from cas_models.continuous_time.regulators import (
    SSModelCTP,
    SSModelCTPIDInt,
    SSModelCTPIInt,
)
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTARXSISO,
    StateSpaceModelDTDelay,
    StateSpaceModelDTFromCT,
    StateSpaceModelDTFromCTRK4,
    StateSpaceModelDTTFSISO,
)
from cas_models.transformations import (
    connect_feedback_system,
    connect_systems,
)

CASADI_TYPES = (cas.SX, cas.MX, cas.DM)


def assert_same_value(value, loaded, name, exact=True):
    """Check an attribute of a loaded model against the original.
    Functions are compared by their serialization only if exact."""
    if isinstance(value, cas.Function):
        assert isinstance(loaded, cas.Function), name
        assert str(loaded) == str(value), name
        if exact:
            assert loaded.serialize() == value.serialize(), name
    elif isinstance(value, CASADI_TYPES):
        assert type(loaded) is type(value), name
        assert loaded.shape == value.shape, name
        assert loaded.sparsity() == value.sparsity(), name
        assert str(loaded) == str(value), name
    elif isinstance(value, dict):
        assert isinstance(loaded, dict), name
        assert list(loaded) == list(value), name
        for key in value:
            assert_same_value(
                value[key], loaded[key], f"{name}[{key!r}]", exact
            )
    elif isinstance(value, (list, tuple)):
        assert type(loaded) is type(value), name
        assert len(loaded) == len(value), name
        for i, (v, lv) in enumerate(zip(value, loaded)):
            assert_same_value(v, lv, f"{name}[{i}]", exact)
    elif isinstance(value, (StateSpaceModelCT, StateSpaceModelDT)):
        # A sub-model in a recipe (cas_models.recipes)
        assert_same_model(value, loaded, exact)
    else:
        assert loaded == value, name


def assert_same_model(model, loaded, exact=True):
    """Check a loaded model is the same as the original: class,
    attributes, the functions' results, and that the loaded model's
    symbolic attributes use its own params' symbols. Functions are
    compared by their serialization only if exact."""
    assert type(loaded) is type(model)
    assert list(vars(loaded)) == list(vars(model))
    for name, value in vars(model).items():
        assert_same_value(value, getattr(loaded, name), name, exact)

    # The state and output functions give the same results
    rng = np.random.default_rng(0)
    for func in ("state_func", "output_func"):
        f = getattr(model, model._attr_names[func])
        f_loaded = getattr(loaded, model._attr_names[func])
        # Positive arguments: time constants, and stable integrations
        args = [rng.uniform(0.5, 1.5, f.size_in(i)) for i in range(f.n_in())]
        for out, out_loaded in zip(f.call(args), f_loaded.call(args)):
            np.testing.assert_array_equal(out_loaded.full(), out.full())

    # Symbols in other attributes (e.g. num, den) are the params' symbols
    params = list(loaded.params.values())
    for name, value in vars(loaded).items():
        if name != "params" and isinstance(value, (cas.SX, cas.MX)):
            for v in cas.symvar(value):
                assert any(cas.is_equal(v, p) for p in params), (name, v)


def round_trip(obj):
    return pickle.loads(pickle.dumps(obj))


# Models to test, one function each, so every test gets new symbols


def ct_custom():
    """Nonlinear, with a vector parameter p"""
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 2)
    u = cas.SX.sym("u")
    p = cas.SX.sym("p", 2)
    rhs = cas.vertcat(-p[0] * x[0] ** 2 + u, x[0] - p[1] * x[1])
    f = cas.Function("f", [t, x, u, p], [rhs], ["t", "x", "u", "p"], ["rhs"])
    h = cas.Function("h", [t, x, u, p], [x[1]], ["t", "x", "u", "p"], ["y"])
    return StateSpaceModelCT(f, h, n=2, params={"p": p}, name="custom")


def ct_static_nonlinearity():
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 0)
    u = cas.SX.sym("u", 2)
    k = cas.SX.sym("k")
    h = cas.Function(
        "h", [t, x, u, k], [k * u[0] * u[1]], ["t", "x", "u", "k"], ["y"]
    )
    return StateSpaceModelCTStaticNonlinearity(h, nu=2, params={"k": k})


def ct_from_abcd_sparse():
    A = cas.DM([[-1.0, 0.0, 0.0], [0.5, -2.0, 0.0], [0.0, 0.0, -3.0]])
    B = cas.DM([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    C = cas.DM([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    D = cas.DM.zeros(2, 2)
    return StateSpaceModelCTFromABCD(A, B, C, D, input_names=["a", "b"])


def ct_from_sympy():
    R, L, C = symbols("R L C")
    sys = StateSpace(
        Matrix([[-R / L, -1 / L], [1 / C, 0]]),
        Matrix([[1 / L], [0]]),
        Matrix([[0, 1]]),
        Matrix([[0]]),
    )
    return SSModelCTFromSympySS(sys, name="RLC")


def ct_pi_feedback():
    plant = SSModelCTLinearFOSISO(K=1, T1=2, name="plant")
    ctrl = SSModelCTPIInt(name="ctrl")
    return connect_feedback_system(plant * ctrl, model_class=StateSpaceModelCT)


def dt_custom():
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", 2)
    uk = cas.SX.sym("uk")
    a = cas.SX.sym("a")
    xkp1 = cas.vertcat(a * xk[0] + uk, cas.sin(xk[0]) + 0.5 * xk[1])
    F = cas.Function(
        "F", [t, xk, uk, a], [xkp1], ["t", "xk", "uk", "a"], ["xkp1"]
    )
    H = cas.Function(
        "H", [t, xk, uk, a], [xk[1]], ["t", "xk", "uk", "a"], ["yk"]
    )
    return StateSpaceModelDT(F, H, n=2, dt=0.5, params={"a": a})


def dt_tf_symbolic():
    b0, a1 = cas.SX.sym("b0"), cas.SX.sym("a1")
    return StateSpaceModelDTTFSISO(
        num=cas.vertcat(0, b0), den=cas.vertcat(1, a1)
    )


def dt_arx_real_poles():
    """A an expression of the poles, as in ffdtools.models.ARXPaths"""
    p0, p1 = cas.SX.sym("p_0"), cas.SX.sym("p_1")
    return StateSpaceModelDTARXSISO(
        A=cas.vertcat(-(p0 + p1), p0 * p1), nb=2, nk=1
    )


def dt_connected_arx():
    """Two ARX paths summed, as in ffdtools.models.ARXPaths"""
    inputs = ["CRAC1", "CRAC2"]
    paths = [
        StateSpaceModelDTARXSISO(na=2, nb=2, input_name="u", output_name="y"),
        StateSpaceModelDTARXSISO(na=1, nb=1, input_name="u", output_name="y"),
    ]
    connections = {f"{name}_u": name for name in inputs}
    connections["R11"] = [f"{name}_y" for name in inputs]
    return connect_systems(
        paths,
        connections,
        StateSpaceModelDT,
        input_names=inputs,
        output_names=["R11"],
        keys=inputs,
        verbose_names=True,
        name="R11",
    )


MODELS = {
    # Continuous time
    "CT custom": ct_custom,
    "CT static nonlinearity": ct_static_nonlinearity,
    "CT from ABCD sparse": ct_from_abcd_sparse,
    "CT direct transmission": lambda: SSModelCTDirectTransmission(
        D=cas.DM([[1.0, 0.0], [0.5, 2.0]])
    ),
    "CT FO no gain": lambda: SSModelCTLinearFONoGainSISO(),
    "CT integrator": lambda: SSModelCTLinearIntegratorSISO(K=2.0),
    "CT FO": lambda: SSModelCTLinearFOSISO(),
    "CT O2 symbolic": lambda: SSModelCTLinearO2SISO(
        name="G", input_name="Q", output_name="T"
    ),
    "CT O2 fixed": lambda: SSModelCTLinearO2SISO(K=2.0, T1=3.0, T2=5.0),
    "CT O2 mixed": lambda: SSModelCTLinearO2SISO(K=2.0),
    "CT O2 underdamped": lambda: SSModelCTLinearO2UnderdampedSISO(),
    "CT from sympy": ct_from_sympy,
    "CT P": lambda: SSModelCTP(),
    "CT PI": lambda: SSModelCTPIInt(Kc=0.5, Ti=30.0),
    "CT PID": lambda: SSModelCTPIDInt(),
    "CT PI feedback": ct_pi_feedback,
    "CT sum": lambda: SSModelCTLinearFOSISO() + SSModelCTLinearFOSISO(K=2),
    # Discrete time
    "DT custom": dt_custom,
    "DT from CT RK4": lambda: StateSpaceModelDTFromCTRK4(
        SSModelCTLinearO2SISO(), 1.0
    ),
    "DT from CT cvodes": lambda: StateSpaceModelDTFromCT(
        SSModelCTLinearFOSISO(), 1.0
    ),
    "DT TF": lambda: StateSpaceModelDTTFSISO(
        num=[0.2, 0.1], den=[1, -1.4, 0.49]
    ),
    "DT TF symbolic": dt_tf_symbolic,
    "DT delay": lambda: StateSpaceModelDTDelay(
        nk=3, nu=2, G=[[1.0, 0.5], [0.0, 2.0]]
    ),
    "DT ARX symbolic": lambda: StateSpaceModelDTARXSISO(na=2, nb=2, nk=1),
    "DT ARX fixed": lambda: StateSpaceModelDTARXSISO(
        A=cas.DM([-1.2, 0.35]), B=cas.DM([0.5, 0.2]), nk=0
    ),
    "DT ARX real poles": dt_arx_real_poles,
    "DT connected ARX": dt_connected_arx,
    "DT sum": lambda: (
        StateSpaceModelDTTFSISO(num=[0, 1.0], den=[1, -0.5])
        + StateSpaceModelDTARXSISO(na=1, nb=1)
    ),
}


# Models whose functions serialize differently after loading: CasADi
# 3.8.2 adds 27 bytes to the serialization of a function with a cvodes
# integrator in each serialize-deserialize round trip, without pickle
# (the results are the same)
NOT_EXACT = {"DT from CT cvodes"}


@pytest.mark.parametrize("name", MODELS)
def test_pickle_round_trip(name):
    """pickle.dumps() and pickle.loads() without a CasADi context"""
    model = MODELS[name]()
    assert_same_model(model, round_trip(model), exact=name not in NOT_EXACT)


def test_pickle_file(tmp_path):
    model = dt_connected_arx()
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert_same_model(model, loaded)
    assert loaded.params.keys() == model.params.keys()


def test_pickle_twice():
    """A loaded model pickles again"""
    model = dt_tf_symbolic()
    assert_same_model(model, round_trip(round_trip(model)))


def test_deepcopy():
    model = dt_tf_symbolic()
    copied = copy.deepcopy(model)
    assert_same_model(model, copied)
    assert copied.F is not model.F


def test_symbols_shared_within_model():
    model = round_trip(dt_tf_symbolic())
    assert cas.depends_on(model.num, model.params["b0"])
    assert cas.depends_on(model.den, model.params["a1"])


def test_symbols_shared_between_models():
    """Models sharing a symbol keep it shared when pickled together in
    the caller's context, and not when pickled without one"""
    K = cas.SX.sym("K")
    model_a = StateSpaceModelDTTFSISO(num=cas.vertcat(0, K), den=[1, -0.5])
    model_b = StateSpaceModelDTTFSISO(num=cas.vertcat(0, K), den=[1, -0.8])

    loaded_a, loaded_b = round_trip((model_a, model_b))
    assert not cas.is_equal(loaded_a.params["K"], loaded_b.params["K"])

    with cas.global_pickle_context():
        s = pickle.dumps((model_a, model_b))
    with cas.global_unpickle_context():
        loaded_a, loaded_b = pickle.loads(s)
    assert cas.is_equal(loaded_a.params["K"], loaded_b.params["K"])
    assert_same_model(model_a, loaded_a)
    assert_same_model(model_b, loaded_b)


def test_caller_context_stays_active():
    """Models pickled in the caller's context leave it active for the
    CasADi objects pickled after them, and models pickled without one
    load in it"""
    model = dt_tf_symbolic()
    x = cas.SX.sym("x")
    plain = pickle.dumps(model)  # without a context

    with cas.global_pickle_context():
        s = pickle.dumps((model, x, 2 * x))
    with cas.global_unpickle_context():
        loaded_plain = pickle.loads(plain)
        # Needs the caller's context still active
        loaded, loaded_x, loaded_2x = pickle.loads(s)
    assert_same_model(model, loaded_plain)
    assert_same_model(model, loaded)
    assert cas.depends_on(loaded_2x, loaded_x)


def test_sx_still_needs_context():
    """Only models pickle without a context: CasADi's own rule for SX is
    unchanged, and no context is left active"""
    pickle.dumps(dt_tf_symbolic())
    with pytest.raises(Exception, match="casadi context"):
        pickle.dumps(cas.SX.sym("x"))
