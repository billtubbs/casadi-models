"""Unit tests for prediction_errors and prediction_errors_ms in
src/cas_models/discrete_time/simulate.py"""

import casadi as cas
import numpy as np
import pytest

from cas_models.continuous_time.models import SSModelCTLinearO2SISO
from cas_models.discrete_time.models import StateSpaceModelDTFromCTRK4
from cas_models.discrete_time.simulate import (
    make_n_step_simulation_function_from_model,
    prediction_errors,
    prediction_errors_ms,
)


@pytest.fixture
def o2_system_data():
    """Second-order SISO model discretized with RK4 and data simulated
    from it with known parameters and a random step input."""
    dt = 0.5
    model = StateSpaceModelDTFromCTRK4(SSModelCTLinearO2SISO(), dt)
    true_params = {"K": 2.0, "T1": 3.0, "T2": 8.0}

    nT = 103  # not a multiple of the segment lengths tested
    t = dt * np.arange(nT + 1)
    rng = np.random.default_rng(0)
    U = np.repeat(rng.choice([-1.0, 1.0], size=nT // 10 + 1), 10)[:nT]
    U = U.reshape(nT, 1)
    x0 = np.zeros(model.n)

    sim = make_n_step_simulation_function_from_model(model, nT)
    X, Y = sim(t, U, x0, *true_params.values())
    return model, true_params, t, U, X.full(), Y.full(), x0


def test_prediction_errors_zero_at_true_params(o2_system_data):
    model, true_params, t, U, X, Y, x0 = o2_system_data
    E = prediction_errors(model, t, U, Y, x0, true_params)
    assert E.shape == Y.shape
    assert np.allclose(E.full(), 0.0)


def test_prediction_errors_params_as_list(o2_system_data):
    model, true_params, t, U, X, Y, x0 = o2_system_data
    params = {"K": 1.0, "T1": 2.0, "T2": 5.0}
    E1 = prediction_errors(model, t, U, Y, x0, params)
    E2 = prediction_errors(model, t, U, Y, x0, list(params.values()))
    assert np.array_equal(E1.full(), E2.full())
    with pytest.raises(ValueError):
        prediction_errors(model, t, U, Y, x0, {"K": 1.0, "T1": 2.0})


def test_prediction_errors_data_shapes(o2_system_data):
    model, true_params, t, U, X, Y, x0 = o2_system_data
    U_long = np.vstack([U, U[-1:]])  # nT + 1 rows, like a data frame
    with pytest.raises(ValueError, match="U has"):
        prediction_errors(model, t, U_long, Y, x0, true_params)
    with pytest.raises(ValueError, match="Y has"):
        prediction_errors(model, t, U, Y[:-1], x0, true_params)


@pytest.mark.parametrize("n_steps", [1, 10, 25, 103, 200])
def test_prediction_errors_ms_segments(o2_system_data, n_steps):
    """At the true segment initial states, the errors equal the single-
    shooting errors and the continuity constraints are satisfied."""
    model, true_params, t, U, X, Y, x0 = o2_system_data
    nT = len(t) - 1
    params = {"K": 1.5, "T1": 2.0, "T2": 5.0}  # not the true values

    # Simulated states with these parameters
    sim = make_n_step_simulation_function_from_model(model, nT)
    X_sim, _ = sim(t, U, x0, *params.values())

    opti = cas.Opti()
    E, X0 = prediction_errors_ms(
        opti, model, t, U, Y, x0, params, n_steps, X_init=X_sim
    )
    n_seg = -(-nT // n_steps)  # ceil
    assert len(X0) == n_seg
    assert E.shape == Y.shape
    assert opti.g.shape[0] == model.n * (n_seg - 1)

    # Evaluate at the initial guesses (the simulated states)
    E_ss = prediction_errors(model, t, U, Y, x0, params)
    E_ms = opti.debug.value(E, opti.initial())
    assert np.allclose(np.reshape(E_ms, Y.shape), E_ss.full())
    if n_seg > 1:
        # Continuity constraints satisfied (Opti may turn some into
        # bounds, g = x with lbg = ubg, so compare g with its bounds)
        g, lbg, ubg = (
            opti.debug.value(expr, opti.initial())
            for expr in (opti.g, opti.lbg, opti.ubg)
        )
        assert np.allclose(g, lbg) and np.allclose(g, ubg)


def test_prediction_errors_ms_without_X_init(o2_system_data):
    model, true_params, t, U, X, Y, x0 = o2_system_data
    opti = cas.Opti()
    E, X0 = prediction_errors_ms(opti, model, t, U, Y, x0, true_params, 20)
    # Segment initial states start at zero
    for x in X0[1:]:
        assert np.array_equal(opti.debug.value(x, opti.initial()), [0, 0])
    with pytest.raises(ValueError):
        prediction_errors_ms(opti, model, t, U, Y, x0, true_params, 0)


@pytest.mark.parametrize("n_steps", [None, 10, 30])
def test_parameter_estimation(o2_system_data, n_steps):
    """Single and multiple shooting recover the true parameters."""
    model, true_params, t, U, X, Y, x0 = o2_system_data
    params_init = {"K": 1.0, "T1": 1.0, "T2": 4.0}

    opti = cas.Opti()
    p = {name: opti.variable() for name in model.params}
    for name, value in params_init.items():
        opti.set_initial(p[name], value)
    opti.subject_to(p["T1"] >= 0.1)
    opti.subject_to(p["T2"] >= 0.1)

    if n_steps is None:
        E = prediction_errors(model, t, U, Y, x0, p)
    else:
        sim = make_n_step_simulation_function_from_model(model, len(t) - 1)
        X_init, _ = sim(t, U, x0, *params_init.values())
        E, X0 = prediction_errors_ms(
            opti, model, t, U, Y, x0, p, n_steps, X_init=X_init
        )
    opti.minimize(cas.sumsqr(E))
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0})
    sol = opti.solve()

    est = {name: sol.value(var) for name, var in p.items()}
    assert np.isclose(est["K"], true_params["K"], rtol=1e-4)
    # The time constants of the O2 model are interchangeable
    assert np.allclose(
        sorted([est["T1"], est["T2"]]),
        sorted([true_params["T1"], true_params["T2"]]),
        rtol=1e-4,
    )
