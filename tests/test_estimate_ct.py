"""Unit tests for src/cas_models/continuous_time/estimate.py and
make_n_step_simulation_function_RK4 in continuous_time/simulate.py"""

import casadi as cas
import numpy as np
import pytest

from cas_models.continuous_time.estimate import (
    estimate_params,
    prediction_errors,
    prediction_errors_ms,
)
from cas_models.continuous_time.models import SSModelCTLinearO2SISO
from cas_models.continuous_time.simulate import (
    make_n_step_simulation_function_RK4,
    make_sim_step_function_integrator,
)
from cas_models.discrete_time.models import StateSpaceModelDTFromCTRK4
from cas_models.discrete_time.simulate import (
    make_n_step_simulation_function_from_model,
)

TRUE_PARAMS = {"K": 2.0, "T1": 3.0, "T2": 8.0}
PARAMS_INIT = {"K": 1.0, "T1": 1.0, "T2": 4.0}
PARAM_BOUNDS = {"T1": (0.1, None), "T2": (0.1, None)}
QUIET = {"print_level": 0}


@pytest.fixture
def irregular_data():
    """Second-order SISO model and data at irregular sample times,
    simulated with CVODES (an accurate integrator, independent of RK4),
    with the input held between samples."""
    model = SSModelCTLinearO2SISO()
    rng = np.random.default_rng(0)
    nT = 120
    t = np.cumsum(np.r_[0.0, rng.uniform(0.1, 1.5, nT)])
    U = np.repeat(rng.choice([-1.0, 1.0], size=nT // 8 + 1), 8)[:nT, None]

    F = make_sim_step_function_integrator(
        model.f,
        model.n,
        model.nu,
        params=model.params,
        solver="cvodes",
        integrator_opts={"abstol": 1e-10, "reltol": 1e-10},
    )
    p = TRUE_PARAMS.values()
    x = np.zeros(model.n)
    Y = []
    for k in range(nT):
        Y.append(float(model.h(t[k], x, U[k], *p)))
        x = F(t[k], x, U[k], t[k + 1] - t[k], *p).full().ravel()
    Y.append(float(model.h(t[nT], x, U[-1], *p)))
    return model, t, U, np.array(Y)[:, None]


def check_o2_estimates(est, rtol):
    assert np.isclose(est["K"], TRUE_PARAMS["K"], rtol=rtol)
    # The time constants of the O2 model are interchangeable
    assert np.allclose(
        sorted([est["T1"], est["T2"]]),
        sorted([TRUE_PARAMS["T1"], TRUE_PARAMS["T2"]]),
        rtol=rtol,
    )


def test_RK4_simulation_regular_grid():
    """On a regular grid with n_sub = 1, the same as the RK4-discretized
    DT model; with n_sub = 2, the same as the DT model with half the time
    step at every other sample."""
    model = SSModelCTLinearO2SISO()
    dt, nT = 0.5, 20
    t = dt * np.arange(nT + 1)
    U = np.sin(t[:-1])[:, None]
    x0 = [0.1, -0.2]
    p = TRUE_PARAMS.values()

    sim = make_n_step_simulation_function_RK4(model, nT)
    X, Y = sim(t, U, x0, *p)
    sim_dt = make_n_step_simulation_function_from_model(
        StateSpaceModelDTFromCTRK4(model, dt), nT
    )
    X_dt, Y_dt = sim_dt(t, U, x0, *p)
    assert np.allclose(X.full(), X_dt.full(), rtol=0, atol=1e-14)
    assert np.allclose(Y.full(), Y_dt.full(), rtol=0, atol=1e-14)

    sim2 = make_n_step_simulation_function_RK4(model, nT, n_sub=2)
    X2, _ = sim2(t, U, x0, *p)
    t_fine = dt / 2 * np.arange(2 * nT + 1)
    sim_fine = make_n_step_simulation_function_from_model(
        StateSpaceModelDTFromCTRK4(model, dt / 2), 2 * nT
    )
    X_fine, _ = sim_fine(t_fine, np.repeat(U, 2, axis=0), x0, *p)
    assert np.allclose(X2.full(), X_fine.full()[::2], rtol=0, atol=1e-14)

    with pytest.raises(ValueError):
        make_n_step_simulation_function_RK4(model, nT, n_sub=0)


def test_prediction_errors(irregular_data):
    model, t, U, Y = irregular_data
    x0 = np.zeros(model.n)
    E = prediction_errors(model, t, U, Y, x0, TRUE_PARAMS, n_sub=4)
    assert E.shape == Y.shape
    # Small errors (RK4 vs CVODES) at the true parameters, smaller with
    # more substeps
    e4 = np.max(np.abs(E.full()))
    E1 = prediction_errors(model, t, U, Y, x0, TRUE_PARAMS, n_sub=1)
    e1 = np.max(np.abs(E1.full()))
    assert e4 < 1e-4 and e4 < e1 / 10


@pytest.mark.parametrize("n_steps", [7, 120])
def test_prediction_errors_ms(irregular_data, n_steps):
    """At the true segment initial states the multiple-shooting errors
    equal the single-shooting ones and the constraints are satisfied."""
    model, t, U, Y = irregular_data
    x0 = np.zeros(model.n)
    sim = make_n_step_simulation_function_RK4(model, len(t) - 1, n_sub=2)
    X_sim, _ = sim(t, U, x0, *TRUE_PARAMS.values())

    opti = cas.Opti()
    E, X0 = prediction_errors_ms(
        opti, model, t, U, Y, x0, TRUE_PARAMS, n_steps, X_init=X_sim, n_sub=2
    )
    E_ss = prediction_errors(model, t, U, Y, x0, TRUE_PARAMS, n_sub=2)
    E_ms = opti.debug.value(E, opti.initial())
    assert np.allclose(np.reshape(E_ms, Y.shape), E_ss.full())
    assert len(X0) == -(-(len(t) - 1) // n_steps)


@pytest.mark.parametrize("n_steps", [None, 10])
def test_estimate_params(irregular_data, n_steps):
    model, t, U, Y = irregular_data
    params_est, x0_est, sol = estimate_params(
        model,
        t,
        U,
        Y,
        n_steps=n_steps,
        params_init=PARAMS_INIT,
        param_bounds=PARAM_BOUNDS,
        solver_options=QUIET,
        n_sub=4,
    )
    assert sol.stats()["return_status"] == "Solve_Succeeded"
    assert list(params_est) == list(model.params)
    assert x0_est is None
    check_o2_estimates(params_est, rtol=1e-5)


def test_estimate_params_n_sub(irregular_data):
    """The RK4 error in the estimates decreases with n_sub."""
    model, t, U, Y = irregular_data
    errors = []
    for n_sub in [1, 4]:
        est, _, _ = estimate_params(
            model,
            t,
            U,
            Y,
            params_init=PARAMS_INIT,
            param_bounds=PARAM_BOUNDS,
            solver_options=QUIET,
            n_sub=n_sub,
        )
        check_o2_estimates(est, rtol=1e-3)
        errors.append(abs(est["K"] - TRUE_PARAMS["K"]))
    assert errors[1] < errors[0]


def test_estimate_params_x0(irregular_data):
    """Data from a non-zero initial state: estimating x0 recovers the
    parameters and the outputs."""
    model, t, U, _ = irregular_data
    sim = make_n_step_simulation_function_RK4(model, len(t) - 1, n_sub=4)
    _, Y = sim(t, U, [0.5, -1.0], *TRUE_PARAMS.values())
    params_est, x0_est, _ = estimate_params(
        model,
        t,
        U,
        Y.full(),
        estimate_x0=True,
        n_steps=10,
        params_init=PARAMS_INIT,
        param_bounds=PARAM_BOUNDS,
        solver_options=QUIET,
        n_sub=4,
    )
    check_o2_estimates(params_est, rtol=1e-6)
    _, Y_est = sim(t, U, x0_est, *params_est.values())
    assert np.allclose(Y_est.full(), Y.full(), atol=1e-6)


def test_time_check(irregular_data):
    model, t, U, Y = irregular_data
    t_bad = t.copy()
    t_bad[5] = t_bad[4]
    x0 = np.zeros(model.n)
    with pytest.raises(ValueError, match="strictly increasing"):
        prediction_errors(model, t_bad, U, Y, x0, TRUE_PARAMS)
    with pytest.raises(ValueError, match="strictly increasing"):
        prediction_errors_ms(
            cas.Opti(), model, t_bad, U, Y, x0, TRUE_PARAMS, 10
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_params(model, t_bad, U, Y, params_init=PARAMS_INIT)
    with pytest.raises(ValueError, match="unknown parameter"):
        estimate_params(model, t, U, Y, params_init={"T3": 1.0})
