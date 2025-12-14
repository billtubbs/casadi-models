from pathlib import Path
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from cas_models.continuous_time.models import (
    SSModelCTLinearFOSISO,
    SSModelCTLinearO2SISO,
    SSModelCTLinearO2UnderdampedSISO,
    StateSpaceModelCTStaticNonlinearity,
)
from cas_models.continuous_time.simulate import (
    make_step_function,
    make_sim_step_function_RK4_fixed_dt,
)
from cas_models.discrete_time.simulate import make_n_step_simulation_function

PLOT_DIR = Path("plots")
Path.mkdir(PLOT_DIR, exist_ok=True)

models = {
    "fo": SSModelCTLinearFOSISO(K=1, T1=2),
    "o2": SSModelCTLinearO2SISO(K=1, T1=2, T2=2),
    "o2ud": SSModelCTLinearO2UnderdampedSISO(K=1, zeta=0.6, omega_n=1.0),
}

step_function = make_step_function(mag=1.0, t_step=1.0)


def make_source_from_time_function(f, name=None, ny=1, params=None):
    if name is None:
        name = step_function.name()
    if params is None:
        params = {}
    n = 0  # no states
    nu = 0  # no inputs
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)
    y = f(t)
    assert y.shape[0] == ny
    h = cas.Function(
        name,
        [t, x, u, *params.values()],
        [y],
        ["t", "x", "u", *params.keys()],
        ["y"],
    )
    sys = StateSpaceModelCTStaticNonlinearity(h, nu=nu, ny=ny)
    return sys


input_sys = make_source_from_time_function(step_function)

# Number of simulation times
nT = 110

# Sampling period
Ts = 0.1

systems = {}
sim_functions = {}
for name, model in models.items():
    # Connect model to input source
    sys = input_sys * model
    systems[name] = sys
    f = sys.f
    h = sys.h
    n = sys.n
    nu = sys.nu
    ny = sys.ny
    params = sys.params
    F = make_sim_step_function_RK4_fixed_dt(f, n, nu, Ts, params=params)
    simulate = make_n_step_simulation_function(
        F, h, n, nu, ny, nT, params=params
    )
    sim_functions[name] = simulate

# Sample times
t = Ts * np.arange(nT + 1)

# Make output response plots
for name in models:
    sys = systems[name]
    simulate = sim_functions[name]
    x0 = np.zeros(sys.n)
    U = np.empty((nT, sys.nu))

    X, Y = simulate(t, U, x0)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(t, Y[:, 0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}_sr.png", dpi=150)
    plt.savefig(PLOT_DIR / f"{name}_sr.svg")
    plt.close()
