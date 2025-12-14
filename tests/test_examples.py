"""Generate examples used in README.md"""

from pathlib import Path
from cas_models.continuous_time.models import (
    StateSpaceModelCT, SSModelCTLinearFOSISO, SSModelCTLinearO2NoGainSISO
)
from cas_models.transformations import connect_systems_in_series

PLOT_DIR = Path("plots")
Path.mkdir(PLOT_DIR, exist_ok=True)

def test_example_build_models():
    # First order, single-input, single-output, continuous-time
    # state-space model with symbolic parameters
    sys_model = SSModelCTLinearFOSISO()
    print(sys_model.f)
    print(sys_model.h)

    assert str(sys_model.f) == "f:(t,x,u,K,T1)->(rhs) SXFunction"
    assert str(sys_model.h) == "h:(t,x,u,K,T1)->(y) SXFunction"

    # First order SISO system with fixed gain
    sys_model = SSModelCTLinearFOSISO(K=2.5)
    print(sys_model.f)
    print(sys_model.h)

    assert str(sys_model.f) == "f:(t,x,u,T1)->(rhs) SXFunction"
    assert str(sys_model.h) == "h:(t,x,u,T1)->(y) SXFunction"

    # Second-order system with gain = 1
    sys_model_2 = SSModelCTLinearO2NoGainSISO()
    print(sys_model_2.f)
    print(sys_model_2.h)

    assert str(sys_model_2.f) == "f:(t,x[2],u,T1,T2)->(rhs[2]) SXFunction"
    assert str(sys_model_2.h) == "h:(t,x[2],u,T1,T2)->(y) SXFunction"

    # Combine both systems by connecting in series
    sys = connect_systems_in_series([sys_model, sys_model_2], model_class=StateSpaceModelCT)
    print(sys.f)
    print(sys.h)

    assert str(sys.f) == "f:(t,x[3],u,sys1_T1,sys2_T1,T2)->(rhs[3]) SXFunction"
    assert str(sys.h) == "h:(t,x[3],u,sys1_T1,sys2_T1,T2)->(y) SXFunction"

    # Series connections can also be made with the '*' operator
    sys = sys_model * sys_model_2
    assert str(sys.f) == "f:(t,x[3],u,sys1_T1,sys2_T1,T2)->(rhs[3]) SXFunction"
    assert str(sys.h) == "h:(t,x[3],u,sys1_T1,sys2_T1,T2)->(y) SXFunction"


from cas_models.discrete_time.models import StateSpaceModelDTFromCT
from cas_models.discrete_time.simulate import make_n_step_simulation_function_from_model

def test_example_discrete_time_simulation():
    # Continuous time model
    sys_ct = SSModelCTLinearFOSISO(K=1, T1=2)

    # Convert to discrete-time with dt=0.1
    dt = 0.1
    sys_dt = StateSpaceModelDTFromCT(sys_ct, dt)
    print(sys_dt.F)
    print(sys_dt.H)

    assert str(sys_dt.F) == "F:(t,xk,uk)->(xkp1) SXFunction"
    assert str(sys_dt.H) == "H:(t,xk,uk)->(yk) SXFunction"

    # Number of time-steps to simulate
    nT = 100
    simulate = make_n_step_simulation_function_from_model(sys_dt, nT)
    print(simulate)

    assert str(simulate) == "F_sim_100_steps:(t_eval[101],U[100],x0)->(X[101],Y[101]) SXFunction"

    import numpy as np
    import matplotlib.pyplot as plt

    # Simualtion time
    # Note: Simulation outputs include values for nT+1 time instants
    t = dt * np.arange(nT+1)
    t_in = t[:-1]

    # Simulation inputs
    U = np.zeros((nT, 1))
    U[t_in >= 1] = 1.0

    # Initial condition
    x0 = np.zeros(sys_dt.n)
    X, Y = simulate(t, U, x0)

    assert X.shape == (nT+1, sys_dt.nu)  # states
    assert Y.shape == (nT+1, sys_dt.ny)  # outputs

    # Make time-series plot
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(t, Y[:, 0])
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"example_sr.png", dpi=150)
    plt.savefig(PLOT_DIR / f"example_sr.svg")
    plt.close()
