"""Models used by several test modules, one function each, so every test
gets new symbols. (Not test_*.py, so pytest doesn't collect it; the test
modules import it as models_for_tests, since pytest puts tests/ on
sys.path.)"""

import casadi as cas

from cas_models.continuous_time.models import (
    SSModelCTLinearFOSISO,
    StateSpaceModelCT,
)
from cas_models.continuous_time.regulators import SSModelCTPIInt
from cas_models.discrete_time.models import (
    SSModelDTARXSISO,
    StateSpaceModelDT,
)
from cas_models.transformations import (
    connect_feedback_system,
    connect_systems,
)


def ct_custom():
    """Nonlinear, with a vector parameter p"""
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 2)
    u = cas.SX.sym("u")
    p = cas.SX.sym("p", 2)
    rhs = cas.vertcat(-p[0] * x[0] ** 2 + u, x[0] - p[1] * x[1])
    f = cas.Function("f", [t, x, u, p], [rhs], ["t", "x", "u", "p"], ["rhs"])
    h = cas.Function("h", [t, x, u, p], [x[1]], ["t", "x", "u", "p"], ["y"])
    return StateSpaceModelCT(
        f, h, n=2, params={"p": p}, name="custom", state_names=["x", "z"]
    )


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


def dt_connected_arx(inputs=("CRAC1", "CRAC2"), output="R11", **kwargs):
    """Two ARX paths summed, as in ffdtools.models (ARXPaths). kwargs
    (e.g. realization) go to both ARX paths."""
    inputs = list(inputs)
    paths = [
        SSModelDTARXSISO(
            na=2, nb=2, input_name="u", output_name="y", **kwargs
        ),
        SSModelDTARXSISO(
            na=1, nb=1, input_name="u", output_name="y", **kwargs
        ),
    ]
    connections = {f"{name}_u": name for name in inputs}
    connections[output] = [f"{name}_y" for name in inputs]
    return connect_systems(
        paths,
        connections,
        StateSpaceModelDT,
        input_names=inputs,
        output_names=[output],
        keys=inputs,
        verbose_names=True,
        name=output,
    )
