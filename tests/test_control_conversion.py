"""Unit tests for src/cas_models/control_conversion.py module"""

import casadi as cas
import control
import numpy as np
import pytest

from cas_models.continuous_time.models import (
    SSModelCTLinearFOSISO,
    StateSpaceModelCT,
    StateSpaceModelCTFromABCD,
)
from cas_models.control_conversion import from_control, to_control
from cas_models.discrete_time.models import (
    SSModelDTARXSISO,
    StateSpaceModelDTFromABCD,
)
from cas_models.recipes import recipe
from cas_models.transformations import linearize

A = np.array([[0.9, 0.1], [0.0, 0.8]])
B = np.array([[1.0, 0.0], [0.5, 2.0]])
C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
D = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]])
NAMES = {
    "input_names": ["u_a", "u_b"],
    "state_names": ["x_a", "x_b"],
    "output_names": ["y_a", "y_b", "y_c"],
}


def check_matrices(sys, ctsys):
    for M, M_ct in zip(linearize(sys), (ctsys.A, ctsys.B, ctsys.C, ctsys.D)):
        np.testing.assert_array_equal(M, M_ct)


@pytest.mark.parametrize(
    "model_class, kwargs, dt",
    [
        (StateSpaceModelCTFromABCD, {}, 0),
        (StateSpaceModelDTFromABCD, {"dt": 0.5}, 0.5),
        (StateSpaceModelDTFromABCD, {}, True),  # dt None: unspecified
    ],
)
def test_round_trip(model_class, kwargs, dt):
    """Matrices, sample time and names go to python-control and back."""
    sys = model_class(A, B, C, D, name="sys", **NAMES, **kwargs)
    ctsys = to_control(sys)
    assert isinstance(ctsys, control.StateSpace)
    check_matrices(sys, ctsys)
    assert ctsys.dt == dt
    assert ctsys.name == "sys"
    assert ctsys.input_labels == NAMES["input_names"]
    assert ctsys.state_labels == NAMES["state_names"]
    assert ctsys.output_labels == NAMES["output_names"]

    sys2 = from_control(ctsys)
    assert type(sys2) is model_class
    check_matrices(sys2, ctsys)
    assert getattr(sys2, "dt", None) == kwargs.get("dt")
    assert sys2.name == "sys"
    assert sys2.input_names == NAMES["input_names"]
    assert sys2.state_names == NAMES["state_names"]
    assert sys2.output_names == NAMES["output_names"]
    # A plain model of matrices: saved by its matrices, not the round trip
    assert recipe(sys2)["class"] is model_class


def test_to_control_default_names():
    """A model without a name gets python-control's sys[k]."""
    sys = StateSpaceModelCTFromABCD(A, B, C, D)
    ctsys = to_control(sys)
    assert ctsys.name.startswith("sys[")
    assert ctsys.input_labels == sys.input_names
    assert ctsys.output_labels == sys.output_names


def test_from_control_generic_names():
    """python-control's generic names (sys[k], u[i], y[i], x[i]) become
    cas_models' defaults; names given replace the system's."""
    ctsys = control.ss(A, B, C, D)
    sys = from_control(ctsys)
    default = StateSpaceModelCTFromABCD(A, B, C, D)
    assert sys.name is None
    assert sys.input_names == default.input_names
    assert sys.state_names == default.state_names
    assert sys.output_names == default.output_names

    sys = from_control(ctsys, name="g", **NAMES)
    assert sys.name == "g"
    assert sys.input_names == NAMES["input_names"]
    assert sys.state_names == NAMES["state_names"]
    assert sys.output_names == NAMES["output_names"]


def test_from_control_after_minreal():
    """python-control's minreal drops the names; give them back."""
    pytest.importorskip("slycot")
    sys = StateSpaceModelCTFromABCD(
        [[-1.0, 0.0], [0.0, -2.0]],
        [[1.0], [0.0]],
        [[1.0, 1.0]],
        [[0.0]],
        name="g",
        input_names=["u1"],
        output_names=["y1"],
    )
    ctsys_min = to_control(sys).minreal()
    assert ctsys_min.input_labels == ["u[0]"]  # names lost
    sys_min = from_control(
        ctsys_min,
        name=sys.name,
        input_names=sys.input_names,
        output_names=sys.output_names,
    )
    assert sys_min.n == 1
    assert sys_min.name == "g"
    assert sys_min.input_names == ["u1"] and sys_min.output_names == ["y1"]


def test_from_control_transfer_function():
    """A transfer function is converted with control.ss()."""
    sys = from_control(control.tf([1.0], [2.0, 1.0]))  # 1 / (2s + 1)
    assert isinstance(sys, StateSpaceModelCTFromABCD)
    A_, B_, C_, D_ = linearize(sys)
    assert A_.shape == (1, 1)
    np.testing.assert_allclose(A_, [[-0.5]])
    np.testing.assert_allclose(C_ @ B_, [[0.5]])  # Markov parameter
    sys = from_control(control.tf([0.2], [1.0, -0.8], dt=1.0))
    assert isinstance(sys, StateSpaceModelDTFromABCD)
    assert sys.dt == 1.0


def test_from_control_dt_none():
    ctsys = control.ss(A, B, C, D, None)
    with pytest.raises(ValueError, match="unspecified"):
        from_control(ctsys)


def test_to_control_analysis():
    """python-control's analysis on a cas_models model (ARX: the poles
    are the roots of A(q))."""
    sys = SSModelDTARXSISO(A=[-1.5, 0.56], B=[0.3, 0.1], nk=1, dt=1.0)
    poles = np.sort(control.poles(to_control(sys)).real)
    np.testing.assert_allclose(poles[-2:], [0.7, 0.8])


def test_to_control_params():
    sys = SSModelCTLinearFOSISO()  # K, T1 symbolic
    with pytest.raises(ValueError, match="values needed"):
        to_control(sys)
    ctsys = to_control(sys, params={"K": 2.0, "T1": 4.0})
    np.testing.assert_allclose(control.dcgain(ctsys), 2.0)


def test_to_control_nonlinear():
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 1)
    u = cas.SX.sym("u")
    f = cas.Function("f", [t, x, u], [-x + u**2], ["t", "x", "u"], ["rhs"])
    h = cas.Function("h", [t, x, u], [x], ["t", "x", "u"], ["y"])
    with pytest.raises(ValueError, match="isn't linear"):
        to_control(StateSpaceModelCT(f, h, n=1))
