"""Test static nonlinearity models (stateless systems).

Tests that StateSpaceModelCTStaticNonlinearity and its discrete-time
conversions work correctly for systems with no dynamics.
"""

import pytest
import numpy as np
import casadi as cas
from cas_models.continuous_time.models import StateSpaceModelCTStaticNonlinearity
from cas_models.discrete_time.models import StateSpaceModelDTFromCTRK4


def test_static_nonlinearity_ct():
    """Test continuous-time static nonlinearity model"""
    # Create output function: [u0*u1, u0+u1]
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 0)  # Empty state
    u = cas.SX.sym("u", 2)
    y = cas.vertcat(u[0] * u[1], u[0] + u[1])
    h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

    # Create model
    model = StateSpaceModelCTStaticNonlinearity(
        h, nu=2, ny=2, name="multiplier"
    )

    # Verify dimensions
    assert model.n == 0, "Should have no states"
    assert model.nu == 2
    assert model.ny == 2

    # Verify f returns empty vector
    rhs = model.f(0.0, [], [3.0, 4.0])
    rhs_array = np.array(rhs).flatten()
    assert rhs_array.shape == (0,), "f should return empty state derivative"

    # Verify h computes correctly
    y_out = model.h(0.0, [], [3.0, 4.0])
    y_out = np.array(y_out).flatten()
    assert np.allclose(y_out, [12.0, 7.0])


def test_static_nonlinearity_dt_from_ct():
    """Test discrete-time model created from static CT model"""
    # Create CT model
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 0)
    u = cas.SX.sym("u", 2)
    y = cas.vertcat(u[0] * u[1], u[0] + u[1])
    h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

    model_ct = StateSpaceModelCTStaticNonlinearity(
        h, nu=2, ny=2, name="multiplier"
    )

    # Create DT model via RK4
    dt = 1.0
    model_dt = StateSpaceModelDTFromCTRK4(model_ct, dt)

    # Verify dimensions
    assert model_dt.n == 0
    assert model_dt.nu == 2
    assert model_dt.ny == 2
    assert model_dt.dt == dt

    # Verify F returns empty state (no state evolution)
    xkp1 = model_dt.F(0.0, [], [3.0, 4.0])
    xkp1_array = np.array(xkp1).flatten()
    assert xkp1_array.shape == (0,), "F should return empty state"

    # Verify H is same as CT (stateless, so time doesn't matter)
    yk = model_dt.H(0.0, [], [3.0, 4.0])
    yk_array = np.array(yk).flatten()
    assert np.allclose(yk_array, [12.0, 7.0])


def test_static_nonlinearity_with_params():
    """Test static nonlinearity with parameters"""
    # Create output function with parameter: y = a*u
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 0)
    u = cas.SX.sym("u", 1)
    a = cas.SX.sym("a")
    y = a * u
    h = cas.Function("h", [t, x, u, a], [y], ["t", "x", "u", "a"], ["y"])

    # Create model with parameter
    model = StateSpaceModelCTStaticNonlinearity(
        h, nu=1, ny=1, params={"a": a}, name="gain"
    )

    # Verify parameter is stored
    assert "a" in model.params
    assert model.params["a"].name() == "a"

    # Test with parameter value
    y_out = model.h(0.0, [], [5.0], 2.0)  # a=2.0
    y_out = np.array(y_out).flatten()
    assert np.allclose(y_out, [10.0])


def test_static_nonlinearity_multiple_outputs():
    """Test static nonlinearity with multiple independent outputs"""
    # Create function: [u0^2, sqrt(u1), u0+u1+u2]
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 0)
    u = cas.SX.sym("u", 3)
    y = cas.vertcat(u[0] ** 2, cas.sqrt(u[1]), u[0] + u[1] + u[2])
    h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

    model = StateSpaceModelCTStaticNonlinearity(
        h, nu=3, ny=3, name="multi_output"
    )

    # Test evaluation
    y_out = model.h(0.0, [], [3.0, 4.0, 5.0])
    y_out = np.array(y_out).flatten()
    expected = [9.0, 2.0, 12.0]
    assert np.allclose(y_out, expected)


def test_static_nonlinearity_time_invariance():
    """Verify that static nonlinearity output is independent of time"""
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", 0)
    u = cas.SX.sym("u", 1)
    y = u ** 2
    h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

    model = StateSpaceModelCTStaticNonlinearity(h, nu=1, ny=1)

    # Should give same output regardless of time
    y1 = np.array(model.h(0.0, [], [3.0])).flatten()
    y2 = np.array(model.h(100.0, [], [3.0])).flatten()
    y3 = np.array(model.h(-50.0, [], [3.0])).flatten()

    assert np.allclose(y1, y2)
    assert np.allclose(y2, y3)
    assert np.allclose(y1, [9.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
