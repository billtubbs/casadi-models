"""Unit tests for src/cas_models/continuous_time/simulate.py module"""

import pytest
import numpy as np
import casadi as cas
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.continuous_time.simulate import (
    make_sim_step_function_RK4,
    make_sim_step_function_RK4_fixed_dt,
    make_sim_step_function_integrator,
    make_sim_step_function_integrator_fixed_dt,
    make_steady_state_solver,
    make_step_function,
)
from cas_models.discrete_time.simulate import (
    make_n_step_simulation_function,
)
from cas_models.discrete_time.models import validate_F_function


@pytest.fixture
def cart_pole_system():
    """Cart-pole (inverted pendulum) system.

    A cart with mass M on a frictionless track with a pole of mass m
    and length L attached by a frictionless hinge.

    States: [x, x_dot, theta, theta_dot]
        x: cart position (m)
        x_dot: cart velocity (m/s)
        theta: pole angle from vertical (rad)
        theta_dot: pole angular velocity (rad/s)

    Input: u - force applied to cart (N)

    Outputs: [x, theta]

    Parameters:
        M: cart mass (kg)
        m: pole mass (kg)
        L: pole length (m)
        g: gravitational acceleration (m/s^2)

    Returns:
        tuple: (n, nu, ny, f, h, params, param_values)
    """
    # System dimensions
    n = 4  # number of states
    nu = 1  # number of inputs
    ny = 2  # number of outputs

    # Symbolic parameters
    M = cas.SX.sym("M")  # cart mass
    m = cas.SX.sym("m")  # pole mass
    L = cas.SX.sym("L")  # pole length
    g = cas.SX.sym("g")  # gravity

    params = {"M": M, "m": m, "L": L, "g": g}

    # Default parameter values
    param_values = {
        "M": 1.0,  # kg
        "m": 0.1,  # kg
        "L": 0.5,  # m
        "g": 9.81,  # m/s^2
    }

    # State and input variables
    t = cas.SX.sym("t")
    x = cas.SX.sym("x", n)
    u = cas.SX.sym("u", nu)

    # Unpack states
    x_pos = x[0]  # cart position
    x_dot = x[1]  # cart velocity
    theta = x[2]  # pole angle
    theta_dot = x[3]  # pole angular velocity

    # System dynamics (inverted pendulum equations)
    # Total mass
    m_total = M + m

    # Common terms
    sin_theta = cas.sin(theta)
    cos_theta = cas.cos(theta)

    # Equations of motion
    # Denominator term
    denom = m_total - m * cos_theta**2

    # Cart acceleration
    x_ddot = (u + m * sin_theta * (L * theta_dot**2 + g * cos_theta)) / denom

    # Pole angular acceleration
    theta_ddot = (
        -u * cos_theta
        - m * L * theta_dot**2 * cos_theta * sin_theta
        + m_total * g * sin_theta
    ) / (L * denom)

    # State derivatives: [x_dot, x_ddot, theta_dot, theta_ddot]
    rhs = cas.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)

    # Create CasADi function for ODE right-hand side
    f = cas.Function(
        "cart_pole_f",
        [t, x, u, M, m, L, g],
        [rhs],
        ["t", "x", "u", "M", "m", "L", "g"],
        ["rhs"],
    )

    # Output function: y = [x, theta]
    y = cas.vertcat(x_pos, theta)

    h = cas.Function(
        "cart_pole_h",
        [t, x, u, M, m, L, g],
        [y],
        ["t", "x", "u", "M", "m", "L", "g"],
        ["y"],
    )

    return n, nu, ny, f, h, params, param_values


def test_cart_pole_equilibrium(cart_pole_system):
    """Test that cart-pole is at equilibrium when upright with no force."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Equilibrium state: cart at origin, pole upright, no velocities
    x_eq = np.array([0.0, 0.0, 0.0, 0.0])
    u_eq = 0.0
    t = 0.0

    # Evaluate dynamics at equilibrium
    rhs = f(t, x_eq, u_eq, *param_values.values())

    # At equilibrium, derivatives should be zero
    assert np.allclose(rhs, np.zeros((n, 1)), atol=1e-10)

    # Check output
    y = h(t, x_eq, u_eq, *param_values.values())
    assert np.allclose(y, np.array([[0.0], [0.0]]))


def test_make_sim_step_function_RK4(cart_pole_system):
    """Test RK4 integration for cart-pole system."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create RK4 step function
    F_rk4 = make_sim_step_function_RK4(f, n, nu, params=params, name="F_rk4")

    # Check function signature
    assert F_rk4.name() == "F_rk4"
    assert F_rk4.n_in() == 4 + len(params)  # t, x, u, dt, + params
    assert F_rk4.n_out() == 1  # xf

    # Initial state: small perturbation from equilibrium
    x0 = np.array([0.0, 0.0, 0.1, 0.0])  # small initial angle
    u = 0.0
    t = 0.0
    dt = 0.01

    # Take one step
    x1 = F_rk4(t, x0, u, dt, *param_values.values())

    # Check that state evolved (angle should increase due to gravity)
    assert x1.shape == (n, 1)
    assert not np.allclose(x1, x0.reshape(-1, 1))

    # With no control, pole should fall to the right (theta > 0)
    # theta_dot should be positive (falling in positive theta direction)
    assert float(x1[3]) > 0  # angular velocity increases


def test_make_sim_step_function_integrator_rk(cart_pole_system):
    """Test CasADi RK integrator for cart-pole system."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create integrator with RK solver
    F_rk = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        name="F_rk",
        solver="rk",
        integrator_opts={"number_of_finite_elements": 4},
    )

    # Check function signature
    assert F_rk.name() == "F_rk"
    assert F_rk.n_in() == 4 + len(params)  # t, x, u, dt, + params
    assert F_rk.n_out() == 1  # xf

    # Initial state: small perturbation from equilibrium
    x0 = np.array([0.0, 0.0, 0.1, 0.0])  # small initial angle
    u = 0.0
    t = 0.0
    dt = 0.01

    # Take one step
    x1 = F_rk(t, x0, u, dt, *param_values.values())

    # Check that state evolved
    assert x1.shape == (n, 1)
    assert not np.allclose(x1, x0.reshape(-1, 1))

    # With no control, pole should fall to the right (theta > 0)
    # theta_dot should be positive
    assert float(x1[3]) > 0  # angular velocity increases


def test_make_sim_step_function_integrator_cvodes(cart_pole_system):
    """Test CasADi CVodes integrator for cart-pole system."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create integrator with CVodes solver
    F_cvodes = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        name="F_cvodes",
        solver="cvodes",
        integrator_opts={"abstol": 1e-8, "reltol": 1e-6},
    )

    # Check function signature
    assert F_cvodes.name() == "F_cvodes"
    assert F_cvodes.n_in() == 4 + len(params)  # t, x, u, dt, + params
    assert F_cvodes.n_out() == 1  # xf

    # Initial state: small perturbation from equilibrium
    x0 = np.array([0.0, 0.0, 0.1, 0.0])  # small initial angle
    u = 0.0
    t = 0.0
    dt = 0.01

    # Take one step
    x1 = F_cvodes(t, x0, u, dt, *param_values.values())

    # Check that state evolved
    assert x1.shape == (n, 1)
    assert not np.allclose(x1, x0.reshape(-1, 1))

    # With no control, pole should fall to the right (theta > 0)
    # theta_dot should be positive
    assert float(x1[3]) > 0  # angular velocity increases


def test_integrator_comparison(cart_pole_system):
    """Compare RK4, RK, and CVodes integrators on same initial condition."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create all three integrators
    F_rk4 = make_sim_step_function_RK4(f, n, nu, params=params)

    F_rk = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="rk",
        integrator_opts={"number_of_finite_elements": 4},
    )

    F_cvodes = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="cvodes",
        integrator_opts={"abstol": 1e-10, "reltol": 1e-10},
    )

    # Initial condition
    x0 = np.array([0.0, 0.0, 0.1, 0.0])
    u = 0.0
    t = 0.0
    dt = 0.01

    # Take one step with each integrator
    x1_rk4 = F_rk4(t, x0, u, dt, *param_values.values())
    x1_rk = F_rk(t, x0, u, dt, *param_values.values())
    x1_cvodes = F_cvodes(t, x0, u, dt, *param_values.values())

    # Results should be similar (but not identical due to different methods)
    # CVodes with tight tolerances should be most accurate
    assert np.allclose(x1_rk4, x1_rk, rtol=1e-3, atol=1e-5)
    assert np.allclose(x1_rk4, x1_cvodes, rtol=1e-3, atol=1e-5)
    assert np.allclose(x1_rk, x1_cvodes, rtol=1e-3, atol=1e-5)


def test_integrator_with_control_input(cart_pole_system):
    """Test integrators with non-zero control input."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create RK4 integrator
    F_rk4 = make_sim_step_function_RK4(f, n, nu, params=params)

    # Initial state: pole tilted to the right
    x0 = np.array([0.0, 0.0, 0.1, 0.0])

    # Apply force to the right (positive direction)
    u_right = 5.0

    # Apply force to the left (negative direction)
    u_left = -5.0

    t = 0.0
    dt = 0.01

    # Simulate with right force
    x1_right = F_rk4(t, x0, u_right, dt, *param_values.values())

    # Simulate with left force
    x1_left = F_rk4(t, x0, u_left, dt, *param_values.values())

    # Cart should move in direction of applied force
    # Right force should increase cart position
    assert float(x1_right[1]) > 0  # positive velocity

    # Left force should decrease cart position
    assert float(x1_left[1]) < 0  # negative velocity

    # Forces should affect the system differently
    assert not np.allclose(x1_right, x1_left)


def test_make_n_step_simulation_function(cart_pole_system):
    """Test multi-step simulation function."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create single-step integrator with dt as parameter
    sim_step = make_sim_step_function_RK4(f, n, nu, params=params)

    # Fixed time step
    dt = 0.1

    # Create F without dt argument (fixed dt version)
    t = cas.SX.sym("t")
    xk = cas.SX.sym("xk", n)
    uk = cas.SX.sym("uk", nu)
    xkp1 = sim_step(t, xk, uk, dt, *params.values())

    F = cas.Function(
        "F",
        [t, xk, uk, *params.values()],
        [xkp1],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )

    # Number of time steps
    nT = 10

    # Create multi-step simulation function
    sim_func = make_n_step_simulation_function(
        F, h, n, nu, ny, nT, params=params, name="cart_pole_sim"
    )

    # Check function signature
    assert sim_func.name() == "cart_pole_sim"
    assert sim_func.n_in() == 3 + len(params)  # t_eval, U, x0, + params
    assert sim_func.n_out() == 2  # X, Y

    # Time vector
    t_eval = np.linspace(0, 1.0, nT + 1)

    # Control inputs (all zeros)
    U = np.zeros((nT, nu))

    # Initial state
    x0 = np.array([0.0, 0.0, 0.1, 0.0])

    # Run simulation
    X, Y = sim_func(t_eval, U, x0, *param_values.values())

    # Check outputs
    assert X.shape == (nT + 1, n)  # states at all time steps
    assert Y.shape == (nT + 1, ny)  # outputs at all time steps

    # Initial state should match
    assert np.allclose(X[0, :], x0)

    # Initial output should be [x, theta]
    assert np.allclose(Y[0, :], [0.0, 0.1])

    # System should evolve (pole should fall)
    assert not np.allclose(X[-1, :], x0)

    # Pole angle should increase (falling)
    assert abs(float(Y[-1, 1])) > abs(float(Y[0, 1]))


def test_multi_step_integrator_comparison(cart_pole_system):
    """Compare RK4 and RK integrators over multiple steps with step input."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Simulation parameters
    nT = 20  # number of time steps
    dt = 0.05  # time step
    t_eval = np.linspace(0, nT * dt, nT + 1)

    # Initial state: pole at 45 degrees
    x0 = np.array([0.0, 0.0, np.pi / 4, 0.0])

    # Control input: step force applied halfway through
    U = np.zeros((nT, nu))
    U[nT // 2 :, 0] = 10.0  # Apply 10N force in second half

    # Create RK4 integrator
    sim_step_rk4 = make_sim_step_function_RK4(f, n, nu, params=params)

    t_sym = cas.SX.sym("t")
    xk_sym = cas.SX.sym("xk", n)
    uk_sym = cas.SX.sym("uk", nu)
    xkp1_rk4 = sim_step_rk4(t_sym, xk_sym, uk_sym, dt, *params.values())

    F_rk4 = cas.Function(
        "F_rk4",
        [t_sym, xk_sym, uk_sym, *params.values()],
        [xkp1_rk4],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )

    # Create CasADi RK integrator
    sim_step_rk = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="rk",
        integrator_opts={"number_of_finite_elements": 4},
    )

    xkp1_rk = sim_step_rk(t_sym, xk_sym, uk_sym, dt, *params.values())

    F_rk = cas.Function(
        "F_rk",
        [t_sym, xk_sym, uk_sym, *params.values()],
        [xkp1_rk],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )

    # Simulate with both integrators
    X_rk4, Y_rk4 = make_n_step_simulation_function(
        F_rk4, h, n, nu, ny, nT, params=params, name="sim_rk4"
    )(t_eval, U, x0, *param_values.values())

    X_rk, Y_rk = make_n_step_simulation_function(
        F_rk, h, n, nu, ny, nT, params=params, name="sim_rk"
    )(t_eval, U, x0, *param_values.values())

    # Results should be very similar (RK4 methods)
    assert np.allclose(X_rk4, X_rk, rtol=1e-3, atol=1e-4)
    assert np.allclose(Y_rk4, Y_rk, rtol=1e-3, atol=1e-4)

    # System should respond to the step input
    # Position should change when force is applied
    pos_before_step = Y_rk4[nT // 2, 0]
    pos_after_step = Y_rk4[-1, 0]
    assert abs(pos_after_step - pos_before_step) > 0.01


def test_cvodes_integrator_multi_step(cart_pole_system):
    """Test CVodes integrator over multiple steps with step input."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Simulation parameters
    nT = 20  # number of time steps
    dt = 0.05  # time step
    t_eval = np.linspace(0, nT * dt, nT + 1)

    # Initial state: pole at 45 degrees
    x0 = np.array([0.0, 0.0, np.pi / 4, 0.0])

    # Control input: step force applied halfway through
    U = np.zeros((nT, nu))
    U[nT // 2 :, 0] = 10.0  # Apply 10N force in second half

    # Create CVodes integrator
    sim_step_cvodes = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="cvodes",
        integrator_opts={"abstol": 1e-8, "reltol": 1e-8},
    )

    t_sym = cas.SX.sym("t")
    xk_sym = cas.SX.sym("xk", n)
    uk_sym = cas.SX.sym("uk", nu)
    xkp1_cvodes = sim_step_cvodes(t_sym, xk_sym, uk_sym, dt, *params.values())

    F_cvodes = cas.Function(
        "F_cvodes",
        [t_sym, xk_sym, uk_sym, *params.values()],
        [xkp1_cvodes],
        ["t", "xk", "uk", *params.keys()],
        ["xkp1"],
    )

    # Simulate with CVodes
    X_cvodes, Y_cvodes = make_n_step_simulation_function(
        F_cvodes, h, n, nu, ny, nT, params=params, name="sim_cvodes"
    )(t_eval, U, x0, *param_values.values())

    # Check that simulation completed successfully
    assert X_cvodes.shape == (nT + 1, n)
    assert Y_cvodes.shape == (nT + 1, ny)

    # System should respond to the step input
    # Position should change when force is applied
    pos_before_step = Y_cvodes[nT // 2, 0]
    pos_after_step = Y_cvodes[-1, 0]
    assert abs(pos_after_step - pos_before_step) > 0.01

    # Pole should move due to initial angle
    assert abs(Y_cvodes[-1, 1]) > 0  # angle not zero at end


def test_step_function():
    """Test the step function generator."""
    # Create step function: magnitude 2.0 at t=0.5
    step = make_step_function(mag=2.0, t_step=0.5)

    # Check function signature
    assert step.n_in() == 1  # t
    assert step.n_out() == 1  # y

    # Before step
    assert float(step(0.0)) == 0.0
    assert float(step(0.25)) == 0.0
    assert float(step(0.49)) == 0.0

    # At and after step
    assert float(step(0.5)) == 2.0
    assert float(step(0.75)) == 2.0
    assert float(step(1.0)) == 2.0


def test_integrator_numerical_stability(cart_pole_system):
    """Test that CVodes integrator remains numerically stable over time."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create CVodes integrator with tight tolerances
    F = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="cvodes",
        integrator_opts={"abstol": 1e-10, "reltol": 1e-10},
    )

    # Initial state: pole at 30 degrees with no velocity
    x0 = np.array([0.0, 0.0, np.pi / 6, 0.0])

    # Simulate without control for moderate time
    u = 0.0
    t = 0.0
    dt = 0.01
    nT = 50

    # Take multiple steps
    x = x0
    states = [x0]
    for k in range(nT):
        x = np.array(F(t + k * dt, x, u, dt, *param_values.values())).flatten()
        states.append(x)

    # Check that states don't blow up (remain bounded)
    states = np.array(states)

    # All states should remain finite
    assert np.all(np.isfinite(states))

    # Cart position should remain reasonable (within 10 meters)
    assert np.all(np.abs(states[:, 0]) < 10.0)

    # Cart velocity should remain reasonable (within 50 m/s)
    assert np.all(np.abs(states[:, 1]) < 50.0)

    # Pole angle might flip over but should remain bounded
    assert np.all(np.abs(states[:, 2]) < 4 * np.pi)  # max 2 full rotations

    # Angular velocity should remain reasonable
    assert np.all(np.abs(states[:, 3]) < 100.0)  # rad/s


def test_make_sim_step_function_RK4_with_dt_parameter(cart_pole_system):
    """Test that dt can be varied as a symbolic parameter."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create RK4 step function
    F = make_sim_step_function_RK4(f, n, nu, params=params)

    # Initial state
    x0 = np.array([0.0, 0.0, 0.1, 0.0])
    u = 0.0
    t = 0.0

    # Try different time steps
    dt_small = 0.001
    dt_large = 0.1

    x1_small = F(t, x0, u, dt_small, *param_values.values())
    x1_large = F(t, x0, u, dt_large, *param_values.values())

    # Larger time step should result in larger state change
    delta_small = np.linalg.norm(np.array(x1_small) - x0.reshape(-1, 1))
    delta_large = np.linalg.norm(np.array(x1_large) - x0.reshape(-1, 1))

    assert delta_large > delta_small


def test_make_sim_step_function_RK4_fixed_dt(cart_pole_system):
    """Test RK4 integration with fixed dt."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create fixed-dt RK4 integrator
    dt = 0.01
    F_fixed = make_sim_step_function_RK4_fixed_dt(
        f, n, nu, dt, params=params, name="F_fixed"
    )

    # Verify function signature using validate_F_function
    validate_F_function(F_fixed, n, nu, params=params)

    # Check function has correct signature (no dt argument)
    assert F_fixed.n_in() == 3 + len(params)  # t, x, u, + params
    assert F_fixed.n_out() == 1  # xf

    # Initial state
    x0 = np.array([0.0, 0.0, 0.1, 0.0])
    u = 0.0
    t = 0.0

    # Take one step with fixed-dt version
    x1_fixed = F_fixed(t, x0, u, *param_values.values())

    # Compare with variable-dt version using same dt
    F_variable = make_sim_step_function_RK4(f, n, nu, params=params)
    x1_variable = F_variable(t, x0, u, dt, *param_values.values())

    # Results should be identical
    assert np.allclose(x1_fixed, x1_variable, rtol=1e-10, atol=1e-12)


def test_make_sim_step_function_integrator_fixed_dt_rk(cart_pole_system):
    """Test CasADi RK integrator with fixed dt."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create fixed-dt RK integrator
    dt = 0.01
    F_fixed = make_sim_step_function_integrator_fixed_dt(
        f,
        n,
        nu,
        dt,
        params=params,
        name="F_fixed_rk",
        solver="rk",
        integrator_opts={"number_of_finite_elements": 4},
    )

    # Verify function signature using validate_F_function
    validate_F_function(F_fixed, n, nu, params=params)

    # Check function has correct signature (no dt argument)
    assert F_fixed.n_in() == 3 + len(params)  # t, x, u, + params
    assert F_fixed.n_out() == 1  # xf

    # Initial state
    x0 = np.array([0.0, 0.0, 0.1, 0.0])
    u = 0.0
    t = 0.0

    # Take one step with fixed-dt version
    x1_fixed = F_fixed(t, x0, u, *param_values.values())

    # Compare with variable-dt version using same dt
    F_variable = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="rk",
        integrator_opts={"number_of_finite_elements": 4},
    )
    x1_variable = F_variable(t, x0, u, dt, *param_values.values())

    # Results should be very close
    assert np.allclose(x1_fixed, x1_variable, rtol=1e-6, atol=1e-8)


def test_make_sim_step_function_integrator_fixed_dt_cvodes(cart_pole_system):
    """Test CasADi CVodes integrator with fixed dt."""
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create fixed-dt CVodes integrator
    dt = 0.01
    F_fixed = make_sim_step_function_integrator_fixed_dt(
        f,
        n,
        nu,
        dt,
        params=params,
        name="F_fixed_cvodes",
        solver="cvodes",
        integrator_opts={"abstol": 1e-10, "reltol": 1e-10},
    )

    # Verify function signature using validate_F_function
    validate_F_function(F_fixed, n, nu, params=params)

    # Check function has correct signature (no dt argument)
    assert F_fixed.n_in() == 3 + len(params)  # t, x, u, + params
    assert F_fixed.n_out() == 1  # xf

    # Initial state
    x0 = np.array([0.0, 0.0, 0.1, 0.0])
    u = 0.0
    t = 0.0

    # Take one step with fixed-dt version
    x1_fixed = F_fixed(t, x0, u, *param_values.values())

    # Compare with variable-dt version using same dt
    F_variable = make_sim_step_function_integrator(
        f,
        n,
        nu,
        params=params,
        solver="cvodes",
        integrator_opts={"abstol": 1e-10, "reltol": 1e-10},
    )
    x1_variable = F_variable(t, x0, u, dt, *param_values.values())

    # Results should be very close
    assert np.allclose(x1_fixed, x1_variable, rtol=1e-6, atol=1e-8)


def test_fixed_dt_functions_with_n_step_simulation(cart_pole_system):
    """Test that fixed-dt functions work directly with
    make_n_step_simulation_function.
    """
    n, nu, ny, f, h, params, param_values = cart_pole_system

    # Create fixed-dt integrator
    dt = 0.05
    F_fixed = make_sim_step_function_RK4_fixed_dt(
        f, n, nu, dt, params=params, name="F"
    )

    # Use directly with n-step simulation (no wrapping needed)
    nT = 10
    sim_func = make_n_step_simulation_function(
        F_fixed, h, n, nu, ny, nT, params=params, name="sim"
    )

    # Time vector
    t_eval = np.linspace(0, nT * dt, nT + 1)

    # Control inputs
    U = np.zeros((nT, nu))
    U[nT // 2 :, 0] = 5.0  # Apply force in second half

    # Initial state
    x0 = np.array([0.0, 0.0, np.pi / 4, 0.0])

    # Run simulation
    X, Y = sim_func(t_eval, U, x0, *param_values.values())

    # Verify results
    assert X.shape == (nT + 1, n)
    assert Y.shape == (nT + 1, ny)

    # System should evolve
    assert not np.allclose(X[-1, :], x0)

    # Position should change due to force
    assert abs(float(Y[-1, 0])) > 0.01


@pytest.fixture
def cascaded_nonlinear_system():
    """A simple 2-state cascaded nonlinear system with a known analytical SS.

    Dynamics:
        dx1/dt = -a*(x1 - u)
        dx2/dt = -b*x2 + x1^2

    Output: y = x2

    Analytical steady state for constant input u:
        x1_ss = u,  x2_ss = u^2 / b,  y_ss = u^2 / b

    The Jacobian d(f)/d(x) is lower-triangular with diagonal [-a, -b], so it
    is always full-rank for positive a and b.

    Parameters: a, b (positive real scalars)
    """
    n, nu, ny = 2, 1, 1

    t_sx = cas.SX.sym("t")
    x_sx = cas.SX.sym("x", n)
    u_sx = cas.SX.sym("u", nu)
    a_sx = cas.SX.sym("a")
    b_sx = cas.SX.sym("b")

    params = {"a": a_sx, "b": b_sx}
    param_values = {"a": 2.0, "b": 3.0}

    rhs = cas.vertcat(
        -a_sx * (x_sx[0] - u_sx[0]),
        -b_sx * x_sx[1] + x_sx[0] ** 2,
    )
    f = cas.Function(
        "f",
        [t_sx, x_sx, u_sx, a_sx, b_sx],
        [rhs],
        ["t", "x", "u", "a", "b"],
        ["rhs"],
    )

    y = x_sx[1:2]
    h = cas.Function(
        "h",
        [t_sx, x_sx, u_sx, a_sx, b_sx],
        [y],
        ["t", "x", "u", "a", "b"],
        ["y"],
    )

    return n, nu, ny, f, h, params, param_values


def test_make_steady_state_solver_raises_on_free_states(cart_pole_system):
    """A clear ValueError is raised when the model has structurally free states.

    The cart-pole has x[0] (cart position) absent from all rhs components,
    making the Jacobian rank-deficient.  The error should name the offending
    index and suggest auto_reduce=True.
    """
    n, nu, ny, f, h, params, param_values = cart_pole_system
    model = StateSpaceModelCT(f=f, h=h, n=n, nu=nu, ny=ny, params=params)

    with pytest.raises(ValueError, match=r"indices \[0\]"):
        make_steady_state_solver(model)

    # The message should also mention auto_reduce.
    with pytest.raises(ValueError, match="auto_reduce=True"):
        make_steady_state_solver(model)


def test_make_steady_state_solver_auto_reduce_cart_pole(cart_pole_system):
    """auto_reduce=True solves for the constrained states and pins the free ones.

    For the cart-pole with u=0, the upright equilibrium is
    x=[x0_pos, 0, 0, 0]: cart position unchanged from x0, all others zero.
    """
    n, nu, ny, f, h, params, param_values = cart_pole_system
    model = StateSpaceModelCT(f=f, h=h, n=n, nu=nu, ny=ny, params=params)
    solve_ss = make_steady_state_solver(model, auto_reduce=True)

    u = np.array([0.0])
    x_pos_init = 2.5  # arbitrary non-zero cart position
    x0 = np.array([x_pos_init, 0.05, 0.1, 0.0])
    x_ss, y_ss = solve_ss(x0, u, param_values)

    # Free state (cart position) is returned unchanged from x0.
    assert x_ss[0] == pytest.approx(x_pos_init)

    # Constrained states converge to the upright equilibrium.
    assert x_ss[1] == pytest.approx(0.0, abs=1e-8)  # x_dot
    assert x_ss[2] == pytest.approx(0.0, abs=1e-8)  # theta
    assert x_ss[3] == pytest.approx(0.0, abs=1e-8)  # theta_dot

    # Output y_ss should match h(x_ss, u).
    y_check = np.array(h(0, x_ss, u, *param_values.values())).flatten()
    assert np.allclose(y_ss, y_check, atol=1e-10)


def test_make_steady_state_solver_known_ss(cascaded_nonlinear_system):
    """Solver recovers the analytical steady state of a nonlinear system.

    For input u the exact equilibrium is x1_ss=u, x2_ss=u^2/b.  The test
    verifies the residual f(x_ss, u)=0, the state values, and that y_ss
    matches h(x_ss, u).
    """
    n, nu, ny, f, h, params, param_values = cascaded_nonlinear_system
    model = StateSpaceModelCT(f=f, h=h, n=n, nu=nu, ny=ny, params=params)
    solve_ss = make_steady_state_solver(model)

    u_val = 2.0
    u = np.array([u_val])
    x0 = np.zeros(n)
    x_ss, y_ss = solve_ss(x0, u, param_values)

    b = param_values["b"]
    assert np.allclose(x_ss[0], u_val, atol=1e-8)
    assert np.allclose(x_ss[1], u_val**2 / b, atol=1e-8)

    residual = np.array(f(0, x_ss, u, *param_values.values())).flatten()
    assert np.allclose(residual, np.zeros(n), atol=1e-8)

    y_check = np.array(h(0, x_ss, u, *param_values.values())).flatten()
    assert np.allclose(y_ss, y_check, atol=1e-10)
    assert np.allclose(y_ss[0], u_val**2 / b, atol=1e-8)


def test_make_steady_state_solver_warm_start(cascaded_nonlinear_system):
    """Solver tracks the SS correctly across a sweep of inputs using warm-starting.

    Each solve is initialised from the previous solution, matching the
    typical usage pattern documented in the function's Notes section.
    """
    n, nu, ny, f, h, params, param_values = cascaded_nonlinear_system
    model = StateSpaceModelCT(f=f, h=h, n=n, nu=nu, ny=ny, params=params)
    solve_ss = make_steady_state_solver(model)

    b = param_values["b"]
    x_prev = np.zeros(n)

    for u_val in [1.0, 2.0, 3.0]:
        u = np.array([u_val])
        x_ss, y_ss = solve_ss(x_prev, u, param_values)

        assert np.allclose(x_ss[0], u_val, atol=1e-8)
        assert np.allclose(x_ss[1], u_val**2 / b, atol=1e-8)
        assert np.allclose(y_ss[0], u_val**2 / b, atol=1e-8)

        residual = np.array(f(0, x_ss, u, *param_values.values())).flatten()
        assert np.allclose(residual, np.zeros(n), atol=1e-8)

        x_prev = x_ss
