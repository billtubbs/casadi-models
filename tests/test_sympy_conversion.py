import pytest
import numpy as np
import casadi as cas

from scipy import signal
from sympy import Matrix
from sympy.physics.control.lti import StateSpace
from cas_models.sympy_conversion import (
    sympy2casadi,
    make_casadi_and_sympy_vars,
    convert_sympy_state_space_to_casadi_SX,
)
from cas_models.continuous_time.models import SSModelCTFromSympySS
from cas_models.continuous_time.simulate import (
    make_sim_step_function_integrator,
)


# TODO: Add more complex tests later
@pytest.fixture
def RLC_variables():
    # Scalar variables
    var_names = {"R": (), "L": (), "C": ()}
    sympy_vars, casadi_vars = make_casadi_and_sympy_vars(var_names)
    return sympy_vars, casadi_vars


@pytest.fixture
def RLC_circuit_ABCD(RLC_variables):
    """State space matrices for RLC circuit.
    See: https://docs.sympy.org/latest/tutorials/physics/control/
         electrical_problems.html
    """
    sympy_vars, casadi_vars = RLC_variables

    R = sympy_vars["R"]
    L = sympy_vars["L"]
    C = sympy_vars["C"]
    A_sym = Matrix([[-R / L, -1 / L], [1 / C, 0]])
    B_sym = Matrix([[1 / L], [0]])
    C_sym = Matrix([[0, 1]])
    D_sym = Matrix([[0]])

    return sympy_vars, casadi_vars, A_sym, B_sym, C_sym, D_sym


def test_sympy2casadi(RLC_variables):
    sympy_vars, casadi_vars = RLC_variables
    R = sympy_vars["R"]
    L = sympy_vars["L"]
    C = sympy_vars["C"]

    sympy_expr = R / L
    result = sympy2casadi(
        sympy_expr, sympy_vars.values(), casadi_vars.values()
    )
    assert repr(result) == "SX((R/L))"

    sympy_expr = Matrix([[-R / L, -1 / L], [1 / C, 0]])
    result = sympy2casadi(
        sympy_expr, sympy_vars.values(), casadi_vars.values()
    )
    assert str(result) == ("\n[[(-(R/L)), (-1/L)], \n [(1./C), 00]]")
    result = sympy2casadi(
        sympy_expr, sympy_vars.values(), casadi_vars.values(), sparsify=False
    )
    assert str(result) == ("\n[[(-(R/L)), (-1/L)], \n [(1./C), 0]]")


def test_convert_sympy_state_space_to_casadi_SX(RLC_circuit_ABCD):
    sympy_vars, casadi_vars, A_sym, B_sym, C_sym, D_sym = RLC_circuit_ABCD

    ss_sympy = StateSpace(A_sym, B_sym, C_sym, D_sym)

    A_cas, B_cas, C_cas, D_cas = convert_sympy_state_space_to_casadi_SX(
        ss_sympy, sympy_vars.values(), casadi_vars.values()
    )

    assert repr(A_cas) == "SX(\n[[(-(R/L)), (-1/L)], \n [(1./C), 00]])"
    assert repr(B_cas) == "SX([(1./L), 00])"
    assert repr(C_cas) == "DM([[00, 1]])"
    assert repr(D_cas) == "DM(00)"


def test_SSModelCTFromSympySS(RLC_circuit_ABCD):
    _, _, A_sym, B_sym, C_sym, D_sym = RLC_circuit_ABCD

    ss_sympy = StateSpace(A_sym, B_sym, C_sym, D_sym)

    ss_cas = SSModelCTFromSympySS(
        ss_sympy,
        name="RLC_circuit",
        input_names=["V_in"],
        output_names=["V_out"],
    )

    assert str(ss_cas) == (
        "SSModelCTFromSympySS("
        "f=Function(f:(t,x[2],u,C,L,R)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,C,L,R)->(y) SXFunction), "
        "n=2, nu=1, ny=1, params={'C': SX(C), 'L': SX(L), 'R': SX(R)}, "
        "name='RLC_circuit', input_names=['V_in'], "
        "state_names=['x1', 'x2'], output_names=['V_out'])"
    )


def test_SSModelCTFromSympySS_simulation_comparison(RLC_circuit_ABCD):
    """Compare CasADi simulation with SciPy simulation for 10 time steps."""
    _, _, A_sym, B_sym, C_sym, D_sym = RLC_circuit_ABCD

    # Numerical parameter values
    param_values = {"R": 1.0, "L": 0.5, "C": 2.0}

    # Create numerical matrices from SymPy
    A_num = np.array(A_sym.subs(param_values)).astype(float)
    B_num = np.array(B_sym.subs(param_values)).astype(float)
    C_num = np.array(C_sym.subs(param_values)).astype(float)
    D_num = np.array(D_sym.subs(param_values)).astype(float)

    # Create SciPy state-space model
    sys_scipy = signal.StateSpace(A_num, B_num, C_num, D_num)

    # Create CasADi model from SymPy
    ss_sympy = StateSpace(A_sym, B_sym, C_sym, D_sym)
    ss_cas = SSModelCTFromSympySS(
        ss_sympy,
        name="RLC_circuit",
        input_names=["V_in"],
        output_names=["V_out"],
    )

    # Create simulation step function for CasADi model
    F_cas = make_sim_step_function_integrator(
        ss_cas.f, ss_cas.n, ss_cas.nu, params=ss_cas.params, name="F"
    )

    # Simulation parameters
    n_steps = 10
    dt = 0.1
    x0 = np.array([0.0, 0.0])  # Initial state

    # Create time vector and step input
    # Input is zero for k=0,1, then steps to 1.0 at k=2
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    U = np.zeros(n_steps + 1)
    U[2:] = 1.0  # Step at k=2

    # SciPy simulation using lsim with zero-order hold (ZOH) on inputs
    t_scipy, y_scipy, X_scipy = signal.lsim(
        sys_scipy, U, t, X0=x0, interp=False
    )
    assert np.allclose(t_scipy, t)
    Y_scipy = y_scipy.reshape(-1, 1)  # Ensure 2D array for outputs

    # Extract parameter values in the order expected by the functions
    param_vals = [param_values[key] for key in ss_cas.params.keys()]

    # CasADi simulation
    X_cas = []
    Y_cas = []
    xk = cas.DM(x0)
    for k in range(n_steps + 1):
        tk = t[k]

        # Get current input value
        uk = U[k]

        # Compute output
        yk = ss_cas.h(tk, xk, uk, *param_vals)

        X_cas.append(xk.full().flatten())
        Y_cas.append(yk.full().flatten())

        # Integrate one step
        xk = F_cas(t[k], xk, uk, dt, *param_vals)

    # Convert results to arrays
    X_cas = np.stack(X_cas)
    Y_cas = np.stack(Y_cas)
    assert X_cas.shape == X_scipy.shape
    assert Y_cas.shape == Y_scipy.shape

    # Compare state trajectories
    np.testing.assert_allclose(
        X_cas,
        X_scipy,
        atol=1e-5,
        err_msg="State trajectories don't match between CasADi and SciPy",
    )

    # Compare output trajectories
    np.testing.assert_allclose(
        Y_cas,
        Y_scipy,
        atol=1e-5,
        err_msg="Output trajectories don't match between CasADi and SciPy",
    )
