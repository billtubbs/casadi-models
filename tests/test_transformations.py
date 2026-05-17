"""Unit tests for src/cas_models/transformations.py module"""

import pytest
import numpy as np
import casadi as cas
from cas_models.continuous_time.models import (
    StateSpaceModelCT,
    SSModelCTDirectTransmission,
    SSModelCTLinearFONoGainSISO,
    SSModelCTLinearFOSISO,
    is_ss_ct,
)
from cas_models.continuous_time.regulators import SSModelCTPIInt
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTFromCT,
    StateSpaceModelDTTFSISO,
    is_ss_dt,
)
from cas_models.discrete_time.simulate import (
    make_n_step_simulation_function_from_model,
)
from cas_models.transformations import (
    connect_feedback_system,
    connect_systems,
    connect_systems_in_parallel,
    connect_systems_in_series,
)


# Fixtures for continuous-time systems
@pytest.fixture
def ct_sys1_with_gain():
    """CT system with gain (K, T1)"""
    return SSModelCTLinearFOSISO(
        input_name="u1", state_names=["x1"], output_name="y1"
    )


@pytest.fixture
def ct_sys2_no_gain():
    """CT system without gain (T1 only)"""
    return SSModelCTLinearFONoGainSISO(
        input_name="u2", state_names=["x2"], output_name="y2"
    )


@pytest.fixture
def ct_sys3_no_gain():
    """CT system without gain (T1 only)"""
    return SSModelCTLinearFONoGainSISO(
        input_name="u3", state_names=["x3"], output_name="y3"
    )


# Fixtures for discrete-time systems
@pytest.fixture
def dt_sys1_with_gain():
    """DT system: G1(z) = K / (z - a) with K and a as parameters"""
    # Transfer function: K / (z - a) => num = [0, K], den = [1, -a]
    K = cas.SX.sym("K")
    a = cas.SX.sym("a")
    return StateSpaceModelDTTFSISO(
        num=cas.vertcat(0, K),
        den=cas.vertcat(1, -a),
        input_name="u1",
        state_names=["x1"],
        output_name="y1",
    )


@pytest.fixture
def dt_sys2_no_gain():
    """DT system: G2(z) = 1 / (z - b) with b as parameter"""
    b = cas.SX.sym("b")
    return StateSpaceModelDTTFSISO(
        num=cas.vertcat(0, 1),
        den=cas.vertcat(1, -b),
        input_name="u2",
        state_names=["x2"],
        output_name="y2",
    )


@pytest.fixture
def dt_sys3_no_gain():
    """DT system: G3(z) = 1 / (z - c) with c as parameter"""
    c = cas.SX.sym("c")
    return StateSpaceModelDTTFSISO(
        num=cas.vertcat(0, 1),
        den=cas.vertcat(1, -c),
        input_name="u3",
        state_names=["x3"],
        output_name="y3",
    )


# Test data for parameterized tests
@pytest.fixture(params=["ct", "dt"])
def system_type(request):
    """Fixture to parameterize tests for both CT and DT systems"""
    return request.param


@pytest.fixture
def sys1_with_gain(system_type, ct_sys1_with_gain, dt_sys1_with_gain):
    """Return CT or DT system based on parameterization"""
    if system_type == "ct":
        return ct_sys1_with_gain
    else:
        return dt_sys1_with_gain


@pytest.fixture
def sys2_no_gain(system_type, ct_sys2_no_gain, dt_sys2_no_gain):
    """Return CT or DT system based on parameterization"""
    if system_type == "ct":
        return ct_sys2_no_gain
    else:
        return dt_sys2_no_gain


@pytest.fixture
def sys3_no_gain(system_type, ct_sys3_no_gain, dt_sys3_no_gain):
    """Return CT or DT system based on parameterization"""
    if system_type == "ct":
        return ct_sys3_no_gain
    else:
        return dt_sys3_no_gain


@pytest.fixture
def model_class(system_type):
    """Return appropriate model class based on system type"""
    return StateSpaceModelCT if system_type == "ct" else StateSpaceModelDT


@pytest.fixture
def expected_parallel_default(system_type):
    """Expected string representation for parallel connection with defaults"""
    if system_type == "ct":
        return (
            "StateSpaceModelCT("
            "f=Function(f:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
            "h=Function(h:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(y[2]) SXFunction), "
            "n=2, nu=2, ny=2, "
            "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, "
            "name='sys1_sys2', "
            "input_names=['u1', 'u2'], state_names=['x1', 'x2'], "
            "output_names=['y1', 'y2'])"
        )
    else:
        # DT systems: K, a, b don't conflict, so no sys1_/sys2_ prefix
        # dt=None is valid - indicates DT model with unspecified time step
        return (
            "StateSpaceModelDT("
            "F=Function(F:(t,xk[2],uk[2],K,a,b)->(xkp1[2]) SXFunction), "
            "H=Function(H:(t,xk[2],uk[2],K,a,b)->(yk[2]) SXFunction), "
            "n=2, nu=2, ny=2, dt=None, "
            "params={'K': SX(K), 'a': SX(a), 'b': SX(b)}, "
            "name='sys1_sys2', "
            "input_names=['u1', 'u2'], state_names=['x1', 'x2'], "
            "output_names=['y1', 'y2'])"
        )


def test_connect_nonlinear_systems_in_parallel(
    system_type,
    sys1_with_gain,
    sys2_no_gain,
    sys3_no_gain,
    model_class,
    expected_parallel_default,
):
    """Test parallel connection for both CT and DT systems"""
    # With defaults - using new generalized function
    sys_combined = connect_systems_in_parallel(
        [sys1_with_gain, sys2_no_gain], model_class
    )

    assert str(sys_combined) == expected_parallel_default

    # With custom keys
    sys_combined = connect_systems_in_parallel(
        [sys1_with_gain, sys2_no_gain, sys3_no_gain],
        model_class,
        keys=["a", "b", "c"],
    )

    # Verify basic properties
    assert sys_combined.n == 3
    assert sys_combined.nu == 3
    assert sys_combined.ny == 3
    assert sys_combined.input_names == ["u1", "u2", "u3"]
    assert sys_combined.state_names == ["x1", "x2", "x3"]
    assert sys_combined.output_names == ["y1", "y2", "y3"]

    # With one constant and a shared parameter (only for CT)
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=2)
        sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
        sys3 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
        sys_combined = connect_systems_in_parallel(
            [sys1, sys2, sys3], StateSpaceModelCT, keys=["a", "b", "c"]
        )
        assert str(sys_combined) == (
            "StateSpaceModelCT("
            "f=Function(f:(t,x[3],u[3],T1)->(rhs[3]) SXFunction), "
            "h=Function(h:(t,x[3],u[3],T1)->(y[3]) SXFunction), "
            "n=3, nu=3, ny=3, params={'T1': SX(T1)}, name='sys1_sys2_sys3', "
            "input_names=['a_u', 'b_u', 'c_u'], "
            "state_names=['a_x', 'b_x', 'c_x'], "
            "output_names=['a_y', 'b_y', 'c_y'])"
        )


def test_connect_nonlinear_systems_in_series(
    system_type, sys1_with_gain, sys2_no_gain, model_class
):
    """Test series connection for both CT and DT systems"""
    # With defaults
    sys_combined = connect_systems_in_series(
        [sys1_with_gain, sys2_no_gain], model_class
    )

    # Verify basic properties
    assert sys_combined.n == 2
    assert sys_combined.nu == 1
    assert sys_combined.ny == 1
    assert sys_combined.input_names == ["u1"]
    assert "x1" in sys_combined.state_names
    assert "x2" in sys_combined.state_names

    # With custom keys
    sys_combined = connect_systems_in_series(
        [sys1_with_gain, sys2_no_gain], model_class, keys=["in", "out"]
    )
    assert sys_combined.n == 2
    assert sys_combined.nu == 1
    assert sys_combined.ny == 1
    assert sys_combined.input_names == ["u1"]

    # With verbose names (only test CT as DT params are different)
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(
            input_name="u1", state_names=["x1"], output_name="y1"
        )
        sys2 = SSModelCTLinearFONoGainSISO(
            input_name="u2", state_names=["x2"], output_name="y2"
        )
        sys_combined = connect_systems_in_series(
            [sys1, sys2],
            StateSpaceModelCT,
            keys=["in", "out"],
            verbose_names=True,
        )
        assert str(sys_combined) == (
            "StateSpaceModelCT("
            "f=Function(f:(t,x[2],u,in_K,in_T1,out_T1)->(rhs[2]) SXFunction), "
            "h=Function(h:(t,x[2],u,in_K,in_T1,out_T1)->(y) SXFunction), "
            "n=2, nu=1, ny=1, "
            "params={'in_K': SX(K), 'in_T1': SX(T1), 'out_T1': SX(T1)}, "
            "name='sys1_sys2', "
            "input_names=['in_u1'], state_names=['out_x2', 'in_x1'], "
            "output_names=['out_y2'])"
        )

        # With one constant and a shared parameter
        sys1 = SSModelCTLinearFOSISO(K=2)
        sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
        sys_combined = connect_systems_in_series(
            [sys1, sys2],
            StateSpaceModelCT,
            keys=["in", "out"],
            verbose_names=True,
        )
        assert str(sys_combined) == (
            "StateSpaceModelCT("
            "f=Function(f:(t,x[2],u,in_out_T1)->(rhs[2]) SXFunction), "
            "h=Function(h:(t,x[2],u,in_out_T1)->(y) SXFunction), "
            "n=2, nu=1, ny=1, "
            "params={'in_out_T1': SX(T1)}, name='sys1_sys2', "
            "input_names=['in_u'], state_names=['out_x', 'in_x'], "
            "output_names=['out_y'])"
        )


def test_connect_vs_series(system_type, model_class):
    """Test that connect gives same result as series for simple cascade."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )

    # Connect in series using connect_nonlinear_systems
    connected_via_connect = connect_systems(
        [sys1, sys2],
        connections={"sys2_u": "sys1_y"},  # sys1 output to sys2 input
        model_class=model_class,
        input_names=["sys1_u"],
        output_names=["sys2_y"],
    )

    # Connect using series function
    connected_via_series = connect_systems_in_series(
        [sys1, sys2],
        model_class=model_class,
    )

    # Should have same dimensions
    assert connected_via_connect.n == connected_via_series.n
    assert connected_via_connect.nu == connected_via_series.nu
    assert connected_via_connect.ny == connected_via_series.ny

    # Test with same inputs
    t_val = 0.0
    x_val = cas.DM.zeros(2, 1)
    u_val = cas.DM([1.0])

    if system_type == "ct":
        y_connect = connected_via_connect.h(t_val, x_val, u_val)
        y_series = connected_via_series.h(t_val, x_val, u_val)
    else:
        y_connect = connected_via_connect.H(t_val, x_val, u_val)
        y_series = connected_via_series.H(t_val, x_val, u_val)

    assert np.allclose(np.array(y_connect), np.array(y_series))


def test_connect_connection_formats(system_type, model_class):
    """Test different connection format options."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )

    # Test list of tuples format
    connected_list = connect_systems(
        [sys1, sys2],
        connections=[("sys2_y", "sys1_u")],
        model_class=model_class,
    )
    assert connected_list.n == 2
    assert connected_list.nu == 1
    assert "sys2_u" in connected_list.input_names

    # Test dictionary format
    connected_dict = connect_systems(
        [sys1, sys2],
        connections={"sys1_u": "sys2_y"},
        model_class=model_class,
    )
    assert connected_dict.n == 2
    assert connected_dict.nu == 1
    assert "sys2_u" in connected_dict.input_names


def test_connect_summing_junction(system_type, model_class):
    """Test summing junction with multiple outputs to one input."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
        sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )
        sys3 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 0.5]), den=cas.DM([1, -0.2])
        )

    # Test weighted sum with dict format
    connected_weighted = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": {"sys2_y": 1.0, "sys3_y": -0.5},  # Weighted sum
        },
        model_class=model_class,
    )
    assert connected_weighted.n == 3
    assert connected_weighted.nu == 2
    assert "sys1_u" not in connected_weighted.input_names

    # Test unit gains with list format
    connected_unit = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": ["sys2_y", "sys3_y"],  # Sum with unit gains
        },
        model_class=model_class,
    )
    assert connected_unit.n == 3
    assert connected_unit.nu == 2
    assert "sys1_u" not in connected_unit.input_names


def test_connect_trimming(system_type, model_class):
    """Test input/output trimming."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )

    # Connect with explicit input/output selection
    connected = connect_systems(
        [sys1, sys2],
        connections={"sys1_u": "sys2_y"},
        model_class=model_class,
        input_names=["sys2_u"],  # Only expose sys2_u
        output_names=["sys1_y"],  # Only expose sys1_y
    )

    assert connected.nu == 1
    assert connected.ny == 1
    assert connected.input_names == ["sys2_u"]
    assert connected.output_names == ["sys1_y"]


def test_connect_feedback_and_empty(system_type, model_class):
    """Test feedback connections and edge cases."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )

    # Test no connections (parallel case)
    parallel = connect_systems(
        [sys1, sys2],
        connections=[],
        model_class=model_class,
    )
    assert parallel.n == 2
    assert parallel.nu == 2  # All inputs external
    assert parallel.ny == 2

    # Test closed-loop feedback (all inputs connected)
    feedback_dict = connect_systems(
        [sys1, sys2],
        connections={"sys1_u": "sys2_y", "sys2_u": "sys1_y"},
        model_class=model_class,
    )
    assert feedback_dict.n == 2
    assert feedback_dict.nu == 0  # No external inputs
    assert feedback_dict.ny == 2
    assert feedback_dict.input_names == []

    # Test feedback with list format
    feedback_list = connect_systems(
        [sys1, sys2],
        connections=[("sys2_y", "sys1_u"), ("sys1_y", "sys2_u")],
        model_class=model_class,
    )
    assert feedback_list.n == 2
    assert feedback_list.nu == 0

    # Verify system type
    if system_type == "ct":
        assert is_ss_ct(feedback_list) is True
    else:
        assert is_ss_dt(feedback_list) is True


def test_connect_error_handling(system_type, model_class):
    """Test error handling for invalid connections."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )

    # Test error on non-existent input
    with pytest.raises(
        ValueError, match="Connection target input 'sys3_u' not found"
    ):
        connect_systems(
            [sys1, sys2],
            connections={"sys3_u": "sys1_y"},
            model_class=model_class,
        )

    # Test error on non-existent source (output or input)
    with pytest.raises(ValueError, match="Connection source 'sys3_y'"):
        connect_systems(
            [sys1, sys2],
            connections={"sys1_u": "sys3_y"},
            model_class=model_class,
        )

    # Test error on duplicate connections in list format
    with pytest.raises(
        ValueError, match="Duplicate connection target 'sys1_u'"
    ):
        connect_systems(
            [sys1, sys2],
            connections=[
                ("sys2_y", "sys1_u"),
                ("sys1_y", "sys1_u"),  # Duplicate target
            ],
            model_class=model_class,
        )


def test_connect_complex_example(system_type, model_class):
    """Complex example combining multiple connection features."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
        sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
        )
        sys3 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 0.5]), den=cas.DM([1, -0.2])
        )

    # Complex connections with summing junction, feedback, and I/O trimming
    connected = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": {"sys2_y": 1.0, "sys3_y": -0.5},  # Summing junction
            "sys2_u": "sys1_y",  # Feedback from sys1
        },
        model_class=model_class,
        input_names=["sys3_u"],  # Only sys3 input is external
        output_names=["sys1_y", "sys2_y"],  # Expose sys1 and sys2 outputs
    )

    assert connected.n == 3  # All three state variables
    assert connected.nu == 1  # Only sys3_u is external
    assert connected.ny == 2  # sys1_y and sys2_y
    assert connected.input_names == ["sys3_u"]
    assert connected.output_names == ["sys1_y", "sys2_y"]


def test_connect_input_to_input(system_type, model_class):
    """Test input-to-input connections (external inputs on RHS)."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0, name="sys1")
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5, name="sys2")
        sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25, name="sys3")
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5]), name="sys1"
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3]), name="sys2"
        )
        sys3 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 0.5]), den=cas.DM([1, -0.2]), name="sys3"
        )

    # Test simple input-to-input connection
    # sys1_u gets its value from external input sys2_u
    connected = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": "sys2_u",  # Input from another external input
        },
        model_class=model_class,
    )
    assert connected.n == 3
    assert connected.nu == 2  # sys2_u and sys3_u are external
    assert "sys2_u" in connected.input_names
    assert "sys3_u" in connected.input_names
    assert "sys1_u" not in connected.input_names  # Connected

    # Test input-to-input with summing junction
    # sys1_u gets sum of two external inputs
    connected = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": ["sys2_u", "sys3_u"],  # Sum of two inputs
        },
        model_class=model_class,
    )
    assert connected.n == 3
    assert connected.nu == 2
    assert "sys2_u" in connected.input_names
    assert "sys3_u" in connected.input_names
    assert "sys1_u" not in connected.input_names

    # Test weighted sum of inputs
    connected = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": {"sys2_u": 1.0, "sys3_u": -0.5},  # Weighted sum
        },
        model_class=model_class,
    )
    assert connected.n == 3
    assert connected.nu == 2
    assert "sys1_u" not in connected.input_names

    # Test mixed output and input sources
    connected = connect_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": {"sys2_y": 1.0, "sys3_u": 0.5},  # Mix output & input
            "sys2_u": "sys3_u",  # Input from external input
        },
        model_class=model_class,
    )
    assert connected.n == 3
    assert connected.nu == 1  # Only sys3_u is external
    assert connected.input_names == ["sys3_u"]

    # Verify the functions actually execute
    t_val = 0.0
    x_val = cas.DM.zeros(3, 1)
    u_val = cas.DM([1.0])

    if system_type == "ct":
        y_result = connected.h(t_val, x_val, u_val)
        f_result = connected.f(t_val, x_val, u_val)
    else:
        y_result = connected.H(t_val, x_val, u_val)
        f_result = connected.F(t_val, x_val, u_val)

    assert y_result.shape == (3, 1)  # All outputs
    assert f_result.shape == (3, 1)  # All states


def test_connect_input_to_input_errors(system_type, model_class):
    """Test error handling for invalid input-to-input connections."""
    if system_type == "ct":
        sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0, name="sys1")
        sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5, name="sys2")
    else:
        sys1 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5]), name="sys1"
        )
        sys2 = StateSpaceModelDTTFSISO(
            num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3]), name="sys2"
        )

    # Test error when trying to use a connected input as a source
    # sys1_u is connected, so it can't be used as a source for sys2_u
    with pytest.raises(
        ValueError,
        match="Connection source input 'sys1_u' .* must be an external input",
    ):
        connect_systems(
            [sys1, sys2],
            connections={
                "sys1_u": "sys2_y",  # sys1_u is now connected
                "sys2_u": "sys1_u",  # ERROR: Can't use sys1_u as source
            },
            model_class=model_class,
        )

    # Test error when source input doesn't exist
    with pytest.raises(
        ValueError,
        match="Connection source 'sys3_u' .* not found",
    ):
        connect_systems(
            [sys1, sys2],
            connections={
                "sys1_u": "sys3_u",  # sys3_u doesn't exist
            },
            model_class=model_class,
        )


def test_mixed_ct_dt_systems_error():
    """Test that mixing CT and DT systems raises appropriate errors."""
    ct_sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    ct_sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    dt_sys1 = StateSpaceModelDTTFSISO(
        num=cas.DM([0, 1.0]), den=cas.DM([1, -0.5])
    )
    dt_sys2 = StateSpaceModelDTTFSISO(
        num=cas.DM([0, 2.0]), den=cas.DM([1, -0.3])
    )

    # Test parallel connection with mixed systems
    with pytest.raises(
        ValueError,
        match="Cannot combine systems with different types",
    ):
        connect_systems_in_parallel([ct_sys1, dt_sys1], StateSpaceModelCT)

    # Test series connection with mixed systems
    with pytest.raises(
        ValueError,
        match="Cannot combine systems with different types",
    ):
        connect_systems_in_series([ct_sys1, dt_sys1], StateSpaceModelCT)

    # Test general connection with mixed systems
    with pytest.raises(
        ValueError,
        match="Cannot combine systems with different types",
    ):
        connect_systems(
            [ct_sys1, dt_sys1],
            connections={"sys2_u": "sys1_y"},
            model_class=StateSpaceModelCT,
        )

    # Test with all DT systems but wrong model_class shouldn't raise in validation
    # (it would fail later during construction, but validation only checks compatibility)
    # Actually, let's verify CT systems also fail when mixed
    with pytest.raises(
        ValueError,
        match="Cannot combine systems with different types",
    ):
        connect_systems(
            [dt_sys1, ct_sys2],
            connections=[],
            model_class=StateSpaceModelDT,
        )


def test_connect_systems_algebraic_loop_silent_failure():
    """connect_systems silently returns an incorrect result for an algebraic loop.

    Two pure-gain systems connected in negative feedback form an algebraic
    loop: fwd_u depends on fbk_y, which depends on fwd_y, which depends on
    fwd_u — all instantaneously.

    Correct closed-loop gain: G_fwd / (1 + G_fwd * G_fbk) = 2 / (1 + 1) = 1.0
    Actual output with y_sp=1: 2.0 — fbk_y is treated as 0, breaking the loop.

    Crucially, no exception is raised either at construction time or when the
    output function is called with numeric values. The result is silently wrong.

    TODO: connect_systems should detect algebraic loops at construction time
    and raise a ValueError, as connect_feedback_system already does.
    """
    fwd = SSModelCTDirectTransmission(D=cas.SX([[2.0]]), name="fwd")
    fbk_gain = SSModelCTDirectTransmission(D=cas.SX([[0.5]]), name="fbk")

    # Construction succeeds without error despite the algebraic loop
    sys_cl = connect_systems(
        [fwd, fbk_gain],
        connections={
            "fwd_u": {"y_sp": 1.0, "fbk_y": -1.0},
            "fbk_u": "fwd_y",
        },
        model_class=StateSpaceModelCT,
        input_names=["y_sp"],
        output_names=["fwd_y"],
    )

    # Calling the output function with y_sp=1 also succeeds — no exception
    y = sys_cl.h(0.0, cas.DM.zeros(sys_cl.n), cas.DM([1.0]))

    # Result is wrong: 2.0 instead of the correct 1.0
    correct = 2.0 / (1.0 + 2.0 * 0.5)  # = 1.0
    assert float(y) != pytest.approx(correct)
    assert float(y) == pytest.approx(2.0)


# --- Tests for connect_feedback_system ---


def test_connect_feedback_system_unity_properties():
    """Unity negative feedback: structural properties (names, dimensions)."""
    plant = SSModelCTLinearFOSISO(K=2.0, T1=1.0, name="plant")
    sys_cl = connect_feedback_system(plant, model_class=StateSpaceModelCT)

    assert sys_cl.n == 1
    assert sys_cl.nu == 1
    assert sys_cl.ny == 1
    assert sys_cl.input_names == ["y_sp"]
    assert sys_cl.output_names == ["y"]
    assert sys_cl.state_names == ["x"]
    assert sys_cl.name == "fbk plant"
    assert is_ss_ct(sys_cl)


def test_connect_feedback_system_pi_plant():
    """PI controller + first-order plant — the README/notebook example."""
    plant = SSModelCTLinearFOSISO(K=1, T1=2, name="plant")
    ctrl = SSModelCTPIInt(Kc=1, Ti=2, name="ctrl")
    sys_cl = connect_feedback_system(
        ctrl * plant, model_class=StateSpaceModelCT
    )

    assert sys_cl.n == 2
    assert sys_cl.nu == 1
    assert sys_cl.ny == 1
    assert sys_cl.input_names == ["y_sp"]
    assert sys_cl.output_names == ["y"]
    assert "plant_x" in sys_cl.state_names
    assert "ctrl_x" in sys_cl.state_names
    assert repr(sys_cl) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u)->(y) SXFunction), "
        "n=2, nu=1, ny=1, params={}, name='fbk ctrl_plant', "
        "input_names=['y_sp'], state_names=['plant_x', 'ctrl_x'], "
        "output_names=['y'])"
    )


def test_connect_feedback_system_simulation():
    """Verify closed-loop steady-state via discrete-time simulation."""
    dt = 0.01

    # Unity feedback, K=2, T1=1 → closed-loop y_ss = K/(1+K) = 2/3 for r=1
    plant = SSModelCTLinearFOSISO(K=2.0, T1=1.0, name="plant")
    sys_cl = connect_feedback_system(plant, model_class=StateSpaceModelCT)
    sys_dt = StateSpaceModelDTFromCT(sys_cl, dt)
    nT = 1000
    simulate = make_n_step_simulation_function_from_model(sys_dt, nT)
    t = dt * np.arange(nT + 1)
    _, Y = simulate(t, np.ones((nT, 1)), np.zeros(sys_dt.n))
    assert abs(float(Y[-1]) - 2.0 / 3.0) < 1e-4

    # PI controller eliminates steady-state offset → y_ss → 1.0
    plant2 = SSModelCTLinearFOSISO(K=1, T1=2, name="plant")
    ctrl = SSModelCTPIInt(Kc=1, Ti=2, name="ctrl")
    sys_cl2 = connect_feedback_system(
        ctrl * plant2, model_class=StateSpaceModelCT
    )
    sys_dt2 = StateSpaceModelDTFromCT(sys_cl2, dt)
    nT2 = 2000
    simulate2 = make_n_step_simulation_function_from_model(sys_dt2, nT2)
    t2 = dt * np.arange(nT2 + 1)
    _, Y2 = simulate2(t2, np.ones((nT2, 1)), np.zeros(sys_dt2.n))
    assert abs(float(Y2[-1]) - 1.0) < 1e-3


def test_connect_feedback_system_custom_input_names():
    """Custom reference signal name via input_names."""
    plant = SSModelCTLinearFOSISO(K=1.0, T1=1.0, name="plant")
    sys_cl = connect_feedback_system(
        plant, model_class=StateSpaceModelCT, input_names=["r"]
    )
    assert sys_cl.input_names == ["r"]
    assert sys_cl.output_names == ["y"]


def test_connect_feedback_system_scalar_gain():
    """Non-unity scalar gain in the feedback path."""
    plant = SSModelCTLinearFOSISO(K=2.0, T1=1.0, name="plant")
    sys_cl = connect_feedback_system(
        plant, sys2=2.0, model_class=StateSpaceModelCT
    )

    assert sys_cl.n == 1
    assert sys_cl.nu == 1
    assert sys_cl.ny == 1
    assert sys_cl.input_names == ["y_sp"]
    assert sys_cl.name == "fbk plant gain"


def test_connect_feedback_system_errors():
    """Error handling for invalid arguments."""
    plant = SSModelCTLinearFOSISO(K=1.0, T1=1.0, name="plant")

    with pytest.raises(ValueError, match="model_class must be specified"):
        connect_feedback_system(plant, model_class=None)

    # sys2 dimension mismatch: use a two-output system as feedback for a SISO forward path
    plant_a = SSModelCTLinearFOSISO(K=1.0, T1=1.0, name="a")
    plant_b = SSModelCTLinearFOSISO(K=1.0, T1=1.0, name="b")
    mimo_feedback = connect_systems_in_parallel(
        [plant_a, plant_b], StateSpaceModelCT
    )  # ny=2, nu=2
    with pytest.raises(
        ValueError, match="sys2 must have same number of inputs"
    ):
        connect_feedback_system(
            plant, sys2=mimo_feedback, model_class=StateSpaceModelCT
        )


def test_connect_feedback_system_detects_algebraic_loop():
    """Raises ValueError when sys1 has direct feedthrough (algebraic loop).

    Two pure-gain systems in feedback form an algebraic loop: neither
    sys1 nor sys2 has any state, so the output always depends directly
    on the input and the closed-loop gain cannot be resolved symbolically.
    connect_feedback_system detects this and raises ValueError rather than
    silently returning an incorrect result.
    """
    fwd = SSModelCTDirectTransmission(D=cas.SX([[2.0]]), name="fwd")
    fbk = SSModelCTDirectTransmission(D=cas.SX([[0.5]]), name="fbk")

    with pytest.raises(ValueError, match="Algebraic loop"):
        connect_feedback_system(fwd, fbk, model_class=StateSpaceModelCT)

    # Unity feedback on a pure-gain forward path is also an algebraic loop
    with pytest.raises(ValueError, match="Algebraic loop"):
        connect_feedback_system(fwd, model_class=StateSpaceModelCT)

    # A plant with no direct feedthrough must NOT raise even in unity feedback
    plant = SSModelCTLinearFOSISO(K=2.0, T1=1.0, name="plant")
    sys_cl = connect_feedback_system(plant, model_class=StateSpaceModelCT)
    assert sys_cl.n == 1
