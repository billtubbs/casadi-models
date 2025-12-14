"""Unit tests for src/cas_models/transformations.py module"""

import pytest
import numpy as np
import casadi as cas
from cas_models.continuous_time.models import (
    StateSpaceModelCT,
    SSModelCTLinearFONoGainSISO,
    SSModelCTLinearFOSISO,
    is_ss_ct,
)
from cas_models.transformations import (
    connect_nonlinear_systems,
    connect_nonlinear_systems_in_parallel,
    connect_nonlinear_systems_in_series,
)


def test_connect_nonlinear_systems_in_parallel():
    sys1 = SSModelCTLinearFOSISO(
        input_name="u1", state_names=["x1"], output_name="y1"
    )
    sys2 = SSModelCTLinearFONoGainSISO(
        input_name="u2", state_names=["x2"], output_name="y2"
    )

    # With defaults - using new generalized function
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2], StateSpaceModelCT
    )

    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u[2],K,sys1_T1,sys2_T1)->(y[2]) SXFunction), "
        "n=2, nu=2, ny=2, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, "
        "name='sys1_sys2', "
        "input_names=['u1', 'u2'], state_names=['x1', 'x2'], "
        "output_names=['y1', 'y2'])"
    )

    # With custom keys
    sys3 = SSModelCTLinearFONoGainSISO(
        input_name="u3", state_names=["x3"], output_name="y3"
    )
    sys_combined = connect_nonlinear_systems_in_parallel(
        [sys1, sys2, sys3], StateSpaceModelCT, keys=["a", "b", "c"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[3],u[3],K,a_T1,b_T1,c_T1)->(rhs[3]) SXFunction), "
        "h=Function(h:(t,x[3],u[3],K,a_T1,b_T1,c_T1)->(y[3]) SXFunction), "
        "n=3, nu=3, ny=3, "
        "params={'K': SX(K), 'a_T1': SX(T1), 'b_T1': SX(T1), 'c_T1': SX(T1)}, "
        "name='sys1_sys2_sys3', "
        "input_names=['u1', 'u2', 'u3'], "
        "state_names=['x1', 'x2', 'x3'], "
        "output_names=['y1', 'y2', 'y3'])"
    )

    # With one constant and a shared parameter
    sys1 = SSModelCTLinearFOSISO(K=2)
    sys2 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys3 = SSModelCTLinearFONoGainSISO(T1=sys1.params["T1"])
    sys_combined = connect_nonlinear_systems_in_parallel(
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


def test_connect_nonlinear_systems_in_series():
    sys1 = SSModelCTLinearFOSISO(
        input_name="u1", state_names=["x1"], output_name="y1"
    )
    sys2 = SSModelCTLinearFONoGainSISO(
        input_name="u2", state_names=["x2"], output_name="y2"
    )

    # With defaults
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], StateSpaceModelCT
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,sys1_T1,sys2_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,sys1_T1,sys2_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'sys1_T1': SX(T1), 'sys2_T1': SX(T1)}, "
        "name='sys1_sys2', "
        "input_names=['u1'], state_names=['x2', 'x1'], "
        "output_names=['y2'])"
    )

    # With custom keys
    sys_combined = connect_nonlinear_systems_in_series(
        [sys1, sys2], StateSpaceModelCT, keys=["in", "out"]
    )
    assert str(sys_combined) == (
        "StateSpaceModelCT("
        "f=Function(f:(t,x[2],u,K,in_T1,out_T1)->(rhs[2]) SXFunction), "
        "h=Function(h:(t,x[2],u,K,in_T1,out_T1)->(y) SXFunction), "
        "n=2, nu=1, ny=1, "
        "params={'K': SX(K), 'in_T1': SX(T1), 'out_T1': SX(T1)}, "
        "name='sys1_sys2', "
        "input_names=['u1'], state_names=['x2', 'x1'], "
        "output_names=['y2'])"
    )

    # With verbose names
    sys_combined = connect_nonlinear_systems_in_series(
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
    sys_combined = connect_nonlinear_systems_in_series(
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


def test_connect_vs_series():
    """Test that connect gives same result as series for simple cascade."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Connect in series using connect_nonlinear_systems
    connected_via_connect = connect_nonlinear_systems(
        [sys1, sys2],
        connections={"sys2_u": "sys1_y"},  # sys1 output to sys2 input
        model_class=StateSpaceModelCT,
        input_names=["sys1_u"],
        output_names=["sys2_y"],
    )

    # Connect using series function
    connected_via_series = connect_nonlinear_systems_in_series(
        [sys1, sys2],
        model_class=StateSpaceModelCT,
    )

    # Should have same dimensions
    assert connected_via_connect.n == connected_via_series.n
    assert connected_via_connect.nu == connected_via_series.nu
    assert connected_via_connect.ny == connected_via_series.ny

    # Test with same inputs
    t_val = 0.0
    x_val = cas.DM.zeros(2, 1)
    u_val = cas.DM([1.0])

    y_connect = connected_via_connect.h(t_val, x_val, u_val)
    y_series = connected_via_series.h(t_val, x_val, u_val)

    assert np.allclose(np.array(y_connect), np.array(y_series))


def test_connect_connection_formats():
    """Test different connection format options."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Test list of tuples format
    connected_list = connect_nonlinear_systems(
        [sys1, sys2],
        connections=[("sys2_y", "sys1_u")],
        model_class=StateSpaceModelCT,
    )
    assert connected_list.n == 2
    assert connected_list.nu == 1
    assert "sys2_u" in connected_list.input_names

    # Test dictionary format
    connected_dict = connect_nonlinear_systems(
        [sys1, sys2],
        connections={"sys1_u": "sys2_y"},
        model_class=StateSpaceModelCT,
    )
    assert connected_dict.n == 2
    assert connected_dict.nu == 1
    assert "sys2_u" in connected_dict.input_names


def test_connect_summing_junction():
    """Test summing junction with multiple outputs to one input."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)

    # Test weighted sum with dict format
    connected_weighted = connect_nonlinear_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": {"sys2_y": 1.0, "sys3_y": -0.5},  # Weighted sum
        },
        model_class=StateSpaceModelCT,
    )
    assert connected_weighted.n == 3
    assert connected_weighted.nu == 2
    assert "sys1_u" not in connected_weighted.input_names

    # Test unit gains with list format
    connected_unit = connect_nonlinear_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": ["sys2_y", "sys3_y"],  # Sum with unit gains
        },
        model_class=StateSpaceModelCT,
    )
    assert connected_unit.n == 3
    assert connected_unit.nu == 2
    assert "sys1_u" not in connected_unit.input_names


def test_connect_trimming():
    """Test input/output trimming."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Connect with explicit input/output selection
    connected = connect_nonlinear_systems(
        [sys1, sys2],
        connections={"sys1_u": "sys2_y"},
        model_class=StateSpaceModelCT,
        input_names=["sys2_u"],  # Only expose sys2_u
        output_names=["sys1_y"],  # Only expose sys1_y
    )

    assert connected.nu == 1
    assert connected.ny == 1
    assert connected.input_names == ["sys2_u"]
    assert connected.output_names == ["sys1_y"]


def test_connect_feedback_and_empty():
    """Test feedback connections and edge cases."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Test no connections (parallel case)
    parallel = connect_nonlinear_systems(
        [sys1, sys2],
        connections=[],
        model_class=StateSpaceModelCT,
    )
    assert parallel.n == 2
    assert parallel.nu == 2  # All inputs external
    assert parallel.ny == 2

    # Test closed-loop feedback (all inputs connected)
    feedback_dict = connect_nonlinear_systems(
        [sys1, sys2],
        connections={"sys1_u": "sys2_y", "sys2_u": "sys1_y"},
        model_class=StateSpaceModelCT,
    )
    assert feedback_dict.n == 2
    assert feedback_dict.nu == 0  # No external inputs
    assert feedback_dict.ny == 2
    assert feedback_dict.input_names == []

    # Test feedback with list format
    feedback_list = connect_nonlinear_systems(
        [sys1, sys2],
        connections=[("sys2_y", "sys1_u"), ("sys1_y", "sys2_u")],
        model_class=StateSpaceModelCT,
    )
    assert feedback_list.n == 2
    assert feedback_list.nu == 0
    assert is_ss_ct(feedback_list) is True


def test_connect_error_handling():
    """Test error handling for invalid connections."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)

    # Test error on non-existent input
    with pytest.raises(
        ValueError, match="Connection target input 'sys3_u' not found"
    ):
        connect_nonlinear_systems(
            [sys1, sys2],
            connections={"sys3_u": "sys1_y"},
            model_class=StateSpaceModelCT,
        )

    # Test error on non-existent output
    with pytest.raises(ValueError, match="Connection source output 'sys3_y'"):
        connect_nonlinear_systems(
            [sys1, sys2],
            connections={"sys1_u": "sys3_y"},
            model_class=StateSpaceModelCT,
        )

    # Test error on duplicate connections in list format
    with pytest.raises(
        ValueError, match="Duplicate connection target 'sys1_u'"
    ):
        connect_nonlinear_systems(
            [sys1, sys2],
            connections=[
                ("sys2_y", "sys1_u"),
                ("sys1_y", "sys1_u"),  # Duplicate target
            ],
            model_class=StateSpaceModelCT,
        )


def test_connect_complex_example():
    """Complex example combining multiple connection features."""
    sys1 = SSModelCTLinearFOSISO(K=1.0, T1=1.0)
    sys2 = SSModelCTLinearFOSISO(K=2.0, T1=0.5)
    sys3 = SSModelCTLinearFOSISO(K=0.5, T1=0.25)

    # Complex connections with summing junction, feedback, and I/O trimming
    connected = connect_nonlinear_systems(
        [sys1, sys2, sys3],
        connections={
            "sys1_u": {"sys2_y": 1.0, "sys3_y": -0.5},  # Summing junction
            "sys2_u": "sys1_y",  # Feedback from sys1
        },
        model_class=StateSpaceModelCT,
        input_names=["sys3_u"],  # Only sys3 input is external
        output_names=["sys1_y", "sys2_y"],  # Expose sys1 and sys2 outputs
    )

    assert connected.n == 3  # All three state variables
    assert connected.nu == 1  # Only sys3_u is external
    assert connected.ny == 2  # sys1_y and sys2_y
    assert connected.input_names == ["sys3_u"]
    assert connected.output_names == ["sys1_y", "sys2_y"]
