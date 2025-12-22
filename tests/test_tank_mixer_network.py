"""Test complex tank network with mixer connections.

This test validates that multi-stage connections work correctly:
- External inputs → tank flow outputs
- Tank flow outputs → mixer inputs
- Mixer outputs → tank inputs

This exercises the iterative connection-filling logic in connect_systems.
"""

import pytest
import numpy as np
import casadi as cas
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTFromCTRK4,
)
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.transformations import connect_systems


def test_four_tank_network_with_mixer():
    """Test a 4-tank network where tank outlets feed a mixer that feeds another tank.

    Network topology:
        Feed → Tank1 ──┬──> Tank2 ──> Mixer ──> Tank4
                       └──> Tank3 ──┘

    This tests:
    1. Input-to-input connections (tank_1_v_dot_out → tank_2/3_v_dot_in)
    2. Output-to-input connections for mixer flows (tank_2/3_v_dot_out → mixer)
    3. Output-to-input connections for mixer concentrations (tank_2/3_conc_out → mixer)
    4. Mixer output-to-tank input connections (mixer_v_dot_out → tank_4_v_dot_in)
    5. Steady-state mass and volume conservation
    """

    # Create simple tank model
    class SimpleTankDT(StateSpaceModelDTFromCTRK4):
        """Minimal mixing tank for testing"""

        def __init__(self, dt, A=1.0, name=None):
            # Create continuous-time model
            model_ct = SimpleTankCT(A=A, name=name)
            super().__init__(model_ct, dt)
            self.A = A

    class SimpleTankCT(StateSpaceModelCT):
        """Continuous-time tank with volume and mass states"""

        def __init__(self, A=1.0, name=None):
            self.A = A

            t = cas.SX.sym("t")
            x = cas.SX.sym("x", 2)  # [L, m]
            u = cas.SX.sym("u", 3)  # [v_dot_in, conc_in, v_dot_out]

            L, m = x[0], x[1]
            v_dot_in, conc_in, v_dot_out = u[0], u[1], u[2]

            # Dynamics
            dL_dt = (v_dot_in - v_dot_out) / A
            dm_dt = v_dot_in * conc_in - v_dot_out * (m / (A * L))

            rhs = cas.vertcat(dL_dt, dm_dt)
            f = cas.Function("f", [t, x, u], [rhs], ["t", "x", "u"], ["rhs"])

            # Outputs: [L, m, conc_out]
            conc_out = m / (A * L)
            y = cas.vertcat(L, m, conc_out)
            h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

            super().__init__(
                f=f,
                h=h,
                n=2,
                nu=3,
                ny=3,
                params=None,
                name=name,
                input_names=["v_dot_in", "conc_in", "v_dot_out"],
                state_names=["L", "m"],
                output_names=["L", "m", "conc_out"],
            )

    # Create stateless mixer
    class SimpleMixerDT(StateSpaceModelDT):
        """Stateless mixer combining two streams"""

        def __init__(self, dt, n_in=2, name=None):
            t = cas.SX.sym("t")
            xk = cas.SX.sym("xk", 0)  # No states
            uk = cas.SX.sym("uk", 2 * n_in)

            # Extract flows and concentrations
            flows = uk[0::2]  # Even indices
            concs = uk[1::2]  # Odd indices

            # Outputs
            v_dot_out = cas.sum1(flows)
            conc_out = cas.sum1(flows * concs) / v_dot_out

            F = cas.Function(
                "F", [t, xk, uk], [xk], ["t", "xk", "uk"], ["xkp1"]
            )
            H = cas.Function(
                "H",
                [t, xk, uk],
                [cas.vertcat(v_dot_out, conc_out)],
                ["t", "xk", "uk"],
                ["yk"],
            )

            input_names = []
            for i in range(n_in):
                input_names.extend([f"v_dot_in_{i + 1}", f"conc_in_{i + 1}"])

            super().__init__(
                F=F,
                H=H,
                n=0,
                nu=2 * n_in,
                ny=2,
                dt=dt,
                params=None,
                name=name,
                input_names=input_names,
                state_names=[],
                output_names=["v_dot_out", "conc_out"],
            )

    # Setup parameters
    dt = 1.0
    A = 20.0  # Tank area

    # Create systems
    systems = [
        SimpleTankDT(dt, A=A, name="tank_1"),
        SimpleTankDT(dt, A=A, name="tank_2"),
        SimpleTankDT(dt, A=A, name="tank_3"),
        SimpleTankDT(dt, A=A, name="tank_4"),
        SimpleMixerDT(dt, n_in=2, name="mixer"),
    ]

    # Define connections
    connections = {
        "tank_1_v_dot_out": [
            "tank_2_v_dot_in",
            "tank_3_v_dot_in",
        ],  # Input-to-input
        "tank_2_conc_in": "tank_1_conc_out",  # Output-to-input
        "tank_3_conc_in": "tank_1_conc_out",
        "mixer_v_dot_in_1": "tank_2_v_dot_out",  # External input-to-input
        "mixer_v_dot_in_2": "tank_3_v_dot_out",
        "mixer_conc_in_1": "tank_2_conc_out",  # Output-to-input (critical!)
        "mixer_conc_in_2": "tank_3_conc_out",
        "tank_4_v_dot_in": "mixer_v_dot_out",  # Mixer output-to-tank input
        "tank_4_conc_in": "mixer_conc_out",
    }

    # Connect systems
    connected = connect_systems(
        systems,
        connections,
        StateSpaceModelDT,
        name="tank_network",
        verbose_names=True,
    )

    # Verify external inputs
    assert connected.nu == 7
    assert set(connected.input_names) == {
        "tank_1_v_dot_in",
        "tank_1_conc_in",
        "tank_2_v_dot_in",
        "tank_2_v_dot_out",
        "tank_3_v_dot_in",
        "tank_3_v_dot_out",
        "tank_4_v_dot_out",
    }

    # Test steady-state operation
    density = 0.75  # tons/m³
    t = 0.0

    # States: all tanks at steady concentration
    xk = np.array(
        [
            1.0,
            density * A * 1.0,  # tank_1: L=1m, m consistent
            2.0,
            density * A * 2.0,  # tank_2
            3.0,
            density * A * 3.0,  # tank_3
            4.0,
            density * A * 4.0,  # tank_4
        ]
    )

    # Inputs: equal flows in/out for steady state
    uk = np.array(
        [
            2.0,  # tank_1_v_dot_in
            density,  # tank_1_conc_in
            1.0,  # tank_2_v_dot_in (half of tank_1 output)
            1.0,  # tank_2_v_dot_out
            1.0,  # tank_3_v_dot_in (other half)
            1.0,  # tank_3_v_dot_out
            2.0,  # tank_4_v_dot_out (mixer combines both streams)
        ]
    )

    # Compute next state
    xkp1 = connected.F(t, xk, uk)
    xkp1 = np.array(xkp1).flatten()

    # Check steady state: all states should be unchanged
    assert xkp1.shape == xk.shape
    np.testing.assert_allclose(
        xkp1,
        xk,
        rtol=1e-10,
        atol=1e-10,
        err_msg="States changed in steady-state operation",
    )

    # Verify outputs
    yk = connected.H(t, xk, uk)
    yk = np.array(yk).flatten()

    # Check mixer output flow
    mixer_v_dot_out_idx = connected.output_names.index("mixer_v_dot_out")
    assert yk[mixer_v_dot_out_idx] == pytest.approx(2.0)

    # Check mixer output concentration
    mixer_conc_out_idx = connected.output_names.index("mixer_conc_out")
    assert yk[mixer_conc_out_idx] == pytest.approx(density)

    # Check tank 4 outputs
    tank_4_L_idx = connected.output_names.index("tank_4_L")
    tank_4_m_idx = connected.output_names.index("tank_4_m")
    assert yk[tank_4_L_idx] == pytest.approx(4.0)
    assert yk[tank_4_m_idx] == pytest.approx(density * A * 4.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
