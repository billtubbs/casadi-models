"""Continuous-time regulator (controller) models.

Classes
-------
SSModelCTPIInt
    PI controller in interactive (series) form.
SSModelCTPIDInt
    PID controller in interactive (series) form.
"""

import casadi as cas

from cas_models.continuous_time.models import SSModelCTFromABCDSISO
from cas_models.param_utils import make_symbolic_vars_from_kwargs


class SSModelCTPIInt(SSModelCTFromABCDSISO):
    def __init__(
        self,
        Kc=None,
        Ti=None,
        input_name=None,
        state_names=None,
        output_name=None,
        name=None,
    ):
        """Construct a state space model of a continuous time
        proportional-integral (PI) controller.

        This PI is in interactive (series) form with the following
        transfer function:

                    Kc (Ti s + 1)
            Gc(s) = -------------
                        Ti s

        State-space realisation (state x = integral of error):

            dx/dt = e
                u = Kc * e + (Kc / Ti) * x

        which gives A=0, B=1, C=Kc/Ti, D=Kc.
        """
        if input_name is None:
            input_name = "e"
        if output_name is None:
            output_name = "u"
        params = make_symbolic_vars_from_kwargs(Kc=Kc, Ti=Ti)
        Kc = params["Kc"]
        Ti = params["Ti"]
        A = cas.sparsify(cas.SX(0))
        B = cas.SX(1)
        C = cas.SX(Kc / Ti)
        D = cas.SX(Kc)
        super().__init__(
            A,
            B,
            C,
            D,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )


class SSModelCTPIDInt(SSModelCTFromABCDSISO):
    def __init__(
        self,
        Kc=None,
        Ti=None,
        Td=None,
        Tf=None,
        input_name=None,
        state_names=None,
        output_name=None,
        name=None,
    ):
        """Construct a state space model of a continuous time
        proportional-integral-derivative (PID) controller.

        This PID is in interactive (series) form with the following
        transfer function:

                    Kc (Ti s + 1) (Td s + 1)
            Gc(s) = ------------------------
                       Ti s (Tf s + 1)

        where Tf is the derivative filter time constant.
        """
        if input_name is None:
            input_name = "e"
        if output_name is None:
            output_name = "u"
        params = make_symbolic_vars_from_kwargs(Kc=Kc, Ti=Ti, Td=Td, Tf=Tf)
        Kc = params["Kc"]
        Ti = params["Ti"]
        Td = params["Td"]
        Tf = params["Tf"]
        A = cas.sparsify(cas.blockcat([[0, 1], [0, -1 / Tf]]))
        B = cas.sparsify(cas.vertcat(0, 1))
        C = cas.horzcat(Kc, -Kc * Td * Ti**2 + Kc * Td + Kc * Ti)
        D = cas.SX(Kc * Td * Ti)
        super().__init__(
            A,
            B,
            C,
            D,
            name=name,
            input_name=input_name,
            state_names=state_names,
            output_name=output_name,
        )
