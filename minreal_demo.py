# Is there an existing procedure in the slycot numerical package that performs
# the same function as minreal in Matlab?
# Yes, the routine is slycot.tb01pd. It wraps the underlying Fortran
# subroutine TB01PD from the SLICOT Library and serves as the computational
# engine behind Python Control's minreal function.
# How slycot.tb01pd Works
# • Reduction Modes (job). You can configure the routine to extract a
#   controllable realization (job='C'), an observable realization (job='O'),
#   or a fully minimal realization (job='M') by stripping both uncontrollable
#   and unobservable states.
# • Orthogonal Transformations. Like MATLAB's minreal, it applies orthogonal
#   similarity transformations to isolate and remove redundant states,
#   returning the reduced system matrix \(A_{r}\) in upper block Hessenberg
#   form.
# • Tolerance Control (tol). You can pass a custom numerical tolerance to catch near-uncontrollable or near-unobservable modes, or set tol <= 0 to let SLICOT compute a default machine-precision threshold automatically.


import control as ct
import numpy as np
import slycot

from cas_models.continuous_time.models import (
    SSModelCTDirectTransmission,
    SSModelCTFromABCDSISO,
)
from cas_models.transformations import (
    connect_feedback_system,
    connect_systems_in_parallel,
    connect_systems_in_series,
    linearize,
    negate_system,
)

# Example 1. state-space matrices (n=2 states, m=1 input, p=1 output)
A = np.array([[-1.0, 0.0], [0.0, -2.0]])
B = np.array([[1.0], [0.0]])  # Second state is uncontrollable
C = np.array([[1.0, 1.0]])

n, m, p = 2, 1, 1

# job='M' computes a minimal realization; tol=0.0 uses default tolerance
Ar, Br, Cr, nr = slycot.tb01pd(n, m, p, A, B, C, job="M", tol=0.0)

# Slice the returned matrices to the new reduced order (nr)
A_min = Ar[:nr, :nr]
B_min = Br[:nr, :m]
C_min = Cr[:p, :nr]

print("Example 1")
for label, (m1, m2) in {
    "A": (A, A_min),
    "B": (B, B_min),
    "C": (C, C_min),
}.items():
    print(f"{label}: {m1.shape} -> {m2.shape}")
print()

# Example 2: MATLAB's minreal example
# (https://www.mathworks.com/help/control/ref/dynamicsystem.minreal.html)
#   g = zpk([],1,1); h = tf([2 1],[1 0]);
#   cloop = inv(1+g*h) * g     % 3 states
#   cloopmin = minreal(cloop)  % s / (s^2 + s + 1), 1 state removed
# Built from casadi-models state-space models, so nothing cancels on the
# way (converting a transfer function with ct.ss() already gives a minimal
# realization, which is why this example first showed no reduction).

g = SSModelCTFromABCDSISO(1.0, 1.0, 1.0, 0.0, name="g")  # 1 / (s - 1)
h = SSModelCTFromABCDSISO(
    0.0, 1.0, 1.0, 2.0, name="h"
)  # (2s + 1) / s = 2 + 1/s
gh = connect_systems_in_series([h, g], name="gh")
T = connect_feedback_system(gh, name="T")  # gh / (1 + gh)
one = SSModelCTDirectTransmission(nu=1, name="one")
S = connect_systems_in_parallel(  # 1 - gh / (1 + gh) = (1 + gh)^-1
    [one, negate_system(T)], input_names=["r"], name="S"
)
cloop = connect_systems_in_series([S, g], verbose_names=True, name="cloop")

A, B, C, D = linearize(cloop)
n, m, p = A.shape[0], B.shape[1], C.shape[0]
print("Example 2")
print(
    f"cloop: {n} states {cloop.state_names}, eigenvalues {np.linalg.eigvals(A)}"
)

Ar, Br, Cr, nr = slycot.tb01pd(n, m, p, A, B, C, job="M", tol=0.0)
A_min, B_min, C_min = Ar[:nr, :nr], Br[:nr, :m], Cr[:p, :nr]
print(
    f"States reduced from {n} to {nr}, eigenvalues {np.linalg.eigvals(A_min)}"
)
print("Minimal closed-loop transfer function (MATLAB: s / (s^2 + s + 1)):")
print(ct.tf(ct.ss(A_min, B_min, C_min, D)))
