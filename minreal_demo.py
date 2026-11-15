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


import numpy as np
import control as ct
import slycot

# Example 1. state-space matrices (n=2 states, m=1 input, p=1 output)
A = np.array([[-1.0, 0.0], [0.0, -2.0]])
B = np.array([[1.0], [0.0]])  # Second state is uncontrollable
C = np.array([[1.0, 1.0]])

n, m, p = 2, 1, 1

# job='M' computes a minimal realization; tol=0.0 uses default tolerance
Ar, Br, Cr, nr = slycot.tb01pd(n, m, p, A, B, C, job='M', tol=0.0)

# Slice the returned matrices to the new reduced order (nr)
A_min = Ar[:nr, :nr]
B_min = Br[:nr, :m]
C_min = Cr[:p, :nr]

print("Example 1")
for label, (m1, m2) in {"A": (A, A_min), "B": (B, B_min), "C": (C, C_min)}.items():
    print(f"{label}: {m1.shape} -> {m2.shape}")
print()

# Example 2

# 1. Define the plant g = 1 / (s - 1) and controller h = (2s + 1) / s
g = ct.zpk([], [1], 1)
h = ct.tf([2, 1], [1, 0])

# 2. Form the non-minimal closed-loop transfer function
# Equivalent to MATLAB's: cloop = inv(1 + g*h) * g
# cloop_tf = ct.minreal(1 / (1 + g * h), tol=0) * g
# Or simply:
cloop_tf = (1 / (1 + g * h)) * g

print("Example 2")
print("Unreduced Closed-Loop Transfer Function:")
print(cloop_tf)

# 3. Convert to state-space so Slycot can process the matrices
cloop_ss = ct.ss(cloop_tf)
A, B, C, D = cloop_ss.A, cloop_ss.B, cloop_ss.C, cloop_ss.D
n, m, p = A.shape[0], B.shape[1], C.shape[0]

# 4. Run slycot.tb01pd directly to compute the minimal realization
# job='M' removes both uncontrollable and unobservable states
Ar, Br, Cr, nr = slycot.tb01pd(n, m, p, A, B, C, job='M', tol=0.0)

# 5. Slice the padded output arrays to the new reduced order (nr)
A_min = Ar[:nr, :nr]
B_min = Br[:nr, :m]
C_min = Cr[:p, :nr]

# 6. Reconstruct the minimal state-space model and convert back to a transfer function
cloop_min_ss = ct.ss(A_min, B_min, C_min, D)
cloop_min_tf = ct.tf(cloop_min_ss)

print(f"\nStates reduced from {n} to {nr}.")
print("\nMinimal Closed-Loop Transfer Function:")
print(cloop_min_tf)
