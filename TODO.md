# TODO

## Report `dlqe` inconsistencies to python-control

Raise an issue at <https://github.com/python-control/python-control/issues>
about `control.dlqe` (found with python-control 0.10.2 while writing
`steady_state_kf_gain` in `src/cas_models/discrete_time/kalman.py`).

Before posting:

- [ ] Search existing issues for `dlqe` (it may already be reported).
- [ ] Add the MATLAB version (and Octave / control package versions)
      used for the reference values.
- [ ] Run the MATLAB/Octave snippet below (not yet run as written; in
      MATLAB, replace `assert(a, b, tol)` with
      `assert(all(abs(a - b) < tol, 'all'))`).
- [ ] Consider two separate issues: the docstring bug (2) is clear-cut;
      the gain difference (1) may be treated as an intended convention.

### Notation

- Kf = P C' (C P C' + R)^-1: correction gain of the Kalman filter in
  filtering form, x(k|k) = x(k|k-1) + Kf (y(k) - C x(k|k-1)).
- Kp = A Kf: correction gain in prediction form,
  x(k+1|k) = A x(k|k-1) + B u(k) + Kp (y(k) - C x(k|k-1)).
- P = P(k+1|k), the solution of the discrete algebraic Riccati
  equation.

### Issue 1: the gain differs from MATLAB's and Octave's `dlqe`

For the same inputs `dlqe(A, G, C, Q, R)`:

| Output | python-control | MATLAB / Octave |
|---|---|---|
| Gain | `L` = Kp = A Kf | `M` = Kf |
| P | P(k+1\|k) | P(k+1\|k) (same) |
| Poles | `E` = eig(A - L C) | `E` = eig(A - A M C) (same) |
| P(k\|k) | not returned | `Z` |

- python-control's docstring defines `L` through
  x_e[n+1] = A x_e[n] + B u[n] + L (y[n] - C x_e[n] - D u[n]), so the
  code agrees with its own docs; the problem is MATLAB compatibility.
- `control.matlab.dlqe` is the same function as `control.dlqe`, so
  code ported from MATLAB silently gets A Kf instead of Kf.
- Recovering Kf from L needs A^-1, which doesn't exist if A has an
  eigenvalue at zero (e.g. delay states). Kf has to be computed from P.
- Reference values below: system adapted from the MATLAB `dlqe`
  documentation, confirmed with MATLAB (old unit test of
  `kalman_gain_ss.m`) and Octave's `dlqe` (M, P, Z and E all agree).

### Issue 2: the docstring's Riccati equation is the continuous-time one

The `dlqe` docstring says `P` solves

```text
A P + P A^T - (P C^T + G N) R^{-1}  (C P + N^T G^T) + G Q G^T = 0
```

which is the continuous-time equation, correct in `lqe` (line 99 on
main) and apparently copied into `dlqe`. The code is right: it calls
`dare(A.T, C.T, G @ QN @ G.T, RN)` and the returned `P` solves the
discrete-time equation.

- Main (commit `bfe9380`):
  <https://github.com/python-control/python-control/blob/bfe9380eedcfbc6e57c0a7117e77e633575e3a18/control/stochsys.py#L235-L239>
- Release 0.10.2:
  <https://github.com/python-control/python-control/blob/0.10.2/control/stochsys.py#L229-L233>

Suggested replacement (same `= 0` style, keeping the N term):

```text
A P A^T - P - (A P C^T + G N) (C P C^T + R)^{-1} (C P A^T + N^T G^T) + G Q G^T = 0
```

Also suggest:

- Stating the gain explicitly in the Returns section,
  `L = (A P C^T + G N) (C P C^T + R)^{-1}`, with a note that MATLAB's
  `dlqe` returns M = P C^T (C P C^T + R)^{-1}.
- Using the parameter names `QN`, `RN`, `NN` in the equations (both
  `lqe` and `dlqe` use `Q`, `R`, `N`).

### Python snippet (all asserts pass with python-control 0.10.2)

```python
import numpy as np
import control

A = np.array([[-0.9, 0.7], [-0.3, 0.1]])
G = np.eye(2)
C = np.array([[1.0, 1.0]])
Q = np.diag([1.0, 3.0])
R = np.array([[0.1]])

L, P, E = control.dlqe(A, G, C, Q, R)

# Kalman filter (filtering form) gain, returned by MATLAB/Octave dlqe as M
M = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
assert np.allclose(M, [[0.5716], [0.4184]], atol=5e-5)  # MATLAB/Octave M

# 1. control.dlqe returns L = A M (prediction form gain), not M
assert not np.allclose(L, M)
assert np.allclose(L, A @ M)
assert np.allclose(np.sort(E), np.sort(np.linalg.eigvals(A - L @ C)))

# 2. P solves the discrete-time Riccati equation ...
assert np.allclose(
    A @ P @ A.T - A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + R) @ C @ P @ A.T
    + G @ Q @ G.T,
    P,
)
# ... not the continuous-time one given in the dlqe docstring
assert not np.allclose(
    A @ P + P @ A.T - P @ C.T @ np.linalg.inv(R) @ C @ P + G @ Q @ G.T, 0
)
```

### MATLAB / Octave snippet (same system)

```matlab
A = [-0.9, 0.7; -0.3, 0.1];
G = eye(2);
C = [1, 1];
Q = diag([1, 3]);
R = 0.1;

[M, P, Z, E] = dlqe(A, G, C, Q, R);

% M is the filtering form gain, not A*M as returned by control.dlqe
assert(M, P*C'/(C*P*C' + R), 1e-12)
assert(M, [0.5716; 0.4184], 5e-5)
assert(max(abs(A*M - [-0.2216; -0.1297])) < 5e-5)  % = control.dlqe's L

% P solves the discrete-time Riccati equation (same P as control.dlqe)
assert(P, [4.7687 0.9438; 0.9438 3.2369], 5e-5)

% Z = P(k|k) and E are the poles of the estimator, eig(A - A*M*C)
assert(Z, P - M*C*P, 1e-12)
assert(sort(E), sort(eig(A - A*M*C)), 1e-12)
```

Octave output for this system (control package):
M = [0.5716; 0.4184], P = [4.7687 0.9438; 0.9438 3.2369],
Z = [1.5032 -1.4460; -1.4460 1.4879], E = [-0.44603; -0.0026922].
