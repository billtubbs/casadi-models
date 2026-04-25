import os
import numpy as np
import pandas as pd
import control as con


# Results will be saved to file
results = {}
test_data_dir = "tests/data"
os.makedirs(test_data_dir, exist_ok=True)

# Simulation time
nT = 12
Ts = 0.5
t = Ts * np.arange(nT + 1)
results["t"] = t

# Input signal
u = np.zeros(nT + 1)
u[t >= 1.0] = 1.0
results["u"] = u

G1 = con.tf([1], [1, 1])
G2 = con.tf([2], [2.5, 1])

# Underdamped second order system
zeta = 0.5
omega = 1.6
num = [omega**2]
den = [1, 2 * zeta * omega, omega**2]
G3 = con.tf(num, den)

# PI controller parameters
Kc = 0.5
Ti = 2.5
Gc = con.tf([Kc * Ti, Kc], [Ti, 0])

# Systems
ct_systems = {
    "G1": G1,
    "G2": G2,
    "G3": G3,
    "G1 * G2": G1 * G2,  # in series
    "G1 * G2 + G3": G1 * G2 + G3,  # in parallel
    "H2": con.feedback(Gc * G2, 1),  # closed-loop system
}

# Simulate the systems
for name, G in ct_systems.items():
    t_out, y_out = con.forced_response(G, T=t, U=u)
    results[f"y_{name}"] = y_out

results = pd.DataFrame(results)
print(results)

# Save simulation results
filename = "results_ct.csv"
filepath = os.path.join(test_data_dir, filename)
results.to_csv(filepath)
print(f"Test data saved to {filepath!r}")
