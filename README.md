# CasADi Dynamic Models

Tools to define and manipulate simple linear dynamic models in [CasADi](https://web.casadi.org)
for use in non-linear dynamic modelling and optimization.  The models may have symbolic 
parameters allowing them to be included in optimization problems.

While CasADi is primarily used for solving non-linear optimization problems, it is often 
convenient to add linear dynamical systems to introduce simple identifiable dynamics.

Furthermore, it is sometimes necessary or convenient to use non-linear constrained optimization
to identify the parameters of linear dynamic models.

I'm not aware of a system identification toolbox for CasADi models so I developed these
tools to identify and simulate simple dynamical systems for my projects.

Note this is a work-in-progress and only a small subset of possible models have been
implemented.

## Example

```python
from cas_models.continuous_time.models import (
    SSModelCTLinearFOSISO, 
    SSModelCTLinearO2NoGainSISO, 
    connect_nonlinear_systems_in_series
)

# First order, single-input, single-output, continuous-time
# state-space model with symbolic parameters
sys_model = SSModelCTLinearFOSISO()
print(sys_model.f)
print(sys_model.h)
```
```lang-none
f:(t,x,u,K,T1)->(rhs) SXFunction
h:(t,x,u,K,T1)->(y) SXFunction
```

Alternatively, some or all model parameters can be set to constants.
```python
# First order SISO system with fixed gain
sys_model = SSModelCTLinearFOSISO(K=2.5)
print(sys_model.f)
print(sys_model.h)
```
```lang-none
f:(t,x,u,T1)->(rhs) SXFunction
h:(t,x,u,T1)->(y) SXFunction
```

```python
# Second-order system with gain = 1
sys_model_2 = SSModelCTLinearO2NoGainSISO()
print(sys_model_2.f)
print(sys_model_2.h)
```
```lang-none
f:(t,x[2],u,T1,T2)->(rhs[2]) SXFunction
h:(t,x[2],u,T1,T2)->(y) SXFunction
```

```python
# Combine both systems by connecting in series
sys = connect_nonlinear_systems_in_series([sys_model, sys_model_2])
print(sys.f)
print(sys.h)
```
```lang-none
f:(t,x[3],u,sys1_T1,sys2_T1,T2)->(rhs[3]) SXFunction
h:(t,x[3],u,sys1_T1,sys2_T1,T2)->(y) SXFunction
```

Note that I prefer to use `cas` to refer to the casadi package rather than `ca`.
