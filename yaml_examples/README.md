# Example model.yaml files

Child classes: their `__init__` arguments, without the defaults (prototype). Connected models: the base-class form, saved with `save()`.

## First order, symbolic parameters

`SSModelCTLinearFOSISO()` → `ct_fo_symbolic/`

```yaml
{cas_models_format: 1, class: SSModelCTLinearFOSISO}
```

## Second order, fixed parameters and names

`SSModelCTLinearO2SISO(K=2.0, T1=3.0, T2=5.0, name="G", input_name="Q", output_name="T")` → `ct_o2_fixed/`

```yaml
cas_models_format: 1
class: SSModelCTLinearO2SISO
args: {K: 2.0, T1: 3.0, T2: 5.0, name: G, input_name: Q, output_name: T}
```

## Second order, K fixed, time constants symbolic

`SSModelCTLinearO2SISO(K=2.0)` → `ct_o2_mixed/`

```yaml
cas_models_format: 1
class: SSModelCTLinearO2SISO
args: {K: 2.0}
```

## Integrator

`SSModelCTLinearIntegratorSISO(K=0.5)` → `ct_integrator/`

```yaml
cas_models_format: 1
class: SSModelCTLinearIntegratorSISO
args: {K: 0.5}
```

## Second order underdamped

`SSModelCTLinearO2UnderdampedSISO(K=1.0, zeta=0.3, omega_n=2.0)` → `ct_o2_underdamped/`

```yaml
cas_models_format: 1
class: SSModelCTLinearO2UnderdampedSISO
args: {K: 1.0, zeta: 0.3, omega_n: 2.0}
```

## PI controller

`SSModelCTPIInt(Kc=0.5, Ti=30.0, name="ctrl")` → `ct_pi/`

```yaml
cas_models_format: 1
class: SSModelCTPIInt
args: {Kc: 0.5, Ti: 30.0, name: ctrl}
```

## PID controller, symbolic

`SSModelCTPIDInt()` → `ct_pid/`

```yaml
{cas_models_format: 1, class: SSModelCTPIDInt}
```

## From A, B, C, D (3 states, 2 inputs, 2 outputs)

`StateSpaceModelCTFromABCD(A, B, C, cas.DM.zeros(2, 2), input_names=["a", "b"])` → `ct_abcd/`

```yaml
cas_models_format: 1
class: StateSpaceModelCTFromABCD
args:
  A:
    DM:
    - [-1.0, 0.0, 0.0]
    - [0.5, -2.0, 0.0]
    - [0.0, 0.0, -3.0]
  B:
    DM:
    - [1.0, 0.0]
    - [0.0, 0.0]
    - [0.0, 1.0]
  C:
    DM:
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 2.0]
  D:
    DM:
    - [0.0, 0.0]
    - [0.0, 0.0]
  input_names: [a, b]
```

## From A, B, C, D (6 states: A larger than 5 x 5, in a file)

`StateSpaceModelCTFromABCD(A, B, C, D)` → `ct_abcd_6_states/`

```yaml
cas_models_format: 1
class: StateSpaceModelCTFromABCD
args:
  A:
    file: A.tsv
    shape: [6, 6]
  B:
    DM: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  C:
    DM:
    - [0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
      0.16666666666666666, 0.16666666666666666]
  D:
    DM: [0.0]
```

`A.tsv`:

```
-0.965	0.082	0.033	-0.13	0.091	0.045
-0.054	-0.942	0.036	0.029	0.003	0.055
-0.074	-0.016	-1.048	0.06	0.004	-0.029
-0.078	-0.026	0.001	-1.028	0.129	0.101
-0.271	-0.189	-0.017	-0.042	-0.979	0.022
0.212	-0.111	-0.038	0.204	0.065	-0.934
```

## Direct transmission

`SSModelCTDirectTransmission(D=cas.DM([[1.0, 0.0], [0.5, 2.0]]))` → `ct_direct/`

```yaml
cas_models_format: 1
class: SSModelCTDirectTransmission
args:
  D:
    DM:
    - [1.0, 0.0]
    - [0.5, 2.0]
```

## ARX, symbolic

`StateSpaceModelDTARXSISO(na=2, nb=2, nk=1)` → `dt_arx_symbolic/`

```yaml
cas_models_format: 1
class: StateSpaceModelDTARXSISO
args: {na: 2, nb: 2}
```

## ARX, fitted values

`StateSpaceModelDTARXSISO(A=cas.DM([-1.62, 0.65]), B=cas.DM([0.012, 0.009]), nk=1)` → `dt_arx_fixed/`

```yaml
cas_models_format: 1
class: StateSpaceModelDTARXSISO
args:
  A:
    DM: [-1.62, 0.65]
  B:
    DM: [0.012, 0.009]
```

## Transfer function

`StateSpaceModelDTTFSISO(num=[0.2, 0.1], den=[1, -1.4, 0.49])` → `dt_tf/`

```yaml
cas_models_format: 1
class: StateSpaceModelDTTFSISO
args:
  num: [0.2, 0.1]
  den: [1, -1.4, 0.49]
```

## Delay with a gain matrix

`StateSpaceModelDTDelay(nk=3, nu=2, G=[[1.0, 0.5], [0.0, 2.0]], dt=1.0)` → `dt_delay/`

```yaml
cas_models_format: 1
class: StateSpaceModelDTDelay
args:
  nk: 3
  nu: 2
  G:
    DM:
    - [1.0, 0.5]
    - [0.0, 2.0]
  dt: 1.0
```

## Discretised with RK4 (a nested model)

`StateSpaceModelDTFromCTRK4(SSModelCTLinearO2SISO(K=2.0), 1.0)` → `dt_from_ct_rk4/`

```yaml
cas_models_format: 1
class: StateSpaceModelDTFromCTRK4
args:
  model_ct:
    class: SSModelCTLinearO2SISO
    args: {K: 2.0}
  dt: 1.0
```

## Discretised with cvodes

`StateSpaceModelDTFromCT(SSModelCTLinearFOSISO(), 1.0, integrator_opts={"abstol": 1e-10})` → `dt_from_ct_cvodes/`

```yaml
cas_models_format: 1
class: StateSpaceModelDTFromCT
args:
  model_ct: {class: SSModelCTLinearFOSISO}
  dt: 1.0
  integrator_opts: {abstol: 1.0e-10}
```

## Connected ARX paths (as in ffdtools ARXPaths)

`connect_systems([ARX(2, 2, 1), ARX(1, 1, 1)], ...)` → `dt_connected_arx/`

```yaml
cas_models_format: 1
class: StateSpaceModelDT
args:
  F: {file: F.casadi}
  H: {file: H.casadi}
  n: 6
  nu: 2
  params:
    CRAC1_flow_a_0: [1, 1]
    CRAC1_flow_a_1: [1, 1]
    CRAC1_flow_b_0: [1, 1]
    CRAC1_flow_b_1: [1, 1]
    CRAC2_flow_a: [1, 1]
    CRAC2_flow_b: [1, 1]
  name: R11_intake
  input_names: [CRAC1_flow, CRAC2_flow]
  state_names: [CRAC1_flow_x1, CRAC1_flow_x2, CRAC1_flow_x3, CRAC1_flow_x4, CRAC2_flow_x1,
    CRAC2_flow_x2]
  output_names: [R11_intake]
```

`F.casadi`: CasADi serialization (not readable)

`H.casadi`: CasADi serialization (not readable)

## PI feedback loop

`connect_feedback_system(plant * ctrl, model_class=StateSpaceModelCT)` → `ct_pi_feedback/`

```yaml
cas_models_format: 1
class: StateSpaceModelCT
args:
  f: {file: f.casadi}
  h: {file: h.casadi}
  n: 2
  params:
    Kc: [1, 1]
    Ti: [1, 1]
  name: fbk_ctrl_plant
  input_names: [y_sp]
  state_names: [plant_x, ctrl_x]
  output_names: [y]
```

`f.casadi`: CasADi serialization (not readable)

`h.casadi`: CasADi serialization (not readable)
