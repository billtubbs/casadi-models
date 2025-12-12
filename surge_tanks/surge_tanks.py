import casadi as cas


def surge_tank_dxdt(t, x, u, D, V0):

    # System states: 
    #  x[0] : Tank level, L [m]
    #  x[1] : Total mass of suspended mineral in tank, m [tons]
    nx = 2

    # Inputs
    #  u[0] : volumetric flowrate into tank, v_dot_in [m^3/hr]
    #  u[1] : density of fluid entering tank, rho_in [tons/m^3]
    #  u[2] : volumetric flowrate out of tank, v_dot_out [m^3/hr]
    nu = 3

    A = cas.pi * D

    # Define the ODE right-hand side
    x = cas.MX.sym('x', nx)  
    u = cas.MX.sym('u', nu)

    dL_dt = (u[0] - u[2]) / A
    dm_dt = u[0] * u[1] - u[2] * x[1] / (x[0] * A)

    rhs = vertcat(dL_dt, dm_dt)