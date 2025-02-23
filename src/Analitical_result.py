from time_dep_diff import TimeDependentDiffusion
from SOR_diff import SORDiffusion
import numpy as np
import matplotlib.pyplot as plt

#Compaison with analitical - SOR and Gauss-Seidel

# Parameters 
x_length = 1.0
y_length = 1.0
n_steps = 50
time_step_num = 10000
initial_condition = 1
omega = 1
mask = None 
tolerance = 1e-9

diffusion_solver = SORDiffusion(
    x_length=x_length,
    y_length=y_length,
    n_steps=n_steps,
    time_step_num=time_step_num,
    omega=omega,
    mask=mask
)


c, end_time, tol = diffusion_solver.solve(tolerance)

# Compare the numerical solution with the analytical solution at the final time step
# In this example, we are comparing the solution at time step `time_step_num - 1` (final step)
diffusion_solver.compare_solutions_full_range(time=end_time)




#same stuff with different omega:


# Parameters 
x_length = 1.0
y_length = 1.0
n_steps = 50
time_step_num = 10000
initial_condition = 1
omega = 1.87
mask = None 
tolerance = 1e-9

diffusion_solver = SORDiffusion(
    x_length=x_length,
    y_length=y_length,
    n_steps=n_steps,
    time_step_num=time_step_num,
    omega=omega,
    mask=mask
)


c, end_time, tol = diffusion_solver.solve(tolerance)

# Compare the numerical solution with the analytical solution at the final time step
# In this example, we are comparing the solution at time step `time_step_num - 1` (final step)
diffusion_solver.compare_solutions_full_range(time=end_time)


