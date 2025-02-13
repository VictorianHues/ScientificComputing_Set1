import numpy as np


from numba import jit, njit




@njit
def fast_solve(c, time_step_num, n_steps, time_step_size, diffusion_coefficient, x_step_size):
    for t in range(0, time_step_num - 1):
        new_c = c[t].copy()  
        
        for i in range(n_steps):
            for j in range(1, n_steps - 1):
                new_c[i, j] = (
                    c[t, i, j] +
                    (time_step_size * diffusion_coefficient / x_step_size**2) *
                    (c[t, (i+1)% n_steps, j] + c[t, (i-1)% n_steps, j] + c[t, i, j+1] + c[t, i, j-1 ] - 4 * c[t, i, j])
                )

        c[t+1] = new_c

        # Top and Bottom Boundaries
        c[t+1, :, -1] = c[t,:,-1]
        c[t+1, :, 0] = 0.0

    return c



# @jit
def SOR(c, omega, mask=None, tolerance= None):
    time_step_num, width, height = c.shape
    if mask is None:
        mask = np.ones([width, height])
    for t in range(0, time_step_num - 1):
        
        for i in range(width):
            for j in reversed(range(1, height - 1)):
                c[t+1, i, j] = mask[i,j] * (omega / 4.0 * (c[t, (i+1)% width, j] + c[t+1, (i-1)% width, j] + c[t+1, i, j+1] + c[t, i, j-1 ])
                    + (1-omega) * c[t, i, j] )
                
        
                    


        # Top and Bottom Boundaries
        c[t+1, :, -1] = c[t,:,-1]
        c[t+1, :, 0] = 0.0

        if tolerance is not None:
            eps = np.max(np.abs(c[-1] - c[-2]))
            if eps < tolerance:
                break

    return c, t


# def SOR_until_tolerance(c, omega, tolerance, mask=None):
#     num_epochs = 0
#     eps = 1e100
#     while eps > tolerance and num_epochs < 1e6:
#         SOR(c, omega, mask)

#         eps = np.max(np.abs(c[-1] - c[-2]))
#     return np.max

