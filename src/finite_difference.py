import numpy as np


from numba import jit, njit, prange




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


@njit
def jacobi(c_old, c_new, max_iter, n_steps, tolerance):
    for iteration in range(max_iter):
                
        # Top and Bottom Boundaries
        c_old[:, -1] = 1.0
        c_old[:, 0] = 0.0

        # Jacobi update rule
        for i in range(n_steps):
            for j in range(1, n_steps - 1):
                c_new[i, j] = 0.25 * (
                    c_old[(i+1) % n_steps, j] +
                    c_old[(i-1) % n_steps, j] +
                    c_old[i, j+1] +
                    c_old[i, j-1]
                )

        # Stopping criterion
        delta = np.max(np.abs(c_new - c_old))
        if delta < tolerance:
            return  iteration

        # Update step
        c_old = c_new.copy()



@njit
def SOR_calc(c, t, i, j, width, omega, mask):
    c[t+1, i, j] = (mask[i,j] 
                    * (omega / 4.0 * (c[t, i+1, j] + c[t+1, i-1, j] + c[t+1, i, (j-1)% width] 
                                        + c[t, i, (j+1)% width ])+ (1-omega) * c[t, i, j] ))



@njit
def SOR(c, omega, mask=None, tolerance= None):
    print(omega)
    time_step_num, width, height = c.shape
    if mask is None:
        mask = np.ones(shape=c.shape[1:])
    for t in range(0, time_step_num - 1):
        
        for i in prange(1, height -1):
            c[t+1, i, -1] = c[t, i, -1] # to avoid adding sink on left side
            for j in prange(width):
                SOR_calc(c, t, i, j, width, omega, mask)

        # Top and Bottom Boundaries
        c[t+1, -1] = c[t, -1]
        c[t+1, 0] = 0.0

        if tolerance is not None:
            eps = np.max(np.abs(c[t+1] - c[t]))
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

