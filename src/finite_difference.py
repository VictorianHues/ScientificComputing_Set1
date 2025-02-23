import numpy as np
from numba import jit, njit, prange




@njit
def fast_solve(c, time_step_num, n_steps, time_step_size, diffusion_coefficient, x_step_size):
    """2nd order finite difference method for time-dependent diffusion
    
    
    params:
        c:              grid of concentration values shape [time_steps x grid_size x grid_size]
        time_step_num:  number of iterations
        n_steps:        grid size
        time_step_size: delta t from report
        c_step_size:    delta x
        diffusion_coefficient - D

    returns:
        c:      modified grid c    
    """
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
    """Jacobi finite difference method for time-independent diffusion
    
    params:
        c:          grid of concentration values shape [time_steps x grid_size x grid_size]
        mask:       grid of sinks shape [grid_size x grid_size]
        mask:       grid of sinks shape [grid_size x grid_size]
        tolerance:  stop when changes between iterations are smaller than tolerance

    returns:
        c:      modified grid c
        t:      last timestep
        tol:    change from previous-to-last iteration to last iteration

    """
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
            return  iteration, delta

        # Update step
        c_old = c_new.copy()
    return iteration, delta



@njit
def SOR(c, omega, mask=None, tolerance= None, insulated=None):
    """SOR finite difference method for time-independent diffusion
    
    params:
        c:          grid of concentration values shape [time_steps x grid_size x grid_size]
        omega:      parameter of SOR
        mask:       grid of sinks shape [grid_size x grid_size]
        mask:       grid of sinks shape [grid_size x grid_size]
        tolerance:  stop when changes between iterations are smaller than tolerance

    returns:
        c:      modified grid c
        t:      last timestep
        tol:    change from previous-to-last iteration to last iteration

    """
    time_step_num, width, height = c.shape
    if mask is None:
        mask = np.ones(shape=c.shape[1:])
    if insulated is None:
        insulated = np.zeros(shape=c.shape[1:])
    for t in range(0, time_step_num - 1):
        # Top and Bottom Boundaries
        c[t+1, -1] = c[t, -1]
        c[t+1, 0] = 0.0
        for i in range(1, height -1):
            c[t+1, i, -1] = c[t, i, -1] # to avoid adding sink on left side
            for j in range(width):
                c0 = c[t, i, j]
                c1 = c0 if insulated[i+1, j] else c[t, i+1, j]
                c2 = c0 if insulated[i-1, j] else c[t+1, i-1, j]
                c3 = c0 if insulated[i, (j-1)%width] else c[t+1, i, (j-1)% width] 
                c4 = c0 if insulated[i, (j+1)%width] else c[t, i, (j+1)% width ]
                c[t+1, i, j] = mask[i,j] * (omega / 4.0 * (c1 + c2 + c3+ c4)
                    + (1-omega) * c0 )
                
        
                    


        # # Top and Bottom Boundaries
        # c[t+1, 0] = c[t, 0]
        # c[t+1, -1] = 0.0

        if tolerance is not None:
            eps = np.max(np.abs(c[t+1] - c[t]))
            if eps < tolerance:
                break

    return c, t, eps


