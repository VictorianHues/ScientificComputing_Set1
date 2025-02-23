import numpy as np
import matplotlib.pyplot as plt
import time


from SOR_diff import SORDiffusion
from jacobi_iteration import Jacobi
from numba import njit, prange
from finite_difference import fast_solve, SOR


def iter_to_convergence_SOR(tolerances, omega, max_steps=100000, mask=None, grid_size = 50):
    """"
    simulate diffusion on a 2d grid to convergence at specified tolerances with the SOR method.
    periodic boundary conditions with source at top and sink at bottom. 
    
    params:
        tolerances: list of different tolerance values to test
        omega:      parameter for SOR
        max_steps:  maximum number of iterations
        grid_size:  size of grid
        mask:       binary mask - 0 for nodes that are sinks
        insulated:  binary mask - 1 for nodes that are insulated    

    returns:
        Ns:         number of iterations to reach converges, same shape as tolerances
    """
    Ns = np.zeros_like(tolerances)
    for i in range(len(tolerances)):
        tolerance = tolerances[i]
        
        x_length = 1.0
        y_length = 1.0
        n_steps = grid_size
        time_step_num = max_steps
        # omega = 0.1
        # """

        sor_diffusion = SORDiffusion(
                                    x_length, 
                                    y_length, 
                                    n_steps, 
                                    time_step_num, 
                                    omega, 
                                    lambda x, y: 1,
                                    mask)
        
        solution, t, tol = sor_diffusion.solve(tolerance=tolerance)
        # sor_diffusion.plot_animation()    
        Ns[i] = t
    return Ns
    
    
def iter_to_convergence_Jacobi(tolerances):
    """"
    simulate diffusion on a 2d grid to convergence at specified tolerances with the Jacobi method.
    periodic boundary conditions with source at top and sink at bottom. 
    
    params:
        tolerances: list of different tolerance values to test

    returns:
        Ns:         number of iterations to reach converges, same shape as tolerances
    """
    Ns = np.zeros_like(tolerances)
    for i, tolerance in enumerate(tolerances):
        
        n_steps = 50
        time_step_num = 100000
        # omega = 0.1
        # """
        jacobi_diffusion = Jacobi(
                                                n_steps, 
                                                tolerance,
                                                time_step_num)
        
        solution, t, tol = jacobi_diffusion.solve()
        print(t)
        # sor_diffusion.plot_animation()    
        Ns[i] = t
    return Ns


def tol_after_N_Jacobi(Ns):
    """"
    simulate diffusion on a 2d grid for a specified number of iterations with the Jacobi method.
    periodic boundary conditions with source at top and sink at bottom. 
    
    params:
        Ns:         list of different iteration counts to test

    returns:
        tolerances:         tolerance / maximum difference after N iterations, same shape as Ns
    """
    tolerances = np.zeros_like(Ns, dtype=np.float64)
    for i, N in enumerate(Ns):
        
        n_steps = 50
        time_step_num = 100000
        # omega = 0.1
        # """
        jacobi_diffusion = Jacobi(
                                                n_steps, 
                                                0,
                                                N)
        
        solution, t, tol = jacobi_diffusion.solve()
        # print(t, tol)
        # sor_diffusion.plot_animation()    
        tolerances[i] = tol
    return tolerances


def tol_after_N_SOR(Ns, omega,  grid_size=50, max_steps=100000, mask=None):
    """"
    simulate diffusion on a 2d grid for a specified number of iterations with the SOR method.
    periodic boundary conditions with source at top and sink at bottom. 
    
    params:
        Ns:         list of different iteration counts to test
        omega:      parameter for SOR
        max_steps:  maximum number of iterations
        grid_size:  size of grid
        mask:       binary mask - 0 for nodes that are sinks 

    returns:
        tolerances:         tolerance / maximum difference after N iterations, same shape as Ns
    """
    tolerances = np.zeros_like(Ns, dtype=np.float64)
    for i, N in enumerate(Ns):
        
        x_length = 1.0
        y_length = 1.0
        n_steps = grid_size
        # time_step_num = max_steps
        # omega = 0.1
        # """
        sor_diffusion = SORDiffusion(
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                N, 
                                                omega, 
                                                lambda x, y: 1,
                                                mask)
        
        solution, t, tol = sor_diffusion.solve(tolerance=0)
        # sor_diffusion.plot_animation()    
        tolerances[i] = tol
    return tolerances
    
    
    
    
def iter_to_convergence_plot():
    """"
    Run multiple finite difference diffusion experiments and plot the results.
    This experiment measures the number of iterations to reach different tolerance levels for different omega of SOR and the Jacobi method
    
    """
    omegas = [ 0.7, 1, 1.5, 1.7, 1.9]
    # omegas = np.linspace(0.5,1.93, 10)
    tolerances = 1/10**np.linspace(3,8,50)
    for omega in omegas:
        num_steps = iter_to_convergence_SOR(tolerances, omega)
        plt.plot(tolerances, num_steps, label=r'SOR, $\omega$ = {}'.format(omega))
    
    num_steps = iter_to_convergence_Jacobi(tolerances)
    plt.plot(tolerances, num_steps, label='Jacobi')
        
    plt.grid()
    plt.legend()
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('iteration')
    plt.title('Number of Iterations to Converge')
    plt.savefig('plots/num_iter_vs_tol.png', dpi=600)
    plt.show()
    
def iter_to_convergence_plot_inv():
    """"
    Run multiple finite difference diffusion experiments and plot the results.
    This experiment measures the tolerance level reached after a set of number of iteration steps for different omega of SOR and the Jacobi method
    The convergence rate is measured as the slope of the data before reaching a high convergence
    """
    omegas = [ 0.7, 1, 1.5, 1.7, 1.9]
    # omegas = np.linspace(0.5,1.93, 10)
    Ns = (10**np.linspace(2,4.5,50)).astype(np.int64)
    for omega in omegas:
        tolerances = tol_after_N_SOR(Ns, omega)
        plt.plot(Ns, tolerances, label=r'SOR, $\omega$ = {}'.format(omega))
        num_pts = np.argmax(tolerances<1e-12)
        lin_fit = np.polyfit(Ns[:num_pts], np.log10(tolerances[:num_pts]), deg=1)
        print(omega, lin_fit[0], lin_fit[1])
    
    tolerances = tol_after_N_Jacobi(Ns)
    plt.plot(Ns, tolerances, label='Jacobi')
    num_pts = np.argmax(tolerances<1e-12)
    lin_fit = np.polyfit(Ns[:num_pts], np.log10(tolerances[:num_pts]), deg=1)
    print('jacobi', lin_fit[0], lin_fit[1])
        
    plt.grid()
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\varepsilon^k$', rotation=0, fontsize=14)
    plt.xlabel('iteration', fontsize=14)
    # plt.title('Number of Iterations to Converge')
    plt.savefig('plots/num_iter_vs_tol_inv.png', dpi=600)
    # plt.show()
    
    
def min_val_approximation(x, y, dx = 1):
    
    """"
    This method attempts to increase the accuracy of the location of the minimum of a dataset y(x) by fitting a parabola 
    to the neighborhood of the minimal datapoint and finding its vertex.    
    """
    i = np.argmin(y)
    # print(omegas[i-dw:i+dw+1], Ns[i-dw:i+dw+1])
    p = np.polyfit(x[i-dx:i+dx+1], y[i-dx:i+dx+1], 2)
    x_min = - p[1] / (2*p[0])
    y_min = np.polyval(p, x_min)
    
    return x_min, y_min
    
    
def optimal_omega_plot(mask = None, title='No Obstructions', file ='plots/opt_omega_full.png', grid_sizes=[50], tolerance=1e-8 ):
    """"
    Run multiple finite difference diffusion experiments and plot the results.
    This experiment measures number of iterations to reach a set tolerance level for different omega of SOR and finds the omega for which this number of iterations is minimal.
    """
    omegas = np.linspace(0.01, 2, 500)
    Ns = np.zeros_like(omegas)
    tolerance = [tolerance]
    
    plt.figure(figsize=[10,3])
    plt.tight_layout()
    
    plt.subplots_adjust(bottom=0.15)
    for grid_size in grid_sizes:
        for i, omega in enumerate(omegas):
            N = iter_to_convergence_SOR(tolerance, omega, max_steps=100000, mask = mask,grid_size=grid_size)
            Ns[i] = N
            
        plt.plot(omegas, Ns, label=grid_size)
        w_min, N_min = min_val_approximation(omegas, Ns)
        # print(p, N_min)
        print('Minimum omega at {:.5f} with {} samples'.format( w_min, N_min))
    plt.grid()
    if len(grid_sizes) >1:
        plt.legend(title='grid size')
    # plt.legend()
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel('iterations to reach tolerance', fontsize=14)
    # plt.title(title + r', $\omega_{min} = $' + '{:.2f}'.format(w_min))
    # plt.ylim([2e2,9e4])
    plt.savefig(file, dpi=600)

    
    
def optimal_omega_plot_inv(mask = None, title='No Sinks', file ='plots/opt_omega_tol.png', grid_size = 50, num_iter=1000 ):
    """"
    Run multiple finite difference diffusion experiments and plot the results.
    This experiment measures the tolerance level reached after a set of number of iteration steps for different omega of SOR and finds the omega for which this tolerance is minimal.
    """
    omegas = np.linspace(0.01, 2, 100)
    tolerances = np.zeros_like(omegas)
    num_iter = [num_iter]
    
    for i, omega in enumerate(omegas):
        tol = tol_after_N_SOR(num_iter, omega, mask = mask, grid_size=grid_size)
        tolerances[i] = tol
        
    plt.figure(figsize=[10,3])
    plt.tight_layout()
    
    plt.subplots_adjust(bottom=0.15)
    plt.plot(omegas, tolerances)
    w_min, tol_min = min_val_approximation(omegas, tolerances)
    # print(p, N_min)
    print('Minimum omega at {:.5f}'.format( w_min))
    plt.grid()
    # plt.legend()
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\varepsilon$')
    plt.title(title + r',  $\omega_{min} = $' + '{:.2f}'.format(w_min))
    plt.ylim([1e-16,1e-3])
    plt.yscale('log')
    plt.savefig(file, dpi=600)

    
    
def optimal_omega_vs_N_plot(mask = None, title='No Obstructions', file ='plots/opt_omega_vs_N.png', grid_sizes=[50], tolerance=1e-9 ):
    """"
    Run multiple finite difference diffusion experiments and plot the results.
    This experiment finds the optimal omega for different grid sizes.
    """
    omegas = np.linspace(1.8, 2, 40)
    Ns = np.zeros_like(omegas)
    tolerance = [tolerance]
    
    plt.figure(figsize=[10,3])
    plt.tight_layout()
    
    plt.subplots_adjust(bottom=0.15)
    # this runs for multiple hours, use saved data to plot instead.
    # opt_omegas = np.zeros_like(grid_sizes).astype(np.float64)
    # for j, grid_size in enumerate(grid_sizes):
    #     for i, omega in enumerate(omegas):
    #         N = iter_to_convergence_SOR(tolerance, omega, max_steps=10000, mask = mask,grid_size=grid_size)
    #         Ns[i] = N
            
    #     # plt.plot(omegas, Ns, label=grid_size)
    #     w_min, N_min = min_val_approximation(omegas, Ns)
    #     # print(p, N_min)
    #     print(grid_size, 'Minimum omega at {:.5f} with {} samples'.format( w_min, N_min))
    #     opt_omegas[j] = w_min
    import pandas as pd
    data = pd.read_csv('data/opt_omega_vs_N_data.csv')
    grid_sizes = data['grid_size']
    opt_omegas = data['omega']


    plt.grid()
    plt.plot(grid_sizes, opt_omegas, linestyle = '', marker='D')
    # if len(grid_sizes) >1:
    #     plt.legend(title='grid size')
    # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(r'grid size', fontsize=20)
    plt.ylabel(r'$\omega_{min}$', fontsize=20)
    # plt.title(title + r', $\omega_{min} = $' + '{:.2f}'.format(w_min))
    plt.ylim([1.8,2])
    plt.xlim([10,400])
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(file, dpi=600)

        

def a_maze():
    """"
    This experiment runs the simulation with a maze of insulated cells and animates the results.
    """
    import imageio as iio
    maze_arr = iio.imread('plots/maze.png')
    mask = (maze_arr[:,:,0]//255).astype(np.float64)
    # mask = np.flip(mask, axis=0)
    insulated = 1-mask

    x_length = 1.0
    y_length = 1.0
    n_steps = 53
    time_step_num = 100000
    omega = 1.87
    # """
    sor_diffusion = SORDiffusion(
                                            x_length, 
                                            y_length, 
                                            n_steps, 
                                            time_step_num, 
                                            omega, 
                                            lambda x, y: 1, 
                                            mask=mask,
                                            insulated = insulated)
    
    c, t, tol = sor_diffusion.solve(1e-9)
    print('required {} iterations'.format(t))

    sor_diffusion.plot_animation(skip_n=100)


    fig, ax = plt.subplots(figsize=[6,5])
    fig.subplots_adjust(left=0.07, bottom=0.06, right=0.85, top=0.95)
    heatmap = ax.imshow(c[t], cmap="hot", origin="lower",aspect='equal' ) #extent=[0,x_length, 0, y_length])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Concentration", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    plt.savefig('plots/maze_heatmap.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    
    a_maze()
    
    iter_to_convergence_plot_inv()

    optimal_omega_plot_inv(num_iter=5000, grid_size=100)

    grid_sizes = [40,50,60,100]

    optimal_omega_plot(grid_sizes=grid_sizes, tolerance=1e-12)

    grid_sizes = np.linspace(25,400,20).astype(np.int64)
    optimal_omega_vs_N_plot(grid_sizes=grid_sizes)
    
    