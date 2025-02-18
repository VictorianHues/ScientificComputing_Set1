import numpy as np
import matplotlib.pyplot as plt
import time


from SOR_diff import SORDiffusion
from jacobi_iteration import Jacobi
from numba import njit, prange
from finite_difference import fast_solve, SOR

def main():
    x_length = 1.0
    y_length = 1.0
    n_steps = 50
    time_step_num = 100000
    omega = 1.8
    # """
    sor_diffusion = SORDiffusion(
                                            x_length, 
                                            y_length, 
                                            n_steps, 
                                            time_step_num, 
                                            omega, 
                                            lambda x, y: 1)
    
    solution, end_time = sor_diffusion.solve(1e-16)

    sor_diffusion.plot_animation()
    #sor_diffusion.plot_y_slice(0)

    sor_diffusion.plot_single_frame(end_time)
    sor_diffusion.plot_analytical_sol()

    sor_diffusion.compare_solutions_full_range(end_time)

def iter_to_convergence_SOR(tolerances, omega, max_steps=100000, mask=None):
    Ns = np.zeros_like(tolerances)
    for i in range(len(tolerances)):
        tolerance = tolerances[i]
        
        x_length = 1.0
        y_length = 1.0
        n_steps = 50
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
        
        solution, t = sor_diffusion.solve(tolerance=tolerance)
        sor_diffusion.plot_animation()    
        Ns[i] = t
    return Ns

# @njit(parallel=True)  
# def iter_to_convergence_SOR(tolerances, omega, max_steps=100000, mask=None):
#     Ns = np.zeros_like(tolerances)
#     for i in prange(len(tolerances)):
#         tolerance = tolerances[i]
        
#         x_length = 1.0
#         y_length = 1.0
#         n_steps = 50
#         time_step_num = max_steps
#         # omega = 0.1
#         # """
        
#         x_points = np.linspace(0, x_length, n_steps)
#         y_points = np.linspace(0, y_length, n_steps)

#         x_step_size = x_length / (n_steps - 1)
#         y_step_size = y_length / (n_steps - 1)

#         c = np.zeros((time_step_num, n_steps, n_steps))

#         c[0, -1, :] = 1.0
#         c[0, 0, :] = 0.0

#         solution, t = SOR(c, omega, mask=mask, tolerance=tolerance)
#         if tolerance is not None:
#             print('finished after ', t, ' iterations')


#         # sor_diffusion = SORDiffusion(
#         #                             x_length, 
#         #                             y_length, 
#         #                             n_steps, 
#         #                             time_step_num, 
#         #                             omega, 
#         #                             lambda x, y: 1,
#         #                             mask)
        
#         # solution, t = sor_diffusion.solve(tolerance=tolerance)
#         # sor_diffusion.plot_animation()    
#         Ns[i] = t
#     return Ns
    
    
def iter_to_convergence_Jacobi(tolerances):
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
        
        solution, t = jacobi_diffusion.solve()
        print(t)
        # sor_diffusion.plot_animation()    
        Ns[i] = t
    return Ns
    
    
def iter_to_convergence_plot():
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
    plt.ylabel('N')
    plt.title('Number of Iterations to Converge')
    plt.savefig('plots/num_iter_vs_tol.png', dpi=600)
    plt.show()
    
    
def min_val_approximation(x, y, dx = 1):
    i = np.argmin(y)
    # print(omegas[i-dw:i+dw+1], Ns[i-dw:i+dw+1])
    p = np.polyfit(x[i-dx:i+dx+1], y[i-dx:i+dx+1], 2)
    x_min = - p[1] / (2*p[0])
    y_min = np.polyval(p, x_min)
    
    return x_min, y_min
    
    
def optimal_omega_plot(mask = None, title='No Obstructions', file ='plots/opt_omega_full.png' ):
    omegas = np.linspace(0.01, 2, 500)
    Ns = np.zeros_like(omegas)
    tolerance = [1e-8]
    
    for i, omega in enumerate(omegas):
        N = iter_to_convergence_SOR(tolerance, omega, max_steps=100000, mask = mask)
        Ns[i] = N
        
    plt.figure(figsize=[10,3])
    plt.tight_layout()
    
    plt.subplots_adjust(bottom=0.15)
    plt.plot(omegas, Ns)
    w_min, N_min = min_val_approximation(omegas, Ns)
    # print(p, N_min)
    print('Minimum omega at {:.5f}'.format( w_min))
    plt.grid()
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\omega$')
    plt.ylabel('N')
    plt.title(title + r',  $\omega_{min} = ${.2f}'.format(w_min))
    plt.ylim([2e2,9e4])
    plt.savefig(file, dpi=600)
        

if __name__ == '__main__':
    start_time = time.time()
    main()
    #iter_to_convergence_plot()
    #optimal_omega_plot()

    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    
    