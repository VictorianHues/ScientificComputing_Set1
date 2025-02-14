import numpy as np
import matplotlib.pyplot as plt
from SOR_diff import SORDiffusion
from jacobi_iteration import Jacobi
from time_dep_diff_tools import plot_y_slice_time_magnitudes

def main():
    x_length = 1.0
    y_length = 1.0
    n_steps = 50
    time_step_num = 10000
    omega = 0.1
    # """
    sor_diffusion = SORDiffusion(
                                            x_length, 
                                            y_length, 
                                            n_steps, 
                                            time_step_num, 
                                            omega, 
                                            lambda x, y: 1)
    
    solution = sor_diffusion.solve(1e-4)
    sor_diffusion.plot_animation()
    # sor_diffusion.plot_y_slice(0)
    
    
def iter_to_convergence_SOR(tolerances, omega, max_steps=100000):
    Ns = np.zeros_like(tolerances)
    for i, tolerance in enumerate(tolerances):
        
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
                                                lambda x, y: 1)
        
        solution, t = sor_diffusion.solve(tolerance=tolerance)
        # sor_diffusion.plot_animation()    
        Ns[i] = t
    return Ns
    
    
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
    # plt.show()
    
    
def min_val_approximation(x, y, dx = 1):
    i = np.argmin(y)
    # print(omegas[i-dw:i+dw+1], Ns[i-dw:i+dw+1])
    p = np.polyfit(x[i-dx:i+dx+1], y[i-dx:i+dx+1], 2)
    x_min = - p[1] / (2*p[0])
    y_min = np.polyval(p, x_min)
    
    return x_min, y_min
    
    
def optimal_omega_plot():
    omegas = np.linspace(1.7, 2, 100)
    Ns = np.zeros_like(omegas)
    tolerance = [1e-8]
    
    for i, omega in enumerate(omegas):
        N = iter_to_convergence_SOR(tolerance, omega, max_steps=10000)
        Ns[i] = N
        
    plt.plot(omegas, Ns)
    w_min, N_min = min_val_approximation(omegas, Ns)
    # print(p, N_min)
    print('Minimum omega at {:.5f}'.format( w_min))
    plt.grid()
    plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$\omega$')
    plt.ylabel('N')
    plt.title(r'Number of Iterations to Reach the specified tolerance for different $\omega$')
    plt.savefig('plots/opt_omega.png', dpi=600)
        

if __name__ == '__main__':
    # main()
    # iter_to_convergence_plot()
    optimal_omega_plot()
    
    