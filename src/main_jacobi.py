import numpy as np
import matplotlib.pyplot as plt
import time


from SOR_diff import SORDiffusion
from jacobi_iteration import Jacobi
from numba import njit, prange
from finite_difference import fast_solve, SOR

def main():
    n_steps = 50

    jacobi_diffusion = Jacobi(n_steps, epsilon=1e-16)

    solution, end_iteration = jacobi_diffusion.solve()

    print(f"Converged after {end_iteration} iterations")
    print(solution)

    jacobi_diffusion.plot_last_frame()
    jacobi_diffusion.plot_analytical_sol()

    jacobi_diffusion.compare_solutions()

    jacobi_diffusion.compare_solutions_full_range()








if __name__ == '__main__':
    main()