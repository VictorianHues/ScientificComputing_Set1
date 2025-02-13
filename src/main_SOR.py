import numpy as np

from SOR_diff import SORDiffusion
from time_dep_diff_tools import plot_y_slice_time_magnitudes

def main():
    x_length = 1.0
    y_length = 1.0
    n_steps = 100
    time_step_num = 1000
    omega = 1.7
    # """
    sor_diffusion = SORDiffusion(
                                            x_length, 
                                            y_length, 
                                            n_steps, 
                                            time_step_num, 
                                            omega, 
                                            lambda x, y: 1)
    
    solution = sor_diffusion.solve()
    sor_diffusion.plot_animation()
    # sor_diffusion.plot_y_slice(0)


if __name__ == '__main__':
    main()