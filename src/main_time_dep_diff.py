import numpy as np

from time_dep_diff import TimeDependentDiffusion
from time_dep_diff_tools import plot_y_slice_time_magnitudes

def main():
    time_step_size = 0.00001
    x_length = 1.0
    y_length = 1.0
    n_steps = 100
    total_time = 1.0
    diffusion_coefficient = 1.0
    """
    time_diffusion = TimeDependentDiffusion(time_step_size, 
                                            x_length, 
                                            y_length, 
                                            n_steps, 
                                            total_time, 
                                            diffusion_coefficient, 
                                            lambda x, y: 1)
    
    solution = time_diffusion.solve()
    time_diffusion.plot_animation()

    y_slice = 0.5
    time_diffusion.plot_y_slice(y_slice)

    analytical_solution = time_diffusion.analytical_solution(100, 1.0, 100)

    print(analytical_solution)

    time_index = 100
    time_diffusion.compare_solutions(time_index)

    time_diffusion_lin = TimeDependentDiffusion(time_step_size, 
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                total_time, 
                                                diffusion_coefficient, 
                                                lambda x, y: y)
    
    time_diffusion_lin.solve()
    time_diffusion_lin.plot_animation()

    time_diffusion_sin = TimeDependentDiffusion(time_step_size, 
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                total_time, 
                                                diffusion_coefficient, 
                                                lambda x, y: np.sin(np.pi * x))
    
    time_diffusion_sin.solve()
    time_diffusion_sin.plot_animation()
    """
    time_test_array = [1.0, 0.1, 0.01, 0.001]

    plot_y_slice_time_magnitudes(time_step_size, 
                                 x_length, 
                                 y_length, 
                                 n_steps, 
                                 diffusion_coefficient,
                                 time_test_array)


if __name__ == '__main__':
    main()