import numpy as np
import gc

from time_dep_diff import TimeDependentDiffusion, plot_y_slice_time_magnitudes


def time_dep_diff_uniform(time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient):
    time_diffusion = TimeDependentDiffusion(time_step_size,
                                            x_length,
                                            y_length,
                                            n_steps,
                                            total_time,
                                            diffusion_coefficient,
                                            lambda x, y: 1)
    
    time_diffusion.solve()
    time_diffusion.plot_animation(save_name="uniform_diffusion.gif")

    y_slice = 0.5
    time_diffusion.plot_y_slice(y_slice)

    time_index = time_diffusion.time_step_num-1
    time_diffusion.compare_solutions(time_index)

    time_diffusion.compare_solutions_full_range(time_index)

    del time_diffusion
    gc.collect()

def time_dep_diff_linear_xy(time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient):
    time_diffusion_lin = TimeDependentDiffusion(time_step_size, 
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                total_time, 
                                                diffusion_coefficient, 
                                                lambda x, y: y)
    
    time_diffusion_lin.solve()
    time_diffusion_lin.plot_animation(save_name="lin_diffusion.gif")

    del time_diffusion_lin
    gc.collect()

def time_dep_diff_sin(time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient):
    time_diffusion_sin = TimeDependentDiffusion(time_step_size, 
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                total_time, 
                                                diffusion_coefficient, 
                                                lambda x, y: np.sin(np.pi * x))
    
    time_diffusion_sin.solve()
    time_diffusion_sin.plot_animation(save_name="sin_diffusion.gif")

    del time_diffusion_sin
    gc.collect()

def time_dep_diff_time_magnitudes(time_test_array, time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient):
    plot_y_slice_time_magnitudes(time_step_size, 
                                 x_length, 
                                 y_length, 
                                 n_steps, 
                                 diffusion_coefficient,
                                 time_test_array)
    

def main():
    return




if __name__ == '__main__':
    time_step_size = 0.00001
    x_length = 1.0
    y_length = 1.0
    n_steps = 100
    total_time = 1.0
    diffusion_coefficient = 1.0

    time_dep_diff_uniform(time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient)

    time_dep_diff_linear_xy(time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient)

    time_dep_diff_sin(time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient)


    time_test_array = [1.0, 0.1, 0.01, 0.001]

    time_dep_diff_time_magnitudes(time_test_array, time_step_size, x_length, y_length, n_steps, total_time, diffusion_coefficient)