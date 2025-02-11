import numpy as np
import matplotlib.pyplot as plt

from time_dep_diff import TimeDependentDiffusion

def plot_y_slice_time_magnitudes(time_step_size, 
                                 x_length, 
                                 y_length, 
                                 n_steps, 
                                 diffusion_coefficient,
                                 time_array):
    
    plt.figure(figsize=(10, 6))
    plt.title("Time Slice Magnitudes")
    plt.xlabel("X")
    plt.ylabel("Concentration")

    for time in time_array:
        print("Time: ", time)
        time_diffusion = TimeDependentDiffusion(time_step_size, 
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                time, 
                                                diffusion_coefficient, 
                                                lambda x, y: 1)
        
        solution = time_diffusion.solve()

        x_idx = np.abs(time_diffusion.x_points - n_steps//2).argmin()

        plt.plot(time_diffusion.y_points, 
                 solution[-1, x_idx, :], 
                 label=f"Time: {time}")
    
    plt.legend()
    plt.show()


    
