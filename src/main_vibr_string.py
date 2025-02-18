import numpy as np
import math

from vibr_string import VibratingString


def main():
    length = 1.0
    spatial_points = 1001
    total_time = 1.0
    time_step_size = 0.0001
    c = 1.0

    sine_constant = 2
    

    string = VibratingString(length, 
                             spatial_points, 
                             total_time, 
                             time_step_size, 
                             c, 
                             lambda x: np.sin(sine_constant * np.pi * x))

    string.solve()
    string.plot_heat_map()
    string.plot_animation()

    mean = 0.5
    stdDev = 0.1
    
    string = VibratingString(length, 
                             spatial_points, 
                             total_time, 
                             time_step_size, 
                             c, 
                             lambda x: np.exp(-((x - mean)**2) / (2 * stdDev**2)))

    string.solve()
    string.plot_heat_map()
    string.plot_animation()



    sine_constant = 5

    string = VibratingString(length, 
                             spatial_points, 
                             total_time, 
                             time_step_size, 
                             c, 
                             lambda x: np.sin(sine_constant * np.pi * x))

    string.solve()
    string.plot_heat_map()
    string.plot_animation()

    mask_start = 1/5
    mask_end = 2/5

    string = VibratingString(length, 
                             spatial_points, 
                             total_time, 
                             time_step_size, 
                             c, 
                             lambda x: np.sin(sine_constant * np.pi * x),
                             mask_start,
                             mask_end)

    string.solve()
    string.plot_heat_map()
    string.plot_animation()

if __name__ == "__main__":
    main()