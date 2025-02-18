import numpy as np

from vibr_string import VibratingString

def vibrating_string_sin(length, 
                         spatial_points, 
                         total_time, 
                         time_step_size, 
                         c, 
                         sin_constant,
                         save_name = None,
                         mask_start = None,
                         mask_end = None):
    string = VibratingString(length, 
                             spatial_points, 
                             total_time, 
                             time_step_size, 
                             c, 
                             lambda x: np.sin(sin_constant * np.pi * x),
                             mask_start,
                             mask_end)

    string.solve()
    string.plot_heat_map()
    string.plot_animation(save_name)

    string.plot_time_steps()


def vibrating_string_exp(length,
                            spatial_points,
                            total_time,
                            time_step_size,
                            c,
                            mean,
                            stdDev,
                            save_name = None):
        string = VibratingString(length, 
                                spatial_points, 
                                total_time, 
                                time_step_size, 
                                c, 
                                lambda x: np.exp(-((x - mean)**2) / (2 * stdDev**2)))
    
        string.solve()
        string.plot_heat_map()
        string.plot_animation(save_name)


        string.plot_time_steps()


def main():
    length = 1.0
    spatial_points = 1001
    total_time = 1.0
    time_step_size = 0.0001
    c = 1.0

    vibrating_string_sin(length, 
                         spatial_points, 
                         total_time, 
                         time_step_size, 
                         c, 
                         2.0,
                         save_name = 'vibrating_string_sin_2.gif',)
    
    vibrating_string_sin(length, 
                         spatial_points, 
                         total_time, 
                         time_step_size, 
                         c, 
                         5.0,
                         save_name = 'vibrating_string_sin_5.gif',)
    
    vibrating_string_sin(length, 
                         spatial_points, 
                         total_time, 
                         time_step_size, 
                         c, 
                         5.0,
                         save_name = 'vibrating_string_sin_5_masked.gif',
                         mask_start = 1/5,
                         mask_end = 2/5)


    mean = 0.5
    stdDev = 0.1
    
    vibrating_string_exp(length,
                        spatial_points,
                        total_time,
                        time_step_size,
                        c,
                        mean,
                        stdDev,
                        save_name = 'vibrating_string_exp.gif')

if __name__ == "__main__":
    main()