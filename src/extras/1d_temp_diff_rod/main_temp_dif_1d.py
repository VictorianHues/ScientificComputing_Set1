import numpy as np

from temp_diffus_1d import TemperatureDiffusion1D

def main():
    length = 1.0
    spatial_points = 101
    total_time = 5.0
    time_step_size = 0.001

    alpha = 0.01
    const_temp = 100.0

    mean = 0.5
    std_dev = 0.1

    temp_diffusion = TemperatureDiffusion1D(length,
                                            spatial_points,
                                            total_time,
                                            time_step_size,
                                            alpha,
                                            const_temp,
                                            lambda x: np.exp(-((x - mean)**2) / (2 * std_dev**2)))

    temp_diffusion.solve()

    temp_diffusion.plot_heat_map()
    temp_diffusion.plot_animation()

    sine_constant = 2.0

    temp_diffusion = TemperatureDiffusion1D(length,
                                            spatial_points,
                                            total_time,
                                            time_step_size,
                                            alpha,
                                            const_temp,
                                            lambda x: np.sin(sine_constant * np.pi * x))
    
    temp_diffusion.solve()

    temp_diffusion.plot_heat_map()
    temp_diffusion.plot_animation()


if __name__ == "__main__":
    main()