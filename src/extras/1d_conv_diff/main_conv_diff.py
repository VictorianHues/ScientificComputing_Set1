import numpy as np

from conv_diffusion_1d import ConvectionDiffusion1D

def main():
    length = 1.0 
    spatial_points = 101
    total_time = 10.0
    time_step = 0.001

    convection_velocity = 0.0 # Convection velocity
    diffusion_coeff = 0.01 # Diffusion coefficient

    mean = 0.5
    stdDev = 0.1

    convection_diffusion = ConvectionDiffusion1D(length, 
                                                 spatial_points, 
                                                 total_time, 
                                                 time_step, 
                                                 convection_velocity, 
                                                 diffusion_coeff, 
                                                 initial_condition_func=lambda x: np.exp(-((x-mean)**2)/(2*stdDev**2)))


    convection_diffusion.solve()
    convection_diffusion.plot_heat_map()
    convection_diffusion.plot_animation()


if __name__ == "__main__":
    main()