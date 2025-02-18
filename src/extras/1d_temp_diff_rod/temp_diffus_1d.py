import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

class TemperatureDiffusion1D:
    def __init__(self, 
                 length : float, 
                 spatial_points : int, 
                 total_time : float, 
                 time_step : float, 
                 alpha : float, 
                 const_temp : float,
                 initial_condition_function : callable,
                 mask_start : Optional[float] = None,
                 mask_end : Optional[float] = None):
        self.length = length
        self.spatial_points = spatial_points
        self.spatial_step = length / (spatial_points - 1)
        self.total_time = total_time
        self.time_step = time_step
        self.alpha = alpha
        self.const_temp = const_temp
        self.initial_condition_func = initial_condition_function

        self.x_points = np.linspace(0, length, spatial_points)

        self.time_steps = int(total_time / time_step)

        if mask_start is None:
            mask_start = 0
        if mask_end is None:
            mask_end = self.length
        self.mask = (self.x_points > mask_start) & (self.x_points < mask_end)

        self.__initialize_u_grid()

    
    def __initialize_u_grid(self):
        self.u_grid = np.zeros((self.time_steps, self.spatial_points))

        for i in range(self.spatial_points):
            if self.mask[i]:
                self.u_grid[0, i] = self.initial_condition_func(self.x_points[i])

        self.u_grid[:, 0] = self.const_temp

    def solve(self):
        """
        for i in range(1, self.spatial_points - 1):
            self.u_grid[1, i] = ((self.alpha * self.time_step / self.spatial_step**2)
                                 * (self.u_grid[n, i+1] - 2*self.u_grid[n, i] + self.u_grid[n, i-1])
                                 + self.u_grid[n, i])
                                 """

        for n in range(1, self.time_steps-1):
            for i in range(1, self.spatial_points-1):
                self.u_grid[n+1, i] = ((self.alpha * self.time_step / self.spatial_step**2)
                                       * (self.u_grid[n, i+1] - 2*self.u_grid[n, i] + self.u_grid[n, i-1])
                                       + self.u_grid[n, i])
            
            self.u_grid[n+1, -1] = self.u_grid[n+1, -2]

    def plot_heat_map(self):
        plt.figure()
        plt.imshow(self.u_grid, aspect='auto', cmap='hot', extent=[0, self.length, self.total_time, 0])
        plt.colorbar(label='Temperature')

        plt.title('Temperature Diffusion in 1D')
        plt.xlabel('Length')
        plt.ylabel('Time')

        plt.show()
        


    def plot_animation(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(self.x_points, self.u_grid[0])
        ax.set_ylim(0, self.const_temp)
        ax.set_xlim(0, self.length)

        self.close = False

        def close_event(event):
            self.close = True

        fig.canvas.mpl_connect('close_event', close_event)

        for n in range(self.time_steps-1):
            if self.close:
                break
            if n % 10 == 0:
                line.set_ydata(self.u_grid[n])
                fig.canvas.draw()
                plt.pause(0.01)

        plt.ioff()
        plt.show()