import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from numba import njit, prange

@njit
def update_grid(Psi_grid, 
                c, 
                time_step_size, 
                spatial_step_size, 
                n, i):
    Psi_grid[n+1, i] = (c**2 * time_step_size**2 / spatial_step_size**2 * 
                        (Psi_grid[n, i+1] - 2*Psi_grid[n, i] + Psi_grid[n, i-1])
                        + 2*Psi_grid[n, i] - Psi_grid[n-1, i])

@njit(parallel=True)
def solve(Psi_grid, 
          c, 
          time_step_size, 
          spatial_step_size, 
          time_steps, 
          spatial_points):
    for n in range(1, time_steps - 1):
        for i in prange(1, spatial_points - 1):
            update_grid(Psi_grid, 
                        c, 
                        time_step_size, 
                        spatial_step_size, 
                        n, i)

    Psi_grid[:, 0] = 0
    Psi_grid[:, -1] = 0

class VibratingString:
    def __init__(self, 
                 length : float, 
                 spatial_points : int, 
                 total_time : float, 
                 time_step_size : float, 
                 c : float, 
                 initial_condition_func : callable, 
                 mask_start : Optional[float] = None, 
                 mask_end : Optional[float] = None):
        self.length = length
        self.spatial_points = spatial_points
        self.spatial_step_size = length / (spatial_points - 1)
        self.total_time = total_time
        self.time_step_size = time_step_size
        self.c = c
        self.initial_condition_func = initial_condition_func

        self.x_points = np.linspace(0, length, spatial_points)

        self.time_steps = int(total_time / time_step_size)

        # Create a mask to apply the initial condition only to a portion of the string
        if mask_start is None:
            mask_start = 0
        if mask_end is None:
            mask_end = self.length
        self.mask = (self.x_points > mask_start) & (self.x_points < mask_end)

        self.__initialize_psi_grid()


    def __initialize_psi_grid(self):
        self.Psi_grid = np.zeros((self.time_steps, self.spatial_points))

        for i in range(self.spatial_points):
            if self.mask[i]:
                self.Psi_grid[0, i] = self.initial_condition_func(self.x_points[i])
        self.Psi_grid[1, :] = self.Psi_grid[0, :]

        # Set Boundary Conditions
        self.Psi_grid[:, 0] = 0
        self.Psi_grid[:, -1] = 0

    def solve(self):
        solve(self.Psi_grid, self.c, self.time_step_size, self.spatial_step_size, self.time_steps, self.spatial_points)


    def plot_heat_map(self):
        plt.figure()
        plt.imshow(self.Psi_grid, aspect='auto', cmap='hot', extent=[0, self.length, self.total_time, 0])
        plt.colorbar(label='Psi')

        plt.xlabel('X')
        plt.ylabel('Time')
        plt.title('Vibrating String')

        plt.show()


    def plot_animation(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(self.x_points, self.Psi_grid[0])
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, self.length)

        self.close = False

        def close_event(event):
            self.close = True

        fig.canvas.mpl_connect('close_event', close_event)

        update_interval = 100

        for n in range(self.time_steps-1):
            if self.close:
                break

            if n % update_interval == 0:
                line.set_ydata(self.Psi_grid[n])
                fig.canvas.draw()
                plt.pause(self.time_step_size )

        plt.ioff()
        plt.show()
