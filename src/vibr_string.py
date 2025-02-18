import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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
            update_grid(Psi_grid, c, time_step_size, spatial_step_size, n, i)
        
        # Enforce boundary conditions at each time step
        Psi_grid[n+1, 0] = 0
        Psi_grid[n+1, -1] = 0

class VibratingString:
    def __init__(self, 
                 length: float, 
                 spatial_points: int, 
                 total_time: float, 
                 time_step_size: float, 
                 c: float, 
                 initial_condition_func: callable, 
                 mask_start: Optional[float] = None, 
                 mask_end: Optional[float] = None):
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
        plt.figure(figsize=(6, 5))
        plt.imshow(self.Psi_grid, aspect='auto', cmap='hot', extent=[0, self.length, self.total_time, 0])
        plt.colorbar(label='Y Position')

        plt.xlabel('X Position')
        plt.ylabel('Time')
        plt.title('Vibrating String Heatmap')
        plt.show()

    def plot_animation(self, gif_filename="vibrating_string.gif"):
        base_dir = os.path.dirname(os.path.dirname(__file__)) 
        animations_dir = os.path.join(base_dir, 'animations')
        print(animations_dir)
        os.makedirs(animations_dir, exist_ok=True)

        gif_filepath = os.path.join(animations_dir, gif_filename)

        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot(self.x_points, self.Psi_grid[0], color='b')
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, self.length)

        def update(n):
            line.set_ydata(self.Psi_grid[n])
            return line,

        update_interval = max(1, self.time_steps // 200)  # Keep ~200 frames for smooth animation
        ani = animation.FuncAnimation(fig, update, frames=range(0, self.time_steps, update_interval), blit=True)

        ani.save(gif_filepath, writer='pillow', fps=120)

        plt.show()
