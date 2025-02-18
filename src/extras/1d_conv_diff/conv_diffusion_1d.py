import numpy as np
import matplotlib.pyplot as plt

class ConvectionDiffusion1D:
    def __init__(self,
                length,
                spatial_points,
                total_time,
                time_step,
                convection_velocity,
                diffusion_coeff,
                initial_condition_func,
                mask_start=None,
                mask_end=None):
            self.length = length
            self.spatial_points = spatial_points
            self.spatial_step = length / (spatial_points - 1)
            self.total_time = total_time
            self.time_step = time_step
            self.convection_velocity = convection_velocity
            self.diffusion_coeff = diffusion_coeff
            self.initial_condition_func = initial_condition_func

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


    def solve(self):
        for n in range(0, self.time_steps - 1):
            for i in range(1, self.spatial_points - 1):
                convection = - (self.convection_velocity * self.time_step / self.spatial_step) * (self.u_grid[n, i] - self.u_grid[n, i-1])
                diffusion = (self.diffusion_coeff * self.time_step / self.spatial_step**2) * (self.u_grid[n, i+1] - 2*self.u_grid[n, i] + self.u_grid[n, i-1])
                
                self.u_grid[n+1, i] = self.u_grid[n, i] + convection + diffusion

    
    def plot_heat_map(self):
        plt.figure()
        plt.imshow(self.u_grid, aspect='auto', cmap='hot', extent=[0, self.length, self.total_time, 0])
        plt.colorbar(label='Temperature')

        plt.title('Convection-Diffusion 1D')
        plt.xlabel('Length')
        plt.ylabel('Time')

        plt.show()

    def plot_animation(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(self.x_points, self.u_grid[0])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, self.length)

        self.close = False

        def close_event(event):
            self.close = True

        fig.canvas.mpl_connect('close_event', close_event)

        for n in range(self.time_steps-1):
            if self.close:
                break
            line.set_ydata(self.u_grid[n])
            fig.canvas.draw()
            plt.pause(0.01)

        plt.ioff()
        plt.show()