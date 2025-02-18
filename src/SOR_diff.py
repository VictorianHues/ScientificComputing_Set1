import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from finite_difference import fast_solve, SOR

from scipy.special import erfc



class SORDiffusion:
    def __init__(self,
                 x_length : float,
                 y_length : float,
                 n_steps : int,
                 time_step_num : float,
                 omega : float,
                 initial_condition_func : callable,
                 mask = None):
        self.x_length = x_length
        self.y_length = y_length
        self.n_steps = n_steps
        self.time_step_num = time_step_num
        self.omega = omega
        self.initial_condition_func = initial_condition_func
        self.mask = mask

        self.x_points = np.linspace(0, self.x_length, self.n_steps)
        self.y_points = np.linspace(0, self.y_length, self.n_steps)

        self.x_step_size = self.x_length / (self.n_steps - 1)
        self.y_step_size = self.y_length / (self.n_steps - 1)

        self.c = np.zeros((self.time_step_num, self.n_steps, self.n_steps))

        self.c[0, -1, :] = self.initial_condition_func(self.x_points, self.y_points)
        self.c[0, 0, :] = 0.0
 
        # if self.omega > 2.0:
        #     raise ValueError(f"SOR becomes unstable for omega > 2, please use a smaller value. Current value: {self.omega}")

    def solve(self, tolerance = None):
        
        _, self.end_time = SOR(self.c, self.omega, mask=self.mask, tolerance=tolerance)
        if tolerance is not None:
            print('finished after ', self.end_time, ' iterations')
        return self.c, self.end_time
    
    def plot_animation(self):
        fig, ax = plt.subplots()
        heatmap = ax.imshow(self.c[0], cmap="hot", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Equilibrium Diffusion")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Concentration")

        def update(frame):
            heatmap.set_array(self.c[frame]) 
            ax.set_title(f"Equilibrium Diffusion (frame = {frame})")
            return heatmap,

        ani = animation.FuncAnimation(fig, update, frames=range(0, self.end_time, 100), interval=50, blit=False)
        plt.show()

    def plot_y_slice(self, x_val):
        x_idx = np.abs(self.x_points - x_val).argmin()

        plt.plot(self.y_points, self.c[-1, x_idx, :])
        plt.xlabel("Y")
        plt.ylabel("Concentration")
        plt.title(f"Concentration Profile at X = {x_val} and time = {self.time_step_num}")
        plt.show()

    def plot_single_frame(self, time):
        fig, ax = plt.subplots(figsize = (10,8))
        heatmap = ax.imshow(self.c[time,:,:], cmap="hot", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Numerical-Analytical Difference")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Difference")

        plt.show()
    
    def plot_analytical_sol(self):
        y_values = np.linspace(0, 1, self.n_steps)
        analytical_solution = np.tile(y_values[:, np.newaxis], (1, self.n_steps))

        fig, ax = plt.subplots(figsize = (10,8))
        heatmap = ax.imshow(analytical_solution, cmap="hot", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Numerical-Analytical Difference")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Difference")

        plt.show()


    def compare_solutions_full_range(self, time):
        y_values = np.linspace(0, 1, self.n_steps)
        analytical_solution = np.tile(y_values[:, np.newaxis], (1, self.n_steps))

        numerical = self.c[time, :, :]
        print(f"Time: {time}")

        difference = np.abs(numerical - analytical_solution)
        rmse = np.sqrt(np.mean(difference**2))
        print(f"Root Mean Squared Error: {rmse}")

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(difference, cmap="coolwarm", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Numerical-Analytical Dif at {time} (RMSE: {rmse})")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Difference")

        plt.show()