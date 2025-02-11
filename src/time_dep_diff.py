import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import concurrent.futures

from scipy.special import erfc

class TimeDependentDiffusion:
    def __init__(self,
                 time_step_size : float,
                 x_length : float,
                 y_length : float,
                 n_steps : int,
                 total_time : float,
                 diffusion_coefficient : float,
                 initial_condition_func : callable):
        self.time_step_size = time_step_size
        self.x_length = x_length
        self.y_length = y_length
        self.n_steps = n_steps
        self.total_time = total_time
        self.diffusion_coefficient = diffusion_coefficient
        self.initial_condition_func = initial_condition_func

        self.x_points = np.linspace(0, self.x_length, self.n_steps)
        self.y_points = np.linspace(0, self.y_length, self.n_steps)

        self.x_step_size = self.x_length / (self.n_steps - 1)
        self.y_step_size = self.y_length / (self.n_steps - 1)

        self.time_step_num = int(self.total_time / self.time_step_size)

        self.c = np.zeros((self.time_step_num, self.n_steps, self.n_steps))

        self.c[0, :, -1] = self.initial_condition_func(self.x_points, self.y_points)
        self.c[0, :, 0] = 0.0

        stable_val = (4 * self.diffusion_coefficient * self.time_step_size) / (self.x_step_size/2**2)
        if stable_val > 1.0:
            raise ValueError(f"Unstable solution, please use smaller diffusion coefficient or time step size. Current value: {stable_val}")


    def solve(self):
        for t in range(1, self.time_step_num - 1):
            new_c = self.c[t].copy()  
            
            for i in range(1, self.n_steps - 1):
                for j in range(self.n_steps - 1):
                    new_c[i, j] = (
                        self.c[t, i, j] +
                        (self.time_step_size * self.diffusion_coefficient / self.x_step_size**2) *
                        (self.c[t, i+1, j] + self.c[t, i-1, j] + self.c[t, i, j+1 % self.n_steps] + self.c[t, i, j-1 % self.n_steps] - 4 * self.c[t, i, j])
                    )

            # Left and Right Boundaries
            #new_c[0, :] = new_c[-2, :]
            #new_c[-1, :] = new_c[1, :]

            self.c[t+1] = new_c

            # Top and Bottom Boundaries
            self.c[t+1, :, -1] = self.initial_condition_func(self.x_points, self.y_points)
            self.c[t+1, :, 0] = 0.0

        return self.c
    
    def plot_animation(self):
        fig, ax = plt.subplots()
        heatmap = ax.imshow(self.c[0], cmap="hot", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Time-Dependent Diffusion")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Concentration")

        def update(frame):
            heatmap.set_array(self.c[frame].T) 
            ax.set_title(f"Time-Dependent Diffusion (t = {frame/self.time_step_num})")
            return heatmap,

        ani = animation.FuncAnimation(fig, update, frames=self.time_step_num, interval=50, blit=False)

        plt.show()

    def plot_y_slice(self, x_val):
        x_idx = np.abs(self.x_points - x_val).argmin()

        plt.plot(self.y_points, self.c[-1, x_idx, :])
        plt.xlabel("Y")
        plt.ylabel("Concentration")
        plt.title(f"Concentration Profile at X = {x_val} and time = {self.total_time}")
        plt.show()

    def analytical_solution(self, x, t, N=100):  
        sum = np.zeros_like(x, dtype=np.float64)

        for i in range(N):
            term1 = erfc((1 - x + 2 * i) / (2 * np.sqrt(self.diffusion_coefficient * t)))
            term2 = erfc((1 + x + 2 * i) / (2 * np.sqrt(self.diffusion_coefficient * t)))

            sum += term1 - term2

        return sum

    def compare_solutions(self,time_index):
        selected_time = self.time_step_size * time_index  # Corresponding time

        numerical = self.c[time_index, self.n_steps // 2, :]  # Extract middle column, range of y values at x

        analytical_solution = self.analytical_solution(self.x_points, selected_time)

        error = np.abs(numerical - analytical_solution)
        rmse = np.sqrt(np.mean(error**2) / self.n_steps)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(self.x_points, numerical, 'bo-', label="Numerical (Simulation)")
        ax1.plot(self.x_points, analytical_solution, 'r--', label="Analytical (erfc solution)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("Concentration")
        ax1.set_title(f"Comparison at t = {selected_time:.2f}, Root Mean Squared Error = {rmse:.5f}")
        ax1.legend(loc='upper left')
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.x_points, error, 'g-', label="Error")
        ax2.set_ylabel("Error")
        ax2.legend(loc='lower left')

        plt.show()