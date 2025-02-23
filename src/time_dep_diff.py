import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.special import erfc
from numba import jit, njit, prange

@njit
def finite_dif_method(c_mesh_new, 
                      t, i, j, 
                      c_mesh, 
                      time_step_size, 
                      diffusion_coefficient, 
                      x_step_size, 
                      n_steps):
    c_mesh_new[i, j] = (
        c_mesh[t, i, j] +
        (time_step_size * diffusion_coefficient / x_step_size**2) *
        (c_mesh[t, (i+1)% n_steps, j] + c_mesh[t, (i-1)% n_steps, j] + c_mesh[t, i, j+1] + c_mesh[t, i, j-1 ] - 4 * c_mesh[t, i, j])
    )
    

@njit(parallel=True)
def fast_solve(c, 
               time_step_num, 
               n_steps, 
               time_step_size, 
               diffusion_coefficient, 
               x_step_size):
    for t in range(0, time_step_num - 1):
        new_c = c[t].copy()  
        
        for i in prange(n_steps):
            for j in range(1, n_steps - 1):
                finite_dif_method(new_c, 
                                  t, i, j, 
                                  c, 
                                  time_step_size, 
                                  diffusion_coefficient, 
                                  x_step_size, 
                                  n_steps)

        c[t+1] = new_c

        # Top and Bottom Boundaries
        c[t+1, :, -1] = c[t,:,-1]
        c[t+1, :, 0] = 0.0

    return c

def plot_y_slice_time_magnitudes(time_step_size, 
                                 x_length, 
                                 y_length, 
                                 n_steps, 
                                 diffusion_coefficient,
                                 time_array):
    
    plt.figure(figsize=(10, 6))
    plt.title("Time Slice Magnitudes")
    plt.xlabel("X")
    plt.ylabel("Concentration")

    for time in time_array:
        print("Time: ", time)
        time_diffusion = TimeDependentDiffusion(time_step_size, 
                                                x_length, 
                                                y_length, 
                                                n_steps, 
                                                time, 
                                                diffusion_coefficient, 
                                                lambda x, y: 1)
        
        solution = time_diffusion.solve()

        x_idx = np.abs(time_diffusion.x_points - n_steps//2).argmin()

        plt.plot(time_diffusion.y_points, 
                 solution[-1, x_idx, :], 
                 label=f"Time: {time}")
    
    plt.legend()
    plt.show()


class TimeDependentDiffusion:
    """
    class that stores all simulation parameters and offers some default methods for visualization  
    for the SOR method for time-independent diffusion
    """
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

        print(f"Time step number: {self.time_step_num}")
        print(f"X steps: {self.n_steps}")
        print(f"Y steps: {self.n_steps}")

        self.c = np.zeros((self.time_step_num, self.n_steps, self.n_steps))

        self.c[0, :, -1] = self.initial_condition_func(self.x_points, self.y_points)
        self.c[0, :, 0] = 0.0

        stable_val = (4 * self.diffusion_coefficient * self.time_step_size) / (self.x_step_size**2)
        if stable_val > 1.0:
            raise ValueError(f"Unstable solution, please use smaller diffusion coefficient or time step size. Current value: {stable_val}")


    def solve(self):
        self.c = fast_solve(self.c, self.time_step_num, self.n_steps, self.time_step_size, self.diffusion_coefficient, self.x_step_size)
        
        return self.c
    

    def plot_animation(self, save_name="animation.gif"):
        base_dir = os.path.dirname(os.path.dirname(__file__)) 
        animations_dir = os.path.join(base_dir, 'animations')

        os.makedirs(animations_dir, exist_ok=True)

        gif_filepath = os.path.join(animations_dir, save_name)


        fig, ax = plt.subplots()
        heatmap = ax.imshow(self.c[0], cmap="hot", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Time-Dependent Diffusion")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Concentration")

        def update(frame):
            heatmap.set_array(self.c[frame].T) 
            return heatmap,

        ani = animation.FuncAnimation(fig, update, frames=range(0, self.time_step_num, 100), interval=50, blit=False)

        ani.save(gif_filepath, writer='pillow', fps=120)

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

    def compare_solutions_full_range(self, time_index):
        selected_time = self.time_step_size * time_index  # Corresponding time

        analytical_solution = self.analytical_solution(self.x_points, selected_time)

        numerical = self.c[time_index, :, :]

        difference = np.abs(numerical - analytical_solution)

        rmse = np.sqrt(np.mean(difference**2))

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(np.transpose(difference), cmap="coolwarm", origin="lower", extent=[0, self.x_length, 0, self.y_length])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Numerical-Analytical Difference at t = {selected_time:.2f}, RMSE = {rmse:.5f}")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Difference")

        plt.show()