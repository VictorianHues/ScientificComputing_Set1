import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from finite_difference import jacobi
class Jacobi:
    """
    class that stores all simulation parameters and offers some default methods for visualization  
    for the Jacobi method for time-independent diffusion
    """
    def __init__(self, 
                 n_steps, 
                 epsilon=1e-5, 
                 max_iter=100000):
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.max_iter = max_iter
        
        # Initialize grid (two matrices c_old and c_new)
        self.c_new = np.zeros((n_steps, n_steps))
        self.c_new[:, -1] = 1.0 
        self.c_new[:, 0] = 0.0   
        self.c_old = self.c_new.copy()

    def solve(self):
        self.max_iter, tol = jacobi(self.c_old, self.c_new, self.max_iter, self.n_steps, self.epsilon)
        # for iteration in range(self.max_iter):
            
        #     # Top and Bottom Boundaries
        #     self.c_old[:, -1] = 1.0
        #     self.c_old[:, 0] = 0.0

        return self.c_new, self.max_iter
    

    def plot_last_frame(self):
        fig, ax = plt.subplots(figsize = (10,8))
        heatmap = ax.imshow(np.transpose(self.c_new[:,:]), cmap="hot", origin="lower", extent=[0, 1.0, 0, 1.0])
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
        heatmap = ax.imshow(analytical_solution, cmap="hot", origin="lower", extent=[0, 1.0, 0, 1.0])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Numerical-Analytical Difference")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Difference")

        plt.show()

    def compare_solutions(self):
        analytical_solution = np.linspace(0, 1, self.n_steps)

        numerical = np.transpose(self.c_new[self.n_steps // 2, :])  # Extract middle column, range of y values at x

        error = np.abs(numerical - analytical_solution)
        rmse = np.sqrt(np.mean(error**2) / self.n_steps)

        x_points = np.linspace(0, 1, self.n_steps)  # Extract x_points from c_new

        fig, ax1 = plt.subplots(figsize=(8, 4))

        fig.subplots_adjust(bottom = 0.15)

        ax1.plot(x_points, numerical, 'bo-', label="Numerical", markersize = 3)
        ax1.plot(x_points, analytical_solution, 'r--', label="Analytical")
        ax1.set_xlabel("x", fontsize=16)
        ax1.set_ylabel("Concentration", fontsize=16)
        #ax1.set_title(f"Comparison at equilibrium, Root Mean Squared Error = {rmse:.5f}")
        ax1.legend(loc='upper left', fontsize=14)
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(x_points, error, 'g-', label="Error")
        ax2.set_ylabel("Error", fontsize=16)
        ax2.legend(loc='lower right', fontsize=14)

        plt.show()


    def compare_solutions_full_range(self):
        y_values = np.linspace(0, 1, self.n_steps)
        analytical_solution = np.tile(y_values[:, np.newaxis], (1, self.n_steps))

        numerical = np.transpose(self.c_new[:, :])

        difference = np.abs(numerical - analytical_solution)
        rmse = np.sqrt(np.mean(difference**2))
        print(f"Root Mean Squared Error: {rmse}")

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(difference, cmap="coolwarm", origin="lower", extent=[0, 1.0, 0, 1.0])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Numerical-Analytical Dif (RMSE: {rmse})")

        cbar = plt.colorbar(heatmap)
        cbar.set_label("Difference")

        plt.show()