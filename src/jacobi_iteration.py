import numpy as np
from finite_difference import jacobi
from cpp_modules import fast_finite_difference as ffd
import time
class Jacobi:
    def __init__(self, n_steps, epsilon=1e-5, max_iter=100000):
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.max_iter = max_iter
        
        reset_state(self)

    def solve(self):
        
        self.max_iter = jacobi(self.c_old, self.c_new, self.max_iter, self.n_steps, self.epsilon)
        # for iteration in range(self.max_iter):
            
        #     # Top and Bottom Boundaries
        #     self.c_old[:, -1] = 1.0
        #     self.c_old[:, 0] = 0.0

        #     # Jacobi update rule
        #     for i in range(self.n_steps):
        #         for j in range(1, self.n_steps - 1):
        #             self.c_new[i, j] = 0.25 * (
        #                 self.c_old[(i+1) % self.n_steps, j] +
        #                 self.c_old[(i-1) % self.n_steps, j] +
        #                 self.c_old[i, j+1] +
        #                 self.c_old[i, j-1]
        #             )

        #     # Stopping criterion
        #     delta = np.max(np.abs(self.c_new - self.c_old))
        #     if delta < self.epsilon:
        #         return self.c_new, iteration

        #     # Update step
        #     np.copyto(self.c_old, self.c_new) 

        return self.c_new, self.max_iter
    
def reset_state(self):
    
    # Initialize grid (two matrices c_old and c_new)
    self.c_new = np.zeros((self.n_steps, self.n_steps))
    self.c_new[:, -1] = 1.0 
    self.c_new[:, 0] = 0.0   
    self.c_old = self.c_new.copy()


def bench_functions(self):

    t = 0
    num_runs = 10
    self.max_iter = 10000
    max_iter = jacobi(self.c_old, self.c_new, self.max_iter, self.n_steps, self.epsilon)
    for i in range(num_runs):
        reset_state(self)
        start = time.time()
        iter = jacobi(self.c_old, self.c_new, self.max_iter, self.n_steps, self.epsilon)
        t += time.time() - start
        if(iter != max_iter):
            print(iter, max_iter)
    print(t)
            
    t = 0
    for i in range(num_runs):
        reset_state(self)
        start = time.time()
        iter = ffd.fast_jacobi(self.c_old, self.c_new, self.max_iter, self.n_steps, self.epsilon)
        t += time.time() - start
        if(iter != max_iter):
            print(iter, max_iter)

    print(t)



    
# Example usage
if __name__ == '__main__':
    solver = Jacobi(50)
    bench_functions(solver)
    # c_new, iteration = solver.solve()
    # print(f"Converged after {iteration} iterations")
    # print(c_new)
