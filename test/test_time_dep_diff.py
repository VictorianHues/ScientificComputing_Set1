import unittest
import numpy as np
from src.time_dep_diff import TimeDependentDiffusion

class TestTimeDependentDiffusion(unittest.TestCase):

    def setUp(self):
        self.time_step_size = 0.00001
        self.x_length = 1.0
        self.y_length = 1.0
        self.n_steps = 100
        self.total_time = 1.0
        self.diffusion_coefficient = 1.0
        self.initial_condition_func = lambda x, y: np.sin(np.pi * x)

    def test_initialization(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, self.x_length, self.y_length, self.n_steps, self.total_time, self.diffusion_coefficient, self.initial_condition_func)
        self.assertEqual(diffusion.time_step_size, self.time_step_size)
        self.assertEqual(diffusion.x_length, self.x_length)
        self.assertEqual(diffusion.y_length, self.y_length)
        self.assertEqual(diffusion.n_steps, self.n_steps)
        self.assertEqual(diffusion.total_time, self.total_time)
        self.assertEqual(diffusion.diffusion_coefficient, self.diffusion_coefficient)
        self.assertTrue(np.array_equal(diffusion.x_points, np.linspace(0, self.x_length, self.n_steps)))
        self.assertTrue(np.array_equal(diffusion.y_points, np.linspace(0, self.y_length, self.n_steps)))

    def test_stability_error(self):
        with self.assertRaises(ValueError):
            TimeDependentDiffusion(0.01, 
                                   self.x_length, 
                                   self.y_length, 
                                   self.n_steps, 
                                   self.total_time, 
                                   self.diffusion_coefficient,
                                   self.initial_condition_func)

        

    def test_solve(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, self.x_length, self.y_length, self.n_steps, self.total_time, self.diffusion_coefficient, self.initial_condition_func)
        solution = diffusion.solve()
        self.assertEqual(solution.shape, (diffusion.time_step_num, diffusion.n_steps, diffusion.n_steps))

    def test_analytical_solution(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, self.x_length, self.y_length, self.n_steps, self.total_time, self.diffusion_coefficient, self.initial_condition_func)
        analytical = diffusion.analytical_solution(diffusion.x_points, 0.5)
        self.assertEqual(analytical.shape, diffusion.x_points.shape)

    def test_compare_solutions(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, self.x_length, self.y_length, self.n_steps, self.total_time, self.diffusion_coefficient, self.initial_condition_func)
        diffusion.solve()
        try:
            diffusion.compare_solutions(10)
        except Exception as e:
            self.fail(f"compare_solutions raised an exception: {e}")

    def test_compare_solutions_full_range(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, 
                                           self.x_length, 
                                           self.y_length, 
                                           self.n_steps, 
                                           self.total_time, 
                                           self.diffusion_coefficient, 
                                           self.initial_condition_func)
        diffusion.solve()
        try:
            diffusion.compare_solutions_full_range(10)
        except Exception as e:
            self.fail(f"compare_solutions_full_range raised an exception: {e}")

    def test_plot_animation(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, self.x_length, self.y_length, self.n_steps, self.total_time, self.diffusion_coefficient, self.initial_condition_func)
        diffusion.solve()
        try:
            diffusion.plot_animation()
        except Exception as e:
            self.fail(f"plot_animation raised an exception: {e}")

    def test_plot_y_slice(self):
        diffusion = TimeDependentDiffusion(self.time_step_size, self.x_length, self.y_length, self.n_steps, self.total_time, self.diffusion_coefficient, self.initial_condition_func)
        diffusion.solve()
        try:
            diffusion.plot_y_slice(0.5)
        except Exception as e:
            self.fail(f"plot_y_slice raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()