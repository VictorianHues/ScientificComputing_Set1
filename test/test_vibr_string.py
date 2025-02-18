import unittest
import numpy as np
from src.vibr_string import VibratingString

class TestVibratingString(unittest.TestCase):

    def test_initial_conditions(self):
        def initial_condition(x):
            return np.sin(np.pi * x)

        string = VibratingString(length=1.0, spatial_points=100, total_time=1.0, time_step_size=0.01, c=1.0, initial_condition_func=initial_condition)
        string.solve()

        for i in range(string.spatial_points):
            if string.mask[i]:
                self.assertAlmostEqual(string.Psi_grid[0, i], initial_condition(string.x_points[i]))
            else:
                self.assertEqual(string.Psi_grid[0, i], 0)

    def test_boundary_conditions(self):
        def initial_condition(x):
            return np.sin(np.pi * x)

        string = VibratingString(length=1.0, spatial_points=100, total_time=1.0, time_step_size=0.01, c=1.0, initial_condition_func=initial_condition)
        string.solve()

        self.assertTrue(np.all(string.Psi_grid[:, 0] == 0))
        self.assertTrue(np.all(string.Psi_grid[:, -1] == 0))

    def test_wave_propagation(self):
        def initial_condition(x):
            return np.sin(np.pi * x)

        string = VibratingString(length=1.0, spatial_points=100, total_time=1.0, time_step_size=0.01, c=1.0, initial_condition_func=initial_condition)
        string.solve()

        self.assertTrue(np.any(string.Psi_grid != 0))

    def test_mask_application(self):
        def initial_condition(x):
            return np.sin(np.pi * x)

        string = VibratingString(length=1.0, spatial_points=100, total_time=1.0, time_step_size=0.01, c=1.0, initial_condition_func=initial_condition, mask_start=0.2, mask_end=0.8)
        string.solve()

        for i in range(string.spatial_points):
            if 0.2 < string.x_points[i] < 0.8:
                self.assertAlmostEqual(string.Psi_grid[0, i], initial_condition(string.x_points[i]))
            else:
                self.assertEqual(string.Psi_grid[0, i], 0)

    def test_plot_heat_map(self):
        def initial_condition(x):
            return np.sin(np.pi * x)

        string = VibratingString(length=1.0, spatial_points=100, total_time=1.0, time_step_size=0.01, c=1.0, initial_condition_func=initial_condition)
        string.solve()

        try:
            string.plot_heat_map()
        except Exception as e:
            self.fail(f"plot_heat_map raised an exception: {e}")

    def test_plot_animation(self):
        def initial_condition(x):
            return np.sin(np.pi * x)

        string = VibratingString(length=1.0, spatial_points=100, total_time=1.0, time_step_size=0.01, c=1.0, initial_condition_func=initial_condition)
        string.solve()

        try:
            string.plot_animation()
        except Exception as e:
            self.fail(f"plot_animation raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()