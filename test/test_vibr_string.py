import unittest
import numpy as np
import noise
import logging

from vibr_string import VibratingString
from func_processor import FunctionProcessor

class test:
    def setUp(self):
        self.intervals = 50
        class TestVibratingString(unittest.TestCase):
            def setUp(self):
                self.length = 10.0
                self.spatial_points = 100
                self.total_time = 5.0
                self.time_step_size = 0.01
                self.c = 1.0
                self.initial_condition_func = FunctionProcessor(lambda x: np.sin(np.pi * x / self.length))
                self.mask_start = 2.0
                self.mask_end = 8.0

            def test_initialization(self):
                vs = VibratingString(self.length, self.spatial_points, self.total_time, self.time_step_size, self.c, self.initial_condition_func, self.mask_start, self.mask_end)
                
                self.assertEqual(vs.length, self.length)
                self.assertEqual(vs.spatial_points, self.spatial_points)
                self.assertEqual(vs.total_time, self.total_time)
                self.assertEqual(vs.time_step_size, self.time_step_size)
                self.assertEqual(vs.c, self.c)
                
                np.testing.assert_array_equal(vs.x_points, np.linspace(0, self.length, self.spatial_points))
                self.assertEqual(vs.time_steps, int(self.total_time / self.time_step_size))
                
                expected_mask = (vs.x_points > self.mask_start) & (vs.x_points < self.mask_end)
                np.testing.assert_array_equal(vs.mask, expected_mask)
                
                self.assertEqual(vs.Psi_grid.shape, (vs.time_steps, self.spatial_points))
                self.assertTrue(np.all(vs.Psi_grid[0, ~vs.mask] == 0))
                self.assertTrue(np.all(vs.Psi_grid[:, 0] == 0))
                self.assertTrue(np.all(vs.Psi_grid[:, -1] == 0))

            def test_initial_condition_application(self):
                vs = VibratingString(self.length, self.spatial_points, self.total_time, self.time_step_size, self.c, self.initial_condition_func, self.mask_start, self.mask_end)
                
                for i in range(vs.spatial_points):
                    if vs.mask[i]:
                        self.assertEqual(vs.Psi_grid[0, i], self.initial_condition_func.get_func()(vs.x_points[i]))
                    else:
                        self.assertEqual(vs.Psi_grid[0, i], 0)

        if __name__ == '__main__':
            unittest.main()