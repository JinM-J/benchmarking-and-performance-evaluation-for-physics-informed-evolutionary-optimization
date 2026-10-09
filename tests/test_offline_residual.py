"""Offline boundary derivative tests; no training or reference data."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.offline_residual import condition_value, condition_metrics
from problems.f08 import F08
from reproduction.evaluate_checkpoint import replay_problem


class CubicPrediction:
    def predict(self, queries):
        return queries[:, 0]**3 + 2 * queries[:, 1]**3


class BoundaryDiagnosticTests(unittest.TestCase):
    def test_second_and_third_derivatives_at_boundaries_and_interior(self):
        problem = F08()
        queries = np.array([[0., 0.], [.02, .02], [.5, .5], [1.98, .98], [2., 1.]])
        for axis, factor in (('x', 1.), ('t', 2.)):
            column = problem.physics.coordinate_names.index(axis)
            np.testing.assert_allclose(
                condition_value(problem, CubicPrediction(), queries, (axis, axis), 32),
                6 * factor * queries[:, column], rtol=0, atol=1e-9)
            np.testing.assert_allclose(
                condition_value(problem, CubicPrediction(), queries, (axis, axis, axis), 32),
                np.full(len(queries), 6 * factor), rtol=0, atol=1e-8)

    def test_declared_f8_boundary_orders_are_reported_without_duplicate_values(self):
        metrics = condition_metrics(F08(), CubicPrediction(), 32)
        self.assertIn('bc_dxx_rmse', metrics)
        self.assertIn('bc_dxxx_rmse', metrics)
        values = [row for row in metrics['condition_components'] if row['quantity'] == 'bc_u']
        self.assertEqual(len(values), 1)

    def test_recorded_ks_profile_keeps_first_derivative_diagnostics(self):
        problem = replay_problem('f08', 'ks_value_first_derivative')
        orders = [len(row.derivative) for row in problem.physics.boundary_conditions()]
        self.assertEqual(orders, [1])
        metrics = condition_metrics(problem, CubicPrediction(), 32)
        self.assertIn('bc_dx_rmse', metrics)
        self.assertNotIn('bc_dxx_rmse', metrics)

    def test_profile_cannot_change_another_problem(self):
        with self.assertRaisesRegex(ValueError, 'another problem'):
            replay_problem('f06', 'ks_value_first_derivative')


if __name__ == '__main__':
    unittest.main()
