import unittest
import numpy as np
from build_busan import fit_od, paths_between, assign


class BusanTest(unittest.TestCase):
    def test_balances_marginals_without_same_station_trips(self):
        cost = np.array([[0., 2, 6], [2, 0, 4], [6, 4, 0]])
        b, a = np.array([30., 40, 30]), np.array([20., 30, 40])
        od, error, _ = fit_od(cost, b, a, 20)
        np.testing.assert_allclose(od.sum(axis=1), b, atol=1e-5)
        np.testing.assert_allclose(od.sum(axis=0), a * 100/90, atol=1e-5)
        np.testing.assert_array_equal(od.diagonal(), 0)
        self.assertLess(error, 1e-5)

    def test_transfer_cost_and_directional_assignment(self):
        graph = {1: [(2, 2., 0, 0)],
                 2: [(1, 2., 0, 1), (3, 5., None, None)],
                 3: [(2, 5., None, None), (4, 3., 1, 0)],
                 4: [(3, 3., 1, 1)]}
        names, costs, paths = paths_between(graph, {'A': [1], 'B': [2, 3], 'C': [4]})
        self.assertEqual(costs[0, 2], 10)
        self.assertEqual(costs[1, 2], 3)  # no transfer penalty when entering here
        od = np.zeros((3, 3))
        od[0, 2], od[2, 0] = 100, 25
        np.testing.assert_array_equal(assign(od, paths, 2), [[100, 25], [100, 25]])

    def test_reject_disconnected_graph(self):
        with self.assertRaises(ValueError):
            fit_od(np.array([[0., np.inf], [np.inf, 0]]), np.ones(2), np.ones(2), 20)

    def test_journey_assumption_changes_profile(self):
        costs = np.abs(np.arange(5)[:, None]-np.arange(5)[None, :]) * 10.
        b = np.array([10., 30, 40, 30, 10])
        short, _, _ = fit_od(costs, b, b, 10)
        long, _, _ = fit_od(costs, b, b, 30)
        self.assertGreater((long*costs).sum(), (short*costs).sum())


if __name__ == '__main__':
    unittest.main()
