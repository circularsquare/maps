import unittest

import numpy as np

from build_small_cities import line_assignment


class SmallCityTest(unittest.TestCase):
    def test_each_trip_crosses_every_intermediate_edge(self):
        od = np.zeros((4, 4))
        od[0, 3] = 100
        od[2, 0] = 25
        np.testing.assert_array_equal(
            line_assignment(od),
            [[100, 25], [100, 25], [100, 0]],
        )

    def test_assignment_conserves_station_flow(self):
        od = np.array([
            [0, 2, 3],
            [4, 0, 5],
            [6, 7, 0],
        ], dtype=float)
        loads = line_assignment(od)
        net = np.r_[loads[0, 0] - loads[0, 1],
                    loads[0, 1] - loads[0, 0] + loads[1, 0] - loads[1, 1],
                    loads[-1, 1] - loads[-1, 0]]
        np.testing.assert_array_equal(net, od.sum(axis=1) - od.sum(axis=0))


if __name__ == "__main__":
    unittest.main()
