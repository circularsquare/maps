import unittest

import numpy as np

from build_metro_stations import station_margins


class MetroStationTest(unittest.TestCase):
    def test_station_margins_count_each_trip_at_two_ends(self):
        pairs = np.array([[0, 1], [0, 2], [2, 1]])
        hourly = np.array([[3, 2], [4, 0], [1, 5]], dtype=float)
        board, alight = station_margins(pairs, hourly, 3)
        np.testing.assert_array_equal(board, [9, 0, 6])
        np.testing.assert_array_equal(alight, [0, 11, 4])
        self.assertEqual(board.sum(), alight.sum())


if __name__ == "__main__":
    unittest.main()
