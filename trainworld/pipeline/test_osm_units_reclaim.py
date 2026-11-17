"""T-3428: coastal land no unit covers joins the unit it borders; inland strips and far land stay out."""
import unittest

from shapely.geometry import box

import osm_units as ou

LAT = 22.5


class Reclaim(unittest.TestCase):
    def test_reclaimed_strip_joins_longest_border(self):
        a = box(0.00, 0.00, 0.10, 0.02)     # unit A, above
        b = box(0.00, -0.02, 0.04, 0.00)    # unit B below, short border with the strip
        sea = box(-0.5, -0.5, 0.5, -0.0)    # old sea; the strip below A's south edge is land
        sea = sea.difference(box(0.04, -0.02, 0.10, 0.00)).difference(b)
        got = ou.reclaimed([a, b], sea, LAT)
        self.assertEqual([j for j, _ in got], [0])

    def test_open_inland_strip_stays_out(self):
        a = box(0.0, 0.0, 0.1, 0.1)
        sea = box(-0.5, -0.5, 0.0, 0.5)     # sea only to the west; land everywhere else
        self.assertEqual(ou.reclaimed([a], sea, LAT), [])

    def test_far_land_stays_out(self):
        a = box(0.0, 0.0, 0.1, 0.1)
        sea = box(0.3, -1, 1, 1).union(box(-1, -1, 0.2, -0.5))
        self.assertEqual(ou.reclaimed([a], sea, LAT), [])

    def _bay(self):
        a = box(0.00, 0.00, 0.10, 0.02)
        sea = box(-0.5, -0.5, 0.5, 0.0)
        return a, sea

    def test_piece_in_foreign_land_is_dropped(self):
        a, sea = self._bay()
        sea = sea.difference(box(0.04, -0.02, 0.10, 0.00))     # a strip of reclaimed land below A
        self.assertEqual([j for j, _ in ou.reclaimed([a], sea, LAT)], [0])
        monaco = box(0.0, -0.1, 0.2, 0.0)
        self.assertEqual(ou.reclaimed([a], sea, LAT, monaco), [])

    def test_piece_in_own_country_is_kept(self):
        a, sea = self._bay()
        sea = sea.difference(box(0.04, -0.02, 0.10, 0.00))
        far = box(0.3, 0.3, 0.4, 0.4)
        got = ou.reclaimed([a], sea, LAT, far)
        self.assertEqual([j for j, _ in got], [0])

    def test_straddling_piece_keeps_own_part(self):
        a, sea = self._bay()
        sea = sea.difference(box(0.02, -0.02, 0.10, 0.00))     # strip x 0.02..0.10
        got = ou.reclaimed([a], sea, LAT, box(0.07, -0.1, 0.2, 0.0))   # abroad east of x=0.07
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][1].bounds[2], 0.07, places=3)
        # a piece mostly abroad goes whole, though its own part is above the noise floor
        got = ou.reclaimed([a], sea, LAT, box(0.03, -0.1, 0.2, 0.0))
        self.assertEqual(got, [])
        # and its own part under the noise floor goes too
        got = ou.reclaimed([a], sea, LAT, box(0.0203, -0.1, 0.2, 0.0))
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()
