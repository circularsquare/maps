"""T-3287: dead units between built land are picked (a line between towns, a town touching built land, a hole), in
rounds; an island or a far unit is not; a corridor cuts the units it crosses."""
import unittest

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import box

import gap_fill as gf

FWD = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)


def moll(g):
    return shapely.transform(g, lambda a: np.column_stack(FWD.transform(a[:, 0], a[:, 1])))


def town(name, lon, lat, pop):
    x, y = FWD.transform(lon, lat)
    return (name, lon, lat, x, y, pop)


class Pick(unittest.TestCase):
    def test_between_two_towns(self):
        # built A (lon 0-1), dead B (1-2), built C (2-3), towns of 60,000 in A and C about 170 km apart
        a, b, c = moll(box(0, 50, 1, 51)), moll(box(1, 50, 2, 51)), moll(box(2, 50, 3, 51))
        picks = gf.pick([("B", b)], [a, c], [town("Ta", 0.5, 50.5, 60e3), town("Tc", 2.5, 50.5, 60e3)])
        self.assertIn("B", picks)
        self.assertIn("between Ta and Tc", picks["B"][0])

    def test_far_apart_or_small_towns_not_between(self):
        a, b, c = moll(box(0, 50, 1, 51)), moll(box(1, 50, 5, 51)), moll(box(5, 50, 6, 51))
        land = moll(box(-1, 49, 7, 52).difference(box(0, 50, 6, 51)))  # unbuilt land all round, so no hole either
        # 400 km apart: over PAIR_KM
        picks = gf.pick([("B", b)], [a, c], [town("Ta", 0.5, 50.5, 60e3), town("Tc", 5.5, 50.5, 60e3)], [land])
        self.assertNotIn("B", picks)
        a, b, c = moll(box(0, 50, 1, 51)), moll(box(1, 50, 2, 51)), moll(box(2, 50, 3, 51))
        land = moll(box(-1, 49, 4, 52).difference(box(0, 50, 3, 51)))
        picks = gf.pick([("B", b)], [a, c], [town("Ta", 0.5, 50.5, 40e3), town("Tc", 2.5, 50.5, 40e3)], [land])
        self.assertNotIn("B", picks)

    def test_strip_between_coasts_is_a_hole(self):
        # land on two sides only, both built (an isthmus between two zones): a hole
        a, b, c = moll(box(0, 50, 1, 51)), moll(box(1, 50, 2, 51)), moll(box(2, 50, 3, 51))
        self.assertIn("a hole", gf.pick([("B", b)], [a, c], [])["B"][0])

    def test_hole(self):
        # B ringed by built land on every side
        ring = moll(box(0, 50, 3, 53).difference(box(1, 51, 2, 52)))
        b = moll(box(1, 51, 2, 52))
        picks = gf.pick([("B", b)], [ring], [])
        self.assertIn("a hole", picks["B"][0])

    def test_island_is_no_hole(self):
        # B is mostly coast, with a short land border to built land: land under a fifth of its border
        a, b = moll(box(0, 50, 1, 50.1)), moll(box(1, 50, 2, 51))
        picks = gf.pick([("B", b)], [a], [])
        self.assertNotIn("B", picks)

    def test_town_needs_built_neighbour(self):
        a, b = moll(box(0, 50, 1, 51)), moll(box(1, 50, 2, 51))
        far = moll(box(5, 50, 6, 51))
        self.assertIn("B", gf.pick([("B", b)], [a], [town("Tb", 1.5, 50.5, 80e3)]))
        self.assertNotIn("B", gf.pick([("B", b)], [far], [town("Tb", 1.5, 50.5, 80e3)]))

    def test_rounds(self):
        # A built, B dead with a town (round 1), C dead holding a town beside B only (round 2)
        a, b, c = moll(box(0, 50, 1, 51)), moll(box(1, 50, 2, 51)), moll(box(2, 50, 3, 51))
        picks = gf.pick([("B", b), ("C", c)], [a], [town("Tb", 1.5, 50.5, 80e3), town("Tc", 2.5, 50.5, 80e3)])
        self.assertTrue(picks["B"][0].startswith("round 1"))
        self.assertTrue(picks["C"][0].startswith("round 2"))


class Corridor(unittest.TestCase):
    def test_split(self):
        strip = gf.corridor_strip([[(0.0, 50.5), (3.0, 50.5)]], 20.0, FWD)
        units = [("big", "Big", moll(box(0, 49, 3, 53))), ("small", "Small", moll(box(1, 50.45, 1.1, 50.55))),
                 ("other", "Other", moll(box(5, 49, 6, 50)))]
        out, corridor = gf.corridor_split(units, {"big", "small"}, strip)
        ids = {u[0] for u in out}
        self.assertEqual(ids, {"big~r", "big", "small", "other"})
        self.assertEqual(corridor, {"big~r", "small"})
        piece = next(g for i, _, g in out if i == "big~r")
        self.assertLess(piece.area, strip.area * 1.01)
        rest = next(g for i, _, g in out if i == "big")
        self.assertAlmostEqual((piece.area + rest.area) / units[0][2].area, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
