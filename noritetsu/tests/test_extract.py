"""What extract.py must keep doing, checked against tests/fixture.osm.

The fixture deliberately contains things that must NOT come through: abandoned track, a
station platform way, a one-node way, a bus route, and an untagged highway crossing node.
A tag filter is easy to widen by accident, so these are asserted as absences.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import extract

FIXTURE = str(Path(__file__).resolve().parent / "fixture.osm")


def quiet(_msg):
    pass


class TestExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ways, cls.rels, cls.needed = extract.pass_ways_and_relations(FIXTURE, quiet)
        cls.stops = extract.pass_stops(FIXTURE, quiet)
        ids = np.unique(np.concatenate(
            cls.needed + [np.fromiter(cls.stops.keys(), dtype=np.int64)]))
        cls.ids = ids
        cls.nid, cls.nx, cls.ny = extract.pass_coords(FIXTURE, ids, quiet)

    def test_only_ridable_track_ways(self):
        self.assertEqual(set(self.ways), {100})          # not 101 abandoned, 103 platform
        tags, nodes = self.ways[100]
        self.assertEqual(tags["railway"], "rail")
        self.assertEqual(tags["name"], "山手線")
        self.assertEqual(tags["electrified"], "contact_line")
        self.assertEqual(list(nodes), [1, 2, 3])

    def test_one_node_way_dropped(self):
        self.assertNotIn(102, self.ways)

    def test_way_tags_are_pruned(self):
        tags, _ = self.ways[100]
        self.assertLessEqual(set(tags), extract.WAY_TAGS)

    def test_rail_routes_and_master_kept_bus_dropped(self):
        self.assertEqual(set(self.rels), {200, 201})
        tags, members = self.rels[200]
        self.assertEqual(tags["colour"], "#9ACD32")
        self.assertEqual(tags["ref"], "JY")
        self.assertEqual(members, [("n", 10, "stop"), ("n", 11, "stop"), ("w", 100, "")])
        master_tags, master_members = self.rels[201]
        self.assertEqual(master_tags["type"], "route_master")
        self.assertEqual(master_members, [("r", 200, "")])

    def test_stops_from_either_tagging_scheme(self):
        self.assertEqual(set(self.stops), {10, 11})       # not 12, a highway crossing
        tags, lon, lat = self.stops[10]
        self.assertEqual(tags["name:en"], "Tokyo")
        self.assertAlmostEqual(lon, 139.7671, places=4)
        self.assertAlmostEqual(lat, 35.6812, places=4)

    def test_coords_cover_way_nodes_and_relation_stops(self):
        self.assertEqual(list(self.ids), [1, 2, 3, 10, 11])
        self.assertEqual(list(self.nid), [1, 2, 3, 10, 11])

    def test_coords_are_fixed_point_and_sorted(self):
        self.assertTrue(np.all(np.diff(self.nid) > 0))
        self.assertEqual(self.nx.dtype, np.int32)
        # 1e-7 degree fixed point: node 1 is at 139.7671, 35.6812
        self.assertEqual(self.nx[0], 1397671000)
        self.assertEqual(self.ny[0], 356812000)


if __name__ == "__main__":
    unittest.main()
