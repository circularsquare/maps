"""build_model.fold_spurs and transit_tail on made-up lines.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_model as bm


def st(sid, lon, lat, junction=False, lines=()):
    return {"id": sid, "name": sid, "lon": lon, "lat": lat, "lines": set(lines),
            "junction": junction}


# Along latitude 51, 0.001 degrees of longitude is about 70 m.
FORK, TIP = (0.010, 51.0), (0.010, 51.006)          # the spur runs ~670 m north of the fork


def spur_line(v=False):
    """A - fork - up the spur to J, and J - down the spur - fork - B. With v, A and B lie
    south of the fork on both sides, so the legs leave it in a V: a switchback."""
    up = [[0.010, 51.0 + 0.0005 * i] for i in range(13)]
    south = 50.99 if v else 51.0
    a_to_j = [[0.0, south], [0.005, (south + 51.0) / 2]] + up
    j_to_b = [list(p) for p in reversed(up)] + [[0.015, (south + 51.0) / 2], [0.020, south]]
    line = {"id": "L", "name": "L", "sections": [["A", "J", 1.4], ["J", "B", 1.4]],
            "display": ["A", "J", "B"], "chain": {"A|J": 1.0, "J|B": 1.0},
            "borrowed": ["J|B"]}
    stations = {"A": st("A", 0.0, 51.0, lines="L"), "B": st("B", 0.020, 51.0, lines="L"),
                "C": st("C", 0.010, 51.02, lines="M"),
                "J": st("J", *TIP, junction=True, lines=("L", "M"))}
    geoms = {"L": {"A|J": bm.Pts(a_to_j, list(range(len(a_to_j)))),
                   "J|B": bm.Pts(j_to_b, list(range(100, 100 + len(j_to_b))))},
             # line M goes on north from J: J is a real junction off the spur
             "M": {"J|C": bm.Pts([list(TIP), [0.010, 51.01], [0.010, 51.02]])}}
    other = {"id": "M", "sections": [["J", "C", 1.6]], "display": ["J", "C"]}
    return line, stations, geoms, other


class TestFoldSpurs(unittest.TestCase):
    def test_spur_is_cut_out(self):
        line, stations, geoms, other = spur_line()
        bm.fold_spurs([line, other], stations, geoms, lambda m: None)
        self.assertEqual([s[:2] for s in line["sections"]], [["A", "B"]])
        self.assertEqual(line["display"], ["A", "B"])
        g = geoms["L"]["A|B"]
        self.assertTrue(all(abs(p[1] - 51.0) < 1e-9 for p in g), "no point up the spur")
        self.assertAlmostEqual(line["km"], 1.4, delta=0.05)   # 0.020 deg of longitude
        self.assertEqual(line["chain"], {"A|B": 2.0})
        self.assertEqual(line["borrowed"], ["A|B"])
        self.assertNotIn("L", stations["J"]["lines"])
        self.assertEqual(len(g.ids), len(g))

    def test_no_other_line_needed(self):
        # gb's junctions are usually a node of only the line that ends there
        line, stations, geoms, other = spur_line()
        stations["J"]["lines"] = {"L"}
        bm.fold_spurs([line], stations, geoms, lambda m: None)
        self.assertEqual(len(line["sections"]), 1)

    def test_detour_beside_the_direct_section_goes(self):
        # Peterborough to Lincoln: A - B directly, and A - J - B up a spur and back
        line, stations, geoms, other = spur_line()
        direct = ["A", "B", 1.41]
        line["sections"].append(direct)
        geoms["L"]["A|B"] = bm.Pts([[0.0, 51.0], [0.010, 51.0], [0.020, 51.0]])
        bm.fold_spurs([line, other], stations, geoms, lambda m: None)
        self.assertEqual(line["sections"], [direct])
        self.assertEqual(set(geoms["L"]), {"A|B"})
        self.assertEqual(line["chain"], {})

    def test_switchback_is_left(self):
        # Chodov-úvrať: the legs leave the fork together, trains reverse up the spur
        line, stations, geoms, other = spur_line(v=True)
        bm.fold_spurs([line, other], stations, geoms, lambda m: None)
        self.assertEqual(len(line["sections"]), 2)

    def test_a_stop_is_never_folded(self):
        line, stations, geoms, other = spur_line()
        stations["J"]["junction"] = False
        bm.fold_spurs([line, other], stations, geoms, lambda m: None)
        self.assertEqual(len(line["sections"]), 2)


class TestSplitFarPieces(unittest.TestCase):
    def test_far_piece_is_its_own_line_near_one_stays(self):
        stations = {s: st(s, lon, 51.0, lines="R") for s, lon in
                    (("A", 0.0), ("B", 0.1), ("C", 0.13), ("D", 0.2),     # C - D 2 km from B
                     ("E", 1.0), ("F", 1.1))}                              # E - F ~60 km away
        line = {"id": "rL", "name": "Ligne", "name_en": "", "src": "rinf", "km": 21.0,
                "km_official": 21.0, "display": ["A", "B", "C", "D", "E", "F"],
                "sections": [["A", "B", 7.0], ["C", "D", 7.0], ["E", "F", 7.0]]}
        geoms = {"rL": {"A|B": [[0, 51], [0.1, 51]], "C|D": [[0.13, 51], [0.2, 51]],
                        "E|F": [[1.0, 51], [1.1, 51]]}}
        reg_ways = {1: {"rL"}}
        got = bm.split_far_pieces([line], stations, geoms, reg_ways, {}, lambda m: None)
        self.assertEqual(list(got), ["rL"])
        new_id = got["rL"][0]
        self.assertEqual([s[:2] for s in line["sections"]], [["A", "B"], ["C", "D"]])
        self.assertEqual(set(geoms[new_id]), {"E|F"})
        self.assertNotIn("E|F", geoms["rL"])
        self.assertIn(new_id, stations["E"]["lines"])
        self.assertNotIn("rL", stations["E"]["lines"])
        self.assertAlmostEqual(line["km_official"], 14.0)
        self.assertTrue(new_id.startswith("r"))

    def test_keep_whole(self):
        stations = {s: st(s, lon, 51.0, lines="R") for s, lon in
                    (("A", 0.0), ("B", 0.1), ("E", 1.0), ("F", 1.1))}
        line = {"id": "rL", "name": "Keep", "src": "rinf", "km": 14.0, "display": [],
                "sections": [["A", "B", 7.0], ["E", "F", 7.0]]}
        got = bm.split_far_pieces([line], stations, {"rL": {}}, {}, {}, lambda m: None,
                                  keep_whole={"Keep"})
        self.assertEqual(got, {})


class FakeIndex:
    def __init__(self, hits):
        self.hits = hits

    def along(self, xy):
        return self.hits


class TestTransit(unittest.TestCase):
    def test_border_to_border(self):
        xy = np.array([[0.0, 51.0], [0.01, 51.0], [0.02, 51.0], [0.03, 51.0]])
        ids = np.array([1, 2, 3, 4])
        p0 = {"id": "e1", "countries": {"fr", "gb"}}
        p1 = {"id": "e2", "countries": {"fr", "be"}}
        hits = [(500.0, p0, 0, 0.005, 51.0, 0.0), (1900.0, p1, 2, 0.025, 51.0, 0.0)]
        members = [("n", 99, "stop")]                 # a stop this extract does not have
        t = bm.transit_tail([(ids, xy)], members, {}, FakeIndex(hits), "fr")
        self.assertEqual((t["bp0"]["id"], t["bp"]["id"]), ("e1", "e2"))
        self.assertEqual(len(t["geom"]), len(t["ids"]))
        self.assertAlmostEqual(t["km"], 1.4, delta=0.05)
        # Not this country's border points: nothing.
        self.assertIsNone(bm.transit_tail([(ids, xy)], members, {}, FakeIndex(hits), "de"))
        # Every stop resolved here (not a train running on abroad): nothing.
        self.assertIsNone(bm.transit_tail([(ids, xy)], members, {99: "x"}, FakeIndex(hits), "fr"))


if __name__ == "__main__":
    unittest.main()
