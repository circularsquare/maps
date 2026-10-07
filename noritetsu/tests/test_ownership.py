"""What ownership.py's small pieces must keep doing: the fixed rule's order, the foot.json
encoding, and how pieces along a section become footprint entries.

    python -m unittest discover -s tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import ownership


class TestFixedRule(unittest.TestCase):
    def test_lowest_ref_by_natural_sort_empty_last(self):
        lines = [{"id": "m3", "ref": "10", "name": "b"}, {"id": "m1", "ref": "2", "name": "z"},
                 {"id": "m2", "ref": "", "name": "a"}, {"id": "m4", "ref": "A3", "name": "c"},
                 {"id": "m0", "ref": "2", "name": "y"}]
        got = [l["id"] for l in sorted(lines, key=ownership.owner_key)]
        self.assertEqual(got, ["m0", "m1", "m3", "m4", "m2"])


class TestFootJson(unittest.TestCase):
    def test_round_trip_and_short_forms(self):
        foot = {7: [[7, 0.0, 0.25, 0.0, 0.25], [3, 0.5, 0.75, 0.25, 1.0]],
                9: [[4, 1.0, 0.0, 0.0, 1.0]],
                11: [[5, 0.1, 0.2, 0.3, 0.6]],
                12: []}
        with tempfile.TemporaryDirectory() as d:
            ownership.write(Path(d), "xx", foot)
            raw = (Path(d) / "foot.json").read_text(encoding="utf-8")
            back = ownership.read(Path(d) / "foot.json")
        self.assertIn('"9":[[4,10000,0]]', raw)            # whole section: three numbers
        self.assertIn('"7":[[7,0,2500,2500],[3,5000,7500]]', raw)
        self.assertEqual(back[12], [])
        for g, v in foot.items():
            for x, y in zip(v, back[g]):
                self.assertEqual(x[0], y[0])
                np.testing.assert_allclose(x[1:], y[1:], atol=1e-4)


class TestRuns(unittest.TestCase):
    def test_consecutive_pieces_on_one_section_are_one_entry(self):
        km = np.array([1.0, 2.0, 1.0, 1.0])            # section 3's km is 1 (index -1: none)
        a0 = np.array([0.0, 0.25, 0.5, 0.75])
        a1 = np.array([0.25, 0.5, 0.75, 1.0])
        ps = np.array([1, 1, -1, 2])
        f0 = np.array([0.0, 0.5, 0.0, 1.0])
        f1 = np.array([0.5, 1.0, 0.0, 0.5])
        got = ownership.runs(a0, a1, ps, f0, f1, km)
        self.assertEqual(got, [[1, 0.0, 1.0, 0.0, 0.5], [2, 1.0, 0.5, 0.75, 1.0]])

    def test_a_jump_along_the_owner_breaks_the_entry(self):
        km = np.array([0.0, 10.0])
        got = ownership.runs(np.array([0.0, 0.5]), np.array([0.5, 1.0]), np.array([1, 1]),
                             np.array([0.0, 0.8]), np.array([0.1, 0.9]), km)
        self.assertEqual(len(got), 2)


import borders  # noqa: E402


@unittest.skipUnless(borders.NAMES.exists() and borders.SHAPES.exists(),
                     "needs religiondots' outlines and Natural Earth")
class TestAbroad(unittest.TestCase):
    """Only another country's land (or water nearer it) is abroad, not a coast the simplified
    outline cut off or the water a tunnel crosses (PATH under the Hudson, 2026-10-05)."""

    def mask(self, region, pts):
        lon = np.array([p[0] for p in pts])
        lat = np.array([p[1] for p in pts])
        return ownership.abroad_mask(region, ownership.merc(lon, lat)).tolist()

    def test_home_land_and_water_stay(self):
        got = self.mask("us", [(-74.0020, 40.7338),     # PATH under Greenwich Village
                               (-74.0200, 40.7320),     # under the Hudson
                               (-73.9700, 40.7400)])    # under the East River
        self.assertEqual(got, [False, False, False])

    def test_neighbour_land_is_abroad(self):
        got = self.mask("us", [(-79.3832, 43.6532),     # Toronto
                               (-106.4245, 31.6904)])   # Ciudad Juárez
        self.assertEqual(got, [True, True])

    def test_land_a_register_outline_cuts_out(self):
        self.assertEqual(self.mask("ru", [(34.10, 44.95)]), [False])   # Simferopol, built with ru
        if (ownership.ROOT / "data" / "raw" / "ua" / "outline.geojson").exists():
            self.assertEqual(self.mask("ua", [(34.10, 44.95)]), [True])


if __name__ == "__main__":
    unittest.main()
