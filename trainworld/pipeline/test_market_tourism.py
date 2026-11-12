"""T-3280: the resort towns' table is cited and matches the config file, and a town's attraction is shared over
the sites inside its outline by their jobs (else its nearest site), only in the pack whose outline holds it."""
import csv
import json
import unittest
from pathlib import Path

import numpy as np

import market_zones as M

HERE = Path(__file__).parent


class Table(unittest.TestCase):
    def test_rows_are_cited_and_match_the_config(self):
        with open(HERE / "market_tourism.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        cfg = json.loads((HERE / "market_tourism.json").read_text(encoding="utf-8"))
        self.assertEqual(len({r["code"] for r in rows}), len(rows))
        for r in rows:
            self.assertTrue(r["url"].startswith("http"), r["name"])
            self.assertGreater(int(r["visitors"]), 0, r["name"])
            self.assertNotIn(r["year"][-4:], ("2020", "2021", "2022"), r["name"])
        self.assertEqual([t["code"] for t in cfg["towns"]], [r["code"] for r in rows])
        self.assertEqual([t["visitors"] for t in cfg["towns"]], [int(r["visitors"]) for r in rows])


class Sites(unittest.TestCase):
    def setUp(self):
        self.was = M.TOURISM
        square = [[[[139.0, 35.2], [139.1, 35.2], [139.1, 35.3], [139.0, 35.3], [139.0, 35.2]]]]
        M.TOURISM = {"on": True, "beta": 2.0, "towns": [{"code": "x", "name": "Test", "visitors": 365000, "lon": 139.05, "lat": 35.25, "shape": square}]}

    def tearDown(self):
        M.TOURISM = self.was

    def test_shared_by_jobs_inside_the_outline(self):
        lon = np.array([139.02, 139.08, 139.2, 139.05])
        lat = np.array([35.22, 35.28, 35.25, 35.25])
        jobs = np.array([100.0, 300.0, 500.0, 0.0])
        tw = M.site_tourism("tyo", lon, lat, jobs)  # Hakone's square lies in the Tokyo pack's outline
        self.assertAlmostEqual(tw.sum(), 2.0 * 1000)
        np.testing.assert_allclose(tw, [500.0, 1500.0, 0.0, 0.0])

    def test_nearest_site_when_none_inside_and_none_in_another_pack(self):
        tw = M.site_tourism("tyo", np.array([139.3, 139.6]), np.array([35.3, 35.6]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(tw, [2000.0, 0.0])
        self.assertEqual(M.site_tourism("osa", np.array([139.05]), np.array([35.25]), np.array([1.0])).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
