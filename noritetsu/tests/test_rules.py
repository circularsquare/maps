"""Each country's rules (rules/<cc>.py) as build_model.country_rules reads them.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_model as bm

FLAGS = ("SERVICE_IF_ALL_ROUTES_ARE", "METRO_DUP", "ROUTE_SHARE_BY_LENGTH")


def countries():
    return sorted(p.stem for p in (ROOT / "rules").glob("*.py")
                  if p.stem not in ("__init__", "shared"))


class TestCountryRules(unittest.TestCase):
    def test_every_file_loads_with_hooks_of_the_right_type(self):
        for cc in countries():
            with self.subTest(cc=cc):
                r = bm.country_rules(cc)
                self.assertIsNotNone(r)
                if hasattr(r, "looks_like_service"):
                    self.assertIsInstance(
                        r.looks_like_service({"name": "x"}, "x", ""), bool)
                for f in FLAGS:
                    if hasattr(r, f):
                        self.assertIsInstance(getattr(r, f), bool, f)
                if hasattr(r, "PLATFORM_SUFFIX"):
                    self.assertTrue(hasattr(r.PLATFORM_SUFFIX, "sub"))

    def test_no_file_means_defaults(self):
        self.assertIsNone(bm.country_rules("zz"))
        self.assertIsNone(bm.country_rules(None))
        self.assertFalse(bm.looks_like_service({"name": "EC 112"}, "train", "zz"))
        self.assertEqual(bm.plain_name({"name": "Box Hill 3"}, "zz"), "Box Hill 3")

    def test_named_trains(self):
        svc = lambda cc, name, **t: bm.looks_like_service({"name": name, **t}, "train", cc)
        self.assertTrue(svc("jp", "のぞみ"))
        self.assertFalse(svc("jp", "東海道本線"))
        self.assertFalse(svc("jp", "上野東京ライン"))
        self.assertTrue(svc("at", "EC 112"))           # rules.shared.EU_TRAIN
        self.assertFalse(svc("at", "ICE 43"))          # a DB interval line
        self.assertFalse(svc("de", "ICE 10"))
        self.assertTrue(svc("de", "ICE 1001"))
        self.assertTrue(svc("se", "Juna PYO 276"))     # rules.shared.FI_TRAIN
        self.assertTrue(svc("ca", "Maple Leaf"))       # rules.shared.US_TRAIN
        self.assertFalse(svc("ca", "Corridor", network="VIA Rail"))
        self.assertTrue(svc("us", "Coast Starlight"))
        self.assertFalse(svc("us", "Northeast Regional"))
        # never for anything but route=train
        self.assertFalse(bm.looks_like_service({"name": "EC 112"}, "subway", "at"))

    def test_service_if_all_routes_are(self):
        routes = {1: ({"name": "Juna PYO 273"}, []), 2: ({"name": "Juna PYO 276"}, [])}
        self.assertTrue(bm.is_service({"name": "Juna 7"}, "train", [1, 2], routes, "fi"))
        routes = {1: ({"name": "のぞみ"}, []), 2: ({"name": "ひかり"}, [])}
        self.assertFalse(bm.is_service({"name": "東海道新幹線"}, "train", [1, 2], routes, "jp"))

    def test_platform_names(self):
        self.assertEqual(bm.plain_name({"name": "Box Hill 3"}, "au"), "Box Hill")
        self.assertEqual(bm.plain_name({"name": "Central, Platform 23"}, "au"), "Central")
        self.assertEqual(bm.plain_name({"name": "Stop 35: X", "railway": "tram_stop"}, "au"),
                         "Stop 35: X")


if __name__ == "__main__":
    unittest.main()
