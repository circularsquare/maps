"""T-2776: market_zones.naming_gone finds the zones whose market packs name a place no registered zone holds.

    python -m unittest test_market_naming_gone      (from pipeline/)
"""
import json
import tempfile
import unittest
from pathlib import Path

import market_zones


class NamingGone(unittest.TestCase):
    def test_finds_packs_naming_a_gone_place(self):
        with tempfile.TemporaryDirectory() as d:
            packs = Path(d)
            files = {
                "aaa": {"place": 1, "commute_place": [2, 3], "everyday_place": [2]},  # 3 is gone
                "bbb": {"place": 2, "commute_place": [1], "everyday_place": [1, 2]},
                "ccc": {"place": 4, "everyday_place": [9]},  # 9 is gone, everyday only
                "ddd": {"place": 5},  # no commutes at all
            }
            for cid, m in files.items():
                (packs / f"{cid}.market.json").write_text(json.dumps(m, separators=(",", ":")), encoding="utf-8")
            # spaced JSON too, in case a file was written another way
            (packs / "eee.market.json").write_text(json.dumps({"place": 6, "commute_place": [1, 7]}), encoding="utf-8")
            reg = {"cities": [{"id": c, "place": p} for c, p in (("aaa", 1), ("bbb", 2), ("ccc", 4), ("ddd", 5), ("eee", 6), ("fff", 8))]}
            old = market_zones.PACKS
            market_zones.PACKS = packs
            try:
                self.assertEqual(market_zones.naming_gone(reg), ["aaa", "ccc", "eee"])
            finally:
                market_zones.PACKS = old


if __name__ == "__main__":
    unittest.main()
