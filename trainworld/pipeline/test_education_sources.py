"""Source regressions for T-150; no network access required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from shapely.geometry import box

import education_sources as e


class EducationSources(unittest.TestCase):
    def test_source_filters_and_colocated_schools(self):
        def row(key, **changes):
            return {"attributes": dict(NCESSCH=key, SCH_NAME=key, SURVYEAR="2023-2024",
                       STATUS="1", VIRTUAL="Not Virtual", MEMBER=100, LATCOD=5,
                       LONCOD=5, **changes)}
        rows = [row("a"), row("b")]
        for key, field, value in [("closed", "STATUS", "2"),
                                  ("virtual", "VIRTUAL", "Full Virtual"),
                                  ("missing", "MEMBER", -1), ("outside", "LONCOD", 20)]:
            r = row(key)
            r["attributes"][field] = value
            rows.append(r)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "page.json"
            p.write_text(json.dumps({"features": rows}))
            with patch.object(e.b, "fetch", return_value=str(p)):
                schools, excluded = e.public_schools(box(0, 0, 10, 10))
            self.assertEqual([r["building"] for r in schools], ["NCES:a", "NCES:b"])
            self.assertEqual(sum(r["enrollment"] for r in schools), 200)
            self.assertEqual(len(excluded), 4)
            p.write_text(json.dumps({"features": [row("a"), row("a")]}))
            with patch.object(e.b, "fetch", return_value=str(p)), self.assertRaisesRegex(ValueError, "duplicate"):
                e.public_schools(box(0, 0, 10, 10))

    def test_full_county_fips_and_connecticut_fallback(self):
        # Same three-digit county in NY and NJ must get different fallbacks.
        blocks = pd.DataFrame({"geoid": ["360010000010001", "340010000010001", "090010000010001"], "pop": [100, 100, 100]})
        rows = []
        for key, share in [("36001000002", 0.2), ("34001000002", 0.4), ("09-000002", 0.6)]:
            r = {"key": key, "B01001_E001": 100}
            for i in [3, 4, 5, 6, 27, 28, 29, 30]:
                r[f"B01001_E{i:03d}"] = 0
            r["B01001_E004"] = share * 100
            rows.append(r)
        with patch.object(e.wfh, "extract", return_value=pd.DataFrame(rows)):
            weights, fallback = e.child_weights(blocks)
        np.testing.assert_allclose(weights, [20, 40, 40])
        self.assertEqual(fallback, 3)

    def test_repacking_is_idempotent_and_preserves_live_arrays(self):
        h = {"arrays": [{"name": "h3", "dtype": "u64", "per": "cell", "count": 2, "offset": 0},
                         {"name": "school_children", "dtype": "f32", "per": "cell", "count": 2, "offset": 24}]}
        cells = np.array([42, 43], dtype="<u8").tobytes()
        b = cells + b"oldbytes" + np.array([1, 2], dtype="<f4").tobytes() + b"orphaned"
        values = {"school_children": np.array([3, 4], dtype="<f4")}
        first = e.replace_cell_arrays(h, b, values)
        saved = copy.deepcopy(h)
        second = e.replace_cell_arrays(h, first, values)
        self.assertEqual(first, second)
        self.assertEqual(saved, h)
        self.assertEqual(first[:16], cells)
        self.assertEqual(len(first), 24)


if __name__ == "__main__":
    unittest.main()
