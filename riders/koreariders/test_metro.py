"""Small regression checks for the metro import's unit and direction boundaries."""
import unittest
from build_metro import build, distribute_express, reverse


def fixture():
    return {"day": "weekday", "build": {"sample": 1}, "hours": [7, 8],
            "line_meta": {"2": {"display": "2", "display_en": "Line 2", "color": "#00a84d"}},
            "segments": [dict(line="2", ca="0201", cb="0202", a="A", b="B", ae="A", be="B",
                              p=[37, 127, 37.01, 127.01], h=[100, 200]),
                         dict(line="2", ca="0202", cb="0201", a="B", b="A", ae="B", be="A",
                              p=[37.01, 127.01, 37, 127], h=[20, 40])]}


class MetroTest(unittest.TestCase):
    def test_reverse_links_and_daily_units(self):
        result = build(fixture(), {})
        self.assertEqual(len(result["features"]), 1)
        p = result["features"][0]["properties"]
        self.assertEqual((p["daily"], p["daily_down"], p["daily_up"]), (360, 300, 60))
        self.assertNotIn("down", p)

    def test_ring_closure(self):
        data = fixture()
        data["segments"] = [data["segments"][0]]
        data["segments"][0].update(ca="0243", cb="0201")
        p = build(data, {})["features"][0]["properties"]
        self.assertEqual(p["daily_down"], 300)
        self.assertEqual(p["from"], "A")

    def test_reject_incompatible_sources(self):
        for change in (lambda d: d.update(day="nye"),
                       lambda d: d["build"].update(sample=10),
                       lambda d: d["segments"][0].update(hx=[1000, 0]),
                       lambda d: d["segments"][0].update(h=[-1, 4])):
            data = fixture()
            change(data)
            with self.assertRaises(ValueError):
                build(data, {})

    def test_express_across_local_stops_without_double_counting(self):
        data = fixture()
        first = dict(data["segments"][0], n=[2, 2], nx=[1, 1], hx=[10, 20])
        second = dict(first, ca="0202", cb="0203", a="B", b="C", ae="B", be="C",
                      p=[37.01, 127.01, 37.02, 127.02])
        express = dict(first, cb="0203", b="C", be="C", p=[37, 127, 37.02, 127.02],
                       n=[1, 1], nx=[1, 1], h=[50, 70], hx=[50, 70])
        data["segments"] = [first, second, express, reverse(express)]
        result = build(data, {})
        self.assertEqual(len(result["features"]), 2)
        for f in result["features"]:
            p = f["properties"]
            self.assertEqual(p["daily_down"], 420)
            self.assertEqual(p["daily_up"], 120)
            self.assertEqual(p["daily"], 540)

    def test_express_does_not_cross_lines(self):
        data = fixture()
        express = dict(data["segments"][0], line="9", h=[10, 20], hx=[10, 20], n=[1, 1], nx=[1, 1])
        data["segments"].append(express)
        with self.assertRaisesRegex(ValueError, "No local path"):
            distribute_express(data)

    def test_reverse_shape_reused(self):
        result = build(fixture(), {"2|37.01000,127.01000|37.00000,127.00000":
                                  [[37.01, 127.01], [37.005, 127.008], [37, 127]]})
        f = result["features"][0]
        self.assertEqual(f["properties"]["geometry_source"], "track")
        self.assertEqual(f["geometry"]["coordinates"][0], [127, 37])


if __name__ == "__main__":
    unittest.main()
