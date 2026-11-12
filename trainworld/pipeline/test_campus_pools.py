"""T-3273: an institution's count is shared over its campuses in the world, members' counts come off
a system's, untagged campuses join by name, open universities are left out."""
import unittest

import numpy as np
import pandas as pd
from shapely.geometry import box

import campus_pools as cp
import tier0


def sites(rows):
    return pd.DataFrame(rows, columns=["kind", "name", "lat", "lon", "m2", "wikidata"])


COUNTS = {"Q1": (10000.0, ""), "Q2": (3000.0, ""), "Q9": (50000.0, ""), "Q7": (80000.0, ""), "Q8": (60000.0, "")}
PARENTS = {"Q3": {"Q1": 10000.0}, "Q4": {"Q1": 10000.0}, "Q2": {"Q9": 50000.0}, "Q5": {"Q9": 50000.0},
           "Q7": {"Q8": 60000.0}, "Q8": {"Q7": 80000.0}}
LABELS = {"Q1": ["Ming Chuan University", "銘傳大學"], "Q2": ["Kennesaw State University"],
          "Q6": ["National Open University"], "Q9": ["University System of Georgia"]}


class CampusPools(unittest.TestCase):
    def table(self, extra=()):
        w = sites([("university", "Ming Chuan Kinmen", 24.5, 118.4, 5e4, "Q3"),     # Q1's branch, far away
                   ("university", "銘傳大學 桃園校區", 25.0, 121.3, 15e4, ""),       # untagged, joins Q1 by name
                   ("university", "Kennesaw State University", 34.0, -84.6, 8e5, "Q2"),
                   ("university", "Kennesaw State University Marietta Campus", 33.9, -84.5, 7e5, "Q5"),
                   ("university", "Ming Chuan University", 51.0, 0.0, 1e4, ""),          # same name, too far
                   ("college", "National Open University Kinmen", 24.4, 118.3, 1e4, "Q6"),
                   ("university", "A", 10.0, 10.0, 1e4, "Q7"), ("university", "B", 10.0, 11.0, 1e4, "Q8"),
                   *extra])
        counts = dict(COUNTS, Q6=(90000.0, ""))
        return cp.build(w, counts, PARENTS, LABELS)[0], counts

    def test_institution_of(self):
        self.assertEqual(tier0.institution_of("Q1", COUNTS, PARENTS)[:2], ("Q1", 10000.0))
        self.assertEqual(tier0.institution_of("Q3", COUNTS, PARENTS)[2], "parent Q1")
        self.assertEqual(tier0.institution_of("", COUNTS, PARENTS)[0], "")

    def test_pools_names_members_and_open(self):
        t, _ = self.table()
        p = t["pools"]
        self.assertEqual(len(p["Q1"]["sites"]), 2)                 # the branch and the campus found by name
        self.assertEqual(p["Q1"]["count"], 10000.0)
        self.assertEqual(sum(s[2] for s in p["Q2"]["sites"]), 150.0)  # Marietta joins Kennesaw State, not the system
        self.assertNotIn("Q9", p)
        self.assertNotIn("Q6", p)                                   # an open university by its label
        self.assertIn("Q6", t["open"])
        self.assertEqual(p["Q7"]["less"], 0.0)                      # a pair naming each other is left alone

    def test_member_counts_come_off(self):
        t, _ = self.table([("university", "System office", 33.7, -84.4, 1e4, "Q9")])
        self.assertEqual(t["pools"]["Q9"]["less"], 3000.0)
        self.assertEqual(t["pools"]["Q9"]["count"], 47000.0)

    def test_word_boundaries_and_cjk(self):
        n = cp._Names({"kennesaw state university": "Q2", "銘傳大學": "Q1"})
        self.assertEqual(n.held("Kennesaw State University Marietta"), {"Q2"})
        self.assertEqual(n.held("XKennesaw State University"), set())
        self.assertEqual(n.held("銘傳大學金門分部"), {"Q1"})

    def test_generic_label_dropped(self):
        w = sites([("university", "Fachhochschule Nord", 50.0, 8.0, 1e4, "Q11"),
                   ("university", "Fachhochschule Süd", 49.0, 8.0, 1e4, "Q12")])
        names = cp.name_index({"Q10": ["Fachhochschule Ost", "Fachhochschule"]}, {"Q10"}, w, {})
        self.assertNotIn("fachhochschule", names)                    # one word
        self.assertIn("fachhochschule ost", names)
        names = cp.name_index({"Q10": ["Hochschule Nord"]}, {"Q10"}, w, {})
        self.assertIn("hochschule nord", names)                      # "Fachhochschule Nord" holds it at no word boundary
        names = cp.name_index({"Q10": ["Fachhochschule Nord"]}, {"Q10"},
                              sites([("u", "Fachhochschule Nord A", 0, 0, 1, "Q11"), ("u", "Fachhochschule Nord B", 0, 0, 1, "Q12")]), {})
        self.assertNotIn("fachhochschule nord", names)               # held by two other ids' sites

    def test_pack_takes_its_world_share(self):
        t, counts = self.table()
        uni = sites([("university", "Ming Chuan Kinmen", 24.5, 118.4, 5e4, "Q3"),
                     ("college", "National Open University Kinmen", 24.4, 118.3, 1e4, "Q6")])
        rows, out = tier0.university_sizes(uni, counts, PARENTS, t, box(118, 24, 119, 25))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(rows.weight.iloc[0]), 10000.0 * 5 / 20)
        rows, _ = tier0.university_sizes(uni, counts, PARENTS)           # no table: the whole count, as before
        self.assertAlmostEqual(float(rows.weight.iloc[0]), 10000.0)


if __name__ == "__main__":
    unittest.main()
