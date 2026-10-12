import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from shapely.geometry import box

import enrich_universities as u


class UniversitySources(unittest.TestCase):
    def test_nonoverlapping_levels_online_exclusion_and_child_campus(self):
        directory = pd.DataFrame([dict(UNITID=i, INSTNM=str(i), LATITUDE=5.0, LONGITUD=5.0,
            ACT="G" if i == 2 else "A", CYACTIVE=1, NEWID=1 if i == 3 else -2,
            DEGGRANT=1, F1SYSNAM="same system") for i in range(1, 5)])
        enrollment = pd.DataFrame([dict(UNITID=i, EFDELEV=level, EFDETOT=total,
            EFDEEXC=online if i != 4 else -1) for i in range(1, 5)
            for level, total, online in [(1,1000,200),(2,900,150),(3,600,100),(11,300,50),(12,100,20)]])
        records, excluded = u.institutions(directory, enrollment, box(0,0,10,10))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["enrollment"], 580)
        self.assertEqual(records[0]["exclusively_online"], 120)
        self.assertEqual(len(excluded), 3)
        # Equal coordinates/system do not collapse distinct reporting institutions.
        directory.loc[directory.UNITID == 2, "ACT"] = "A"
        records, _ = u.institutions(directory, enrollment, box(0,0,10,10))
        self.assertEqual(sum(r["enrollment"] for r in records), 1160)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            u.institutions(directory, pd.concat([enrollment,enrollment.iloc[:1]]), box(0,0,10,10))

    def test_not_applicable_online_blank_requires_full_in_person_evidence(self):
        directory=pd.DataFrame([dict(UNITID=1,INSTNM="One",LATITUDE=5.0,LONGITUD=5.0,
            ACT="A",CYACTIVE=1,NEWID=-2,DEGGRANT=1,F1SYSNAM="")])
        enrollment=pd.DataFrame([dict(UNITID=1,EFDELEV=3,EFDETOT=100,EFDEEXC=np.nan,
            XEFDEEXC="A",EFDENON=100)])
        rows,_=u.institutions(directory,enrollment,box(0,0,10,10))
        self.assertEqual(rows[0]["enrollment"],100)
        enrollment.loc[0,"EFDENON"]=90
        rows,excluded=u.institutions(directory,enrollment,box(0,0,10,10))
        self.assertFalse(rows)
        self.assertEqual(excluded[0]["reason"],"invalid_enrollment_or_online_count")

    def test_adults_exclude_all_under18_and_use_full_county_fallback(self):
        rows = []
        for key, children in [("36001000002",20),("34001000002",40),("09-000002",60)]:
            row = {"key":key,"B01001_E001":100}
            row.update({f"B01001_E{i:03d}":0 for i in [3,4,5,6,27,28,29,30]})
            row["B01001_E003"] = children
            rows.append(row)
        blocks = pd.DataFrame({"geoid":["360010000010001","340010000010001","090010000010001"],"pop":[100]*3})
        with patch.object(u.wfh,"extract",return_value=pd.DataFrame(rows)):
            weights, fallback = u.adult_weights(blocks)
        np.testing.assert_allclose(weights,[80,60,60])
        self.assertEqual(fallback,3)


if __name__ == "__main__":
    unittest.main()
