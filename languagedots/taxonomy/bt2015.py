"""Bhutan: GNH 2015 survey, Table A1.5, mother tongue by dzongkhag (sources/bt_gnh.py, sources/bt.md).

Labels as the table prints them (column heads shortened in the build). Classification checked
against Glottolog: Dzongkha (dzon1239), Chocangacakha (choc1275), Layakha (laya1253), Lakha
(lakh1240) and Brokpake (brok1248) sit in its South Tibetic; Bumthangkha (bumt1240), Khengkha
(khen1241), Kurtokha (kurt1248), Chalikha (chal1267), Nyenkha (Upper Mangdep, nyen1254),
Dzalakha (dzal1238) and Dakpakha (dakp1242) are the conventional East Bodish; Olekha (olek1239),
Gongduk (gong1251) and Lhokpu (lhok1238) are branches of their own directly under Sino-Tibetan;
Tshangla (tsha1245) is cn.txt's node.

The table's "Monpakha" is drawn as Olekha, the Black Mountain Monpa language: its answers sit in
Trongsa, Dagana and Wangdue Phodrang, around the Black Mountains, not in the east where India's
Tawang Monpa (Dakpa) lives. "Others" (4.28% nationally, 24% of Tsirang, 20% of Dagana) is an
unnamed remainder, very likely mostly Tamang, Gurung, Rai and Limbu, but nothing says so: `other`.

Non-Bhutanese rows are node ids already (origin mixes), passed through as EXTRA_NODES.
"""
from pathlib import Path

import pandas as pd

ST = "sinotibetan"
NAMES = {
    "Dzongkha": f"{ST}.tibetic.dzongkha",
    "Cho-cha nga-chakha": f"{ST}.tibetic.chocangacakha",
    "Tshangla": f"{ST}.tshangla",
    "Bumthangkha": f"{ST}.eastbodish.bumthang",
    "Khengkha": f"{ST}.eastbodish.kheng",
    "Kurtop": f"{ST}.eastbodish.kurtop",
    "Nyenkha": f"{ST}.eastbodish.nyenkha",
    "Dzala": f"{ST}.eastbodish.dzala",
    "Dakpa": f"{ST}.eastbodish.dakpa",
    "Chali kha": f"{ST}.eastbodish.chali",
    "Monpakha": f"{ST}.olekha",
    "Brokpa": f"{ST}.tibetic.brokpa",
    "Lakha": f"{ST}.tibetic.lakha",
    "Bokha": f"{ST}.tibetic.tibetan",
    "Nepali": "indoeuropean.indoaryan.pahari.eastern.nepali",
    "Lhokpu": f"{ST}.lhokpu",
    "Gongduk": f"{ST}.gongduk",
    "Lepcha": f"{ST}.lepcha",
    "Layap": f"{ST}.tibetic.layakha",
    "English": "indoeuropean.germanic.english",
    "Others": "other",
}

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "bt.csv"
EXTRA_NODES = sorted(set(pd.read_csv(_CSV, usecols=["source_category"])["source_category"])
                     - set(NAMES)) if _CSV.exists() else []


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    if label in EXTRA_NODES:
        return label
    raise KeyError(f"bt2015: unmapped label {label!r}")
