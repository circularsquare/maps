"""Kiribati: everyone drawn as Gilbertese (Kiribati language) per island, on the 2020 census
island populations -> data/normalized/ki.csv.

    python sources/ki_pop.py

NO HOME-LANGUAGE QUESTION: the 2020 census asks only "Does X speak English at home?" (yes/no,
questionnaire module D9, religiondots' data/raw/ki/census_report_2020.pdf p140) and literacy. No
language table is published. Haiti-style build (sources/ht.md), rows `derived`.

THE POPULATION: the 2020 census island profiles (religiondots' data/raw/ki/island_profile_2020.xlsx,
read only), one sheet per island; row 3 ("Population (Census)") carries 2015 and 2020 for the
island, the outer islands, South Tarawa with Betio and All Kiribati.

CHECKS: 24 islands, each mapped to a religiondots unit (both ways); the islands sum to 119,438,
the 2020 census total (census report Table G-1); every sheet's All Kiribati figure agrees.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

XLSX = RD / "data" / "raw" / "ki" / "island_profile_2020.xlsx"
OUT = HERE / "data" / "normalized" / "ki.csv"
TOTAL_2020 = 119_438
# island-profile sheet name -> religiondots unit
SHEETS = {
    "Banaba": "banaba", "Makin": "makin", "Butaritari": "butaritari", "Marakei": "marakei",
    "Abaiang": "abaiang", "North Tarawa": "ntarawa", "Betio": "betio",
    "South Tarawa": "starawa", "Maiana": "maiana", "Kuria": "kuria", "Aranuka": "aranuka",
    "Abemama": "abemama", "Nonouti": "nonouti", "NTabiteuea": "ntabiteuea",
    "STabiteuea": "stabiteuea", "Beru ": "beru", "Nikunau": "nikunau", "Onotoa": "onotoa",
    "Tamana": "tamana", "Arorae": "arorae", "Kiritimati": "kiritimati", "Teraina": "teeraina",
    "Tabuaeran ": "tabuaeran", "Kanton": "kanton",
}


def main():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    lk = pd.read_csv(RD / "data" / "geo" / "ki" / "ki_lookup.csv")
    assert set(SHEETS.values()) == set(lk["unit"]), set(SHEETS.values()) ^ set(lk["unit"])
    assert set(SHEETS) | {"Check"} == set(wb.sheetnames), set(wb.sheetnames) ^ set(SHEETS)
    name = dict(zip(lk["unit"], lk["island"]))
    out = []
    for sheet, unit in SHEETS.items():
        rows = list(wb[sheet].iter_rows(min_row=1, max_row=3, values_only=True))
        years = [c for c in rows[1] if c is not None]
        vals = [c for c in rows[2] if c is not None]
        assert vals[0] == "Population (Census)" and years[:2] == [2015, 2020], (sheet, rows)
        assert vals[-1] == TOTAL_2020, (sheet, vals)
        out.append(dict(geo_id=unit, geo_level="island", geo_name=name[unit],
                        source_category="Gilbertese", count=int(vals[2]), tier="derived",
                        year=2020))
    df = pd.DataFrame(out)
    assert df["count"].sum() == TOTAL_2020, df["count"].sum()
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} islands, {TOTAL_2020:,} people")


if __name__ == "__main__":
    main()
