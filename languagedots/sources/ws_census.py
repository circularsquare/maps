"""Samoa: everyone with Samoan citizenship drawn as Samoan, per traditional district, from the
2021 census -> data/normalized/ws.csv.

    python sources/ws_census.py

NO LANGUAGE OR ETHNICITY QUESTION in the 2021 census tables (coverage sweep 2026-10-03: the
49-sheet workbook has citizenship only). Haiti-style build (sources/ht.md): the one language
nearly everyone grows up with, rows `derived`.

THE TABLES (religiondots' copies, read only):
  * village totals: religiondots' normalized ws.csv (2021 religion table, which has no
    not-stated cell, so its categories sum to each village's population), folded onto the 25
    traditional districts by religiondots' ws_lookup.csv, the units its hex layer uses;
  * non-citizens: the census workbook's Table 8a ("NO NOT A CITIZEN OF SAMOA", 1,218 people),
    by census district, folded onto the same 25 units through ws_lookup's census_district.
Non-citizens' nationality is not published, so their language is unknown: they are not drawn
(`gap`). Samoan citizens (born in Samoa, born abroad to Samoan parents, naturalised) are drawn as
Samoan.

CHECKS: 339 villages; the villages sum to Table 8a's national total 205,557; Table 8a's census
districts sum to its national row in both columns; every census district maps to a unit.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

XLSX = RD / "data" / "raw" / "ws" / "CensusTablesEXCELFiles.xlsx"
OUT = HERE / "data" / "normalized" / "ws.csv"
TOTAL, NONCIT = 205_557, 1_218


def main():
    import openpyxl
    lk = pd.read_csv(RD / "data" / "geo" / "ws" / "ws_lookup.csv")
    rd = pd.read_csv(RD / "data" / "normalized" / "ws.csv", keep_default_na=False,
                     na_values=[""])
    vil = rd.groupby("geo_id")["count"].sum()
    assert len(vil) == 339 and vil.sum() == TOTAL, (len(vil), vil.sum())
    assert set(vil.index) == set(lk["geo_id"]), set(vil.index) ^ set(lk["geo_id"])
    pop = vil.groupby(lk.set_index("geo_id")["unit"]).sum()

    d2u = lk.drop_duplicates("census_district").set_index("census_district")["unit"]
    assert lk.groupby("census_district")["unit"].nunique().max() == 1
    rows = list(openpyxl.load_workbook(XLSX, read_only=True)["Table 8a"].iter_rows(
        values_only=True))
    nat = next(r for r in rows if r[0] == "Samoa")
    # columns: place, Total(T,M,F), born in Samoa(T,M,F), born abroad(T,M,F),
    # naturalised(T,M,F), not a citizen(T,M,F)
    assert nat[1] == TOTAL and nat[13] == NONCIT, nat
    dist = {}
    for r in rows:
        lab = r[0]
        if isinstance(lab, str) and lab.startswith("        ") and not lab.startswith("         "):
            dist[lab.strip()] = (int(r[1]), int(r[13]))
    assert sum(t for t, _ in dist.values()) == TOTAL and sum(n for _, n in dist.values()) == NONCIT
    missing = set(dist) - set(d2u.index)
    assert not missing, missing
    non = pd.Series({d2u[k]: 0 for k in dist})
    tot8 = pd.Series({d2u[k]: 0 for k in dist})
    for k, (t, n) in dist.items():
        non[d2u[k]] += n
        tot8[d2u[k]] += t
    assert (tot8.sort_index() == pop.sort_index()).all(), (tot8 - pop)
    out = []
    for u, n in pop.items():
        out.append(dict(geo_id=u, geo_level="district", geo_name=u, source_category="Samoan",
                        count=int(n - non[u]), tier="derived", year=2021,
                        note=f"2021 population {n}, less {non[u]} non-citizens"))
    df = pd.DataFrame(out)
    assert df["count"].sum() == TOTAL - NONCIT
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} districts, {df['count'].sum():,} people "
          f"({NONCIT:,} non-citizens not drawn)")


if __name__ == "__main__":
    main()
