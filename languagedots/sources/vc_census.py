"""Saint Vincent and the Grenadines: Vincentian Creole and English per enumeration district from
the 2012 census ethnicity table -> data/normalized/vc.csv.

    python sources/vc_census.py

NO CENSUS LANGUAGE QUESTION (2012; the US Census Bureau's country geodatabase carries every
published table and none is on language). Built as Barbados and Grenada (sources/bb.md,
sources/gd.md): everyone on the national creole except white Vincentians on English. Every row
`derived`.

THE TABLE: USCB, "Saint Vincent and the Grenadines" subnational census tables (2021-09
release; religiondots' data/raw/vc/saint_vincent_and_the_grenadines_uscb_202109.xlsx, read only),
sheet "Ethnicity and Religion", 2012 census by enumeration district (ADM_LEVEL 2, 221 rows):
ETH_TPOP and ETH_BLK, ETH_INDG, ETH_WHT, ETH_EIND, ETH_MIX, ETH_POR, ETH_OTHR. GEO_MATCH is the
id religiondots' ED layer is keyed on.

CHECKS: 221 EDs; the ethnic categories sum to ETH_TPOP on every ED; the EDs sum to the national
row (109,188) column by column; the ids equal religiondots' ed ids both ways.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

XLSX = RD / "data" / "raw" / "vc" / "saint_vincent_and_the_grenadines_uscb_202109.xlsx"
OUT = HERE / "data" / "normalized" / "vc.csv"
ETH = ["ETH_BLK", "ETH_INDG", "ETH_WHT", "ETH_EIND", "ETH_MIX", "ETH_POR", "ETH_OTHR"]
TOTAL = 109_188


def main():
    import openpyxl
    rows = list(openpyxl.load_workbook(XLSX, read_only=True)["Ethnicity and Religion"]
                .iter_rows(values_only=True))
    df = pd.DataFrame(rows[2:], columns=rows[0])
    nat = df[df["ADM_LEVEL"] == 0].iloc[0]
    ed = df[df["ADM_LEVEL"] == 2].copy()
    assert len(ed) == 221 and nat["ETH_TPOP"] == TOTAL
    for c in ETH + ["ETH_TPOP"]:
        ed[c] = ed[c].astype(int)
        assert ed[c].sum() == nat[c], c
    bad = ed[ed[ETH].sum(axis=1) != ed["ETH_TPOP"]]
    assert bad.empty, bad[["GEO_MATCH", "ETH_TPOP"]]
    rd = pd.read_csv(RD / "data" / "normalized" / "vc.csv", keep_default_na=False, na_values=[])
    rd_ids = set(rd.loc[rd["geo_level"] == "ed", "geo_id"])
    assert set(ed["GEO_MATCH"]) == rd_ids, set(ed["GEO_MATCH"]) ^ rd_ids
    out = []
    for _, r in ed.iterrows():
        w = int(r["ETH_WHT"])
        for lab, n in (("White", w), ("Everyone else", int(r["ETH_TPOP"]) - w)):
            if n:
                out.append(dict(geo_id=r["GEO_MATCH"], geo_level="ed", geo_name=r["AREA_NAME"],
                                source_category=lab, count=n, tier="derived", year=2012))
    o = pd.DataFrame(out)
    assert o["count"].sum() == TOTAL
    o.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {o['geo_id'].nunique()} EDs, {TOTAL:,} people, "
          f"{int(nat['ETH_WHT']):,} white on English")


if __name__ == "__main__":
    main()
