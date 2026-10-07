"""Micronesia (FSM): 2010 census, Table B10A "Language mainly spoken at home", persons 3+, by
state -> data/normalized/fm.csv; placement layer data/geo/fm/fm_hexes.gpkg (state units).

    python sources/fm_census.py

SOURCE: FSM 2010 Census Basic Tables (stats.gov.fm), sheet "Basic Tables", rows "Language
mainly spoken at home 3+ years" .. "Other Languages" (Table B10A), read from religiondots'
download data/raw/fm/fsm_basic_tables_2010.xlsx (read-only). A real home-language question,
so rows are `measured`. The 2023 census basic tables (religiondots' other files) have no
language or ethnicity table. Table B08 (ethnicity) of the same census is the cross-check.

CHECKS: the 15 languages sum to the 3+ total in every state and nationally; states sum to the
national column.

PLACEMENT: religiondots' fm_hexes.gpkg (Yap and Pohnpei by municipality, Chuuk and Kosrae whole)
re-keyed to the four states through fm_lookup.csv's `state` column (read-only).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402

XLSX = RD / "data" / "raw" / "fm" / "fsm_basic_tables_2010.xlsx"
OUT = ROOT / "data" / "normalized" / "fm.csv"
HEX = ROOT / "data" / "geo" / "fm" / "fm_hexes.gpkg"
STATES = ["Yap", "Chuuk", "Pohnpei", "Kosrae"]
HEAD = "Language mainly spoken at home 3+ years"


def num(v):
    return 0 if v in (None, "-", "") else int(v)


def main():
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, read_only=True)["Basic Tables"]
    rows = [r for r in ws.iter_rows(values_only=True)]
    i = next(k for k, r in enumerate(rows) if r[0] == HEAD)
    total = [num(v) for v in rows[i][1:6]]
    langs = []
    for r in rows[i + 1:]:
        if r[0] is None or str(r[0]).startswith("Source"):
            break
        langs.append((str(r[0]).strip(), [num(v) for v in r[1:6]]))
    assert len(langs) == 15, [l for l, _ in langs]
    for j in range(5):
        s = sum(v[j] for _, v in langs)
        assert s == total[j], (j, s, total[j])
    for _, v in langs:
        assert v[0] == sum(v[1:]), v
    print(f"B10A: {len(langs)} languages, {total[0]:,} people 3+; states "
          + ", ".join(f"{s} {t:,}" for s, t in zip(STATES, total[1:])))
    out = []
    for lab, v in langs:
        for s, n in zip(STATES, v[1:]):
            if n > 0:
                out.append(dict(geo_id=s.lower(), geo_level="state", geo_name=s,
                                source_category=lab, count=n, tier="measured", year=2010,
                                source_id="fsm_census_2010_b10a"))
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(df)} rows, {df['count'].sum():,} people")
    hexes()


def hexes():
    import geopandas as gpd
    g = gpd.read_file(RD_GEO / "fm" / "fm_hexes.gpkg")
    lut = pd.read_csv(RD_GEO / "fm" / "fm_lookup.csv", dtype=str)
    st = dict(zip(lut["unit"], lut["state"].str.lower()))
    missing = set(g["unit"]) - set(st)
    assert not missing, missing
    g["unit"] = g["unit"].map(st)
    assert sorted(g["unit"].unique()) == sorted(s.lower() for s in STATES)
    print("  hexes by state (Kontur people): "
          + ", ".join(f"{k} {v:,.0f}" for k, v in g.groupby("unit")["pop"].sum().items()))
    HEX.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(HEX, driver="GPKG")


if __name__ == "__main__":
    main()
