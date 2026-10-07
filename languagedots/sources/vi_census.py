"""US Virgin Islands, 2020 Island Areas Census, language spoken at home -> data/normalized/vi.csv.

    python sources/vi_census.py [--fetch]

SOURCE. U.S. Census Bureau, 2020 Island Areas Censuses, U.S. Virgin Islands Demographic and
Housing Characteristics Summary File (DHC), public domain:
  https://www2.census.gov/programs-surveys/decennial/2020/data/island-areas/us-virgin-islands/
      demographic-and-housing-characteristics-file/vi2020.dhc.zip
with its table matrix (which table sits in which segment, and in what order):
  https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/
      island-areas-tech-docs/dhc/2020-iac-dhc-usvi-table-matrix.xlsx
The layout is Guam's (sources/gu_census.py reads the same product); this file reads it on its own.

THE TABLE. PBG5 "Age by language spoken at home for the population 5 years and over in
households", 11 cells: two age bands (5-17, 18+), each split into speak only English, Spanish,
"French, Haitian, or Cajun" and other languages. Published at block group (summary level 150),
the finest level any language table reaches; nothing is suppressed. These four groups are all
the 2020 USVI census publishes on language: PCT22-PCT24 (tract and up) and the Detailed
Cross-Tabulations (CT25, CT46, ...) use the same four. One answer per person: the questionnaire
asks whether the person speaks a language other than English at home and, if so, which, so
"English" is English only and an English-and-Spanish home counts under Spanish.

CHECKS (asserted)
  1. PBG5 adds up in every record: each band's groups sum to the band, the bands to the total
  2. block groups summed per tract equal PCT22 (a second table: the same four groups by age,
     published at tract) in every tract and every group, exactly; tracts sum to the three islands
     and the islands to the territory, in both tables
  3. the 92 block groups are all published and their people sum to the territory's 80,430
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "vi"
NORM = HERE / "data" / "normalized"
BASE = "https://www2.census.gov/programs-surveys/decennial/2020/"
DHC = "data/island-areas/us-virgin-islands/demographic-and-housing-characteristics-file/"
DOC = "technical-documentation/island-areas-tech-docs/"
FILES = {
    "vi2020.dhc.zip": BASE + DHC + "vi2020.dhc.zip",
    "2020-iac-dhc-readme.pdf": BASE + DHC + "2020-iac-dhc-readme.pdf",
    "2020-iac-dhc-usvi-table-matrix.xlsx": BASE + DOC + "dhc/2020-iac-dhc-usvi-table-matrix.xlsx",
    "2020-iac-dhc-geographic-header-record-usvi.xlsx": BASE + DOC + "dhc/2020-iac-dhc-geographic-header-record-usvi.xlsx",
    "2020-iac-usvi-dct-list-of-tables.xlsx": BASE + DOC + "detailed-cross-tabulations/2020-iac-usvi-dct-list-of-tables.xlsx",
}

# the four groups, cells 3-6 (5-17) and 8-11 (18+) of PBG5 and of PCT22 alike; 2 and 7 are the bands
LANGS = ["Speak only English", "Speak Spanish", "Speak French, Haitian, or Cajun", "Speak other languages"]
GEO_COLS = {2: "SUMLEV", 4: "GEOCOMP", 7: "LOGRECNO", 8: "GEOID", 86: "BASENAME", 87: "NAME"}
N_BG, N_TRACTS = 92, 32
TOTAL_5PLUS = 80_430


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        p = RAW / name
        if p.exists() and p.stat().st_size > 0:
            continue
        print(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        p.with_suffix(p.suffix + ".part").write_bytes(data)
        p.with_suffix(p.suffix + ".part").replace(p)


def table_position(table):
    """(segment, first field index, cells) of a table in the segment files, from the table matrix."""
    t = pd.read_excel(RAW / "2020-iac-dhc-usvi-table-matrix.xlsx", sheet_name="Table Segments", header=None)
    t = t.iloc[2:, 5:9].dropna()
    t.columns = ["seg", "table", "cells", "order"]
    t = t.astype({"seg": int, "cells": int, "order": int})
    row = t[t["table"] == table]
    assert len(row) == 1, table
    seg, order = int(row["seg"].iloc[0]), int(row["order"].iloc[0])
    before = t[(t["seg"] == seg) & (t["order"] < order)]
    assert sorted(before["order"]) == list(range(1, order)), f"{table}: gap in segment order"
    # segment files lead with FILEID, STUSAB, CHARITER, CIFSN, LOGRECNO
    return seg, 5 + int(before["cells"].sum()), int(row["cells"].iloc[0])


def read_table(z, geo, table):
    seg, first, n = table_position(table)
    with z.open(f"vi{seg:05d}2020.dhc") as fh:
        s = pd.read_csv(fh, sep="|", header=None, dtype=str)
    out = s[[4] + list(range(first, first + n))].copy()
    out.columns = ["LOGRECNO"] + list(range(1, n + 1))
    out = out.dropna(subset=[1])
    if (out[1] == ".").any():
        raise SystemExit(f"{table}: suppressed cells ('.') appeared; this script assumes none")
    for c in range(1, n + 1):
        out[c] = out[c].astype(int)
    return geo.merge(out, on="LOGRECNO", validate="one_to_one")


def groups(p):
    """Check 1 on a PBG5/PCT22-shaped table, then add one column per group (both bands summed)."""
    bad = (p[[3, 4, 5, 6]].sum(axis=1) != p[2]) | (p[[8, 9, 10, 11]].sum(axis=1) != p[7]) \
        | (p[2] + p[7] != p[1])
    if bad.any():
        raise SystemExit(f"check 1: {int(bad.sum())} records do not add up")
    for i, lab in enumerate(LANGS):
        p[lab] = p[3 + i] + p[8 + i]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    NORM.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(RAW / "vi2020.dhc.zip")
    with z.open("vigeo2020.dhc") as fh:
        geo = pd.read_csv(fh, sep="|", header=None, dtype=str, encoding="latin-1")
    geo = geo[list(GEO_COLS)].rename(columns=GEO_COLS)
    geo = geo[geo["GEOCOMP"] == "00"]

    b = groups(read_table(z, geo, "PBG5"))
    t = groups(read_table(z, geo, "PCT22"))
    print(f"check 1 ok: PBG5 adds up in all {len(b)} records, PCT22 in all {len(t)}")

    vi = t[t["SUMLEV"] == "040"].iloc[0]
    print(f"US Virgin Islands, people aged 5+ in households: {vi[1]:,}")
    for lab in LANGS:
        print(f"    {lab:36s} {vi[lab]:>7,}  {vi[lab] / vi[1]:7.2%}")

    # check 3
    bg = b[b["SUMLEV"] == "150"].copy()
    if len(bg) != N_BG or bg[1].sum() != TOTAL_5PLUS or vi[1] != TOTAL_5PLUS:
        raise SystemExit(f"check 3: {len(bg)} block groups, {bg[1].sum():,} people")
    bg["geo_id"] = bg["GEOID"].str[9:]
    assert bg["geo_id"].str.len().eq(12).all() and bg["geo_id"].is_unique

    # check 2
    tr = t[t["SUMLEV"] == "140"].copy()
    tr["geo_id"] = tr["GEOID"].str[9:]
    if len(tr) != N_TRACTS:
        raise SystemExit(f"expected {N_TRACTS} tracts, got {len(tr)}")
    s = bg.assign(tract=bg["geo_id"].str[:11]).groupby("tract")[LANGS].sum()
    ref = tr.set_index("geo_id")[LANGS]
    d = s.reindex(ref.index, fill_value=0) - ref
    if d.abs().to_numpy().sum() or set(s.index) - set(ref.index):
        raise SystemExit(f"check 2: block groups do not sum to PCT22's tracts\n{d[d.abs().sum(axis=1) > 0]}")
    isl = t[t["SUMLEV"] == "050"].assign(geo_id=lambda x: x["GEOID"].str[9:]).set_index("geo_id")
    for name, tab in (("PCT22", tr), ("PBG5", bg)):
        si = tab.assign(c=tab["geo_id"].str[:5]).groupby("c")[LANGS].sum()
        if (si.reindex(isl.index) - isl[LANGS]).abs().to_numpy().sum() or (si.sum() - vi[LANGS]).abs().sum():
            raise SystemExit(f"check 2: {name} does not sum to the islands and the territory")
    print(f"check 2 ok: {N_BG} block groups (PBG5) sum exactly to PCT22 in all {N_TRACTS} tracts and "
          f"four groups; both sum to the 3 islands and the territory")
    for g, r in isl.iterrows():
        print(f"    {r['BASENAME']:12s} {r[1]:>7,}  " + "  ".join(f"{r[lab] / r[1]:6.1%}" for lab in LANGS))

    p1 = read_table(z, geo, "P1")
    p1 = p1[p1["SUMLEV"].isin(["040", "150"])]
    tot = int(p1.loc[p1["SUMLEV"] == "040", 1].iloc[0])
    # completeness judged by people, not record count (the header can leave units out): the
    # block groups' total population must be the territory's
    if p1.loc[p1["SUMLEV"] == "150", 1].sum() != tot:
        raise SystemExit("check 3: block groups' P1 does not sum to the territory; a block group is missing")
    print(f"total population (P1) {tot:,}; aged 5+ in households {vi[1]:,} ({vi[1] / tot:.1%}); "
          f"left out {tot - vi[1]:,}")

    bg = bg[bg[1] > 0]
    long = bg.melt(id_vars=["geo_id"], value_vars=LANGS, var_name="source_category", value_name="count")
    long["geo_level"] = "block_group"
    long = long[long["count"] > 0]
    assert long["count"].sum() == TOTAL_5PLUS
    long[["geo_id", "geo_level", "source_category", "count"]].to_csv(NORM / "vi.csv", index=False)
    print(f"wrote vi.csv: {len(long)} rows over {bg['geo_id'].nunique()} block groups with people; "
          f"{long['count'].sum():,} people")


if __name__ == "__main__":
    main()
