"""Reader for the 2020 Island Areas Censuses' Demographic and Housing Characteristics summary
files (CNMI and American Samoa; Guam has its own copy in gu_census.py). Public domain, U.S.
Census Bureau:

  https://www2.census.gov/programs-surveys/decennial/2020/data/island-areas/
      commonwealth-of-the-northern-mariana-islands/demographic-and-housing-characteristics-file/mp2020.dhc.zip
      american-samoa/demographic-and-housing-characteristics-file/as2020.dhc.zip
  with each territory's table matrix (which table is in which segment and order):
  https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/island-areas-tech-docs/dhc/
      2020-iac-dhc-cnmi-table-matrix.xlsx, 2020-iac-dhc-american-samoa-table-matrix.xlsx

Used by sources/mp_census.py and sources/as_census.py (sources/mp.md, sources/as.md).
"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
BASE = "https://www2.census.gov/programs-surveys/decennial/2020/"
GEO_COLS = {2: "SUMLEV", 4: "GEOCOMP", 7: "LOGRECNO", 8: "GEOID", 86: "BASENAME", 87: "NAME"}
AREAS = {
    "mp": dict(folder="commonwealth-of-the-northern-mariana-islands", matrix="2020-iac-dhc-cnmi-table-matrix.xlsx"),
    "as": dict(folder="american-samoa", matrix="2020-iac-dhc-american-samoa-table-matrix.xlsx"),
}


def raw_dir(cc):
    return HERE / "data" / "raw" / cc


def fetch(cc):
    a = AREAS[cc]
    d = raw_dir(cc)
    d.mkdir(parents=True, exist_ok=True)
    files = {
        f"{cc}2020.dhc.zip": BASE + f"data/island-areas/{a['folder']}/demographic-and-housing-characteristics-file/{cc}2020.dhc.zip",
        a["matrix"]: BASE + "technical-documentation/island-areas-tech-docs/dhc/" + a["matrix"],
    }
    for name, url in files.items():
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            continue
        print(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)


def open_zip(cc):
    return zipfile.ZipFile(raw_dir(cc) / f"{cc}2020.dhc.zip")


def geo(cc, z):
    with z.open(f"{cc}geo2020.dhc") as fh:          # latin-1: Manu'a, Hagatna spellings
        g = pd.read_csv(fh, sep="|", header=None, dtype=str, encoding="latin-1")
    g = g[list(GEO_COLS)].rename(columns=GEO_COLS)
    return g[g["GEOCOMP"] == "00"]


def table_position(cc, table):
    """(segment, first field index, cells) of a table in the segment files, from the matrix."""
    t = pd.read_excel(raw_dir(cc) / AREAS[cc]["matrix"], sheet_name="Table Segments", header=None)
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


def read_table(cc, z, g, table):
    """The table's cells (columns 1..n) joined to the geographic header `g`; records the Bureau
    prints as "." (not tabulated) are dropped."""
    seg, first, n = table_position(cc, table)
    with z.open(f"{cc}{seg:05d}2020.dhc") as fh:
        s = pd.read_csv(fh, sep="|", header=None, dtype=str)
    out = s[[4] + list(range(first, first + n))].copy()
    out.columns = ["LOGRECNO"] + list(range(1, n + 1))
    out = out.dropna(subset=[1])
    out = out[out[1] != "."]
    for c in range(1, n + 1):
        out[c] = out[c].astype(int)
    return g.merge(out, on="LOGRECNO", validate="one_to_one")
