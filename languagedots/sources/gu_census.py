"""Guam, 2020 Island Areas Census, language spoken at home -> data/normalized/gu.csv.

    python sources/gu_census.py [--fetch]

SOURCE. U.S. Census Bureau, 2020 Island Areas Censuses, Guam Demographic and Housing
Characteristics Summary File (DHC), public domain:
  https://www2.census.gov/programs-surveys/decennial/2020/data/island-areas/guam/
      demographic-and-housing-characteristics-file/gu2020.dhc.zip
with its table matrix (which table sits in which segment, and in what order):
  https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/
      island-areas-tech-docs/dhc/2020-iac-dhc-guam-table-matrix.xlsx
api.census.gov carries the same tables (dataset dec/dhcgu) but refuses unkeyed requests; the
summary file needs no key.

THE TABLE. PCT25 "Age by language spoken at home for the population 5 years and over in
households (excluding people in military housing units)", 27 cells: two age bands (5-17, 18+),
each split into speak only English and 11 language groups. Published for Guam, the 19 villages
(county subdivisions), 55 tracts and their parts. One answer per person: the Island Areas
questionnaire asks whether the person speaks a language other than English at home and, if so,
which, so "English" is English only and a Chamorro-and-English home counts under Chamorro.

THE GRAIN. Tracts (summary level 140): the finest level PCT25 is published at. Villages (060)
and tract-in-village parts (080) are read as checks.

SUPPRESSION. The Bureau prints "." for PCT25 (and every other sample table) in six tracts
(9516, 9519.01, 9519.02, 9524, 9534, 9554; 10,934 people in all) and three villages (Hagåtña,
Tamuning, Umatac). Guam's own row is published, so Guam minus the 49 published tracts is exactly
the six tracts' people, language by language. They become one unit, SUPPRESSED_UNIT, whose
counts are the census's; only where inside the six tracts each person goes is borrowed
(countries/gu.py places it by each tract's published total population, P1).

WHAT THIS WRITES
  gu.csv                 geo_id (11-digit tract GEOID, or SUPPRESSED_UNIT), geo_level,
                         source_category, count (both age bands summed)
  gu_suppressed_p1.csv   the six suppressed tracts' total population (P1), for placement

CHECKS (asserted)
  1. every cell group of PCT25 adds up: each age band's languages sum to the band, the bands to
     the total, in every record at every level
  2. the suppressed cells are exactly the pinned ones; Guam minus the published tracts, and minus
     the published villages, is non-negative in every row
  3. where a tract (or village) and all its tract-in-village parts (080) are published, the parts
     sum to it exactly (the two levels are the same people, cut two ways)
  4. a second table of the same census: PBG5 (the same universe in six coarser groups, published
     at block group) summed per tract equals PCT25 collapsed to those six groups, in every
     published tract whose block groups are all published
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
RAW = HERE / "data" / "raw" / "gu"
NORM = HERE / "data" / "normalized"
BASE = "https://www2.census.gov/programs-surveys/decennial/2020/"
FILES = {
    "gu2020.dhc.zip": BASE + "data/island-areas/guam/demographic-and-housing-characteristics-file/gu2020.dhc.zip",
    "2020-iac-dhc-guam-table-matrix.xlsx": BASE + "technical-documentation/island-areas-tech-docs/dhc/"
                                                   "2020-iac-dhc-guam-table-matrix.xlsx",
    "2020-iac-dhc-readme.pdf": BASE + "data/island-areas/guam/demographic-and-housing-characteristics-file/"
                                      "2020-iac-dhc-readme.pdf",
}

# PCT25's cells in order, as data.census.gov labels them (dec/dhcgu groups/PCT25.json). The same
# 12 rows repeat for 5-17 (cells 3-14) and 18+ (cells 16-27); cell 1 is the total, 2 and 15 the bands.
LANGS = ["Speak only English", "Speak Chamorro", "Speak Carolinian", "Speak Palauan", "Speak Chuukese",
         "Speak Philippine languages", "Speak other Pacific Island languages", "Speak Chinese",
         "Speak Japanese", "Speak Korean", "Speak other Asian languages", "Speak other languages"]
# PBG5's six groups (5-17 cells 3-8, 18+ cells 10-15), and which PCT25 rows make each
PBG5 = {"English only": ["Speak only English"],
        "Chamorro": ["Speak Chamorro"],
        "Philippine languages": ["Speak Philippine languages"],
        "Other Pacific Island languages": ["Speak Carolinian", "Speak Palauan", "Speak Chuukese",
                                           "Speak other Pacific Island languages"],
        "Asian languages": ["Speak Chinese", "Speak Japanese", "Speak Korean", "Speak other Asian languages"],
        "Other languages": ["Speak other languages"]}
GEO_COLS = {2: "SUMLEV", 4: "GEOCOMP", 7: "LOGRECNO", 8: "GEOID", 86: "BASENAME", 87: "NAME"}
N_VILLAGES, N_TRACTS = 19, 55
# PCT25 prints "." (not shown) for these, all levels of the same cells: the Bureau's
# suppression of small or complementary cells. Pinned so a re-release that changes them stops.
SUPPRESSED_TRACTS = ["66010951600", "66010951901", "66010951902", "66010952400", "66010953400", "66010955400"]
SUPPRESSED_VILLAGES = ["0600000US6601034800", "0600000US6601071600", "0600000US6601078750"]  # Hagåtña, Tamuning, Umatac
SUPPRESSED_UNIT = "66010SUPPR"   # the six tracts as one unit: Guam minus every published tract


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
    """(segment, first field index) of a table in the segment files, from the table matrix."""
    t = pd.read_excel(RAW / "2020-iac-dhc-guam-table-matrix.xlsx", sheet_name="Table Segments", header=None)
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
    with z.open(f"gu{seg:05d}2020.dhc") as fh:
        s = pd.read_csv(fh, sep="|", header=None, dtype=str)
    out = s[[4] + list(range(first, first + n))].copy()
    out.columns = ["LOGRECNO"] + list(range(1, n + 1))
    out = out.dropna(subset=[1])
    out = out[out[1] != "."]             # "." = not tabulated for that summary level
    for c in range(1, n + 1):
        out[c] = out[c].astype(int)
    return geo.merge(out, on="LOGRECNO", validate="one_to_one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    NORM.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(RAW / "gu2020.dhc.zip")
    with z.open("gugeo2020.dhc") as fh:          # latin-1: Hagåtña
        geo = pd.read_csv(fh, sep="|", header=None, dtype=str, encoding="latin-1")
    geo = geo[list(GEO_COLS)].rename(columns=GEO_COLS)
    geo = geo[geo["GEOCOMP"] == "00"]

    p = read_table(z, geo, "PCT25")
    young = {lab: 3 + i for i, lab in enumerate(LANGS)}
    adult = {lab: 16 + i for i, lab in enumerate(LANGS)}
    # check 1
    bad = (p[list(young.values())].sum(axis=1) != p[2]) | (p[list(adult.values())].sum(axis=1) != p[15]) \
        | (p[2] + p[15] != p[1])
    if bad.any():
        raise SystemExit(f"check 1: {int(bad.sum())} PCT25 records do not add up")
    for lab in LANGS:
        p[lab] = p[young[lab]] + p[adult[lab]]
    print(f"check 1 ok: PCT25 adds up in all {len(p)} records")

    gu = p[p["SUMLEV"] == "040"]
    assert len(gu) == 1
    gu = gu.iloc[0]
    print(f"Guam, people aged 5+ in households outside military housing: {gu[1]:,}")
    for lab in LANGS:
        print(f"    {lab:40s} {gu[lab]:>8,}  {gu[lab] / gu[1]:7.2%}")

    # ---- suppression ----
    all_tr = set(geo.loc[geo["SUMLEV"] == "140", "GEOID"])
    all_vi = set(geo.loc[geo["SUMLEV"] == "060", "GEOID"])
    tr = p[p["SUMLEV"] == "140"].copy()
    vi = p[p["SUMLEV"] == "060"].copy()
    sup_tr = sorted(g[9:] for g in all_tr - set(tr["GEOID"]))
    sup_vi = sorted(all_vi - set(vi["GEOID"]))
    if len(all_tr) != N_TRACTS or len(all_vi) != N_VILLAGES + 1:   # +1: "County subdivision not defined"
        raise SystemExit(f"expected {N_TRACTS} tracts and {N_VILLAGES} villages, got {len(all_tr)} and {len(all_vi)}")
    if sup_tr != SUPPRESSED_TRACTS or sup_vi != SUPPRESSED_VILLAGES:
        raise SystemExit(f"suppressed cells changed: tracts {sup_tr}, villages {sup_vi}")
    p1 = read_table(z, geo, "P1")
    p1_tr = p1[p1["SUMLEV"] == "140"].assign(geo_id=lambda d: d["GEOID"].str[9:]).set_index("geo_id")[1]
    print(f"suppressed in PCT25: {len(sup_tr)} tracts (total population {p1_tr[sup_tr].sum():,}: "
          + ", ".join(f"{t} {p1_tr[t]:,}" for t in sup_tr) + f") and {len(sup_vi)} villages")

    # check 2: the residual is non-negative in every row, at both levels, and the village residual
    # holds the tract residual's villages (Hagåtña = 9534, Umatac = 9554, Tamuning holds 9519.01,
    # 9519.02 and part of 9524; 9516 is in Barrigada, which is published, so it is not in it)
    res_tr = pd.Series({lab: gu[lab] - tr[lab].sum() for lab in LANGS})
    res_vi = pd.Series({lab: gu[lab] - vi[lab].sum() for lab in LANGS})
    if (res_tr < 0).any() or (res_vi < 0).any():
        raise SystemExit(f"check 2: negative residual\n{res_tr}\n{res_vi}")
    ratio = res_tr.sum() / p1_tr[sup_tr].sum()
    print(f"check 2 ok: residual over the suppressed tracts {res_tr.sum():,} people "
          f"({ratio:.2f} of their total population; Guam {gu[1] / p1.loc[p1['SUMLEV'] == '040', 1].iloc[0]:.2f}); "
          f"over the suppressed villages {res_vi.sum():,}")
    for lab in LANGS:
        print(f"    {lab:40s} tracts {res_tr[lab]:>6,}   villages {res_vi[lab]:>6,}")

    # check 3: where a tract and all its tract-in-village parts are published, the parts sum to it;
    # same for villages
    # "All its parts" is judged by total population (P1), not by record count: the geographic
    # header leaves out some parts altogether (9532 and 9543 each lack one, 27 and 91 people).
    parts = p[p["SUMLEV"] == "080"].copy()
    p1_all = p1.set_index("GEOID")[1]
    parts["P1"] = parts["GEOID"].map(p1_all)
    for d in (parts,):
        d["tract"] = "1400000US" + d["GEOID"].str[9:14] + d["GEOID"].str[-6:]
        d["village"] = "0600000US" + d["GEOID"].str[9:19]
    n_ok = {}
    for key, d in (("tract", tr), ("village", vi)):
        have = parts.groupby(key)["P1"].sum()
        whole = [u for u in d["GEOID"] if u in have.index and have[u] == p1_all[u]]
        s = parts[parts[key].isin(whole)].groupby(key)[LANGS].sum()
        ref = d.set_index("GEOID").loc[whole, LANGS]
        dd = (s.loc[whole] - ref)
        diff = dd.abs().to_numpy().sum()
        if diff:
            raise SystemExit(f"check 3: tract-in-village parts do not sum to their {key} (abs diff {diff:,})\n"
                             f"{dd[dd.abs().sum(axis=1) > 0].T}")
        n_ok[key] = (len(whole), len(d))
    print(f"check 3 ok: tract-in-village parts sum exactly to {n_ok['tract'][0]} of {n_ok['tract'][1]} "
          f"published tracts and {n_ok['village'][0]} of {n_ok['village'][1]} published villages "
          f"(the rest have a part suppressed or missing from the header)")

    # check 4: PBG5 by block group, summed per tract, against PCT25 collapsed, where all of a
    # published tract's block groups are published
    b = read_table(z, geo, "PBG5")
    b = b[b["SUMLEV"] == "150"].copy()
    b["tract"] = "1400000US" + b["GEOID"].str[9:9 + 11]
    b["P1"] = b["GEOID"].map(p1_all)
    groups = list(PBG5)
    for i, g in enumerate(groups):
        b[g] = b[3 + i] + b[10 + i]
    assert (b[groups].sum(axis=1) == b[1]).all(), "PBG5 does not add up"
    have = b.groupby("tract")["P1"].sum()
    whole = [u for u in tr["GEOID"] if u in have.index and have[u] == p1_all[u]]
    bs = b[b["tract"].isin(whole)].groupby("tract")[groups].sum().reindex(whole, fill_value=0)
    t5 = pd.DataFrame({g: tr.set_index("GEOID").loc[whole, rows].sum(axis=1) for g, rows in PBG5.items()})
    diff = (bs - t5).abs()
    if diff.to_numpy().sum():
        raise SystemExit(f"check 4: PBG5 by block group != PCT25 by tract:\n{diff[diff.sum(axis=1) > 0]}")
    print(f"check 4 ok: PBG5 by block group sums to PCT25 in all six groups in {len(whole)} of "
          f"{len(tr)} published tracts (the rest have a block group suppressed or missing from the header)")

    tr["geo_id"] = tr["GEOID"].str[9:]
    tr = tr[tr[1] > 0]
    long = tr.melt(id_vars=["geo_id"], value_vars=LANGS, var_name="source_category", value_name="count")
    sup = pd.DataFrame({"geo_id": SUPPRESSED_UNIT, "source_category": LANGS, "count": res_tr[LANGS].to_numpy()})
    long = pd.concat([long, sup], ignore_index=True)
    long["geo_level"] = "tract"
    long = long[long["count"] > 0]
    assert long["count"].sum() == gu[1]
    long[["geo_id", "geo_level", "source_category", "count"]].to_csv(NORM / "gu.csv", index=False)
    pd.DataFrame({"geo_id": sup_tr, "p1": [int(p1_tr[t]) for t in sup_tr]}).to_csv(NORM / "gu_suppressed_p1.csv",
                                                                                   index=False)
    print(f"wrote gu.csv: {len(long)} rows over {tr['geo_id'].nunique()} published tracts with people and "
          f"one unit '{SUPPRESSED_UNIT}' for the {len(sup_tr)} suppressed tracts together; "
          f"{long['count'].sum():,} people; gu_suppressed_p1.csv (their total population, for placement)")
    print("  villages, for the record:")
    for _, r in vi.sort_values(1, ascending=False).iterrows():
        print(f"    {r['BASENAME']:32s} {r[1]:>7,}")


if __name__ == "__main__":
    main()
