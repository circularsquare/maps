"""Micronesia — boundaries for the 33 drawn units, at two tiers.

Writes data/geo/fm/fm_units.gpkg and data/geo/fm/fm_lookup.csv.

OCHA COD-AB Micronesia (`cod-ab-fsm`, 2019), the **ADM2 municipality** shapefile, on §12's
Chile rule. 75 polygons, p-coded FM101-FM120 (Yap), FM201-FM240 (Chuuk), FM301-FM311 (Pohnpei)
and FM401-FM404 (Kosrae), each carrying its ADM1 parent, a `MAIN_OUTER` flag and the statistics
office's own `STATS_MCOD`.

**THE DRAWN TIER IS MIXED, BECAUSE THE SOURCE IS.** Yap's and Pohnpei's 2023 workbooks publish
religion by municipality and Chuuk's and Kosrae's do not (`sources/fm.py`, `sources/fm.md` §4).
So this layer is Yap's 20 municipalities, Pohnpei's 11, and Chuuk and Kosrae **dissolved whole**
from their 40 and 4 ADM2 polygons. Every drawn unit is a unit some FSM table counted.

**THE JOIN IS BY NAME, SCOPED BY STATE, AND THE P-CODE THEN TESTS IT.** Benin's rule (§9bb):
never join on the p-code, so the p-code stays free to check the result. It pays here — COD's
FM101..FM120 and FM301..FM311 run in **exactly** the census's own print order, so a name join
that put any municipality in the wrong place would show up as a rank that moved. Two of the 31
names need a fold and both are one letter: COD writes `Mwokilloa` for the census's `Mwoakilloa`
and `Sapwuafik` for `Sapwuahfik`.

**AND COD'S `MAIN_OUTER` REPRODUCES YAP'S OWN TABLE STRUCTURE.** The workbook prints Table B6 in
two blocks, `YAP PROPER` (ten) and `OUTER ISLANDS` (ten); COD independently flags the same ten
Main and the same ten Outer. Neither file knows about the other, so that agreement is evidence
about the pairing and it is asserted.

**MICRONESIA DOES NOT CROSS THE ANTIMERIDIAN**, unlike Kiribati and Fiji. It runs 137°E to 163°E
over 2,700 km of ocean, so the ordinary bounding-box width check works here and is used
([[reference_antimeridian]]). What it cannot help with is that a state polygon is mostly water:
Chuuk reaches from the Western Islands to the Mortlocks and Yap out to Satawal, so the placement
grid has to be a population surface and not an area weight (`sources/fm_grid.py`).

Usage:
    python sources/fm_geo.py --fetch    one ~1.7 MB zip from HDX
    python sources/fm_geo.py            rebuild from data/raw/fm/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "fm")
OUT_DIR = os.path.join(ROOT, "data", "geo", "fm")
OUT = os.path.join(OUT_DIR, "fm_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "fm_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "fm.csv")

ZIP_URL = ("https://data.humdata.org/dataset/dc71c13f-e848-4ddc-9074-17e608464b63/resource/"
           "7348d022-6726-438c-9b1c-0b5524b7dbfd/download/fsm_admbnda_shp.zip")
ZIP_NAME = "fsm_admbnda_SHP.zip"
SHP2 = "fsm_admbnda_adm2.shp"
SHP1 = "fsm_admbnda_adm1.shp"

EXPECTED_ADM2 = 75
EXPECTED_ADM1 = 4
PER_STATE_ADM2 = {"Yap": 20, "Chuuk": 40, "Pohnpei": 11, "Kosrae": 4}

# The country runs Ngulu (about 137.5 E) to Kosrae (about 163.1 E) and Kapingamarangi
# (about 1.0 N) to the Hall Islands (about 9.9 N). Nothing here is near 180.
EXPECTED_LON = (135.0, 165.0)
EXPECTED_LAT = (0.0, 12.0)
MAX_SPAN_DEG = 27.0

# COD spells two of Pohnpei's outer atolls without the census's vowel. Nothing else differs.
NAME_ALIAS = {
    "mwoakilloa": "mwokilloa",
    "sapwuahfik": "sapwuafik",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s)


def key(name):
    k = fold(name)
    return NAME_ALIAS.get(k, k)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise SystemExit(f"HDX returned something that is not a zip ({len(r.content):,} bytes)")
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def build():
    import geopandas as gpd
    import pandas as pd

    from fm import YAP_MAIN, YAP_OUTER, POHNPEI_MUNIS, COARSE_STATES

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"{src} is missing — run `python sources/fm_geo.py --fetch`")
    g = gpd.read_file("zip://" + src + "!" + SHP2)
    if len(g) != EXPECTED_ADM2:
        raise SystemExit(f"{len(g)} ADM2 polygons, expected {EXPECTED_ADM2} — COD reissued it")
    g = g.to_crs(4326)
    a1 = gpd.read_file("zip://" + src + "!" + SHP1).to_crs(4326)
    if len(a1) != EXPECTED_ADM1:
        raise SystemExit(f"{len(a1)} ADM1 polygons, expected {EXPECTED_ADM1}")

    # --- the antimeridian check, per country: Micronesia genuinely does not cross 180, so
    # the ordinary width assertion is the right one here (Kiribati and Fiji need per-polygon).
    minx, miny, maxx, maxy = g.total_bounds
    if maxx - minx > MAX_SPAN_DEG:
        raise SystemExit(f"the ADM2 layer spans {maxx - minx:.1f}° of longitude, more than "
                         f"{MAX_SPAN_DEG}° — a polygon is torn across 180 "
                         "[[reference_antimeridian]]")
    if not (EXPECTED_LON[0] <= minx and maxx <= EXPECTED_LON[1]
            and EXPECTED_LAT[0] <= miny and maxy <= EXPECTED_LAT[1]):
        raise SystemExit(f"bounds {minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f} are not FSM")
    print(f"  {len(g)} ADM2 polygons, {minx:.2f}E..{maxx:.2f}E, {miny:.2f}N..{maxy:.2f}N — "
          f"{maxx - minx:.1f}° wide and nowhere near 180")

    counts = g["ADM1NAME"].value_counts().to_dict()
    if counts != PER_STATE_ADM2:
        raise SystemExit(f"ADM2 per state is {counts}, expected {PER_STATE_ADM2}")

    rows, parts = [], []

    # --- the two fine states, joined by NAME inside their own state.
    for state, names in (("Yap", YAP_MAIN + YAP_OUTER), ("Pohnpei", POHNPEI_MUNIS)):
        sub = g[g["ADM1NAME"] == state]
        by = {}
        for idx, r in sub.iterrows():
            by.setdefault(key(r["ADM2_NAME"]), []).append(idx)
        dup = {k: v for k, v in by.items() if len(v) > 1}
        if dup:
            raise SystemExit(f"{state}: COD repeats a municipality name: {dup}")
        folded = 0
        for nm in names:
            hit = by.get(key(nm))
            if not hit:
                raise SystemExit(f"{state}: no COD polygon for census municipality {nm!r} "
                                 f"(key {key(nm)!r}) — add it to NAME_ALIAS")
            idx = hit[0]
            r = sub.loc[idx]
            if fold(nm) != fold(r["ADM2_NAME"]):
                folded += 1
            uid = f"{state.lower()}-{nm.lower()}"
            rows.append((uid, "municipality", nm, state, r["ADM2_PCODE"],
                         str(r["STATS_MCOD"]), r["MAIN_OUTER"], 1))
            parts.append((uid, sub.loc[[idx], "geometry"]))
        print(f"  {state}: joined {len(names)}/{len(names)} on the name inside the state "
              f"({folded} via NAME_ALIAS)")

        # THE FREE INDEPENDENT CHECK. COD's p-codes were minted in the office's own order and
        # the census prints its columns in that same order, so the two sequences must agree
        # exactly. A name that matched the wrong municipality moves a rank and shows up here
        # even though every total still reconciles. [[reference_name_join_wrong_neighbour]]
        got = [pc for _, _, _, st, pc, _, _, _ in rows if st == state]
        want = sorted(got)
        if got != want:
            bad = [(n, p) for (_, _, n, st, p, _, _, _) in rows
                   if st == state]
            raise SystemExit(f"{state}: the census print order and COD's p-code order "
                             f"DISAGREE, so the name join put at least one municipality in "
                             f"the wrong place: {bad}")
        print(f"    p-code order {got[0]}..{got[-1]} matches the census print order exactly")

    # AND YAP'S OWN TABLE STRUCTURE, from the other file. The workbook prints ten under
    # `YAP PROPER` and ten under `OUTER ISLANDS`; COD flags ten Main and ten Outer.
    main = {n for (_, _, n, st, _, _, mo, _) in rows if st == "Yap" and mo == "Main"}
    outer = {n for (_, _, n, st, _, _, mo, _) in rows if st == "Yap" and mo == "Outer"}
    if main != set(YAP_MAIN) or outer != set(YAP_OUTER):
        raise SystemExit(f"COD's Main/Outer split of Yap is {sorted(main)} / {sorted(outer)}, "
                         f"and the workbook's two blocks are {YAP_MAIN} / {YAP_OUTER}")
    print(f"  Yap: COD's MAIN_OUTER flag reproduces the workbook's own YAP PROPER / OUTER "
          f"ISLANDS split, {len(main)} and {len(outer)}, from a file that has never seen it")

    # --- the two coarse states, dissolved whole.
    for state in COARSE_STATES:
        sub = g[g["ADM1NAME"] == state]
        if len(sub) != PER_STATE_ADM2[state]:
            raise SystemExit(f"{state}: {len(sub)} ADM2 to dissolve, expected "
                             f"{PER_STATE_ADM2[state]}")
        merged = sub.geometry.union_all()
        uid = state.lower()
        rows.append((uid, "state", state, state, sub["ADM1_PCODE"].iloc[0], "", "", len(sub)))
        parts.append((uid, gpd.GeoSeries([merged], crs=g.crs)))
        # The dissolve must not have lost or invented land. COD publishes ADM1 separately.
        a = a1[a1["ADM1NAME"] == state].geometry.union_all()
        ratio = merged.area / a.area if a.area else 0.0
        if not 0.98 <= ratio <= 1.02:
            raise SystemExit(f"{state}: the {len(sub)} dissolved ADM2 cover {ratio:.4f} of "
                             f"COD's own ADM1 polygon — the dissolve dropped or added land")
        print(f"  {state}: {len(sub)} ADM2 dissolved to one unit, {ratio:.4f} of COD's "
              "own ADM1 polygon")

    # --- and the unit set has to be exactly what fm.csv counted.
    if not os.path.exists(NORM):
        raise SystemExit(f"{NORM} is missing — run `python sources/fm.py` first")
    df = pd.read_csv(NORM, dtype=str, keep_default_na=False)
    df["count"] = df["count"].astype(int)
    want_ids = set(df["geo_id"])
    got_ids = {u for u, *_ in rows}
    if want_ids != got_ids:
        raise SystemExit(f"fm.csv and this layer disagree about the unit set:\n"
                         f"  only in fm.csv: {sorted(want_ids - got_ids)}\n"
                         f"  only in the layer: {sorted(got_ids - want_ids)}")
    lvl = dict(zip(df["geo_id"], df["geo_level"]))
    bad = [(u, l, lvl[u]) for u, l, *_ in rows if lvl[u] != l]
    if bad:
        raise SystemExit(f"geo_level disagrees between fm.csv and this layer: {bad}")

    out = gpd.GeoDataFrame(
        {"unit": [u for u, _ in parts]},
        geometry=pd.concat([s.reset_index(drop=True) for _, s in parts],
                           ignore_index=True), crs=g.crs)
    meta = {u: (l, n, st, pc, mc, mo, k) for u, l, n, st, pc, mc, mo, k in rows}
    out["level"] = out["unit"].map(lambda u: meta[u][0])
    out["name"] = out["unit"].map(lambda u: meta[u][1])
    out["state"] = out["unit"].map(lambda u: meta[u][2])
    out["pcode"] = out["unit"].map(lambda u: meta[u][3])
    out = out[["unit", "level", "name", "state", "pcode", "geometry"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, driver="GPKG", layer="units")
    os.replace(tmp, OUT)

    with open(LOOKUP + ".part", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "level", "name", "state", "pcode", "stats_mcod",
                    "main_outer", "adm2_parts"])
        for u, l, n, st, pc, mc, mo, k in rows:
            w.writerow([u, u, l, n, st, pc, mc, mo, k])
    os.replace(LOOKUP + ".part", LOOKUP)

    per = df.groupby("geo_id")["count"].sum()
    print(f"\n    {'unit':<26} {'tier':<13} {'people':>7}")
    for u, l, n, st, *_ in sorted(rows, key=lambda r: -per[r[0]]):
        label = n if l == "state" else f"{st}/{n}"
        print(f"    {label:<26} {l:<13} {per[u]:>7,}")
    print(f"    {'':<26} {'':<13} {per.sum():>7,}")
    print(f"\nwrote {OUT} ({len(out)} units)")
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        build()
