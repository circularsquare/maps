"""Myanmar — boundaries for the 15 States/Regions the census tabulates religion on.

Writes data/geo/mm/mm_states.gpkg and data/geo/mm/mm_lookup.csv.

OCHA COD-AB Myanmar (which is MIMU's own boundary set, redistributed), the **shapefile**
bundle rather than the geodatabase on §12's Chile rule.

**THE COUNTING GEOGRAPHY IS NOT THE ADMINISTRATIVE GEOGRAPHY, AND THE FEATURE COUNT SAYS SO
BEFORE ANY JOIN IS ATTEMPTED — §12's Kenya rule.** COD's ADM1 has **18 features** and the
census tabulates religion on **15 rows**, because the standard p-code set splits two of them:

    MMR007 Bago (East)   + MMR008 Bago (West)                        -> Bago
    MMR014 Shan (South)  + MMR015 Shan (North) + MMR016 Shan (East)  -> Shan

**MIMU's own transcription confirms this is the intended aggregation rather than a guess**:
its religion sheet codes Bago `MMR111` and Shan `MMR222`, two codes that exist precisely
because the census reports those states whole. Those aggregate codes are what this file uses
as the drawn `unit`, so the id on the map is the code of the thing actually counted.

**THE DISSOLVE IS A RULE, NOT A LIST** (§12 — derive alias maps, do not hard-code them):
strip a trailing parenthesised qualifier from COD's `adm1_name` and group on what is left, so
`Bago (East)` and `Bago (West)` land together and a future release that splits a third state
is handled rather than silently dropped.

**Three checks, and the second is the one that would catch a wrong grouping:**

  * the 15 groups pair 1:1 with the census's 15 rows, with one romanisation difference
    (DOP writes `Ayeyawady`, MIMU and OCHA write `Ayeyarwady`);
  * **every multi-member group must be CONTIGUOUS** — its members have to touch each other.
    Grouping on a stripped name is a string operation, and this is the geometric fact that
    says the string operation grouped real neighbours;
  * the 15 dissolved polygons must **tile with no overlap**, which is what rules out §12's
    Korea trap where overlapping ADM1 polygons hand a `keep="first"` sjoin the wrong unit.

And the free independent check, as in Cambodia: **the census's printed row order reproduces
COD's p-code order** on all 15, sorting each group by its lowest member code. The two have
different origins, so agreement is evidence rather than a tautology.

Usage:
    python sources/mm_geo.py --fetch    one ~34 MB zip from HDX
    python sources/mm_geo.py            rebuild from data/raw/mm/
"""

import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mm")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mm")
OUT = os.path.join(OUT_DIR, "mm_states.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mm_lookup.csv")
PCODES = os.path.join(ROOT, "data", "normalized", "mm_pcodes.csv")

ZIP_URL = ("https://data.humdata.org/dataset/3ac9b527-dff2-4b9f-a16e-476aa821896a/"
           "resource/8823f5a3-4e2c-499d-bccc-b89c4a5a1cd5/download/"
           "mmr_admin_boundaries.shp.zip")
ZIP_NAME = "mmr_admin_boundaries.shp.zip"

EXPECTED_FEATURES = 18
EXPECTED_UNITS = 15


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and zipfile.is_zipfile(dest):
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    """Romanisation-tolerant key. `Ayeyawady` (DOP) vs `Ayeyarwady` (MIMU/OCHA) is the only
    difference across the fifteen, so `r` is dropped; uniqueness is asserted on both sides."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s.lower()).replace("r", "")


def base_name(s):
    """`Bago (East)` -> `Bago`. The dissolve rule, applied to COD's names."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()


def _read_adm1():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()]
    shp = [n for n in names if re.search(r"adm(?:in)?1\.shp$", n, re.I)]
    if len(shp) != 1:
        raise SystemExit(f"expected one admin1 shapefile in the bundle, found {shp}")
    g = gpd.read_file(f"zip://{src}!{shp[0]}", engine="fiona")
    if len(g) != EXPECTED_FEATURES:
        raise SystemExit(
            f"{shp[0]}: {len(g)} features, expected {EXPECTED_FEATURES} -- COD splits Bago "
            "in two and Shan in three against the census's 15 rows, so a different count "
            "means the split has changed and the dissolve below must be re-derived")
    cols = {c.upper(): c for c in g.columns}
    name_col = next((cols[k] for k in ("ADM1_NAME", "ADM1_EN") if k in cols), None)
    code_col = cols.get("ADM1_PCODE")
    if not name_col or not code_col:
        raise SystemExit(f"no adm1 name/pcode column in {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g, name_col, code_col


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, name_col, code_col = _read_adm1()
    print(f"COD ADM1: {len(g)} polygons, crs={g.crs}")

    if not os.path.exists(PCODES):
        raise SystemExit(f"missing {PCODES} -- run sources/mm.py first")
    cen = pd.read_csv(PCODES, dtype=str)
    if len(cen) != EXPECTED_UNITS:
        raise SystemExit(f"{len(cen)} census states, expected {EXPECTED_UNITS}")

    # ---- the dissolve, by rule ----
    g = g.copy()
    g["base"] = g[name_col].map(base_name)
    groups = g.groupby("base")
    print(f"\n  the dissolve — {len(g)} COD features -> {len(groups)} groups:")
    for base, sub in groups:
        if len(sub) > 1:
            members = ", ".join(f"{c} {n}" for n, c in
                                zip(sub[name_col], sub[code_col].astype(str)))
            print(f"    {base:<16} <- {members}")
    if len(groups) != EXPECTED_UNITS:
        raise SystemExit(f"the dissolve gives {len(groups)} units, expected "
                         f"{EXPECTED_UNITS}")

    # ---- contiguity: a multi-member group's members must touch ----
    import itertools
    proj = g.to_crs(32646)
    bad = []
    for base, sub in groups:
        if len(sub) < 2:
            continue
        idx = list(sub.index)
        touching = {i: False for i in idx}
        for a, b in itertools.combinations(idx, 2):
            if proj.geometry[a].buffer(50).intersects(proj.geometry[b].buffer(50)):
                touching[a] = touching[b] = True
        loners = [g[name_col][i] for i in idx if not touching[i]]
        if loners:
            bad.append((base, loners))
    print(f"  {'OK ' if not bad else 'BAD'} every multi-member group is contiguous "
          f"({len(bad)} failures) {bad}")
    if bad:
        raise SystemExit("a dissolve group's members do not touch -- grouping on a stripped "
                         "name has put unrelated polygons together")

    diss = g.dissolve(by="base", aggfunc={code_col: "min"}).reset_index()
    diss = diss.rename(columns={code_col: "min_pcode"})

    # ---- pair to the census ----
    poly = {}
    for base, code in zip(diss["base"], diss["min_pcode"]):
        k = fold(base)
        if k in poly:
            raise SystemExit(f"COD folds {base!r} onto {poly[k][0]!r} -- collision")
        poly[k] = (base, code)
    seen = {}
    for nm in cen["geo_name"]:
        k = fold(nm)
        if k in seen:
            raise SystemExit(f"the census folds {nm!r} onto {seen[k]!r} -- collision")
        seen[k] = nm

    pairs, missing = {}, []
    for gid, nm, pcode in zip(cen["geo_id"], cen["geo_name"], cen["pcode"]):
        k = fold(nm)
        if k in poly:
            pairs[gid] = (nm, poly[k][0], poly[k][1], pcode)
        else:
            missing.append((gid, nm))
    used = {v[1] for v in pairs.values()}
    spare = [b for b in diss["base"] if b not in used]

    print("\n  the join, both ways (§12):")
    print(f"    census states              {len(cen):>4}")
    print(f"    dissolved COD units        {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}  {missing}")
    print(f"    polygons with no census    {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("join FAILED")

    renamed = [(v[0], v[1]) for v in pairs.values() if v[0] != v[1]]
    print(f"\n    {len(pairs) - len(renamed)}/{len(pairs)} names agree; {len(renamed)} "
          "differ by romanisation:")
    for a, b in sorted(renamed):
        print(f"      DOP {a!r:<16} OCHA {b!r}")

    # ---- the independent check: print order against COD's own p-code order ----
    order = {c: i for i, c in enumerate(sorted(v[2] for v in pairs.values()))}
    bad = []
    for gid in sorted(pairs):
        pos = int(gid.split("-")[1]) - 1
        if order[pairs[gid][2]] != pos:
            bad.append((pairs[gid][0], pos + 1, order[pairs[gid][2]] + 1))
    print(f"\n    the census's printed row order reproduces COD's ADM1_PCODE order on "
          f"{len(pairs) - len(bad)}/{len(pairs)} —\n    DOP's table position and OCHA's code "
          "attribute have different origins, so agreement is\n    evidence rather than a "
          "tautology, and it is what rules out a transposed row.")
    for nm, i, j in bad:
        print(f"      {nm!r}: printed at #{i}, coded at #{j}")
    if bad:
        raise SystemExit("the census row order is NOT p-code order")

    # ---- do the 15 tile cleanly? (§12, the Korea trap) ----
    dp = diss.to_crs(32646)
    tot = dp.geometry.area.sum() / 1e6
    union = dp.geometry.union_all().area / 1e6
    print(f"\n    sum of the 15 areas {tot:,.0f} km2 vs the area of their union "
          f"{union:,.0f} km2\n    difference {tot - union:,.1f} km2 — 0 means they tile "
          "with no overlap (§12, Korea).")
    if tot - union > 1.0:
        raise SystemExit(f"the dissolved states overlap by {tot - union:,.1f} km2 -- a "
                         "centroid sjoin would assign hexes by row order")

    # `unit` is MIMU's own code for the thing the census counted — MMR111 for Bago and
    # MMR222 for Shan, which is why it is taken from mm.py's bridge and not from COD.
    out = diss.merge(
        pd.DataFrame({"base": [v[1] for v in pairs.values()],
                      "unit": [v[3] for v in pairs.values()],
                      "name": [v[0] for v in pairs.values()]}),
        on="base", how="left")
    if out["unit"].isna().any():
        raise SystemExit("a dissolved polygon came out of the join with no census code")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "min_pcode", "geometry"]].to_file(
        OUT, layer="states", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[g][3] for g in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
