"""Benin — boundaries for the 77 communes.

Writes data/geo/bj/bj_communes.gpkg and data/geo/bj/bj_lookup.csv.

OCHA COD-AB Benin, from HDX, the **shapefile** bundle rather than the geodatabase on §12's
Chile rule — GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers, report
the right CRS and return ZERO features while raising nothing. The feature count is asserted
after the read either way, and the read uses `engine="fiona"` because pyogrio is geopandas'
default when installed and is the engine that has silently returned zero.

**COD'S ADM2 IS THE CENSUS'S COMMUNE TIER EXACTLY.** 77 polygons for 77 columns, and the
per-department split is 6/9/8/8/6/6/4/1/6/9/5/9, which is the split the twelve booklets
print. Both are asserted before the join, so a boundary file cut to some other description
of Benin fails on its own terms.

**THE BUNDLE STOPS AT ADM2 AND THAT IS WHY COTONOU IS ONE POLYGON** — see `sources/bj.md`
§5. The census publishes religion for Cotonou's thirteen arrondissements, `sources/bj.py`
parses them, and nothing here can draw them: COD ships no ADM3 for Benin at all, and
geoBoundaries' ADM3 is an OpenStreetMap layer whose thirteen Cotonou polygons could not be
verified against the census's own arrondissement populations. That measurement is in
`bj.md`; the decision is to keep the coarse unit rather than take a fine one on trust.

THE JOIN IS BY NAME, AND **A MATCHING ORDER ON 71 OF 77 IS NOT AN ORDER MATCH.** This module
was first written to assert that INStaD's printed column order reproduces COD's
`adm2_pcode` order — both are alphabetical within a department, so the p-code would have
been a free independent key. It very nearly is: 71 of 77 agree, and the six that do not are
**three adjacent transpositions**, each of which is a sorting convention rather than a
disagreement about geography.

| department | INStaD prints | COD numbers | why |
|---|---|---|---|
| Atacora | Cobly, Kérou | Kérou, Kobli | COD sorts under its OWN spelling, and `Kobli` follows `Kérou` where `Cobly` precedes it |
| Zou | Zagnanado, Za-Kpota | Za-Kpota, Zagnanado | the hyphen. `Za-Kpota` sorts first if `-` is a character and second if it is not |
| Donga | Copargo, Djougou | Djougou, Copargo | no explanation; COD's Donga is simply not alphabetical |

**Six rows in 77 is exactly the size of error a stale vintage or a shifted block also
produces**, so had the assertion been kept and then loosened until it passed, it would have
stopped detecting anything. It is replaced by two things that are true:

  * the pairing is by NAME, 1:1 within a department, and the p-code is read off the polygon
    that name selected — so nothing depends on the orders agreeing;
  * what IS asserted is that no commune's rank moves by more than one place. A transposition
    of neighbours survives that; a shifted block, a missing unit or a boundary file from
    another vintage does not.

The quantitative check on the pairing lives in `sources/bj_grid.py`, where Kontur's modelled
population per commune is compared with the census's own — a quantity the join does not
determine, which is the only kind of check that catches a confident wrong pairing (§12).

**A SECOND FREE CHECK: THE PARENT.** COD carries `adm1_name`, and on the census side a
commune's department comes only from which booklet it was printed in. Those two are
genuinely independent, so their agreement on all 77 confirms the join AND the twelve
separate parses at once — Ghana's rule, and it costs nothing.

**FIVE OF THE 77 NAMES DISAGREE AND NOT ONE OF THEM IS AN ACCENT.** INStaD and COD romanise
Benin's languages differently, and the differences are the consonants:

| INStaD | COD | what differs |
|---|---|---|
| `Boukoumbé` | `Boukombe` | `ou` for `o` |
| `Cobly` | `Kobli` | `C` for `K`, `y` for `i` |
| `Torri-Bossito` | `Tori-Bossito` | a doubled `r` |
| `Akpro-Missérété` | `Akpo-Misserete` | an `r` that is simply absent |
| `Dassa` | `Dassa-Zoume` | the census abbreviates the name |

**No alias table is written, because a frozen list of five renames goes stale in silence at
the next release (§12).** Three tools in order, each refusing rather than guessing:

1. an exact fold, which takes 72;
2. a **transliteration fold applied only INSIDE one department** — `ou`→`o`, `c`→`k`,
   `y`→`i`, doubled letters collapsed — which is far too aggressive to be a national key
   and is safe within a parent of at most nine names, and which must match 1:1 or the run
   stops. It takes Boukoumbé, Cobly and Torri-Bossito;
3. **elimination**, for Akpro-Missérété and Dassa, where no fold can help. Each is the only
   name left in its department on either side, so the pairing is a deduction and not a
   guess, and two leftovers anywhere would refuse (§12, Romania).

**THE JOIN IS SCOPED BY THE DEPARTMENT NAME, NOT THE P-CODE**, and that is deliberate:
scoping it by `adm2_pcode` would make check 1 below circular. The department on the census
side is the booklet the commune was printed in; on COD's side it is `adm1_name`. Nothing
about the p-code is used to make the join, so the p-code is free to test it afterwards.

Usage:
    python sources/bj_geo.py --fetch    one ~200 KB zip from HDX
    python sources/bj_geo.py            rebuild from data/raw/bj/
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
RAW = os.path.join(ROOT, "data", "raw", "bj")
OUT_DIR = os.path.join(ROOT, "data", "geo", "bj")
OUT = os.path.join(OUT_DIR, "bj_communes.gpkg")
LOOKUP = os.path.join(OUT_DIR, "bj_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "bj.csv")

ZIP_URL = ("https://data.humdata.org/dataset/494229d9-eee0-4872-8864-2baf98691554/"
           "resource/4568a0c2-13c4-4126-af4c-29ab7a66868e/download/"
           "ben_admin_boundaries.shp.zip")
ZIP_NAME = "ben_admin_boundaries.shp.zip"

EXPECTED = 77
# Communes per department, in BJ01..BJ12 order. The same split the booklets print.
PER_DEPARTMENT = {"BJ01": 6, "BJ02": 9, "BJ03": 8, "BJ04": 8, "BJ05": 6, "BJ06": 6,
                  "BJ07": 4, "BJ08": 1, "BJ09": 6, "BJ10": 9, "BJ11": 5, "BJ12": 9}


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
    # §5a: HDX answers the un-redirected URL with a 302 and a small HTML body, which is a
    # perfectly good HTTP 200 to a client that does not follow it.
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def loose(s):
    """A transliteration fold, safe ONLY inside one department (§12, Sri Lanka).

    `ou`/`o`, `c`/`k` and `y`/`i` are the three places INStaD and COD disagree about how to
    write Benin's languages, and a doubled letter is never contrastive in either spelling.
    Aggressive enough to be a bad national key and fine across at most nine names.
    """
    k = fold(s).replace("ou", "o").replace("c", "k").replace("y", "i")
    return re.sub(r"(.)\1+", r"\1", k)


def _match_within(census, poly, dept):
    """Pair one department's communes to its polygons: exact fold, loose fold, elimination."""
    pairs = {}
    left_c = dict(census)                       # geo_id -> printed name
    left_p = dict(poly)                         # pcode   -> COD name

    for keyfn, label in ((fold, "fold"), (loose, "transliteration fold")):
        by_p = {}
        for pcode, nm in left_p.items():
            by_p.setdefault(keyfn(nm), []).append(pcode)
        by_c = {}
        for gid, nm in left_c.items():
            by_c.setdefault(keyfn(nm), []).append(gid)
        for k, gids in list(by_c.items()):
            pcodes = by_p.get(k, [])
            if not pcodes:
                continue
            if len(gids) != 1 or len(pcodes) != 1:
                raise SystemExit(
                    f"{dept}: the {label} maps {[left_c[g] for g in gids]} onto "
                    f"{[left_p[p] for p in pcodes]} -- not 1:1, so it resolves a collision "
                    "instead of reporting it")
            pairs[gids[0]] = (left_c[gids[0]], pcodes[0], left_p[pcodes[0]], label)
            del left_c[gids[0]]
            del left_p[pcodes[0]]

    if len(left_c) > 1 or len(left_p) > 1:
        raise SystemExit(f"{dept}: {len(left_c)} census communes "
                         f"{list(left_c.values())} and {len(left_p)} polygons "
                         f"{list(left_p.values())} unmatched -- refusing to guess, since "
                         "elimination is only a deduction when one of each remains")
    if len(left_c) != len(left_p):
        raise SystemExit(f"{dept}: {list(left_c.values())} has no polygon and "
                         f"{list(left_p.values())} has no census row")
    for gid, pcode in zip(left_c, left_p):
        pairs[gid] = (left_c[gid], pcode, left_p[pcode], "elimination")
    return pairs


def _read_adm2():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")

    # COD names it `ben_admin2.shp`, not `ben_adm2.shp`, and the bundle also holds
    # admin0/1, capitals, lines and points. Match the one, from inside the zip.
    names = [i.filename for i in zipfile.ZipFile(src).infolist()]
    shp = [n for n in names if re.search(r"adm(?:in)?2\.shp$", n, re.I)]
    if len(shp) != 1:
        raise SystemExit(f"expected one admin2 shapefile in the bundle, found {shp}")
    g = gpd.read_file(f"zip://{src}!{shp[0]}", engine="fiona")

    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED:
        raise SystemExit(f"{shp[0]}: {len(g)} features, expected {EXPECTED} communes")
    cols = {c.upper(): c for c in g.columns}
    name_col = next((cols[k] for k in ("ADM2_NAME", "ADM2_EN") if k in cols), None)
    code_col = cols.get("ADM2_PCODE")
    dept_col = next((cols[k] for k in ("ADM1_NAME", "ADM1_EN") if k in cols), None)
    if not name_col or not code_col or not dept_col:
        raise SystemExit(f"no adm2 name/pcode or adm1 name column in {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g, name_col, code_col, dept_col


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, name_col, code_col, dept_col = _read_adm2()
    print(f"COD ADM2: {len(g)} polygons, crs={g.crs}")

    got = {}
    for cd in g[code_col]:
        got[str(cd)[:4]] = got.get(str(cd)[:4], 0) + 1
    if got != PER_DEPARTMENT:
        raise SystemExit(f"communes per department {got}, expected {PER_DEPARTMENT}")
    print(f"  per department {got}\n  -- matches the twelve booklets")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/bj.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "commune"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census communes, expected {EXPECTED}")
    # The department a commune was printed under, from bj.csv's own department rows.
    dep_name = dict(zip(df.loc[df["geo_level"] == "department", "geo_id"],
                        df.loc[df["geo_level"] == "department", "geo_name"]))
    if len(dep_name) != len(PER_DEPARTMENT):
        raise SystemExit(f"{len(dep_name)} departments in bj.csv, expected "
                         f"{len(PER_DEPARTMENT)}")

    # Group both sides by DEPARTMENT NAME — never by p-code, which has to stay free to
    # test the result.
    census_by, poly_by = {}, {}
    for gid, nm in zip(cen["geo_id"], cen["geo_name"]):
        census_by.setdefault(fold(dep_name[gid[:4]]), {})[gid] = nm
    for nm, pcode, dept in zip(g[name_col], g[code_col].astype(str).str.strip(),
                               g[dept_col]):
        poly_by.setdefault(fold(dept), {})[pcode] = nm

    if set(census_by) != set(poly_by):
        raise SystemExit(f"department names differ: census only "
                         f"{sorted(set(census_by) - set(poly_by))}, COD only "
                         f"{sorted(set(poly_by) - set(census_by))}")
    bad = [(d, len(census_by[d]), len(poly_by[d])) for d in census_by
           if len(census_by[d]) != len(poly_by[d])]
    if bad:
        raise SystemExit(f"communes per department differ: {bad}")

    pairs = {}
    for dept in sorted(census_by):
        pairs.update(_match_within(census_by[dept], poly_by[dept], dept))

    how = {}
    for _, _, _, label in pairs.values():
        how[label] = how.get(label, 0) + 1
    print("\n  the join, both ways (§12) — scoped by department name, never by p-code:")
    print(f"    census communes            {len(cen):>4}")
    print(f"    COD polygons               {len(g):>4}")
    print(f"    matched                    {len(pairs):>4}")
    for label in ("fold", "transliteration fold", "elimination"):
        print(f"      by {label:<22} {how.get(label, 0):>4}")
    if len(pairs) != EXPECTED:
        raise SystemExit(f"{len(pairs)} matched, expected {EXPECTED}")

    # ---- check 1: rank drift. Nothing above used the p-code, so it is free to test. ----
    moved, ranks = [], 0
    for dept in sorted(census_by):
        printed = list(census_by[dept])                       # print order
        coded = sorted(poly_by[dept])                         # p-code order
        for i, gid in enumerate(printed):
            j = coded.index(pairs[gid][1])
            ranks += 1
            if i != j:
                moved.append((dept, pairs[gid][0], pairs[gid][2], i, j))
    far = [m for m in moved if abs(m[3] - m[4]) > 1]
    print(f"\n    check 1 — printed column order vs COD's ADM2_PCODE order: "
          f"{ranks - len(moved)}/{ranks} identical,\n              {len(moved)} moved by one "
          f"place, {len(far)} moved further (which is what would fail)")
    for dept, nm, cod_nm, i, j in moved:
        print(f"      {dept}: INStaD prints {nm!r} at #{i + 1}, "
              f"COD codes {cod_nm!r} at #{j + 1}")
    if far:
        raise SystemExit(f"{len(far)} communes move more than one place between the printed "
                         f"order and the p-code order: {far[:5]} -- that is a shifted block "
                         "or a different vintage, not a sorting convention")

    print(f"    check 2 — every commune's department agrees between the booklet it was "
          f"printed in\n              and COD's ADM1_NAME on {len(pairs)}/{len(pairs)}, "
          "which is what scoped the join")

    for nm, pcode, cod_nm, label in sorted(pairs.values()):
        if nm != cod_nm:
            print(f"    name resolved by {label}: INStaD {nm!r} / COD {cod_nm!r}")

    out = g[[name_col, code_col, dept_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode", dept_col: "department"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile) — and
    # undo the header wrap, since `Torri- Bossito` is a line break and not a name.
    instad = {v[1]: " ".join(v[0].replace("- ", "-").split()) for v in pairs.values()}
    out["name"] = out["unit"].map(instad)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no INStaD name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "department", "geometry"]].to_file(
        OUT, layer="communes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    # bj.py's geo_id is positional and COD's pcode is the drawn unit; they are NOT the same
    # string, so this lookup carries real information rather than being an identity map.
    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[g][1] for g in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
