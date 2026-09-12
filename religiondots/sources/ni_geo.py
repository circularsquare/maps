"""Nicaragua — boundaries for the 153 municipios.

Writes data/geo/ni/ni_municipios.gpkg and data/geo/ni/ni_lookup.csv.

OCHA COD-AB Nicaragua (`cod-ab-nic`, version 02, valid 2023-11-27), the **shapefile** bundle
rather than the geodatabase on §12's Chile rule — GDAL's OpenFileGDB driver has been seen to
open a .gdb, list its layers, report the right CRS and return ZERO features while raising
nothing. Read with `engine="fiona"`, because pyogrio is geopandas' default when installed and
is the engine that has silently returned zero. The feature count is asserted either way.

**THE CODE JOIN LOOKS PERFECT, IS AVAILABLE, AND IS WRONG. DO NOT RESTORE IT.** COD's
`adm2_pcode` is `NI` + a four-digit municipality code in INIDE's own format, and 145 of the
153 match INIDE's 2005 codes exactly. That is the §12 shape-2 trap — *a confident wrong
pairing* — in its most inviting form, because the eight that do not match are visible and the
**five that DO match while naming different places are not**:

    INIDE 9105 = Waspám              COD NI9105 = Mulukukú
    INIDE 6545 = El Coral            COD NI6545 = San Francisco de Cuapa
    INIDE 0515 = El Jícaro           COD NI0515 = Jícaro                    (same place)
    INIDE 2020 = San Juan de Río Coco COD NI2020 = San Juan del Río Coco    (same place)
    INIDE 5525 = Municipio de Managua COD NI5525 = Managua                  (same place)

**The first line is the whole argument for this file.** Waspám is the Río Coco Miskito
municipality: 38,926 people, 43.6% Moravian, one of the four units that carry the category
Nicaragua is being drawn for. Mulukukú is an interior mining-triangle municipality created in
2005, 0.3% Moravian. A code join sends Waspám's Moravians inland and puts Mulukukú's mestizo
Catholic profile on the Honduran border — and **every total in `ni.py` still reconciles**,
because a permutation of units preserves every sum. Ten municipalities were renumbered
between the 2005 census and COD's 2023 vintage, mostly around municipalities created in the
2000s, and the renumbering cascaded.

**SO THE JOIN IS ON NAME, AND THE CODE IS DEMOTED TO EVIDENCE.** Names are unique on both
sides — no duplicates at all — and 150 of 153 fold to the same string; the three that do not
are spelling, listed in `ALIASES` with the reason. The independent confirmation is the
**department**: all ten renumberings stay inside the same department, so a name pairing that
also agrees on the two-digit department prefix is two signals rather than one. That is checked
on all 153.

**AND THE DATA ITSELF IS THE THIRD WITNESS.** Moravian Nicaragua is the Caribbean coast and
nothing else. After the join this file checks that the Moravian-heavy municipalities really
are the eastern ones, using COD's own centroid longitudes — a test of the pairing that uses
neither name nor code, and the one that would catch a systematic east/west swap.

**ADM2 IS THE FLOOR AND THE COUNTS GO FINER.** INIDE's REDATAM serves religion at 2,579
comarcas (`sources/ni.py`), and this bundle stops at 153 municipios: OCHA's own description
says *"structured into 2 levels"* and geoBoundaries 404s on NIC ADM3. The counting geography
is limited by the BOUNDARIES here, which is the reverse of the usual situation.

Usage:
    python sources/ni_geo.py --fetch    one ~27 MB zip from HDX
    python sources/ni_geo.py            rebuild from data/raw/ni/
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
RAW = os.path.join(ROOT, "data", "raw", "ni")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ni")
OUT = os.path.join(OUT_DIR, "ni_municipios.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ni_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ni.csv")

ZIP_URL = ("https://data.humdata.org/dataset/98e9bcfe-6565-4372-8422-f57d5b209448/"
           "resource/54c55d85-dc48-4b2f-acb9-5a19b16a59f0/download/"
           "nic_admin_boundaries.shp.zip")
ZIP_NAME = "nic_admin_boundaries.shp.zip"
EXPECTED = 153

# INIDE's spelling -> COD's, for the three that are the same place written differently.
# Each is a rendering difference and none of them is a judgement call:
ALIASES = {
    # INIDE keeps the article, COD drops it. El Jícaro, Nueva Segovia.
    "El Jícaro": "Jícaro",
    # `de` vs `del`; the river is the Río Coco, both name the same Madriz municipality.
    "San Juan de Río Coco": "San Juan del Río Coco",
    # INIDE distinguishes the municipality from the department, which are both `Managua`.
    "Municipio de Managua": "Managua",
}

# Moravian Nicaragua is the Caribbean coast. Used only as a check on the join, never to
# place anything: the top Moravian municipalities must come out east of this meridian.
MORAVIAN_EAST_OF = -85.0
MORAVIAN_TOP_N = 6


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


def _read_adm2():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()]
    # anchored so the edge-matched `nic_admin2_em.shp` cannot match
    shp = [n for n in names if re.search(r"adm(?:in)?2\.shp$", n, re.I)]
    if len(shp) != 1:
        raise SystemExit(f"expected one admin2 shapefile in the bundle, found {shp}")
    g = gpd.read_file(f"zip://{src}!{shp[0]}", engine="fiona")

    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED:
        raise SystemExit(f"{shp[0]}: {len(g)} features, expected {EXPECTED} municipios")
    cols = {c.lower(): c for c in g.columns}
    for want in ("adm2_name", "adm2_pcode", "adm1_pcode", "center_lon"):
        if want not in cols:
            raise SystemExit(f"no {want} column in {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g, cols


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, cols = _read_adm2()
    print(f"COD ADM2: {len(g)} polygons, crs={g.crs}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/ni.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "municipio"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census municipios, expected {EXPECTED}")

    pcode = g[cols["adm2_pcode"]].astype(str).str.strip()
    cname = g[cols["adm2_name"]].astype(str).str.strip()
    a1 = g[cols["adm1_pcode"]].astype(str).str.strip()
    lon = g[cols["center_lon"]].astype(float)

    # every COD pcode is NI + a code whose first two digits are its own department's
    bad = [(p, d) for p, d in zip(pcode, a1) if not p.startswith("NI") or p[2:4] != d[2:]]
    if bad:
        raise SystemExit(f"COD pcodes not under their own department: {bad[:5]}")

    # ---- names are the join key, so they must be unique on both sides ----
    poly = {}
    for p, nm, ln in zip(pcode, cname, lon):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD name {nm!r} appears twice -- the name join is unsafe")
        poly[k] = (nm, p, float(ln))
    census = {}
    for gid, nm in zip(cen["geo_id"], cen["geo_name"]):
        k = fold(ALIASES.get(nm, nm))
        if k in census:
            raise SystemExit(f"INIDE name {nm!r} appears twice -- the name join is unsafe")
        census[k] = (gid, nm)
    print(f"  names are unique on both sides: {len(census)} INIDE, {len(poly)} COD")

    pairs, missing = {}, []
    for k, (gid, nm) in census.items():
        if k in poly:
            pairs[gid] = (nm, poly[k][1], poly[k][0], poly[k][2])
        else:
            missing.append((gid, nm))
    used = {v[1] for v in pairs.values()}
    spare = [(nm, p) for nm, p, _ in poly.values() if p not in used]

    print("\n  the join, both ways (§12) — ON NAME, see the module docstring:")
    print(f"    census municipios          {len(cen):>4}")
    print(f"    COD polygons               {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for gid, nm in missing[:10]:
        print(f"      no polygon: {gid} {nm!r}")
    for nm, p in spare[:10]:
        print(f"      no census : {p} {nm!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- witness 1: the department must agree, on every pair ----
    dep_bad = [(gid, nm, gid[:2], p[2:4]) for gid, (nm, p, _, _) in pairs.items()
               if gid[:2] != p[2:4]]
    print(f"\n    witness 1 — INIDE's department prefix agrees with COD's on "
          f"{len(pairs) - len(dep_bad)}/{len(pairs)}")
    for gid, nm, a, b in dep_bad[:8]:
        print(f"      {nm!r}: INIDE dept {a}, COD dept {b}")
    if dep_bad:
        raise SystemExit("a name pairing crosses a department boundary -- two municipalities "
                         "share a name across departments, or the pairing is wrong")

    # ---- witness 2: what the code join WOULD have done, reported not assumed ----
    by_code = {p[2:]: nm for nm, p, _ in poly.values()}
    renum = sorted((gid, nm, by_code.get(gid)) for gid, (nm, _, _, _) in pairs.items()
                   if by_code.get(gid) is not None and fold(by_code[gid]) != fold(nm)
                   and fold(ALIASES.get(nm, nm)) != fold(by_code[gid]))
    absent = sorted(gid for gid in pairs if gid not in by_code)
    print(f"\n    witness 2 — the code join is NOT used. Joining on INIDE's own code would "
          f"have\n    silently mispaired {len(renum)} municipalities and dropped "
          f"{len(absent)}:")
    for gid, nm, other in renum:
        print(f"      {gid}: INIDE {nm!r} -> COD's NI{gid} is {other!r}")
    if not renum:
        raise SystemExit(
            "the code join no longer mispairs anything. Either COD has re-aligned its "
            "pcodes to the 2005 census (good news, and this file should be simplified "
            "deliberately rather than by accident), or the census read has changed. "
            "STOP and check which -- do not delete this assertion.")

    # ---- witness 3: the data's own geography. Moravians are the Caribbean coast. ----
    tot = (df[df["source_category"] == "Total"].set_index("geo_id")["count"]).to_dict()
    mor = (df[df["source_category"] == "Morava"].set_index("geo_id")["count"]).to_dict()
    share = {gid: mor[gid] / tot[gid] for gid in pairs if tot.get(gid)}
    top = sorted(share, key=share.get, reverse=True)[:MORAVIAN_TOP_N]
    print(f"\n    witness 3 — the {MORAVIAN_TOP_N} most Moravian municipios, against COD's "
          f"own centroid longitude:")
    west = []
    for gid in top:
        nm, p, _, ln = pairs[gid]
        flag = "" if ln > MORAVIAN_EAST_OF else "   <-- WEST"
        if ln <= MORAVIAN_EAST_OF:
            west.append((nm, ln))
        print(f"      {nm:<28} {100 * share[gid]:5.1f}% Moravian   lon {ln:8.3f}{flag}")
    if west:
        raise SystemExit(
            f"{len(west)} of the most Moravian municipalities sit west of "
            f"{MORAVIAN_EAST_OF}: {west}. The Moravian Church in Nicaragua is the Caribbean "
            "coast; if its dots are landing in the Pacific half the join is permuted.")
    print("      the Moravian coast lands on the Caribbean side, which neither the names "
          "nor the\n      codes were used to establish.")

    out = g[[cols["adm2_name"], cols["adm2_pcode"], "geometry"]].rename(
        columns={cols["adm2_name"]: "name", cols["adm2_pcode"]: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    inide = {v[1]: v[0] for v in pairs.values()}
    out["name"] = out["unit"].map(inide)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no INIDE name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="municipios", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[k][1] for k in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
