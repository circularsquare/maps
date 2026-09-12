"""Vanuatu — boundaries for the 66 area councils and urban municipalities.

Writes data/geo/vu/vu_councils.gpkg and data/geo/vu/vu_lookup.csv.

OCHA COD-AB Vanuatu (`cod-ab-vut`, reviewed 31 October 2024), the **shapefile** bundle rather
than the geodatabase on §12's Chile rule.

**COD's ADM2 IS THE CENSUS'S OWN TIER, TO THE UNIT.** 66 polygons against Table 3.5's 64 rural
area councils plus Port Vila and Luganville, and **every name matches outright** — 66 of 66 on
a folded comparison, nothing spare on either side. That is rare enough here to be worth saying
plainly: no aliases, no renumbering, no manual pairs. §9bd's Fiji had a shared office id to
join on; Vanuatu has no published id at all and does not need one, because the two name sets
are identical.

**AND BECAUSE THE JOIN IS FREE, IT GETS THE WITNESS WITH ACTUAL POWER.** A name join that
matches everything is exactly the one nobody checks, and the failure it cannot see is the
wrong twin — two different places sharing a name, paired confidently and silently. So this
file asserts something the names cannot fake: **COD independently files each ADM2 under an
ADM1 province, and the census independently groups its area councils under provinces by the
order they are printed in Table 3.5.** Those two assignments come from different
organisations and must agree on all 64 rural councils. They do.

**NO ANTIMERIDIAN PROBLEM, AND THAT IS ASSERTED RATHER THAN ASSUMED.** Vanuatu runs 166.5°E to
170.2°E — 3.7 degrees wide, nowhere near 180° — so unlike Fiji (§9bd §7) plain EPSG:4326 is
correct here and `vu_grid.py` needs none of that machinery. The span is checked anyway,
because a torn polygon is silent.

Usage:
    python sources/vu_geo.py --fetch    one ~3.3 MB zip from HDX
    python sources/vu_geo.py            rebuild from data/raw/vu/
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "vu")
OUT_DIR = os.path.join(ROOT, "data", "geo", "vu")
OUT = os.path.join(OUT_DIR, "vu_councils.gpkg")
LOOKUP = os.path.join(OUT_DIR, "vu_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "vu.csv")

ZIP_URL = ("https://data.humdata.org/dataset/67887590-51b0-4d98-bc12-666ebedf7704/"
           "resource/7606ac81-d36f-48f4-bdad-dadf34ebe26a/download/"
           "vut_admin_boundaries.shp.zip")
ZIP_NAME = "vut_admin_boundaries.shp.zip"
SHP = "vut_admin2.shp"
EXPECTED = 66

# Vanuatu is 3.7 degrees wide. Anything approaching this means a torn polygon.
MAX_SPAN_DEG = 20.0

# The census prints provinces in capitals; COD spells them normally.
PROV_FOLD = {"torba": "TORBA", "sanma": "SANMA", "penama": "PENAMA",
             "malampa": "MALAMPA", "shefa": "SHEFA", "tafea": "TAFEA"}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    print(f"  {os.path.getsize(dest):,} bytes")
    with open(dest, "rb") as fh:
        if fh.read(2) != b"PK":
            raise SystemExit(f"{dest} is not a zip -- HDX served something else")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zpath):
        raise SystemExit(f"missing {zpath} -- run with --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/vu.py first")

    g = gpd.read_file("zip://" + zpath + "!" + SHP, engine="fiona")
    if len(g) != EXPECTED:
        raise SystemExit(f"COD ADM2 has {len(g)} features, expected {EXPECTED}")
    g = g.to_crs("EPSG:4326")
    b = g.total_bounds
    print(f"COD-AB ADM2: {len(g)} area councils, crs={g.crs}")
    print(f"  bounds lon {b[0]:.3f}..{b[2]:.3f}, lat {b[1]:.3f}..{b[3]:.3f} "
          f"— {b[2] - b[0]:.2f}° wide")
    if b[2] - b[0] > MAX_SPAN_DEG:
        raise SystemExit(f"Vanuatu came out {b[2] - b[0]:.1f}° wide; it is 3.7°. A polygon "
                         "is torn across the antimeridian -- see sources/fj_geo.py")
    widest = max(r.geometry.bounds[2] - r.geometry.bounds[0] for r in g.itertuples())
    print(f"  and no single polygon is wider than {widest:.2f}°: nothing is torn")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "area_council"]
    cen_name = dict(zip(df["geo_id"], df["geo_name"]))
    if len(cen_name) != EXPECTED:
        raise SystemExit(f"{NORM} has {len(cen_name)} units, expected {EXPECTED}")
    # the province each council is filed under, from Table 3.5's printed order
    cen_prov = {}
    for gid, note in zip(df["geo_id"], df["note"]):
        m = re.search(r"province=([A-Z]+)", str(note))
        if m:
            cen_prov[gid] = m.group(1)

    cod_name = {fold(r.adm2_name): str(r.adm2_name).strip() for r in g.itertuples()}
    cod_prov = {fold(r.adm2_name): str(r.adm1_name).strip() for r in g.itertuples()}
    cen_fold = {fold(k): k for k in cen_name}

    missing = sorted(set(cen_fold) - set(cod_name))
    spare = sorted(set(cod_name) - set(cen_fold))
    print("\n  the join, both ways (§12) — ON NAME, because neither side publishes an id:")
    print(f"    census units               {len(cen_fold):>4}")
    print(f"    COD polygons               {len(cod_name):>4}")
    print(f"    matched                    {len(set(cen_fold) & set(cod_name)):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for k in missing[:6]:
        print(f"      no polygon: {cen_fold[k]!r}")
    for k in spare[:6]:
        print(f"      no census : {cod_name[k]!r}")
    if missing or spare:
        raise SystemExit("join FAILED")
    if len(cen_fold) != EXPECTED or len(cod_name) != EXPECTED:
        raise SystemExit("a folded name collided -- two units share one key")

    # ---- the witness: two organisations' province assignments must agree ----
    #
    # The names alone cannot catch a wrong twin. COD files each ADM2 under an ADM1; the
    # census files each area council under a province by where it prints in Table 3.5.
    # Independent, and they have to agree.
    print("\n    witness — COD's ADM1 against the province Table 3.5 prints each council "
          "under:")
    disagree = []
    checked = 0
    for k, gid in cen_fold.items():
        if gid not in cen_prov:          # Port Vila and Luganville are urban, not in a
            continue                     # province block of the table
        checked += 1
        if PROV_FOLD.get(fold(cod_prov[k])) != cen_prov[gid]:
            disagree.append((cen_name[gid], cen_prov[gid], cod_prov[k]))
    print(f"      agree on {checked - len(disagree)}/{checked} rural area councils "
          f"({len(disagree)} disagreements)")
    for nm, a, b_ in disagree[:8]:
        print(f"        MISMATCH {nm!r}: census {a} vs COD {b_}")
    if disagree:
        raise SystemExit(f"{len(disagree)} councils pair on name while the two sources file "
                         "them under different provinces -- §12's shape-2 failure, and "
                         "exactly the wrong-twin case a name join cannot see. Resolve by "
                         "hand.")
    for nm in ("Port Vila", "Luganville"):
        print(f"      {nm} is urban and outside the province blocks; COD files it under "
              f"{cod_prov[fold(nm)]}")

    out = g[["adm2_name", "adm2_pcode", "adm1_name", "geometry"]].copy()
    out["unit"] = out["adm2_pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    out["name"] = out["adm2_name"].map(lambda n: cen_name[cen_fold[fold(n)]])
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no census name")
    if out["unit"].duplicated().any():
        raise SystemExit("COD ADM2 has duplicate pcodes")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "adm1_name", "geometry"]].to_file(
        OUT, layer="councils", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    pairs = sorted(((cen_fold[fold(n)], u) for n, u in zip(out["adm2_name"], out["unit"])),
                   key=lambda t: t[1])
    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit"])
        w.writerows(pairs)
    print(f"wrote {LOOKUP} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
