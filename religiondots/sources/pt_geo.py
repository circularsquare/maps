"""Portugal — boundaries for the 3,092 freguesias.

Writes data/geo/pt/pt_freguesias.gpkg.

**THE CHEAPEST BOUNDARY JOIN IN THE PROJECT, AND IT NEEDED NO DOWNLOAD.** GISCO LAU 2021 has
been on disk since Poland (§9e). Portugal's LAU *is* the freguesia — 3,092 polygons, six-digit
`LAU_ID`, and those digits are character-for-character INE's own `geocod`. The join is:

    census freguesias      3,092
    GISCO PT polygons      3,092
    unmatched, either way      0

No name resolution, no concordance, no re-cutting, no Kontur. `sources.md` §9e wrote that
"the GISCO LAU file is the boundary answer for most of Europe"; Portugal is the case where
that is true without a single qualification, and it is worth recording that such a case
exists, because the last ten countries have all needed work here.

**The names are checked anyway** (§12: a count match is not a join). INE's `geodsg` and
GISCO's `LAU_NAME` agree on every unit once case and accents are folded, which is what
promotes "the codes line up" to "the codes mean the same thing".

WHAT IS *NOT* SOLVED, and it is the reason a `place_weight` may follow. The freguesia is both
the counting unit and the placement unit, so dots spread uniformly inside it (§8.2). That is
harmless over most of the country — 3,092 units for 10.3 million people, a median of about
20 km² — but the Alentejo freguesias are enormous and nearly empty (Alcácer do Sal's are
several hundred km² at single-digit people per km²) and uniform scatter there paints cork
forest. Kontur would fix it exactly as it did for Kenya and Ethiopia. Not done yet; the
freguesia tier is already fine enough that this is an improvement rather than a correction.

Usage:
    python sources/pt_geo.py        (no --fetch: GISCO LAU 2021 is already local)
"""

import os
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("OMP_NUM_THREADS", "2")   # reference/scipy_eats_all_cores

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO_LAU = os.path.join(ROOT, "data", "geo", "lau2021")
SHP = os.path.join(GEO_LAU, "shp4326", "LAU_RG_01M_2021_4326.shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "pt")
OUT = os.path.join(OUT_DIR, "pt_freguesias.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "pt.csv")

EXPECTED = 3_092


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    if not os.path.exists(SHP):
        raise SystemExit(f"missing {SHP} -- the GISCO LAU 2021 shapefile arrived with "
                         "Poland (sources.md §9e); unzip LAU_RG_01M_2021_4326.shp.zip "
                         f"into {os.path.join(GEO_LAU, 'shp4326')}")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/pt.py first")

    g = gpd.read_file(SHP, columns=["CNTR_CODE", "LAU_ID", "LAU_NAME",
                                    "POP_2021", "AREA_KM2"])
    pt = g[g["CNTR_CODE"] == "PT"].copy()
    pt["kod"] = pt["LAU_ID"].astype(str).str.strip()
    print(f"  GISCO LAU 2021, CNTR_CODE == 'PT': {len(pt):,} polygons")
    if len(pt) != EXPECTED:
        raise SystemExit(f"{len(pt)} PT polygons, expected {EXPECTED}")
    if pt["kod"].nunique() != len(pt):
        raise SystemExit("duplicate LAU_ID among the Portuguese polygons")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    cen = (df[df["geo_level"] == "freguesia"][["geo_id", "geo_name"]]
           .drop_duplicates("geo_id").set_index("geo_id")["geo_name"].to_dict())

    geo_keys, cen_keys = set(pt["kod"]), set(cen)
    missing, extra = cen_keys - geo_keys, geo_keys - cen_keys
    print("\n  the join, both ways (§12 — a count match is not a join):")
    print(f"    matched                        {len(cen_keys & geo_keys):>6,}")
    print(f"    census freguesias with no polygon {len(missing):>3,}")
    print(f"    polygons with no census freguesia {len(extra):>3,}")
    if missing or extra:
        raise SystemExit(f"the join is not exact: {sorted(missing)[:5]} / "
                         f"{sorted(extra)[:5]}")
    print("    OK  the two sets are identical, with no derivation at all")

    # independent verification of the key: the NAMES must agree (§12)
    gname = dict(zip(pt["kod"], pt["LAU_NAME"]))
    dis = [k for k in cen_keys if fold(cen[k]) != fold(gname[k])]
    print(f"    {'OK ' if not dis else 'BAD'} names agree on "
          f"{len(cen_keys) - len(dis):,} of {len(cen_keys):,} units")
    for k in dis[:10]:
        print(f"        {k}: INE {cen[k]!r} vs GISCO {gname[k]!r}")
    if dis:
        raise SystemExit("name verification FAILED — the codes match and the units do not")

    out = pt[["kod", "LAU_NAME", "POP_2021", "AREA_KM2", "geometry"]].rename(
        columns={"LAU_NAME": "name", "POP_2021": "pop_2021", "AREA_KM2": "area_km2"})
    out["level"] = "freguesia"

    # Reported, never used for allocation: GISCO's POP_2021 is its own estimate and is not
    # the census figure (10,562,178 against INE's 10,343,066). §9i's rule — two population
    # numbers that measure different things must not be asserted equal.
    print(f"\n  area: median {out['area_km2'].median():.1f} km², "
          f"max {out['area_km2'].max():.0f} km² "
          f"({out.loc[out['area_km2'].idxmax(), 'name']})")
    print(f"  GISCO POP_2021 sums to {out['pop_2021'].sum():,.0f}; INE's census total is "
          "10,343,066 — different measurements, reported only")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="freguesias", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} polygons)")


if __name__ == "__main__":
    main()
