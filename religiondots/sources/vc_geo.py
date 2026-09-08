"""Saint Vincent — the 221 enumeration districts. No placement grid, and that is a finding.

Writes:
    data/geo/vc/vc_eds.gpkg     221 polygons keyed as vc.py keys them
                                (serves as BOTH `units` and `place`; §8.2 uniform placement)

Usage:
    python sources/vc_geo.py    # no --fetch: the boundaries ship inside vc.py's geodatabase

**THE JOIN IS FREE, AS EVERYWHERE IN THE USCB SERIES.** `GEO_MATCH` keys the counts to the
boundaries by construction: **221 table keys, 221 geo keys, 221 matched, 0 unmatched.** Sixth
country in the series, sixth exact join, and the only download this module needs is the one
`sources/vc.py` already made.

**THE BOUNDARY PROVENANCE IS THE ODDEST IN THE PROJECT.** USCB's Metadata sheet says the
enumeration-district geometry was *"downloaded from the Feasibility Study & Environmental
Assessment for Georgetown Coastal Defense, Figure 3: Census Districts in Saint Vincent"* — a
coastal engineering report. The SVG Statistical Office does not publish the boundaries, so the
only public copy is a figure in a document about sea defences, digitised by USCB. Total area
comes to **384.6 km² against the country's 389** (98.9%), a complete cover with ordinary
generalisation, and the exact join is what actually vouches for it.

---

## KONTUR WAS BUILT, MEASURED AND REMOVED — a grid has a resolution floor

Every country drawn here since Kenya has been placed on Kontur's 400 m H3 grid, so it was the
default for this one too. **It was built, and it is the wrong tool for Saint Vincent**, which
is worth recording because the reason generalises rather than being about this country.

A Kontur r8 hex is about **0.16 km²**. The median enumeration district here is **0.66 km²**.
So the grid is only about four cells across the *typical counting unit* — and once
hex-to-unit assignment is done by centre point, a great many units get nothing at all:

| measured, 2026-09-07 | |
|---|---|
| Kontur hexes over the whole country | **509** (417 after assignment) |
| populated EDs with **no** hex | **78 of 219 — 36%** |
| per-ED Kontur/census ratio | **p10 0.00, median 0.45, p90 2.68** |
| national Kontur/census ratio | 0.88x |

**A weighting that is absent for a third of the units and scatters over an order of magnitude
for the rest is not a weighting, it is noise.** Using it would mean two different placement
rules applied essentially at random across the island — Kontur where a hex happened to land,
equal-shares where it did not — with no reason to think the first is better than the second.

**So placement is §8.2's uniform-within-unit, and the units are already fine enough for that
to be right.** A median ED is 0.66 km², which is a few hundred metres across; scattering a
handful of dots uniformly inside one is honest at any zoom this map reaches.

**What that costs, stated rather than hidden.** The ED areas are extremely skewed — the
largest is **44 km²** against a median of 0.66, a factor of 67 — and the big ones are the
interior of Saint Vincent, which is the Soufrière massif and close to uninhabited. Uniform
scatter there puts dots on a volcano. The bound on the damage is small: at 1:1,000 an ED of a
thousand people draws one dot, so this is a question of one or two dots sitting up a mountain,
not of a misread population distribution.

**The general rule, which is new to this project: a population grid must be finer than the
counting tier to be worth anything, and Kontur r8 stops paying at roughly 1 km² per unit.**
Every earlier customer was far above that line — Kenya's counties, Ethiopia's woredas,
Bosnia's municipalities at a median 304 km² — so the floor had never been reached. Do not
reach for Kontur reflexively on a fine-geography country; measure the hexes-per-unit first,
and if a large share of units would come back empty, uniform placement is not a fallback but
the better answer.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "vc")
RAW = os.path.join(ROOT, "data", "raw", "vc")
NORM = os.path.join(ROOT, "data", "normalized", "vc.csv")

GDB = os.path.join(RAW, "Saint_Vincent_and_The_Grenadines.gdb")
LAYER_ADM2 = "VC_GEOG_ADM2_2012_uscb_202109"

UNITS_OUT = os.path.join(GEO, "vc_eds.gpkg")

EXPECTED_UNITS = 221
UTM = 32620                      # UTM 20N, metres


def build_units():
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    if not os.path.isdir(GDB):
        raise SystemExit(f"missing {GDB} -- run `python sources/vc.py --fetch` first; the "
                         "boundaries ship inside the same geodatabase as the counts")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/vc.py first")

    layers = [n for n, _ in pyogrio.list_layers(GDB)]
    if LAYER_ADM2 not in layers:
        raise SystemExit(f"{LAYER_ADM2} not in the geodatabase; layers are {layers}")

    g = gpd.read_file(GDB, layer=LAYER_ADM2)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    print(f"  {LAYER_ADM2}: {len(g)} polygons")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} polygons, got {len(g)}")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    ed = (df[df["geo_level"] == "ed"][["geo_id", "geo_name"]]
          .drop_duplicates("geo_id").copy())
    if len(ed) != EXPECTED_UNITS:
        raise SystemExit(f"{len(ed)} census EDs, expected {EXPECTED_UNITS}")

    for label, s in (("geodatabase", g["GEO_MATCH"]), ("census", ed["geo_id"])):
        if s.duplicated().any():
            raise SystemExit(f"{label} GEO_MATCH is not unique")

    only_c = set(ed["geo_id"]) - set(g["GEO_MATCH"])
    only_g = set(g["GEO_MATCH"]) - set(ed["geo_id"])
    print("\n  the GEO_MATCH join, both ways (§12 — a count match is not a join):")
    print(f"    matched  {len(set(ed['geo_id']) & set(g['GEO_MATCH'])):>4}")
    if only_c or only_g:
        print("    census keys with no polygon:", sorted(only_c)[:8])
        print("    polygons with no census row:", sorted(only_g)[:8])
        raise SystemExit("the key join is not 1:1 -- which should be impossible here")
    print(f"    OK  all {len(ed)} enumeration districts join on GEO_MATCH")

    out = g.merge(ed, left_on="GEO_MATCH", right_on="geo_id", how="inner")
    keep = ["geo_id", "geo_name", "AREA_NAME", "ADM1_NAME", "PARISH", "geometry"]
    out = out[[c for c in keep if c in out.columns]].rename(
        columns={"geo_id": "unit", "AREA_NAME": "geo_name_gdb"})
    out["area_km2"] = out.to_crs(UTM).area.values / 1e6

    a = out["area_km2"]
    print(f"    area: median {a.median():.2f} km², min {a.min():.3f}, max {a.max():.0f}, "
          f"total {a.sum():,.0f} km² (SVG is 389)")
    print(f"    a Kontur r8 hex is ~0.16 km², so the median unit is ~{a.median()/0.16:.0f} "
          f"hexes and {int((a < 0.16).sum())} units are smaller than one hex — which is why "
          f"there is no grid here; see the module docstring")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="eds", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} enumeration districts "
          f"(serves as both units and placement)")
    return out


def main():
    if "--fetch" in sys.argv:
        print("nothing to fetch: the boundaries are inside sources/vc.py's geodatabase, "
              "and there is no placement grid (see the module docstring)")
    build_units()


if __name__ == "__main__":
    main()
