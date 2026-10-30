"""T-1857: Australia's country keys for planned rows (metros and zones): 2021 census population by
SA1, everything else tier 0's (docs/data-register.md, Australia).

    python pipeline/australia.py <id>     the census and GHSL people in a planned row's outline

The census figures come from the ABS's own ArcGIS service (geo.abs.gov.au, the Digital Atlas layers,
CC BY 4.0): the 2021 General Community Profile table G01 joined to the ASGS Edition 3 SA1 polygons,
one layer, queried by the outline's box and cached per city. SA1s hold 200 to 800 people.
"""
import os
import sys

import numpy as np
from shapely.geometry import shape
from shapely.validation import make_valid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arcgis  # noqa: E402

G01_SA1 = "https://geo.abs.gov.au/arcgis/rest/services/Hosted/ABS_2021_Census_G01_SA1/FeatureServer/0"
# the polygons generalised to about 20 m: they only sort GHSL cells and test points against the outline
SIMPLIFY_DEG = 0.0002
SOURCE = ("ABS 2021 Census of Population and Housing, G01 persons by SA1 (ASGS Edition 3, geo.abs.gov.au), "
          "spread inside each SA1 by GHSL GHS-POP E2020")


def sa1s(city, bounds):
    """[(SA1 code, polygon, people)] of every SA1 meeting the box, cached in data/raw/tier0/australia/."""
    feats = arcgis.features(os.path.join("tier0", "australia", f"sa1_{city}"), G01_SA1, fields="objectid,sa1_code_2021,tot_p_p",
                            bbox=bounds, page=2000, params={"orderByFields": "objectid", "maxAllowableOffset": SIMPLIFY_DEG})
    out = []
    for f in feats:
        if not f.get("geometry"):
            continue  # migratory, offshore and no-usual-address SA1s have no polygon
        p = f["properties"]
        out.append((p["sa1_code_2021"], make_valid(shape(f["geometry"])), float(p["tot_p_p"] or 0)))
    return out


def plan_population(city, boundary, sig):
    """tier0's `population` hook (Canada's DA hook, T-564, by SA1): each SA1 whose representative point
    is inside the outline, its census people spread over its cells by GHSL; cells in no such SA1 keep
    GHSL's people."""
    import england
    allu = sa1s(city, boundary.bounds)
    units = [(c, g) for c, g, _ in allu if boundary.contains(g.representative_point())]
    tot = {c: n for c, _, n in allu}
    ghsl = np.asarray(sig["pop_ghsl"], float)
    idx, _ = england.cell_units(sig["lat"], sig["lon"], units)
    out = england.spread_totals(sig["lat"], sig["lon"], units, tot, ghsl)
    out[idx < 0] = ghsl[idx < 0]
    print(f"  census population by SA1: {len(units):,} of {len(allu):,} SA1s in the box, {out.sum():,.0f} people "
          f"(GHSL {ghsl.sum():,.0f}; {int((idx < 0).sum()):,} cells in no SA1 keep GHSL's)", flush=True)
    return out, SOURCE


def plan_hooks(row):
    """Any planned Australian row, metro or zone: population by SA1. Jobs, work from home and the
    curve stay tier 0's (the 2021 census's place of work and its journey-to-work flows were taken in
    the Sydney and Melbourne lockdowns, and the flows are in TableBuilder only)."""
    return {"population": plan_population}


if __name__ == "__main__":
    import json
    from shapely.ops import unary_union
    city = sys.argv[1]
    import build_city as b
    with open(os.path.join(b.WORK, f"{city}.outline.geojson"), encoding="utf-8") as f:
        g = unary_union([make_valid(shape(x["geometry"])) for x in json.load(f)["features"]])
    allu = sa1s(city, g.bounds)
    inside = [n for c, s, n in allu if g.contains(s.representative_point())]
    print(f"{city}: {len(inside):,} SA1s inside, {sum(inside):,.0f} census people")
