"""Singapore — the placement layer: 332 URA subzones carrying census resident population.

Writes data/geo/sg/sg_subzones.gpkg. `countries.py` uses it to weight where a planning area's
dots land, never to change how many there are. Singapore has no separate unit layer -- **these
subzones ARE the geography**, each labelled with the planning area the religion table counts
it in, which is Hong Kong's and Tonga's wiring.

**THERE IS NO KONTUR HERE AND THAT IS THE POINT.** Every other small country on this map is
placed on the Kontur population grid, and Singapore is the one place where something strictly
better exists: SingStat publishes *Resident Population by Planning Area/Subzone of Residence,
Ethnic Group and Sex (Census of Population 2020)*, which is the SAME CENSUS, the SAME YEAR and
the SAME UNIVERSE-DEFINING WORD -- residents -- as the religion table this weights. A modelled
global surface built from building footprints would be worse on all three counts, and worse in
a specific and knowable direction: Kontur counts everybody physically present, so it would put
weight in the Tuas and Sungei Kadut worker dormitories, where the RESIDENT population the
religion table counts is 70 and 750 people respectively.
Weighting residents by a surface that is mostly non-residents would push Singapore's dots into
exactly the places its census did not ask.

WHAT IT COSTS, STATED. The population table is all ages and the religion table is 15 and over,
so a subzone with an unusually young population is weighted very slightly high. Across the 55
planning areas the 15+ share runs from 0.76 to 0.90 against a national 0.86, which moves dots
inside a planning area by a couple of per cent and never between planning areas. Rounding to
the nearest 10 costs another handful of people per subzone.

THE TWO JOINS ARE BOTH ASSERTED, and one of them nearly went wrong.

  1. **subzone name -> polygon.** All 332 subzone names are distinct across the whole country,
     so the join is unique, but it is still made on (planning area, subzone) and both sides
     must match as SETS -- not "most of them matched"
     ([[reference_name_join_wrong_neighbour]]).
  2. **subzone -> planning area, inside the population CSV.** The CSV is a FLAT list in which
     a planning area is a header row reading `<name> - Total` and its subzones follow it
     unindented, so the parent is carried positionally. **`Changi- Total` is printed with no
     space before the hyphen** -- the only one of the 55 that is -- so the obvious
     `endswith(" - Total")` misses it, Changi is never opened as a planning area, and its
     three subzones (Changi Airport, Changi Point, Changi West) are silently attributed to the
     PREVIOUS header, Central Water Catchment. The visible symptom is Central Water Catchment
     holding 3,700 people when its own total row says nil, and nothing else in the file
     complains. Matched on `\\s*-\\s*Total$` and the planning-area set is then asserted to be
     URA's 55 exactly, which is what catches it.

THE `Others` UNIT. The religion table names 30 planning areas and lumps the other 25 into a
single `Others` row of 25,756 people, so those 25 planning areas' subzones all carry
`unit = "Others"` here and share one mixture. Their dots therefore land across Rochor, Newton,
Singapore River, Southern Islands, Changi and the rest in proportion to who actually lives
there, which is the best available and is honest about being one unit.

Usage:
    python sources/sg_geo.py --fetch    two files from data.gov.sg, ~3.2 MB, seconds
    python sources/sg_geo.py            rebuild data/geo/sg/sg_subzones.gpkg
"""

import collections
import csv
import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sg")
GEO = os.path.join(ROOT, "data", "geo", "sg")
NORM = os.path.join(ROOT, "data", "normalized", "sg.csv")
OUT = os.path.join(GEO, "sg_subzones.gpkg")

# Master Plan 2019 Subzone Boundary (No Sea), URA. The religion table's own footnote says
# "Planning areas refer to areas demarcated in the Urban Redevelopment Authority's Master
# Plan 2019", so this is the matching vintage and not merely the newest one.
SUBZONE_DS = "d_8594ae9ff96d0c708bc2af633048edfb"
SUBZONE_NAME = "MasterPlan2019SubzoneBoundaryNoSea.geojson"
# Resident Population by Planning Area/Subzone of Residence, Ethnic Group and Sex, COP 2020.
POP_DS = "d_e7ae90176a68945837ad67892b898466"
POP_NAME = "sg_resident_population_by_subzone_2020.csv"

EXPECTED_SUBZONES = 332
EXPECTED_PLANNING_AREAS = 55
RESIDENT_POPULATION = 4_044_210        # Table 1.1; the population CSV's own `Total` row
NIL = "-"
MAX_SPAN_DEG = 1.0                     # Singapore is 0.4 degrees wide

TOTAL_RX = re.compile(r"^(.*?)\s*-\s*Total$")


def _key(s):
    return re.sub(r"\s+", " ", (s or "").strip()).upper()


def _num(v, where):
    v = (v or "").strip()
    if v == NIL:
        return 0
    try:
        return int(v.replace(",", ""))
    except ValueError:
        raise SystemExit(f"!! {where}: {v!r} is neither a number nor {NIL!r}")


def _download(dataset, dest, min_bytes):
    import requests

    if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
        print("already have", dest)
        return
    poll = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset}/poll-download"
    print("GET", poll)
    j = requests.get(poll, timeout=120).json()
    # The two dataset families answer differently: a table poll carries `status`, a geospatial
    # one carries only `url`. Read the url and do not require the status field.
    url = j.get("data", {}).get("url")
    if j.get("code") != 0 or not url:
        raise SystemExit(f"!! poll-download did not return a URL: {j}")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(dest + ".part", dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _download(SUBZONE_DS, os.path.join(RAW, SUBZONE_NAME), 1_000_000)
    _download(POP_DS, os.path.join(RAW, POP_NAME), 10_000)


def read_population():
    """(planning area, subzone) -> resident population, from the flat CSV."""
    path = os.path.join(RAW, POP_NAME)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))[1:]

    pa, groups, grand = None, collections.OrderedDict(), None
    for r in rows:
        label, tot = r[0].strip(), _num(r[1], r[0])
        if label == "Total":
            grand = tot
            continue
        m = TOTAL_RX.match(label)
        if m:
            pa = m.group(1).strip()
            if pa in groups:
                raise SystemExit(f"!! planning area {pa!r} has two header rows")
            groups[pa] = {"total": tot, "sub": collections.OrderedDict()}
            continue
        if pa is None:
            raise SystemExit(f"!! subzone {label!r} appears before any planning-area header")
        groups[pa]["sub"][label] = tot

    if grand is None:
        raise SystemExit("!! the population CSV has no grand `Total` row")
    if grand != RESIDENT_POPULATION:
        raise SystemExit(f"!! the population CSV totals {grand:,}, expected the census's "
                         f"{RESIDENT_POPULATION:,}")
    if len(groups) != EXPECTED_PLANNING_AREAS:
        raise SystemExit(
            f"!! parsed {len(groups)} planning areas, expected {EXPECTED_PLANNING_AREAS}. "
            "The `<name> - Total` header match has slipped and subzones are being carried "
            "onto the wrong parent; see this file's docstring on `Changi- Total`.")

    worst = max(((p, g["total"] - sum(g["sub"].values())) for p, g in groups.items()),
                key=lambda x: abs(x[1]))
    print(f"population CSV: {len(groups)} planning areas, "
          f"{sum(len(g['sub']) for g in groups.values())} subzones, {grand:,} residents")
    print(f"  worst planning area vs its own subzones: {worst[1]:+} ({worst[0]}), "
          "which is rounding to the nearest 10")
    if abs(worst[1]) > 100:
        raise SystemExit(f"!! {worst[0]} is off by {worst[1]}; that is not rounding, a "
                         "subzone is on the wrong parent")
    return groups


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gj = os.path.join(RAW, SUBZONE_NAME)
    if not os.path.exists(gj) or not os.path.exists(os.path.join(RAW, POP_NAME)):
        raise SystemExit("missing raw files — run with --fetch first")

    groups = read_population()

    with open(gj, encoding="utf-8-sig") as fh:
        fc = json.load(fh)
    feats = fc.get("features", [])
    print(f"\nURA subzone boundaries: {len(feats)} features")
    if len(feats) != EXPECTED_SUBZONES:
        raise SystemExit(f"!! expected {EXPECTED_SUBZONES} subzones, found {len(feats)}")

    # ---- the units the religion table actually names -------------------------------------
    residual = "Others"
    with open(NORM, encoding="utf-8") as fh:
        named = {r["geo_id"] for r in csv.DictReader(fh)
                 if r["geo_level"] == "planning_area"} - {residual}
    if len(named) != 30:
        raise SystemExit(f"!! {len(named)} named planning areas in {NORM}, expected 30 — "
                         "run sources/sg.py first")
    named_key = {_key(n): n for n in named}
    missing = sorted(n for k, n in named_key.items() if k not in {_key(p) for p in groups})
    if missing:
        raise SystemExit(f"!! the religion table names planning areas the population table "
                         f"and URA do not have: {missing}")

    # ---- join 1: the two flat name sets must match as SETS --------------------------------
    pop_pa = {_key(p) for p in groups}
    ura_pa = {_key(f["properties"]["PLN_AREA_N"]) for f in feats}
    if pop_pa != ura_pa:
        raise SystemExit(f"!! planning-area sets differ.\n  population only: "
                         f"{sorted(pop_pa - ura_pa)}\n  URA only: {sorted(ura_pa - pop_pa)}")
    pop_sz = {(_key(p), _key(s)) for p, g in groups.items() for s in g["sub"]}
    ura_sz = {(_key(f["properties"]["PLN_AREA_N"]), _key(f["properties"]["SUBZONE_N"]))
              for f in feats}
    if pop_sz != ura_sz:
        raise SystemExit(f"!! subzone sets differ.\n  population only: "
                         f"{sorted(pop_sz - ura_sz)[:8]}\n  URA only: "
                         f"{sorted(ura_sz - pop_sz)[:8]}")
    print(f"  join: {len(ura_sz)} (planning area, subzone) pairs match exactly, both ways")

    pop_by = {(_key(p), _key(s)): v for p, g in groups.items() for s, v in g["sub"].items()}

    rows = []
    for f in feats:
        p = f["properties"]
        pa_key = _key(p["PLN_AREA_N"])
        unit = named_key.get(pa_key, residual)
        rows.append({
            "unit": unit,
            "planning_area": p["PLN_AREA_N"].strip().title(),
            "subzone": p["SUBZONE_N"].strip().title(),
            "subzone_code": p.get("SUBZONE_C", "").strip(),
            "pop": float(pop_by[(pa_key, _key(p["SUBZONE_N"]))]),
            "geometry": f["geometry"],
        })

    gdf = gpd.GeoDataFrame.from_features(
        [{"type": "Feature",
          "properties": {k: v for k, v in r.items() if k != "geometry"},
          "geometry": r["geometry"]} for r in rows],
        crs="EPSG:4326")

    n_res = int((gdf["unit"] == residual).sum())
    print(f"\n  {gdf['unit'].nunique()} units: 30 named planning areas + `{residual}`")
    print(f"  `{residual}` is {n_res} subzones in "
          f"{EXPECTED_PLANNING_AREAS - 30} planning areas, "
          f"{gdf.loc[gdf['unit'] == residual, 'pop'].sum():,.0f} residents of all ages")

    w, s, e, n = gdf.total_bounds
    print(f"  bbox {w:.4f} {s:.4f} {e:.4f} {n:.4f}  ({e - w:.3f}° x {n - s:.3f}°)")
    if not (0.05 < e - w < MAX_SPAN_DEG and 0.02 < n - s < MAX_SPAN_DEG):
        raise SystemExit("!! Singapore does not fit in a degree; geometry is torn "
                         "[[reference_antimeridian]]")

    os.makedirs(GEO, exist_ok=True)
    gdf.to_file(OUT, driver="GPKG", layer="subzones")
    print(f"\nwrote {OUT}\n  {len(gdf):,} subzones, {gdf['pop'].sum():,.0f} residents")

    # ---- the check that actually tests the join -------------------------------------------
    #
    # Both sides are the SAME CENSUS, so unlike the Kontur checks elsewhere on this map this
    # one is allowed to be tight. Per planning area, residents aged 15+ (the religion table)
    # against residents of all ages (the population table) must be a near-perfect straight
    # line through the origin, because the only thing separating them is the local age
    # structure. A subzone attached to the wrong planning area shows up as a unit whose ratio
    # leaves the 15+ share band, and a scrambled join collapses r.
    with open(NORM, encoding="utf-8") as fh:
        rel = {r["geo_id"]: float(r["count"]) for r in csv.DictReader(fh)
               if r["source_category"] == "Total"
               and r["geo_level"] == "planning_area"}
    grid = gdf.groupby("unit")["pop"].sum().to_dict()
    pairs = [(rel[u], grid[u]) for u in sorted(rel)]
    n = len(pairs)
    ma = sum(a for a, _ in pairs) / n
    mb = sum(b for _, b in pairs) / n
    num = sum((a - ma) * (b - mb) for a, b in pairs)
    den = (sum((a - ma) ** 2 for a, _ in pairs)
           * sum((b - mb) ** 2 for _, b in pairs)) ** 0.5
    r = num / den
    share = {u: rel[u] / grid[u] for u in rel if grid[u] > 0}
    lo, hi = min(share.values()), max(share.values())
    print(f"\n  aged 15+ vs all ages, over {n} units: r = {r:.5f}")
    print(f"  {'unit':<18}{'15+':>10}{'all ages':>11}{'15+ share':>11}")
    for u in sorted(share, key=share.get):
        print(f"  {u:<18}{rel[u]:>10,.0f}{grid[u]:>11,.0f}{share[u]:>11.3f}")
    print(f"  national 15+ share {rel and sum(rel.values()) / sum(grid.values()):.3f}; "
          f"band {lo:.3f}-{hi:.3f}")
    if r < 0.99 or lo < 0.60 or hi > 1.00:
        raise SystemExit(
            f"!! r = {r:.4f}, 15+ share band {lo:.3f}-{hi:.3f}. Two tables of one census "
            "cannot disagree by that much — a subzone is on the wrong planning area.")


if __name__ == "__main__":
    main()
