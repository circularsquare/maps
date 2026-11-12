"""T-3240: each market place's long-trip pull, the `places` table of pipeline/market_pull.json.

    python pipeline/market_pull.py           the table worked out again and written into market_pull.json
    python pipeline/market_pull.py --dry     printed, not written

A place's *excess* is how much more important it is than its people say (docs/regional.md 5.1): the best Natural
Earth 10m populated-places `scalerank` inside its zone's outline against the rank a place of its people has (a line
through every ranked place's rank against ln people), clipped at 0 and at the config's `cap`. Two trims for map
balance (Natural Earth ranks remote centres high): the cap shrinks with the people within `trim.radius_km` of the
place's centre (its own included), from the full cap at `trim.people[1]` to none at `trim.people[0]`, on a log
scale; `hand` sets a place's excess outright. Beach and resort towns (`resorts`, pipeline/market_resorts.json from
Wikidata) give their place at least `resorts.excess` times the share of its people living in them over
`resorts.share` (at most 1). The rule reads the excess as a pull `exp(gamma x excess)` (market_zones.py,
intercity.rs, marketRule.ts).

Reads data/raw/world_field/ne_10m_populated_places_simple.zip, data/packs/<id>.boundary.geojson and the market
rows (people and centres). Rerun before a world rebuild when the zones change; the rebuild reads the table.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import market_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).parent / "market_pull.json"
RESORTS = Path(__file__).parent / "market_resorts.json"
NE = ROOT / "data/raw/world_field/ne_10m_populated_places_simple.zip"
PACKS = ROOT / "data/packs"


def load():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def km(lon, lat, lons, lats):
    p, q = math.radians(lat), np.radians(lats)
    h = np.sin((q - p) / 2) ** 2 + math.cos(p) * np.cos(q) * np.sin(np.radians(lons - lon) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.minimum(h, 1)))


def outlines(ids):
    """Each zone's outline (shapely), those with a boundary file, and a tree over them."""
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    geoms, of = [], []
    for z in ids:
        p = PACKS / f"{z}.boundary.geojson"
        if not p.exists():
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        for f in (g["features"] if g.get("type") == "FeatureCollection" else [g]):
            geoms.append(shape(f.get("geometry", f)).buffer(0))
            of.append(z)
    return geoms, of, STRtree(geoms)


def locate(points, geoms, of, tree):
    """The zone holding each (lon, lat), or None."""
    from shapely.geometry import Point
    out = []
    for lon, lat in points:
        pt, hit = Point(lon, lat), None
        for k in tree.query(pt):
            if geoms[k].contains(pt):
                hit = of[k]
                break
        out.append(hit)
    return out


def ranks(geoms, of, tree):
    """{zone: [best scalerank, its point's name]} over Natural Earth's populated places."""
    import pyogrio
    df = pyogrio.read_dataframe(f"zip://{NE}!ne_10m_populated_places_simple.shp", columns=["name", "scalerank", "latitude", "longitude"], read_geometry=False)
    best = {}
    for z, (name, r) in zip(locate(zip(df.longitude, df.latitude), geoms, of, tree), zip(df.name, df.scalerank)):
        if z is not None and (z not in best or r < best[z][0]):
            best[z] = [int(r), name]
    return best


def table(cfg, rows):
    """The places table {zone: [excess, why]} and the rank line (a, b): rank = a + b ln(people)."""
    ids = list(rows)
    geoms, of, tree = outlines(ids)
    best = ranks(geoms, of, tree)
    pop = np.array([max(rows[z]["pop"], 1.0) for z in ids])
    lon, lat = np.array([rows[z]["lon"] for z in ids]), np.array([rows[z]["lat"] for z in ids])
    have = [k for k, z in enumerate(ids) if z in best]
    b, a = np.polyfit(np.log(pop[have]), [best[ids[k]][0] for k in have], 1)
    t = cfg["trim"]
    lo, hi = math.log(t["people"][0]), math.log(t["people"][1])
    resort_people = {}
    if RESORTS.exists():
        res = json.loads(RESORTS.read_text(encoding="utf-8"))["towns"]
        for z, r in zip(locate([(r["lon"], r["lat"]) for r in res], geoms, of, tree), res):
            if z is not None:
                resort_people[z] = resort_people.get(z, 0.0) + (r.get("pop") or cfg["resorts"]["pop_unknown"])
    out = {}
    for k, z in enumerate(ids):
        ex, why = 0.0, []
        if z in best:
            ex = a + b * math.log(pop[k]) - best[z][0]
            if ex > 0:
                why.append(f"rank {best[z][0]} ({best[z][1]})")
            near = float(pop[km(lon[k], lat[k], lon, lat) < t["radius_km"]].sum())
            cap = cfg["cap"] * min(max((math.log(near) - lo) / (hi - lo), 0.0), 1.0)
            if cap < cfg["cap"] and ex > cap:
                why.append(f"trimmed from {min(ex, cfg['cap']):.2f}, {near / 1e6:.2f}M within {t['radius_km']:.0f} km")
            ex = min(max(ex, 0.0), cap)
        if z in resort_people:
            r = cfg["resorts"]["excess"] * min(1.0, resort_people[z] / pop[k] / cfg["resorts"]["share"])
            if r > ex:
                ex = r
                why.append(f"resorts {resort_people[z] / pop[k]:.0%} of its people")
        if z in cfg["hand"]:
            ex = float(cfg["hand"][z][0])
            why.append("by hand: " + cfg["hand"][z][1])
        if ex >= 0.01 or z in cfg["hand"]:
            out[z] = [round(ex, 3), "; ".join(why)]
    return out, (round(float(a), 3), round(float(b), 4))


def main():
    cfg = load()
    rows = market_rows.load(ROOT)
    places, fit = table(cfg, rows)
    cfg["rank_fit"] = list(fit)
    cfg["places"] = dict(sorted(places.items()))
    pulls = sorted(places.items(), key=lambda kv: -kv[1][0])
    print(f"rank = {fit[0]} {fit[1]:+} ln(people); {len(places)} places with an excess, {sum(1 for v in places.values() if 'trimmed' in v[1])} trimmed, "
          f"{sum(1 for v in places.values() if 'resorts' in v[1])} by their resorts")
    for z, (e, why) in pulls[:15]:
        print(f"  {z} {e:.2f} {why}")
    if "--dry" not in sys.argv:
        part = CONFIG.with_suffix(".json.part")
        part.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        part.replace(CONFIG)
        print("wrote", CONFIG)


if __name__ == "__main__":
    main()
