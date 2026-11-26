"""Every registered zone scored on shape and on commutes crossing its line (T-4364; read-only on data/).

    venv\\Scripts\\python.exe -I trainworld\\notes\\T-4364\\rank_zones.py [--jobs 4] [--out <csv>]

From the maps root; claim the cores first (`python cores.py claim 4 rank_zones --pid <pid>`); about 10 minutes at 4.
Writes one row per registered pack to artifacts/trainworld/zone-ranking.csv (default):

- hull: area over its convex hull, the merge rule's compactness (docs/zones.md 2, 75%), of the pack's own
  boundary (data/packs/<id>.boundary.geojson, land only, as the game draws it), in Mollweide as world_plan.
- land_hull: area over the land in its convex hull (the hull less OSM's coastline sea, osm_water.sea), so a coast,
  a bay or islands do not count against it; the ranking uses this one.
- cross: the share of commutes with an end in the zone that cross its line, from the joint commute solve's figures
  in the pack header (gravity.trips within; gravity.joint.commuters_out and commuters_in across, long commutes
  included): (out + in) / (trips + out + in). A pack in no joint solve has none (blank).
- score: the mean of the two percentiles (1 - land_hull and cross, 1 = worst of all), blank without cross.
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
PACKS = os.path.join(TW, "data", "packs")
REG = os.path.join(TW, "app", "src", "game", "registry.json")
NAMES = os.path.join(TW, "app", "src", "game", "zone-names.json")
PLAN = os.path.join(TW, "pipeline", "world_plan.json")
OUT = os.path.join(os.path.dirname(TW), "artifacts", "trainworld", "zone-ranking.csv")


def one(cid):
    try:
        return _one(cid)
    except Exception as e:  # a bad outline is one blank row, not a failed run
        print(f"{cid}: {e}")
        return {"id": cid}


def _one(cid):
    import shapely
    import world_plan as wp
    import osm_water
    out = {"id": cid}
    try:
        h = json.load(open(os.path.join(PACKS, f"{cid}.json"), encoding="utf-8"))
    except OSError:
        return out
    out["pop"] = round((h.get("totals") or {}).get("pop") or 0)
    g = h.get("gravity") or {}
    j = g.get("joint") or {}
    if j and g.get("trips"):
        o, i, t = j.get("commuters_out", 0), j.get("commuters_in", 0), g["trips"]
        out["cross"] = (o + i) / (t + o + i)
        out["out_share"] = o / (t + o)
    try:
        gj = json.load(open(os.path.join(PACKS, f"{cid}.boundary.geojson"), encoding="utf-8"))
    except OSError:
        return out
    geom = shapely.union_all([shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in gj["features"]])
    geom = wp.polygonal(geom)
    if geom.is_empty:
        return out
    m = wp.to_moll(geom)
    hull = m.convex_hull
    out["km2"] = m.area / 1e6
    out["hull"] = m.area / hull.area
    hull_ll = geom.convex_hull
    sea = osm_water.sea(hull_ll.bounds)
    sea_in = 0.0
    mv = shapely.make_valid(m)
    for x in sea:  # the split sea polygons do not overlap, so their parts add up
        try:
            s = wp.polygonal(shapely.make_valid(x).intersection(hull_ll))
            if not s.is_empty:
                sea_in += shapely.make_valid(wp.to_moll(s)).difference(mv).area
        except shapely.errors.GEOSException:
            out["sea_error"] = 1
    land = max(hull.area - sea_in, m.area)
    out["land_hull"] = m.area / land
    return out


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT
    reg = json.load(open(REG, encoding="utf-8"))["cities"]
    names = json.load(open(NAMES, encoding="utf-8"))
    plan = {r["id"]: r for r in json.load(open(PLAN, encoding="utf-8"))["metros"]}
    ids = [c["id"] for c in reg]
    rows = []
    with Pool(jobs) as p:
        for k, r in enumerate(p.imap_unordered(one, ids, chunksize=8)):
            rows.append(r)
            if k % 250 == 0:
                print(f"{k}/{len(ids)}")
    by = {c["id"]: c for c in reg}
    for r in rows:
        c = by[r["id"]]
        pr = plan.get(r["id"]) or {}
        r["name"] = (names.get(r["id"]) or {}).get("name") or c.get("name")
        r["plan_name"] = pr.get("name", "")
        r["country"] = c.get("country", "")
        r["kind"] = pr.get("kind") or ("metro" if pr.get("units") else "hand")
        r["status"] = pr.get("status", "")
    # percentiles: 1 is the worst
    def pct(key, bad):
        have = sorted((bad(r[key]), r["id"]) for r in rows if r.get(key) is not None)
        n = max(1, len(have) - 1)
        return {i: k / n for k, (_, i) in enumerate(have)}
    ph = pct("land_hull", lambda v: -v)
    pc = pct("cross", lambda v: v)
    for r in rows:
        if r["id"] in ph and r["id"] in pc:
            r["score"] = (ph[r["id"]] + pc[r["id"]]) / 2
    rows.sort(key=lambda r: -(r.get("score") or -1))
    cols = ["id", "name", "plan_name", "country", "kind", "status", "pop", "km2", "hull", "land_hull", "cross", "out_share",
            "score"]
    tmp = path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(tmp, path)
    print("->", path, len(rows))


if __name__ == "__main__":
    main()
