"""T-3012: zones whose land is in more than one piece. Read-only over data/packs and the registry.

    python checks/split_zones.py [--out FILE] [--ids a,b,c] [--min-km2 N] [--touch-m N]
                                 (from trainworld/, ..\\venv's Python; about 2-3 minutes over every zone, one core)

Every registered city and zone (app/src/game/registry.json) has its land in `data/packs/<id>.boundary.geojson`;
this reads those, not `outlines.json`, whose rings carry the sea off each coast and so join across water. A zone's
parts are grouped into pieces (parts within TOUCH_M of each other are one piece: slivers between boundary files).
Each piece but the biggest is judged by the straight line from its nearest point to the nearest other piece,
sampled every 250 m against Natural Earth's countries:

  tiny     the piece is under MIN_KM2 (a rounding sliver or an islet)
  seam     the nearest piece is within 500 m (a river, a road, a gap between two boundary files)
  water    under half the line is land: an island or a piece across a strait, which `contiguous` zones allow
           (WATER_GAP_KM, ARCHIPELAGO_KM in world_plan.py)
  exclave  the piece lies in a unit whose own land is in separate parts (Golmud's Tanggula); needs the plan's state
  foreign  half the land under the line or more belongs to another country than the main piece's: an exclave the
           source itself draws (Kaliningrad, Ceuta, Nakhchivan, Cabinda, Llivia)
  land     the line runs over the zone's own country: the zone is in pieces on land. These are the findings.

For each land piece the output names the zones that touch it most and the zones standing in the gap, which
is what a mend (move the piece, or the gap's zone) would need. Output: `../artifacts/trainworld/split-zones.json`
(`counts` first, then `zones`, worst first by the biggest land piece's area). `material` marks a zone with a land piece of 20 km2 or more 2 km or more from its nearest piece; the smaller ones are
slivers, river and bay cuts and the odd cut by a pack. Exit 1 if any zone is material.
A zone crossing the antimeridian is read as drawn in longitude and may show false pieces.
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyogrio
import shapely
from shapely import STRtree
from shapely.geometry import LineString, shape
from shapely.ops import nearest_points

TW = Path(__file__).resolve().parents[1]
PACKS = TW / "data/packs"
REG = TW / "app/src/game/registry.json"
NE = TW / "data/raw/world_field/ne_10m_admin_0_countries.zip"
TOUCH_M = 100.0
MIN_KM2 = 1.0
STEP_M = 250.0
SEAM_M = 500.0
MAT_KM2 = 20.0  # a land piece this big, this far from the rest, is material: the zone is in pieces a player sees
MAT_GAP_KM = 2.0
M_LAT = 110574.0
M_LON = 111320.0


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def projector(lat0):
    k = M_LON * np.cos(np.radians(lat0))

    def fwd(c):
        return np.column_stack([c[:, 0] * k, c[:, 1] * M_LAT])

    def inv(c):
        return np.column_stack([c[:, 0] / k, c[:, 1] / M_LAT])
    return fwd, inv


def flat(g):
    """The simple parts of g, collections inside collections (make_valid's) opened."""
    for p in shapely.get_parts(g):
        if p.geom_type in ("GeometryCollection", "MultiPolygon"):
            yield from flat(p)
        else:
            yield p


def pieces_of(g, touch):
    """[(piece geometry)] of g's parts, parts within `touch` metres one piece, biggest first."""
    parts = list(shapely.get_parts(g))
    if len(parts) < 2:
        return [g]
    tree = STRtree(parts)
    root = list(range(len(parts)))

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a
    for a, b in zip(*tree.query(parts, predicate="dwithin", distance=touch)):
        root[find(int(a))] = find(int(b))
    by = collections.defaultdict(list)
    for k, p in enumerate(parts):
        by[find(k)].append(p)
    return sorted((shapely.union_all(v) for v in by.values()), key=lambda p: -p.area)


def exclaves(zones):
    """A land piece lying in a unit that has parts of its own far apart is that unit's own exclave (Golmud's Tanggula),
    which the source itself draws: kind "exclave". Needs the plan and its state (data/work/worldplan/state.pkl)."""
    import pickle
    sys.path.insert(0, str(TW / "pipeline"))
    import world_plan as wp
    todo = [z for z in zones if any(r["kind"] == "land" for r in z["found"])]
    if not todo or not (wp.WORK / "state.pkl").exists():
        return

    class Here(pickle.Unpickler):
        def find_class(self, mod, name):
            return wp.Unit if name == "Unit" else super().find_class(mod, name)
    st = Here(open(wp.WORK / "state.pkl", "rb")).load()
    geo = {u.id: u.geom for u in st["units"]}
    rows = {m["id"]: m for m in json.loads(wp.OUT.read_text(encoding="utf-8"))["metros"]}
    for z in todo:
        r = rows.get(z["id"])
        ids = [i for v in (r or {}).get("units", {}).values() for i in v if i in geo]
        for row in z["found"]:
            if row["kind"] != "land":
                continue
            x, y = wp.FWD.transform(*row["at"])
            pt = shapely.Point(x, y)
            for i in ids:
                g = geo[i]
                if g.distance(pt) < 1000 and len(pieces_of(g, TOUCH_M)) > 1:
                    row["kind"] = "exclave"
                    break


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "-h" in sys.argv or "--help" in sys.argv:
        sys.exit(__doc__)
    out = Path(arg("--out", TW.parent / "artifacts/trainworld/split-zones.json"))
    only = set(arg("--ids").split(",")) if arg("--ids") else None
    min_km2 = float(arg("--min-km2", MIN_KM2))
    touch = float(arg("--touch-m", TOUCH_M))
    t0 = time.time()
    reg = json.loads(REG.read_text(encoding="utf-8"))["cities"]
    meta = {c["id"]: c for c in reg}
    ne = pyogrio.read_dataframe(f"zip://{NE}", columns=["ADM0_A3"])
    ne_geom, ne_iso = list(ne.geometry), list(ne.ADM0_A3)
    ne_tree = STRtree(ne_geom)

    def land_of(lon, lat):
        """The ADM0_A3 of the land at each (lon, lat), '' on none."""
        pts = shapely.points(lon, lat)
        res = np.full(len(pts), "", dtype=object)
        pi, gi = ne_tree.query(pts, predicate="intersects")
        for a, b in zip(pi, gi):
            if not res[a]:
                res[a] = ne_iso[int(b)]
        return res

    geoms, found, order = {}, {}, []
    for c in reg:
        cid = c["id"]
        f = PACKS / f"{cid}.boundary.geojson"
        if not f.exists() or (only and cid not in only):
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        gs = [shape(ft["geometry"]) for ft in doc["features"] if ft.get("geometry")]
        g = shapely.make_valid(shapely.union_all(gs)) if len(gs) > 1 else shapely.make_valid(gs[0])
        polys = [p for p in flat(g) if p.geom_type == "Polygon" and not p.is_empty]
        if not polys:
            continue
        geoms[cid] = shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]
        order.append(cid)
    print(f"read {len(order)} boundaries, {time.time() - t0:.0f} s", flush=True)
    tree = STRtree([geoms[i] for i in order])
    total_kind = collections.Counter()
    zones = []
    for n, cid in enumerate(order):
        g = geoms[cid]
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(order)} {time.time() - t0:.0f} s", flush=True)
        if shapely.get_num_geometries(g) < 2:
            continue
        t1 = time.time()
        b = g.bounds
        fwd, inv = projector((b[1] + b[3]) / 2)
        gp = shapely.transform(g, fwd)
        pcs = pieces_of(gp, touch)
        if len(pcs) < 2:
            continue
        main = pcs[0]
        # the country of the main piece: the land under its representative points
        rp = shapely.get_coordinates(shapely.transform(main.representative_point(), inv))
        home = land_of(rp[:, 0], rp[:, 1])[0]
        rows = []
        small = [p for p in pcs[1:] if p.area / 1e6 < min_km2]
        if small:
            total_kind["tiny"] += len(small)
        big = pcs[:1] + [p for p in pcs[1:] if p.area / 1e6 >= min_km2]
        ptree = STRtree(pcs)
        for p in big[1:]:
            idx, dist = ptree.query_nearest(p, exclusive=True, return_distance=True, all_matches=False)
            a, bb = nearest_points(p, pcs[int(idx[0])])
            gap = float(dist[0])
            row = {"area_km2": round(p.area / 1e6, 2), "gap_km": round(gap / 1e3, 2),
                   "at": [round(v, 4) for v in shapely.get_coordinates(shapely.transform(p.representative_point(), inv))[0]]}
            if gap <= SEAM_M:
                row["kind"] = "seam"
            else:
                m = max(int(gap // STEP_M), 1)
                xy = np.array([[a.x + (bb.x - a.x) * t, a.y + (bb.y - a.y) * t] for t in np.linspace(0, 1, m + 1)[1:-1]])                     if m > 1 else np.zeros((0, 2))
                if len(xy):
                    ll = inv(xy)
                    land = land_of(ll[:, 0], ll[:, 1])
                    n_land = int((land != "").sum())
                    foreign = int(((land != "") & (land != home)).sum())
                    row["land_share"] = round(n_land / len(xy), 2)
                    if n_land < 0.5 * len(xy):
                        row["kind"] = "water"
                    elif foreign >= 0.5 * n_land:
                        row["kind"] = "foreign"
                    else:
                        row["kind"] = "land"
                else:
                    row["land_share"] = 1.0
                    row["kind"] = "land"
            total_kind[row["kind"]] += 1
            if row["kind"] == "land":
                ring = p.buffer(150)
                touching = collections.Counter()
                for j in tree.query(shapely.transform(ring, inv), predicate="intersects"):
                    oid = order[int(j)]
                    if oid == cid:
                        continue
                    og = shapely.transform(geoms[oid], fwd)
                    touching[oid] = og.boundary.intersection(ring).length / 2 / 1e3
                row["touching"] = [{"id": i, "name": meta[i]["name"], "border_km": round(v, 1)}
                                   for i, v in touching.most_common(4) if v > 0.05]
                line = LineString([a, bb]).buffer(2500)
                between = collections.Counter()
                for j in tree.query(shapely.transform(line, inv), predicate="intersects"):
                    oid = order[int(j)]
                    if oid == cid:
                        continue
                    between[oid] = shapely.transform(geoms[oid], fwd).intersection(line).area / 1e6
                row["between"] = [{"id": i, "name": meta[i]["name"], "km2_in_corridor": round(v)}
                                  for i, v in between.most_common(4) if v > 1]
            rows.append(row)
        zones.append({"id": cid, "name": meta[cid]["name"], "country": meta[cid].get("country"),
                      "land_iso": home, "area_km2": round(main.area / 1e6), "pieces": len(pcs), "tiny_pieces": len(small), "found": rows})
        if time.time() - t1 > 10:
            print(f"  {cid} {meta[cid]['name']}: {shapely.get_num_geometries(g)} parts, {time.time() - t1:.0f} s", flush=True)
    if "--no-units" not in sys.argv:
        exclaves(zones)
        n_ex = sum(r["kind"] == "exclave" for z in zones for r in z["found"])
        total_kind["land"] -= n_ex
        total_kind["exclave"] = n_ex
    bad = [z for z in zones if any(r["kind"] == "land" for r in z["found"])]
    for z in bad:
        z["found"] = [r for r in z["found"] if r["kind"] == "land"] + [r for r in z["found"] if r["kind"] != "land"]
        z["worst_km2"] = max(r["area_km2"] for r in z["found"] if r["kind"] == "land")
        z["material"] = any(r["kind"] == "land" and r["area_km2"] >= MAT_KM2 and r["gap_km"] >= MAT_GAP_KM for r in z["found"])
    bad.sort(key=lambda z: (not z["material"], -z["worst_km2"]))
    mat = [z for z in bad if z["material"]]
    by_country = collections.Counter(z["country"] for z in bad)
    doc = {"task": "T-3012", "params": {"touch_m": touch, "min_km2": min_km2, "step_m": STEP_M, "material_km2": MAT_KM2,
                          "material_gap_km": MAT_GAP_KM},
           "counts": {"zones_read": len(order), "multi_piece_zones": len(zones), "zones_split_on_land": len(bad),
                      "zones_material": len(mat), "material_by_country": dict(collections.Counter(z["country"] for z in mat).most_common()),
                      "pieces_by_kind": dict(total_kind), "split_by_country": dict(by_country.most_common())},
           "zones": bad}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)
    print(f"{len(order)} zones read, {len(zones)} with more than one piece, {len(bad)} split on land, {len(mat)} material "
          f"({dict(total_kind)}), {time.time() - t0:.0f} s -> {out}")
    for z in (mat or bad)[:15]:
        w = [r for r in z["found"] if r["kind"] == "land"]
        print(f"  {z['id']:6s} {z['name'][:28]:28s} {z['country']}  main {z['area_km2']} km2, "
              f"{len(w)} land piece(s), biggest {z['worst_km2']} km2, gap {max(r['gap_km'] for r in w)} km")
    sys.exit(1 if mat else 0)


if __name__ == "__main__":
    main()
