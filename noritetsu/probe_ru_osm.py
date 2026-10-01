"""Russia stage 1: the state of OSM's railway data in Russia, on the extract noritetsu builds from.

    python probe_ru_osm.py > data/raw/ru/probe_osm.txt

After `python extract.py --region ru --pbf data/raw/russia-YYMMDD.osm.pbf`. Measures:

1. Where the track is: km by the country outline it falls in (religiondots'
   country_shapes.geojson, the outline tools/build_regions.py uses), so what Geofabrik's extract
   carries beyond Russia, and how much is in Crimea, is known before any clip.
2. Heavy-rail main and branch track (build_tiles.rank_of < 2) inside Russia: the part a
   route=train relation runs over ("passenger-used"), and of each the share in a
   route=railway/route=tracks relation (infra.pkl) and the share with a `name` on the way.
3. route=train relations: how many, suburban (электрички) against long-distance, by the
   service tag, the name and the network.
4. Urban rail: route relations by city/network.
5. Infrastructure relations: count, ref and name shapes.
"""
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "proc" / "ru"
SHAPES = ROOT.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"
CRIMEA_BOX = (32.4, 44.3, 36.7, 46.25)     # with ru's outline, which holds Crimea


def main():
    import json

    import build_tiles as bt
    from shapely import contains_xy
    from shapely.geometry import shape
    from shapely.prepared import prep

    with open(D / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(D / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(D / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    c = np.load(D / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]

    feats = json.loads(SHAPES.read_text(encoding="utf-8"))["features"]
    near = {"ru", "ua", "by", "kz", "ee", "lv", "lt", "pl", "fi", "no", "ge", "az", "mn", "cn",
            "kp", "am", "md"}
    outlines = {}
    for f in feats:
        cc = f["properties"]["cc"]
        if cc in near:
            g = shape(f["geometry"])
            outlines[cc] = g.union(outlines[cc]) if cc in outlines else g

    def way_geom(nodes):
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size < 2:
            return 0.0, None, None
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        km = float(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32,
                            np.diff(lat) * 110.57).sum())
        return km, float(lon[len(lon) // 2]), float(lat[len(lat) // 2])

    # where is each way (by its middle vertex)
    wids = list(ways)
    mids = np.full((len(wids), 2), np.nan)
    kms = np.zeros(len(wids))
    for i, wid in enumerate(wids):
        km, x, y = way_geom(ways[wid][1])
        kms[i] = km
        if x is not None:
            mids[i] = (x, y)
    where = np.array(["?"] * len(wids), dtype=object)
    for cc in ["ru"] + sorted(k for k in outlines if k != "ru"):
        free = where == "?"
        ok = np.zeros(len(wids), bool)
        ok[free] = contains_xy(outlines[cc], mids[free, 0], mids[free, 1])
        where[ok] = cc
    crimea = ((mids[:, 0] >= CRIMEA_BOX[0]) & (mids[:, 0] <= CRIMEA_BOX[2])
              & (mids[:, 1] >= CRIMEA_BOX[1]) & (mids[:, 1] <= CRIMEA_BOX[3]) & (where == "ru"))
    loc = dict(zip(wids, where))
    in_crimea = set(np.array(wids)[crimea].tolist())

    print("1. ALL RAILWAY WAYS IN THE EXTRACT, km by the outline they fall in (middle vertex)")
    by = Counter()
    for i, wid in enumerate(wids):
        by[where[i]] += kms[i]
    for cc, km in by.most_common():
        print(f"   {cc:<4} {km:10,.0f} km")
    print(f"   of ru, in the Crimea box: {kms[crimea].sum():,.0f} km "
          f"({int(crimea.sum())} ways)")

    # --- relations
    in_route = defaultdict(set)
    for rid, (tags, members) in rels.items():
        if tags.get("type") != "route":
            continue
        for ty, ref, _role in members:
            if ty == "w":
                in_route[ref].add(tags.get("route"))
    in_infra = set()
    for rid, (tags, members) in infra.items():
        for ty, ref, _ in members:
            if ty == "w":
                in_infra.add(ref)

    tot = Counter()
    names = Counter()
    usage = Counter()
    for i, wid in enumerate(wids):
        tags, nodes = ways[wid]
        if where[i] != "ru":
            continue
        kind = bt.KIND[tags["railway"]]
        rank = bt.rank_of(kind, tags)
        km = kms[i]
        if kind in ("rail", "narrow_gauge", "heritage"):
            usage[(tags.get("usage") or "-", tags.get("service") or "-")] += km
        if kind not in ("rail", "narrow_gauge", "heritage"):
            if rank < 2:
                tot[("urban", "all")] += km
                if wid in in_route:
                    tot[("urban", "pax")] += km
            continue
        if rank >= 2:
            continue
        pax = "train" in in_route.get(wid, ())
        name = (tags.get("name") or "").strip()
        scopes = ("all", "pax") if pax else ("all",)
        if wid in in_crimea:
            scopes = scopes + ("crimea",) + (("crimea_pax",) if pax else ())
        for scope in scopes:
            tot[(scope, "km")] += km
            if wid in in_infra:
                tot[(scope, "infra")] += km
            if name:
                tot[(scope, "named")] += km
            if rank == 0:
                tot[(scope, "main")] += km
            if tags.get("railway:traffic_mode") == "freight":
                tot[(scope, "freight")] += km
        if name:
            names[name] += km

    print("\n2. HEAVY RAIL in ru (rail, narrow gauge, preserved), rank < 2 (main + branch)")
    for scope, label in (("all", "all main + branch track"),
                         ("pax", "passenger-used (in a route=train relation)"),
                         ("crimea", "  of all, in Crimea"),
                         ("crimea_pax", "  of passenger-used, in Crimea")):
        k = tot[(scope, "km")]
        print(f"  {label}: {k:,.0f} km")
        for key, what in (("infra", "in a route=railway/tracks relation"),
                          ("named", "with a name on the way"),
                          ("main", "usage=main"),
                          ("freight", "railway:traffic_mode=freight")):
            print(f"    {what:<40} {tot[(scope, key)]:>9,.0f} km  "
                  f"{100 * tot[(scope, key)] / k if k else 0:5.1f}%")
    print(f"  usage/service on all rail in ru (km): "
          f"{[(u, round(v)) for u, v in usage.most_common(14)]}")
    print(f"\n  way names on main + branch track: {len(names)} distinct; top 40 by km:")
    for n, km in names.most_common(40):
        print(f"    {km:8.1f}  {n}")
    print(f"\nURBAN in ru (subway, light rail, tram, monorail, funicular), rank < 2: "
          f"{tot[('urban', 'all')]:,.0f} km, in a route relation {tot[('urban', 'pax')]:,.0f}")

    # --- route=train relations in ru (any member way in ru)
    def rel_where(members):
        cs = Counter(loc.get(r) for ty, r, _ in members if ty == "w" and r in loc)
        return cs.most_common(1)[0][0] if cs else None

    def rel_crimea(members):
        return any(r in in_crimea for ty, r, _ in members if ty == "w")

    trains = {rid: v for rid, v in rels.items()
              if v[0].get("type") == "route" and v[0].get("route") == "train"}
    ru_trains = {rid: v for rid, v in trains.items() if rel_where(v[1]) == "ru"}
    print(f"\n3. route=train relations: {len(trains)} in the extract, {len(ru_trains)} mostly in ru, "
          f"{sum(1 for v in ru_trains.values() if rel_crimea(v[1]))} touching Crimea")
    svc = Counter(v[0].get("service") or "-" for v in ru_trains.values())
    print("   service tag:", svc.most_common())
    shape_ = Counter()
    for v in ru_trains.values():
        nm = v[0].get("name") or ""
        m = re.match(r"^\s*([^\d:\s]+(?:\s[^\d:\s]+)?)", nm)
        shape_[m.group(1) if m else "(none)"] += 1
    print("   name opening words:", shape_.most_common(20))
    nets = Counter(v[0].get("network") or "-" for v in ru_trains.values())
    print("   network:", nets.most_common(25))
    ops = Counter(v[0].get("operator") or "-" for v in ru_trains.values())
    print("   operator:", ops.most_common(25))

    def classify(t):
        s = (t.get("service") or "").lower()
        nm = (t.get("name") or "").lower()
        net = (t.get("network") or "").lower()
        if s in ("commuter", "regional", "suburban", "local") or any(
                w in nm for w in ("электрич", "пригород", "ласточка", "аэроэкспресс", "мцд",
                                  "экспресс")) or any(w in net for w in ("пригород", "мцд", "цппк")):
            return "suburban"
        if s in ("long_distance", "high_speed", "night", "national", "international") or \
                re.search(r"поезд\s*№|№\s*\d|сапсан|ласточк|^\d+[а-я]?\b", nm):
            return "long-distance"
        return "unclear"
    cl = Counter(classify(v[0]) for v in ru_trains.values())
    print("   classified (service tag, then name/network words):", cl.most_common())
    masters = sum(1 for v in rels.values() if v[0].get("type") == "route_master"
                  and v[0].get("route_master") == "train")
    print(f"   route_master=train in the extract: {masters}")
    samp = sorted(ru_trains.items())[:: max(1, len(ru_trains) // 40)][:40]
    for rid, (t, _m) in samp:
        print(f"     r{rid} {classify(t):<13} svc={t.get('service')!r} net={t.get('network')!r} "
              f"name={t.get('name')!r}")

    # --- urban
    urban = Counter()
    for rid, (t, m) in rels.items():
        if t.get("type") == "route" and t.get("route") in ("subway", "light_rail", "monorail",
                                                            "tram", "funicular"):
            if rel_where(m) != "ru":
                continue
            key = t.get("network") or t.get("operator") or re.split(r"[\d:]", t.get("name") or "?")[0]
            urban[(t.get("route"), key.strip())] += 1
    print(f"\n4. URBAN route relations in ru by (route, network or operator): {len(urban)} groups")
    for (k, n), v in sorted(urban.items(), key=lambda kv: (kv[0][0], -kv[1])):
        if k != "tram" or v >= 6:
            print(f"   {k:<10} {v:4d}  {n}")
    trams = sum(v for (k, _n), v in urban.items() if k == "tram")
    print(f"   tram: {trams} relations in {sum(1 for (k, _n) in urban if k == 'tram')} groups")

    # --- infra relations
    ru_inf = {rid: v for rid, v in infra.items() if rel_where(v[1]) == "ru"}
    print(f"\n5. INFRA relations: {len(infra)} in the extract, {len(ru_inf)} mostly in ru; "
          f"{dict(Counter(v[0].get('route') for v in ru_inf.values()))}")
    refs = Counter(re.sub(r"\d", "9", v[0].get("ref") or "-")[:12] for v in ru_inf.values())
    print("   ref shapes:", refs.most_common(10))
    keys = Counter(k for t, _m in ru_inf.values() for k in t)
    print("   tag keys:", keys.most_common(25))
    for rid, (t, m) in sorted(ru_inf.items(), key=lambda kv: -len(kv[1][1]))[:30]:
        print(f"     r{rid} ways={sum(1 for x in m if x[0] == 'w'):5d} ref={t.get('ref')!r} "
              f"name={t.get('name')!r} op={t.get('operator')!r}")


if __name__ == "__main__":
    main()
