"""Tie OpenStreetMap ferry lines to the statistics ports they serve.

Inputs: raw/osm_ferries.json (route=ferry ways), raw/osm_ferry_terminals.json
(amenity=ferry_terminal), data/ports_2024.geojson (port statistics on C02
points), and C02 itself.

1. Way ends are snapped to the nearest ferry terminal within SNAP_KM, and
   terminals lying on a way (within STOP_KM of a vertex) count as intermediate
   stops, so a multi-stop line drawn as one way still reaches its island ports.
2. Each anchor is assigned to a statistics port: AREA_PORT first (everything
   within a radius of a named terminal), then TERMINAL_PORT (terminals named
   after their destination), then a port sharing a 2+ character place name with
   the terminal within NAME_KM, else the nearest port within NEAR_KM. C02 gives
   one point per port district, which in a big port can sit 10 km from the pier
   (新潟, 苫小牧), so names come before distance.
3. Ways meeting at an unanchored end (out at sea) are joined into one route.
4. When several anchors of one route fall to the same port by distance alone,
   only the nearest keeps it; the rest stay as their own terminal. Otherwise a
   mainland pier the statistics don't count (宮島口) would merge into the island
   port and the route would vanish as a loop.
5. Routes with an anchor more than JAPAN_KM from any C02 port are international,
   and routes whose line names match SIGHTSEEING are sightseeing. Both are kept
   in the output, flagged, and left out of the partner counts.

Island piers are often fishing harbours (漁港), which the port statistics don't
cover, so such an anchor stays as its bare terminal.

Writes data/osm_ferry_ways.geojson and prints how many ports have one partner
port, several, or none, then the terminals assigned to each busy port for review.
"""
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
DATA = os.path.join(HERE, "data")
SNAP_KM = 1.0
STOP_KM = 0.3
NAME_KM = 20.0
NEAR_KM = 8.0
JAPAN_KM = 30.0
REVIEW_MIN = 400_000

GENERIC = ["フェリーターミナル", "旅客ターミナル", "ターミナル", "フェリー", "高速船", "旅客", "桟橋",
           "乗り場", "のりば", "乗船場", "待合所", "渡船場", "渡場", "渡し", "営業所", "発着所", "港"]
# not 観光船: 名鉄海上観光船 runs the scheduled 三河湾 island routes
SIGHTSEEING = ["クルーズ", "Cruise", "cruise", "遊覧", "めぐり", "巡り", "海賊船", "遊船", "屋形船",
               "川下り", "疏水船", "イルカ", "潮流体験", "Transit Steamer"]

# terminal name -> (prefecture, statistics port), for terminals named after where they go
TERMINAL_PORT = {
    "桜島フェリーターミナル": ("鹿児島", "鹿児島"),
}
# (terminal name, radius km, prefecture, statistics port): everything within the radius of
# that terminal (the one nearest the port, if the name repeats) belongs to the port
AREA_PORT = [
    ("北埠頭旅客ターミナル", 2.0, "鹿児島", "鹿児島"),  # 本港 north/south piers and 新港; 桜島港 is 3.4 km east
    ("佐渡汽船フェリーターミナル", 1.5, "新潟", "新潟"),  # 10 km from the C02 point
]


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def norm(s):
    s = s or ""
    for a, b in (("奧", "奥"), ("ヶ", "ケ"), ("　", ""), (" ", ""), ("（", "("), ("）", ")")):
        s = s.replace(a, b)
    return s


def base_name(port):
    return norm(port).split("(")[0]


def strip_generic(name):
    s = norm(name)
    for g in GENERIC:
        s = s.replace(g, "|")
    return s


def shares_place(a, b):
    """True if a and b share a run of 2+ characters (no '|' separators)."""
    for i in range(len(a) - 1):
        chunk = a[i:i + 2]
        if "|" not in chunk and chunk in b:
            return True
    return False


class Grid:
    def __init__(self, items, cell=0.02):
        self.cell = cell
        self.cells = defaultdict(list)
        for it in items:
            self.cells[(int(it["lat"] // cell), int(it["lon"] // cell))].append(it)

    def near(self, lat, lon, radius_km):
        ci, cj = int(lat // self.cell), int(lon // self.cell)
        reach = max(1, int(radius_km / 111 / self.cell) + 1)
        out = []
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                for it in self.cells.get((ci + di, cj + dj), ()):
                    d = km(lat, lon, it["lat"], it["lon"])
                    if d <= radius_km:
                        out.append((d, it))
        return sorted(out, key=lambda x: x[0])


def load():
    with open(os.path.join(DATA, "ports_2024.geojson"), encoding="utf-8") as f:
        stat_ports = [
            {"key": ft["properties"]["c02_code"], "name": ft["properties"]["port"],
             "pref": ft["properties"]["prefecture"], "dom_total": ft["properties"]["dom_total"],
             "lon": ft["geometry"]["coordinates"][0], "lat": ft["geometry"]["coordinates"][1]}
            for ft in json.load(f)["features"]
        ]
    c02 = gpd.read_file(os.path.join(RAW, "C02-14", "C02-14_GML", "C02-14-g_PortAndHarbor.shp"), encoding="cp932")
    c02_ports = [{"key": r.C02_004, "name": r.C02_005, "lon": r.geometry.x, "lat": r.geometry.y}
                 for r in c02.itertuples()]
    with open(os.path.join(RAW, "osm_ferry_terminals.json"), encoding="utf-8") as f:
        terminals = []
        for e in json.load(f)["elements"]:
            c = e if "lat" in e else e.get("center")
            if c:
                terminals.append({"id": f"{e['type'][0]}{e['id']}", "name": e.get("tags", {}).get("name", ""),
                                  "lat": c["lat"], "lon": c["lon"]})
    with open(os.path.join(RAW, "osm_ferries.json"), encoding="utf-8") as f:
        ways = [e for e in json.load(f)["elements"] if e["type"] == "way" and len(e.get("geometry", [])) >= 2]
    return stat_ports, c02_ports, terminals, ways


def main():
    stat_ports, c02_ports, terminals, ways = load()
    port_grid, c02_grid, term_grid = Grid(stat_ports), Grid(c02_ports), Grid(terminals)
    by_key = {p["key"]: p for p in stat_ports}
    by_prefname = {(p["pref"], p["name"]): p for p in stat_ports}

    areas = []
    for tname, radius, pref, port in AREA_PORT:
        target = by_prefname.get((pref, port))
        cands = [t for t in terminals if norm(t["name"]) == norm(tname)]
        if not target or not cands:
            print(f"  ! area override not found: {tname} -> {pref} {port}")
            continue
        t = min(cands, key=lambda t: km(t["lat"], t["lon"], target["lat"], target["lon"]))
        areas.append((t["lat"], t["lon"], radius, target))

    def resolve(lat, lon, term):
        """-> (stat port or None, how, distance km)"""
        for alat, alon, radius, target in areas:
            if km(lat, lon, alat, alon) <= radius:
                return target, "area", km(lat, lon, target["lat"], target["lon"])
        if term:
            fixed = TERMINAL_PORT.get(norm(term["name"]))
            if fixed and fixed in by_prefname:
                p = by_prefname[fixed]
                return p, "override", km(lat, lon, p["lat"], p["lon"])
            stripped = strip_generic(term["name"])
            for d, p in port_grid.near(lat, lon, NAME_KM):
                b = base_name(p["name"])
                if len(b) >= 2 and shares_place(b, stripped):
                    return p, "name", d
        hits = port_grid.near(lat, lon, NEAR_KM)
        if hits:
            return hits[0][1], "near", hits[0][0]
        return None, None, None

    def anchor(lat, lon, term, kind):
        port, how, dist = resolve(lat, lon, term)
        return {
            "kind": kind, "lat": lat, "lon": lon, "term": term["id"] if term else None,
            "tname": term["name"] if term else "", "stat": port["key"] if port else None, "how": how,
            "dist": dist, "foreign": not c02_grid.near(lat, lon, JAPAN_KM),
            "own": term["id"] if term else f"{lat:.3f},{lon:.3f}",
        }

    def end(pt):
        return (round(pt["lat"], 5), round(pt["lon"], 5))

    way_ends = [(end(w["geometry"][0]), end(w["geometry"][-1])) for w in ways]
    touching = defaultdict(list)
    for i, (a, b) in enumerate(way_ends):
        touching[a].append(i)
        touching[b].append(i)

    end_anchor = {}
    for k, idx in touching.items():
        hits = term_grid.near(k[0], k[1], SNAP_KM)
        term = hits[0][1] if hits else None
        a = anchor(k[0], k[1], term, "end")
        a["anchored"] = term is not None or a["stat"] is not None or len(idx) != 2
        end_anchor[k] = a

    stops = defaultdict(dict)  # way index -> terminal id -> anchor
    for i, w in enumerate(ways):
        endpoint_terms = {end_anchor[e]["term"] for e in way_ends[i]}
        for pt in w["geometry"][1:-1]:
            for _, t in term_grid.near(pt["lat"], pt["lon"], STOP_KM):
                if t["id"] not in endpoint_terms and t["id"] not in stops[i]:
                    stops[i][t["id"]] = anchor(t["lat"], t["lon"], t, "stop")

    parent = list(range(len(ways)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k, idx in touching.items():
        if not end_anchor[k]["anchored"]:
            for j in idx[1:]:
                parent[find(j)] = find(idx[0])
    comps = defaultdict(list)
    for i in range(len(ways)):
        comps[find(i)].append(i)

    partners = defaultdict(set)
    assigned = defaultdict(set)  # stat key -> terminal labels, for review
    features = []
    counts = defaultdict(int)
    for cid, members in comps.items():
        anchors = [end_anchor[e] for i in members for e in way_ends[i] if end_anchor[e]["anchored"]]
        anchors += [a for i in members for a in stops[i].values()]
        seen, uniq = set(), []
        for a in anchors:
            ident = a["term"] or a["own"]
            if ident not in seen:
                seen.add(ident)
                uniq.append(dict(a))
        anchors = uniq
        if not anchors or all(a["foreign"] for a in anchors):
            continue

        by_stat = defaultdict(list)
        for a in anchors:
            if a["stat"]:
                by_stat[a["stat"]].append(a)
        for stat, group in by_stat.items():
            if len(group) < 2:
                continue
            keeper = min(group, key=lambda a: (a["how"] == "near", a["dist"]))
            for a in group:
                if a is keeper or a["how"] != "near":
                    continue
                if km(a["lat"], a["lon"], keeper["lat"], keeper["lon"]) > 1.0:
                    a["stat"], a["how"] = None, "demoted"

        way_names = [ways[i].get("tags", {}).get("name") or "" for i in members]
        international = any(a["foreign"] for a in anchors)
        sightseeing = any(word in n for n in way_names for word in SIGHTSEEING)
        counts["routes"] += 1
        counts["international"] += international
        counts["sightseeing"] += sightseeing and not international
        keys = {a["stat"] or a["own"] for a in anchors}
        for a in anchors:
            if a["stat"]:
                assigned[a["stat"]].add(f"{a['tname'] or '(no terminal)'} [{a['how']} {a['dist']:.1f}km]")
                if not international and not sightseeing:
                    partners[a["stat"]] |= keys - {a["stat"]}
        labels = sorted({(by_key[a["stat"]]["name"] if a["stat"] else (a["tname"] or a["own"])) for a in anchors})
        for i in members:
            w = ways[i]
            tags = w.get("tags", {})
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in w["geometry"]]},
                "properties": {"osm_id": w["id"], "name": tags.get("name"), "operator": tags.get("operator"),
                               "route_group": cid, "ports": " / ".join(labels),
                               "international": international, "sightseeing": sightseeing},
            })

    path = os.path.join(DATA, "osm_ferry_ways.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)
    print(f"{len(features)} ways in {counts['routes']} joined routes "
          f"({counts['international']} international, {counts['sightseeing']} sightseeing) -> {path}")

    # A port with exactly one partner gives that route's passengers. Where both ends are
    # counted ports, the two figures are the same route counted twice, so their gap measures
    # how well the shortcut works.
    pairs = {}
    for sp in stat_ports:
        ks = partners[sp["key"]]
        if len(ks) == 1:
            other = next(iter(ks))
            pairs.setdefault(tuple(sorted((sp["key"], other))), set()).add(sp["key"])
    rows = []
    for (a, b), singles in sorted(pairs.items()):
        pa, pb = by_key.get(a), by_key.get(b)
        known = pa if a in singles else pb
        other = pb if known is pa else pa
        both = pa is not None and pb is not None and {a, b} <= singles
        gap = ""
        if both:
            gap = round(abs(pa["dom_total"] - pb["dom_total"]) / max(pa["dom_total"], pb["dom_total"]) * 100, 1)
        rows.append({
            "port": f"{known['pref']} {known['name']}", "passengers": known["dom_total"],
            "other_end": f"{other['pref']} {other['name']}" if other else (b if known is pa else a),
            "other_passengers": other["dom_total"] if other else "",
            "both_single": "yes" if both else "", "gap_pct": gap,
        })
    path_pairs = os.path.join(DATA, "routes_from_ports.csv")
    with open(path_pairs, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gaps = sorted(r["gap_pct"] for r in rows if r["both_single"])
    print(f"\n{len(rows)} routes read straight off the port statistics -> {path_pairs}")
    if gaps:
        print(f"  {len(gaps)} of them counted at both ends; gap between the two counts: "
              f"median {gaps[len(gaps) // 2]}%, worst {gaps[-1]}%")
        for r in sorted((r for r in rows if r["both_single"]), key=lambda r: -r["gap_pct"])[:8]:
            print(f"    {r['gap_pct']:>5}%  {r['port']} {r['passengers']:,} vs {r['other_end']} {r['other_passengers']:,}")

    classes = {"no OSM line": [], "one partner port": [], "two or more partner ports": []}
    for sp in stat_ports:
        n = len(partners[sp["key"]])
        cls = "no OSM line" if n == 0 else "one partner port" if n == 1 else "two or more partner ports"
        classes[cls].append(sp)
    total = sum(sp["dom_total"] for sp in stat_ports)
    for cls, ports in classes.items():
        pax = sum(sp["dom_total"] for sp in ports)
        print(f"  {cls:<27} {len(ports):>4} ports  {pax:>12,} boardings+landings  {pax / total:.1%}")

    def names(keys):
        return ", ".join(sorted(by_key[k]["name"] if k in by_key else k for k in keys))

    print("\nbusiest ports with no OSM line:")
    for sp in sorted(classes["no OSM line"], key=lambda s: -s["dom_total"])[:20]:
        print(f"  {sp['dom_total']:>10,}  {sp['pref']} {sp['name']}")
    print("\nbusiest one-partner ports:")
    for sp in sorted(classes["one partner port"], key=lambda s: -s["dom_total"])[:30]:
        print(f"  {sp['dom_total']:>10,}  {sp['pref']} {sp['name']}  -> {names(partners[sp['key']])}")
    print(f"\nreview: ports with {REVIEW_MIN:,}+ boardings+landings and their partners")
    for sp in sorted(stat_ports, key=lambda s: -s["dom_total"]):
        if sp["dom_total"] < REVIEW_MIN:
            break
        print(f"  {sp['pref']} {sp['name']} ({sp['dom_total']:,}): {names(partners[sp['key']])[:160]}")


if __name__ == "__main__":
    main()
