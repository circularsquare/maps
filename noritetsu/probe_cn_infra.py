"""Stage-1 measurements for mainland China: can OSM's infrastructure relations be the register?

    python probe_cn_infra.py > data/raw/cn/probe_infra.txt

Measures, on the extract noritetsu builds from (data/proc/cn, after cn_register.py --clip):

1. Heavy-rail main and branch track km (build_tiles.rank_of < 2), and the part of it that a
   passenger route relation (route=train in rels.pkl) runs over -- "passenger-used".
2. Of each: the share lying in some route=railway/route=tracks relation (infra.pkl), in one
   whose ref starts with a four-digit national line code ("0002 京沪线"), and the share whose
   own `name` tag is a line name (ends in 线 or 铁路, not a structure).
3. The infrastructure relations themselves: how many, with a code, high-speed by name.
4. Wikidata's station chains (data/raw/cn/wd_*.json, cn_register.py --fetch) against OSM's
   stations: how many chained stations have an OSM station of the same name within 2 km.
5. Metro systems: OSM route=subway/light_rail/monorail relations by network.
"""
import json
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "proc" / "cn"
RAW = ROOT / "data" / "raw" / "cn"
CODE = re.compile(r"^\s*(\d{4})\b")
LINE_NAME = re.compile(r"(线|線|铁路|鐵路|支线|联络线|客专|客运专线)$")
STRUCT = re.compile(r"(隧道|大桥|特大桥|桥|高架)$")
HS = re.compile(r"(高速铁路|高铁|客运专线|客专|城际)")


def main():
    import build_tiles as bt
    with open(D / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(D / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(D / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    with open(D / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    c = np.load(D / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]

    def way_km(nodes):
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size < 2:
            return 0.0, None
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        km = float(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32,
                            np.diff(lat) * 110.57).sum())
        return km, (float(lon[len(lon) // 2]), float(lat[len(lat) // 2]))

    # --- ways in passenger routes, by route kind
    in_route = defaultdict(set)
    route_kinds = Counter()
    for rid, (tags, members) in rels.items():
        if tags.get("type") != "route":
            continue
        k = tags.get("route")
        route_kinds[k] += 1
        for ty, ref, _role in members:
            if ty == "w":
                in_route[ref].add(k)
    # --- ways in infra relations
    in_infra, in_code, infra_name = set(), set(), {}
    hs_rel = 0
    code_rel = 0
    route_type = Counter()
    for rid, (tags, members) in infra.items():
        route_type[tags.get("route")] += 1
        nm = tags.get("name") or ""
        m = CODE.match(tags.get("ref") or "") or CODE.match(nm)
        if m:
            code_rel += 1
        if HS.search(nm) or tags.get("highspeed") == "yes":
            hs_rel += 1
        for ty, ref, _role in members:
            if ty == "w":
                in_infra.add(ref)
                infra_name.setdefault(ref, nm)
                if m:
                    in_code.add(ref)

    tot = Counter()
    for wid, (tags, nodes) in ways.items():
        kind = bt.KIND[tags["railway"]]
        if kind not in ("rail", "narrow_gauge", "heritage"):
            if bt.rank_of(kind, tags) < 2:
                km, _ = way_km(nodes)
                tot[("urban", "all")] += km
                if wid in in_route:
                    tot[("urban", "pax")] += km
                    if tags.get("name"):
                        tot[("urban", "pax_named")] += km
            continue
        rank = bt.rank_of(kind, tags)
        if rank >= 2:
            continue
        km, _mid = way_km(nodes)
        pax = "train" in in_route.get(wid, ())
        name = (tags.get("name") or "").strip()
        named = bool(LINE_NAME.search(name)) and not STRUCT.search(name)
        for scope in ("all", "pax") if pax else ("all",):
            tot[(scope, "km")] += km
            if wid in in_infra:
                tot[(scope, "infra")] += km
            if wid in in_code:
                tot[(scope, "code")] += km
            if named:
                tot[(scope, "named")] += km
            if named or wid in in_infra:
                tot[(scope, "either")] += km
            if tags.get("highspeed") == "yes":
                tot[(scope, "hs")] += km
                if wid in in_infra:
                    tot[(scope, "hs_infra")] += km
                if named:
                    tot[(scope, "hs_named")] += km
            if rank == 0:
                tot[(scope, "main")] += km
                if wid in in_infra:
                    tot[(scope, "main_infra")] += km

    print("HEAVY RAIL (rail, narrow gauge, preserved), rank < 2")
    for scope, label in (("all", "all main + branch track"),
                         ("pax", "passenger-used (in a route=train relation)")):
        k = tot[(scope, "km")]
        print(f"  {label}: {k:,.0f} km")
        for key, what in (("infra", "in a route=railway/tracks relation"),
                          ("code", "  ... whose ref/name starts with a 4-digit line code"),
                          ("named", "with a line name on the way itself"),
                          ("either", "in a relation OR named"),
                          ("main", "usage=main"), ("main_infra", "  usage=main in a relation"),
                          ("hs", "highspeed=yes"), ("hs_infra", "  highspeed=yes in a relation"),
                          ("hs_named", "  highspeed=yes named")):
            base = tot[(scope, "hs")] if key.startswith("hs_") else (
                tot[(scope, "main")] if key == "main_infra" else k)
            print(f"    {what:<52} {tot[(scope, key)]:>9,.0f} km  "
                  f"{100 * tot[(scope, key)] / base if base else 0:5.1f}%")
    print(f"\nURBAN (subway, light rail, tram, monorail, funicular): {tot[('urban', 'all')]:,.0f} km, "
          f"of which in a route relation {tot[('urban', 'pax')]:,.0f}, named "
          f"{tot[('urban', 'pax_named')]:,.0f}")
    print(f"\nroute relations by kind: {dict(route_kinds)}")

    print(f"\nINFRA RELATIONS: {len(infra)} ({dict(route_type)}); with a 4-digit code "
          f"{code_rel}; high-speed by name or tag {hs_rel}")
    refs = Counter()
    for rid, (tags, members) in infra.items():
        r = tags.get("ref") or ""
        refs[re.sub(r"\d", "9", r)[:12]] += 1
    print("  ref shapes:", refs.most_common(12))
    samp = sorted(infra.items(), key=lambda kv: kv[1][0].get("ref") or "zzz")
    for rid, (tags, members) in samp[:25]:
        print(f"   r{rid} ref={tags.get('ref')!r} name={tags.get('name')!r} "
              f"op={tags.get('operator')!r} hs={tags.get('highspeed')!r} "
              f"ways={sum(1 for m in members if m[0] == 'w')}")
    keys = Counter(k for tags, _m in infra.values() for k in tags)
    print("  tag keys:", keys.most_common(30))
    hs_names = sorted({t.get("name") for t, _m in infra.values()
                       if HS.search(t.get("name") or "")})
    print(f"  high-speed-looking names ({len(hs_names)}):", " ".join(hs_names[:80]))

    # --- top way names on passenger-used heavy rail that are in no relation
    orphan = Counter()
    for wid, (tags, nodes) in ways.items():
        if tags["railway"] not in ("rail", "narrow_gauge", "preserved"):
            continue
        if "train" in in_route.get(wid, ()) and wid not in in_infra \
                and bt.rank_of(bt.KIND[tags["railway"]], tags) < 2:
            orphan[tags.get("name") or "(unnamed)"] += way_km(nodes)[0]
    print("\npassenger-used heavy rail in no relation, by way name (km):")
    for n, km in orphan.most_common(40):
        print(f"  {km:8.1f}  {n}")

    # --- route=train relation names
    tn = Counter()
    ex = []
    for rid, (tags, members) in rels.items():
        if tags.get("route") == "train":
            nm = tags.get("name") or ""
            tn[re.sub(r"\d+", "9", nm)[:14]] += 1
            if len(ex) < 40 and rid % 7 == 0:
                ex.append((nm, tags.get("ref"), tags.get("network"), tags.get("operator")))
    print("\nroute=train name shapes:", tn.most_common(30))
    for e in ex:
        print("   ", e)

    # --- metros by network
    nets = Counter()
    for rid, (tags, members) in rels.items():
        if tags.get("type") == "route" and tags.get("route") in ("subway", "light_rail",
                                                                  "monorail", "tram"):
            nets[(tags.get("route"), tags.get("network") or
                  re.split(r"\d|[A-Z]|号|線|线", tags.get("name") or "?")[0])] += 1
    print(f"\nurban route relations by (kind, network or name prefix): {len(nets)} groups")
    for (k, n), v in sorted(nets.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {k:<10} {v:4d}  {n}")

    # --- Wikidata station chains against OSM stations
    wd_match(stops)


def nkey(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", "", s)
    for suf in ("火车站", "高铁站", "站"):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
            break
    return s


def wd_match(stops):
    tail = lambda u: u.rsplit("/", 1)[-1]
    adj = json.loads((RAW / "wd_adjacency.json").read_text("utf-8"))["rows"]
    strows = json.loads((RAW / "wd_stations.json").read_text("utf-8"))["rows"]
    lines = json.loads((RAW / "wd_lines.json").read_text("utf-8"))["rows"]
    lab = defaultdict(dict)
    pt = {}
    for r in strows:
        s = tail(r["s"])
        if r.get("lab"):
            lab[s].setdefault(r["lang"], r["lab"])
        if r.get("coord") and s not in pt:
            m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", r["coord"])
            if m:
                pt[s] = (float(m.group(1)), float(m.group(2)))
    linelab = defaultdict(dict)
    for r in lines:
        if r.get("lab"):
            linelab[tail(r["x"])].setdefault(r["lang"], r["lab"])
    per = defaultdict(set)
    for r in adj:
        per[tail(r["line"])].add((tail(r["s"]), tail(r["a"])))
    # chained = one connected piece of 3+ stations
    chained = {}
    for ln, e in per.items():
        par = {}

        def f(x):
            while par.setdefault(x, x) != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for a, b in e:
            par[f(a)] = f(b)
        if len({f(x) for x in list(par)}) == 1 and len(par) >= 3:
            chained[ln] = set(par)
    # OSM stations by name key
    osm = defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("railway") in ("station", "halt") or tags.get("public_transport") == "station":
            if tags.get("name"):
                osm[nkey(tags["name"])].append((lon, lat))
    import math

    def found(s):
        names = [lab[s].get(l) for l in ("zh-cn", "zh-hans", "zh", "zh-hant")]
        names = [n for n in names if n]
        p = pt.get(s)
        for n in names:
            for lon, lat in osm.get(nkey(n), ()):
                if p is None:
                    return True
                d = math.hypot((lon - p[0]) * math.cos(math.radians(lat)) * 111.32,
                               (lat - p[1]) * 110.57)
                if d <= 2.0:
                    return True
        return False
    st_all = {s for v in chained.values() for s in v}
    ok = {s for s in st_all if found(s)}
    print(f"\nWIKIDATA: {len(per)} lines with adjacency, {len(chained)} chained (3+ stations, one "
          f"piece), {len(st_all)} stations on them; {len(ok)} ({100 * len(ok) / max(1, len(st_all)):.1f}%) "
          f"have an OSM station of the same name within 2 km; {sum(1 for s in st_all if s in pt)} "
          f"have a point")
    full = sum(1 for v in chained.values() if len(v & ok) >= 0.9 * len(v))
    print(f"  chained lines with 90%+ of their stations found in OSM: {full}")
    big = sorted(chained.items(), key=lambda kv: -len(kv[1]))[:40]
    for ln, v in big:
        nm = linelab[ln].get("zh-cn") or linelab[ln].get("zh-hans") or linelab[ln].get("zh") or ln
        print(f"   {len(v):4d} stations, {len(v & ok):4d} in OSM  {nm}")


if __name__ == "__main__":
    main()
