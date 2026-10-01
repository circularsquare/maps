"""Russia stage 1: which tariff points do OSM's passenger train routes actually stop at?

    python probe_ru_stops.py > data/raw/ru/probe_stops.txt

Tariff Guide No. 4 flags a point П/Б/О when passenger operations are permitted there, which is
not the same as a train calling today. OSM's route=train relations (3,900 in Russia, most of
them suburban) list their stops, so: an ESR point found in OSM (data/raw/ru/osm_esr.json) is
"served" when a stop or platform member of some route=train relation lies within 400 m of it.
Counted by the TR-4 passenger flag and by section type, for Russia + Crimea.
"""
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "proc" / "ru"
RAW = ROOT / "data" / "raw" / "ru"
OCCUPIED_2022 = {"Донец (Р)", "ЛУГАН (Р)", "МЕЛИТ (Р)"}
R_M = 400


def main():
    import xlrd
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(ROOT))
    import probe_ru_tr4 as t4

    rp, op = t4.book2()
    secs = [s for s in json.loads((RAW / "tr4_sections.json").read_text("utf-8"))
            if s["sheet"] not in OCCUPIED_2022]
    osm_esr = json.loads((RAW / "osm_esr.json").read_text("utf-8"))
    with open(D / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(D / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    c = np.load(D / "coords.npz")
    cid, cx, cy = c["id"], c["x"] / 1e7, c["y"] / 1e7

    def xy(lon, lat):
        return (lon * 111.32 * math.cos(math.radians(lat)), lat * 110.57)

    kinds = Counter()
    pts = []
    for rid, (tags, members) in rels.items():
        if tags.get("type") != "route" or tags.get("route") != "train":
            continue
        sub = "suburban" if (tags.get("service") in ("regional", "commuter") or
                             re.search(r"пригород|электропоезд|дизельпоезд|мцд",
                                       (tags.get("name") or "").lower())) else "long"
        for ty, ref, role in members:
            if ty == "n" and (role.startswith("stop") or role.startswith("platform")):
                if ref in stops:
                    lon, lat = stops[ref][1], stops[ref][2]
                else:
                    i = np.searchsorted(cid, ref)
                    if i >= cid.size or cid[i] != ref:
                        continue
                    lon, lat = cx[i], cy[i]
                pts.append((xy(lon, lat), sub))
                kinds[sub] += 1
    tree = cKDTree(np.array([p[0] for p in pts]))
    subs = [p[1] for p in pts]
    print(f"route=train stop/platform node members: {len(pts):,} ({dict(kinds)})")

    served = {}
    for code, rows in osm_esr.items():
        lon, lat = rows[0][2], rows[0][3]
        hits = tree.query_ball_point(xy(lon, lat), R_M / 1000)
        served[code] = (bool(hits), {subs[h] for h in hits})

    codes = {}
    for s in secs:
        for p in s["points"]:
            codes.setdefault(p["esr"], (p["name"], s["type"]))
    tab = Counter()
    for code, (name, typ) in codes.items():
        flag = t4.passenger(code, rp, op) or "absent"
        if code not in osm_esr:
            tab[(flag, "not in OSM")] += 1
            continue
        hit, which = served[code]
        tab[(flag, "served" if hit else "no route stops")] += 1
        if hit:
            tab[(flag, "served by long-distance")] += "long" in which
    print(f"\nTR-4 points (Russia + Crimea): {len(codes):,}; with an OSM esr:user node "
          f"{sum(1 for k in codes if k in osm_esr):,}")
    crim = {p["esr"] for s in secs if s["sheet"] == "Крым (Р)" for p in s["points"]}
    print(f"  Crimea: {len(crim)} points, {len(crim & set(osm_esr))} found in OSM by their "
          f"post-2014 code (none by the Ukrainian code in osm.sbin.ru's crimea.csv)")
    for flag in ("stop", "none", "absent"):
        tot = sum(v for (f, k), v in tab.items() if f == flag and k != "served by long-distance")
        print(f"  TR-4 flag {flag!r:<9} {tot:6,}: served {tab[(flag, 'served')]:6,}  "
              f"(long-distance too {tab[(flag, 'served by long-distance')]:,}), "
              f"no route stops nearby {tab[(flag, 'no route stops')]:6,}, "
              f"no ESR node in OSM {tab[(flag, 'not in OSM')]:6,}")

    # per section: share of its passenger-flagged points that are served
    sec_tab = Counter()
    unserved = []
    for s in secs:
        flagged = [p for p in s["points"] if t4.passenger(p["esr"], rp, op) == "stop"
                   and p["esr"] in osm_esr]
        if not flagged:
            continue
        n_srv = sum(1 for p in flagged if served[p["esr"]][0])
        L = max((p["km"][0] or 0) for p in s["points"])
        share = n_srv / len(flagged)
        b = "all" if share == 1 else "none" if share == 0 else "some"
        sec_tab[(b, "n")] += 1
        sec_tab[(b, "km")] += L
        if b == "none":
            unserved.append((L, s["id"], s["name"], s["type"]))
    print("\nsections by how many of their passenger-flagged points OSM routes serve:")
    for b in ("all", "some", "none"):
        print(f"  {b:<5} {sec_tab[(b, 'n')]:5,} sections, {sec_tab[(b, 'km')]:8,} tariff km")
    unserved.sort(reverse=True)
    print("  longest with none served (freight-only, or routes missing in OSM):")
    for x in unserved[:40]:
        print(f"    {x[0]:5d} km  {x[1]:<7} {x[2]}  [{x[3]}]")


if __name__ == "__main__":
    main()
