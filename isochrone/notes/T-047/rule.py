"""T-047: the sample's measured gain beside each tile's 1 km figures, and the rule's world count.

    python isochrone/notes/T-047/rule.py <compare.json> <world.csv>
"""
import csv
import json
import sys

# the rule (docs/fine.md, "Where the fine level goes")
URBAN, MIXED, RUGGED, ROAD = 0.01, 0.10, 0.25, 0.10


def picked(t):
    return t["urban"] >= URBAN or t["mixed"] >= MIXED or t["rugged"] >= RUGGED or t["road"] >= ROAD


def main(cj, wc):
    world = {}
    with open(wc) as f:
        for r in csv.DictReader(f):
            world[(int(r["lat"]), int(r["lon"]))] = {k: float(v) for k, v in r.items()}
    sample = json.load(open(cj))
    print("%-22s %-8s %5s %5s %5s %5s %5s | %5s %5s %5s %5s | %s" % (
        "tile", "kind", "land", "urban", "mixed", "rugd", "road", "band", "glo", "only", "gain", "pick"))
    rows = []
    for s in sample:
        w = world.get((s["lat"], s["lon"]))
        p = s["pairs"]
        if not w or "l1_our/f_plain" not in p:
            continue
        a, g = p["l1_our/f_plain"], p.get("l1_our/f_glo", p["l1_our/f_plain"])
        gain = g["bandDiff"] + g["onlyOne"]
        pk = picked(w)
        rows.append((gain, pk))
        print("%-22s %-8s %5.2f %5.3f %5.3f %5.2f %5.2f | %5.2f %5.2f %5.2f %5.2f | %s" % (
            s["label"], s["kind"], w["land"], w["urban"], w["mixed"], w["rugged"], w["road"],
            a["bandDiff"], g["bandDiff"], g["onlyOne"], gain, "yes" if pk else "no"))
        for k in ("l1_map/f_road", "l1_map/f_road_glo"):
            if k in p:
                print("    %-18s band %.2f only %.3f relMedian %.3f ratio %.2f" % (k, p[k]["bandDiff"], p[k]["onlyOne"], p[k]["relMedian"] or 0, p[k]["ratio"]))
    yes = sorted(g for g, k in rows if k)
    no = sorted(g for g, k in rows if not k)
    print("\npicked %d, gain median %.2f min %.2f; left %d, gain median %.2f max %.2f" % (
        len(yes), yes[len(yes) // 2], yes[0], len(no), no[len(no) // 2] if no else 0, no[-1] if no else 0))
    n = sum(1 for t in world.values() if picked(t))
    by = {"urban": 0, "mixed": 0, "rugged": 0, "road": 0}
    for t in world.values():
        if t["urban"] >= URBAN: by["urban"] += 1
        if t["mixed"] >= MIXED: by["mixed"] += 1
        if t["rugged"] >= RUGGED: by["rugged"] += 1
        if t["road"] >= ROAD: by["road"] += 1
    print("world: %d land tiles, %d picked (%s)" % (len(world), n, by))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
