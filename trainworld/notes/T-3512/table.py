"""Settle times per run and edit, and whether each edit's end figures match across runs (rail, trips, intercity, stations hash, every line's riders and loads)."""
import json
import sys
import glob
import os

OUT = os.path.join(os.path.dirname(__file__), "out")
names = sys.argv[1:]
runs = {n: json.load(open(os.path.join(OUT, n + ".json"), encoding="utf-8")) for n in names}
for n, j in runs.items():
    print("%-12s load %6.1f  " % (n, j.get("final") or 0) + "  ".join("%s %6.1f" % (e["kind"], e["settle"]) for e in j["edits"]) + ("  ERR " + j["error"][:80] if j.get("error") else ""))


def key(f):
    return json.dumps([f["rail"], f["trips"], f["icRail"], f["icTrips"], f["stH"], f["split"], sorted(f["lines"].items()), f["points"]], sort_keys=True)


ref = runs[names[0]]
for n in names[1:]:
    j = runs[n]
    same = []
    for a, b in zip(ref["edits"], j["edits"]):
        same.append("%s:%s" % (a["kind"], "same" if key(a["fig"]) == key(b["fig"]) else "DIFF rail %r vs %r" % (a["fig"]["rail"], b["fig"]["rail"])))
    print(n, "vs", names[0], "load:", "same" if key(ref["fig0"]) == key(j["fig0"]) else "DIFF", " ".join(same))
