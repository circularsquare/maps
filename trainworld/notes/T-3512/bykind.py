"""Two runs' end figures compared by edit kind and its order among edits of that kind (full key: rail, trips,
intercity, stations hash, split, every line's riders/boardings/loads hash, points)."""
import json
import os
import sys
from collections import defaultdict

OUT = os.path.join(os.path.dirname(__file__), "out")


def key(f):
    return json.dumps([f["rail"], f["trips"], f["icRail"], f["icTrips"], f["stH"], f["split"], sorted(f["lines"].items()), f["points"]], sort_keys=True)


def bykind(j):
    seen, out, prev = defaultdict(int), {}, None
    for e in j["edits"]:
        k = e["kind"] if e["kind"] != "undo" else "undo-of-" + prev
        out[(k, seen[k])] = e["fig"]
        seen[k] += 1
        prev = e["kind"] if e["kind"] != "undo" else prev
    return out


a, b = (json.load(open(os.path.join(OUT, n + ".json"), encoding="utf-8")) for n in sys.argv[1:3])
print("load", "same" if key(a["fig0"]) == key(b["fig0"]) else "DIFF")
ka, kb = bykind(a), bykind(b)
for k in ka:
    if k in kb:
        print(k, "same" if key(ka[k]) == key(kb[k]) else "DIFF rail %r vs %r" % (ka[k]["rail"], kb[k]["rail"]))
