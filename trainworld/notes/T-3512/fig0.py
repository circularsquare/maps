"""The load's final figures of each run: rail trips, all trips, intercity rail, stations hash."""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), "out")
for n in sys.argv[1:]:
    j = json.load(open(os.path.join(OUT, n + ".json"), encoding="utf-8"))
    f = j.get("fig0") or {}
    print("%-12s rail %r trips %r icRail %r stH %r lines %d" % (n, f.get("rail"), f.get("trips"), f.get("icRail"), f.get("stH"), len(f.get("lines", {}))))
