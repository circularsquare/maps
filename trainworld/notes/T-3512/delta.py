"""Each edit's rail trips minus the load's, per run, side by side (runs on different game json keep the same deltas
if the edits move the figures the same)."""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), "out")
runs = [json.load(open(os.path.join(OUT, n + ".json"), encoding="utf-8")) for n in sys.argv[1:]]
print("load rail", [r["fig0"]["rail"] for r in runs])
for k in range(len(runs[0]["edits"])):
    row = [r["edits"][k]["fig"]["rail"] - r["fig0"]["rail"] for r in runs]
    print("%-5s" % runs[0]["edits"][k]["kind"], " ".join("%.6f" % v for v in row), "diff %.2e" % (max(row) - min(row)))
