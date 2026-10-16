"""Seoul Line 9's express share of riders in a calib run (seoulriders' published 40.2%, person-stops
over the line's segments). python line9.py <run.json>"""
import json
import sys

run = json.load(open(sys.argv[1]))
net = json.load(open(run["net"]))
L = run["lines"]
tot = {}
for i, l in enumerate(net["lines"]):
    name = " ".join(str(l.get(k, "")) for k in ("name", "ref", "id", "pattern"))
    if "9" in str(l.get("name", "")) or "9" in str(l.get("ref", "")):
        print(i, {k: l[k] for k in l if k not in ("stops", "shape", "geom", "coords", "times")}, "load", L["load"][i], "pkm", L["pkm"][i], "board", L["board"][i])
