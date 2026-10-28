import json, sys, collections
MB = 1 << 20
d = json.load(open(sys.argv[1]))
key = sys.argv[2] if len(sys.argv) > 2 else "final"
r = d[key]
rep = [x for x in r["diag"]["full"] if not x["aux"] and x["kind"] == "done" and x["cities"]][-1]
tags = sorted(rep["tags"], key=lambda t: -max(t[1], t[3]))
for name, live, mx, ap in tags[:40]:
    print(f"{name:60} live {live/MB:8.1f} max {mx/MB:8.1f} atpeak {ap/MB:8.1f}")
by = collections.Counter()
for name, live, mx, ap in rep["tags"]:
    by[name.split("@")[0]] += live
print()
for k, v in by.most_common(30):
    print(f"{k:50} {v/MB:8.1f}")
