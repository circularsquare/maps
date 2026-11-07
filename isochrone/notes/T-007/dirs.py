"""T-007: how the served crossing treats the cells proto.py splits (which directions are crossed).

    python isochrone/notes/T-007/dirs.py <proto out dir> [on|off]
"""
import glob
import json
import os
import sys

d = sys.argv[1]
roads = sys.argv[2] if len(sys.argv) > 2 else "on"
tot = {"split": 0, "all4": 0, "some": 0, "none": 0, "slots2": 0, "slots3+": 0, "listed_only": 0}
pat = {}
for fn in glob.glob(os.path.join(d, "p_%s_*.json" % roads)):
    r = json.load(open(fn))
    bl = dict((c, b) for c, b in r["blocked"])
    for cell, sds, masks, smap in r["split"]:
        tot["split"] += 1
        tot["slots2" if len(sds) == 2 else "slots3+"] += 1
        b = bl.get(cell, 0)
        if b == 16:
            tot["none"] += 1
        elif b:
            tot["some"] += 1
            k = "".join(ch for i, ch in enumerate("EWSN") if b >> i & 1)
            pat[k] = pat.get(k, 0) + 1
        else:
            tot["all4"] += 1
    tot["listed_only"] += len(r["listed"])
print(tot)
print("not crossed:", sorted(pat.items(), key=lambda x: -x[1]))
