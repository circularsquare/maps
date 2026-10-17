"""T-558: read the no-rail fit grid; per walk cost, the constant matching today's walking and the
band profile it gives. Usage: python fit_read.py fit.json"""
import json
import sys

import numpy as np

f = json.load(open(sys.argv[1]))
vars_ = f["vars"]
near = np.array(f["near"])  # var, layer, band, mode(trips, walk, rail, car, other)
allv = np.array(f["all"])[0]
edges = f["edges"]
names = [f"<{e}" for e in edges] + [f"{edges[-1]}+"]
groups = {"commute": [0], "everyday": [1, 2, 3, 4, 5, 6]}
for g, ls in groups.items():
    trips_all = allv[ls].sum(axis=0)[:, 0]
    today = near[0][ls].sum(axis=0)  # band x mode
    tot_trips = trips_all.sum()
    tw = today[:, 1].sum()
    print(f"\n== {g}: trips {tot_trips:,.0f}, today walk {tw / tot_trips * 100:.2f}% (near pairs hold {today[:, 0].sum() / tot_trips * 100:.1f}% of trips)")
    print("today walk % by band:", " ".join(f"{n}:{(today[b, 1] / trips_all[b] * 100) if trips_all[b] else 0:.1f}" for b, n in enumerate(names[:9])))
    # group variants by (t0, w2)
    keys = {}
    for k, v in enumerate(vars_):
        if v["today"]:
            continue
        keys.setdefault((v["t0"], v["w2"]), []).append((v["ac"], k))
    for (t0, w2), lst in sorted(keys.items()):
        lst.sort()
        ws = [near[k][ls].sum(axis=0)[:, 1].sum() for _, k in lst]
        a = [x for x, _ in lst]
        # interpolate the constant giving today's total
        if not (ws[0] <= tw <= ws[-1]):
            print(f"t0 {t0} w2 {w2}: out of range ({ws[0] / tot_trips * 100:.1f}-{ws[-1] / tot_trips * 100:.1f}%)")
            continue
        i = next(i for i in range(1, len(ws)) if ws[i] >= tw)
        fr = (tw - ws[i - 1]) / (ws[i] - ws[i - 1])
        asc = a[i - 1] + fr * (a[i] - a[i - 1])
        prof = (1 - fr) * near[lst[i - 1][1]][ls].sum(axis=0)[:, 1] + fr * near[lst[i][1]][ls].sum(axis=0)[:, 1]
        err = np.abs(prof - today[:, 1]).sum() / tw * 50
        print(f"t0 {t0:4.0f} w2 {w2:4.1f}: asc {asc:5.2f}  misplaced {err:4.1f}% of walkers  walk % by band: " +
              " ".join(f"{(prof[b] / trips_all[b] * 100) if trips_all[b] else 0:.1f}" for b in range(9)))
