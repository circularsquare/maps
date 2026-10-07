"""The app's crediting (creditRun + mapBack + mergeSpans), for a ride over whole sections.

    python ride_sim.py <data dir> cc <ridden line id> <station name a> <station name b> <name filter>
Rides every section of the line between the two stations (by display order) and prints the
percentage of each line whose name contains the filter."""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
sys.path.insert(0, str(ROOT))
import ownership

d, cc, lid, sa, sb, flt = Path(sys.argv[1]) / sys.argv[2], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
lines = json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]
st = json.loads((d / "stations.json").read_text(encoding="utf-8"))["stations"]
foot = ownership.read(d / "foot.json")
km = {}
for l in lines:
    for a, b, k, g in l["sections"]:
        km[g] = k
L = next(l for l in lines if l["id"] == lid)
name = {s: st[s]["n"] for s in st}
disp = L["display"]
ia = next(i for i, s in enumerate(disp) if name[s] == sa)
ib = next(i for i, s in enumerate(disp) if name[s] == sb)
lo, hi = sorted((ia, ib))
seg = {frozenset(p) for p in zip(disp[lo:hi], disp[lo + 1:hi + 1])}
ridden = [g for a, b, k, g in L["sections"] if frozenset((a, b)) in seg]


def merge(sp, k):
    tol = min(0.15 / k, 0.1) if k > 0 else 0
    sp = sorted([min(x), max(x)] for x in sp)
    out = []
    for x, y in sp:
        if out and x <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], y)
        else:
            out.append([x, y])
    if out and out[0][0] <= tol:
        out[0][0] = 0
    if out and out[-1][1] >= 1 - tol:
        out[-1][1] = 1
    return out


fo = lambda g: foot.get(g, [[g, 0, 1, 0, 1]])
raw = defaultdict(list)
for g in ridden:
    raw[g].append((0, 1))
    for t, f, to, a, b in fo(g):
        raw[t].append((f, to))
R = {t: merge(v, km[t]) for t, v in raw.items()}
for l in lines:
    if flt not in l["name"]:
        continue
    tot = done = 0.0
    for a, b, k, g in l["sections"]:
        tot += k
        if g in ridden:
            done += k
            continue
        out = []
        for t, f, to, x, y in fo(g):
            r = R.get(t)
            if not r or abs(to - f) < 1e-9:
                continue
            lo_, hi_ = min(f, to), max(f, to)
            for p, q in r:
                i, j = max(p, lo_), min(q, hi_)
                if j <= i:
                    continue
                p1 = x + (i - f) / (to - f) * (y - x)
                p2 = x + (j - f) / (to - f) * (y - x)
                out.append((p1, p2))
        done += k * min(1, sum(q - p for p, q in merge(out, k)))
    print(f"{100 * done / tot:5.1f}%  {l['name']}")
