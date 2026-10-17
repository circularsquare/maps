"""T-558: tables from a t558 run. Usage: python real_read.py real.json"""
import json
import sys

import numpy as np

f = json.load(open(sys.argv[1]))
vars_ = f["vars"]
none = np.array(f["none"])  # var, layer, band, mode
real = np.array(f["real"])
edges = f["edges"]
c = f["checks"]
print("checks:", c)
summ = f["summary"]
fc = sum(s["free_commute"][2] for s in summ)
ff = sum(s["final_commute"][2] for s in summ)
fe = sum(s["free_everyday"][2] for s in summ)
fef = sum(s["final_everyday"][2] for s in summ)
tc = sum(s["free_commute"][1] for s in summ)
print(f"model commute trips {tc:,.0f}; rail free-flow {fc:,.0f}, after crowding {ff:,.0f} ({(ff / fc - 1) * 100:+.1f}%); everyday rail free {fe:,.0f}, crowded {fef:,.0f} ({(fef / fe - 1) * 100:+.1f}%)")


def name(v):
    return "today" if v["today"] else f"l{v['l']} w2 {v['w2']} t0 {v['t0']} ac {v['ac']} ae {v['ae']}"


groups = {"all": list(range(7)), "commute": [0], "everyday": list(range(1, 7)), "commute to paid parking": [7], "everyday to paid parking": [8]}
# coarse bands as T-551: 0-2, 2-5, 5-10, 10-20, 20-50, 50+ (fine band index ranges)
coarse = {"0-2": range(0, 4), "2-5": range(4, 8), "5-10": [8], "10-20": [9], "20-50": [10], "50+": [11]}
for net, a in [("real network", real), ("no rail", none)]:
    print(f"\n#### {net}")
    for g, ls in groups.items():
        print(f"== {g}  (walk / rail / car / other, % of all trips)")
        for k, v in enumerate(vars_):
            x = a[k][ls].sum(axis=(0, 1))
            print(f"  {name(v):40s} trips {x[0]:12,.0f}  " + " / ".join(f"{x[m] / x[0] * 100:5.2f}" for m in range(1, 5)))
for g, ls in groups.items():
    print(f"\n== {g}, real network by band: walk % | rail % of all")
    print("  " + " " * 40 + "  ".join(f"{b:>11s}" for b in coarse))
    for k, v in enumerate(vars_):
        x = real[k][ls].sum(axis=0)
        cells = []
        for b, ix in coarse.items():
            y = x[list(ix)].sum(axis=0)
            cells.append(f"{y[1] / y[0] * 100:4.1f} |{y[2] / y[0] * 100:5.1f}" if y[0] else "   -")
        print(f"  {name(v):40s}" + "  ".join(f"{s:>11s}" for s in cells))
    x = real[0][ls].sum(axis=0)
    print("  share of trips by band: " + " ".join(f"{b}:{x[list(ix)].sum(axis=0)[0] / x[:, 0].sum() * 100:.1f}" for b, ix in coarse.items()))
# fine bands, everyday walk
print("\n== everyday walk % by fine band, real network")
names = [f"<{e}" for e in edges] + [f"{edges[-1]}+"]
for k, v in enumerate(vars_):
    x = real[k][1:7].sum(axis=0)
    print(f"  {name(v):40s}" + " ".join(f"{n}:{x[b, 1] / x[b, 0] * 100:.1f}" for b, n in enumerate(names[:9])))
# who rail takes: change against today, real network
print("\n== change against today on the real network, millions of trips a day (walk, rail, car, other)")
for g, ls in groups.items():
    t = real[0][ls].sum(axis=(0, 1))
    for k, v in enumerate(vars_[1:], 1):
        x = real[k][ls].sum(axis=(0, 1))
        print(f"  {g:9s} {name(v):40s} " + " ".join(f"{(x[m] - t[m]) / 1e6:+7.3f}" for m in range(1, 5)))
