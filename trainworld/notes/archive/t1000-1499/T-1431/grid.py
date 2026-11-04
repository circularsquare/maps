"""T-1431: grid over the inter-city fare weight (flat or a distance taper) and the plane's base and per-km.

Score: rail of rail + air on the ten pairs, the corridor read as the mean of centre and suburb access,
against the published (or estimated) figure, in log-odds so 1% and 99% count fairly.
"""
import math, sys, itertools
sys.argv = [sys.argv[0], "today"]
exec(open(__file__.replace("grid.py", "fares.py"), encoding="utf-8").read().split("only = sys.argv")[0])

def lo(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))

pre = []
for za, zb, name, tmin, pub in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B)
    c = car(A, B, km)
    pre.append(dict(name=name, km=km, fkm=1.05 * km, car=c, beta=0.1 * math.sqrt(min(60 / c, 1)), w=fw(A["country"]),
                    pt=plane_parts(A, B, km), tmin=tmin, pub=pub))

def run(wf, base, perkm, detail=False):
    err, rows_ = 0, []
    for q in pre:
        f = wf(q["km"])
        rr, aa, cc = [], [], []
        for acc in (30, 130):
            r = acc + WAIT + q["tmin"] + f * FW_IN * rail_curve(q["fkm"])
            pl = q["pt"] + f * q["w"] * (base + perkm * q["km"])
            sr, sc, sa = split(q["beta"], [r, q["car"], pl])
            rr.append(sr); aa.append(sa); cc.append(sc)
        sr, sa, sc = sum(rr) / 2, sum(aa) / 2, sum(cc) / 2
        rra = sr / (sr + sa)
        err += abs(lo(rra) - lo(q["pub"][2] / 100))
        rows_.append((q["name"], 100 * sr, 100 * sa, 100 * sc, 100 * rra, q["pub"][2], 100 * rr[0] / (rr[0] + aa[0]), 100 * rr[1] / (rr[1] + aa[1])))
    return err / len(pre), rows_

weights = {"x1": lambda d: 1}
for k in (0.6, 0.5, 0.4, 0.33):
    weights[f"x{k}"] = (lambda k: lambda d: k)(k)
for d0, e in ((50, 0.2), (50, 0.3), (50, 0.4), (100, 0.5)):
    weights[f"({d0}/km)^{e}"] = (lambda d0, e: lambda d: taper(d, d0, e))(d0, e)

res = []
for (wn, wf), base, perkm in itertools.product(weights.items(), (100, 75, 60, 50, 40, 30, 20, 10), (0.03, 0.05, 0.07, 0.09)):
    e, _ = run(wf, base, perkm)
    res.append((e, wn, base, perkm))
res.sort()
print("best 25 (mean log-odds error):")
for r in res[:25]:
    print("  %.2f  %-14s $%d + %dc" % (r[0], r[1], r[2], round(100 * r[3])))
print("today's plane at each weight:")
for wn in weights:
    print("  %.2f  %s" % (next(r[0] for r in res if r[1] == wn and r[2] == 100 and r[3] == 0.05), wn))
print("best plane per weight:")
for wn in weights:
    b = next(r for r in res if r[1] == wn)
    print("  %.2f  %-14s $%d + %dc" % (b[0], wn, b[2], round(100 * b[3])))

for wn, base, perkm in [(a.split("|")[0], int(a.split("|")[1]), float(a.split("|")[2])) for a in sys.argv[2:]] or []:
    pass

def show(wn, base, perkm):
    e, rows_ = run(weights[wn], base, perkm)
    print(f"\n{wn}, plane ${base} + {round(100*perkm)}c: error {e:.2f}")
    for r in rows_:
        print("  %-22s rail %5.1f air %5.1f car %5.1f | r/(r+a) %5.1f (centre %5.1f, suburb %5.1f) pub %s" % (r[0], r[1], r[2], r[3], r[4], r[6], r[7], r[5]))

show("x1", 100, 0.05)
for wn, base, perkm in [("x0.4", 100, 0.05), ("x0.4", 40, 0.05), ("x0.5", 40, 0.05), ("x0.4", 50, 0.05), ("(50/km)^0.3", 40, 0.05), ("(50/km)^0.3", 50, 0.05), ("(50/km)^0.3", 100, 0.05), ("x1", 30, 0.05)]:
    show(wn, base, perkm)
