"""T-1521 round 2: shares on T-1431's ten pairs at market rates. Rail at the new starting curve, the
plane $100 + 5c, the car at its country's fuel, upkeep and tolls, every mode's money at 0.4 of the
rider's weight on longer trips, weights from GDP per head at market rates (4.4 x (gdp / 90,000)^-0.7)."""
import sys, math
HERE = r"C:\Users\anita\projects\maps\trainworld\notes\T-1431" + "\\"
sys.argv = [sys.argv[0]]
src = open(HERE + "income2.py", encoding="utf-8").read().split('print("car money')[0]
__file__ = HERE + "income2.py"
exec(src)

GDP_USD = {"US": 90027, "JP": 35951, "CN": 13862, "FR": 48986, "ES": 38627, "GB": 57602, "IN": 2702}
def w_mkt(c):
    return 4.4 * (GDP_USD[c] / 90000) ** -0.7
FUEL = {"US": 1.220, "JP": 1.075, "CN": 1.384, "FR": 2.415, "ES": 1.982, "GB": 2.336}
TOLL = {"JP": 21 / 149.66, "CN": 0.5 / 7.19, "FR": 0.104 / 0.88}
def car_km(c):
    return FUEL[c] * 0.085 + UPKEEP + TOLL.get(c, 0.)

import os
UPKEEP = float(os.environ.get("UPKEEP", "0"))
OLD_CURVE = rail_curve
def new_curve(fkm):  # India PPP-era units; K x curve is the same at either basis
    return OLD_CURVE(fkm) if fkm <= 20 else OLD_CURVE(20) + 0.25 / 2.8 * (fkm - 20)

def run3(q, base=100., perkm=0.05, long_m=0.4, car_real=True, curve=new_curve, mkt=True, n=3):
    c = q["country"]
    w = w_mkt(c) if mkt else q["w"]
    cu = q["cu"] * (car_km(c) / 0.15 if car_real else 1.)
    g = groups(c, n)
    tw = [p * y ** EPS for p, y in g]
    tot = sum(tw)
    out = []
    for acc in (30, 130):
        s3 = [0, 0, 0]
        for (p, y), t in zip(g, tw):
            m = y ** -VOT_E * long_m
            car = q["cm"] + cu * w * m
            beta = 0.1 * math.sqrt(min(60 / car, 1))
            r = acc + WAIT + q["tmin"] + m * FW_IN * curve(q["fkm"])
            pl = q["pt"] + m * w * (base + perkm * q["km"])
            s = split(beta, [r, car, pl])
            for i in range(3): s3[i] += s[i] * t / tot
        out.append(s3)
    sr, sc, sa = [(out[0][i] + out[1][i]) / 2 for i in range(3)]
    return sr, sa, sc

CASES = [
    ("built today (T-1431)", dict(base=30, long_m=1., car_real=False, curve=OLD_CURVE, mkt=False)),
    ("round 1 (PPP, plane $150)", dict(base=150, long_m=1., car_real=False, mkt=False)),
    ("market, real fares only", dict(long_m=1., car_real=False)),
    ("market + real car", dict(long_m=1.)),
    ("market + real car + 0.4", dict()),
]
print("weights at market:", {c: round(w_mkt(c), 2) for c in GDP_USD}, " car $/km:", {c: round(car_km(c), 3) for c in FUEL})
print("car $ each per pair (real):", {q["name"]: round(q["cu"] * car_km(q["country"]) / 0.15, 1) for q in pre})
print(f"{'pair':22s} | " + " | ".join(l for l, _ in CASES) + " | published")
for q in pre:
    row = []
    for l, kw in CASES:
        sr, sa, sc = run3(q, **kw)
        row.append(f"{100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%")
    print(f"{q['name']:22s} | " + " | ".join(row) + f" | {q['pub']}")
print("error:", " | ".join(f"{err(lambda q: run3(q, **kw)):.2f}" for _, kw in CASES))
print("\ninfo: plane base sweep, full model:", {b: round(err(lambda q: run3(q, base=b)), 2) for b in (60, 80, 100, 120, 150)})
print("info: long-trip money factor sweep:", {f: round(err(lambda q: run3(q, long_m=f)), 2) for f in (0.3, 0.4, 0.5, 0.7)})
