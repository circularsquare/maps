"""T-1431 round 3: three income groups, every mode at the rider's own minutes a dollar (the car's
money too, Anita 2026-10-26). Refit the plane's base. Same pairs and harness as income.py.
"""
import json, math, sys
HERE = __file__.replace("income2.py", "")
sys.argv = [sys.argv[0]]
src = open(HERE + "income.py", encoding="utf-8").read()
exec(src.split("pre = []")[0].replace('__file__.replace("income.py", "")', repr(HERE)))

def car_parts(a, b_, km):
    """(minutes, dollars) of the market's car: drive and parking in minutes, running and crossings in US$ (shared by two)."""
    rt, rk, fkm, fn = drv.get(tuple(sorted((a["zone"], b_["zone"]))), [1, 1, 0, 0])
    if km > 1500: rt, rk = 1, 1
    mins = (15 + km * 1.25 / 85 * 60) * rt + 0.5 * (max(0, a["park"]) + max(0, b_["park"])) / 2
    usd = km * 1.25 * rk * 0.15 / 2
    if (a["country"] == "GB") != (b_["country"] == "GB"):
        mins += 60; usd += 150 / 2
    if fn:
        mins += fn * 30 + fkm / 35 * 60; usd += (fn * 50 + 0.5 * fkm) / 2
    return mins, usd

pre = []
for za, zb, name, tmin, pub in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B)
    cm, cu = car_parts(A, B, km)
    pre.append(dict(name=name, km=km, fkm=1.05 * km, cm=cm, cu=cu, w=fw(A["country"]), pt=plane_parts(A, B, km),
                    tmin=tmin, pub=pub, country=A["country"]))

def run(q, base, perkm=0.05, n=3, car_same=True, mix=True, trips_e=EPS, rx=1.0):
    g = groups(q["country"], n) if mix else [(1, 1)]
    tw = [p * y ** trips_e for p, y in g]
    tot = sum(tw)
    out = []
    for acc in (30, 130):
        s3 = [0, 0, 0]
        for (p, y), t in zip(g, tw):
            m = y ** -VOT_E
            car = q["cm"] + q["cu"] * (q["w"] * m if car_same else 3)
            beta = 0.1 * math.sqrt(min(60 / car, 1))
            r = acc + WAIT + q["tmin"] + rx * m * FW_IN * rail_curve(q["fkm"])
            pl = q["pt"] + m * q["w"] * (base + perkm * q["km"])
            s = split(beta, [r, car, pl])
            for i in range(3): s3[i] += s[i] * t / tot
        out.append(s3)
    sr, sc, sa = [(out[0][i] + out[1][i]) / 2 for i in range(3)]
    return sr, sa, sc

def lo(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))

def err(fn):
    e = 0
    for q in pre:
        sr, sa, sc = fn(q)
        e += abs(lo(sr / (sr + sa)) - lo(q["pub"][2] / 100))
    return e / len(pre)

print("car money per pair (US$ shared), minutes, at the country's weight:")
for q in pre:
    print(f"  {q['name']:22s} car {q['cm']:5.0f} min + ${q['cu']:5.1f} x {q['w']:.2f} = {q['cm'] + q['cu'] * q['w']:5.0f} (at 3: {q['cm'] + 3 * q['cu']:5.0f})")
print("\nplane base sweep, 3 groups, car at the rider's weight (error); car at 3 for comparison:")
for base in (100, 80, 60, 50, 45, 40, 35, 30, 25, 20):
    print(f"  ${base}: {err(lambda q: run(q, base)):.2f}   car at 3: {err(lambda q: run(q, base, car_same=False)):.2f}")
best = min(range(0, 101, 5), key=lambda b: err(lambda q: run(q, b)))
print("best base:", best)

print("\nrail / air / car of all, rail of rail + air")
for q in pre:
    cells = []
    for lab, fn in (("today", lambda q: run(q, 100, mix=False, car_same=False)),
                    ("3 groups, car at 3, $30", lambda q: run(q, 30, car_same=False)),
                    ("3 groups, same weight, $100", lambda q: run(q, 100)),
                    (f"3 groups, same weight, ${best}", lambda q: run(q, best)),
                    ("3 groups, same weight, $30", lambda q: run(q, 30)),
                    ("3 groups, same weight, $40", lambda q: run(q, 40))):
        sr, sa, sc = fn(q)
        cells.append(f"{100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%")
    print(f"{q['name']:22s} | " + " | ".join(cells) + f" | pub {q['pub']}")
print(f"columns: today | 3 groups car at 3 $30 | same weight $100 | same weight ${best} | same $30 | same $40")

print("\ninfo only: rail's long fares at x the starting curve (a player's dearer curve), plane base refit")
for rx in (1.5, 2.0, 2.5):
    b = min(range(0, 161, 5), key=lambda bb: err(lambda q: run(q, bb, rx=rx)))
    print(f"  rail x{rx}: best plane ${b}, error {err(lambda q: run(q, b, rx=rx)):.2f}")
    for q in pre:
        if 280 < q["km"] < 700:
            sr, sa, sc = run(q, b, rx=rx)
            print(f"     {q['name']:22s} {100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%  pub {q['pub']}")
