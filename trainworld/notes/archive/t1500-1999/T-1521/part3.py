"""T-1521 part 3: shares on T-1431's ten pairs with rail's new starting curve and the plane at a real
airfare, nothing else changed. T-1431's round 3 harness (income2.py: three groups, every mode's money
at the rider's weight)."""
import sys
HERE = r"C:\Users\anita\projects\maps\trainworld\notes\T-1431" + "\\"
sys.argv = [sys.argv[0]]
src = open(HERE + "income2.py", encoding="utf-8").read().split('print("car money')[0]
__file__ = HERE + "income2.py"
exec(src)

OLD_CURVE = rail_curve
S_PAST_20 = 0.25 / 2.8  # New York's 25 cents a km past 20 km, in the India curve
def new_curve(fkm):
    return OLD_CURVE(fkm) if fkm <= 20 else OLD_CURVE(20) + S_PAST_20 * (fkm - 20)

def cells(curve, base, perkm=0.05):
    global rail_curve
    rail_curve = curve
    out = {}
    for q in pre:
        out[q["name"]] = run(q, base, perkm)
    e = err(lambda q: run(q, base, perkm))
    rail_curve = OLD_CURVE
    return out, e

CASES = [("built today: old curve, plane $30 + 5c", OLD_CURVE, 30, 0.05),
         ("new curve, plane $30 + 5c", new_curve, 30, 0.05),
         ("new curve, plane $100 + 5c", new_curve, 100, 0.05), ("new curve, plane $150 + 5c", new_curve, 150, 0.05)]
res = [(lab, *cells(c, b, k)) for lab, c, b, k in CASES]
print("rail fare in New York terms (x2.8) at the pairs' fare km, old -> new:")
for q in pre:
    print(f"  {q['name']:22s} {q['fkm']:5.0f} km  India ${OLD_CURVE(q['fkm']):6.2f} -> ${new_curve(q['fkm']):6.2f}; in {q['country']} x{FW_IN / q['w']:.2f}")
print()
print(f"{'pair':22s} | " + " | ".join(lab for lab, _, _ in res) + " | published")
for q in pre:
    row = []
    for lab, out, e in res:
        sr, sa, sc = out[q["name"]]
        row.append(f"{100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%")
    print(f"{q['name']:22s} | " + " | ".join(row) + f" | {q['pub']}")
print("error (mean |log-odds| of rail of rail + air):", " | ".join(f"{e:.2f}" for _, _, e in res))
print("\nplane base sweep with the new curve (info only, not built):")
for b in (60, 80, 100, 120, 150):
    print(f"  ${b}: {cells(new_curve, b)[1]:.2f}")

def run2(q, base, perkm=0.05, carx=1.0, mf=1.0, n=3):
    """income2.run with the car's money times carx and every mode's money weight times mf."""
    g = groups(q["country"], n)
    tw = [p * y ** EPS for p, y in g]
    tot = sum(tw)
    out = []
    for acc in (30, 130):
        s3 = [0, 0, 0]
        for (p, y), t in zip(g, tw):
            m = y ** -VOT_E * mf
            car = q["cm"] + carx * q["cu"] * q["w"] * m
            beta = 0.1 * math.sqrt(min(60 / car, 1))
            r = acc + WAIT + q["tmin"] + m * FW_IN * new_curve(q["fkm"])
            pl = q["pt"] + m * q["w"] * (base + perkm * q["km"])
            s = split(beta, [r, car, pl])
            for i in range(3): s3[i] += s[i] * t / tot
        out.append(s3)
    sr, sc, sa = [(out[0][i] + out[1][i]) / 2 for i in range(3)]
    return sr, sa, sc

print("\ncar money per traveller at the pairs (game $, shared by two):")
for q in pre:
    print(f"  {q['name']:22s} ${q['cu']:6.1f}  ({q['cu'] / q['km']:.3f} a km of distance)")
print("\ninfo only, not built: new curve, plane $150 + 5c, and")
INFO = [("car money x2", dict(carx=2.0)), ("money weight x0.4 (all modes)", dict(mf=0.4)),
        ("both", dict(carx=2.0, mf=0.4))]
print(f"{'pair':22s} | " + " | ".join(l for l, _ in INFO))
for q in pre:
    row = []
    for l, kw in INFO:
        sr, sa, sc = run2(q, 150, **kw)
        row.append(f"{100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%")
    print(f"{q['name']:22s} | " + " | ".join(row))
print("error:", " | ".join(f"{err(lambda q: run2(q, 150, **kw)):.2f}" for _, kw in INFO))
