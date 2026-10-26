"""T-1431 round 2: travellers split by income group, each with its own fare weight (Anita's idea).

Each group's minutes a dollar is the country's times (group income / mean)^-0.5, the exponent T-1277
already uses across countries. Groups from the World Bank's income shares by quintile
(SI.DST.FRST.20 .. SI.DST.05TH.20, ../artifacts/trainworld/T-1431/wb_quintiles.json). How many
long trips each group makes: (group income / mean)^0.7, the market rule's `eps` (trip rate by
income, intercity.rs `s_inc = inc^eps`), or equal. Same pairs and harness as fares.py.
"""
import json, math, sys, itertools
HERE = __file__.replace("income.py", "")
sys.argv = [sys.argv[0]]
exec(open(HERE + "fares.py", encoding="utf-8").read().split("only = sys.argv")[0])

wb = json.load(open(r"C:\Users\anita\projects\maps\artifacts\trainworld\T-1431\wb_quintiles.json", encoding="utf-8"))[1]
ORDER = ["SI.DST.FRST.20", "SI.DST.02ND.20", "SI.DST.03RD.20", "SI.DST.04TH.20", "SI.DST.05TH.20"]
ISO2 = {"CHN": "CN", "ESP": "ES", "FRA": "FR", "GBR": "GB", "JPN": "JP", "USA": "US"}
shares = {}
for r in wb:
    shares.setdefault(ISO2[r["countryiso3code"]], {})[r["indicator"]["id"]] = r["value"]
QUINT = {c: [v[k] / sum(v.values()) for k in ORDER] for c, v in shares.items()}

def groups(c, n):
    """(population share, income / mean) for n equal groups, from the quintile Lorenz curve (income flat within a quintile)."""
    q = QUINT[c]
    lor = [0]
    for s in q: lor.append(lor[-1] + s)
    L = lambda p: lor[min(int(p * 5), 4)] + (p * 5 - min(int(p * 5), 4)) * q[min(int(p * 5), 4)]
    return [(1 / n, (L((i + 1) / n) - L(i / n)) * n) for i in range(n)]

EPS = 0.7      # the market rule's trip rate by income
VOT_E = 0.5    # T-1277's fare weight by income

def shares_for(q, base, perkm, f=1.0, n=5, trips_e=EPS, mix=True):
    out = []
    for acc in (30, 130):
        g = groups(q["country"], n) if mix else [(1, 1)]
        tw = [p * y ** trips_e for p, y in g]
        tot = sum(tw)
        acc_s = [0, 0, 0]
        for (p, y), t in zip(g, tw):
            m = y ** -VOT_E * f
            r = acc + WAIT + q["tmin"] + m * FW_IN * rail_curve(q["fkm"])
            pl = q["pt"] + m * q["w"] * (base + perkm * q["km"])
            s = split(q["beta"], [r, q["car"], pl])
            for i in range(3): acc_s[i] += s[i] * t / tot
        out.append(acc_s)
    sr, sc, sa = [(out[0][i] + out[1][i]) / 2 for i in range(3)]
    return sr, sa, sc

pre = []
for za, zb, name, tmin, pub in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B); c = car(A, B, km)
    pre.append(dict(name=name, km=km, fkm=1.05 * km, car=c, beta=0.1 * math.sqrt(min(60 / c, 1)), w=fw(A["country"]),
                    pt=plane_parts(A, B, km), tmin=tmin, pub=pub, country=A["country"]))

def lo(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))

def err(fn):
    e = 0
    for q in pre:
        sr, sa, sc = fn(q)
        e += abs(lo(sr / (sr + sa)) - lo(q["pub"][2] / 100))
    return e / len(pre)

for c in QUINT:
    print(c, "quintiles income/mean:", [round(y, 2) for _, y in groups(c, 5)], "terciles:", [round(y, 2) for _, y in groups(c, 3)],
          "fare weight x by quintile:", [round(y ** -VOT_E, 2) for _, y in groups(c, 5)],
          "trip-weighted mean of weight x:", round(sum(y ** EPS * y ** -VOT_E for _, y in groups(c, 5)) / sum(y ** EPS for _, y in groups(c, 5)), 3))

V = {
    "today": lambda q: shares_for(q, 100, 0.05, mix=False),
    "mix": lambda q: shares_for(q, 100, 0.05),
    "mix $40": lambda q: shares_for(q, 40, 0.05),
    "mix equal trips": lambda q: shares_for(q, 100, 0.05, trips_e=0),
    "mix $40 equal trips": lambda q: shares_for(q, 40, 0.05, trips_e=0),
    "mix $40, 3 groups": lambda q: shares_for(q, 40, 0.05, n=3),
    "mix $40, 10 groups": lambda q: shares_for(q, 40, 0.05, n=10),
    "taper $40": lambda q: shares_for(q, 40, 0.05, f=taper(q["km"], 50, 0.3), mix=False),
    "mix + taper $40": lambda q: shares_for(q, 40, 0.05, f=taper(q["km"], 50, 0.3)),
    "mix + taper $60": lambda q: shares_for(q, 60, 0.05, f=taper(q["km"], 50, 0.3)),
}
print()
for vn, fn in V.items():
    print(f"{vn:22s} error {err(fn):.2f}")
print("\nbest plane base with the mix alone (5c a km):")
for base in (100, 80, 60, 50, 40, 30, 20, 10, 0):
    print(f"  ${base}: {err(lambda q: shares_for(q, base, 0.05)):.2f}   mix + taper: {err(lambda q: shares_for(q, base, 0.05, f=taper(q['km'], 50, 0.3))):.2f}")

print("\nrail / air / car of all, then rail of rail + air")
for q in pre:
    cells = []
    for vn in ("today", "mix", "mix $40", "taper $40", "mix $40, 3 groups", "mix equal trips", "mix + taper $40"):
        sr, sa, sc = V[vn](q)
        cells.append(f"{100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%")
    print(f"{q['name']:22s} {q['km']:5.0f} | " + " | ".join(cells) + f" | pub {q['pub']}")
print("columns: today | mix | mix $40 | taper $40 | mix $40 3 groups | mix equal trips | mix + taper $40")

print("\nmix alone, plane $30 / $35 / $40 / $50 (5 groups): rail / air / car, rail of rail + air")
for q in pre:
    cells = []
    for base in (30, 35, 40, 50):
        sr, sa, sc = shares_for(q, base, 0.05)
        cells.append(f"{100*sr:3.0f}/{100*sa:3.0f}/{100*sc:3.0f} {100*sr/(sr+sa):3.0f}%")
    print(f"{q['name']:22s} | " + " | ".join(cells) + f" | pub {q['pub'][2]}")
for base in (30, 35):
    print(f"mix ${base}: error {err(lambda q: shares_for(q, base, 0.05)):.2f}; 3 groups {err(lambda q: shares_for(q, base, 0.05, n=3)):.2f}; equal trips {err(lambda q: shares_for(q, base, 0.05, trips_e=0)):.2f}")

print("\nper group, Tokyo - Fukuoka centre, mix $100: rail/car/plane")
q = pre[3]
for p, y in groups("JP", 5):
    m = y ** -VOT_E
    r = 30 + WAIT + q["tmin"] + m * FW_IN * rail_curve(q["fkm"]); pl = q["pt"] + m * q["w"] * (100 + 0.05 * q["km"])
    print(f"  income x{y:.2f} weight x{m:.2f}: rail {r:.0f} plane {pl:.0f} car {q['car']:.0f} ->", [round(100 * s) for s in split(q["beta"], [r, q["car"], pl])])
