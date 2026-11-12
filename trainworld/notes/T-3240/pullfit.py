"""T-3240: an importance pull on the market's destinations, from the rows, every place of the world.

A place's pull P = exp(gamma * max(0, R0 - rank)), rank its best Natural Earth scalerank (place_rank.json,
none = no pull); US places P = 1 when --us-off. Each origin's departures are kept: its trips to every
place are scaled by P_q * k_p, k_p = dep_p / (dep_p + sum_q T_pq (P_q - 1)) (the unplaced share, dead
space and the ground's dropped fade, keeps its weight). Ground and tail alike. Rough: place-centre
distances, near pairs' each-way figure as each direction.
    python pullfit.py gamma R0 [--us-off] [--eps E]      (--eps: income exponent in place of 0.7)
"""
import json, math, sys, collections
import numpy as np
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist"
TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rule = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
rank = json.load(open(HERE + r"\place_rank.json", encoding="utf-8"))
OPEN = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE NO IS CH LI GB".split())

args = [a for a in sys.argv[1:] if not a.startswith("--")]
gamma, R0 = (float(args[0]), float(args[1])) if args else (0., 8.)
us_off = "--us-off" in sys.argv
eps_new = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else None

Z = list(rows)
R = [rows[z] for z in Z]
n = len(Z)
ix = {z: i for i, z in enumerate(Z)}
pix = {r["place"]: i for i, r in enumerate(R)}
lat = np.radians([r["lat"] for r in R]); lon = np.radians([r["lon"] for r in R])
cc = np.array([r["country"] for r in R])
inc = np.minimum(np.array([r["inc"] for r in R]), 1.)
g = np.array([r["g"] for r in R]); t = np.array([r["t"] for r in R])
a = np.array([r["a"] for r in R]); ta = np.array([r["ta"] for r in R])
pop = np.array([r["pop"] for r in R]); dep = np.array([r["dep"] for r in R])
if eps_new is not None:
    # departures and attraction scale with income^eps (origin s, destination a and ta); far trips' normaliser
    # is a world sum, so its change is ignored (rough)
    f = np.where(inc > 0, inc, 1.) ** (eps_new - rule["eps"])
    g, t, a, ta, dep = g * f, t * f, a * f, ta * f, dep * f
openm = np.array([c in OPEN for c in cc])

# directed trips T[p, q], p -> q, a weekday
d = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    h = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
    d[i] = 2 * 6371.0088 * np.arcsin(np.sqrt(np.minimum(h, 1)))


def on(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


T = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    di = d[i].astype(np.float64)
    gr = on(di, *rule["onset_km"]) * np.maximum(di, rule["d0_km"]) ** -rule["A"] * (1 - on(di, *rule["tail_onset_km"]))
    tl = on(di, *rule["tail_onset_km"]) * np.maximum(di, rule["D1_km"]) ** -rule["G"]
    rel = np.minimum(inc / inc[i], 1) ** rule["eps_t"] if inc[i] > 0 else 1
    gen = rule["border"] * inc[i] ** rule["border_eps"]
    b = np.where(cc == cc[i], 1., np.where(openm & openm[i], max(0.5, gen), gen))
    T[i] = b * (g[i] * gr * a + t[i] * tl * ta * rel)
    T[i, i] = 0
    near = rows[Z[i]].get("near", [])
    for q, v in near:
        j = pix.get(q)
        if j is not None:
            T[i, j] = v * (f[i] if eps_new is not None else 1)
    far = d[i] >= rule["near_km"]
    # near pairs not in i's own row: the other row's figure
for i in range(n):
    for q, v in rows[Z[i]].get("near", []):
        j = pix.get(q)
        if j is not None and j != i and not any(x[0] == R[i]["place"] for x in rows[Z[j]].get("near", [])):
            T[j, i] = v

rk = np.array([rank.get(z, [99])[0] for z in Z], dtype=float)
if "--excess" in sys.argv:
    # importance beyond headcount: the rank a place of its people would have (a line through every ranked
    # place's rank against log people), less its own rank; R0 is the cap on that excess
    has = rk < 99
    bfit = np.polyfit(np.log(pop[has]), rk[has], 1)
    expect = np.polyval(bfit, np.log(np.maximum(pop, 1)))
    excess = np.where(has, expect - rk, 0.)
    print(f"rank = {bfit[1]:.2f} {bfit[0]:+.2f} ln(people)")
    P = np.exp(gamma * np.clip(excess, 0, R0))
    for z in ["war", "kra", "tro", "poz", "msw", "lod", "kat", "skw", "ber", "mun", "ham", "lon", "man", "nyc", "bos", "wdc", "phl", "tyo", "osa", "bei", "sha", "nag", "jnj", "liu"]:
        i = ix[z]; print(f"   {z} rank {rk[i]:.0f} expected {expect[i]:.1f} pull {P[i]:.2f}", end=";")
    print()
else:
    P = np.exp(gamma * np.maximum(0, R0 - rk))
if us_off:
    for c_ in (sys.argv[sys.argv.index("--off") + 1].split(",") if "--off" in sys.argv else ["US"]): P[cc == c_] = 1.
PH = [float(x) for x in sys.argv[sys.argv.index("--phase") + 1].split(",")] if "--phase" in sys.argv else None
W = (P[None, :] - 1) * (on(d, *PH) if PH else 1.) + 1  # the pull fades in over the phase window
OFFC = sys.argv[sys.argv.index("--off") + 1].split(",") if "--off" in sys.argv else (["US"] if us_off else [])
W = np.where(np.isin(cc, OFFC)[:, None], 1., W)  # an exempt country's residents see no pull
k = dep / (dep + (T * (W - 1)).sum(1))
Tn = T * W * k[:, None]


def each(a_, b_):
    i, j = ix[a_], ix[b_]
    return 0.5 * (Tn[i, j] + Tn[j, i]), 0.5 * (T[i, j] + T[j, i])


TARGET = {("war", "kra"): 25000, ("war", "tro+gda+stz"): 25000, ("ber", "mun"): 22000, ("ham", "mun"): 10000,
          ("lon", "man"): 18000, ("bei", "sha"): 100000, ("tyo", "osa"): 175000, ("jnj", "nag"): 6000,
          ("nyc", "bos"): 53000, ("nyc", "wdc"): 41000}
err = []
print(f"gamma {gamma} R0 {R0} us_off {us_off} eps {eps_new or rule['eps']}")
for (x, y), tgt in TARGET.items():
    nw = old = 0.
    for yy in y.split("+"):
        e = each(x, yy); nw += e[0]; old += e[1]
    nw, old = 2 * nw, 2 * old
    err.append(math.log(nw / tgt))
    print(f"  {x}-{y:12s} both ways {old:9.0f} -> {nw:9.0f}  target {tgt:7d}  x{nw/tgt:.2f}")
print(f"  rms log error {math.sqrt(sum(e*e for e in err)/len(err)):.3f} (targets rough)")
for x, y in [("war", "skw"), ("war", "lod"), ("kra", "kat"), ("jnj", "yzo"), ("nyc", "phl"), ("lon", "bir")]:
    e = each(x, y); print(f"  {x}-{y} both ways {2*e[1]:9.0f} -> {2*e[0]:9.0f}")
# national rates, domestic, a person a weekday
for c in ["US", "JP", "DE", "PL", "CN", "GB", "FR"]:
    m = cc == c
    sub, sub0, dd = Tn[np.ix_(m, m)], T[np.ix_(m, m)], d[np.ix_(m, m)]
    pc = pop[m].sum()
    b100 = lambda M: M[(dd >= 100) & (dd < 300)].sum() / pc
    b300 = lambda M: M[dd >= 300].sum() / pc
    print(f"  {c}: 100-300 {b100(sub0):.4f} -> {b100(sub):.4f}, 300+ {b300(sub0):.4f} -> {b300(sub):.4f}")
np.save(HERE + r"\last_P.npy", P)
json.dump({"Z": Z, "k": k.tolist(), "P": P.tolist()}, open(HERE + r"\last_pull.json", "w"))
if "--pl" in sys.argv:
    BIG = [ix[z] for z in ["war", "kra", "tro", "gda", "poz", "msw", "lod", "kat"]]
    for z in ["osr", "sdl", "kic", "rad", "nwo", "skw", "olz", "rzs"]:
        i = ix[z]
        for lab, M in (("before", T), ("after", Tn)):
            row = M[i]
            tot = dep[i]
            own = row[i]; near = row[(d[i] < 100) & (np.arange(n) != i)].sum(); big = row[BIG].sum() - (row[i] if i in BIG else 0)
            print(f"  {z} {lab}: departures {tot:7.0f}; own zone {own/tot:4.0%}, others under 100 km {near/tot:4.0%}, the 8 big cities {big/tot:4.0%} ({big:6.0f}), Warsaw {row[ix['war']]:6.0f}")
