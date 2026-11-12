"""T-3240 code stage: the pull's scenario on the market rows (rows only, every place of the world), with the rule as
built: excess from pipeline/market_pull.json's table, gamma at home (times country_gamma by destination country),
gamma_x across a border, faded in over phase_km, no pull for residents of `off`, each origin's departures kept.
    python scen.py [--gx G] [--cn F] [--gamma G] [--fr24] [--pairs] [--nopull]
Saves the matrices for other scripts in scen_<tag>.npz when --save TAG.
"""
import json, math, sys
import numpy as np
TW = r"C:\Users\anita\projects\maps\trainworld"
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\pull"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rule = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
cfg = json.load(open(TW + r"\pipeline\market_pull.json", encoding="utf-8"))
arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
gamma, gx = arg("--gamma", cfg["gamma"]), arg("--gx", cfg["gamma_x"])
cg = dict(cfg["country_gamma"]); cg["CN"] = arg("--cn", cg.get("CN", 1.0))
OPEN = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE NO IS CH LI GB".split())

Z = list(rows); R = [rows[z] for z in Z]; n = len(Z)
ix = {z: i for i, z in enumerate(Z)}; pix = {r["place"]: i for i, r in enumerate(R)}
lat = np.radians([r["lat"] for r in R]); lon = np.radians([r["lon"] for r in R])
cc = np.array([r["country"] for r in R])
inc = np.minimum(np.array([r["inc"] for r in R]), 1.)
g = np.array([r["g"] for r in R]); t = np.array([r["t"] for r in R])
a = np.array([r["a"] for r in R]); ta = np.array([r["ta"] for r in R])
pop = np.array([r["pop"] for r in R]); dep = np.array([r["dep"] for r in R])
openm = np.array([c in OPEN for c in cc])
d = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    h = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
    d[i] = 2 * 6371.0088 * np.arcsin(np.sqrt(np.minimum(h, 1)))
on = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)
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
for i in range(n):
    for q, v in R[i].get("near", []):
        j = pix.get(q)
        if j is not None:
            T[i, j] = v
            if not any(x[0] == R[i]["place"] for x in R[j].get("near", [])):
                T[j, i] = v
ex = np.array([cfg["places"].get(z, [0.])[0] for z in Z])
ex = np.minimum(ex, arg("--capx", 9.))
if "--nopull" in sys.argv:
    ex[:] = 0
gdom = np.array([gamma * cg.get(c, 1.0) for c in cc])  # by destination country
same = cc[:, None] == cc[None, :]
P = np.where(same, np.exp(gdom[None, :] * ex[None, :]), np.exp(gx * ex[None, :])).astype(np.float32)
W = (P - 1) * on(d, *cfg["phase_km"]) + 1
W[np.isin(cc, cfg["off"])] = 1.
del P
k = dep / (dep + (T * (W - 1)).sum(1))
Tn = T * W * k[:, None]
del W
print(f"gamma {gamma} gamma_x {gx} CN {cg['CN']}; {int((ex > 0).sum())} places with an excess")


def each(x, y, M=Tn):
    tot = 0.
    for yy in y.split("+"):
        i, j = ix[x], ix[yy]
        tot += M[i, j] + M[j, i]
    return tot  # both ways


if "--pairs" in sys.argv:
    PAIRS = [("war", "kra", "25,000"), ("war", "tro+gda+stz", "25,000"), ("ber", "mun", "22,000"), ("ham", "mun", "10,000"),
             ("bei", "sha", "65-160k"), ("jnj", "nag", "6,000"), ("war", "lod", ""), ("kra", "kat", ""), ("jnj", "yzo", ""),
             ("nyc", "bos", "53,000"), ("nyc", "wdc", "41,000"), ("lon", "man", ""), ("tyo", "osa", "")]
    for x, y, tg in PAIRS:
        print(f"  {x}-{y:12s} {each(x, y, T):9.0f} -> {each(x, y):9.0f}   {tg}")
    print("  big pairs at home:")
    for x, y in [("ber", "ham"), ("fra", "col"), ("mun", "stu"), ("col", "dus"), ("mil", "tur"), ("rom", "nap"), ("mil", "rom"), ("mad", "bar"), ("mad", "val"),
                 ("mad", "sev"), ("par", "lyo"), ("par", "lil"), ("lyo", "mar"), ("bei", "tia"), ("cho", "chg"), ("mos", "sai"), ("ist", "ank"), ("mum", "pun"), ("del", "agr")]:
        if x in ix and y in ix:
            print(f"  {x}-{y:4s} {d[ix[x], ix[y]]:5.0f} km {each(x, y, T):9.0f} -> {each(x, y):9.0f}  x{each(x, y) / max(each(x, y, T), 1):.1f}")
    az = json.load(open(HERE + r"\airport_zone.json"))
    fr = json.load(open(r"C:\Users\anita\projects\maps\flights\routes_fr24.json", encoding="utf-8"))["routes"]
    air = {}
    for a_, b_, nn, *_ in fr:
        za, zb = az.get(a_), az.get(b_)
        if za and zb:
            air[frozenset((za, zb))] = air.get(frozenset((za, zb)), 0) + nn
    print("  across a border (both ways a weekday; FR24 air a day, both ways, annual / 365):")
    for x, y in [("par", "bru"), ("vie", "bud"), ("ber", "war"), ("sin", "kua"), ("ber", "pra"), ("mun", "vie"), ("zur", "mil"), ("cop", "sto"),
                 ("par", "ams"), ("mad", "lis"), ("hon", "shz"), ("hon", "gua"), ("ban", "sin"), ("par", "ber"), ("war", "lon"), ("par", "lon")]:
        print(f"  {x}-{y:4s} {d[ix[x], ix[y]]:5.0f} km {each(x, y, T):9.0f} -> {each(x, y):9.0f}   air {air.get(frozenset((x, y)), 0) / 365:7.0f}")
for c in ["US", "JP", "GB", "DE", "PL", "FR", "CN", "IN", "IT", "ES"]:
    m = cc == c
    sub, sub0, dd = Tn[np.ix_(m, m)], T[np.ix_(m, m)], d[np.ix_(m, m)]
    pc = pop[m].sum()
    b1 = lambda M: M[(dd >= 100) & (dd < 300)].sum() / pc
    b3 = lambda M: M[dd >= 300].sum() / pc
    abroad = lambda M: M[m][:, ~m].sum() / pc
    print(f"  {c}: 100-300 {b1(sub0):.4f} -> {b1(sub):.4f}, 300+ {b3(sub0):.4f} -> {b3(sub):.4f}, abroad {abroad(T):.4f} -> {abroad(Tn):.4f}")
if "--save" in sys.argv:
    np.savez(HERE + rf"\scen_{sys.argv[sys.argv.index('--save') + 1]}.npz", T=T, Tn=Tn, d=d, ex=ex)
