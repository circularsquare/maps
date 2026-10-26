"""T-1411: the proposal's variants on real pairs, by the game's market rule (see pairs.py)."""
import json, math, struct, sys

TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
gdp = json.load(open(TW + r"\app\src\game\countryIncome.json", encoding="utf-8"))["gdp"]

def fw(c):
    return 5 * (gdp.get(c, gdp["IN"]) / 60000) ** -0.5

b = open(TW + r"\data\packs\market-drives.bin", "rb").read()
p = 4
n = struct.unpack_from("<I", b, p)[0]; p += 4
ids = []
for i in range(n):
    L = b[p]; p += 1
    ids.append(b[p:p + L].decode()); p += L
counts = struct.unpack_from("<%dH" % n, b, p); p += 2 * n
drv, keys = {}, []
for a in range(n):
    for k in range(counts[a]):
        o, qt, qk = struct.unpack_from("<Hbb", b, p); p += 4
        key = tuple(sorted((ids[a], ids[o])))
        keys.append(key)
        drv.setdefault(key, [math.exp(qt / 64), math.exp(qk / 64), 0, 0])
nf = struct.unpack_from("<I", b, p)[0]; p += 4
for i in range(nf):
    ki, fk, fn = struct.unpack_from("<IHB", b, p); p += 7
    d = drv.get(keys[ki])
    if d: d[2], d[3] = fk, fn

R = 6371.0088
def gc(a, b_):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b_["lat"], b_["lon"]))
    u = [math.cos(la1) * math.cos(lo1), math.cos(la1) * math.sin(lo1), math.sin(la1)]
    v = [math.cos(la2) * math.cos(lo2), math.cos(la2) * math.sin(lo2), math.sin(la2)]
    return 2 * R * math.asin(min(math.dist(u, v) / 2, 1))

def car(a, b_, km):
    rt, rk, fkm, fn = drv.get(tuple(sorted((a["zone"], b_["zone"]))), [1, 1, 0, 0])
    if km > 1500: rt, rk = 1, 1
    drive = (15 + km * 1.25 / 85 * 60) * rt
    run = km * 1.25 * rk * 0.15 / 2 * 3
    park = 0.5 * (max(0, a["park"]) + max(0, b_["park"])) / 2
    cross = 0
    if (a["country"] == "GB") != (b_["country"] == "GB"): cross += 60 + 150 / 2 * 3
    if fn: cross += fn * 30 + fkm / 35 * 60 + (fn * 50 + 0.5 * fkm) / 2 * 3
    return drive + run + park + cross

def plane(a, b_, km, w):
    best = None
    for x in a["air"]:
        for y in b_["air"]:
            direct = b_["place"] in x[1] or a["place"] in y[1]
            tot = x[0] + y[0] + 90 + km / 750 * 60 + 30 + (0 if direct else 60) + (100 + 0.05 * km) * w
            best = tot if best is None else min(best, tot)
    return best

def within(w):
    if w <= 30: return 1.0
    if w >= 360: return 0.0
    return math.log(360 / w) / math.log(12)
def span_pen(H):
    m = 2000
    phi = round(sum(within((i + 0.5) / m * H) for i in range(m)) / m * 100) / 100
    return 0.0 if phi >= 1 else -math.log(phi) / 0.10
def wait_now(H): return H / 2 + span_pen(H)
def wait_new(H): return min(H / 2, 5 + H / 6, 30)

FW_IN = fw("IN")
rail_fare = lambda tkm: FW_IN * (0.5357 + 0.03571 * tkm)

PAIRS = [
    ("nyc", "bos", "New York - Boston", 215, 370),
    ("nyc", "wdc", "New York - Washington", 170, 362),
    ("par", "lyo", "Paris - Lyon", 117, 427),
    ("tyo", "osa", "Tokyo - Osaka", 147, 553),
    ("mad", "bar", "Madrid - Barcelona", 150, 621),
    ("lon", "cit", "London - Edinburgh", 260, 632),
    ("par", "mar", "Paris - Marseille", 185, 750),
    ("bei", "sha", "Beijing - Shanghai", 270, 1318),
    ("tyo", "fuk", "Tokyo - Fukuoka", 286, 1106),
    ("bei", "gua", "Beijing - Guangzhou", 450, 2298),
]
ACC = 30  # a 10 minute walk at each end, 1.5 a minute

def split(beta, r, c, a):
    xs = [r, c, a]
    m = min(xs)
    v = [math.exp(-beta * (x - m)) for x in xs]
    s = sum(v)
    return [x / s for x in v]

VARIANTS = [
    ("today", wait_now, None, 0),
    ("long wait, plane +25", wait_new, None, 25),
    ("long wait, plane +25, plane fare at 3/$", wait_new, 3.0, 25),
    ("today waits, plane fare at 3/$", wait_now, 3.0, 0),
]
res = []
for za, zb, name, tmin, tkm in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B)
    c = car(A, B, km)
    beta = 0.1 * math.sqrt(min(60 / c, 1))
    for H in (60, 120):
        for vn, wf, pw, padd in VARIANTS:
            pl = plane(A, B, km, pw if pw else fw(A["country"])) + padd
            r = ACC + wf(H) + tmin + rail_fare(tkm)
            sr, sc, sa = split(beta, r, c, pl)
            res.append(dict(pair=name, km=round(km), H=H, variant=vn, rail=round(r), car=round(c), plane=round(pl),
                            rail_pct=round(100 * sr, 1), car_pct=round(100 * sc, 1), air_pct=round(100 * sa, 1),
                            rail_of_rail_air=round(100 * sr / (sr + sa), 1)))
for x in res:
    print(f"{x['pair']:22s} {x['km']:5d} H{x['H']:3d} {x['variant']:40s} rail {x['rail']:5d} car {x['car']:5d} plane {x['plane']:5d} | rail {x['rail_pct']:5.1f} car {x['car_pct']:5.1f} air {x['air_pct']:5.1f} | r/(r+a) {x['rail_of_rail_air']:5.1f}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
