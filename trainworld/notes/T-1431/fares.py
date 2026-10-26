"""T-1431: plane and long rail fares, variants on real pairs (the market rule as in notes/T-1411/variants.py).

Rail: a direct line at the real best train time, fare km = 1.05 x the great circle (the game charges
the straight line between stops), the starting curve at India's weight (every country's default).
Waits: T-1430's planned wait at an hourly train (15); the plane's flat 25.
"""
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

def plane_parts(a, b_, km):
    """(time minutes without fare) of the best airport pair; the fare is added by the variant."""
    best = None
    for x in a["air"]:
        for y in b_["air"]:
            direct = b_["place"] in x[1] or a["place"] in y[1]
            t = x[0] + y[0] + 90 + 25 + km / 750 * 60 + 30 + (0 if direct else 60)
            best = t if best is None else min(best, t)
    return best

FW_IN = fw("IN")
WAIT = 15  # T-1430 at an hourly train

PAIRS = [  # rows, name, best train minutes, published rail of all / air of all / rail of rail+air (None = unknown)
    ("sha", "bei", "Shanghai - Beijing", 270, (None, None, 86)),
    ("bei", "gua", "Beijing - Guangzhou", 450, (None, None, 25)),
    ("tyo", "osa", "Tokyo - Osaka", 147, (64, None, 85)),
    ("tyo", "fuk", "Tokyo - Fukuoka", 286, (None, None, 11)),
    ("par", "lyo", "Paris - Lyon", 117, (None, None, 92)),
    ("par", "mar", "Paris - Marseille", 185, (None, None, 75)),
    ("mad", "bar", "Madrid - Barcelona", 150, (64, None, 78)),
    ("lon", "cit", "London - Edinburgh", 260, (None, None, 45)),
    ("nyc", "wdc", "New York - Washington", 170, (20, 5, 79)),
    ("nyc", "bos", "New York - Boston", 215, (4.6, 4, 53)),
]

def split(beta, xs):
    m = min(xs)
    v = [math.exp(-beta * (x - m)) for x in xs]
    s = sum(v)
    return [x / s for x in v]

# A variant gives (rail fare minutes, plane fare minutes) from (fare km, great-circle km, country weight,
# rail time without fare, plane time without fare).
def rail_curve(fkm):  # India curve dollars
    return 0.5357 + 0.03571 * fkm

def plane_usd(km, base=100, perkm=0.05):
    return base + perkm * km

def taper(d, d0, e):
    return (d0 / max(d, d0)) ** e

VARIANTS = {
    "today": lambda fkm, km, w, rt, pt: (FW_IN * rail_curve(fkm), w * plane_usd(km)),
    "plane $50+5c": lambda fkm, km, w, rt, pt: (FW_IN * rail_curve(fkm), w * plane_usd(km, 50)),
    "plane $50+8c": lambda fkm, km, w, rt, pt: (FW_IN * rail_curve(fkm), w * plane_usd(km, 50, 0.08)),
    "rail per-km halves past 200": lambda fkm, km, w, rt, pt: (FW_IN * (0.5357 + 0.03571 * (min(fkm, 200) + 0.5 * max(0, fkm - 200))), w * plane_usd(km)),
    "fare cap = trip time": lambda fkm, km, w, rt, pt: (min(FW_IN * rail_curve(fkm), rt), min(w * plane_usd(km), pt)),
    "fare cap = half trip time": lambda fkm, km, w, rt, pt: (min(FW_IN * rail_curve(fkm), 0.5 * rt), min(w * plane_usd(km), 0.5 * pt)),
    "inter-city weight x0.5": lambda fkm, km, w, rt, pt: (0.5 * FW_IN * rail_curve(fkm), 0.5 * w * plane_usd(km)),
    "inter-city weight x0.4": lambda fkm, km, w, rt, pt: (0.4 * FW_IN * rail_curve(fkm), 0.4 * w * plane_usd(km)),
    "inter-city weight x0.33": lambda fkm, km, w, rt, pt: (0.33 * FW_IN * rail_curve(fkm), 0.33 * w * plane_usd(km)),
    "taper (50/km)^0.3": lambda fkm, km, w, rt, pt: (taper(km, 50, 0.3) * FW_IN * rail_curve(fkm), taper(km, 50, 0.3) * w * plane_usd(km)),
    "taper (50/km)^0.4": lambda fkm, km, w, rt, pt: (taper(km, 50, 0.4) * FW_IN * rail_curve(fkm), taper(km, 50, 0.4) * w * plane_usd(km)),
    "taper (100/km)^0.5": lambda fkm, km, w, rt, pt: (taper(km, 100, 0.5) * FW_IN * rail_curve(fkm), taper(km, 100, 0.5) * w * plane_usd(km)),
    "x0.4 + plane $75+5c": lambda fkm, km, w, rt, pt: (0.4 * FW_IN * rail_curve(fkm), 0.4 * w * plane_usd(km, 75)),
    "x0.4, plane per-km 8c": lambda fkm, km, w, rt, pt: (0.4 * FW_IN * rail_curve(fkm), 0.4 * w * plane_usd(km, 100, 0.08)),
}

only = sys.argv[1:] or list(VARIANTS)
out = []
print("country weights:", {c: round(fw(c), 2) for c in ("US", "GB", "JP", "FR", "ES", "CN", "IN")})
print("default scales:", {c: round(FW_IN / fw(c), 2) for c in ("US", "GB", "JP", "FR", "ES", "CN")})
for za, zb, name, tmin, pub in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B)
    fkm = 1.05 * km
    c = car(A, B, km)
    beta = 0.1 * math.sqrt(min(60 / c, 1))
    w = fw(A["country"])
    pt = plane_parts(A, B, km)
    print(f"\n{name} ({A['country']}): {km:.0f} km, car {c:.0f}, beta {beta:.4f}, plane time {pt:.0f}, w {w:.2f}, pub {pub}")
    for vn in only:
        f = VARIANTS[vn]
        cells = []
        for acc in (30, 130):
            rt = acc + WAIT + tmin
            rf, pf = f(fkm, km, w, rt, pt)
            r, pl = rt + rf, pt + pf
            sr, sc, sa = split(beta, [r, c, pl])
            cells.append(dict(acc=acc, rail=r, plane=pl, rf=rf, pf=pf, sr=sr, sc=sc, sa=sa))
            out.append(dict(pair=name, variant=vn, acc=acc, km=km, car=c, rail=r, plane=pl, rail_fare=rf, plane_fare=pf,
                            rail_pct=100 * sr, car_pct=100 * sc, air_pct=100 * sa, r_ra=100 * sr / (sr + sa)))
        x, y = cells
        print(f"  {vn:30s} centre: rail {x['rail']:5.0f} (fare {x['rf']:4.0f}) plane {x['plane']:5.0f} (fare {x['pf']:4.0f}) -> "
              f"{100*x['sr']:4.0f}/{100*x['sa']:4.0f}/{100*x['sc']:4.0f} r/ra {100*x['sr']/(x['sr']+x['sa']):4.0f}"
              f" | suburb {100*y['sr']:4.0f}/{100*y['sa']:4.0f}/{100*y['sc']:4.0f} r/ra {100*y['sr']/(y['sr']+y['sa']):4.0f}")
json.dump(out, open(r"C:\Users\anita\projects\maps\artifacts\trainworld\T-1431\fares.json", "w"), indent=1)
