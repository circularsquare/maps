"""T-1411: rail against plane and car on real pairs, by the game's market rule.

Mirrors app/src/game/marketRule.ts (car, plane, split) and the sim's rail costs
(half-headway wait, the span tolerance penalty, rides to stations, the fare at India's
weight on the default curve). Rail's ride time and track km are real best trains.
"""
import json, math, struct, sys

TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
gdp = json.load(open(TW + r"\app\src\game\countryIncome.json", encoding="utf-8"))["gdp"]
reg = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))
RULE = reg["market"]["rule"]

def fw(c):
    return 5 * (gdp.get(c, gdp["IN"]) / 60000) ** -0.5

# drives table
b = open(TW + r"\data\packs\market-drives.bin", "rb").read()
assert b[:4] == b"TWD1"
p = 4
n = struct.unpack_from("<I", b, p)[0]; p += 4
ids = []
for i in range(n):
    L = b[p]; p += 1
    ids.append(b[p:p + L].decode()); p += L
counts = struct.unpack_from("<%dH" % n, b, p); p += 2 * n
drv = {}
keys = []
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
    c = math.dist(u, v)
    return 2 * R * math.asin(min(c / 2, 1))

def car(a, b_, km):
    d = drv.get(tuple(sorted((a["zone"], b_["zone"]))), [1, 1, 0, 0])
    rt, rk, fkm, fn = d
    drive = (15 + km * 1.25 / 85 * 60) * rt
    run = km * 1.25 * rk * 0.15 / 2 * 3
    park = 0.5 * (max(0, a["park"]) + max(0, b_["park"])) / 2
    cross = 0
    if (a["country"] == "GB") != (b_["country"] == "GB"): cross += 60 + 150 / 2 * 3
    if fn: cross += fn * 30 + fkm / 35 * 60 + (fn * 50 + 0.5 * fkm) / 2 * 3
    return drive + run + park + cross, drive, rt

def plane(a, b_, km):
    best = None
    for x in a["air"]:
        for y in b_["air"]:
            direct = b_["place"] in x[1] or a["place"] in y[1]
            fm = km / 750 * 60 + 30 + (0 if direct else 60)
            fare = 100 + 0.05 * km
            tot = x[0] + y[0] + 90 + fm + fare * fw(a["country"])
            if best is None or tot < best["total"]:
                best = dict(total=tot, acc=x[0] + y[0], direct=direct, flight=fm, fare=fare, fw=fw(a["country"]))
    return best

# span penalty: even headway H, riders arriving evenly, tolerance log-uniform 30 min - 6 h
def within(w):
    if w <= 30: return 1.0
    if w >= 360: return 0.0
    return math.log(360 / w) / math.log(12)
def span_pen(H):
    m = 2000
    phi = sum(within((i + 0.5) / m * H) for i in range(m)) / m
    phi = round(phi * 100) / 100
    return 0.0 if phi >= 1 else (-math.log(phi) / 0.10 if phi > 0 else float("inf"))

def wait_now(H): return H / 2 + span_pen(H)
def wait_new(H): return min(H / 2, 5 + H / 6, 30)

FW_IN = fw("IN")
def rail_fare(track_km):
    return FW_IN * (0.5357 + 0.03571 * track_km)

# real best trains: minutes, track km
PAIRS = [
    ("nyc", "bos", "New York - Boston", 215, 370),
    ("nyc", "wdc", "New York - Washington", 170, 362),
    ("par", "lyo", "Paris - Lyon", 117, 427),
    ("tyo", "osa", "Tokyo - Osaka", 147, 553),
    ("bei", "sha", "Beijing - Shanghai", 270, 1318),
    ("mad", "bar", "Madrid - Barcelona", 150, 621),
    ("lon", "cit", "London - Edinburgh", 260, 632),
]
ACCESS = {"centre": 15 + 15, "suburb": 65 + 65}  # 10 min walks at 1.5; rides 15 + 2.66 x 15 + 10 each end

def shares(beta, r, c, a):
    v = {k: math.exp(-beta * x) if x is not None else 0 for k, x in (("rail", r), ("car", c), ("air", a))}
    m = min(x for x in (r, c, a) if x is not None)
    v = {k: (math.exp(-beta * (x - m)) if x is not None else 0) for k, x in (("rail", r), ("car", c), ("air", a))}
    s = sum(v.values())
    return {k: v[k] / s for k in v}

out = []
print("span penalty by headway:", {H: round(span_pen(H), 1) for H in (30, 45, 60, 90, 120, 180, 240)})
print("wait now:", {H: round(wait_now(H), 1) for H in (15, 30, 60, 120, 180, 240)})
print("wait new:", {H: round(wait_new(H), 1) for H in (15, 30, 60, 120, 180, 240)})
print("fw IN %.2f" % FW_IN)
for za, zb, name, tmin, tkm in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B)
    c, cdrive, rt = car(A, B, km)
    f = plane(A, B, km)
    beta = 0.1 * math.sqrt(min(60 / c, 1))
    fare = rail_fare(tkm)
    print(f"\n{name}: {km:.0f} km, car {c:.0f} (drive {cdrive:.0f}, rt {rt:.2f}), beta {beta:.4f}")
    print(f"  plane {f['total']:.0f}: airport drives {f['acc']:.0f}, 90 terminal, flight {f['flight']:.0f}, fare ${f['fare']:.0f} x {f['fw']:.2f} = {f['fare']*f['fw']:.0f}, direct {f['direct']}")
    print(f"  rail ride {tmin}, fare {fare:.0f} (${0.5357+0.03571*tkm:.0f} India-curve; {country_scale if False else ''})")
    rec = dict(pair=name, km=km, car=c, plane=f["total"], plane_parts=f, beta=beta, fare=fare, ride=tmin, cases=[])
    for acc_name, acc in ACCESS.items():
        for H in (30, 60, 120, 240):
            for rule in ("now", "new"):
                w = wait_now(H) if rule == "now" else wait_new(H)
                r = acc + w + tmin + fare
                pa = f["total"] + (0 if rule == "now" else 25)
                s = shares(beta, r, c, pa)
                rec["cases"].append(dict(access=acc_name, H=H, rule=rule, rail=r, wait=w, s=s))
                print(f"  {acc_name:6s} H{H:3d} {rule}: rail {r:5.0f} (wait {w:5.1f}) plane {pa:5.0f} -> rail {100*s['rail']:5.1f} air {100*s['air']:5.1f} car {100*s['car']:5.1f}; rail of rail+air {100*s['rail']/(s['rail']+s['air']):5.1f}")
    out.append(rec)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "pairs.json", "w"), indent=1, default=str)
