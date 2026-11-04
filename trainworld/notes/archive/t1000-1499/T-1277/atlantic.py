"""T-1275: transatlantic trips and rail's share against the plane, from the market rows and the
registry's rule (a mirror of app/src/game/marketRule.ts and intercity.rs's pair rule)."""
import json
import math

G = "C:/Users/anita/projects/maps/trainworld/app/src/game/"
reg = json.load(open(G + "registry.json", encoding="utf-8"))
R = reg["market"]["rule"]
rows = json.load(open(G + "market-rows.json", encoding="utf-8"))
byplace = {r["place"]: r for r in rows.values()}

EARTH_R = 6371008.8
OPEN = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE NO IS CH LI GB".split())


def unit(r):
    lo, la = math.radians(r["lon"]), math.radians(r["lat"])
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def km(a, b):
    u, v = unit(a), unit(b)
    c = math.dist(u, v)
    return 2 * EARTH_R / 1000 * math.asin(min(c / 2, 1))


def ground(d):
    on = min(1, max(0, (d - R["onset_km"][0]) / (R["onset_km"][1] - R["onset_km"][0])))
    return 0 if on == 0 else on / max(d, R["d0_km"]) ** R["A"]


def tail(d):
    on = min(1, max(0, (d - R["tail_onset_km"][0]) / (R["tail_onset_km"][1] - R["tail_onset_km"][0])))
    return 0 if on == 0 else on / max(d, R["D1_km"]) ** R["G"]


inc = lambda r: min(r["inc"], 1)


def border(p, q):
    if p["country"] == q["country"]:
        return 1
    gen = R["border"] * inc(p) ** R["border_eps"]
    return max(0.5, gen) if p["country"] in OPEN and q["country"] in OPEN else gen


def directed(p, q, d):
    rel = min(inc(q) / inc(p), 1) ** R["eps_t"] if inc(p) > 0 else 1
    gr = p["g"] * ground(d) * q["a"] if p["g"] > 0 else 0
    tl = p["t"] * tail(d) * q["ta"] * rel if p["t"] > 0 else 0
    return border(p, q) * (gr + tl)


def eachway(p, q):
    d = km(p, q)
    return 0.5 * (directed(p, q, d) + directed(q, p, d)), d


def air_min(p, q, d):
    best = None
    for x in p.get("air") or []:
        for y in q.get("air") or []:
            direct = q["place"] in x[1] or p["place"] in y[1]
            v = x[0] + y[0] + 90 + d / 750 * 60 + 30 + (0 if direct else 60) + (100 + 0.05 * d) * 3
            if best is None or v < best[0]:
                best = (v, direct, x[0] + y[0])
    return best


def car_min(d, park):
    return 15 + d * 1.25 / 85 * 60 + (d * 1.25 * 0.15 * 3 + park) / 2


nyc, lon = rows["nyc"], rows["lon"]
d = km(nyc, lon)
t, _ = eachway(nyc, lon)
a = air_min(nyc, lon, d)
cm = car_min(d, 0.5 * (nyc["park"] + lon["park"]))
bt = 0.1 * math.sqrt(min(60 / cm, 1))
print(f"New York - London: {d:.0f} km, trips a weekday each way {t:.0f}, plane {a[0]:.0f} perceived min (direct {a[1]}, access {a[2]:.0f}), car minutes for the scale {cm:.0f}, scale {bt:.4f}/min")

# metro areas: places within 150 km of each centre
def near(c, r_km):
    return [r for r in byplace.values() if km(r, c) <= r_km]

for rad in (100, 200, 400):
    A, B = near(nyc, rad), near(lon, rad)
    s = sum(eachway(p, q)[0] for p in A for q in B)
    print(f"places within {rad} km: NY side {len(A)} ({sum(p['pop'] for p in A)/1e6:.1f}M people), London side {len(B)} ({sum(q['pop'] for q in B)/1e6:.1f}M), trips a weekday each way {s:.0f}")

# whole continents
NA = [r for r in byplace.values() if r["country"] in ("US", "CA")]
EU = [r for r in byplace.values() if -25 < r["lon"] < 45 and r["lat"] > 34 and r["country"] not in ("TR", "RU", "MA", "DZ", "TN", "EG", "LY", "IL", "LB", "SY", "JO", "CY", "IR", "IQ")]
s = sum(eachway(p, q)[0] for p in NA for q in EU)
print(f"US+Canada places {len(NA)} ({sum(p['pop'] for p in NA)/1e6:.0f}M), Europe places {len(EU)} ({sum(q['pop'] for q in EU)/1e6:.0f}M): trips a weekday each way {s:.0f}")
sgb = sum(eachway(p, q)[0] for p in NA for q in EU if q["country"] == "GB")
print(f"  of them with Great Britain {sgb:.0f}")
AVG_DAY = (5 + 1.69 + 1.48) / 7
print(f"average day factor {AVG_DAY:.3f}")
json.dump({"d": d, "t_pair": t, "air": a[0], "car_scale_min": cm, "bt": bt}, open("C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/c3ba3638-10f7-4fdb-afc0-1d96a1000507/scratchpad/ss-econ/atlantic.json", "w"))
