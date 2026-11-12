"""T-3240: the market's domestic longer trips by country and straight-line band, from the rows.

Near pairs (under 250 km) from the rows' `near` (each way, place-centre distance for the band);
far pairs by the rule at place centres (marketRule.ts `directed`, both ways summed).
Prints trips a person a weekday, one way (each direction counted, as NHTS and MLIT count trips).
    python bands.py [CC ...]
"""
import json, math, collections, sys
TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rule = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
R = list(rows.values())
byp = {r["place"]: r for r in R}


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(1, h)))


def on(d, lo, hi):
    return min(1., max(0., (d - lo) / (hi - lo)))


def ground(d):
    o = on(d, *rule["onset_km"])
    return 0. if o == 0 else o * max(d, rule["d0_km"]) ** -rule["A"] * (1 - on(d, *rule["tail_onset_km"]))


def tail(d):
    o = on(d, *rule["tail_onset_km"])
    return 0. if o == 0 else o * max(d, rule["D1_km"]) ** -rule["G"]


def directed(p, q, d):
    ip, iq = min(p["inc"], 1), min(q["inc"], 1)
    rel = min(iq / ip, 1) ** rule["eps_t"] if ip > 0 else 1
    gr = p["g"] * ground(d) * q["a"] if p["g"] > 0 else 0
    tl = p["t"] * tail(d) * q["ta"] * rel if p["t"] > 0 else 0
    return gr + tl  # domestic: border 1


EDGES = [0, 50, 100, 150, 200, 300, 500, 700, 1000, 1e9]
want = [] if __name__ != "__main__" else sys.argv[1:] or ["US", "JP", "DE", "FR", "GB", "PL", "CN", "ES", "IT", "KR"]
for c in want:
    ids = [r for r in R if r["country"] == c]
    pop = sum(r["pop"] for r in ids)
    dep = sum(r["dep"] for r in ids)
    band = collections.Counter()
    selfp = 0.
    for p in ids:
        near = dict((q, t) for q, t in p.get("near", []))
        for q in ids:
            if q is p:
                selfp += near.get(p["place"], 0.)
                continue
            d = km(p, q)
            if d < rule["near_km"]:
                t = near.get(q["place"])
                if t is None:
                    t = dict(q.get("near", [])).get(p["place"], 0.)
            else:
                t = 0.5 * (directed(p, q, d) + directed(q, p, d))
            for i in range(len(EDGES) - 1):
                if EDGES[i] <= d < EDGES[i + 1]:
                    band[i] += t  # p -> q each way; summing over ordered pairs counts both ways
                    break
    print(f"{c}: places {len(ids)}, people {pop/1e6:.1f}M, departures {dep/pop:.4f} a person a weekday; within a place (25 km+) {selfp/pop:.4f}")
    s100 = sum(band[i] for i in range(2, len(EDGES) - 1))
    s300 = sum(band[i] for i in range(5, len(EDGES) - 1))
    print("   " + "  ".join(f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}: {band[i]/pop:.5f}" for i in range(len(EDGES) - 1)))
    print(f"   100 km+ between places {s100/pop:.4f} ({s100*365/1e6:.0f}M a year at weekday rate), 300 km+ {s300/pop:.4f}; 100-300 {(s100-s300)/pop:.4f}")
