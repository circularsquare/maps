"""T-1504: the market's longer trips a day between China's places, from the rows alone, as
`Market::place_eachway` works them out (far pairs at the centres' distance; near pairs the rows'
sums). Prints the named pairs and the totals between the places given (default the 24 biggest).

    python notes/T-1504/market_pairs.py [ids...]
"""
import json
import math
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
reg = json.loads((root / "app/src/game/registry.json").read_text(encoding="utf-8"))
rows = json.loads((root / "app/src/game/market-rows.json").read_text(encoding="utf-8"))
R = reg["market"]["rule"]


def ground(d):
    on = min(max((d - R["onset_km"][0]) / (R["onset_km"][1] - R["onset_km"][0]), 0), 1)
    return 0 if on == 0 else on / max(d, R["d0_km"]) ** R["A"]


def tail(d):
    on = min(max((d - R["tail_onset_km"][0]) / (R["tail_onset_km"][1] - R["tail_onset_km"][0]), 0), 1)
    return 0 if on == 0 else on / max(d, R["D1_km"]) ** R["G"]


def km(p, q):
    r = math.pi / 180
    a = [math.cos(p["lat"] * r) * math.cos(p["lon"] * r), math.cos(p["lat"] * r) * math.sin(p["lon"] * r), math.sin(p["lat"] * r)]
    b = [math.cos(q["lat"] * r) * math.cos(q["lon"] * r), math.cos(q["lat"] * r) * math.sin(q["lon"] * r), math.sin(q["lat"] * r)]
    c = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 2 * 6371.0088 * math.asin(min(c / 2, 1))


def directed(p, q, d):
    rel = min(q["inc"] / p["inc"], 1) ** R["eps_t"] if p["inc"] > 0 else 1
    return p["g"] * ground(d) * q["a"] + p["t"] * tail(d) * q["ta"] * rel


def eachway(i, j):
    p, q = rows[i], rows[j]
    d = km(p, q)
    if d < R["near_km"]:
        near = dict((int(k), v) for k, v in (p.get("near") or []))
        if q["place"] in near:
            return near[q["place"]], d, "near(row)"
        return 0.5 * (directed(p, q, d) + directed(q, p, d)), d, "near(centres)"
    return 0.5 * (directed(p, q, d) + directed(q, p, d)), d, "far"


if __name__ == "__main__":
    cn = sorted([k for k, v in rows.items() if v.get("country") == "CN"], key=lambda k: -rows[k]["pop"])
    ids = sys.argv[1:] or cn[:24]
    for a, b in [("bei", "sha"), ("gua", "shz"), ("wuh", "chs"), ("bei", "tia"), ("sha", "nan"), ("bei", "gua"), ("sha", "hag"), ("wuh", "gua")]:
        t, d, how = eachway(a, b)
        print(f"{a}-{b}: {d:7.0f} km  {t:10.0f} trips a day each way  ({how})")
    tot = far = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            t, d, how = eachway(a, b)
            tot += 2 * t
            if d >= 250:
                far += 2 * t
    print(f"{len(ids)} places: {tot:,.0f} trips a day both ways between them, {far:,.0f} of them on pairs 250 km or more apart")
    dep = sum(rows[k]["dep"] for k in ids)
    pop = sum(rows[k]["pop"] for k in ids)
    print(f"their departures {dep:,.0f} a day, people {pop:,.0f} ({dep / pop:.4f} a person)")
    alldep = sum(rows[k]["dep"] for k in cn)
    allpop = sum(rows[k]["pop"] for k in cn)
    print(f"all {len(cn)} CN places: departures {alldep:,.0f} a day, people {allpop:,.0f}")
