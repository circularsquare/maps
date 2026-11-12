"""T-3240: two places' longer trips a weekday each way (marketRule.ts placeTrips), by zone id.
    python pairs.py a-b c-d ...      or   python pairs.py --list CC
"""
import json, math, sys
sys.path.insert(0, r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist")
from bands import rows, rule, km, directed, byp  # noqa
G = r"C:\Users\anita\projects\maps\trainworld\app\src\game"
hand = json.load(open(G + r"\zone-names.json", encoding="utf-8"))
OPEN = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE NO IS CH LI GB".split())


def border(p, q):
    if p["country"] == q["country"]:
        return 1.
    gen = rule["border"] * min(p["inc"], 1) ** rule["border_eps"]
    return max(0.5, gen) if p["country"] in OPEN and q["country"] in OPEN else gen


def each_way(a, b):
    p, q = rows[a], rows[b]
    d = km(p, q)
    if d < rule["near_km"]:
        t = dict(p.get("near", [])).get(q["place"])
        if t is None:
            t = dict(q.get("near", [])).get(p["place"], 0.)
        return d, t
    return d, 0.5 * (border(p, q) * directed(p, q, d) + border(q, p) * directed(q, p, d))


if __name__ != "__main__":
    pass
elif sys.argv[1] == "--list":
    for z, r in sorted(rows.items(), key=lambda x: -x[1]["pop"]):
        if r["country"] == sys.argv[2]:
            print(f"{z} {hand.get(z, {}).get('name', '')!s:<20} pop {r['pop']/1e6:5.2f}M dep {r['dep']:8.0f} ({r['dep']/r['pop']:.4f})")
    sys.exit()
for arg in (sys.argv[1:] if __name__ == "__main__" else []):
    a, b = arg.split("-")
    d, t = each_way(a, b)
    print(f"{a}-{b}: {d:5.0f} km, {t:9.0f} each way a weekday, {2*t:9.0f} both ways; x365x1.17 = {2*t*365*1.166/1e6:.2f}M a year both ways")
