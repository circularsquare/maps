"""Look up market places and their rows by name. python places.py <name substr> [...]"""
import json, sys, math
G = r"C:\Users\anita\projects\maps\trainworld\app\src\game"
reg = json.load(open(G + r"\registry.json", encoding="utf-8"))
rows = json.load(open(G + r"\market-rows.json", encoding="utf-8"))
names = json.load(open(G + r"\zone-names.json", encoding="utf-8"))
places = reg["market"]["places"]
zones = {}
for k, v in reg.items():
    if isinstance(v, list):
        for z in v:
            if isinstance(z, dict) and "place" in z and "id" in z:
                zones[z["place"]] = z
byplace = {r["place"]: (zid, r) for zid, r in rows.items()}

def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))

def show(pid):
    z = zones.get(pid, {})
    zid = z.get("id")
    disp = names.get(zid, {}).get("name") if zid else None
    print(f"place {pid}: market name {places[pid]['name']!r} regionName {places[pid]['regionName']!r} zone {zid} shown {disp!r} country {z.get('country')} localTrips {z.get('localTrips')}")
    if pid in byplace:
        zid2, r = byplace[pid]
        print(f"   row {zid2}: pop {r['pop']:.0f} dep {r['dep']:.0f} dep/pop {r['dep']/r['pop']:.4f} inc {r['inc']:.3f} zones {r['zones']}")
        near = sorted(r["near"], key=lambda x: -x[1])
        for q, t in near[:25]:
            o = byplace.get(q)
            d = km(r, o[1]) if o else float('nan')
            print(f"     -> {q:5d} {places[q]['regionName']:<32} {d:6.0f} km  {t:10.0f} trips/weekday (both ways?)")
        print("   keys", [k for k in r.keys() if k not in ("near", "commute", "air")])

for arg in sys.argv[1:]:
    if arg.isdigit():
        show(int(arg)); continue
    for pid, p in enumerate(places):
        if arg.lower() in p["name"].lower():
            show(pid)
