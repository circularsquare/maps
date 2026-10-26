"""T-279 research: routed truth from OSRM's public demo (table service), one request per region.
Anchors: big places at least 700 km apart; each anchor's table holds up to 24 places of its
landmass within 450 km, spread out. Polite: one request per 6 s, a plain User-Agent, cached.
python -I osrm_tables.py <market-rows.json> <out dir>"""
import json, math, os, sys, time, urllib.request

rows = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
cache_f = os.path.join(OUT, "tables.json")
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
R = 6371.0088


def gc(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(h)))


P = [(k, (r["lon"], r["lat"]), r["road"], r["pop"], r["country"]) for k, r in rows.items()]
P.sort(key=lambda p: -p[3])
anchors = []
for p in P:
    if all(gc(p[1], a[1]) > 700 for a in anchors):
        anchors.append(p)
if len(sys.argv) > 3 and sys.argv[3] == "round2":
    # a second set between the first: places at least 350 km from every first anchor and each other
    first = list(anchors)
    anchors = []
    for p in P:
        if all(gc(p[1], a[1]) > 350 for a in first) and all(gc(p[1], a[1]) > 500 for a in anchors):
            anchors.append(p)
print(len(anchors), "anchors", flush=True)

for a in anchors:
    if a[0] in cache:
        continue
    cand = [p for p in P if p[2] == a[2] and gc(p[1], a[1]) < 450]
    if len(cand) < 4:
        cache[a[0]] = {"ids": [], "skip": "few"}
        continue
    pick = [a]
    while len(pick) < min(24, len(cand)):
        best = max((p for p in cand if p not in pick), key=lambda p: min(gc(p[1], q[1]) for q in pick) * (1 + 0.15 * math.log10(max(p[3], 1e4) / 1e4)))
        pick.append(best)
    coords = ";".join(f"{p[1][0]:.5f},{p[1][1]:.5f}" for p in pick)
    url = f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=duration,distance"
    req = urllib.request.Request(url, headers={"User-Agent": "trainworld-research/1.0 (road detour study)"})
    j = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as f:
                j = json.load(f)
            break
        except Exception as e:  # noqa: BLE001
            print("retry", a[0], e, flush=True)
            time.sleep(30)
    if j is None or j.get("code") != "Ok":
        print("failed", a[0], j and j.get("code"), flush=True)
        time.sleep(6)
        continue
    cache[a[0]] = {"ids": [p[0] for p in pick], "dur": j["durations"], "dist": j["distances"],
                   "snap": [s.get("distance") for s in j.get("sources", [])]}
    json.dump(cache, open(cache_f, "w"))
    print(a[0], a[4], len(pick), flush=True)
    time.sleep(6)
print("done", flush=True)
