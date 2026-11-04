"""Medical facilities enrich_medical drops for a pack (no cell within SNAP_M): medmiss.py <id>"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import numpy as np
import enrich_medical as em
b = em.b
city = sys.argv[1]
print("in CITIES:", city in b.CITIES)
cfg = b.CITIES[city]
boundary = b.build_boundary(city, cfg)
kept, merged = em.facilities(city, boundary)
stem = os.path.join(b.PACKS, city)
header = json.load(open(stem + ".json"))
buf = open(stem + ".bin", "rb").read()
arr = {a["name"]: a for a in header["arrays"]}
pop = np.frombuffer(buf, dtype="<f4", count=arr["pop"]["count"], offset=arr["pop"]["offset"]).sum(dtype=np.float64)
table = em.load_table()
hosp = [r for r in kept if r["kind"] == "hospital"]
em.size_hospitals(hosp, pop, em.country_of(city, header), table)
for r in kept:
    r["size"] = em.VISITS_PER_BED * r["beds_used"] if r["kind"] == "hospital" else em.SIZE[r["kind"]]
a = arr["h3"]
cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"])
index = {int(c): i for i, c in enumerate(cells)}
tot, miss = 0.0, 0.0
for r in sorted(kept, key=lambda r: -r["size"]):
    tot += r["size"]
    cell, snap = em.air.cell_of(index, r["lat"], r["lon"], em.SNAP_M)
    if cell is None:
        miss += r["size"]
        print(f"  off: {r['kind']:8} {r['size']:>10,.0f} {r['name'][:50]} ({r['lat']:.4f}, {r['lon']:.4f}) beds {r.get('beds_used')}")
print(f"{len(kept)} kept, {miss / tot:.1%} of visits off every cell")
