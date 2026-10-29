"""T-1765: the origin world_plan's rule (densest GHSL pixel of the unit the zone leads with, outside metro packs) gives
each named zone row, without a replan. python new_origins.py id [id...] -> JSON lines {id: [lon, lat]} in out file argv: --out f"""
import json
import pickle
import sys

sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import world_plan as wp  # noqa: E402
from shapely import contains_xy  # noqa: E402

args = sys.argv[1:]
out = None
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    args = args[:i] + args[i + 2:]


class Here(pickle.Unpickler):
    def find_class(self, mod, name):
        return wp.Unit if name == "Unit" else super().find_class(mod, name)


with open(wp.WORK / "state.pkl", "rb") as f:
    st = Here(f).load()
by = {u.id: u for u in st["units"]}
pop = wp.Pop()
plan = json.loads(wp.OUT.read_text(encoding="utf-8"))
rows = {r["id"]: r for r in plan["metros"] if r.get("units")}
zone_ids = {i for i, r in rows.items() if r.get("kind") == "zone"}
pk = {k: v for k, v in wp.packs().items() if k not in zone_ids}
res = {}
for cid in args:
    r = rows[cid]
    iso = list(r["units"])[0]
    lead = next(i for i in r["units"][iso] if i in by and by[i].name == r["source"]["unit_names"][0])
    u = by[lead]
    x, y, p = pop.inside(u.geom)
    keep = np.ones(len(x), bool)
    for q in pk.values():
        if q["geom"].intersects(u.geom):
            keep &= ~contains_xy(q["geom"], x, y)
    x, y, p = x[keep], y[keep], p[keep]
    i = int(np.argmax(p))
    lon, lat = wp.INV.transform(x[i], y[i])
    res[cid] = [round(lon, 4), round(lat, 4)]
    print(cid, r["name"], iso, "lead", u.name, "old", r["origin"], "new", res[cid], "pixel pop", round(float(p[i])), flush=True)
if out:
    json.dump(res, open(out, "w"))
