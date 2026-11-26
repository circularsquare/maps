"""The merge rule's travel (world_plan.gravity_flows, GHSL on 20 km cells) from groups of units to every row near them.
usage: partners.py ISO <groups.json>   groups: {"label": ["unit name", ...]}; every row (current units) and every
other group is one area; prints each group's travel shares, biggest first. Read-only."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\notes\T-4364\kit")
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import shapely  # noqa: E402
import world_plan as wp  # noqa: E402

iso = sys.argv[1]
groups = json.load(open(sys.argv[2], encoding="utf-8"))
by = pe.load_units(iso)
names = {}
for u in by.values():
    names.setdefault(u.name, []).append(u)
doc = shared_lock.read_json(pe.PLAN)
gid = {}
for lab, ns in [(k, v) for k, v in groups.items() if not k.startswith("_")]:
    for n in ns:
        cs = names[n]
        if len(cs) > 1 and n in groups.get("_ids", {}):
            cs = [x for x in cs if x.id == groups["_ids"][n]]
        if len(cs) > 1 and "_near" in groups:
            pt = wp.to_moll(shapely.Point(*groups["_near"]))
            cs = [min(cs, key=lambda x: x.geom.distance(pt))]
        (u,) = cs
        gid[u.id] = lab
area = shapely.union_all([by[u].geom for u in gid]).buffer(220_000)
pop = wp.Pop()
pts = {}
for lab in groups:
    if lab.startswith("_"):
        continue
    g = shapely.union_all([by[u].geom for u, l in gid.items() if l == lab])
    pts[lab] = pop.inside(g)
for r in doc["metros"]:
    us = [u for u in (r.get("units") or {}).get(iso, []) if u in by and u not in gid]
    if not us:
        continue
    g = shapely.union_all([by[u].geom for u in us])
    if g.intersects(area):
        pts[r["id"]] = pop.inside(g)
fl = wp.gravity_flows(pts)
for lab in groups:
    if lab.startswith("_"):
        continue
    t = fl.get(lab, {})
    tot = sum(t.values()) or 1
    top = sorted(t.items(), key=lambda kv: -kv[1])[:6]
    print(lab, f"{pts[lab][2].sum():,.0f} people:", ", ".join(f"{k} {v / tot:.2f}" for k, v in top))
