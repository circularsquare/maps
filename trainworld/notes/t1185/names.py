"""T-1185/T-1189 rebuild, after the join: the hand-set names (app/src/game/zone-names.json, T-897's
Chinese name rules) of the old zone rows, carried to the new zone rows. A new zone takes an old
zone's hand-set name when each holds the other's origin (densest square km), so the name still
says the city at its centre ("Xinxiang" for Muye's zone); its registry name becomes an alias. Rows
that kept their id keep their entry. With --write, the carried entries are added (none
overwritten) and the removed ids' entries dropped; without it, printed only. The rest of the new
zones need a look against docs/zones.md "Names".

    python trainworld/notes/t1185/names.py [--write]
    (--plan= --ext= --old= --names=: a dry test on a scratch plan and its extents)
"""
import json
import os
import sys

import shapely
from shapely.geometry import Point, shape

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
sys.stdout.reconfigure(encoding="utf-8")
opt = {a.split("=")[0]: a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
NAMES = opt.get("--names", os.path.join(TW, "app", "src", "game", "zone-names.json"))
with open(opt.get("--plan", os.path.join(TW, "pipeline", "world_plan.json")), encoding="utf-8") as f:
    rows = {r["id"]: r for r in json.load(f)["metros"] if r.get("kind") == "zone" and "CHN" in (r.get("units") or {})}
with open(opt.get("--ext", os.path.join(TW, "data", "work", "worldplan", "extents.geojson")), encoding="utf-8") as f:
    new_g = {ft["properties"]["id"]: shape(ft["geometry"]) for ft in json.load(f)["features"] if ft["properties"].get("id") in rows}
with open(opt.get("--old", os.path.join(HERE, "old_zones.geojson")), encoding="utf-8") as f:
    old = {ft["properties"]["id"]: (shape(ft["geometry"]), Point(ft["properties"]["origin"])) for ft in json.load(f)["features"]}
with open(NAMES, encoding="utf-8") as f:
    names = json.load(f)
ids_old = list(old)
tree = shapely.STRtree([old[i][0] for i in ids_old])
carried, plain = {}, []
for i, r in rows.items():
    if i in old:
        continue
    o = Point(r["origin"])
    hit = [ids_old[k] for k in tree.query(o, predicate="intersects")]
    src = next((h for h in hit if i in new_g and new_g[i].intersects(old[h][1])), None)
    if src and isinstance(names.get(src), dict) and names[src].get("name"):
        e = names[src]
        al = [a for a in dict.fromkeys(e.get("aliases", []) + [r["name"]]) if a != e["name"]]
        carried[i] = {"name": e["name"], **({"aliases": al} if al else {})}
        print(f"  {i} {r['name']} -> {e['name']} (from {src})")
    else:
        plain.append(i)
removed = [i for i in old if i not in rows]
dead = [i for i in removed if i in names]
print(f"{len(rows)} zone rows: {sum(i in old for i in rows)} kept their id; {len(carried)} new ones take a hand-set name; "
      f"{len(plain)} keep the planner's name; {len(dead)} hand-set entries belong to removed ids "
      f"({len({names[d].get('name') for d in dead} - {c['name'] for c in carried.values()} - {None})} of their names not carried)")
if "--write" in sys.argv:
    for i, e in carried.items():
        names.setdefault(i, e)
    for i in dead:
        names.pop(i, None)
    tmp = NAMES + ".part"
    with open(tmp, "w", encoding="utf-8") as f:  # the file's own layout: one zone a line
        f.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in names.items()) + "\n}\n")
    os.replace(tmp, NAMES)
    print(f"written {NAMES}")
