"""T-2344, Crimea's zone split by hand. The zones step gives Russia Crimea's councils (DE_FACTO_TAKE) and zone_groups
merges them into one zone of 2.02M over the 1.5M cap (Sevastopol and Yevpatoriia, under the floor with no neighbour
within the cap, join the smallest neighbour). Three zones instead, by where each council lies:
- Sevastopol: the city of Sevastopol (its ADM1) and the south coast below 44.8 N west of 34.7 E (Yalta, Alushta,
  Bakhchysarai);
- Kerch: east of 34.7 E (Kerch, Feodosiia, Sudak, Lenine, Kirovske, Sovietskyi, Nyzhniohirskyi);
- Simferopol: the rest (Simferopol, Yevpatoriia, Saky, Dzhankoi, the north), keeping the merged row's id and order.
Origins at the three city centres (T-2299: origins off their towns). People from GHS-POP over the units.

    python crimea_patch.py          dry: prints the rows, writes nothing
    python crimea_patch.py --go     under the plan lock: writes world_plan.json and extents.geojson
"""
import contextlib
import json
import sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
from shapely.geometry import mapping  # noqa: E402
import shapely  # noqa: E402
import shared_lock  # noqa: E402
import world_plan as wp  # noqa: E402

TW = r"C:\Users\anita\projects\maps\trainworld"
PLAN = TW + r"\pipeline\world_plan.json"
EXT = TW + r"\data\work\worldplan\extents.geojson"
GO = "--go" in sys.argv
if "--plan" in sys.argv and not GO:  # a dry plan (world_plan zones --out) instead of the live one
    PLAN = sys.argv[sys.argv.index("--plan") + 1]
    EXT = PLAN[:-5] + ".extents.geojson"
ORIGIN = {"Simferopol": [34.1024, 44.9521], "Sevastopol": [33.5254, 44.6054], "Kerch": [36.4700, 45.3570]}

import pickle  # noqa: E402


class Here(pickle.Unpickler):
    def find_class(self, mod, name):
        return wp.Unit if name == "Unit" else super().find_class(mod, name)


with open(wp.WORK / "state.pkl", "rb") as f:
    st = Here(f).load()
taken = {u.id: u for u in wp.taken_units(st, "RUS")}
pop = wp.Pop()


def piece(u):
    if u.parents and u.parents[0][1] == "Sevastopol":
        return "Sevastopol"
    c = wp.to_ll(u.geom.centroid)
    if c.x >= 34.7:
        return "Kerch"
    return "Sevastopol" if c.y < 44.8 else "Simferopol"


with shared_lock.held("plan", "T-2344 Crimea by hand") if GO else contextlib.nullcontext():
    doc = shared_lock.read_json(PLAN)
    ext = shared_lock.read_json(EXT)
    rows = [r for r in doc["metros"] if r.get("kind") == "zone" and set(r.get("units", {}).get("RUS", [])) & set(taken)]
    assert rows and all(r.get("status") == "planned" for r in rows), [(r["id"], r.get("status")) for r in rows]
    assert all(set(r["units"]["RUS"]) <= set(taken) for r in rows)
    ids = [i for r in rows for i in r["units"]["RUS"]]
    assert set(ids) == set(taken), (len(ids), len(taken))
    base = max(rows, key=lambda r: r["people"])
    used = {r["id"] for r in doc["metros"]} | {c["id"] for c in json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["cities"]}
    order = max(r.get("order", 0) for r in doc["metros"])
    parts = {}
    for i in ids:
        parts.setdefault(piece(taken[i]), []).append(taken[i])
    new = []
    for name in ("Simferopol", "Sevastopol", "Kerch"):
        us = parts[name]
        g = wp.polygonal(shapely.union_all([u.geom for u in us]))
        n = {u.id: float(pop.inside(u.geom)[2].sum()) for u in us}
        us.sort(key=lambda u: -n[u.id])
        ll = wp.to_ll(g)
        r = json.loads(json.dumps(base))
        if name == "Simferopol":
            r["id"] = base["id"]
        else:
            r["id"] = wp.make_id(name, used)
            used.add(r["id"])
            order += 1
            r["order"] = order
        r.update({"name": name, "people": round(sum(n.values())), "origin": ORIGIN[name], "units": {"RUS": [u.id for u in us]},
                  "bbox": [round(v, 3) for v in ll.bounds]})
        r["source"] = {**base.get("source", {}), "unit_names": [u.name for u in us],
                       "hand": {"task": "T-2344", "why": "Crimea's one zone of 2.02M split in three by where its councils lie"}}
        r.pop("gate_relax", None)
        new.append((r, ll))
        print(r["id"], r["order"], name, r["people"], len(us), r["origin"], [u.name for u in us[:4]])
    drop = {r["id"] for r in rows}
    doc["metros"] = [r for r in doc["metros"] if r["id"] not in drop] + [r for r, _ in new]
    feats = {f["properties"]["id"]: f for f in ext["features"]}
    proto = feats[base["id"]]
    for i in drop:
        feats.pop(i, None)
    for r, ll in new:
        f = json.loads(json.dumps(proto))
        f["properties"].update({k: r[k] for k in ("id", "name", "people", "order") if k in f["properties"]})
        f["geometry"] = mapping(shapely.set_precision(ll, 1e-5))
        feats[r["id"]] = f
    ext["features"] = list(feats.values())
    if GO:
        shared_lock.write_atomic(EXT, json.dumps(ext))
        shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print("written")
    else:
        print("dry: nothing written")
