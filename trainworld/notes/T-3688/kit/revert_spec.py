"""Undo one applied plan_edit spec (plan rows, extents, zone-names) from the backup taken before it, leaving every
other row as it is now: the spec's touched rows go back to the backup's copies, its new rows go, its retired rows
come back. Only for a spec whose retire step has not run (nothing registered or moved yet).
usage: revert_spec.py <spec.json> <backup stage folder> [--go]"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402

spec = json.load(open(sys.argv[1], encoding="utf-8"))
bk = sys.argv[2]
go = "--go" in sys.argv
rel = lambda p: os.path.join(bk, os.path.relpath(p, pe.TW))  # noqa: E731
touched = {m[2] for m in spec.get("moves", []) if m[2]} | {m[3] for m in spec.get("moves", [])} | set(spec.get("retire", {})) \
    | set(spec.get("retire", {}).values()) | set(spec.get("set", {})) | set(spec.get("replan", []))
new = set(spec.get("new", {}))
old_doc = json.load(open(rel(pe.PLAN), encoding="utf-8"))
old = {r["id"]: r for r in old_doc["metros"]}
back = touched - new
print("rows back to the backup:", sorted(back), "removed:", sorted(new))
with shared_lock.held("plan", "revert_spec T-3688"):
    doc = shared_lock.read_json(pe.PLAN)
    have = {r["id"] for r in doc["metros"]}
    out = []
    for r in old_doc["metros"]:  # backup order, today's rows, the touched ones from the backup
        if r["id"] in back:
            out.append(old[r["id"]])
    cur = {r["id"]: r for r in doc["metros"]}
    metros = [old[r["id"]] if r["id"] in back else r for r in doc["metros"] if r["id"] not in new]
    for k in back - have:  # retired rows: back in their backup place
        i = [r["id"] for r in old_doc["metros"]].index(k)
        prev = old_doc["metros"][i - 1]["id"]
        j = [r["id"] for r in metros].index(prev)
        metros.insert(j + 1, old[k])
    doc["metros"] = metros
    ext = shared_lock.read_json(pe.EXT)
    # backup.py does not keep extents.geojson: the touched rows' features again from their backup units, as plan_edit
    # draws them (union of units in lon/lat)
    import shapely
    import world_plan as wp
    from shapely.geometry import mapping
    by = pe.load_units(spec["iso"])
    oext = {}
    for k in back:
        r = old[k]
        g = shapely.set_precision(wp.to_ll(wp.polygonal(shapely.union_all([by[u].geom for u in r["units"][spec["iso"]]]))), 1e-5)
        oext[k] = {"type": "Feature", "properties": {"order": r["order"], "id": k, "name": r["name"], "people": r["people"]},
                   "geometry": mapping(g)}
    feats = [oext[f["properties"].get("id")] if f["properties"].get("id") in back else f for f in ext["features"]
             if f["properties"].get("id") not in new]
    feats += [oext[k] for k in back - {f["properties"].get("id") for f in feats} if k in oext]
    ext["features"] = feats
    print("plan rows", len(cur), "->", len(metros), "; extents", len(json.load(open(pe.EXT))["features"]), "->", len(feats))
    if go:
        shared_lock.write_atomic(pe.EXT, json.dumps(ext))
        shared_lock.write_atomic(pe.PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
# zone-names: put back the entries the spec dropped or set, from the backup
names = spec.get("names", {})
ks = set(names.get("drop", [])) | set(names.get("aliases", {})) | set(names.get("set", {}))
if ks:
    ob = {l.split(":")[0]: l.rstrip().rstrip(",") for l in open(rel(pe.NAMES), encoding="utf-8").read().splitlines() if l.startswith('"')}
    lines = open(pe.NAMES, encoding="utf-8").read().splitlines()
    lines = [l for l in lines if l.split(":")[0].strip('"') not in ks]
    for k in ks:
        if f'"{k}"' in ob:
            lines.insert(1, ob[f'"{k}"'] + ",")
    body = "\n".join(lines) + "\n"
    body = body.replace(",\n}", "\n}")
    json.loads(body)
    print("zone-names back:", sorted(ks))
    if go:
        open(pe.NAMES + ".part", "w", encoding="utf-8").write(body)
        os.replace(pe.NAMES + ".part", pe.NAMES)
print("written" if go else "(dry)")
