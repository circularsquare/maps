"""T-1185/T-1189 rebuild: China's old zone rows out of the way of the replanned ones.

    python trainworld/notes/t1185/retire.py save     before the replan: the plan's Chinese zone rows
                                                     (ids and units) into old_zones.json beside this file,
                                                     their extents and origins into old_zones.geojson
    python trainworld/notes/t1185/retire.py go [--dry]
                                                     after it: every old zone id unregistered through the
                                                     runner's own unregister (entry, market pack, market
                                                     rows); the zone rows that kept their units set back
                                                     to planned (they rebuild with the township census);
                                                     the removed ids' packs, caches and res9 files moved to
                                                     data/work/pre-T1185/removed/, and every zone row's
                                                     tier 0 signal cache to data/work/pre-T1185/signals/
Run with no runner or join live. The backup (notes/t1150/run/backup.py T1185 pre CN,HK,MO,TW) first.
"""
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402

PLAN = os.path.join(TW, "pipeline", "world_plan.json")
SAVED = os.path.join(HERE, "old_zones.json")
DEST = os.path.join(TW, "data", "work", "pre-T1185")


def zone_rows():
    return {r["id"]: r for r in shared_lock.read_json(PLAN)["metros"]
            if r.get("kind") == "zone" and "CHN" in (r.get("units") or {})}


def move(paths, sub, dry):
    n = 0
    for p in paths:
        out = os.path.join(DEST, sub, os.path.relpath(p, TW))
        if not dry:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.move(p, out)
        n += 1
    return n


if sys.argv[1] == "save":
    if os.path.exists(SAVED):
        sys.exit(f"{SAVED} exists")
    rows = zone_rows()
    with open(SAVED, "w", encoding="utf-8") as f:
        json.dump({i: r["units"]["CHN"] for i, r in rows.items()}, f)
    ext = shared_lock.read_json(os.path.join(TW, "data", "work", "worldplan", "extents.geojson"))
    ext["features"] = [ft for ft in ext["features"] if ft["properties"].get("id") in rows]
    for ft in ext["features"]:
        ft["properties"]["origin"] = rows[ft["properties"]["id"]]["origin"]
    with open(os.path.join(HERE, "old_zones.geojson"), "w", encoding="utf-8") as f:
        json.dump(ext, f, ensure_ascii=False)
    print(f"{len(rows)} Chinese zone rows saved to {SAVED}, their extents and origins to old_zones.geojson (names.py)")
    sys.exit()

dry = "--dry" in sys.argv
with open(SAVED, encoding="utf-8") as f:
    old = json.load(f)
now = zone_rows()
kept = sorted(i for i in now if i in old and sorted(now[i]["units"]["CHN"]) == sorted(old[i]))
removed = sorted(i for i in old if i not in now)
new = sorted(i for i in now if i not in old)
print(f"old {len(old)}, now {len(now)}: kept {len(kept)}, removed {len(removed)}, new {len(new)}")
if dry:
    print("(dry: nothing written)")
else:
    with shared_lock.held("plan", "t1185 retire"):
        doc = shared_lock.read_json(PLAN)
        for r in doc["metros"]:
            if r.get("id") in kept and r.get("units"):
                r["status"] = "planned"
                r.pop("run", None)
        shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    import world_run
    run = world_run.Run({})
    have = {c["id"] for c in run.registry()["cities"]}
    todo = [i for i in old if i in have]
    run.unregister(todo)
    print(f"{len(todo)} unregistered; {len(kept)} kept rows set planned")
gone = []
for i in removed:
    for pat in (f"data/packs/{i}.*", f"data/work/{i}.*", f"data/work/res9/{i}.*"):
        gone += glob.glob(os.path.join(TW, pat))
print(f"removed ids: {move(gone, 'removed', dry)} files to {DEST}\\removed")
sig = [p for i in now for p in glob.glob(os.path.join(TW, "data", "work", f"{i}.tier0_signals.npz"))]
print(f"signal caches: {move(sig, 'signals', dry)} to {DEST}\\signals")
stale = []
for i in new:
    for pat in (f"data/packs/{i}.*", f"data/work/{i}.*"):
        stale += glob.glob(os.path.join(TW, pat))
print(f"files already under a new id (an id used long ago): {move(stale, 'stale', dry)} to {DEST}\\stale")
