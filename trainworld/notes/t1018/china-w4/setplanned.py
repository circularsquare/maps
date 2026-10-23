"""Set rows back to planned (for a replan), backing up the plan and extents first: setplanned.py id,id"""
import json, os, shutil, sys

TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402

ids = set(sys.argv[1].split(","))
plan = os.path.join(TW, "pipeline", "world_plan.json")
bk = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bk")
os.makedirs(bk, exist_ok=True)
shutil.copy2(plan, os.path.join(bk, "world_plan.json"))
shutil.copy2(os.path.join(TW, "data", "work", "worldplan", "extents.geojson"), os.path.join(bk, "extents.geojson"))
with shared_lock.held("plan", "t788b setplanned"):
    doc = shared_lock.read_json(plan)
    for r in doc["metros"]:
        if r.get("id") in ids and r.get("units"):
            print(r["id"], r["status"], "-> planned")
            r["status"] = "planned"
            r.pop("run", None)
    shared_lock.write_atomic(plan, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
