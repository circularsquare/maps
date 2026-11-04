"""T-1150 rebuild: relax the gate's pop_vs_plan on the rows whose 2020 township census lies outside
its 0.7-1.5 of the plan's GHSL people (census_vs_plan.py; county_check.py agrees with the townships
within 1%). Run with the world plan free (no runner or join holding it).
    python trainworld/notes/t1150/prerelax.py id,id,...        (--dry to print only)"""
import json
import os
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402

WHY = ("T-1150: the pack holds the 2020 census by township, which GHSL's older counts miss in China "
       "(the boom cities short, the shrinking ones long); the townships agree with the county census within 1%")
ids = set(sys.argv[1].split(","))
plan = os.path.join(TW, "pipeline", "world_plan.json")
with shared_lock.held("plan", "t1150 prerelax"):
    doc = shared_lock.read_json(plan)
    for r in doc["metros"]:
        if r.get("id") in ids:
            print(r["id"], r["status"], "pop_vs_plan relaxed")
            r.setdefault("gate_relax", {})["pop_vs_plan"] = WHY
    if "--dry" not in sys.argv:
        shared_lock.write_atomic(plan, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
