"""Drop a sliver extract from planned rows: fixex.py id:extract[,id:extract...]"""
import json, os, sys
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402
drop = dict(x.split(":") for x in sys.argv[1].split(","))
plan = os.path.join(TW, "pipeline", "world_plan.json")
with shared_lock.held("plan", "t897 fixex"):
    doc = shared_lock.read_json(plan)
    for r in doc["metros"]:
        if r.get("id") in drop and r["status"] == "planned":
            before = list(r["extracts"])
            r["extracts"] = [e for e in r["extracts"] if e != drop[r["id"]]]
            print(r["id"], before, "->", r["extracts"])
    shared_lock.write_atomic(plan, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
