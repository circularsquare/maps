"""Set parked rows back to built (their packs are on disk) so the next run gates them again,
optionally relaxing one gate check with a reason.
setbuilt.py id,id,... [check "reason"]"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402

ids = set(sys.argv[1].split(","))
relax = sys.argv[2:4] if len(sys.argv) >= 4 else None
plan = os.path.join(TW, "pipeline", "world_plan.json")
with shared_lock.held("plan", "t897 setbuilt"):
    doc = shared_lock.read_json(plan)
    for r in doc["metros"]:
        if r.get("id") in ids and r.get("units"):
            ok = all(os.path.exists(os.path.join(TW, "data", "packs", f"{r['id']}.{e}")) for e in ("json", "bin"))
            print(r["id"], r["status"], "-> built" if ok else "pack missing, left")
            if relax:
                r.setdefault("gate_relax", {})[relax[0]] = relax[1]
            if ok and r["status"] == "parked":
                r["status"] = "built"
                (r.get("run") or {}).pop("reason", None)
    shared_lock.write_atomic(plan, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
