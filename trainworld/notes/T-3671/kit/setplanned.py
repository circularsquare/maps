"""Set shipped (or parked) rows back to planned under the plan lock, so the next run rebuilds them on today's code
(T-3455: osm_units.reclaimed adds the coastal land their old outlines miss). usage: setplanned.py a,b,c <task> [--go]"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import shared_lock

PLAN = r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json"
ids = sys.argv[1].split(",")
task = sys.argv[2]
go = "--go" in sys.argv


def apply(doc):
    n = 0
    for r in doc["metros"]:
        if r.get("units") and r["id"] in ids:
            ok = r["status"] in ("shipped", "parked")
            print(r["id"], r["name"], r["status"], "-> planned" if ok else "left")
            if ok:
                r["status"] = "planned"
                r.setdefault("source", {})["rebuilt_for"] = task
                n += 1
    return n


if not go:
    apply(shared_lock.read_json(PLAN))
else:
    with shared_lock.held("plan", f"{task} setplanned.py"):
        doc = shared_lock.read_json(PLAN)
        if apply(doc):
            shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
            print("written")
