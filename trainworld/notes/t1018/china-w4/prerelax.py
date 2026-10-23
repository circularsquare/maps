"""Relax pop_vs_fua on planned metro rows: prerelax.py regions [ids]
regions: solve regions (comma list) whose planned metro rows are relaxed; ids: extra rows (comma list,
Taiwan's metros, built by the same planner rule: whole townships round a GHSL urban area)."""
import json, os, sys
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402
want = set(sys.argv[1].split(","))
extra = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
REASON = ("T-897: a Chinese metro is whole counties and districts round a small GHSL urban area (FUA), "
          "so it holds well over the FUA's people; pop_vs_plan still guards it")
REASON_TW = ("T-897: a Taiwanese metro is whole townships round a GHSL urban area (FUA), as China's are, "
             "so it can hold well over the FUA's people; pop_vs_plan still guards it")
plan = os.path.join(TW, "pipeline", "world_plan.json")
with shared_lock.held("plan", "t897 prerelax"):
    doc = shared_lock.read_json(plan)
    n = 0
    for r in doc["metros"]:
        hit = (r.get("solve_region") in want) or (r.get("id") in extra)
        if hit and r.get("kind") != "zone" and r.get("units") and r["status"] == "planned":
            r.setdefault("gate_relax", {})["pop_vs_fua"] = REASON_TW if r.get("iso2") == "TW" else REASON
            n += 1
            print(r["id"], end=" ")
    shared_lock.write_atomic(plan, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
print()
print(n, "metro rows relaxed")
