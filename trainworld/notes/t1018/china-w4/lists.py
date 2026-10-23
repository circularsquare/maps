"""Wave 4 run lists: ids_<run>.txt and wave_ids.txt; prints each run's regions as the runner chains them."""
import json, os, sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
HERE = os.path.dirname(os.path.abspath(__file__))
plan = json.load(open(r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json", encoding="utf-8"))["metros"]
runs = {"a": ["liaoning"], "b": ["jilin", "heilongjiang"], "c": ["gansu", "qinghai", "tibet"],
        "d": ["innermongolia", "ningxia"], "e": ["xinjiang"]}
MOVE = {"lsh": "b"}  # Lishu (Siping, Jilin; solve region liaoning) builds from Jilin's extract: run B's
allids = []
for k, regs in runs.items():
    rs = sorted((r for r in plan if r["status"] == "planned" and r.get("solve_region") and (MOVE.get(r["id"]) == k or
                 (r["id"] not in MOVE and r["solve_region"] in regs))), key=lambda r: r["order"])
    ids = [r["id"] for r in rs]
    allids += ids
    open(os.path.join(HERE, f"ids_{k}.txt"), "w").write(",".join(ids))
    ex = {}
    for r in rs:
        ex.setdefault(r["solve_region"], set()).update(r["extracts"])
    print(k, len(ids), {g: sorted(v) for g, v in ex.items()})
open(os.path.join(HERE, "wave_ids.txt"), "w").write(",".join(allids))
print("all", len(allids))
