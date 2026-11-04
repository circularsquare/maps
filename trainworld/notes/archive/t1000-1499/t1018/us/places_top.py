"""Each row's biggest places (from data/packs/<id>.places.json) and its county count: places_top.py id,id"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
plan = {r["id"]: r for r in json.load(open(TW + r"\pipeline\world_plan.json", encoding="utf-8"))["metros"] if r.get("id")}
for i in sys.argv[1].split(","):
    p = json.load(open(TW + rf"\data\packs\{i}.places.json", encoding="utf-8"))["places"]
    top = sorted(p, key=lambda x: -(x[4] or 0))[:6]
    r = plan[i]
    print(i, r.get("kind") or "metro", r["name"], len(r["units"]["USA"]), "|", ", ".join(f"{x[0]} {x[4]:,}" for x in top))
