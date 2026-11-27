"""Exit 0 when every listed plan row is shipped after its join and every listed hand pack (no plan row) is registered;
else print each one's state and exit 1. usage: shipped.py a,b,c"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
ids = sys.argv[1].split(",")
rows = {r["id"]: r for r in json.load(open(TW + r"\pipeline\world_plan.json", encoding="utf-8"))["metros"] if r.get("units")}
reg = {c["id"] for c in json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["cities"]}
bad = 0
for i in ids:
    if i in rows:
        st = rows[i]["status"]
        ok = st == "shipped" and i in reg
        why = (rows[i].get("run") or {}).get("reason", "") if st == "parked" else ""
        print(f"{i}: {st}{', registered' if i in reg else ', NOT registered'}{'; ' + why if why else ''}")
    else:
        ok = i in reg
        print(f"{i}: hand pack, {'registered' if ok else 'NOT registered'}")
    bad += not ok
sys.exit(1 if bad else 0)
