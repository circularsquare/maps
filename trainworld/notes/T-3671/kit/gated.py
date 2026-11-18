"""Exit 0 when every listed plan row is gated (built, gate passed, registered, waiting for its join); else print each
row's status and parked reason and exit 1. usage: gated.py a,b,c"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
PLAN = r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json"
ids = sys.argv[1].split(",")
rows = {r["id"]: r for r in json.load(open(PLAN, encoding="utf-8"))["metros"] if r.get("units")}
bad = 0
for i in ids:
    r = rows.get(i)
    st = r["status"] if r else "missing"
    why = (r.get("run") or {}).get("reason", "") if r else ""
    print(f"{i}: {st}{'; ' + why if st == 'parked' else ''}")
    bad += st != "gated"
sys.exit(1 if bad else 0)
