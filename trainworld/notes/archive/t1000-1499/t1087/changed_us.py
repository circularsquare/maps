"""Which plan rows a US variant changes, by status, with names and solve regions."""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
R = pickle.load(open(os.path.join(HERE, "us_out.pkl"), "rb"))
doc = json.load(open(r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json", encoding="utf-8"))
rows = {r["id"]: r for r in doc["metros"] if "USA" in (r.get("units") or {})}
V = R[sys.argv[1]]
sets = {frozenset(v) for v in V["groups"].values()}
for st in ("shipped", "built", "planned"):
    ch = [rid for rid, ids in R["before"]["groups"].items() if rows[rid]["status"] == st and frozenset(ids) not in sets]
    regs = sorted({rows[r].get("solve_region") for r in ch})
    print(f"{st}: {len(ch)} changed: " + ", ".join(f"{r} {rows[r]['name']}" for r in ch) + f"; solve regions {regs}")
