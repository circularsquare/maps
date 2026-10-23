"""Shown names that repeat: every registered pack's shown name (zone-names.json, else the registry's) and every
planned row's name, against a names file: repeats.py names.json"""
import collections, json, sys
sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
reg = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["cities"]
zn = json.load(open(TW + r"\app\src\game\zone-names.json", encoding="utf-8"))
plan = json.load(open(TW + r"\pipeline\world_plan.json", encoding="utf-8"))["metros"]
add = json.load(open(sys.argv[1], encoding="utf-8"))
shown = collections.defaultdict(list)
for c in reg:
    shown[((add.get(c["id"]) or {}).get("name") or (zn.get(c["id"]) or {}).get("name") or c["name"]).split(",")[0]].append(c["id"])
have = {c["id"] for c in reg}
for r in plan:
    if r.get("id") and r["id"] not in have and r.get("status") == "planned":
        shown[r["name"].split(",")[0]].append(r["id"] + "(planned)")
for i, v in add.items():
    n = v["name"].split(",")[0]
    print(i, v["name"], "->", shown[n])
