"""Wave status: counts by status per solve region, claims, parked reasons. wstat.py [regions]"""
import collections, json, sys
sys.stdout.reconfigure(encoding="utf-8")
plan = json.load(open(r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json", encoding="utf-8"))["metros"]
want = set((sys.argv[1] if len(sys.argv) > 1 else "south-atlantic,east-south-central,west-south-central,"
            "east-north-central,west-north-central,mountain,pacific,northeast").split(","))
rows = [r for r in plan if r.get("solve_region") in want]
for reg in sorted(want):
    rs = [r for r in rows if r["solve_region"] == reg]
    print(reg, len(rs), dict(collections.Counter(r["status"] for r in rs)))
print("all", len(rows), dict(collections.Counter(r["status"] for r in rows)))
cl = collections.Counter((r.get("run") or {}).get("claim") for r in rows if (r.get("run") or {}).get("claim"))
print("claims", dict(cl))
for r in rows:
    if r["status"] == "parked":
        print("PARKED", r["id"], r["name"], str((r.get("run") or {}).get("reason"))[:300].replace("\n", " "))
