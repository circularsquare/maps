"""Build step times per row of a solve region, from the plan's run records: buildtimes.py [region]"""
import json, statistics as st, sys
sys.stdout.reconfigure(encoding="utf-8")
plan = json.load(open(r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json", encoding="utf-8"))["metros"]
reg = sys.argv[1] if len(sys.argv) > 1 else "south-atlantic"
out = []
for r in plan:
    run = r.get("run") or {}
    if r.get("solve_region") == reg and isinstance(run, dict) and run.get("steps"):
        s = run["steps"]
        out.append((r.get("kind") or "metro", r["id"], round(r["people"] / 1e6, 2), round(sum(s.values())),
                    {k: round(v) for k, v in s.items()}, run.get("run_batch"), r["status"]))
for x in sorted(out):
    print(*x)
for k in ("zone", "metro"):
    t = [x[3] for x in out if x[0] == k]
    if t:
        print(k, len(t), "median", st.median(t), "s, max", max(t))
