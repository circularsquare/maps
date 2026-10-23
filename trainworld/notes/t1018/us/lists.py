"""US run lists: lists.py a=13,12 b=48 ...   (run name = state FIPS codes of the rows' lead county)
Writes ids_<run>.txt and wave_ids.txt (planned US rows only, plan order); prints each run's extracts and the
rows whose extracts reach past the run's own states (they chain runner regions: check two runs never share an
extract)."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
plan = json.load(open(r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json", encoding="utf-8"))["metros"]
runs = {a.split("=")[0]: set(a.split("=")[1].split(",")) for a in sys.argv[1:]}
allids, seen = [], {}
for k, sts in runs.items():
    rs = sorted((r for r in plan if r.get("iso2") == "US" and r["status"] == "planned" and r.get("units")
                 and r["units"]["USA"][0][:2] in sts), key=lambda r: r["order"])
    ids = [r["id"] for r in rs]
    allids += ids
    open(os.path.join(HERE, f"ids_{k}.txt"), "w").write(",".join(ids))
    ex = sorted({e for r in rs for e in r["extracts"]})
    for e in ex:
        if e in seen:
            print(f"  WARNING: {e} in runs {seen[e]} and {k}")
        seen.setdefault(e, k)
    print(k, len(ids), "rows,", sum(r.get("kind") != "zone" for r in rs), "metros; extracts", [e.split("/")[-1] for e in ex])
    for r in rs:
        other = sorted({u[:2] for u in r["units"]["USA"]} - sts)
        if other or len(r["extracts"]) > 1:
            print("   ", r["id"], r["name"], "counties in states", other, "extracts", [e.split("/")[-1] for e in r["extracts"]])
open(os.path.join(HERE, "wave_ids.txt"), "w").write(",".join(allids))
print("all", len(allids))
