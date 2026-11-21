"""Print a hang.mjs run's worker lines per wait: logs.py name [regex]"""
import json, re, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/ca3a5e8b-220b-4b3d-a5e9-87f500e687dc/scratchpad/crowdfast/out/"
r = json.load(open(OUT + sys.argv[1] + ".json", encoding="utf-8"))
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else re.compile(r"T3513tmp|T2729|T3512|T1733|T3599")
print("load", r.get("load"), "error", r.get("error"))
for i, s in enumerate(x for x in r["steps"] if x["op"] == "wait"):
    print(f"--- wait {i}: {s['settle']:.1f} s rail {s['fig']['rail']}")
    for t, x in s["log"]:
        if pat.search(x):
            print(f"{t/1000:7.1f} {x[:260]}")
