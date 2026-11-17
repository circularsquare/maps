"""A hang.mjs run's hang dump, long values cut, and the page log after the last op: showdump.py name"""
import json, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/out/"
d = json.load(open(OUT + sys.argv[1] + ".json", encoding="utf-8"))
h = d["hang"]
def short(v):
    s = json.dumps(v)
    return s if len(s) <= 100 else s[:100] + "..."
dump = h["dump"]
for k in ("cur", "view", "client"):
    o = dump.get(k) or {}
    print("==", k)
    if isinstance(o, dict):
        print("   ", ", ".join(f"{a}={short(b)}" for a, b in o.items() if a not in ("results", "lastKey", "layout")))
for i, e in enumerate(dump.get("log") or []):
    print("== log", i, ", ".join(f"{a}={short(b)}" for a, b in e.items() if a not in ("results", "layout")))
print("--- page log after the op (W lines dropped)")
n = 0
for t, x in h["log"]:
    if not x.startswith("W "):
        print("   ", t, x[:220]); n += 1
        if n > 80: break
print("worker lines:", sum(1 for t, x in h["log"] if x.startswith("W ")))
for t, x in [l for l in h["log"] if l[1].startswith("W ")][:40]:
    print("   ", t, x[:220])
