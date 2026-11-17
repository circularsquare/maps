"""Compare the figures at each wait of hang.mjs runs: cmp.py a.json b.json [grep]"""
import json, re, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/out/"
def load(n):
    return json.load(open(OUT + n + ".json", encoding="utf-8"))
def key(f):
    return (f["rail"], f["trips"], f["icRail"], f["stH"], json.dumps(f["lines"], sort_keys=True))
runs = [load(n) for n in sys.argv[1:3]]
pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
for r in runs:
    print(r["name"], "load", r.get("load"), "error", (r.get("error") or "")[:120], "hang", bool(r.get("hang")), "errorsN", r.get("errorsN"))
print("fig0 same:", key(runs[0]["fig0"]) == key(runs[1]["fig0"]), runs[0]["fig0"]["rail"], runs[1]["fig0"]["rail"])
w = [[s for s in r["steps"] if s["op"] == "wait"] for r in runs]
for i, (a, b) in enumerate(zip(*w)):
    fa, fb = a["fig"], b["fig"]
    same = key(fa) == key(fb)
    print(f"wait {i}: {a['settle']:.1f} s vs {b['settle']:.1f} s  rail {fa['rail']!r} vs {fb['rail']!r}  same={same}")
    if not same:
        for k in ("rail", "trips", "icRail", "stH"):
            if fa[k] != fb[k]:
                print("   ", k, fa[k], fb[k])
        diff = [l for l in fa["lines"] if fa["lines"][l] != fb["lines"].get(l)]
        print("    lines differing", len(diff), diff[:5])
    if pat:
        for r, s in ((runs[0]["name"], a), (runs[1]["name"], b)):
            for t, x in s["log"]:
                if pat.search(x) and not x.startswith("W "):
                    print("     ", r, t, x[:200])
