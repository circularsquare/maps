"""Each run's waits in a line: summ.py name..."""
import json, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/out/"
keys = {}
for n in sys.argv[1:]:
    d = json.load(open(OUT + n + ".json", encoding="utf-8"))
    waits = [s for s in d["steps"] if s["op"] == "wait"]
    k = lambda f: (f["rail"], f["trips"], f["icRail"], f["stH"], json.dumps(f["lines"], sort_keys=True))
    sig = [k(s["fig"]) for s in waits]
    keys[n] = sig
    print(n, "load", d.get("load"), "error", (d.get("error") or "-")[:60], "hang", bool(d.get("hang")), "errorsN", d.get("errorsN"),
          "fig0", d["fig0"]["rail"], "| " + "  ".join(f"{s['settle']:.0f}s {s['fig']['rail']:.3f}" for s in waits))
names = list(keys)
ref = keys[names[0]]
for n in names[1:]:
    same = [a == b for a, b in zip(ref, keys[n])]
    print(n, "vs", names[0], "same at each wait:", same)
