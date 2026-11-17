"""Seconds to each final by the op before it, per run: tbl.py name..."""
import json, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/out/"
for n in sys.argv[1:]:
    d = json.load(open(OUT + n + ".json", encoding="utf-8"))
    row, last = [], None
    for s in d["steps"]:
        if s["op"] == "wait":
            w = s.get("work") or {}
            local = w.get("localShownAt"); arr = w.get("arrivedAt")
            loc = f" (local {round((local - arr) / 1000)})" if local and arr else ""
            row.append(f"{last} {s['settle']:.1f}{loc}")
        else:
            op = s["op"]
            last = op.get("k", op["t"])
    print(n, "load", d.get("load"), "|", " | ".join(row), "| error", (d.get("error") or "-")[:50])
