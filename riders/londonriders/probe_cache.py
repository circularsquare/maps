"""Diagnostic: per-TfL-line, count cache files and routes returned.

If a line has 0 routes across all its timetable files, we know the API simply
isn't returning schedule data for that line via /Line/{id}/Timetable/{stop}.
We'd need a different endpoint (e.g. arrivals/predictions) for those.
"""
import json
import os
from collections import defaultdict

CACHE_DIR = "data/api_cache"


def main():
    by_line = defaultdict(lambda: {
        "files": 0, "routes": 0, "scheds": 0, "kj": 0,
        "errs": 0, "names": defaultdict(int),
    })

    for fname in sorted(os.listdir(CACHE_DIR)):
        if not (fname.startswith("tt_") and fname.endswith(".json")):
            continue
        rest = fname[3:-5]
        if "__" not in rest:
            continue
        line_id = rest.split("__", 1)[0]
        rec = by_line[line_id]
        rec["files"] += 1
        try:
            with open(f"{CACHE_DIR}/{fname}", encoding="utf-8") as f:
                tt = json.load(f)
        except (json.JSONDecodeError, OSError):
            rec["errs"] += 1
            continue
        if "error" in tt:
            rec["errs"] += 1
            continue
        for route in (tt.get("timetable", {}) or {}).get("routes", []) or []:
            rec["routes"] += 1
            for s in route.get("schedules", []) or []:
                rec["scheds"] += 1
                rec["names"][s.get("name") or "<none>"] += 1
                rec["kj"] += len(s.get("knownJourneys", []) or [])

    print(f"{'line':22s} {'files':>5} {'routes':>7} {'scheds':>7} {'kj':>6} {'errs':>5}  schedule names")
    for line_id in sorted(by_line.keys()):
        r = by_line[line_id]
        names = ", ".join(f"{n!r}({c})" for n, c in sorted(r["names"].items(), key=lambda kv: -kv[1]))
        print(f"{line_id:22s} {r['files']:>5} {r['routes']:>7} {r['scheds']:>7} {r['kj']:>6} {r['errs']:>5}  {names}")


if __name__ == "__main__":
    main()
