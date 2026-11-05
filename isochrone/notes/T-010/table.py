"""T-010: tables from compare.json.   python isochrone/notes/T-010/table.py <compare.json> [rule ...]"""
import json
import sys

raw = open(sys.argv[1], "rb").read()
d = json.loads(raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8"))   # PowerShell's > writes UTF-16
rules = sys.argv[2:] or list(d[0]["rules"])
for part in ("all", "other", "urban"):
    print("\n%s cells: median |time difference| / band share / 100 m over 1 km time" % part)
    print("%-14s" % "tile" + "".join("%-18s" % r for r in rules))
    tot = {r: [] for r in rules}
    for t in d:
        row = "%-14s" % t["label"]
        for r in rules:
            v = t["rules"][r][part]
            if v["rel"] is None:
                row += "%-18s" % "-"
                continue
            row += "%-18s" % ("%3.0f%% %.2f %.2f" % (100 * v["rel"], v["band"], v["ratio"]))
            tot[r].append(v)
        print(row)
    row = "%-14s" % "mean"
    for r in rules:
        vs = tot[r]
        if vs:
            row += "%-18s" % ("%3.0f%% %.2f %.2f" % (100 * sum(v["rel"] for v in vs) / len(vs), sum(v["band"] for v in vs) / len(vs), sum(v["ratio"] for v in vs) / len(vs)))
    print(row)
