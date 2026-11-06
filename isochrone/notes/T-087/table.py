"""T-087: print compare.mjs output as a table.
    python isochrone/notes/T-087/table.py out.json [rule,rule,...] [metric]
metric: cell (default) = "rel% bias% ratio", rel, bias, ratio
rel = median |100 m - 1 km| / 1 km; bias = median ln(100 m / 1 km) in % (negative: the 1 km map is slower);
ratio = summed 100 m time / summed 1 km time (below 1: the 1 km map is too slow).
"""
import json
import sys

data = json.load(open(sys.argv[1]))
rules = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(data[0]["rules"])
metric = sys.argv[3] if len(sys.argv) > 3 else "cell"
w = max(len(r) for r in rules) + 2
print("%-10s" % "tile" + "".join("%-*s" % (max(w, 16), r) for r in rules))


def cell(v):
    if metric == "rel":
        return "%.1f%%" % (100 * v["rel"])
    if metric == "bias":
        return "%+.1f%%" % (100 * v["bias"])
    if metric == "ratio":
        return "%.3f" % v["ratio"]
    return "%2.0f%% %+3.0f%% %.2f" % (100 * v["rel"], 100 * v["bias"], v["ratio"])


means = {r: [0, 0, 0, 0] for r in rules}
for t in data:
    print("%-10s" % t["name"] + "".join("%-*s" % (max(w, 16), cell(t["rules"][r])) for r in rules))
    for r in rules:
        v = t["rules"][r]
        m = means[r]
        m[0] += v["rel"]; m[1] += v["bias"]; m[2] += v["ratio"]; m[3] += 1
print("%-10s" % "mean" + "".join("%-*s" % (max(w, 16), cell({"rel": means[r][0] / means[r][3], "bias": means[r][1] / means[r][3],
                                                             "ratio": means[r][2] / means[r][3]})) for r in rules))
