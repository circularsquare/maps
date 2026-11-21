"""T-4008: two calib outputs the same apart from timings (and the paths and settings echoed). python same.py a.json b.json"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d in (a, b):
    for k in ("ms", "net", "pack", "set"):
        d.pop(k, None)


def diff(x, y, at=""):
    if type(x) != type(y):
        return [at]
    if isinstance(x, dict):
        out = []
        for k in set(x) | set(y):
            out += diff(x.get(k), y.get(k), f"{at}.{k}")
        return out
    if isinstance(x, list):
        if len(x) != len(y):
            return [at + f" (len {len(x)} vs {len(y)})"]
        out = []
        for i, (p, q) in enumerate(zip(x, y)):
            out += diff(p, q, f"{at}[{i}]")
        return out
    return [] if x == y else [at]


d = diff(a, b)
print("identical" if not d else f"{len(d)} differences, first: {d[:5]}")
