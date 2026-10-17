"""Print fit621's table for points given in a JSON file (cached runs only; missing cities skipped).

    python show621.py <points.json> [cities]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit621 as F  # noqa: E402

pts = json.load(open(sys.argv[1]))
cities = sys.argv[2].split(",") if len(sys.argv) > 2 else F.ALL
cache = F.load_cache()
for a in pts:
    p = dict(F.START)
    p.update(a)
    print({k: v for k, v in a.items() if "@CA" in k or "@KR" in k} or a)
    F.table({c: cache[F.key(c, p)] for c in cities if F.key(c, p) in cache})
