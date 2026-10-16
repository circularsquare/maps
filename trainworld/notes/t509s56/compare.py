"""Before/after table from the cache: python compare.py [after.json]"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit

cache = fit.load_cache()
before = fit.today()
after = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(fit.HERE, "best.json")))
print("| City | Score | Stations | Operators | Areas | Everyday (NHTS) | Station level | Commute rail (census) | Everyday rail |")
print("|---|---|---|---|---|---|---|---|---|")
sums = [0.0, 0.0]
for c in fit.ALL:
    a, b = cache[fit.key(c, before)], cache[fit.key(c, after)]
    if c in fit.FIT:
        sums[0] += a["total"]
        sums[1] += b["total"]
    f = lambda k: f"{a['parts'].get(k, 0):.2f} -> {b['parts'].get(k, 0):.2f}"
    cr = lambda r: r["commute"][1] / r["commute"][0]
    er = lambda r: r["everyday"][1] / r["everyday"][0]
    meas = f" ({a['rail_share_measured']:.1%})" if a["rail_share_measured"] > 0 else ""
    print(f"| {c} | {a['total']:.2f} -> {b['total']:.2f} | {f('stations')} | {f('operators')} | {f('areas')} | {f('everyday')} | "
          f"x{math.exp(a['station_level']):.2f} -> x{math.exp(b['station_level']):.2f} | {cr(a):.1%} -> {cr(b):.1%}{meas} | {er(a):.1%} -> {er(b):.1%} |")
print(f"\nfitted six: {sums[0]:.3f} -> {sums[1]:.3f}; objective with penalty {fit.objective({c: cache[fit.key(c, before)] for c in fit.FIT}, before):.3f} -> "
      f"{fit.objective({c: cache[fit.key(c, after)] for c in fit.FIT}, after):.3f}")
for c in ["nyc"]:
    for tag, p in (("before", before), ("after", after)):
        r = cache[fit.key(c, p)]
        print(tag, {k: (round(v["rail"], 3), round(v["walk"], 3)) for k, v in (r.get("ev_rows") or {}).items()})
        print(tag, "operators", {k: round(math.exp(v), 2) for k, v in r["operators"].items()})
