"""T-3246: what the groups move, off against on. python effects.py off.json on.json [net.json]"""
import json
import sys

a, b = (json.load(open(f)) for f in sys.argv[1:3])
net = json.load(open(sys.argv[3] if len(sys.argv) > 3 else a["net"]))
print("lines keys:", list(a["lines"].keys()), "areas keys:", list(a["areas"].keys()))
ca, cb = a["day"]["commute"], b["day"]["commute"]
ea, eb = a["day"]["everyday"], b["day"]["everyday"]
print(f"commute rail {ca[1]:,.0f} -> {cb[1]:,.0f} ({100 * (cb[1] / ca[1] - 1):+.2f}%), walk {ca[2]:,.0f} -> {cb[2]:,.0f}")
print(f"everyday rail {ea[1]:,.0f} -> {eb[1]:,.0f} ({100 * (eb[1] / ea[1] - 1):+.2f}%), walk {ea[2]:,.0f} -> {eb[2]:,.0f}")
tr = ca[1] + ea[1]
print(f"all rail {tr:,.0f} -> {cb[1] + eb[1]:,.0f} ({100 * ((cb[1] + eb[1]) / tr - 1):+.2f}%)")
for k in a["operators"]:
    x, y = a["operators"][k], b["operators"][k]
    print(f"  {k:10s} {x:12,.0f} -> {y:12,.0f} {100 * (y / x - 1):+.2f}%")
ba, bb = a["lines"]["board"], b["lines"]["board"]
names = [f"{l.get('operator', '?')} {l.get('name', l.get('id', i))}" for i, l in enumerate(net["lines"])]
rows = sorted(((100 * (y / x - 1), names[i] if i < len(names) else str(i), x) for i, (x, y) in enumerate(zip(ba, bb)) if x > 20000), key=lambda r: r[0])
print("lines over 20k boardings, most lost:")
for r in rows[:8]:
    print(f"  {r[1]:40s} {r[2]:10,.0f} {r[0]:+.2f}%")
print("most gained:")
for r in rows[-8:]:
    print(f"  {r[1]:40s} {r[2]:10,.0f} {r[0]:+.2f}%")
