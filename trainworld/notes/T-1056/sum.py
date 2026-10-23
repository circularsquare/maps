"""T-1055/T-1056: calib outputs side by side. python sum.py <dir> <city> <run,..>"""
import json
import sys

D, city, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
BANDS = ["<0.5", "0.5-1", "1-2", "2-5", ">=5 km"]
res = {}
for r in runs:
    j = json.load(open(f"{D}/{city}_{r}.json"))
    res[r] = j
    cm, ev = j["day"]["commute"], j["day"]["everyday"]
    rail, tot, walk = cm[1] + ev[1], cm[0] + ev[0], cm[2] + ev[2]
    print(f"{city} {r}: rail {rail / 1e6:.3f}M/day (commute {cm[1] / 1e6:.3f}M, everyday {ev[1] / 1e6:.3f}M), "
          f"rail of rail+car {100 * rail / (tot - walk):.2f}%, rail of all {100 * rail / tot:.2f}%, walk {100 * walk / tot:.2f}%")
    t = [[0.0] * 12, [0.0] * 12]
    for p in j["periods"]:
        for layer in (0, 1):
            for k, x in enumerate(p.get("t1056", [[0] * 12, [0] * 12])[layer]):
                t[layer][k] += x
    for layer, name in ((0, "commute"), (1, "everyday")):
        a = t[layer]
        if a[0] > 0:
            bands = ", ".join(f"{BANDS[b]} {a[3 + 2 * b] / max(a[2 + 2 * b], 1e-9) * 100:.1f}% of {a[2 + 2 * b] / a[0] * 100:.1f}%" for b in range(5))
            print(f"   {name} (one way): walking further {a[1] / a[0] * 100:.2f}% of rail; by direct distance: {bands}")
if len(runs) > 1:
    a, b = res[runs[0]], res[runs[-1]]
    net = json.load(open(a["net"], encoding="utf-8"))
    names = {}
    for i, d in enumerate(a["st_map"]):
        names.setdefault(d, net["stations"][i].get("name", str(i)))
    lnames = [l.get("name") or l.get("route") for l in net["lines"]]
    ea = [x + y for x, y in zip(a["entries"]["commute"], a["entries"]["everyday"])]
    eb = [x + y for x, y in zip(b["entries"]["commute"], b["entries"]["everyday"])]
    ch = sorted(range(len(ea)), key=lambda i: -abs(eb[i] - ea[i]))[:8]
    print(f"   entries {runs[0]} -> {runs[-1]}: total {sum(ea):,.0f} -> {sum(eb):,.0f}; biggest changes (demand station: before -> after):")
    print("   " + "; ".join(f"{names.get(i, i)}: {ea[i]:,.0f} -> {eb[i]:,.0f}" for i in ch))
    la, lb = a["lines"]["board"], b["lines"]["board"]
    ch = sorted(range(len(la)), key=lambda i: -abs(lb[i] - la[i]))[:6]
    print("   line boardings: " + "; ".join(f"{lnames[i]}: {la[i]:,.0f} -> {lb[i]:,.0f}" for i in ch))
