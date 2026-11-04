"""compare two measure.mjs JSONs: rail, per zone, stations, lines (final and refined)"""
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for key in ("final", "refined"):
    A, B = a.get(key), b.get(key)
    if not A or not B:
        continue
    print(f"== {key}: rail {A['rail']:.2f} -> {B['rail']:.2f} ({100*(B['rail']/A['rail']-1):+.4f}%), active {A['active']} {B['active']}")
    print("   memory", [(m['wasm'], m['heap'], m['peakNow']) for m in A['memory']], "->", [(m['wasm'], m['heap'], m['peakNow']) for m in B['memory']])
    ca, cb = A['cities'], B['cities']
    diff = [(cb[k]['rail'] - ca[k]['rail'], k, ca[k]['rail']) for k in ca if k in cb and cb[k]['rail'] != ca[k]['rail']]
    print(f"   zones differing in rail: {len(diff)} of {len(ca)}")
    for d, k, r in sorted(diff, key=lambda x: -abs(x[0]))[:10]:
        print(f"     {k}: {r:.1f} {d:+.1f} ({100*d/max(r,1e-9):+.3f}%)")
    act = A['active']
    if act in ca and act in cb:
        print(f"   in view {act}: rail {ca[act]['rail']:.1f} -> {cb[act]['rail']:.1f}, trips {ca[act]['trips']:.1f} -> {cb[act]['trips']:.1f}")
    sa, sb = A['stations'], B['stations']
    worst = sorted(((abs(sb[k] - sa[k]) / max(sa[k], 1), k, sa[k], sb[k]) for k in sa if k in sb and sa[k] > 1000), reverse=True)[:5]
    print("   stations (1000+) worst:", [(k, round(x, 1), round(y, 1), f"{100*r:.2f}%") for r, k, x, y in worst])
    la, lb = A['lines'], B['lines']
    worst = sorted(((abs(lb[k] - la[k]) / max(la[k], 1), k, la[k], lb[k]) for k in la if k in lb and la[k] > 1000), reverse=True)[:5]
    print("   lines (1000+) worst:", [(k, round(x, 1), round(y, 1), f"{100*r:.2f}%") for r, k, x, y in worst])
