"""List the gaps between consecutive meetings whose stray is over 100 m, to pick the threshold."""
import sys, collections
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import relate

relate.STRAY_RATIO = 1.0  # keep everything together so every gap is measured once
d, cx, routes = relate.load(sys.argv[1])
rows = []
seen = set()
for li in range(len(d["lines"])):
    L = routes[li]
    for r in relate.classify(d, cx, routes, li):
        key = tuple(sorted((li, r["o"])))
        for m in r["meet"]:
            if "gap" in m:
                st, run, lstep = m["gap"]
                rows.append((st / run, st, run, lstep, d["lines"][li]["name"], d["lines"][r["o"]]["name"], d["nodes"][m.get("Onode", 0)].get("name", "")[:24], key in seen))
        seen.add(key)
rows.sort()
hist = collections.Counter()
for q, st, run, lstep, a, b, at, dup in rows:
    hist[min(10, int(q / 0.025))] += 1
print("ratio histogram (0.025 bins):", [hist[k] for k in range(11)])
for q, st, run, lstep, a, b, at, dup in rows:
    if dup or st < 100 or q < 0.02 or q > 0.4:
        continue
    print(f"{q:5.3f} stray {st:6.0f} run {run:6.0f} Lstep {lstep:6.0f}  {a:<24} {b:<24} to {at}")
