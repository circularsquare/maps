"""T-1140: replay the plan's US merge log (waves 3 and 4, T-1088's replan) and score every join of
100,000 people or more under the new test: the joiner's partner ratio (ACS, both ways) and the joined
zone's hull ratio against the bigger part's. Read-only.

    python replay.py [names...]
"""
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402

D = C.load("us_data.pkl")
P = C.load("us_prep.pkl")
d = D["USA"]
U = d["units"]
F = P["USA"]["acs"]
S = C.Shapes(U)
byname = {u.name: i for i, u in U.items()}
rec = d["record"]
kept = set(rec["kept"])
zrows = [r for r in d["rows"] if r.get("kind") == "zone"]
owner = {}
for r in zrows:
    if r["id"] in kept:
        for i in r["units"]["USA"]:
            owner[i] = ("row", r["id"])
for f in d["fixed"]:
    for i in f["units"]:
        owner.setdefault(i, ("f", f["id"]))
names = {("row", r["id"]): r["name"] for r in zrows} | {("f", f["id"]): f["name"] for f in d["fixed"]}
groups = {}
for r in zrows:
    if r["id"] not in kept:
        for i in r["units"]["USA"]:
            groups[i] = [i]
            owner[i] = ("g", i)
pat = re.compile(r"^(.+?) \(([\d,]+)\) joins (.+?) \((.+)\)$")
want = sys.argv[1:]
rows = []
for ln in rec["merges"]:
    m = pat.match(ln)
    if not m:
        continue
    a, n, b, how = m.group(1), int(m.group(2).replace(",", "")), m.group(3), m.group(4)
    ia, ib = byname.get(a), byname.get(b)
    if ia not in groups or ib not in groups:
        continue
    ga, gb = groups[ia], groups[ib]
    if n >= 100_000:
        to = C.area_flows(F, ga, owner)
        top = to.most_common(1)
        r = to.get(("g", ib), 0.0) / top[0][1] if top else 0.0
        topn = (U[top[0][0][1]].name if top and top[0][0][0] == "g" else names.get(top[0][0], "?")) if top else "-"
        sa, sb = S.zone(ga), S.zone(gb)
        hu = C.hull_ratio(S.zone(ga + gb))
        big = max((sa, sb), key=lambda g: g.area)
        hb = C.hull_ratio(big)
        need = min(0.75, hb - 0.05)
        verdict = "partner" if r >= 0.5 else ("compact" if hu >= need else "REFUSED")
        rows.append((a, n, b, sum(U[i].free for i in gb), how, r, topn, hu, hb, verdict))
    groups[ib] = gb + ga
    for i in ga:
        owner[i] = ("g", ib)
    del groups[ia]
for a, n, b, nb, how, r, topn, hu, hb, v in rows:
    if want and not any(w in a or w in b for w in want):
        continue
    print(f"{a[:28]:28} {n / 1e6:4.2f}M -> {b[:28]:28} {nb / 1e6:4.2f}M ({how[:24]:24}) partner {r:4.2f} (top {topn[:24]}) "
          f"hull {hu:.2f} (bigger part {hb:.2f}) {v}")
print(len(rows), "joins of 100,000 or more;", sum(x[-1] == "REFUSED" for x in rows), "refused,",
      sum(x[-1] == "partner" for x in rows), "by partner,", sum(x[-1] == "compact" for x in rows), "by shape")
