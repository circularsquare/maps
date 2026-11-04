"""Before/after by country: summary.py <m_before.json> <m_after.json> <hdr_before.json> <hdr_after.json>
(matstats outputs: local layer's trips, mean km, over 50 km, home zone; stats.py outputs: header
commuters_out and long_out)."""
import json, sys, collections
mb, ma, hb, ha = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:5])
by = collections.defaultdict(list)
for i in ma:
    if i in mb and "error" not in ma[i] and "error" not in mb[i]:
        by[hb[i]["cc"]].append(i)


def agg(ids, m):
    t = sum(m[i]["trips"] for i in ids)
    return t, sum(m[i]["mean"] * m[i]["trips"] for i in ids) / t, sum(m[i]["over50"] * m[i]["trips"] for i in ids) / t, \
        sum(m[i]["home"] * m[i]["trips"] for i in ids) / t


print("country packs | local commutes before > after | mean km | over 50 km | home zone | between zones (+ add-back) before > after")
for c, ids in sorted(by.items(), key=lambda kv: -len(kv[1])):
    t0, m0, o0, h0 = agg(ids, mb)
    t1, m1, o1, h1 = agg(ids, ma)
    x0 = sum(hb[i].get("out") or 0 for i in ids)
    x1 = sum(ha[i].get("out") or 0 for i in ids)
    l1 = sum(ha[i].get("long_out") or 0 for i in ids)
    print(f"{c} {len(ids)} | {t0:,.0f} > {t1:,.0f} | {m0:.2f} > {m1:.2f} | {o0:.2%} > {o1:.2%} | {h0:.1%} > {h1:.1%} | "
          f"{x0:,.0f} > {x1:,.0f} (+{l1:,.0f})")
print("\nzones whose figures moved a lot (mean down by over a third or 5 km, or up by over 25%):")
rows = []
for c, ids in by.items():
    for i in ids:
        a, b = mb[i], ma[i]
        if (a["mean"] - b["mean"] > max(5, a["mean"] / 3)) or b["mean"] > a["mean"] * 1.25:
            rows.append((c, i, a["mean"], b["mean"], a["over50"], b["over50"], a["home"], b["home"]))
for r in sorted(rows, key=lambda r: r[2] - r[3], reverse=True):
    print(f"  {r[0]} {r[1]}: mean {r[2]:.1f} > {r[3]:.1f} km, over 50 km {r[4]:.1%} > {r[5]:.1%}, home {r[6]:.1%} > {r[7]:.1%}")
