"""T-3240 scenario: the ground curve's destinations weighted by (ta/a)^theta (the tail's density pull,
the mean of u^0.75 over a place's people), each origin's ground departures kept (renormalised over
its own country's places, abroad ignored). Far tail unchanged. From the rows only: a rough look.
    python pull.py theta
"""
import sys, collections
sys.path.insert(0, r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist")
from bands import rows, rule, km, ground, tail, byp  # noqa
from pairs import each_way  # noqa

th = float(sys.argv[1])
w = {z: (r["ta"] / r["a"]) ** th if r["a"] > 0 else 1. for z, r in rows.items()}
by_c = collections.defaultdict(list)
for z, r in rows.items():
    by_c[r["country"]].append(z)


def ground_each(a, b):
    p, q = rows[a], rows[b]
    d = km(p, q)
    if a == b:
        return dict(p.get("near", [])).get(p["place"], 0.)
    if d < rule["near_km"]:
        return each_way(a, b)[1]
    return 0.5 * (p["g"] * ground(d) * q["a"] + q["g"] * ground(d) * p["a"])


W = {}
for c, zs in by_c.items():
    for a in zs:
        num = den = 0.
        for b in zs:
            t = ground_each(a, b)
            if t:
                num += t * w[b]; den += t
        W[a] = num / den if den else 1.

PAIRS = "war-kra war-tro war-gda war-poz war-msw war-kat war-lod kra-kat war-skw nyc-phl nyc-bos nyc-wdc lon-man lon-bir ber-ham fra-col jnj-nag jnj-liu jnj-yzo liu-nag".split()
for s in PAIRS:
    a, b = s.split("-")
    if a not in rows or b not in rows:
        print(s, "missing"); continue
    d, t = each_way(a, b)
    g = ground_each(a, b)
    f = 0.5 * (w[b] / W[a] + w[a] / W[b])
    new = t - g + g * f
    print(f"{s:8s} {d:5.0f} km  each way {t:8.0f} -> {new:8.0f} (x{new/t:.2f})   ta/a {rows[a]['ta']/rows[a]['a']:.2f} {rows[b]['ta']/rows[b]['a']:.2f}")
