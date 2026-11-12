"""Where a place's longer trips go: within itself, its country by band, abroad, and to no place (dead space).
    python split.py war ber ...
"""
import sys, collections
sys.path.insert(0, r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist")
from bands import rows, rule, km, directed  # noqa
from pairs import border  # noqa

for a in sys.argv[1:]:
    p = rows[a]
    near = dict(p.get("near", []))
    out = collections.Counter()
    for b, q in rows.items():
        d = km(p, q)
        if b == a:
            out["within"] += near.get(p["place"], 0.)
            continue
        if d < rule["near_km"]:
            t = near.get(q["place"]) or dict(q.get("near", [])).get(p["place"], 0.)
        else:
            t = border(p, q) * directed(p, q, d)
        if q["country"] != p["country"]:
            out["abroad"] += t
        else:
            out["home <100" if d < 100 else "home 100-250" if d < 250 else "home 250-500" if d < 500 else "home 500+"] += t
    tot = sum(out.values())
    print(f"{a}: departures {p['dep']:.0f}, to places {tot:.0f}, to no place {p['dep']-tot:.0f}  | " + ", ".join(f"{k} {v:.0f} ({v/p['dep']:.0%})" for k, v in sorted(out.items())))
