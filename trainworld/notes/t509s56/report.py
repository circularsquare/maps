"""Stage 5/6 table from score_<tag>.json per city: python report.py <tag> [top]"""
import json
import math
import os
import sys

CALIB = r"C:\Users\anita\projects\maps\trainworld\data\work\calib"
tag = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 4
print("| City | Score | Stations | Operators | Areas | Everyday | Station level | Commute rail (measured) | Everyday rail | Other c/e |")
print("|---|---|---|---|---|---|---|---|---|---|")
rows = {}
for c in ["nyc", "lon", "par", "seo", "tyo", "sdj", "tor"]:
    p = os.path.join(CALIB, c, f"score_{tag}.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    rows[c] = s
    pa = s["parts"]
    d = s["day"]
    cm, ev = d["commute"], d["everyday"]
    print(f"| {c} | {s['total']:.2f} | {pa['stations']:.2f} | {pa['operators']:.2f} | {pa['areas']:.2f} | {pa.get('everyday', float('nan')):.2f} | "
          f"x{math.exp(s['station_level']):.2f} | {cm[1] / cm[0]:.1%} ({s['rail_share_measured']:.1%}; areas {s['rail_share_model']:.1%}) | "
          f"{ev[1] / ev[0]:.1%} | {cm[3] / cm[0]:.1%} / {ev[3] / ev[0]:.1%} |")
fit = [c for c in rows if c != "tor"]
print(f"\nsum of the fitted six: {sum(rows[c]['total'] for c in fit):.3f}")
for c, s in rows.items():
    print(f"\n{c}:")
    st = [r for r in s["station_rows"] if not r["hub"]]
    st.sort(key=lambda r: -abs(math.log(max(r["model"], 1) / r["measured"])) * math.sqrt(r["measured"]))
    print("  stations: " + "; ".join(f"{r['name'][:24]} x{r['model'] / r['measured']:.2f} ({r['measured'] / 1e3:.0f}k)" for r in st[:top]))
    ops = sorted(s["operators"].items(), key=lambda kv: -abs(kv[1]["log_ratio"]))
    if ops:
        print("  operators: " + "; ".join(f"{k} x{math.exp(v['log_ratio']):.2f}" for k, v in ops[:top]))
    ar = s["area_rows"]
    if ar:
        tot = sum(a["commuters"] for a in ar)
        def lg(p):
            p = min(max(p, 0.01), 0.99)
            return math.log(p / (1 - p))
        ar.sort(key=lambda a: -a["commuters"] * (lg(a["model"]) - lg(a["measured"])) ** 2)
        print("  areas: " + "; ".join(f"{a['name'][:16]} {a['model']:.0%} vs {a['measured']:.0%}" for a in ar[:top]))
    for k, v in (s.get("everyday_rows") or {}).items():
        print(f"  everyday {k}: rail {v['rail']:.1%} (survey {v['rail_measured']:.1%}), walk {v['walk']:.1%} (survey {v['walk_measured']:.1%})")
