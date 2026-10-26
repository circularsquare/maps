"""T-1000: python t1000_options.py <city> "<m,X,s[,r0];...>"  (cap m, cut X km or 0 for none, add-back
share s to jobs r0-150 km): baseline, pull spread, then each option. Run with the maps venv."""
import sys
import time
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from t1000lib import Pack, tstats, fmt2, lead2, pull_spread

city = sys.argv[1]
specs = [tuple(float(v) for v in s.split(",")) for s in sys.argv[2].split(";")]
t0 = time.time()
p = Pack(city)
print(f"== {city}: {p.nz} zones, workers {p.W.sum():,.0f}, curve d^-{p.g.get('decay_pow', 0):.1f} exp(-d/{p.g['decay_km']}), joint {p.joint}, header mean {p.g['mean_trip_km']:.1f} km")
if p.joint:
    A, B, it, err = p.furness(tol=1e-6)
    print(f"  re-solved alone: {it} steps, err {err:.1e}")
else:
    A, B = p.row, p.col
Tb = A[:, None] * p.F * B[None, :]
print("  baseline:", fmt2(tstats(p, Tb)), flush=True)
b, wj = pull_spread(p, B)
q = np.percentile(b, [1, 5, 50, 95, 99])
print("  pull (B/J over its mean) percentiles 1/5/50/95/99 by zone:", " ".join(f"{v:.3g}" for v in q))
for m in (3, 10, 30, 100):
    out = wj[(b > m) | (b < 1 / m)].sum() / wj.sum()
    print(f"    jobs in zones pulling beyond {m}x either way: {out:.2%}")
for sp in specs:
    m, X, s = sp[:3]
    r0 = sp[3] if len(sp) > 3 else (X if X > 0 else 50.0)
    T, it, lost = lead2(p, m, X if X > 0 else None, s, r0=r0)
    ab = f", add-back {s:.1%} to {r0:g}-150 km" if s > 0 else ""
    print(f"  cap {m:g}x, cut {'none' if X <= 0 else f'{X:g} km'}{ab}: ({it} steps, workers with no job in reach {lost:.2%})", fmt2(tstats(p, T, Tb)), flush=True)
    del T
print(f"  done in {time.time() - t0:.0f} s")
