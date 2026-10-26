"""T-1000: python t1000_clusters.py <city> [gap_km] [--cap] [--opts]: baseline, settlement clusters,
long trips by cluster pair; --cap fill per cluster under the 20x cap; --opts the first option sweep."""
import sys
import time
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from t1000lib import Pack, fmt

city = sys.argv[1]
gap = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 25.0
opts = "--opts" in sys.argv
t0 = time.time()
p = Pack(city)
print(f"== {city}: {p.nz} zones, workers {p.W.sum():,.0f}, pop {p.pop.sum():,.0f}, curve d^-{p.g.get('decay_pow', 0):.1f} exp(-d/{p.g['decay_km']}), joint {p.joint}, header mean {p.g['mean_trip_km']:.1f} km; built in {time.time() - t0:.0f} s")
cl = p.clusters(gap_km=gap)
names = p.cluster_names(cl)
if p.joint:
    A, B, it, err = p.furness()
    print(f"re-solved alone: {it} steps, err {err:.1e}")
else:
    A, B = p.row, p.col
base = p.metrics(A, B, cl)
print("baseline:", fmt(base))
T = p._T
nc = cl.max() + 1
Wc = np.bincount(cl, weights=p.W, minlength=nc)
Jc = np.bincount(cl, weights=p.J, minlength=nc)
Pc = np.bincount(cl, weights=p.pop, minlength=nc)
print(f"clusters (seeds of 100+ commute ends, joined within {gap:g} km): {nc}; the 12 biggest by workers:")
home_out = np.array([T[cl == c][:, cl != c].sum() for c in range(nc)])
work_in = np.array([T[cl != c][:, cl == c].sum() for c in range(nc)])
TD = T * p.D
for c in range(min(nc, 12)):
    rows = cl == c
    mres = TD[rows].sum() / max(T[rows].sum(), 1e-9)
    print(f"  {c:3d} {names[c][:22]:22s} pop {Pc[c]:9,.0f} workers {Wc[c]:8,.0f} jobs {Jc[c]:8,.0f} jobs/workers {Jc[c] / max(Wc[c], 1):5.2f}  "
          f"residents working in another cluster {home_out[c] / max(Wc[c], 1):6.1%}, jobs filled from another {work_in[c] / max(Jc[c], 1):6.1%}, residents' mean {mres:6.1f} km")
small = slice(12, None)
print(f"  the other {max(nc - 12, 0)}: workers {Wc[small].sum():,.0f}, jobs {Jc[small].sum():,.0f}")
# long trips by cluster pair
far = p.D > 50
Tf = np.where(far, T, 0)
pairs = {}
ci = cl
M = np.zeros((nc, nc))
np.add.at(M, (ci[:, None].repeat(p.nz, 1).ravel(), ci[None, :].repeat(p.nz, 0).ravel()), Tf.ravel()) if p.nz < 3000 else None
if p.nz >= 3000:
    for c in range(nc):
        r = Tf[cl == c].sum(axis=0)
        M[c] = np.bincount(cl, weights=r, minlength=nc)
tf = Tf.sum()
print(f"trips over 50 km: {tf:,.0f} ({tf / T.sum():.1%}); the top cluster pairs (home -> work):")
flat = np.argsort(-M.ravel())[:10]
for k in flat:
    h, w = divmod(k, nc)
    if M[h, w] <= 0:
        break
    sel = (cl[:, None] == h) & (cl[None, :] == w) & far
    dm = (T[sel] * p.D[sel]).sum() / T[sel].sum()
    print(f"  {names[h][:18]:18s} -> {names[w][:18]:18s} {M[h, w]:9,.0f} ({M[h, w] / tf:5.1%} of them), mean {dm:5.0f} km;"
          f" home jobs/workers {Jc[h] / max(Wc[h], 1):.2f}, work {Jc[w] / max(Wc[w], 1):.2f}")
same = sum(M[c, c] for c in range(nc))
print(f"  of the trips over 50 km, {same / tf if tf else 0:.1%} stay inside one cluster")
if "--cap" in sys.argv:
    A4, B4, it, err = p.furness(clamp=20.0, tol=1e-6)
    T4 = A4[:, None] * p.F * B4[None, :]
    col = np.bincount(cl, weights=T4.sum(axis=0), minlength=nc)
    print("under the 20x cap: commuters working in each cluster against its jobs")
    for c in range(min(nc, 12)):
        print(f"  {names[c][:22]:22s} jobs {Jc[c]:9,.0f} filled {col[c]:9,.0f} ({col[c] / max(Jc[c], 1):6.1%}), residents' mean {(T4[cl == c] * p.D[cl == c]).sum() / max(T4[cl == c].sum(), 1e-9):5.1f} km")
    big = np.argsort(-Jc)
    for c in big[:20]:
        if c >= 12:
            print(f"  (also) {names[c][:22]:22s} jobs {Jc[c]:9,.0f} workers {Wc[c]:9,.0f} filled {col[c]:9,.0f} ({col[c] / max(Jc[c], 1):6.1%})")
if not opts:
    sys.exit()
print("-- options --")
# A: jobs rebalanced to workers per cluster
Jr = p.J.copy()
for c in range(nc):
    m = cl == c
    if Jc[c] > 0:
        Jr[m] = p.J[m] * Wc[c] / Jc[c]
    else:
        Jr[m] = p.W[m]
A1, B1, it, err = p.furness(J=Jr)
print(f"A rebalance per cluster ({it} steps, err {err:.1e}):", fmt(p.metrics(A1, B1, cl, J=Jr)))
# A + 0.4 between clusters
F4 = p.F * np.where(cl[:, None] != cl[None, :], 0.4, 1.0)
A2, B2, it, err = p.furness(J=Jr, F=F4)
print(f"A + 0.4 across clusters ({it} steps, err {err:.1e}):", fmt(p.metrics(A2, B2, cl, F=F4, J=Jr)))
del F4
for m in (1.0, 3.0, 10.0):
    A3, B3, it, err = p.furness(clamp=m)
    print(f"B pull capped at {m:g}x ({it} steps):", fmt(p.metrics(A3, B3, cl)))
print(f"done in {time.time() - t0:.0f} s")
