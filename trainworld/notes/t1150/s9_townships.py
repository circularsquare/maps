"""T-1150: townships furthest off in the pack's homes, and the same with the ratio2 variant."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *

HERE = os.path.dirname(os.path.abspath(__file__))
v = np.load(os.path.join(HERE, "sha_variants_pop.npz"))
tidx = np.load(os.path.join(HERE, "sha_tidx.npy"))
tw = json.load(open(os.path.join(HERE, "sha_townships.json"), encoding="utf-8"))
area = np.load(os.path.join(HERE, "sha_area.npy"))
truth = np.array([t[2] for t in tw], float)
ok = tidx >= 0
km2 = np.bincount(tidx[ok], weights=area[ok], minlength=len(tw))
m = {k: np.bincount(tidx[ok], weights=v[k][ok], minlength=len(tw)) for k in ("pack", "ratio2")}
diff = m["pack"] - truth
o = np.argsort(-np.abs(diff))
print("township | census 2020 | km2 | census /km2 | pack | pack off | ratio2 | ratio2 off")
for i in o[:30]:
    print(f"{tw[i][1]} | {truth[i]:,.0f} | {km2[i]:.1f} | {truth[i] / max(km2[i], .01):,.0f} | {m['pack'][i]:,.0f} | {m['pack'][i] / truth[i] - 1:+.0%} | "
          f"{m['ratio2'][i]:,.0f} | {m['ratio2'][i] / truth[i] - 1:+.0%}")
for k in m:
    r = m[k] / np.maximum(truth, 1)
    big = truth > 20000
    print(f"{k}: townships off by more than 30%: {(np.abs(r[big] - 1) > 0.3).sum()} of {big.sum()}; by more than 50%: {(np.abs(r[big] - 1) > 0.5).sum()}")
print("densest townships by census:", ", ".join(f"{tw[i][1]} {truth[i] / km2[i]:,.0f}" for i in np.argsort(-truth / np.maximum(km2, .01))[:8]))
