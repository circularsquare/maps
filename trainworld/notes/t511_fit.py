import numpy as np, pandas as pd
from scipy.optimize import minimize
R = r"C:\Users\anita\projects\maps\trainworld\data\raw\t511"
a = pd.read_csv(R + r"\acsdt5y2023-b08201.dat", sep="|", dtype={"GEO_ID": str})
nat = a[a.GEO_ID == "0100000US"].iloc[0]
E = lambda r, i: r[f"B08201_E{i:03d}"]
# persons by household size: 1, 2, 3, 4+ (4+ at 4.6)
size = {7: 1, 13: 2, 19: 3, 25: 4.6}
p0 = sum(E(nat, b + 1) * s for b, s in size.items())
pall = sum(E(nat, b) * s for b, s in size.items())
h0, hall = E(nat, 2), E(nat, 1)
s0 = p0 / h0
s1 = (pall - p0) / (hall - h0)
print(f"US: no-car hh {h0/hall:.3f}, persons {p0/pall:.3f}, size no-car {s0:.2f}, car {s1:.2f}, ratio {s0/s1:.3f}")
t = a[a.GEO_ID.str.startswith("1400000US")]
tot = E(t, 1).astype(float)
t = t[tot >= 50]
tot = E(t, 1).astype(float)
us_c = (E(t, 3) + 2 * E(t, 4) + 3 * E(t, 5) + 4.3 * E(t, 6)) / tot
us_h = E(t, 2) / tot
e = pd.read_csv(R + r"\ts045_lsoa2021.csv")
e = e[e["all"] >= 50]
en_c = (e.c1 + 2 * e.c2 + 3.2 * e.c3) / e["all"]
en_h = e.c0 / e["all"]
c = np.concatenate([us_c, en_c]); h = np.concatenate([us_h, en_h]); w = np.concatenate([tot, e["all"]])
def f(x, c):
    a_, b_ = x
    return np.exp(-a_ * c ** b_)
def loss(x, c, h, w):
    return np.sum(w * (f(x, c) - h) ** 2) / w.sum()
for name, cc, hh, ww in [("pooled", c, h, w), ("US", us_c.to_numpy(), us_h.to_numpy(), tot.to_numpy()), ("England", en_c.to_numpy(), en_h.to_numpy(), e["all"].to_numpy())]:
    r = minimize(loss, [1.0, 1.0], args=(cc, hh, ww), method="Nelder-Mead")
    print(name, r.x, "rmse", np.sqrt(r.fun), "poisson rmse", np.sqrt(loss([1, 1], cc, hh, ww)))
x = minimize(loss, [1.0, 1.0], args=(c, h, w), method="Nelder-Mead").x
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1), (1, 1.2), (1.2, 1.5), (1.5, 2), (2, 3)]:
    for nm, cc, hh, ww in [("US", us_c, us_h, tot), ("EN", en_c, en_h, e["all"])]:
        m = (cc >= lo) & (cc < hi)
        if m.sum() < 20: continue
        print(f"c {lo}-{hi} {nm} n={m.sum()} obs {np.average(hh[m], weights=ww[m]):.3f} fit {np.average(f(x, cc[m]), weights=ww[m]):.3f} poisson {np.average(np.exp(-cc[m]), weights=ww[m]):.3f}")
# England national: c and h0
print("England c", (e.c1 + 2*e.c2 + 3.2*e.c3).sum()/e["all"].sum(), "h0", e.c0.sum()/e["all"].sum())
