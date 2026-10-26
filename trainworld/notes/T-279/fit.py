"""T-279 research: fit drive-time models on the OSRM pairs, held out by region (leave one table
out), and score them in minutes and in rail's odds against the car.
python -I fit.py <scratch dir>"""
import json, math, os, sys
import numpy as np
from scipy.optimize import least_squares

S = sys.argv[1]
recs = json.load(open(os.path.join(S, "pairs.json")))
ROWS = json.load(open(r"C:\Users\anita\projects\maps\trainworld\app\src\game\market-rows.json", encoding="utf-8"))


def clean(r):
    a, b = ROWS[r["a"]], ROWS[r["b"]]
    if r["snap_m"] >= 3000 or r["osrm_min"] <= 0 or r["osrm_km"] < r["gc_km"]:
        return False
    if (r["ca"] == "GB") != (r["cb"] == "GB"):
        return False  # the market adds the Channel crossing itself
    if (a["country"] == "JP" and a["lat"] > 41.4) != (b["country"] == "JP" and b["lat"] > 41.4):
        return False  # and the Tsugaru ferry
    if {r["ca"], r["cb"]} == {"KR", "CN"}:
        return False  # one landmass in the market through North Korea; no road truth
    return True


recs = [r for r in recs if clean(r)]
tab = [r for r in recs if r["cluster"] != "t279"]
old = [r for r in recs if r["cluster"] == "t279"]
print("table pairs", len(tab), "clusters", len({r["cluster"] for r in tab}), "t279 pairs", len(old))


def X(rs, cols):
    return np.array([[r[c] if not callable(c) else c(r) for c in cols] for r in rs], float)


d = lambda r: r["gc_km"]
FEATS = {
    "ldet": lambda r: math.log(max(r["land4_km"], r["gc_km"]) / r["gc_km"]),
    "water": lambda r: r["water_frac"],
    "slope": lambda r: r["slope_mean"] / 10,
    "mtn": lambda r: r["frac_mtn"],
    "hwf": lambda r: r["hw_frac"],
    "lgd": lambda r: math.log(r["gc_km"] / 200),
}


class Rule:
    name = "rule (today)"
    def fit(self, rs): return self
    def pred(self, rs):
        g = X(rs, [d])[:, 0]
        return 15 + 1.25 * g / 85 * 60, 1.25 * g


class Level:
    """T = a + b d, K = c d (one speed and one detour, refit)."""
    name = "rule refit (a + b d)"
    def fit(self, rs):
        g, T, K = X(rs, [d])[:, 0], X(rs, ["osrm_min"])[:, 0], X(rs, ["osrm_km"])[:, 0]
        f = lambda p: np.log(p[0] + p[1] * g) - np.log(T)
        self.p = least_squares(f, [15, 0.8], bounds=([0, 0.1], [60, 3]), loss="soft_l1", f_scale=0.1).x
        self.c = math.exp(np.median(np.log(K / g)))
        return self
    def pred(self, rs):
        g = X(rs, [d])[:, 0]
        return self.p[0] + self.p[1] * g, self.c * g


class Formula:
    """T = a + d exp(b0 + b.x); log(K / d) = c0 + c.x."""
    def __init__(self, feats):
        self.feats = feats
        self.name = "formula: " + ", ".join(feats)
    def fit(self, rs):
        g, T, K = X(rs, [d])[:, 0], X(rs, ["osrm_min"])[:, 0], X(rs, ["osrm_km"])[:, 0]
        x = X(rs, [FEATS[f] for f in self.feats]) if self.feats else np.zeros((len(rs), 0))
        f = lambda p: np.log(p[0] + g * np.exp(p[1] + x @ p[2:])) - np.log(T)
        p0 = np.r_[10, math.log(0.8), np.zeros(x.shape[1])]
        lo = np.r_[0, -5, -np.inf * np.ones(x.shape[1])]; hi = np.r_[60, 5, np.inf * np.ones(x.shape[1])]
        self.p = least_squares(f, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1).x
        A = np.column_stack([np.ones(len(rs)), x])
        self.c = np.linalg.lstsq(A, np.log(K / g), rcond=None)[0]
        return self
    def pred(self, rs):
        g = X(rs, [d])[:, 0]
        x = X(rs, [FEATS[f] for f in self.feats]) if self.feats else np.zeros((len(rs), 0))
        return self.p[0] + g * np.exp(self.p[1] + x @ self.p[2:]), g * np.exp(np.column_stack([np.ones(len(rs)), x]) @ self.c)


class Raster:
    """T = a + b * least-cost minutes on the 1/60 degree raster; K = c * that path's km."""
    def __init__(self, v):
        self.v = v
        self.name = f"raster router {v}"
    def fit(self, rs):
        m, T = X(rs, [self.v + "_min"])[:, 0], X(rs, ["osrm_min"])[:, 0]
        km, K = X(rs, [self.v + "_km"])[:, 0], X(rs, ["osrm_km"])[:, 0]
        f = lambda p: np.log(p[0] + p[1] * m) - np.log(T)
        self.p = least_squares(f, [10, 1], bounds=([0, 0.1], [60, 5]), loss="soft_l1", f_scale=0.1).x
        self.c = math.exp(np.median(np.log(K / km)))
        return self
    def pred(self, rs):
        return self.p[0] + self.p[1] * X(rs, [self.v + "_min"])[:, 0], self.c * X(rs, [self.v + "_km"])[:, 0]


class Country:
    """T = a + b_country d (a country's own speed where it has 15+ training pairs in it)."""
    name = "speed by country"
    def fit(self, rs):
        self.lv = Level().fit(rs)
        self.b = {}
        for c in {r["ca"] for r in rs}:
            sub = [r for r in rs if r["ca"] == c and r["cb"] == c]
            if len(sub) >= 15:
                g, T = X(sub, [d])[:, 0], X(sub, ["osrm_min"])[:, 0]
                self.b[c] = float(np.median((T - self.lv.p[0]) / g))
        return self
    def pred(self, rs):
        T0, K0 = self.lv.pred(rs)
        T = np.array([self.lv.p[0] + r["gc_km"] * (self.b.get(r["ca"], self.lv.p[1]) + self.b.get(r["cb"], self.lv.p[1])) / 2 for r in rs])
        return T, K0


def car(T, K):
    return T + K * 0.15 * 3 / 2


def score(rs, T, K):
    Tt, Kt = X(rs, ["osrm_min"])[:, 0], X(rs, ["osrm_km"])[:, 0]
    ct = car(Tt, Kt)
    b = 0.10 * np.sqrt(np.minimum(1, 60 / ct))
    lo = b * (car(T, K) - ct)  # >0: the model's car is dearer, so rail's odds too high
    e = np.abs(T - Tt)
    return dict(n=len(rs), med_min=float(np.median(e)), p90_min=float(np.percentile(e, 90)),
                med_pct=float(np.median(np.abs(T / Tt - 1)) * 100), p90_pct=float(np.percentile(np.abs(T / Tt - 1), 90) * 100),
                med_odds=float(np.exp(np.median(np.abs(lo)))), p90_odds=float(np.exp(np.percentile(np.abs(lo), 90))),
                bias=float(np.median(T / Tt)))


MODELS = [Rule(), Level(), Country(), Formula([]), Formula(["ldet"]), Formula(["ldet", "water"]), Formula(["ldet", "slope"]),
          Formula(["ldet", "slope", "hwf"]), Formula(["ldet", "slope", "mtn", "hwf"]), Formula(["ldet", "slope", "hwf", "lgd"]),
          Formula(["slope", "hwf"]),
          Raster("v5"), Raster("v6"), Raster("v7"), Raster("v8")]
clusters = sorted({r["cluster"] for r in tab})
results = {}
print(f"{'model':45s} | held-out by region: min med / p90 | % med / p90 | odds med / p90 | bias || T-279 pairs: min med / p90 | odds med / p90")
for m in MODELS:
    Tp = np.zeros(len(tab)); Kp = np.zeros(len(tab))
    for c in clusters:
        tr = [r for r in tab if r["cluster"] != c]
        te_i = [i for i, r in enumerate(tab) if r["cluster"] == c]
        mm = m.fit(tr)
        T, K = mm.pred([tab[i] for i in te_i])
        Tp[te_i] = T; Kp[te_i] = K
    s = score(tab, Tp, Kp)
    full = m.fit(tab)
    To, Ko = full.pred(old)
    so = score(old, To, Ko)
    results[m.name] = {"heldout": s, "t279": so, "params": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in vars(full).items() if k in ("p", "c", "b")}}
    for i, r in enumerate(old):
        r.setdefault("pred", {})[m.name] = [float(To[i]), float(Ko[i])]
    print(f"{m.name:45s} | {s['med_min']:5.1f} / {s['p90_min']:5.1f} | {s['med_pct']:4.1f} / {s['p90_pct']:4.1f} | x{s['med_odds']:.2f} / x{s['p90_odds']:.2f} | {s['bias']:.2f} || "
          f"{so['med_min']:5.1f} / {so['p90_min']:5.1f} | x{so['med_odds']:.2f} / x{so['p90_odds']:.2f}")
json.dump({"results": results, "t279": old}, open(os.path.join(S, "fit_results.json"), "w"), indent=1)
