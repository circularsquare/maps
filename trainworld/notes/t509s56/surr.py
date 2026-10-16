"""Per-city quadratic surrogates of the score in (asc_rail, everyday_asc_rail, everyday_walk_d0_km), from
every cached run with w_walk 1.5 and other_asc 10; the joint optimum, leave-one-out optima, per-city
optima and weighting variants on them. python surr.py"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit

V = ["asc_rail", "everyday_asc_rail", "everyday_walk_d0_km"]
cache = fit.load_cache()


def feats(x):
    x = np.atleast_2d(x)
    cols = [np.ones(len(x))] + [x[:, i] for i in range(3)] + [x[:, i] * x[:, j] for i, j in itertools.combinations_with_replacement(range(3), 2)]
    return np.stack(cols, 1)


S = {}
for c in fit.ALL:
    X, y = [], []
    for r in cache.values():
        p = r["p"]
        if r["city"] == c and abs(p["w_walk"] - 1.5) < 1e-9 and abs(p["other_asc"] - 10) < 1e-9:
            X.append([p[v] for v in V])
            y.append(r["total"])
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(feats(X), y, rcond=None)
    res = y - feats(X) @ coef
    S[c] = coef
    print(f"{c}: {len(y)} points, rms residual {np.sqrt(np.mean(res ** 2)):.3f} (scores {y.min():.2f}-{y.max():.2f})")

x0 = np.array([fit.PARAMS[v][0] for v in V])
rng = np.array([fit.PARAMS[v][1] for v in V])
lo, hi = np.array([-1.0, -2.5, 1.2]), np.array([1.5, 1.0, 2.8])


def obj(x, w):
    return sum(wc * float(feats(x) @ S[c]) for c, wc in w.items()) + sum(w.values()) * fit.PENALTY * float((((x - x0) / rng) ** 2).sum())


def best(w, pen=True):
    f = (lambda x: obj(x, w)) if pen else (lambda x: sum(wc * float(feats(x) @ S[c]) for c, wc in w.items()))
    r = min((minimize(f, s, bounds=list(zip(lo, hi))) for s in [x0, np.array([0.25, -1.1, 1.85])]), key=lambda r: r.fun)
    return r.x


def show(name, x):
    pred = {c: float(feats(x) @ S[c]) for c in fit.ALL}
    print(f"{name:28s} asc {x[0]:+.2f} ev {x[1]:+.2f} walk {x[2]:.2f} | " + " ".join(f"{c} {pred[c]:.2f}" for c in fit.ALL))
    return x


out = {}
eq = {c: 1.0 for c in fit.FIT}
out["joint"] = show("joint (six, penalty)", best(eq)).tolist()
for c in fit.FIT:
    out[f"loo_{c}"] = show(f"without {c}", best({k: 1.0 for k in fit.FIT if k != c})).tolist()
out["nyc_half"] = show("New York at half weight", best(dict(eq, nyc=0.5))).tolist()
for c in fit.ALL:
    show(f"{c} alone (no penalty)", best({c: 1.0}, pen=False))
show("today", x0)
json.dump(out, open(os.path.join(fit.HERE, "surr.json"), "w"), indent=1)
