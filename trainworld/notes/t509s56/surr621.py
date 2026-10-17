"""T-621 stage 2: per-city quadratic surrogates over fit621.py's cache, the joint optimum on them,
and the leave-one-city-out refits (as T-509 stage 6's surr.py).

    python surr621.py [radius]     # needs fit621.py's environment (FIT_DIR, FIT_FIXED)

Each city's surrogate is a full quadratic in the values that reach it (`fit621.resolve`), fitted
on the cached runs within `radius` plausible ranges of the best point's values; the joint
objective is fit621's (New York half, penalty by reach). Writes surr.json in FIT_DIR.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import itertools
import json
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit621 as F  # noqa: E402

RADIUS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
best = json.load(open(os.path.join(F.HERE, F.BEST)))
NAMES = list(F.PARAMS)
x0 = np.array([best[k] for k in NAMES])
# the range of each effective input, for scaling and the radius
RNG = {"asc_rail": 1.0, "everyday_asc_rail": 1.0, "other_asc": 10.0, "country_asc_rail": 1.0, "asc_walk": 1.0, "everyday_asc_walk": 1.0}


def feats(city):
    s = F.resolve(city, best)
    return [k for k in s if k in RNG]


def design(v):
    v = np.atleast_2d(v)
    cols = [np.ones(len(v))] + [v[:, i] for i in range(v.shape[1])]
    cols += [v[:, i] * v[:, j] for i, j in itertools.combinations_with_replacement(range(v.shape[1]), 2)]
    return np.column_stack(cols)


cache = list(F.load_cache().values())
models = {}
for c in F.ALL:
    fk = feats(c)
    b = F.resolve(c, best)
    xs, ys = [], []
    for r in cache:
        if r["city"] != c or any(abs(r["set"].get(k, 0.0) - F.FIXED[k]) > 1e-9 for k in F.FIXED):
            continue
        v = [(r["set"].get(k, b[k]) - b[k]) / RNG[k] for k in fk]
        if max(abs(x) for x in v) <= RADIUS:
            xs.append(v)
            ys.append(F.total(c, r))
    xs, ys = np.array(xs), np.array(ys)
    A = design(xs)
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    res = ys - A @ coef
    models[c] = (fk, b, coef)
    print(f"{c}: {len(ys)} runs over {fk}, rms residual {np.sqrt(np.mean(res ** 2)):.4f}, need {A.shape[1]}")


def surr(c, p):
    fk, b, coef = models[c]
    s = F.resolve(c, p)
    v = np.array([(s[k] - b[k]) / RNG[k] for k in fk])
    return float((design(v) @ coef)[0])


def obj(x, fit):
    p = dict(zip(NAMES, x))
    # parameters reaching no fitted city stay at the best point (they are undetermined)
    return sum(F.WEIGHT.get(c, 1.0) * surr(c, p) for c in fit) + F.penalty(p)


# SURR_FIX=k=v,...: parameters held at a value (a variant's optimum)
for _kv in filter(None, os.environ.get("SURR_FIX", "").split(",")):
    _k, _v = _kv.split("=")
    x0[NAMES.index(_k)] = float(_v)
FIXED_NAMES = {kv.split("=")[0] for kv in filter(None, os.environ.get("SURR_FIX", "").split(","))}


def solve(fit):
    live = [i for i, k in enumerate(NAMES) if any(c in fit for c in F.reaches(k)) and k not in FIXED_NAMES]

    def f(y):
        x = x0.copy()
        x[live] = y
        return obj(x, fit)
    r = minimize(f, x0[live], method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-4, "fatol": 1e-6})
    x = x0.copy()
    x[live] = r.x
    return dict(zip(NAMES, [round(float(v), 3) for v in x])), float(r.fun)


out = {}
p, o = solve(F.FIT)
out["joint"] = {"p": p, "objective": o, "scores": {c: surr(c, p) for c in F.ALL}}
print("joint", json.dumps(p), f"{o:.4f}", {c: round(surr(c, p), 3) for c in F.ALL})
for c in F.FIT:
    fit = [x for x in F.FIT if x != c]
    q, oq = solve(fit)
    moves = {k: round(q[k] - p[k], 3) for k in NAMES if abs(q[k] - p[k]) > 0.005}
    out["loo_" + c] = {"p": q, "objective": oq, "dropped_score": surr(c, q), "moves": moves}
    print(f"without {c}: moves {moves}; {c} scores {surr(c, q):.3f} (joint {surr(c, p):.3f})")
json.dump(out, open(os.path.join(F.HERE, os.environ.get("SURR_OUT", "surr.json")), "w"), indent=1)
