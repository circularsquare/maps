"""T-185 question 4: candidate fixes for reverse commuting, re-solved in numpy on the shipped zones.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_candidates.py SET [city ...]

Each candidate is a deterrence F_ij (and how far the work end is constrained), scored with
t185_common.evaluate: class table misplaced, county pairs misplaced, length misplaced (T-162's two),
between-state share, job margin error. Results go to CACHE/cand.json, keyed "city|candidate";
a key already there is skipped, so a set can be rerun or extended.

Inputs every city has: zone residents (commuters), zone jobs, distances, region per zone.
  grav  pw km k             d^-pw exp(-d/km) x k across regions (T-162's model)
  part  km k alpha          the same, work end partly constrained (alpha 0 = origin-constrained)
  dens  km0 beta k          decay length per home zone: km0 (rho_i / 2000)^-beta, rho = commuters
                            living + working per km2 in the zone
  out   km k gamma          x min(1, J_j / J_i)^gamma: a trip to a zone with fewer jobs than home
  rank  a                   intervening opportunities, rank form (Noulas et al. 2012): u^-a, u = jobs
                            at home + jobs nearer than j + half of j's
  rad                       radiation (Simini et al. 2012), doubly constrained
  radO                      radiation, origin-constrained (its own form)
  hyb   km k a              gravity x u^-a (distance and intervening opportunities together)
"""
import json
import os
import sys
import time

import numpy as np

import t185_common as tc

SETS = {
    "base": [("grav", 1.0, 40.0, None), ("grav", 1.0, 40.0, 1.0), ("grav", 1.0, 29.0, 1.0)],
    # the main candidate: gravity x u^-a, one strength a (a = 0 is today's model)
    "hyb": [("hyb", km, k, a) for k in (None, 1.0) for km in (40.0, 60.0) for a in (-0.3, -0.15, 0.15, 0.3, 0.6)],
    # the same with the decay length refitted so the mean trip equals LODES's
    "hybL": [("hybL", a, k) for k in (None, 1.0) for a in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.4)],
    # control: plain gravity shape d^-pw with the length refitted the same way (no opportunities)
    "powL": [("powL", pw, k) for k in (None, 1.0) for pw in (0.6, 0.8)],
    # one shared length per shape, for a single value in every pack
    "shape": [("grav", 0.8, 30.0, None), ("grav", 0.8, 26.0, None), ("grav", 0.7, 25.0, None), ("grav", 0.6, 22.0, None)],
    # T-198: Washington at its refitted factor
    "t198": [("grav", 0.8, 30.0, 0.45)],
    "sch": [("sch", L, m) for m in ("dc", "sc") for L in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4)],
    "rad": [("rad",), ("radO",)],
    "rank": [("rank", a) for a in (0.8, 1.2)],
    "part": [("part", 40.0, None, al) for al in (0.0, 0.5)],
    "dens": [("dens", 40.0, be, None) for be in (-0.15, 0.15)],
    "out": [("out", 40.0, None, g) for g in (0.1, 0.25)],
}


def key(spec, k_city):
    s = list(spec)
    s = ["K" if v is None else v for v in s]
    return " ".join(str(v) for v in s)


def build(c, spec):
    kind = spec[0]
    kk = lambda v: tc.PACK_K.get(c.name, tc.BORDER_US) if v is None else v
    alpha = 1.0
    if kind == "grav":
        _, pw, km, k = spec
        F = c.gravity_F(pw, km, kk(k))
    elif kind == "part":
        _, km, k, alpha = spec
        F = c.gravity_F(1.0, km, kk(k))
    elif kind == "dens":
        _, km0, be, k = spec
        rho = (c.pop + c.jobs) / 4.0
        km_i = np.clip(km0 * (np.maximum(rho, 50.0) / 2000.0) ** -be, 5.0, 300.0)
        F = c.gravity_F(1.0, 40.0, kk(k), km_i=km_i)
    elif kind == "out":
        _, km, k, g = spec
        F = c.gravity_F(1.0, km, kk(k))
        J = c.jobs.astype(np.float32) + 50.0
        F *= np.minimum(1.0, J[None, :] / J[:, None]) ** np.float32(g)
    elif kind in ("hybL", "powL"):
        if kind == "hybL":
            _, a, k = spec
            pw = 1.0
            u = c.opportunities() + (c.jobs.astype(np.float32)[:, None] + 0.5 * c.jobs.astype(np.float32)[None, :])
            G = (np.maximum(u, 50.0) / np.float32(1e5)) ** np.float32(-a)
            del u
        else:
            _, pw, k = spec
            G = np.float32(1.0)
        lo, hi = np.log(8.0), np.log(600.0)
        for _ in range(9):
            km = float(np.exp(0.5 * (lo + hi)))
            F = c.gravity_F(pw, km, kk(k)); F *= G
            T = c.furness(F, tol=1e-3)
            m = float(np.einsum("ij,ij->", T, c.d, dtype=np.float64) / T.sum(dtype=np.float64))
            del F, T
            lo, hi = (lo, np.log(km)) if m > c.obs_mean else (np.log(km), hi)
        km = float(np.exp(0.5 * (lo + hi)))
        c.last_km = km
        F = c.gravity_F(pw, km, kk(k)); F *= G
    elif kind in ("rank", "hyb"):
        u = c.opportunities() + (c.jobs.astype(np.float32)[:, None] + 0.5 * c.jobs.astype(np.float32)[None, :])
        u = np.maximum(u, 50.0)
        if kind == "rank":
            F = u ** np.float32(-spec[1])
        else:
            _, km, k, a = spec
            F = c.gravity_F(1.0, km, kk(k))
            F *= (u / np.float32(1e5)) ** np.float32(-a)
        del u
    elif kind == "sch":
        # Schneider (1959) intervening opportunities: P(j) = exp(-L V) (1 - exp(-L n_j)), V = jobs
        # nearer than j including the home zone's own (home zone first, V = 0)
        _, L, mode = spec
        V = c.opportunities() + c.jobs.astype(np.float32)[:, None]
        np.fill_diagonal(V, 0.0)
        F = np.exp(np.float32(-L) * V)
        del V
        F *= -np.expm1(-L * c.jobs.astype(np.float64)).astype(np.float32)[None, :]
        alpha = "sc" if mode == "sc" else 1.0
    elif kind in ("rad", "radO"):
        s = c.opportunities()
        m = c.jobs.astype(np.float32)[:, None] + 1.0
        n = c.jobs.astype(np.float32)[None, :] + 1.0
        F = m * n / ((m + s) * (m + n + s))
        if kind == "radO":
            alpha = "sc"
    else:
        raise ValueError(kind)
    return F, alpha


def main(args):
    sets = args[0].split(",")
    names = args[1:] or tc.CITIES
    out = os.path.join(tc.CACHE, "cand.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    for name in names:
        specs = [s for st in sets for s in SETS[st]]
        todo = [s for s in specs if f"{name}|{key(s, None)}" not in res]
        if not todo:
            continue
        c = tc.load(name)
        for spec in todo:
            t0 = time.time()
            F, alpha = build(c, spec)
            if alpha == "sc":  # singly (origin) constrained on the model's own form, F already has n_j
                o = c.pop * c.jobs.sum() / c.pop.sum()
                F *= (o / np.maximum(F.sum(1, dtype=np.float64), 1e-30)).astype(np.float32)[:, None]
                T = F
            else:
                T = c.furness(F, alpha=alpha)
            r = c.evaluate(T)
            if spec[0] in ("hybL", "powL"):
                r["km"] = c.last_km
            del F, T
            res = json.load(open(out)) if os.path.exists(out) else {}
            res[f"{name}|{key(spec, None)}"] = r
            json.dump(res, open(out, "w"))
            print(f"  {name} {key(spec, None)}: rel {r['rel_off']:.1%} class {r['class_off']:.1%} county {r['county_off']:.1%} "
                  f"length {r['tld_off']:.1%} cross {r['cross']:.1%} (LODES {c.obs_cross:.1%}) jobs {r['job_margin_off']:.1%} "
                  f"mean {r['mean']:.1f} (LODES {c.obs_mean:.1f}) km {r.get('km', 0):.0f}; C->S x{r['ratio'][0][2]:.2f} S->C x{r['ratio'][2][0]:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        c._s = None
        del c


if __name__ == "__main__":
    main(sys.argv[1:])
