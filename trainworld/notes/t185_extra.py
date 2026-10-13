"""T-185: the central city's own boundary (the London comparison) and LODES's side of the relative
yardstick. Run after t185_candidates.py (it reads the refitted decay lengths from cand.json).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_extra.py [city ...]

out = share of the central county's resident commuters who work outside it; in = share of its
jobs held by people living outside it. London 2011 (GLA CIS2014-16, census WU01EW): out 6.8%
(271k of ~3.98M resident workers), in 17.6% (790k of 4.5M).
"""
import json
import os
import sys

import numpy as np

import t185_candidates as cand
import t185_common as tc

CENTRAL = {"nyc": ["36005", "36047", "36061", "36081", "36085"], "phl": ["42101"], "bos": ["25025"],
           "wdc": ["11001"], "bal": ["24510"], "pvd": ["44007"], "hfd": ["09003"], "spf": ["25013"]}


def outin(C, idx):
    C = np.asarray(C)
    m = np.zeros(len(C), bool); m[idx] = True
    res = C[m].sum()
    jobs = C[:, m].sum()
    inside = C[np.ix_(m, m)].sum()
    return float(1 - inside / res), float(1 - inside / jobs)


def main(names):
    out = os.path.join(tc.CACHE, "extra.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    cj = json.load(open(os.path.join(tc.CACHE, "cand.json")))
    for name in names:
        c = tc.load(name)
        r = {"rel_lodes": c.rel_obs(), "ratio_share_lodes": None}
        if name in CENTRAL:
            idx = [c.counties.index(f) for f in CENTRAL[name]]
            r["central"] = {"lodes": outin(c.obs_C, idx)}
            specs = {"now": ("grav", 1.0, 40.0, None), "k1": ("grav", 1.0, 40.0, 1.0),
                     "pre": ("grav", 1.0, 29.0, 1.0),
                     "io-0.2": ("hybL", -0.2, None), "io+0.2": ("hybL", 0.2, None), "io+0.1": ("hybL", 0.1, None),
                     "pow0.8": ("powL", 0.8, None), "pow0.6": ("powL", 0.6, None), "pow0.8k1": ("powL", 0.8, 1.0)}
            for lab, spec in specs.items():
                if spec[0] == "hybL":
                    km = cj[f"{name}|{cand.key(spec, None)}"]["km"]
                    u = c.opportunities() + (c.jobs.astype(np.float32)[:, None] + 0.5 * c.jobs.astype(np.float32)[None, :])
                    F = c.gravity_F(1.0, km, tc.PACK_K.get(name, tc.BORDER_US))
                    F *= (np.maximum(u, 50.0) / np.float32(1e5)) ** np.float32(-spec[1])
                    del u
                elif spec[0] == "powL":
                    km = cj[f"{name}|{cand.key(spec, None)}"]["km"]
                    F = c.gravity_F(spec[1], km, tc.PACK_K.get(name, tc.BORDER_US) if spec[2] is None else spec[2])
                else:
                    F, _ = cand.build(c, spec)
                T = c.furness(F)
                C = c.score(T)["C"]
                del F, T
                r["central"][lab] = outin(C, idx)
                print(f"  {name} {lab}: out {r['central'][lab][0]:.1%} in {r['central'][lab][1]:.1%} "
                      f"(LODES {r['central']['lodes'][0]:.1%} / {r['central']['lodes'][1]:.1%})", flush=True)
        res[name] = r
        json.dump(res, open(out, "w"))
        c._s = None
        del c


if __name__ == "__main__":
    main(sys.argv[1:] or tc.CITIES)
