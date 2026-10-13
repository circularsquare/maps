"""T-185: what the candidates move, in T-159/T-162's terms. New York county pairs (the rail ones)
and Center City residents' work places (zone approximation: a zone's share of residents in
T-159's Center City box), for the shipped gravity, no factor, and the decay-shape candidates.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_pairs.py
"""
import json
import os

import numpy as np

import t159_flows as tf9
import t185_candidates as cand
import t185_common as tc

VARS = {"now": ("grav", 1.0, 40.0, None), "k1": ("grav", 1.0, 40.0, 1.0),
        "pow0.8": ("powL", 0.8, None), "pow0.6": ("powL", 0.6, None), "pow0.6k1": ("powL", 0.6, 1.0)}
NYC_PAIRS = [("Hudson", "New York"), ("Bergen", "New York"), ("Essex", "New York"), ("Passaic", "New York"),
             ("Union", "New York"), ("Rockland", "Bergen"), ("Kings", "New York"), ("Queens", "New York"),
             ("Bronx", "New York"), ("Westchester", "New York"), ("Nassau", "New York"), ("Suffolk", "New York"),
             ("Fairfield", "New York"), ("New York", "Hudson"), ("New York", "Westchester"), ("New York", "Nassau"),
             ("New York", "Kings"), ("New York", "New York")]


def solve(c, spec, cj):
    if spec[0] == "powL":
        km = cj[f"{c.name}|{cand.key(spec, None)}"]["km"]
        k = tc.PACK_K.get(c.name, tc.BORDER_US) if spec[2] is None else spec[2]
        return c.furness(c.gravity_F(spec[1], km, k))
    F, _ = cand.build(c, spec)
    return c.furness(F)


def main():
    cj = json.load(open(os.path.join(tc.CACHE, "cand.json")))
    c = tc.load("nyc")
    names = {f: n for f, n in tc.bc.CITIES["nyc"]["counties"].items()}
    idx = {}
    for i, f in enumerate(c.counties):
        n = names[f]
        idx[n.replace(" NY", "").replace(" NJ", "").replace(" CT", "")] = i
    O = c.obs_C / c.obs_C.sum(1, keepdims=True)
    res = {}
    for lab, spec in VARS.items():
        C = c.score(solve(c, spec, cj))["C"]
        res[lab] = C / C.sum(1, keepdims=True)
    print("| home -> work | LODES | " + " | ".join(VARS) + " |")
    print("|---|---|" + "---|" * len(VARS))
    for h, w in NYC_PAIRS:
        if h not in idx or w not in idx:
            print(f"  missing {h} or {w}: {sorted(idx)}")
            continue
        i, j = idx[h], idx[w]
        print(f"| {h} -> {w} | {O[i, j]:.1%} | " + " | ".join(f"{res[v][i, j]:.1%} ({res[v][i, j] / O[i, j]:.2f}x)" for v in VARS) + " |")
    del c
    # Center City, zone approximation
    c = tc.load("phl")
    b = c.blocks
    cc = tf9.in_cc(b["lat"].to_numpy(), b["lon"].to_numpy())
    ok = c.bzone >= 0
    pz = np.bincount(c.bzone[ok], weights=(b["pop"].to_numpy() * cc)[ok], minlength=len(c.pop))
    pt = np.bincount(c.bzone[ok], weights=b["pop"].to_numpy()[ok], minlength=len(c.pop))
    share = np.where(pt > 0, pz / np.maximum(pt, 1e-9), 0.0)
    cn = {f: i for i, f in enumerate(c.counties)}
    groups = {"Philadelphia": [cn["42101"]], "PA suburbs": [cn[f] for f in ("42017", "42029", "42045", "42091")],
              "NJ": [i for f, i in cn.items() if f.startswith("34")]}
    print("\n| Center City residents work in | LODES (blocks, t185_centercity) | " + " | ".join(VARS) + " |")
    print("|---|---|" + "---|" * len(VARS))
    rows = {g: [] for g in groups}
    for lab, spec in VARS.items():
        T = solve(c, spec, cj)
        v = (share.astype(np.float32) @ T) @ c.Q
        v = v / v.sum()
        for g, cols in groups.items():
            rows[g].append(f"{v[cols].sum():.1%}")
        del T
    lodes = {"Philadelphia": "71.9%", "PA suburbs": "22.2%", "NJ": "5.9%"}
    for g in groups:
        print(f"| {g} | {lodes[g]} | " + " | ".join(rows[g]) + " |")


if __name__ == "__main__":
    main()
