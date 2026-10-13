"""T-159: is the cross-river overshoot the water or the state line? Model county flows against
ACS (and LODES for New York) for pairs split by kind: land state border, water within one state,
water plus state border, same side. Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t159_borders.py SCRATCH

SCRATCH is t159_flows.py's output folder (model_class_flows.npy).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, HERE)
import t159_flows as tf  # noqa: E402

OUT = sys.argv[1]


def report(name, M, A, L, idx, pairs):
    homeM, homeA = M.sum(1), A.sum(1)
    homeL = L.sum(1) if L is not None else None
    print(f"\n{name}: | home -> work | kind | model share | ACS share | model/ACS |" + (" LODES share | model/LODES |" if L is not None else ""))
    for h, w, kind in pairs:
        i, j = idx[h], idx[w]
        m, a = M[i, j] / homeM[i], A[i, j] / homeA[i]
        s = f"| {h} -> {w} | {kind} | {m:.2%} | {a:.2%} | {m / a:.2f}x |" if a > 0 else f"| {h} -> {w} | {kind} | {m:.2%} | 0 | |"
        if L is not None:
            l = L[i, j] / homeL[i]
            s += f" {l:.2%} | {m / l:.2f}x |"
        print(s)


def phl():
    Mc = np.load(os.path.join(OUT, "model_class_flows.npy"))
    nc = tf.NC
    M = Mc.reshape(nc, 4, nc, 4).sum((1, 3))
    acs = pd.read_csv(os.path.join(tf.RAW, "acsflows", "phl_county_flows.csv"), dtype={"h": str, "w": str})
    A = np.zeros((nc, nc))
    for r in acs.itertuples():
        if r.w in tf.CIDX:
            A[tf.CIDX[r.h], tf.CIDX[r.w]] += r.n
    idx = {tf.COUNTIES[f]: tf.CIDX[f] for f in tf.CF}
    pairs = [
        ("New Castle", "Delaware", "land state border"), ("Delaware", "New Castle", "land state border"),
        ("New Castle", "Chester", "land state border"), ("Chester", "New Castle", "land state border"),
        ("Cecil", "New Castle", "land state border"), ("New Castle", "Cecil", "land state border"),
        ("Cecil", "Chester", "land state border"), ("Chester", "Cecil", "land state border"),
        ("Salem", "New Castle", "water + state"), ("New Castle", "Salem", "water + state"),
        ("Gloucester", "Delaware", "water + state"), ("Delaware", "Gloucester", "water + state"),
        ("Burlington", "Bucks", "water + state"), ("Bucks", "Burlington", "water + state"),
        ("Philadelphia", "Camden", "water + state"), ("Camden", "Philadelphia", "water + state"),
        ("Philadelphia", "Delaware", "same side"), ("Chester", "Delaware", "same side"),
        ("Camden", "Burlington", "same side"), ("Camden", "Gloucester", "same side"),
    ]
    report("Philadelphia pack (ACS 2016-2020)", M, A, None, idx, pairs)


def nyc():
    import county_flows as cf
    import realnet
    M = cf.model_flows()
    acs = cf.acs_flows()
    order = realnet.COUNTY_ORDER
    I = {f: i for i, f in enumerate(order)}
    A = np.zeros((len(order), len(order)))
    for r in acs.itertuples():
        if r.w in I:
            A[I[r.h], I[r.w]] += r.n
    L = np.load(os.path.join(realnet.WORK, "lodes_county_flows.npy"))
    idx = {realnet.COUNTIES[f]: I[f] for f in order}
    pairs = [
        ("Rockland", "Bergen", "land state border"), ("Bergen", "Rockland", "land state border"),
        ("Orange NY", "Passaic", "land state border"), ("Passaic", "Orange NY", "land state border"),
        ("Orange NY", "Sussex", "land state border"), ("Sussex", "Orange NY", "land state border"),
        ("Rockland", "Westchester", "water within NY (Hudson, 1 bridge)"), ("Westchester", "Rockland", "water within NY (Hudson, 1 bridge)"),
        ("Orange NY", "Dutchess", "water within NY (Hudson, 2 bridges)"), ("Dutchess", "Orange NY", "water within NY (Hudson, 2 bridges)"),
        ("Orange NY", "Westchester", "water within NY (Hudson)"), ("Westchester", "Nassau", "water within NY (Sound, no bridge)"),
        ("Nassau", "Westchester", "water within NY (Sound, no bridge)"), ("Suffolk", "Westchester", "water within NY (Sound, no bridge)"),
        ("Richmond", "Kings", "water within NY (Narrows, 1 bridge)"), ("Kings", "Richmond", "water within NY (Narrows, 1 bridge)"),
        ("Kings", "New York", "water within NY (East River, many)"), ("Queens", "New York", "water within NY (East River, many)"),
        ("Queens", "Bronx", "water within NY (East River, 2 bridges)"), ("Bronx", "Queens", "water within NY (East River, 2 bridges)"),
        ("Hudson", "Union", "water within NJ (Newark Bay)"), ("Union", "Hudson", "water within NJ (Newark Bay)"),
        ("Monmouth", "Middlesex", "water within NJ (Raritan Bay/River)"), ("Middlesex", "Monmouth", "same side, land"),
        ("Hudson", "New York", "water + state (Hudson)"), ("Bergen", "New York", "water + state (Hudson)"),
        ("New York", "Hudson", "water + state (Hudson)"), ("New York", "Bergen", "water + state (Hudson)"),
        ("Bergen", "Westchester", "water + state (Hudson)"), ("Westchester", "Bergen", "water + state (Hudson)"),
        ("Richmond", "Union", "water + state (Arthur Kill)"), ("Richmond", "Middlesex", "water + state (Arthur Kill)"),
        ("Union", "Richmond", "water + state (Arthur Kill)"), ("Middlesex", "Richmond", "water + state (Arthur Kill)"),
        ("Bergen", "Passaic", "same side, land"), ("Essex", "Union", "same side, land"), ("Nassau", "Queens", "same side, land"),
    ]
    report("New York pack (ACS 2016-2020; LODES 2023)", M, A, L, idx, pairs)


if __name__ == "__main__":
    phl()
    nyc()
