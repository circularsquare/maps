"""T-1150 shared helpers (read-only on trainworld/data)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json
import numpy as np
import h3

TW = r"C:\Users\anita\projects\maps\trainworld"
ART = r"C:\Users\anita\projects\maps\artifacts\trainworld\t1150"
sys.path.insert(0, os.path.join(TW, "pipeline"))


def load_sig(city):
    z = np.load(os.path.join(TW, "data", "work", f"{city}.tier0_signals.npz"))
    return {k: z[k] for k in z.files}


def cell_area_km2(ids):
    return np.array([h3.cell_area(h3.int_to_str(int(c)), "km^2") for c in ids])


def sha_units():
    import china
    ds = china.districts("sha")
    return ds  # [(name, geom, pop)]


def sha_rebuild(sig, pop_weight=None, job_weight=None):
    """Res-9 pop and jobs as the sha build makes them: district totals spread by the weights
    (default: GHSL pop, tier 0's unweighted proxy blend)."""
    import china, england, tier0
    ds = sha_units()
    units = [(d[0], d[1]) for d in ds]
    pw = sig["pop_ghsl"] if pop_weight is None else pop_weight
    pop = england.spread_totals(sig["lat"], sig["lon"], units, {d[0]: d[2] for d in ds}, pw)
    s2 = dict(sig)
    s2["pop"] = pop
    jw = tier0.jobs_from(s2, 1.0, tier0.JOB_BLEND) if job_weight is None else job_weight
    jobs = england.spread_totals(sig["lat"], sig["lon"], units, china.JOBS["sha"], jw)
    idx, _ = england.cell_units(sig["lat"], sig["lon"], units)
    return pop, jobs, idx, ds
