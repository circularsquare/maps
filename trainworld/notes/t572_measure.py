"""T-573: bounding the height weight in tier 0's GHSL non-residential term, measured on London,
Toronto and Berlin. Read-only on packs and caches; writes ../artifacts/trainworld/T-573-measure.json.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t572_measure.py

"Misplaced" is half the L1 distance between two shares of jobs by 2 km zone (notes/t360_jobs.py), as
T-460 measured it: London the proxy on its cached signals against BRES by LSOA spread inside each
LSOA by today's proxy (fixed); Toronto the proxy on t0tor's signals against the real pack's jobs
(StatCan place of work, the res-9 copy). Berlin: its pack's jobs rule (Zensus people, Pendlerrechnung
jobs by Gemeinde spread by the proxy), the largest and 10th cell.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys

import h3
import numpy as np
from shapely import make_valid, union_all
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, HERE)
import build_city as b  # noqa: E402
import england  # noqa: E402
import germany  # noqa: E402
import tier0  # noqa: E402
from read_pack import load_pack  # noqa: E402
import t360_jobs as t3  # noqa: E402

OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-573-measure.json")


def nres_variant(sig, h_cap=None, share_cap=None, pow_=tier0.HEIGHT_POW):
    v, s = sig["nres_v"], sig["nres_s"]
    h = np.where(s > 0, v / np.maximum(s, 1e-9), 0.0)
    if h_cap:
        h = np.minimum(h, h_cap)
    w = v * np.power(np.maximum(h, 1e-9) / 10.0, pow_)
    if share_cap:
        w = cap_share(w, share_cap)
    return w


def cap_share(w, c):
    """No cell over share c of the total; the excess to the others in proportion (as T-387)."""
    w = w.astype(float).copy()
    T = w.sum()
    capped = np.zeros(len(w), bool)
    for _ in range(50):
        over = (w > c * T) & ~capped
        if not over.any():
            break
        capped |= over
        w[capped] = c * T
        free = w[~capped].sum()
        w[~capped] *= (T - c * T * capped.sum()) / free
    return w


def proxy(sig, nres, blend=None):
    blend = blend or tier0.JOB_BLEND
    osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
    parts = {"nres": nres, "osm": osm, "pop": sig["pop"]}
    return sum(blend[k] * parts[k] / max(parts[k].sum(), 1e-9) for k in blend)


def zone_totals(sig, v, origin, keys):
    return np.bincount(np.searchsorted(keys, t3.zone_keys(sig["lat"], sig["lon"], origin)), weights=v, minlength=len(keys))


VARIANTS = [("today", {})] + [(f"h<={h}", {"h_cap": h}) for h in (20, 25, 30, 35, 40, 50, 60)] + \
           [(f"share<={c:.2%}", {"share_cap": c}) for c in (0.0025, 0.005, 0.01, 0.02)]


def main():
    out = {}
    # London
    lon = dict(np.load(os.path.join(b.WORK, "lon.tier0_signals.npz")))
    bnd = union_all([make_valid(shape(f["geometry"])) for f in json.load(open(england.raw("lon_itl3.geojson"), encoding="utf-8"))["features"]])
    ref = england.bres_jobs("lon", bnd, lon, tier0.jobs_from(lon, 1.0))[0]
    o_l = b.CITIES["lon"]["origin"]
    zk_l = np.unique(t3.zone_keys(lon["lat"], lon["lon"], o_l))
    ref_l = zone_totals(lon, ref, o_l, zk_l)
    # Toronto
    tor = dict(np.load(os.path.join(ROOT, "data", "work", "t287", "work", "t0tor.tier0_signals.npz")))
    h, a = load_pack("tor", os.path.join(ROOT, "data", "work", "res9"))
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(x))) for x in a["h3"]])
    pk = {"lat": ll[:, 0], "lon": ll[:, 1]}
    o_t = b.CITIES["tor"]["origin"]
    zk_t = np.union1d(t3.zone_keys(tor["lat"], tor["lon"], o_t), t3.zone_keys(pk["lat"], pk["lon"], o_t))
    ref_t = zone_totals(pk, a["jobs"].astype(float), o_t, zk_t)
    # Berlin, as its build: Zensus people in place of GHSL's before the proxy
    ber = dict(np.load(os.path.join(b.WORK, "ber.tier0_signals.npz")))
    kreise = b.CITIES["ber"]["kreise"]
    ber["pop"] = germany.zensus_population("ber", None, ber, kreise)[0]
    for name, kw in VARIANTS:
        row = {}
        row["lon_misplaced"] = t3.miss(zone_totals(lon, proxy(lon, nres_variant(lon, **kw)), o_l, zk_l), ref_l)
        row["tor_misplaced"] = t3.miss(zone_totals(tor, proxy(tor, nres_variant(tor, **kw)), o_t, zk_t), ref_t)
        w = proxy(ber, nres_variant(ber, **kw))
        jobs = germany.pendler_jobs("ber", None, ber, w, kreise)[0]
        o = np.argsort(-jobs)
        row.update(ber_top=float(jobs[o[0]]), ber_top_at=(round(float(ber["lat"][o[0]]), 4), round(float(ber["lon"][o[0]]), 4)),
                   ber_2nd=float(jobs[o[1]]), ber_10th=float(jobs[o[9]]))
        for c, sig in (("lon", lon), ("tor", tor), ("ber", ber)):
            nv = nres_variant(sig, **kw)
            row[f"{c}_nres_top_share"] = float(nv.max() / nv.sum())
        out[name] = row
        print(f"{name:>12}: London {row['lon_misplaced']:.4%}  Toronto {row['tor_misplaced']:.4%}  Berlin top {row['ber_top']:,.0f} at {row['ber_top_at']}, "
              f"2nd {row['ber_2nd']:,.0f}, 10th {row['ber_10th']:,.0f}; NRES top share lon {row['lon_nres_top_share']:.4f} tor {row['tor_nres_top_share']:.4f} "
              f"ber {row['ber_nres_top_share']:.4f}", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
