"""T-287: tier 0 Toronto (data/work/t287/packs/t0tor, built by notes/t287_build.py) against the
real pack (data/packs/tor, StatCan census inputs) and the census itself. Read-only on both packs.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t287_compare.py [--blends]

Everything is aggregated the same way for both packs: a cell belongs to the census subdivision
(municipality) or division holding its centre. Commute flows: the game's gravity solved in numpy on
each pack's own 2 km zones (tier0.gravity), then cell shares to municipality pairs, set against the
2021 census flows between the 28 municipalities (98-10-0459-01, both ends inside). "Misplaced" is
half the L1 distance between two shares: the part of the total that would have to move.
--blends also scores jobs-proxy blends from the tier 0 signals (in-sample on Toronto).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")

import json
import sys

import h3
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree, make_valid
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import canada  # noqa: E402
import tier0  # noqa: E402
from read_pack import load_pack  # noqa: E402

SCRATCH = os.path.join(ROOT, "data", "work", "t287")
OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-287-compare.json")
KING_BAY = (43.6487, -79.3817)
CURVES = {"Toronto's own fit, d^-1.2 exp(-d/20)": (1.2, 20.0), "US, d^-0.8 exp(-d/30)": (0.8, 30.0)}


def miss(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    return 0.5 * float(np.abs(a / a.sum() - b_ / b_.sum()).sum())


class Geo:
    def __init__(self):
        cds = set(b.CITIES["tor"]["counties"])
        feats = [f for f in canada.csd_features() if f["properties"]["CSDUID"][:4] in cds]
        self.ids = [f["properties"]["CSDUID"] for f in feats]
        self.names = {f["properties"]["CSDUID"]: f["properties"]["CSDNAME"] for f in feats}
        self.tree = STRtree([make_valid(shape(f["geometry"])) for f in feats])

    def of(self, lat, lon):
        """Municipality index per point (-1 outside every one: the nearest)."""
        pts = shapely.points(lon, lat)
        out = np.full(len(pts), -1)
        pi, gi = self.tree.query(pts, predicate="within")
        out[pi] = gi
        lost = np.nonzero(out < 0)[0]
        if len(lost):
            out[lost] = self.tree.nearest(pts[lost])
        return out


def pack(city, packs):
    h, a = load_pack(city, packs)
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    return h, a, ll


def flows_table(geo, a, csd, census, curves):
    """Municipality-pair shares from the gravity on the pack's 2 km zones, for each curve."""
    zone, gx, gy = tier0.zones_of(a["x_m"].astype(float), a["y_m"].astype(float))
    nz, nc = len(gx), len(geo.ids)
    home = np.bincount(zone, weights=a["commute_home"].astype(float), minlength=nz)
    work = np.bincount(zone, weights=a["commute_work"].astype(float), minlength=nz)
    P = np.zeros((nz, nc)); Q = np.zeros((nz, nc))
    np.add.at(P, (zone, csd), a["commute_home"].astype(float))
    np.add.at(Q, (zone, csd), a["commute_work"].astype(float))
    P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
    d = tier0.zone_distances(gx, gy)
    out = {}
    for name, (pw, km) in curves.items():
        T = tier0.gravity(home, work, d, pw, km)
        C = P.T @ T.astype(np.float64) @ Q
        C /= C.sum()
        mean, med = tier0.trip_stats(T, d)
        own_m, own_c = np.diag(C) / C.sum(1), np.diag(census) / census.sum(1)
        to_tor = geo.ids.index("3520005")
        out[name] = dict(misplaced=0.5 * float(np.abs(C - census).sum()), own=float(np.trace(C)), mean_km=mean, median_km=med,
                         to_toronto={geo.names[geo.ids[i]]: [round(float(C[i, to_tor] / C[i].sum()), 3), round(float(census[i, to_tor] / census[i].sum()), 3)]
                                     for i in np.argsort(-census.sum(1))[:6]})
    return out


def main(args):
    geo = Geo()
    rh, ra, rll = pack("tor", os.path.join(ROOT, "data", "packs"))
    th, ta, tll = pack("t0tor", os.path.join(SCRATCH, "packs"))
    rc, tc = geo.of(rll[:, 0], rll[:, 1]), geo.of(tll[:, 0], tll[:, 1])
    nc = len(geo.ids)
    res = {}

    def by(csd, v):
        return np.bincount(csd, weights=np.asarray(v, float), minlength=nc)

    # census reference figures
    fl = canada.csd_flows(b.CITIES["tor"])
    ci = {c: i for i, c in enumerate(geo.ids)}
    census_jobs = np.array([fl[fl.w == c].n.sum() for c in geo.ids], float)
    f2 = fl[fl.h.isin(ci) & fl.w.isin(ci)]
    O = np.zeros((nc, nc))
    np.add.at(O, (f2.h.map(ci).to_numpy(), f2.w.map(ci).to_numpy()), f2.n.to_numpy(float))
    O /= O.sum()

    # ---------------- population and jobs by municipality
    rows = []
    rp, tp = by(rc, ra["pop"]), by(tc, ta["pop"])
    rj, tj = by(rc, ra["jobs"]), by(tc, ta["jobs"])
    sig = dict(np.load(os.path.join(SCRATCH, "work", "t0tor.tier0_signals.npz")))
    sc = geo.of(sig["lat"], sig["lon"])
    gp = by(sc, sig["pop_ghsl"])
    for i, c in enumerate(geo.ids):
        rows.append(dict(csd=geo.names[c], cd=c[:4], pop_real=rp[i], pop_t0=tp[i], pop_ghsl_unscaled=gp[i],
                         jobs_real=rj[i], jobs_t0=tj[i], census_usual_place=census_jobs[i]))
    t = pd.DataFrame(rows).sort_values("pop_real", ascending=False)
    t["pop_err"] = t.pop_t0 / t.pop_real - 1
    t["jobs_err"] = t.jobs_t0 / t.jobs_real - 1
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"))
    cdt = t.groupby("cd")[["pop_real", "pop_t0", "pop_ghsl_unscaled", "jobs_real", "jobs_t0", "census_usual_place"]].sum()
    print(cdt.to_string(float_format=lambda v: f"{v:,.0f}"))
    w = t.pop_real / t.pop_real.sum()
    res["pop"] = dict(total_real=float(rp.sum()), total_t0=float(tp.sum()), total_ghsl_unscaled=float(gp.sum()),
                      csd_misplaced=miss(tp, rp), csd_misplaced_unscaled=miss(gp, rp), cd_misplaced_unscaled=miss(cdt.pop_ghsl_unscaled, cdt.pop_real),
                      csd_mean_abs_err_popweighted=float((w * t.pop_err.abs()).sum()), worst=t.loc[t.pop_err.abs().idxmax(), ["csd", "pop_err"]].tolist())
    wj = t.jobs_real / t.jobs_real.sum()
    res["jobs"] = dict(total_real=float(rj.sum()), total_t0=float(tj.sum()), census_usual_place=float(census_jobs.sum()),
                       csd_misplaced_vs_pack=miss(tj, rj), csd_misplaced_vs_census=miss(tj, census_jobs), real_pack_vs_census=miss(rj, census_jobs),
                       csd_mean_abs_err_jobsweighted=float((wj * t.jobs_err.abs()).sum()), table=t.to_dict("records"))
    # 2 km zones (one grid for both: the real pack's frame and origin are the same)
    def zones(a):
        return (np.floor(a["x_m"] / 2000).astype(np.int64) * 100000 + np.floor(a["y_m"] / 2000).astype(np.int64))
    zr, zt = zones(ra), zones(ta)
    keys = np.union1d(zr, zt)
    zjr = np.bincount(np.searchsorted(keys, zr), weights=ra["jobs"].astype(float), minlength=len(keys))
    zjt = np.bincount(np.searchsorted(keys, zt), weights=ta["jobs"].astype(float), minlength=len(keys))
    zpr = np.bincount(np.searchsorted(keys, zr), weights=ra["pop"].astype(float), minlength=len(keys))
    zpt = np.bincount(np.searchsorted(keys, zt), weights=ta["pop"].astype(float), minlength=len(keys))
    res["jobs"]["zone2km_misplaced"] = miss(zjt, zjr)
    res["pop"]["zone2km_misplaced"] = miss(zpt, zpr)
    # near the centre
    def near(a, ll, km):
        import enrich_airports as air
        dist = air.metres(KING_BAY[0], KING_BAY[1], ll[:, 0], ll[:, 1])
        return float(a["jobs"][dist <= km * 1000].sum()), float(a["pop"][dist <= km * 1000].sum())
    res["centre"] = {f"{k} km": dict(jobs_real=near(ra, rll, k)[0], jobs_t0=near(ta, tll, k)[0]) for k in (1, 2, 3, 5, 10)}
    print("jobs near King and Bay:", json.dumps(res["centre"]))
    print("pop:", json.dumps({k: v for k, v in res["pop"].items()}))
    print("jobs:", json.dumps({k: v for k, v in res["jobs"].items() if k != "table"}))

    # ---------------- jobs-proxy blends (in-sample)
    if "--blends" in args:
        # the signals' cells to the real pack's 2 km zones and municipalities
        lon0, lat0 = b.CITIES["tor"]["origin"]
        sx = b.R_EARTH * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0))
        sy = b.R_EARTH * np.radians(sig["lat"] - lat0)
        zs = np.floor(sx / 2000).astype(np.int64) * 100000 + np.floor(sy / 2000).astype(np.int64)
        keys2 = np.union1d(keys, zs)
        zr2 = np.bincount(np.searchsorted(keys2, zr), weights=ra["jobs"].astype(float), minlength=len(keys2))
        zi = np.searchsorted(keys2, zs)
        osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
        parts = {"nres_h": tier0.nres_signal(sig), "nres_v": sig["nres_v"], "nres_s": sig["nres_s"], "osm": osm, "pop": sig["pop"]}
        bl = []
        for n_kind in ("nres_h", "nres_v", "nres_s"):
            for a_ in np.arange(0, 1.01, 0.1):
                for o_ in np.arange(0, 1.01 - a_, 0.1):
                    p_ = 1 - a_ - o_
                    v = a_ * parts[n_kind] / parts[n_kind].sum() + o_ * parts["osm"] / parts["osm"].sum() + p_ * parts["pop"] / parts["pop"].sum()
                    bl.append(dict(nres=n_kind, w_nres=round(a_, 1), w_osm=round(o_, 1), w_pop=round(p_, 1),
                                   zone=miss(np.bincount(zi, weights=v, minlength=len(keys2)), zr2), csd=miss(by(sc, v), rj),
                                   csd_census=miss(by(sc, v), census_jobs),
                                   centre3=float(v[tier0_near(sig, 3)].sum() / v.sum() * rj.sum())))
        bt = pd.DataFrame(bl).sort_values("zone")
        print(bt.head(15).to_string(index=False))
        for name, q in (("nres_v only", (bt.nres == "nres_v") & (bt.w_nres == 1)), ("nres_s only", (bt.nres == "nres_s") & (bt.w_nres == 1)),
                        ("osm only", (bt.w_osm == 1) & (bt.nres == "nres_v")), ("pop only", (bt.w_pop == 1) & (bt.nres == "nres_v")),
                        ("nres_h only", (bt.nres == "nres_h") & (bt.w_nres == 1)),
                        ("0.5/0.3/0.2 V", (bt.nres == "nres_v") & (bt.w_nres == 0.5) & (bt.w_osm == 0.3)),
                        ("default 0.5/0.3/0.2 height-weighted", (bt.nres == "nres_h") & (bt.w_nres == 0.5) & (bt.w_osm == 0.3))):
            print(name, bt[q].to_dict("records"))
        res["blends"] = bt.to_dict("records")

    # ---------------- commute flows
    curves = dict(CURVES)
    g = th["gravity"]
    curves[f"tier 0 fitted to the 10.5 km median, d^-{g.get('decay_pow', 1.0)} exp(-d/{g.get('decay_km')})"] = (float(g.get("decay_pow", 1.0)), float(g.get("decay_km")))
    res["flows_real_inputs"] = flows_table(geo, ra, rc, O, curves)
    res["flows_t0_inputs"] = flows_table(geo, ta, tc, O, curves)
    for k in ("flows_real_inputs", "flows_t0_inputs"):
        for name, r in res[k].items():
            print(f"{k} {name}: misplaced {r['misplaced']:.1%}, own {r['own']:.1%} (census {np.trace(O):.1%}), mean {r['mean_km']:.1f} km, "
                  f"median {r['median_km']:.1f} km; to Toronto (model, census) {r['to_toronto']}")
    hc = by(rc, ra["commute_home"]), by(tc, ta["commute_home"])
    res["home_end_csd_misplaced"] = miss(hc[1], hc[0])
    print(f"home end by municipality misplaced {res['home_end_csd_misplaced']:.1%}")

    # ---------------- other purposes
    oth = {}
    for name in ("school_enrollment", "university_enrollment", "airport_trips", "errand_jobs", "leisure_jobs", "medical_visits", "park_draw"):
        if name in ra and name in ta:
            oth[name] = dict(real=float(ra[name].sum()), t0=float(ta[name].sum()),
                             csd_misplaced=miss(by(tc, ta[name]), by(rc, ra[name])) if ra[name].sum() > 0 and ta[name].sum() > 0 else None)
    res["other"] = oth
    print(json.dumps(oth, indent=1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)


def tier0_near(sig, km):
    import enrich_airports as air
    return air.metres(KING_BAY[0], KING_BAY[1], sig["lat"], sig["lon"]) <= km * 1000


if __name__ == "__main__":
    main(sys.argv[1:])
