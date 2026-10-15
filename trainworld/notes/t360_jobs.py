"""T-360: tier 0's London jobs (the proxy in the stage a pack) against BRES employment by LSOA, and
the proxy's spread rules scored on London and Toronto. Read-only on the packs.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t360_jobs.py

"Misplaced" is half the L1 distance between two shares (as t287_compare). Toronto's reference is
the real pack's jobs (StatCan place of work) on the tier 0 signals rebuilt in data/work/t287/work.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")

import json
import sys

import numpy as np
from shapely import make_valid, union_all
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
import tier0  # noqa: E402
import enrich_airports as air  # noqa: E402
from read_pack import load_pack  # noqa: E402

OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-360-jobs.json")
BANK = (51.5133, -0.0890)
KING_BAY = (43.6487, -79.3817)


def miss(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    return 0.5 * float(np.abs(a / a.sum() - b_ / b_.sum()).sum())


def zone_keys(lat, lon, origin):
    lon0, lat0 = origin
    x = b.R_EARTH * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = b.R_EARTH * np.radians(lat - lat0)
    return np.floor(x / 2000).astype(np.int64) * 100000 + np.floor(y / 2000).astype(np.int64)


def parts(sig):
    osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
    out = {"osm": osm / osm.sum(), "pop": sig["pop"] / sig["pop"].sum()}
    for hp in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0):
        v = tier0.nres_signal(sig, hp)
        out[f"nres_h{hp:g}"] = v / v.sum()
    return out


def blends(P, score):
    rows = []
    for k in [k for k in P if k.startswith("nres")]:
        for a in np.arange(0, 1.01, 0.1):
            for o in np.arange(0, 1.01 - a, 0.1):
                p = max(0.0, 1 - a - o)
                v = a * P[k] + o * P["osm"] + p * P["pop"]
                rows.append(dict(nres=k, w_nres=round(a, 1), w_osm=round(o, 1), w_pop=round(p, 1), **score(v)))
    return rows


def main():
    res = {}
    # ---------------- London
    sig = dict(np.load(os.path.join(b.WORK, "lon.tier0_signals.npz")))
    with open(england.raw("lon_itl3.geojson"), encoding="utf-8") as f:
        boundary = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    jpr = tier0.worldbank("GB")[0]
    P = float(sig["pop"].sum())
    t0 = tier0.jobs_from(sig, P * jpr)
    bres, year = england.bres_jobs("lon", boundary, sig, tier0.jobs_from(sig, 1.0))
    with open(england.raw("lad_dec2023_uk_bgc.geojson"), encoding="utf-8") as f:
        lads = [(ft["properties"]["LAD23CD"], ft["properties"]["LAD23NM"], make_valid(shape(ft["geometry"]))) for ft in json.load(f)["features"]]
    lads = [x for x in lads if boundary.contains(x[2].representative_point())]
    li, _ = england.cell_units(sig["lat"], sig["lon"], [(c, g) for c, _, g in lads])
    li = np.where(li < 0, 0, li)
    nl = len(lads)
    gl = np.array([c.startswith("E09") for c, _, _ in lads])
    core = np.array([n in ("Westminster", "City of London") for _, n, _ in lads])

    def by_lad(v):
        return np.bincount(li, weights=v, minlength=nl)
    dist = air.metres(BANK[0], BANK[1], sig["lat"], sig["lon"])
    zk = zone_keys(sig["lat"], sig["lon"], b.CITIES["lon"]["origin"])
    _, zi = np.unique(zk, return_inverse=True)
    units = england.lsoas("lon", boundary)
    ui, _ = england.cell_units(sig["lat"], sig["lon"], units)
    ui = np.where(ui < 0, len(units), ui)

    def summary(v, ref):
        vl, rl = by_lad(v), by_lad(ref)
        return dict(total=float(v.sum()), greater_london=float(vl[gl].sum()), westminster_city=float(vl[core].sum()),
                    gl_share=float(vl[gl].sum() / v.sum()), core_share=float(vl[core].sum() / v.sum()),
                    **{f"bank_{k}km": float(v[dist <= k * 1000].sum()) for k in (1, 2, 5, 10)},
                    lad_misplaced=miss(vl, rl), zone_misplaced=miss(np.bincount(zi, weights=v), np.bincount(zi, weights=ref)),
                    lsoa_misplaced=miss(np.bincount(ui, weights=v), np.bincount(ui, weights=ref)))
    res["bres_year"] = year
    res["tier0"] = summary(t0, bres)
    res["bres"] = summary(bres, bres)
    res["pop"] = dict(total=P, greater_london=float(by_lad(sig["pop"])[gl].sum()), jobs_per_resident=jpr)
    lt = [dict(lad=n, bres=float(r), tier0=float(t), err=float(t / max(r, 1) - 1)) for (_, n, _), r, t in zip(lads, by_lad(bres), by_lad(t0))]
    res["lads"] = sorted(lt, key=lambda r: -r["bres"])
    print(json.dumps({k: res[k] for k in ("bres_year", "tier0", "bres", "pop")}, indent=1))
    for r in res["lads"][:12]:
        print(f"  {r['lad']:<28} BRES {r['bres']:>10,.0f}  tier0 {r['tier0']:>10,.0f}  {r['err']:+.0%}")

    Pl = parts(sig)
    bz, bl = np.bincount(zi, weights=bres), by_lad(bres)

    def score_lon(v):
        vl = by_lad(v)
        return dict(lon_zone=miss(np.bincount(zi, weights=v), bz), lon_lad=miss(vl, bl), lon_core=float(vl[core].sum() / bl[core].sum() * bl.sum()),
                    lon_bank2=float(v[dist <= 2000].sum() / bres[dist <= 2000].sum() * bres.sum()))
    rows = blends(Pl, score_lon)

    # ---------------- Toronto, same rules
    tsig = os.path.join(ROOT, "data", "work", "t287", "work", "t0tor.tier0_signals.npz")
    if os.path.exists(tsig):
        import h3
        ts = dict(np.load(tsig))
        h, a = load_pack("tor", os.path.join(ROOT, "data", "packs"))
        ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
        origin = b.CITIES["tor"]["origin"]
        rz, sz = zone_keys(ll[:, 0], ll[:, 1], origin), zone_keys(ts["lat"], ts["lon"], origin)
        keys = np.union1d(rz, sz)
        rjz = np.bincount(np.searchsorted(keys, rz), weights=a["jobs"].astype(float), minlength=len(keys))
        si = np.searchsorted(keys, sz)
        rd = air.metres(KING_BAY[0], KING_BAY[1], ll[:, 0], ll[:, 1])
        sd = air.metres(KING_BAY[0], KING_BAY[1], ts["lat"], ts["lon"])
        r2 = float(a["jobs"][rd <= 2000].sum() / a["jobs"].sum())
        Pt = parts(ts)
        trows = blends(Pt, lambda v: dict(tor_zone=miss(np.bincount(si, weights=v, minlength=len(keys)), rjz),
                                          tor_core2=float(v[sd <= 2000].sum() / r2)))
        for r, t in zip(rows, trows):
            r.update({k: v for k, v in t.items() if k.startswith("tor")})
    for r in rows:
        r["both_zone"] = r["lon_zone"] + r.get("tor_zone", 0.0)
    res["blends"] = rows

    # ---------------- concentration: the blend's density to a power (cells are near equal area)
    grows = []
    for k, a_, o_ in (("nres_h1.5", 0.5, 0.3), ("nres_h2", 0.3, 0.5), ("nres_h1.5", 0.3, 0.5), ("nres_h0", 0.3, 0.5)):
        for g in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5):
            r = dict(nres=k, w_nres=a_, w_osm=o_, gamma=g)
            v = (a_ * Pl[k] + o_ * Pl["osm"] + (1 - a_ - o_) * Pl["pop"]) ** g
            r.update(score_lon(v / v.sum()))
            if os.path.exists(tsig):
                vt = (a_ * Pt[k] + o_ * Pt["osm"] + (1 - a_ - o_) * Pt["pop"]) ** g
                vt /= vt.sum()
                r.update(tor_zone=miss(np.bincount(si, weights=vt, minlength=len(keys)), rjz), tor_core2=float(vt[sd <= 2000].sum() / r2))
            grows.append(r)
    res["gamma"] = grows

    def show(title, rs):
        print(title)
        for r in rs:
            print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    show("concentration", grows)
    cur = [r for r in rows if r["nres"] == "nres_h1.5" and r["w_nres"] == 0.5 and r["w_osm"] == 0.3]
    show("current rule (height^1.5, 0.5/0.3/0.2)", cur)
    show("best on London zones", sorted(rows, key=lambda r: r["lon_zone"])[:8])
    if "tor_zone" in rows[0]:
        show("best on Toronto zones", sorted(rows, key=lambda r: r["tor_zone"])[:5])
        show("best on both", sorted(rows, key=lambda r: r["both_zone"])[:8])
        for k in [k for k in Pl if k.startswith("nres")]:
            show(f"best for {k}", sorted([r for r in rows if r["nres"] == k], key=lambda r: r["both_zone"])[:1])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
