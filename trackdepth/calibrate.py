"""Calibrate the model's error bars with held-out anchors.

Anchors (elevator-chain platforms from station_levels.csv, and groups of documented depths from
depths.csv) are split into 2 folds by a stable hash of their id. For each fold the model is built
with the other fold's anchors and scored on this fold's: the model's rail height (or depth below
street) there against the anchor, error / sqrt(model sd^2 + anchor sd^2). The pooled result says
how often an anchor that the model had not seen fell inside 1 and 2 model sds (68% / 95% if the
error bars are honest) and whether the model runs too deep or too shallow (median signed error).

    python calibrate.py                        # current constants in build.py
    python calibrate.py --tl "-1:6.5:3,-2:10:6"  # try other TUNNEL_LEVEL entries (level:mean:sd)
    python calibrate.py --corr 1000            # try another CORR
    python calibrate.py --docs-only            # fold only the document anchors; elevators stay in

Rows that an elevator anchor covers are never scored as documents (the elevator is the better
measurement and is scored itself), and when that elevator anchor is held out the document row is
not promoted to an anchor either, so a held-out truth never leaks back in. Scratch work only;
nothing here writes data/.
"""
import argparse
import collections
import hashlib
import math

import numpy as np

import build
import validate


def fold_of(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 2


def elev_key(an):
    return f"E{an['station_mrn']}|{an['line']}|{an['level_group']}"


def mode_level(ways, way_of, idx):
    c = collections.Counter(build.level_of(ways[way_of[i]]["tags"]) for i in idx)
    return c.most_common(1)[0][0]


MIN_N = 8   # anchors a level needs before its mean and sd are re-fitted


def unit_table(ctx):
    """Every anchor unit with its OSM level and depth below street, from one solve with all of them."""
    ways, xs, ys, way_of, seqs, ground, gsm = ctx
    *_, info = build.solve(ways, xs, ys, way_of, seqs, ground, gsm, nsamples=2, quiet=True)
    units = []
    for e in info["elev"]:
        idx = e["idx"]
        units.append((elev_key(e), mode_level(ways, way_of, idx), float(np.nanmean(gsm[idx])) - e["rail"]))
    for g in info["doc_groups"]:
        units.append((g["gid"], g["level"], g["depth"]))
    return units


def fit_levels(units, defaults):
    """Prior mean := median depth of the anchors at that level, sd := their standard deviation."""
    out = dict(defaults)
    for lv in defaults:
        d = np.array([u[2] for u in units if u[1] == lv])
        if len(d) >= MIN_N:
            out[lv] = (round(float(np.median(d)), 1), round(float(d.std()), 1))
    return out


def run_fold(f, ctx, nsamples, docs_only):
    ways, xs, ys, way_of, seqs, ground, gsm = ctx
    z, sd, pmu, psd, cls, used, info = build.solve(
        ways, xs, ys, way_of, seqs, ground, gsm,
        doc_ok=lambda gid: fold_of(gid) != f,
        elev_ok=None if docs_only else (lambda an: fold_of(elev_key(an)) != f),
        nsamples=nsamples, quiet=True)
    depth = gsm - z
    res = []   # (kind, name, level, truth depth, model depth, model sd, anchor sd, z)
    for e in info["elev"]:
        if not e["held_out"]:
            continue
        idx = e["idx"]
        zm = float(np.mean(z[idx])); msd = float(np.mean(sd[idx]))
        err_depth = -(zm - e["rail"])           # model depth minus anchor depth
        g = float(np.nanmean(gsm[idx]))
        res.append(("elevator", f"{e['station']} ({e['line']})", mode_level(ways, way_of, idx),
                    g - e["rail"], g - zm, msd, e["sd_used"], err_depth / math.hypot(msd, e["sd_used"])))
    for g in info["doc_groups"]:
        if not g["held_out"]:
            continue
        for row in g["rows"]:
            r = validate.compare(row, g["idx"], xs, ys, depth, sd)
            if r is None:
                continue
            model, msd, err, zz, _ = r
            res.append(("document", f"{row['id']} {row['place'][:40]}", g["level"], row["depth"], model, msd,
                        row["sd"], zz))
    return res, info


def summarize(res, label):
    if not res:
        print(f"{label}: none")
        return
    zs = np.array([r[7] for r in res]); err = np.array([r[4] - r[3] for r in res])
    print(f"{label}: n={len(res)}  inside 1 sd {np.mean(np.abs(zs) <= 1):.0%}  inside 2 sd "
          f"{np.mean(np.abs(zs) <= 2):.0%}  median signed error {np.median(err):+.1f} m "
          f"(median |z| {np.median(np.abs(zs)):.2f}, rms z {np.sqrt(np.mean(zs ** 2)):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", help="TUNNEL_LEVEL overrides, level:mean:sd,...")
    ap.add_argument("--corr", type=float)
    ap.add_argument("--sdscale", type=float, help="multiply every TUNNEL_LEVEL sd (after --tl)")
    ap.add_argument("--elev-sd", type=float, help="sd of the elevated-track prior (m)")
    ap.add_argument("--refit", action="store_true",
                    help="per fold, fit TUNNEL_LEVEL means and sds to the other fold's anchors")
    ap.add_argument("--nsamples", type=int, default=40)
    ap.add_argument("--docs-only", action="store_true")
    ap.add_argument("--rows", action="store_true", help="print every held-out unit")
    a = ap.parse_args()
    if a.tl:
        for item in a.tl.split(","):
            lv, mu, s = item.split(":")
            build.TUNNEL_LEVEL[int(lv)] = (float(mu), float(s))
    if a.sdscale:
        build.TUNNEL_LEVEL = {k: (m, s * a.sdscale) for k, (m, s) in build.TUNNEL_LEVEL.items()}
    if a.elev_sd:
        build.CLASS["elevated"] = (build.CLASS["elevated"][0], a.elev_sd)
    if a.corr:
        build.CORR = a.corr
    print("TUNNEL_LEVEL", build.TUNNEL_LEVEL, "CORR", build.CORR, flush=True)

    ways = build.load_tracks()
    xs, ys, way_of, seqs = build.build_points(ways)
    ground = build.sample_ground(xs, ys)
    gsm = build.smooth_along(seqs, ground)
    ctx = (ways, xs, ys, way_of, seqs, ground, gsm)

    allres = []
    defaults = dict(build.TUNNEL_LEVEL)
    units = unit_table(ctx) if a.refit else None
    for f in (0, 1):
        if a.refit:
            # constants fitted on the other fold's anchors only, so the scoring stays held out
            build.TUNNEL_LEVEL = fit_levels([u for u in units if fold_of(u[0]) != f], defaults)
            print(f"fold {f} fitted on the other fold:", build.TUNNEL_LEVEL, flush=True)
        res, info = run_fold(f, ctx, a.nsamples, a.docs_only)
        print(f"fold {f}: {sum(1 for r in res if r[0] == 'elevator')} elevator, "
              f"{sum(1 for r in res if r[0] == 'document')} document rows held out", flush=True)
        allres += res
    if a.rows:
        for r in sorted(allres, key=lambda r: (r[2] if r[2] is not None else 0, r[1])):
            print(f"  {r[0]:8} {r[1][:62]:62} lvl {r[2]!s:>4} truth {r[3]:6.1f} model {r[4]:6.1f} "
                  f"sd {r[5]:4.1f} +-{r[6]:3.1f} z {r[7]:5.1f}")
    if a.refit:
        print("fitted on all anchors:", fit_levels(units, defaults))
    summarize(allres, "all held out")
    summarize([r for r in allres if r[0] == "elevator"], "  elevator anchors")
    summarize([r for r in allres if r[0] == "document"], "  document rows")
    for lv in sorted({r[2] for r in allres if r[2] is not None}):
        sub = [r for r in allres if r[2] == lv]
        summarize(sub, f"  OSM level {lv}")
        t = np.array([r[3] for r in sub])
        print(f"      anchor depths: median {np.median(t):.1f}, mean {t.mean():.1f}, sd {t.std():.1f}, "
              f"range {t.min():.1f} to {t.max():.1f}")


if __name__ == "__main__":
    main()
