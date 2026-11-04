"""Compare the model (data/tracks.json) with documented depths (evidence/depths.csv).

Uses rows with coordinates whose measure is a depth below or height above the street. Rows are
placed on OSM track by build.place_doc_rows(): the row's line, the right kind of track, and where
that line has several OSM `level` values at the place, the level picked by order (lower or express
-> lowest level). Rows that cannot be placed are listed, not guessed.

Prints one line per row (with the OSM level it matched and whether the build used it as an anchor)
and the share inside 1 and 2 sd. z = (model - document) / sqrt(model sd^2 + document sd^2), where
the document sd is its confidence class plus half of any quoted range; the strict version leaves
out the confidence class (only the quoted range, 0.5 m if none), as this script used to.
Rows the build anchored on (document or elevator) are in-sample checks; held-out coverage is
calibrate.py's job.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd

import build

ROOT = pathlib.Path(__file__).parent


def load_points():
    tracks = json.loads((ROOT / "data" / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    xs, ys, names, levels, clss, services, depth, sd = [], [], [], [], [], [], [], []
    for t in tracks:
        x, y = build.to_utm.transform(t["lon"], t["lat"])
        for i in range(len(x)):
            xs.append(x[i]); ys.append(y[i]); names.append(t["name"]); levels.append(t["level"])
            clss.append(t["cls"]); services.append(bool(t["service"]))
            g = t["ground"][i]
            depth.append(np.nan if g is None else g - t["z"][i]); sd.append(t["sd"][i])
    return (np.array(xs), np.array(ys), names, levels, clss, services, np.array(depth), np.array(sd))


def compare(row, idx, xs, ys, depth, sd):
    """Model depth and sd at the matched point nearest the row's coordinates, and the z scores."""
    ax, ay = build.to_utm.transform(row["lon"], row["lat"])
    idx = [i for i in idx if not np.isnan(depth[i])]
    if not idx:
        return None
    i = min(idx, key=lambda j: math.hypot(xs[j] - ax, ys[j] - ay))
    err = depth[i] - row["depth"]
    z = err / math.hypot(sd[i], row["sd"])
    z_strict = err / math.hypot(sd[i], row["half"] or 0.5)
    return depth[i], sd[i], err, z, z_strict


def coverage(zs, label):
    zs = np.array(zs)
    if not len(zs):
        return
    print(f"{label}: {len(zs)} rows; inside 1 sd {np.mean(np.abs(zs) <= 1):.0%}, "
          f"inside 2 sd {np.mean(np.abs(zs) <= 2):.0%}; median |z| {np.median(np.abs(zs)):.1f}")


def main():
    xs, ys, names, levels, clss, services, depth, sd = load_points()
    docs = build.doc_rows()
    placed = build.place_doc_rows(docs, xs, ys, names, levels, clss, services)
    status = {}
    p = ROOT / "data" / "doc_rows.csv"
    if p.exists():
        status = dict(pd.read_csv(p).set_index("id")["status"])
    print(f"{'id':7} {'place':38} {'doc':>6} {'model':>6} {'sd':>5} {'err':>6} {'z':>5} {'lvl':>4} {'use':8} conf")
    matched, skipped = [], []
    for row, (idx, lvl, why) in zip(docs, placed):
        res = compare(row, idx, xs, ys, depth, sd) if not why and idx else None
        if res is None:
            skipped.append((row, why or "no ground under the matched points"))
            continue
        model, msd, err, z, zs = res
        use = status.get(row["id"], "?")
        matched.append((row, z, zs, err, use, lvl))
        print(f"{row['id']:7} {row['place'][:38]:38} {row['depth']:6.1f} {model:6.1f} {msd:5.1f} {err:6.1f} "
              f"{z:5.1f} {'' if lvl is None else int(lvl):>4} {use:8} {row['conf']}")
    print(f"\n{len(skipped)} rows not placed:")
    for row, why in skipped:
        print(f"  {row['id']:7} {row['place'][:48]:48} {why}")
    print()
    coverage([m[1] for m in matched], "all placed rows")
    coverage([m[2] for m in matched], "  strict (document sd = range only)")
    for u in ("anchor", "elevator"):
        coverage([m[1] for m in matched if m[4] == u], f"in-sample, {u}")
    coverage([m[1] for m in matched if m[4] not in ("anchor", "elevator")], "not used as anchor")
    errs = np.array([m[3] for m in matched])
    print(f"median signed error (model - document) {np.median(errs):+.1f} m")


if __name__ == "__main__":
    main()
