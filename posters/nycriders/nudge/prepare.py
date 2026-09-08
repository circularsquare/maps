"""
Export everything the nudge editor needs, so the browser never has to redo the
hard half of ribbons.py.

The editor's job is only geometry: move a node, watch the stripes follow. The
things that are genuinely fiddly — recovering shared track from the GTFS
vertices, orienting the runs consistently, stacking the fan, solving the bundle
centres — all depend on the route data and not at all on where a node sits, so
they run once here and get baked into the export as plain numbers. What the page
re-does per frame is the small part: offset a polyline, ramp it at the ends,
mitre the joints. That keeps one implementation of the hard part rather than two
that can drift.

The node positions exported are POST-smoothing, which is what the sheet actually
draws, so a nudge in the editor is a nudge of the final position and needs no
mental correction.

    python nudge/prepare.py
    python nudge/serve.py          # then open http://127.0.0.1:8799/nudge/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import frame as fr          # noqa: E402
import palette              # noqa: E402
import chains               # noqa: E402
import defaults as D        # noqa: E402
import ribbons              # noqa: E402
import render as rd         # noqa: E402

BUILD = HERE.parent / "build"
STATS = HERE.parent.parent.parent / "riders" / "nycriders" / "stats.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=D.SHEET)
    ap.add_argument("--pad-km", type=float, default=D.PAD_KM)
    ap.add_argument("--min-width", type=float, default=D.MIN_WIDTH)
    ap.add_argument("--max-width", type=float, default=D.MAX_WIDTH)
    ap.add_argument("--width-gamma", type=float, default=D.WIDTH_GAMMA)
    ap.add_argument("--gap", type=float, default=D.GAP)
    ap.add_argument("--centre-reg", type=float, default=D.CENTRE_REG)
    ap.add_argument("--smooth-passes", type=int, default=D.SMOOTH_PASSES)
    ap.add_argument("--smooth-cap", type=float, default=D.SMOOTH_CAP)
    ap.add_argument("--bubble-min", type=float, default=D.BUBBLE_MIN)
    ap.add_argument("--bubble-max", type=float, default=D.BUBBLE_MAX)
    ap.add_argument("--base-scale", type=float, default=0.25,
                    help="resolution of the backdrop PNG, as a fraction of the sheet")
    ap.add_argument("--theme", default="light", choices=["dark", "light"])
    args = ap.parse_args()

    BUILD.mkdir(exist_ok=True)
    data, feats = ribbons.load_features(STATS)
    runs, node_runs, graph = ribbons.build_runs(feats)

    lon = np.array([c[0] for f in feats.values() for c in f["coords"]])
    lat = np.array([c[1] for f in feats.values() for c in f["coords"]])
    w_in, h_in = (float(v) for v in args.sheet.lower().split("x"))
    f = fr.fit_sheet(lon, lat, 0.0, w_in, h_in, pad_km=args.pad_km)

    vmax = max(v["value"] for v in feats.values())
    lo, hi = args.min_width, args.max_width
    widths_m = {k: (0.0 if v["value"] < 0.5
                    else lo + (hi - lo) * (v["value"] / vmax) ** args.width_gamma) * f.m_per_inch
                for k, v in feats.items()}
    rel = ribbons.fan_relative(runs, widths_m, args.gap * hi * f.m_per_inch)
    centres = ribbons.solve_centres(runs, node_runs, rel, reg=args.centre_reg)
    offs = ribbons.absolute(rel, centres)

    # node positions, smoothed exactly as render.py smooths them
    keys = list(graph["coord"].keys())
    ll = np.array([graph["coord"][k] for k in keys], float)
    px, py = fr.project(ll[:, 0], ll[:, 1], 0.0)
    xy_node = {k: np.array([px[i], py[i]]) for i, k in enumerate(keys)}
    if args.smooth_passes > 0:
        xy_node = ribbons.smooth_nodes(
            xy_node, graph, ribbons.node_half_widths(runs, offs),
            passes=args.smooth_passes, cap=args.smooth_cap)

    xy = [np.array([xy_node[n] for n in run["nodes"]]) for run in runs]
    joints = ribbons.joint_normals(runs, node_runs, xy)
    nudges = ribbons.load_nudges(HERE / "nudges.json")
    forced, dropped = ribbons.nudge_sets(nudges)
    cs = chains.build(runs, offs, xy, joints, forced=forced, dropped=dropped)

    st = ribbons.station_totals(data, None)
    sval = np.array([s[2] for s in st])
    sr = (args.bubble_min + (args.bubble_max - args.bubble_min)
          * np.sqrt(sval / sval.max())) * f.m_per_inch
    sx, sy = fr.project(np.array([s[0] for s in st]), np.array([s[1] for s in st]), 0.0)
    stations = [[round(float(sx[i]), 1), round(float(sy[i]), 1), round(float(sr[i]), 1)]
                for i in range(len(st))]

    # a backdrop so it is possible to tell where you are
    base_scale = args.base_scale
    W, H = f.size_px(base_scale)
    rd.render_base(f, W, H, args.theme, BUILD / "nudge_base.png")

    payload = {
        "frame": f.to_dict(),
        "theme": args.theme,
        "base": {"png": "../build/nudge_base.png", "w": W, "h": H},
        "chains": cs,
        "stations": stations,
        "spline": {"samples": chains.SPLINE_SAMPLES},
    }
    out = BUILD / "nudge_data.json"
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    mb = out.stat().st_size / 1e6
    ctrl = sum(len(c["ctrl"]) for c in cs)
    dense = sum(len(c["keys"]) for c in cs)
    print(f"{out.name}: {len(cs):,} route chains, {ctrl:,} control points "
          f"of {dense:,} vertices, {mb:.1f} MB")
    print(f"backdrop {W} x {H}")
    report = ribbons.edit_report(nudges, cs)
    print(report if report else f"hand edits: none yet ({HERE / 'nudges.json'})")


if __name__ == "__main__":
    main()
