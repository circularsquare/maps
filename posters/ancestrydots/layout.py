"""
Plan where the insets sit on the ancestrydots poster.

Insets are packed into rails along the four edges, in the style of a 19th-c
atlas plate: a column down each side, a row along the top and part of the
bottom. Free placement was tried first and rejected — it scattered boxes over
sparse-but-interesting country, and sparse areas are exactly where a dot map is
worth reading.

Each rail is rigid: the order within it is fixed, boxes are flush to their
edge, and the single free parameter is the rail's offset along its own axis.
That offset is chosen by scoring against a summed-area table of dot density,
so each rail slides to wherever it hides the fewest dots.

The left, right and bottom rails land on open ocean. The top rail cannot —
there is populated Canada up there — so it is the only one that covers dots,
and its offset is chosen to minimise the damage.

    python layout.py
    python layout.py --extend-left 3 --trim-right 3
    python layout.py --alaska-scale 0.26

Writes build/layout_plan.png and build/layout.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

from dotraster import splat, over
from insets import CITIES, KM_PER_IN_MAIN

HERE = Path(__file__).parent
BUILD = HERE / "build"

MAIN = CRS.from_proj4("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
                      "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
LAT_S, LAT_N, LON_W, LON_E = 24.55, 53.55, -125.24, -59.75

# Alaska, Hawaii, Puerto Rico and Newfoundland live in insets.CITIES alongside
# the city windows, each with its own magnification in insets.MAG, so sizes are
# defined in exactly one place.

# Each rail is a list of ROWS. Side rails stack their rows down the edge; the
# top and bottom rails are a single row. Two columns on the right shortens that
# rail from 16.1 in to about 11, which is what pays for trimming the Atlantic.
RAILS = [
    ("left",   "v", [["alaska"], ["seattle"], ["sf"], ["la"], ["honolulu"],
                     ["hawaii"]]),
    ("right",  "v", [["newfoundland"], ["boston", "nyc"], ["philly", "dc"],
                     ["atlanta", "miami"], ["pr"]]),
    ("top",    "h", [["chicago", "detroit", "toronto", "montreal"]]),
    ("bottom", "h", [["dallas", "houston"]]),
]

CELLS_PER_IN = 20
STEP_IN = 0.1
GAP_IN = 0.15


def extent(tf):
    lons = np.linspace(LON_W, LON_E, 400)
    lats = np.linspace(LAT_S, LAT_N, 400)
    x, y = tf.transform(
        np.concatenate([lons, lons, np.full(400, LON_W), np.full(400, LON_E)]),
        np.concatenate([np.full(400, LAT_S), np.full(400, LAT_N), lats, lats]))
    return x.min(), x.max(), y.min(), y.max()


def box_sum(sat, r0, c0, r1, c1):
    return sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-in", type=float, default=36.0)
    # Tightened once the right rail went to two columns. 1 in of Pacific is all
    # the left rail needs, and drops Haida Gwaii. 3 in comes off the Atlantic;
    # 4 in is too far — it pushes the rail's inner column onto the New England
    # coast and costs 10,496 dots. Losing St John's here is deliberate:
    # Newfoundland gets its own 1x inset instead.
    ap.add_argument("--extend-left", type=float, default=1.0)
    ap.add_argument("--trim-right", type=float, default=3.0)
    ap.add_argument("--band", type=float, default=3.87)
    ap.add_argument("--margin", type=float, default=0.25,
                    help="inset inset from the sheet edge, inches")
    ap.add_argument("--pull", type=float, default=25.0,
                    help="cost per grid cell of a rail sitting away from the "
                         "mean position of the cities it shows")
    ap.add_argument("--plan-dpi", type=int, default=60)
    args = ap.parse_args()

    tf = Transformer.from_crs("EPSG:4326", MAIN, always_xy=True)
    x0, x1, y0, y1 = extent(tf)
    m_per_in = (x1 - x0) / args.width_in
    x0 -= args.extend_left * m_per_in
    x1 -= args.trim_right * m_per_in
    map_w_in, map_h_in = (x1 - x0) / m_per_in, (y1 - y0) / m_per_in
    sheet_h_in = map_h_in + args.band
    print(f"map   {map_w_in:.2f} x {map_h_in:.2f} in  ({KM_PER_IN_MAIN:.0f} km/inch)")
    print(f"sheet {map_w_in:.2f} x {sheet_h_in:.2f} in  (band {args.band} in)\n")

    LANDMARKS = {
        "Key West FL": (24.55, -81.78), "Brownsville TX": (25.90, -97.50),
        "Edmonton AB": (53.55, -113.49), "Campbell River BC": (50.03, -125.24),
        "Cape Breton NS": (46.90, -59.75),
        "Cape Scott BC": (50.78, -128.39), "San Diego CA": (32.72, -117.16),
        # St John's is deliberately outside — Newfoundland has its own inset.
    }
    lost = [nm for nm, (la, lo) in LANDMARKS.items()
            if not (x0 <= tf.transform(lo, la)[0] <= x1
                    and y0 <= tf.transform(lo, la)[1] <= y1)]
    print("frame check: " + ("all landmarks in" if not lost
                             else "OUTSIDE -> " + ", ".join(lost)) + "\n")

    # --- sizes and true locations ----------------------------------------
    size, where = {}, {}
    for name, label, lon, lat, w, h in CITIES:
        size[name] = (label, w, h)
        # Alaska, Hawaii and Puerto Rico are outside the frame, so their true
        # positions must not pull their rail or draw a locator box
        if x0 <= tf.transform(lon, lat)[0] <= x1 and \
                y0 <= tf.transform(lon, lat)[1] <= y1:
            where[name] = (lon, lat)

    # --- density grid -----------------------------------------------------
    d = np.load(BUILD / "dots_na.npz")
    lon, lat, idx = d["lon"], d["lat"], d["idx"]
    X, Y = tf.transform(lon, lat)
    W_c, H_c = int(map_w_in * CELLS_PER_IN), int(map_h_in * CELLS_PER_IN)
    cx = ((X - x0) / (x1 - x0) * W_c).astype(np.int32)
    cy = ((y1 - Y) / (y1 - y0) * H_c).astype(np.int32)
    ok = (cx >= 0) & (cx < W_c) & (cy >= 0) & (cy < H_c)
    grid = np.zeros((H_c, W_c), dtype=np.int64)
    np.add.at(grid, (cy[ok], cx[ok]), 1)
    sat = np.pad(grid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    print(f"density grid {W_c} x {H_c} cells, {grid.sum():,} dots\n")

    # --- pack each rail ---------------------------------------------------
    gap_c = int(GAP_IN * CELLS_PER_IN)
    margin_c = int(args.margin * CELLS_PER_IN)
    step = max(1, int(STEP_IN * CELLS_PER_IN))
    placed, blocked = [], []

    for rail, axis, rows in RAILS:
        # Normalise so each rail squares off. On a side rail every box takes the
        # rail's widest width and its own row's tallest height; on a top/bottom
        # rail every box takes the tallest height. A wider window just shows
        # more surrounding country at the same magnification, so nothing is lost.
        flat = [n for row in rows for n in row]
        if axis == "v":
            col_w = max(size[n][1] for n in flat)
            dims = {n: (col_w, max(size[m][2] for m in row))
                    for row in rows for n in row}
        else:
            row_h = max(size[n][2] for n in flat)
            dims = {n: (size[n][1], row_h) for n in flat}

        row_w = [sum(int(dims[n][0] * CELLS_PER_IN) for n in row)
                 + gap_c * (len(row) - 1) for row in rows]
        row_h_c = [int(dims[row[0]][1] * CELLS_PER_IN) for row in rows]
        if axis == "v":
            run = sum(row_h_c) + gap_c * (len(rows) - 1)
            limit, lo, hi = H_c, margin_c, H_c - run - margin_c
        else:
            run = max(row_w)
            limit, lo, hi = W_c, margin_c, W_c - run - margin_c

        if hi < lo:
            print(f"  {rail:6s} DOES NOT FIT: needs {run / CELLS_PER_IN:.1f} in "
                  f"of a {limit / CELLS_PER_IN:.1f} in edge")
            continue

        # pull each rail toward the mean position of the cities it shows, so
        # Dallas/Houston sit under Texas rather than in a corner
        anchors = [where[n] for n in flat if n in where]
        if anchors:
            ax, ay = tf.transform(*zip(*anchors))
            mean_c = float(np.mean((np.array(ax) - x0) / (x1 - x0) * W_c))
            mean_r = float(np.mean((y1 - np.array(ay)) / (y1 - y0) * H_c))
        else:
            mean_c = mean_r = None

        best, best_cost = None, np.inf
        for off in range(lo, hi + 1, step):
            boxes, cursor = [], off
            for row, rw, rh in zip(rows, row_w, row_h_c):
                inner = margin_c if rail in ("left", "top") else None
                if axis == "v":
                    c_start = margin_c if rail == "left" else W_c - margin_c - rw
                    cx_ = c_start
                    for n in row:
                        wc = int(dims[n][0] * CELLS_PER_IN)
                        boxes.append((n, cursor, cx_, rh, wc))
                        cx_ += wc + gap_c
                    cursor += rh + gap_c
                else:
                    r0 = inner if rail == "top" else H_c - margin_c - rh
                    cx_ = off
                    for n in row:
                        wc = int(dims[n][0] * CELLS_PER_IN)
                        boxes.append((n, r0, cx_, rh, wc))
                        cx_ += wc + gap_c
            if any(not (b[2] + b[4] + gap_c <= t[1] or t[1] + t[3] + gap_c <= b[2] or
                        b[1] + b[3] + gap_c <= t[0] or t[0] + t[2] + gap_c <= b[1])
                   for b in boxes for t in blocked):
                continue
            cost = sum(box_sum(sat, r, c, r + h, c + w) for _, r, c, h, w in boxes)
            if mean_c is not None:
                mid = off + run / 2
                cost += args.pull * abs(mid - (mean_r if axis == "v" else mean_c))
            if cost < best_cost:
                best, best_cost = boxes, cost

        if best is None:
            print(f"  {rail:6s} NO ROOM (collides with an already-placed rail)")
            continue
        total = 0
        for n, r, c, h, w in best:
            covered = int(box_sum(sat, r, c, r + h, c + w))
            total += covered
            blocked.append((r, c, h, w))
            placed.append({"name": n, "label": size[n][0], "rail": rail,
                           "x_in": round(c / CELLS_PER_IN, 2),
                           "y_in": round(r / CELLS_PER_IN, 2),
                           "w_in": round(w / CELLS_PER_IN, 2),
                           "h_in": round(h / CELLS_PER_IN, 2),
                           "covers_dots": covered})
        print(f"  {rail:6s} {len(best)} insets, {run / CELLS_PER_IN:5.1f} in run, "
              f"covers {total:6,} dots")
        for p in placed[-len(best):]:
            print(f"           {p['name']:9s} at {p['x_in']:5.1f},{p['y_in']:5.1f}  "
                  f"{p['covers_dots']:6,} dots")

    grand = sum(p["covers_dots"] for p in placed)
    print(f"\n{len(placed)} insets placed, {grand:,} of {grid.sum():,} dots "
          f"covered ({100 * grand / grid.sum():.2f}%)")

    (BUILD / "layout.json").write_text(json.dumps(
        {"map_in": [round(map_w_in, 2), round(map_h_in, 2)],
         "sheet_in": [round(map_w_in, 2), round(sheet_h_in, 2)],
         "band_in": args.band, "km_per_inch": KM_PER_IN_MAIN,
         "insets": placed},
        indent=2), encoding="utf-8")

    # --- draw -------------------------------------------------------------
    dpi = args.plan_dpi
    W, H = int(map_w_in * dpi), int(map_h_in * dpi)
    palette = json.loads((BUILD / "palette.json").read_text(encoding="utf-8"))
    lut = np.zeros((0x10000, 3), dtype=np.uint8)
    for i, e in enumerate(palette):
        hx = e["color"].lstrip("#")
        lut[i] = [int(hx[j:j + 2], 16) for j in (0, 2, 4)]

    px = (X - x0) / (x1 - x0) * W
    py = (y1 - Y) / (y1 - y0) * H
    keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    rgb, alpha = splat(px[keep], py[keep], idx[keep], W, H, 0.8, lut, ss=2,
                       rng=np.random.default_rng(0))
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[:] = [0x0d, 0x0d, 0x0f]
    sheet = Image.new("RGB", (W, int(sheet_h_in * dpi)), "#08080a")
    sheet.paste(Image.fromarray(over(rgb, alpha, base)), (0, 0))

    dr = ImageDraw.Draw(sheet)
    dr.rectangle([0, H, W - 1, int(sheet_h_in * dpi) - 1],
                 fill="#101014", outline="#2a3038")
    dr.text((10, H + 8), f"LEGEND BAND  {map_w_in:.1f} x {args.band} in", fill="#8a929c")

    for p in placed:
        bx, by = p["x_in"] * dpi, p["y_in"] * dpi
        bw, bh = p["w_in"] * dpi, p["h_in"] * dpi
        dr.rectangle([bx, by, bx + bw, by + bh], fill="#0d0d0f", outline="#f0c674", width=2)
        dr.text((bx + 4, by + 3), p["label"], fill="#f0c674")
        # thin locator box on the map at the window's true footprint
        if p["name"] in where:
            lo_, la_ = where[p["name"]]
            ax, ay = tf.transform(lo_, la_)
            sx = (ax - x0) / (x1 - x0) * W
            sy = (y1 - ay) / (y1 - y0) * H
            hw = p["w_in"] / 5.0 * dpi / 2
            hh = p["h_in"] / 5.0 * dpi / 2
            dr.rectangle([sx - hw, sy - hh, sx + hw, sy + hh],
                         outline="#f0c674", width=1)

    out = BUILD / "layout_plan.png"
    sheet.save(out)
    print(f"\nwrote {out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
