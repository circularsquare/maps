"""Render dry-summer / wet-summer maps from the IMERG monthly climatology (2001-2022).

Input:  raw/IMERG-Final.CLIM.2001-2022.MM.V07B.tif, 3600x1800 float32, mm/day, -9999.9 fill.
Output: season.png and annual.png (3600x1800), and months.bin (12 x 360 x 720 uint16, mm/month,
        0.5 deg, 65535 = no data) for the hover chart.
"""
import json
import os

import numpy as np
import tifffile
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
LAND = os.path.join(HERE, "..", "citybrowser", "data", "ne_50m_land.geojson")
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float32)
W, H = 3600, 1800

EPS = 5.0          # mm/month added to both sides so near-zero months don't explode the ratio
BG = np.array([17, 17, 17], dtype=np.float32)


def load():
    out = np.empty((12, H, W), dtype=np.float32)
    for m in range(12):
        a = tifffile.imread(os.path.join(RAW, f"IMERG-Final.CLIM.2001-2022.{m + 1:02d}.V07B.tif"))
        a = a.astype(np.float32)
        a[a < 0] = np.nan
        out[m] = a * DAYS[m]
        print(f"month {m + 1}: loaded", flush=True)
    return out


# Diverging ramp: dry summer (negative) brown-orange -> neutral -> wet summer (positive) teal-blue.
# Positions are log2(summer / winter). The middle +-4 (16x) carries most of the colour change;
# 16x..64x only darkens a little further, so the extremes stay readable without washing out
# the middle.
STOPS = [
    (-6.0, (95, 30, 15)),
    (-4.0, (150, 55, 20)),
    (-2.4, (205, 110, 40)),
    (-1.0, (225, 185, 120)),
    (0.0, (200, 200, 195)),
    (1.0, (130, 195, 200)),
    (2.4, (40, 140, 185)),
    (4.0, (25, 65, 150)),
    (6.0, (30, 30, 95)),
]


# Annual total, on log10(mm/year): dry dark sand -> olive -> green -> teal -> pale blue.
ANNUAL_STOPS = [
    (np.log10(30), (45, 38, 30)),
    (np.log10(150), (100, 82, 55)),
    (np.log10(400), (145, 140, 70)),
    (np.log10(900), (75, 165, 95)),
    (np.log10(1800), (40, 170, 170)),
    (np.log10(3500), (110, 190, 240)),
    (np.log10(7000), (235, 245, 255)),
]


def ramp(t, stops=STOPS):
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=np.float32)
    t = np.clip(t, xs[0], xs[-1])
    return np.stack([np.interp(t, xs, cs[:, i]) for i in range(3)], axis=-1)


def colourise_annual(annual, ocean):
    rgb = ramp(np.log10(np.maximum(np.nan_to_num(annual), 1)), ANNUAL_STOPS)
    k = np.where(ocean, 0.45, 1.0)
    rgb = BG + (rgb - BG) * k[..., None]
    rgb[np.isnan(annual)] = BG
    return rgb


def colourise(logratio, annual, ocean):
    rgb = ramp(logratio)
    # Fade toward the background where the year is too dry for a ratio to mean much.
    wet = np.clip((np.log10(np.maximum(annual, 1)) - np.log10(80)) / (np.log10(400) - np.log10(80)), 0, 1)
    wet = np.nan_to_num(wet)
    k = 0.15 + 0.85 * wet
    k = np.where(ocean, k * 0.32, k)
    rgb = BG + (rgb - BG) * k[..., None]
    rgb[np.isnan(logratio)] = BG
    return rgb


def land_mask_and_outline():
    with open(LAND, encoding="utf-8") as f:
        gj = json.load(f)
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    rings = []
    for feat in gj["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for i, ring in enumerate(poly):
                pts = [((x + 180) * 10, (90 - y) * 10) for x, y in ring]
                md.polygon(pts, fill=255 if i == 0 else 0)
                rings.append(pts)
    return np.array(mask) > 0, rings


def main():
    p = load()
    annual = np.nansum(p, axis=0)
    annual[np.all(np.isnan(p), axis=0)] = np.nan
    lat = 90 - (np.arange(H) + 0.5) / 10
    north = (lat >= 0)[:, None]

    # Months are 0-based: Jun..Aug = 5..7, Dec..Feb = 11, 0, 1.
    jja = np.nansum(p[5:8], axis=0)
    djf = np.nansum(p[[11, 0, 1]], axis=0)
    summer = np.where(north, jja, djf)
    winter = np.where(north, djf, jja)

    # "Summer" flips at the equator; fade the ratio to neutral within 3 degrees of it so the
    # flip is not a hard seam through the Amazon and Congo.
    fade = np.clip(np.abs(lat) / 3.0, 0, 1)[:, None]
    views = {
        "season": fade * np.log2((summer + 3 * EPS) / (winter + 3 * EPS)),
        "annual": None,
    }

    land, rings = land_mask_and_outline()
    ocean = ~land
    for name, lr in views.items():
        if lr is None:
            rgb = colourise_annual(annual, ocean)
        else:
            rgb = colourise(np.where(np.isnan(annual), np.nan, lr), annual, ocean)
        img = Image.fromarray(rgb.round().astype(np.uint8))
        d = ImageDraw.Draw(img)
        for pts in rings:
            d.line(pts + [pts[0]], fill=(90, 90, 90), width=1)
        img.save(os.path.join(HERE, f"{name}.png"), optimize=True)
        print(f"wrote {name}.png", flush=True)

    # 0.5 deg month grid for the hover chart.
    coarse = np.nanmean(p.reshape(12, H // 5, 5, W // 5, 5), axis=(2, 4))
    q = np.where(np.isnan(coarse), 65535, np.clip(np.round(coarse), 0, 65534)).astype("<u2")
    q.tofile(os.path.join(HERE, "months.bin"))
    print("wrote months.bin", q.shape, flush=True)


if __name__ == "__main__":
    main()
