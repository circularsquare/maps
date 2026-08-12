"""Render the citydirections prototype.

  out/arrows_us.png    one arrow per metro, on an Albers US map
  out/medallions.png   the actual intra-city wealth field for the largest metros,
                       drawn as circular medallions -- a preview of what these
                       would look like as bubbles on a world map
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from core import weighted_rank

mpl.rcParams["font.family"] = "DejaVu Sans"

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"

STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
ALBERS = "EPSG:5070"
WEALTH_CMAP = "viridis"     # dark = poor, yellow = rich
R2_FULL = 0.12              # r^2 at which an arrow is drawn fully opaque

# Opacity ramp for the medallions, in people per km^2 of smoothed population
# density. Log-scaled because urban density spans orders of magnitude. Cities'
# built-up cores sit well above the top of the ramp and stay fully opaque; only
# the exurban fringe fades out.
DENSITY_LO, DENSITY_HI = 30.0, 600.0
EDGE_FEATHER = 0.07         # fraction of the radius over which the rim fades

# Alaska, Hawaii and Puerto Rico metros would otherwise land off the CONUS map.
CONUS_BOX = (-125.5, 24.0, -66.5, 49.5)


def conus(df):
    lo_x, lo_y, hi_x, hi_y = CONUS_BOX
    return df[df.cent_lon.between(lo_x, hi_x) & df.cent_lat.between(lo_y, hi_y)]


def bearing_cmap(sat=0.62, val=0.74):
    """Cyclic colour wheel at constant lightness.

    Matplotlib's cyclic maps (twilight, twilight_shifted) run through near-white
    at one end of the cycle, so arrows at that bearing disappear against a light
    background. Holding saturation and value fixed keeps every direction equally
    legible, at the cost of hue spacing being less perceptually uniform.
    """
    hue = np.linspace(0, 1, 256, endpoint=False)
    rgb = mpl.colors.hsv_to_rgb(np.column_stack([hue, np.full(256, sat), np.full(256, val)]))
    return mpl.colors.ListedColormap(rgb, name="bearing")


BEARING_CMAP = bearing_cmap()


def states():
    path = DATA / "cb_2023_us_state_20m"
    if not path.exists():
        print("fetching state boundaries ...")
        raw = urllib.request.urlopen(STATES_URL, timeout=120).read()
        zipfile.ZipFile(io.BytesIO(raw)).extractall(path)
    gdf = gpd.read_file(path)
    return gdf[~gdf.STUSPS.isin(["AK", "HI", "PR"])].to_crs(ALBERS)


def arrows_us(metros):
    metros = conus(metros)
    tf = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    mx, my = tf.transform(metros.cent_lon.to_numpy(), metros.cent_lat.to_numpy())

    fig, ax = plt.subplots(figsize=(16, 10))
    states().plot(ax=ax, facecolor="#f2f0ec", edgecolor="#ffffff", linewidth=1.1, zorder=1)

    theta = np.radians(metros.bearing.to_numpy())
    # Arrow length encodes gradient steepness. Units here are metres (Albers), and
    # the floor keeps weak metros visible rather than collapsing them to a dot.
    length = (2.2 + 9.0 * metros.strength.to_numpy()) * 55_000
    u, v = np.sin(theta) * length, np.cos(theta) * length
    alpha = np.clip(metros.r2.to_numpy() / R2_FULL, 0.25, 1.0)
    colour = BEARING_CMAP(metros.bearing.to_numpy() / 360.0)
    colour[:, 3] = alpha

    ax.quiver(mx, my, u, v, color=colour, angles="xy", scale_units="xy", scale=1,
              width=0.0042, headwidth=3.4, headlength=3.9, headaxislength=3.4,
              zorder=4)
    ax.scatter(mx, my, s=15, c="#22201e", zorder=5, linewidths=0)

    big = metros.nlargest(26, "pop")
    bx, by = tf.transform(big.cent_lon.to_numpy(), big.cent_lat.to_numpy())
    for x, y, name in zip(bx, by, big.name):
        ax.annotate(name.split("-")[0].split(",")[0], (x, y),
                    xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=7.6, color="#3a3733", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="#f2f0ec")])

    ax.set_xlim(-2_450_000, 2_400_000)
    ax.set_ylim(150_000, 3_310_000)
    ax.set_axis_off()
    ax.set_aspect("equal")

    ax.text(0, 1.115, "Which way are the rich neighbourhoods?", transform=ax.transAxes,
            fontsize=22, color="#1d1b19", va="top", ha="left")
    ax.text(0, 1.068, "Direction of the population-weighted household-income gradient in each US metro. Arrow length is how steep that gradient is.\n"
            "Faded arrows are metros where wealth is barely arranged directionally at all (low R²) — New York and Los Angeles are the clearest cases.\n"
            "ACS 2019–23 median household income by census block group.",
            transform=ax.transAxes, fontsize=9.4, color="#6a655f", va="top",
            ha="left", linespacing=1.6)

    # cyclic compass legend, inset over the Pacific
    lax = ax.inset_axes([0.015, 0.03, 0.135, 0.26], projection="polar")
    ang = np.linspace(0, 2 * np.pi, 361)
    lax.scatter(ang, np.ones_like(ang), c=BEARING_CMAP(ang / (2 * np.pi)),
                s=30, linewidths=0)
    lax.set_theta_zero_location("N")
    lax.set_theta_direction(-1)
    lax.set_rlim(0, 1.3)
    lax.set_yticks([])
    lax.set_xticks(np.radians([0, 90, 180, 270]))
    lax.set_xticklabels(["N", "E", "S", "W"], fontsize=8, color="#4a4642")
    lax.grid(False)
    lax.patch.set_alpha(0)
    lax.spines["polar"].set_visible(False)

    fig.savefig(OUT / "arrows_us.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote out/arrows_us.png")


def medallion(ax, sub, row, cell_km=0.4, bandwidth_km=2.0):
    """One city's wealth field, smoothed onto a grid and clipped to a circle.

    Block-group medians are noisy at their native size, so the raw field reads as
    speckle. Smoothing is population-weighted -- a dense block group pulls the
    local value harder than a sparse one -- and cells with almost nobody near
    them are dropped, which is what carves out water, parks and airports and
    leaves a recognisable city silhouette.
    """
    R = row.radius_km
    x, y, z = sub.x.to_numpy(), sub.y.to_numpy(), sub["rank"].to_numpy()
    w = sub["pop"].to_numpy(float)

    edges = np.arange(-R, R + cell_km, cell_km)
    pop_grid = np.histogram2d(x, y, bins=[edges, edges], weights=w)[0]
    val_grid = np.histogram2d(x, y, bins=[edges, edges], weights=w * z)[0]
    sigma = bandwidth_km / cell_km
    ps = gaussian_filter(pop_grid, sigma, mode="constant")
    vs = gaussian_filter(val_grid, sigma, mode="constant")

    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(ps > 1e-9, vs / ps, np.nan)

    # Smoothing pulls everything toward the middle; re-rank so the colour ramp
    # keeps spanning poorest-to-richest and the colourbar label stays true.
    # Weighted by population, so empty cells cannot drag the ranking around.
    ok = np.isfinite(img)
    if ok.sum() > 10:
        img[ok] = weighted_rank(img[ok], ps[ok])

    # Opacity from population density, so the city dissolves into the page
    # instead of ending at a hard threshold. Smoothstep on log density.
    density = ps / cell_km ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (np.log10(np.maximum(density, 1e-9)) - np.log10(DENSITY_LO)) / \
            (np.log10(DENSITY_HI) - np.log10(DENSITY_LO))
    t = np.clip(t, 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)

    # Feather the rim too, otherwise a metro that fills its circle gets a hard
    # edge exactly where the 50 km trim happened to fall.
    cx, cy = np.meshgrid(edges[:-1] + cell_km / 2, edges[:-1] + cell_km / 2, indexing="ij")
    r = np.hypot(cx, cy)
    alpha *= np.clip((R - r) / (EDGE_FEATHER * R), 0.0, 1.0)
    alpha[~ok] = 0.0

    rgba = plt.get_cmap(WEALTH_CMAP)(np.nan_to_num(img))
    rgba[..., 3] = alpha

    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", extent=[-R, R, -R, R],
              interpolation="bilinear", zorder=2)
    ax.add_patch(Circle((0, 0), R, facecolor="none", edgecolor="#eae7e2",
                        linewidth=1.1, zorder=3))

    th = np.radians(row.bearing)
    L = R * 0.72
    ax.annotate("", xy=(np.sin(th) * L, np.cos(th) * L), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.30,head_length=0.62",
                                color="white", linewidth=3.4, alpha=0.95,
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate("", xy=(np.sin(th) * L, np.cos(th) * L), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.55",
                                color="#d81e3f", linewidth=1.7,
                                shrinkA=0, shrinkB=0), zorder=5)

    ax.set_xlim(-R * 1.06, R * 1.06)
    ax.set_ylim(-R * 1.06, R * 1.06)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"{row['name'].split('-')[0].split(',')[0]}   "
                 f"$\\bf{{{row.compass}}}$",
                 fontsize=11, pad=6, color="#1d1b19")
    # tucked into the empty corner outside the inscribed circle
    ax.text(0.015, 0.02, f"R² {row.r2:.2f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.8, color="#a09a92")


def medallions(metros, field, n=16, ncol=4):
    top = metros.nlargest(n, "pop").reset_index(drop=True)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.35 * nrow))
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        medallion(ax, field[field.cbsa == int(row.cbsa)], row)
    for ax in axes.ravel()[len(top):]:
        ax.set_axis_off()

    fig.suptitle("The wealth field inside each metro", fontsize=17, y=0.997,
                 x=0.012, ha="left", color="#1d1b19")
    fig.text(0.012, 0.973,
             "Census block group median household income, population-weighted, smoothed to 2 km, then re-ranked within each metro so the colour ramp "
             "always spans that metro's own poorest to richest.\nOpacity is population density, so each city fades out through its own suburbs rather "
             "than ending at a hard edge. Arrow is the fitted gradient direction.",
             fontsize=9.2, color="#6a655f", ha="left", va="top", linespacing=1.5)

    cax = fig.add_axes([0.30, -0.014, 0.40, 0.0085])
    mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(WEALTH_CMAP),
                              orientation="horizontal")
    cax.set_xticks([0, 0.5, 1])
    cax.set_xticklabels(["poorest in metro", "median", "richest in metro"],
                        fontsize=8.2, color="#6a655f")
    cax.tick_params(length=0)

    fig.tight_layout(rect=[0, 0.006, 1, 0.962])
    fig.savefig(OUT / "medallions.png", dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote out/medallions.png")


def main():
    metros = pd.read_csv(OUT / "metros.csv")
    field = pd.read_csv(OUT / "field.csv")
    arrows_us(metros)
    medallions(metros, field)


if __name__ == "__main__":
    main()
