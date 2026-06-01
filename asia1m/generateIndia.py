"""Render India's 2011-census admin divisions onto the asia1m canvas.

Produces one PNG per layer, each pixel-aligned with the asia1m map
(same projection / extent / figsize as generateAsia.py), so they drop
straight into Aseprite as separate layers.

India is built on 2011-census vintage on purpose: that is the vintage of
the IIPS district projections that helper1m uses, so districts (640) and
subdistricts (~5,900) line up with the population data.

Boundaries: SHRUG 2.1 open polygons (Development Data Lab) - all 2011
census vintage, all keyed by pc11 ids, and they nest cleanly because
state / district / subdistrict come from one source.
  Download:  https://www.devdatalab.org/shrug_download/  (free, short form)
  The pack contains state.shp, district.shp, subdistrict.shp (+ .gpkg).
  Drop the three .shp sets (with .shx/.dbf/.prj) into data/asia1m/india/.

Run from the maps/ directory:  python asia1m/generateIndia.py
The canvas is the full 2x asia1m size, so each render is slow (~minutes).
"""
import os

import geopandas
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

# (input shapefile, output png) - SHRUG's native filenames; comment out a
# line to skip that layer.
LAYERS = [
    ("data/asia1m/india/state.shp",       "asia1m/indiadiv_states.png"),
    ("data/asia1m/india/district.shp",    "asia1m/indiadiv_districts.png"),
    ("data/asia1m/india/subdistrict.shp", "asia1m/indiadiv_subdistricts.png"),
]

# Projection params - MUST stay identical to generateAsia.py, or the render
# will not line up with the rest of the asia1m canvas in Aseprite.
CRS = ccrs.AlbersEqualArea(95, 0, 0, 0, (6, 42))
EXTENT = [47.7, 129.2, -9, 65]          # lon/lat bounds, PlateCarree
FIGSIZE = (189.65, 159.69)              # 2x zoom; use (95, 80) for 1x


def render(shp_path, out_path):
    df = geopandas.read_file(shp_path)
    if df.crs is None:
        df = df.set_crs("EPSG:4326")
    df = df.to_crs(CRS)

    fig, ax = plt.subplots(subplot_kw={"projection": CRS}, figsize=FIGSIZE)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    plt.tight_layout()

    df.plot(ax=ax, edgecolor="#63538d", facecolor="none", zorder=10,
            rasterized=True, antialiased=False, linewidth=0.004)

    plt.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(df)} features)")


def main():
    for shp_path, out_path in LAYERS:
        if not os.path.exists(shp_path):
            print(f"missing: {shp_path}")
            print("  download SHRUG open polygons (see header), drop into data/asia1m/india/")
            continue
        print(f"rendering {shp_path} ...")
        render(shp_path, out_path)


if __name__ == "__main__":
    main()
