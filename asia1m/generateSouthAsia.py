"""Render a South Asia admin-division reference onto the asia1m canvas.

One combined transparent PNG, pixel-aligned with the asia1m map (same
projection / extent / figsize as generateAsia.py at 1x), to drop into
Aseprite as a single tracing layer.

Layers drawn (the levels already started in asia1m.ase):
  - India        districts      (SHRUG district.shp, 2011-census vintage)
  - Bangladesh   subdistricts   (upazilas,  bgd_admin3.shp)
  - Pakistan     subdistricts   (tehsils,   pak_admin3.shp)
  - Sri Lanka    subdistricts   (DS divisions / division secretariats, lka_admin3.shp)

Run from the maps/ directory:  C:\\Python39\\python.exe asia1m/generateSouthAsia.py
1x canvas (95x80 in @ dpi 100 = 9500x8000 px).
"""
import os

import geopandas
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

# Shapefiles to draw onto the one canvas. Comment a line out to skip a layer.
LAYERS = [
    "data/asia1m/india/district.shp",        # India districts
    "data/asia1m/bangladesh/bgd_admin3.shp",  # Bangladesh subdistricts (upazilas)
    "data/asia1m/pakistan/pak_admin3.shp",    # Pakistan subdistricts (tehsils)
    "data/asia1m/srilanka/lka_admin3.shp",    # Sri Lanka subdistricts (DS divisions)
]

OUT_PATH = "asia1m/southasiadiv.png"
EDGE = "#4a8c3f"                       # green border lines

# Projection params - MUST stay identical to generateAsia.py, or the render
# will not line up with the rest of the asia1m canvas in Aseprite.
CRS = ccrs.AlbersEqualArea(95, 0, 0, 0, (6, 42))
EXTENT = [47.7, 129.2, -9, 65]         # lon/lat bounds, PlateCarree
FIGSIZE = (95, 80)                     # 1x -> 9500x8000 px at dpi 100
                                       # (2x is (189.65, 159.69))


def main():
    fig, ax = plt.subplots(subplot_kw={"projection": CRS}, figsize=FIGSIZE)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_axis_off()
    plt.tight_layout()

    for shp_path in LAYERS:
        if not os.path.exists(shp_path):
            print(f"missing: {shp_path}  (skipped)")
            continue
        df = geopandas.read_file(shp_path)
        if df.crs is None:
            df = df.set_crs("EPSG:4326")
        df = df.to_crs(CRS)
        df.plot(ax=ax, edgecolor=EDGE, facecolor="none", zorder=10,
                rasterized=True, antialiased=False, linewidth=0.004)
        print(f"  drew {shp_path}  ({len(df)} features)")

    plt.savefig(OUT_PATH, transparent=True)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
