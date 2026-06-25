import numpy as np
from scipy.ndimage import convolve, binary_dilation
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import glob
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from PIL import Image

# ── projections ────────────────────────────────────────────────────────────────
crs_albers = ccrs.AlbersEqualArea(95, 0, 0, 0, (6, 42))
plat = ccrs.PlateCarree()

# Set up the figure just to get the Albers extent/bounds
# 2x: figsize matches generateAsia.py 2x render (189.65, 159.69) -> 18965x15969
fig, ax = plt.subplots(subplot_kw={"projection": crs_albers}, figsize=(189.65, 159.69))
ax.set_extent([47.7, 129.2, -9, 65], crs=plat)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()

plt.tight_layout()
fig.savefig('tmp2x.png', dpi=100)  # force a render first
bbox = ax.get_position()  # in figure-fraction coordinates
ax_width_px = int(bbox.width * 189.65 * 100)   # fig width inches * dpi
ax_height_px = int(bbox.height * 159.69 * 100)  # fig height inches * dpi
print('mask output size:', ax_width_px, ax_height_px)

# ── load and mosaic tiles ──────────────────────────────────────────────────────
tile_files = glob.glob("../data/asia1m/waterdata/*.tif")
datasets = [rasterio.open(f) for f in tile_files]
mosaic, src_transform = merge(datasets, bounds=(0, -15, 155, 85), res=0.003)
seasonality = mosaic[0]

src_crs = CRS.from_epsg(4326)
dst_crs = CRS.from_proj4(crs_albers.proj4_init)

# ── step 1: reproject to supersampled resolution using nearest-neighbor ─────────
# 2x: SCALE=2 keeps the *fine* grid identical to the 1x run (which used SCALE=4 on
# a half-size axes), so source sampling fidelity is unchanged; only the final
# downsample block is 2x2 instead of 4x4.
SCALE = 2
fine_height = ax_height_px * SCALE
fine_width  = ax_width_px * SCALE
print('fine grid:', fine_width, fine_height)

fine_transform = rasterio.transform.from_bounds(x0, y0, x1, y1, fine_width, fine_height)

fine_output = np.zeros((fine_height, fine_width), dtype=np.float32)
reproject(
    source=seasonality.astype(np.float32),
    destination=fine_output,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=fine_transform,
    dst_crs=dst_crs,
    resampling=Resampling.nearest,   # ← nearest, not average
)
del mosaic, seasonality

# ── step 2: threshold at fine resolution ──────────────────────────────────────
fine_water = (fine_output > 10).astype(np.uint8)  # 1 = water, 0 = not water
del fine_output

# ── step 3: 2x2 majority-vote downsample ──────────────────────────────────────
blocks = fine_water.reshape(ax_height_px, SCALE, ax_width_px, SCALE)
votes  = blocks.sum(axis=(1, 3))          # shape (H, W), range 0-4
# 1x used >=3 of 16 (~0.19 coverage); the water-biased analog for a 2x2 block is
# "any subpixel water" (>=1 of 4 = 0.25), preserving thin coast/rivers. Isolated
# specks are cleaned by the morphology stage (plotwaterdata2_2x.py).
water_mask = votes >= 1

# ── render mask ────────────────────────────────────────────────────────────────
rgba = np.zeros((ax_height_px, ax_width_px, 4), dtype=np.uint8)
rgba[water_mask] = [217, 246, 255, 255]   # water
Image.fromarray(rgba).save('globalwater2x.png')
print('saved globalwater2x.png')
