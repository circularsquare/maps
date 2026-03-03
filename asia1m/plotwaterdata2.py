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
fig, ax = plt.subplots(subplot_kw={"projection": crs_albers}, figsize=(95, 80))
ax.set_extent([47.7, 129.2, -9, 65], crs=plat)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()

img = np.array(Image.open('globalwater.png'))
print('meow')
water_mask = (img[:, :, 0] == 217).astype(np.int8)
del img

#kernel for counting neighbors (excludes center)
kernelo = np.array([[0,1,0],
                   [1,0,1],
                   [0,1,0]], dtype=np.uint8)
kernel = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]], dtype=np.uint8)
kernel5 = np.array([[1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1]], dtype=np.uint8)
kernel7 = np.array([[1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],], dtype=np.uint8)
from scipy.ndimage import generate_binary_structure
struct = generate_binary_structure(2, 2)  # 3x3 all-ones kernel
def dilate(mask):
    """Returns 1 wherever mask has a 1 in any of the 8 neighbors."""
    out = np.zeros_like(mask)
    out[1:,  :]  += mask[:-1, :]   # up
    out[:-1, :]  += mask[1:,  :]   # down
    out[:,  1:]  += mask[:, :-1]   # left
    out[:, :-1]  += mask[:,  1:]   # right
    out[1:,  1:] += mask[:-1, :-1] # up-left
    out[1:, :-1] += mask[:-1, 1:]  # up-right
    out[:-1, 1:] += mask[1:, :-1]  # down-left
    out[:-1,:-1] += mask[1:,  1:]  # down-right
    return (out > 0).astype(np.int8)
def orr(x, y):
    return np.clip(x + y, 0, 1)

# ── step 1: remove isolated water pixels (0 or 1 water neighbors) ─────────────
water_neighbor_count = convolve(water_mask.astype(np.uint8), kernel, mode='constant', cval=0)
water = water_mask * (water_neighbor_count > 1)
print('meow')
# - step 1.1: remove isolated islands
land = (1 - water)
touches_land = dilate(land) 
water = orr(water, land * (1-touches_land))

# ── step 2: border = water pixels touching land ────────────────────────────────
land = 1 - water
touches_land = dilate(land)#convolve(land_mask.astype(np.uint8), kernel, mode='constant', cval=0) > 0
border = water * touches_land
print('meow')
print(border.mean())
print(water.mean())

# step 3: re-remove isolated water
def step3(wb):
    (water, border) = wb
    water = water * (1 - border)
    water_neighbor_count = convolve(water.astype(np.uint8), kernel5, mode='constant', cval=0)
    border_neighbor_count = convolve(border.astype(np.uint8), kernel5, mode='constant', cval=0)
    water = water * (water_neighbor_count + border_neighbor_count * 0.2 >= 3)
    water_neighbor_count7 = convolve(water.astype(np.uint8), kernel7, mode='constant', cval=0)
    water = water * (water_neighbor_count7 > 2)
    # ── step 3.5: re-add border
    land = (1 - water) * (1 - border)
    touches_land = dilate(land)
    border = np.clip(border + water * touches_land, 0, 1)
    print('3')
    return (water, border)
# ── step 4: keep only border pixels that touch BOTH water and land ─────────────
def step4(wb):
    (water, border) = wb
    water = water * (1 - border)
    water_neighbor_count = convolve(water.astype(np.uint8), kernel, mode='constant', cval=0)
    water_neighbor_count5 = convolve(water.astype(np.uint8), kernel5, mode='constant', cval=0)
    border_neighbor_count = convolve(border.astype(np.uint8), kernel5, mode='constant', cval=0)
    border = border * (water_neighbor_count * 5 + border_neighbor_count + water_neighbor_count5 * 0.2 > 9)
    land = (1 - water) * (1 - border)
    land_neighbor_count_o = convolve(land.astype(np.uint8), kernelo, mode='constant', cval=0)
    border = border * (land_neighbor_count_o < 3)
    print('4')
    return (water, border)

(water, border) = step3((water, border))
(water, border) = step4(step4(step4((water, border))))
(water, border) = step3((water, border))
(water, border) = step4(step4(step4(step4((water, border)))))
(water, border) = step3((water, border))
(water, border) = step4(step4((water, border)))

# ── render ─────────────────────────────────────────────────────────────────────
rgba = np.zeros((7970, 9303, 4), dtype=np.uint8)
rgba[water.astype(bool), :] = [217, 246, 255, 255]
rgba[border.astype(bool), :] = [100, 160, 210, 255]

# rgba[touches_land.astype(bool), :] = [200, 1, 1, 255]
# rgba[land.astype(bool), :] = [1, 200, 1, 255]

#Image.fromarray(rgba).save('test.png')
# -----------------------------------------------------
print('meowww')
ax.imshow(rgba, extent=[x0, x1, y0, y1], origin='upper',
          transform=crs_albers, 
          zorder=10, interpolation='none')
    

plt.tight_layout()
plt.savefig('globalwaterf.png')
