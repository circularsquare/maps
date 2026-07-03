"""Terrain sources for ascentshed.

synthetic() reproduces the prototype's test surface: one broad massif (high
volume, modest prominence), two tall thin spires (high prominence, low volume),
a cluster of nested medium hills, a regional tilt, and several octaves of
fractal noise. The mismatch between volume-rank and prominence-rank is the whole
point of the test.

load_dem(path) reads a real GeoTIFF (full topo+bathy, e.g. GEBCO). Bathymetry is
NOT masked -- oceanic islands then score base-to-peak. Returns z plus a per-row
cell-area weight (cos(lat)) when the raster is geographic.
"""
import numpy as np
from scipy import ndimage


def _bump(shape, cy, cx, h, sigma):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return h * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))


def synthetic(n=320, seed=7):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, n))
    z += _bump((n, n), 0.42 * n, 0.30 * n, h=1400, sigma=0.16 * n)   # broad massif
    z += _bump((n, n), 0.30 * n, 0.70 * n, h=2100, sigma=0.040 * n)  # tall spire
    z += _bump((n, n), 0.66 * n, 0.78 * n, h=1950, sigma=0.045 * n)  # tall spire
    z += _bump((n, n), 0.72 * n, 0.32 * n, h=900,  sigma=0.10 * n)   # medium cluster
    z += _bump((n, n), 0.78 * n, 0.42 * n, h=650,  sigma=0.05 * n)
    z += _bump((n, n), 0.62 * n, 0.22 * n, h=600,  sigma=0.05 * n)
    yy, xx = np.mgrid[0:n, 0:n]
    z += 0.6 * xx + 0.3 * yy                                          # regional tilt
    noise = np.zeros((n, n))
    for octave in range(5):
        s = 2 ** octave
        noise += ndimage.gaussian_filter(rng.standard_normal((n, n)),
                                          sigma=2.0 * s) * (s * 6.0)
    z += noise
    z += rng.standard_normal((n, n)) * 1e-4                           # tie-break jitter
    return z


def load_dem(path, downsample=1, jitter=1e-4, seed=0):
    """Read a GeoTIFF DEM (topo+bathy). Optional integer downsample for speed.
    Adds tiny jitter to break flats. Returns (z, cellarea2d)."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        transform = ds.transform
        crs = ds.crs
        if downsample > 1:                       # decimate on read (low memory)
            h, w = ds.height // downsample, ds.width // downsample
            z = ds.read(1, out_shape=(h, w),
                        resampling=Resampling.nearest).astype(float)
        else:
            z = ds.read(1).astype(float)
    rng = np.random.default_rng(seed)
    z = z + rng.standard_normal(z.shape) * jitter
    # cell area weight: cos(lat) for geographic rasters, else uniform
    cellarea = np.ones_like(z)
    try:
        if crs and crs.is_geographic:
            n = z.shape[0]
            top = transform.f
            dy = transform.e * downsample
            lats = top + (np.arange(n) + 0.5) * dy
            cellarea = np.repeat(np.cos(np.radians(lats))[:, None], z.shape[1], axis=1)
            cellarea = np.clip(cellarea, 1e-6, None)
    except Exception:
        pass
    return z, cellarea
