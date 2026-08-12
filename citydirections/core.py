"""Shared maths for citydirections.

Both the US pipeline (build.py, ACS block groups) and the global one
(build_rwi.py, Meta Relative Wealth Index) reduce to the same problem: given
points carrying some measure of wealth inside one city, which way is rich?
Everything that is not data ingest lives here so the two agree by construction.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

KM_PER_DEG_LAT = 110.574

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(bearing):
    return COMPASS[int(round(bearing / 22.5)) % 16]


def to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular km east/north of an origin. Fine over a single city."""
    x = (lon - lon0) * KM_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat - lat0) * KM_PER_DEG_LAT
    return x, y


def weighted_rank(values, weights):
    """Weighted percentile rank in [0, 1]."""
    order = np.argsort(values, kind="stable")
    w = weights[order]
    cum = np.cumsum(w) - 0.5 * w          # midpoint of each unit's weight slice
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = cum / w.sum()
    return ranks


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, v)


def wls(design, target, weights):
    """Weighted least squares. Returns coefficients and weighted R^2."""
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)
    resid = target - design @ coef
    ss_res = np.sum(weights * resid ** 2)
    ss_tot = np.sum(weights * (target - np.average(target, weights=weights)) ** 2)
    return coef, (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def density_peak(x, y, weight, bandwidth_km=4.0, cell_km=1.0):
    """Smoothed density peak, used as a city centre for the radial term only."""
    xe = np.arange(x.min() - 2, x.max() + 2 + cell_km, cell_km)
    ye = np.arange(y.min() - 2, y.max() + 2 + cell_km, cell_km)
    grid = np.histogram2d(x, y, bins=[xe, ye], weights=weight)[0]
    grid = gaussian_filter(grid, sigma=bandwidth_km / cell_km, mode="constant")
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2


def fit_wealth_field(x, y, value, weight):
    """Which way is rich, for one city's points.

    `value` is any monotone measure of wealth -- dollars, an index, anything --
    because it is immediately converted to a weighted percentile rank and
    z-scored. That is what lets ACS median household income and Meta's RWI feed
    the same estimator, and what makes the result comparable between cities on
    different data.

    Direction comes from a weighted plane fit, which needs no city centre. That
    matters: "downtown" is the hardest thing to define consistently across
    cities, and anchoring on a bad centre would rotate every arrow the same way.

    Returns the fit plus the per-point rank field.
    """
    weight = np.asarray(weight, dtype=float)
    rank = weighted_rank(np.asarray(value, dtype=float), weight)
    z = rank - np.average(rank, weights=weight)
    spread = np.sqrt(np.average(z ** 2, weights=weight))
    if not np.isfinite(spread) or spread == 0:
        return None
    z = z / spread

    # Direction: centre-free plane fit.
    (a, bx, by), r2 = wls(np.column_stack([np.ones_like(x), x, y]), z, weight)
    bearing = np.degrees(np.arctan2(bx, by)) % 360        # compass, 0 = N
    strength = float(np.hypot(bx, by) * 10.0)             # sd per 10 km

    # Radial term: rich core vs rich suburbs. This one does need a centre.
    cx, cy = density_peak(x, y, weight)
    r = np.maximum(np.hypot(x - cx, y - cy), 0.5)
    theta = np.arctan2(y - cy, x - cx)
    design = np.column_stack([np.ones_like(r), np.log(r), np.cos(theta), np.sin(theta)])
    (_, gamma, ccx, ccy), r2_polar = wls(design, z, weight)

    return dict(
        bearing=float(bearing), compass=compass(bearing), strength=strength,
        r2=float(r2), radial=float(gamma), r2_polar=float(r2_polar),
        bearing_polar=float(np.degrees(np.arctan2(ccx, ccy)) % 360),
        peak_dx=float(cx), peak_dy=float(cy), rank=rank, z=z,
    )
