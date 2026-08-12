"""Global pipeline via Meta's Relative Wealth Index.

*** NEGATIVE RESULT. Do not use this output for the map. ***

The code is correct and the pipeline runs -- 624 cities, same estimator as the US
side -- but RWI does not measure what this project needs, and the failure is
systematic rather than noisy. Kept so the finding is not rediscovered from
scratch; see README for the evidence table.

In short: within a single large city, RWI does not track wealth, and in cities
with low-density affluent districts it is *inverted*. Dar es Salaam's Tandale, an
informal settlement, scores 1.59 while Masaki, the diplomatic quarter, scores
0.72. Lagos ranks Mushin above Ikoyi. Nairobi cannot separate Kibera from Karen.
Across 624 cities, adding distance-from-centre lifts median R^2 from 0.068 to
0.355 and 98% come out "rich core" with a median coefficient of -0.996 -- RWI's
within-city variance is nearly all a radial urbanisation gradient.

This is not a defect in RWI. It is validated for relative poverty at aggregate
and rural scale, where dense/connected genuinely means richer. Inside a metro
that relationship reverses, because affluence there looks like low-density
leafy plots, which read as *less* developed to satellite and connectivity
features. Using RWI for intra-urban gradients is outside its design envelope.

Two further limits, independent of the above:

1. **No population weights.** RWI ships lat/lon/rwi and nothing else, so every
   cell counts equally -- settled-area weighting, not population weighting.
2. **No high-income countries.** LMICs only.

Outputs
-------
  out/cities_rwi.csv   one row per city
  out/field_rwi.csv    the per-cell wealth field
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

from core import fit_wealth_field, to_local_xy

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
CITIES_CSV = HERE.parent / "worldcities" / "worldcities.csv"

MIN_CITY_POP = 300_000
MIN_CELLS = 40          # below this a plane fit is not worth reporting
EARTH_R = 6371.0088


ABSORB_FRAC = 1.0       # absorb satellites within this multiple of the parent radius


def city_radius_km(pop):
    """Analysis radius from city population. Deliberately gentle (cube-root-ish):
    metro footprint grows far more slowly than population."""
    return np.clip(13.0 * (pop / 1e6) ** 0.36, 10.0, 50.0)


def to_ecef(lat, lon):
    """Lat/lon to 3-D Cartesian, so a KD-tree gives true great-circle neighbours
    without wrapping problems at the antimeridian or distortion near the poles."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) * EARTH_R


def load_cities():
    c = pd.read_csv(CITIES_CSV)
    c = c[c.population.notna() & (c.population >= MIN_CITY_POP)].copy()
    c = c.sort_values("population", ascending=False).reset_index(drop=True)
    c["radius_km"] = city_radius_km(c.population.to_numpy(float))

    # The list contains satellites of big metros as separate entries (Ecatepec
    # and Chimalhuacan next to Mexico City, Soacha next to Bogota, Gurgaon next
    # to Delhi). These are overwhelmingly the *poor* peripheries, so letting them
    # claim their own cells strips the low end out of the parent metro and leaves
    # a truncated, disproportionately rich sample -- which silently reverses the
    # parent's gradient. Absorb anything inside the parent's own analysis radius.
    # Iterating largest-first means the biggest city always wins.
    xyz = to_ecef(c.lat.to_numpy(), c.lng.to_numpy())
    tree = cKDTree(xyz)
    absorbed = np.zeros(len(c), dtype=bool)
    for i in range(len(c)):
        if absorbed[i]:
            continue
        for j in tree.query_ball_point(xyz[i], c.radius_km.iloc[i] * ABSORB_FRAC):
            if j > i:
                absorbed[j] = True
    print(f"{len(c):,} cities >= {MIN_CITY_POP:,}; absorbed {absorbed.sum():,} satellites")
    return c[~absorbed].reset_index(drop=True)


def assign_cells(cells, cities):
    """Give every RWI cell to the nearest city that actually reaches it."""
    city_xyz = to_ecef(cities.lat.to_numpy(), cities.lng.to_numpy())
    cell_xyz = to_ecef(cells.latitude.to_numpy(), cells.longitude.to_numpy())
    k = min(4, len(cities))
    dist, idx = cKDTree(city_xyz).query(cell_xyz, k=k)

    owner = np.full(len(cells), -1)
    radii = cities.radius_km.to_numpy()
    for n in range(k):                       # nearest first, so ties go to it
        cand, d = idx[:, n], dist[:, n]
        hit = (owner < 0) & (d <= radii[cand])
        owner[hit] = cand[hit]
    return owner


def main():
    OUT.mkdir(exist_ok=True)
    cities = load_cities()
    cells = pd.read_csv(DATA / "rwi_all.csv")
    print(f"{len(cells):,} RWI cells")

    cells["owner"] = assign_cells(cells, cities)
    cells = cells[cells.owner >= 0]
    print(f"{len(cells):,} cells fall inside a city ({cells.owner.nunique():,} cities)")

    rows, fields = [], []
    for owner, g in cells.groupby("owner"):
        if len(g) < MIN_CELLS:
            continue
        city = cities.iloc[owner]
        x, y = to_local_xy(g.latitude.to_numpy(), g.longitude.to_numpy(), city.lat, city.lng)
        w = np.ones(len(g))                  # see caveat 1 in the module docstring
        fit = fit_wealth_field(x, y, g.rwi.to_numpy(float), w)
        if fit is None:
            continue
        rank = fit.pop("rank")
        fit.pop("z")
        rows.append(dict(city_id=int(city.id), name=city.city_ascii,
                         country=city.country, iso3=city.iso3,
                         lat=city.lat, lon=city.lng, pop=city.population,
                         radius_km=city.radius_km, n_cells=len(g), **fit))
        fields.append(pd.DataFrame(dict(city_id=int(city.id), lat=g.latitude.to_numpy(),
                                        lon=g.longitude.to_numpy(), x=x, y=y, pop=w,
                                        rank=rank, rwi=g.rwi.to_numpy())))

    out = pd.DataFrame(rows).sort_values("pop", ascending=False)
    out["core"] = np.where(out.radial < 0, "rich core", "rich suburbs")
    out.to_csv(OUT / "cities_rwi.csv", index=False)
    pd.concat(fields).to_csv(OUT / "field_rwi.csv", index=False)

    print(f"\n{len(out)} cities -> out/cities_rwi.csv, out/field_rwi.csv\n")
    show = out.head(30)[["name", "country", "compass", "strength", "r2", "radial", "n_cells"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))


if __name__ == "__main__":
    main()
