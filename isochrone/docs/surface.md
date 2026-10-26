# Our walking surface

Today's off-road walking speed, built by us at 1 km from primary layers with the Malaria Atlas
Project's recipe (research/friction.md 1). It is the base later stages swap layers into: roads
(T-010), 10,000 BC (T-012), 90 m terrain (T-011). The page does not use it yet (T-028); it
still reads MAP's own surface. Agreement with MAP: notes/T-009.md.

## Grid

MAP's: 30 arc-seconds, 85 N to 60 S, 43200 x 17400, EPSG:4326. Every layer below is one
GeoTIFF on this grid in `data/surface/` (gitignored), tiled 512 x 512, deflate.

## Inputs

| Input | Where | Licence |
|---|---|---|
| Copernicus Global Land Cover LC100 v3, 2019, discrete classes, 100 m (1/1008 degree), 80 N to 60 S | `data/raw/lc100/cgls-lc100-2019-cog.tif` (2.4 GB), a COG copy at https://s3-west.nrp-nautilus.io/public-land-cover/cgls-lc100-2019-cog.tif of Zenodo 10.5281/zenodo.3939050 | CC BY 4.0 |
| GMTED2010 mean elevation, 30" | `data/raw/gmted/mn30_grd/` (ESRI grid, from USGS `mn30_grd.zip`, 277 MB) | public domain |

## Land cover speed

| LC100 class | km/h |
|---|---|
| 111, 113 closed needleleaf; 115, 116 closed mixed or unknown | 3.24 |
| 112 closed evergreen broadleaf | 1.62 |
| 114 closed deciduous broadleaf | 4.00 |
| 121, 123, 125 open needleleaf or mixed forest | 4.00 |
| 124 open deciduous broadleaf, 126 open forest unknown | 4.86 |
| 122 open evergreen broadleaf | 3.00 |
| 20 shrubs | 4.20 |
| 30 herbaceous | 4.86 |
| 40 cultivated | 2.50 |
| 50 urban | 5.00 |
| 60 bare or sparse | 3.00 |
| 70 snow and ice | 1.62 |
| 90 herbaceous wetland | 2.00 |
| 100 moss and lichen | 4.86 |
| 80 inland water, 200 sea | water |
| 0 unknown, 255 none | left out |

MAP's survey values. Classes MODIS has no match for take the speed MAP mostly gives the same
pixels (shrubs as open shrubland, open broadleaf as savanna, moss and lichen as grassland);
open needleleaf and open evergreen broadleaf are guesses. The table lives in
`tools/surface/common.py` (`LC_SPEED`). A 1 km pixel covers 8.4 x 8.4
LC100 pixels; each 100 m pixel counts by its exact area overlap. The land part's speed is
the area-weighted mean of time per metre (0.06 / km/h) over its land pixels, turned back into
km/h: the crossing time of the land is kept. Water shares are kept apart, so wetland (90) is
slow land and never water.

## Terrain

GMTED's grid sits half a pixel off MAP's, so elevation is resampled bilinearly. Slope is the
angle from the four neighbours (as Earth Engine's `ee.Terrain.slope`), sea as 0 m. Then MAP's
formulas, unchanged:

    elev_factor  = 1.016 * exp(-0.0001072 * elevation_m)
    slope_factor = min(6 * exp(-3.5 * |tan(slope) + 0.05|), 5) / 5
    speed        = landcover_kmh * elev_factor * slope_factor

## Water and the far north

A pixel is water when inland water and sea together cover half of it or more (`WATER_MIN` in
`combine.py`). Where LC100 has no class (north of 80 N, and class 0 over Svalbard and some
Arctic islands) land is where GMTED is above 0 m, walked as snow and ice (1.62 km/h).

No roads in this surface: MAP puts every road pixel at 5 km/h, our roads come as their own
layer (T-010).

## Outputs (`data/surface/`)

| File | Type | What |
|---|---|---|
| `lc_speed.tif` | uint16 | land part's land cover speed, km/h x 1000; 0 = no land |
| `lc_water.tif` | uint8 x 2 | band 1 inland water share, band 2 sea share, x 250 |
| `lc_class.tif` | uint8 | LC100 class at the pixel centre, for checks |
| `elev.tif` | int16 | metres; sea 0, -32768 north of 84 N |
| `slope.tif` | uint16 | slope degrees x 100 |
| `terrain_factor.tif` | uint16 | elev_factor x slope_factor x 10000 |
| `walk_today.tif` | uint16 | off-road walking km/h x 1000; 0 = water or no land |

uint16 rather than the page's uint8 x 40 because steep high ground goes well under 0.1 km/h,
where steps of 0.025 km/h would be 25% errors.

## Rebuild

Plain `python` from the maps root (C:\Python39), in order:

| Step | Time |
|---|---|
| `python isochrone/tools/surface/elevation.py` | ~4.5 min |
| `python isochrone/tools/surface/landcover.py [workers]` (claim cores first, `cores.py`) | ~25 min on 4 |
| `python isochrone/tools/surface/combine.py` | ~4 min |
| `python isochrone/tools/surface/compare.py` (check against MAP, PNGs to `maps/artifacts/isochrone/t009_*.png`) | ~4 min |

Changing the speed table needs landcover.py again; the factors or the water rule only
combine.py. Outputs take 1.4 GB; inputs kept are LC100 (2.3 GB) and GMTED (0.4 GB).
