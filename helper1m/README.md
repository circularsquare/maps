# helper1m

Tool for building 1-million-people-per-region maps. Click admin divisions to see populations, with linear extrapolation from the last two censuses and a shift-click running sum.

## Running

The viewer is a static page that `fetch()`es its GeoJSON, so `file://` won't
work — serve the folder and open it over HTTP:

```
python -m http.server 8000     # from helper1m/, then open http://localhost:8000/
```

Pick a country from the list, click an admin division for its population
(linear extrapolation from the last two data points), shift-click to add to a
running sum.

The country, admin level, ticked regions and map position are remembered per
country in `localStorage`, so a refresh drops you back where you were. So are
the basemap, the density-fill opacity and the sidebar width, which are global.
Untick everything and tick one region to work through a province at a time.

Keys 1-4 switch admin level. Drag the sidebar's right edge to resize it, down
to 100 px, and double-click the edge to reset it. The map does not rotate or
tilt — north stays up, by drag, pinch or keyboard alike.

Hovering a region shows its name, every division it sits inside up to the top
level (town, district, city, province), and its estimated population. The basemap is
Streets (OpenStreetMap), Topo (OpenTopoMap), or Elevation: AWS terrain tiles
coloured blue, green, yellow, orange, red, white, stretched over whatever
elevations are currently on screen, with that range shown under the switch.
Red to white gets a larger share of the range than the lower colours, because
the mountains in any view are a long tail of high elevations. The viewer
needs MapLibre GL JS 5 for the Elevation layer.

## Layout

- `index.html`, `viewer.js`, `style.css` — MapLibre viewer (country-agnostic).
- `countries.json` — index of available countries.
- `countries/<country_id>/` — per-country output: `meta.json` + one `adm{N}.geojson` per admin level, with population timeseries baked into feature properties. A level with too many features to fetch at once is written as `adm{N}/<adm1 code>.geojson` instead and declared `"split"` in `meta.json`; the viewer then loads only the adm1 regions on screen. A country may also have a `composition.json` (below).
- `scripts/build_country.py` — generic: reads shapefile(s) + long-format `population.csv`, writes `adm{N}.geojson` under `countries/<id>/`.
- `scripts/<country_id>/` — country-specific fetchers that produce `data/<id>/population.csv` in the long format `code,level,year,pop`. Every country gets its own fetcher — data sources differ too much to generalize.
- `data/` — gitignored. Raw shapefiles, response caches, canonical `population.csv`.

## Composition pies

A country can ship `countries/<id>/composition.json`, named in its `meta.json` as
`"composition": "composition.json"`. It holds, for each admin unit at whichever
levels the source actually reaches, how its people divide between a fixed set of
groups, plus the mean position of those people. The viewer draws one pie per unit
over its shape, sized by the unit's own population, and adds the top few groups to
the hover tooltip. A level the file has nothing for draws nothing, and a country
without the file never shows the panel.

The panel sits under the basemap controls: a checkbox to draw them, a size slider,
and the list of groups. Clicking a group hides it and the pies renormalise over the
rest, which is how you read a minority pattern under a large majority;
shift-clicking one isolates it. The setting is remembered per country.

```json
{ "label": "Nationality", "year": 2020,
  "groups": [{ "key": "han", "en": "Han", "cn": "汉族", "color": "#8ba0b6" }],
  "levels": { "3": { "<unit code>": { "t": 1000, "x": 104.1, "y": 35.2,
                                      "g": [0, 5], "k": [900, 100] } } } }
```

`g` indexes `groups` and `k` is the count, biggest first. China's is built by
`scripts/china/ethnicity.py` from `maps/chinaethnicity/`.

## Countries

- [china](scripts/china/README.md) — four levels (province, prefecture, county,
  township) on 2016–18 boundaries, all dissolved out of one township shapefile
  so they nest exactly, and the same file asia1m's base map was drawn from.
  2020 township counts are recovered by zonal-summing the ASPECT 100 m grid,
  which is the printed census township table spread over cells; 2010 is
  back-cast from a county census panel and 2024 forward-cast from provincial
  year-end population, so the current-year estimate follows the decline since
  2021 rather than the 2010s growth. Every province matches its published total
  for all three years. Note that a district can read well above the figure
  published against its name, because the census reports development zones as
  their own rows while the land stays legally the district's — Hefei's Shushan
  is 1,047,150 by that reckoning and 1,874,930 by its own boundary. Fetcher:
  `scripts/china/`. Composition: nationality at province, prefecture and county,
  from `maps/chinaethnicity/`, carried onto these boundaries through the same
  ASPECT grid rather than joined by county name, because the two boundary
  vintages disagree about where the urban districts end
  (`scripts/china/ethnicity.py`).
- **india** — three levels (state/UT, district, subdistrict) on 2011-census
  boundaries (SHRUG 2.1 open polygons). No post-2011 census exists, so
  populations come from the IIPS district projections (Dhar 2022, Table 8) at
  five-year steps 2011–2031: states summed from districts, subdistricts scaled
  by their 2011 census share. Fetcher: `scripts/india/fetch.py`.
- [indonesia](scripts/indonesia/README.md) — BPS (main site + 514 regency
  subdomains). Abandoned mid-build — the source site proved too unfriendly, so
  that map was finished by hand instead.

## Adding a country

1. Write a fetcher at `scripts/<id>/fetch.py` that produces `data/<id>/population.csv` with columns `code,level,year,pop` (code = BPS PCODE or similar unique per admin unit, level = 1/2/3).
2. Add boundaries under `data/<id>/boundaries/` or reuse an existing path.
3. Create `countries/<id>/meta.json` with view config.
4. Run `python scripts/build_country.py <id>`. `build_country.py` supports a
   composite `code_col` (a list of columns joined to form the unit code) and a
   `SIMPLIFY_TOL` for geometry simplification.
5. Add the country to `countries.json`.

> Geo scripts (`build_country.py`, the fetchers) need a working
> geopandas/shapely stack. On this machine, run them with `C:\Python39\python.exe`
> — the project venv is broken for geo work.
