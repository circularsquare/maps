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

## Layout

- `index.html`, `viewer.js`, `style.css` — MapLibre viewer (country-agnostic).
- `countries.json` — index of available countries.
- `countries/<country_id>/` — per-country output: `meta.json` + one `adm{N}.geojson` per admin level, with population timeseries baked into feature properties.
- `scripts/build_country.py` — generic: reads shapefile(s) + long-format `population.csv`, writes `adm{N}.geojson` under `countries/<id>/`.
- `scripts/<country_id>/` — country-specific fetchers that produce `data/<id>/population.csv` in the long format `code,level,year,pop`. Every country gets its own fetcher — data sources differ too much to generalize.
- `data/` — gitignored. Raw shapefiles, response caches, canonical `population.csv`.

## Countries

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
