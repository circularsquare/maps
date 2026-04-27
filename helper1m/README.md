# helper1m

Tool for building 1-million-people-per-region maps. Click admin divisions to see populations, with linear extrapolation from the last two censuses and a shift-click running sum.

## Layout

- `index.html`, `viewer.js`, `style.css` — MapLibre viewer (country-agnostic).
- `countries.json` — index of available countries.
- `countries/<country_id>/` — per-country output: `meta.json` + one `adm{N}.geojson` per admin level, with population timeseries baked into feature properties.
- `scripts/build_country.py` — generic: reads shapefile(s) + long-format `population.csv`, writes `adm{N}.geojson` under `countries/<id>/`.
- `scripts/<country_id>/` — country-specific fetchers that produce `data/<id>/population.csv` in the long format `code,level,year,pop`. Every country gets its own fetcher — data sources differ too much to generalize.
- `data/` — gitignored. Raw shapefiles, response caches, canonical `population.csv`.

## Countries

- [indonesia](scripts/indonesia/README.md) — BPS (main site + 514 regency subdomains)

## Adding a country

1. Write a fetcher at `scripts/<id>/fetch.py` that produces `data/<id>/population.csv` with columns `code,level,year,pop` (code = BPS PCODE or similar unique per admin unit, level = 1/2/3).
2. Add boundaries under `data/<id>/boundaries/` or reuse an existing path.
3. Create `countries/<id>/meta.json` with view config.
4. Run `python scripts/build_country.py <id>`.
5. Add the country to `countries.json`.
