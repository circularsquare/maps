# Notices

What the code, the data and the bundled files are licensed under, and what they came from.

## The code

The game's code (TypeScript, Rust and Python in this repository) is licensed under the GNU Affero General Public
License version 3, see `LICENSE`. Copyright (c) 2026 Anita Chen.

The licence covers the code. It does not cover:

- the name "trainsworld" and the logo and preview picture (`trainworld/app/public/favicon.svg`, `preview.png`)
- the four building sounds. They are Pixabay Content Licence files and are not in this repository. The game plays
  without them.
- the privacy page, the terms page, the Ko-fi and Discord links and the credits links. They name the site and its
  owner. If you host a fork, replace them with your own.

## The game's data

The game reads about 6 GB of data (city packs, terrain, outlines, real rail networks) that is not in this repository.
The dev server fetches what it needs from https://data.anita.garden/trainworld/. It is built by the scripts in
`trainworld/pipeline/` from the sources below, and it is offered under the Open Database License (ODbL), the licence
of the OpenStreetMap data inside it: https://opendatacommons.org/licenses/odbl/1-0/. It is not under the code licence.
Keep the credits below with any copy of it.

The JSON and CSV files inside this repository that hold data (for example `trainworld/app/src/game/*.json` and
`trainworld/pipeline/*.json`, `*.csv`) are made from the same sources and are under the ODbL as well, each with the
credit of its source below. `trainworld/docs/data-register.md` lists, country by country, what each one is made from.
Note references in the docs, such as `notes/T-1234.md`, point to working notes that are not published.

All data was adapted for this game. Credits as shown on the game's about page:

- Map data (c) OpenStreetMap contributors, under the Open Database License. https://www.openstreetmap.org/copyright
- Basemap (c) OpenMapTiles, served by OpenFreeMap. Rail lines from OpenStreetMap and national rail registers.
- (c) EuroGeographics for the administrative boundaries.
- Population and built-up areas from GHSL, (c) European Union 1995-2025, and Kontur Population (CC BY 4.0).
- Boundaries from geoBoundaries (Spain's from the Instituto Geografico Nacional, CC BY 4.0), Natural Earth and the
  Office for National Statistics. Contains OS data (c) Crown copyright and database right 2025.
- Contains public sector information licensed under the Open Government Licence v3.0 (ONS, Nomis, Department for
  Education, National Records of Scotland, NISRA) and the Open Government Licence (Ontario).
- Contains information licensed under the Open Government Licence - Toronto. Toronto timetables from the Toronto
  Transit Commission, GO Transit and UP Express (Metrolinx).
- Powered by TfL Open Data. Contains Network Rail Infrastructure Limited data, used under the Open Government Licence
  v3.0. Great Britain National Rail timetables via aubin.app.
- Adapted from Statistics Canada, 2021 Census. This does not constitute an endorsement by Statistics Canada of this
  product.
- Australian Bureau of Statistics, 2021 Census and statistical areas (CC BY 4.0).
- US Census Bureau, NCES, FAA and the Bureau of Transportation Statistics.
- New York subway, LIRR and Metro-North timetables from MTA open data, NJ Transit rail timetables, and PATH
  timetables from the Port Authority of New York and New Jersey. The game is not affiliated with or endorsed by the
  MTA, NJ Transit or the Port Authority.
- Paris timetables (c) Ile-de-France Mobilites, under the Open Database License.
- China township populations from the ASPECT population grid (Ju et al. 2025, Scientific Data), CC BY 4.0.
- INSEE and geo.api.gouv.fr (Licence Ouverte 2.0). (c) Statistisches Bundesamt (Destatis), Zensus 2022. (c) BKG 2025,
  dl-de/by-2-0. Pendleratlas of the statistical offices of the Federation and the Laender.
- Processed from Japan's 2020 census and 2021 economic census (Statistics Bureau of Japan, e-Stat) and National Land
  Numerical Information (MLIT).
- IBGE. Metro de Sao Paulo origin and destination survey 2017. Ministry of the Interior and Safety (Korea). Census of
  India 2011. Shanghai Municipal Statistics Bureau.
- World Bank (CC BY 4.0), Maddison Project Database 2023 (CC BY 4.0), Eurostat, Wikidata (CC0) and OurAirports.
- Large lakes from Natural Earth. Terrain classes from GEDTM30 (OpenGeoHub, CC BY 4.0). Hill shading from Terrain
  Tiles on AWS Open Data.
- Satellite imagery layer: Sentinel-2 cloudless by EOX IT Services GmbH (contains modified
  Copernicus Sentinel data 2024, CC BY-NC-SA 4.0), GSI Japan (Geospatial Information Authority of Japan) and the USGS
  National Map. These are fetched by the page from their own servers and are not part of the data files.

Two items carry conditions beyond attribution. The administrative-boundary fills from EuroGeographics (NUTS) are
non-commercial and non-transferable, so do not republish the data files commercially. The data files also hold a
layer from about 15 national sources of car ownership figures, some with no licence stated. Both are on the
maintainer's list to remove at the next pack rebuild.

## Files in this repository

| What | Where | Licence |
|---|---|---|
| Zen Maru Gothic, Huninn, Hanzi Round (cut from Chill Round M by ChillType), Noto Sans, Nunito | `trainworld/app/public/fonts/` | SIL Open Font License 1.1, texts `OFL-*.txt` beside them |
| Country, region and symbol flags | `trainworld/app/public/flags/` | flag-icons (MIT), circle-flags by HatScripts (MIT), some drawn here, Japanese prefecture and a few other flags from Wikimedia Commons (public domain or CC0). Texts `LICENSE-*` in the folder, details in `trainworld/tools/flags.mjs` |
| RTL text plugin | `trainworld/app/public/rtl/` | BSD-3-Clause (Mapbox), text in `LICENSE-mapbox-gl-rtl-text.txt` |

## Libraries

Installed with npm (`trainworld/app/package-lock.json` pins versions):

| Package | Licence |
|---|---|
| maplibre-gl 6.13 | BSD-3-Clause |
| pmtiles 3.2 | BSD-3-Clause |
| preact 11 and @preact/signals 2 | MIT |
| earcut 3 | ISC |
| intl-messageformat 12 | BSD-3-Clause |
| vite 8 (build) | MIT |
| typescript 7 (build) | Apache-2.0 |

Installed with cargo (`trainworld/sim/Cargo.lock` pins versions): wasm-bindgen, serde, serde_json and dlmalloc, each
under MIT or Apache-2.0, with their own dependencies. Run `cargo about` or `cargo license` in `trainworld/sim` for the
full list.

Fonts, the basemap and the satellite and terrain tiles are fetched by the page at run time from their own servers
under their own terms.
