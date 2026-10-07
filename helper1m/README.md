# helper1m

Tool for building 1-million-people-per-region maps. Click admin divisions to see populations, with linear extrapolation from the last two censuses and a shift-click running sum.

## Running

The viewer is a static page that `fetch()`es its GeoJSON, so `file://` won't
work — serve the folder and open it over HTTP:

```
./serve.sh          # or ./serve.sh 8001; then open http://localhost:8000/
```

It just runs `python -m http.server 8000` from `helper1m/`, wherever you call
it from.

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

`g` indexes `groups` and `k` is the count, biggest first. A group may carry a
`title`, shown on hover in the group list. China's is built by
`scripts/china/ethnicity.py` from `maps/chinaethnicity/`, India's by
`scripts/india/language.py`. Pakistan's, Afghanistan's and Iran's language pies
come from maps/languagedots through one shared builder, `scripts/language_common.py`,
and share one palette, `scripts/language_colors.csv`.

## Countries

- [afghanistan](scripts/afghanistan/README.md) — two levels (province,
  district) on COD-AB v03 boundaries. Figures are NSIA's (now GSIA's) own
  settled-population estimates for the solar years 1403, 1404 and 1405
  (2024–2026), a fixed-rate projection from a 2003-05 household listing, since
  there has been no census since 1979. They leave out the 1.5 million Kuchi
  nomads and read about 17% under the UN's total. NSIA's 457 administrative
  units, including temporary districts and ones created since 2023, are mapped
  onto COD's 401 districts by name. New districts carved from several old ones
  are split back to those districts by what each lost in NSIA's own yearly
  tables. Provinces match NSIA's province table to within 4 people.
  UNFPA/OCHA's COD-PS (48.6 million) is available with
  `fetch.py --source codps`. Fetcher: `scripts/afghanistan/`. Composition:
  language by province only, and a proxy: maps/languagedots' figure, the 2006-07
  MRRD profiles' village-majority language on NSIA's 1404 population, so
  minorities inside mixed villages vanish and Hazaragi is counted as Dari. "Not
  described" is the part each profile leaves out, all of Takhar and Kunduz
  (`scripts/afghanistan/language.py`).
- [bangladesh](scripts/bangladesh/README.md) — four levels (division, zila,
  upazila, union) on the OCHA COD-AB v03 boundaries, whose 507 level-3 units
  are the 495 upazilas and 12 city corporations exactly as the 2022 census
  reports them. 2022 is parsed from the census National Report Volume I and
  matches BBS's zila table exactly; 2011 is rebuilt from the 5,161 unions and
  wards of the 2011 census (USCB transcription), so every 2011 zila and
  division total still comes out exact. Level 4 (4,936 units) takes 2022
  unions from BBS's Union Statistics volume and paurashavas and city wards from
  the National Report, drawn on the 2011 union outlines because no newer ones
  are open; unions split or created since 2011 are grouped with what they came
  from. Five city corporations split by ward, Dhaka North and South by thana
  (Pallabi + Mirpur + Rupnagar, 1.32 million, is the largest unit), Gazipur
  into 7. Both years are enumerated counts. Fetcher: `scripts/bangladesh/`.
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
  by their 2011 census share. Fetcher: `scripts/india/fetch.py`. Composition:
  mother tongue at all three levels, from Census 2011 table C-16, which goes
  down to the sub-district (`scripts/india/fetch_c16.py` downloads it,
  `scripts/india/language.py` builds it). Hindi's varieties (Bhojpuri,
  Rajasthani, Magahi and so on) get their own colours; how people split
  "Hindi" from a variety differs by state, so Bihar shows far more Bhojpuri
  than eastern Uttar Pradesh. The palette is `scripts/india/language_colors.csv`,
  hand-editable; the script only appends rows for new groups.
- [iran](scripts/iran/README.md) — three levels (31 provinces, 429 counties,
  1,049 districts) on the units of the 1395 (2016) census, with counties from
  OCHA COD-AB, whose 429 counties are the census's. 2016 is SCI's census count
  by settlement. 2011 is the 1390 census carried onto the 2016 units village by
  village through the village code both censuses share, so counties split since
  2011 get a 2011 figure. 2024 is SCI's 1403 provincial estimate; inside a
  province each county grows at its province's rate plus half its own 2011-16
  lead, raked back to SCI's total, so provinces are SCI's and the county split
  is a model. Districts are OpenStreetMap's 2026 polygons cut on the county
  lines and regrouped onto the 1395 districts through SCI's 1400 settlement
  file; eight pairs are merged. Tehran's Central District is one unit of 8.7
  million. Fetcher: `scripts/iran/`. Composition: language at home by province
  only, from a survey, not a count: maps/languagedots' World Values Survey 2005
  and 2020 shares (about 4,200 adults, many provinces 10-50 interviews) on the
  2016 population (`scripts/iran/language.py`).
- [indonesia](scripts/indonesia/README.md) — BPS (main site + 514 regency
  subdomains). Abandoned mid-build — the source site proved too unfriendly, so
  that map was finished by hand instead.
- [kazakhstan](scripts/kazakhstan/README.md) — two levels (20 oblasts and
  republican cities, 224 rayons and cities) on the 1 January 2026 map, with
  OpenStreetMap outlines matched to the official KATO classifier. 2025 and 2026
  are the statistics office's own January estimates by rayon, and 2021 is the
  census, carried onto today's rayons village by village where a rayon has been
  split since. So the seven rayons created in 2022–24 have a 2021 figure, and
  Taraz's 2025 takeover of 14 villages is backed out of the 2025 figures. Every
  region matches its published total in all three years. Shymkent is one unit
  because no outline exists for its current five districts. Astana's newest
  district, Saraishyk, is drawn as the part of the city that none of its other
  five districts covers. Fetcher: `scripts/kazakhstan/`.
- [kyrgyzstan](scripts/kyrgyzstan/README.md) — three levels (oblast, rayon/city,
  aiyl aimak/town) on the units of the 2024-25 reform, which halved the aiyl
  aimaks and enlarged Bishkek, Osh and Jalal-Abad (now Manas). Populations are
  the NSC's village-level workbook for the start of 2026, with 2025 and 2024
  from Wayback copies of the same file. Aiyl aimak register figures are scaled
  to the NSC rayon totals, so every level sums exactly. COD's 2018 polygons are
  regrouped onto the new units by matching 2024 villages to 2026 villages by
  name, with GeoNames points and border-plus-growth for the rest; that rule
  reproduced Bishkek's published annexation list. The estimate runs off 2025 to
  2026 on identical units; 2024 is history only. Bare mountain land outside any
  aiyl aimak shows as zero. Licence CC BY-NC-SA 4.0. Fetcher:
  `scripts/kyrgyzstan/`.
- [nepal](scripts/nepal/README.md) — three levels (province, district, palika)
  on OCHA COD-AB boundaries, with the 22 national-park polygons dropped from the
  palika level because the census counts nobody in them. 2021 is the census
  count by palika, matched to the boundaries by name within each district. 2026
  and 2031 are NSO's own projection from that census, which NSO publishes for
  every palika through the API behind its results site, so the current-year
  estimate is NSO's figure. Both sources keep the institutional population
  (barracks, prisons, monasteries; 0.8%) at district level only, and it is
  spread over each district's palikas pro rata, so every district and province
  matches NSO exactly in every year. The projection carries 2016-2021 migration
  forward, so fast-emptying hill palikas keep emptying. Fetcher:
  `scripts/nepal/fetch.py`.
- [pakistan](scripts/pakistan/README.md) — three levels (province/territory,
  district, tehsil) on the 2023 census's own units. PBS's 2023 Table 1 prints
  every tehsil, taluka, sub-division and sub-tehsil with its 2023 count and its
  2017 count re-tabulated on the same unit, so both years are census counts on
  one boundary set and every province matches PBS's headline total (241.5
  million in 2023). Tehsil polygons are OpenStreetMap's 2023 tehsils, holes
  filled from OCHA's COD-AB. Where OSM and the census split a district
  differently, units are merged to the common piece rather than split:
  Balochistan sub-tehsils are folded into their tehsil, and Karachi's seven
  districts are one unit each at tehsil level, because its sub-divisions match
  no published lines. Azad Kashmir (32 tehsils) and Gilgit-Baltistan (10
  districts) come from their governments' reprints of the same two censuses, on
  COD's lines. Fetcher: `scripts/pakistan/`. Composition: mother tongue from
  2023 census Table 11 at tehsil level for the four provinces and Islamabad,
  joined by census unit through the same crosswalk as the population
  (`scripts/pakistan/language.py`). Table 11 leaves out the 1.04 million
  counted by head only, most of them in Quetta, Islamabad and Rawalpindi.
  Gilgit-Baltistan and Azad Kashmir are not in Table 11 and draw no pie
  (languagedots' models of them are behind `MODELLED_NORTH`, off).
- [russia](scripts/russia/README.md) — two levels (federal subject, municipal
  district / okrug) on OpenStreetMap's municipal boundaries of October 2026,
  with Moscow's 12 administrative okrugs and St Petersburg's 18 districts
  standing in for their small intra-city municipalities. Populations are
  Rosstat's municipal tables for 1 January 2024 and 2025, fetched through the
  Wayback Machine because rosstat.gov.ru fails TLS; 2024 is carried onto the
  2025 units by name, since about 190 districts became okrugs or merged that
  year. OSM is a year newer than the table, so 24 polygons hold two or three
  Rosstat units, mostly Krasnoyarsk towns folded into their districts in 2025,
  and Nizhny Novgorod includes Kstovsky. Subjects are the sum of their units and
  match Rosstat's subject rows; 2021 is the census, by subject. Internationally
  recognised borders: Crimea and Sevastopol are a switch in `fetch.py`, off by
  default. Fetcher: `scripts/russia/`. Composition: nationality (ethnic group)
  at both levels from the 2021 census by settlement (tochno.st, CC BY), carried
  onto today's municipalities by settlement coordinates; 11.6% "not stated",
  most of it in the big cities (`scripts/russia/ethnicity.py`).
- [srilanka](scripts/srilanka/README.md) — three levels (province, district,
  Divisional Secretariat division) on the 340 DS divisions of the 2024 census,
  drawn from the census department's own DS polygons on ArcGIS Online, with
  districts and provinces dissolved from them. 2024 is the census count by GN
  division summed to DS; 2012 is the 2012 census by DS division (Table A1 of the
  district reports), carried onto today's units by name. Nine DS divisions
  created since 2012 (five in Nuwara Eliya, three in Galle, Kaltota in
  Ratnapura) get their parent's 2012 count split by 2024 shares, so they show
  the parent's growth. Every district matches the census in both years. GN
  divisions (14,000) are not drawn: 2012 has no GN table, only printed maps.
  Fetcher: `scripts/srilanka/`.
- [tajikistan](scripts/tajikistan/README.md) — two levels (5 regions, 65 cities
  and districts), the units the Agency on Statistics publishes every year.
  Boundaries are OSM (Geofabrik, October 2026) with the current district names.
  Bokhtar, Khorugh and Istiqlol have no OSM boundary and are cut out of their
  districts from their built-up outlines, which for Bokhtar is too tight. Most
  cities include the rural district they absorbed, drawn with the old district
  outline. Populations are the 1 January bulletin for 2021-2025, read from the
  PDFs' text layer, with two typos fixed. 2010 and 2020 are census counts, left
  out for Dushanbe and Rudaki because part of Rudaki went to Dushanbe after the
  2020 census. Dushanbe is one unit. Every region and the national total match
  the bulletin within rounding. Fetcher: `scripts/tajikistan/`.
- [turkmenistan](scripts/turkmenistan/README.md) — two levels (5 velayats,
  Ashgabat and Arkadag; 53 etraps and cities) on OpenStreetMap's October 2026
  map, which already has the five etraps created since the census. 2022 is the
  census (17 December 2022, 7,057,841), read settlement by settlement from
  section 1's tables and carried onto today's etraps by placing each of its
  1,810 towns and villages on the OSM village of the same name; 3.7% of people,
  in settlements OSM lacks, are spread by Kontur. Every velayat matches the
  census. No figure by velayat or etrap has been published since, so 2026 is
  the census times the UN's national growth (WPP 2024, +6.1%), the same for
  every unit: the estimate keeps the census's proportions. Arkadag had 567
  people at the census and its figure is far too low. Fetcher:
  `scripts/turkmenistan/`.
- [uzbekistan](scripts/uzbekistan/README.md) — two levels (14 regions, 199
  districts and cities) on OCHA COD-AB 2018b boundaries, joined to the
  Statistics Committee's SIAT district series (permanent population, 1 January,
  2011–2026) by SOATO code. Seven newer districts and the 2020–23 transfers
  between units are handed back to the 2018 polygons, so Tashkent city is its
  2018 extent and reads about 132,000 below its census figure. Each region's
  districts are scaled in every year by the 2026 census/SIAT ratio
  (`CENSUS_LEVEL` in `fetch.py`), so the level is the census's and the trend
  SIAT's; Tashkent region alone is 1.19x, and its suburbs probably carry more
  of that than the flat factor gives them. COD's city polygons are small and
  often a few km off the town, so click a city together with its district.
  Fetcher: `scripts/uzbekistan/`.

## Adding a country

1. Write a fetcher at `scripts/<id>/fetch.py` that produces `data/<id>/population.csv` with columns `code,level,year,pop` (code = the boundary file's unit code as a string, level = 1/2/3...). Each unit needs at least two years, since the estimate is a line through the last two.
2. Add boundaries under `data/<id>/boundaries/` or reuse an existing path.
3. Describe them in `scripts/<id>/boundaries.json`, keyed by level, with the
   same fields as the built-in `SHAPEFILES` entries in `build_country.py`
   (`candidates` relative to the repo root, `code_col`, `name_col`,
   `parent_col`, `group_col`, and optionally `extra_cols`, `simplify`,
   `split_by`). China, India and Indonesia predate this and still live in
   `SHAPEFILES`. `code_col` may be a list of columns joined to form the code.
4. Create `countries/<id>/meta.json` with view config.
5. Run `python scripts/build_country.py <id>`.
6. Add the country to `countries.json`.

> Geo scripts (`build_country.py`, the fetchers) need a working
> geopandas/shapely stack. On this machine, run them with `C:\Python39\python.exe`
> — the project venv is broken for geo work.
