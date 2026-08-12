# citybrowser — TODO

Interactive world city map, ~11k bubbles, hover card per city.
Part scraped, part hand-curated over months.

## Decided

- Scope: all ~11k GHS-UCDB urban centres. Hand-added historical / isolated cities later.
- Two-layer DB: `base.json` (generated, never hand-edited) + `overrides.json`
  (field-level hand patches, each recording the value it replaced) → `cities.json`.
- Own stable ids (`c00001`), crosswalk to GHS id + Wikidata QID so ids never shift.
- Edit mode is local: `serve.py`, open map at `?edit=1`, PATCH writes `overrides.json`.
  Curation lives in git. No hosting, no auth.
- Untouched cities render gray, so progress is visible on the map itself.
- No lead paragraph, no page thumbnail. Facts are read and written by hand.
- Not optimising for speed anywhere. Slow and polite is fine.

## Open questions

## GDP per capita: settled, but coverage is thin

**Hard pipeline rule: never divide a source's GDP by GHS population.** Always
take GDP *per capita* directly from one consistent boundary. Mixing numerator
and denominator across definitions gives 2–4x errors on merged Chinese/Indian
blobs and on fragments like Fort Worth. Within one boundary the error is 5–25%
and defensible. The build should refuse to compute a ratio across sources.

Fallback chain, all CC BY 4.0:

1. **OECD Functional Urban Areas** — 562 FUAs, 33 countries, GDPpc direct in
   USD PPP, latest 2020–2023. SDMX bulk CSV:
   `https://sdmx.oecd.org/public/rest/data/OECD.CFE.EDS,DSD_FUA_ECO@DF_ECONOMY,/all?format=csvfile`
2. **Eurostat metro regions** (`met_10r_3gdp`, ~243) then **NUTS3**
   (`nama_10r_3gdp`, 1,343) — per-inhabitant direct. DBnomics mirrors it.
3. **OECD TL3** (2,612 small regions) / **DOSE v2.14** (1,667 admin-1 regions,
   83 countries, Zenodo 20035157) — best reported non-modelled global set.
4. **Kummu et al. 2025** — note this ships **GPKG/CSV at admin-2, 43,501 units**
   (Zenodo 13943886), not just a raster. Join UCDB centroid to the admin-2
   polygon; that sidesteps the raster-footprint ambiguity entirely. This is the
   global backstop.

**Coverage reality:** ~900 centres confirmed today, 2,000–3,500 optimistic, out
of 11,686. OECD countries hold only 1,685 centres (14.4%). China (1,974) and
India (1,925) are 33% of the roster with essentially no dedicated city GDP, and
nor do Nigeria (425), Pakistan (299), Bangladesh (239), Egypt (214), DRC (189).
So GDP is a sometimes-field. The card must look right with it absent.

### National statistical offices: surveyed, and mostly not worth it

The dedicated-source path has a sharp cost curve. Where to stop:

**Worth doing — one download each, genuinely sub-provincial:**
- **Brazil** IBGE PIB dos Municípios — 5,570 municípios, 2023, per capita
  direct, **no API key**. `apisidra.ibge.gov.br/values/t/5938/n6/all/v/37/p/2023`
  (table 5938; table 21 is discontinued). Bulk is over `ftp://`, not https.
- **Indonesia** BPS PDRB — 514 kabupaten/kota, **2025** (6-month lag, fastest
  anywhere), per capita direct, free key. Note `www.bps.go.id` 403s fetchers;
  use `webapi.bps.go.id`.

**Not worth it for this project:**
- **China** — the City Statistical Yearbook is a ¥358 print book, no free
  download. `data.stats.gov.cn` carries GDP for just **36 cities** and 403s from
  outside the mainland. The real route is **31 provincial bureau scrapers**.
- **Japan** — 市町村民経済計算 is not published nationally (0 hits on e-Stat).
  47 prefectural sites, Excel/PDF, one prefecture suspended since 2018.
- **India** — no city or municipal GDP from any official source, at all. Only
  26 of 36 states compile district product, with differing methodologies.
- **Korea** 17 provincial KOSIS orgIds; **Mexico** has no municipal GDP (VACB is
  establishment-based, not GDP); **Nigeria** has 22 states, 2013–2017 only.
- Turkey / Russia / South Africa / Vietnam are **province-level** — no finer
  than what OECD/DOSE already give us.

**⚠️ US changed under us:** BEA **discontinued all metro-area statistics** with
the Feb 2026 release. `MAGDP*` now error. Only GDP-by-county survives (3,127
units), per capita is no longer published, so an MSA figure means aggregating
counties over an OMB CBSA crosswalk and dividing. OECD FUA already covers US
metros with per capita direct — use that instead and skip BEA.

**So the GDP plan is: OECD FUA + Eurostat met (~700 rich-world metros, two
CSVs) → optionally Brazil + Indonesia later → Kummu admin-2 for everything
else, flagged as modelled.** Do not build country scrapers for this.

**Do not use the Oxford Economics GCFS sample** (900 cities) found in a
third-party GitHub repo — it is proprietary IP redistributed with no licence
grant. Brookings GMM18 (300 metros, 2016) states no licence either and its
underlying data is also Oxford Economics; cross-check only, never ship.

**Wikidata is not a GDP source**: only 712 items carry P2131 at all, of which
**6** are instances of city. That path is dead.

**GHS-UCDB ships its own GDP** (`SC_GDP_SUM_*`, from Kummu) — convenient, but
it is national GDP redistributed over grids, so it misses the urban
productivity premium: it puts Bangkok above London. Same shape of problem as
population. Fine as a consistent fallback, wrong as a displayed figure.

## GHS re-export: done

Fetched directly from the JRC open-data FTP — no manual download needed. The
themed sub-packages are small and public; the 1.69 GB GeoPackage is only needed
if we ever draw footprints.

```
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A/
  GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A/V1-0/...zip               7 MB
  GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A/V1-0/...zip 6 MB
  GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A/V1-2/...zip          31 MB
```

- Mojibake **gone**: 0 bad names in the fresh export vs 329 in the local copy.
- Comma-delimited, real UTF-8, no thousands-separator trap.
- **Coordinates found** in the GeoPackage's `UC_centroids` table as
  `PWCentroidX/Y` — *population-weighted* centroids, better than geometric.
  ESRI:54009 Mollweide metres, reprojected to WGS84 → `data/ucdb_centroids.json`.
  Validated: Quito 4.8 km, Reykjavík 4.7 km, Tokyo 3.7 km from city centre.
  Guangzhou is 44.9 km off, which is the merge blob doing exactly what we
  expect. Confirms the plan: GHS centroid for **matching**, Wikidata P625 for
  **display**.

**Schema changed — do not mix with the old copy.** Fresh export keys on
`ID_UC_G0` with 11,422 rows; `cityhistory/data/ghs_ucdb.csv` keys on
`ID_MTUC_G0` with 11,686. Different vintage, different ids.

- [ ] Gitignore `citybrowser/data/` — source zips and derived intermediates,
      all regenerable. See [[project_maps_untracked]].

## Population: GHS-UCDB is NOT the displayed number

Checked against the raw file. It merges and splits inconsistently, so there is
no offset a reader could learn (ratios vs Wikipedia swing 0.6x–3.2x):

| GHS row | pop | area km² | what it actually is |
|---|---|---|---|
| "Guangzhou" | 42,987,704 | 6454 | Shenzhen + Guangzhou + Foshan + Dongguan + … |
| "New Delhi" | 31,422,508 | | Delhi + Ghaziabad + Gurugram + Noida |
| "Fort Worth" | 102,173 | 60 | a fragment |
| "North Richland Hills" | 523,625 | 268 | ← the actual Fort Worth blob |
| "Atlanta" | 602,708 | 292 | dense core only, metro is ~6M |
| Hong Kong | 4,807,599 | | split 3 ways, official is 7.5M |

**UN WUP 2025 is not an escape hatch.** As of the Nov 2025 revision the UN
dropped national administrative definitions and adopted the GHSL Degree of
Urbanisation rule (contiguous ≥1500/km², total ≥50k) — the *same* rule GHS uses.
So WUP 2025 and GHS-UCDB are the same family, not independent sources, and
`cityhistory/data/stadester/wup2025.json` inherits that. UN WUP 2025 puts New
York at 13.9M, down from 18.8M in WUP 2018, against an MSA of 19.9M.

**Resolution: separate the two jobs, because they have opposite requirements.**

- **Bubble size** needs cross-city *consistency*. Nobody eyeballs a bubble and
  says "that's 6M too small." → **GHS urban centre.**
- **The displayed number** needs to match what the reader would find if they
  googled it. → **Wikidata/Wikipedia, with its definition stated.**

So the card shows a small labelled set — city proper, metro, urban centre —
which is what a Wikipedia infobox does anyway, and no single figure has to carry
the weight. GHS's member-city list goes on hover so "Guangzhou 43M" explains
itself. Any of them can be overridden.

GHS keeps three jobs it is genuinely good at: the 11,686-row **roster**,
**area/density**, and the **1975–2030 recent history**. Merge/split/relabel
mistakes get fixed through the override layer as she curates.

- [ ] Re-export GHS from source — the local CSV is mojibaked (329 names with
      `?`/`�`: `Klaip?da`, `Timi?oara`) **and** has no coordinates. One fresh
      download fixes both. Note: raw values use `.` as thousands separator.

## Climate: settled

**CHELSA V2.1, 30 arc-sec, 1981–2010.** Beats WorldClim on every axis that
matters here: CC0 vs CC BY-NC-SA, 1981–2010 vs 1970–2000, better precipitation,
and half the bytes at the same resolution.

30 arc-sec rather than ~5 km because the whole content of the chart is the band
height, and a 5 km pixel in steep terrain carries ~1.6–3.3 °C of terrain error
vs ~0.4–0.8 °C at 1 km. La Paz, Quito, Kathmandu would be visibly wrong.

```
https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/{var}/CHELSA_{var}_{MM}_1981-2010_V.2.1.tif
```
`var` = tasmin / tasmax / pr · `MM` = 01–12 · tasmin+tasmax ≈ 2.9 GB, pr ≈ 2.9 GB
Scaling: °C = `v * 0.1 - 273.15` · mm = `v * 0.01`

Try `/vsicurl/` range reads first (we need ~0.003% of the bytes); fall back to
downloading tasmin+tasmax only if that thrashes.

Caveat to carry onto the card: tropical-mountain precipitation is unreliable in
both datasets (~50% underestimate of peaks). Temperature is sound.

## Smoke test: passed, after one real failure

First attempt hung: 10 minutes, zero countries written. Cause was the type
filter `?city wdt:P31/wdt:P279* wd:Q486972`. Measured on Gabon (26 settlements):

| query shape | time | rows |
|---|---|---|
| `P31/P279*` subclass walk | **502 @ 43s** | died |
| no type filter | 0.6s | 26 |
| + aggregation | 0.8s | 25 |
| + label service | 0.8s | 25 |
| explicit `P31` VALUES list | 3.1s | **only 9** |

So the walk is ~70x slower, and the "cheap" enumerated-type alternative
silently drops two thirds of the data — a quiet correctness bug, not a loud
one. **No type filter.** The label service is free *because* it wraps an
aggregated subquery. `diag_query.py` re-runs this if WDQS behaviour shifts.

Re-run: 5 countries, 764 settlements, ~6s total query time. Field coverage —
name 100%, pop 100%, coords 100%, admin 96%, **elevation 16%**.

**Consequence of dropping the type filter:** the pool contains non-settlements
— "West Africa" (429M), "Guinea" the country, "Kindia Region". Mostly harmless,
but a region's coordinate can sit near its namesake city and shadow it. Fix
costs nothing: stage 3 calls `wbgetentities` for aliases anyway, and that
returns P31, so **filter by type locally there** — zero extra requests.

## Altitude: use GHS, not Wikidata

Wikidata P2044 is only **16%** populated, useless for a per-city colour scale.
`GE_ELV_AVG_2025` in the GEOGRAPHY theme is **100%** (11,422/11,422) and
validated against ten known cities — La Paz 3868 vs 3640, Quito 2801 vs 2850,
Lhasa 3658 vs 3656, Amsterdam 1 vs 2. All within tolerance.

Same theme also carries `GE_ECO_CLA_2025` (ecoregion, e.g. "Samoan tropical
moist forests") and soil class — possible extra card fields, free.

## Build

- [x] `fetch_wikidata.py` — reviewed, rewritten, smoke-tested. Ready for the
      full run (~258 countries).
- [ ] `fetch_geonames.py` — alt-name candidate pool (not auto-selected; feeds the edit UI)
- [ ] `sample_climate.py` — monthly tmin/tmax/prec at each city point
- [ ] `build.py` — merge base + overrides → `cities.json`, flag stale overrides
- [ ] `serve.py` — local edit server
- [ ] `index.html` — map, bubbles, hover card

## Card fields

name · up to 3 alt names · country · subdivision · coords · altitude ·
population + sparse history (log line) · gdp per capita · languages (pie) ·
climate band (monthly min–max) · up to 3 facts · photo slot · religions (maybe)

## Colour

Coloured text throughout, and the same value gets the same colour in every city.

- **languages** — fixed colour per language, identical across all cards
- **altitude** — hypsometric: green → yellow → orange-red → pink / pale grey
- **gdp per capita** — dark blue → red rainbow
- **population** — green (low) → yellow (mid) → orange → red → purple (highest)

House style is `riders/nycriders/index.html`. No transitions.
