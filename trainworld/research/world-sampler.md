# World sampler: London, Berlin, Paris, Shanghai, Mumbai

T-286, 2026-10-13. A plan for Anita's review: which demand inputs carry worldwide, which need a
country's own data, and the order to build the five cities in. Nothing built or downloaded. Sources
were checked on their download or catalogue pages where marked "checked"; the rest is from earlier
research (research/demand-data.md, research/education-data.md) or memory, marked "not checked".
Low fidelity throughout, as Anita asked: an input with no easy source is left out, not hunted.

## 1. Every input, every city

Key: **G** a global source works; **N** the national census or register has it cheaply; **out**
left out at low fidelity; **sub** needs a substitute. "Tier 0" is the global-only adapter of
section 2.

| Input | London | Berlin | Paris | Shanghai | Mumbai |
|---|---|---|---|---|---|
| Population, small grain | **N** Census 2021 population by output area (~300 people), Nomis TS001, open (OGL); OA polygons from ONS Open Geography | **N** Zensus 2022, residents per 100 m grid square, open with source credit (checked) | **N** INSEE Filosofi 200 m squares (2019; 2021 out Feb 2026), open, Licence Ouverte (checked) | **G** WorldPop R2025A 100 m (2015-2030, CC BY 4.0, checked) or GHSL POP, scaled to the 16 districts' 2020 census totals (Shanghai Statistics Bureau bulletin, PDF, checked) | **G** the same grid. Census 2011 is 15 years old (ward totals for Greater Mumbai exist on OpenCity, public domain, checked, but no official ward polygons); the 2027 census is not out |
| Jobs at work, small grain | **N** BRES employee jobs by LSOA (~1,500 people), Nomis open access, rounded (checked). Employer survey, so no lockdown problem | **N** for the ring only: Federal Employment Agency (BA) insured employees at work per municipality, open. Berlin is one municipality, so inside it **sub**: tier 0's split | **N** jobs at place of work by commune (Paris by arrondissement), census, open; not published below commune (checked). Inside a commune tier 0's split | **sub** tier 0 (GHSL non-residential, OSM places, population). District employment totals from yearbooks optional | **sub** tier 0. Informal and home-based work is most of Mumbai's, so keep a large population share in the split |
| Commute flows to fit the curve | **N** Census 2011 flows MSOA to MSOA (WU01EW, open, pre-pandemic; not checked this pass). 2021's (Nomis, MSOA, open, checked) only as a lockdown check | **N** Pendleratlas flows between municipalities (statistical offices, interactive; checked) and BA flows between districts (xlsx, checked). Berlin being one municipality makes the inner fit weak: take Paris's or London's curve and check it on Berlin-Brandenburg flows | **N** INSEE mobilités professionnelles 2022, commune to commune (Paris by arrondissement), open (checked). The same fit as Toronto's | **sub** no open flows. Fit the curve's length to a published mean commute: 10.2 km, Shanghai's 7th comprehensive transport survey (checked, via press); CAUPD-Baidu's yearly commute report covers ~45 Chinese cities | **sub** no flows. Census 2011 table B-28: "other workers" by distance band (0-1, 2-5, 6-10, 11-20, 21-30, 31-50, 51+ km) and mode, Maharashtra's urban rows (checked). Fit the curve to that distance distribution |
| Border factor | none needed (no land border in reach) | none needed: Poland is 80 km out. Berlin is a city-state: whether the Berlin-Brandenburg line cuts commuting can be read off the Pendleratlas later. Factor 1 at first | none needed | none needed (province lines factor 1) | none needed |
| Work from home | **N** one share, ONS (Labour Force Survey or Opinions and Lifestyle), at both ends as Toronto | **N** one share: Eurostat LFS "usually works from home" by country (lfsa_ehomp; a NUTS-2 version exists, code not confirmed) | **N** the same Eurostat share. The census counts teleworkers at their workplace (only ~4% "no travel" in RP 2022, checked), so no lockdown correction | **out** (small; set 0) | **out**. Home-based workers already make no trip in B-28 ("no travel") |
| Everyday destinations | **G** OSM shops, services, food, venues per cell (Toronto's code, from the Geofabrik extract) | **G** OSM | **G** OSM | **G** OSM, but thin: the Shanghai extract is 25 MB (checked). Overture Places (CDLA permissive) could top it up; its China coverage not checked | **G** OSM, thin outside the main streets; same note |
| Schools with enrolment | **N** Get Information about Schools: every school in England with pupils and coordinates, open (OGL; fields not opened this pass) | **sub** schools are per state in Germany, so tier 0: OSM schools sized by building area | **N** Education ministry's school directory with coordinates (checked) plus pupils per school (primary, collège, lycée files; checked in part), Licence Ouverte; join on school code | **sub** tier 0 | **sub** tier 0. UDISE+ has every school with coordinates (an India Data Portal copy, checked) but enrolment is a separate table behind the portal: later upgrade |
| Universities | **N** HESA enrolments by provider, CC BY (checked); one point per provider as IPEDS does | **N** students per institution, Destatis GENESIS 21311 (not checked); else tier 0 | **N** higher-education ministry's students by institution and site, open (not checked); else tier 0 | **sub** tier 0: OSM campuses, Wikidata student counts | **sub** tier 0 |
| Airports and passengers | **G** OurAirports + Wikidata patronage (the existing fallback). Heathrow connecting share by judgement as Pearson's (0.75) | **G** same; BER 1 | **G** same; CDG 0.75 | **G** same; CAAC's yearly airport bulletin has every airport (Pudong 76.8M in 2024, checked via press) | **G** same; AAI's monthly traffic tables. Navi Mumbai's new airport needs a current figure |
| Medical | **G** OSM (existing script, from the extract) | **G** OSM | **G** OSM | **G** OSM; hospitals are huge outpatient sites in China, beds from area is fine | **G** OSM; thin, leave as is |
| Parks | **G** OSM, English names work | **G** OSM; add "Park", "Volkspark" to the reserve name rule or reserves at a quarter | **G** OSM; "parc", "bois", "jardin" | **G** OSM; "gongyuan" (公园) | **G** OSM; "udyan", "garden", "maidan" |
| Institutions | **out** (UK counts communal establishments by OA, cheap later) | **out** | **out** | **out** | **out** |
| Routed walks and drives | **G** routes.py, Geofabrik England or Greater London | **G** Berlin + Brandenburg | **G** Ile-de-France | **G** Shanghai (+ Jiangsu, Zhejiang if the extent reaches them); check snapping distance | **G** India western zone (210 MB, checked); expect long snaps in informal settlements |
| Water mask | **G** OSM water (T-041) | **G** OSM | **G** OSM | **G** OSM; the Huangpu, the Yangtze estuary | **G** OSM; creeks and mangroves (mangroves as water, judgement) |
| Regional market and non-city zones | ITL3 (ONS, 182 units; NUTS stopped updating UK regions after Brexit, checked) | NUTS-3 Kreise (Berlin is one, DE300) | NUTS-3 départements (Ile-de-France is 8) | county-level units (districts and counties, 300k-1.5M people). geoBoundaries ADM3 for China exists; its licence not confirmed (checked) | sub-districts (taluka); districts are 2-11M people, too big. Census 2011 sub-district polygons (the SHRUG copy used before) |
| World frame | a Europe world: one conic tangent near 50.7 N covers London, Paris and Berlin (48.9-52.5 N) with under 0.1% scale error | same world | same world | its own world near 31 N | its own world near 19 N |

At low fidelity every city starts outside any regional market (`market: -1`, as Toronto did).

## 2. The general methods

**Global for every country** (all free; licences in section 4):

- **Population**: GHSL GHS-POP or WorldPop R2025A at 100 m, scaled to the best admin totals the
  country publishes (any census has them at district level). Kontur (H3 resolution 8, ~400 m) is
  too coarse for cells but stays the outside world's source.
- **Jobs proxy (T-012)**: GHSL non-residential built-up surface (100 m, 1975-2030, CC BY 4.0, on
  AWS and Earth Engine, checked), blended with OSM job places (Toronto's 80/20 rule inside a tract
  already uses OSM places) and population. Totals come from population times the country's
  employment rate, since the gravity is balanced and only the spread matters. Where informal work
  is large (Mumbai) the population share is larger.
- **Everyday destinations, medical, parks, routes, water**: OSM, read from Geofabrik extracts
  (Overpass was down for Toronto and is no less flaky elsewhere). Parks need the reserve-name words
  per language.
- **Schools**: OSM schools sized by building area, with pupils = children (WorldPop's age grids)
  x a national enrolment rate (UNESCO), spread over the schools by area.
- **Universities**: OSM campuses, sized by Wikidata's student count (P2196) where present, else by
  area.
- **Airports**: OurAirports + Wikidata patronage (already the US fallback), connecting share by
  judgement for hubs.
- **Admin units for zones**: NUTS-3 / ITL3 in Europe, geoBoundaries ADM2-3 elsewhere.

**Inherently national**: commute flows (no global source; the Tsinghua synthetic global OD,
1,625 cities on a 0.5-5 km grid, has no stated licence and is model output, so a check at most),
work from home, jobs below the city (registers), school and university enrolment where the
country publishes them.

**The commute curve without flows: fit to a trip-length figure.** Most countries publish something
about commute distance even with no OD: India's census by distance band, China's city commute
reports, national travel surveys' mean commute. Keep the power at a default and fit the length so
the model's distance distribution (or mean) matches. Where nothing exists, take the fitted curve of
the most similar fitted city. This is one small function beside `notes/t277_fit.py`, which
generalises to "flows between any units plus their polygons" for the countries that have flows.

**Recommendation: a tier 0 adapter** (`pipeline/tier0.py`), any city from a bounding set of admin
units, an OSM extract and a curve: population grid scaled to admin totals, the GHSL-OSM job
proxy, OSM everything else, OurAirports-Wikidata airports, OSM water, a fallback curve. National
data then upgrades single inputs by replacing one function, not the adapter:

| Upgrade | London | Berlin | Paris | Shanghai | Mumbai |
|---|---|---|---|---|---|
| population grid | OA census | Zensus 100 m | Filosofi 200 m | district totals | (2027 census) |
| jobs | BRES LSOA | BA per municipality | commune jobs | - | - |
| curve | 2011 MSOA flows | municipal flows | commune flows | mean commute | B-28 bands |
| schools | GIAS | - | ministry files | - | UDISE+ later |
| universities | HESA | Destatis | ministry | - | - |

Canada's adapter (930 lines) becomes the template for "tier 0 plus upgrades", and the first city
also measures tier 0 against the national truth (section 3), which answers T-012 with real numbers.

## 3. Order and effort

**Before any of the five** (shared, one stage each):

1. **Frames and the renderer origin (T-126, T-127).** The registry has one world with one frame
   at New York; the renderer's single origin is New York's. London is 5,570 km away, so both must
   come from the world. Simplest: a world per region (Europe, Shanghai, Mumbai), each its own
   save with its own frame origin, and the renderer anchored at the world's origin. Small code if
   one save stays one world (open call 1).
2. **OSM water mask (T-041)** for any country, with `read_pack.py`'s water checks. The global sea
   polygons are one 906 MB file; cut per world once.
3. **The tier 0 adapter**, checked first on a pack we already have (New Haven or Toronto rebuilt
   from tier 0 and compared with LODES or StatCan: jobs by zone, trips misplaced). Most code is
   Canada's, generalised: the OSM reads, places per cell, schools and campuses at points, airports.

**First city: London.** England has the best open checks for tier 0: population by OA, workplace
jobs by LSOA from an employer survey (no lockdown problem), pre-pandemic MSOA flows, schools and
universities nationally. So London is both a playable city and the measure of how far tier 0 is
from the truth, before Shanghai and Mumbai, which only have tier 0. English names also keep the
parks rule and place search working.

Then **Paris** (France's data is open and national, commune flows like Canada's municipalities,
no lockdown correction; joins London's Europe world), **Berlin** (Zensus grid and the
municipality-level register, the city-state border test; the same world), then **Shanghai** and
**Mumbai**, tier 0 with a trip-length fit, each its own world.

Rough cost, in agent stages (a stage is one task and one report):

| | New code | Reused | Stages |
|---|---|---|---|
| Frames and renderer | registry worlds, renderer anchor | the conic code (T-125) | 1 |
| OSM water | a water source for `water.build_mask` | the rasterizer, checks | 1 |
| Tier 0 adapter | grid reading (GeoTIFF windows), job proxy, school and campus sizing, curve fallback | Canada's OSM reads, spreading, pack writer, gravity, routes.py, places.py | 1-2 |
| London | UK upgrades: OA, BRES, flows fit, GIAS, HESA | tier 0, Toronto's fit script | 1-2 |
| Paris | France upgrades: Filosofi, commune jobs, flows fit, schools | as London | 1 |
| Berlin | Zensus grid, BA per municipality | as London | 1 |
| Shanghai | district totals, mean-commute fit, CJK labels check | tier 0 | 1 |
| Mumbai | B-28 fit, sub-district units | tier 0 | 1 |

**One city's stage contains:** footprint by whole units and why; the adapter's config; build;
`read_pack.py`, water and pack images; tier 0 against the national inputs where both exist
(jobs per zone, curve); registry entry with `market: -1`; a browser check (jump, homes, water
refusal, a test line with every purpose); docs/demand-worldwide.md entries, the "Outside the US"
table's new column, notes/T-###.md.

## 4. Risks

- **Lockdown-year censuses.** England and Wales 2021 (21 March) counted 31% of workers at home;
  ONS warns the travel-to-work data is affected (checked). Avoided by BRES for jobs and 2011 flows
  for the curve. France is not affected the same way: its census counts teleworkers at their usual
  workplace (correcting T-286's brief, which grouped France with the UK). Germany's Zensus 2022
  has no workplace question and the BA register is employer-based. China (Nov 2020) and India
  (2011) have no commuting tables to be affected.
- **China's open data.** No small-area census tables in machine-readable form (district totals in
  PDF bulletins; township tables possibly in yearbooks, not checked), no OD, no school register.
  Government sites can be slow or blocked from abroad. Chinese POI sources (Amap, Baidu) use the
  offset GCJ-02 coordinates; stay with OSM (WGS84), which is thin. Mapping by foreigners is legally
  sensitive in China; using published OSM and global grids avoids collecting anything.
- **India's open data.** The 2011 census is 15 years old; ward polygons are unofficial; UDISE+
  sits behind a captcha portal; most work is informal, which a built-up job proxy misses.
  censusindia.gov.in is slow at times.
- **Bot walls.** Not met this pass on Nomis, INSEE, zensus2022.de, the BA or Geofabrik. StatCan's
  www12 was the Toronto precedent. Note the URL and move on, as before.
- **Licences.** OSM is ODbL (packs derived from it are already shipped for the US); GHSL,
  WorldPop, Kontur and geoBoundaries' gbOpen are CC BY 4.0; OurAirports public domain, Wikidata
  CC0; UK OGL, France Licence Ouverte, Zensus "free with source credit"; Overture Places CDLA
  permissive. **Avoid GADM** (non-commercial). The Tsinghua OD and geoBoundaries' China ADM3 have
  no confirmed licence.
- **Frames and renderer (T-126, T-127)** block all five cities. Labels too: Chinese and
  Devanagari place names need glyphs or system-font fallback on the basemap (as T-057 for Japan),
  and the search and station names read OSM `name`, so `name:en` may be wanted.
- **Demand rules built in the US** (docs/demand-worldwide.md) matter more than the inputs in
  Shanghai and Mumbai: car ownership (the driving alternative is open to everyone), crush loads and
  riders left behind, weekend days. These are not part of this plan, but the first runs there will
  show them.

## 5. Open calls for Anita

1. **open** One save, one world: a Europe world (London, Paris, Berlin), a Shanghai world and a
   Mumbai world as separate games, or one global save with many frames? Separate worlds is much
   less code (T-126, T-127).
2. **open** London first, built as tier 0 and as tier 1 to measure tier 0, or Paris first (the
   simpler national data)? Recommended: London.
3. **open** Extents: Greater London alone (9M) or with the home counties' ITL3 units; Ile-de-France
   (12M) or Paris's inner three départements; Berlin alone or with Brandenburg's ring; Shanghai's
   16 districts; Greater Mumbai or the metropolitan region (26M)? Recommended: Greater London,
   Ile-de-France, Berlin plus its ring of Kreise, Shanghai's districts, the Mumbai metropolitan
   region (its suburban rail runs that far).
4. **open** Should the Europe world get a regional market across its three cities (London, Paris
   and Berlin are 340-1,100 km apart, so little but long trips between them), or stay `-1` like
   Toronto for now? Recommended: `-1` for now.

## Sources checked this pass

- GHSL built-up surface R2023A, non-residential band, CC BY: https://registry.opendata.aws/ghs-built-s/
- WorldPop R2025A 100 m: https://hub.worldpop.org/geodata/listing?id=135
- BRES by LSOA (open access on Nomis): https://integrateddataservice.gov.uk/data/business-register-and-employment-survey-employee-counts-by-lower-super-output-area
- Census 2021 OD data: https://www.ons.gov.uk/census/aboutcensus/censusproducts/origindestinationflowdata
- Census 2021 working from home, lockdown caveat: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/traveltoworkenglandandwales/census2021
- ITL3 and NUTS: https://www.ons.gov.uk/methodology/geography/ukgeographies/eurostat
- HESA 2023/24: https://ckan.publishing.service.gov.uk/dataset/higher-education-student-statistics-uk-2023-24
- Zensus 2022 grid: https://zensus2022.de/EN/Census_results/02-release-of-results.html
- BA commuter atlas and tables: https://statistik.arbeitsagentur.de/DE/Navigation/Statistiken/Interaktive-Statistiken/Pendleratlas/Pendleratlas-Nav.html
- INSEE commuting flows 2022: https://www.insee.fr/fr/statistiques/8582949
- INSEE IRIS bases (no workplace data below commune): https://www.insee.fr/fr/information/2383389
- INSEE Filosofi 200 m: https://www.insee.fr/fr/statistiques/7655475
- INSEE commuting mode 2022 ("no travel" 4%): https://www.insee.fr/fr/statistiques/8581610
- French school locations: https://data.gouv.fr/en/datasets/adresse-et-geolocalisation-des-etablissements-denseignement-du-premier-et-second-degres-1
- Shanghai 2020 census bulletins: https://tjj.sh.gov.cn/tjnj/2020rktjnj/fu02.pdf
- Shanghai mean commute 10.2 km (7th comprehensive transport survey): https://www.thepaper.cn/newsDetail_forward_33619060 ; CAUPD-Baidu's 2025 report gives 9.8 km for 2024: https://www.yicai.com/news/102978803.html (T-291)
- CAAC 2024 airports (press): https://m.jiemian.com/article/12217623.html
- India B-28: https://censusindia.gov.in/nada/index.php/catalog/13954
- Mumbai ward census tables: https://data.opencity.in/dataset/mumbai-ward-wise-census-data
- UDISE+ school details copy: https://ckandev.indiadataportal.com/dataset/udise/resource/457fddf1-982f-4c85-855d-5095578accc1
- Geofabrik China and India extracts: https://download.geofabrik.de/asia/china.html, https://download.geofabrik.de/asia/india.html
- geoBoundaries China listing: https://geo.btaa.org/catalog/70e47c80-9dd7-4786-bb23-dddf710d792f
- Tsinghua global commuting OD: https://arxiv.org/html/2505.17111v1, https://github.com/tsinghua-fib-lab/WorldCommuting-OD
- Overture Places: https://docs.overturemaps.org/guides/places
