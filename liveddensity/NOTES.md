# liveddensity: notes

Per-country histograms of how many people live at each population density. X is the density of
the grid cell a person lives in (people/km², log scale); the value is the number of people in
cells at that density. The page is `index.html`; the data is `data/hist.json`, built by
`build.py`.

## Source

**GHS-POP R2023A** (JRC Global Human Settlement Layer), the newest global GHS-POP release as of
2026-10-25. The GHSL FTP also has `GHS_POP_ARCTIC_R2025A`, which covers the Arctic only, so it is
not used. Population is placed on built-up surface (GHS-BUILT-S), so people do not get spread
over empty land.

- Product: `GHS_POP_E{epoch}_GLOBE_R2023A_54009_100_V1_0`, 100 m, World Mollweide (ESRI:54009),
  one global GeoTIFF per epoch inside a zip (about 5.1 GB zipped, 6.4 GB unzipped; float64,
  LZW, 256×256 tiles, nodata -200). Grid 360820 × 180000 cells, origin (-18041000, 9000000).
- URL pattern:
  `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{epoch}_GLOBE_R2023A_54009_100/V1-0/GHS_POP_E{epoch}_GLOBE_R2023A_54009_100_V1_0.zip`
  (plain HTTP download, no login). Tiles exist too under `.../V1-0/tiles/`; the single file was
  simpler.
- Epochs built: 1975, 2000, 2020, 2025.
- Citation (from DataCite for the DOI): Schiavina, M., Freire, S., MacManus, K. (2023):
  GHS-POP R2023A - GHS population grid multitemporal (1975-2030). European Commission, Joint
  Research Centre (JRC). doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE. Licence CC BY 4.0
  (copyright.txt on the FTP).
- The 1 km product (`..._54009_1000_V1_0`, 307 MB) was downloaded once only to check the block
  alignment, then deleted.

GHS-POP R2023A is spread from census units (CIESIN GPWv4.11 inputs, listed in
`GHS_POP_GLOBE_R2023A_input_metadata.xlsx` in each zip) and scaled to UN WPP 2022 national totals.

**Exception: eleven countries use WorldPop R2025A for 2020 and 2025** (`build.WORLDPOP`,
`worldpop.py`; section "WorldPop for eleven countries" below). Their 1975 and 2000 are still GHSL,
which the page no longer shows. `hist.json` marks them with `"source": "WorldPop R2025A"` on the
country entry (no field = GHSL) and has a top-level `sources` map of citations; the old
`source` and `citation` fields are unchanged.

## Countries

Natural Earth 10m admin 0 countries, read in place from
`../religiondots/data/geo/ne_10m_admin_0_countries.geojson` (this copy is NE 5.1 or later).

Key: `ISO_A3`, else `ISO_A3_EH`, else `ADM0_A3`, after these calls (`KEY_FIX`, `MOVE_PARTS`,
`CUTS` in build.py):

- Somaliland -> Somalia. Northern Cyprus, the UN buffer zone, and the Akrotiri and Dhekelia
  base areas -> Cyprus.
- Kosovo -> `XKX` (the code the EU and World Bank use).
- France and Norway come out as `FRA` and `NOR` via `ISO_A3_EH`. France here includes its
  overseas departments (Guiana, Réunion, Guadeloupe, Martinique, Mayotte), as NE draws it.
- Crimea and Sevastopol -> Ukraine. NE draws it in Russia; moved to follow the UN, so Russia
  and Ukraine line up with WPP (moved as the whole Crimea polygon part of Russia's feature).
- Western Sahara: this NE copy draws Morocco down to Mauritania and leaves `SAH` as only the
  strip east of the berm (about 2,000 people). Everything of Morocco's polygon south of
  27°40′ N is cut off and given to Western Sahara (`ESH`), so it has its full ~0.56 M (2020).
- Siachen Glacier -> India. Baikonur -> Kazakhstan. Guantanamo Bay -> Cuba. Åland -> Finland
  (the UN counts it in Finland). Christmas and Cocos Islands (NE "Indian Ocean Ter.") ->
  Australia. Brazilian Island -> Brazil.
- Dropped as polygons (their few cells go to the nearest country): Southern Patagonian Ice
  Field, Bir Tawil, Bajo Nuevo, Serranilla, Scarborough Reef, Spratly Is., Clipperton,
  Ashmore and Cartier, Coral Sea Is.
- Hong Kong, Macao, Taiwan, Palestine, Greenland, Puerto Rico and the other dependencies stay
  separate as in NE.
- Left as NE draws them: Kashmir (de facto lines), Golan (Israel), East Jerusalem, Abkhazia and
  South Ossetia (Georgia), Transnistria (Moldova).
- Names: NE `NAME_LONG` with plain overrides in `NAMES` (DR Congo, Republic of the Congo, Côte d'Ivoire,
  Türkiye, Russia, South Korea, Laos, Eswatini, Czechia, ...).
- `countries` holds everyone with people in some epoch: 232 entries. Antarctica, Heard and
  McDonald, French Southern Lands, South Georgia, British Indian Ocean Territory and US Minor
  Outlying Islands have no GHSL people and are left out. Vatican City has 9 people.

Polygons are segmentized to 0.05° before projecting, so meridian borders bend correctly in
Mollweide. NE 10m already splits rings at 180°: no ring has a longitude jump over 180°, and the
antimeridian countries come out right (Russia 145.8 M, Fiji 920 k, 2020).

## Sub-national units (subunits.py)

Two kinds of unit, built in the same pass as the countries, with the same bins and grids.
Every unit has a parent ISO3 that is a key in hist.json, and only counts cells of that country.
**Only epoch 2025 is in `data/sub/`** (Anita's call, 2026-10-26; `SUB_YEARS` in build.py). The
other epochs build countries only. 2020 and 2000 were built with units too before that call and
then trimmed out; the 2020 numbers quoted below come from that run.

**States** (`adm1:<id>`, kind `state`): Natural Earth 10m admin 1, read in place from
`../languagedots/data/raw/ne_admin1/ne_10m_admin_1_states_provinces.zip`. 3,493 units.
- Parent: the unit's `adm0_a3` through the same key rules as the countries; Crimea and
  Sevastopol go to Ukraine (NE has them under Russia); every piece south of 27°40′ N in
  Morocco's regions, plus NE's Western Sahara strip, becomes one unit "Western Sahara". Somaliland
  is a state of Somalia; Northern Cyprus, Akrotiri and Dhekelia are states of Cyprus.
- Where NE's admin 1 is really a second level, it is merged up by NE's `region` field: France
  (18 regions), Italy (20 regions), Philippines (17 regions), Slovenia, Latvia, Malta,
  North Macedonia, Uganda, Azerbaijan; the UK by `geonunit` (England, Scotland, Wales,
  Northern Ireland). A few region names are put in English (Brittany, Lombardy, Sicily, ...).
- A populated cell gets the state its centre falls in, if that state belongs to the cell's
  country. Otherwise (coasts, slivers where NE's admin 0 and admin 1 disagree) it gets the
  nearest state of its own country (distance transform inside the window plus 100 km, then the
  nearest polygon of that country). 71.0 M people in 2020, almost all the coastal cells that
  also needed a nearest country. So a country's states always add up to the country: the build
  asserts it on the float sums, and the integer `pop` values in the index are rounded with
  largest remainders so they add up exactly to the country's `pop`.
- Same name twice in one country gets the NE type: "Kyiv (region)", "Kyiv (municipality)",
  "Moscow (region)", "Moscow (federal city)". Fixed by hand: Altai Krai (NE calls it "Altai
  Republic"), Jewish Autonomous Oblast. NE's unnamed island units are "Outlying islands".

**Metros** (`fua:<eFUA_ID>`, kind `metro`): GHSL Functional Urban Areas, **GHS-FUA R2019A**
(`GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg`, city plus commuting zone, delineated on
2015 data), the newest FUA release. 9,031 units. URL:
`https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_FUA_UCDB2015_GLOBE_R2019A/V1-0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip`.
Citation (DataCite): Schiavina, M., Maffenini, L., Moreno-Monroy, A., Veneri, P. (2019):
GHS-FUA R2019A - GHS functional urban areas, derived from GHS-UCDB R2019A (2015). European
Commission, Joint Research Centre (JRC). doi:10.2905/347f0337-f2da-4592-87b3-e25975ec2c95.
Names also use GHS-UCDB R2024A (Mari Rivero, I. et al. 2024, doi:10.2905/1a338be6-7eaf-480c-9664-3a8ade88cbcd).
- GHSL already splits a cross-border FUA into one feature per country (San Diego is the US
  part of "Tijuana"). Parent = the feature's own country (`XKO` -> XKX, Réunion, Guiana,
  Guadeloupe, Martinique, Mayotte -> FRA, `XNC` -> CYP). A metro counts only cells of its
  parent: 4.6 M people in 2020 sat inside a metro polygon but over its country's border.
- Cells outside every metro are in no metro. Metros don't overlap.
- Names: `eFUA_name`, except (315 renamed):
  - "UNNAMED" (126), non-Latin script (122) and a name used by more than one FUA (67) take
    the name of the biggest **GHS-UCDB R2024A** urban centre (`GC_UCN_MAI_2025`, layer
    `UC_centroids`) inside the FUA. That gives San Diego, El Paso, Laredo, Windsor, Johor Bahru,
    Savar and so on, while the FUA that the name belongs to keeps it. UCDB R2024A V1.2 was
    downloaded for this only (`raw/ucdb/`).
  - Left in another script: the Latin part if there is one ("Pirganj (পীরগঞ্জ)" -> Pirganj),
    else Unidecode ("蒲江县" -> Pujiang). Hand fixes in `FUA_NAME_FIX` where that came out
    wrong (Chibayish, Ungsang, Nicosia / North Nicosia, Narsingdi, ...).
  - R2019A splits the Pearl River Delta into several FUAs all called "Guangzhou", and the
    centroid rule then named the biggest one "Sihui" (17 M) and Shenzhen "Yantian District".
    Fixed by hand from where each lies: Guangzhou (with Foshan), Shenzhen, Shenzhen (Bao'an),
    Dongguan, Jiangmen; also Tangerang (a "Jakarta" piece), Gazipur, Owerri. Patched into
    `data/sub/index.json` on 2026-10-27 (names only, histograms unchanged).
  - Still unnamed: "Unnamed, <state>". Still the same name twice in a country: the smaller ones
    get "(<state>)", e.g. "Wuwei (Gansu)", then a number.
- `data/sub/index.json` (1.0 MB) lists units with at least 50,000 people in 2025 (11,089:
  8,173 metros, 2,916 states). `data/sub/<ISO3>.json` holds every unit of that country, as
  `{"hist": {"1km": {year: {id: {o, v}}}, "100m": {...}}}`. All written via temp file + replace. 238 country files, 17.6 MB in all (China 2.3 MB,
  India 2.0 MB, US 0.6 MB).

## City propers (cities.py, cities.csv)

Administrative city limits, 2025 only, both grids, as units with `"kind": "city"` and ids
`city:<iso3>-<name>` (one city can be several OSM relations). They sit in `data/sub/index.json`
(all of them, no 50,000 floor) and in the country files next to the metros and states. A city is
its own unit, not merged into its metro, and counts only cells of its own country (same country
raster as the main build; cells off NE's coast inside a city outline go to the nearest country).

**Which cities.** 334 built, 11 skipped, from three lists (`city_seeds.csv`):
- `metro>3M`: the city each FUA over 3 M people in 2025 is named after (225 FUAs; Shenzhen
  (Bao'an) is Shenzhen's second FUA and shares its city). 217 built.
- `capital`: national capitals of the 162 countries in hist.json with over 1 M people (Bolivia
  has two). 158 built.
- `GaWC`: GaWC 2024 Alpha and Beta tiers, 120 cities, all built. The list is from
  https://gawc.lboro.ac.uk/gawc-worlds/the-world-according-to-gawc/world-cities-2024/ and saved
  as `gawc_2024.csv` (all 335 classified cities with their tiers).
- `added` (5): Edinburgh (GaWC has it only at Gamma-), Kyoto, Venice, Mecca, Honolulu.
Overlaps count once: 334 cities carry 500 source tags. Each row's sources are in `cities.csv`.

**Boundaries.** OSM relations, found with Nominatim and Overpass, polygons from
polygons.openstreetmap.fr, each checked against Wikidata's area (`logs/city_areas.csv`).
Default: the legal city or municipality the name refers to. Exceptions and other calls are in
the `note` column of `cities.csv`, one line each. The main ones:
- Tokyo: the 23 special wards. London: Greater London. Paris: the commune. Delhi: the NCT
  (and New Delhi, the capital, is its own unit: the NDMC area). Manila: City of Manila.
- Chinese cities: urban districts (区). Where those still take in big rural areas, the core
  districts: Beijing the six core districts (1,488 km2), Shanghai the seven central districts
  plus Pudong, Tianjin the six central districts, Chongqing the nine core districts, Guangzhou
  the seven older districts, Wuhan seven, Chengdu five, Nanjing six, and so on. Whole city where
  every district is urban (Shenzhen) or there are no districts (Dongguan, Cixi).
- City-wide governments where the name is the region: Cairo and Alexandria governorates,
  Istanbul (the metropolitan municipality = the province, 5,300 km2 of land), Kinshasa,
  Karachi Division, Lima Province, Mexico City (all of CDMX), Hanoi, Brasília (the Federal
  District), Auckland, Honolulu (City and County = Oahu), Canberra (the ACT).
- Groups of units for cities without one outline: Greater Mumbai (two districts), Metropolitan
  Lagos (16 LGAs), Kano (8 LGAs), Ibadan (5 LGAs), Onitsha and Owerri LGAs, Ankara (the 9
  central districts, not the 25,600 km2 province), Kingston + St Andrew, Nouakchott (3 regions),
  Hyderabad (Pakistan) 3 talukas.
- Australian cities use the council that holds the name (City of Sydney 0.29 M, City of
  Melbourne 0.23 M, City of Perth 27 k, City of Brisbane 1.3 M): Greater Sydney etc. have no
  government.
- Brussels: the Brussels-Capital Region, not the City of Brussels commune.
- Ho Chi Minh City: the pre-July-2025 city from geoBoundaries (VNM ADM1, `raw/geob/`), because
  OSM already has the merger with Binh Duong and Ba Ria-Vung Tau, a 6,800 km2 province.
- Nearest available unit, noted as such: Dhaka (Dhaka Metropolitan area; OSM has no City
  Corporation outlines), Cumilla and Gazipur (sadar upazilas), Pune (city taluka), Kanpur and
  Lucknow (tehsils), Port-au-Prince (arrondissement), Maseru and Juba (district / county),
  Amman, Tunis, Riyadh, Jeddah, Dammam (governorates). Baku: OSM's outline is the central city
  only (165 km2, 1.2 M of about 2.3 M).

**Skipped (11), no usable outline in OSM:** Ahmedabad, Surat, Varanasi, Asansol (Indian
municipal corporations missing), Mansoura, Luxor, Asyut (no city outline; governorates are
mostly countryside), Antananarivo (urban commune missing), Gitega, Lilongwe (only the province
or district) and Sri Jayawardenepura Kotte (Colombo is covered).

**Raster.** The 2025 100 m tiles that cover the cities (132 of GHSL's 1,000 km tiles, 3.3 GB,
`..._54009_100_V1_0_R{row}_C{col}.zip`) joined in a VRT on the global grid, so the 1 km blocks
line up with the main build. Deleted afterwards.

**Rebuild:** edit `cities.csv`, then `python -I cities.py units`, `cities.py tiles`,
`cities.py hist` (a few minutes on one core after the tiles are in). `cities.py check` writes
the outline areas next to Wikidata's.

## Method (build.py)

- 5000 × 5000-cell windows (500 km), 6 worker processes, one thread each.
- Each window: read population, drop nodata and zeros, rasterize country ids (GDAL rasterize,
  cell centres) from polygons clipped to the window.
- Populated cells outside every polygon go to the nearest country: rasterize the window plus a
  1000-cell (100 km) margin, then a Euclidean distance transform with indices gives each such
  cell its nearest country cell. Anything over 100 km from every polygon goes to the nearest
  polygon anywhere (shapely STRtree `query_nearest`). Nothing is dropped; the build asserts it.
- 100m density = cell population × 100. 1km density = sum of the aligned 10 × 10 block, which
  every cell of the block gets, and each cell's people still count for its own country. Windows
  start at multiples of 10 cells from the grid origin, which is also the 1 km product's origin
  (-18041000, 9000000).
- Bins: log10 density, lo -1, step 0.02, n 350; below lo into bin 0, above the top into the last.
  People are accumulated per country per bin in float64 and rounded when written.
- `data/hist.json` is merged per epoch (an existing file keeps its other epochs) and written via
  a temp file + `os.replace`. Schema as the page expects: `source`, `citation`, `years`,
  `grids`, `bins`, `countries` {ISO3: {name, pop: {year: n}}}, `hist` {grid: {year: {ISO3:
  {o, v}}}}.
- Logs go to `logs/build_{year}.log`, totals and reassignments to `logs/summary_{year}.json`.
  While it runs it saves `logs/state_{year}.npz` every minute and resumes from it after a crash
  (only when the countries, windows and input file are the same).

## Rebuild

    venv/Scripts/python.exe -I liveddensity/subunits.py          # only if the unit sources change
    venv/Scripts/python.exe -I liveddensity/run_epochs.py 2020 2025 2000 1975
    venv/Scripts/python.exe -I liveddensity/cities.py hist        # after the 2025 epoch (tiles first)
    venv/Scripts/python.exe -I liveddensity/worldpop.py run 2020  # the WorldPop countries
    venv/Scripts/python.exe -I liveddensity/worldpop.py run 2025
    venv/Scripts/python.exe -I liveddensity/worldpop.py write     # always last
    venv/Scripts/python.exe -I liveddensity/check.py 2020 <owid population.csv>
    venv/Scripts/python.exe -I liveddensity/check_sub.py 2020

`build.py` and `cities.py hist` write GHSL numbers for every country, the WorldPop ones too
(build.py also drops their `source` field for 2020 and 2025), so `worldpop.py write` must come
after them.

`run_epochs.py` does each epoch in turn: download the zip (resumes, restarts a stalled
transfer), unzip the GeoTIFF and delete the zip, claim cores in `../cores.py`, run `build.py`,
release, delete the GeoTIFF. It fetches the next zip while the current epoch builds. About 45-50
min per epoch on 6 cores with the sub-national units (2025), about 20-35 min for countries alone; downloads run
3-5 MB/s and stall now and then. Reading the GeoTIFF straight out of the zip (`/vsizip/`) works
but was about 4x slower, so the runner unzips.

`build.py <year> <tif>` on its own does one epoch; `--bbox lon0 lat0 lon1 lat1` runs a test area
and writes nothing. Needs rasterio (1.4.3) and Unidecode (1.3.8), both installed into the maps
venv for this, plus numpy, scipy, shapely, pyproj, pyogrio. `raw/` and `logs/` are gitignored;
`logs/units.pkl` (the prepared units) and `logs/sub_pops.json` (every unit's people in every
built epoch, which `data/sub/index.json` is made from) are build state.

## Checks (2026-10-26)

**Nothing lost.** Every epoch: sum over countries = raster total, to rounding.

| epoch | GHSL raster total | sum of countries | WPP 2024 world |
|---|---|---|---|
| 1975 | 4,069,437,259 | 4,069,437,262 | 4,070,735,279 |
| 2000 | 6,148,899,024 | same | 6,171,702,992 |
| 2020 | 7,840,952,947 | 7,840,952,941 | 7,887,001,289 |
| 2025 | 8,191,988,536 | same | (projection, not checked) |

**Nearest-country reassignment** (people in cells outside every NE polygon): 2020 71.0 M (0.9%),
of whom 57 k were over 100 km from any polygon. Biggest: China 9.0 M, Philippines 8.3 M,
Indonesia 7.5 M, India 4.0 M, Brazil 3.0 M, US 1.8 M, Hong Kong 1.3 M. Mostly coastal cities and
reclaimed land that NE 10m's coastline misses. 1975: 36.9 M; 2000: 55.6 M; 2025: 73.9 M.

**Country totals, 2020, against UN WPP 2024** (via Our World in Data's population grapher).
GHSL is scaled to WPP 2022, so a few % of drift is expected.

| | GHSL | WPP 2024 | diff |
|---|---|---|---|
| China | 1,425.2 M | 1,426.1 M | -0.1% |
| India | 1,396.2 M | 1,402.6 M | -0.5% |
| United States | 336.1 M | 339.4 M | -1.0% |
| Indonesia | 271.8 M | 274.8 M | -1.1% |
| Brazil | 213.4 M | 208.7 M | +2.3% |
| Nigeria | 208.1 M | 214.0 M | -2.7% |
| Russia | 145.8 M | 146.4 M | -0.4% |
| Japan | 125.2 M | 126.3 M | -0.8% |
| Egypt | 107.4 M | 109.3 M | -1.7% |
| DR Congo | 92.6 M | 96.0 M | -3.6% |
| Germany | 83.4 M | 83.6 M | -0.3% |
| Ukraine | 43.9 M | 44.7 M | -1.7% |
| Morocco | 36.6 M | 36.6 M | +0.2% |
| Kenya | 52.2 M | 52.2 M | +0.1% |
| Australia | 25.7 M | 25.7 M | -0.3% |
| Philippines | 112.2 M | 112.1 M | +0.1% |
| Fiji | 920 k | 915 k | +0.6% |
| Iceland | 367 k | 367 k | 0.0% |
| Western Sahara | 556 k | 549 k | +1.1% |

Bigger gaps (countries over 1 M) come from GHSL's own inputs, not the country assignment:
Saudi Arabia +16%, North Macedonia +13%, Eritrea +9%, Togo +8%, Serbia +8% (WPP 2024 revised
these down after new censuses); Yemen -11%, Côte d'Ivoire -7%, Namibia -7%, Palestine -7%
(WPP counts East Jerusalem in both Israel and Palestine). In 1975 Saudi Arabia is +73%
(GHSL's back-cast, against WPP 2024's much lower history) and the US is -3.5%.

**US 1km peak.** Smoothed over 0.1 log units, the US 1km histogram peaks at 1,780/km² (2020),
inside the expected 1,000-2,000. 54% of Americans live in 1 km cells of 1,000-10,000/km².

**1 km alignment.** 10 × 10 sums of the 100 m grid match GHSL's own 1 km product exactly (0.000%
cell-by-cell difference) over the Netherlands, Lagos, Dhaka and Iowa; shifting by 5 cells gives
42-77% differences (`check_1km.py`).

**Small country and archipelago.**
- Singapore 5.80 M vs WPP 5.62 M (+3.3%). 261 k of that were cells off NE's coastline
  (reclaimed land in the north around Woodlands and in the east), assigned by nearest. Some cells
  along the Johor Strait may have gone to the wrong side.
- Macao 771 k vs 683 k (+13%). NE's Macao is small; part of Hengqin (China) is nearer to Macao's
  polygon than to China's, so roughly 70-90 k Chinese people count as Macao. Hong Kong +0.6%.
- Philippines 112.2 M vs 112.1 M (+0.1%), although 8.3 M of it were reassigned coastal cells.

## Sub-national checks (2025, `check_sub.py 2025`)

- **States add up**: in every country the states' integer `pop` sums to the country's `pop`
  exactly (0 mismatches); the float histograms agree to under one person.
- **Metros against other figures**:
  - New York 20.20 M; NY-Newark-Jersey City MSA 19.94 M (Census 2024 estimate), +1%.
  - Houston 7.82 M; Houston MSA 7.80 M (2024 estimate), 0%.
  - Dhaka 26.1 M; UN World Urbanization Prospects 2018 puts Dhaka at about 24.6 M in 2025, +6%.
    The FUA reaches out to Gazipur and Narayanganj.
- **Clipping at borders**: San Diego (the US part of GHSL's Tijuana FUA) 3.48 M and Tijuana
  1.71 M stay apart; El Paso 1.00 M / Ciudad Juárez 1.25 M; Detroit 4.04 M / Windsor 0.39 M;
  Singapore 5.95 M / Johor Bahru 2.17 M. 4.6 M people (2020) sat inside a metro polygon but over
  the border, and count for no metro.

## City-proper checks (2025)

GHSL 2025 inside each outline, against the latest official count I know of:

| city (unit used) | GHSL 2025 | official | gap |
|---|---|---|---|
| New York City | 8.45 M | 8.48 M (Census 2024 estimate) | 0% |
| Tokyo, 23 wards | 9.17 M | 9.73 M (Tokyo Metropolitan Government, 2024) | -6% |
| Paris (commune) | 2.25 M | 2.11 M (INSEE 2022) | +7% |
| London (Greater London) | 9.79 M | 8.95 M (ONS mid-2023) | +9% |
| Shanghai, 7 central districts + Pudong | 18.2 M | 12.4 M (2020 census) | +47% |
| Lagos, 16 metropolitan LGAs | 10.8 M | 7.9 M in the 2006 census, none since | n/a |
| São Paulo (municipality) | 12.0 M | 11.45 M (IBGE 2022) | +5% |
| Greater Mumbai | 15.1 M | 12.4 M (2011 census, none since) | +22% over 14 years |
| Cairo Governorate | 9.44 M | about 10.3 M (CAPMAS 2023) | -8% |
| Mexico City (CDMX) | 7.78 M | 9.21 M (INEGI 2020) | -16% |
| Seoul | 9.87 M | 9.33 M (resident registration 2024) | +6% |
| Berlin | 3.63 M | 3.89 M (2024) | -7% |
| San Francisco | 0.83 M | 0.81 M (Census 2023 estimate) | +2% |

Most are within 10%, which is about what spreading census units over built-up land can do at
city edges. The big misses are GHSL's own allocation, not the outlines (the outline areas match
the official ones): GHSL puts far more people in central Shanghai than the 2020 census does
(Putuo 2.1 M vs 1.24 M, Yangpu 2.1 M vs 1.24 M), and fewer in Mexico City and more in the
State of Mexico around it, though Mexico's national total is right. Kerala's corporations come out
2-3x their 2011 census figures (Kozhikode 2.7 M, Kollam 1.1 M, Thiruvananthapuram 2.5 M), the same
pile-up as in the dense-country section below.

## Republic of the Congo, Somalia, Djibouti, Yemen

(The first look, with GHSL. Congo, Somalia and Yemen now use WorldPop for 2020 and 2025; see
"WorldPop for eleven countries".) At 1 km their medians are 30k, 18k, 16k and 9k/km² (2025). Evidence used: the metro and
outside-metro histograms here; Africapolis 2025 (OECD/SWAC, `../cityhistory/data/
Africapolis_agglomeration_2025.xlsx`, population and built-up km² per agglomeration, 2020); and
the census inputs GHSL lists in `GHS_POP_GLOBE_R2023A_input_metadata.xlsx`. GHS-POP spreads each
census unit's people over the built-up surface it detects inside that unit, so with big units
and missed buildings the people pile up on the cells it did see. "Occupied km²" below is the
area of the 1 km cells that hold anyone. All figures 2020.

| | GHSL people | occupied km² | mean /km² | Africapolis people | built-up km² | /km² |
|---|---|---|---|---|---|---|
| Brazzaville | 2.76 M | 159 | 17,300 | 2.03 M | 236 | 8,600 |
| Pointe-Noire | 0.68 M | 154 | 4,400 | 1.06 M | 224 | 4,700 |
| Mogadishu | 3.56 M | 117 | 30,500 | 1.87 M | 164 | 11,500 |
| Hargeisa | 1.31 M | 61 | 21,400 | 1.00 M | 83 | 12,000 |
| Baidoa | 0.33 M | 16 | 20,900 | 0.25 M | 29 | 8,700 |
| Kismayo | 0.32 M | 13 | 24,400 | 0.15 M | 18 | 8,300 |
| Djibouti | 0.46 M | 49 | 9,500 | 0.57 M | 58 | 9,800 |

- **Republic of the Congo: both.** Real: it is one of the most urban countries in Africa;
  Africapolis puts 3.1 M of about 5.7 M people in Brazzaville and Pointe-Noire, and GHSL has
  73% of people in metros, while the rest of the country sits at a median of 1,500/km². The
  artefact is inside Brazzaville. GHSL gives it 2.76 M people on 159 km² of occupied land (1 km
  median 47,000/km²), against Africapolis's 2.03 M on 236 km² of built-up land. It also gives
  Pointe-Noire only 0.68 M against 1.06 M. GHSL's Congo input is the 2007 census at 12
  departments, so the split between the two cities is extrapolated, and Brazzaville's people are
  packed onto too little land. So the right shape is "most people in two cities", but at roughly
  half the densities shown.
- **Somalia: mostly artefact.** Every city has more people on less land than Africapolis has:
  1.3-2.1× the people on 0.55-0.75× the area, so 1.8-2.9× the density. Outside metros, 9.2 M people
  sit at a median of 5,100/km², and 36% of them in 1 km cells of 10,000/km² or more. That
  can't describe a country where a large share of people are pastoralists. The input is a 2005
  UNDP estimate at 74 districts averaging 8,600 km² (no census since 1975), so whole districts'
  people land on the few settlements the satellites picked up. IDP camps around Mogadishu and
  Baidoa really are dense, but not on this scale.
- **Djibouti: mostly real, rural tail artefact.** Djibouti city matches Africapolis closely
  (9,500 vs 9,800/km² on about the same land), and three quarters of Djiboutians are urban.
  But the 0.63 M outside the FUA sit at a median of 14,100/km² (part of that is Balbala, the
  city's own suburb, which the 2015 FUA outline cuts off). The rest is nomads and small towns
  from 6 input regions (2009 census) piled onto a handful of towns.
- **Yemen: both, mostly artefact outside the cities.** Sana'a (3.4 M in 2020, 7,800/km² mean
  over 439 km²) is in line with UN estimates of about 3 M and with a compact old city. But 71%
  of Yemenis live outside metros, and GHSL puts them at a median of 4,500/km², with a third in
  cells of 10,000/km² or more. Yemen's countryside is tens of thousands of small stone villages,
  which are hard to see from space. The input is the 2004 census at 333 districts, carried
  forward 16 years through war and displacement, so each district's people sit on the villages
  that were detected.

## WorldPop for eleven countries (2026-10-27)

The section above is why. A scan of every country (1 km medians from GHSL against WorldPop
R2025A's own 1 km mosaic, each source's census inputs, and the people GHSL and WorldPop put
around 364 Africapolis agglomerations of 250 k or more) found that the two grids agree where the
census input is good: the US, Netherlands, Germany, UK, France, Japan, China, Brazil, Mexico,
Russia, Australia, Colombia and Türkiye all have medians within 0.7-1.4× of each other. Where
GHSL's input is a few big units or an old census, GHSL is 3-8× denser, and the city checks side
with WorldPop. WorldPop is not good enough to use everywhere: it flattens the densest cities
(Hong Kong 1 km median 25,700 against GHSL's 46,800, where district averages alone are 30-57 k;
Bangladesh's top tenth starts at 10,600 against 44,700), its Morocco misses most of Tangier,
Meknes, Safi and Tétouan, it only covers 2015-2030, and it is an alpha release. So only the
countries where GHSL is clearly worse were swapped. Meta's HRSL and WorldPop's older 2020
constrained grid were also tried for Congo: both spread the same 2007 census, so they put
1.7 M in Brazzaville and 0.2-0.5 M in Pointe-Noire, no better than GHSL.

**Source.** WorldPop Global2 R2025A v1, constrained, 100 m (3 arc-seconds), one file per country
and year: `https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/{ISO3}/v1/100m/constrained/{iso3}_pop_{year}_CN_100m_R2025A_v1.tif`.
Random-forest dasymetric spread of c.2010 and c.2020 census rounds, on a settlement layer from
GHSL BUILT-S/BUILT-V plus Google Open Buildings V3 and Microsoft building footprints; national
totals matched to UN WPP 2024 (1 January). CC BY 4.0. Citation (WorldPop hub): Bondarenko M. et
al. (2025), Constrained estimates of 2015-2030 total number of people per grid square at a
resolution of 3 arc (approximately 100m at the equator) R2025A version v1, WorldPop, University
of Southampton, doi:10.5258/SOTON/WP00839. Release statement:
`https://data.worldpop.org/repo/prj/Global_2015_2030/R2025A/doc/Global2_Release_Statement_R2025A_v1.pdf`;
census inputs: `.../R2025A/doc/census/global2_census_data_sources_R2025A_v1.xlsx`.

**The countries** (census inputs from both sources' metadata sheets; 1 km median person, 2025):

| country | GHSL input | WorldPop input | 1 km median GHSL → WorldPop | confidence GHSL was wrong |
|---|---|---|---|---|
| Republic of the Congo | 12 departments, 2007 census | 57 units, 2023 census (preliminary) | 29,500 → 9,300 | high: the 2023 census has Brazzaville 2.15 M, Pointe-Noire 1.42 M |
| Somalia | 74 districts, 2005 estimate | 74 districts, 2021 estimate (COD-PS) | 17,800 → 5,100 | high |
| Equatorial Guinea | 7 units, 2015 | 18 units, 2015 | 15,500 → 2,200 | high: GHSL has Bata and Malabo at 1.6-2.1× Africapolis |
| South Sudan | 78 units, 2008 (pre-independence) | 28 units, 2022 | 3,400 → 390 | high |
| DR Congo | 181 units, 2008 | 164 units, 2020 | 7,100 → 1,300 | medium |
| Eritrea | 6 units, 2012 | 6 units, 2012 | 2,800 → 490 | medium |
| Zambia | 150 units, 2010 | 116 units, 2022 census | 1,100 → 300 | medium |
| Sierra Leone | 160 units, 2004 | 208 units, 2021 | 3,200 → 1,000 | medium |
| Niger | 66 units, 2012 | 266 units, 2023 | 1,500 → 540 | medium |
| Yemen | 333 districts, 2004 census | 33 units, 2016 projection | 9,300 → 2,800 | medium (both inputs weak) |
| Afghanistan (added later the same day) | 400 districts, 2011 estimate | 34 provinces, 2021 estimate | 5,100 → 1,300 | medium (below) |

**Afghanistan and North Korea**, checked because both sat high on the page.
- *Afghanistan: switched.* There has been no census since 1979; NSIA's 2025-26 estimate has
  about 25% of people urban (about 9 M of 35.7 M; the UN total is higher, 43.8 M). GHSL puts
  50% of Afghans in 1 km cells of 5,000/km² or more and 38% at 10,000 or more; outside the
  metros, 20% (6 M people) are at 10,000/km² or more, which is a town, not a village. WorldPop has
  23% at 5,000 or more, about the urban share. Mud-brick qala villages are the colour of the
  ground and GHSL's built-up layer misses many of them (the Yemen problem); WorldPop adds
  Google and Microsoft building footprints, so it spreads village people over 166,000 km² of
  occupied 1 km cells against GHSL's 53,000. Against NSIA's 2025-26 city estimates (Kabul
  5.33 M, Kandahar 0.73 M, Herat 0.67 M, Mazar 0.57 M, Jalalabad 0.32 M), GHSL has Kabul city
  5.94 M, Herat metro 1.52 M, Kandahar 0.90 M, Mazar 0.75 M, Jalalabad 0.62 M; WorldPop has
  4.25 M, 0.91 M, 0.53 M, 0.52 M, 0.46 M. So WorldPop is low in Kabul (its input is only by
  province) and GHSL high in the smaller cities; the country curve, which is mostly villages,
  is the reason for the switch. Kabul city's 1 km median goes 30,900 → 19,500.
- *North Korea: kept on GHSL.* Both grids use the same input (2008 census, 181 units), put
  nearly the same people on the same land in every city (Pyongyang city 3.35 M on 729 km² of
  occupied cells vs 3.20 M on 749 km²; Hamhung, Chongjin, Sariwon and Wonsan likewise), and
  match the census (Pyongyang 3.26 M in 2008, 2.82 M of them urban). They differ only in how
  tightly people sit inside that land: Pyongyang's 1 km median is 19,500 in GHSL and 13,500 in
  WorldPop. The national median (GHSL 10,700, WorldPop 4,700) swings a lot because it falls on the
  edge between towns and villages: the census has 60.6% urban, GHSL puts 51% of people at
  10,000/km² or more and WorldPop 32%. North Korean towns are compact apartment blocks and farm
  villages are clustered, so GHSL's shape is believable; nothing points to it being wrong.

Afghanistan's totals: 2025 GHSL 44.23 M → WorldPop 43.25 M (UN 43.84 M); 2020 38.76 → 38.43 M;
states add up exactly.

Candidates not swapped (yet): Cameroon (GHSL 7,100 vs 2,000; 58 units 2005 vs 10 units 2023),
Nicaragua, Saudi Arabia, Chad (medium-low); Djibouti (the city is right, only the rural tail is
squeezed; both sources use the same 6 regions). Over 2× apart but GHSL kept: Morocco (WorldPop is
broken), Western Sahara, Honduras, Dominican Republic, Angola, Hong Kong (GHSL has
the better input). One line in `build.WORLDPOP` adds a country.

**Method (`worldpop.py`).** Each WorldPop file is resampled with GDAL's `sum` resampling
(rasterio `WarpedVRT`) onto GHS-POP's own 100 m Mollweide grid, over a window that starts on a
1 km block of the global grid, so the 100 m and 1 km binning is exactly build.py's. Totals are
kept: -0.000% to -0.020% per country and year. Every person in a country's WorldPop file counts
for that country (WorldPop's LSIB borders, not Natural Earth); neighbours keep their GHSL people,
so along a border the two can overlap or leave a gap of a few cells, and a 1 km block on a
border only sums this country's people (Brazzaville's blocks no longer include Kinshasa's).
In 2025, states (NE admin 1; cells outside the country's NE states go to the nearest one, as in
build.py), metros (FUA cells of that country) and cities (`logs/city_units.pkl` outlines) are
binned from the same raster. `run` writes `logs/worldpop/<ISO3>_<year>.json`; `write` puts them
into `hist.json`, `data/sub/<ISO3>.json`, `data/sub/index.json` and `logs/sub_pops.json`, with
states rounded by largest remainders so they add up to the country exactly. About 30 min on one
core for 2025 (DR Congo alone 10 min), less for 2020; 0.7 GB of downloads in `raw/worldpop/`,
deleted afterwards (`worldpop.py clean`).

**Checks (2026-10-27).**
- Totals: WorldPop / UN WPP 2024 (mid-year, Our World in Data) is 0.981-0.991 in every
  country and year, the gap between 1 January and 1 July. Against the old GHSL figure, 2025:
  Yemen 35.96 → 41.17 M, DR Congo 108.8 → 111.0 M, Niger 29.2 → 27.5 M, Eritrea 3.94 → 3.57 M,
  Somalia 20.15 → 19.33 M, others within 5%.
- States add up exactly to the country in all ten (2025). Every other country, and 1975 and
  2000 for these ten, are byte-for-byte unchanged in `hist.json`.
- Cities against the reference in `cities.csv` (GHSL → WorldPop): Brazzaville 3.19 → 2.23 M
  (2023 census 2.15 M), Mogadishu 3.88 → 2.43 M (2.12 M in 2015), Lusaka 3.55 → 2.43 M (2.20 M,
  2022 census), Malabo 0.68 → 0.39 M (0.30 M, 2018), Juba county 0.38 → 0.53 M (0.50 M),
  Niamey 1.70 → 1.59 M (1.50 M, 2024). Worse with WorldPop: Kinshasa 13.48 → 9.56 M (an
  estimate of 14.6 M for 2020; Africapolis 11.9 M for 2025), Freetown 1.49 → 0.54 M (1.06 M in
  the 2015 census), Hargeisa metro 1.39 → 0.58 M (Africapolis about 1.0 M). Asmara is low in both
  (0.39 → 0.53 M against an estimate of 0.96 M for the Central region). Sana'a goes up, 3.66 →
  4.43 M.
- 1 km medians of the cities (2025, GHSL → WorldPop): Brazzaville 53,700 → 16,200, Kinshasa
  49,000 → 24,500, Mogadishu 67,600 → 30,900, Lusaka 16,200 → 8,100, Freetown 29,500 → 7,100,
  Sana'a 33,900 → 37,200. Dhaka (GHSL) is 93,300.

## Thumbnails (2026-10-27)

One 192 x 192 WebP per place (every country in hist.json and every unit in `data/sub/index.json`),
2025 only, at `data/thumbs/<id>.webp` with every character outside `A-Za-z0-9_.-` replaced by `_`
(`USA.webp`, `fua_7244.webp`, `city_usa-new-york.webp`, `adm1_GAZ_00_.webp` for `adm1:GAZ+00?`;
index.html applies the same regex). `thumbs.py` makes them; `data/` is gitignored, the files go to
R2 at `popdensity/data/thumbs/` and the page reads them from `${BASE}data/thumbs/`.
`python -I thumbs.py` is resumable (a place with a file is skipped; `--force` redoes all,
`--only id,id` redoes those, `clean` deletes the downloads). Log: `logs/thumbs.log`.

**Picture.** The outline in a Lambert azimuthal equal-area projection centred on the place, square,
8% margin on each side (at least 4 km across, so a tiny place is not blown up past about 5 pixels
per 100 m cell). Each pixel is the mean density of the source cells under it (k x k sample points per
pixel, k up to 10), coloured with the page's ramp (`RAMP` and `rampAt` ported to numpy, same
rounding), 10 to 400k people/km2. Zero is `#0a0a0a`. The page's ramp starts at 10/km2, so below
that the colour fades to the background by 1/km2 (otherwise every thinly peopled land pixel is the
palest green and a country looks like one flat sheet). Pixels outside the outline are drawn at 30%.
The outline is a 1 px line, `#cdcdcd` at 90%, drawn at 4x and averaged down. No text.

**Sources.** Level by pixel size p = frame / 192: p under 1 km reads the 100 m tiles (cell people x
100; `GHS_POP_E2025_..._54009_100_V1_0_R{r}_C{c}`, about 230 of the 1000 km tiles, fetched in batches of
up to 40 and deleted when the next batch does not need them); 1 to 5 km the 1 km product
(`..._54009_1000_V1_0`, 323 MB); 5 to 25 km a 5 km block mean of it; 25 km and over a 25 km block
mean. 10,029 places use 100 m, 1,408 the 1 km, 159 the 5 km and 8 the 25 km level. The 100 m
cells are averaged down (never upsampled) where the frame is over 19 km across; under that a cell
shows as several pixels.
The eleven `build.WORLDPOP` countries (AFG added 2026-10-27) use WorldPop R2025A 2025 for the
country, its states, metros and cities: `make_patch` resamples the country's 100 m file onto the
GHS-POP grid with a sum, as `worldpop.py` does, into a scratch GeoTIFF plus 1 km sums. Inside the
country's Natural Earth polygon, or wherever WorldPop has people, a pixel takes WorldPop;
elsewhere (the neighbours, drawn dim) it takes GHSL.

**Outlines** are what the build used: Natural Earth countries with `build.load_countries` (all the
border calls), `logs/units.pkl` states and metros, `logs/city_units.pkl` cities. A state, metro or
city has what lies inside another country's polygon cut off (172 units), as the build only counted
the parent country's cells. A unit with no polygon of its own (`adm1:XXX-all`) is drawn as its
country. Parts of an outline split at 180 degrees are projected one by one and merged after (Russia,
Fiji, the Aleutians), so nothing is torn. A small closing buffer (3% of a pixel) removes the seam.

**Which parts are framed.** A place with islands or exclaves far apart is framed on the people:
the part with the most people (people counted on the 5 km or 1 km level), every part with at least
10% of them, and every part reached from those by hops of at most max(15% of the main part's
diagonal, 100 km). Places whose parts all lie within 200 km are drawn whole. What is left out is
off the picture, not drawn. Calls on the awkward ones:
- United States: the contiguous 48 states (149 of 344 parts, 5,455 km across). Alaska (distance to
  the 48 states about 850 km), Hawaii and the territories are out of frame.
- Russia: all of it, 9,390 km across, centred on 95 E 60 N; Chukotka crosses 180 E and joins up.
- France: metropolitan France and Corsica, 1,364 km. The overseas departments are out (14% of the
  area, 4% of the people).
- Indonesia: the whole archipelago, 6,039 km (255 of 264 parts: only a few far outliers out).
- Kiribati: Tarawa only, 21 km (see scattered atolls below); the other islands are out.
- Fiji: all but three small outliers, 589 km, over 180 degrees.
- Japan 87 of 109 parts (the far southern islands out), Norway (Svalbard and Jan Mayen out), Chile
  (Easter Island out), Portugal (Azores and Madeira out), Spain (Canary Islands out), Ecuador
  (Galapagos out). Malaysia is whole: Borneo holds about 20% of the people, so it is kept.
- Units use the same rule: Tokyo Prefecture without the Izu and Ogasawara islands, Honolulu is
  Oahu (the north-western Hawaiian Islands are out), Hawaii state 6 of 14 parts.
- Scattered atolls: where the kept parts cover under 1% of their frame (`SPARSE`), the picture is
  cut down to the part with most people plus parts within 15 km (`refine_sparse`, run once on
  `plans.pkl`). Done for Kiribati (country and state), Marshall Islands (4 km, the minimum frame),
  Cook Islands (Rarotonga, 14 km), Tonga (Tongatapu, 39 km), Tuvalu (Funafuti, 11 km), Wallis and
  Futuna, American Samoa (Tutuila), Saint Helena, Maldives (Male and the atolls round it, 47 km),
  Lakshadweep and San Andres. Without it these were a few specks in a frame 150 to 1,100 km
  across.

**Known limits.** NE 10m's coastline and the 2015 FUA outlines are what they are: along a coast the
line can sit a pixel or two off where the people stop. Sparse land under 10 people/km2 is drawn
dimmer than the page's ramp would draw it. A 100 m cell that is empty inside a populated pixel is
averaged in, so metro edges look speckled at coarse pixel sizes.

## Known limits

- NE 10m's coastline is coarse for small, heavily reclaimed places; nearest assignment is right
  for nearly everyone but blurs places like Macao and Singapore by a few %. A finer admin-0
  layer would fix it.
- Bins are 0.02 log10 wide (about 4.7%) and unsmoothed, so small countries' curves are noisy.
