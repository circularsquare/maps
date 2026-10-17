# Data register by country

What each country has for every demand input, what the game uses, and which procedure (archetype)
its packs follow. Started 2026-10-16 (T-501).

**Reading it.** One row per country, the inputs split over four tables. A cell is a status mark,
then the source and its grain:

- **U** used by the country's packs
- **N** usable, not used yet: the task that would use it
- **G** gated: exists, but behind a key, an application, a licence or a bot wall (why)
- **?** said to exist, not opened yet (task where one is worth it)
- **skip** usable, judged not worth it (why)
- **-** none found

Sources' URLs and the measurements behind each choice live in the country's note (last column of
the first table); the rules themselves are in demand-worldwide.md (per country) and demand.md (the
US). **Updating it**: a city task that uses, finds or rules out a source changes that cell, filing a
P3 task for anything marked N. A new country gets a row in every table before its first build.

## Countries and their procedure

| Country | Packs | Archetype | Notes |
|---|---|---|---|
| United States | nyc phl bos wdc bal pvd nhv hfd spf wor wnd nlc lit | A | docs/demand.md, notes/T-142.md |
| Canada | tor | A | notes/T-277.md |
| England (UK) | lon | A | notes/T-288.md (T-360 part); calibration (T-509): U TfL NUMBAT 2024 station entries and line boarders, U ORR Table 1410 station usage 2024-25, U Census 2011 QS701EW travel to work by MSOA (2021 TS061 skip: lockdown), U GB-wide GTFS (aubin.app, Transitous) for National Rail; G Network Rail and RDG timetable feeds (login) |
| France | par | A, by commune | notes/T-289.md; calibration (T-509): U IDFM GTFS, U IDFM validations per ZdC per day, U INSEE RP 2022 transit share to work by commune (no rail split); G EGT 2018-20 result tables (omnil.fr 403 to scripts) |
| Germany | ber | A, by Gemeinde | notes/T-290.md |
| Japan | tyo osa ngo fuk spk sdj hij, 38 zones | A | notes/T-420.md, T-432, T-463; calibration (T-509): U KSJ S12 station counts, U 2020 census table 18 rail share by municipality of residence (e-Stat e18.xlsx); from them a country rail constant of +0.5 on every Japanese pack (Sendai's fit; docs/demand-worldwide.md Japan) |
| Brazil | sao | B | notes/T-458.md |
| China | sha | C, curve from a mean | notes/T-291.md |
| South Korea | seo | C, curve from a mean | notes/T-457.md; calibration (T-509): U seoulriders' timetable, card gate and KRIC station counts; G census commuting mode by si-gun-gu (KOSIS) |
| India | mum | C, curve from bands | notes/T-292.md |
| Nigeria | lag | D | notes/T-459.md |

## Population and geography

OSM completeness c is T-456's: OSM errand and leisure places per 1,000 residents over 5 and 3,
capped at 1 (errands / leisure). It decides how far the everyday fallback trusts OSM.

| Country | Population | Boundaries | OSM completeness c |
|---|---|---|---|
| United States | U 2020 census blocks, institutions off (PL 94-171 P5) | U TIGER counties | not needed (LODES sectors); OSM about half of employer shops |
| Canada | U 2021 dissemination areas (98-10-0015-01); N dissemination blocks, T-282 | U Esri Canada copies of StatCan files; G official files (www12 bot check), T-282 | 0.94 / 0.92 |
| England | U Census 2021 by LSOA (TS001), GHSL inside; skip output areas (LSOA already changed little, T-360) | U ONS ITL3 and LAD polygons | 0.98 / 1 (BRES in use) |
| France | U legal population by commune (geo.api.gouv.fr); N Filosofi 200 m squares, T-384 | U geo.api.gouv.fr communes | 1 / 1 |
| Germany | U Zensus 2022 100 m grid scaled to Gemeinde totals (VG250-EW) | U BKG VG250-EW | 1 / 1 |
| Japan | U 2020 census 500 m mesh | U KSJ N03 municipalities | Tokyo 0.56 / 0.68; other packs about 0.33-1 |
| Brazil | U IBGE 2022 census tracts | U IBGE tracts dissolved | 0.24 / 0.17 |
| China | U 2020 census by district (city bulletin), GHSL inside | U OSM admin_level 6; skip geoBoundaries ADM2 (1:4M) | 0.07 / 0.13 |
| South Korea | U resident registration by si-gun-gu (MOIS, open CSV); skip eup-myeon-dong (same file, little gain at 2 km); G 2020 census table (KOSIS: no account without a Korean ID; browser download only) | U OSM relations by Korean name | 0.51 / 0.93 |
| India | U Census 2011 by sub-district, grown by district to 2020; 2027 census not out | U OSM admin_level 5 and 6 by name; G SHRUG polygons (non-commercial, checks only) | 0.04 / 0.05 |
| Nigeria | U GHSL alone; - no credible count (census 2006); ? GRID3 100 m estimates, T-507 | U OSM LGAs | 0.02 / 0.01 |

## Jobs

| Country | Jobs at work | Jobs by industry (everyday destinations) | Residents by industry |
|---|---|---|---|
| United States | U LODES WAC, block | U LODES WAC sectors, block | N LODES RAC sectors, block, T-500 |
| Canada | U 2021 census workers by tract of work (98-10-0498-01), lockdown at-home workers put back | U 98-10-0492-01, tract x sector (leisure; errands fail the concentration test) | ? census industry by tract of residence, T-500 |
| England | U BRES 2024, LSOA | U BRES by LSOA x SIC division | N Census 2021 industry by LSOA of residence (TS060), T-500 |
| France | U census flows (MOBPRO 2022) summed by commune of work | N Flores 2021, commune x A38 (trade mixed with wholesale), T-503 | ? MOBPRO detail file, T-500 |
| Germany | U Pendlerrechnung 2024 by Gemeinde of work (Berlin one unit; - no Bezirk table found) | - Regionaldatenbank 13111 has 4 groups by Gemeinde, too coarse | ? not looked for |
| Japan | U 2021 economic census 500 m mesh, scaled to census workers by municipality of work (public service left out, T-461) | U the same mesh by major industry, 9-1B splits by municipality | N 2020 census industry by municipality of residence, T-500 |
| Brazil | U OD 2017 survey, workers by zone of first job | ? survey's job sector, not checked | ? survey's job sector, not checked |
| China | U 5th economic census by district (legal-person units) | ? economic census by district x sector, T-503 | - |
| South Korea | tier 0 proxy; G Census on Establishments by si-gun-gu (KOSIS: no account without a Korean ID; browser download only), T-504 | G the same census by sector, T-504 | - |
| India | tier 0 proxy (population 0.5); N 6th Economic Census 2013 by ward, T-388 | - | - |
| Nigeria | tier 0 proxy (population 0.5, land cap) | - | - |

## Commuting

Work at home: people whose workplace is their home, every working day. Telework: part-week home
days of people with a workplace. Several countries have one figure for both.

| Country | Commute flows (the curve) | Travel survey | Work at home | Telework |
|---|---|---|---|---|
| United States | U LODES OD, fitted | U NHTS 2017 and 2022: weekend factors, medical rates, long trips | U ACS by tract and industry | U in ACS's figure |
| Canada | U 2021 census flows between municipalities (98-10-0459-01, lockdown), fitted; N 2016 flows for a refit, T-282 | ? Toronto's TTS, not checked | U LFS May 2024, one share (24.7%) | U in the LFS share |
| England | U 2011 census flows (WU01EW), MSOA, fitted; 2021 flows lockdown, checks only | G National Travel Survey microdata (UK Data Service licence), not checked | - (Census 2021 lockdown) | U ONS OPN 2024, one share (0.27) |
| France | U MOBPRO 2022 commune flows, fitted | ? Ile-de-France EGT, not checked | U Eurostat LFS usually at home | U LFS sometimes (0.12 in all) |
| Germany | U Pendleratlas, ten largest flows per Gemeinde, fitted | G MiD 2017 microdata (application), not checked | U Eurostat LFS usually at home | U LFS sometimes (0.10 in all) |
| Japan | U 2020 census table 4, municipality flows, fitted; joint gravity across packs | N 2018 Tokyo person trip survey (everyday rates), T-450, waits on calibration | U census table 4, share by municipality of work | U MLIT 2024 telework survey by region |
| Brazil | - no census flows; U OD 2017 home-to-job distances, fitted | U Metro OD 2017 microdata; its everyday rates are T-450's evidence | U survey (9.1%) | in the survey |
| China | - no open flows; U published mean commute (10.2 km, Shanghai survey via press; CAUPD-Baidu report for about 45 cities) | U the survey's mean only | U 0 | U 0 |
| South Korea | G 2020 census commuting by si-gun-gu (KOSIS: no account without a Korean ID; browser download only), T-504; U mobile-data mean commute (Statistics Korea, 20.4 km) | ? KTDB household survey, not checked | U 0 (no figure checked) | U 0 |
| India | U Census 2011 B-28 distance bands by district, both parameters fitted | - | U B-28 "no travel" (20.1%) | - |
| Nigeria | - none; U the borrowed curve (T-314) | - (one 2010 paper's mean trip, too weak to fit) | U 0 | U 0 |

## Car ownership

People living in a household with no car, per cell (T-511, `pipeline/nocar.py`; the rule in
demand.md 4.3). Where only cars per household are known, households without a car come from one
curve fitted on US tracts and English LSOAs (demand-worldwide.md "Driving alternative"). Every
country has a figure (T-562): the rows below, then the three rule rows at the end in that order;
`nocar.py --plan` prints each planned country's. No open world register exists: WHO's GHO counts
all registered vehicles with motorcycles (2016, one total), the World Bank's IRF series stopped
about 2011, OICA's vehicles-in-use pages are gone (404) with no licence stated.

| Country | Homes without a car |
|---|---|
| United States | U ACS 2019-23 B08201 by tract, people in no-vehicle households (nearest populated block's tract) |
| Canada | U StatCan light-duty registrations 2023 per 1,000 x census household size, national; - census asks no vehicle question |
| England | U Census 2021 TS045 by LSOA (households, made people) |
| France | U Eurostat passenger cars per 1,000 (2024) x household size, national; N INSEE census housing base by commune or IRIS, T-512 |
| Germany | U Eurostat passenger cars per 1,000 (2024) x household size, national |
| Japan | U AIRIA private cars per household by prefecture (March 2025); - no small-area figure found |
| Brazil | U Senatran automóveis per 1,000 (2024) x household size, national; ? OD 2017 survey's car ownership, T-512 |
| China | U Ministry of Public Security cars (2024) per 1,000 x household size, national; ? Shanghai yearbook cars per 100 households, T-512 |
| South Korea | U MOLIT passenger cars per 1,000 (2024) x household size, national; G KOSIS small areas, T-504 |
| India | U MoRTH cars, jeeps and taxis per 1,000 (2022) x household size, national |
| Nigeria | U NBS private vehicles per 1,000 (2018) x household size, national; N DHS 2024 by state, T-570 |
| Wales | U Census 2021 TS045 by LSOA (Nomis, households, made people) |
| Scotland | U Census 2022 UV405 by data zone 2022 (NRS topic tables zip; maps.gov.scot boundaries; OGL) |
| Northern Ireland | U Census 2021 car or van availability by data zone 2021 (NISRA ArcGIS layer, counts people in households) |
| Hong Kong, Singapore, Taiwan | U Transport Department, LTA, Highway Bureau private cars per 1,000 x household size, national (the GDP rule cannot hold for city states; Taiwan has no World Bank figures) |
| Rule 1: DHS countries | U DHS API (open, no registration) HC_TRNS_H_CAR, households possessing a private car, national, latest survey from 2015 (rule 3: from 2010), 51 planned countries; N the same by region, T-570 |
| Rule 2: Eurostat countries | U road_eqs_carhab cars per 1,000 x ilc_lvph01 household size, latest year (CC BY 4.0); 20 planned countries |
| Rule 4: everything else | U households with a car from World Bank GDP per head PPP (NY.GDP.PCAP.PP.KD), logit fit (notes/t562_fit.py); no GDP (Cuba, North Korea, Venezuela): the World Bank income group's median; 46 planned countries, 116 rows; N census or register figures for the larger ones, T-571 (Mexico's INEGI 2020 car availability by municipality among them) |

## Travel surveys

Everyday outing rates and distance scales by purpose (T-509 stage 4, `pipeline/everyday_rates.py`);
a country without one takes the surveys' median.

| Country | Rates | Lengths |
|---|---|---|
| United States | U NHTS 2017 microdata, weekday legs by purpose (on disk); N NHTS 2022 | U NHTS New York CBSA legs under 46 km, road miles / 1.3 |
| Canada | - (TTS is Toronto-only, not opened); worldwide | worldwide |
| England (UK) | U NTS0504b weekday trips by purpose, 2015-19; G NTS and LTDS microdata (UKDS) | U NTS0413 London residents' errand lengths; social worldwide (no cut-off in the published means) |
| France | U EMP 2019 weekday motifs table (SDES); N EMP microdata (open, media/5052) for medical and exact lengths; G EGT 2018-20 (omnil.fr 403) | U EMP declared km x the Paris area factor |
| Germany | U MiD 2017 Ergebnisbericht by Zweck, made weekday; G MiD tables tool (login) | U SrV 2018 Berlin Steckbrief |
| Japan | U Tokyo PT 2018 press tables (private split); U national survey 2021 (MLIT Excel) as a check; G Tokyo PT tabulations (application form, login) | U Tokyo welfare bureau 2021, straight-line km by purpose |
| Brazil | U Sao Paulo OD 2017 microdata | U the same, straight line |
| China | -; worldwide | worldwide |
| South Korea | U KTDB 2016 national O/D report, capital-area sample counts; G KTDB 2021 survey (account) | U Gyeonggi transport DB 2016 bands (Seoul, Gyeonggi) |
| India | -; worldwide | worldwide |
| Nigeria | -; worldwide | worldwide |

## Destinations

| Country | Schools | Universities | Medical | Airports |
|---|---|---|---|---|
| United States | U NCES CCD public schools; N NCES PSS private, T-142 | U IPEDS | U OSM (relations skipped, T-203) | U FAA enplanements, DB1B connecting share |
| Canada | U Ontario school file (per province) | U COU and Ontario college files (per province); N partner campuses, T-282 | U OSM | U StatCan 23-10-0253-01; connecting by judgement |
| England | U GIAS | N HESA Table 1 by provider, T-382 | U OSM | U Wikidata; Heathrow 0.7 by judgement |
| France | N ministry directory with pupils, T-385 | ? ministry students by site, T-505 | U OSM | U Wikidata; CDG 0.75 by judgement |
| Germany | U tier 0 OSM (school files per Land) | ? Destatis 21311 by institution, T-505 | U OSM | U Wikidata |
| Japan | U KSJ P29 points, mesh children 0-14 | U tier 0 (short in some zones, T-462) | U OSM, beds almost never tagged (sized by rule, T-387) | U Wikidata; Narita 0.8 by judgement |
| Brazil | U tier 0 OSM; ? INEP school census, T-506 | U tier 0 | U OSM; ? CNES beds by establishment, T-506 | U GRU via the press, Congonhas Wikidata |
| China | U tier 0 OSM | U tier 0 | U OSM, one hospital with beds (sized by rule, T-387) | U CAAC bulletin via the press |
| South Korea | U tier 0 OSM | U tier 0 (3 of 217 sites with Wikidata counts) | U OSM, no beds tagged (sized by rule, T-387) | U Wikidata; Incheon 0.85 by judgement |
| India | U tier 0 OSM; G UDISE+ enrolment (behind the portal) | U tier 0 | U OSM, one hospital with beds (sized by rule, T-387) | U the operator's total via the press |
| Nigeria | U tier 0 OSM; ? GRID3 schools, T-507 | U tier 0 (World Bank rate from 2011) | U OSM, no beds; ? GRID3 health facilities, T-507 | U FAAN via the press |

Everywhere: places and parks from OSM, water from OSM (TIGER and StatCan in the US and Canada),
world market from the packs. U hospital beds per 1,000 people from the World Bank (SH.MED.BEDS.ZS,
CC BY 4.0, the country's latest year, cached in `pipeline/hospital_beds.json`; none for Taiwan,
which takes the world figure) scale every pack's hospital beds (T-387). ? Overture Places as a second place source where c is under 0.3
(Brazil, China, India, Nigeria), T-508. Tier 0 alone covers any input marked tier 0 above; its rules are in
demand-worldwide.md "Tier 0".

## Every country: the world plan

Which metros get a pack and which units each holds (`pipeline/world_plan.py`, T-536; the rule in
adding-a-city.md 1). Downloads in `data/raw/worldplan/`, fetched when missing.

| Input | Source | Terms |
|---|---|---|
| Which cities | U GHSL Urban Centre Database R2024A, 2025 population (11,422 centres), from `citybrowser/data/` | EC reuse with acknowledgement |
| Commuting areas | U GHS-FUA R2019A (`GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0`, 9,031 areas, 2015 epoch; still the latest release, checked 2026-10-16) | "Reuse of this data is authorised with proper acknowledgment of the source" (GHSL datasets page; EC reuse policy) |
| People | U GHS-POP E2025 1 km, Mollweide (R2023A) | as above |
| Units, Europe | U NUTS-3 2024 (Eurostat GISCO, 1:3M; EU, EFTA, candidates); U ONS ITL3 January 2025 for the UK | NUTS: "© EuroGeographics for the administrative boundaries"; GISCO's own download clauses not read yet (they restrict some admin datasets to non-commercial use), check before publishing |
| Units, elsewhere | U geoBoundaries gbOpen, simplified, ADM1-4 (the level per country is in `world_plan.json` `unit_sources`, with each file's licence: CC BY, ODbL or public domain) | per country; GADM ruled out (non-commercial) |
| Country of a centre | U Natural Earth admin 0, 1:10M | public domain |
| Extracts | U Geofabrik `index-v1.json` | ODbL (OSM) |
| Time zones | U timezonefinder 6.6 (offline, timezone-boundary-builder) | MIT; data ODbL |
| Outline of a planned city | U OSM `boundary=administrative` relations from the row's Geofabrik extracts, matched to the plan's units (`osm_units.py`, T-537) | ODbL |
| Informal work (jobs blend) | U ILO modelled vulnerable employment, World Bank SL.EMP.VULN.ZS (T-537); - informal employment SL.ISV.IFRM.ZS, archived by the World Bank | CC BY 4.0 |

## Archetypes

Four procedures are in use. Each names what it needs; everything else comes from tier 0.

- **A. Census flows.** Needs a commute flow table between small units (municipalities or finer)
  and jobs at work by the same or finer units. Jobs from the table (or an employer register), the
  curve fitted on the flows with the pack's own margins. Grain varies: blocks, tracts, LSOAs or
  500 m meshes (US, Canada, England, Japan), communes or Gemeinden (France, Germany, where a big
  city is one or a few units and tier 0's proxy shapes the jobs inside it). Worked examples:
  Toronto, London, Paris, Berlin, Tokyo.
- **B. Travel survey.** Needs a household travel survey's microdata with home and job zones. Jobs
  by survey zone scaled to the census, work at home from the survey, both curve parameters fitted
  to home-to-job distances. Population from census small areas. Worked example: Sao Paulo.
- **C. District totals.** Needs population (and ideally jobs) by district from a census or
  register, and a published commute figure: a distance-band table (fit both parameters, India) or
  a mean or median (power 1.4, length fitted; China, South Korea). Without a jobs table, tier 0's
  proxy with a larger population share where work is informal. Worked examples: Shanghai, Mumbai,
  Seoul.
- **D. Global only.** Nothing national is credible or open. GHSL population unscaled, the jobs
  proxy with population 0.5 and the land cap, a borrowed curve (`"curve": "borrowed"`,
  `tier0.borrowed_curve`, T-314), work at home 0. Worked example: Lagos.

**Deciding for a new country.** Fill its row in every table first (an hour of searching at most;
look for mirrors when an official server walls, and never take a source from memory). Then take
the first that holds:

1. A commute flow table between units no bigger than municipalities, open: **A**. Check the
   jobs column next: jobs at work by small area keep A's full grain; otherwise jobs from the flows'
   destination totals (Paris).
2. A household travel survey's microdata for the metro, open: **B** (even if the census is good:
   one file gives jobs, work at home and the curve).
3. Population by district from a census or register within about ten years, and some published
   commute figure: **C**. Take jobs by district if any table has them (China's economic census).
4. Otherwise **D**.

Then work down the columns: residents and jobs by industry, schools, universities, medical and
airports each take the national source if it is U-grade elsewhere, else tier 0. Follow the
worked example's paragraph in adding-a-city.md "Outside the US" for the chain of commands.
A gated source does not change the archetype until it is in hand; it goes in its cell with the
task that would use it.
