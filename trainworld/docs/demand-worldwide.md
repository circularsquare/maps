# Demand assumptions to recheck for cities around the world

Anita's list (2026-10-12): every demand rule must hold up worldwide, so for a city with a very
different density pattern, commute culture, car ownership or income. Each entry says what the
game does now, why it might not carry over, and what would check it; a task that adds or changes a
demand rule updates its entry. Demand is simple on purpose (Anita, 2026-10-17, T-636): no fitted
calibration, Anita's two ceiling tests the targets, real cities loose checks. The measurements
behind the entries are in the notes each cites and in notes/demand-measurements.md.

Which sources each country has, which are used, and which procedure its packs follow:
[data-register.md](data-register.md). The sections below hold the rules those sources feed.

## The first country outside the US: Canada (Toronto, T-277, 2026-10-13)

What Canada's data changed, rule by rule (sources in notes/T-277.md, the adapter in
`pipeline/canada.py`, the substitution table in docs/adding-a-city-by-hand.md "Outside the US"):

- **Institutions** are not taken off homes: Canada publishes people in collective dwellings only
  for larger areas, not by dissemination area. Its incarceration rate is about a fifth of the
  US's, so the error is smaller than in a US pack; a big care home or prison stays a home.
- **Job locations** come from the census's workers by census tract of work, not an employer
  register. The 2021 census was taken in a lockdown (35.6% of the Toronto CMA's workers at home,
  counted at home): their jobs are put back at workplaces by employment income band, and "no
  fixed workplace" workers added in proportion. Downtown Toronto is probably still under its
  pre-pandemic job count; for today's commuting with 24.7% at home it is plausible. Not
  reweighted by OSM offices (DECISIONS-to-2026-10-19 2026-10-13).
- **Work from home** is one share for the whole pack (Labour Force Survey, May 2024, Toronto CMA
  24.7%), applied as "the same fraction of 2021's at-home workers commutes again" at both ends,
  Hamilton and Oshawa included (28.3% and 27.3% at home in 2021; their own current share is
  probably a little lower). US packs use ACS by tract and industry.
- **The distance curve** fitted on Canada's census flows between municipalities is much steeper
  than the US one: d^-1.2 exp(-d/20 km) (notes/T-277.md). Partly real (a strongly
  centred city, as the "distance curve shape" entry below predicted), partly the lockdown: the
  commuters of May 2021 were the ones still going in, more local than office commuters. Recheck
  on the 2026 census flows.
- **Everyday destinations** are jobs by sector by census tract of work (98-10-0492-01, NAICS 44-45
  and 81, 71 and 72; T-453), with the lockdown caveat of the jobs above.
- **Job places within a tract** are 80% OSM job places (with commercial and industrial land at one
  place per 5,000 m², judgement) and 20% population.
- **Airports** have no connecting share from a national source (no DB1B): Pearson takes 0.75 by
  judgement, the share of the US hubs DB1B measures (Dulles 0.74), the others 1.
- **Schools and universities** come from provincial files (Ontario's), so each province needs its
  own; part-time college students are not counted.
- **Borders**: Ontario is one region with no factor; the pack does not reach the US border.

## Tier 0: global sources only (`pipeline/tier0.py`, T-287, 2026-10-14)

The adapter for countries with no small-area census inputs, with a country's own inputs in place
of its proxies where it has them. Measured by building Toronto from tier 0 against the real pack
(notes/T-287.md has every table; London's in notes/T-288.md and T-360):

- **Population** is GHSL GHS-POP (100 m) scaled to district totals; areas built up after the GHSL
  epoch come out short (Toronto's newest municipalities 15-27%), so use the latest epoch and the
  newest district totals for fast-growing cities. GHS-POP spreads each of GHSL's own census units
  over its residential volume (GHS-BUILT-V total less NRES), so people per residential m³ is flat
  inside a unit and steps at its edge, 10-30x where those units are old and coarse (China). Each
  100 m pixel's people per residential m³ is held within 2x of its neighbourhood's (a 2 km
  Gaussian; `POP_BOUND`, `POP_BOUND_SIGMA_M`, T-1150), then the people are scaled back to GHS-POP's
  total over the pack; a pixel with no residential volume keeps GHS-POP's figure. Where GHSL's
  units are sound this moves almost nothing (notes/T-1150.md). A country's census by small area is
  spread by this bounded figure too.
  Packs built before T-1150 keep the unbounded GHS-POP until their `<id>.tier0_signals.npz` is
  deleted and the pack rebuilt.
- **Jobs** are a proxy: population x the World Bank employment rate, spread 30% by GHSL
  non-residential volume weighted by height^2, 50% by OSM job places, 20% by population (T-360,
  the best joint fit on Toronto's census and London's BRES). The height in that weight stops at 40 m (`HEIGHT_CAP_M`,
  T-573): above it a cell's weight grows with its volume alone, since GHSL's mean height on a
  100 m pixel with one tower or hangar on little floor (Berlin's Treptowers, 70 m) otherwise gave
  that cell a city's share. OSM industrial, commercial and retail land counts one job
  place per 5,000 m², one area's weight capped at 1 km² (`land_cap_km2`) and spread over a 0.003
  degree (~330 m) grid inside it (`land_spread_deg`), in every build (T-460); a config may set
  either to 0 or another value. Where the jobs are this proxy alone (T-583), one OSM place takes at
  most 0.75% of all jobs and one cell at most 500 x the square root of the jobs in thousands
  (`PLACE_JOBS_CAP`, `CELL_JOBS_K`) through the two building parts; the jobs the caps leave over go
  to the population part, and the pack's jobs source says how many. Only packs where OSM and GHSL
  are thin reach them. A country with workplace counts by small area (England's
  BRES by LSOA) replaces the total and the proxy only spreads each area's count (tier0's `jobs`
  hook). Why it may not carry over: an old mid-rise centre with a large commuter shed defeats it
  (London's proxy gives Westminster and the City a third of their BRES jobs, and no blend fixes it
  without wrecking Toronto, whose centre is towers). The total is the residents' rate, so a pack
  that draws commuters from beyond its edge is short (London -19%). Where much work is informal
  or at home (Mumbai), buildings miss it, so the population share rises: in a planned city it is
  the country's ILO vulnerable employment share held to 0.2-0.5, the other two shares moving with
  it to 0.2 and 0.3 at the top (T-537; a national figure with farming in it, so it overstates a
  middle-income city's informal work); where industry is
  large, industrial land overdraws (Brampton +40%). Check, for the next big old centre (Paris):
  the proxy's core against any published central job count.
- **Where workers live** is population x one rate (no source by residence), so student and
  retiree areas send too many commuters.
- **The distance curve without flows**: with no flow table and no distance bands, the city
  borrows one curve (`"curve": "borrowed"`, `tier0.borrowed_curve`): of the fitted cities'
  curves (`pipeline/fitted_curves.json`), the one that misplaces the fewest commuters beyond each
  city's own, averaged by country; today d^-1.8 exp(-d/80 km). Held out a country at a time it
  beat nearest by population and vehicles, by income group, a world median and a length scaled to
  the city's size (T-314). Why it may not carry over: no fitted country has the US's flat d^-0.8
  curve, so a sprawling car city would be served worst. A **non-city zone** with no flow table (a
  countryside zone of a covered country, T-723; Anita, T-719 decision 5) borrows by the same rule
  over the fitted Japanese prefecture zones only (`"curve": "borrowed_zone"`): today d^-1.4
  exp(-d/16 km), where the metros' curve would send rural commuters far too far. Why it may not carry over:
  every fitted zone is Japanese, so a country with longer rural commutes (Spain's interior,
  dispersed farm work) may sit on too short a curve. Where a distance distribution exists (India's census bands, a travel
  survey), fit both parameters; reported (road) distances are divided by 1.3, the routed drive
  factor of Mumbai's commutes at the band edges (T-292). A published median or mean alone fits
  only the length at power 1.4 (`tier0.fit_curve`); medians are often of exact or road distances
  and fit too flat a curve (Toronto's 10.5 km gave 95 km), so check one against any flow table.
- **Schools** are OSM sites sized by the square root of site area with pupils from the national
  share aged 0-14 (OSM's school polygons are uneven, and site area is a weak guide to pupils:
  London's came out far more misplaced than Toronto's). Public and private alike. One site holds at most 3,000 pupils (`SCHOOL_SITE_MAX`, T-574, about a large
  secondary school) and the sites in one pack cell 6,000 together (`SCHOOL_CELL_MAX`: a campus
  mapped as several schools), the excess going to the other sites in proportion; where OSM maps too few
  schools for that (Lagos 861 sites for 5.0M pupils, Mumbai 803 for 6.2M), the pupils no site
  can hold go to school where they live, in proportion to children.
  Why it may not carry over: a city whose schools OSM barely maps gets mostly local school trips,
  which is right for primary pupils and short for secondary ones. A country register with pupils
  and coordinates (England's GIAS) replaces the sites and sizes (tier0's `schools` hook).
- **Population by small area**: GHSL scaled to district totals is already close at the game's
  2 km scale (London against Census 2021 by LSOA), so finer census counts (tier0's `population`
  hook) matter little for commuting. **District totals matter in old dense centres**: GHSL put
  12-19% too many people in London's central boroughs before scaling (T-288).
- **Universities** are OSM campuses with Wikidata student counts, which few OSM campuses carry,
  so most are sized by area. A campus item with no count takes its institution's (part of, parent
  organisation, occupant or operator: UBC Vancouver's is UBC's 71,201), whole, campuses outside the
  pack not taken off (T-3216). Sites named as open or distance universities or research centres
  are left out (IGNOU, CNEA's Centro Atómico Ezeiza; T-3226). Every site has a cap from how
  counted campuses' students go with their area (1,872 x hectares^0.31): a count is cut to 14
  times that and to 100,000 and then caps its sites, so a whole system's count on one campus
  (UNAM's 360,883 on CU) or a scaled-up count (IIT Bombay's 176,000) can't stand; a site sized
  by area takes at most 5.4 times that and 50,000. The total is population x the national tertiary students a resident
  (UNESCO enrolment via the World Bank, SE.TER.ENRL over SP.POP.TOTL; T-288), and the counts and
  areas only spread it (counts alone sized London at several times its students), none past its
  cap; students no site can take study near home, by adults, as schools' do. Why it may not
  carry over: students gather in university cities, so a national rate undercounts a city with
  many campuses (Toronto -20% against Ontario's files) and overcounts one with few. Check: London
  against HESA (T-382). A campus OSM maps as an outline has its students spread over the cells
  under it, half by GHSL non-residential volume and half by area (`CAMPUS_BUILT_SHARE`; T-3214:
  one cell under the outline's inner point put all of UBC's 45,000 2.2 km from its bus exchange);
  a point, or an outline too small to hold a res-11 cell centre, stays in its one cell. Packs pick
  this up when their OSM education cache is made again (a row's rebuild deletes it).
- **Airports**: patronage from Wikidata (it matched StatCan's for Toronto's three airports).

## France (Paris, `pipeline/france.py`, T-289, 2026-10-15)

Tier 0 with two INSEE upgrades; notes/T-289.md has the tables.

- **Jobs**: INSEE's census flows (Mobilités professionnelles 2022, every flow, Paris by
  arrondissement) summed by commune of work, spread inside each commune by tier 0's blend. The
  census counts teleworkers at their usual workplace, so no lockdown correction. Communes are the
  finest unit published; inside Paris an arrondissement is 0.9-16 km². Tier 0's proxy alone falls
  well short in Paris's core, if less than in London's.
- **Population**: the legal population by commune (geo.api.gouv.fr serves the latest), GHSL
  inside each commune. Filosofi's 200 m squares would refine it (T-384).
- **Curve**: fitted on the same flows, d^-1.3 exp(-d/20 km). The fit is flat near it (tier 0's
  one-parameter rule within half a point).
- **Work from home**: one national share, 0.12, from Eurostat's LFS (lfsa_ehomp: 11.1% usually,
  24.4% sometimes, France 2025), judged at 0.6 and 0.2 of their days. Ile-de-France works at home
  more than France as a whole; not modelled.
- **Schools**: tier 0's OSM sites; the ministry's files with pupils are T-385.

## China (Shanghai, `pipeline/china.py`, T-291, 2026-10-15)

Tier 0 with district totals; notes/T-291.md has the tables.

- **Districts**: OSM admin_level 6 relations, clipped to land by the OSM coastline (Chinese district
  lines run far out to sea). Relations that do not assemble are rebuilt from their admin_level 8
  units or as the province's remainder. geoBoundaries' China ADM2 is too coarse (1:4M, 2017).
- **Population**: the 2020 census by township (T-1150): helper1m's China table, the ASPECT 100 m
  grid (built on 40,718 townships' 2020 counts) summed over 2018 township outlines and scaled to
  the census counties, copied to `data/raw/tier0/china_townships/townships.fgb`. Each township's
  count is spread inside it by GHSL's residential volume (total less NRES; tier 0's bounded GHSL
  where a township has none); a township the outline cuts brings the share
  of its GHSL people inside; cells in no township keep GHSL's people at the pack's census-to-GHSL
  ratio. Shanghai's result is then held to its bulletin's district totals. Every planned mainland
  metro row takes the townships (`china.plan_hooks`); the countryside zones stay on GHSL alone.
  GHSL has about a quarter too many people in Shanghai and puts them in blocks by its own older
  units, which district totals cannot fix. Packs built before T-1150 keep the district or GHSL
  figures until rebuilt (notes/T-1150.md "Rebuild").
- **Jobs**: the 2023 economic census, persons employed in legal-person units by district (each
  province's bulletin no. 1 has the table), the proxy inside each. Individual businesses are left
  out of that table. The proxy alone has London's old-core shortfall and overdraws the industrial
  outer districts.
- **Curve**: power 1.4, length fitted to the published mean commute (10.2 km, the city's 2024-2025
  transport survey) taken as straight-line, as CAUPD-Baidu's yearly commute report (location data,
  about 45 cities) confirms; that report is the general source for other Chinese cities.
- **Work from home**: 0.
- **Airports**: CAAC's yearly airport bulletin (passengers / 2) in place of Wikidata.
- **Names**: OSM `name:en`, then `name:zh-Latn-pinyin`, for places (two thirds of Shanghai's have
  one); the Chinese name stays a search alias.
- **OSM is thinner** (one hospital with beds, few job places and medical sites for 25M people):
  medical and everyday destinations are likely short; not checked.

## India (Mumbai, `pipeline/india.py`, T-292, 2026-10-15)

Tier 0 with sub-district totals and a curve fitted to distance bands; notes/T-292.md has the tables.

- **Units**: OSM boundary relations by name, districts at admin_level 5 and talukas at 6. OSM's lines
  between neighbouring talukas need not follow the 2011 ones (Kalyan, Bhiwandi, Ulhasnagar and
  Ambarnath): where GHSL's share of a unit is far off its census share but a group's sum agrees,
  the group is scaled as one (`india.GROUPS`). SHRUG's PC11 polygons are non-commercial: checks only.
- **Population**: the latest census is 2011. Its Primary Census Abstract by sub-district (one
  workbook a district on censusindia.gov.in/nada), grown to 2020 by each district's 2001-2011
  growth (table A-2), GHSL inside each (GHSL alone has the island city high and the fast-growing
  suburbs low).
- **Jobs**: no workplace table; the proxy with population 0.5 of the blend (`job_blend`) for
  informal and home-based work. It puts Greater Mumbai's share of jobs below its share of workers
  and overdraws port and industrial talukas (T-388 would add the economic census).
- **Curve**: Census 2011 table B-28 ('other workers' by distance band and mode, by district) for
  the pack's districts; both parameters fitted (`tier0.fit_curve_bands`), the reported distances
  read as travel distance and divided by 1.3: d^-1.65 exp(-d/32.7 km). Every Indian district has
  this table, so this is the rule for India.
- **Work from home**: B-28's 'no travel' share of other workers (20.1% in Mumbai's districts).
- **Airports**: the operator's yearly total via the press (passengers / 2) through `boardings`.
- **Names**: OSM `name` is usually already Latin in Maharashtra.
- **OSM is thin**: few school sites for the pupils, one hospital with beds (the rest sized by the
  "Medical" rule below), long drive detours between near zones (creeks and hills).

## Nigeria (Lagos, `pipeline/nigeria.py`, T-459, 2026-10-16)

Tier 0 from global sources alone, the data-poor test; notes/T-459.md has the tables.

- **Units**: OSM local government areas (admin_level 6), every one in the config's states plus named
  ones beyond. Holes in the dissolved outline are lagoon no LGA covers, so they are filled.
- **Population**: GHSL E2020 unscaled. The last census is 2006 and the state's own figures are not
  credible; UN WUP and Africapolis bracket GHSL within about 10%, so scaling would only move it
  inside that range. GHSL lags the fast fringe (Ikorodu): use the latest epoch once the game moves
  past 2020.
- **Jobs**: the proxy with population 0.5 of the blend (as Mumbai). **Where OSM maps few places,
  land use dominates the OSM signal**, and one big industrial area would be one huge job lump at
  its centroid (the Dangote refinery). The tier 0 default for every city (T-460) caps one area's
  weight at 1 km² and spreads it over a 0.003 degree grid (~330 m) inside it, as in "Tier 0"
  above. Markets (Balogun, Alaba, Oshodi) get almost nothing from buildings or OSM; their jobs come
  only from the population share.
- **Everyday destinations**: OSM maps very few errand and leisure places, so the destinations rest
  almost wholly on jobs and residents (completeness 0.02, "Everyday destinations" below).
- **Curve**: no flows or distance table; borrowed by the rule (T-314), d^-1.8 exp(-d/80 km).
- **Work from home**: 0 (no source); informal work near home is in the population share instead.
- **Universities**: the World Bank's tertiary rate is from 2011 (0.0088 a resident).
- **Medical**: no OSM hospital carries beds; all are sized by the "Medical" rule below, to Nigeria's
  World Bank figure of 2004.

## Brazil (Sao Paulo, `pipeline/brazil.py`, T-458, 2026-10-16)

Tier 0 with census tracts and a household travel survey; notes/T-458.md has the tables.

- **Units, population**: IBGE's 2022 census tracts with their people (the "malha com atributos"
  GeoPackage, one per state, open); the pack's municipalities by code, the outline their tracts
  dissolved (GHSL alone runs low in the fast-growing western suburbs).
- **Jobs, work from home, curve: a household travel survey** (the Metro's OD 2017 microdata), the
  rule for any city with one: workers by survey zone of their first job, scaled from the survey's
  population to the census's, spread inside each zone by the proxy; work from home is the share of
  workers whose job is at home (9.1%); both curve parameters fitted to workers' home-to-job
  straight-line distances, with the bands under 4 km as one (2 km zones cannot split them):
  d^-1.8 exp(-d/19 km), mean 8.5 km as the survey's. Other Brazilian metros have published OD
  surveys (the odbr R package gathers them); IBGE's census has no workplace table by small area.
- **Everyday rates** (T-509) are Brazil's from the OD 2017 microdata ("Everyday trip rates"
  below): errands 0.106, social 0.057, medical 0.044 outings a resident; the decay lengths are the
  worldwide set (T-637).
- **Names**: OSM `name` is Portuguese; no `latin_names`.

## Germany (Berlin, `pipeline/germany.py`, T-290, 2026-10-15)

Tier 0 with three national upgrades; notes/T-290.md has the tables.

- **Units**: whole Kreise and their Gemeinden from BKG's VG250-EW (land parts), which also carries
  each Gemeinde's population (the Länder's figure on the Zensus 2022 base). Berlin and Hamburg are
  one Gemeinde each.
- **Population**: the Zensus 2022 100 m grid (residents per square, every square in one 153 MB CSV),
  scaled to each Gemeinde's current population.
- **Jobs**: the Länder's Pendlerrechnung (Pendleratlas): everyone employed, by Gemeinde of work
  (in-commuters plus those working where they live), so civil servants and the self-employed are
  in, unlike the BA's insured employees. Spread by tier 0's blend inside each Gemeinde, so inside
  Berlin the jobs have the proxy's shape.
- **Curve**: fitted on the same atlas's flows, which publish each Gemeinde's ten largest in- and
  out-flows and its own workers (IOP), scored on those pairs only: d^-1.4 exp(-d/28 km). Berlin's
  own internal commuters are one cell, so the fit says little inside Berlin.
- **Work from home**: 0.10, Eurostat LFS (lfsa_ehomp: 13.0% usually, 11.7% sometimes, Germany
  2025), judged at 0.6 and 0.2 of their days.
- **Schools and universities**: tier 0 (schools are per Land in Germany). Berlin holds more than
  its share of Germany's students, so the national tertiary rate likely leaves it short (no
  figure checked).
- **OSM is rich** in job places, schools and medical sites.

## South Korea (Seoul, `pipeline/korea.py`, T-457, 2026-10-16)

Tier 0 with one national upgrade; notes/T-457.md has the sources.

- **Units**: si-gun-gu, with the non-autonomous gu of the large si, as OSM relations matched by
  Korean name inside their province. OSM follows boundary changes within months (Incheon's July
  2026 gu), so a table from an earlier year needs its units mapped onto today's.
- **Population**: the Ministry of the Interior and Safety's resident registration count by
  si-gun-gu (monthly, open CSV, down to eup-myeon-dong), the census month's. Registered foreigners
  are left out. GHSL is far off in the new towns of the 2010s and high in old dense districts:
  district totals matter here.
- **Jobs**: tier 0's proxy (the Census on Establishments by si-gun-gu is on KOSIS only, behind a
  key or the interactive site).
- **Curve**: length fitted to Statistics Korea's mobile-data mean commute (20.4 km in the capital
  area, read as one way along the route, 15.7 km straight-line): at tier 0's power 1.4 that needs
  a 156 km length, close to a pure power curve.
- **Work from home**: 0 (telework is uncommon; no figure checked).
- **Universities**: Wikidata has student counts for few sites; the national tertiary rate scales
  the total, so only the spread by site area is weak.

## Australia (world plan rows, `pipeline/australia.py`, T-1857, 2026-10-30)

Tier 0 with one national upgrade, for every row; notes/T-1857.md has the sources.

- **Population**: the 2021 census by SA1 (200-800 people), GHSL inside each, as Canada's DAs.
- **Jobs**: tier 0's proxy. The 2021 census's place of work and journey to work were counted in the
  Sydney, Melbourne and Canberra lockdowns (August 2021), with home workers coded at home, so they would
  empty the city centres; the flows are also behind TableBuilder's account.
- **Curve**: borrowed (metros) and the zone curve (zones); **work from home** 0, as tier 0.

## Mexico (world plan rows, `pipeline/mexico.py`, T-2080, T-2081, 2026-11-05)

Tier 0 with two national upgrades, for every row once its build takes the module (T-2436); notes/T-2080.md
has the sources and checks.

- **Population**: the 2020 census. Urban AGEBs (median 0.25 km2) with their people, and rural localities
  from ITER as points, each over the cells nearer it than any other within 2 km; GHSL inside each. The units
  add up to the census exactly; GHSL runs 10-15% under it in the big cities.
- **Jobs**: DENUE's units at their points, each at its employee band's mean size, plus the jobs DENUE leaves
  out (farm work, street trade, domestic work: tier 0's World Bank jobs a resident less DENUE's national
  workers a resident, 0.151) over the row's people by tier 0's proxy; the country's total is tier 0's.
- **Curve**, **work from home** and everyday destinations: as tier 0.

## Japan (Tokyo, `pipeline/japan.py`, T-420, 2026-10-15)

Tier 0 with Japan's national inputs, written to serve every Japanese pack (research/japan.md);
notes/T-420.md has the tables.

- **Units**: whole prefectures, dissolved from KSJ N03 municipalities (designated cities by ward,
  the census's unit); in Hokkaido, subprefectures (N03's 振興局 names, `japan.SUBPREFECTURES`), and
  a single municipality where the flows say it belongs (Otaru with Sapporo). N03 draws every pier and rock apart; parts under about 0.02 km² are dropped
  from the outline, and land of no municipality only where it touches one (Tokyo Bay's landfill).
  The 38 non-city zones (T-420 stage 3) also leave out the Russian-administered southern Kurils
  and uninhabited parts under 5 km² more than 50 km from other land. Wards redrawn since the 2020
  census (Hamamatsu, 2024) are read as their whole city on both sides (`japan.CITY_OF_WARD`).
- **Population**: the 2020 census 500 m mesh, each mesh spread over its cells by GHSL; a mesh the
  outline cuts on land keeps its share of land inside. Suppressed meshes are already counted in
  their neighbour, so the mesh totals are the census's (Greater Tokyo within 0.01%).
- **Jobs**: the 2021 economic census 500 m mesh (persons employed, government included), spread
  inside each mesh by tier 0's proxy and scaled per municipality to the 2020 census's employed
  people by place of work (the economic census counts dispatched and multiple jobholders and no
  farms).
- **Curve**: fitted on the census's municipality flows with the pack's own margins; only the steep
  end keeps the census's share working in their own municipality: d^-2.2 exp(-d/80 km).
  Keihanshin's and Chukyo's flows fit it as well as any other (T-431), so it is Japan's metro
  curve. The regional metros fit a little flatter and shorter, each its own (T-432): Fukuoka
  d^-2.0 exp(-d/32), Sendai d^-1.8 exp(-d/50), Hiroshima d^-1.9 exp(-d/32); Sapporo ties Tokyo's
  curve and keeps it. Each non-city zone fits its own over a fixed grid (power 1.2-2.4, length
  16-160 km; `japan_zones.fit`) on its at-home-corrected margins: most fit flatter and shorter
  still (d^-1.4 exp(-d/16) the commonest).
- **Work from home**, per cell: the census's share working at home in the cell's municipality of
  work (table 4; 0.5% in Chiyoda, 21% in Suginami: shopkeepers and farmers work at home in the
  suburbs, office workers do not, T-465), then MLIT's 2024 telework survey share of the region's
  employed, judged at 2 days in 5, on the rest: Greater Tokyo 0.15 (37.5%), Keihanshin 0.098
  (Kinki's 24.4%), Chukyo 0.078 (19.6%), every other pack 0.067 (16.8% outside the three).
- **Commutes between packs**: Japan's 45 packs are one joint commute gravity (docs/zones.md 4), each
  pair across packs on the home pack's curve, cut at 150 km, times 0.4, so the commuters between
  packs match the 2020 census's between the same prefectures less telework. Prefecture lines
  inside a pack take no factor, as the curves were fitted. Why it may not carry over: a symmetric
  crow-flies gravity misses one-way pulls (Shiga into Kyoto and Osaka) and mountain ranges.
- **Schools**: KSJ P29 points (no pupil counts); each 500 m mesh's children 0-14 x 0.9 go to the
  nearest elementary, junior high, compulsory or secondary school. Universities stay tier 0 (where
  no campus has a Wikidata count, as in Fukuoka, the national total is split by campus area).
- **Names**: OSM's `name:en`, else romaji from a kana reading (`japan.kana_romaji`, Hepburn with
  long vowels written short, so ASCII like every other name), else from the kanji's reading in a
  dictionary (`japan.kanji_romaji`, UniDic through fugashi), else kanji. The dictionary takes the analysis with most of the name in
  proper nouns, reads a split-off first 上 下 東 西 南 北 中 新 元 小 as in place names, a final 町
  as cho, ケ between kanji as ga, and writes a block as "N-chome"; about one name in four or five
  it gives is misread (against `name:en`, T-426). The kanji stays as a search alias.
- **Rail constant**: none (T-636), as everywhere. Japan's cycling to stations and its trains'
  share of commuting (Tokyo's 2020 census 52%) are not in the time and money the choice reads, so
  the packs' rail falls short of the counts where habit carries it. Check: Tokyo's and Sendai's
  station entries against the model's level.

## Homes (docs/demand-sources.md 4.1)

- **People in institutions are not homes** (T-234). The US packs take the 2020 census's
  institutional group quarters (prisons and jails, juvenile facilities, nursing homes, other
  institutions; PL 94-171 P5 per block) off every block's population, keep dorms, military
  quarters and other non-institutional places, and cap LODES job holders on those blocks at the
  people left. 0.5-1.3% of each pack's people. Why it may not carry over: censuses differ in
  where they count these people. Some count prisoners at their home address, not the prison
  (several US states now reallocate them for redistricting, though the 2020 PL file does not);
  register-based censuses (the Nordic countries, the Netherlands) count people at their
  registered address, which for long-stay inmates may be the institution or the old home; some
  publish institutional population only for large areas or not by type. Where no small-area
  figure exists, a global fallback is OSM `amenity=prison` and `amenity=nursing_home` polygons
  with a capacity estimate, and a population grid (GHSL, WorldPop) that spreads census totals by
  built-up area will put prisoners and residents of big care homes wherever the census put them,
  usually the institution's district. Elsewhere, where nursing homes are rare (most of Asia and
  Africa, where the old live with family), the rule matters less; where incarceration is far
  lower (Europe, Japan), prisons matter less too. Check, per country: does its census or register
  count prisoners at the prison, and is there an institutional population by small area; for one
  big prison per country, compare the population grid's cells over it with its capacity.

## Commute flows (SPEC 4, docs/demand-kernel.md 4.4)

- **Distance curve shape.** Work gravity uses a curve d^-a exp(-d/L) carried in each pack: in the
  US d^-0.8 exp(-d/30 km) (T-198, from T-185), one value for the nine US packs, fitted on LODES
  (notes/T-198.md); elsewhere a curve fitted on the country's flows or distance bands, or borrowed
  ("Tier 0" above). The flat US near end reflects that people living among many jobs still take
  jobs about as far down the list of nearest ones as anyone else (two-earner households, matching
  by skill). Why it may not carry over: centre-heavy cities fit steeper curves (Toronto d^-1.2
  exp(-d/20 km), London d^-1.85 exp(-d/30 km), T-360), and sprawling car cities (Los Angeles,
  Houston) may need both parameters longer; informal-job cities have no workplace register to fit
  to. Check: one commuting-flow table per new country, fit the two parameters, compare with
  T-185's relative measures (local job ratio, nearest-job rank), not absolute density.
- **The curve flat inside 2 km** (T-1091, Anita 2026-10-23). Every curve but the US one was fitted
  on units coarser than the 2 km squares (boroughs, municipalities, communes, distance bands with
  everything under 4 km in one), so nothing fitted its near end; extended inward from there, the
  steep ones kept two to three times the census's walking commutes in the home square (London
  18.5% walking against the census's 9.4%, Tokyo 23.9% against 7.5%). So a job in the home square
  weighs as one in the next square: f(d) = f(max(d, 2 km)), one rule for every non-US pack
  (`kernel::flat_near_km`). The US curve, fitted on LODES block pairs, matches short commutes
  already (3.2% in the home square against LODES's 3.5%) and keeps its near end. Why it may not
  carry over: where the units are small enough to see the near end (Japan's municipalities, the
  Mumbai and Sao Paulo surveys' bands), the flows want more short commutes than the flat end gives
  (2-3 points more commuters misplaced in Japan, 4-8 in Mumbai and Sao Paulo, under 1 in Europe and
  Toronto); the gap between those flows and the census's walking is the short-trip modes the game
  lacks (bikes, buses: T-634). Check, for a country with fine flows: its own-unit share with and
  without the flat end.
- **Border factor.** Cross-state US commutes are multiplied by 0.23 (Washington 0.45), fitted to
  LODES; outside the US the policy is national borders only, weaker inside Schengen (T-181). It
  does not stop NJ to NYC commuting (Hudson to Manhattan stays near the real figure); it removes
  the excess the plain gravity produced. Anita is unsure state lines matter much in practice.
  Neither the distance curve, water or toll crossings nor a factor between metros explains the
  gap, so in the US the effect is the state line itself, inside metros too (T-171, T-185, T-198).
  Check: for each new country, whether its regional
  borders show any measurable effect in its flow data, and whether a city-state (Berlin, Vienna,
  Mexico City) behaves like DC (a weak factor) rather than a state.
- **Doubly constrained balancing, with a pull cap** (T-1000). Every zone's jobs are filled from
  homes, but no zone's jobs pull more than 20 times harder (or more weakly) than its pack's mean;
  past that they stay part unfilled, or take extra workers. Holds where job counts are reliable
  and most workers commute within the pack. Where jobs data is a proxy (GHSL, T-012; tier 0's
  building volume) a cluster of towns can have far more jobs than workers next to one with far
  fewer, and without the cap the balance sent workers hundreds of km between them (Golmud, Lhasa).
  The cap leaves those proxy jobs part empty instead: station counts at desert mines and oil
  towns show commuters, not proxy jobs. Check, for a new country: the share of jobs held at the
  cap (`gravity.pull.held`), high only in thin, proxy-job country and on islands.
- **Long commutes, a set rate** (T-1000). 1% of every zone's workers commute to jobs 50-150 km
  away, a judgement for "low" (Japan's census has 1.6% of commuters crossing a prefecture line).
  Why it may not carry over: belts of commuter towns 50-100 km out (the Northeast Corridor,
  London's home counties) commute more than that, and the balanced layer carries some of theirs;
  remote country less. Check: the share of a country's commuters over 50 km against its census.
- **Job locations from registers.** LODES places jobs at the reporting establishment; head-office
  lumps (T-043) exist in other countries' registers too, and many countries have no workplace
  data at all (the GHSL proxy would replace it).

## Mode choice

### The curve, islands, damping and propensity

- **The ratio curve** (T-918, docs/demand-mode.md 4.3). Rail against the car is 1 / (1 + q^3) on
  q, rail's perceived cost over the car's, fading to nothing from 3 to 4 times either way. It has
  no constant to fit and no unit: a ratio reads the same whatever the currency, speeds or trip
  lengths of a country, and it spreads with the trip's length by itself. Why it may not carry over:
  where trips are long and slow everywhere (India, Lagos), a ratio judges a 30 minute saving on a 3
  hour trip as worth little, which may undersell a fast intercity line there; the power 3 is
  Anita's choice for play, set against her two ceiling tests, not fitted to any survey. Check: a
  country's rail share by trip length against its travel survey, short and long trips apart.
- **Islands no road reaches** (T-2214, docs/demand-mode.md 4.3 "Between two land parts"). Between
  two land parts of a pack (T-2213: land joined only by a drivable OSM way, from the pack's water
  mask and OSM roads, the same for every country) there is no car and no walk; the player's network
  is the only way, its share held only by the ratio curve's 3-4x fade against a car on a bridge
  that is not there, and the rest no way serves. No constant. Car ferries count as no road, one
  rule everywhere (Anita, 2026-11-03). Why it may not carry over: where road ferries are part of
  the road network (Sweden's vägfärjor, Norway's riksvei ferries, the Philippines' RoRo routes,
  Indonesia's ASDP crossings) an island keeps a real car the model takes away, and a player's
  boat there fills with drivers who in fact cross by car; OSM ways a car can use but the public
  cannot reach still join (a private bridge, a service track over a dam); the gravity sends trips
  across water as if it were land, so a busy strait's island draws too many (Vashon about 3 times
  its ferry's riders, notes/T-2207.md 3). Check: per country, the people on parts other than the
  first (`checks/t2213_islands.py`) against its known car-free islands, and an island's trips
  across against its ferry's riders.
- **Cost damping** (T-495, docs/demand-mode.md 4.3). The logit's 0.10 a perceived minute falls as
  sqrt(60 / T) past T = 60, T the car's perceived minutes (even where no car can go), for walking's
  share and the regional market (rail against the car is the ratio curve's). The form and power follow UK WebTAG's cost damping and long-distance mode choice studies;
  set on European and US rail-air shares, not fitted. Why it may not carry over: the car's minutes
  are the US car's (speed, fuel, occupancy), so where long drives are slower (India, Lagos) a
  trip's scale falls further than its distance warrants; tastes for long trips (luggage, families
  driving) differ by country. Check: rail, air and car shares on a few corridors per country
  (MLIT's Inter-Regional Travel Survey, Eurostat, NEC FUTURE) at two lengths, so the slope as well
  as the level is seen.
- **Rail propensity** (T-636, docs/demand-mode.md 4.3). None: `asc_rail` is 0 everywhere, so rail
  and the car are judged on time and money alone, by the ratio curve; Anita chose a sensible and
  clear model over fitting the real cities (DECISIONS-to-2026-10-19 2026-10-17). A rail constant for the
  everyday layer exists as a parameter at 0 (`everyday_asc_rail`), and the country table
  (`api::COUNTRY_ASC_RAIL`) is empty. The real cities' station entries are the loose check
  (notes/T-636.md). Why it may not carry over: with no constant, habit, income, bus competition and what the car is
  good for show up only where time and money show them; countries where rail is the norm beyond
  its minutes (Japan, Switzerland) will look low, and where bus riders crowd the train (Seoul,
  London) high. Check: a new country's station entries against the model's level, and its census
  rail share by area where it has one; a country constant only where a real network's counts show
  a gap the worldwide value cannot close.

### Driving and parking

- **Driving alternative and parking at home** (T-854, docs/demand-mode.md 4.3).
  Everyone can drive; drive times and costs are the US car's, and the car stands for every way of
  getting about but rail and walking. Parking at the home end, 0.4 of the destination's paid-space
  rule on the home zone's residents per km² of land, needs only the pack's `pop`, so it is one rule
  in every country, with no car-ownership data anywhere. Why it may not carry over: density
  cannot see income, so a dense poor district where few own cars (Lagos, Mumbai) drives as much as a
  dense rich one, and a city of minibuses, motorbike taxis or bicycles is priced as if those riders
  drove; car ownership's real spread between countries (Paris's 65% of households without a car
  against Houston's few) shows only through density. Check: a city's rail share by area against its
  census mode share where it has one (T-509), dense cores against suburbs.
- **Routed drives** (T-275, docs/demand-mode.md 4.3). A local drive's road is the straight line
  times a routed factor over OSM roads per pair of 2 km zones (within 12 km) or of supercells beyond
  (every public road but service ways, shortest distance), times one scale, 1.3 / 1.210, that puts
  New York's commute-weighted factor back at the 1.3 its density speeds were fitted with; the nine
  US packs come to 1.17-1.26 before the scale. A drive between places (the regional and market
  layers, T-279, T-2424, docs/regional.md 5.1) is routed once per pair of places on a world 1/60
  degree lattice from GRIP4 road density, round water the lattice has no road over, times each
  zone's own detour towards the other place, so a coastal line round a lake (Toronto to Buffalo)
  competes with a car that goes round too, zone by zone; a car crosses water only on a fixed link
  or a car ferry OSM maps, at a ferry's high base and per-km price. Why it
  may not carry over: the scale ties every city to New York's speed fit; shortest-distance routes
  undercount more where drivers keep to a sparse motorway grid (Los Angeles, Houston) and less in
  old street grids (European centres); OSM's classification differs by country (rural roads
  tagged `track` are not in the drive graph), and residential
  coverage is thin in parts of Africa and Asia, where the nearest road joined can be kilometres
  off. Ferries that carry cars (Hong Kong, Istanbul, Seattle, the Baltic) are not roads locally, so
  a city across a strait has only its bridges and tunnels; between places a car ferry counts only
  where OSM tags it for cars, and a ferry so tagged where few cars really sail (or one left untagged)
  shifts the pair. Check: per new country, the
  commute-weighted factor the pipeline reports and the median snapping distance (US packs:
  94-213 m); a city's ACS-like car commute times against the model's, as T-090 did; the market
  drives' log of car ferry lines and pairs with a ferry.
- **Long drives** (T-370, T-938, docs/demand-mode.md 4.3). Each end's density speeds (16.5-88 km/h)
  last its own first 10 km of road (2 local, 8 main); anything beyond is open highway at 110 km/h.
  Why it may not carry over: 110 km/h assumes a motorway between the ends (US, Canada, western Europe); where long
  roads are two-lane or congested end to end (much of India, Indonesia, Lagos) the middle is far
  slower, and a sprawling metro whose dense core is wider than 10 km (Mexico City, Jakarta) is
  slow for longer. The speeds are as fast as or faster than measured ones where checked (T-1147):
  NHTS 2017's door-to-door car times in eight US metros at every trip length, and the published
  average speeds of London (TomTom, the DfT), New York (TomTom) and Tokyo (the road traffic
  census). Check: a few typical door-to-door drives of 50-130 km per new country against the
  model's (notes/T-295.md has the US, Toronto and London ones), and the city's published average
  driving speed against the ceiling harness's `drive` output (notes/T-1147.md).
- **Parking** (T-380, T-372, T-440, docs/demand-mode.md 4.3). One rule for the commute, everyday and
  regional layers. In minutes, not money: 7 to park (T-636, T-637; with the car ownership of $2.10 a trip
  and $0.1365 a km, T-938 and T-1755, local layers only), plus for a paid
  space 15 x ln(d / 3,000) per one-way trip (capped at 50), d the destination zone's jobs per km²
  of land; a share 1 / (1 + d / 8,000) parks free; congestion charges from a short list of
  charging areas (`kernel::CORDONS`: Manhattan, central London). Checked against real prices in New
  York, London and Toronto (within about half of each) and US and Toronto free-parking shares.
  By income (T-2278, Anita 2026-11-03: "yes lets do parking cost scale with income. less than
  linear"): in a country under $35,000 of GDP per head at market rates the paid space and the
  parking at home are times (GDP / 35,000)^0.5 (Delhi 0.28, Ho Chi Minh City 0.38, Mexico City 0.63,
  Beijing 0.63), 1 above. Colliers' 2009 CBD prices rise about in proportion to GDP below $40,000 and
  not at all above it (notes/T-2278.md), and a dollar weighs GDP^-0.5 minutes as fares do (T-2670),
  which leaves 1 - 0.5 (T-2677; `kernel::PARK_INCOME_E`, held to the fares' power by a test).
  The local layers and the cross-zone outings use it (`Params::park_income`); the market's longer
  trips keep their own parking.
  Why it may not carry over: prices relative to wages differ (Tokyo, Hong Kong and Singapore
  dearer; much of the US Sun Belt, Africa and South Asia cheaper, with informal kerb parking),
  and free parking at work is commoner in North America than in Europe or East Asia; the 15 and
  the 8,000 are the first country knobs. Other cordons (Stockholm, Gothenburg, Singapore's ERP,
  Milan's Area C, Durham) join the list when a pack covers them. Check: a day's parking price in
  a city's main and secondary centres against the rule, and a free-parking-at-work survey where
  one exists.

### Walking and riding to a station

- **Routed walks** (T-275, T-1164). A walk to a station goes over water on a 50 m raster from the
  pack's water mask, each metre of water counting 3 metres of walking (water too wide for that
  within the bound is no walk), with every OSM way a person may walk on that is a bridge opened
  at the plain cost. The 3 is one number for every place, standing for the bus, ferry and bike
  that cross water in real cities. Why it may not
  carry over: the water mask is TIGER's in the US and OSM's elsewhere (T-041; within 0.6% of TIGER at New York); OSM's `foot=` tagging
  on bridges is good in Europe and patchy elsewhere, and trunk bridges are assumed to have no
  footway unless `foot=yes`, which drops real footways in countries that tag them on the road
  itself; walking ferries (Istanbul, Hong Kong's Star Ferry, Venice) are no walk; a bridge
  passing over an island joins the island (Roosevelt Island over the Queensboro Bridge).
  Barriers other than water (motorways, rail yards, gated estates, rivers in deep cuttings) are
  not barriers. Check: the probes in notes/T-275.md for a few crossings per city
  (`pack_walk --probe`).
- **Walking and riding to a station** (T-371, T-389, docs/demand-mode.md 4.3). The walk is 6 km/h on a
  path 1.3x the straight line at weight 1.5; a commute or everyday trip of 5 km or more may
  instead ride from home (car, bus, bike
  or a lift, one mode) to the cheapest station of each line within 15 km, at `p_ride` 15 + 2.66 x
  the density-model drive (twice the whole-way drive's weight, T-916) + the cheaper of parking by the station zone's job density and a
  drop-off of 10. Nothing in it is per country: drive speeds come from density, parking from job
  density, and nobody lacks the ride for want of a car. Why it may not carry over: `p_ride` stands
  for getting the car out and walking from the car park, a North American picture; where the ride
  is mostly a bus, a share-auto or a motorbike taxi (Mumbai, Jakarta, Lagos) its fixed cost
  includes a wait and should be larger, and where cycling is the usual way to the station (the
  Netherlands, Japan) the ride is cheaper and shorter than here. The work or activity end walks
  only; errands of 5 km or more ride like commutes, though a lift or taxi to the shops may be
  rarer than one to work. The regional layer (T-373, docs/regional.md 5.3) uses the same rides:
  commutes between zones at the home end, inter-city trips at both ends with only a drop-off
  (`d_drop`) at the far end, which stands for a taxi or a pick-up; where taxis are cheap and common
  (much of Asia and Latin America) the far end is easier than this, and where long trips end at a
  rental car or a long walk it is harder. The
  weights are fitted on New York only (T-372: 53-66% of commuter rail riders ride there). Check: the share of rail riders
  reaching the station by car or bus per sampler city (UK NTS for London, Metrolinx surveys for
  Toronto) against the model's home-end ride share (rail trips at the ride access entries, `Subzones::ride`).
- **Where a selection's train riders are drawn** (T-1920, docs/demand-kernel.md step 5, `flows`). A
  subzone's commuters by rail at a selection's far end go to its cells within the walk limit
  (`pair_walk_max`, 80 perceived minutes for both ends' walks) of the selection's nearest end, or
  to all its cells by jobs or homes when none is. Nothing in it is per country: it reads the same
  walks and rides as mode choice. Why it may not carry over: where stations are dense (Tokyo,
  Paris) nearly every cell is in reach and nothing moves; where stations are far apart a far
  subzone's riders gather on its cells nearer the station, more than in New York (notes/T-1920.md).
- **Walking the whole way** (T-558, T-636, docs/demand-mode.md 4.3 "Walking"). A mode in the choice
  beside rail and the car (or other), at 1.5 perceived minutes a minute at every length, with a
  bonus of 10 perceived minutes against the car (`walk_bonus_min`) for what a short drive costs beyond
  its road time, one value for commutes and everyday trips and for every country. A commute walks
  between six home and six job cells of its zones, not their 2 km centres (T-587), so its walk
  shares by distance follow the everyday layer's. Why it may not
  carry over: walking tolerance differs by climate, culture, street design and safety, and the
  bonus also stands for how much a short drive costs beyond its road time, which differs with car
  ownership and parking. Check: a country's travel survey walk shares by distance (NHTS,
  England's NTS, France's EMP, Japan's person-trip surveys) and its census walk-to-work share.

- **Trams** (T-2926). A tram line is a rail line: it boards with no penalty (a bus pays 6 perceived
  minutes), each tram section counts as three quarters of a rail car in seats, rated and crush loads,
  and a cell keeps at most 3 stops served only by tram lines among its 6 nearest stations, so a street of
  close stops does not crowd out the metro a short walk away. A stop a metro or rail line also serves is
  an ordinary station. All three are one number for every place. Why they may not carry over: where
  trams run in mixed traffic with long waits at signals or are unreliable, the missing boarding penalty
  makes them too attractive; where trams are the main mode (Melbourne, Zurich) a cap of 3 undercounts
  how many tram stops people choose among. Check: a city's tram boardings per route-km against its
  published ridership (Melbourne, Zurich, Toronto streetcars).

### Value of time, income and crowding

- **Value of time and fare sensitivity** (T-1277, T-1166, T-1521, T-2670). A fare dollar costs 4.4 x
  (GDP per head / $90,000)^-0.5 perceived minutes, by the country the trip starts in: the US 4.4, the
  UK 5.5, Japan 7.0, China and Mexico 11.2, India 25, Nigeria 38. GDP per head in current US dollars at market rates
  (World Bank NY.GDP.PCAP.CD, `app/src/game/countryIncome.json` `usd`, Taiwan from the IMF; game
  dollars are US dollars). T-1521 had 0.7 (T-1277's 0.5 at PPP); T-2670 (Anita, 2026-11-06, notes/T-2668.md
  option N: "seems like a good compromise") has 0.5 at market rates for both the minutes a dollar and
  the fare scale. Evidence (notes/T-2668.md 2b): between countries the value of time rises with real
  income at about 0.5-1 (Wardman et al. 2016 1.0 at PPP, 0.72 on GDP per head at market rates; Shires
  and de Jong 2009 0.47-0.67); 0.5 at market rates is 0.32 on real income, the level of within-country
  studies of transit riders (0.1-0.4), the low end. Real short metro fares across 46 cities scale at
  about income^0.66 (0.56-0.75), so the default fares now sit a little under that line; Anita prefers
  the gentler scaling. Fares are
  the player's curve times each country's scale, by default its income over the start country's to
  the same power (T-1906; the US's 11.6 times its income over the US's in a save from before), so at the default scales a fare
  weighs the same everywhere and a poorer country brings in less per rider; a scale the player sets
  moves inter-city riders from that country too (T-1286, every country's scale over its income's).
  Train running costs follow income at the same 0.5 (T-2669). In the local layers the car's ownership money stays at 3
  minutes a dollar everywhere; parking is counted in minutes, times its income factor at 1 minus
  this power (Parking, above; T-2677). The regional market's longer trips split each country's people
  into three income groups (T-1431, below). Why it may not carry over: incomes differ within a
  country as much as between (a Mumbai suburb against Bandra), which the local layers ignore, and
  their car's money should scale too. Check: a country's transit fares against its income, and its
  fare elasticity where a survey publishes one. **Starting fares** (T-1521, T-1670, T-1906, T-2670): the
  starting curve is one straight line set once for the start country, $2.50 plus 20
  cents a km in the US ($6.50 at 20 km, $22.50 at 100; Anita: about $2.50 in the US, $1 in Japan),
  elsewhere times (its income over the US's)^0.5, rounded: Japan $1.55, Germany $2.05, Mexico and China
  $1.00, India $0.45, the same in every game wherever it starts. Simple on purpose (Anita); real short
  metro fares sit at 0.5-2x these (Tokyo $1.20, Mexico City $0.26, Mumbai $0.23, London $3.83; notes/T-2668.md). Check: a country's published fare at 50-150 km against it (notes/T-1521.md).
  **High-speed fares** (T-1626): the fare above the base is times the player's high-speed multiplier
  (default 1.5) by the share of the trip's km on high-speed lines, in route choice, mode choice and
  the market alike. 1.5 is about JR's Nozomi ticket over its ordinary one (notes/T-1616.md); Europe's
  and China's high-speed fares sit at a similar premium over conventional ones. Why it may not carry
  over: where high-speed rail is priced near conventional rail (Spain's Avlo and other low-cost
  services) the default overcharges. Check: a country's high-speed fare a km against its ordinary one.
  Ferry lines take the player's Ferry multiplier and bus lines the Bus multiplier (T-3028) the same way (default 1.00, T-2195; "Ferries as lines" below).
  **Transfer fare** (T-1801): the player's, $0 by default, paid on each change of line, times the
  country's scale and weighed at its `fare_w` like the curve; the regional market changes free. At
  $0 it is a free change everywhere; a player who sets one models a system that charges each
  boarding.
- **Income groups on longer trips** (T-1431, docs/regional.md 5.1). A country's travellers are three
  equal groups by income from the World Bank's income shares by quintile (SI.DST.FRST.20 to
  SI.DST.05TH.20, 170 countries with a survey from 1992 to 2025; the rest take their subregion's
  median). A group weighs every mode's money (car running cost and crossings, plane fare, rail fare)
  at the country's minutes a dollar times (its income over the mean)^-0.5 (T-1277's exponent between
  countries at PPP; within a country prices are the same) times 0.4 (T-1521, about US DOT's value of
  intercity over local time), and makes longer trips in proportion to (its income over the mean)^0.7, the
  market's trip rate by income between countries. No new fitted number; the plane's base was
  refitted with it ($30), then set to published airfares by T-1521 ($100). Not in the local layers: there it would cost about 2.4 times the solve
  time (notes/T-1431.md). Why it may not carry over: a quintile table is a national average (cities
  are richer than the countryside), survey years differ by decades, and the trip rate by income
  within a country is taken from the one between countries. Check: a country's long-distance
  travel survey by income band.
- **Waiting and crowding penalties.** A wait counts as plain minutes, half the headway, with no
  weight (Anita, T-585), where the literature (mostly US and European) weighs a platform wait
  1.5-2.5 times a minute aboard and a long wait at home before a known departure less. The
  crowding penalty's shape is from that literature too, but it starts only at 70% of a car's rated
  load of 160 (Anita, T-710, kept by T-2186; the literature starts it where the seats run out, 44
  a car), counts a minute aboard 3 at the rated load and runs on, the same curve, to about 10.8 at
  crush (240, 150% of rated); riders in Tokyo or Mumbai tolerate crowding very differently. A change costs a
  fixed 2 perceived minutes beyond its wait and walk (Anita, T-717), where the literature puts
  5-15 perceived minutes on one, wait and walk often included. Why it may not carry over: where trains are
  sparse and timetabled (rural lines; the market's inter-city trips have a planned wait, T-1430) riders plan for the departure and the half
  headway overstates the wait, while on unreliable or exposed platforms a minute feels longer.
  Check: station entries against the counts, where a wait weight shows up as too much or too
  little pull to frequent lines.
- **Route choice by relative difference** (Anita, T-710, docs/demand-mode.md "Access and
  egress"). Riders share station pairs by a logit whose scale is 20 over the trip's rough length
  (25 + 3 perceived minutes a straight km, held at 60), in four distance bands. One rule
  everywhere, set by judgement, not fitted. Why it may not carry over: the trip's length comes
  from the straight line at an assumed rail speed, so a country with slow rail (or long access, as
  where riders ride to stations) has longer real trips than the rule assumes and gets a choice
  slightly too sharp; where many near-equal parallel routes exist (New York, Tokyo) riders stay
  spread, as they should. Check: the trip explorer's share on the clearly faster route of a few
  trips with unequal options, and station entries against the counts.
- **Crowding rounds** (T-434). At most three, and a period stops once a round moves its loads
  under 1%. Measured on New York and Tokyo-sized grids, where the peaks still move a few percent at
  round three; a far more crowded network (Mumbai) may need a fourth peak round.
- **Riders left behind** (T-188, docs/demand-kernel.md step 7). A boarding's cost includes the
  average extra wait for full trains, worked out per period with the boarding account's rule:
  arrivals peaking mid-period at the US busiest-hour factors, trains taking riders up to crush
  (240 a 20 m car, 150% of its rated 160, T-2186; a bus a third of a car, 80, T-2919), the extra wait in plain minutes like any wait (T-585) and
  capped at 60 minutes. Riders left behind wait at their station: nobody changes onto the line
  they left back towards the stop they came from, so nobody rides a full line the other way to
  board upstream (Anita, T-575; a line passing a station twice may be changed onto there towards
  other stops, T-1268), though
  Tokyo riders do (gyaku-nori), and riding out on another line and back stays open. Why it may
  not carry over: crush loads differ (Tokyo's worst lines run at twice the rated load, 320 a car,
  and Mumbai packs more, pushers included, so "full" comes later; London and Chinese metros reach
  only about 1.3 times theirs, notes/T-2147.md; many European systems run emptier); where riders are
  routinely left behind (Mumbai suburban, Lagos, Chinese metros with
  queue control at the station entrance) they plan for it, and the perceived weight of that wait
  may be lower, while elsewhere one missed train feels much worse than its minutes; the peak
  shape within a period is US commute timing. Check: published crush capacities per car for the
  country's stock, and left-behind or platform-queue surveys where they exist (London's and Hong
  Kong's operators publish some).
- **The trains' room, as a queue** (Anita, T-1070, T-1255, docs/demand-kernel.md step 7). After
  the rounds riders keep their trip only if, arriving at the platform with everyone who would ride
  if the trains had room, they would wait no longer than 30 minutes for a train with room; the rest
  go by car or on foot. Room is the same crush everywhere, 240 a car (150% of the rated 160, as
  Japan and Korea publish peak loads and New York's cars reach; T-2186, notes/T-2147.md), the
  arrivals the US busiest-hour shapes, and the 30 minutes one figure everywhere. Why it may not
  carry over: where a line runs past 150% for real (Tokyo's worst lines at 200%, Mumbai's peaks,
  with pushers) the cap holds rail lower than it is; where
  riders shift their trips to the peak's shoulders rather than queue, the shape overstates the
  queue; and patience on a platform differs by place (Indian and Chinese metros queue long at
  security and gates already). Check: peak-hour line loads against the operator's stated
  capacity, platform-queue or left-behind surveys (London's and Hong Kong's operators publish
  some), and whether the busiest lines' daily riders come out near their published counts.

## Other purposes

### Everyday trips

- **Everyday trip rates and distance scales** (T-509 stage 4, `pipeline/everyday_rates.py`, the
  pack's `everyday` block; notes/T-509.md section 12). Each country's weekday outings a resident
  from its national travel survey: legs by purpose, the leg home under the activity it leaves,
  halved (surveys with a "return home" purpose spread those trips over the purposes first),
  errands = shopping, personal business and escorting, social = leisure, visits, meals out and
  exercise, business left out. US (NHTS 2017) 0.577 / 0.424 / 0.044 (errands / social / medical),
  England (NTS) 0.554 / 0.291, France (EMP 2019) 0.497 / 0.375, Germany (MiD 2017) 0.518 / 0.365,
  Japan (Tokyo PT 2018) 0.273 / 0.122 / 0.037, Korea (KTDB 2016) 0.189 / 0.110, Brazil (Sao
  Paulo OD 2017) 0.106 / 0.057 / 0.044; elsewhere (Canada, China, India, Nigeria) the median,
  0.497 / 0.291 / 0.044. The distance scales are one worldwide set (T-637, `everyday::DECAY`;
  a pack's block carries none; the values in demand-sources.md 4.1), set for the share of trips
  under 2 km, not the mean, since surveys agree on that share (about half of everyday trips in New
  York and London alike; notes/T-637.md). Why the rates may not carry over: surveys
  differ in what they count (short walks, trip chains, ages 5 or 6 and up, a weekday or the whole
  week), medical is split out only in the US, Brazil and Japan (elsewhere 0.044 is taken out of
  errands), and poorer countries' surveys (Sao Paulo 2.0 trips a resident) run far below rich
  ones', so the median is a rich-country default for Lagos or Mumbai. Why the scales may not: a
  city where errands are a long drive (US exurbs, car-built new towns) has fewer short trips than
  New York or London. Check: a new country's national survey by purpose and its share of
  everyday trips under 2 km, and its rail share by purpose against the game's everyday rail
  share.
- **Central destinations** (T-1092, notes/T-1092.md): a fifth of errands and social outings are
  drawn to zones by attraction times its attraction per km² of land, at a 6 km scale, the rest by
  the local scales above. One share and one scale everywhere, from the attraction arrays every
  pack carries, so it needs no data of its own. Why it holds up: high-order shops and venues
  cluster in the densest places in every city measured (OSM in Paris and London; NHTS trips up
  the density gradient run 7-9 km against 1-2). Why it may not carry over: the density weight
  squares the attraction, so an error in the attraction proxy (tier 0's blend of jobs and OSM
  places, an estimated jobs lump) is amplified in where long trips end; and a polycentric city
  with weak centres gets its tail spread over many small ones. Check: a country's survey share of
  everyday trips over 5 km, and which zones take the most central trips (notes/T-1092.md) for a lump that is not a real centre.
- **Everyday samples** (T-397, T-1874; demand-sources.md 4.1). A cell's everyday trips go to a
  few sampled cells: one per 8 trips, 2 to 16, and a school's or university's homes one per 30,
  2 to 100, each sample's share varied by a seeded factor (uniform 0.5-1.5) and the cell's total
  kept. A method with no data of its own, so it carries over as it is. Why it may not: a site far
  bigger than New York's largest campuses still has 100 homes, each standing for a wider area.
- **Social destinations** are population, leisure jobs and parks (46 / 46 / 8%, T-196). ATUS 2023
  puts only 23% of US leisure outings at someone else's home against the model's 46% population
  draw (the population draw also stands for neighbourhood leisure LODES misses), worth a look
  with any non-US survey.
- **Everyday destinations** (T-453, notes/T-453.md and T-456.md). Errands are drawn by jobs in
  retail and personal services, leisure by jobs in food, accommodation, arts and recreation (LODES
  CNS07 + CNS19 and CNS17 + CNS18), as relative weights scaled to 15% and 8% of the pack's jobs.
  - *Where a country publishes jobs by sector by small area* (the config's `everyday` hook): a
    cell's jobs x its unit's sector share. Japan: the economic census 2021 500 m mesh's persons
    employed in I, M and N, I split to retail (56-61) and N to 78-79 and 80 by the municipality's
    table 9-1B, head-office units left out. England: BRES by LSOA, SIC 45, 47, 95, 96 and 55, 56,
    90-93 (94 left out: professional bodies' offices in central London). Canada: the census's
    usual place of work by census tract of work, NAICS 44-45, 81 and 71, 72.
  - *A source is used only if plausible* (T-456 section 7): its total 8-20% of the pack's jobs for
    errands, 4-15% for leisure; and under 5% of it in 2 km zones holding over 3 times their share
    of both residents and the purpose's OSM places (+1 a zone), skipped where OSM is under 30%
    complete. The test runs on 2 km zones (at LSOAs it failed BRES on noise). A country's source is judged on its
    best-mapped metro (Japan's on Tokyo) and its sparser packs run only the level test, since
    their OSM misses roadside malls and resort hotels. Toronto's retail and other services fail
    it (suburban business parks with retail jobs), so Toronto's errands use the fallback.
  - *Elsewhere* (Paris, Berlin, Shanghai, Mumbai, Seoul, Sao Paulo, Lagos): per purpose, c =
    min(1, OSM places per 1,000 residents / R), R = 5 errands, 3 leisure; each 2 km zone's base =
    0.75 x its share of jobs + 0.25 x its share of residents for errands (jobs alone for leisure);
    weight = base^(1 - c/2) x (OSM places + 1)^(c/2), normalised; inside a zone by OSM places,
    else jobs. Thin OSM (Lagos c = 0.02) leaves the base; complete OSM gives the geometric mean.
  - Why it may not carry over: small-area counts are rounded or suppressed; a sector table can mix
    wholesale or head offices into retail (the check catches the big cases); the fallback's base
    is the jobs proxy where a country has no workplace counts, and its OSM half follows mapping
    effort. Check: a household travel survey's shopping destinations by zone (Sao Paulo's OD 2017).
- **Parks** (T-196). OSM parks take 8% of the social/leisure destination draw, from ATUS 2023
  (7.6% of US leisure outings end outdoors away from home); each place draws sqrt(area) x
  D / (D + 2,000), D the residents per km² within 2 km of its edge; nature reserves count fully only when
  named as parks, else at a quarter. Why it may not carry over: the outdoor share depends on
  climate and culture (Nordic and Australian cities likely more, hot or very dense ones less, and
  ATUS is a year's average with no season); the reserve rule reads English names ("Park",
  "Recreation Area"), so other languages need their own words or the quarter for every reserve;
  OSM tagging differs by country (UK commons and recreation grounds are counted, but whether a
  country maps its urban parks as `leisure=park` at all needs a look); the half-point of 2,000
  residents per km² is absolute (about a US inner suburb), so in dense cities (Mumbai, Dhaka,
  Seoul) almost every park saturates and only size separates them, which is probably right; in
  US exurbs a big state park is discounted to a tenth. The size exponent 0.5 and the 2 km reach are
  judgement, not fitted. Check: a national time-use survey with a location code (the UK and most
  of the EU have HETUS-style diaries) for the share; park visit counts (mobile-phone or ticketed
  parks) in one city per country for the size and population terms.

### School, airport, medical and attractions

- **School and university.** NCES and IPEDS are US-only. Abroad: national school registers where
  they exist, OSM school and university points with an enrolment estimate otherwise.
- **Airport** (T-142): FAA enplanements are US-only; OurAirports is global but needs a passenger
  figure.
- **Medical** (T-142, T-387; `enrich_medical.py`, numbers in `pipeline/hospital_beds.json`). OSM
  hospitals, clinics, doctors and dentists everywhere; a hospital draws 1,000 visits a year a bed.
  Beds are OSM's `beds` where tagged; else 0.588 x the square root of the site's floor area in m²
  (its buildings' footprint x levels, 6.34 levels where a building has neither levels nor height,
  or 1.139 m² of floor per m² of site where the site's area gives more), between 20 and 1,500; else,
  for a hospital mapped as a point, 45. Those numbers pool the packs where OSM tags beds well
  (Baltimore, Boston, Philadelphia, Springfield, Washington), New York held out to check them.
  Every pack's hospitals are then scaled together to the World Bank's hospital beds per 1,000
  people (SH.MED.BEDS.ZS, the country's latest year; the world figure where the country has none,
  as for Taiwan) times the pack's population, and no site keeps more than 5% of the pack's beds or
  1,500, whichever is more, and never over 4,000 (the excess goes to the others in proportion).
  The scale sets hospitals against clinics and doctors; the shape only shares beds among
  hospitals. A pack with no OSM facility at all (rural Chinese counties, T-897) spreads its visits
  by jobs, so the purpose stays. Why it may not carry over: floor area is a weak guide, little better than giving
  every hospital the same size (notes/T-387.md); point hospitals are small private ones in Mumbai and Lagos but large ones in New York;
  some countries' latest World Bank year is old (Nigeria 2004, 0.5); and a national rate
  undercounts a city that holds its region's referral hospitals. Check: a national register of
  beds by hospital where one is open (Brazil's CNES, T-506).
- **Attractions** (T-263, `pipeline/attractions_json.py`, docs/demand-sources.md 4.1). One Wikidata
  list for the world: places of attraction classes with coordinates, sized by Wikidata's visitors a
  year (P1174) where given, else by a budget shared by Wikipedia page views (English plus the
  country's own editions, T-1766): 6 outings a year for each person within 25 km (GHSL), shared
  among the places within 25 km by views^1.3 with 0.03 views a resident for places not listed, a
  place's share s counting as s / (1 + s / 0.05), so the lone listed place of a town does not take
  most of its people's outings (T-1905), plus
  one world pool of tourists (World Bank international arrivals plus one domestic trip a resident a
  year, 0.6 listed places a trip) shared by views^0.5. Global constants, except where a country's
  people hardly read Wikipedia: estimates in mainland China times 5 (blocked there) and South Korea
  times 2.5 (Namuwiki), observation towers excepted (T-1935; Anita: per-country attraction factors
  are fine). Kept
  from 100,000; visits a day drawn half from homes and half from leisure jobs with the chance halving
  every 10 km. An attraction's staff draw no social outings: within 1 km of it, a cell's leisure jobs
  above 2,000, up to one per 1,000 visitors a year, leave social's weight (T-1932; a theme park's own
  jobs otherwise drew a city's social outings as if they were restaurants). A lake, hill or island that Wikidata also tags a tourist attraction counts (West Lake;
  in practice China's AAAAA class); a presidential palace or headquarters counts only with a figure.
  Waterfalls, caves, hot springs, opera houses, Ferris wheels and casinos (also hotels) count too
  (T-2666), and an attraction takes the nearest cell within 2 km. Of 13,304 places, 1,295 carry a
  figure (the US 275, Italy 193, France 156, China 21, Japan 25, Nigeria none); against them the
  budget is within 3x for 75%, within 2x for 54%. Why it may not
  carry over: the country factors rest on few figures (China 18, Korea 7), and another country
  where Wikipedia is little read would need its own; views follow fame abroad as well as visits, so a
  famous place with no figure runs high (the Oriental Pearl Tower 5.5M against about 3.4M, Delhi's
  Red Fort 7.4M); the staff rule's 2,000 and 1,000 are judgement, set on Disneyland and the Met; domestic
  tourism is one trip a person everywhere, rich or poor; a figure can be a whole institution's or a
  slip (the build estimates any above 25M); a country whose places have few editions (Nigeria,
  Brazil) gets few attractions. Hotels are not in the packs, so leisure jobs stand for where
  visitors stay. Check: a national tourism board's visitor table for a country's top places,
  against the file's figures.

### Timing, weekends and superholidays

- **Trip timing.** Period shares and departure profiles come from US data (ACS B08302, NHTS).
  Work hours, siesta patterns, school hours and late evenings differ by country.
- **Weekends** (T-260, T-262, docs/demand-timing.md 4.2). A Saturday's and a Sunday's trips per
  weekday trip by purpose come from NHTS 2017 (Saturday, then Sunday: work 0.37 and 0.24, shopping
  and errands 1.39 and 1.05, social and leisure 1.54 and 1.67, school 0.11 and 0.06, university
  0.30 and 0.14, medical 0.30 and 0.18, trips of 50 miles or more 1.69 and 1.48) and TSA
  checkpoint counts (airport 0.92 and 1.11); each day's timing is the weekday's shifted by NHTS's
  day/weekday ratio per period. A holiday runs as a Sunday. England's NTS 0504b gives nearly the
  same for the weekend as a whole, so the shape holds at least for rich car-owning countries. Why
  it may not carry over: the weekend is not Saturday and Sunday everywhere, so each country's legal
  weekend days decide which days run on these rates, its day of rest on Sunday's (T-262,
  docs/world.md "Calendar"), but six-day working
  weeks in parts of Asia and Africa are not in a legal calendar; where Saturday work and
  school are common (Japan's and Korea's Saturday schools until recently, retail and informal work
  everywhere) weekend work and school run higher; where shops close on Sundays (Germany, Austria,
  parts of Switzerland) Sunday shopping collapses further than the US Sunday's 1.05 while
  Saturday's rises; and where public holidays cluster (Golden Week, Chuseok, Chinese New
  Year) the long-distance factor is a holiday matter, not a weekend one (T-264, below). The calendar's
  game day 1 being a Monday is a game choice, not a rule. The weekend also keeps the weekday's
  crowded route choices, which matters more where weekday crowding is extreme (Tokyo, Mumbai), and
  runs the weekday's trains (T-2335): where real weekend service is much thinner than a weekday's,
  its crowding reads lighter than real. Check: each new country's national travel survey by day of week and purpose (most publish it:
  MiD Germany, ENTD France, the Japanese person trip surveys, the Korean household travel survey),
  and its legal weekend days.
- **Superholidays' days away** (T-264, docs/world.md "Calendar"): a Sunday with local outings
  (errands, leisure, school, university, medical) at 0.7 and attractions at 1.5; work, airport and
  longer trips a Sunday's. Found from any country's calendar by a name list, so every country the
  `holidays` package covers has them. One strength everywhere, a game choice (Anita, 2026-10-26),
  not a measurement. Why it may not carry over: how many leave town differs (China's and
  Vietnam's migrant cities empty far more than New York at Thanksgiving), and where a holiday's
  name is not in the list (a national day off that people travel on) nothing happens. Check: a
  country's rail ridership over its long holiday, and whether its big holiday is in the list.
- **Superholidays' going-out and coming-back days** (T-1373, docs/demand-market.md "A superholiday's
  lean"): a weekday and a Sunday with the longer trips twice a weekday's, leaning towards the
  smaller place going out and back coming back, up to 70/30, a place's size its people times the
  square root of its people per 6 km zone. Population and density alone, from any country's
  zones; no birthplace table (Anita, 2026-10-26: home trips by population with a boost towards
  rural areas, small towns and smaller cities). Anchors for the x 2, from memory: China's rail over
  the 40 days of Spring Festival travel about 1.2 times normal with peaks near 2, JR's Obon and
  New Year peaks 1.3-1.6 on the Shinkansen, US air on the Sunday after Thanksgiving about 1.25.
  Why it may not carry over: where people go home is a country's migration history, not its
  sizes (Shenzhen's go to Hunan and Guangxi, not its neighbours; a country whose cities grew from
  their own countryside has its home trips short), and one strength everywhere overstates a
  Thanksgiving next to a Spring Festival. Check: a census table of birthplace against residence
  by province or state, where a country's flows look wrong in play.
- **Superholiday trips to touristy places** (T-1374, docs/demand-market.md "A superholiday's
  lean"): going out, a zone pair's longer trips lean further towards the zone with more attraction
  visits a year per resident (none below 3, fully from 30; up to 1.5 times on top of the places'
  lean, 90/10 at most), coming back the other way. From attractions.json (Wikidata, every
  country), no tourism table. Why it may not carry over: attractions.json shares a budget by
  Wikipedia page views, so a place few read about reads as untouristy (mainland China, where
  Wikipedia is blocked, reads low); beaches, ski resorts and most lakes are not attractions there
  at all. Check: a country's top destinations by its tourism statistics against the zones that
  come out touristy.

## Network and stations

- **Station merge and walking transfer distances** (100 m merge, 300 m walks). Fine for US
  stations; very large hubs (Shinjuku, Châtelet) have transfers over much longer underground walks.
- **Complexes as transfer points** (T-161, with the player's links, T-197). Any two stations of a
  complex are a walking transfer however far apart, costed like every walk between stations:
  the distance between the platforms' centres (snapshot stations; within one demand station too,
  T-717) x 1.3 detour at 1.2 m/s, weighted 1.39, plus the boarding's usual wait and the change's
  fixed 2 perceived minutes (Anita, T-717: "mostly like the time it takes to walk between the
  centers of the stations"). Set by judgement to agree with the existing walks, not fitted. Why it
  may not carry over: in deep, crowded hubs (Shinjuku, Châtelet, Moscow's linked stations,
  London's Bank) a long interchange is slower than 1.2 m/s on the straight line x 1.3 (stairs,
  escalators, crowds, levels the plan view does not see), and riders weigh it more heavily. A
  change across a platform costs its wait and the fixed 2, as in a cross-platform design (Hong
  Kong, Singapore); riders elsewhere may dislike changing more than that (the literature's 5-15
  perceived minutes a change). Check: published
  interchange times for a few big hubs (London's TfL interchange walk times, Tokyo station transfer
  guides) against what the rule gives for the same platforms.
- **Passages** (T-1281). A change between two stations the player joined by a passage walks the
  straight line between the platforms' centres at 2 m/s (moving walkways, no detour), weighted
  1.39 like any transfer walk, with the boarding's wait and the fixed 2 perceived minutes. Set by
  judgement (Anita's yes, 2026-10-25), not fitted. Why it may not carry over: real long passages
  (Paris's Chatelet, Tokyo's Otemachi, Hong Kong's Central-Hong Kong walkway) have stairs, gates
  and crowds that slow them below 2 m/s, and riders dislike them more than their minutes say.
  Check: a few published passage times against the rule.
- **Changes inside a station building** (T-2220, Anita's yes, 2026-11-03: "changes inside it a
  little faster"; T-2276, 2026-11-03: "reduce to 1 minute instead of 2. also it should count as a
  waiting room"). At a station the player has drawn a building round, a change's fixed 2 perceived
  minutes count 1, and every wait there counts as in a waiting room (0.8); the walk between
  platforms and any transfer fare are unchanged (with the walk cut too, a building that left one
  line's platforms outside swung a parallel line 83% on New York). Flat judgement figures the
  player buys; they read nothing local. Why it may not carry over: some real hubs are slow to
  change in however big their roof (Shinjuku, Chatelet, Penn Station's levels), where signage,
  stairs and crowds, not the roof, decide the time. Check: published interchange times inside a few
  big single-roof stations (Berlin Hbf, Tokyo, Grand Central) against the rule's.
- **Station upgrades in demand** (T-1281, Anita's yes, 2026-10-25). A park and ride takes 10
  perceived minutes off a ride's fixed 15 to or from that station; a waiting room counts the
  half-headway wait there at 0.8 of its minutes, in route choice as in mode choice. Both are flat
  judgement figures, the player's to buy, and read nothing local. Why they may not carry over: the
  value of a car park depends on car ownership and parking elsewhere (in a city where few drive
  it does little, which the ride's other costs already show); the shelter discount stands for
  seats, heat and cover, worth more in harsh climates. Check: rail shares at a city's park and
  ride stations against nearby ones without.
- **No train slower than walking** (Anita, T-1059 in metres, T-2265 in real minutes, 2026-11-03:
  "comparing actual train time to actual walking time ... should only be to rule out ridiculous
  train trips that are literally longer time than wlaking"; docs/demand-kernel.md step 5). Within
  walking's 10 km reach, a pair of stations is no rail option for a trip when its real minutes
  (the walk or drive to the first station, the waits at half the headway, the rides, the walks
  between stations and platforms, the walk from the last) are more than walking the whole way's
  real minutes (over water at its cost where the pack routes walks). No fare, crowding, weights or
  fixed change minutes on either side, and a trip with any quicker pair keeps its whole share.
  Refused pairs carrying at most 1% of a trip's station-pair weight are let be (T-2307, for speed;
  the same everywhere). A rule of sense rather than a fitted number, and it holds anywhere: nobody takes a train that takes
  longer than walking there. What can differ by place: straight lines stand for both walks, so where
  streets are far from straight on one side only (a station across a motorway or rail yard, a gated
  campus, hill towns with long stairs) the rule can keep or refuse a pair a real rider would not;
  and where walking stands for cycling or a short bus (Amsterdam, Tokyo's suburbs) the real
  alternative is quicker than the walk, so the rule is if anything kind to the train. Everyday
  outings across a zone line have no such check (the regional layer has no walks to compare).
  Check: in a city's trip explorer, a few short trips beside a station that the rule refuses.
- **The service span** (T-842, docs/demand-kernel.md step 4). Riders' tolerance for the wait until
  the next train is log-uniform from 30 minutes to 6 hours (Anita's range, 2026-10-21; the shape
  and the median of 1.7 h are judgement, not fitted), and a stop where only a share `phi` of a
  period's riders have a train due within it costs -ln(`phi`) / 0.10 minutes more to board. It
  reads only the player's timetable, so it holds anywhere the other rules do. Why it may not carry
  over: the arrival shape it weighs riders by is the US one per period (the busiest-hour factors),
  and where people plan around a timetable (long-distance and rural lines, Swiss or Japanese
  regional clock-face service, a country whose last trains run past midnight) the tolerance is
  longer than a turn-up-and-go rider's; on long trips the cost ratio (4.3) already makes the
  extra minutes weigh less. Check: night and late-evening rail shares on a real network with
  service ending early (a commuter line without evening trains) against the model's.
- **Bus corridors** (T-1464, Anita's yes to the design, 2026-10-26; docs/demand-kernel.md step 2).
  A home within 300 m of a bus route walks its distance to the route plus 100 m and rides at the bus's run time (T-2904: a hop takes the
  car's time at 90% of its speed, never under 30 km/h for a bus lane, plus 28 s for the stop; about 19 km/h at
  400 m stops in a dense centre, 22 in open country), on the network's fare curve, bus kilometres included. Every boarding of a bus, a change
  from a train or another bus onto it included, costs 6 perceived minutes on top of the wait
  (T-1576; unreliability, crowding at the stop, a stop not worth the wait for a short hop). Judgement
  figures, not fitted; the 6 minutes and 300 m were chosen so a drawn route's riders per route-km
  land in the range of busy real routes (notes/T-1466.md). Why they may not carry over: bus speeds
  run from about 10 km/h in dense Asian and African centres to 25-30 in North American suburbs, and
  the model gives every dense centre a bus lane's speed (Seoul's median lanes, Anita 2026-11-09), so
  buses where traffic holds them up in reality are faster here than real ones;
  where buses are informal or fares flat and separate (minibus taxis, many Latin American cities),
  the one fare curve and the free change to rail overstate the feeder; a 300 m corridor is short
  where blocks are huge or long where people walk further to a bus (low car ownership). Check: a
  city's bus ridership per route-km against the model's on a similar drawn network.
- **Ferries as lines** (T-2195, design notes/T-2177.md 3.6; docs/demand-kernel.md step 4). A ferry
  line is a line like any other: its ports are stations walked to as any station, the minutes aboard
  are the boats' run (32 km/h in open water, 10 in narrow channels, 0.15 m/s² at each port), the
  wait half the headway, a change to or from it a change like any other, and no boarding cost (a
  ferry is boarded from a waiting room, as a train is). A boat holds two train cars (480 at crush).
  Its kilometres pay the one fare curve, the part above the base times the player's Ferry multiple
  (1.00 by default), and its trips are the network's. One rule everywhere, no constant. Why it may not
  carry over: real ferries are priced every which way (Staten Island's is free, NYC Ferry's and
  Sydney's apart from the metro, Istanbul's the metro's), so one curve and one multiple stand for all
  of them; a real boat holds 150 to 4,500 riders; demand's walks over water (three times a metre on
  land, standing for small ferries and bikes) stay, so a water crossing has its walk beside the
  player's ferry. Check: a drawn harbour ferry's weekday riders against the real one's at a similar
  headway (Staten Island about 70,000 a weekday before 2020 at a boat every 15-30 minutes).
- **Ferry fans** (T-2196, Anita 2026-11-03; docs/demand-mode.md "Ferry fans"). A tenth of visits
  and attractions trips (not errands, commutes, school, university, medical or airports) count each
  minute aboard a ferry at 0.1; everything else in their choice is anyone's. One share, one weight,
  no country constant. Why it holds: harbour ferries carry visitors and day-trippers beside their
  commuters wherever they run (the Staten Island ferry's tourists, Sydney's Manly ferry, Istanbul's
  Bosphorus ferries, Hong Kong's Star Ferry), and the ride is the outing as much as the way there.
  Why it may not carry over: where a ferry is a plain commuter crossing with nothing to see (a short
  river shuttle) the fans overstate its weekend riders; a sightseeing cruise that goes nowhere is no
  trip at all here; a long scenic crossing could draw fans past a direct train beside it, bounded by
  the tenth. Check: a harbour ferry's Sunday riders against its weekday's, where the operator
  publishes both.

## Intercity

- **The world market's long trips** (T-377, rule T-376, docs/regional.md 5.1). Every zone joins one
  market whose normalisers run over the world field: the NHTS ground curve (0.107 departures,
  onset 25-50 km, max(d, 25)^-2.37, fading out over 300-700 km) and a long-haul tail (0.00473, from
  300-700 km, d^-0.70, toward dense places), both scaled by GDP per head^0.7, a poorer destination damped by its income over
  the origin's, a border factor 0.26 x origin income^1.5, except 0.5 between two countries of the
  open-border area (the EU 27, Norway, Iceland, Switzerland, Liechtenstein and the UK; T-809).
  Fitted on NHTS 2017, US, UK, Japanese and Indian residents' trips abroad and flights pairs
  (notes/T-376.md); the area's 0.5 checked on London - Paris, Paris - Brussels and Amsterdam -
  Brussels (notes/T-809.md). Why it may not carry over: one factor for the whole area, island or
  land pair alike, so the land pairs come out low; every other border, open or not (Mercosur, ASEAN, the US-Canada line), keeps the one
  general rule; Japan's residents go abroad 6x too often; the ground curve comes from one dense, car-owning
  region; attraction is people, so a giant metro takes most of a smaller neighbour's 80-300 km trips
  (New York - Philadelphia about 2.5-6x plausible, notes/T-379.md, T-438). Check: each country's national travel survey by distance band (MiD Germany, ENTD France,
  NTS England, Japan's Inter-Regional Travel Survey) and its cross-border counts; another open
  area, or a factor per country pair, where they disagree (with country tailoring).
- **Attraction inside a pack** (T-377). `population + 2 x jobs` rescaled to the pack's people, so
  a pack counts as many people as the world field around it, its jobs only moving where in it trips
  end. Where Kontur and a pack's census differ (the Northeast's near pairs rose 2-7% on the switch
  from the zones' own normaliser) the pack's near pairs move by the gap.
- **A zone's stations** (T-1506, T-1511, docs/regional.md 5.2 and 5.3). Each zone end keeps its 8
  best-weighted stations, then up to 10 with a slot each for track none of those is on, and its
  rides take the 6 cheapest lines and then up to 2 more for track those miss, so an inter-city
  station not linked to a big city's metro is still in the lists of the zones near it and of those
  that ride to it. Why it may not carry over: a city with
  more than two unlinked rail networks near one zone (separate suburban, metro and high-speed
  operators with no shared stations) still loses some. Check: the trip explorer's served share on
  a country's main inter-city pairs, against the share of trips with a station in reach.
- **An inter-city trip's planned wait** (T-1430, docs/regional.md 5.1). The market's inter-city
  trips wait at their first boarding the least of half the headway, 5 minutes plus a sixth of the
  headway, and 30 minutes, with no tolerance penalty: a planned trip leaves on average a quarter of
  the headway from the time wanted, weighted two thirds of a platform wait, plus a few minutes of
  arriving early; past a train every 3 hours one leaves at another time. Changes, commutes and
  outings across zone lines keep the half headway. Set by reasoning on Anita's call ("a 2 hour wait
  is fine"), not fitted (notes/T-1411.md). Why it may not carry over: where long-distance trains
  need booking days ahead, or run once a day, a 30 minute cap undersells the cost of a sparse
  timetable; where trains are late often, arriving early costs more. Check: rail's share on
  corridors served every 2 hours or less often against published shares.
- **No car between landmasses** (T-377, T-375's fixed links). Far pairs on two landmasses fly
  (T-378); near ones (under 250 km) keep their trips fixed; a ferry is no link.
- **Air door to door** (T-378, docs/regional.md 5.1). Between far places (250 km and more) that both
  have airports: drives to the airports, 135 minutes there for both ends together (90 before departure and 45 after landing, Monmousseau et al. 2021; T-1564, Anita 2026-10-27), 25 minutes
  until the flight (its frequency unmodelled: rail's planned first wait at a flight every 2 hours,
  T-1430), 750 km/h plus 30 minutes, 60 more with no direct flight, and a fare of $100 + 5 cents a km (T-1521, published average airfares at market rates, one world fare) at each
  income group's minutes a dollar (T-1431, above; a corridor's split between car and plane is
  the mean of its two ways). The fare is one simple rule on purpose (Anita: predictable for the player).
  Its base was fitted on ten corridors' published rail shares of rail and air (notes/T-1431.md), so
  it sits as far under real airfares (notes/T-712.md: about $108 in Japan at 766 km, $200 in the US)
  as the starting rail curve sits under real long-distance rail fares; a player who raises long fares
  to real levels meets the plane.
  Airports from OurAirports (global, public domain), their size and direct routes from FR24 (two
  days in May 2026, offline). Why it may not carry over: fares (the US pays about twice Japan and
  Europe; low-cost India, subsidised routes), airport times (US security is slower than a Japanese
  domestic flight), and a single 1M-passenger floor. Check: each country's air and rail share on
  its big corridors (MLIT for Japan, Eurostat's air O-D counts, DB1B).
- **Cars carried over water** (T-712). Where a fixed link joins two landmasses but takes no cars at
  road prices, the car pays the crossing: the Channel (Le Shuttle or the Dover ferries) $150 a car
  and 60 minutes beyond the straight drive, between Great Britain and anywhere else; the Tsugaru
  Strait (the Seikan Tunnel is rail only; the Aomori-Hakodate ferry, 3 h 40) $170 and 240 minutes,
  between Hokkaido (Japan north of 41.4°) and anywhere else; the fare shared by two like the car's
  running cost. Toll bridges (Øresund, the Great Belt, Honshu-Shikoku) stay free like every toll
  road (T-270). Why it may not carry over: each crossing is named by hand; a new fixed link that
  carries no cars (a rail tunnel) needs its own entry.
- **The longer trip's car** (T-276, T-1521). Drive 15 min + 1.25 x the great circle at 85 km/h,
  plus the car's running cost a road km and the local parking rule at the destination, both split
  over two occupants, the money at each income group's minutes a dollar (T-1431). The running cost
  is the two places' countries' mean (`countryIncome.json` `car`, notes/T-1521/money_table.py):
  gasoline at the pump (GlobalPetrolPrices, octane 95, 2026-10-19; a country without one takes its
  subregion's median) at 8.5 litres a 100 km (IEA's world average for new cars, 7.1, plus the 20%
  real-world gap the European Commission measures), plus the motorway toll a km where tolls go by
  distance on most of the network (Japan, China, Korea, Taiwan, France, Italy, Portugal, Greece,
  Croatia, Serbia, North Macedonia, Turkey, India, Mexico, Brazil); vignettes and the few tolled
  corridors elsewhere (the US, Spain, Poland) count nothing. The US $0.10 a km, Japan $0.23, China
  $0.19, France $0.32, Germany $0.20. Upkeep is left out (AAA: about $0.07 a km): riders weigh what
  they pay on the trip. Why it may not carry over: a pump price is one day's, tolls are national
  rates on sample routes, and motorway speeds, occupancy and car ownership vary by country too. Within 1,500 km the drive is routed (T-279,
  docs/regional.md 5.1) on three world layers: GRIP4 road density (roads as of its 2018 release, so
  newer motorways are missing), GEDTM30 slope and the world water
  mask, fitted to 13,470 OSRM drives in 43 countries (6.8% off at the median, held out by region;
  the rule 10.0%). Free flow: dense corridors come out 10-13% fast against traffic. A ferry over open
  water, one simple price everywhere. Check: a fuel price and toll regime per country; a few routed
  drives between its biggest places against OSRM's.
- **Commutes between zones** (T-276, docs/zones.md 4). One gravity over every zone pack, each
  pack's own curve and border factor inside it, region pairs across packs from the pack holding
  both, else the country's admin-1 default (US 0.23, Japan 0.5), else the pack-line default 0.5
  (`pack_line_default`; Japan's census fit was 0.4, raised by Anita in T-1163), national borders
  0.05 (US-Canada, judgement), 0 where a border is closed to crossings (Ukraine with Russia and Belarus,
  T-2344). A line through one continuous city (10+ km of it with GHS-POP 1 km
  pixels of 1,500+ people on both sides in one urban centre) eases up to 1 by that centre's people
  on the smaller side, 50,000 to 200,000 (T-1192, `seamless`, `pipeline/seams.py`; 41 pairs, 38 in
  China). Why it may not carry over: GHS-POP's 1 km grid and the 1,500 threshold find Chinese
  cities cut by county lines but no suburb-to-suburb line in Europe or the US (Baltimore -
  Washington has none); a country whose cities sprawl at lower density gets no lift.
  Checked, not fitted, against LODES jobs between packs (notes/T-268.md). Why it may not carry over: where
  commuting crosses national borders freely (Schengen: Basel, Geneva, Luxembourg) 0.05 is far too
  strong (the long trips' open area, T-809, does not apply to commutes; T-810); a country whose pack lines do not cut commuting as Japan's do would want a lower or
  higher country value in `admin1_default`. Check: the country's census commuting flows between its zones.
  **Which packs share a gravity** (T-539): those whose outlines come within 50 km of each other,
  chained, formed by `intercity.py --group auto` (pairs more than 50 km apart carry well under 1%
  of the commuters crossing a pack line). Why it may not carry over: two cities 50-150 km apart
  with nothing between them exchange no one, and a pack line between packs in a country with no
  fitted factor takes the default 0.5 (T-584, T-1163). Check: the country's census commuting flows between the packs,
  when one is fitted.
- **Everyday outings across zone lines** (T-1163, T-1175, docs/zones.md 4a). The everyday draw
  (errands, social, school, university, medical) runs over the neighbouring packs' zones within
  80 km as well as the home's own, undamped inside a country and at the national factor across a
  border; each zone keeps its own pack's share and the rest travel in the regional layer. It uses
  only what every pack already has (its everyday attractions and rates), so it carries over as the
  everyday layer does. Why it may not: walks across a line are straight lines (no pack's water
  routes span two packs), so a line along a river (much of the Pearl River delta) walks more than it
  should; where a zone line is also a hard barrier (a closed border, a sea strait between packs) the
  draw still crosses it at the national factor or 1. Check: on a line through one built-up area,
  about 25-30% of the arrivals at places within 3 km of the line come from the other side
  (Guangzhou - Foshan, notes/T-1163.md).
- **Zones** (T-276, T-719, T-788, docs/zones.md). Non-city zones are built from a country's units
  (US county, Canadian census division, NUTS-3), merged within their region to at least about 0.5M
  people (Anita, 2026-10-20); the unit differs in size by country, and where admin-2 units are tiny
  (French communes, Japanese municipalities) a coarser statistical unit is needed.
- **Zone pairs in chunks** (T-780, docs/regional.md 5.1). The longer trips' 6 km zone pairs under
  about 1 trip a day are dealt in chunks of that size, each zone's trips with each place kept. It
  saves most where many small zones trade a little over long distances (all of Japan); a sparse
  country with few zones gains little and moves little. Why it may
  not carry over: a station fed by few small pairs (a rural stop on a long-distance line) shows the
  sampling most, by a quarter or so of its market riders on Japan. Check: `t756-reach.mjs compare
  <label> chunks` on the country's network. The chunk grows with the trip's distance and place pairs
  250 km or more apart are dealt over both places' zones (T-2641, regional.md 5.1): about 1.7-1.9 times
  on the market's pass over North America, 1.1 on all Japan; a far city's stations get their
  long-distance riders in lumps, a few small lines (1,000-3,000 riders a day) move 2-5%.
  `?perfOff=farChunks` turns it off.
