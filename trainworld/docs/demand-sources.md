[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: where trips come from (4.1)

Cells, jobs, homes and the seven purposes' inputs.

### 4.1 Where trips come from

A **doubly constrained gravity model** from homes to jobs with a distance-decay curve. Tiers say
how good the inputs are. The US cities are tier 1; every other place is built by tier 0
(`pipeline/tier0.py`, T-287), with a country's own inputs in place of its proxies where it has
them (SPEC.md's city list, [data-register.md](data-register.md)):

| Tier | Inputs | Used by |
|---|---|---|
| 0 | GHSL population grid scaled to district totals + a jobs proxy (GHSL non-residential volume weighted by height, OSM job places, population) + OSM, Wikidata and OurAirports for the other purposes | every place outside the US |
| 1 | population + real jobs counts by small area | the US cities (LODES) |
| 2 | a real home-to-work OD matrix | not an input; a **check** on the fit where it exists (LODES OD for the US) |

**Cells** are the unit of local demand: H3 resolution 9 (about 0.1 km², 170 m edge), only cells
with people or jobs. Resolution 8 (about 0.74 km², Kontur's) is too coarse for walk catchment:
half a cell is about 4 minutes of walking. Each census block is spread over the cells its polygon
covers, by land area (T-013). Grain matters most in city centres (Anita): res 9 where it is dense,
**res 8 where it is sparse** (T-044, `pipeline/merge_cells.py`, the last pipeline step): a res-8
parent whose cells make under 2,500 trip ends a weekday (commutes and the everyday rates of 4.1's
purposes, both ends) becomes one cell, merging only the cells inside one 2 km zone, so a parent cut
by a zone edge is one cell per zone and every zone's sums, the gravity and the routes are those of
the res-9 pack. A merged cell sums every weight, keeps the parent's H3 index (a parent cut by a zone
edge gives two cells with one index) and sits at its cells' trip-end-weighted centre. In a parent at
or over 2,500, the children under 1,250 trip ends each that touch each other inside one zone merge
into one cell the same way, keeping the res-9 index of their busiest child, while the busy children
stay at res 9 (T-466). A cell under 1,250 that neither rule merged joins, across its parent's
edge, the touching merged cell of its zone with the fewest trip ends; such cells that touch only
each other merge among themselves under the busiest one's index (T-510). Joining is one pass (no
cell joins a joiner), and no group a lone cell joins may reach more than 700 m from its centre; a
group past that gives up its farthest joiners. A group stays unmerged if its centre falls on water
or outside its zone square. Last, any cell (merged or not) still under 50 trip ends joins the
nearest cell of its zone at or over 50, which keeps its index and its place; a zone with none
becomes one cell at its busiest (T-850b). In a pack with walk parts (`walk_part`, a zone's cells split by bank,
T-700), each part of a zone counts as the zone in all of this: no merged cell spans two banks, and
every zone keeps all its parts, which the walk table's layout needs; a parent merged in two parts of
one zone gives each its busiest child's index (T-725). The pack carries each zone's res-9 cell count (`zone_cells9`), which the sim takes as the
zone's land. Enrichments and routes run on the res-9 pack, kept in `data/work/res9/`.

**Jobs data is real job counts at the place of work**, not a proxy, where it exists:

- US: LODES WAC (Workplace Area Characteristics), jobs per census block by sector and earnings.
  Covers wage and salary jobs under unemployment insurance plus federal civilian; leaves out the
  self-employed, military and informal work. Block figures carry deliberate noise, and some
  employers' jobs are placed by a model among their sites, so aggregate a little before trusting
  one block.
- Japan: Economic Census 2021 mesh, workers per 500 m square.
- England and Wales: Census 2021 workplace population by output area. **Caution: the census was in
  March 2021 lockdown**; many people working from home count at home. Usable for shape, biased
  against office districts.
- Australia: destination zones; Italy: 2011 census sections; EU: JRC ENACT 1 km (2011).
- Everyone else, China included: tier 0.

Full list with grain and access in `research/demand-data.md`. Population for the US: 2020
decennial census by block. **The home end of commutes is where job holders live**, not everyone:
in the US, LODES RAC (jobs by home block, the same jobs WAC counts at work), so a neighbourhood
of students, retirees or children sends fewer commuters than its population (T-054). Elsewhere,
population until a source by residence is found.

**People in institutions are not homes** (T-234, Anita: "why are there homes on rikers
island?"). The census counts people where they are held or cared for: prisons and jails,
juvenile facilities, nursing homes and other institutions (psychiatric hospitals, hospice). They
make no commutes, errands, outings or school trips, so the 2020 census's institutional group
quarters (redistricting file PL 94-171, table P5, per block, every state) are taken off each
block's population before anything is built on it: residents for everyday outings, child and
adult homes for school and university, the medical and airport origins, the market's sites. The
park draw's nearby-residents term still counts them until parks are next fetched from OSM
(T-235). All four institutional types go, nursing homes included:
their residents rarely leave. The non-institutional types stay: college dorms (students
travel, and feed university trips from their own block), military quarters and other
non-institutional places (shelters, group homes). LODES also puts job holders on such blocks,
so on a block with an institution the commute home end is capped at the residents left; elsewhere
LODES stands. Per city: [T-234](../notes/T-234.md). `pipeline/group_quarters.py` holds the source
and the rule; `build_city.py` applies it as it reads the blocks.

**People who work at their own home make no commute and are not counted** (Anita). LODES places
a remote worker at the employer's office, so commute trips are scaled down by the share working
from home (US: ACS "worked from home", by tract at the home end, by industry at the work end). Jobs whose work is at other people's homes
(home care aides) are real commutes and are spread over where people live (T-014).

**Every registered city has the same non-work purposes from the same national sources** (T-142):
everyday outings from LODES sectors, school attendance from NCES public schools, university
attendance from IPEDS, all built by `pipeline/enrich_all.py <city>` onto a finished pack (and by
`build_city.py` for a new one). No city has a city-specific source (Anita, 2026-10-12, T-181).
A pack without these arrays opens
commute-only. Per-city inputs and totals, and per-capita checks against New York:
[T-142](../notes/T-142.md).

**Everyday outings (T-066; rates T-509).** All residents generate home-based shopping/errands,
social/leisure and medical outings, each an outward and return leg, at their country's weekday
rates from its national travel survey, a worldwide median where it has none
(`pipeline/everyday_rates.py`, written into the pack's `everyday` header block; the sim's defaults
0.3, 0.2 and 0.04 only for a pack without one). The US's, from NHTS 2017: errands 0.577, social
0.424, medical 0.044 a resident a weekday. A survey's legs for a purpose, the leg home counted
under the activity it leaves, are halved into outings, so its legs are kept whatever its trip
chains. Errands are shopping, personal business and escorting; social is leisure, visits, meals
out and religious; business trips have no game purpose. Shopping attracts trips by retail and other-service employment (LODES CNS07+CNS19). Social
destinations mix three independently normalized draws: population (visits to friends and family)
46%, food/recreation jobs (CNS17+CNS18) 46%, and **parks 8%** (T-196; a pack without parks keeps
50/50). Employees are relative draw, never customer capacity. Outside the US (T-453) the same
sectors come from a country's jobs by sector by small area where published and plausible (Japan's
economic census mesh, England's BRES by LSOA, Canada's census tracts of work); elsewhere each 2 km
zone's jobs (errands: 75% jobs, 25% residents) blended with its OSM shops or food and leisure places
as far as the pack's OSM is complete ([demand-worldwide.md](demand-worldwide.md), "Everyday
destinations").

**Parks (T-196).** `pipeline/enrich_parks.py <city>` takes OpenStreetMap's parks, commons,
recreation grounds, beach resorts, botanical gardens and named beaches in the pack's boundary,
nature reserves named as parks ("Keney Park", "... Recreation Area") as parks, and other named
nature reserves at a quarter; playgrounds, dog parks, pitches, golf courses, cemeteries, private
and unnamed places are left out, and a place inside a larger kept one merges into it. Each place
draws factor x sqrt(area in ha) x D / (D + 2,000), D the residents per km² of land within 2 km
of its edge (the place itself left out): the square root is the diminishing return on size (a
park 100 times larger draws 10 times as much), and the population term discounts parks few people
live near without multiplying the ones in dense places, which the gravity already sends more trips
to (Anita: "by size and by population nearby"). It is a density, not a count, because a count
within 2 km grows with a park's edge (a huge state park outdrew Central Park). The draw is spread over the pack cells inside the place (the nearest cell within 500 m
for a small one) as the `park_draw` array. The 8% share is the American Time Use Survey 2023:
7.6% of leisure outings (eating out, socialising and leisure, sport and recreation) end
"outdoors away from home". Parks take their share evenly from the other two draws, so the 50/50
balance T-066 chose between them stays. Parks change where social outings go, not how many there
are; no season (ATUS is a year's average). Per city: [T-196](../notes/T-196.md).
The pipeline spreads original sector locations by the same block land-area method as population;
office/headquarters commute corrections do not relocate these customer weights.

Destination gravity is origin-constrained: zone attraction times exp(-distance/scale), normalized
per origin zone, with one worldwide set of scales (T-637, `everyday::DECAY`): errands 1.0 km,
social 1.0 km, school 1.4 km, medical 2.0 km, university 3.5 km, airport 20 km, set for the share
of short trips rather than the mean ([T-637](../notes/T-637.md)). Same-zone distance is 1.04 km.
**Errands and social outings mix two draws** (T-1092, `everyday::CENTRAL_SHARE`,
[notes/T-1092.md](../notes/T-1092.md)): 80% by that rule, and 20% to central destinations, each
zone weighed by its attraction times its attraction per km² of land (`ncell` x 0.1053 km², at most
the square) at a 6 km scale. One scale cannot give the surveys' mix of many short trips and a long
tail; the density weight sends the tail into dense centres, where high-order shops and venues
cluster, instead of spreading it evenly. Within the zone the cell is drawn by attraction, as for
the local part, and the two parts share the cell's stratified samples (the central part last).
Deterministic stratified destination samples per cell/purpose integrate that distribution: one
per 8 of the cell's trips a day for the purpose, rounded up, 2 to 16 (T-397; none for a cell
with no budget for that purpose),
then choose an actual destination cell by attraction weight. This is an approximation to
continuous demand, not sampled individual travellers. It is stable across networks and reloads.
**Everyday trips below 150 m cell-centre distance vanish entirely**, with no replacement or
denominator contribution, including walking. The cutoff is limited by cell grain; real within-cell
journeys are not separately represented. Existing commuting behavior stays unchanged.
Local weight tapers smoothly from 1 at 40 km to 0 at 80 km; trips at 80 km or more disappear
from this layer. Only within-pack everyday outings are generated here. The regional market's
longer trips start at 25 km (docs/zones.md 5), so the two overlap from 25 to 80 km; the local
decays leave under 0.3% of errand, social, school and medical outings past 20 km, so the double
count is negligible, and airport trips, which the market does not place at airports, keep their
reach (decision T-276).

After station access is rebuilt, eligible cell edges aggregate into purpose-specific subzone
pairs, retaining weighted length and walking shares. Outward and return mode/route choices run
independently on each period's directed graph with the same fares, access, road costs, loading and
frozen combined crowding as all other demand. MSA retains direction-specific rail probabilities
for cell, selection and period queries. City mode splits, line loads and fare income include these
trips. Local bubbles count **trip legs** (a commuter is two); their purpose/timing queries reconcile to those same endpoint totals. Commute
station catchments remain explicitly commuter counts. Source-neutral cell-weight names allow
other countries' destination inputs later; runtime enables the layer for any pack that carries
the arrays (`errand_jobs` and the rest). A purpose whose arrays hold no sites (no retail,
leisure, school, university, airport or medical site; a school or university pair with either
side zero) is an empty slot with no trips; with no leisure sites, social uses residents (and
parks) alone. Residents must be positive, and a pair with only one of its two arrays is an error.

**School attendance (T-140/T-150/T-181, metro public schools).** Every pack's schools are the
NCES EDGE/CCD 2023-24 public schools in its counties (`pipeline/enrich_schools.py <city>`),
charter schools included. Distinct NCES school IDs at shared coordinates keep their own enrollment,
so a building holding several schools counts each once. Operational physical schools with
positive known enrollment are included; virtual/unknown categories are omitted. Missing
enrollment/sites contribute no invented pupils.

Origins use 2020 block residents weighted by ACS 2023 tract ages 5–17 plus two fifths of
under-fives, approximating ages 3–4, spread by block land area across all the pack's counties.
Missing shares use full-state county FIPS fallbacks, or the pack's age share for Connecticut gaps.
Each destination cell generates
0.9 attendance outings per enrolled pupil on a typical school day. This attendance factor is
provisional. Deterministic home samples per school cell (one per 8 attendance trips, 2 to 16) use child weights and a 1.4 km
distance-decay scale (T-637). Unlike customer trips, school demand is destination-constrained by
enrollment. The same 150 m exclusion, local-distance taper and directional assignment apply.

School attendance joins bubbles, purpose/timing inspection, fares, loads and crowding as the
model purpose School, shown to the player together with University as "School and university"
(T-192). Private schools are not modelled (NCES's Private School Survey is the
national source for them, T-142).
Estimated homes are not observed student journeys or catchment/grade assignments.
There is no school calendar, escort demand, school-bus choice or student-fare policy yet.
Source selection and international follow-ups: [education data](../research/education-data.md).

**University attendance (T-152, metro IPEDS; every city T-142).** IPEDS 2023 directory and Fall
Enrollment records supply the active degree-granting reporting institutions. The layer uses degree/certificate-seeking undergraduates plus graduate students and excludes
exclusively distance-education students. Nondegree undergraduates are omitted. Structural
blank online counts are accepted as zero only when IPEDS marks them not applicable and reports the
whole cohort as non-distance; unknown counts remain excluded. One budget is attached to each
reporting UNITID's directory coordinate, avoiding child-campus and merged-institution duplicates.
The exception is an institution that absorbed others in the data year (IPEDS `ACT` "C" with
`NEWID` pointing at it): its total is split over its own and the absorbed campuses' points by
each one's own eligible enrollment the year before (IPEDS 2022): Connecticut State Community
College, which in 2023 reported all twelve former community colleges at Hartford. Otherwise this
is not campus-level allocation: a multi-campus institution with one UNITID (Northern Virginia
Community College, Montgomery College, Community College of Rhode Island) sits at one point.

Adult homes use block population multiplied by the ACS tract age-18+ share. A provisional 60%
typical-day attendance factor, 3.5 km distance-decay scale (T-637) and separate day/evening
schedule. The purpose joins the same bubbles, fares, assignment, loads and
crowding as other local trips as the model purpose University (shown with School as "School and
university", T-192). Homes, attendance and timing are
estimates; staff remain in Work. Universities in other countries need country/source adapters.

**Airport ground trips (T-142, every city).** Air passengers travelling between an airport and
their home or hotel. `pipeline/enrich_airports.py <city>` takes each airport with scheduled
service in the pack's counties from OurAirports (global), its CY2024 boardings from the FAA
(national; Wikidata's patronage figure, halved, where the FAA lacks it), and sets its daily
outings to boardings x O&D share x pack share / 366. The O&D share is the part of boardings that
start the passenger's air journey there rather than connect (BTS DB1B Market, 2024 Q2,
domestic). The pack share is the airport's catchment inside the pack's counties: every US county
within 300 km by its 2020 population x exp(-d / 20 km) from its centre of population. The outings sit on the airport's passenger
terminals (OSM `aeroway=terminal` within 4 km, split by footprint; cargo, general aviation and
military terminals left out by name; the airport's point where none is mapped). One outing is a
trip to the terminal and one from it (a passenger leaving, a passenger arriving): boardings and
alightings are about equal, so legs a day are twice the outings. The other end is half
residents, a quarter jobs (business visitors) and a quarter leisure jobs (hotels, food,
attractions), each normalised, none within 1.5 km of a terminal (its own jobs are not its
passengers' ends), drawn with a 20 km exponential decay and normalised per airport, so each
airport keeps its total before the 150 m exclusion and the 40-80 km taper. Two stratified
destination samples a home cell, not sixteen: the destinations are a handful of airports, and
each sample costs the same at opening. Airport staff are commuters already (LODES) and are not
counted again. Trips to an airport in another pack (Springfield to Bradley, Washington to BWI)
and connecting passengers are not modelled. Per city: [T-142](../notes/T-142.md).

**Medical visits (T-142, every city).** Patients and visitors at hospitals, clinics, doctors and
dentists; staff are commuters already. `pipeline/enrich_medical.py <city>` takes OSM
`amenity=hospital|clinic|doctors|dentist` (or the same `healthcare=*`) in the pack's counties and
sizes each in visits a year (judgement scaled to national figures): a hospital 1,000 per bed
(beds by T-387's worldwide rule, docs/demand-worldwide.md "Medical": OSM `beds`, else from the
site's floor area, else a pooled median, all scaled to the World Bank's national beds per 1,000
people), a clinic 10,000, a doctors' practice 4,000, a dentist 3,000. A hospital mapped twice
(a building or point inside its campus) counts once. Residents make medical outings at their
country's rate (above; the US 0.044 a day, NHTS), to these places by size and a 2.0 km exponential
decay (T-637), origin-constrained like errands but with two destination samples a home cell (the
city's medical trips in aggregate are what the game shows). OSM's
doctors and dentists are mapped unevenly; where they are missing, their visits go to the nearest
mapped places instead. Per city: [T-142](../notes/T-142.md).

Unmodelled purposes are labelled explicitly. Activity chains, visitor origins and cross-pack everyday outings
remain later work. Research: [T-138](../notes/T-138.md); implementation and checks:
[T-066](../notes/T-066.md), [T-140](../notes/T-140.md).
