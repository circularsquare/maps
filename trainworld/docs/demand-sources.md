[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: where trips come from (4.1)

Cells, jobs, homes and the seven purposes' inputs.

### 4.1 Where trips come from

A **doubly constrained gravity model** from homes to jobs with a distance-decay curve. Tiers say
how good the inputs are. **The registered cities have playable tier 1 packs; tier 0, from global
sources only, is built (`pipeline/tier0.py`, T-287) and measured on Toronto** (notes/T-287.md):

| Tier | Inputs | Status |
|---|---|---|
| 0 | GHSL population grid scaled to district totals + a jobs proxy (GHSL non-residential volume weighted by height, OSM job places, population) + OSM, Wikidata and OurAirports for the other purposes | built; no city uses it yet |
| 1 | population + real jobs counts by small area | build |
| 2 | a real home-to-work OD matrix | not an input; used as a **check** on the tier-1 fit where it exists (LODES OD for the US) |

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
or outside its zone square. The pack carries each zone's res-9 cell count (`zone_cells9`), which the sim takes as the
zone's land. Together the rules remove 47-68% of the cells of a metro outside the US, 70-78% of
a US one, 56-70% of a Japanese prefecture and 78-79% of a rural US county. Enrichments and routes run on the res-9 pack,
kept in `data/work/res9/`.

Subway Builder also uses gravity: LODES gives it homes and jobs, then a distance-based gravity
model picks each commuter's workplace (`research/competitors.md`). Tier 1 done well is at its level.

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
of students, retirees or children sends fewer commuters than its population (T-054: county
totals now within 10% of ACS commuters, the Bronx was 23% over). Elsewhere, population until a
source by residence is found.

**People in institutions are not homes** (T-234, Anita: "why are there homes on rikers
island?"). The census counts people where they are held or cared for: prisons and jails,
juvenile facilities, nursing homes and other institutions (psychiatric hospitals, hospice). They
make no commutes, errands, outings or school trips, so the 2020 census's institutional group
quarters (redistricting file PL 94-171, table P5, per block, every state) are taken off each
block's population before anything is built on it: residents for everyday outings, child and
adult homes for school and university, the medical and airport origins, the market's sites. The
park draw's nearby-residents term still counts them until parks are next fetched from OSM
(T-235): the total moves under 1%, but a small park beside a big prison draws up to about four
times too much. All four institutional types go, nursing homes included:
their residents rarely leave. The non-institutional types stay: college dorms (students
travel, and feed university trips from their own block), military quarters and other
non-institutional places (shelters, group homes). LODES also puts job holders on such blocks
(Rikers Island: 676 on blocks where 27 people live outside the jails), so on a block with an
institution the commute home end is capped at the residents left; elsewhere LODES stands. Over the
nine packs this takes 362,000 people (0.8%) and 66,000 job holders (0.3%) off homes; per city:
[T-234](../notes/T-234.md). `pipeline/group_quarters.py` holds the source and the rule;
`build_city.py` applies it as it reads the blocks.

**People who work at their own home make no commute and are not counted** (Anita). LODES places
a remote worker at the employer's office, so commute trips are scaled down by the share working
from home (US: ACS "worked from home", by tract at the home end, by industry at the work end). Jobs whose work is at other people's homes
(home care aides) are real commutes and are spread over where people live (T-014).

**Every registered city has the same non-work purposes from the same national sources** (T-142):
everyday outings from LODES sectors, school attendance from NCES public schools, university
attendance from IPEDS, all built by `pipeline/enrich_all.py <city>` onto a finished pack (and by
`build_city.py` for a new one). No city has a city-specific source: New York's five boroughs used
NYC DOE building enrollment until T-181 (Anita, 2026-10-12) and now use NCES like everywhere
else. A pack without these arrays opens
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
within 2 km grows with a park's edge: counted, Harriman State Park (19,000 ha) outdrew Central Park
five times; by density Central Park draws 17, Harriman 15. The draw is spread over the pack cells inside the place (the nearest cell within 500 m
for a small one) as the `park_draw` array. The 8% share is the American Time Use Survey 2023:
7.6% of leisure outings (eating out, socialising and leisure, sport and recreation) end
"outdoors away from home" (95% interval 6.4-8.7%); friends' homes are 23% there and restaurants,
gyms and other places 70%. Parks take their share evenly from the other two draws, so the 50/50
balance T-066 chose between them stays. Parks change where social outings go, not how many there
are; no season (ATUS is a year's average). Per city: [T-196](../notes/T-196.md).
The pipeline spreads original sector locations by the same block land-area method as population;
office/headquarters commute corrections do not relocate these customer weights.

Destination gravity is origin-constrained: zone attraction times exp(-distance/scale), normalized
per origin zone, with the pack's country's scales (US: errands 2.15 km, social 2.42 km, medical
3.53 km), each set so the game's mean outing length matches the survey's mean straight-line leg
under 46 km (the game's mean is about 0.7 + 1.7 x the scale at 2 km zones in every pack);
same-zone distance is 1.04 km.
Deterministic stratified destination samples per cell/purpose integrate that distribution: one
per 8 of the cell's trips a day for the purpose, rounded up, 2 to 16 (T-397; none for a cell
with no budget for that purpose),
then choose an actual destination cell by attraction weight. This is an approximation to
continuous demand, not sampled individual travellers. It is stable across networks and reloads.
**Everyday trips below 150 m cell-centre distance vanish entirely**, with no replacement or
denominator contribution, including walking. The cutoff is limited by cell grain; real within-cell
journeys are not separately represented. Existing commuting behavior stays unchanged.
Local weight tapers smoothly from 1 at 40 km to 0 at 80 km; trips at 80 km or more disappear
from this layer. Only within-pack everyday outings are generated here. Since T-276 the regional
market's longer trips start at 25 km (docs/zones.md 5), so the two overlap from 25 to 80 km; the
local outings' 1.3-3.7 km decays leave under 2% of social outings past 25 km, so the double count is
negligible, and airport trips (20 km decay), which the market does not place at airports, keep
their reach (decision T-276).

After station access is rebuilt, eligible cell edges aggregate into purpose-specific subzone
pairs, retaining weighted length and walking shares. Outward and return mode/route choices run
independently on each period's directed graph with the same fares, access, road costs, loading and
frozen combined crowding as all other demand. MSA retains direction-specific rail probabilities
for cell, selection and period queries. City mode splits, line loads and fare income include these
trips. Local bubbles count **trip legs**, twice the former commute-person count before adding
everyday travel; their purpose/timing queries reconcile to those same endpoint totals. Commute
station catchments remain explicitly commuter counts. Source-neutral cell-weight names allow
other countries' destination inputs later; runtime enables the layer for any pack that carries
the arrays (`errand_jobs` and the rest). A purpose whose arrays hold no sites (no retail,
leisure, school, university, airport or medical site; a school or university pair with either
side zero) is an empty slot with no trips; with no leisure sites, social uses residents (and
parks) alone. Residents must be positive, and a pair with only one of its two arrays is an error.

**School attendance (T-140/T-150/T-181, metro public schools).** Every pack's schools are the
NCES EDGE/CCD 2023-24 public schools in its counties (`pipeline/enrich_schools.py <city>`): 230
to 1,370 schools in each of the other cities, and in New York 5,064 schools with 2,825,244 enrolled pupils, charter
schools included. 100% of eligible enrollment matched to cells everywhere, with at most two snaps
(New York one, 237 m). Distinct NCES school IDs at shared coordinates keep their own enrollment,
so a building holding several schools counts each once. Operational physical schools with
positive known enrollment are included; virtual/unknown categories are omitted. Missing
enrollment/sites contribute no invented pupils. Until T-181, New York's five boroughs used NYC
DOE building enrollment instead (844,956 pupils in 1,213 buildings); NCES has 975,570 there,
the difference being mostly charter schools (141,775 pupils), which the DOE building report did
not carry (notes/T-181.md).

Origins use 2020 block residents weighted by ACS 2023 tract ages 5–17 plus two fifths of
under-fives, approximating ages 3–4, spread by block land area across all the pack's counties.
Missing shares use full-state county FIPS fallbacks, or the pack's age share for Connecticut gaps.
The New York pack contains about 3.87 million estimated child-home weights. Each destination cell generates
0.9 attendance outings per enrolled pupil on a typical school day. This attendance factor is
provisional. Deterministic home samples per school cell (one per 8 attendance trips, 2 to 16) use child weights and a 2.5 km
distance-decay scale. Unlike customer trips, school demand is destination-constrained by
enrollment. The same 150 m exclusion, local-distance taper and directional assignment apply.

School attendance joins bubbles, purpose/timing inspection, fares, loads and crowding as the
model purpose School, shown to the player together with University as "School and university"
(T-192). Private schools are not modelled (NCES's Private School Survey is the
national source for them, T-142).
Estimated homes are not observed student journeys or catchment/grade assignments.
There is no school calendar, escort demand, school-bus choice or student-fare policy yet.
Source selection and international follow-ups: [education data](../research/education-data.md).

**University attendance (T-152, metro IPEDS; every city T-142).** IPEDS 2023 directory and Fall
Enrollment records supply, in New York, 261 active degree-granting reporting institutions and
836,714 eligible students (the other cities: 8 to 76 institutions, 48k to 282k students). The
layer uses degree/certificate-seeking undergraduates plus graduate students and excludes
exclusively distance-education students. Nondegree undergraduates are omitted. Structural
blank online counts are accepted as zero only when IPEDS marks them not applicable and reports the
whole cohort as non-distance; unknown counts remain excluded. One budget is attached to each
reporting UNITID's directory coordinate, avoiding child-campus and merged-institution duplicates.
The exception is an institution that absorbed others in the data year (IPEDS `ACT` "C" with
`NEWID` pointing at it): its total is split over its own and the absorbed campuses' points by
each one's own eligible enrollment the year before (IPEDS 2022). This is Connecticut State
Community College, which in 2023 reported all twelve former community colleges at Hartford; split,
Hartford keeps 5 of them, New Haven gets Gateway and Naugatuck Valley, and New York's Fairfield
County gets Housatonic and Norwalk (5,330 students New York had not counted). Otherwise this is
not campus-level allocation: a multi-campus institution with one UNITID (Northern Virginia
Community College, Montgomery College, Community College of Rhode Island) sits at one point.

Adult homes use block population multiplied by the ACS tract age-18+ share. A provisional 60%
typical-day attendance factor, 6 km distance-decay scale and separate day/evening schedule produce
about 1.0 million trip legs a day in New York. The purpose joins the same bubbles, fares, assignment, loads and
crowding as other local trips as the model purpose University (shown with School as "School and
university", T-192). Homes, attendance and timing are
estimates; staff remain in Work. Universities in other countries need country/source adapters.
Existing school staff commutes remain in Work.

**Airport ground trips (T-142, every city).** Air passengers travelling between an airport and
their home or hotel. `pipeline/enrich_airports.py <city>` takes each airport with scheduled
service in the pack's counties from OurAirports (global), its CY2024 boardings from the FAA
(national; Wikidata's patronage figure, halved, where the FAA lacks it), and sets its daily
outings to boardings x O&D share x pack share / 366. The O&D share is the part of boardings that
start the passenger's air journey there rather than connect (BTS DB1B Market, 2024 Q2, domestic:
0.66 at BWI, 0.74 at Dulles, 0.82-0.83 at Philadelphia and National, 0.90 at the three New York
airports, 0.98-1.00 elsewhere). The pack share is the airport's catchment inside the pack's
counties: every US county within 300 km by its 2020 population x exp(-d / 20 km) from its
centre of population (BWI 0.66 to Baltimore, the rest to Washington; Bradley 0.57 to Hartford;
Tweed 0.47 to New Haven; the big hubs 0.94-1.00). The outings sit on the airport's passenger
terminals (OSM `aeroway=terminal` within 4 km, split by footprint; cargo, general aviation and
military terminals left out by name; the airport's point where none is mapped). One outing is a
trip to the terminal and one from it (a passenger leaving, a passenger arriving): boardings and
alightings are about equal, so legs a day are twice the outings. The other end is half
residents, a quarter jobs (business visitors) and a quarter leisure jobs (hotels, food,
attractions), each normalised, none within 1.5 km of a terminal (its own jobs are not its
passengers' ends), drawn with a 20 km exponential decay and normalised per airport, so each
airport keeps its total before the 150 m exclusion and the 40-80 km taper (to sampling
precision: every airport with over 1,000 outings a day within 0.4% of its input, except
Westchester, 4% short, and Long Island MacArthur, 11% short, whose catchments reach past the
80 km cut). Two stratified
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
(a building or point inside its campus) counts once. Residents make 0.04 medical outings a day
(0.08 legs, against NHTS 2022's 0.091 medical/dental legs per person a day: 4.02% of 2.27 trips),
to these places by size and a 5 km exponential decay, origin-constrained like errands but with
two destination samples a home cell (the city's medical trips in aggregate are what the game
shows; against sixteen, the walk share moves at most 0.3 points and the mean length 0.1 km). OSM's
doctors and dentists are mapped unevenly; where they are missing, their visits go to the nearest
mapped places instead. Per city: [T-142](../notes/T-142.md).

Private schools are not modelled yet (candidate national source: [T-142](../notes/T-142.md));
unmodelled purposes are labelled explicitly. Activity chains, visitor origins and cross-pack everyday outings
remain later work. Research: [T-138](../notes/T-138.md); implementation and checks:
[T-066](../notes/T-066.md), [T-140](../notes/T-140.md).
