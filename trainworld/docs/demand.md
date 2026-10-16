[Back to the spec](../SPEC.md)

## 4. Demand: local trips

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
stay at res 9 (T-466). A group stays unmerged if its centre falls on water or outside its zone
square. The pack carries each zone's res-9 cell count (`zone_cells9`), which the sim takes as the
zone's land. Together the two rules remove 41-57% of the cells of a metro outside the US, 65-73% of
a US one, about half to 60% of a Japanese prefecture and 72-74% of a rural US county. Enrichments and routes run on the res-9 pack,
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
the arrays (`errand_jobs` and the rest).

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
(OSM `beds`, else its site's building floor area, footprint x levels, at the pack's own tagged
hospitals' beds per floor square metre, about one bed per 190 m² in New York, else their
median), a clinic 10,000, a doctors' practice 4,000, a dentist 3,000. A hospital mapped twice
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

### 4.2 Time of day

Five periods: morning peak 6-10, midday 10-16, evening peak 16-20, evening 20-24, night 0-6. They
are the player's scheduling periods too (section 6.3: high demand in the peaks, medium midday and
evening, low at night), defined once in `sim/src/track/params.rs` (`PERIOD_HOURS`,
`PERIOD_LEVEL`) for schedules and demand alike. The hours are local time (T-309,
docs/world.md "Local time"): each line's periods, boarding hours and kind of day are its first
stop's region's, so London's morning peak is London's 6-10. The solve itself has no clock, so
demand's period q is every region's own period q.

Each commuter makes a trip to work and a trip home a day. Of the day's trips to work, 70 / 12 / 6 /
4 / 8% start in the five periods; of the trips home, 2 / 18 / 58 / 16 / 6% (judgement for now). A
trip home goes back by the station pair the morning's trip used, so a period's rail trips are its
share of trips to work on the morning's station pairs plus its share of trips home on the same
pairs reversed. Crowding is judged at each period's busiest hour: 1.4 times the period's mean hour
in the peaks, 1.15 midday, 1.3 in the evening, 1.5 at night. Everyday outings share these combined-
load busiest-hour factors; their five-period departure shares are provisional:

| Purpose/direction | 6–10 | 10–16 | 16–20 | 20–24 | 0–6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Errands outward | 10% | 55% | 30% | 4% | 1% |
| Errands return | 6% | 49% | 36% | 8% | 1% |
| Social outward | 5% | 35% | 42% | 17% | 1% |
| Social return | 2% | 25% | 43% | 27% | 3% |
| School outward | 94% | 4% | 1.5% | 0.4% | 0.1% |
| School return | 1% | 82% | 15% | 1.8% | 0.2% |
| University outward | 65% | 27% | 7% | 0.9% | 0.1% |
| University return | 4% | 42% | 42% | 11% | 1% |
| To the airport | 30% | 35% | 22% | 6% | 7% |
| From the airport | 12% | 33% | 28% | 24% | 3% |
| Medical outward | 30% | 55% | 12% | 2% | 1% |
| Medical return | 8% | 60% | 25% | 5% | 2% |

The inspector uses actual purpose/mode/period selection results. Bars show departures per hour,
dividing period counts by 4/6/4/4/6 hours; outward and return rows remain separate.

**Kinds of day** (T-260, notes/T-170.md). Two: weekday and weekend. Game day 1, a new game's
first, is a Monday, so days 6 and 7 of each week are the weekend (`day_kind_at` in
`sim/src/track/params.rs`, `app/src/game/days.ts`); no holidays yet. The tables above and the
solve are a weekday's. **A weekend is worked out from the weekday's solve, not solved**: without
crowding a trip's mode and route depend on its period's service only, and the solve keeps each
period's loads, boardings and alightings and each city's trips, rail and walking by purpose
(T-158). So the weekend's riders of a purpose in period q are the weekday's times
`rate x (weekend outward + return share) / (weekday outward + return share)`
(`sim/src/demand/days.rs` `load_factor`); the views are built from those on the main thread and
every query (bubbles, catchments, selections, longer trips, transfers) weighs its periods the
same way in the workers, with the everyday purposes' per-direction weekend shares where it reads
directions. The weekend keeps the weekday's crowded route and mode choices (fewer riders would
crowd less), and the same trains run every day (Anita: no separate weekend schedules for now;
T-261 would need its own solve).

Weekend trips per weekday trip, from NHTS 2017 (person-weighted, Mon-Fri against Saturday and
Sunday; England's NTS agrees for work, shopping and leisure) and, for airports, TSA checkpoint
throughput 2025 by day of week: work 0.30, shopping and errands 1.22, social and leisure 1.60,
school 0.08, university 0.22, airport 1.01, medical 0.24, the regional market's longer trips 1.58
(NHTS trips of 50 miles or more). Weekend period shares are the weekday's above times NHTS's
weekend/weekday ratio in each period, renormalised; school, university, airport and medical keep
their weekday timing (too few weekend trips in NHTS to shape them):

| Weekend, purpose/direction | 6–10 | 10–16 | 16–20 | 20–24 | 0–6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| To work | 54.6% | 16.1% | 11.6% | 8.7% | 9.0% |
| Home from work | 4.5% | 17.5% | 36.6% | 28.9% | 12.5% |
| Errands outward | 9.8% | 63.2% | 23.0% | 3.2% | 0.8% |
| Errands return | 6.6% | 58.6% | 27.7% | 6.6% | 0.5% |
| Social outward | 7.6% | 47.8% | 30.7% | 13.4% | 0.5% |
| Social return | 1.5% | 38.8% | 38.1% | 18.2% | 3.4% |
| Longer trips | 15.1% | 34.9% | 23.5% | 23.9% | 2.6% |

**The average day** (T-464) is a third day the views and queries can be asked for: 5 weekdays and
2 weekend days over 7, taken on each purpose's rate and period shares (`days.rs` `AVERAGE`), so
every figure linear in them is the same blend of the weekday's and the weekend's. It is the demand
views' default; the clock, the money and the boarding account know only the two kinds.

On New York's real network a weekend day has 0.77 of a weekday's local rail legs (6.39M against
8.29M; commutes 1.04M against 3.46M, social 2.60M against 1.62M), its busiest period is midday,
and each purpose keeps its weekday rail share. Costs: no extra solve or opening; the weekend's
views about 7 ms on the main thread a publish; the boarding account twice the work (one account
per kind of day); weekend bubbles are asked for only while shown (about 0.6 s in each solving
worker). Weekday results are bit-identical to before (notes/T-170.md).

### 4.3 Mode choice and catchment

Per OD pair and period, reserve the short-trip walking share first, then compare rail with
driving for the remaining trips by **generalised cost** in minutes. Buses are out of scope
(Anita, 2026-10-09), except as part of "other", the choice of homes without a car (below).

- rail = walk to a station + wait (half the headway, capped) + ride + transfers x penalty + walk
  from station + fare / value of time
- driving = door-to-door road time (below; a few constants, real road travel times are not a
  priority) + its cost

Rail share = logistic in the difference (0.05 per perceived minute), plus a **rail propensity**
constant, one worldwide value: **`asc_rail` 0.4** for commutes and **0.4 - 1.1 = -0.7 for everyday
trips** (`everyday_asc_rail` -1.1; T-509, fitted jointly on New York, London, Paris, Seoul, Tokyo
and Sendai's real networks): an errand or a visit takes the train less often than a commute
between the same places (New York's survey: 5-7% of errand and social legs by rail against 21%
of work legs). **Cost damping** (T-495, `Params::mode_beta`): past 60 perceived minutes of
the car's trip T, the scale is 0.05 x sqrt(60 / T), so a minute counts for less on a long trip
(0.029 at 3 hours, 0.014 at 13); one rule for commutes, everyday trips and the regional layer
(docs/regional.md 5.1), T the car's perceived minutes door to door including parking, never the
rail's, so the scale does not change with the network. In the local layers only the choice
between modes is damped (the walks and station pairs behind rail's cost are weighed at 0.05, since
they are shared by every destination); the regional layer damps the whole rail trip, access
included. The propensity is not scaled. It absorbs habit, income and what the car is good for
(trip chains, shopping) in one number (car ownership is its own input, below). **Catchment is soft** (T-090): there
is no circle the player sees. A farther station makes the walk, and so the whole trip, worse, and
the station loses riders where that trip gets worse than driving. Value of time is per country so
fares compare across countries. The US uses $20/hour (3 perceived minutes per dollar, a gameplay
judgement). The base fare is charged once per trip, including transfers; the per-km rate uses
actual track length. Distance fare affects route choice as well as mode choice, and crowding
multiplies ride time without multiplying fare (T-067).

**Walking.** A share of all trips is reserved for walking by distance:
`w = 1 / (1 + exp((d - d0) / 0.4 km))`, d0 1.7 km for commutes and 1.9 km for everyday trips
(`everyday_walk_d0_km`, T-509); rail and driving then compete for `1 - w`.
Thus `walk = w`, `rail = (1 - w) * r`, and `drive = (1 - w) * (1 - r)`, where `r` is the
rail-versus-road result. This prevents attractive rail or expensive parking from taking the
walk share of very short trips. Commutes use the gravity's 2 km zone-centre distance, so a short
trip across a zone boundary can still inherit a coarse walking share. Everyday purposes use the
sampled cell-centre distance. Either distance is times the two zones' walk factor where water
lies between them (T-275, below), so a home across a river from its job is as far as the walk
round by a bridge. The curve gives 84% walking within one 2 km zone, 32% to the next
zone and 6% two zones away; the remainder is offered to rail and driving. With no rail 4.8% of
New York's commutes walk (4.9% with the straight line; ACS: about 6%). Judgement until fitted.
notes/T-153.md.

Weights (perceived minutes per real minute), from Subway Builder's published constants where they
exist: riding 1.0, the walk to and from stations 1.5 (below), walking between platforms on a
transfer 1.39, platform wait 1.37, waiting at home before a known departure 0.4, congested driving
1.33 (also on a ride to a station), plus parking (below).

**Driving** (T-090, T-275; `place_min_per_km`, `drive_min_f`, `drive_factor` in
`sim/src/demand/kernel.rs`). The road is the straight line times the two zones' routed factor
(below) times 1.074, or 1.3x the straight line in a pack without routes. From each end it runs 2 km
on that end's local streets, then 8 km on that end's main roads, and anything left in the middle
is open highway at 100 km/h (T-370); a road too short for all that splits it evenly between its two
ends (at most half each for the local streets, then half the rest each for the main roads). Each
place has a speed for its streets and main roads, by its density (commuters living and working per
km² of land in its 2 km zone): local streets 40 km/h and main roads 80 km/h in open country, both
falling towards 15 km/h where it is dense, half way at 5,000 per km². So a dense end slows only its
own last 10 km: 3 km inside Manhattan takes about 13 minutes, Brooklyn to Midtown (10 km) about
30, 25 km between suburbs about 28, and 60 km from a suburb into a 30,000/km² centre about 71.

**Parking** (T-380, T-372; `Params::park_parts`). At the destination every driver pays 3 perceived
minutes to park and any congestion charge there; a paid space costs `15 x ln(d / 3,000)` more per
one-way trip, capped at 50, with `d` the jobs per km² of land in the destination's 2 km zone (none
below 3,000: 18 at 10,000, 40 at 42,000, 50 from 85,000). A share `1 / (1 + d / 8,000)` of the
drivers to that zone park free (80% at 2,000, 50% at 8,000, 16% at 43,000). Rail's share is the mix
of two logit results, one against the free driver and one against the paying one, weighted by
that share, not one result at the mean price. Congestion charges (`kernel::CORDONS`, a place's
charging area and its charge, half the day's charge at 3 minutes a dollar) apply to the share of a
zone's jobs inside the area: Manhattan south of 60th Street 13.5 minutes ($9), central London 35.1
(£18); nowhere else. A ride to a station pays the mean over drivers there. The same rule serves all
three layers: the commutes and the everyday layer call `park_parts` (the everyday layer takes the
mix of two logit results per pair, `rail_share_parked`); the regional layer's market pack holds each
site's mean over drivers (`site_park`), written by `pipeline/intercity.py park_by_cell`, which
mirrors `park_parts` and `cordon_by_zone` and is used for every zone's pack.

**Routed walking and driving** (T-275, notes/T-275.md). Walks and drives go round water and
over bridges, from data that exists for any city: OpenStreetMap roads and bridges (Geofabrik
extracts) and the pack's own water mask. `pipeline/routes.py <city>` adds them to a finished
pack, in place, and must be rerun whenever the pack's zones change (`build_city.py`, a gravity
re-solve); `routes.py --check` lists the packs whose routes are missing or stale. A pack without
them, or with them made for other zones, has every walk and drive straight x 1.3, as before.
- **Walks.** A barrier raster at 50 m (`walk_row`, `walk_x`, header `routes.walk`): a pixel is a
  barrier when 3 of its 4 water-mask pixels are water, every OSM way a person may walk on that
  crosses as a bridge (footways, paths, streets; not motorways, trunk roads only with `foot=yes`)
  is land, and islets under 0.1 km² are water. Stations are the player's, so the walk to them is
  searched in the game: from each station over the raster's pixels within its 3 km bound
  (eight neighbours, never across a barrier's corner), and a cell's walk is its straight line
  times how much longer the search's path is than the free path between the same two pixels,
  exactly the straight line where nothing is in the way, and none (no access) where water cuts
  it off within the bound. The cells whose walk differs are kept per station position for the
  worker (`routes::station_walks`), shared by a city's commute and everyday populations, so an
  edit searches only the stations it moved or added. The walk to and from stations, the
  access subzones and the trip explorer all use these distances. Between zones, the walk's
  factor to every zone within 2 squares (`walk_zone`) comes from the same search run from each
  zone's representative cell over the other zone's cells, both directions averaged.
- **Drives.** Per zone, the routed factor (road metres over straight metres, commuter-weighted
  over the other zone's cells) to every zone within 6 squares (`drive_near`), and between
  supercells of 4 x 4 squares per region beyond (`drive_far`, `drive_super`). The roads are
  every public road but service ways; each place joins its nearest road point whose straight
  line crosses no water, at 1.3x that line. The speeds stay the density model's above: only the
  road's length comes from the routes. The factor is multiplied by `road_routed_scale`,
  1.3 / 1.210: shortest-distance routes come to 1.17-1.26x the straight line, commute- and
  distance-weighted (New York 1.210), while the density speeds were fitted on New York with 1.3,
  so New York keeps its fitted level and other places their difference from it (and the longer,
  faster roads drivers really take are covered). Hoboken to Midtown is now 7.5 km by road (5.0
  before), St George to Wall St 22 km (12), Stamford to Huntington 83 km (30).
- **Not routed**: walking transfers between stations (300 m and complexes) stay straight x 1.3,
  because every population must build the same graph from the same stations to share its
  strategy searches (T-119) and none of the real New York network's 80 pairs within 300 m has
  water between them; ferries are no walk or drive (T-280); the regional market's drives
  (T-276's redesign).
- **The map's walking minutes** (T-312, `app/src/game/walks.ts`): the closest-stations lists of
  the station inspector and the bubble hover walk the same barrier raster, read on the main thread
  from the pack's `walk_row` and `walk_x` (a Range request, about 250 KB a city) with a TypeScript
  copy of `WalkGrid::field` and `Field::eff` (same constants, same search), so a hover never waits
  for a demand worker mid-solve. Minutes = routed metres x `walk_detour` 1.3 / 6 km/h, the model's walk. It does
  not check `zones_check`: a pack whose routes are stale still has its barrier here. Change the
  search in `routes.rs` and `walks.ts` together.

**The interface** (for other work, T-276 included). Byte tables, factor = `exp(q / 64)`. A zone's
window slot is `(dx + W) * (2W + 1) + (dy + W)` for the other zone's square offset (dx, dy), W 6
for driving and 2 for walking; the pack stores, zone by zone, a byte for each zone of its window
that exists in slot order then region order (a square cut by a region border holds one zone per
region), and the sim expands them to slot x region-count + region. `drive_far` is
supercells x supercells, supercells numbered in order of their first zone. The header's `routes`
block holds the windows, the supercell size, the zone count and `zones_check` (a sum over zones of
position and region, `routes::zones_check`; the tables are used only when it matches).
**Lookup**: `Params::drive_factor(i, j)` (`Routes::drive_factor` x `road_routed_scale`) and
`Gravity::walk_share_pair(z, i, j)`. Across pack edges nothing is stored; the pipeline's
`routes.drive_factors(lon, lat, extracts)` gives the same factor between any list of places from
the same roads (unscaled).

Route choice is not all-or-nothing shortest path. NIMBY Rails' best-known complaint is riders
waiting 50 minutes for a fast train that arrives one minute earlier. Wait time counts at its
weight and the crowding loop (4.4) spreads riders.

**Access and egress.** Riders spread over the ways they could reach a station (walking or riding
to each one) by a logit on total cost (scale 0.2 per perceived minute: an option 5 minutes worse
gets 37% of the better one's weight), and rail as a whole is priced at the logsum, so a second
usable station helps a little and a slightly better one does not take everyone.

**Walking to a station** (Anita, 2026-10-14, T-371). A true walk at both ends: **6 km/h along a
path 1.3x the straight line, or round water by a bridge where one is in the way (above; 13 minutes
a straight km), each minute weighing 1.5** (`w_walk`, fitted on New York in T-372;
notes/T-372.md). So a station 500 m away costs 9.8
perceived minutes, 1 km 19.5, 2 km 39. Between walks the logit above applies (a station 500 m
further gets 14% of the nearer one's riders).

**Riding to a station** (Anita, 2026-10-14, T-295, T-371, T-389). Car, bus, bike or a lift are one
mode, the ride, called driving in the game. It is offered only at the home end of a trip of at
least 5 km (`ride_min_km`: shorter, the ride cannot beat driving the whole way), from the home
zone's home-weighted centre: for a commute, 5 km between its two zones' centres; for an everyday
trip, its own mean distance between home and activity cells, at the home end both ways (the trip
out boards there, the trip home leaves the train there). It costs **`p_ride` 15 + 1.33 x the drive
to the station + the cheaper of parking at the station and a drop-off, `d_drop` 10** perceived
minutes: the drive is the whole-trip drive's rule (below), from the home zone to the station's
zone, and parking is a destination's mean over drivers (Parking, above). In a suburb
a ride to a station 1 km off costs about 21, 3 km 26, 10 km 35, so past about 1 km it is cheaper
than walking and access flattens out: a long commute's rail cost grows slowly with the distance to
the station, as Anita's curve does. Which stations: within 15 km straight line (`ride_max_m`), the
cheapest station of each line, then those of the 6 cheapest lines (`ride_lines`), so a rider
skips the local stop for the line they want. Homes beyond every walk (3 km) are one subzone per
6 km square (3 x 3 zones) with rides only, from its home-weighted centre, keeping each zone's
trip total but taking the most populous zone's distance decay; the work or activity end walks
only. Car parks are automatic and unlimited. `p_ride`, `d_drop` and `w_walk` are fitted on the real
New York network (T-372: about 55-65% of commuter rail riders ride to their station).
The regional layer's rides (commutes between zones, inter-region trips at both ends) come from the
same function, `access_options` ([regional.md](regional.md) 5.3).

**Homes without a car** (Anita, 2026-10-16, T-511). Each cell carries the people living in a
household with no car (the pack's `nocar`, `pipeline/nocar.py`); a home subzone's share is their
share of its homes. In the commutes and the everyday layer that share of each origin's trips
cannot drive. It chooses among three by one logit (`Params::nocar_choice`), with the same
propensity on rail:
- **rail**, walking to stations at the home end too (no ride; where the home has rides, its rail
  is priced on the walks alone, and a home with no station on foot gets no rail for it);
- **walking the whole way**, 15 perceived minutes a km of the drive's road (`walk_trip_min`: 6
  km/h weighed 1.5);
- **other** (Anita, 2026-10-16, T-513), standing in for bus, taxi and bike: the drive's road at
  12 km/h door to door (`other_kmh`, 5 minutes a km) plus a wait of 8 minutes weighed 1.37
  (`other_wait_min`, `w_wait`: 11 perceived) plus `other_asc`, no parking. The speed and wait are
  fixed for every city (a city bus's average speed with stops, and half a 15-minute headway with
  some walk to the stop); `other_asc` is 10 perceived minutes (`kernel::OTHER_ASC`, set on New
  York's boroughs against ACS; T-509's joint fit of six cities kept it).

The scale is damped on other's minutes, as the car's on the car's. The walking share reserved by
distance stays as it is for everyone. Rail's share of a pair's trips that do not walk is
`(1 - c) r_car + c r_nocar`, c the home's share without a car; the people with a car choose as
above. Trips by other are counted with driving in every view (docs/ui-demand-views.md); the sim keeps
them apart (`summary` index 10, per layer: trips by other with no rail, from the zone pairs or the
everyday samples, less the no-car rail trips times other's share of the non-rail choice, which
in a logit is the same with or without rail).
The regional layer has no such group yet. Shares by country: the US by tract (ACS B08201),
England by LSOA (Census 2021 TS045), Japan by prefecture (AIRIA cars per household), elsewhere
national cars per 1,000 people and household size (docs/demand-worldwide.md "Driving
alternative"). A pack without `nocar` has everyone able to drive.

Two **computational bounds** keep the work down and are set where little rail is left: stations
are looked for within 3 km straight line of a walk (`walk_cutoff_m`), and a pair of places is
offered rail only if its access and egress together cost at most 80 perceived minutes
(`pair_walk_max`; a 3 km walk costs 59).

Driving is the alternative for the whole trip. Roughly calibrated on the real New York network
(section 9, notes/T-078.md, T-090.md, T-371.md).

### 4.4 Cost of computing it

Measured in `sim/src/demand/` (method and all numbers in notes/T-005.md, T-019.md and T-020.md)
on the real New York pack (21.7M people, 10.2M jobs; 74,920 cells, and 202,337 since blocks are
spread over their area, T-013) and a synthetic city of 153k cells, with a hand-made 434-station,
25-line network.

**Commuting does not enumerate cell pairs.** 75k-150k cells make 5.6-23 billion pairs. Truncating by
distance does not save it: commutes average 15-20 km, and a cutoff that keeps them keeps billions
of pairs. Even pairing only cells within walk of a station is 90-180M pairs on this network
(1.3-2.6 s native per period) and grows with the square of the area the network covers. So demand
works on two levels:

1. **Gravity between 2 km square zones, once per city.** Doubly constrained, commuting residents to
   commuting jobs (4.1), decay by distance: **f(d) = d^-0.8 x exp(-d / 30 km), x k between zones in
   different first-level regions** (the US packs' curve, carried in each pack's gravity block). The regions are US states (a cell's state is the one holding
   most of its census blocks' people and jobs); a square a state line cuts is one zone per state,
   both at the square's position. k is the pack's `border_factor`: **0.23 for the US packs**,
   fitted with the decay length on New York and Philadelphia LODES OD (T-162: each city's own k
   came to 0.235 and 0.219, and 0.228 and 0.207 at T-198's curve), and **0.45 for Washington**, its
   own fit (0.454 at this curve, 0.486 at T-162's), since
   DC is a city counted as a state: with 0.23, DC residents working in Maryland or Virginia were
   3-4% of DC's commuters against LODES's 13.5% each (T-181; `build_city.py` `CITIES[...]
   ["border_factor"]` overrides `BORDER_FACTOR_US`). It stands for everything that keeps people
   working in their own state (separate labour markets, residency rules, tolls); the crow-fly
   gravity without it sent 2-4x LODES across state lines, land borders and rivers alike, while
   water inside one state was about right (T-159). A pack with no factor has 1. T-171 tested the
   alternatives on five packs' LODES (notes/T-171.md 2): per-state-pair factors tie it (they
   only separate DC from real states), a fade near the line is score-neutral, and a factor only
   between metros or a water or toll-crossing term in its place does clearly worse.

   **Borders outside the US** (Anita, 2026-10-12, T-181; no non-US pack exists yet). National
   borders get a factor; inside the Schengen area it is weaker (closer to 1) than across other
   national borders. First-level regions (states, provinces, prefectures, Länder) get none,
   factor 1, by default. The values are fitted on the country's own commuting data when its first
   pack is built. This does not contradict the US packs: there the first-level factor is a
   measurement, not a default, because LODES shows US states cut commuting to about a quarter
   (T-159, T-162). Elsewhere an admin-1 line is not assumed to: Tokyo's prefectures exchange
   millions of commuters, and many admin-1 layers have city-states (Berlin, Vienna, Mexico City)
   where a state-sized factor would be far too strong, as Washington shows. A country whose own
   OD data shows an admin-1 effect can still fit one. The mechanism is the same per-pack factor:
   a pack's `region` per cell becomes the country, and a pack spanning Schengen and non-Schengen
   borders would need a factor per region pair (not built).

   **The curve** (T-198, from T-185's diagnosis). T-006 fitted d^-1 on New York; T-162 kept that
   shape and moved its length from 29 to 40 km to give back the trip length the border factor
   removes. T-185 found the steep near end kept residents of job-rich places too close to home:
   in LODES they commute out to suburban jobs (reverse commuting) about as far down the list of
   nearest jobs as anyone else, and the model sent them 17-26% too short. The flatter
   d^-0.8 exp(-d / 30 km), one value for every US pack, lengthens job-centre residents' trips by
   0.6-1.3 km in all nine packs (still 9-19% short) and moves centre-to-suburb commuting from
   0.73-0.87x LODES to 0.88-0.97x, undershooting as Anita asked. Trips misplaced by length, county
   pair, nearest-job rank and centre/suburb class fall in eight packs (sum of the four 17-35% to
   12-27%); New York, the densest and most centre-heavy, is about neutral (16.2% to 16.7%: its
   own-county share falls from 42.0% to 40.1%, LODES 42.8%). Per pack: notes/T-198.md. It does not
   depend on the network, so edits never rerun it. The pipeline solves it (Furness, the sim crate's own code run natively, ~3 s)
   and ships a zone id per cell and two factors per zone in the pack (notes/T-004.md); opening a
   city rebuilds zone trips in ~2 ms in WASM instead of solving them for 8 s. Zone trips spread to
   cells by commuters at each end. **Joint gravity** (T-276, docs/zones.md 4): the shipped
   factors are those of one solve over every zone pack (`bin/joint_gravity.rs`, about 12 minutes,
   run by pipeline/intercity.py), so a pack's zone trips are the part of each commute that stays in
   it; `zone_home` and `zone_work` (per zone, the trips kept) let cells count their commuters per
   zone (`DemandApi::home_k`, `work_k`) instead of the pack's total over its people. Each pack's
   homes still send as many commuters as it has jobs (scaled per pack, as alone: scaled to the
   union, packs with net commuting from outside every zone would balance through their
   neighbours, and the solve barely converged). Furness stops at 500 iterations, max row error
   2e-4 (one Boston zone; the rest far below).
2. **Access per cell**: the 6 nearest stations by walk within the walk's 4 km bound (4.3), found by
   scattering each station over the cells near it. 7-13 ms native. With routes (T-275) each
   station's walk is searched once per position (0.6-1.2 ms a station: opening the real New York
   network's 867 stations costs 0.5-0.6 s native and 1.0 s in WASM, once per worker; an edit
   searches only its new or moved stations and costs what it did, 31 ms in WASM with or without).
3. **Access subzones**: the cells of one zone that share their nearest station. Each carries its
   perceived walk to each of its stations, averaged over its cells as the mode choice weighs them
   (a logsum at its 0.05 scale: the walk's cost grows faster than its length, and the near cells
   carry most of the rail), used at both ends of a trip. Cells with no station within 4 km are in
   no subzone: their commuters drive or walk. 2,750 subzones on the hand-made network, 5,032 on New
   York's real one (2,056 and 3,702 with the 2.5 km edge before T-090); access and subzones
   together are rebuilt on every edit in 16-27 ms on the hand-made network and ~57 ms on the real
   one in WASM.
4. **Station-to-station times**: one reverse strategy search per destination station over station
   nodes plus one node per stop per direction per line (T-069). At each station, useful lines
   share their departure frequency: boarding costs the weighted wait (half the combined headway:
   the first 5 minutes at 1.37, beyond that at 0.4) plus their frequency-weighted downstream cost
   and a transfer penalty, taken off once. A line joins only if it improves expected trip cost;
   wrong branches and off, slow, expensive or crowded alternatives add no useful frequency when
   they worsen that cost. A rider leaving
   a train may walk to another station within 300 m straight line and board there (walk at its
   weight; walk edges start only after a ride, so walking between stations is never a rail trip).
   Snapshot stations within 100 m of each other are one demand station (side-by-side platforms of
   two lines; never two neighbours on one line). **Station complexes are transfer points** (T-161):
   the clock worker sends each constructed station's complex with the snapshot (game/complexes.ts
   over constructed stations only, with the player's links, T-197; `DemandApi.set_complexes`), and
   a rider leaving a train at one may walk to any other demand station of the same complex however
   far, at the same cost as any walking transfer: the centre-to-centre distance x 1.3 detour at
   1.2 m/s, weighted 1.39, then the boarding's usual wait and 5-minute transfer penalty. No extra
   fixed penalty: the 5 minutes already fall on every change, at one platform or after a walk, so
   a walk of 120 m costs 3 perceived minutes more than a change across the platform and a linked
   pair 1 km apart 25 more. The members stay separate demand stations joined by walk edges rather
   than one merged station: each keeps its own catchment and position, a long link is not a free
   teleport, and transfers stay counted platform to platform for the complex map (T-176). The
   trip explorer shows such a change as a walk between the two stations. On the real New York save
   this changes nothing: the 100 m merge (single linkage, so a chain of platforms is one demand
   station) already makes its 855 complexes 873 demand stations, and the 20 pairs of demand
   stations sharing a complex are at most 269 m apart, inside the 300 m walk (Penn Station's 24
   platforms are 2 demand stations, Grand Central's 16 are 3, Times Sq's 10 are 2;
   `complexes_real_network`, notes/T-161.md). It matters for members more than 300 m apart: long
   platforms end to end, a long chained complex the player builds, and linked stations.
   Destination strategies and boarding frequencies
   are kept for loading. Each destination's search is independent, so only stations someone can
   end a rail trip at are searched towards (those in the access lists of subzones with jobs,
   `egress_stations`): a population living near two stations of an 875-station network needs two
   searches. The tables are shared by the populations of a worker (`kernel::with_skims`, T-119):
   populations whose graphs are the same to the bit (a city's commutes and everyday layer; on the
   real save also New York and the regional market) search once per period and round. Each city
   converts station positions to its own frame, so cities can differ in the last bits and bring a
   graph each. A worker keeps two tables; when a third graph comes, the table with the fewest
   searched destinations gives way, and only to one with at least as many to search (a cheaper one
   is searched for that call alone), so the order populations are solved in does not change what
   is searched (T-405). A table is stations² costs plus, per searched destination, each node's next
   node and the settle order (u16 ids below 65,535 nodes) and one bit per boarding route node for
   the departures accepted there (their trains an hour are the graph's): about 4 bytes a node, 18 MB
   for New York's on the real save (T-398).
5. **Mode choice per subzone pair.** The access logit factorises: each origin subzone gets one row
   over all stations, the logsum of its access options followed by every station-to-station cost
   (at most 6 options x stations multiply-adds); each pair then multiplies that row into its
   destination's egress weights, up to 6 multiply-adds, one ln and one logit per pair. Pairs whose
   two walks together pass `pair_walk_max` (80 perceived minutes, 4.3) are skipped without being
   visited: each zone's destinations are kept cheapest walk first, and the destination zones by
   their cheapest walk, so an origin stops at the first that is too far. That bound is what holds
   the pairs where the 2.5 km edge had them: hand-made network 4.2M pairs (4.1M before T-090;
   with the 4 km reach and no pair bound 9.3M), real network 13.8M (13.6M; 31.8M). What a pair
   reads about its destination is packed in visiting order, its egress list padded to 6 with
   weight-0 entries. An origin first lists its pairs within the bound, zone by zone, then takes
   them eight at a time in SIMD lanes (`lanes.rs`, T-036: wasm SIMD in the browser, SSE2
   natively), logsum, drive, logit and the spread of each pair's rail trips onto its egress
   stations in one pass. Each lane does the scalar expressions in their order, a pad adds exactly
   0, and every sum keeps the pairs' order, so the results are those of one pair at a time to the
   bit (the trip explorer's `pair_choice` still computes one pair alone). Rail trips go back onto access
   stations once per origin subzone and are kept by station pair (over the stations in the city's
   access lists only, since no other station starts or ends a commute: a city a few stations reach
   keeps a few rows, not the whole network's, T-398), and by subzone and access
   station at each end for the demand views (T-084, notes/T-078.md). The views ask for them on
   demand, never pushed: commuters by mode per cell (`cellModes`), a station's catchment and far
   end (`stationRiders`, with the home riders split into those who walked and those who rode to
   it, commutes only: everyday trips and the regional layer are not split, T-374), and where the commuters living or working in any set of cells go, every
   mode (`flows`, T-098 for the bubbles: the gravity's zone trips from the selection, rail along
   the kept station pairs, walk and drive the rest; totals exact and equal to `cellModes`, far
   rail by place blurred like `stationRiders`; 10-20 ms in WASM for one cell or thousands, the
   far cells holding the last 1% left out as a cut total; notes/T-098.md). A selection's everyday
   purposes (`EverydayApi::selection_sparse_day`) visit only the selection's destination samples
   (a purpose's samples laid out by the selected end are read by range, the others scanned for
   selected cells before any is weighed) and answer only the far cells they reach; the main
   thread adds the workers' answers up over those cells (`workers/selectionMerge.ts`). A click on
   a bubble on the real New York save shows its far end 80-170 ms after the click when no solve
   is running (first click of a session about 300 ms); it took 0.5-0.9 s (1.5 s first) when every
   sample of the city was weighed and every worker sent three numbers per cell per purpose
   (T-285, notes/T-285.md).
6. **Loading**: each destination's strategy is walked from farthest settled nodes towards it.
   Station flow splits by accepted departure frequency, retaining each line's individual segment
   loads and boarding/alighting counts (T-069). Transfers are counted beside the loads (T-176,
   "Transfers" below).
7. **Crowding**: a penalty per segment from the loads, repeat 4-6, average the loads (method of
   successive averages). Each round costs as much as the first pass. The riders projects worked
   this out; read memory `reference_transit_crowding_assignment.md` (in
   `~/.claude/projects/c--Users-anita-projects-maps/memory/`) first: the penalty starts where seats
   run out, never saturates, and averaging is what stops it ringing. **At most three rounds**
   (T-022): the first round leaves the segment that takes the overflow at its worst (963% of crush
   on the real New York network against 310% before it). No crush cap: an overloaded line is the
   player's signal (notes/T-022.md). **Each period stops early once settled** (T-434,
   `kernel::settled`): a round is skipped when the round before moved the period's summed segment
   loads (every population's, rider weighted: the sum of the changes over the sum of the loads) by
   under 1%. The first crowding round always runs; the peaks keep all three, quiet periods stop
   after one or two (notes/T-434.md). A round in which no segment is over its seats has the
   free-flow estimate's factors (all one), so it would solve the same pass again: it is skipped,
   with exactly the same result (T-119). A skipped round still counts as a round, so every period
   reports the same round numbers.

   **Riders left behind** (T-188, `kernel::left_behind`, `round_costs`). Each round also prices
   the wait for full trains. From the same frozen loads (every population's segment loads,
   boardings and alightings per route node; the worker sends all three), each direction of each
   line runs the boarding account's rule (docs/track-and-service.md 6.6) on the period's own
   trains: an even headway at the period's frequency, riders arriving at each stop by the
   account's shape for the period (fastest mid-period, the busiest hour at the busiest-hour
   factor), each train letting off the stop's alighting share and taking riders up to crush, the
   rest waiting a headway more for the next train of the same line and direction. A queue left
   at the period's end clears at the trains' average spare room. The rider-minutes waited
   beyond the first train, over the stop's boardings, are that boarding's average extra wait
   (capped at 60 minutes, `LEFT_WAIT_CAP_MIN`, for a line whose queue never clears), added to
   the line's cost at that boarding at the platform wait's weight (1.37) in the strategy search,
   so riders spread to other lines, other stations or the car. The free-flow estimate has none.
   Decisions (DECISIONS 2026-10-13):
   - **Inside the solve, not read back from the clock worker's account.** The account is worked
     out from a finished solve's riders on the main thread's schedule; feeding it back would need
     a second solve after every solve (and a third, to settle), and a new game would open on
     waits from no account. Run per period inside each round, it settles with the crowding by the
     same averaging (the waits come from the averaged loads, as the crowding factors do; no extra
     damping), costs nothing measurable (a loop over trains and stops), and needs no state at
     startup. The two are the same rule on the same trains; demand's differs only in using
     even headways and in each period standing alone (no queue carried in from the period before).
   - **Per period, not per hour.** Demand's periods are the solve's unit; the account's hours are
     not mapped back. The arrival shape gives the period its mid-period peak.
   - **Weekdays only.** The weekend is worked out from the weekday solve (4.2), keeping its
     route and mode choices, so it keeps the weekday's waits too; with fewer riders its trains
     are emptier, so this errs towards too few weekend riders on lines full on weekdays.
   On the real New York save (commutes and everyday purposes, native, three rounds;
   `left_behind_real_network`, notes/T-188.md): rail trips a day fall 0.1% (commutes 3,463,591
   to 3,460,098, everyday 4,831,084 to 4,825,736); riders left behind in the evening peak fall
   from 40,600 to 30,400 and in the morning from 5,600 to 4,500; the LIRR Hempstead Branch loses
   4.2% of its boardings and the Babylon Branch 2.3%; no busiest line moves more than 0.8%.

**Cost.** One period on the 202,337-cell New York pack and the hand-made network, one stage (the
free-flow estimate or one crowding round), one thread: 0.25-0.3 s native, 0.27 s in WASM (0.23-0.24
before T-090's wider reach: the extra origin subzones each build their station row and visit the
destination zones, and the drive costs a little more per pair; a day is 15% longer in WASM). An
edit reruns everything except gravity. Periods are independent, so the game runs them in a pool
of workers ([architecture.md](architecture.md) 7): with three workers in Chrome the mode split updated 0.49-0.55 s after a
new network reached them and the third crowding round landed by 1.9-2.0 s at T-078; T-090 adds
about 15% (not measured in Chrome). Each worker holds its own copy of the city (one for its commutes and everyday layer together): 18 MB of WASM
memory open, about 23 MB after solving (notes/T-026.md). New York's real 867-station network is
13.8M pairs, 1.0-1.7 s a stage in WASM, 24-25 s for a whole day on one worker (20 s before T-090).

**Accuracy of subzones** against one subzone per cell with the same model (T-090, hand-made
network, free flow): rail total within 0.2%, station entries and exits within 5.0% and 6.3% (sum
of absolute differences), segment loads within 3.0%. The logit roughly halved the station error
of all-or-nothing choice at any grouping, which is why the 5-minute walk bands are gone; what is
left is the grouping (a subzone's walk is one figure for all its cells). A plain mean of the
cells' perceived walks put rail 3.3% under, since the walk's cost is convex.

**If a bigger network makes it slow** (pairs grow with the square of the subzone count): add
workers first. Skipping zone pairs under 0.2-0.5 trips cuts pairs 1.1-4x but drops long trips
(segment loads 3.5-5.4% off, T-020). **If there is time to spare**,
10-minute walk bands are the accuracy lever: 2.4x the pairs, every error about a fifth smaller.

**Which cities are open** (T-149). A worker holds only the city packs the game needs: the viewed
city and every city with a built station inside its footprint or within 4.5 km of it (the 3 km
walk plus a margin for edge cells and pack frames, `app/src/workers/cityReach.ts`). The ride's
15 km does not widen it: a pack's trips run between two stations of its own pack and the work or
activity end walks, so a station only a ride from the pack carries none of its riders (T-394; the
regional market carries riders between zones). Any
other city has no subzone with a station, so all its trips walk or drive and it adds nothing to
any line's load, fare or crowding: leaving it unloaded changes no number elsewhere. A worker
check solved a Midtown line with all nine packs open and with New York alone; every line load,
boarding, crowding figure, New York and market summary and corridor was bit-identical, and the
other eight cities had zero rail (notes/T-149.md). Packs arrive in `open` messages after the
WASM starts, the viewed city's first; a solve covers the cities open when it starts, and the
client sends one only when every city its network needs is open in every worker. A city that
construction reaches later joins at the next solve, which the client sends at once; a city only
viewed or on screen opens bare instead (below). Opening is cheap for a commute
pack (16 ms New York, 1-7 ms the others, natively); the cost is each city's everyday purposes
(natively New York 0.6 s, Washington 0.3 s, Philadelphia 0.2 s, Boston 0.2 s, Baltimore 0.1 s,
the four small ones 0.02-0.06 s, since T-397's sample counts; in Chrome, measured before T-397
with about twice the samples, New York 1.3-1.9 s in each of five workers, Philadelphia 1.1-1.2 s
and Washington 1.7-1.8 s). A zone's **market pack** (T-377, `<id>.market.json`) opens in the
solving workers once the zone is viewed, beside the view or reached, or a built station lies
within 30 km of its box (T-396's rule, per zone); before that no solve waits for it or covers it,
and a solve sent without a market pack open since is sent again. The workers compose the market
from the packs they hold (`MarketPacks.market`) whenever one opens or closes. **Packs close when
nobody needs them** (T-301): a zone that is
neither viewed, beside the view, reached by a station nor part of the load in hand closes in every
worker (bare or not) once it has been so for 60 s, or at once when more than two such zones are
open (the least recently needed first); a zone's market pack closes 60 s after the zone is in none
of those sets and no station is in its reach. A closed zone carries no rider on any line, so the shown
figures stand (a bare zone's city figures stay shown); the next solve leaves it out, and a worker
solving with it closes it when that solve ends. It reopens as it first opened (download, mostly
from the browser's cache, and opening). The bare copy of a zone that a shown solve covers closes at
once. WASM memory never shrinks, so closing bounds a worker's peak (the freed space takes the next
pack) rather than lowering it. `?perfOff=close` keeps everything open. A new game
opens only New York; a save with construction in several cities pays
each of their openings once, in every worker. Only the cells with a budget for a purpose hold its
destination samples and their pair indices (T-395; about half of all cells x purposes have none),
and a cell holds one sample per 8 of its trips (T-397): natively New York's everyday layer holds
2.3M samples, about 18 MB with their pair indices. A sample's
pair index is held only once a network gives its layer a pair, written in place on the next
network, and the pairs are found through a 4-byte-a-slot table sized from the last network's count
(T-398). A city's everyday layer is opened beside its commutes (`EverydayApi::beside`, T-399) and
shares their cells, zone grid, gravity and routes; its own are the zones' sums of residents and
jobs, and the station access and subzones built on those weights (not the commute weights), so
each network sets up the two layers' subzones separately. Natively a solving worker on the real
save holding three periods has about 156 MB of heap after a solve and 188 MB at its peak; with
all nine Northeast cities open, 282 and 318 MB (notes/T-399.md).
T-172 made the everyday opening cheap: zone centres sit on the zone grid, so the destination
gravity's decay comes from one table per purpose by grid offset instead of a square root and an
exp per zone pair (the sampled destinations agree with before to 7 significant figures in every
purpose's trips, walk share and trip length), and the long loops of the opening and of
`set_network` run in functions called once per zone or per 16k samples, because a browser
optimises a WebAssembly function only between calls (a single long call stays in the baseline
compiler throughout: in Chrome that made the opening 3-5x slower than natively). Startup
timings: notes/T-172.md, notes/T-174.md.

**Bare zones** (T-325, T-336). A zone viewed or on screen beside the view that no built station
reaches opens **bare**: its pack goes to the auxiliary worker alone, which sets its commute and
everyday layers on an empty network and solves every period once. With no station in reach every
trip walks or drives, so these are exactly the figures any solve gives it. No solving worker opens
it and no solve is started or waited for, so a pan never re-solves. The `opened` reply carries
each period's city summary and purpose parts, from which the page builds the zone's city view as
from a solve's results. Bubbles ask the auxiliary worker (`bare`): commuters by mode, and the
everyday legs once per solving worker's periods (`cell_modes_mask_day`), added up in period order
as a solved zone's are. A bubble click asks it `bareFlows` (the commuters' far end, and per
solving worker's periods the work timing and everyday far end: `purpose_timing_mask_day`,
`selection_sparse_mask_day`) and `bareSquares` (the selection as the market's 600 m squares), which
go with `icSel` to the solving workers, holding the market, for longer trips and commutes between
zones. `checks/t325-bare-check.mjs` finds every one of these equal to the bit to the same zone
solved with no station in it.

- **At startup** (T-351) the viewed zone's pack downloads before the network's reach (the first
  network and the footprints) is known, and is sent once downloaded: bare if the reach is known by
  then and leaves the zone out, else to every worker. A new game (`?new=1`) has no station, so its
  reach is known at once: the viewed zone opens bare, the solving workers open only the market and
  solve it (under 0.1 s), and the loading card and held clock go when the bare figures arrive,
  which are final as they come (`holdDone`). New York's city figures and bubbles are identical to
  the bit to the full solve's; the network's own totals leave the bare zone out, as after a pan.
  Card gone 2.6-2.8 s after navigation (dev server, warm), against 6.6-7.1 s before; the bare
  opening is 2.0-2.2 s of it, 0.4 s more without the everyday warm-up, which stays. A saved game's
  network arrives after its viewed pack has downloaded (about 2.1 s against 0.65 s for the real
  New York save), so a save opens as before; waiting for the reach would cost every save that
  reaches its viewed zone about 1.4 s.
- **The first station** (T-351): with no station in the game, picking the track, station or
  blueprint tool opens the bare viewed zone in every solving worker as well (`prefetchViewed`,
  2.4-2.6 s for New York), with no solve started for it, so a first line's solve need not wait:
  its first estimate came 2.0-2.1 s after the build, as when the zone was open everywhere; built
  with no tool picked first, 4.4-4.6 s. notes/T-351.md.
- A station built within reach downloads the pack again (the page keeps no copy) and opens it in
  every worker; the solve waits for it and covers it from then on. Its bare figures serve until a
  shown solve covers it. The auxiliary worker keeps its bare copy (WASM memory never shrinks).
- **Memory and time** (real New York save, five solving workers, a 13-step pan): each solving
  worker stayed at 354 MB, against 855-955 MB when every zone on the way opened in every worker
  and was solved (10 solves); the auxiliary worker grew from 76 to 477 MB; the pool from about
  4.6 GB to 2.25 GB. A bare zone's bubbles came 1.5-3.6 s after arriving, against a final result
  16.6-22.6 s after. A click on a bare zone's bubble drew its far end in 155 ms. Opening bare took
  0.7-1.4 s (Toronto, Philadelphia) in Node.

**Everyday warm-up** (T-173). Even so, five workers opening New York's everyday layer at the same
time from baseline code took 7-8 s each instead of 1.3-1.8 s in most page loads (one worker alone
1.6 s, three 4.1 s each: as if they shared one core; never with Chrome's
`--no-wasm-dynamic-tiering`; not the module cache, efficiency mode or the tiering budget). So
worker 0 opens a small synthetic everyday layer first (`warm.rs`, 22,500 cells, about 150 ms;
ready 0.36-0.66 s after navigation, about when New York has downloaded), and the client holds the other solving workers' city packs
until it is ready (`demandClient.ts` `held`). Compiled code is shared by a page's workers, so
they all open on optimised code: New York 1.3-1.8 s in every worker in 10 of 10 page loads,
against 3.4-7.3 s in 7 of 10 without (`?perfOff=stagger`, same build).

**The world market** (T-377, [regional.md](regional.md) 5.1, docs/zones.md 4-5). Longer trips
follow T-376's rule: ground and tail departures, each over its own normaliser against the world
field, both stored per 6 km zone in the place's pack, so the departures to places with no zone
(`info()[4]`, by place `elsewhere(q)`, a selection's in its last two numbers) need no outside
world. The car's perceived minutes add 15 cents a road km and the destination zone's parking
(`site_park`: the local parking rule's mean over drivers at each site's 2 km zone, job-weighted), both halved for two occupants
(`car_min`). T-378: the plane (`air_min`, [regional.md](regional.md) 5.1) is priced per pair of
places when the market is built (`PairRule`: car or not, the plane's share of the trips not by
rail, the weight of every mode but rail against the reference, the reference's minutes with no
car); a served pair's rail share is `sum / (rest + sum)` against its zones' car time (or the
plane's, with no car), and the split of the rest by plane and car is the pair's
(`corridor_modes`), which the client applies to every figure it shows. Commutes between zones: in the home place's pack,
`commute_home` (its 6 km zone), `commute_place` and `commute_work` (the work place and its zone)
and `commute_trips`, from `bin/joint_gravity.rs` by way of `data/packs/intercity.json`
(pipeline/market_zones.py); at opening merged by zone pair, at `set_network` kept between served
zones (`cpairs`), and in each period's pass chosen
once per commuter on its home-to-work station pairs (logit 0.05 damped as in 4.3, drive x 1.33 + parking at work,
access weights at that scale, the home end with rides, [regional.md](regional.md) 5.3), outward legs at `TO_WORK[q]`, the trip home at `TO_HOME[q]` on the
same pairs reversed; their own assignment (`c_avg`) is the work purpose's share of the market's
loads (`commute_loads`; the worker moves it from longer trips to work), with their own corridors
(`commute_corridors`), parts (`part_summary`: longer trips and rail, then commute legs and rail)
and a tail on `selection_flows` (legs of commuters living in the selection by work zone, of those
working there by home zone). A station's far-zone rows hold both parts and take the longer trips'
weekend factor (a known approximation: commuters are counted 1.58x rather than 0.30x on a weekend
there, T-296). Measurements on the real New York network: notes/T-276.md.

**A composition opens without its zone pairs** (T-174, T-377). A market is the world (the
registry's frame, rule and every place's row) plus the packs of the places a worker holds
(`MarketPacks`, each parsed once). The figures that need all pairs (the day's trips, each
corridor's, the departures to places with no zone) come from the rows, the same in every
composition, so opening one parses its packs and visits no pair (all 20 places: 0.66 s natively).
`set_network_ll` takes stations as longitude and latitude; each place's sites see them in the
place's own frame for access and rides (each place on its own, so its figures never depend on which
other packs are open), the graph in the first place's. It builds pairs only between zones some
station serves whose stations share a track component (T-409: the rail graph's, every line and
walking transfer at any frequency; the commutes between zones follow the same rule) on one
landmass, in the composition's zone order, so every rail figure is the same sum. All-mode trips
for a selection are computed row by row for its zones when asked, against every open zone and, by
place, every place not open (`selection_all_day`, over any periods: the worker asks it of the
composition of every pack it holds and adds each part's rail, `selection_rail_day`). Answers lay
out every place and every 6 km zone of the world (`zone_count`, the rows' zone counts in place
order); a zone not open has no figures of its own and sits at its place's centre. Positions the
app reads are in the market frame (the New York equirectangular one) whatever the place.

**The market's distances are on the ground** (T-128). Each zone centre goes through its pack's
frame to the unit sphere and zone pairs are measured by great circle (`zone_km`; the same as
`track::frame::great_circle`); far places by their centres'. The decay and the drive time follow.
An old single pack (`sites` with a market column: the Northeast's `data/packs/intercity.json`,
which the joint commute gravity reads and writes, and the tests' fixtures) opens as a closed world
over its own zones with the ground curve alone.

The Northeast recomputes every open city population and the separate regional market on service or
fare changes; per-city dirty filtering remains future work. At each period's crowding round, the
worker freezes their combined segment flows and feeds that same vector to all three populations.
Capacities and busiest-hour factors agree, so shared crowding affects every rider. The free-flow
estimate lands when all five periods agree on version and round; three refinements replace it the
same way. A faster worker cannot mix a later round with another worker's first estimate. A newer
network supersedes work after the population in progress (a pass is up to a second of one call).

Longer-trip queries (T-118, [regional.md](regional.md) 5.2) come from the regional API, not the local ones. Each period
keeps rail trips per directed served zone pair (T-174) and, per station, rail trips starting and ending there by
far zone (T-120; one f32 per station, direction and served zone, the only zones a rail trip reaches, T-398;
the market's whole period is about 7 MB on the real save), both from the same
assignment pass that loads the lines. `selection_flows` sums the market's zone pairs (each
selected zone's row computed when asked, T-174) for the 600 m
squares a selection of local cells lies in (a partly selected square counts its selected share;
homes by population share of the zone's departures, jobs by attraction share of its arrivals);
`station_flows` reads the per-station row. Both answer by market (the inspector's places) and by
far zone (all modes and rail, starts and ends; the bubbles), then every zone's own x and y, which
are the same on every worker and so are not added up by the client. Both answer over the periods a
worker holds, versioned and round-aware like `flows`. Tests: place and zone totals equal the
selection's trips, rail never above all, zone positions are the zones', station rows equal entries
and exits. A period's regional solve costs about 8 ms on the small real-pack profile
(3.7 ms before).

Each solving worker holds both local APIs and the regional API; stations transform from the loaded
game's world frame (T-125) through longitude and latitude into each pack's own frame (the regional
market takes longitude and latitude and puts them in each place's frame, T-377; its zone distances
are great circles, T-128). Line and station loads and fares include all
populations. The top bar's denominator and the cell, flow and station-catchment queries remain the
focused city's local commutes. Query XY is returned in world coordinates; cell and zone ids remain
pack-local. Focus changes reset caches, layers and commuter selections, and reject late replies,
including a switch away and back.

**Loads by purpose** (T-158, the map's purpose filter, [ui-demand-views.md](ui-demand-views.md)). Every period's result also
carries its segment loads, boardings and alightings split by purpose: work is the local city
populations together, then the everyday layer's six purposes (four before T-142's airports and
medical visits), then the regional market as
longer trips. The populations were already separate; the everyday layer's purposes are kept apart
inside its pass. Its pairs are stored purpose by purpose, so one station-pair matrix is filled
by a purpose's pairs and then packed into that purpose's columns
(origin stations and trips, by destination), and the matrix reused for the next. One walk of each
destination's strategy then carries every purpose at once, a lane per purpose (T-433,
`assign_lanes`): each lane does the kernel's assignment arithmetic in its order, so a purpose's
loads are those of assigning it alone to the bit, and the total is the purposes' sum, which is
what crowding reads. Each purpose's loads are averaged across crowding rounds like the total, so
they still add up after every round. Nothing is solved again per purpose and mode choice never
sees the split. The kept purpose loads are 3 arrays per purpose per period (one float a route
node each), and each result message carries 8 purposes x 3 arrays. One purpose's local trip legs
per cell for the bubbles come from `EverydayApi.cell_modes_of(kind)`, asked of the solving
workers on demand (`purposeModes`) and answered only from a finished round, like stations; work
alone is the commute sums the bubbles already have.

**Transfers** (T-176, for the complex map, [ui-inspectors.md](ui-inspectors.md)). A station's flow is everyone there, so
loading alone forgets which line a rider arrived on. `kernel::assign_with_transfers` keeps it: when
a route node's riders leave the train at a station that is not the destination (at their own
station, or by a walking transfer to one within 300 m or in the same complex, T-161), the riders and the edge they arrived by
are noted on that station's list for this destination. A station is loaded after every node that
feeds it (the walk goes farthest first), so when it splits its flow over its useful departures the
list is whole, and each arrival splits the same way as everyone there: riders x that departure's
share, added to the slot of (arriving edge, departure). Slots are laid out once per graph
(`xfer_index`; every edge from a route node to a station, then that station's boarding edges in
order), so a slot is a pair of route nodes, line, direction and stop in and out: platform to
platform, merged demand stations included. The loads are `assign`'s to the bit; the transfers are
counted beside them. Every population does it (commutes, the everyday pass with all its purposes
in one array, the regional market), each round, averaged across rounds with the loads' weights.
The everyday walk notes an arrival's riders summed over its purposes' lanes, so its transfers
equal the purpose-by-purpose count to float rounding, not to the bit (T-433).
Check: at every station the transfers leaving equal its boardings less its first boardings, and
on the real network the commutes' transfers equal all boardings less entries
(2,278,897 in periods 0 and 2; 593k of them walk between demand stations).
Query: `transfers(sel)` on each API gives (route node in, route node out, riders) for every
pair with either end at a route node marked in `sel`, for each period held; the worker adds every
population's and period's by pair and answers the `transfers` message with the route nodes of one
complex, on request only (never pushed with results). The client adds the workers' answers and
names each end by station and line, both runs together (`complexTransfers(members)`).
Cost on the real New York network (native, period 0): one commute assignment 36 ms, 42 ms counting
transfers (riders arriving at a station x its useful departures, about half of them, for each
destination); whole free-flow stages best of five, counted against not, commute 1,027 against
1,024 ms and everyday 337 against 339 ms, within the noise, so they are counted every round rather
than on the last only. Memory: 28,538 slots, 111 kB a period and population plus a 43 kB index
(about 0.5 MB a worker holding one period of New York commutes, its everyday purposes and the
market). The New York save's Penn Station answer is 886 pairs, 10.6 kB, 9 ms in the worker.
Loads with and without counting are bit-identical natively (`transfers_real_network`, two periods,
free flow and three rounds, commutes and everyday) and in the WASM worker (five periods, three
rounds, the market too; notes/T-147.md).

**The trip explorer** (T-195, `sim/src/demand/trip.rs`, the interface in [ui-inspectors.md](ui-inspectors.md)). One trip
between two points, as the commute model decides it in each period: `DemandApi.trip(q, from, to)`
takes the cell nearest each point (within 600 m, else the point is outside the city), reads the
trip as a commute from a home in the first to work in the second, and answers with the model's
own numbers for that pair of subzones (4.4 step 5): the walk share (by the two zones' distance),
the drive (real minutes between the subzones' centres, its parking, its perceived cost), the
rail logsum over every pair of access and egress stations and each pair's share of the trip's
rail riders, rail's utility against the road and the chance of walking, rail and driving. The
expressions and their order are `mode_choice`'s, so the figures are the same f32 numbers, on the
period's own graph (the costs of its last crowding round; the published loads are the rounds'
average). Rail is a strategy, not one path: riders spread over the station pairs by the access
logit and, at each station, over the useful departures by frequency. So the answer also walks the
destination strategies `assign` loads, from each station pair carrying at least 0.5% of the
trip's rail riders, and lists every route (a sequence of boardings, with a walk between stations
where the strategy takes one) carrying at least 0.5%, at most eight, biggest first, each with
its share, legs, waits (half the combined headway of the useful trains at the station),
in-vehicle minutes with and without crowding, and distance fare. Strategies are searched towards
the destination's egress stations only (at most six searches, a few ms; each search is
independent, so they are the solve's). It changes no state and is answered between stages, like
the other queries: each solving worker answers for the periods it holds, only from a round that
all its periods hold (`trip` message, the cells' positions in world metres). A trip whose end has
no subzone (no station within 4 km), or whose cheapest walks pass `pair_walk_max`, or with no
connecting service in the period gets no rail; its drive is priced between the two cells.

Checks: over every destination subzone, an origin's rail trips from these pair choices add up to
the solve's own for that origin subzone, and the station pairs' shares to its station-pair
matrix, within 1e-6 (f32 summation order) on the synthetic city, free flow and crowded
(`pair_choice_reproduces_mode_choice`), and within 6e-6 for the three origins of
`trip_real_network` on the real New York network; the WASM worker's answers for those three
trips equal the native ones to 1.7e-7 relative (rail and drive chances identical). On the real
save in Chrome a trip is answered in 3-14 ms. Park Slope to Midtown in the morning peak: train
91%, drive 9% (the drive is 36 minutes plus parking that counts as 43); the busiest route is the
F from 7 Av to 47-50 Sts-Rockefeller Ctr with 17% of the trip's train riders, the rest spread over
neighbouring stations at both ends and the 42 St stations. Jersey City (Journal Square) to Wall
St: train 62% (PATH to World Trade Center, then the 1 or a walk). Mineola to Hicksville: drive
63%, train 37%. Everyday purposes choose the same way on the same rail and road costs but with
the walk share and the drive's distance from the sampled cells, and their own subzone averages;
the explorer shows the commute.

**The pool and the bubbles' chain** (T-119). The pool is half the logical cores less one, at most
5, at least 1 workers that solve, and one more that holds no period ([architecture.md](architecture.md) 7). Periods go to workers
by index (worker i of n: periods q with q mod n = i), so five workers solve a stage of all five periods at once, unless
that puts both periods of a demand level on one worker (2 workers: both peaks), when they are dealt
out level by level (high, then medium, then low: 0, 1, 4 and 2, 3), so an edit to one level's trains
has its two periods solved side by side (T-124; `periodsOfWorker`, `?perfOff=spread` by index alone). The bubbles ask for the
commuters by mode right after a round is published, when every worker is already part way into
the next round and cannot answer a question about the round before from its live state. So each
worker sends its subzone sums for a round (`sub_sums`, per city) unasked, from the moment all its
periods had that round and ahead of its last result for it; the client adds them up when the
round is published. The commuters by mode per cell (`cell_modes`) read only those sums and the
network's subzones, so the worker with no periods answers them whatever round the others are in.
Stations, selections, transfers, trips and longer-trip queries read a period's state, so a worker
answers them only at a round all its periods hold; a selection's far end spread (`flows`) reads
only the sums it is handed and answers at any round. While the running solve's estimates are
shown (startup, a zone just opened), a cached result of the same network is, or nothing is shown
yet, the client asks about the round after the published one (round 0 before any), marked `wait`:
a worker short of that round keeps the question and answers it the moment it reaches the round,
just before that round's last result, so the client holds every answer when it publishes the
round. Such a solve is sent with `barrier`: a worker starts a crowding round only on `go`, which
the client sends to every solving worker as it publishes the round before, so a worker with fewer
periods (the default pool splits five periods 2, 2, 1) waits at the boundary holding its round
instead of running ahead past the round asked about (rounds counted per group of clusters, T-402:
"Clusters" below). That costs the final about 0.7 s and changes
no figure; solves after an edit run without it. A click then waits at most one stage, and its
figures are at most one crowding round ahead of the bubbles; the answer for the next round is
asked for at once, so each publish finds its answer waiting, and the final replaces it as always.
A worker already past the round asked about answers nothing and the question goes to the newer
round. After an edit questions about a zone being solved wait for its final (the workers hold a
network the map does not show yet); others are answered at once ("Clusters" below). `?perfOff=ahead` turns all of this off. T-327, notes/T-327.md.

**Clusters** (T-326, `workers/demandClusters.ts`). Each worker splits the network into clusters
that share no rider and solves each on its own sub-network: the cluster's stations (in snapshot
order) and its lines, with their own route nodes, mapped back to the whole network's in every
result and answer. A cluster is joined up from populations, stations and lines: a line joins its
stations; a station complex joins its stations; stations within 1 km join (merged platforms are
100 m, walking transfers 300 m); a city joins every station within 30 km of its cells' bounding box
(the 4 km walk and the 15 km ride from a home group's centre); the regional market, when open,
joins each of its own 6 km zones to every station within 30 km of the box of that zone's sites
(its rides reach about 17 km; `MARKET_REACH_M`, `IntercityApi.zone_boxes`), one node per zone, so
every station in a zone's access lists is in that zone's cluster, and two networks meet through
the market only where one 6 km zone reaches both (T-409). Clusters are
worked out afresh for every network, so a station or a population that reaches two of them merges
them. Each cluster the market reaches holds a part of the market of its own (`IntercityApi.beside`,
sharing the composition's sites and zones, kept by the cluster's market zones' world ids; a part
keeps the composition it was made from while its key lasts, so a pack opened elsewhere re-solves
nothing), set on the cluster's network alone: its pairs are those between zones its own stations serve, and a pair whose ends lie
in two clusters is in neither, its trips fixed in the pack's totals like any other pair with no
rail path. The market's figures are its parts' added up: rail from each, the trips, pairs and
corridor trips (which need no network) once. A market zone no station reaches is in no cluster;
with none reached the market is a cluster of its own with no lines, solved once and kept. Lines and
stations no population reaches are in no cluster and carry nobody. `?perfOff=split` joins the
market's zones into one node, as before T-409.
Every population of a cluster crowds on the cluster's own loads, which are the whole network's
there, since no other population puts a rider on its lines; so the figures are the same as one
solve of everything. A worker keeps what each population last solved, and a cluster whose
sub-network, populations and rounds are unchanged since keeps its result untouched: after an edit
only the clusters it touched are solved, and the others' figures are read from their populations
as they stand. Every worker splits the same way from the same inputs. `?perfOff=clusters` solves
everything as one cluster, as before T-326.

Within a cluster that is solved, a period whose own inputs are unchanged keeps its result too
(T-124). Periods share no rider and each reads only its own demand level's run times and trains an
hour (the trips home in a period go back by that period's own station pairs; the day's money, the
purposes and the weekend are worked out from the five periods' results afterwards), so a change to
one level's trains leaves the other levels' periods exactly as a fresh solve would give them. The
populations decide it themselves (`set_network`, `Service::same_periods`): with stations, lines, stops,
cars and complexes unchanged they keep station access and subzones, and every period whose level's
run times and trains an hour are unchanged; `set_fares` with the same fare keeps them as well. The
worker keeps a period only when every population of the cluster kept it and each holds it from a whole
solve of the cluster's period with the same inputs (`psig`, per period like `sig`), since the crowding
is the cluster's; the other periods are solved from free flow. A train-count change at one level
re-solves two periods (one at the low level), a new station or a fare change all five.
`?perfOff=periods` re-solves every period of a cluster solved.

After an edit (a solve without the round barrier) the clusters to solve go one after another, the
zone in view's first (T-393), each to its last round; each but the last is sent as soon as it is
done, as one result per period with that cluster's figures alone (`result.cluster`), and the page
puts its lines, their stations, its zones' views and, if the market is in it, the regional market's
into the shown result at once. The network-wide totals wait for the solve's final: once every
cluster is done, the solve's result for every round is the final figures, read once (after an
edit only the final is shown, and new lines take round 0's, T-324). A cluster's network
is set up only when its turn comes, so until then it still holds the shown result's. A worker keeps
the clusters of its last six networks, and answers a question about any of them at once while the
asked zone's cluster holds that network solved to its last round, whatever it is solving
meanwhile: a click in a zone whose cluster is kept, or already published, needs no wait. Questions
about a published cluster's zones go to its solve. The status tag and the top bar's "before your
last change" speak only for a zone still being solved (`plan`, `demandSolving`). With the barrier
(startup, a zone just opened) the clusters to solve go in two groups when there are two or more and
the zone in view's is one of them (T-402): that cluster alone first, through every round, each
round sent as its own cluster result and shown where the solve's estimates are (a view of the
cluster alone when nothing else is shown, else only its final, put in as after an edit); then the
other clusters together, round by round, whose results are the solve's, with the first group's
figures read from its populations. Each group counts its own rounds: `go` names the group
(`plan.ahead` tells the page which cluster is first), and a worker tells which group a question
means by its zone, so a question about a zone in the second group waits for that group's round
and one about the first group's zone after its final is answered at once. Otherwise all clusters
step through the rounds together. The worker adds
cities up in the registry's order, never the order their packs opened in, so a network's figures
are the same to the bit however it was loaded. T-392, T-393, notes/T-326.md.

**Cost on the real combined network** (T-119: the 109-line New York save plus one Philadelphia
line, 875 demand stations, 5,039 subzones, 13.9M subzone pairs a period; native, one period, free
flow or one round, three populations): 1.75 s before. Strategy searches took 0.28 s for each of
the three populations (Philadelphia's has 35 subzones but searched towards all 875 stations); New
York's mode choice 0.81 s; the market's pair loop 0.06 s. Now 1.25 s: one search (0.29 s), the
mode choice, and Philadelphia's 2 destinations. Results are identical to floating point noise
(max difference 2e-16 of each array's largest value over line loads, boardings, entries and the
summaries of all five periods). Since T-172 (fixing T-121) the market's free-flow estimate also
routes with the distance fare, as the local populations' does, so round 0 shares their search
too; before, its own round-0 search towards every station a far zone can reach took 1.8 s natively
a period on the 109-line save (against 0.1 s for a crowding round), and in five Chrome workers
at once made round 0 four times as long as a crowding round (10 s against 2.3 s).

In Chrome on this machine (16 threads), the real network plus the Philadelphia line:

| | before (3 workers) | 3 workers | 5 workers |
|---|---|---|---|
| free-flow estimate published | 4.5 s | 3.2 s | 1.7 s |
| round three published | 17.2 s | 11.0 s | 6.1 s |
| bubbles drawn | 17.4 s (after round three) | 3.4 s | 1.9 s |
| WASM memory per worker | 129-132 MB | 120-134 MB | 120 MB |

Each period's stage in a worker takes about 1.5 s (2.1-2.3 s before). Historical single-city
measurements above retain their context.

**Memory follow-up** (T-123). The regional JSON pack now parses into flat numeric rows, skipping
metadata; temporary rows are consumed before constructing pairs. Immutable pair and lookup
vectors release their unused capacity. This preserves all trip parameters and results. On the
current full-network fixture, opening the five solving workers plus the auxiliary uses
360 MB against 596 MB before; each solving worker after the solve uses 127–128 MB against
137 MB. Estimated pool memory falls by 46 MB (6.4%) to about 668 MB, including the auxiliary's
opening allocation. These are WASM linear-memory measurements, not browser RSS. Live network,
strategy and period arrays still set the full-network peak. Comparison: notes/T-123.md.
