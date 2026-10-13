# Intercity demand: how it works, how realistic it is, and options

Research report for T-268, 2026-10-13. No code changed. Working tables and how each number was
made: [notes/T-268.md](../notes/T-268.md). The choices in section 5 are Anita's.

## 1. How it works today

The game has two demand layers that never overlap.

**Local layer (each city pack on its own).** Commutes come from a work gravity inside the pack
(d^-0.8 exp(-d/30 km) between 2 km zones, every zone's jobs filled from the pack's own homes,
cross-state pairs times 0.23, Washington 0.45). It has no distance cut, but it stops at the pack's
edge: nobody commutes from one pack to another. Everyday outings (shopping, social, school,
university, airport, medical) also stay inside the pack, and they taper away between 40 and
80 km: at 40 km they are all local, at 80 km none are.

**Regional market (`sim/src/demand/intercity.rs`, built by `pipeline/intercity.py`).** One market
for all nine packs plus Mercer County as a town:

- **Trips made**: every resident makes 0.015 trips a day (one way). With 46.6M people that is
  about 690,000 trips a day, both directions together. One purpose, no business or leisure split.
- **Where they go**: each 6 km zone shares its trips among every other represented zone in
  proportion to `population + 2 x jobs` times a distance curve `exp(-d/180 km) / d^1.2`. The
  curve is multiplied by an onset that is 0 at 40 km and 1 at 80 km, the mirror of the local
  taper, so a trip of 60 km is half local outing, half market trip. Each zone's trips are shared
  only among the represented places: the zone's whole budget always lands somewhere in the nine
  packs and Mercer. The two directions of a pair are averaged, so A to B equals B to A.
- **Rail or car**: a logit at 0.025 per perceived minute between the best rail strategy and the
  car. Car time is 15 min plus the straight-line distance x 1.25 at 85 km/h (68 km/h effective).
  The car pays nothing else: no tolls, parking, fuel or traffic. Rail pays the walk to the station
  (fast-walk catchment only, no driving to the station), the wait, the ride, transfers, the
  player's fare (the same fare table as local trips) and crowding.
- **Timing**: 20/35/25/15/5% over the five periods, flatter than commutes. Weekends carry 1.58
  times a weekday's longer trips (T-260, from NHTS 2017).
- **Not modelled**: commuting between packs; air and coach; purposes; tolls, parking, running
  costs and congestion on the car side; income; induced travel (a better railway does not create
  trips, it only takes them from the car); park-and-ride; anything outside the nine packs and
  Mercer.

The commuting gap is the big one. The NJ Transit Northeast Corridor line from Trenton, Metro-North
from New Haven, MARC between Baltimore and Washington and the MBTA from Providence are mostly
commuter markets crossing a pack edge. The game has none of those commutes: it shows only the
leisure and business trips of the regional market on those corridors.

## 2. How realistic is it

### Total size

| Measure | Model | Real |
|---|---|---|
| Long trips a resident a day | 0.015 (40 km and up, damped below 80 km) | about 0.08 legs of 50 miles or more a day (NHTS 2017: 2.5% of all trips) |
| Trips a day between places in the corridor | 447,000 between places, 690,000 with long trips inside a metro | about 1.16 million (NEC FUTURE 2012 model: 425M intercity trips a year, no auto commuting) |
| Share by car | logit, depends on the network | about 81% of corridor intercity trips, rail 12% (NEC Intercity Travel Study via New Geography) |

The total is the right order but low, about 0.4-0.6 of the NEC FUTURE corridor figure. The real
long-trip rate is much higher than 0.015, but most of those trips go to places outside the nine
packs (the rest of the country, by car and air), so 0.015 for the represented places is not
crazy.

I-95 traffic counts were not usable as an all-mode check: no clean segment count turned up in
this pass, and every count on I-95 or the Turnpike is mostly local and commuting traffic.
NEC FUTURE's model figures are the better all-mode figure (the NEC Commission's related
Intercity Travel Study took its car trips from E-ZPass records along I-95).

### Pairs

Model all-mode trips are the market's own (whole pack to whole pack). "Model rail" is not a model
output: rail depends on the player's network, and the only real save (T-007) has New York's lines
alone. It is the trips with both ends within 20 km of the centres, times the rail share the
game's logit gives one centre-to-centre trip on a train like today's best regular service. It is
the right order for what a player who rebuilt the corridor's main line would see.

Real all-mode figures are NEC FUTURE's 2012 metro-to-metro trips (Figure 5-1, read off a chart,
non-commuting). Real rail figures are estimates from NEC FUTURE and ridership counts; pair-level
Amtrak counts are not published. All per day, both directions.

| Pair | Model all modes | Real all modes | Judgement | Model rail (rough) | Real rail (estimate) | Judgement |
|---|---:|---:|---|---:|---:|---|
| New York - Philadelphia | 131,000 | ~57,000 | 2.3x high | ~9,300 | 4,000-6,000 | right order, high |
| New York - Boston | 31,000 | ~50,000 | 0.6x | ~400 | ~3,000 | 8x low |
| New York - Washington | 22,500 | ~20,000 | right | ~1,400 | ~4,000 | 3x low |
| Philadelphia - Washington | 27,000 | not published | (plausible) | ~1,600 | 2,000-3,000 | right order |
| New York - New Haven | 23,700 | not published | (plausible for non-commute) | ~1,200 | 3,000-6,000, mostly commuters | 3-5x low |
| New Haven - Hartford | 990 | not published; 129k jobs cross this pack line | far too low | ~120 | 1,000-2,000 (Hartford Line about 2,000 a weekday) | 10-20x low |
| Boston - Providence | 27,700 | not published; 222k jobs cross this pack line | low once commutes count | ~2,200 | 6,000-10,000, mostly commuters | 3-4x low |
| New York - Trenton (Mercer) | 14,000 | not published; 175k jobs cross this line | low once commutes count | ~2,200 | 15,000-25,000, mostly commuters (NJT Trenton, Hamilton, Princeton Jct) | ~10x low |
| Washington - Baltimore (extra) | 51,000 | ~6,300 non-commute; 414k jobs cross this line | 8x high non-commute, commutes missing | | | |
| Washington - Boston (extra) | 500 | ~4,900 (76% by air) | 10x low | | | |

Readings:

- **Between big cities 130-330 km apart the market is the right order**, New York - Washington
  almost exactly, but it piles trips onto the nearest big neighbour (New York - Philadelphia 2.3x)
  and starves longer pairs (New York - Boston 0.6x, Washington - Boston 0.1x).
- **Rail is too weak on long pairs** (New York - Boston and New York - Washington 3-8x low),
  because the car is free beyond its time and four hours of driving into Manhattan costs nothing.
- **Everything that is really commuting is missing**, which makes the four commuter corridors
  3-20x low on rail. T-214 (New Haven - Hartford about 1k trips) is this: 990 all-mode trips a
  day against 129,000 LODES jobs held across that pack line.
- **Drive times are about right** on average (New York - Philadelphia 133 min, Google-style
  typical 1h50-2h15), but there is no congestion near centres and no car cost at all.

## 3. What's wrong and why, most important first

1. **Commuting stops at the pack edge.** Each pack fills its jobs from its own homes, and the
   market is non-commute by design, so a commute from New Haven to Stamford, Trenton to Newark,
   Pawtucket to Boston or Columbia to Silver Spring does not exist. LODES 2023 has about 1.8
   million jobs held across our pack lines (notes/T-268.md), most of them short commutes between
   neighbouring packs. This is the biggest real rail market in the corridor and the one a player
   is most likely to build for. It also makes the game depend on where the pack lines were drawn.
2. **The market is a closed world.** Each zone's 0.015 trips a day must land inside the
   represented places. New Haven has nowhere to send them but New York (its own Hartford is too
   close for the onset); Baltimore sends most of its to Washington. Real people also go to
   Albany, the Poconos, Florida and everywhere else. This is why New York - Philadelphia is 2.3x
   and Washington - Baltimore 8x high, and why adding a city later will move other cities'
   numbers.
3. **The distance curve.** It falls too fast beyond about 200 km (exp(-d/180) is 3% at 630 km),
   and the 40-80 km onset removes most trips between neighbouring cities 50-70 km apart (New
   Haven - Hartford 56 km keeps 40%, Boston - Providence 65 km keeps 60%). Real long-trip counts
   fall more slowly: NEC FUTURE's Washington - Boston, 630 km apart, has about a quarter of
   New York - Washington's trips, where the model gives it 2%.
4. **The car costs only time.** No tolls, parking, fuel or traffic, while rail pays a fare. The
   local layer is fairer (driving weighted 1.33, parking up to 40 min by job density), so the same
   trip into Manhattan is judged by two different car models depending on its length. Rail into
   dense centres is under-rewarded; this is most of why long-pair rail is low.
5. **No driving to the station.** Only fast-walk access, so suburban zones get no rail at all.
   Real intercity and commuter rail leans on park-and-ride (Metropark, Route 128, BWI, Princeton
   Junction, New Carrollton). The spec already plans it.
6. **No air or coach.** Small today, since all-mode totals already include those travellers and
   only car competes. It matters once a player builds fast rail over 300 km: rail would then win
   trips from a car that real travellers would have flown.
7. **One kind of traveller.** No business vs leisure split, no income. Business travellers value
   time more and travel on weekdays; leisure trips are weekend-heavy and price-sensitive. NHTS shows
   long-trip rates rising strongly with income. Mostly flavour for the game.
8. **No induced travel.** A great new line moves trips off the road but never creates any. Real
   corridors grow when service improves; NEC FUTURE's alternatives add 13% total trips. For game
   feel, "build it and they come" is satisfying and currently missing for intercity.
9. **The trip rate (0.015) is a judgement.** Hidden today by problem 2; it needs a fresh value
   once the world is opened.

## 4. Options

All options keep the market's structure (zones, served pairs only, the offline derived block), so
performance stays where it is. None uses city-specific data.

### Option 1. Retune the market in place

**What**: change the formula, not the design.

- An **outside world**: each zone's normaliser gets an extra term for everything not represented,
  so a share of each zone's trips leaves the map. The term is the population at each distance from
  the zone (a national or global population raster, GHSL), so a city near the map's edge sends
  more away than one in the middle.
- A **flatter long tail** and an **earlier onset** (for instance 25-50 km), fitted to NHTS's
  distribution of long trips by distance.
- A **trip rate** from NHTS (US) for trips of 50 km and up, applied to the open world.
- **Car costs**: running cost per km plus a parking cost by destination job density, the same rule
  as the local layer, split over about two people a car for long trips; tolls as a cost per km on
  limited-access roads where a country has them, or left out.

**Data**: NHTS 2017/2022 (national, open), GHSL population (global). Japan has its national
Inter-Regional Travel Survey, Europe national travel surveys, to refit when a country is added.
**Cost**: small. A new FORMULA, `intercity.py --derive`, a day or two with checks against the
table above. **Game feel**: same as now with better numbers. New York - Philadelphia falls,
New York - Boston and long-pair rail rise. Commuter corridors stay wrong.

### Option 2. Option 1 plus commutes between packs

**What**: one commute gravity over all packs and towns together, with the same curve and border
rule as inside a pack, solved offline. The part of each commute that stays inside a pack stays in
the local layer, as now; the part that crosses a pack line becomes a second purpose of the
regional layer, "commute", with commute timing (peaks), a weekend factor of 0.30 and the local
layer's commute costs (driving weighted 1.33, parking by job density). Towns like Mercer then have
commuters too.

**Data**: what the packs already use (census population, LODES or other jobs locations; abroad,
any jobs layer the packs have). LODES OD and ACS county-to-county flows as checks, not inputs.
**Cost**: medium to large. The pack gravity is solved per pack today; this needs a joint offline
solve over the union of zones (about 46M people, pairs cut at 100-150 km) and a pack format change
so each pack knows how many of each zone's homes and jobs it keeps. The regional layer gains a
purpose; its pairs grow near pack seams, which the T-174 design handles. Every pack's commute
numbers move a little near its edges, so the usual demand checks are needed.
**Game feel**: big. Commuter rail into New York from New Haven and Trenton, Providence to Boston
and Baltimore to Washington becomes a real business with morning and evening peaks. Pack lines
stop mattering to commuters. T-214 is fixed at the root.

### Option 3. Option 2 plus a richer market

**What**: on top of Option 2, the market splits into business, leisure and visiting friends,
each with its own attraction (jobs, jobs and population, population), value of time, timing and
weekend factor (business weekday-heavy, leisure weekend-heavy); coach and air join car and rail in
the logit (coach on the road with a low fare and a time penalty; air between airports the packs
already have, over about 300 km, with a fixed airport time); and a little induced travel, total
trips rising with how good the best mode is (a logsum term), so a great line grows its market.
Income could scale trip rates by area.

**Data**: NHTS for purpose shares and rates by purpose; airport list is in the packs (T-142);
ACS income by tract, gridded GDP abroad. **Cost**: large, mostly design and tuning. **Game feel**:
the richest: weekend leisure trains, business trains at peak, a reason to build fast rail against
flying, and growth after building. Also the most numbers to explain in the panels.

### Option 4. Measured flows as the base (not recommended)

**What**: take a published all-mode OD table and spread it over zones instead of a gravity.
The candidate is FHWA's 2020 NextGen NHTS OD data (583 zones nationwide, all modes, from phone
location data). **Why not**: US only, a COVID year, zones far coarser than our places, and its
download terms are unclear (it is published through the nhts.ornl.gov viewer). Worth using as a
check for Option 1's curve if it can be downloaded, never as the model.

## 5. Recommendation (the choices are Anita's)

**Option 2**, then the car-cost and air parts of Option 3 when fast rail over 300 km matters.
Reasons: the largest error is the missing commutes, they are where players build first, and the
fix also removes the dependence on where pack lines fall. Option 1 alone makes the numbers
plausible but leaves the commuter corridors 3-20x low.

**How the layers should split the work.** By purpose, not by distance or pack:

- Commutes: one gravity everywhere, no distance onset. Inside a pack it stays local; crossing a
  pack line it goes to the regional layer. Long commutes across packs belong in the regional
  layer for routing (it is the only layer that spans packs) but are made by the commute rule, so
  there is no seam at the pack edge.
- Non-commute trips: local outings up to the onset, the market beyond it, as now (perhaps with an
  earlier onset, Option 1).
- Long trips inside one big pack (Long Island to New Jersey, New York - New York, 184,000 a day)
  stay in the market as now.

**The border factor (overlaps T-171, which is checking whether it is behind PATH's low riders;
not redone here).** Option 2 would carry the factor across pack lines, where it bites hard:
Boston - Providence crosses Massachusetts - Rhode Island, Mercer - Philadelphia crosses New
Jersey - Pennsylvania, Baltimore - Washington crosses Maryland - DC (New Haven - Hartford and
Mercer - New York stay inside one state). The fitted 0.23 is real in
LODES (T-162), but it probably stands for things the gravity cannot see: the Hudson and the
Delaware are state lines with few, tolled crossings, and a straight-line distance ignores them.
A gravity on road travel time instead of straight distance might explain much of the drop without
a factor (docs/demand-worldwide.md already lists this check). Whatever T-171 concludes should
apply to the joint commute gravity too.

## 6. If a source was blocked

- NEC Intercity Travel Study summary (2015), the source of the 81% car share: the NEC Commission
  link returns 404 and the Wayback Machine is unreachable from here.
  http://nec-commission.com/app/uploads/2018/04/2015-09-14_NEC-Intercity-Travel-Summary-Report_Website.pdf
  It would give pair-level trips for 25 market pairs, the best check there is.
- BTS "America on the Go" long-distance reports return 403:
  https://www.bts.gov/archive/publications/america_on_the_go/us_business_travel/entire

## Sources

- NEC FUTURE Tier 1 Final EIS, Volume 2, chapter 5 (tables 5-1, 5-8, 5-9, 5-10, 5-15):
  https://www.fra.dot.gov/necfuture/pdfs/feis/volume_2/c05.pdf and the web version
  https://www.fra.dot.gov/necfuture/tier1_eis/feis/volume_2/chapter_05.aspx
- NEC FUTURE Tier 1 Draft EIS chapter 5, Figure 5-1 (trips by mode between metro pairs, 2012):
  https://www.fra.dot.gov/necfuture/pdfs/tier1_deis/c05.pdf
- Wendell Cox, "Cars, Not Trains or Planes Dominate Northeast Corridor Travel" (summarising the NEC
  Intercity Travel Study): https://www.newgeography.com/content/006407-cars-not-trains-or-planes-dominate-northeast-corridor-travel
- Rail Passengers Association, Amtrak fact sheet: Northeast Regional, 2024:
  https://www.railpassengers.org/site/assets/files/3453/5.pdf
- Hartford Line ridership: https://en.wikipedia.org/wiki/Hartford_Line and CRCOG's overview
  https://crcogct.gov/wp-content/uploads/2026/06/04e.-Hartford-Line-Passenger-Rail-Overview-2026-06.pdf
- NJ Transit station boardings: https://en.wikipedia.org/wiki/Trenton_Transit_Center and the FY2018
  NJT weekday chart copy https://www.myleszhang.org/files/2019/03/Weekday-Ridership-Statistics-21z2n4i-1h26kh2.pdf
- MBTA Providence/Stoughton Line: https://en.wikipedia.org/wiki/Providence/Stoughton_Line and
  https://ecori.org/providence-rail-hub-thriving-but-future-uncertain/
- Long trips as 2.5% of person trips (NHTS 2017):
  https://caee.utexas.edu/prof/kockelman/public_html/priyankatrb2026pt1.pdf
- BTS long-distance travel (2001 NHTS: 2.6 billion trips of 50+ miles a year, about 89% by car,
  16% business): https://www.bts.gov/archive/publications/america_on_the_go/long_distance_transportation_patterns/entire
- FHWA 2020 NextGen NHTS OD data: https://www.fhwa.dot.gov/policyinformation/nextgen-nhts-origin-dest-data-2020.cfm
- Japan, Inter-Regional Travel Survey (MLIT): https://www.mlit.go.jp/report/press/sogo23_hh_000184.html
- LODES 2023 OD (in `data/raw/`): https://lehd.ces.census.gov/data/
