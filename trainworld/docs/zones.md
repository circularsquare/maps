[Back to the spec](../SPEC.md)

# Zones (T-276)

Written 2026-10-13. Anita's decisions behind it: option 2 of `research/intercity-review.md`
(retune the regional market and add commutes between cities) inside a world divided into zones;
Mercer County joins New York. Which parts are built is in section 9.

## 1. What a zone is

The world is divided into **zones**, and every place is in exactly one zone or in dead space.

- A **city zone** is a metro area with a hand-drawn extent: the nine Northeast city packs and
  Toronto (T-277).
- A **non-city zone** is land between and around city zones that is not part of any metro: the
  country between Providence, Hartford, Springfield and Boston, say.
- **Dead space** is everything else: Nunavut, and for now most of the world. Nothing there makes or
  attracts a trip in any layer, so a line through it carries only riders going through.

Both kinds of zone are the same thing in the code: a **pack** in the registry
(`app/src/game/registry.json`, its `cities` list), built by the same pipeline from the same
national sources, with the same cells (H3 resolution 9), the same local demand layer and the same
bubbles. They differ only in how their extent is chosen and a `kind` in the registry (`"city"`, the
default, or `"zone"`). So everything a city has, a non-city zone has too: it opens when it is viewed
or reached by a station, has a city inspector, a boundary line, a place list, its own bubbles, and a
place in the regional market. Anita: "so these are basically cities i guess. theyre like. non urban
areas."

Why one kind of pack rather than a coarse second model: everything that works for cities (bubbles,
selection, the trip explorer, loading, reach, saves) works for non-city zones with no new code
paths, and a zone can become a city zone later by redrawing extents, not by switching model. The
cost of fine cells in thin country is small: cells exist only where people or jobs are.

## 2. How non-city zones are built

**Unit.** A non-city zone is one **county-sized statistical unit**: a US county, a Canadian census
division, a NUTS-3 region in Europe; elsewhere the country's second-level unit of 100k-1M people.
One unit per zone, so zone lines are lines that every national dataset already uses; a unit of
under 100,000 people joins the neighbouring non-city zone it shares the longest border with. A
unit belongs to a city zone (and so is not a non-city zone) when that city's pack lists it.

**Boundary file.** A zone's drawn boundary is its units dissolved. In the US the units are TIGER/Line
2020 counties, which include the water inside each county out to the state and county lines at sea
(T-352), so an outline and the outside shade run straight offshore instead of hugging the coast.
Neighbouring zones share exact simplified borders, so their boundaries do not overlap.

**Which units are modelled (the extent).** A unit not in any city zone is modelled when it
**touches two different city zones**: it lies between them. Today that is Worcester MA, Windham CT,
New London CT and Litchfield CT in New England. Modelled space stays between cities for now (Anita,
2026-10-14, T-300): no ring of units round each city (in the Northeast that ring would be about 30
more counties and 9 million people). Everything not modelled is dead space. **Japan is modelled
whole** (research/japan.md decision 2): every prefecture outside the metro packs is a zone, and in
Hokkaido each subprefecture is a unit under the 100,000 rule (Hiyama joins Oshima; Rumoi and Soya
join Kamikawa; Hidaka joins Tokachi; Nemuro joins Okhotsk; Shiribeshi less Otaru joins Iburi), 38
zones from `pipeline/japan_zones.json` (T-420 stage 3). Their outlines leave out the
Russian-administered southern Kurils and uninhabited islets more than 50 km from other land.

**Mercer County (Trenton) is part of New York's city zone**, not a non-city zone (Anita leaned
this way). Reasons: it is in the New York combined statistical area; LODES holds 175,000 jobs
between Mercer and New York's counties against 88,000 with Philadelphia's; and as a zone of
387,000 people pinched between two big packs, most of its commuting would cross a zone line. A
county between two city zones whose commuting runs mostly to one of them may join it this way; the
choice is made per county when a city's extent is drawn, as extents always have been (pragmatic,
no study). Its place in the market (id 2, "Trenton") stays in the registry with no sites, because
market ids are never renumbered.

**Data.** The same national sources as the country's city zones, at the same grain: in the US 2020
census blocks less institutions (T-234), LODES WAC and RAC, NCES, IPEDS, FAA, OSM. Where a country
has no small-area jobs (tier 0, docs/demand-sources.md 4.1), population from Kontur (H3 resolution 8,
global, each hexagon spread over its seven resolution 9 children) or the national census, and jobs
from the GHSL non-residential built-up layer (T-012). A country's non-city zones are built by that
country's city adapter (Toronto's for Canada, T-277) with `kind: "zone"`.

**Names.** A non-city zone is named after its unit ("Worcester County" in the US, the census
division's name in Canada); a merged zone after its most populous unit.

## 3. How the demand layers split the work

By purpose and by zone, with no seam a commuter can see:

| Trips | Layer | Notes |
|---|---|---|
| Commutes with both ends in one zone | the zone's local layer | the joint commute gravity's within-zone part (section 4) |
| Commutes crossing a zone line | the regional layer, purpose "commute" | the same gravity's cross-zone part, at the market's 6 km zones; commute timing, weekday-heavy |
| Everyday outings, school, university, airport, medical | the zone's local layer, inside the zone | as now; outings crossing a zone line under the market's onset are not modelled (short trips near the line: small) |
| Longer trips (visits, business, leisure) | the regional layer, purpose "longer trips" | the retuned market (section 5), within and between zones, a share leaving the map |
| Anything starting or ending in dead space | none | except the market's trips that leave the map, counted as leaving and not routed |

A non-city zone's local layer is the same as a city's: commutes and everyday purposes inside it,
"local-ish" only in that its trips are longer and its cells thinner.

## 4. Commutes between zones

**One commute gravity for every zone together.** The local gravity (doubly constrained, 2 km zones,
f(d) = d^-0.8 exp(-d / 30 km)) is solved offline once over every zone pack's gravity zones at once,
by a native bin (`joint_gravity`) with the kernel's own code. Every home's commuters are shared
over every job in reach, in its own zone or another, so moving a zone line moves no commuter.

- **Within one pack** a pair uses the pack's own decay table and border factor, exactly as the pack
  does at runtime.
- **Across packs** a pair uses the great-circle distance between the two gravity zones' centres,
  cut at 150 km (beyond it the decay is under a thousandth of a 10 km pair's, and LODES's long
  links are mostly head-office addresses, notes/T-268.md), and a **border factor per region pair**:
  the factor of the pack that holds both regions (New Jersey and Pennsylvania take Philadelphia's,
  Maryland and DC Washington's 0.45), else the country's default (US states 0.23). T-171 found the
  factor should stay as fitted per pack rather than become one global value; this keeps every
  pack's own fit for the pairs it was fitted on. National borders take a factor per country pair
  (US-Canada 0.05 for now, judgement from Windsor-Detroit's few thousand cross-border commuters
  against a gravity that would send several times more; fitted when a zone pair across the border
  is built).
- The solve gives each pack new balancing factors (`row`, `col` in its gravity block) and the
  within-pack trip totals of each gravity zone (`home_trips`, `work_trips`), so the pack's runtime
  rebuilds exactly the within-zone part. A pack's cells count commuters from those totals rather
  than from the pack's total trips over its people (which assumed every commute stays in the pack).
- The cross-pack part is spread from gravity zones to the market's 6 km zones by each gravity zone's
  homes (at the home end) and jobs (at the work end) in each, and shipped in the market pack as a
  list of home zone, work zone and commuters a day.
- **The balance** (T-297) is solved per group of packs joined by any pair within the cut (groups
  share no commuter). Each group runs Furness steps with Anderson mixing of the last 12 steps, until
  every zone's jobs are met to 1e-8 relative (its homes exactly), about 40 steps. The tolerance is
  tight because weakly joined packs and regions leave slow modes: a looser residual leaves factors
  percents off (notes/T-297.md).
- **Groups form themselves** (T-539, `intercity.py --group auto [ids]`): two registered packs are in
  one group when their outlines (`<id>.boundary.geojson`) come within 50 km of each other, chained,
  across national borders too; a pack with no neighbour that close stands alone and is not solved.
  Within a group the 150 km cut still applies. Today this gives two groups. The Northeast (the
  registry's cities with a `market` id; cross-pack commutes through `data/packs/intercity.json`) is
  solved its own way and must hold exactly those cities. Every other group runs `joint_gravity
  --packs` (distances in a conic at the packs' mean origin; at most 150 steps, the best kept) and
  writes its cross-pack commutes, by 6 km square of each pack's own frame, to
  `data/work/<first member>.joint_commutes.json` (Japan's 45 packs: `tyo`), which `market_zones.py`
  writes into each zone's market pack; an older file holding any of the same packs is removed.
  Japan's balance stops at about 8e-4, all of it in Okinawa's outlying islands, a block joined to
  little else (notes/T-463.md). A tier 0 pack's single region is its own, and its header names its
  country. A group's solve grows with its zone pairs (within packs plus across within the cut):
  each Furness step sums every pair twice, about 100M pairs a second a thread, so Japan's 0.5
  billion pairs take about 25 minutes on one thread and the Northeast's 0.2 billion 4 minutes.
  `intercity.py` refuses a group over 200,000 zones (about three Japans); with outlines under the
  gate's 40,000 km² no group in the world plan comes near it (notes/T-539.md).

**Fitting and checking.** The curve and the factors are each pack's fitted values, so nothing new
is fitted for commutes between zones in the US. LODES OD 2023 is the check, not an input: commuters
between each pair of zones against LODES's jobs between the same counties (the table in
notes/T-268.md: Baltimore - Washington 414k, Boston - Providence 222k, New Haven - New York 134k,
Hartford - New Haven 129k ...), aiming for the right order, not a fit. Other countries fit the
curve and their border factors on their census commuting flows (Canada's census commuting flows by
census subdivision, England and Wales WU01EW, Japan's census commuting tables) when their first
zone pair is built.

**At runtime** (the regional layer, `intercity.rs`). Commute pairs are built only between zones a
station serves, as the market's pairs are. Each period carries C(h, w) x the share of trips to work
leaving then, from h to w, and C(h, w) x the share of trips home, on the same station pairs
reversed: the local layer's 70/12/6/4/8 and 2/18/58/16/6. Mode choice is the local commute's: a
logit at 0.10 per perceived minute, driving weighted 1.33 plus parking at the work end (5 minutes
plus the work zone's parking, the local rule's mean over drivers, `site_park` job-weighted over the 6 km zone), against
the best rail strategy with its walks, waits, transfers and fares, with the local commute's rail
constant: `asc_rail` 0.4 plus the home zone's country's (Japan +0.5; demand-mode.md 4.3; T-509). The
longer trips keep a constant of 0 until they are fitted on corridor counts. The weekend has 0.30 of a
weekday's commutes with the weekend's work timing (days.rs `WORK`). Their loads join the
work purpose's (T-158), so the purpose filter's "Work" shows them.

## 5. The world market (longer trips)

Every zone is a place of one world market (T-377; the rule in [regional.md](regional.md) 5.1):

- **Normalisers over the world field** (T-375), never over the zone list: a share of every zone's
  trips goes to places with no zone, and adding a zone moves only the trips that went to where it
  now is. Each zone's own pack carries its 6 km zones' two normalisers (ground and tail), so a new
  zone needs only its own pack and the registry rows rebuilt (`pipeline/market_zones.py`, seconds).
- **The trip rate, the onset and the curve** from NHTS 2017 for the ground; the long-haul tail,
  income and border factors fitted in T-376.
- **Near and far**: places under 250 km apart (centre to centre) pair 6 km zone to 6 km zone; farther
  ones place to place, spread inside each end.
- **The car pays more than time.** Running cost per km and the local layer's parking at the
  destination, each split over the car's occupants (two on a long trip, NHTS), at the game's value
  of time (3 perceived minutes a dollar). No tolls (T-270); no car between landmasses.
- Everyday outings still taper by the complement of the market's onset, so the two layers meet
  without a gap or a double count.

## 6. On the map

**Over a non-city zone** it is the focused zone, exactly as a city is focused: the zone holding the
view's centre. Its bubbles are drawn as a city's are, and the top bar names it. A centre outside
every boundary still belongs to the nearest zone when it is within 40 km of that zone's boundary
(T-318; about the market's onset). Before the boundaries near it have loaded
the rule is 120 km from a zone's centre, past the biggest metro's far edge.

**Over dead space**, farther than that from every zone (London while it has no pack), the view
belongs to no zone. The top bar shows no name and "–" for the split. Nothing is loaded or solved
for the view (no pack, not even the market), no "Loading data" card or status shows, the clock is
not held, and the focused zone stays where it was. A built station that reaches a zone still loads
that zone. One helper answers it, `placeAt` in `game/zones.ts`, kept in `viewedZone`
(`game/cities.ts`).

**Zones not opened** [T-729]: each game keeps the zones it has opened, in order (the save header's
`opened`). A zone not on the list is loaded for viewing exactly as dead space is above (nothing
fetched, no card, the clock not held), though the top bar still names it with "–" for the split, and
it is drawn as one grey bubble at every zoom (docs/ui-demand-views.md "Cities not opened"). A click
on the bubble adds it and opens it bare, wherever the view is; a built station within reach (4.5 km)
adds it too, so the list only ever stops a zone opening for viewing, never for a solve, and no demand
figure depends on it. Nothing removes a zone from the list. A new game opens its start city only (a
start on the map, none); a save from before T-729 starts with the zone in view as it opens and the
zones its stations reach. `?opened=all` treats every zone as opened and leaves the list alone, for
checks that pan.

**Nearby zones** are drawn too: every opened zone whose boundary is on screen (at most
three, nearest first) shows its own bubbles beside the focused zone's, so a zone line is not a cliff
in the bubbles. A click on a nearby zone's bubble focuses that zone and selects the bubble.

Bubbles need a zone's results: an opened zone on screen is opened (its pack downloaded) like a viewed city.
With no station within reach (T-325) it opens **bare**, in the auxiliary demand worker only: its
bubbles and city figures come from a no-rail pass there (all walking and driving, exactly what a
solve would give it), and no solve covers it or is started for it. A click on its bubble draws the
far end as a solved zone's does (T-336): local trips from the auxiliary worker, longer trips and
commutes between zones from the solving workers, which hold the market. A new game opens the
viewed city bare from the start (T-351); a saved game's viewed city opens in every worker, since
its pack downloads before the network's reach is known (docs/demand-runtime.md "Bare zones").
A station built within reach downloads the pack again and opens it in every worker; the next solve
waits for it and covers it from then on, and its bare figures serve until a shown solve does.
Detail: [demand-runtime.md](demand-runtime.md), "Bare zones".

**Zoomed far out** (8.5 and below) every opened zone is one bubble, as if it were one place: its trips a
day at homes (Homes) or at destinations (Destinations) in every layer, local trips, commutes
between zones and longer trips, coloured by train share, sized against the busiest zone whose
centre is on screen (New York's local trips would otherwise dwarf every other zone). An opened zone not loaded in this session has no rail and its
totals come from a small summary written by the pipeline (`data/packs/zones.json`: each zone's
trips a day by purpose with no rail). A click on one selects the zone and draws one bubble per
other zone for the commutes and longer trips between them, as T-186 does for cities.

**Selection, the inspector and loading.** A non-city zone's city inspector is a city's. Search finds
zones by name. Loading treats a zone as a city: opened when viewed or reached, its files fetched on
demand. **Startup must not fetch every zone's files.** Each registry entry carries a `box` (west,
south, east, north round its boundary and water mask; `pipeline/zone_boxes.py`), so where a zone is
is known without its files (T-302, T-400):

- **Water masks** (clock worker): a zone's mask is fetched when the view comes within 20 km of its box
  (only the zone under the centre when the view is over 400 km wide), and for every zone whose box,
  widened by 2 km, a loaded network's track meets; the network then opens over those masks (read from
  the save's nodes and curve points before the load, so it is derived once; a zone only the drawn
  curves reach still costs a second opening; T-737). A
  preview or an edit that makes track (a route, new PIs, a node moved) is first worked out, and if
  its geometry comes within 2 km of a zone whose mask is not in, it waits for the mask and is worked
  out again, so ground over water is never accepted because a mask was late. A mask that cannot be
  fetched is reported, and the edit is worked out without it.
- **Outlines**: one source and line layer hold the outlines of the zones whose boxes meet the view,
  fetched as the view comes near them at zoom 5 to 10.5 (they fade out at 10.5); a zone's footprint
  and outside shade are built when it is focused or near the view.
- **Place lists**: a first search or the station tool fetches those of the zones whose boxes lie
  within 30 km of the view's centre (else the nearest zone's); a search of three or more letters with
  fewer than five results fetches the next four nearest zones' lists, until it has enough or all are
  in. A station named after a place fetches the lists of the zones within 6 km of it.
- **Boundaries** for the focus rule and the demand pool's reach test (T-404): a place asked about
  (the view's centre, a station, a search hit) fetches the boundaries of the zones whose boxes lie
  within 40 km of it (`ZONE_REACH_M`), and the zones beside the view those whose boxes meet the
  screen; a network's stations fetch those whose boxes lie within 4.5 km of one (`REACH_M`).
  Until all of a place's candidates are in, it is placed by the nearest centre as before any
  boundary loads, so its zone never flips as boundaries arrive one by one. A place no box comes
  within 40 km of is dead space at once, with nothing fetched.

## 7. Other countries and borders

Every rule above is per country: the unit (US county, Canadian census division, NUTS-3), the data
adapters, the border factors. Toronto (T-277) is a city zone of Canada with Canadian census
divisions around it as its non-city zones once modelled space reaches them. The joint commute
gravity spans countries: a pair across the US-Canada border takes the national border factor, and
the market's longer trips across any border T-376's factor (0.26 times the origin's income factor
to the power 1.5). A world with separate clusters of zones (the Northeast, Toronto, London, Tokyo)
needs nothing special: the commute gravity has no pairs between them, the market's longer trips
between them are place to place, and clusters merge only through track (T-362 decision 6).

## 8. Performance

- **Opening.** A non-city zone is a pack the size of a small city (Worcester County 860k people,
  about Providence's), opened only when viewed or reached, like a city: 0.05-0.3 s natively with its
  everyday layer. Commute packs open from shipped factors as now (a few ms).
- **A zone's market pack** is opened only once the zone is viewed or reached, or a station lies
  within 30 km of its box, and closes with the zone (T-396, T-377; demand.md "Which cities are
  open"), so the market a worker holds grows with what is viewed and reached, not with the
  registry. Its pairs are built only between zones a station serves, so a solve's pair loop grows
  with the network's reach; pairs to every other place come from the registry rows (one row per
  zone, a few hundred bytes).
  Cross-zone commute pairs add a second pass over served pairs to the market's per-period work
  (the market pass was 0.06 s natively a period on the real New York network).
- **The joint gravity** is offline only (`joint_gravity`, about a minute natively on three
  threads over the 13 packs' 25.6k zones, about three over 65k); a runtime edit never reruns it.
- **Startup** must not grow with the number of zones: zones add registry entries, and their files
  are fetched on demand (section 6). A mask installed while a big network is loaded reworks the
  whole network once (about 0.2 s on the real New York network). A zone opened only for its bubbles opens bare in the auxiliary worker (T-325):
  0.3-1.5 s, and its everyday layer once rather than in every solving worker.
- **Closing** (T-301; demand-runtime.md "Which cities are open"). A zone nobody views, sees beside the view
  or reaches closes after 60 s, or at once beyond two such zones, so a long pan holds a few zones'
  packs at a time, not every zone it passed.

## 9. What is built (T-276)

All five stages are built; numbers and checks in [notes/T-276.md](../notes/T-276.md).

1. Mercer County in New York's pack (207,766 cells, 21.97M people).
2. The market retuned: 0.107 departures a weekday, onset 25-50 km, `max(d, 25)^-2.37`, the outside
   world (0.76M of 5.13M departures leave the map), the car's running cost and parking.
3. Commutes between zones: `joint_gravity`, 1.51M commuters across pack lines among the 13 zones
   (1.12M among the nine cities before the non-city zones), in the regional layer at commute timing and the local commute's mode choice.
4. Non-city zones Worcester, Windham, New London and Litchfield Counties (`wor`, `wnd`, `nlc`,
   `lit`); the focus follows the zone under the view, and up to three nearby zones' bubbles show
   beside the focused zone's (app/src/game/zones.ts, map/zoneFocus.ts, map/commuterView.ts).
5. The far-zoomed per-zone bubble: local trips plus the regional layer's, coloured by train share.

Not built, each its own task: park-and-ride (T-295), the market's routed drives (T-279), the
commute part's own weekend factor in a station's far-zone rows (T-296), commutes shown apart from longer trips (T-298), the market's seam trips against NEC
FUTURE (T-299), and
fetching zones' water masks and boundaries on demand rather than at startup (T-302; today every
registered zone's are fetched at startup, as cities' are).
