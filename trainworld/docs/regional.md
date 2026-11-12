[Back to the spec](../SPEC.md)

# 5. Demand: the regional layer

The regional layer carries what crosses zones or goes far [T-117, T-276, T-377]: **inter-city
trips** (visits, business, leisure, 25 km and more) within and between every zone of the world,
and **commutes between zones**. The design and its reasons are in [zones.md](zones.md) and
notes/T-362.md; the kernel's side (`sim/src/demand/intercity.rs`: compositions, served pairs,
queries) in [demand-market.md](demand-market.md) "The world market" and after. Every registered zone is a
**place** of the market (`place` in the registry); place ids are stable and new zones append;
Trenton's (2) is kept with no sites (Mercer is in New York's pack).

## 5.1 The total travel market, then rail's share

Generate **total** all-mode trips, then let the player's network earn rail's share.

### Sites, pairs and what is solved

- **Sites**: population- and job-weighted endpoint sites at 600 m grain from every zone pack,
  grouped into zones of 6 km, each zone's in its own pack, `data/packs/<id>.market.json`
  (`pipeline/market_zones.py`; the Northeast's in the market frame, every other zone's in its own
  pack frame).
- **Inter-city trips** (T-376's rule, constants in `pipeline/longhaul.json`, copied into the
  registry's `market.rule`): a resident makes two kinds of departures a weekday, each shared by its
  own normaliser over the **world field** (T-375: Kontur H3 cells with country, GDP per head and
  landmass), never over the zone list, so adding a zone moves only the trips that went where it now
  is:
  - *ground*: `0.107 x s` departures, weight
    `a_j x b x onset(25-50 km) x (1 - onset(300-700 km)) x max(d, 25)^-2.37` (T-1603: the tail's
    window, so the ground is gone past 700 km; its normaliser is the plain curve, so the
    dropped trips are not given back to nearer places);
  - *tail*: `0.00473 x s x u^0.503 / 1.2413` departures, weight
    `a_j x u_j^0.75 x min(y_j / y_i, 1) x b x onset(300-700 km) x max(d, 171)^-0.703`;

  with `s = min(y / y_US, 1)^0.7` (GDP per head at PPP), `u` the people of the site's H3 r5 cell
  over 100,000, `a_j` people times `s_j` (inside a pack its `population + 2 x jobs`, rescaled to
  the pack's people), and `b` 1 within a country, 0.5 between two countries of the open-border
  area (the EU 27, Norway, Iceland, Switzerland, Liechtenstein and the UK, by the rows' `country`;
  `OPEN_AREA_BORDER`, T-809), else `0.26 x min(y_i / y_US, 1)^1.5` (the origin's income). Each
  6 km zone's two normalisers are computed offline at its centre.
- **The long-trip pull** (T-3240, `pipeline/market_pull.json`; off, `"on": false`, until the world
  rebuild after 2026-11-11 turns it on and rebuilds every pack's market file, so its packs, rows and
  registry rule never mix with and without it): a destination place's
  attraction, ground and tail alike, is times `1 + (exp(gamma x px) - 1) x fade(300-700 km)`, the tail's own
  window: the pull fills the 300-700 km valley between the two curves and leaves trips under 300 km as they
  are (Berlin - Hamburg, Rome - Naples, Madrid - Valencia, Istanbul - Ankara were at or over their
  published figures without it; Warsaw - Kraków and Warsaw - Tricity stay under theirs). A local factor,
  `near_gamma` (T-3254, Anita 2026-11-12, "a very slight poland-only pull"): within Poland a place's
  attraction is also times `1 + (exp(0.2 x px) - 1)` at every distance, unfaded, for its own residents
  (Warsaw 1.7x), added to the pull above: Warsaw - Kraków 3.6k -> 5.1k a weekday both ways, under its
  7.2k by rail alone in H1 2019. `px`, its
  *excess*, is how much more important the place is than its people say: the best Natural Earth 10m
  populated-places `scalerank` inside its zone's outline against the rank a place of its people has
  (`20.80 - 1.08 ln(people)`), clipped to 0-3. Trimmed for map balance: the cap of 3 shrinks on a log
  scale from 5M people within 300 km of the centre (own included) to none at 0.5M (Darwin, Anchorage,
  Reykjavík go to 0, Auckland to 2.3); Leh and Magong (Penghu) are set to 0 by hand. Beach and resort towns
  (`pipeline/market_resorts.json`: Wikidata's seaside resorts and resort towns and Wikipedia's
  seaside-resorts categories) give their place at least `1.5 x min(1, their people's share / 0.1)`.
  `gamma` is 0.6 within a country (China's 0.6 of that, 0.36), 0.25 across a border (FR24 airport
  pairs); residents of the US, Japan, the UK and Korea see no pull (their pairs fit without), nor do
  Russia's and Turkey's (their 300 km+ trips are over the published figures without it). Fitted so every
  checked pair and national 300 km+ rate is at or under its published figure, the lower where sources
  disagree (Anita, 2026-11-12; notes/T-3240.md section 7). The normalisers carry it over the world field (each r5 cell its place's excess, by
  outline), so each origin's departures are kept and only where they go changes. `pipeline/market_pull.py`
  rebuilds the excess table, `pipeline/market_resorts.py` the resort list.
- **Pairs**: places whose centres are under **250 km** apart pair 6 km zone to 6 km zone at their
  own distance; farther places pair at the distance between the places' centres, so their 6 km
  pairs add up to the places' trips exactly, each end's share by its zones' departures over the
  normalisers and its attraction (DECISIONS-to-2026-10-19 2026-10-15, T-377). Opposite totals are averaged;
  pairs under 0.01 trips a day each way are pruned. Distances are great circles [T-128]. Local
  outings taper over 40-80 km, so the two layers meet without a gap or double count.
- **Market rows**: each place's row (`app/src/game/market-rows.json` by zone id, T-776: its zones' sums, centre, country, income,
  landmass, parking, its airports' drive times and direct flights, and its trips with every place
  under 250 km, every zone pair added up) gives any pair of places its trips and modes with no pack
  open; `bin/market_pack.rs` writes them from the places' packs (a row reads its own and those under
  250 km, so only the rows near a changed pack are written again, T-929).
- **Commutes between zones**: the cross-pack part of the joint commute gravity
  ([demand-kernel.md](demand-kernel.md) 4.4, [zones.md](zones.md) 4), solved over each group of packs whose
  outlines come within 50 km of each other (T-539), spread from 2 km gravity zones to the
  market's zones and kept where at least one a day travels between two 6 km zones (each home zone's
  dropped commuters given to its kept pairs). They travel at the local commute's timing (70/12/6/4/8%
  to work, the return 2/18/58/16/6% on the same station pair reversed) and 0.30 of a weekday's at
  the weekend.
- **Everyday outings across a zone line** (T-1175, T-1176; [zones.md](zones.md) 4a): errands,
  social, school, university and medical outings drawn offline over the neighbouring packs' zones,
  by 6 km pair and purpose with each pair's walking share with no rail, kept at one a day or more.
  Built between served zones as the commutes are; each purpose's outward and return timing; the home
  end walks or rides from 5 km, the activity end walks. Walking takes its part of the everyday
  layer's three-way logit (its weight from the pair's walking share), rail and the car split the
  rest on the ratio curve with the commutes' car, and rail takes its share of the walkers. Their
  loads are their purposes'; a weekend scales their counts by errands' factor and their loads by
  each purpose's.
- **What is solved**: pairs are built only between 6 km zones some station serves, whose stations
  share a track component, with a car or a plane to compete with: on one landmass (T-375's, fixed
  links joined), or far places that both have airports. Every other pair's trips are fixed: counted
  in the totals, shown, never in the solve. The market's day and its corridors (place pairs) come
  from the rows, the same in every part of it.
- **Zone pairs in chunks** [T-780]: a zone's trips with the zones of each place that come after it
  in the market's zone order are dealt in chunks of about 1 trip a day each way (`MARKET_CHUNK`),
  by systematic sampling over that place's zones by their trips from a start fixed by the two
  zones' ids (`solved_pairs`). A pair of at least a chunk is always solved; a smaller one carries a
  chunk or is not solved. The chunk grows with distance [T-2641]: `MARKET_CHUNK` times the distance to
  the place's first zone over 50 km, from 1 to 16 times. Place pairs at least 250 km apart are dealt
  once over both places' zones: each served origin zone's run of the far place's zones follows the
  last one's, from a start fixed by the two places, so an origin zone with less than a chunk to a far
  place gets a whole chunk or none. A zone's trips with a place under 250 km away are exact; with a
  far place they are exact on average over the place's origin zones, so a far city's long-distance
  riders reach its stations in lumps (one zone's station can show twice its far riders and the next
  none), while totals by line and by city hold. The dealing runs over every
  destination zone, served or not, so a newly served destination zone adds pairs without dealing the
  others again; a far place pair's run depends on the served origin zones before it. Mode and station choice within a pair are unchanged. A
  selection's rail with a place is its sampled pairs' rail spread over the place's zones it could
  ride to, by their trips; a station's far-zone rows stay the sampled pairs'. `?perfOff=chunks`
  solves every pair, `?perfOff=farChunks` deals them in the same chunk at every distance.
- **Elsewhere**: a place's or a 6 km zone's departures less its trips with every place are its
  trips with places that have no zone; shown, never solved, their modes not modelled.

### Mode choice

**Mode choice.** Inter-city trips choose rail, car and plane in one logit [T-378]. **Income groups**
[T-1431]: a place's longer trips come from its country's people in three equal groups by income
(the World Bank's income shares by quintile, `countryIncome.json` `quint`, a country with no survey
taking its subregion's median); a group's income over the mean is 0.35-0.49 in the poorest third
and 1.6-1.9 in the richest. Each group's riders weigh every mode's money, the car's running cost and
crossings, the plane's fare and rail's fare, at the same minutes a dollar: the departing country's
(T-1277) times (the group's income over the mean)^-0.5, times 0.4 (T-1521: a longer trip's time is
worth more than a local one's, about US DOT's guidance, $14.10 an hour locally against $20.40 by
road and $47.20 by air or high-speed rail). A group makes longer trips in proportion to
(its income over the mean)^0.7, the rule's `eps`. A zone pair's choice is worked out for each group
and added up by their trips; the place pair's split between car and plane likewise. Rail's fare is
the player's curve as set (a trip pays it times its starting country's scale) [T-1277]. A curve
dollar costs a local trip's riders at their default scales `marketCurveWeight` perceived minutes:
India's 51 a dollar of India's units, over the start country's income scale in a game from T-1906 on,
whose curve is in that country's dollars (4.38 in a US start, 6.96 in a Japan start; T-3239). **The
route search charges fares at a longer trip's weight, 0.4 of that** (`marketFareWeight`, Anita's
option A, T-3239: every country's three groups average 0.40 by their trips), so a longer trip picks
between a high-speed line and a slower one at about the weight its groups choose rail at. Each
trip's fare is weighed again by its group's factor and by its origin's fare scale over its default
(on the expected fare km of each station pair's strategy as searched, so routes stay as the search
gives them) [T-1286]. Commutes and outings between zones keep one group, the car's money at 3
minutes a dollar and the fare at their home's weight in mode choice, but share the longer trips'
route search, so they too pick routes at 0.4 (the lie; notes/T-3239.md). All of it is damped by the trip's length as the local walking share is
([demand-mode.md](demand-mode.md) 4.3, T-495): 0.10 x sqrt(60 / T) per perceived minute, T the
car's perceived minutes over the zone pair's distance, even across a sea where no car goes (so
rail against air never depends on it). The scale covers the whole
rail trip, access included. Rail's cost is chosen as in the local layers (T-718, demand-mode.md
4.3, T-710): the pair's riders spread over its station pairs (up to 10 access stations at each end, 5.2)
by their whole cost, access and egress included, at the station choice's scale for the zones'
distance (0.33 a perceived minute from 12 km, so nearly every pair), softened past 60 minutes by
sqrt(60 / T) with T the car's perceived minutes (`rel_soft`: 0.19 at 3 hours by car, 0.12 at 8),
so a long trip spreads over more of its stations, and rail is priced at that logsum before the
damped mode choice. Each zone's access weights (built at 0.10, 5.3) are taken to
that scale by raising them to the power of the scale over 0.10. Rail's resulting share against
air by hours on the train: notes/T-712.md.

- *Car*: the rule is 15 minutes plus 1.25 times the great-circle distance at 85 km/h, plus the
  car's running cost a road km (T-1521: the two places' countries' mean of `countryIncome.json`
  `car`, gasoline at the pump at 8.5 litres a 100 km plus motorway tolls where they go by distance;
  the market row's `cu`, 15 cents where a row has none) and the destination's parking (the local rule, docs/demand-mode.md 4.3, baked into the
  market pack's `site_park`), both split over two occupants, the running cost at each income group's
  minutes a dollar (T-1431); no tolls at borders, distance tolls inside the running cost
  (T-270, T-1521), and North Korea's roads are open like any other's. Only within a landmass.
  *Routed drives* (T-279, `market-drives.bin` from `pipeline/market_drives.py`): per pair of places
  on one landmass within 1,500 km, a least-cost path on the world's 1/60 degree lattice, each cell's
  speed from GRIP4 road density (110 km/h with at least 25 m of highway a km², else 85 with 25 m of
  highway and primary road, 55 with 100 m of any road, 20 with less) slowed by GEDTM30 slope (divided
  by 1 + (slope - 3°) / 6, highways excepted), open water at 6 km/h within 12 km of a shore, Wikidata's
  fixed links as land; the drive is 24 + 1.252 x the path's minutes and its road 1.107 x the path's km.
  The table keeps each as a ratio to the rule's at the places' centres. T-2424: each place's search
  is read also at the other place's market zones, so each zone has its own ratios towards that place
  over the centres' (kept where 10% or more off; zones within 5 km of the other centre, or on a
  different number of ferries, keep the centres'). Every drive between two zones of the two places
  (each zone pair, commutes and outings between zones, the trip explorer) is the rule's at its own
  distance times the places' ratios times zone i's own towards j's place and zone j's towards i's,
  so zones at a lake's end drive round less and zones across its widest part more; the corridor
  split keeps the centres'. Within one place, beyond 1,500 km, across the Channel or the Tsugaru
  Strait, or with no table, the rule. A car takes a *ferry* only where OSM maps a car ferry
  (`route=ferry` with `motor_vehicle` or `motorcar` yes): 90 minutes (check-in, the wait for a
  sailing, loading and unloading) plus the crossing at 30 km/h, and a fare of $80 plus 50 cents a
  km per car, shared by the two, so a ferry is clearly worse than a bridge (the path uses it where
  that costs less than going round). Where the car is carried over water by the market's own
  crossings (T-712, `crossing_cost`) it pays the crossing's fare per car (shared by the two) and its
  minutes beyond the straight drive: the Channel, $150 and 60 minutes, between a place in Great
  Britain (the UK less Northern Ireland, which is on Ireland's landmass, T-2423) and one elsewhere;
  the Tsugaru Strait's ferry, $170 and 240 minutes, between Hokkaido (Japan north of 41.4°) and
  elsewhere. Commutes between zones pay them too. Airports' access drives keep the rule.
- *Plane*: between places whose centres are at least 250 km apart (the far pairs) and that both
  have airports. Door to door: the drive to an airport at each end (the car's drive time,
  people-weighted over the place's zones), 135 minutes at the airports (both ends together, T-1564), 25 minutes until the flight
  (its frequency is not modelled: rail's planned first wait at a flight every 2 hours, T-1430), the flight at 750 km/h on
  the great circle between the places' centres plus 30 minutes, 60 more with no direct flight, and
  a fare of $100 plus 5 cents a km (T-1521, published average economy fares at market rates, one world fare) at each income group's minutes a dollar (the corridor's split is the mean of its two ways); the best pair of the two places' airports. Not built by
  the player, no capacity, not drawn.
- *Rail*: the player's network. Fares, service, transfers and shared crowding affect it. An
  inter-city trip is planned, so its first boarding waits the least of half the headway, 5 minutes
  plus a sixth of the headway, and 30 minutes, with no tolerance penalty (T-1430; the same as the
  local wait up to a train every 15 minutes, 15 at an hourly train, 25 at one every 2 hours). A
  change of trains on the way waits half the headway as in the local layers, and commutes and
  everyday outings between zones wait as they do. The search costs each station as a trip's start
  both ways (`Skims::entry_r`); the riders' loads follow the station's ordinary strategy.

The car's and the plane's costs are by place (the car's at the places' centres and their mean
parking), so a pair's split of its trips not by rail between car and plane is the same whoever
builds what, computed once, and kept (`corridor_modes`); the solve weighs every mode but rail
against the zone pair's own car time with that split. The main thread holds a copy of the
place-pair rule (`app/src/game/marketRule.ts`, T-1148: car, crossings, plane, split, the places'
trips) for the trip explorer, held to the sim's by `checks/t1148-market-rule.mjs`. A place's airports (`pipeline/market_zones.py`):
OurAirports' large and medium airports with scheduled service within 150 km of its centre carrying
at least 1M passengers a year, the biggest eight; a route of 50,000 passengers a year is a direct
flight (both from the flights project's FR24 file, read offline only). Commutes between zones
choose rail or car as local commutes do, on the ratio curve of their costs (demand-mode.md 4.3,
T-918; no walking between zones), the drive weighted 1.33 plus the work end's parking; their loads are the work purpose's [T-158]. Inter-city trips' period
shares are 20/35/25/15/5%; their weekend factor 1.58.

Not modelled yet: coach, a car between landmasses (a ferry where the market has no car at all,
T-1396), induced travel, traffic on routed drives.

## 5.2 Where a long-distance trip starts and ends

Origin access is population-weighted; destination access uses population and jobs. Each site offers
nearby stations through the local layer's walk and, where it applies, rides (5.3). Each zone end
keeps its 8 best-weighted stations, then up to 10 in all, the best station of each track component
none of those is on (an inter-city line not linked to a city's metro), best first; a station only
bus routes or lines with no trains stop at takes no such slot [T-1506]. The longer trips, commutes and outings between zones and the trip explorer use the same
lists. They feed the common strategy graph. Local feeder lines, transfers and the
intercity service receive actual loads and boardings. No express-only route or station-city
affiliation bypasses the player's local network.

**Seeing them** [T-118]: the local views cannot show them (a Philadelphia bubble's trips end in New
York, off screen), so a selected bubble set and a selected station list them by place and draw
them by far zone ([ui-demand-views.md](ui-demand-views.md) "Inter-city trips"). A selection of
local cells is mapped to the market through its 600 m squares: the sites were built from the same
cells, so each selected cell falls in one site's square; a square counts the selected share of its
people (homes) or attraction (destinations), and a site's share of its zone's departures is its
population over the zone's (of arrivals, its attraction over the zone's). The numbers are the
market's own, summed from its zone pairs; a place whose pack is not open is counted by place only
(a far one exactly, a near one as the places' trips times the selected zones' shares of their
place's departures and attraction). A station's riders come from the same assignment that
loads the lines, so they add up to its entries and exits.

## 5.3 Park-and-ride

The regional layer rides to stations as the local layer does ([demand-mode.md](demand-mode.md) 4.3, T-371),
through the same `kernel::access_options` [T-373]: from each 2 km zone of the market's sites
(their homes and jobs, as a local zone's, give the drive's speeds and the parking), the cheapest
station of each of the 6 cheapest lines within 15 km, then up to 2 more, the cheapest line of each
track component none of those 6 is on, its station needing a line with trains [T-1511]: so a zone in
a city of many metro lines still rides to an inter-city station not linked to the metro, and that
station can take its slot in the access list (5.2). The local layers keep the 6 alone. Each ride
costs `p_ride` 15 + 2.66 x the drive (T-916) + at the station a cost that depends on the end:

- **Commutes between zones**: at the home end, the cheaper of parking at the station and a
  drop-off (`d_drop` 10), when the two zones' centres are 5 km or more apart (`ride_min_km`); the
  work end walks. Everyday outings across a zone line the same, the activity end walking.
- **Inter-city trips**: at the home end as a commute's; at the far end a drop-off only
  (`d_drop`), since the traveller has no car there.

A site's access at each end is its walks (the mean of their weights at 0.10 a minute) nested
with its zone's rides at the station choice's scale from 12 km (0.33 per
minute, T-718), so a
ride dearer than the walk adds little, and a site beyond every walk is reached by its rides alone.
Zones that only a ride reaches are served (5.1), so a lone intercity station draws on everyone
within the ride's reach. Player-built feeder lines take part in regional routing.

## 5.4 Size

A place's market pack holds its sites, its 6 km zones' normalisers and its commutes to other
places (sizes: notes/regional-history.md). The solving workers open it with the zone (viewed,
beside the view, reached, or a station within 30 km of its box) and close it as they close the
zone; a worker composes the market from the packs it holds. It runs in the same worker pool as the
zone packs, periods divided among workers. Each period advances all populations from one frozen
aggregate flow vector per crowding round. Details and checks: [T-117](../notes/T-117.md),
[T-276](../notes/T-276.md), [T-377](../notes/T-377.md).
