[Back to the spec](../SPEC.md)

# Zones (T-276)

Anita's decisions behind it (2026-10-13): option 2 of `research/intercity-review.md` (retune the
regional market and add commutes between cities) inside a world divided into zones.

## 1. What a zone is

The world is divided into **zones**, and every place is in exactly one zone or in dead space.

- A **city zone** is a metro area: a world plan row (adding-a-city.md 1) or a hand-drawn extent
  (the nine Northeast city packs, Toronto).
- A **non-city zone** (countryside zone) is land around city zones that is not part of any metro:
  the rest of a covered country (section 2), or in the US the counties between city zones.
- **Dead space** is everything else: countries not covered yet, and very empty units inside
  covered ones. Nothing there makes or attracts a trip in any layer, so a line through it carries
  only riders going through.

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
**touches two different city zones**: it lies between them. Today that is Worcester MA (Windham,
New London and Litchfield CT went into Hartford's pack, T-1113). Countries planned whole (below)
leave no such gaps. Everything not modelled is dead space. **Japan is modelled
whole** (research/japan.md decision 2): every prefecture outside the metro packs is a zone, and in
Hokkaido each subprefecture is a unit under the 100,000 rule (Hiyama joins Oshima; Rumoi and Soya
join Kamikawa; Hidaka joins Tokachi; Nemuro joins Okhotsk; Shiribeshi less Otaru joins Iburi), 38
zones from `pipeline/japan_zones.json` (T-420 stage 3). Their outlines leave out the
Russian-administered southern Kurils and uninhabited islets more than 50 km from other land.

**Whole countries** (Anita, 2026-10-19, T-719; built for Spain, T-723) are covered by zone rows in
the world plan (`world_plan.py zones --countries ES`), built by the runner like metro rows. Every
unit of the plan's unit level (NUTS-3 in Europe) that no plan row holds and no pack holds half of is
a candidate. A unit under **20 people a km²** stays dead space for now unless it holds 0.8 of the
zone floor (0.4M at 0.5M; `DEAD_KEEP`, T-903: Bergen, Trondheim, Oulu, Inverness); Spain's dead
space is Soria, Teruel, Cuenca, Huesca, Zamora, Palencia, Ávila, Cáceres and Segovia. Units **under 100,000** join a
zone of their own region (NUTS-2; elsewhere the first-level unit), smallest first: the one they
share the longest border with, or for an island the most populous zone of the region within
150 km (so small islands join their province's largest: La Gomera, La Palma and El Hierro
Tenerife, Fuerteventura Gran Canaria, Menorca Mallorca). A small unit with neither stays its own
zone (Ceuta, Melilla). **Zones hold at least about 0.5M people** (Anita, 2026-10-20, T-788):
countries planned since then keep merging within the region until each zone has **500,000**,
from 100,000 into the neighbour with the nearest people centre that keeps the two within **1.5M**
and makes no zone that **wraps** another zone, a metro or a pack (T-1087; below), and into a zone of
another region of the country when its own has none left (a small region, or Osttirol cut off from
Tirol); a zone of 100,000 or more crosses water only within 10 km, and islands and enclaves that
can join nothing stay as they are (the Isle of Wight, Åland). Rows built before the floor keep
their size: Spain, Italy, France, Portugal and Greece, and T-788's first wave (Germany, the UK,
Ireland and the Benelux at 0.1-1.2M). A European country's units outside Europe (France's
overseas départements) are not zoned. A zone row has `kind` "zone", is named after its most populous unit (before
any "/"), and has its origin at its densest square km. A replan of the metros keeps zone rows as
they are. Countryside zones use the zone curve (section 4) and global sources only, unless their
country's zone rule names its module (the US, below). Which countries are covered, and with how many zones:
docs/data-register.md.
**No ring zones** (Anita, 2026-10-23, T-1087: zones keep together areas that mostly commute within
themselves, "kinda like voronoi"). A zone grown to the floor joins, of its neighbours keeping the two
within the cap, the one whose people centre (the mean position of its people) is nearest its own, so
zones gather round their own towns; with none within the cap, the smallest. A join that makes a zone
**wrap** another zone, a metro row or a built pack is refused: A wraps B when at least **0.5** of B's
area lies inside A's convex hull and A holds at least **0.5** of B's border with zones, metros and
packs (coast, dead space and other countries do not count). A zone with no join within the cap, or
whose every join would wrap, stays as it is once it holds 80% of the floor; a smaller one joins
anyway. Then each zone still wrapping something is cut in two across it (its units by direction from
the wrapped area's people centre, split where the halves hold about the same people and each is in
one piece, by land or across its own water joins) and the halves regroup by the same rule, except
that a half whose every join would wrap stays as it is, under the floor if need be (rejoining would
rebuild the ring); three rounds at most. A single unit wrapping a zone takes it when both fit under
the cap. A replan regroups only
planned rows. notes/T-1087.md.
**Commuter belts** (Anita, 2026-10-23, T-1088: "we can let metros take neighboring counties but it
doesnt have to be a hard rule"). Where a country's commuting between units is on disk
(`world_plan.COMMUTE_FLOWS`: the US's ACS 2016-2020 county flows), a metro row not yet built takes
each free unit sending at least **25%** of its resident workers into it, when that metro is where it
sends most, the unit touches the metro by land (or touches a unit joining it) and the metro stays
under its cap; a unit it cannot take is grouped into zones as usual. The zones step adds the belts
(`zones --countries`, recorded in the plan's `belt` source and the country's `belts`), so a metro
replan is followed by a zones replan. Built metros, and every metro elsewhere, keep their extents.
**China and Taiwan** (Anita, 2026-10-22, T-897) have their own rules (`world_plan.ZONE_RULES`):
zones of at least **1M** (cap 2.5M) from geoBoundaries counties merged within their province, no
dead space by density, and **no exclaves**: a zone's parts touch by land or across at most 10 km of
water (another region's zone across water only for a zone under 80% of the floor, as by land; T-1015),
island zones of one region join within 50 km (Penghu, Kinmen), and a zone that can join
nothing with under 10,000 people is dead space. Hong Kong, Macau and Taiwan are countries of their
own, never in a Chinese zone. Units follow the de facto lines (`world_plan.DE_FACTO`: Arunachal
Pradesh is India's, Aksai Chin China's). Chinese rows solve their joint commutes by province
(`solve_region`; Beijing, Tianjin and Hebei as one, Shanghai with Jiangsu), each held against its
neighbours as countries are (section 7). notes/T-897.md.
**The US and Puerto Rico** (T-1004, T-1015) take Census 2020 counties as units (Puerto Rico's
municipios; GEOIDs, the state as region; `world_plan.census_units`). Their metro rows are planned
alone (`world_plan.py metros --countries US`, `FUA_COUNTRIES`), by Europe's rule: a GHS-FUA over 1M
people, from US counties only (none lent to a Mexican or Canadian metro, none of theirs taken), with
Riverside and Imperial counties a metro row by hand (`METRO_EDITS`) and San Diego and San Francisco
named for their cities rather than GHSL's Tijuana and San Jose (`METRO_NAMES`); a replan of the
other metros keeps them. Zones (`zones --countries US,PR`) follow China's rules at a **0.5M** floor
(cap 1.5M) within their state. In Alaska only the road and rail belt is zoned (Anchorage,
Matanuska-Susitna, Kenai Peninsula, Fairbanks North Star, Denali, Southeast Fairbanks, Chugach,
Copper River; `zoned_only`), the rest dead space; Guam, the Northern Marianas, American Samoa and
the US Virgin Islands are not zoned. Every US and Puerto Rico row, zones included, takes the `usa`
module (archetype A; Puerto Rico C, population only). US rows solve their joint commutes by Census
division, Delaware, Maryland and DC with the Northeast (`SOLVE_REGIONS`); Puerto Rico is its own
country. A zone named after a county name another zone has carries its state ("Jefferson County, KY").
notes/T-1004.md.

**Mercer County (Trenton) is part of New York's city zone**, not a non-city zone (Anita leaned
this way). Reasons: it is in the New York combined statistical area; LODES holds 175,000 jobs
between Mercer and New York's counties against 88,000 with Philadelphia's; and as a zone of
387,000 people pinched between two big packs, most of its commuting would cross a zone line. A
county between two city zones whose commuting runs mostly to one of them may join it this way; the
choice is made per county when a city's extent is drawn, as extents always have been (pragmatic,
no study). Its place in the market (id 2, "Trenton") stays in the registry with no zone, because
place ids are never renumbered.

**Data.** The same national sources as the country's city zones, at the same grain: in the US 2020
census blocks less institutions (T-234), LODES WAC and RAC, NCES, IPEDS, FAA, OSM. Where a country
has no small-area jobs (tier 0, docs/demand-sources.md 4.1), population from Kontur (H3 resolution 8,
global, each hexagon spread over its seven resolution 9 children) or the national census, and jobs
from the GHSL non-residential built-up layer (T-012). A country's non-city zones are built by that
country's city adapter (Toronto's for Canada, T-277) with `kind: "zone"`.

**Names.** A non-city zone is named after its unit ("Worcester County" in the US, the census
division's name in Canada); a merged zone after its most populous unit. That is the registry's
name, which every join rewrites; the game shows its own (T-825, `app/src/game/zoneNames.ts`):

- **One short name.** A city zone shows its biggest city ("Rotterdam", not GHSL's "Rotterdam [The
  Hague]"), or a metro name only where that is clearly what people call the whole zone ("Ruhr",
  "Kansai"; not "Rhine-Main" for Frankfurt, which people call Frankfurt). A countryside zone keeps
  its unit's name (a county, department, prefecture, province) unless that name is a statistical
  label nobody uses or a compound: NUTS groupings ("Overig Groningen", "Essex Thames Gateway", Ireland's
  "South-West"), Poland's adjectival subregions ("Lubelski"), "X and Y" or "X, Y" merges, and a
  unit that holds a much bigger city than its name says ("Aargau" holding Basel). Those show their
  biggest city, or a well-known name for the whole ("Dodecanese", "Cyclades", "Teesside"). English
  exonyms where English uses one (Florence, Venice, Geneva, Corfu). A US zone whose shown name another
  zone has, or a planned row will have, carries its state ("Columbus, GA", "Rome, GA"), as US county
  names in the plan do ("Polk County, FL").
- **Automatically**, a zone with no hand-set name drops a bracketed part ("[The Hague]"), a
  trailing parenthesis ("Halle (Saale)") and NUTS's " CC" ("Devon CC").
- **Hand-set names** live in `app/src/game/zone-names.json` (zone id to `name` and extra `aliases`),
  which no pipeline step writes. A newly joined country's zones need a look against this rule.
- **Aliases** are every other name search finds a zone by: the hand-set ones, the registry's name
  and its parts, and the six biggest cities and towns (20,000 people or more) in the zone's place list with
  their OSM other names (`app/src/game/zone-aliases.json`, written by `pipeline/zone_aliases.py`
  from `data/packs/<id>.places.json`; run by `world_run.py join`). Both the map search and the
  new game sheet match aliases ignoring case and accents, show the zone's own name with "matched
  The Hague" beside it, and rank a match through an alias with one through the name, the name
  first among equals.

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
  does at runtime (with thin country's 4 km blocks too, docs/demand-kernel.md 1, T-852). Across
  packs a block stands at the middle of its four squares. Every pack the solve balances takes the
  curve's flat near end of today's rule (T-1091: flat inside 2 km on every curve but the US one) and
  ships it in its gravity block, so a rerun join applies it; a fixed neighbour keeps its own.
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
- **Long commutes are set, not emergent** (T-1000, Anita 2026-10-23). 1% of every gravity zone's
  workers (`joint_gravity` `LONG_SHARE`) commute to jobs 50-150 km from home (`LONG_KM`; 50 km is
  about where a daily commute becomes a trip between towns, 150 km the cut), shared over the jobs
  in that ring by jobs times the home pack's curve and border factors, in its own pack or another,
  and not balanced, so no job-rich oasis draws more of them. They join the list above, a pair whose
  work zone is in the home's own pack included, kept apart from the balanced pairs when pairs under
  1 a day are dropped. The balance takes the rest: each such zone keeps 99% of its workers and each
  pack's jobs are scaled to its homes' total, so the balanced shares by pair do not move. The
  balance itself has the pull cap (docs/demand-kernel.md 1): a zone's jobs pull at most 20 times
  harder or more weakly than its pack's mean, its pack's other zones filled at one ratio so the
  pack's net exchange stays nil.
- **The balance** (T-297) is solved per group of packs joined by any pair within the cut (groups
  share no commuter). Each group runs Furness steps with Anderson mixing of the last 12 steps, until
  every column inside the pull band meets its pack's fill of its jobs to 1e-8 relative (its homes
  exactly), about 40 steps; a group of thin packs held hard at the band takes several hundred. The tolerance is
  tight because weakly joined packs and regions leave slow modes: a looser residual leaves factors
  percents off (notes/T-297.md).
- **Groups form themselves** (T-539, `intercity.py --group auto [ids]`): two registered packs are in
  one group when their outlines (`<id>.boundary.geojson`) come within 50 km of each other, chained,
  across national borders too; a pack with no neighbour that close stands alone and is not solved.
  Within a group the 150 km cut still applies. Every group runs `joint_gravity --packs`
  (distances in a conic at the packs' mean origin; at most 300 steps, the best kept), the
  Northeast's 13 hand-built packs too since T-1014 (solve region "northeast", their hand-made
  rows' `intercity.SOLVE_REGION_PACKS`), and
  writes its cross-pack commutes, by 6 km square of each pack's own frame, to
  `data/work/<first member>.joint_commutes.json` (Japan's 45 packs: `tyo`), which `market_zones.py`
  writes into each zone's market pack; an older file holding any of the same packs is removed.
  Japan's balance stops at about 8e-4, all of it in Okinawa's outlying islands, a block joined to
  little else (notes/T-463.md). A tier 0 pack's single region is its own, and its header names its
  country. A group's solve grows with its zone pairs (within packs plus across within the cut):
  each Furness step sums every pair twice, about 100M pairs a second a thread, so Japan's 0.5
  billion pairs take about 25 minutes on one thread and the Northeast's 0.2 billion 4 minutes.
- **A group spanning countries is solved a country at a time** (T-786, T-781): each country's
  packs in the group are one solve unit, and the group's packs of other countries within the
  150 km cut take part held at their factors as they stand (`joint_gravity --fixed`): their
  zones are in the unit's sums across the cut, but are neither balanced nor written. The unit's
  file (`<first member>-<country>`) lists its packs as `solved` and holds every commute with an
  end in them, the fixed neighbours' into the country too; each pair of packs takes its commutes
  from the newest file covering it, and a file a newer one covers whole is removed. A country's
  join so costs its own solve only, however long the chain of covered countries; a neighbour's
  own commutes and factors stay as its last solve left them until it is solved again. National
  borders take the 0.05 default, so under 0.02% of commuters cross one (notes/T-786.md).
  A fixed neighbour's rows are not rebalanced, so one whose factors predate the pull cap (no
  `gravity.pull` block) can pour workers into capped zones across the line that its own solve had
  held near zero; solving that neighbour again with the cap takes the flow back (notes/T-1000.md).
  A fixed neighbour solved before the flat near end keeps its own curve for its homes' pairs
  across the line; nearly all those pairs are 2 km or more apart, where the two curves agree.
  A solve region in `intercity.SOLVE_ALONE` (Xinjiang, T-998) holds no neighbours: its commutes
  with them come from their solves. `intercity.py` refuses one solve over 400,000 zones (about six Japans).

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
  zone needs only its own pack and the market rows rebuilt (`pipeline/market_zones.py`, seconds).
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
that zone. A network whose stations reach no zone is still solved, with no pack, so its lines
show their figures, nobody on them (T-845). One helper answers it, `placeAt` in `game/zones.ts`, kept in `viewedZone`
(`game/cities.ts`).

**Realms** [T-975, T-1006; `game/realms.ts`, `map/realmOverlay.ts`, `pipeline/realms.py`,
notes/T-975.md]: zones are grouped into country-sized realms, a hand list in
`app/src/game/realms.json` (each an id, a name and its countries, or for a country split between
realms its first-level units: China in seven, Russia in two, the US in four by Census region with
Delaware, Maryland and DC in the south, and Puerto Rico a realm of its own). Players see only a realm's
name; the code word is "realm" because "region" already means a zone in the code.
`pipeline/realms.py` (run by the join) writes each registry row's `realm` (its country's; in a split
country the realm whose land holds its centre) and `packs/realms.geojson`, each realm's land
(Natural Earth admin 0 cut along geoBoundaries ADM1 or the Census states), which only that
assignment uses: the app reads no realm shapes. A realm is **shown** while the game has a station or
track in it (blueprint or built; a station in the zone whose coarse outline holds it, else the zone
with the nearest centre; track in the zones it crosses, "Track" below), it holds the start zone, or the player showed it this session (its circle, a click on its land, a
search pick of a city or place in it, or the view resting 0.4 s at zoom 10 or closer with its centre
on its land after a move [T-1151, `AUTO_SHOW_ZOOM`]); a click is not saved, so a reload hides a realm again unless
the game has built there. A game started on the map (no start zone) shows every realm while it has
no station or track [T-1133], so the player sees where a game can start; its first station or track
ends that, leaving the realms the rule above shows (that "all shown" is not the player showing them).
What is not shown is one shade over the whole world, land and water
alike, with the shown realms' zones cut out along their coarse outlines (`packs/outlines.json`;
shared borders cancel and gaps the zones enclose stay in the cut), so its edge follows the zone
outlines and two hidden realms meeting have no edge (`inactive` in `game/shadeTable.ts`: 11% of the
outside shade's colour, 22.5% black in the dark theme, drawn over the outside shade: half way from
the outside shade alone to the double shade it was). The cut is made in a worker (T-1026): when the
shown realms change, the circles and cities change at once and the shade keeps its old shape until
the new one arrives, swapped in one frame. Each hidden
realm has a circle at the centre of its zone nearest the mean of its zones' centres, and with the
select tool its land has a hover naming it with "Click to load" [T-1151]. Its zones are
dead space for the view: no bubbles, labels, outlines, loading greys or bars, no name in the top
bar, and no pack, boundary or water mask fetched for viewing. Demand, the market and money never
read it, so every figure is the same whatever is shown; a selection's far ends still draw over a
dark realm. Before the first game is in, and when `outlines.json` is missing, every zone shows; a
zone with no realm always shows.

**Track** [T-1047, `game/trackZones.ts`]: a zone is crossed by track when a point of the track's
drawn line, tested every 1 km, lies in its coarse outline. A zone built track crosses is in play as
one a station reaches, for all the player sees: it joins the opened list, its realm is shown, and its
figures are worked out. Blueprint track that is not switched off shows its realms only, as a
blueprint station does. With no station within reach none of its people can board, so a solve would
give it the figures it has with no rail (T-149): it joins no solve, and opens bare in the auxiliary
worker instead, once the load in hand is done, one zone at a time, nearest the view first and after
any zone opening for the view. It then closes as a spare zone does, its figures still shown. A
station built within its reach makes it a reached zone like any other.

**The opened list** [T-729, T-801, T-1047]: each game's save header keeps `opened`, the zones it has
started in, a station has reached or its built track crosses, in order. Zoomed out, a zone not in it
is a dot (T-976); nothing reads it for loading. A save without it loads as any other.

**Zoomed out, nothing loads for the view** [T-871]: the viewed zone's pack and the zones' exact
outlines load only above zoom 7.5 (`ZONE_LOAD_ZOOM`, `game/zones.ts`), one level out from where a
zone's own bubbles show (8.5). Further out the overview's one bubble per zone needs no pack (a zone
not open shows the registry's local trips), outlines are the coarse ones, and only zones a station
reaches are open, with their markets, so demand figures are the same at every zoom. A viewed zone
left behind closes as any pack does (T-301: after a minute, or at once past two spare).

**Nearby zones** are drawn too: every zone whose boundary is on screen (at most
three, nearest first) shows its own bubbles beside the focused zone's, so a zone line is not a cliff
in the bubbles. A click on a nearby zone's bubble focuses that zone and selects the bubble.

Bubbles need a zone's results: an opened zone on screen is opened (its pack downloaded) like a viewed city.
With no station within reach (T-325) it opens **bare**, in the auxiliary demand worker only: its
bubbles and city figures come from a no-rail pass there (all walking and driving, exactly what a
solve would give it), and no solve covers it or is started for it. A click on its bubble draws the
far end as a solved zone's does (T-336): local trips from the auxiliary worker, longer trips and
commutes between zones from the solving workers, which open the zone's market for the selection
(T-801). A new game opens the
viewed city bare from the start (T-351); a saved game's viewed city opens in every worker, since
its pack downloads before the network's reach is known (docs/demand-runtime.md "Bare zones").
A station built within reach downloads the pack again and opens it in every worker; the next solve
waits for it and covers it from then on, and its bare figures serve until a shown solve does.
Detail: [demand-runtime.md](demand-runtime.md), "Bare zones".

**Zoomed far out** (8.5 and below) a zone not opened is a dot, left out near a bigger zone's mark so
dense realms are not crowded (T-1133, ui-demand-views.md "Zoomed far out"). Every opened zone is one bubble, as if it were one place: its trips a
day at homes (Homes) or at destinations (Destinations) in every layer, local trips, commutes
between zones and longer trips, coloured by train share, sized against the busiest zone whose
centre is on screen (New York's local trips would otherwise dwarf every other zone). An opened zone not loaded in this session has no rail and its
totals come from a small summary written by the pipeline (`data/packs/zones.json`: each zone's
trips a day by purpose with no rail). A click on one selects the zone and draws one bubble per
other zone for the commutes and longer trips between them, as T-186 does for cities.

**Selection, the inspector and loading.** A non-city zone's city inspector is a city's. Search finds
zones by name or alias (section 2, "Names"). Loading treats a zone as a city: opened when viewed or reached, its files fetched on
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
adapters, the border factors. The joint commute gravity spans countries: a pair across the US-Canada border takes the national border factor, and
the market's longer trips across any border T-376's factor (0.26 times the origin's income factor
to the power 1.5). A world with separate clusters of zones (the Northeast, Toronto, London, Tokyo)
needs nothing special: the commute gravity has no pairs between them, the market's longer trips
between them are place to place, and clusters merge only through track (T-362 decision 6).

## 8. Performance

- **Opening.** A countryside zone is a pack the size of a small city, opened only when viewed or
  reached, like a city; commute packs open from shipped factors.
- **A zone's market pack** is opened only once the zone is reached or selected, or a station lies
  within 30 km of its box, never for viewing alone (T-396, T-377, T-801; demand-runtime.md), so the
  market a worker holds grows with what is reached, not with the registry or the pan. Its pairs are built only between zones a station serves, so a solve's pair loop grows
  with the network's reach; pairs to every other place come from the market rows (one row per
  zone, about 0.9 KB, `game/market-rows.json`), which every solving worker holds once a market pack opens.
  Cross-zone commute pairs add a second pass over served pairs to the market's per-period work.
- **The joint gravity** is offline only (`joint_gravity`); a runtime edit never reruns it.
- **Startup** must not grow with the number of zones beyond the registry index (T-776): a zone adds
  about 310 bytes to the bundled `registry.json` (what the map, the zoomed-out bubbles, the new game
  sheet's search and the save header need before it opens) and about 0.9 KB to the market rows,
  which the page downloads only with the first market pack; its files are fetched
  on demand (section 6). A mask installed under drawn track reworks the water and price of only
  the track within its box; the others' are kept as they were (T-1039). A zone opened only for its bubbles opens bare in the auxiliary worker
  (T-325), its everyday layer once rather than in every solving worker.
- **Closing** (T-301; demand-runtime.md "Which cities are open"). A zone nobody views, sees beside the view
  or reaches closes after 60 s, or at once beyond two such zones, so a long pan holds a few zones'
  packs at a time, not every zone it passed.

