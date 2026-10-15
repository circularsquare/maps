# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Direction** (Anita, 2026-10-14; details in CLAUDE.md "Working here"). Order of work:
1. London stage b (T-360, done 2026-10-15); 2. sampler cities Paris, Shanghai, Mumbai (T-289, T-291,
T-292), from tonight, one at a time; 3. park-and-ride (T-380 parking check, then T-372 and T-374;
T-370, T-371, T-389, T-373 done); 4. worldwide long trips (T-362: T-375 to T-379); 5. the crossover tool (T-338);
6. responsiveness leftovers (T-326, T-327 done); 7. a demand-realism batch (T-270, T-271, T-272);
8. a design for game modes, not yet a task. Performance is open work from 2026-10-15: measured,
ranked by impact and by what lets the game scale (notes/perf-survey.md). Anita plans hundreds of metros: cost per city matters (notes/city-costs.md).

**Where things stand (2026-10-14).** One save spans the Earth (Mercator frame, local time); the
Northeast (nine cities, four county zones, commutes between them), Toronto and London (tier 0)
play. Demand workers are sized to the machine (T-381). History is in ARCHIVE.md.

**Overnight handoff (2026-10-14, for a fresh orchestrator).** Anita approved, for tonight: Paris,
Shanghai and Mumbai, one city agent at a time (Opus), tier 0 per docs/adding-a-city.md, each
ending with its retrospective; record each city's line in notes/city-costs.md and judge its
proposals (adopt the cheap clear ones, decline others with a reason, leave edge cases for Anita
under "Added by agents" in TODO.md). **CPU cap tonight about 10 cores** (back to 6 when she is
up); auto mode. Before the first city, **T-363** (read a region's OSM once) is worth doing: it
saves about an hour of CPU per city. **Disk**: about 32 GB free on 2026-10-14; country extracts
are 1.5-4.5 GB; delete each city's raw downloads once it is verified, and England's .osm.pbf once
T-360 is in ARCHIVE.md. T-360 (London stage b) finished and is committed: tier 0 has new jobs blend and curve rules and `population`, `jobs`, `schools` hooks (notes/T-288.md stage b); its lesson is that tier 0's core jobs and the curve's published median are the weak points to check in each city. Nothing else is in flight. **After each city**: `node trainworld/checks/smoke.mjs`, commit explicit paths, then
rebuild 8800: `git -C maps/.tw-snap checkout --detach <commit>`, then with
`CARGO_TARGET_DIR=maps/.tw-snap-target`, `CARGO_BUILD_JOBS=3` and `%USERPROFILE%\.cargo\bin` on
PATH, `npm run build --prefix maps/.tw-snap/trainworld/app` (a few minutes; background it).

## In progress (batch 22, 2026-10-13)


## In progress (batch 23, 2026-10-14)

- T-303 [P2] Demand loading and responsiveness. Stage 1 (measure and propose, notes/T-303.md) done; the whole proposal approved by Anita 2026-10-14 ("the proposed loading fixes make sense to me, lets try it out"). Built as T-324 to T-328, in that order.

- T-321 [P3] Local time's rest (found in T-309, docs/world.md "Local time"): the money ledger's days are the game clock's (New York's), not each region's; the demand bubbles' catch-up over a day (map/catchUp.ts) follows the clock, not the local time in view; the regional market has no trips between zones of different offsets yet, and when it does (London to Paris), a trip's period needs a rule (its origin's local time). A second offset now has demand: London, UTC+0 (T-288 stage a, 2026-10-14), so the money ledger's days and the catch-up now differ from local time there. Open.




## In progress (batch 25, 2026-10-14)


- T-355 [P2] Startup CPU headroom (found in T-348): in the first 10-14 s after a load (packs opening, basemap tiles parsing, 5 solving workers at full speed, other work on the box) the main thread is descheduled: long tasks of 50-230 ms wall with 3-7 ms of CPU. With 3 solving workers the long tasks about halve, but the final lands at 26-28 s instead of 19-20 s. Anita 2026-10-14: do not rely on using all of a machine's cores; be respectful of players' hardware and run reasonably on older devices. Folded into T-361. notes/T-348.md.
- T-361 [P2] Hardware budget, stage 1: measure and propose (Anita 2026-10-14, see T-355). Today 5 solving workers plus aux plus clock on a 16-thread box, about 354 MB of WASM each and 2.25 GB in all after a pan. Propose a default worker count and memory budget for a modest machine (4 cores, 8 GB), with timings on the real save at 1, 2 and 3 workers, and what it would take to fit (smaller memory per worker, sharing, fewer periods per worker). Design to Anita before building. Measured and proposed 2026-10-14 (notes/T-361.md); waiting on Anita's decisions there. Item 2 done (T-398: a worker's native peak on the real save 353 to 256 MB).
- T-383 [P3] A cached cell-to-unit table per pack (census small area, district, 2 km zone) that every measurement script shares (T-360 retrospective: one script spent 10 minutes on 88k repeated cell-in-polygon lookups). Any tier 0 rule change re-measures on Toronto and London with it. Open.
- T-384 [P3] Paris population from INSEE Filosofi 200 m squares (2021 release) through tier0's `population` hook, in place of commune totals spread by GHSL; measure first (London's LSOA upgrade changed little). Open; notes/T-289.md.
- T-385 [P3] Paris schools from the education ministry's school directory with coordinates joined to pupils per school (primary, collÃ¨ge, lycÃ©e files) through tier0's `schools` hook. Open; notes/T-289.md.
- T-382 [P3] London universities from HESA (T-360 left it): HESA Table 1 enrolments by provider (CSV zip of several hundred MB, CC BY 4.0, hesa.ac.uk/data-and-analysis/students/table-1) placed at each provider's campuses, against tier 0's national-rate total (538k) and its spread; propose the university-city rule for tier 0. Open.
- T-362 [P2] A worldwide baseline for long trips between cities (Anita 2026-10-14: no bubble between London and New York or Toronto). The regional market is one offline pack for the Northeast's zones only (NHTS-fitted); every other zone has `market: -1`. Design a general baseline from global sources (population, distance, borders, maybe air traffic) that covers every pair of zones, with rail's share earned by the network. Design first; research/intercity-review.md is the background. Design stage taking, 2026-10-14; notes/T-362.md.
  - T-437 [P3] Long trips' modes, after T-378: the 0.025 logit has no distance scaling (Paris - Berlin's car 0% beside the plane, real about a tenth; rail's share curve steeper than the studies'), the Channel crossing's price and ferries, and the modes of trips with places that have no zone (shown as all modes only). Proposed; notes/T-378.md.
  - T-438 [P3] The market's destination choice: attraction is people, so a giant metro takes most of a smaller neighbour's 80-300 km trips (New York - Philadelphia 301k a day over 80 km, about 2.5-6x plausible; Washington - Baltimore 7.5x NEC FUTURE). A competing destinations term or a leisure attraction, fitted worldwide (NEC FUTURE pairs plus one non-US survey); calibration, so it waits on Anita's word. Proposed; notes/T-379.md.
  - T-427 [P3] Longer-trip answers sized to what has figures: a selection's and a station's answers lay out every 6 km zone of the world (5,894 now, about 600 kB a selection per worker); with hundreds of zones send only the zones with trips. Open; notes/T-377.md.
  - T-428 [P3] `data/packs/intercity.json` is now only the joint commute gravity's input and output (14 MB in the packs folder, never read by the game since T-377): move it to data/work (`joint_gravity.rs`'s path and `market_zones.py`) and keep it out of what is published. Open.
  - T-429 [P3] Cross-border long trips in Europe: T-376's one border rule gives London - Paris 9,600 trips a day both ways (Eurostar alone about 30,000) and Paris - Berlin 4,300; a factor per country pair (EU, Schengen, the Channel Tunnel) with country tailoring. Open; notes/T-377.md.
- T-356 [P3] A move far from the draw anchor (T-307) brings a whole clock state: the world is rebuilt on the main thread (labels, complexes, network buffers, every list), about 50 ms of main thread per long pan (T-348). Rebuild only what depends on the anchor. Open.
- T-354 [P3] Two smaller main-thread costs (T-348): a pack opening in every worker is copied for each worker in one task (45-70 ms for New York, demandClient `post`), and a solve's final publish takes 70-110 ms (three kinds of day, purposes, bubble sums). Spread each over several tasks. Open.
- T-357 [P3] After T-352's sea outlines (the outside world part is gone with T-377: normalisers run over the world field). Toronto's outline still hugs Lake Ontario (StatCan cartographic subdivisions); a lake-line source if it ever looks wrong beside the US cities. Open.
- T-359 [P3] After T-351: a bare start's card waits 2.0-2.2 s for New York's bare opening in the aux worker alone while the solving workers sit idle (they open only the market); opening it bare split over the pool could bring the card to about 1-1.5 s. Also the in-game "Start a new game" still solves the viewed zone in every worker (already open there). Open; notes/T-351.md.

## In progress (batch 24, 2026-10-14)


- T-344 [P3] A zone's loading bar can stick near 99% (found in T-343): the viewed zone opened in every worker but never solved, because the view moved to dead space before its first solve; its grey and bar stay until something solves it. Rare. Open.


- T-346 [P3] After T-328, a reload with the cache on screen hides the fresh solve's estimates, so bubbles for a purpose other than all, or for the weekend (not cached), wait for the final at about 20 s instead of the first estimate at about 11 s. Show such views from the first estimate, or cache them. Also untested in a browser: a reload after deletions renumber ids (the key is built to cover it). Open; notes/T-328.md.

- T-335 [P3] Split docs/demand.md (about 900 lines) by area like the interface docs (sources and purposes, timing, mode choice and walking, the kernel and its costs), cutting its measurement trails to notes/. Found in T-329. Open.

- T-330 [P3] Line chips without a symbol show the first letter of the name, so in the real save every subway chip reads "S" and PATH "P" (seen in T-323). Default a chip to the line's short name (`lineShortName` in map/networkTips.ts, T-305: "subway 6X" to "6X"), cut to fit, when the player has set no symbol. Open.


## Demand and data
- T-283 [P3] Toronto's Yonge test line lost all its airport riders once routes were added (567 to none, found in T-277): check whether routed walks or drives cut Pearson's terminals off from stations, or whether airport trips now drive. May affect US airports too. Open; notes/T-277.md.

- T-279 [P2] Routes after the zone changes (from T-275): `pipeline/routes.py nyc` (8 minutes) once T-276's New York rebuild (Mercer) is final, and for any new pack (Worcester `wor`, Toronto `tor`) and after any gravity re-solve, joint gravity included: a pack whose zones changed has its routes ignored (straight x 1.3). `routes.py --check` lists them. New York, Worcester, Windham, New London and Litchfield done in T-276 (2026-10-13, `routes.py --check` all current). Left: routed drives for the regional market and between packs (T-276's zones; its car is still the great circle x 1.25): `routes.drive_factors(lon, lat, extracts)` gives them. Open; notes/T-275.md.
- T-295 [P2] Park-and-ride (from T-276): commuters between zones and longer trips reach stations only by the fast walk, so on a test network of five real commuter corridors across zone lines rail carries a third to a fifth of their real riders (notes/T-276.md). A drive-to-station access leg for the regional layer at least (SPEC 5.3; T-034 found it doubled suburban rail in the local layer, so the local layer is a separate question). Anita's design input 2026-10-14 in notes/T-295.md: driving (or bus, bike) to a station for both layers, the access cost flattening as driving takes over, then park-and-ride against driving the whole way. Design stage done 2026-10-14, five decisions for Anita at the top of notes/T-295.md; stages below wait on them.
  - T-440 [P2] T-380's parking rule (`Params::park_parts`, the two-logit mix, cordons) for the everyday layer (`everyday.rs` still has the old 3 + jobs/km² of the square / 2,500 inline) and the regional layer (`pipeline/intercity.py` bakes the old rule into `site_park`; rebuild the market pack). After T-433. Small. Open; notes/T-372.md.
  - T-390 [P3] After T-374: the walked/drove split of a station's riders covers commutes only. Extend it to everyday trips (`everyday.rs` `pass`: tally the home end's entries by walk or ride) and the regional layer (`intercity.rs`), so it matches the station's whole boardings, and then show it in the station hover and draw riders' home zones by how they arrive on the catchment. Small to medium. Open; notes/T-374.md.
- T-296 [P3] A station's far-zone rows (`IntercityApi::station_flows_day`) hold commuters between zones and longer trips together and take the longer trips' weekend factor (1.58), so a weekend overcounts the commuters there (0.30); keep the commute part apart if it shows. Open; docs/demand.md 4.4 "The market since T-276".
- T-298 [P3] Inspectors and longer-trip bubbles show the commutes between zones merged into the longer trips (a selection's far end, the city inspector's places, the far-zoomed bubbles, labelled "longer trips" in places): split them or relabel, as Anita prefers once she has seen them. Also the far-zoomed bubbles: scaled to the busiest zone on screen, so with New York in view (zoom 7) the small zones are dots and Litchfield and Windham lose their labels; a floor or a separate scale for non-city zones. Open; notes/T-276.md.
- T-280 [P3] Ferries (found in T-275): with walks blocked by water, Staten Island's commute rail share fell from 2.9% to 1.9% (ACS 3.7%), since its real link to Manhattan is the Staten Island Ferry; NY Waterway, Boston's harbour ferries and others likewise. A ferry as an access leg or its own line kind. Parked (Anita 2026-10-13: nothing to do about ferries for now); notes/T-275.md.
- T-142 [P3] Continue non-work demand. Done 2026-10-12: all nine cities have everyday, school (NCES), university (IPEDS), airport (FAA, DB1B, OurAirports, OSM terminals) and medical (OSM) purposes. Remaining: private schools (NCES PSS, New York too) and other countries' adapters (airports already take a Wikidata fallback for passenger counts). P2 to P3 in T-259: the nine cities are done and other countries wait for city additions (paused). Open; notes/T-142.md.
- T-269 [P3] realnet's operator count is wrong (found in T-171): `realnet.py compare` counts first boardings at stations labelled with an operator, so PATH's Manhattan stations (merged with subway complexes) count as subway, riders reaching an operator by another train are missed, and commuter rail's returns from Penn and Grand Central are hidden. Count trips using each operator (boardings on its lines less changes between two of its lines) from route-node boardings and transfers, as notes/t171_net.rs.txt and t171_operators.py do. Open; notes/T-171.md 2.5.
- T-270 [P3] The drive across rivers and into dense centres is too cheap (found in T-171): no tolls or congestion charge, parking from the 2 km zone's job density (the WTC / Wall St zone takes in harbour, so 16 perceived minutes against Midtown's 43), no tunnel queues. The Hudson waterfront takes rail at 26.9% against ACS 40.9%; a 30-minute crossing stand-in lifts PATH commuters 125k to 185k. Worldwide-safe start: parking from jobs per kmÂ² of land (the pack has the water mask); tolls from OSM `toll=yes` on bridges and tunnels. Moves calibrated numbers. Open; notes/T-171.md 1.4.
- T-271 [P3] Commuter rail is over on commutes alone (found in T-171): trips using the LIRR 483k and Metro-North 336k a day against 268k and 236k real all-purpose trips; the everyday layer adds more. Bears on T-268's commuter rail between cities. Open; notes/T-171.md 1.2.
- T-272 [P3] The everyday layer's rail is about 1.4x the commute's on every operator (found in T-171): with it every operator carries 1.65-3.1x its real all-purpose count. Check everyday rail shares and volumes against NHTS or operator counts. Open; notes/T-171.md 1.2.
- T-273 [P3] Border factor fade near the line, k(d) = k + (1 - k) exp(-d / 3 km) (T-171's one judgement call): score-neutral on LODES in five packs, brings the Hudson waterfront to Manhattan from 0.67x to 0.92x LODES (about 10-15k PATH riders), but overshoots the shortest crossings elsewhere (Camden city to Center City 5.8x to 8.6x LODES). Anita decides; the change is in notes/T-171.md 2.4.
- T-058 [P3] Fit the commute timing and walking: the period shares of trips to work from ACS
  B08302 (time leaving home, by tract) and of trips home from NHTS; the walk curve from ACS
  B08301 walked by county. Both are judgement today (SPEC 4.2, 4.3). notes/T-026.md.
- T-059 [P3] Trips home chosen from the work end (the return direction's crowding in the choice) instead of the morning's station pairs reversed.
  notes/T-026.md.
- T-043 [P3] County-government and school-district head-office lumps in LODES (listed but left
  alone in notes/T-014.md): check which are real, spread the proven ones as T-014 did.
- T-200 [P3] Boarding a train only to walk away: the demand graph's walking transfers start at route nodes, and a station boards onto a route node at no ride, so a strategy can "board" and at once walk to a station within 300 m, paying a boarding's wait instead of riding (seen by the trip explorer, T-195: Jersey City to Wall St, 0.8% of the trip's rail riders board the E at World Trade Center and walk to Fulton St without riding). Small, but it is a walk between stations priced as a boarding; walks should start only after a ride of at least one stop. Changes loads, so it needs the usual checks. Open.
- T-202 [P3] Rethink the 100 m station merge now that complexes are transfer points (T-161): the merge is single linkage, so a chain of platforms becomes one demand station at their mean position with free changes (the real NYC save's Penn Station, 24 platforms over 218 x 261 m, is 2 demand stations; 855 complexes are 873 demand stations). With complex walks costed by distance, merging only stations within 100 m of each other directly (or not merging at all) would give long hubs walking times between their platforms. Changes loads; needs the usual checks. Open.
- T-203 [P3] Medical sites mapped as multipolygon relations are skipped (found in T-196): `enrich_medical.py` queries with `out geom tags`, which gives relations no members, so a hospital campus mapped as a relation has no polygon and is dropped (the same bug cost T-196 Prospect and Pelham Bay Parks until fixed there). Refetch relations with `out geom` as `enrich_parks.py` does and re-enrich the nine packs; check how many hospitals and beds come back. Open.
- T-235 [P3] Re-run the park draw on T-234's population (found in T-234): `park_draw`'s nearby-residents term still counts people in prisons and nursing homes, because the Overpass park tiles were deleted after T-196 and the public servers answered 504 for ten minutes on 2026-10-13. The nine packs' total draw would fall only 0.2-0.6%, but small parks beside big prisons are overdrawn by up to about 4x (Bilton Road Open Space in Somers, Hartford pack; a place OSM calls "Mci Bridgewater" by the Bridgewater prison, Boston pack). `python pipeline/enrich_parks.py <city>` for each pack when Overpass answers (an hour for nine on a bad day; Springfield's first three tiles are cached in data/raw/t196/); parks do not touch the gravity. Open.
- T-240 [P3] Central Park may draw too few social and leisure visits (Anita: "seems a bit undercounted maybe but hard to get perfect"). Revisit with the park weights (T-196) or the 46/46/8 split when demand is calibrated. Open.
- T-216 [P3] Subway peak loads against measured (found in T-207): on the real New York save only 3% of morning-peak trains reach crush in the boarding account; demand carries 2.74M subway riders a day and T-090 had the busiest links at 76% of their measured busiest hour. Recheck the busiest subway links against the MTA busiest-hour figures after T-198, and whether the crowding penalty (alpha 2, starting at seats) holds peak loads too low. Demand calibration; worldwide note in docs/demand-worldwide.md if a rule changes. Open. Anita 2026-10-12: after the intercity rework.
- T-214 [P3] Intercity rework observations (Anita: "record this as something we noticed"). New Haven to Hartford shows about 1k trips a day; real CTrail Hartford Line carries about 2k riders a day and most people probably drive, so the real total is likely far higher. Collect such observations here or in notes/T-214.md for when the intercity market is redesigned. Open.
- T-262 [P3] Saturday apart from Sunday, and a holiday calendar per country with holidays run as Sundays (from T-170). `DAY_KINDS` is already threaded through the factors, the views, the boarding account and money; NHTS has the Saturday and Sunday rates in research/weekend/nhts_results.json. Also the per-country weekend days (Friday-Saturday in much of the Middle East). Idea; notes/T-170.md.
- T-263 [P3] Stadiums and arenas as points with bursts of demand on generated game dates (Anita: bursts, not consistent; alerts only where the player has built), one column of mode choice per venue added onto that day's riders, with a minute profile for the boarding account. Idea; notes/T-170.md.
- T-264 [P3] Superholidays (Chinese New Year, Golden Week, Obon, Chuseok, Thanksgiving) as long-distance kinds of day with a direction (out of the big cities, then back), after the intercity rework (T-214). Idea; notes/T-170.md.
- T-265 [P3] Signals simulated by clumps of shared track, with lookahead between clumps and a stored repeating pattern per demand level and kind of day (SPEC 2.1). Idea; notes/T-170.md.
- T-187 [P3] Trip chains: people often go from work to a nearby leisure place and then home, rather than to leisure from home. Whether and how to model it. Idea, not a priority (Anita 2026-10-12).

## Track and trains

- T-248 [P3] Since T-244 the track inspector's Construct and level buttons act on one piece of a drawn route; consider "this piece / the whole run between junctions and stations" if Anita finds constructing piece by piece tedious. Open.

- T-339 [P3] An express slowed by a local ahead of it on plain double track (Anita: "ideally express is slowed by local ahead of it but thats whatever"). Today trains pass through each other at no cost. Low priority. Open.
- T-340 [P3] Quad track's rest (from T-257): a station on a quad stroke's own end node is refused when that end is new track; the express pair is always left of the drawn direction. Open.

## Interface

- T-424 [P3] City checks' test-line stations from OSM (T-290 retrospective): keep `railway=station` points in osm_features' point cache and let `checks/city-browser.mjs` rows name stations that are looked up there, instead of hand-placed coordinates; and water test points by polylabel on the cached water polygons, skipping channels under about 50 m wide. Open.

- T-418 [P2] Track drawing like NIMBY Rails (Anita 2026-10-15): the track passes through every clicked point, a click never changes segments already placed (each new segment leaves the last point along its direction), two clicks place a segment by default and continuing needs another click or a held key, and the curve's circle shows faintly. Design and decisions in notes/T-418.md. Stage 1 done 2026-10-15: "Through points" mode beside "Corners" in the Build tab (through points by default), quad still by corners; stages 2 and 3 are T-422 and T-423.
- T-422 [P2] T-418 stage 2: after Anita has tried both drawing modes, settle the default and whether Corners stays (her decision: keep it until quad draws through points; it does since T-423, so Corners can retire), and tidy track-and-service.md 6.1 and ui-building.md to match. Waits on Anita's try.
- T-430 [P3] T-418's straight-then-arc fallback for a click too tight for one tangent arc (left out of T-423). Open idea, only if wanted.
- T-294 [P3] Fares across a period's hours (from T-284): each hour of a period is paid the period's fares spread evenly, while the boarding account's riders peak mid-period, so 07:00 earns as much as 08:00. Weight the period's hours by the account's boardings each hour, keeping the period's total (the worker already reads each hour's slice, `shapeHour`). Small; open; notes/T-284.md.

- T-166 [P3] A station's riders by purpose on the map: under the purpose filter a selected station's catchment shows commuters only (hidden for other purposes). Everyday and school catchments need per-purpose station queries from `EverydayApi`. Open; notes/T-158.md.
- T-039 [P3] Dark theme (SPEC 8).
- T-201 [P3] The trip explorer for the everyday purposes (T-195 shows a trip as a commute): the same query on `EverydayApi` (its walk share and drive distance from the cells, its own subzone averages), and a purpose choice in the panel. Needs a small accessor in everyday.rs. Open.
- T-057 [P3] Kana and kanji on the basemap in Zen Maru Gothic (for cities in Japan, M3): MapLibre draws them in the browser's
  system font today (they never come from glyph tiles). MapLibre 6's style `font-faces` can pin a
  self-hosted Zen Maru file to those codepoints, fetched only when such a label is drawn; check its
  size and the main-thread cost of drawing them. notes/T-056.md.

## Later: the world (M2-M3)

World sampler (T-286, research/world-sampler.md). All waiting for Anita's review of research/world-sampler.md; the shared stages first: T-126 and T-127 (a frame and renderer origin per world), then T-287 (T-041's OSM water is done).

- T-314 [P2] Tier 0's curve without flows: `tier0.fit_curve` fitted only the length to one published median and came out far too flat on Toronto (17.5% of commutes misplaced against 11.3% with Toronto's own curve, 22% fewer riders on a test line; notes/T-287.md). Both parameters are now fitted where a distance-band table exists (`tier0.fit_curve_bands`, Mumbai's B-28, T-292: d^-1.65 exp(-d/32.7 km), KL 0.017 against 0.023 for the best length at power 1.4; the surface is flat between powers 1.6 and 1.75, and 2 km zones cannot separate the 0-1 and 2-5 km bands). Open: write the rule for borrowing the most similar fitted city's curve, and whether a band table's road factor should be the pack's own measured drive factor (Mumbai's near pairs 1.62, against the 1.3 assumed). Open.
- T-315 [P3] Tier 0 universities: few OSM campuses carry a Wikidata id (4 of 63 in Toronto), so most are sized by area (+20% students in Toronto, 10.7% misplaced). Try matching OSM campus names to Wikidata universities with coordinates in the boundary, and splitting a multi-campus institution's count. Open.
- T-316 [P3] Tier 0 boundaries from global admin units (geoBoundaries ADM2-3, NUTS-3/ITL3) instead of a hand-made GeoJSON, and district population totals from the same units where a national source publishes them. Open.
- T-420 [P2] All of Japan (design): 7 metro packs of whole prefectures (Greater Tokyo about 100k cells, ~10 MB) plus one zone per other prefecture, census mesh population and economic census mesh jobs through a country module, in three stages. Design done, with Anita; research/japan.md. Stage 1 done 2026-10-15: `pipeline/japan.py` and Greater Tokyo (`tyo`, 98.6k cells, 36.9M people, 8.3 MB), registered outside the market; a dense synthetic network doubles a worker's peak (242 MB) on either pack; notes/T-420.md. Stage 2 (Keihanshin, Chukyo, Fukuoka, Sapporo, Sendai, Hiroshima; Japan joins the market) waits on T-377 and T-409; stage 3 (the prefecture zones) on T-297, T-301 and T-362's coupling.
- T-441 [P3] Japan pack tooling from T-432's retrospective: KSJ downloads in japan.py by curl or with a timeout per chunk (Python's request stalled for good on N03 05-07 while curl took seconds); `read_pack.py` fails a pack with campuses and 0 university students; one script for a batch's registry line, `zone_boxes.py` and `market_zones.py`. Open; notes/T-432.md.
- T-387 [P3] Tier 0 hospitals where OSM tags beds on almost none (Shanghai 1, Mumbai 1 of 1,827, Keihanshin 1 of 1,390 at 1,063 beds, T-431): every untagged hospital takes the one tagged median (Mumbai 2,844 beds each, so clinics get almost no medical visits). Size by building floor area or a per-country beds-per-resident total instead. Open; notes/T-292.md.
- T-388 [P3] Mumbai jobs by place of work: the 6th Economic Census (2013) has employment by district and Greater Mumbai by ward; through tier0's `jobs` hook if open, since the proxy gives Greater Mumbai 45% of jobs against 52% of the 2011 workers by residence. Open; notes/T-292.md.
- T-386 [P3] tier0's signals log prints GHSL totals for the whole box, which over sea include nodata (Shanghai's nres_v 34.7e12 against 1.6e9 inside); print the totals inside the boundary. Small. Open; notes/T-291.md.

- T-282 [P3] Toronto from the official StatCan files (found in T-277): StatCan's geography server (www12) answered a bot check, so the pack uses dissemination areas from ArcGIS copies and a gravity curve fitted on the lockdown month's 2021 flows. With Anita's downloads (dissemination block boundaries `ldb_000a21a_e.zip`, the 2016 commuting flows 98-400-X2016325; URLs in notes/T-277.md): finer grain, and a refit on pre-pandemic flows. Also the college partner campuses (about 16,000 students) left out of `pipeline/ca_campuses.csv`. Open; notes/T-277.md.

- T-358 [P3] Water between city packs for intercity track (M2, the half of T-041 not done): each pack's mask covers only its own box, so ground track between cities sees no water. `pipeline/osm_water.py` can make masks for any box (sea plus Geofabrik extracts); needs a world or corridor mask format and the app to read it. Open; notes/T-041.md.
- T-060 [P3] City packs on R2: versioned path, compressed, cached for good, one base-URL constant
  in `workers/demandClient.ts` (and the clock worker's pack URL); parse the header in JS so
  `serde_json` leaves the WASM. notes/T-026.md "Serving the pack from R2 later".
- T-011 [P3] Northeast long-distance data: NextGen NHTS OD zones, Amtrak NEC ridership by station
  pair if published, air routes from `flights`.
- T-012 [P3] GHSL non-residential layer as the tier-0 jobs proxy; compare with LODES WAC in New
  York to see how good a proxy it is before relying on it abroad.
- T-033 [P3] Draw track and trains under MapLibre's globe projection (world view): our shaders
  assume mercator, so under globe the network collapses to a speck at zoom 3 and vanishes at 6-10.
  Use `shaderData.vertexShaderPrelude` (`projectTile`) with per-tile projection data, keeping
  centimetre precision at street zoom. notes/T-018.md.

## Performance (open since 2026-10-15; ranked in notes/perf-survey.md)

- T-281 [P3] Opening a big save searches every station's walk round water once per worker (T-275): about 1 ms a station in WASM, 1.0 s for the real New York network's 867 stations (53 ms before); edits are unchanged. A tighter search (padded window without bounds checks, reused buffers) or a 100 m raster for the far part would roughly halve it. notes/T-275.md.
- T-076 [P3] Playing costs 13-15% of a core on any network, nearly all of it Chrome presenting 60
  frames a second (GPU process 9, renderer 4; our draw 0.1-0.2 ms). A lower frame rate when the
  fastest train on screen moves under about half a pixel a frame would roughly halve it. A
  smoothness judgement for Anita; measure the visible step first.
- T-070 [P3] Edits on 10,000 km networks: 23-36 ms of main thread an edit plus deserialising the
  whole state (round trip 130-214 ms). Send what an edit changed (edges, nodes, lines by id)
  instead of the whole state, keep stroke buffers per edge in the worker, upload sub-ranges.
  notes/T-063.md.
- T-439 [P3] Commute pair loop after T-036 (about 37 ns a pair native): the scatter into the y rows (10 ns, scalar to keep each station's sum in order), the zone walk (6 ns) and the per-origin-zone decay row (`fz`, 4% of a solve, rebuilt every period and round) are what is left; the everyday pair loop (10.7%) has the same shape and could take `lanes.rs`. Open; notes/T-036.md.
- T-168 [P3] Add a newly viewed city without re-solving the whole network: a city with no construction needs only its own walk and drive figures, but a pan to one re-solves everything (seconds on a big save, in the background). Open; notes/T-149.md.
- T-419 [P3] A worker's set-up state after T-398: New York's everyday `edge_pair` (28 MB) and pairs (20 MB) and the market's set-up (23 MB: pairs, index, access lists) are now the largest fixed items. Open; notes/T-398.md.
- T-414 [P3] Set-up order and the first solve in WASM (found in T-405): set up in registry order, each worker's first round-0 solve of the real save is 2-3 s slower in the browser (New York's commutes 5-6 s against 2-3 s), same heap size, nothing natively; set up in pack-open order and solved in registry order it was fastest (29.3 s). Find why (likely first-touched memory) and whether a set-up order rule helps every save. Open; notes/T-405.md.
- T-433 [P2] Everyday assignment in one pass for all purposes (found in T-401): the layer assigns once per purpose, 8.6% of a New York solve (12.6% on a 2,000-station network); carry a flow per purpose through one walk of each destination's strategy. Open; notes/T-401.md.
- T-434 [P2] How much the crowding rounds change (found in T-401). Measured 2026-10-15: round 3 still moves New York visibly (fullest train 124% to 96% of crush, 11 lines over 2%), so 2 rounds is out and no other averaging saves a round; proposed: each period stops once a round moves its loads under 1% (peaks keep 3), 15-20% less solve work, no visible change. Approved by Anita 2026-10-15: build it (per period, 1% tolerance, at most 3 rounds). notes/T-434.md.
- T-436 [P3] Start an edit's crowding rounds from the solve before instead of free flow (found in T-434): might need one or two rounds after a small edit, but figures would depend on edit history (an undo would not restore them). Not measured; Anita decides whether that is acceptable. notes/T-434.md.
- T-435 [P3] Market zone pair loop (found in T-401): 7.9% of a New York solve, f64 exp and a random S x S lookup per access pair, about 165 ms a pass; f32 exp and skipping pairs with no finite rail cost should roughly halve it. Open; notes/T-401.md.
