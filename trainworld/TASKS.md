# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Direction** (Anita; details in CLAUDE.md "Working here"): content and gameplay first.
Performance work stays parked until she reports a concrete problem; city additions are paused
(nine is enough for now).

**Where things stand (2026-10-13).** M1's New York sandbox is complete. M2 is under way: nine
Northeast cities play in one save with the regional market (T-116, T-117, T-134, T-137), and
batches 10-19 were Anita's feedback from playing. Batch history is in ARCHIVE.md.

## In progress (batch 22, 2026-10-13)


## In progress (batch 23, 2026-10-14)

- T-303 [P2] Demand loading and responsiveness, stage 1: measure and propose (Anita: "demand loading performance is definitely becoming an issue"; bubbles slow in the first minute while demand re-solves; T-122's frozen copies rejected as too much memory). Measure startup, panning into zones, memory per worker, and clicks during the first minute; propose what to compute first (what is on screen, what the player is likely to look at) and what to leave until asked; folds in T-301 and T-302. A proposal for Anita, no build. Taking, 2026-10-14 (batch 23).

- T-306 [P2] One world, stage 1a, the track model (docs/world.md): the Mercator frame in Rust and TS, ground lengths per piece, metre figures scaled by k, the TWT6 save, convert_frame to Mercator, the 85-degree and antimeridian refusals. Taking, 2026-10-14 (batch 24).
- T-317 [P2] One world, stage 1b's rest: the demand views follow the draw anchor (T-307 left them on `HOME_ANCHOR`, map/geo.ts). The commuter bubble workers take the anchor with their cells and rebuild on its version; the catchment, longer-trip and bubble-selection discs, the hover finders' pointer and the demand canvas's matrix follow; MERC_K in bubbles.ts, commuterView.ts, demandViews.ts and intercityView.ts becomes k at the disc. Needed once a pack far from New York (London, T-288) has bubbles: on the home anchor they are about 16 cm coarse there and wobble a pixel or two at zoom 18. Open.
- T-318 [P2] Names and loading outside registered cities (found in T-307): in a London game the top bar named the place "Boston" (`cityNear` picks the nearest registered city at any distance) and a new world-frame game showed a "Loading data" card there. Name a place only within reach of a city or zone, and load nothing for dead space. Open.
- T-308 [P2] One world, stage 2: convert old saves to the world frame on load (Anita: yes; and where something moves substantially and is easy to correct, correct it). After T-306. Open.
- T-309 [P2] One world, stage 3: local time per region, as NIMBY Rails does (Anita): the clock shows the local time where the player is looking, and each region's demand runs on its own local time. Before London's demand means much. Open.

- T-313 [P2] Station dots (Anita 2026-10-14), zoomed out below 15: dots 65% bigger in radius; one dot per station (no more growing a dot to cover a complex's platform centres), the dots of a complex merged under one black outline the way station boxes are, with the connector between dots that do not touch; the connector made of the same material as the dots (same outline thickness, its white continuous with the dots' white), for player links too. Taking, 2026-10-14 (batch 25).

## Demand and data
n- T-283 [P3] Toronto's Yonge test line lost all its airport riders once routes were added (567 to none, found in T-277): check whether routed walks or drives cut Pearson's terminals off from stations, or whether airport trips now drive. May affect US airports too. Open; notes/T-277.md.

- T-279 [P2] Routes after the zone changes (from T-275): `pipeline/routes.py nyc` (8 minutes) once T-276's New York rebuild (Mercer) is final, and for any new pack (Worcester `wor`, Toronto `tor`) and after any gravity re-solve, joint gravity included: a pack whose zones changed has its routes ignored (straight x 1.3). `routes.py --check` lists them. New York, Worcester, Windham, New London and Litchfield done in T-276 (2026-10-13, `routes.py --check` all current). Left: routed drives for the regional market and between packs (T-276's zones; its car is still the great circle x 1.25): `routes.drive_factors(lon, lat, extracts)` gives them. Open; notes/T-275.md.
- T-295 [P2] Park-and-ride (from T-276): commuters between zones and longer trips reach stations only by the fast walk, so on a test network of five real commuter corridors across zone lines rail carries a third to a fifth of their real riders (notes/T-276.md). A drive-to-station access leg for the regional layer at least (SPEC 5.3; T-034 found it doubled suburban rail in the local layer, so the local layer is a separate question). Open.
- T-296 [P3] A station's far-zone rows (`IntercityApi::station_flows_day`) hold commuters between zones and longer trips together and take the longer trips' weekend factor (1.58), so a weekend overcounts the commuters there (0.30); keep the commute part apart if it shows. Open; docs/demand.md 4.4 "The market since T-276".
- T-297 [P3] The joint commute gravity takes 12 minutes (500 Furness iterations on 23k zones, stopping at 2e-4 on one Boston zone): a faster convergence (over-relaxation, or a tolerance per zone size) would make `pipeline/intercity.py` quicker to rerun. Open; notes/T-276.md.
- T-298 [P3] Inspectors and longer-trip bubbles show the commutes between zones merged into the longer trips (a selection's far end, the city inspector's places, the far-zoomed bubbles, labelled "longer trips" in places): split them or relabel, as Anita prefers once she has seen them. Also the far-zoomed bubbles: scaled to the busiest zone on screen, so with New York in view (zoom 7) the small zones are dots and Litchfield and Windham lose their labels; a floor or a separate scale for non-city zones. Open; notes/T-276.md.
- T-299 [P3] The market's trips between neighbouring packs (New York - Philadelphia 380k a day, Washington - Baltimore 299k) include short trips across their seam since the 25 km onset; NEC FUTURE's metro-to-metro figures are 57k and 6k. Compare the market's trips over 80 km and centre to centre with NEC FUTURE before tuning anything. Open; notes/T-276.md.
- T-280 [P3] Ferries (found in T-275): with walks blocked by water, Staten Island's commute rail share fell from 2.9% to 1.9% (ACS 3.7%), since its real link to Manhattan is the Staten Island Ferry; NY Waterway, Boston's harbour ferries and others likewise. A ferry as an access leg or its own line kind. Parked (Anita 2026-10-13: nothing to do about ferries for now); notes/T-275.md.
- T-142 [P3] Continue non-work demand. Done 2026-10-12: all nine cities have everyday, school (NCES), university (IPEDS), airport (FAA, DB1B, OurAirports, OSM terminals) and medical (OSM) purposes. Remaining: private schools (NCES PSS, New York too) and other countries' adapters (airports already take a Wikidata fallback for passenger counts). P2 to P3 in T-259: the nine cities are done and other countries wait for city additions (paused). Open; notes/T-142.md.
- T-269 [P3] realnet's operator count is wrong (found in T-171): `realnet.py compare` counts first boardings at stations labelled with an operator, so PATH's Manhattan stations (merged with subway complexes) count as subway, riders reaching an operator by another train are missed, and commuter rail's returns from Penn and Grand Central are hidden. Count trips using each operator (boardings on its lines less changes between two of its lines) from route-node boardings and transfers, as notes/t171_net.rs.txt and t171_operators.py do. Open; notes/T-171.md 2.5.
- T-270 [P3] The drive across rivers and into dense centres is too cheap (found in T-171): no tolls or congestion charge, parking from the 2 km zone's job density (the WTC / Wall St zone takes in harbour, so 16 perceived minutes against Midtown's 43), no tunnel queues. The Hudson waterfront takes rail at 26.9% against ACS 40.9%; a 30-minute crossing stand-in lifts PATH commuters 125k to 185k. Worldwide-safe start: parking from jobs per km² of land (the pack has the water mask); tolls from OSM `toll=yes` on bridges and tunnels. Moves calibrated numbers. Open; notes/T-171.md 1.4.
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
- T-257 [P3] Quad track follow-ups (T-255): crossovers between the pairs mid-stretch (today lines change pairs only at a quad stroke's ends); a station on a quad stroke's own end node is refused when that end is new track (two pairs leaving one side of a node with nothing behind, "one-sided"); the express pair is always on the left of the drawn direction (draw the other way for the right). On hold (Anita 2026-10-13: express and quad need more design thought; no further work until she says).

## Interface

- T-294 [P3] Fares across a period's hours (from T-284): each hour of a period is paid the period's fares spread evenly, while the boarding account's riders peak mid-period, so 07:00 earns as much as 08:00. Weight the period's hours by the account's boardings each hour, keeping the period's total (the worker already reads each hour's slice, `shapeHour`). Small; open; notes/T-284.md.

- T-166 [P3] A station's riders by purpose on the map: under the purpose filter a selected station's catchment shows commuters only (hidden for other purposes). Everyday and school catchments need per-purpose station queries from `EverydayApi`. Open; notes/T-158.md.
- T-039 [P3] Dark theme (SPEC 8).
- T-201 [P3] The trip explorer for the everyday purposes (T-195 shows a trip as a commute): the same query on `EverydayApi` (its walk share and drive distance from the cells, its own subzone averages), and a purpose choice in the panel. Needs a small accessor in everyday.rs. Open.
- T-057 [P3] Kana and kanji on the basemap in Zen Maru Gothic (for cities in Japan, M3): MapLibre draws them in the browser's
  system font today (they never come from glyph tiles). MapLibre 6's style `font-faces` can pin a
  self-hosted Zen Maru file to those codepoints, fetched only when such a label is drawn; check its
  size and the main-thread cost of drawing them. notes/T-056.md.

## Later: the world (M2-M3)

World sampler (T-286, research/world-sampler.md). All waiting for Anita's review of research/world-sampler.md; the shared stages first: T-126 and T-127 (a frame and renderer origin per world), T-041 (OSM water), then T-287.

- T-288 [P2] London (first city of the sampler): tier 0 plus England's upgrades (Census 2021 OA population, BRES jobs by LSOA, curve fitted on 2011 MSOA flows, GIAS schools, HESA), and tier 0 measured against them; Europe world, `market: -1`. Waiting for Anita's review of research/world-sampler.md. Plan approved 2026-10-14 (Anita); after T-287 and T-304. Measure tier 0 with `notes/t287_compare.py`'s tables (BRES for jobs): the first out-of-sample check of the jobs proxy's height power 1.5 and blend, which were set on Toronto (notes/T-287.md). Open.
- T-314 [P2] Tier 0's curve without flows: `tier0.fit_curve` fitted only the length to one published median and came out far too flat on Toronto (17.5% of commutes misplaced against 11.3% with Toronto's own curve, 22% fewer riders on a test line; notes/T-287.md). Fit both parameters where a distance distribution exists (Mumbai's census B-28 bands, for T-292), and write the rule for borrowing the most similar fitted city's curve (which city, checked against the published figure). Open.
- T-315 [P3] Tier 0 universities: few OSM campuses carry a Wikidata id (4 of 63 in Toronto), so most are sized by area (+20% students in Toronto, 10.7% misplaced). Try matching OSM campus names to Wikidata universities with coordinates in the boundary, and splitting a multi-campus institution's count. Open.
- T-316 [P3] Tier 0 boundaries from global admin units (geoBoundaries ADM2-3, NUTS-3/ITL3) instead of a hand-made GeoJSON, and district population totals from the same units where a national source publishes them. Open.
- T-289 [P2] Paris: tier 0 plus France's upgrades (Filosofi 200 m population, census jobs by commune, curve fitted on INSEE commune flows 2022, ministry school files); Europe world. Waiting for Anita's review of research/world-sampler.md.
- T-290 [P2] Berlin: tier 0 plus Zensus 2022 100 m population and BA jobs per municipality (Berlin itself split by tier 0), curve from London's or Paris's checked on Pendleratlas flows; Europe world. Waiting for Anita's review of research/world-sampler.md.
- T-291 [P2] Shanghai: tier 0 scaled to the 16 districts' 2020 census totals, curve length fitted to the 10.2 km mean commute, CAAC airport figures; its own world near 31 N; CJK basemap labels checked. Waiting for Anita's review of research/world-sampler.md.
- T-292 [P2] Mumbai: tier 0 with a large population share in the job proxy (informal work), curve fitted to Census 2011 B-28 distance bands, sub-district zones; its own world near 19 N. Waiting for Anita's review of research/world-sampler.md.

- T-282 [P3] Toronto from the official StatCan files (found in T-277): StatCan's geography server (www12) answered a bot check, so the pack uses dissemination areas from ArcGIS copies and a gravity curve fitted on the lockdown month's 2021 flows. With Anita's downloads (dissemination block boundaries `ldb_000a21a_e.zip`, the 2016 commuting flows 98-400-X2016325; URLs in notes/T-277.md): finer grain, and a refit on pre-pandemic flows. Also the college partner campuses (about 16,000 students) left out of `pipeline/ca_campuses.csv`. Open; notes/T-277.md.

- T-041 [P3] Water mask from OSM instead of TIGER, for cities outside the US (M3) and the world
  pack's intercity track (M2): coastline-derived sea polygons plus inland water, reproducing the
  T-030 checks in `read_pack.py`. The osmdata.openstreetmap.de sea file is one 906 MB global zip,
  so this wants a once-per-world preprocessing step. notes/T-030.md.
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

## Parked: performance (until Anita reports a problem)

- T-301 [P3] Zone packs never close (T-276): nearby zones' bubbles open each zone on screen in every solving worker (Philadelphia about 90 MB of WASM memory a worker, New Haven and Springfield together 57), so a long pan round the region grows every worker and every solve. Close packs that are neither viewed, nearby nor reached, or open nearby zones commute-only in the auxiliary worker. Parked until it matters; DECISIONS 2026-10-13 (T-276 agent).
- T-302 [P3] Zones' files on demand (docs/zones.md 6, 8): the clock worker loads every registered zone's water mask at startup and every boundary is fetched for the reach test and outlines; with four non-city zones that is four small masks more (0.04-0.15 MB), but a world of zones needs them fetched for the zones on screen or reached only. Measure startup with `notes/startup-timing.mjs` before and after. Open.
- T-281 [P3] Opening a big save searches every station's walk round water once per worker (T-275): about 1 ms a station in WASM, 1.0 s for the real New York network's 867 stations (53 ms before); edits are unchanged. A tighter search (padded window without bounds checks, reused buffers) or a 100 m raster for the far part would roughly halve it. notes/T-275.md.
- T-076 [P3] Playing costs 13-15% of a core on any network, nearly all of it Chrome presenting 60
  frames a second (GPU process 9, renderer 4; our draw 0.1-0.2 ms). A lower frame rate when the
  fastest train on screen moves under about half a pixel a frame would roughly halve it. A
  smoothness judgement for Anita; measure the visible step first.
- T-070 [P3] Edits on 10,000 km networks: 23-36 ms of main thread an edit plus deserialising the
  whole state (round trip 130-214 ms). Send what an edit changed (edges, nodes, lines by id)
  instead of the whole state, keep stroke buffers per edge in the worker, upload sub-ranges.
  notes/T-063.md.
- T-036 [P3] Faster pair loop: ~35 ns per subzone pair native against ~21 before the logsum. A
  packed per-destination layout (done in T-090) or wasm SIMD over egress stations could make
  10-minute walk bands (2.4x the pairs, every error a fifth smaller) fit the recompute budget.
  Since T-090 a third more origin subzones each build a row over all stations and visit the
  destination zones though few pairs survive there: about 10% of a stage. notes/T-020.md,
  notes/T-090.md.
- T-044 [P3] Adaptive cell grain, only if performance needs it: keep res 9 (or finer) in dense
  centres, merge sparse suburban cells (SPEC 3).
- T-124 [P3] Reuse the periods an edit did not touch: a change to one demand level's trains leaves
  the other levels' periods identical (their inputs are the same), so the workers could keep
  their results. Saves CPU, not time, while each period has a worker; helps the 1-3 worker pools
  of smaller machines. notes/T-119.md.
- T-178 [P3] Everyday layer memory per open city: a worker holding all nine cities' everyday layers uses about 496 MiB of WASM memory (278 MiB before T-142, 475 before its airports and medical visits), mostly destination samples and their pair indices (u32 each, cells x 16 for four purposes, x 2 for airport and medical). Only saves with construction in many cities pay it. Parked until it matters (Anita: performance only on report). Open; notes/T-142.md "Opening cost".
- T-168 [P3] Add a newly viewed city without re-solving the whole network: a city with no construction needs only its own walk and drive figures, but a pan to one re-solves everything (seconds on a big save, in the background). Open; notes/T-149.md.
