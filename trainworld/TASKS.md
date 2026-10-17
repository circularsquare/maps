# Tasks

Open agent work, one line a task; detail in `notes/T-###.md`. Format and rules in `CLAUDE.md`.
P1 = now or blocking, P2 = this milestone, P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Direction** (Anita; CLAUDE.md "Working here"). The whole world, with a demand model that is
sensible and makes building rail matter (2026-10-17: fun over fitting real cities; her two ceiling
tests are the targets, real cities loose checks; DECISIONS 2026-10-17, T-636). Next, in order: 1.
cities: T-614 and T-615 (thin-OSM parks), then T-613, the rest of Europe (72 rows), then the rest
of the world (432 rows); 2. demand: T-637 (everyday trips too long), T-634 (somewhere for short
trips besides the car); 3. interface: T-627, T-633, T-602, T-629; 4. long trips (T-528).
Performance is open work (notes/perf-survey.md). Game modes wait.

**Where things stand (2026-10-17).** 80 zones play, all shipping through the pack gate (T-538). The
unattended runner `pipeline/world_run.py` shipped its 5-city pilot (T-540), the UK batch (T-541)
and, after T-610 and T-611, Cardiff and Belfast; Owerri, Aba and Aboh park on T-614 and T-615.
Anita's plain page on demand: docs/demand-plain.md (keep it current). History is in ARCHIVE.md.

**For a task agent.** Check `git status` and the "taking" marks here first. Several agents work at once:
each owns named files, so leave others' alone. Anita's playtest notes land in TODO.md (hers; read it,
file new items as tasks, never edit it). Asks for Anita go in the final message or notes/for-anita.md.
CPU: 6 of 16 cores while Anita uses the machine, 12 overnight. Pack rebuilds: back up to
data/work/pre-T###/ first; `data/work/pre-T*` and `data/work/res9/` wait for Anita's acceptance.
`serve.py` (maps root) has an uncommitted one-line fix (listen backlog 128, T-490); 8800 runs it. When
the permission check refuses a step, make it safer, review, retry; don't wait for Anita. The
orchestrator's commit recipe is in CLAUDE.md "Before a commit".

## Demand and data
- T-632 [P3] Walking feels too strong to play against (Anita 2026-10-17): the model walks at or under the surveys; showing trips no train could win apart from the demand played against is parked (Anita: try later). notes/T-632.md.
- T-633 [P2] Explain perceived minutes in the game (Anita 2026-10-17: the walk weight stays; communicate it better): where the trip explorer and hovers show "counts as" minutes, say plainly why a walk counts for more. Open.
- T-551 [P2] Demand review for Anita (her playtest notes, 2026-10-16): measured, nothing changed (notes/T-551.md). Waits for Anita, who decides at the deep calibration.
- T-514 [P3] "Other" per cell (T-513): the sim keeps trips by other only as a total, so bubbles and hovers cannot show them by cell; needs other by home and work cell. Open; notes/T-514.md.
- T-637 [P2] Everyday trips run long: 17% under 2 km against NHTS's 43%, so everyday walking sits below the surveys (London social and education 18% against 41-44%) and rail takes errands a real network would not; check the everyday destination choice's distance against the surveys. notes/T-621.md "Stage 2", notes/T-636.md.
- T-638 [P3] Rebuild the market packs on the new parking base (they hold 3 minutes until rebuilt; about 2 minutes on long regional trips). notes/T-636.md.
- T-634 [P2] Short trips need somewhere to go besides the car (a bike mode, or "other" fitted for short trips) before walking is refit on the census and surveys: the fit's lost walkers mostly drive (Paris commute car 46% to 57%, census 36%). notes/T-621.md "Stage 2".
- T-635 [P3] Why Toronto wants more rail than the fitted cities (TTC platform counts, the 2021 lockdown census, worldwide everyday rates); a Canada rail constant only if it holds up. notes/T-621.md.
- T-528 [P2] Fit the regional layer's longer trips (T-509's last part, Anita's design 2026-10-16): rail constant and damping on corridor counts at two lengths per country. Open; notes/T-509.md section 15.
- T-529 [P3] Bikes to stations for car-free homes in Japan (T-509 stages 6-7): they only walk now, part of why Tokyo's commute rail is 35% against 52%. A bike access mode where a survey shows it. Open; notes/T-509.md sections 14-15.
- T-497 [P3] Japan's joint commute gravity, what is left: the pair spread is a quarter off the census (wants a travel-time cost), and the balance stalls at 8e-4 in Okinawa's islands. Open; notes/T-463.md.
- T-479 [P3] The Northeast's market sites in `data/packs/intercity.json` are slightly off the packs' cells after T-453's re-merge at 2,500. Rerun `intercity.py` at the next Northeast rebuild. Open; notes/T-479.md.
- T-461 [P3] Japan's jobs total uses census mesh column 023 (less public service, 3-5% of jobs); decide whether packs count public service, and fix `japan.census_jobs`'s off-by-one comment. Open; notes/T-456.md section 8.
- T-454 [P3] Destinations that respond to the network (rail pulling everyday trips toward places it serves): parked by Anita 2026-10-16.
- T-522 [P3] Everyday survey gaps (T-509 stage 4): France's medical rate and lengths from EMP 2019, NHTS 2022 for the US, London's social scale. Open; notes/T-509.md section 12.
- T-437 [P3] Long trips' modes after T-378 and T-495: the long car's cost (Paris - Berlin car 0.2% against a real tenth), the Channel crossing, ferries, places with no zone. Proposed; notes/T-378.md, notes/T-495.md.
- T-438 [P3] The market's destination choice: a giant metro takes most of a smaller neighbour's 80-300 km trips, and big job centres draw too little for commutes; a competing destinations term. Proposed, waits on Anita; notes/T-379.md, notes/T-500.md.
- T-427 [P3] Longer-trip answers sent only for zones with trips (a selection's answer is about 600 kB per worker at 5,894 zones). Open; notes/T-377.md.
- T-428 [P3] `data/packs/intercity.json` (14 MB) is only the joint gravity's input and output, never read by the game since T-377: move it to data/work, out of what is published. Open; notes/T-428.md.
- T-429 [P3] Cross-border long trips in Europe: T-376's one border rule gives London - Paris 9,600 a day (Eurostar alone about 30,000); a factor per country pair. Open; notes/T-377.md.
- T-605 [P3] Move the measurement trails still inside the demand-*.md paragraphs (before/after tables, how a figure was reached) to notes/, one file at a time, as each is next edited. Open.
- T-283 [P3] Toronto's Yonge test line lost all its airport riders once routes were added (567 to none): are Pearson's terminals cut off from stations by routed walks or drives? May affect US airports. Open; notes/T-277.md.
- T-279 [P2] Routed drives for the regional market and between packs: its car is still 1.25 x the great circle (docs/regional.md 5.1); new packs need `routes.py <id>`. Open; notes/T-275.md.
- T-390 [P3] A station's walked/drove split covers commutes only; extend it to everyday trips and the regional layer, then to the hover and catchment. Open; notes/T-374.md.
- T-296 [P3] A station's far-zone rows mix commuters between zones with longer trips under the weekend factor 1.58, so weekends overcount the commuters (0.30); keep them apart if it shows. Open; notes/T-296.md.
- T-298 [P3] Inspectors and inter-region bubbles show commutes between zones merged into longer trips: split or relabel, as Anita prefers once she has seen them. Open; notes/T-276.md.
- T-280 [P3] Ferries (found in T-275): Staten Island's commute rail share fell to 1.9% (ACS 3.7%) with walks blocked by water; a ferry as access leg or line kind. Parked (Anita 2026-10-13); notes/T-275.md.
- T-142 [P3] Private schools (NCES PSS, New York too); the other non-work purposes are in every US pack and come from tier 0 abroad. Open; notes/T-142.md.
- T-269 [P3] realnet's operator count is wrong (found in T-171): it counts first boardings at labelled stations; count trips using each operator from route-node boardings and transfers. Open; notes/T-171.md 2.5.
- T-270 [P3] The drive across rivers and into dense centres is too cheap (found in T-171): no tolls or tunnel queues; start with OSM `toll=yes`. Moves calibrated numbers. Open; notes/T-171.md 1.4.
- T-271 [P3] Commuter rail is over on commutes alone (found in T-171): LIRR 483k and Metro-North 336k against 268k and 236k real all-purpose trips. Bears on T-268. Open; notes/T-171.md 1.2.
- T-273 [P3] Border factor fade near the line (T-171's one judgement call): the Hudson waterfront goes 0.67x to 0.92x LODES, Camden - Center City overshoots 5.8x to 8.6x. Anita decides; notes/T-171.md 2.4.
- T-058 [P3] Fit commute timing and walking: period shares from ACS B08302 and NHTS, the walk curve from B08301 by county; judgement today (SPEC 4.2, 4.3). Open; notes/T-026.md.
- T-059 [P3] Trips home chosen from the work end (the return direction's crowding in the choice) instead of the morning's station pairs reversed. Open; notes/T-026.md.
- T-043 [P3] County-government and school-district head-office lumps in LODES: check which are real and spread the proven ones as T-014 did. Open; notes/T-014.md.
- T-200 [P3] Boarding a train only to walk away: a strategy can board and at once walk to a station within 300 m, paying a boarding's wait; walks should start after a ride of a stop. Changes loads. Open; notes/T-161.md.
- T-202 [P3] Rethink the 100 m station merge now that complexes are transfer points (T-161): single linkage makes a platform chain one demand station with free changes. Changes loads. Open; notes/T-161.md.
- T-203 [P3] Medical sites mapped as multipolygon relations are skipped (`enrich_medical.py` uses `out geom tags`); refetch as `enrich_parks.py` does and re-enrich the nine packs. Open; notes/T-196.md.
- T-235 [P3] Re-run the park draw on T-234's population: `park_draw` still counts prisons and nursing homes, overdrawing small parks beside them up to 4x. Run `enrich_parks.py <city>` per pack when Overpass answers. Open; notes/T-234.md.
- T-240 [P3] Central Park may draw too few social and leisure visits (Anita). Revisit with the park weights (T-196) or the 46/46/8 split when demand is calibrated. Open; notes/T-240.md.
- T-216 [P3] Subway peak loads against measured (found in T-207): only 3% of morning-peak trains reach crush; recheck the busiest links against MTA busiest-hour figures and the crowding penalty. Calibration; open; notes/T-207.md.
- T-214 [P3] Intercity observation (Anita): New Haven to Hartford shows about 1k trips a day, real CTrail about 2k riders, so the real total is likely far higher. Collect such observations here. Open; notes/T-214.md.
- T-262 [P3] Saturday apart from Sunday, a holiday calendar per country, and per-country weekend days (Friday-Saturday in much of the Middle East). Idea; notes/T-170.md.
- T-263 [P3] Stadiums and arenas as points with bursts of demand on generated game dates (Anita: bursts, alerts only where the player has built). Idea; notes/T-170.md.
- T-264 [P3] Superholidays (Chinese New Year, Golden Week, Obon, Chuseok, Thanksgiving) as long-distance days with a direction, after the intercity rework (T-214). Idea; notes/T-170.md.
- T-265 [P3] Signals simulated by clumps of shared track, with lookahead and a stored repeating pattern per demand level and kind of day (SPEC 2.1). Idea; notes/T-170.md.
- T-187 [P3] Trip chains: work to a nearby leisure place and then home. Whether and how to model it. Idea, not a priority (Anita 2026-10-12).

## Track and trains
- T-623 [P3] A station cut inside an arc, or a ramp overlapping a curve, still shifts the ramp (T-622 holds ramps on straights only). Open.
- T-625 [P2] A line's train count set automatically from its headway (Anita's TODO 2026-10-17), as an option beside today's whole number per demand level. Open.
- T-631 [P2] Trains cost money, bought automatically as a line's fleet grows (Anita's TODO 2026-10-17): a price per train or car, sold back or kept when the fleet shrinks. Open.
- T-629 [P2] At-grade junctions and flat crossings marked with an icon (Anita 2026-10-17): fairly big on blueprints and while drawing, scaling with zoom only part way; smaller and quieter on built track; `crossings` in sim/src/track/net.rs lists flat crossings. Open.
- T-654 [P2] Clicking track that several lines share opens a chooser of those lines (the track view), instead of picking one (Anita 2026-10-17, asked before). Open.
- T-655 [P2] Draw a small ramp where track climbs from level 0 to 1 and a tunnel portal where it drops to -1: discreet but noticeable (Anita 2026-10-17). Open.
- T-656 [P2] Ground-level track drawn across water becomes a bridge by itself: up to level 1 before the water, back down after (ramps on land, docs/track-and-service.md "No track at ground level over water"), instead of refusing (Anita 2026-10-17). Open.
- T-657 [P2] Turnback capacity (Anita 2026-10-17): buildable ways to turn trains faster than reversing on a through platform (15 an hour): tail tracks (sidings beyond the platform) and loops, as real metros do for 2-2.5 minute headways; and whether a line's turnaround may go down to about 2 minutes (20 an hour on one platform, needs crews stepping back). Design first, then build. Open.
- T-248 [P3] The track inspector's Construct and level buttons act on one piece of a drawn route; consider "this piece / the whole run between junctions and stations" if piece by piece is tedious. Open.
- T-339 [P3] An express slowed by a local ahead of it on plain double track (Anita: "ideally... but thats whatever"); trains pass through each other at no cost. Low priority. Open.
- T-340 [P3] Quad track's rest (from T-257): a station on a quad stroke's own end node is refused when that end is new track; the express pair is always left of the drawn direction. Open.

## Interface
- T-646 [P2] When editing stops, a button to add every station on the track between two clicked stops, with a preview of the stops it would add and confirm or cancel (Anita's TODO 2026-10-17). Open.
- T-647 [P2] Loading shown while demand loads (Anita's TODO 2026-10-17; she asked for motion here, an exception to "no animations"): a small spinner in place of each line's crowding dot per demand level, a moving element on the top "loading demand" tag (easy to miss today), and one in a clicked bubble's panel while its far ends load. Open.
- T-649 [P2] Airports drawn as demand bubbles (same colour and zoom scaling) with a small black airplane at the centre, instead of fixed-size icons (Anita's TODO 2026-10-17). Open.
- T-650 [P3] The top demand bar drops the word "Train" before the ridership percent (Anita's TODO 2026-10-17). Open.
- T-651 [P2] Zoomed out, zones not opened look fully red while opened ones look yellowish from their green walkers (Anita's TODO 2026-10-17): the same as T-602, give unopened zones their walk split. Open.
- T-652 [P2] Station boxes about 50% wider (hard to see zoomed out, the waiting people vanish), and a full station's waiting bar may grow past the box (Anita's TODO 2026-10-17). Open.
- T-653 [P2] The mode colour blend turns an even three-way mix white, which is hard to see: blend part way towards a light grey (#ddd or so) in `modeMix` (Anita's TODO 2026-10-17). Open.
- T-626 [P2] Type a fare amount, not only step it (Anita's TODO 2026-10-17). Open.
- T-627 [P2] "Platform full at Nerul Village, high demand, 1 train waits at the ends" made no sense to Anita (seen at headways under 4 minutes): likely a line reversing on a through station's platform, which holds it for the turnaround plus 60 s (3 min + 1 = 4 min, 15 trains an hour; docs/track-and-service.md "Resources"): confirm, then say what limits and what helps (a shorter turnaround, a dead-end terminus with two platform tracks); every capacity alert names its kind plainly (terminus, platform, junction, shared track with other lines, single track, flat crossing) (Anita 2026-10-17); `LineInspector.tsx` FULL/TAKES, `capacityMarkers.ts`. Open.
- T-602 [P2] Walk split for zones not opened (rest of T-576): `zone_summary` (sim) gives a walk count per zone, `pipeline/zones.py` `local_trips()` stores it as `localWalk` in registry.json beside `localTrips`, `intercityView.ts` draws it green. Open.
- T-561 [P3] Page freezes at 1,500x or faster in a fresh game with a line in each of Toronto and Boston (T-550's agent); not reproduced on 2026-10-16's tree. Open until seen again with exact steps; notes/T-561.md.
- T-560 [P3] Crush load per country (found in T-552): every city's trains fill at `api::CRUSH_PER_CAR`'s 160 riders a 20 m car, stranding Tokyo's riders early; take published figures per stock from the pack. Open; notes/T-552.md.
- T-330 [P3] Line chips without a symbol show the name's first letter (every New York subway chip reads "S"); default to the line's short name ("subway 6X" to "6X"). Open. Detail in notes/T-330.md.
- T-430 [P3] T-418's straight-then-arc fallback for a click too tight for one tangent arc (left out of T-423). Open idea, only if wanted.
- T-294 [P3] Fares across a period's hours: each hour is paid the period's fares evenly while riders peak mid-period; weight hours by the boarding account, keeping the total. Small; open; notes/T-284.md.
- T-166 [P3] A station's riders by purpose on the map: under a purpose filter its catchment shows commuters only; needs per-purpose station queries from `EverydayApi`. Open; notes/T-158.md.
- T-543 [P2] Themes stage 2, dark theme legibility: the known spots are done 2026-10-16; open until Anita has played in dark. Spots not yet checked in dark are in notes/T-543.md.
- T-057 [P3] Kana and kanji on the basemap in Zen Maru Gothic: MapLibre draws them in the system font; try its style `font-faces` with a self-hosted file, checking size and main-thread cost. Open; notes/T-056.md.

## World, tier 0 and countries
Tier 0 and country sources: docs/demand-worldwide.md, docs/data-register.md. The unattended city pipeline follows T-535's design (notes/T-535.md "Stages").

- T-578 [P3] The other tier 0 packs (Paris, London, Sao Paulo, Japan, Toronto's test) carry OSM job rows from before T-460's land cap and T-572 to T-574's rules; rebuild each on current rules when next rebuilt. Open; notes/T-572.md.
- T-570 [P3] Regional homes without a car: DHS's HC_TRNS_H_CAR by survey region for the 51 DHS countries, and open census small areas where they exist (Mexico's INEGI). Open; notes/T-562.md.
- T-571 [P3] National no-car figures in `nocar.COUNTRIES` for the larger countries on T-562's GDP rule (Russia, Mexico, Iraq, Iran and others; Vietnam's 65% is too low): census share first, else a register. Open; notes/T-562.md.
- T-563 [P3] Fit an archetype A planned city's own curve from its country's flows at build time (England, France, Germany borrow now): a `curve` mode in tier0 and the key in `plan_hooks`. Open; notes/T-555.md.
- T-564 [P3] Country hooks T-555 could not carry: Korea's registration population and Canada's DA population and jobs as tier 0 hooks. Open; notes/T-555.md.
- T-594 [P3] `pipeline/intercity.py` runs the full Northeast solve on `--help` or any unknown flag; make it print usage instead.
- T-592 [P3] world_run reads the next region's extract while this one gates and joins (reading is the largest phase: Portugal 5.2 of 12.9 minutes). notes/T-540.md.
- T-614 [P2] aba and abo park on `university_rate` 0 (OSM maps no university in the outline): pass a pack with none (T-582 opens it), or take universities from another source. Before T-613. notes/T-610.md.
- T-615 [P2] owe and abo park on the browser check "carries every purpose" (the test line meets no medical or airport trips): pick a test line that reaches them, or relax it for thin packs. Before T-613. notes/T-610.md.
- T-616 [P3] The runner reports a rates-test error (a sim copy without `data`/`app` junctions) as a pack fault: report it as a gate error, and add the junctions to the `TW_SIM_DIR` recipe. notes/T-610.md.
- T-617 [P3] Rebuild gla, lis and pot on T-610's unit matching when next rebuilt (gla +5.6%; lis and pot move from plan polygons to OSM municipalities). notes/T-610.md.
- T-612 [P3] Runner: delete a parked row's pack files once its reason is recorded.
- T-613 [P2] The next runner batch, the rest of Europe from pipeline/world_plan.json (`plan --countries` first for the row count), after T-614 and T-615; publishing past the current set needs T-060 and Anita's word. notes/T-541.md retrospective.
- T-546 [P3] Refit Mumbai's curve with B-28's 0-1 and 2-5 km bands merged, as Sao Paulo's are (d^-1.5 exp(-d/23.3 km) against the shipped d^-1.65 exp(-d/32.7 km)). Open; notes/T-314.md.
- T-315 [P3] Tier 0 universities: few OSM campuses carry a Wikidata id (4 of 63 in Toronto), so most are sized by area; try matching campus names to Wikidata universities. Open; notes/T-315.md.
- T-547 [P3] Read GISCO's download clauses for NUTS 2024 (some files are non-commercial only) before publishing any pack on NUTS-3 units; fall back to geoBoundaries if barred. Open; notes/T-536.md.
- T-462 [P3] Japan zone tooling fixes from T-420 stage 3: `japan_zones.py plan` fails on mismatched municipality codes, `build` refits, `japan.municipalities` takes 53 minutes, Tokachi's students. Open; notes/T-420.md.
- T-548 [P3] Rerun `hospital_beds.py pool` after T-203 if it brings back tagged campuses, then re-enrich the packs (T-544's steps). Open; notes/T-387.md.
- T-388 [P3] Mumbai jobs by place of work from the 6th Economic Census (2013), through tier0's `jobs` hook if open; the proxy gives Greater Mumbai 45% of jobs against 52% of workers. Open; notes/T-292.md.
- T-386 [P3] tier0's signals log prints GHSL totals for the whole box, nodata over sea included; print totals inside the boundary. Small. Open; notes/T-291.md.
- T-503 [P3] Everyday destinations from sector totals by larger unit under T-456's fallback: Paris from INSEE Flores 2021, Shanghai from the economic census. Open; docs/data-register.md, notes/T-453.md 5.
- T-504 [P3] Seoul from KOSIS (needs a Korean ID; downloads work in a browser, so Anita downloads named tables): establishments and 2020 commuting by si-gun-gu, moving Korea from archetype C to A. Open; docs/data-register.md. Detail in notes/T-504.md.
- T-505 [P3] University students by institution from national files: France's ministry and Germany's Destatis GENESIS 21311, neither opened. Open; docs/data-register.md. Detail in notes/T-505.md.
- T-506 [P3] Brazil's registers for tier 0: INEP's school census for schools and DATASUS CNES for hospital beds (also bears on T-387); not checked. Open; docs/data-register.md. Detail in notes/T-506.md.
- T-507 [P3] Nigeria from GRID3 (population, schools, health facilities; not checked), possibly the region's general upgrade over GHSL and OSM; measure against Lagos first. Open; docs/data-register.md. Detail in notes/T-507.md.
- T-512 [P3] Car-free households below country level outside the US, England and Japan: France from INSEE's census housing base, a yearbook for Seoul, Shanghai, Toronto, Sao Paulo; add to `nocar.py`. Open; docs/data-register.md. Detail in notes/T-512.md.
- T-508 [P3] Overture Places as a second place source where OSM's completeness is under 0.3 (Sao Paulo, Shanghai, Mumbai, Lagos); count against T-456's official counts first. Open; docs/data-register.md. Detail in notes/T-508.md.
- T-383 [P3] A cached cell-to-unit table per pack that every measurement script shares (one spent 10 minutes on repeated cell-in-polygon lookups). Open; notes/T-383.md.
- T-384 [P3] Paris population from INSEE Filosofi 200 m squares through tier0's `population` hook, in place of commune totals spread by GHSL; measure first. Open; notes/T-289.md.
- T-385 [P3] Paris schools from the education ministry's directory with pupils per school, through tier0's `schools` hook. Open; notes/T-289.md.
- T-382 [P3] London universities from HESA Table 1 (CC BY 4.0, several hundred MB) placed at each provider's campuses, against tier 0's national-rate total (538k); propose the university-city rule. Open; notes/T-382.md.
- T-357 [P3] Toronto's outline still hugs Lake Ontario (StatCan subdivisions); a lake-line source if it ever looks wrong beside the US cities (after T-352). Open; notes/T-357.md.
- T-282 [P3] Toronto from the official StatCan files (found in T-277; their server bot-checks): with Anita's downloads, finer grain and a pre-pandemic refit; college campuses (16,000 students) left out. Open; notes/T-277.md.
- T-358 [P3] Water between city packs for intercity track (M2, half of T-041): each pack's mask covers only its own box; `osm_water.py` can make masks for any box, but a world or corridor format is needed. Open; notes/T-041.md.
- T-060 [P3] City packs on R2: versioned compressed path, one base-URL constant in `workers/demandClient.ts` and the clock worker, header parsed in JS. Open; notes/T-026.md.
- T-033 [P3] Draw track and trains under MapLibre's globe projection (our shaders assume mercator); use `shaderData.vertexShaderPrelude` with per-tile projection data. Open; notes/T-018.md.
- T-321 [P3] Local time's rest (T-309, docs/world.md "Local time"): the money ledger's days and the bubbles' catch-up follow the game clock, and market trips across offsets need a period rule. Open; notes/T-321.md.

## Performance (open since 2026-10-15; ranked in notes/perf-survey.md)
- T-648 [P2] Adding a stop takes about 100 ms to respond and the map sometimes flashes (Anita's TODO 2026-10-17): measure with checks/t496-edit-latency.mjs, find the flash. Related T-499. Open.
- T-630 [P2] The game lags with station ridership shown, zoomed far out (Anita's TODO 2026-10-17): measure where the frame time goes, then cut what can be cut (fewer markers far out, say). Open.
- T-499 [P3] Edits after T-496: click to drawn is 96 ms on the real save; left ~20 ms worker-to-main transfer (send only what changed, also T-070's idea), `buildState`, shadow name layer, GPU uploads. Open; notes/T-496.md "Left".
- T-281 [P3] Opening a big save searches every station's walk round water per worker: 1.0 s for the real New York network's 867 stations; a tighter search or a 100 m raster would halve it. Open; notes/T-275.md.
- T-076 [P3] Playing costs 13-15% of a core, nearly all Chrome presenting 60 frames a second; a lower frame rate when the fastest train moves under half a pixel a frame would halve it. Smoothness judgement for Anita. Open; notes/T-076.md.
- T-168 [P3] Add a newly viewed city without re-solving the whole network (a pan to one re-solves everything, seconds on a big save). Open; notes/T-149.md.
- T-414 [P3] Set-up order and the first solve in WASM: each worker's first real-save solve is 2-3 s slower in the browser than native; find why and whether an order rule helps. Open; notes/T-405.md.
- T-436 [P3] Start an edit's crowding rounds from the solve before instead of free flow; figures would then depend on edit history, so Anita decides. Not measured; notes/T-434.md.
- T-361 [P3] One demand engine per city, not two (T-361's item 3): the everyday layer builds its own commute engine copy; sharing saves 20-40 MB a worker, moderate risk. Open; notes/T-361.md.
- T-356 [P3] A move far from the draw anchor (T-307) rebuilds the whole world on the main thread, about 50 ms per long pan (T-348); rebuild only what depends on the anchor. Open; notes/T-356.md.
- T-354 [P3] Two smaller main-thread costs (T-348): a pack opening is copied per worker in one task (45-70 ms for New York), and a solve's final publish takes 70-110 ms; spread each. Open; notes/T-354.md.
- T-359 [P3] After T-351: a bare start's card waits 2.0-2.2 s for New York's bare opening in the aux worker alone; splitting it over the pool could bring it to 1-1.5 s. Open; notes/T-351.md.
- T-344 [P3] A zone's loading bar can stick near 99% (found in T-343) when the view moved to dead space before the viewed zone's first solve. Rare. Open; notes/T-344.md.
- T-346 [P3] After T-328, a reload with the cache on screen hides the fresh solve's estimates, so uncached views (other purposes, weekend) wait for the final at about 20 s. Open; notes/T-328.md.
