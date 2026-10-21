# Tasks

Open agent work, one line a task; detail in `notes/T-###.md`. Format and rules in `CLAUDE.md`.
P1 = now or blocking, P2 = this milestone, P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Direction** (Anita; CLAUDE.md "Working here"). The whole world, with a demand model that is
sensible and makes building rail matter (2026-10-17: fun over fitting real cities; her two ceiling
tests are the targets, real cities loose checks; DECISIONS 2026-10-17, T-636). Next, in order: 1.
finish T-613 (the rest of Europe; Russia, Belarus and Ukraine held for later, Anita 2026-10-19), then
the follow-ups below, then whole countries from now on, Spain first (T-723, notes/T-719.md), and the
rest of the world; 2. demand: T-712 with T-721 (city pairs and regional rail against real figures,
for discussion), then T-528 (fit the regional rail constant and damping); T-634; 3. interface:
T-602/T-651, T-646; 4. game modes wait. Performance is open work (notes/perf-survey.md;
`checks/t714-pan.mjs` guards panning, `checks/t724-hover.mjs` pointer moves).

**Where things stand (2026-10-17).** 80 zones play, all shipping through the pack gate (T-538). The
unattended runner `pipeline/world_run.py` shipped its 5-city pilot (T-540), the UK batch (T-541)
and, after T-610 and T-611, Cardiff and Belfast; Owerri, Aba and Aboh park on T-614 and T-615.
Anita's plain page on demand: docs/demand-plain.md (keep it current). History is in ARCHIVE.md.

**Handoff (2026-10-19, evening).** The 2026-10-18 and 2026-10-19 playtest rounds are done (T-669 to
T-722, ARCHIVE.md; Anita's notes in notes/playtest-2026-10-19*.md). The last commit on 8800 holds
the round's end (walking bands T-720, change cost 2 minutes T-717, regional route choice T-718
with the long-trip softening on, circles and money T-722, bullets T-705, the pan check T-714).
T-725 and T-768 are done and T-613 has built every non-held Europe row (nothing running; registry
136 zones, not committed): only nic parks (T-769); state and resume in notes/T-613.md. Russia, Belarus and Ukraine are held (status `held`).
T-700 is done: every pack's walk factors are split by bank through res 9 (packs changed, not committed; backups in
`data/work/pre-T700/` wait for Anita's acceptance). Then, in order (T-764's runner checks are done): T-638's market packs (park 7); T-602/T-651
(zones not opened drawn with their walk split; writes registry.json, so after T-613); T-712 (with
T-721). Spain is covered whole (T-723, 2026-10-20; registry 176 zones, not committed). Batch agents wait once on the runner's process, never on a timer (T-719: wake-ups
were 75% of a batch's tokens).

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
- T-551 [P2] Demand review for Anita (her playtest notes, 2026-10-16): measured, nothing changed (notes/T-551.md). Waits for Anita, who decides at the deep calibration.
- T-514 [P3] "Other" per cell (T-513): the sim keeps trips by other only as a total, so bubbles and hovers cannot show them by cell; needs other by home and work cell. Open; notes/T-514.md.
- T-638 [P3] Rebuild the market packs on the new parking base (they hold 3 minutes until rebuilt, `intercity.py`'s `PARK_MIN` is 7 since T-637; about 2 minutes on long regional trips). notes/T-636.md.
- T-688 [P3] A distance decay inside the 2 km zone (an everyday destination is a zone, then any cell in it): the only way to the surveys' very short trips (NHTS 23% of errands under 0.5 km, the game 12% at best). Item 3 of notes/T-637.md "What Anita decides". Open.
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
- T-429 [P3] Cross-border long trips in Europe: T-809's one open-area factor (0.5) fits London - Paris but leaves land pairs (Paris - Brussels, Amsterdam - Brussels) about half to two thirds low; a factor per country pair if country tailoring wants it. Open; notes/T-809.md.
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
- T-797 [P2] Lines that backtrack (Anita 2026-10-21): a line may reverse partway along its route (at a station or a track end), with throughput limited hard there (a long reversal hold, say) so it stays a stopgap. Design in notes/T-797.md; Anita approved 2026-10-21 with a big reversal icon, 12 reversals an hour a side. B1+B2 done 2026-10-21 (landed a8aede2); B3 (bare track ends) open; loops (T-798) next.
- T-798 [P2] Loop lines (Anita 2026-10-21): a line whose route comes back to its first station runs round continuously with no turnback. Design first (schedules, fleet, demand routing); may share work with T-797. Design in notes/T-797.md; Anita approved 2026-10-21 (loops at ends lift the terminal limit, fine). L1+L2 taking 2026-10-21 (worktree loops).
- T-691 [P3] A station tried on edges 997 and 1130 of the real save (an overlapping pair, 688/1075) is refused with a track-drawing reason ("The curve is too tight... would run on top of existing track"): find where the worker's station trial produces it and give the station's own reason. Found in T-671. Open.
- T-623 [P3] A station cut inside an arc, or a ramp overlapping a curve, still shifts the ramp (T-622 holds ramps on straights only). Open.
- T-663 [P3] The select tool's drag handle covers a blueprint junction's fork badge; and check whether the quiet crossing marks get busy on a big built network (340 on the real save; `QUIET_R`, `QUIET_ZOOM` in gradeMarks.ts). Open.
- T-658 [P2] (raised by Anita 2026-10-19: "somewhat important"; design for her first, taking 2026-10-19) Draw double track as its two tracks when zoomed far in, which needs lines and stations drawn with a minimum width in world units (the two tracks sit at slightly different places); helps reason about terminals and crossovers (Anita 2026-10-17: plan first, not now). Reference NIMBY Rails (research/nimby-track-drawing.md: its double track is two single tracks, which it calls its main usability cost, so keep our one piece and offset each lane in drawing), but show little of it at medium and far zoom and keep the normal single-click default easy; odd setups possible but no priority (Anita 2026-10-18). Design in notes/T-658.md; Anita 2026-10-19: zoom 17 yes, a platform each side yes, in traffic thickness the two tracks draw over the traffic width (which stays behind); turn-back crossovers (stage d) approved too ("yes we can draw turn back crossovers it assumes"), after a. All four stages built 2026-10-19 (a and b committed in d931dcf; c, d and the single-track preview fix waiting on the orchestrator's commit; checks/t658-lanes.mjs; notes/T-658.md "as built"). Left: a node drag's preview of single track still shows two tracks close in (nodeDrag.ts has no edge ids). Then archive.
- T-666 [P3] An automatic bridge near a short first arc or inside a tight multi-corner turn is refused although the land is there (`bridge.rs` cuts an arc only around the ramp's two vertices; 14 such cases in its sweep). Open.
- T-668 [P3] Reshaping or dragging a bridged segment still refuses ground over water: decide whether the drag rebuilds its ramps. Open.
- T-657 [P3] Turnback capacity (Anita 2026-10-17: reversing on a through platform stays at 4 minutes): buildable ways to turn trains faster than reversing on a through platform (15 an hour): tail tracks (sidings beyond the platform) and loops, as real metros do for 2-2.5 minute headways; and whether a line's turnaround may go down to about 2 minutes (20 an hour on one platform, needs crews stepping back). Design first, then build. Anita 2026-10-18: after a while, the first ten lines in a huge city are at capacity at once even at a 4 minute turnaround, which a 30 trains an hour terminal would solve; at 160 km/h top speed 25 an hour (2.5 minutes) is her reference. Open.
- T-248 [P3] The track inspector's Construct and level buttons act on one piece of a drawn route; consider "this piece / the whole run between junctions and stations" if piece by piece is tedious. Open.
- T-339 [P3] An express slowed by a local ahead of it on plain double track (Anita: "ideally... but thats whatever"); trains pass through each other at no cost. Low priority. Open.
- T-340 [P3] Quad track's rest (from T-257): a station on a quad stroke's own end node is refused when that end is new track; the express pair is always left of the drawn direction. Open.

## Interface
Anita's playtest of 2026-10-19 (the 9e466a4 build): notes/playtest-2026-10-19.md, a section a task.
- T-811 [P2] T-809's open-border area in the normalisers: `pipeline/market_zones.py` `Field.normalisers` (line 137) takes the sim's rule (`OPEN_AREA`, 0.5), then `..\venv\Scripts\python.exe pipeline/market_zones.py` from trainworld/ for every zone and the rows, once the Europe batches finish. Until then European places' departures run high (far pairs about 8-17%, tail more) and near pairs' place totals are the old rule's. Open; notes/T-809.md.
- T-810 [P3] Commutes across an open border: `pipeline/borders.json`'s national 0.05 is far too strong inside the EU, Schengen and the UK (Luxembourg, Geneva, Basel); its own value fitted on cross-border commuter counts, then the joint gravity groups rebuilt. Open; notes/T-809.md.
- T-841 [P3] The track inspector cannot re-level a single straight piece (no corners, so no "Elevation of the stretch" control); the multi-select bar can (found in T-815). And the selection bar is about 600 px wide with the elevation buttons, reaching the debug panel on a 1400 px map. Open.
- T-817 [P2] Design speed and high-speed track (Anita 2026-10-21): a Build tab "design speed" the drawn curves are made for (160 km/h default, the player may raise it for rounder curves, mostly on long segments); high-speed track as its own track type that costs more, default design speed about 300-320 km/h. Stage a done 2026-10-21 (landed 5cd004d); stage b designed in notes/T-817.md; Anita 2026-10-21: above 160 km/h needs high-speed track, and the player picks high-speed trains per line (not automatic). b1 (track type) done 2026-10-21, landed 8879594; b2+b3 (speeds above 160 on HSR, per-line high-speed trains, their price and running cost) taking 2026-10-21 (worktree hsr2).
- T-828 [P2] Eurostar is hard to make work (Anita 2026-10-21): a very crowded stretch near the city puts long-distance riders off the whole line. Check how crowding on a shared stretch enters long trips' choice once T-818 reports; notes/T-818.md. Open.
- T-830 [P3] The "Blueprint added" notice once read "86.7 km, $2.90B" after a 3 km segment while the blueprint total was $325M (found in T-817): find what it sums. Open.
- T-836 [P3] Run `pipeline/zone_aliases.py` from `world_run.py join` (fail-soft) so new zones get search aliases, and give each newly joined country's zone names a look against docs/zones.md "Names" (T-825). Open.
- T-837 [P3] Sparse lines' end of service (found in T-835): on lines with one or two trains a period (80 on the real save, mostly LIRR and NJT, 3.8M boardings) riders arriving after a period's last train wait for the next period's first, 29% more waiting over the day than demand's half-headway assumes. Open; notes/T-835.md.
- T-842 [P2] Riders expect the service each stop really gets (Anita 2026-10-21: "if passenger expectations differ from train schedule perhaps we should be mostly shifting the passenger expectations"): demand's wait at a stop per period from the departures that actually pass it (first and last trains, period changes along a long line, sparse lines' end of service, T-837), not the line's period headway. Design note first (demand's route search, cost, what changes in figures); notes/T-835.md has the measurements. Designed 2026-10-21 (notes/T-842.md): service span per stop (a boarding penalty where no train is due within 2 h), stages a and b; waits on Anita's two questions.
- T-803 [P3] More station faces (Anita 2026-10-21): some by time of day and a few rare ones. Open.
- T-719 [P2] Covering whole regions (all of Spain, as Japan) rather than lone metros, timings and speedups: note written 2026-10-19, waits on Anita's decisions (its last section); notes/T-719.md.
- T-771 [P3] A Spain module (T-719 item 6, later tailoring): the transport ministry's mobile-phone trip tables (a curve per zone, jobs at work) and INE's 1 km census grid; would rebuild mad bar val sev mal and the 40 zones. Open; notes/T-719.md.
- T-772 [P3] Spain's 9 dead-space provinces (under 20 people a km², 1.46M people; three at 16.8-18.9) leave Valladolid, León, Burgos and Salamanca as zones in a hole: revisit the cut, or grow very empty units onto a neighbour. Anita 2026-10-20: some dead zones are fine, so low priority. T-788 adds big dead units holding cities: Vestland (Bergen), Trøndelag (Trondheim), Agder, Pohjois-Pohjanmaa (Oulu), Keski-Suomi, Pohjois-Savo, Lõuna-Eesti (Tartu), Latvia's four regions outside Riga, Sweden's north and Värmland; a zone of a dead unit's populated municipalities would fit them. Open; notes/T-723.md, notes/T-788.md.
- T-774 [P3] `world_plan.py`'s metro replan does not know the `held` status (plan_config.ROW_STATES), so a replan would give held rows new ids and orders; check and add it. Open.
Anita's second set of 2026-10-19 notes (the bfe93f9 build): notes/playtest-2026-10-19b.md, a section a task.
- T-702 [P3] Station ghost leftovers after T-695: rare corrections on the real save's bundles of overlapping imported lines (refusals the page cannot see, T-691), jumps while sliding along them, one blink near a track end. notes/T-695.md "Left". Open.
Anita's TODO.md "short term", filed 2026-10-19 (evening).
- T-727e [P3] Real-network graphs for the other ~113 noritetsu regions (`pipeline/realgraph.py`), shipped from R2 beside noritetsu's tiles rather than copied into the build. Open; notes/T-727.md.
- T-755 [P3] `bridge::bridge` finds no room for ramps at Lake Hamana on the traced Tokaido Shinkansen (six corners alone fail too), so `bridge_long` raises that stretch whole; find why the ramps do not place (needs the zone's water raster in a native harness; not tried in T-754). Open; notes/T-727.md.
- T-763 [P3] Penn Station-Emeryville is still 3.0-3.6 s from rest to preview in WASM though search (0.06 s) and fit (0.27-0.36 s) are fast now: the clock worker's preview is 1.2-1.35 s and about 1.5 s more is unaccounted for. Profile the preview round trip. Open; notes/T-727.md.
- T-750 [P3] Add a city's real network to a sandbox game already running (merging in the Rust track model; notes/T-728.md stage 6). Anita wants it, not urgent. Open.
- T-751 [P3] "Expand all platforms on this line" in the line panel, so a line can run longer trains (Anita 2026-10-19, after T-733). Open.
- T-734 [P2] While placing track, a press and drag on a node moves the node (a click on track still branches and places track). Open.
- T-738 [P3] Flight data for long-distance intercity demand (Anita's idea); see T-712's spot check first. Idea.
- T-742 [P3] `checks/t488-money-days.mjs` times out waiting for 00:00-00:10 (line 47) on the current app, with or without T-736; also after importing a save, fares stay $0 for about 40 wall seconds of play (notes/T-736.md). Open.
Anita's playtest of 2026-10-18 (her TODO.md "short term"), filed 2026-10-19. Plans in the notes.
- T-676 [P2] A cuter base map (warm paper land, soft teal water, sage parks, 3 road weights, most POIs gone); the ground tinted by density as a switch, off. Anita judges the palette. Open: stage 1 done, palette waits on Anita; stage 2 (density tint) not built; notes/T-676.md.
- T-678 [P3] Day and night tint on the base map, subtle. After T-676; notes/T-676.md.
- T-682 [P3] Real-world rail layer for reference (noritetsu's tiles), then a trace tool that follows a real line. Stage 1 (view only, Settings > Real network) done 2026-10-19; the trace tool is T-727; notes/T-682.md.
- T-683 [P3] Accounts (optional, skippable), cloud saves and city ridership leaderboards on shonei-server. After T-684; notes/T-684.md.
- T-685 [P3] City stamps at 10% rail share (the goal number): stars done (T-684's round, 2026-10-19), drawn stamps later. Open; notes/T-684.md.
- T-651 [P3] Zoomed out, zones not opened look fully red while opened ones look yellowish from their green walkers (Anita's TODO 2026-10-17): the same as T-602, give unopened zones their walk split. Open.
- T-777 [P3] Market rows per country: `market-rows.json` (0.9 KB a zone, about 1 MB for Europe, 5-15 MB for the world) is fetched and parsed whole by every solving worker once a market pack opens, since the market reads every place's row (pairs to unopened places, departures elsewhere); loading only the countries near the network needs the Rust market to work from partial rows without changing demand. notes/T-776.md. Open.
- T-602 [P3] Walk split for zones not loaded (rest of T-576): `zone_summary` (sim) gives a walk count per zone, `pipeline/zones.py` `local_trips()` stores it as `localWalk` in registry.json beside `localTrips`, `intercityView.ts` draws it green. Open.
- T-561 [P3] Page freezes at 1,500x or faster in a fresh game with a line in each of Toronto and Boston (T-550's agent); not reproduced on 2026-10-16's tree. Open until seen again with exact steps; notes/T-561.md.
- T-560 [P3] Crush load per country (found in T-552): every city's trains fill at `api::CRUSH_PER_CAR`'s 160 riders a 20 m car, stranding Tokyo's riders early; take published figures per stock from the pack. Open; notes/T-552.md.
- T-330 [P3] Line chips without a symbol show the name's first letter (every New York subway chip reads "S"); default to the line's short name ("subway 6X" to "6X"). Open. Detail in notes/T-330.md.
- T-430 [P3] T-418's straight-then-arc fallback for a click too tight for one tangent arc (left out of T-423). Open idea, only if wanted.
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
- T-617 [P3] Rebuild gla, lis and pot on T-610's unit matching when next rebuilt (gla +5.6%; lis and pot move from plan polygons to OSM municipalities). notes/T-610.md.
- T-612 [P3] Runner: delete a parked row's pack files once its reason is recorded.
- T-613 [P2] The next runner batch, the rest of Europe from pipeline/world_plan.json (`plan --countries` first for the row count), after T-614 and T-615; publishing past the current set needs T-060 and Anita's word. 2026-10-19 evening: every non-held Europe row built; 69 rows shipped in all (registry 136 zones, every-pack-opens PASSED on the 27 new), nic parked (T-769), 18 Russia, Belarus and Ukraine rows held (later, Anita). Left: nic (the retrospective's proposals: T-764 done, the rest T-775); state, resume and retrospective in notes/T-613.md.
- T-788 [P2] Whole-country cover for the rest of Europe (33 countries, 360 zone rows planned on Anita's 0.5M floor; Russia, Belarus, Ukraine held), in waves of two or three country runs at once and one join a wave. Taking, 2026-10-20, Europe data supervisor; state per wave in notes/T-788.md.
- T-787 [P3] Portugal's Médio Tejo (meo), Lezíria do Tejo (lez) and Douro (doo) zones park on their outlines: meo built from a 6 km box of 91 people (tier0 then failed in `cap_places` on an empty index), lez at 1.96 times its plan population, doo at 0.09 (T-786's run). Likely the 2024 NUTS-3 units against OSM's admin relations in `osm_units`; fix the match and rerun the three. Open; notes/T-786.md.
- T-783 [P3] Melilla (mea) is registered with country "MA" (market_zones.py's country lookup, T-776), so its trips with the rest of Spain take the cross-border factor. Fix the lookup to use the plan row's country and re-register. Found 2026-10-20.
- T-784 [P3] Low-memory devices: one demand worker and lighter settings (the market's 8 stations a zone end) when `navigator.deviceMemory` is small, and a notice on very big networks (all of Japan's real network takes about 2.3 GB a worker). Measure on a simulated 4 GB machine. Anita 2026-10-20: low priority.
- T-785 [P3] (was a second T-782) France's 80 countryside zones are on global sources and the borrowed zone curve (T-779), though the France module's sources (MOBPRO commune flows for a curve per zone and jobs at work, Filosofi 200 m squares) cover them; later tailoring, rebuilding the zones (and the plan metros) on the module. Open; notes/T-779.md.
- T-769 [P3] Nice (nic) parks on drive_factor_median 2.51 (gate fails over 2.0): see whether its routed drives are real (mountains, the sea) or missing roads (the PACA extract lacks Monaco and Italy, which some zones touch), then fix the extract list or relax the row. Open; notes/T-613.md.
- T-765 [P3] Work at home is 0 in every archetype D pack (24 countries, data-register.md Commuting). Eurostat LFS lfsa_ehomp has a share for the EU and EFTA countries (Germany's 0.10 and France's 0.12 came from it) and ONS OPN 0.27 for Wales, Scotland and Northern Ireland. A constant per country is a demand factor, so propose to Anita before adding any. Open; notes/T-726.md.
- T-766 [P3] `world_run` records each extract's PBF header snapshot time (`osmosis_replication_timestamp`, a header read, `notes/t726_pbfheader.py`) in the row's `run` before it deletes the file, so extract dates need no log archaeology. Open; after T-613 hands back (the runner's files are in use).
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
- T-791 [P2] Moscow's jobs look far too even across the city (Anita 2026-10-20: "definitely noticeable"): the tier 0 proxy (30% GHSL non-residential volume, 50% OSM job places, 20% population) suits its tall residential microdistricts and mid-rise centre badly, and the caps may move the centre's jobs to the population part. Check the pack's jobs by okrug against official figures (Rosstat/Mosstat, open sources only), count what the caps moved, and fix the proxy generally (big cities, tall residential blocks). Do it with the rest of Russia (held, Anita).
- T-790 [P3] The market's corridors (zone pairs, 5.3 MB, plus 2.7 MB of commutes and the 4 MB place-pair mode shares) ride in every period's result: 5-13 ms of main thread a message, 16-17 messages an edit, growing with the square of the zone count. Send them only when their key changes, robust to stale messages and restarted workers, and let `corridor_summary` skip empty pairs in Rust. Open; notes/T-789.md "Zones and lag".
- T-782 [P3] The market's routing on big clusters: since T-780 its two search sweeps a pass are about half its time (all of Japan 62-70 s a daytime pass in WASM). Exact ideas: the kept strategies in one block (an arena) so WASM grows its heap once (768 MB kept halves the second sweep natively, but growing piece by piece cost the first pass minutes), fewer crowding rounds for the market when its loads barely move. notes/T-780.md.
- T-758 [P3] The track derive with water masks is about 4.9 s in WASM for Japan's 709 lines (0.42 s natively without masks): find the water cost. Open; notes/T-737.md.
- T-759 [P3] The clock state on a big network: 360-500 ms of main thread when Japan loads, 130-200 ms when resent mid-play; send only what changed (with T-499, T-070). Open; notes/T-737.md.
- T-499 [P3] Edits after T-648: a stop click to drawn 112-130 ms on the real save (checks/t648-stop-edit.mjs); edges and nodes already left out when unchanged. Left: lines sent whole (send only the changed ones, with T-759), the buffer build 15-20 ms, 25 ms of main thread taking the state (world effects: labels, money charts) and ~40 ms more to the frame. Open; notes/perf-survey.md (T-648).
- T-762 [P3] Every clock state with a planned line runs `blueprint_cost_items`'s trial construction of the whole blueprint (5-20 ms on the real save); work it out only when the blueprint or service changed. Open; notes/perf-survey.md (T-648).
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
