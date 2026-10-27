# Tasks

Open agent work, one line a task; detail in `notes/T-###.md`. Format and rules in `CLAUDE.md`.
P1 = now or blocking, P2 = this milestone, P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.
Lower-priority tasks are under **Someday** at the end, one short line each. They are open tasks
(Anita 2026-10-26): count them when asked what is open, and work on them when nothing above can go
ahead; pull one back up when it becomes next. Anita's shortlist of candidates: `notes/for-anita.md`. Notes of archived
tasks that nothing open or current cites live in `notes/archive/` (older ARCHIVE.md and DECISIONS.md
links say `notes/T-###.md`: look there).

**Orchestrator handoff: start with `notes/handoff-2026-10-26.md`** (state as of 2026-10-26 evening).

**Where things stand (2026-10-26).** The whole world is the goal, demand simple on purpose
(CLAUDE.md "Working here"; plain account in docs/demand-plain.md). Covered whole: Japan, Europe
except Russia, Belarus and Ukraine (held), China with Hong Kong, Macau and Taiwan (countryside
replanned into compact zones on the 2020 census, T-1185), the US with Puerto Rico, and Korea (built
2026-10-25 but Daegu and Tongyeong, being rebuilt in T-1262); about 1,750 zones. Version 2026-10-25b
is published. Under way: the res-9 walk repair (T-1261), which must come before Canada (T-1239)
and any other non-Chinese join. After it: Canada, then
Mexico, Turkey and Vietnam; the replan of the US and Europe waits for Anita (T-1189). Performance on
big networks is the other open front (Performance below).

## World and countries
Sources: docs/demand-worldwide.md, docs/data-register.md. The unattended city pipeline: notes/T-535.md "Stages".
- T-1375 [P3] The zone ring cut halves a ring by people round the centre it wraps, so a thin arm (Whistler and Pemberton, joined to the Fraser Valley zone round Vancouver's row) splits the ring's real body instead (Abbotsford 0.17M and Chilliwack-Mission 0.19M, T-1239): try shedding the arm that alone makes the wrap before halving; a Canada replan after. Open; notes/T-1239.md section 1.
- T-1376 [P3] Les Îles-de-la-Madeleine (lel, 12,000 people, an island zone of Canada) parks in its build: `enrich_parks` finds under 85% of its park draw on a cell (beach and dune parks off the populated cells). Allow a small island zone through (as T-897's one-park islets) or leave it dead space. Open; notes/T-1239.md.
- T-1189 [P2] Replan the built countries by T-1140's merge rule: US and Puerto Rico (162 of 187 rows), Europe (about 393 rows), Taiwan (3), each a dry run and a look, then builds, gates, joins and T-1053 successors. Waits for Anita's word ("we might run into other things that will make us want to rebuild again"; asked again 2026-10-26: not yet). China's part done 2026-10-25 with T-1185. Carries what T-1002, T-1005 part 2, T-1089 and T-903's merges asked (Zhubei split, wraps, rings, small Italian zones); T-1307, T-774 and T-891 first. notes/T-1189.md. What rides along: notes/T-1189.md "What rides along" and "Waits on a rebuild" below.
- T-1193 [P2] T-1178 stage 2, what is left: rerun the solve units holding a pack in T-1192's seamless list outside mainland China (GB's Bristol group, Italy's Milan-Como, Puerto Rico; check whether Taiwan's ran with China's joins in T-1184 and T-1185), after T-1261, rerunning `pipeline/seams.py` first if ids changed; check Guangzhou-Foshan's commuters across near 1.3-1.5M. notes/T-1178.md.
- T-1310 [P3] tier0's cell cache (`data/work/<id>.tier0_signals.npz`) is reused even when the pack's outline changed, so a rebuild for a new outline keeps the old cells (T-1144's first hon run). Key the cache on the outline's hash, or have `world_run run --rows` of a rebuilt row delete it. Open.
- T-1334 [P3] Other coastal districts dropped as Kwai Tsing was (T-1144)? List each built outline's OSM relations at its picked admin level with 0.25-0.5 of their land inside a plan unit and not picked (a coarse unit polygon missing reclaimed land or islands, the leftover under the 10% fill); rebuild any populated ones. Open.
- T-1231 [P2] `world_run.py` breaks every neighbour's market pack when it rebuilds a registered row: it unregisters the row at its region's start and registers it again at a new market place, so the other market packs still listing commutes to the old place make `market_pack` panic ("invalid commute between zones") and the region's rows park at `register --market` (T-1184's first start: 27 rows; T-1113's hfd). The runner could keep a rebuilt row's place, or rewrite the market packs naming the old place (`market_zones.py <ids>`) after unregistering. Open; notes/T-1150.md "Run".
- T-1079 [P2] `world_run.start_vite` takes any server already on `--port` for its own: it polls the URL and returns once anything answers, so when another session's Vite held the port (T-1078's GE run, 2026-10-23) every browser check of the region ran against the wrong tree and 37 rows parked. Fail when the port is taken before starting, or when its own process exits while the URL answers. Open; notes/T-1004.md "Wave 2".
- T-1080 [P2] A world run at home can fail extract reads on Windows' commit limit (74.5 GB here): each parallel GDAL reader in `osm_features` commits a 6 GB node index, so with a join, a big profile and browsers up, Wisconsin's and Nebraska's reads failed "Failed to open dataset" with 31 GB of RAM free (T-1078). Check free commit before a read and wait, or read with fewer parallel readers when it is short. Open; notes/T-1004.md "Wave 2".
- T-891 [P3] A row rebuilt after its outline changes keeps every per-city cache from the old outline (`data/work/<id>.outline.geojson`, `.tier0_signals.npz`, `data/raw/tier0/osm_jobs_<id>.csv.gz`, `raw/t142b`, `raw/t196`), so it rebuilds the old pack: Sydjylland (T-788 wave 4) kept Region Syddanmark's 1.22M until each was moved by hand. `world_run.py build_row` for a planned or parked row could clear the row's caches first. Also a new row reusing a removed row's id takes its caches: T-1239's lnn, gup and say parked on old `osm_jobs_<id>` files (48 deleted by hand). Open.
- T-774 [P3] `world_plan.py`'s metro replan does not know the `held` status (plan_config.ROW_STATES), so a replan would give held rows new ids and orders; check and add it. Open.
- T-964 [P2] Gate a region in chunks as its rows finish, while its later rows build, instead of after the last build (wave 1's northern region: builds 114 min, then the gate 73 min with every build worker idle; about 25-30 min a 100 rows). Open; notes/T-962.md.
- T-930 [P2] The join's re-gate re-checks only what a joint solve can change (mean commute, top zone share, joint borders, other packs unchanged) instead of every figure of every member: the GB join for hih (2026-10-22) spent 9.8 of its 31 minutes there. Open; notes/T-928.md section 4.
- T-1309 [P3] 384 new Chinese zones show the planner's county name: a look at their names (docs/zones.md "Names"). Open.
- T-1218 [P3] `zone_outlines.py` takes about 15 minutes since T-1145 (3.5 for the sea tiles, 10 for 285 windows of sea split), once per join: cache each window's result keyed by its zones' coarse rings, so a join that changes a few zones redoes only their windows. Open.
- T-428 [P3] `data/packs/intercity.json` (14 MB) is only the joint gravity's input and output, never read by the game since T-377: move it to data/work, out of what is published. Open; notes/T-428.md.
- T-941 [P3] The published page's expected 404s: the real-rail layer asks `../noritetsu/regions.json` before R2 (404 on anita.garden every load), and the two Canadian syllabics glyph ranges glyphs.py leaves out; smoke against the live page counts both as console errors. Try R2 first when the page has a tw-data meta, and make smoke's glyph filter match there too. Open.

## Demand
- T-264 [P2] (Anita 2026-10-24: a priority, with holidays' own demand effects, trips to touristy POIs, and telling the player what to expect) Superholidays (Chinese New Year, Golden Week, Obon, Chuseok, Thanksgiving) as long-distance days with a direction. Design written 2026-10-26 (notes/T-264.md); Anita 2026-10-26: go ahead; home trips by population, but with a boost toward rural areas, small towns and smaller cities (no birthplace tables); going-out trips lean to touristy places (T-1374); the heads-up as a calendar icon, not a top-bar line. Stages T-1372 and T-1373 (done 2026-10-26), T-1374 below.
- T-1374 [P2] Superholidays stage 3: going-out trips lean to touristy places (attraction visits per resident), after T-1373 (whose lean column it can weight further). notes/T-264.md
- T-1503 [P3] Superholiday rush days on networks past about 8,000 stations take the longer trips unleaned (the lean's S x S table over 256 MB, `LEAN_OD_BYTES`, T-1373): a sparse or per-destination table if a network that big plays. notes/T-264.md
- T-1504 [P2] The China network's longer trips barely ride rail: about 3,300 line boardings a weekday for the market's longer trips on ten high-speed lines between China's 24 biggest metros (t1156-all-china.save, 39 million rail trips in all), found measuring T-1373. Analysis written 2026-10-26: fares weigh too much on long trips (T-1431's root, x4 alone), the save's "HSR" runs at 160 km/h with 6/3/2 trains and its stations are not linked to the metros, which a zone's 8-station access list then never reaches; sampling is not a cause. Proposal and Anita's questions in notes/T-1504.md; follow-ups T-1506 to T-1508.
- T-1508 [P3] The market's totals on very big far pairs look low (Beijing - Shanghai about 21,000 trips a day both ways by all modes, against 580,000 riders a day on the line in 2019): calibrate against national rail totals; every country. notes/T-1504.md. Open.
- T-1233 [P2] T-1177 leftovers: run `checks/t1177-cross-everyday.mjs` with `IDS=gua,fos` once Guangdong's market packs carry the pairs (compare each purpose's local legs plus the pairs with the old totals); the explorer's Everyday trips block shows only while a network reaches the market (no walking figures for two zones with no line, as there are no corridor figures for the pairs); a city's own purpose counts and homes bubbles stay local legs less the kept share. Open; notes/T-1177.md.
- T-1396 [P3] A car between landmasses by ferry: the market has no car at all between places on two landmasses (London - Dublin, Athens - Heraklion, Barcelona - Palma); T-279's ferry rule (docs/regional.md 5.1) could give them one, which would add pairs with only a car and a plane. Also rerun `pipeline/market_drives.py build` once Canada's market rows land (23 Canadian zones had none on 2026-10-26). notes/T-279.md.

## Performance
Ranked in notes/perf-survey.md and notes/T-1156.md (the all-China save) and notes/T-1200.md (slow CPUs).
- T-1223 [P2] T-1206 stages 2-4: the fine set in `demandClient` (zones not looked at in a while coarse, not only over budget, Anita 2026-10-24; priorities, memory budget, close and reopen at a level, save header and cache key, T-1197's formula, stars from fine only, `?lod=`), the local refine (one population against the cluster's frozen loads), and the status tag and rough-figures text. Stages 0-1 done as T-1222. notes/T-1206.md.
- T-1169 [P2] Demand over many zones: an all-China network (169 zones reached) spends about 3 minutes downloading and opening packs before the solve, and each solving worker holds 2.4-2.6 GB of WASM memory (4 GB ceiling), so all 745 Chinese zones would not fit: memory per opened zone, with T-898 and T-782. Open.
- T-1354 [P2] The market's corridor figures depend on which market packs were open at a worker's first composition: `MarketPacks` keeps the corridors (`corridors_of`) worked out then for every later composition (T-1204), but they differ with the packs open (on the real save, a worker that composed once from all 13 packs and the workers that composed as packs arrived had corridor tables ten rows apart, 1,530,357 and 1,530,367). So a page's longer trips depend on the order packs arrived; maybe also T-1225's load-to-load finals. Make them depend on the world's rows alone, or recompute when a pack adds a near pair. Open; notes/T-1174.md. T-1225's look (2026-10-26, `checks/t328-cache-check.mjs` with `MODE=reload DUMP=1`, per-section hashes in its json): the two hashes did not come back, 26 reloads of the real save gave one (4332e0ad), also with two checks at once. What does differ is a game loaded by import (`MODE=prepare`, hash df1cc482) against a reload (4332e0ad): same corridor table and `railAt`, but every other section differs at float-rounding size (relative ~1e-7: a line's day total 1,261,111.44 against .485, longer trips 1,616.8592 against .8593), in the lines, stations, purposes and cities too, so not longer trips alone; the rail total rounds to 12,189,003 in both. Likely the same cause (different pack arrival, so a different composition and sum order), unproven. To pin it, delay some market packs on purpose and diff the dumps.
- T-1366 [P3] The demand pool's memory cap (T-1205) never adds a solving worker back in a page: a big game's zones opened for a while (or a big game then New York from the menu) leave it at 1-2 workers, solves slower. Grow it back when the projected memory has room for one more, with a margin against flapping (a spawn, reopen and re-dealt periods, as `shrink` in reverse). Open; notes/T-1205.md.
- T-1367 [P3] After zones or a market close, a demand worker's heap still holds them (frees deferred while busy, the old market parts the kept clusters reference) until its next solve, so T-1205's restart waits for one (all-China: heap at rest 2,041 MB after three zones closed, 1,502 at load with the same items open). Free those at the close, or count them out of the need. Done when `checks/t1205-memory.mjs` passes on the all-China save (T-1205's bar; its last run stalled in the solve after the close, unexplained). Open; notes/T-1205.md.
- T-1170 [P2] Moving the draw anchor to another city rebuilds every stroke: 0.7-0.9 s on the China save, and the first click in a new city 1.4-1.6 s; rebuild only what depends on the anchor (with T-356, T-1028, T-1029). On New York at 6x throttling 460-495 ms, the world save at 4x 227-497 (T-1200). Open.
- T-1171 [P2] Unbuilt lines re-quoted on every clock state (`blueprint_cost_items`' trial construction, `line_build_cost` per unbuilt line): about 8.5 ms more an edit per unbuilt line; quote only when a panel asks (Rust, the Build tab, the line panel; with T-762). Open.
- T-1358 [P3] `followTrack` samples every edge's drawn line each km on a game's first network (about 90 ms at 6x on New York); cut it with the same zones found (it cannot wait past the paint: its zones feed demand's rounds), and `toWorld` about 12 ms an edit. Open; notes/T-1357.md.
- T-1505 [P3] After T-1380, a station edit that makes a new name label still spends 100-140 ms at 6x in `StationLabels.place` (likely the layout forced by reading the new label's width, unproven): measure the width without forcing a layout, or place only the names near the edit. Open; notes/T-1029.md "What is left".
- T-1509 [P3] A line whose length or drawing order changes (stops dropped, a line made, undo of either) still posts the line strokes whole to the page (about 35 MB on Japan, 4 of 12 edits of the T-1381 series), since the segments after it move; keep the lines' order stable or post the moved runs (map/strokePost.ts). Open; notes/T-1029.md "What is left".
- T-1021 [P2] Big saves' load still holds the main thread up to about 1.9 s (Japan): zones.ts placeAt/nearestZone/cityReach distanceTo over every station and track edge (about 0.85 s), boundary geometry as boundaries arrive, garbage collection. Open; notes/T-1001.md "T-1008".
- T-1211 [P3] The page's DOM makes every layout cost: 2,200 elements (875 station names on the map), a forced layout 22-37 ms at 1x; the Lines tab opens in 360-400 ms at 6x (each chip reads `currentCSSZoom`, `useCssZoom`), the saves panel 534 ms (`showModal`), the Stations tab still 0.6-0.7 s at 4x with 80 rows (`Chips` measuring). Open; notes/T-1200.md.
- T-955 [P2] The network canvas asks for 4x MSAA with a stencil (overlay.ts:87), about 80 MB at dpr 3 on a phone, though every shader already antialiases analytically: compare `antialias: false` (or false from dpr 2) against today in screenshots of the real save and drop it if they match. Open; notes/T-953.md item 3.
- T-921 [P3] Performance: commute zone pairs under 1 person a day dealt in chunks (T-874's deal with the chunk in people a day, `c = 1 x period share`, so every period deals the same pairs): New York solve 34.3 to 29.6 s (-14%), stations under 0.8%, ceilings unchanged. Measured in notes/T-909.md option 1. Open.
- T-782 [P3] The market's routing on big clusters: since T-780 its two search sweeps a pass are about half its time (all of Japan 62-70 s a daytime pass in WASM). Exact ideas: the kept strategies in one block (an arena) so WASM grows its heap once (768 MB kept halves the second sweep natively, but growing piece by piece cost the first pass minutes), fewer crowding rounds for the market when its loads barely move. notes/T-780.md.

## Interface
Anita's playtest of 8800 (build 40593370), 2026-10-26 morning: the lighter map "feels nice", Traffic view "looks pretty good", related lines "looks great", adding local games to the account worked.
- T-1407 [P2] After a click the ghost stalls about 160 ms at 1x on the real save (0.3-0.6 s at 6x): the clock worker works the edit (stroke buffers 67 ms, the boarding hour 54) before the previews asked meanwhile. Answer waiting previews first, or put the boarding hour after them. Clock worker's state path, so after T-1028/T-1029. Open; notes/T-1394.md.
Anita's playtest of buses and the round, 2026-10-26 (evening). "buses are cool!"
- T-1481 [P3] Marey diagrams (Anita: nothing built yet makes them interesting, but might be nice). Idea. Open.
Anita's answers on T-1411 and T-1412's ideas, 2026-10-26 (afternoon). Not wanted: riders before you pay ("people can just build it and find out"), growth near good stations (confusing), flights or freight you run.
- T-1519 [P3] Income groups in the local layers (Anita's call): three groups as the market's (T-1431) would run the local mode choice three times, about 2.4x the local solve time on New York (72% of it is the mode choice). notes/T-1431.md "Round 3". Open.
Anita's TODO.md "short term", filed 2026-10-26 (night).
- T-1521 [P2] Starting fares match real life. Round 2 built 2026-10-27 in tree t1521 (T-1511 merged), not landed: US dollars at market rates (income rule on market GDP per head to the 0.7, the same minutes per unit of local money as before), rail 25c a km past 20 km in New York, longer trips' money at 0.4, the car at its countries' fuel and distance tolls, the plane $100 + 5c; old saves weigh as before. Harness shares still off (car too weak in Japan and Europe, plane too dear in China); proposals in notes/T-1521.md.
- T-1541 [P2] Corridor rail is low wherever an inter-city line runs alone through a big city (T-1521 round 3's every-zone corridor runs: Shanghai - Beijing 35% of rail + air against 86%, London - Edinburgh 8% against 45%, Northeast rail 0-2% of all trips against 5-20%): rail access to a lone inter-city line, as T-712 found. Diagnosed 2026-10-27, three causes: China is the 15 km ride reach (63% of trips unserved; a metro or 40 km rides give 91-96%), London - Edinburgh is rail against the plane at 4-5 h (the station's own zone ties), the Northeast is the car (T-1542). Proposal (longer trips ride up to 40 km) awaiting Anita; notes/T-1541.md.
- T-1542 [P3] The US car is too strong on corridor runs (NY - Washington 94% car against 64%, Boston 99% against 87%) with real fuel at $0.104 a km; was 45% and 80% before T-1521. Look after T-1541. Open; notes/T-1521.md round 3.
- T-1545 [P3] The market pass still holds about 1.4 GB of temporary s x s tables a worker on all of Japan (route costs, first-wait costs, fare km, trip table; s = 9,249 served stations), so a save bigger than Japan could hit the 4 GB WASM limit again (T-1482). Build them sparse or per column. notes/T-1482.md. Open.
- T-1520 [P2] Cut the market's income-group cost: with T-1431's three groups the market's solve takes +71% (New York) and +87% (China), every population's +19-20%: three logsums a zone pair and the expected fare km table every pass. Try sharing the access part between groups and a cheaper fare-difference table. notes/T-1431.md. Open.
- T-1471 [P3] Drives view: the commutes' table with no rail takes 1.1-1.9 s in the aux worker once per zone and day on New York (a loop over every zone pair, 41M); sum it by square pair from the gravity's rows and columns, or keep it with the pack. docs/demand-views.md "Car trips by pair of squares". Open.
- T-1444 [P3] A game mode with a city's real network run as a rival to win riders from (Anita: a new game mode). After T-1305. Open.
Anita's playtest of buses along roads, 2026-10-26 (night), Shenzhen, Bus 13.
- T-1501 [P3] Bus legs that stay straight look like road legs. A cue (dashed, say) if players are confused about where a bus does not follow the road; the sim already knows (point kind bits, notes/T-1465.md). Open.
- T-1502 [P3] Bus money by region on a road route: `regionMoney.ts` shares a route's road km by the chords between stops, rougher on a winding route; share by the route's own points (notes/T-1465.md). Open.
- T-1466 [P3] Check bus ridership against real buses: a 3.6 km feeder on the real save carried 28,000 riders a day and a 6 km one 13,000 in a new game (t1464-bus.mjs); compare riders per route-km with real city buses before tuning anything (notes/T-1464.md). Taking, 2026-10-27 (tree t1466).
- T-1467 [P3] Buses stage B3 if wanted after playing (notes/T-1445.md): speed by place, a bus lane upgrade, ferries on the same line kind. Open.
- T-1463 [P3] Rename "train share" and the like now that bus-only trips count in it (Anita: later). Open.
- T-1445 [P2] Buses, design first (Anita): draw a line (along roads if that is easy) and stops place themselves, where it meets a train station and otherwise every 400 m or so; maybe no concept of stops at all. A cheap slow feeder; a demand change, so the design goes to her. Design written 2026-10-26, waits on Anita; notes/T-1445.md
- T-1446 [P3] Share a game by link, read only (Anita: yes). Open.
- T-1447 [P3] Copy and paste blueprints between saves (Anita: yes). Open.
Anita's TODO.md "short term", filed 2026-10-26 (afternoon).
- T-1270 [P3] Terrain, stage 4 (stages 1-3 built, ARCHIVE): optional, not recommended first, a 120 km/h cap on mountain-class track at -1 and up. Open; notes/T-1270.md.
- T-676 [P2] A cuter base map (warm paper land, soft teal water, sage parks, 3 road weights, most POIs gone); the ground tinted by density as a switch, off. Anita judges the palette. Open: stage 1 done, palette waits on Anita; stage 2 (density tint) not built; notes/T-676.md.
- T-751 [P3] "Expand all platforms on this line" in the line panel, so a line can run longer trains (Anita 2026-10-19, after T-733). Open.
- T-993 [P3] noritetsu's account delete question (`noritetsu/dist/account.js`, "Delete your account? Local saves will stay.") should say anitabuilder's leaderboard entries go too, since the account is shared (T-972); its privacy page could mention anitabuilder. Open.

## Track and trains
- T-658 [P3] Double track drawn as its two tracks when zoomed far in (Anita 2026-10-19: "somewhat important"; all four stages built 2026-10-19, notes/T-658.md "as built"). Left: a node drag's preview of single track still shows two tracks close in (`map/nodeDrag.ts`); then archive.
- T-691 [P3] A station tried on edges 997 and 1130 of the real save (an overlapping pair, 688/1075) is refused with a track-drawing reason ("The curve is too tight... would run on top of existing track"): find where the worker's station trial produces it and give the station's own reason. Found in T-671. Open.
- T-1072 [P3] Changing a line's stops at an end (a stop added past the last, the first removed) moves the other direction's running trains by the length added or removed: flights are kept as offsets along the run; match them by place on the ground instead. Open; notes/T-1071.md "Limits".
- T-1073 [P3] Undo or redo of a train count takes effect at once, without T-1071's lead, so at a fast clock its departures hop a little; tell the worker the step was a service edit. Open.

## Code health
- T-1114 [P3] A general code cleanup (Anita 2026-10-24), the rest: reviews in notes/T-1114/ (ui.md, app-core.md, sim.md, pipeline.md), most batches landed 2026-10-24. Left: app-core 2, 3, 4, 9 (perf switches, approved), 11; sim G's kernel.rs split and H; pipeline F items 19-20, G (the nocar layer, changes pack bytes, with T-878 and T-926), I. Pipeline G waits on a rebuild: the next data batch.
- T-1183 [P3] After T-1146: take the fixed-rate line path out of the engine (`capacity.rs` `continuous` rounding, `LineData.trains` as an Option, the test-only `Op::Schedule`, `set_line` and `set_schedule` in `wasm.rs`), which about 30 sim tests pin exact trains per hour on, so they need rewriting onto train counts; and regenerate the shipped `data/packs/real/*.save` (they load, converted on load) with `realsave` + `gamesave` at the next data batch (T-1303's staged saves are that regeneration). Waits on a rebuild: the saves with the next data batch.
- T-1304 [P3] `complexNow`'s train list (T-1096) is still worked out but nothing draws it since T-1289 drew the complex with the map's renderer: trim it. Open.

## Waits on a rebuild
Tasks that land only when a pack or country is rebuilt, grouped by that rebuild, so whoever runs it knows what rides along (2026-10-26). The two replan groups are also in notes/T-1189.md "What rides along".

### Europe replan (T-1189's Europe part)
- T-925 [P2] Rebuild the 9 European zones whose outline takes a NUTS polygon where OSM leaves a gap (boe ceu het aan lis les obe osk pot) with geoBoundaries as the fill: NUTS geometry is non-commercial and non-transferable, which clashes with ODbL share-alike on the published packs (T-547). The outline is the pack footprint, so each needs a pack rebuild (cells, water, market, outlines.json); after the Europe batch. Open; notes/T-924.md "Licences".
- T-892 [P3] Portugal's three parked rows (lez, meo, doo); maybe with T-1189. notes/someday.md
- T-617 [P3] Rebuild gla, lis and pot on T-610's unit matching. notes/someday.md
- T-772 [P3] Dead-space units holding cities (Valladolid, Agder...); Anita: low priority. notes/someday.md
- T-785 [P3] France's countryside zones on the France module's sources. notes/someday.md
- T-771 [P3] A Spain module from the ministry's phone trip tables and INE's grid. notes/someday.md
- T-853 [P3] A tertiary share where UNESCO has none (Kosovo's Prizren). notes/someday.md
- T-783 [P3] Melilla registered as Morocco, so its trips take the border factor. notes/someday.md
- T-765 [P3] Work at home is 0 in archetype D packs; a per-country constant needs Anita. notes/someday.md
- T-810 [P3] Commutes across open EU borders far too damped at 0.05. notes/someday.md
- T-429 [P3] Cross-border long trips in Europe low on land pairs (Paris - Brussels). notes/someday.md
- T-932 [P3] Countries solved in turn leave a held neighbour off its margins. notes/someday.md
Also T-1345's Nice (T-769), and T-1344's Paris and London items if those metros are rebuilt.

### US replan (T-1189's US part)
- T-1019 [P3] NCES schools, IPEDS and FAA enplanements for planned US rows. notes/someday.md
- T-1034 [P3] Planned US rows' home end from LODES RAC by block. notes/someday.md
- T-142 [P3] Private schools in US packs (NCES PSS). notes/T-142.md
- T-203 [P3] Medical sites mapped as multipolygon relations are skipped. notes/someday.md
- T-548 [P3] Rerun `hospital_beds.py pool` after T-203. notes/T-387.md
- T-235 [P3] Re-run the park draw on T-234's population. notes/someday.md
T-203, T-548 and T-235 re-enrich the nine US metro packs. Also T-1345's LODES lumps (T-043).

### China's next rebuild
- T-1306 [P3] Altay (aty)'s plan origin is GHSL's densest pixel, desert 130 km from Altay city; its test line carries no riders and it shipped on a relaxed browser check: move the origin to the city and rebuild aty. notes/T-1185.md "Result". Open.
- T-985 [P3] Wei and Linquan rows carry solve_region henan though in Hebei and Anhui. notes/someday.md
- T-996 [P3] GHSL population spikes (Dazhou, Benxi); measure after T-1150's bound. notes/someday.md
- T-1232 [P3] `osm_units.enclaves` can leave land in no outline (Ninghe). notes/someday.md
- T-1258 [P3] China's countryside packs hold few OSM medical sites; a fallback. notes/someday.md

### Korea's next rebuild, and Toronto's (tor was only re-solved in T-1239)
- T-564 [P3] Korea's registration as a tier 0 hook (Canada's DA population done in T-1239; its jobs by tract of work not yet). notes/T-555.md
- T-1243 [P3] Korea's resident registration counts by si-gun-gu for the plan rows. notes/someday.md
Also T-1344's Toronto (T-282, T-357) and Seoul (T-504) items, T-1345's Toronto airport (T-283).

### Russia's build (held)
- T-791 [P3] Moscow's jobs far too even (Anita: "definitely noticeable"); with the rest of Russia, held. notes/someday.md

### The next data batch on every pack
- T-926 [P3] Drop the unread `nocar` layer from the packs and Credits. notes/someday.md
- T-878 [P3] Drop the `nocar` build step from world_run and everyday_rates (with T-926). notes/someday.md
Also T-1114's pipeline G (the nocar layer) and T-1183's regenerated real saves, in Code health.

### Any pack's next rebuild (lands pack by pack)
- T-578 [P3] Older tier 0 packs (Paris, London, Sao Paulo, Japan) on current job rules. notes/someday.md
- T-315 [P3] Tier 0 universities matched to Wikidata by name. notes/T-315.md

## Someday
Short lines; the full context is in each task's notes file.

### Quick wins
Judged 2026-10-26: under about 2 hours of agent work each, self-contained, no data rebuild. Grouped by the files they touch, so one agent can take a batch. None found in the sim's track code.
#### App UI
#### Map
#### Pipeline (between runs: the runner's files are in use while one runs)
- T-612 [P3] Runner: delete a parked row's pack files once its reason is recorded.
- T-766 [P3] world_run records each extract's PBF snapshot time. notes/someday.md
- T-1227 [P3] The join runs `seams.py` first so the seamless list follows replans. notes/someday.md
- T-386 [P3] tier0's signals log totals inside the boundary only. notes/T-291.md
- T-943 [P3] market_zones' Python part at a gate's register, 28 of 33 s. notes/someday.md
- T-986 [P3] Region list upkeep for newly built countries. notes/someday.md

### World and countries
- T-1344 [P3] One-city data sources, 9 folded tasks (Paris Filosofi, schools and Flores; London HESA; Toronto StatCan and outline; Seoul KOSIS; Mumbai jobs and curve): take one when cheap or when its pack is rebuilt. notes/T-1344.md
- T-967 [P3] Fetch a wave's extracts ahead; keep region caches until the join. notes/someday.md
- T-968 [P3] The join's per-pack loops and the gate's register across processes. notes/someday.md
- T-969 [P3] Gate figures made in the build workers, not the serial gate. notes/someday.md
- T-970 [P3] About 8 s less build cost a row (h3, GHSL reuse, fewer processes). notes/someday.md
- T-592 [P3] Read the next region's extract while this one gates and joins. notes/someday.md
- T-570 [P3] Regional car-free homes from DHS and open census small areas. notes/someday.md
- T-571 [P3] National no-car figures for larger countries on T-562's GDP rule. notes/someday.md
- T-563 [P3] Fit an archetype A planned city's own curve at build time. notes/someday.md
- T-784 [P3] Low-memory devices: one demand worker, lighter settings (Anita: low priority). notes/someday.md
- T-847 [P3] A look at each newly joined country's zone names.
- T-1027 [P3] Coarse zone outlines overlap after simplification. notes/someday.md
- T-462 [P3] Japan zone tooling fixes from T-420. notes/someday.md
- T-505 [P3] University students by institution (France, Germany). notes/T-505.md
- T-506 [P3] Brazil's school census and DATASUS for tier 0. notes/T-506.md
- T-507 [P3] Nigeria from GRID3. notes/T-507.md
- T-512 [P3] Car-free households below country level outside the US, England, Japan. notes/T-512.md
- T-508 [P3] Overture Places where OSM's completeness is under 0.3. notes/T-508.md
- T-383 [P3] A shared cached cell-to-unit table per pack for measurement scripts. notes/T-383.md
- T-060 [P3] City packs on R2, the rest: a compressed .bin copy and the header in JS. notes/someday.md
- T-321 [P3] Local time's rest: ledger days and market trips across offsets. notes/T-321.md

### Demand
- T-1345 [P3] One-city calibration checks, 12 folded tasks (Central Park, LIRR and Metro-North, MTA peaks, Hudson fade, LODES lumps, New Haven - Hartford, calib's area files, Nice, Toronto's airport; ACS timing, Japan's gravity and survey gaps as fits). notes/T-1345.md
- T-263 [P3] Stadium events (attractions' stage 3), low priority (Anita 2026-10-24); holiday sightseeing goes with T-264. notes/T-263.md
- T-632 [P3] Walking feels too strong to play against; parked (Anita: try later). notes/T-632.md
- T-688 [P3] A distance decay inside the 2 km zone for very short trips (Anita decides). notes/someday.md
- T-634 [P3] A bike or short-trip mode before walking is refit on surveys. notes/someday.md
- T-529 [P3] Bikes to stations for car-free homes in Japan. notes/someday.md
- T-461 [P3] Japan's jobs total and public service. notes/someday.md
- T-454 [P3] Destinations that respond to the network; parked by Anita. notes/someday.md
- T-437 [P3] Long trips' modes: the long car's cost, the Channel, ferries. notes/someday.md
- T-438 [P3] The market's destination choice: giant metros take neighbours' trips (waits on Anita). notes/someday.md
- T-427 [P3] Longer-trip answers sent only for zones with trips. notes/T-377.md
- T-390 [P3] A station's walked/drove split for everyday and regional trips. notes/someday.md
- T-296 [P3] Weekends overcount commuters in a station's far-zone rows. notes/T-296.md
- T-298 [P3] Inspectors mix commutes between zones into longer trips. notes/someday.md
- T-280 [P3] Ferries as an access leg or line kind; parked (Anita 2026-10-13). notes/someday.md
- T-270 [P3] Drives across rivers and into dense centres too cheap (tolls). notes/someday.md
- T-059 [P3] Trips home chosen from the work end. notes/someday.md
- T-202 [P3] Rethink the 100 m station merge now complexes exist. notes/someday.md
- T-265 [P3] Signals simulated by clumps of shared track. notes/someday.md
- T-187 [P3] Trip chains (work, leisure, home); not a priority (Anita). notes/someday.md
- T-1240 [P3] A leaner per-cell sample index for coarse zones. notes/someday.md
- T-1241 [P3] Everyday station noise in thin zones between draws. notes/someday.md
- T-560 [P3] Crush load per country from published stock figures. notes/someday.md
- T-873 [P3] A train empties fully mid-line in the boarding account. notes/someday.md
- T-738 [P3] Flight data for long-distance demand (Anita's idea). notes/someday.md

### Performance
- T-1172 [P3] Load-progress bookkeeping 5-7% of the main thread in a big load. notes/someday.md
- T-1173 [P3] The clock worker's load is 5.7 s on the China save.
- T-1230 [P3] The clock worker's first boarding hour after a result, 1.3 s. notes/someday.md
- T-898 [P3] City pack opening, the floor of a big load (7 s a worker for Japan). notes/someday.md
- T-899 [P3] A save outside New York still opens New York's pack first. notes/someday.md
- T-1040 [P3] A long piece's ghost is a full trial edit (100-300 ms). notes/someday.md
- T-1031 [P3] The first ghost of each run takes 0.5-0.7 s. notes/someday.md
- T-879 [P3] Long tasks left when panning zoomed out. notes/someday.md
- T-875 [P3] The evening period from midday's choices. notes/someday.md
- T-876 [P3] Tiny far-zone station rows folded into a remainder. notes/someday.md
- T-877 [P3] Market rows' air lists once per airport. notes/someday.md
- T-777 [P3] Market rows per country instead of the whole file. notes/someday.md
- T-758 [P3] The track derive with water masks, 4.9 s for Japan. notes/someday.md
- T-499 [P3] Stop edits: lines sent whole and the buffer build. notes/someday.md
- T-281 [P3] Opening a big save searches every station's walk round water. notes/someday.md
- T-076 [P3] A lower frame rate when trains barely move (Anita's smoothness call). notes/T-076.md
- T-168 [P3] Add a newly viewed city without re-solving the whole network. notes/someday.md
- T-414 [P3] Each worker's first solve slower in the browser than native. notes/someday.md
- T-436 [P3] Start crowding rounds from the previous solve (Anita decides). notes/someday.md
- T-361 [P3] One demand engine per city, not two. notes/T-361.md
- T-354 [P3] Pack copies per worker and the final publish, spread. notes/T-354.md
- T-359 [P3] A bare start's card waits 2 s for New York in one worker. notes/someday.md
- T-344 [P3] A zone's loading bar can stick near 99%. notes/T-344.md
- T-346 [P3] A cached reload hides the fresh solve's estimates. notes/someday.md
- T-1543 [P3] The rough solve's one period split over the workers (each the same period, a share of the origins, the partial results summed on the page): the cold real save's first bubbles 8-9 s here, of which one period's free flow 4.6 s; about 3.5 s a worker split. notes/T-1538.md. Taking, 2026-10-27 (tree t1543).

### Interface
- T-1305 [P3] Real networks share track: corridors, one node per corridor; a few days, design with Anita (later, Anita 2026-10-25). notes/someday.md
- T-750 [P3] Add a city's real network to a running sandbox game (Anita wants it, not urgent). notes/someday.md
- T-685 [P3] Drawn city stamps at 10% rail share (stars done). notes/T-684.md
- T-651 [P3] Unopened zones look fully red zoomed out; give them their walk split (Anita's TODO). notes/someday.md
- T-602 [P3] Walk split for zones not loaded, the data half of T-651. notes/someday.md
- T-678 [P3] Day and night tint on the base map, after T-676. notes/T-676.md
- T-803 [P3] More station faces by time of day (Anita). notes/someday.md
- T-984 [P3] Region outlines fetched per region. notes/T-975.md
- T-956 [P3] WebKit keeps Homes bubbles drawn after Homes is off. notes/someday.md
- T-166 [P3] A station's riders by purpose on the map. notes/someday.md
- T-057 [P3] Kana and kanji on the basemap in Zen Maru Gothic. notes/someday.md

### Track and trains
- T-798 L3 [P3] One-way loops. notes/someday.md
- T-623 [P3] A station cut inside an arc still shifts the ramp. notes/someday.md
- T-666 [P3] An automatic bridge refused near a short first arc or tight turn. notes/someday.md
- T-668 [P3] Dragging a bridged segment refuses ground over water.
- T-248 [P3] Construct and level acting on a whole run between junctions. notes/someday.md
- T-339 [P3] Expresses pass locals at no cost on double track (Anita: "thats whatever"). notes/someday.md
- T-340 [P3] Quad track's rest: end-node stations, express pair side. notes/someday.md
- T-702 [P3] Station ghost leftovers on overlapping imported lines. notes/someday.md
- T-755 [P3] No room for ramps at Lake Hamana on the traced Tokaido Shinkansen. notes/someday.md
- T-763 [P3] Penn Station-Emeryville 3-3.6 s from rest to preview. notes/someday.md
- T-914 [P3] A snapped route Schiphol - Amsterdam Centraal refused over water. notes/someday.md
- T-430 [P3] A straight-then-arc fallback for too-tight clicks, only if wanted.
