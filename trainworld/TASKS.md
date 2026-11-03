# Tasks

Open agent work, one line a task; detail in `notes/T-###.md`. Format and rules in `CLAUDE.md`.
P1 = now or blocking, P2 = this milestone, P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.
Lower-priority tasks are under **Someday** at the end, one short line each. They are open tasks
(Anita 2026-10-26): count them when asked what is open, and work on them when nothing above can go
ahead; pull one back up when it becomes next. Anita's shortlist of candidates: `notes/for-anita.md`. Notes of archived
tasks that nothing open or current cites live in `notes/archive/t0001-0499/`, `t0500-0999/` and
`t1000-1499/` by task id (older ARCHIVE.md and DECISIONS.md links say `notes/T-###.md`: look there).

**Orchestrator handoff: start with `notes/handoff.md`** (live; history in notes/handoff-2026-10-26.md).

**Where things stand (2026-10-30).** The whole world is the goal, demand simple on purpose
(CLAUDE.md "Working here"; plain account in docs/demand-plain.md). Covered whole: Japan, Europe
except Russia, Belarus and Ukraine (held), China with Hong Kong, Macau and Taiwan, the US with
Puerto Rico, Korea, Canada, Vietnam, Thailand and Singapore; 1,897 zones shipped. Version 2026-10-28g
is published. Under way: release prep (the save review and legal fixes are in; text review, T-1840,
and Anita's playtest), and performance on big networks (Performance below). Southeast Asia is next,
with Cambodia and Malaysia (T-1738), then Mexico and Turkey; the replan of the US and Europe waits
for Anita (T-1189).

## World and countries
Sources: docs/demand-worldwide.md, docs/data-register.md. The unattended city pipeline: notes/T-535.md "Stages".
- T-1999 [P3] The planner's 50,000 zone floor (T-1984) would make dead space of shipped places that cannot merge with a neighbour on a replan (Aland, Ceuta, Vieques with Culebra, Nantucket, Trebinje, Gorazde): exempt islands and exclaves with no land neighbour in their region, rather than per-country figures (orchestrator's call 2026-11-02; Anita's rule: no place skipped for coarse geography). Before any replan of Finland, Spain, Puerto Rico, the US or Bosnia. Open. notes/T-1984.md.
- T-2000 [P2] Cambodia's north-west: GHS-POP (E2020 and E2025) puts 3-10x the census people in the Thai border districts (Trapeang Prasat 1.07M in forest villages, Sala Krau 314k against 34,815 in 2019, Phnum Proek 263k against 44,741), so the zones tpe anv phm slk (3.3M in the plan, about 0.5M real) are held. Needs district or commune census totals (2019 census; Open Development Cambodia has only provinces with polygons, its WFS timed out to scripts; citypopulation.de lists communes), then either a `cambodia` module's `admin_totals` (tier0) and a replan of the north-west on census people, or the districts as dead space. Open. notes/T-1738.md 11.
- T-2001 [P3] Other geoBoundaries files that dissolve same-named units into one far-flung feature, as Malaysia's ADM3 (T-1738's `world_plan.GB_SPLIT_FAR`): Portugal's ADM3 (74 features, Azores parishes such as Ribeirinha and Feteira on several islands; Portugal is built on NUTS3, so only if ADM3 is ever used), Indonesia's ADM2 (26 far features, check before Indonesia), Sierra Leone's ADM3 (2). List: `gbscan.py` in t1738b's kit, notes/T-1738.md 11. Open.
- T-2003 [P3] The joint gravity is symmetric, so a one-way border flow comes out both ways: Johor Bahru to Singapore 169,000 commutes a day at the MY-SG factor 0.5 (news estimates 250,000-400,000) but Singapore to Johor Bahru 160,000 as well (few in fact). A direction factor per border pair (borders.json), or leave it: a player sees the volume, rarely the direction. Open. notes/T-1738.md 11.
- T-2002 [P3] A preflight check for GHSL phantom people (T-2000): a plan row with 1 km pixels over about 20,000 people and no GHS-UCDB urban centre, or rural units whose people per built-up m2 stand far above their region's, as a FLAG naming the units. Open.
- T-1974 [P1] Build time speedups before more countries (Anita 2026-11-02: "this week i'd like to try adding more of the world. before we start on that itd probably be good to do some general thinking about build time speedups"). Measure where a country's wall time goes, propose ranked speedups. Design only. Taking, 2026-11-02, t1974 agent. notes/T-1974.md.
- T-1975 [P1] A streamlined country playbook (Anita 2026-11-02: "think about common pitfalls, in general streamline the process so that a new agent adding a country will be able to do so thoroughly and efficiently"). Pitfalls from past country tasks, token costs, a revised docs/adding-a-city.md proposal. Design only. Taking, 2026-11-02, t1975 agent. notes/T-1975.md.
- T-1994 [P3] A city centre reunited by `world_plan.FUA_WHOLE` sheds outlying pieces that then stay as they are, under the floor if need be (on a China replan: Fuding 0.47M and Zhenping 0.82M against China's 1M floor). Let a shed piece regroup by the merge rule without rejoining the city zone it left. Only matters on a replan or a new country with such a city. From T-1984. notes/T-1984.md.
- T-1995 [P3] A city zone under the FUA_WHOLE line is named after its most populous unit, a suburb at times (Newcastle's is "Lake Macquarie"; Australia's display names came by hand in zone-names.json). Name a zone holding a GHSL urban centre of CITY_POP after the biggest such centre, as FUA_WHOLE zones are. From T-1984. notes/T-1984.md.
- T-1978 [P2] T-964 (gate in chunks while the region builds), after T-1976 so chunks do not each pay a register. Saves 10-15 min on a 100-row region (Mexico). From T-1974, waits on Anita's go. notes/T-1974.md, notes/T-962.md.
- T-1979 [P2] A run fetches a country's World Bank indicators (and any module source) once before its builds, and a failed network fetch is retried in place rather than by rebuilding the row: Australia lost 8 builds to `tier0.tertiary_share` resets with 9 builds asking at once. From T-1974, waits on Anita's go. notes/T-1974.md.
- T-1980 [P3] The next region's download and read during this region's builds and gate (T-967's first half), one read ahead at most for the commit limit (T-1080): Australia's 20 min, more for Indonesia, India, Brazil and the US replan. From T-1974, waits on Anita's go. notes/T-1974.md.
- T-1981 [P2] Measure the joint solve's thread scaling (CPU poll during the next country's join, and one unit at 6 and 12 threads): solves are 17-30 min a country, 2.4 h for China. Then decide night joins at 12-15 cores, or (Anita's call, it changes results slightly) a rerun's unit starting from its last factors. From T-1974, waits on Anita's go. notes/T-1974.md.
- T-1988 [P3] A market pack's `air_direct` (the places its airports fly to directly) is as of its last write, so it misses zones registered since: 79 of 151 sampled packs differ from a fresh build in that field only (Vietnam's, Australia's, Europe's). Refresh it for every pack after a join (`market_zones.py --air` rewrites all 2,000 and redoes every row, about 4 min), or only where a new zone's airports are on a pack's direct routes. From T-1976. notes/T-1976.md.
- T-1992 [P3] A cold `zone_outlines` run (empty cache) takes 4.4-5.1 min at 4 jobs where T-1977 aimed at 3; 860-960 CPU-s of it is zone_water's 368 windows (owner_cells 37%, the reach buffers 21%, owners' points 8%). Only changes that keep outlines.json byte-identical (hash in notes/T-1977.md): ridge lines and face tests of `owner_cells` as arrays, a zone's reach shared between windows where its clip is the same. Open; notes/T-1977.md.
- T-1738 [P2] Indonesia taking, 2026-11-02, t1738c agent (notes/T-1738.md 12). Southeast Asia, starting with Vietnam (Anita 2026-10-28: the next compute-heavy task once T-1734, T-1735, T-1374, T-1520 and T-1625 are done or under way and nothing is pressing). docs/adding-a-country.md and the world plan; a data-register row per country. Vietnam done 2026-10-28 (93 rows shipped; Muong Nhe and Con Dao are dead space, T-1744; eleven origins moved into their towns, T-1762); Thailand done 2026-10-29 (67 of 67 shipped: Bangkok and 66 zones) and Singapore (sin shipped; the Johor Bahru border pair waits for Malaysia); Cambodia, Malaysia and Brunei done 2026-11-02 (47 rows shipped; Cambodia's four north-west zones held, T-2000); open for the next countries (Philippines, Indonesia, Myanmar, Laos, Timor-Leste; list in notes/T-1738.md 5). notes/T-1738.md.
- T-1765 [P3] Zones whose origin was set (before T-1762) on the densest pixel anywhere in the zone, over 25 km outside the unit they lead with. Done 2026-10-29, after Vietnam: Canada's five (North Bay 217 km, Riviere-du-Loup 154, Barrie 98, Tracadie 92, Sherbrooke 47), Kerkyra 172, Siedlecki 137, Starogardzki 104: rebuilt and joined, shipped. China's eleven city zones (Kashgar, Linhe, Yizhou-Fukang, Altay, Liangzhou, Zhaoyang, Taobei, Wanzhou, Jiyuan, Zhaodong, Guixi) rebuilt and joined the same day, shipped. Left: China's other 61 (county towns: Guyang 235 km, Nehe 145, Huzhu 129, Lancang 123). Also over 25 km: 5 more in Poland (Sieradzki 70, Leszczynski 52, Grudziadzki 32, Szczecinecko-pyrzycki 29, Nowosadecki 27), 6 in Germany (Mecklenburgische Seenplatte 65, Rendsburg-Eckernforde 35), 2 in Hungary (Hajdu-Bihar 63), Valais 67, Innviertel 41, Zenica 44, Kortrijk 41, Macvanska and Sremska oblast (30-37) in Serbia, Mid Ulster 33, a few in Croatia, Romania and Bulgaria, and Thailand's 7 (28-57 km, mostly in their provincial town); Korea and the US not checked. `python trainworld/checks/zone_new_origins.py <ids> --out f.json` gives the rule's origin for any row without a replan (China's: artifacts/trainworld/t1765-china-new-origins.txt), `python trainworld/checks/zone_origins.py` lists the far ones; lists in artifacts/trainworld/t1762-far-origins-before.txt and -after.txt; notes/T-1738.md 10.
 The runner's market heal reaches only 400 km, so China's run left three packs naming a freed place (healed by hand): widen it before the next big China rebuild.
- T-1375 [P3] The zone ring cut halves a ring by people round the centre it wraps, so a thin arm (Whistler and Pemberton, joined to the Fraser Valley zone round Vancouver's row) splits the ring's real body instead (Abbotsford 0.17M and Chilliwack-Mission 0.19M, T-1239): try shedding the arm that alone makes the wrap before halving; a Canada replan after. Open; notes/T-1239.md section 1.
- T-1376 [P3] Les Îles-de-la-Madeleine (lel, 12,000 people, an island zone of Canada) parks in its build: `enrich_parks` finds under 85% of its park draw on a cell (beach and dune parks off the populated cells). Allow a small island zone through (as T-897's one-park islets) or leave it dead space. Open; notes/T-1239.md.
- T-1189 [P2] Replan the built countries by T-1140's merge rule: US and Puerto Rico (162 of 187 rows), Europe (about 393 rows), Taiwan (3), each a dry run and a look, then builds, gates, joins and T-1053 successors. Waits for Anita's word ("we might run into other things that will make us want to rebuild again"; asked again 2026-10-26: not yet). China's part done 2026-10-25 with T-1185. Carries what T-1002, T-1005 part 2, T-1089 and T-903's merges asked (Zhubei split, wraps, rings, small Italian zones); T-1307, T-774 and T-891 first. notes/T-1189.md. What rides along: notes/T-1189.md "What rides along" and "Waits on a rebuild" below.
- T-1193 [P2] T-1178 stage 2, what is left: rerun the solve units holding a pack in T-1192's seamless list outside mainland China (GB's Bristol group, Italy's Milan-Como, Puerto Rico; check whether Taiwan's ran with China's joins in T-1184 and T-1185), after T-1261, rerunning `pipeline/seams.py` first if ids changed; check Guangzhou-Foshan's commuters across near 1.3-1.5M. notes/T-1178.md.
- T-1310 [P3] tier0's cell cache (`data/work/<id>.tier0_signals.npz`) is reused even when the pack's outline changed, so a rebuild for a new outline keeps the old cells (T-1144's first hon run). Key the cache on the outline's hash, or have `world_run run --rows` of a rebuilt row delete it. Open.
- T-1334 [P3] Other coastal districts dropped as Kwai Tsing was (T-1144)? List each built outline's OSM relations at its picked admin level with 0.25-0.5 of their land inside a plan unit and not picked (a coarse unit polygon missing reclaimed land or islands, the leftover under the 10% fill); rebuild any populated ones. Open.
- T-1080 [P2] A world run at home can fail extract reads on Windows' commit limit (74.5 GB here): each parallel GDAL reader in `osm_features` commits a 6 GB node index, so with a join, a big profile and browsers up, Wisconsin's and Nebraska's reads failed "Failed to open dataset" with 31 GB of RAM free (T-1078). Check free commit before a read and wait, or read with fewer parallel readers when it is short. Open; notes/T-1004.md "Wave 2".
- T-774 [P3] `world_plan.py`'s metro replan does not know the `held` status (plan_config.ROW_STATES), so a replan would give held rows new ids and orders; check and add it. Open.
- T-964 [P2] Gate a region in chunks as its rows finish, while its later rows build, instead of after the last build (wave 1's northern region: builds 114 min, then the gate 73 min with every build worker idle; about 25-30 min a 100 rows). Open; notes/T-962.md.
- T-930 [P2] The join's re-gate re-checks only what a joint solve can change (mean commute, top zone share, joint borders, other packs unchanged) instead of every figure of every member: the GB join for hih (2026-10-22) spent 9.8 of its 31 minutes there. Open; notes/T-928.md section 4.
- T-1309 [P3] 384 new Chinese zones show the planner's county name: a look at their names (docs/zones.md "Names"). Open.
- T-428 [P3] `data/packs/intercity.json` (14 MB) is only the joint gravity's input and output, never read by the game since T-377: move it to data/work, out of what is published. Open; notes/T-428.md.
- T-1635 [P3] On the live page a pack's whole-file fetch right after its Range reads (sed.bin) can fail once with `net::ERR_CACHE_OPERATION_NOT_SUPPORTED` (immutable Cache-Control); the client's retry gets it 0.9 s later, but it logs a console error and smoke counts it. Fetch whole files with `cache: "no-store"` or skip the Range reads when the whole file follows. Found 2026-10-28, scratchpad tlive/result.json. Open.
- T-941 [P3] The published page's expected 404s: the real-rail layer asks `../noritetsu/regions.json` before R2 (404 on anita.garden every load), and the two Canadian syllabics glyph ranges glyphs.py leaves out; smoke against the live page counts both as console errors. Try R2 first when the page has a tw-data meta, and make smoke's glyph filter match there too. Open.

## Demand
- T-1753 [P3] Rapid and express patterns for Tokyo's long suburban lines in the real save (about +2%); needs an open timetable source (ODPT needs an account). notes/T-1750.md. Open.
- T-1503 [P3] Superholiday rush days on networks past about 8,000 stations take the longer trips unleaned (the lean's S x S table over 256 MB, `LEAN_OD_BYTES`, T-1373): a sparse or per-destination table if a network that big plays. notes/T-264.md Since T-1545 the table is gone (the lean is a column per station in the pass's window), so lifting the gate costs only time: a walk a destination a pass.
- T-1504 [P2] The China network's longer trips barely ride rail: about 3,300 line boardings a weekday for the market's longer trips on ten high-speed lines between China's 24 biggest metros (t1156-all-china.save, 39 million rail trips in all), found measuring T-1373. Analysis written 2026-10-26: fares weigh too much on long trips (T-1431's root, x4 alone), the save's "HSR" runs at 160 km/h with 6/3/2 trains and its stations are not linked to the metros, which a zone's 8-station access list then never reaches; sampling is not a cause. Proposal and Anita's questions in notes/T-1504.md; follow-ups T-1506 to T-1508.
- T-1599 [P3] T-1508's option 2, deferred: after T-1603, look at a built Beijing - Shanghai high-speed line at player scale (riders a day, does it feel too empty to a player?). Done when (no refit): the line carries at least about 50,000 riders a day both ways with good service, the busiest of its kind on the map, so a player sees it working; only if it falls short, the tail's existing constants refitted (`pipeline/longhaul_fit.py`) against all-mode targets with rail counted (NHTS, MLIT, China's rail and air, the big pairs), and every country's market data rebuilt. Waits on Anita. notes/T-1508.md.
- T-1233 [P2] T-1177 leftovers: run `checks/t1177-cross-everyday.mjs` with `IDS=gua,fos` once Guangdong's market packs carry the pairs (compare each purpose's local legs plus the pairs with the old totals); the explorer's Everyday trips block shows only while a network reaches the market (no walking figures for two zones with no line, as there are no corridor figures for the pairs); a city's own purpose counts and homes bubbles stay local legs less the kept share. Open; notes/T-1177.md.
- T-1396 [P3] A car between landmasses by ferry: the market has no car at all between places on two landmasses (London - Dublin, Athens - Heraklion, Barcelona - Palma); T-279's ferry rule (docs/regional.md 5.1) could give them one, which would add pairs with only a car and a plane. Also rerun `pipeline/market_drives.py build` once Canada's market rows land (23 Canadian zones had none on 2026-10-26). notes/T-279.md.

## Performance
Ranked in notes/perf-survey.md and notes/T-1156.md (the all-China save) and notes/T-1200.md (slow CPUs).
- T-1730 [P3] T-1609 follow-up: a browser check that an edit right after a local refine solves the cluster whole (the refined periods are marked `~local`), and that a zone going coarse is refined locally too; neither was run. notes/T-1609.md. Open.
- T-1610 [P2] T-1206 leftovers: a clicked station's or bubble's far ends and purposes in a zone still rough wait for its refine (the 8 px ring) instead of reading coarse samples; the fine set's budget corrected by the workers' reported WASM after each solve (now T-1197's rate alone). notes/T-1223.md. Open.
- T-1629 [P2] T-1206 stage 5, split from T-1611 (which did the memory alone): the coarse zones' bias measured with the fine set moving, on the Paris and Shanghai save. notes/T-1206.md, notes/T-1223.md. Open.
- T-1169 [P2] Demand over many zones: memory per worker, so all 723 Chinese zones fit the 4 GB WASM ceiling (est. 4.3 GB held today). Stage 1 (measured breakdown and staged plan) done 2026-10-27, design with Anita; stages 2, 2b and 3 (T-1558, T-1559, T-1545) done 2026-10-27, all of Japan now peaks at 1.33 GB a worker; stages T-1223, T-1560. Done when: an all-China save with every zone reached solves with each worker under 3.2 GB (margin under the 4 GB ceiling), and all of Japan peaks under 2 GB a worker. notes/T-1169.md. Open.
- T-1633 [P3] T-1605's small kind: about 1 all-China cold load in 26 moves one line's commute figures by 0.0016 riders a day (line 118, coarse zones in western Liaoning), invisible. Pin it from a load that records opens; leads: what a reopened coarse zone shares with its earlier copy or the thread (`STATION_WALKS`, skims cache), `RandomState` keys advancing with history. Open; notes/T-1605.md.
- T-1634 [P3] Every all-China load closes about 40 reached zones at ~13 s and reopens them a second later in every worker (`sweep` running before the loaded network's reach is complete): wasted downloads and opening, and a load history that varies. Keep reached zones open while the reach is being worked out. Open; notes/T-1605.md.
- T-1560 [P3] T-1169 stage 5, design only: city zones dealt between solving workers instead of periods (each worker its share of the zones for all five periods, loads added on the page each round as T-1543 adds parts), the route to the whole world. After T-1558, T-1559, T-1545 and T-1223 are measured on an all-China save with every zone reached (Anita 2026-10-27: can wait). notes/T-1169.md section 5 (H). Open.
- T-1354 [P2] The market's corridor figures depend on which market packs were open at a worker's first composition: `MarketPacks` keeps the corridors (`corridors_of`) worked out then for every later composition (T-1204), but they differ with the packs open (on the real save, a worker that composed once from all 13 packs and the workers that composed as packs arrived had corridor tables ten rows apart, 1,530,357 and 1,530,367). So a page's longer trips depend on the order packs arrived; maybe also T-1225's load-to-load finals. Make them depend on the world's rows alone, or recompute when a pack adds a near pair. Open; notes/T-1174.md. T-1225's look (2026-10-26, `checks/t328-cache-check.mjs` with `MODE=reload DUMP=1`, per-section hashes in its json): the two hashes did not come back, 26 reloads of the real save gave one (4332e0ad), also with two checks at once. What does differ is a game loaded by import (`MODE=prepare`, hash df1cc482) against a reload (4332e0ad): same corridor table and `railAt`, but every other section differs at float-rounding size (relative ~1e-7: a line's day total 1,261,111.44 against .485, longer trips 1,616.8592 against .8593), in the lines, stations, purposes and cities too, so not longer trips alone; the rail total rounds to 12,189,003 in both. Likely the same cause (different pack arrival, so a different composition and sum order), unproven. To pin it, delay some market packs on purpose and diff the dumps.
- T-1366 [P3] The demand pool's memory cap (T-1205) never adds a solving worker back in a page: a big game's zones opened for a while (or a big game then New York from the menu) leave it at 1-2 workers, solves slower. Grow it back when the projected memory has room for one more, with a margin against flapping (a spawn, reopen and re-dealt periods, as `shrink` in reverse). Open; notes/T-1205.md.
- T-1367 [P3] After zones or a market close, a demand worker's heap still holds them (frees deferred while busy, the old market parts the kept clusters reference) until its next solve, so T-1205's restart waits for one (all-China: heap at rest 2,041 MB after three zones closed, 1,502 at load with the same items open). Free those at the close, or count them out of the need. Done when `checks/t1205-memory.mjs` passes on the all-China save (T-1205's bar; its last run stalled in the solve after the close, unexplained). Open; notes/T-1205.md.
- T-1358 [P3] `followTrack` samples every edge's drawn line each km on a game's first network (about 90 ms at 6x on New York); cut it with the same zones found (it cannot wait past the paint: its zones feed demand's rounds; since T-1021 it runs in a task of its own right after the state's), and `toWorld` about 12 ms an edit. Open; notes/T-1357.md.
- T-1505 [P3] After T-1380, a station edit that makes a new name label still spends 100-140 ms at 6x in `StationLabels.place` (likely the layout forced by reading the new label's width, unproven): measure the width without forcing a layout, or place only the names near the edit. Open; notes/T-1029.md "What is left".
- T-1509 [P3] A line whose length or drawing order changes (stops dropped, a line made, undo of either) still posts the line strokes whole to the page (about 35 MB on Japan, 4 of 12 edits of the T-1381 series), since the segments after it move; keep the lines' order stable or post the moved runs (map/strokePost.ts). Open; notes/T-1029.md "What is left".
- T-1211 [P3] The page's DOM makes every layout cost: 2,200 elements (875 station names on the map), a forced layout 22-37 ms at 1x; the Lines tab opens in 360-400 ms at 6x (since T-1021 each chip reads `currentCSSZoom` after the paint and measures a new symbol in idle time; not measured again at 6x), the saves panel 534 ms (`showModal`), the Stations tab still 0.6-0.7 s at 4x with 80 rows (`Chips` measuring). Open; notes/T-1200.md.
- T-921 [P3] Performance: commute zone pairs under 1 person a day dealt in chunks (T-874's deal with the chunk in people a day, `c = 1 x period share`, so every period deals the same pairs): New York solve 34.3 to 29.6 s (-14%), stations under 0.8%, ceilings unchanged. Measured in notes/T-909.md option 1. Open.
- T-1647 [P3] Cheaper market searches per node: T-1608's early stop saves only 6% of nodes (every destination has a far longer-trip start), so profile `Search::run`, `entry_costs` and `trip_km` on the native Japan pass (`t1545_dump`) for exact savings. Done when the native pass is 15% faster with figures identical. Open. notes/T-1608.md.

## Interface
Anita's TODO.md short term, filed 2026-11-02 (her 10-31 to 11-02 playtests; verbatim in notes/todo-2026-11-02.md; items may repeat earlier ones).
Her 11-02 second batch (verbatim in notes/todo-2026-11-02b.md):
Her notes from testing the 11-02 landings (verbatim in notes/todo-2026-11-02c.md):
- T-2036 [P3] Kuala Lumpur's new-game camera and gate test line sit in Ampang, east of the centre (origin rule: densest pixel of the named unit); move it to the centre by hand as Thailand's five were (notes/T-1738.md 7), rebuild and join. From T-1738's report. Taking, 2026-11-02, t1738c agent (with Indonesia's join).
- T-2009 [P3] The elevation hovers' "Upkeep ... per km per day" (T-2006, `upkeepTip` in ui/BuildTab.tsx) gives one double track's rate under Quad, as the "$75M per km" line does; Quad lays two doubles, so double both figures or label them. Open.
- T-2043 [P3] Bring checks/t988-cloud.mjs up to T-2037 (it needs a local shonei-server, not run): section 7 expects a fork slot named "T-007-real-nyc at <time>" that the page moves into; now the playing slot splits off in place (same name, new account id, same origin). Its icon list drops "Earlier versions", and its account-backup rows are `.save-version-list li[data-kind=accountBackup]` under `.save-versions-line`, restored under the game's name. Open.
- T-1916 [P3] Drawing beside track: in the real-save sweep (`beside_sweep_real_save`, sim/src/track/wasm/beside_tests.rs) 32 of ~85,000 previews still cross the track followed, where the concentric line itself crosses the run (e.g. the N line by Canal St, -74.0030, 40.7185) and the tool falls back to the plain route; also 46 on-line starts refused as "on top of existing track" beside a parallel line. Open.
- T-1917 [P3] A ramp spread over several legs (T-1895) can move when two pieces are joined under a new platform (the join widens the room it spreads in); harmless while platforms refuse ramps, but check `unjoin_near` joins keep heights. T-1964's drawn sweep: 58 of 6,851 spots near a level-change joint refused for the ramp a join moves (shown as the platform on a ramp), 1 with a curve reason. Open.
- T-1970 [P3] The station tool slides a station off a ramp it is clicked on near the ramp's end (at most half a platform and 2 m, as T-1870's room slide), from the ramps in the edge's drawn heights; today "The platform would be on a ramp." stands. T-1964's drawn sweep: 1,012 of 6,851 spots refused for a ramp. notes/T-1964.md. Open.
- T-1897 [P2] Track costs and rules: at-grade road-crossing cost, are costs realistic (params proposal to Anita), no viaduct over very long or deep water, no cut and cover, rural maintenance realism. Proposal approved 2026-11-02; T-1928, T-1930, T-1931 done, T-1929 (road crossings) open; notes/T-1897.md.
- T-1898 [P2] Blueprint selection: box moves only waypoints inside it, set regular or high speed on a selection, dragging a 320 km/h blueprint's middle point keeps 320. Done in tree .tw-trees/t1898 2026-11-02, to land (WASM interface: adds `set_edges_hsr`); `checks/t1898-select.mjs`.
- T-1948 [P3] A node drag's new lead-in toward a loose end takes at most 0.3 of the leg (`lead_in_capped`), so a 1 km straight with its joint dragged 108 m sideways gets a 4,400 m curve (250 km/h) where about 9,000 m fits; let it use the whole leg when the far end is a track end. Open (found in T-1898).
- T-1949 [P3] Node and pick drag previews draw high-speed track as ordinary track (`previewStrokes` in map/nodeDrag.ts passes no `hsr`; the curve bend does). Open (found in T-1898).
- T-1929 [P2] Ground track pays per road crossed (US$6M, motorways 3 crossings), counted from the basemap's zoom 14 roads on draw and kept per edge; tag and quote show it. Anita 2026-11-02: go. After T-1895 lands (drawing code). notes/T-1897.md 2.
- T-1902 [P2] Trip explorer: effective time against share confusion (Seoul-Busan, Tokyo-Seoul), wording, clickable and labelled waypoints, "walk too long" while people ride, large-city pairs "can't be worked out". Done in tree .tw-trees/t1902 2026-11-02, to land; notes/T-1902.md.
- T-1920 [P3] The views deal a zone's rail commuters over its work cells by jobs, so cells beyond any station's reach get train riders their own trip has none of (5 of 71 New York pairs; the explorer now shows a lifted route). Deal them only over cells within reach? A demand rule: Anita first. notes/T-1902.md.
- T-2045 [P2] During a big load, the zone in view's own figures first inside the whole solve's round 0 (its first period, the others estimated), so a city zoomed into shows demand in seconds, not at round 0 (38 s on Tokyo Seoul Fukuoka Busan, minutes on China); finals to the bit. Proposal for Anita first: notes/T-1904.md "Proposal".
- T-2046 [P3] Seoul on Tokyo Seoul Fukuoka Busan: the first Homes/Destinations switch in the seconds after zooming in waits about 6 s, until its detailed solve lands; main thread idle, cause not found (maybe a headless frame stall). notes/T-1904.md "After".
Anita's playtest of 8800 (build 40593370), 2026-10-26 morning: the lighter map "feels nice", Traffic view "looks pretty good", related lines "looks great", adding local games to the account worked.
- T-1856 [P2] Terrain cells as hexagons, our own lattice (Anita 2026-10-30: yes), after T-1854 lands so price and shape changes stay apart. notes/T-1854.md. Open.
- T-1858 [P3] Loading the all-Japan save takes minutes before the first city shows demand; after it the other cities come fast (Anita 2026-10-30: "that's fine then"). Investigation stopped early; its last note: a demand cache exists that a load might show first. Look only if she asks again.
- T-1861 [P3] Track placement on the all-Japan save still feels a little unresponsive (Anita 2026-10-30: expected for its size, but room to improve). After T-1794; measure with t1407-edit-ghost.mjs on a Japan save. Open.
- T-1849 [P2] Each demand worker's corridor table takes 2.6-3.1 s the first time it has market figures (7.9M pairs, `corridor_modes` 2.2-2.4 s): compute modes only for the 1.9M pairs kept (~1 s). Rust and WASM interface change, own tree, figures checked with `calib --cells --exact`. notes/T-1847.md. Open.
- T-1863 [P2] The settle leaves lines with no shared track unchanged to the bit, as track-and-service.md says (T-1850 stage 2, sim/src/track). Done when `netdiff.mjs` shows 0 changed Tokyo lines after the Ceuta edit. After t1854b leaves sim/src/track. notes/T-1850.md. Open.
- T-1867 [P3] Draw a destination's everyday homes as the smooth spread their samples stand for (T-1866 fix 1, drawing only). On hold: look again once T-1874 has landed and Anita has seen it. notes/T-1866.md. Open.
- T-1875 [P3] `t671-ghost-hover` fails "near track ends" on the real save (14 of 84 hovers, 2 chains): the page's guess says a spot fits and the worker refuses it (a flat crossing the page's marks miss; an end station the worker finds too short). Not from T-1870's change (same candidates before). Page/worker agreement on new blueprint near flat crossings. Open.
- T-1876 [P3] Australian jobs from the 2016 census Working Population Profile by destination zone (ABS DataPacks, CC BY 4.0) instead of tier 0's proxy (2021's place of work was taken in lockdown). notes/T-1857.md. Open.
- T-1878 [P3] The zone planner never splits a zone from its own pieces hundreds of km apart (seen in Australia, T-1857). notes/T-1857.md. Open.
- T-1879 [P3] A metro rule for 1M+ cities GHSL's urban centres miss (Adelaide was made one zone by hand, T-1857). Also add to trainworld/CLAUDE.md's join rule: while a gate or join runs, treat data/terrain like data/packs (a terrain rewrite mid-gate likely parked Tasmania's rows once). Open.
- T-1880 [P2] Do buses act as feeders? (Anita 2026-10-30, for next week: "we should check wherher buses actually act as feeders? like is it possible to make a bus line that is unprofitable by itself but considered with the stuff it links, actually increases total profit?") Measure: a feeder bus on a real save, the network's total profit with and without it. Open.
- T-1881 [P2] The hillshade draws dense centres as hills (Anita 2026-10-30: "tokyo and nagoya seem to have some hills in suspiciously building dense areas... also shanghai"): the track price classes were already flat there (T-1854); the hillshade comes from the public Terrain Tiles, a surface model with buildings. Switch it to bare earth or mask it in built-up areas. notes/T-1854.md. Open.
- T-1882 [P3] checks/t1270-terrain.mjs has 2 stale checks (the quote layout before T-1839; the track key "r" is now "1"). Open.
- T-1889 [P2] Riverbanks priced as hills or mountains: the East River crossing at 42nd St tags "Mountains 2.5x" at zoom 14 and "Hills 1.6x" at 15, from rough cells on the Manhattan and Roosevelt Island banks (T-1854's classes; bank and seawall slopes in the 30 m data). notes/T-1887.md. Open.
- T-1944 [P2] Mid-island Midtown priced as mountains, away from the riverbanks of T-1889: a ground ghost due north on 6th Ave from 34th St to Central Park South (1.8 km) tags "Mountains +3.0x" with a 1.54x multiplier, about 40% of it charged as mountains, and one along 45th St the same (found by T-1929's check, checks/t1929-roads.mjs). Open.
- T-1891 [P1] "Demand ran out of memory" on a brand new New York game after one line (Anita 2026-10-31, Chrome at 16 GB in Task Manager). No leak reproduced (memory flat in every run); the alert said "memory" for any second worker panic, fixed in game/loading.ts, so the real cause is most likely a demand panic, not found yet: needs the panic text (now in the alert's hover) or her save. notes/T-1891.md
- T-1873 [P2] Muting other lines: overlapping low-opacity segments look like a worm; mute as one layer (a blanket over the others, or the layer drawn opaque then faded). After T-1848 lands (line drawing). Open.
- T-1864 [P3] Barrier solves (3 workers) keep clean market clusters, which T-1607's split marks all dirty (T-1850 stage 3). Measure how often it happens first. notes/T-1850.md. Open.
Anita's playtest of buses and the round, 2026-10-26 (evening). "buses are cool!"
- T-1481 [P3] Marey diagrams (Anita: nothing built yet makes them interesting, but might be nice). Idea. Open.
Anita's answers on T-1411 and T-1412's ideas, 2026-10-26 (afternoon). Not wanted: riders before you pay ("people can just build it and find out"), growth near good stations (confusing), flights or freight you run.
- T-1519 [P3] Income groups in the local layers (Anita's call): three groups as the market's (T-1431) would run the local mode choice three times, about 2.4x the local solve time on New York (72% of it is the mode choice). notes/T-1431.md "Round 3". Open.
Anita's TODO.md "short term", filed 2026-10-26 (night).
Anita's TODO.md short term, filed 2026-10-27 (morning). Her priorities: memory per zone and worker high; rest of the world after the finer demand calibration.
Anita's TODO.md short term, filed 2026-10-27 (afternoon).
Anita's TODO.md short term, filed 2026-10-28 (afternoon).
- T-1681 [P3] Doubling up the real save: 122 of 1,093 double-track pieces are refused with PlatformCrossing (another track crossing at the same level where the pair's twin platform goes), found by T-1678's scan. Check whether that is right or the twin station could go elsewhere. Open.
Anita's TODO.md short term, filed 2026-10-28 (evening).
Anita's TODO.md short term, filed 2026-10-28 (night).
Anita's TODO.md short term, filed 2026-10-27 (night).
- T-1650 [P2] Cut a big connected network into parts for demand (Anita 2026-10-28: "one fundamental issue is that it's one connected network; could we find cutting points?"). All of Japan is one cluster, so every edit re-solves the country. Design first: find cuts where few riders cross (weak links between regions), solve each part with the others' crossing flows frozen, and exchange the flows now and then; a cheap version of T-1609's frozen loads for edits. Measure the error against a full solve (Anita accepts lines 10-20% off for about half the time, ideally with the part in view exact). After T-1608, T-1607, T-1609. Also lets T-1665's "constructing..." clear per part (city) instead of per whole solve (Anita 2026-10-28 wants it per city). notes/perf-overview.md. Design written 2026-10-28 (notes/T-1650.md): no cut by region (Japan has no weak links); re-solve the edited populations against a frozen market instead. Anita 2026-10-28: go ahead, T-1735 too; constructing clears zone by zone, intercity lines included. Stages T-1733 to T-1735.
- T-1736 [P3] After 40 minutes of edits on all of Japan the last whole solve differs from a fresh solve of its network in one zone (Okayama: 1 line, 13 stations, about 2e-9 of their figures; T-1734's run `jp-steady1`). Find which state carries over (a level refine's periods, a market kept across networks) and make it equal to the bit. Larger on master too (T-1764, 2026-10-29): after 17 lines in a row on japan-1930.save the whole solve is 0.16% under a fresh one (24,559,288 against 24,599,016 rail; 45 of 46 zones, stations up to 2%). notes/T-1734.md, notes/T-1764.md. Open.
- T-1764 [P2] Touch rule, rule 2 left (rule 1 done 2026-10-29, see ARCHIVE): a line with at most 3 stops in a zone's view (one of its own plus pads) no longer touches it on trains an hour or run times. Measure first: the pads' transfers (a rider alighting at a pad to change to a line whose span starts there), and how many zones it spares on all of Japan (the 40-line run touches 12 zones, 9 on the Tohoku Shinkansen's real trains-an-hour cut). notes/T-1737.md, notes/T-1764.md. Open.
- T-1734 [P2] T-1650 stage 2, an edit's solve runs to its end. Built 2026-10-29 in tree tw/t1734, not landed: newer networks wait on the page until its final, then go locally on top; all of Japan building a line a minute lands a whole solve every 11-14 minutes (China's to the bit, Japan's but one zone at 2e-9, T-1736). The cost: an edit during a whole solve waits up to 14 minutes on Japan (T-1733 alone: 2). A local solve between the whole solve's rounds can't meet both targets at a line a minute (notes/T-1734.md). Anita 2026-10-29: keep T-1733's behaviour for now, but come back to it for further thought (tree tw/t1734 kept; idea space: notes/T-1734.md).
- T-1763 [P3] After T-1644, the overlay at zoom 15 on all of Japan still draws every station box (9,529, three passes, about 1.1 ms) and train (about 0.8 ms) whatever is on screen; cull both to the view with a spatial index (about 1.5 ms a frame; also the 18-19% GPU with the clock running and the map still). notes/T-1644.md. Open.
- T-1760 [P3] A new express shows about a quarter of its market riders until the whole solve lands on all of Japan (a 623 km copy: 26% in play, against 72-85% at free flow on T-1650's network; China's HSR express 64%). Two causes: the first edit after a load keeps the evening's market on another worker, which gets no window; and with crowding the whole solve moves riders off the line it parallels whose pairs have both ends outside the window. Ideas: a window in the shared period (old loads from the worker holding it), the window's pairs over a few crowding rounds, windows sized by the pairs riding the changed lines. notes/T-1735.md. Open.
- T-1471 [P3] Drives view: the commutes' table with no rail takes 1.1-1.9 s in the aux worker once per zone and day on New York (a loop over every zone pair, 41M); sum it by square pair from the gravity's rows and columns, or keep it with the pack. docs/demand-views.md "Car trips by pair of squares". Open.
- T-1444 [P3] A game mode with a city's real network run as a rival to win riders from (Anita: a new game mode). After T-1305. Open.
Anita's playtest of buses along roads, 2026-10-26 (night), Shenzhen, Bus 13.
- T-1501 [P3] Bus legs that stay straight look like road legs. A cue (dashed, say) if players are confused about where a bus does not follow the road; the sim already knows (point kind bits, notes/T-1465.md). Open.
- T-1502 [P3] Bus money by region on a road route: `regionMoney.ts` shares a route's road km by the chords between stops, rougher on a winding route; share by the route's own points (notes/T-1465.md). Open.
- T-1467 [P3] Buses stage B3 if wanted after playing (notes/T-1445.md): a bus lane upgrade, ferries on the same line kind (speed by place done as T-1577). Buses are settled (Anita 2026-10-27): only if she asks. Open.
- T-1463 [P3] Rename "train share" and the like now that bus-only trips count in it (Anita: later). Open.
- T-1445 [P2] Buses, design first (Anita): draw a line (along roads if that is easy) and stops place themselves, where it meets a train station and otherwise every 400 m or so; maybe no concept of stops at all. A cheap slow feeder; a demand change, so the design goes to her. Design written 2026-10-26, waits on Anita; notes/T-1445.md
- T-1446 [P3] Share a game by link, read only (Anita: yes). Open.
- T-1447 [P3] Copy and paste blueprints between saves (Anita: yes). Open.
Anita's TODO.md "short term", filed 2026-10-26 (afternoon).
- T-1270 [P3] Terrain, stage 4 (stages 1-3 built, ARCHIVE): optional, not recommended first, a 120 km/h cap on mountain-class track at -1 and up. Open; notes/T-1270.md.
- T-993 [P3] noritetsu's account delete question (`noritetsu/dist/account.js`, "Delete your account? Local saves will stay.") should say trainsworld's leaderboard entries go too, since the account is shared (T-972); its privacy page could mention trainsworld. Open.
- T-1791 [P3] Line groups, stage 4 only (stages 1-3 landed, ARCHIVE): boardings in the group view; left out of stage 2 as a later stage if wanted. Open; notes/T-1791.md.

## Track and trains
- T-1966 [P2] Trains wait for each other on screen at flat crossings and crossing junction moves (Anita 2026-11-02: "ideally we have trains wait for each other and not collide at flat junctions"). T-1965's guard has it written (`TrainGuard.crossings`, off): on the real save's 340 flat crossings it held 100-150 trains at a time, up to its 20-minute cap, at 960x. Needs each crossing's real occupancy (a train's time over it, not its body between 16 s frames) or a cap per crossing, and junction moves from the node's conflicts. Open; notes/T-1965.md "Flat crossings".
- T-1967 [P3] Opposing trains on single track are drawn through each other between passing points (T-1965's guard covers one direction of a track only). Open.
- T-1845 [P3] T-925's run 1 gate check (pot, 00:40-02:09 2026-10-30, before 4f316064) waited 90 min for a test line's removal answer that never came; on master the same removal answers in 8 ms (`checks/t1407-last-edit.mjs`) and city-browser.mjs now skips it. Cause unknown; look only if a gate hangs again. notes/T-925.md.
- T-1794 [P3] T-1787 leftovers: on shared-track New York a headway edit on its busiest line still takes 150-400 ms (two or three settles of ~15 rounds), and Paris placements p95 237 ms (about 9 holds a run and level, 2-3 ms a profile); options in notes/T-1779.md "T-1787" (a held line's fill want from its fleet alone, a cheaper crawl bisection in run_with). Also (T-1407): a train count +2 on the real save's busiest line settles 63-128 ms in WASM and re-times ~90 of 109 lines a little (their boarding accounts and phase tables all redone), and a ghost asked while it runs waits that long; settling only what moves past what demand or the drawing can see would shrink the edit and its follow-up together. Open; notes/T-1407.md "Left".
- T-1816 [P3] Demand's `transfers` question names the zone in view, so a player who pans to a zone outside the solve and then edits probably gets no transfer pulses until they look back (found in T-1814, untested): ask by the edited network's zones instead. Open; notes/T-1810.md "T-1814".
- T-1842 [P3] `TrackApi.past_day` (the panels' past 24 hours) takes ~40 ms on shared New York, on every account change and 4 times a second while the clock runs: keep each line's tally in `Boarding` and redo only the lines whose account changed (found in T-1815). Open; notes/T-1815.md.
- T-1808 [P3] `pipeline/calib.py run nyc` fails on its `--areas` file: data/work/realnet/cell_county.u8 has 61,427 values against the pack's 54,218 cells (found in T-1801); rebuild it from the current New York pack. Open.
- T-1793 [P3] The drawn platform crowd overstates riders where transfer pulses (T-554) are large: at Jamaica (line 40's terminal, 08:34) pulses plus the steady rate make about 9,750 an hour against 7,297 arrivals, so the platform shows about 1,958 while the train leaves with 1,501. Found in T-1785. Open; artifacts/trainworld/t1785/.
- T-1778 [P3] T-1777's leftovers: the tools' snap under a still pointer is kept in local units, so after a draw-anchor move the ring and ghost sit in the wrong place until the pointer moves; `map.project` placers (bus drawing, touch bar) are not hidden on the globe's far side. Open; notes/T-1777.md "Left".
- T-1776 [P3] Train sounds when zoomed in on trains (Anita's TODO): very light pass-by and acceleration sounds, slightly varied like the track placement sounds, a soft "sheuuu" as one passes, pitch rising a little as it accelerates, shaped by the game's playback speed. Open.
- T-1773 [P3] Shared real networks on their real levels: each piece's level from the graph's tunnel and bridge flags (-1/-2 in tunnel, +1 on a viaduct or over water), ramps of 200 m a level where there is room, else the neighbour's level; today every piece is on level -1 with every edge `sep` (T-1305 stage 3 left it out). Check New York first. Open; notes/T-1305.md.
- T-658 [P3] Double track drawn as its two tracks when zoomed far in (Anita 2026-10-19: "somewhat important"; all four stages built 2026-10-19, notes/T-658.md "as built"). Left: a node drag's preview of single track still shows two tracks close in (`map/nodeDrag.ts`); then archive.
- T-691 [P3] A station tried on edges 997 and 1130 of the real save (an overlapping pair, 688/1075) is refused with a track-drawing reason ("The curve is too tight... would run on top of existing track"): find where the worker's station trial produces it and give the station's own reason. Found in T-671. T-1964's sweep of the real save (every 100 m, 44,978 spots): 2,050 refused for `CrossingOverlap` and 1,374 for `Geom:RadiusTooSmall` alone, about 1,200 more mixed with other reasons (notes/T-1964.md). Open.
- T-1072 [P3] Changing a line's stops at an end (a stop added past the last, the first removed) moves the other direction's running trains by the length added or removed: flights are kept as offsets along the run; match them by place on the ground instead. Open; notes/T-1071.md "Limits".
- T-1073 [P3] Undo or redo of a train count takes effect at once, without T-1071's lead, so at a fast clock its departures hop a little; tell the worker the step was a service edit. Open.

## Code health
- T-1114 [P3] A general code cleanup (Anita 2026-10-24), the rest: reviews in notes/T-1114/ (ui.md, app-core.md, sim.md, pipeline.md), most batches landed 2026-10-24. Left: app-core 2, 3, 4, 9 (perf switches, approved), 11; sim G's kernel.rs split and H; pipeline F items 19-20, G (the nocar layer, changes pack bytes, with T-878 and T-926), I. Pipeline G waits on a rebuild: the next data batch.
- T-1183 [P3] After T-1146: take the fixed-rate line path out of the engine (`capacity.rs` `continuous` rounding, `LineData.trains` as an Option, the test-only `Op::Schedule`, `set_line` and `set_schedule` in `wasm.rs`), which about 30 sim tests pin exact trains per hour on, so they need rewriting onto train counts; (the shipped `data/packs/real/*.save` were regenerated by T-1751, 2026-10-28; only the engine part is left).
- T-1304 [P3] `complexNow`'s train list (T-1096) is still worked out but nothing draws it since T-1289 drew the complex with the map's renderer: trim it. Open.

## Waits on a rebuild
Tasks that land only when a pack or country is rebuilt, grouped by that rebuild, so whoever runs it knows what rides along (2026-10-26). The two replan groups are also in notes/T-1189.md "What rides along".

### Europe replan (T-1189's Europe part)
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
- T-1933 [P2] Airport jobs on the terminals: a census block's jobs over an airport go to its airport_trips cells, so JFK is one cluster instead of ~30 equal bubbles. Pipeline (build_city.py's block spread). Anita 2026-11-02: at a later rebuild. notes/T-1905.md.
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
Also T-1114's pipeline G (the nocar layer), in Code health.

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

### Interface
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
- T-1936 [P3] Run profiles with many holds in one hop are slow: `profile::run_with` works the hop out again for each hold (2 `hop_time` calls each, every one over the holds before it), so T-1924's following waits (4 at most a stretch for this reason) cost the shared New York save ~10% on edits. A single pass over a hop's holds would let the waits sit at every station the slower trains stop at. track-and-service.md 6.2.
- T-1937 [P3] Departure spacing (T-1925) leaves flat crossings out: a crossing's other-route users carry no place on their run (`at` 0), so lines meeting only at a flat crossing are not spread. Give them their crossing offset, then count them in `spacing::place`.
- T-1927 [P3] Passing tracks at stations that reset T-1924's following loss (T-334 option D). Anita 2026-11-02: not yet, "a reach goal to investigate at some point"; passing only at set points may complicate things. notes/T-1900.md.
- T-340 [P3] Quad track's rest: end-node stations, express pair side. notes/someday.md
- T-702 [P3] Station ghost leftovers on overlapping imported lines. notes/someday.md
- T-755 [P3] No room for ramps at Lake Hamana on the traced Tokaido Shinkansen. notes/someday.md
- T-763 [P3] Penn Station-Emeryville 3-3.6 s from rest to preview. notes/someday.md
- T-914 [P3] A snapped route Schiphol - Amsterdam Centraal refused over water. notes/someday.md
- T-430 [P3] A straight-then-arc fallback for too-tight clicks, only if wanted.
