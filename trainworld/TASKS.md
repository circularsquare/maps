# Tasks

Open agent work, one line a task; detail in `notes/T-###.md`. Format and rules in `CLAUDE.md`.
P1 = now or blocking, P2 = this milestone, P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.
Lower-priority tasks are under **Someday** at the end, one short line each. They are open tasks
(Anita 2026-10-26): count them when asked what is open, and work on them when nothing above can go
ahead; pull one back up when it becomes next. Anita's shortlist of candidates: `notes/for-anita.md`. Notes of archived
tasks (except those an open task or the handoff cites) live in `notes/archive/` by task id, in the folders `t0001-0499/`,
`t0500-0999/`, `t1000-1499/`, `t1500-1999/` and `t2000-2499/` (older ARCHIVE.md and DECISIONS.md links say
`notes/T-###.md`: look there).

**Orchestrator handoff: start with `notes/handoff.md`** (live; older ones in notes/archive/).

## World and countries
Sources: docs/demand-worldwide.md, docs/data-register.md. The unattended city pipeline: notes/T-535.md "Stages".
- T-2001 [P3] Other geoBoundaries files that dissolve same-named units into one far-flung feature, as Malaysia's ADM3 (T-1738's `world_plan.GB_SPLIT_FAR`): Portugal's ADM3, Indonesia's ADM2 (check before Indonesia), Sierra Leone's ADM3. List: `gbscan.py` in t1738b's kit, notes/T-1738.md 11. Open. notes/T-2001.md.
- T-2578 [P3] The Mexico-US border takes the national default factor 0.05, though Tijuana-San Diego and Juarez-El Paso are very busy crossings: a pair factor in borders.json (as MY-SG's 0.5), checked against the joint solve's flows and published crossing counts. notes/T-2436.md. Open.
- T-2003 [P3] The joint gravity is symmetric, so a one-way border flow comes out both ways: Johor Bahru to Singapore 169,000 commutes a day at the MY-SG factor 0.5 (news estimates 250,000-400,000) but Singapore to Johor Bahru 160,000 as well (few in fact). A direction factor per border pair (borders.json), or leave it: a player sees the volume, rarely the direction. Open. notes/T-1738.md 11.
- T-2054 [P3] The join makes no terrain grids and no routed drives for new zones, so 250 zones (Cambodia to Indonesia) priced track on flat ground and 542 (Vietnam on) took the drive rule until T-1738's Philippines run built them by hand (playbook 8, `checks/build_inputs.py`). Open. notes/T-1738.md 13. notes/T-2054.md.
- T-2106 [P3] Indonesia's and the Philippines' other zones on census totals: the zone rules name `admin_totals` since T-2038, but only the rebuilt rows carry it. Open. notes/T-1738.md 15. notes/T-2106.md.
- T-2107 [P3] Cambodia's other provinces on the 2019 census: COD-AB KHM (2018) lacks five districts made since and Tbong Khmum's seven are renumbered, so config `KH` covers only the north-west (`only`); fold the new districts into the old polygons by their communes or find newer polygons, then widen `only` and rebuild. Open. notes/T-1738.md 15. notes/T-2107.md.
- T-2002 [P3] A preflight check for GHSL phantom people (T-2000): a plan row with 1 km pixels over about 20,000 people and no GHS-UCDB urban centre, or rural units whose people per built-up m2 stand far above their region's, as a FLAG naming the units. Open.
- T-2099 [P3] A pack's shore clip leaves out GHSL people on stilt villages and coastal pixels whose cells lie outside the outline: pti's cells hold 80-90% of GHSL's people of Jolo, Bongao and Siasi (Pandami 54%), and with census totals (T-2084) that share of the census goes too (pti 1.63M of its units' 1.89M). Open. notes/T-2084.md 2. notes/T-2099.md.
- T-2108 [P3] New Zealand's jobs from Stats NZ's Business Demography employee counts by SA2 (`2024_Business_Demography_employee_count_by_SA2`, ArcGIS Online, CC BY 4.0) in place of the tier 0 proxy; its people are on the 2023 census by SA2 since T-2077. Open. research/next-countries-data.md.
- T-2094 [P3] world_plan's other distances are Mollweide metres, 0.67-1.9x the ground away from longitude 0 (Cook Strait 34 km measures 22, Oamaru-Dunedin 93 measures 176): `WATER_GAP_KM`, `ARCHIPELAGO_KM`, `SCATTER_KM` and the shared-border km in the merge log. `reach_people` and `join_km` are great-circle since T-1738. Open. notes/T-1738.md 14. notes/T-2094.md.
- T-2095 [P3] The first `register --market` of New Zealand's run (2 rows, 03:18 on 2026-11-03, after the Philippines' join) rebuilt all 2,342 places' market rows in 159 s; the next register, 4 minutes later, did 5 in 0.5 s. Open. notes/T-1738.md 14. notes/T-2095.md.
- T-2251 [P3] uji Ujiarpur (0.21M, Bihar) is a 170 km2 remnant south-west of its own town: its origin and gate centre lie inside haj Hajipur's shown outline, so a jump to Ujiarpur town focuses Hajipur and the zone shown as Ujiarpur holds none of it. Open. notes/T-2111.md "Checks". notes/T-2251.md.
- T-2112 [P3] `seams.py` (borders.json's seamless pairs, a line through one continuous city up to factor 1, T-1192) is a manual step no join runs, so every country built since (Southeast Asia, Australia, New Zealand, India) has its pack lines at the plain 0.5, Kerala's coast metros and Delhi's edge zones included. Open. notes/T-2111.md. notes/T-2112.md.
- T-2298 [P3] The gate's Vite serves the main tree's app/, so a landing with a WASM rebuild mid-gate parks every row checked meanwhile ("no page errors", a minified "Nn" thrown 20-odd times a page: 72 of Russia's rows, 2026-11-04). Serve the gate from a frozen copy of app/ (as `TW_SIM_DIR` freezes sim/), or have the runner stop the gate when app/'s WASM changes and rerun the chunk. Open. notes/T-2295.md.
- T-2398 [P3] `world_run.py step <name> --help` runs the step instead of printing help: `step seams --help` ran seams for real and rewrote borders.json's seamless list (put back from HEAD, 2026-11-04). Make `step` pass `--help` through only to scripts that parse it, else print the step's docstring and stop. Open. notes/T-2295.md.
- T-2399 [P3] The gate's browser check broke silently with an app change: from 861369d9 (T-2365, 13:42 on 2026-11-04) the top bar's split hover is a grid and `city-browser.mjs` waited 300 s for "Train " on every zone, parking Jakarta's jak and sra (regex fixed by countries3, 17:10). A smoke check that runs `city-browser.mjs` on one shipped zone would catch the next such change at landing; and a gate whose browser check fails the same way on every row could stop as a run fault (T-775 needs three parks in a row). Open. notes/T-2341.md.
- T-2299 [P3] A preflight check for origins off their town: an origin whose 1 km pixel is far below its zone's densest, or outside every GHS-UCDB centre in a zone that holds one (Konya, Erzurum, Veliky Novgorod, Tobolsk, Lesosibirsk all parked on 0 riders, 2026-11-04); and `source.unit_names` cut at 25 characters. Open. notes/T-2294.md, notes/T-2295.md. notes/T-2299.md.
- T-2326 [P3] The cloud wave's seed should carry `data/work/world_field.*` and a sea-polygon fallback as standard (routes need world_field since T-2213; osmdata.openstreetmap.de did not answer on 2026-11-04, and wave.sh cannot add files once a machine is up): fold replan2's mkseed.py (scratch replan2) into tools/aws for launch.py's wave kind. notes/T-1189.md. Open.
- T-2296 [P3] Plan zones on census totals where an `admin_totals` config exists: Turkey's plan was grouped on GHSL, which holds 0.84M in Adana's four urban ilce against 2.01M registered (Yuregir 56k against 449k), so sey Adana ships as a 2.3M zone over the cap and yum Ceyhan at 0.22M under the floor, and Adana's UCDB centre missed the metro list. Open. notes/T-2294.md. notes/T-2296.md.
- T-2297 [P3] Russia's 2021 census (VPN-2021) by municipal district and okrug on rosstat.gov.ru as an `admin_totals` config for Russia's rows (GHSL alone since T-2295; nothing checks it): find the table and a polygon file whose codes join (OKTMO; geoBoundaries ADM2 has none), `admin_totals.py check RU`, then rebuild rows moving over 10%. Not opened yet. notes/T-2295.md.
- T-2598 [P3] Brazil jobs by municipio from RAIS (the formal-jobs register), the data register's "?" in Brazil's jobs cell (T-2437): check access and whether it beats OSM jobs outside sao. notes/T-2437.md. Open.
- T-2558 [P3] Mexico everyday trips from DENUE's SCIAN codes as an `everyday` hook (errands 46 and 81, leisure 71 and 72), the data register's "P3 task to file" (T-2081). notes/T-2080.md. Open.
- T-2082 [P3] Israel, if Anita picks it as the Middle East country and decides the territory: a population hook by CBS 2022 census statistical area (data.gov.il CSV on CBS's own ArcGIS polygons, `SEMEL_YISHUV` + `STAT_2022`). Open. research/next-countries-data.md.
- T-1994 [P3] A city centre reunited by `world_plan.FUA_WHOLE` sheds outlying pieces that then stay as they are, under the floor if need be (on a China replan: Fuding 0.47M and Zhenping 0.82M against China's 1M floor). Let a shed piece regroup by the merge rule without rejoining the city zone it left. Only matters on a replan or a new country with such a city. From T-1984. notes/T-1984.md.
- T-1995 [P3] A city zone under the FUA_WHOLE line is named after its most populous unit, a suburb at times (Newcastle's is "Lake Macquarie"; Australia's display names came by hand in zone-names.json). Name a zone holding a GHSL urban centre of CITY_POP after the biggest such centre, as FUA_WHOLE zones are. From T-1984. notes/T-1984.md.
- T-2070 [P2] Confirm the read ahead (T-1980) on a real multi-region run (India, an Australia rerun; Brazil's rows chained its five extracts into one region, T-2437): region k+1's download line while region k builds, no two `read` steps at once; what to read is in notes/T-2040.md "What only a real run shows". The chunked gate (T-1978) is confirmed on Mexico's run, 2026-11-05 (notes/T-2436.md "T-2070"). Open. notes/T-2070.md.
- T-1981 [P2] Measure the joint solve's thread scaling (CPU poll during the next country's join, and one unit at 6 and 12 threads): solves are 17-30 min a country, 2.4 h for China. Then decide night joins at 12-15 cores, or (Anita's call, it changes results slightly) a rerun's unit starting from its last factors. From T-1974, waits on Anita's go. Open. notes/T-1981.md.
- T-1988 [P3] A market pack's `air_direct` (the places its airports fly to directly) is as of its last write, so it misses zones registered since: 79 of 151 sampled packs differ from a fresh build in that field only (Vietnam's, Australia's, Europe's). From T-1976. notes/T-1976.md. Open. notes/T-1988.md.
- T-1992 [P3] A cold `zone_outlines` run (empty cache) takes 4.4-5.1 min at 4 jobs where T-1977 aimed at 3; 860-960 CPU-s of it is zone_water's 368 windows (owner_cells 37%, the reach buffers 21%, owners' points 8%). Open; notes/T-1977.md. notes/T-1992.md.
- T-1738 [P2] Southeast Asia, what is left: Myanmar (Yangon, Mandalay, ADM3; OSM thin), Laos and Timor-Leste (no plan rows or unit source yet). Shipped 2026-10-28 to 11-03: Vietnam, Thailand, Singapore, Cambodia, Malaysia, Brunei, Indonesia and the Philippines (516 rows). docs/adding-a-country.md; notes/T-1738.md 5.
- T-1765 [P3] Zones whose origin was set (before T-1762) on the densest pixel anywhere in the zone, over 25 km outside the unit they lead with; Canada's five, Kerkyra, Siedlecki, Starogardzki and China's eleven city zones were rebuilt and shipped 2026-10-29 (2b8cdda6, b50802f3). Left: China's other 61 (county towns: Guyang 235 km, Nehe 145, Huzhu 129, Lancang 123). Open. notes/T-1765.md.
- T-1375 [P3] The zone ring cut halves a ring by people round the centre it wraps, so a thin arm (Whistler and Pemberton, joined to the Fraser Valley zone round Vancouver's row) splits the ring's real body instead (Abbotsford 0.17M and Chilliwack-Mission 0.19M, T-1239): try shedding the arm that alone makes the wrap before halving; a Canada replan after. Open; notes/T-1239.md section 1.
- T-1376 [P3] Les Îles-de-la-Madeleine (lel, 12,000 people, an island zone of Canada) parks in its build: `enrich_parks` finds under 85% of its park draw on a cell (beach and dune parks off the populated cells). Allow a small island zone through (as T-897's one-park islets) or leave it dead space. Open; notes/T-1239.md.
- T-2300 [P3] Taiwan's tig ("Taitung", 1.29M, T-1189's replan) is the Central Range with both foothills: Hualien and Taitung's coast, Puli, Douliu, Pingtung's plain and Kaohsiung's hill districts, joined by shape across regions because the mountain townships fill the hull (compact 0.69-0.82). Open. notes/T-1189.md "Taiwan". notes/T-2300.md.
- T-1193 [P2] T-1178 stage 2, what is left: rerun the solve units holding a pack in T-1192's seamless list outside mainland China (GB's Bristol group, Italy's Milan-Como, Puerto Rico; check whether Taiwan's ran with China's joins in T-1184 and T-1185), T-1261 is done; rerunning `pipeline/seams.py` first if ids changed; check Guangzhou-Foshan's commuters across near 1.3-1.5M. notes/T-1178.md.
- T-1334 [P3] Other coastal districts dropped as Kwai Tsing was (T-1144)? List each built outline's OSM relations at its picked admin level with 0.25-0.5 of their land inside a plan unit and not picked (a coarse unit polygon missing reclaimed land or islands, the leftover under the 10% fill); rebuild any populated ones. Open.
- T-1080 [P2] A world run at home can fail extract reads on Windows' commit limit (74.5 GB here): each parallel GDAL reader in `osm_features` commits a 6 GB node index, so with a join, a big profile and browsers up, Wisconsin's and Nebraska's reads failed "Failed to open dataset" with 31 GB of RAM free (T-1078). Open; notes/T-1004.md "Wave 2". notes/T-1080.md.
- T-930 [P2] The join's re-gate re-checks only what a joint solve can change (mean commute, top zone share, joint borders, other packs unchanged) instead of every figure of every member: the GB join for hih (2026-10-22) spent 9.8 of its 31 minutes there. Open; notes/T-928.md section 4.
- T-1309 [P3] 384 new Chinese zones show the planner's county name: a look at their names (docs/zones.md "Names"). Open.
- T-428 [P3] `data/packs/intercity.json` (14 MB) is only the joint gravity's input and output, never read by the game since T-377: move it to data/work, out of what is published. Open; notes/T-428.md.
- T-1635 [P3] On the live page a pack's whole-file fetch right after its Range reads (sed.bin) can fail once with `net::ERR_CACHE_OPERATION_NOT_SUPPORTED` (immutable Cache-Control); the client's retry gets it 0.9 s later, but it logs a console error and smoke counts it. Fetch whole files with `cache: "no-store"` or skip the Range reads when the whole file follows. Open. notes/T-1635.md.

## Demand
- T-2093 [P3] The rest of T-2088's ridership check. Part 1 done 2026-11-03: Osaka, Nagoya, Fukuoka, Sapporo and Hiroshima from the jpn7 save (`t2088-ridership.mjs jpn7`, S12 circles matched by station position; all 0.35-0.52 of S12, jpn7 itself about 0.72-0.75 of the dedicated saves; notes/T-2088.md). Open. notes/T-2093.md.
- T-1753 [P3] Rapid and express patterns for Tokyo's long suburban lines in the real save (about +2%); needs an open timetable source (ODPT needs an account). notes/T-1750.md. Open.
- T-1503 [P3] Superholiday rush days on networks past about 8,000 stations take the longer trips unleaned (the lean's S x S table over 256 MB, `LEAN_OD_BYTES`, T-1373): a sparse or per-destination table if a network that big plays. notes/T-264.md Since T-1545 the table is gone (the lean is a column per station in the pass's window), so lifting the gate costs only time: a walk a destination a pass.
- T-1599 [P3] T-1508's option 2, deferred (T-1603 is done): look at a built Beijing - Shanghai high-speed line at player scale (riders a day, does it feel too empty to a player?). Waits on Anita. notes/T-1508.md. Open. notes/T-1599.md.
- T-1233 [P2] T-1177 leftovers: run `checks/t1177-cross-everyday.mjs` with `IDS=gua,fos` once Guangdong's market packs carry the pairs (compare each purpose's local legs plus the pairs with the old totals); the explorer's Everyday trips block shows only while a network reaches the market. Open; notes/T-1177.md. notes/T-1233.md.

## Performance
Ranked in notes/perf-survey.md and notes/T-1156.md (the all-China save) and notes/T-1200.md (slow CPUs).
- T-2588 [P3] On Anita's North America save the demand cache's read ahead of the network misses ("none") on every reload: its `saveNetKey` differs each load (ca5f231b, b63a5b86, cd4fb305) while Tokyo's stays the same, so the cache shows only at the network. T-1539's area. Found by tokyocache (T-2579). notes/T-2579.md. Open.
- T-1730 [P3] T-1609 follow-up: a browser check that an edit right after a local refine solves the cluster whole (the refined periods are marked `~local`), and that a zone going coarse is refined locally too; neither was run. notes/T-1609.md. Open.
- T-1610 [P2] T-1206 leftovers: a clicked station's or bubble's far ends and purposes in a zone still rough wait for its refine (the 8 px ring) instead of reading coarse samples; the fine set's budget corrected by the workers' reported WASM after each solve (now T-1197's rate alone). notes/T-1223.md. Open.
- T-1629 [P2] T-1206 stage 5, split from T-1611 (which did the memory alone): the coarse zones' bias measured with the fine set moving, on the Paris and Shanghai save. notes/T-1206.md, notes/T-1223.md. Open.
- T-1633 [P3] T-1605's small kind: about 1 all-China cold load in 26 moves one line's commute figures by 0.0016 riders a day (line 118, coarse zones in western Liaoning), invisible. Pin it from a load that records opens; leads: what a reopened coarse zone shares with its earlier copy or the thread (`STATION_WALKS`, skims cache), `RandomState` keys advancing with history. Open; notes/T-1605.md.
- T-1634 [P3] Every all-China load closes about 40 reached zones at ~13 s and reopens them a second later in every worker (`sweep` running before the loaded network's reach is complete): wasted downloads and opening, and a load history that varies. Keep reached zones open while the reach is being worked out. Open; notes/T-1605.md.
- T-1560 [P3] T-1169 stage 5, design only: city zones dealt between solving workers instead of periods (each worker its share of the zones for all five periods, loads added on the page each round as T-1543 adds parts), the route to the whole world. Open. notes/T-1560.md.
- T-1354 [P2] The market's corridor figures depend on which market packs were open at a worker's first composition (T-1204): on the real save two workers' tables were ten rows apart, so a page's longer trips depend on the order packs arrived. Make them depend on the world's rows alone, or recompute when a pack adds a near pair. Open; notes/T-1174.md. notes/T-1354.md.
- T-1366 [P3] The demand pool's memory cap (T-1205) never adds a solving worker back in a page: a big game's zones opened for a while (or a big game then New York from the menu) leave it at 1-2 workers, solves slower. Grow it back when the projected memory has room for one more, with a margin against flapping (a spawn, reopen and re-dealt periods, as `shrink` in reverse). Open; notes/T-1205.md.
- T-1367 [P3] After zones or a market close, a demand worker's heap still holds them (frees deferred while busy, the old market parts the kept clusters reference) until its next solve, so T-1205's restart waits for one (all-China: heap at rest 2,041 MB after three zones closed, 1,502 at load with the same items open). Open; notes/T-1205.md. notes/T-1367.md.
- T-1505 [P3] After T-1380, a station edit that makes a new name label still spends 100-140 ms at 6x in `StationLabels.place` (likely the layout forced by reading the new label's width, unproven): measure the width without forcing a layout, or place only the names near the edit. Open; notes/T-1029.md "What is left".
- T-1509 [P3] A line whose length or drawing order changes (stops dropped, a line made, undo of either) still posts the line strokes whole to the page (about 35 MB on Japan, 4 of 12 edits of the T-1381 series), since the segments after it move; keep the lines' order stable or post the moved runs (map/strokePost.ts). Open; notes/T-1029.md "What is left".
- T-1211 [P3] The page's DOM makes every layout cost: 2,200 elements (875 station names on the map), a forced layout 22-37 ms at 1x; the Lines tab opens in 360-400 ms at 6x, the saves panel 534 ms (`showModal`), the Stations tab still 0.6-0.7 s at 4x with 80 rows (`Chips` measuring). Open; notes/T-1200.md. notes/T-1211.md.
- T-921 [P3] Performance: commute zone pairs under 1 person a day dealt in chunks (T-874's deal with the chunk in people a day, `c = 1 x period share`, so every period deals the same pairs): New York solve 34.3 to 29.6 s (-14%), stations under 0.8%, ceilings unchanged. Measured in notes/T-909.md option 1. Open.
- T-2136 [P3] The page's world corridor table: 5.9M rows on the real save, 118 MB since T-2485 made its columns narrower (16-bit places, commute legs only where any) and had one worker send it; it grows with the square of the world's places. Left: keep no full table on the page (per-place totals, and a place's rows asked of the market's worker when the city view, inspector, or achievements need them). notes/T-2122.md, notes/T-2136.md, notes/T-2485.md. Open.
- T-2496 [P3] A solving demand worker's WASM on the real save is 367 MB, heap 233 at rest and 314 at a solve's peak, and two workers hold the same packs: account for it per structure in Rust (a native test on the real save's packs: the zones' layers, the composed market, the 34 MB corridor figures, each period's state), then trim what a solve keeps between periods. Rust, tell maps-17 first. Estimate 50-150 MB a worker if there is slack. notes/T-2485.md. Open.

## Interface
Localization (Anita 2026-11-06: Simplified and Traditional Chinese and Japanese first, Spanish and German second; no human reviewer, Claude translates and reviews for meaning; plan in research/localization.md):
- T-2624 [P2] Traffic on the zone in view's lines after a trains-only edit held behind a whole solve: T-2605's aside worker shows the zone's figures and bubbles in about 3 s, but lines already shown keep their figures until the whole final (about 40-120 s on Anita's North America save), since the zone's riders alone read about 25% low (Salt Lake City 50,400 against 67,800). Recommended: the aside worker also solves the shown network, and those lines show shown riders plus the zone's change (others' riders taken as unchanged); the other way, solving the other zones the line reaches too, still misses the market. notes/T-2605.md.
- T-2606 [P3] Cut a connected network where few trips cross, for North America-like saves (Anita 2026-11-06: "in my america save, say, we mostly have people on the west coast and someon the east coast, and the number of trips that cross between is like pretty small. could we maybe do a cut and solve east side and west side and crossing separately? i think we investigated this before but dropped it"). T-1650 dropped geographic cuts for Japan (no weak cut: 26-39% of market trips cross even at 3-6 parts) and found China's cuts clean but its market cheap; North America is the case it didn't measure, and there the market is ~60% of a round (T-2553). Measure first on her trainsworld-north-america-day-383.save (clusters, crossing shares at cuts, what each part's solve would cost), then propose; design before build. Measured and designed 2026-11-06 (cutnet, notes/T-2606.md): one line (LA to Chicago, 0.7 trains an hour) is all that joins east and west; 3.6% of market rail trips cross but 28-42% of its rail-km, all on that line, which separate parts would show at about half its market riders; the parts' passes add back to the whole's (16.8 against 17.1 s), so no whole-solve gain. Recommends no cut (T-1733's freeze plus T-2605); awaiting Anita's read, then archive.
- T-2610 [P2] Network diagram: since T-2599 the Hudson between Exchange Place and lower Manhattan is drawn as land (the sheet fix alone kept it open; the East River at the tip is drawn shut before and after). Station pairs across the river are as far apart or farther, so the carry or the map's land cells decide it. Picture: artifacts/trainworld/smoothlines/nyc-tip-3way.png. notes/T-2599.md. Open.
- T-2619 [P2] Upkeep raised before release (Anita 2026-11-06, after T-2618's research/upkeep-france.md: "i think definitely raise undergrund station maintenance to 200k, that ones quite low. i think we can do ground maintenance to 35k, viaduct to 50k, cut and cover to 100k, bored to 130k."; "we don't have an opportunity to raise maintenance costs much once we release"): per game day, ground 35k a km, viaduct 50k, cut and cover 100k, bored 130k, underground station 200k (new), other stations 50k. Open; taking, 2026-11-06, agent upkeep.
- T-2620 [P2] Keep long lines profitable for a player who tries hard (Anita 2026-11-06: "would we maybe be able to just raise like passengers willingness to pay to compensate? idk. i feel like i do want people to have fun playing this game and part of that is to be able to have long distance lines feasibly be profitable if you try really hard. and players' trains are probably significantly faster than irl so this is not like strictly unrealistic."): measure the T-2619 upkeep's effect on her saves' line profits, and propose a fare-weight (willingness to pay) change with numbers; demand changes go to Anita before they land. Open; taking, 2026-11-06, agent upkeep.
- T-2621 [P2] The regional market (long trips) on a coarser graph (Anita 2026-11-06: "also regarding cut: is there really no way to do this? i guess like im thinking, a long distance line doesn't really care much about the exact local shape of ridership. as long as we get a rough sense of like quantiatively how accessible is each station, shouldnt we kinda be able to compute long distance trips more quickly? idk. i feel like zooming in we just have so many tiny bubbles that surely theres more we can do here?"): measure where the market's time goes, then design and prototype long-trip routing between grouped stations or access summaries; figures within a few % of today at a fraction of the cost. notes/T-2606.md, notes/T-1650.md, notes/T-1647.md. Open; taking, 2026-11-06, agent coarsemkt.
- T-2622 [P2] The 4x clock cap must not depend on the city in view (Anita 2026-11-06: "also it seems like if demand calcultion is 1 day behind i think the 4x time cap might be applying differently depending on what city we have selected. which is weird. we should make that not happen, if its happenign"): today the cap holds while "the zone in view shown rough" among others (docs/ui-panels.md ~270, game/lineFigures.ts); find what differs by view, make the rule the same wherever the camera is. Open; taking, 2026-11-06, agent capview.
- T-2625 [P3] Network diagram at continent scale, 2D condensation: after T-2609's cut bands, land in one network's column and another's row (the Gulf in Florida's rows, Ontario in Toronto's column) stays at the floor, so networks are still small spots; needs a looser fold guard and a solver that settles under it. notes/T-2609.md. Open.
- T-2626 [P2] Quick profit estimate in the edit note, T-2604 stage 1 (Anita 2026-11-06: "ok we can show profit estimate but say like "estimated -$5.3M profit" and can show on + and - buttons if easy but no strong need for that tbh"): notes/T-2604.md's page-only estimate (running cost exact, fares by the wait term). Open; taking, 2026-11-06, agent profitest.
- T-2627 [P3] The join's market step runs under world_run's 1800 s limit, but its local trips over all 3,545 zones took 26 min on a busy machine (T-2595, notes/T-2595.md; Brazil's whole step 1058 s): raise or drop the limit for that step, or speed up the local trips, before a bigger join hits it. Open.
- T-2628 [P2] No terrain shading under water (Anita 2026-11-06: "occaisonally the terrain for a place loads before the water and we see the terrain under the water. woudl we be able to just not show terrain under water ever? i think it might be nice later for like determining how much to penalize water crossing cost but for display purposes we should just consider it as flat"). Open; taking, 2026-11-06, agent hillwater.
- T-2584 [P2] Localization stage 1: every player-facing string into en.json (about 2,000), pieced sentences and plurals rewritten as whole messages, the language passed to number and date formats; `?lang=pseudo` shows any missed string and `checks/i18n.mjs` counts what is left per folder (docs/localization.md); split by folder, best with little other UI work. Open.
- T-2585 [P2] Localization stage 2: Japanese, Simplified and Traditional Chinese through translation plus independent review passes (back-translation for meaning, naturalness, mechanical checks, screenshots), fonts per language, a Language setting, layout pass. Anita's TODO.md, 2026-11-06: "for language selector we can put it in settings, default to whatever users browser language is, and on hover disclose that non-english text was ai translated." After T-2584. Open.
- T-2586 [P3] Localization stage 3: place names in the player's language where data has them and it is cheap (Anita 2026-11-06: "placenames shown in players language if we have them, and if its cheap"): basemap name:ja/zh labels, the place-list aliases Japan/China/Taiwan/Hong Kong already keep, zone names via Wikidata; no world rebuild of place lists. Traditional Chinese is Taiwan's (zh-TW). Anita's TODO.md, 2026-11-06: "also we should try to get placenames translated but that is also lower prio, and we might not want to do it if it costs a lot of resources". After T-2585. Open.
- T-2587 [P3] Localization stage 4: Spanish and German. After T-2585. Open.
Anita's notes 2026-11-05 evening (verbatim in notes/todo-2026-11-05c.md; briefs quote it word for word):
Anita's notes 2026-11-04 ~09:30 (verbatim in notes/todo-2026-11-04b.md; briefs quote it word for word):
Anita's notes 2026-11-04 evening (verbatim in notes/todo-2026-11-04i.md; briefs quote it word for word):
Anita's notes 2026-11-05 late morning (verbatim in notes/todo-2026-11-05b.md; briefs quote it word for word):
- T-2589 [P3] Network diagram sheet speed. Stage 1 done 2026-11-06 (sheetspeed): the same arithmetic faster in JS, in the page New York 10.9 s to 9.3 s, all Japan 49 s to 38 s. Stage 2 done 2026-11-06 (sheetwasm): the carry's solve in WASM (`sim/src/diagram.rs`), bit for bit the JS, about twice as fast (all Japan's carry in the page 20.5 to 11.2 s, New York's 3.9 to 2.2 s; `?perfOff=carryWasm` the JS). Left, aim (all Japan well under 15 s, New York under 4 s) not met: the layout on a country map (about 13 s of all Japan's 41; its sweeps never meet their tolerance, so any speedup changes the drawing a little; waits on the smoothing work in layout.ts), the map (T-2592), the sheet and ground (about 5 s). notes/T-2589.md. Open.
- T-2590 [P3] Delete the old network diagram layout now the sheet is the default (`?sheet=0`: layout A and the blocks stage, `borders.ts` pill regions, `water.ts` and `waterSea.ts` drawing) and the checks built on it (diagram-game.mjs, t2169-game.mjs run on sheet=0; t2160/t2161), once the sheet has been played a while. Takes in T-2582 (blocks take in intercity branches). notes/T-2580.md. Open.
- T-2593 [P3] Network diagram sheet at the edge of the built world: where no zone has a mask at all, `waterReal.ts`'s far grid still guesses toward sea, so land past the built zones may draw as sea. Found by njwater (T-2591). notes/T-2591.md. Open.
- T-2597 [P3] On the sheet, a "zones" message to the diagram worker recomputes the borders through `sheetMap` but posts the water from the first map again (`app/src/workers/diagram.worker.ts` ~line 103), so if zone outlines arrive after the layout, water and borders may come from different outline sets; check whether it ever shows, fix if so. Found by sheetspeed (T-2589). Open.
- T-2592 [P3] The sheet's map step: 2.4 s on New York and 5.1 s on all Japan in the page after T-2589 stage 1 (was 2.9 and 5.5). Limiting it to the area near the network saves little without changing the whole view (New York's drawn grid is barely bigger than its stations' box; Japan's islands fill theirs): what is left is the thin passes' distance transforms and the grid's triangles over 4 or 12 million cells. notes/T-2589.md. Open.
- T-2538 [P3] Network diagram: the routing and separation still lose crossings B kept (after T-2534, all Japan outside dense places 0.76 after B, 0.49 drawn): stations slide along a bent piece's legs past the crossing (most stay kept loosely, a station early or late). notes/T-2534.md. Open.
- T-2539 [P3] Network diagram: t2407's rebuild check moves more since T-2534 (median city station beyond its block 0.071/0.112/0.103 -> 0.186/0.155/0.106 hops; the mock's new line crosses city lines and pulls them to it); decide whether the limit or the rebuild's crossing weights change. notes/T-2534.md. Open.
- T-2546 [P3] A smooth recorded clip for launch (Anita: "just screen recording feels a bit unnprofessional cuz the zooming/panning is kinda not smooth"): a scripted camera path rendered frame by frame to video (ffmpeg is installed), shots picked with her. On hold (Anita, 2026-11-05: "lets do it later once stuff is more finalized"). Open.
- T-2551 [P3] Network diagram: New York's core river bands wind where the layout scatters a river's crossings (from T-2517, not done in T-2541); also t2405 finds hidden stretches on the LIRR by Jamaica (Jamaica - Rosedale and St. Albans) on one New York graph or the other as the layout shifts (T-2541: the save 3; T-2550: the save 0, the game capture 6). All Japan's "every drawn river is one band" passes again with T-2550. notes/T-2541.md, notes/T-2550.md. Open.
Anita's notes 2026-11-05 morning (verbatim in notes/todo-2026-11-05a.md; briefs quote it word for word):
- T-2529 [P3] Network diagram: parallel lines lost in the routing (after T-2520, 80-90% of parallel pairs share a direction after B but 62-70% once drawn): pieces off a direction get two legs chosen separately, so parallel pieces jog differently; couple their leg order or snap. notes/T-2520.md. Open.
- T-2530 [P3] Network diagram: B converges slower since T-2520 (New York's layout 2.8 -> 3.6 s, B 1.37 -> 1.99 s); find a cheaper way to the same choices, and look at all Japan's near pairs drawn on the wrong side (387 -> 481). notes/T-2520.md. Open.
Anita's notes 2026-11-04 late (verbatim in notes/todo-2026-11-04j.md):
- T-2472 [P2] Size summary for Anita: R2, her computer, what an average player downloads, the player's RAM, anything else; and what could shrink. Measured, report in notes/T-2472.md; waiting on Anita (disk deletions are hers to approve, list in section 2). done 2026-11-04, sizes.
- T-2486 [P3] Wire size: `.bin` is 76% of R2 (2.2 of 2.9 GB) and the biggest piece of each city's download, and the first line pulls 14.4 MB of worldwide data; compress `.bin` for whole-file reads, keep Range reads working, load the market files by region. notes/T-2472.md section 6.
- T-2458 [P3] A zone selected in the city view during a load is solved coarse by the load's solve and refined only by a second whole solve after its final (Tokyo save: Seoul selected 8 s before the solve was sent, fine figures about 6 s after the final). Pick its level before the load's solve goes. notes/T-2453.md "Left as they are". Open.
Anita's chat 2026-11-04 ~17:10 (verbatim in notes/todo-2026-11-04h.md; briefs quote it word for word):
Anita's notes 2026-11-04 late afternoon (verbatim in notes/todo-2026-11-04g.md; briefs quote it word for word):
Anita's notes 2026-11-04 evening (verbatim in notes/todo-2026-11-04f.md; briefs quote it word for word):
- T-2417 [P2] Network diagram: a one-station ~2 km extension past Coney Island drew stretched to near Philadelphia. Not reproduced in Node (the real save and T-2407's mock, master and blocks: 0.9 to 4.7 hops); a rebuild's new node with no placed neighbour now starts off the nearest placed node, not the middle of all (between cities); needs Anita's game exported to confirm. Open (fix landed 67468b83; Anita to say if it recurs), notes/T-2407.md.
- T-2425 [P3] Network diagram after T-2407, all Japan: single-line stretches +19% (18,399 -> 21,886 pitches; 4 longer than the longest block), pill overlaps 4 -> 9. A's rounds are not the cause (40 rounds: 21,875); the long stretches (Kosei line, Tsuruga-Higashi-Maizuru, Tatsuno) are stations between blocks, so the stage's box spacing and the sprawling block 25 (Iga-Ueno to Hikone) are the next look; B is slower because turn terms beside long hops between blocks stop its solves converging (whole layout still 78 s against 93 without blocks). notes/T-2425.md. Open (diagnames handed back, 2026-11-04).
- T-2422 [P3] Governors Island (T-2418 fix 2): tunnels join land only at their two ends, not drawn onto the water (a `way_flags` bit, `WAYS_RULE` bumped, every extract re-read: ~1h45, 67 GB worldwide); risk to Hong Kong Island, Haneda, Umihotaru (add checks). Rides with the next data batch. notes/T-2418.md. Open.
- T-2426 [P3] Car ferries OSM leaves untagged for cars: since T-2424 a ferry carries cars between places only with `motor_vehicle`/`motorcar` yes, and China's Bohai Strait ferries (Dalian - Yantai, real Ro-Ro) have no such tag, so Shanghai - Dalian and Seoul's pairs with eastern China now drive round by Beijing (rt 0.87 -> 1.60). Count the untagged `route=ferry` lines over 50 km between two places' landmasses and decide a rule (a length floor, `ferry=*` or `hgv` tags) or leave. notes/T-2424.md. Open.
Anita's notes 2026-11-04 midday (verbatim in notes/todo-2026-11-04c.md; briefs quote it word for word):
- T-2404 [P3] Blueprint lag, fix 4 of notes/T-2388.md: hold an early edit's follow-up while the ghost is being asked, take it when the pointer rests, merge consecutive segments. After T-2403 (done: a New York click still freezes the ghost ~190-200 ms p50, the follow-up itself; notes/T-2403.md). Open.
- T-2393 [P3] With a fuller train scheduler: fractional trains, so trains can wait at the end of the line when demand is low; Marey diagrams including other lines on the same track (T-1481). Idea. Open.
- T-2396 [P3] After T-2282: the dock's tab area keeps one height for every tab, so the shorter Build tab gives the inspector no more room (notes/T-2267.md 3.5). Open.
Anita's playtest with a friend, filed 2026-11-03 evening (verbatim, grouped, in notes/todo-2026-11-03h.md; briefs quote it word for word):
- T-2280 [P3] More tutorial-only longer hovers (T-2263's `guideVerbose` in game/ui.ts): candidates the demand bubble hover (map/bubbles.ts), Train length (inspector/LineInspector.tsx) and the top bar's split (TopBar.tsx, game/splitTip.ts), files T-2214's tree held on 2026-11-03. Only if Anita's playtest of the tutorial asks for more. Open.
- T-2264 [P2] Help panels ("?") explaining demand and effective time, drafted from docs/demand-plain.md; wording for Anita. notes/todo-2026-11-03h.md. Open.
- T-2268 [P2] Station placement: more room for a station when auto-ramping over a line (an easier transfer), and snap a station to the nearest viable spot when one is close. notes/todo-2026-11-03h.md. Open.
- T-2336 [P3] Weekend service later: players optionally set weekend schedules apart, or a citywide or game-wide multiple on all headways (Anita 2026-11-04, notes/todo-2026-11-04a.md). Design to Anita first. Open.
- T-2337 [P3] Idea: load by hour of day as a small plot in the crowding hover, if peak load per hour is cheap to keep (Anita 2026-11-04, notes/todo-2026-11-04a.md). Open.
- T-2321 [P3] Network diagram, a follow-up to T-2289: all Japan has 3 more pill overlaps (neighbours on compressed lines, Nishi-Fujinomiya and Numakubo, Izuma and Kamiichiba) and 4 pairs closer than their spacing that it had not; Tokyo-game 2 close pairs. A gap pass along each piece changed nothing (the pairs sit at piece ends). Figures in notes/T-2289.md. Open.
- T-2293 [P3] Network diagram, a follow-up to T-2290: near stations still drawn on the wrong side of each other after the routing and separation (10 to 18 pairs a city), mostly parallel lines a kilometre apart whose stations the routing places by share of a piece; a side term in the separation's linear terms is the likely way. Figures in notes/T-2290.md. Open. notes/T-2293.md.
- T-2332 [P3] After T-2291: a copied line (`copyLine`, Rust-side) keeps its source's colour, and a bus or ferry line takes its colour at the tool's start (`nextFreeColour` at the first click), before its route is drawn, so neither avoids a near-duplicate beside it; same re-pick as `lineColour.ts` if wanted (a copy sharing the original's colour may be intended). Open.
- T-2292 [P3] Network diagram idea, a follow-up to T-2290: lines that run close together and swap sides on the ground drawn as one parallel bundle with a fixed order (Anita's first reading of line 8, since corrected: "not that egregious"). Only if a real save shows it. Open.
- T-2288 [P3] The airport badge's hover (map/airportMarks.ts) leaves "No route" trips out of its count and bar (T-2215 follow-up). Open.
- T-2320 [P3] Drag a train line's stops into another order in its stops list while editing stops (`setStops` with the new order), if Anita wants it: a reordered list on track mostly makes trains double back, so T-2275 left it out (DECISIONS 2026-11-04). Open.
Her notes from testing the 11-02 landings (verbatim in notes/todo-2026-11-02c.md):
- T-2043 [P3] Bring checks/t988-cloud.mjs up to T-2037 (it needs a local shonei-server, not run): section 7 expects a fork slot named "T-007-real-nyc at <time>" that the page moves into; now the playing slot splits off in place (same name, new account id, same origin). Open. notes/T-2043.md.
- T-1916 [P3] Drawing beside track: in the real-save sweep (`beside_sweep_real_save`, sim/src/track/wasm/beside_tests.rs) 32 of ~85,000 previews still cross the track followed, where the concentric line itself crosses the run, and the tool falls back to the plain route; also 46 on-line starts refused as "on top of existing track" beside a parallel line. Open. notes/T-1916.md.
- T-1917 [P3] A ramp spread over several legs (T-1895) can move when two pieces are joined under a new platform (the join widens the room it spreads in); harmless while platforms refuse ramps, but check `unjoin_near` joins keep heights. T-1964's drawn sweep: 58 of 6,851 spots near a level-change joint refused for the ramp a join moves (shown as the platform on a ramp), 1 with a curve reason. Open.
- T-1970 [P3] The station tool slides a station off a ramp it is clicked on near the ramp's end (at most half a platform and 2 m, as T-1870's room slide), from the ramps in the edge's drawn heights; today "The platform would be on a ramp." stands. T-1964's drawn sweep: 1,012 of 6,851 spots refused for a ramp. notes/T-1964.md. Open.
- T-1948 [P3] A node drag's new lead-in toward a loose end takes at most 0.3 of the leg (`lead_in_capped`), so a 1 km straight with its joint dragged 108 m sideways gets a 4,400 m curve (250 km/h) where about 9,000 m fits; let it use the whole leg when the far end is a track end. Open (found in T-1898).
- T-1949 [P3] Node and pick drag previews draw high-speed track as ordinary track (`previewStrokes` in map/nodeDrag.ts passes no `hsr`; the curve bend does). Open (found in T-1898).
- T-2046 [P3] Seoul on Tokyo Seoul Fukuoka Busan: the first Homes/Destinations switch in the seconds after zooming in waits about 6 s, until its detailed solve lands; main thread idle, cause not found (maybe a headless frame stall). notes/T-1904.md "After".
Anita's playtest of 8800 (build 40593370), 2026-10-26 morning: the lighter map "feels nice", Traffic view "looks pretty good", related lines "looks great", adding local games to the account worked.
- T-1856 [P2] Terrain cells as hexagons, our own lattice (Anita 2026-10-30: yes); T-1854 landed first so price and shape changes stayed apart. Design written (notes/T-1881.md, last section; about a day); T-1881's terrain rebuild landed 2026-11-03 (8212d530), so it can go ahead. Open.
- T-1858 [P3] Loading the all-Japan save takes minutes before the first city shows demand; after it the other cities come fast (Anita 2026-10-30: "that's fine then"). Investigation stopped early; its last note: a demand cache exists that a load might show first. Look only if she asks again.
- T-1861 [P3] Track placement on the all-Japan save still feels a little unresponsive (Anita 2026-10-30: expected for its size, but room to improve). After T-1794; measure with t1407-edit-ghost.mjs on a Japan save. T-2388 (2026-11-04) saw every pointer move cost 100-145 ms on all Japan in drawing runs (the snap projecting every node, the hover's pick 35-40 ms) while a direct probe found 0.3-0.8 ms: cause not found (notes/T-2388.md). Open.
- T-1863 [P2] The settle leaves lines with no shared track unchanged to the bit, as track-and-service.md says (T-1850 stage 2, sim/src/track). Done when `netdiff.mjs` shows 0 changed Tokyo lines after the Ceuta edit. After t1854b leaves sim/src/track. notes/T-1850.md. Open.
- T-1867 [P3] Draw a destination's everyday homes as the smooth spread their samples stand for (T-1866 fix 1, drawing only). On hold: T-1874 has landed; look again once Anita has seen it. notes/T-1866.md. Open.
- T-1875 [P3] `t671-ghost-hover` fails "near track ends" on the real save (14 of 84 hovers, 2 chains): the page's guess says a spot fits and the worker refuses it (a flat crossing the page's marks miss; an end station the worker finds too short). Not from T-1870's change (same candidates before). Page/worker agreement on new blueprint near flat crossings. Open.
- T-1876 [P3] Australian jobs from the 2016 census Working Population Profile by destination zone (ABS DataPacks, CC BY 4.0) instead of tier 0's proxy (2021's place of work was taken in lockdown). notes/T-1857.md. Open.
- T-1878 [P3] The zone planner never splits a zone from its own pieces hundreds of km apart (seen in Australia, T-1857). notes/T-1857.md. Open.
- T-1879 [P3] A metro rule for 1M+ cities GHSL's urban centres miss (Adelaide was made one zone by hand, T-1857); CLAUDE.md's join rule already names data/terrain. Open.
- T-1882 [P3] checks/t1270-terrain.mjs: the track key was fixed to "1" on 2026-11-02 (902af16d); whether its quote-rows check still matches the layout since T-1839 is unknown: run it and fix or archive. Open.
- T-2141 [P3] Lines side by side, tight and loose ribbons by zoom: T-2132's hold keeps a ribbon across gaps where tracks are up to 50 m apart at every zoom, so close in a held band can sit up to 25 m off its track; mix a held and an unheld profile by line width on screen if that shows. notes/T-2058.md 3e.
- T-2150 [P3] Achievements stage 3: the Sea crossing first (a running line whose path crosses 10 km of unbroken water, bridge or tunnel). The sim gives each line its longest unbroken water run at any level (T-1930's run counting, bored tunnels included) in `LineStats`; it changes the WASM interface, so its own worktree. Open. notes/T-2150.md.
- T-2242 [P3] Network diagram spacing left from T-2205: Tokyo's 7 close pairs (mostly key nodes: Iidabashi, Tokyo, Nippori/Tabata; the count swings 4-17 with small settings), and rebuilds leave more than a fresh layout (Tokyo 13 against 7) because of the 0.3 hop cap that keeps edits stable. notes/T-2205.md. Open.
- T-2262 [P3] Float32 plane-metre positions lose 1-2 m in x far from the frame's origin on Japan saves (strokes, trains, stations): jitter at street zoom, and the root of T-2222's zero-length segments (now hidden by T-2222's 8 m minimum). Store positions relative to a nearby origin. notes/T-2058.md 3h. Open.
- T-2259 [P3] Network diagram: the separation can still fold a line when its redraw leaves a node another way (Tokyo's 東中野 in 3 of 16 perturbed samples). notes/T-2255.md. Open.
- T-2260 [P3] Network diagram: hooks at four-way hubs where lines end (New York Penn: no axis crosses all 24 stops; Jamaica, Sutphin). Fixing them would let A's pushes move to its last 3 rounds, which ended doubling back in every sample. notes/T-2255.md. Open.
- T-2261 [P3] Network diagram: the game's graph depends on the camera's draw anchor (float32 local coordinates), so another session can get another layout; build it from pack-frame places so it is the same every time. notes/T-2255.md. Open.
- T-2246 [P3] Network diagram: room for names in the layout (reserve label space at busy interchanges, as Nöllenburg and Wolff do), so 47-50 Sts-Rockefeller Ctr and 7 Av get names at 14 px and Times Sq, Penn, Grand Central and Shinjuku at the opening zoom (tied to T-2242). Done when Times Sq's 15 interchanges are all named at 14 px. notes/T-2162.md. Open. notes/T-2246.md.
- T-2247 [P3] Network diagram: name placement takes 0.5-1 s of New York's layout (whose Node timing nears the 5 s limit on a loaded machine); skip levels that change little or carry unchanged names over. notes/T-2162.md. Open.
- T-2510 [P3] Japan's water masks miss the big rivers (found in T-2497): along the hops crossing them the 25 m mask is dry at the Tama (二子玉川-二子新地), the Sumida (浅草橋-両国) and the Yodo (新大阪-大阪, 十三-中津), and the Arakawa at 赤羽 is 175 m wet; so the diagram's water (and the track model's water pricing) lacks them. Likely the OSM river-area multipolygons in pipeline/osm_water.py's read. A data rebuild; wait until no join runs. notes/T-2497.md. Open.
- T-2511 [P3] Diagram sea fallbacks (after T-2497): Jamaica Bay is not sea on the ground (counting islands under 3 km² as water for the core test did not change it), the Boso side of Tokyo Bay comes out partly land, and the sea ends in a straight cut 40 pitches past the network's box, seen when zoomed out. notes/T-2497.md. Open.
- T-2518 [P3] Diagram river drawing time after T-2516 (whole rivers, channels): all Japan about 1.2 -> 4.2 s, New York 0.9-1.4 s in game, in the layout worker after each rebuild; with T-2498. notes/T-2516.md. Open.
- T-2498 [P3] Diagram water time (after T-2169): the ground's water takes 4.9 s for all Japan (0.7 s New York) and is worked out again after every rebuild that moves a station or the network's middle, holding the layout worker; keep it per tile in mercator so an edit redoes only its tiles. notes/T-2169.md. Open.
- T-2148 [P3] Saves keep every day of every region's ledger (T-2143), about 30 bytes compressed a region-day; a world game of 100 regions over 1,000 days nears the account's 4 MB cap. If saves get big, fold a region's days older than 90 into Monday weeks in the save (the chart already shows weeks past 90 days). Open.
Anita's notes on build 827, 2026-11-03 late morning (verbatim in notes/todo-2026-11-03d.md, her shots in artifacts/trainworld/anita-2026-11-03d/):
- T-2209 [P3] Ferry routes in canals at the 10 m limit: Amsterdam's grachten route (3.0 km) is not found at 2.5 m cells (an 11 m canal, a seam between water polygons), so Centraal to Rijksmuseum goes by the Amstel (4.25 km); 1-4% of Venice's and Amsterdam's routes under 5 m. Perhaps a 1.5 m cell for short canal legs, or seams under a cell closed. notes/T-2177.md "T-2192 results". Open.
- T-2188 [P3] A city line's edit ends with its local solve, no whole solve of the cluster (the "Trips in 18 cities" step): the touch rule already re-solves only the line's zone (city game: Tokyo alone, 0.1 s; the whole solve 8 s), and on the real Tokyo save the local result was within 83 riders of the whole (31.7M). Open. notes/T-2188.md.
- T-2189 [P3] The edit note forgets edits not settled in 5 minutes (`GIVE_UP_MS`, game/editNote.ts): Tokyo's real save on one demand worker settled in 352-361 s and showed no note. Raise it, keeping the level-of-detail hold it now bounds (T-2180) short. Open.
- T-2198 [P3] Mark where faster trains lose time behind slower ones (T-1924's following loss, wire kind 11): it can be minutes a train, yet it shows only in the line panel ("Behind X trains"), never on the map or in the trouble list, and drawn it is trains bunched behind slower ones, which reads as a jam with no mark. Open; notes/T-2185.md. notes/T-2198.md.
- T-2199 [P3] Trains drawn far behind their timetable at a full merge: T-2185's test Y at 100% (`checks/t2185-merge.mjs`, 10 trains a line, cut to 12.5 an hour each) drew 9-15 trains over 30 s late, up to 5-10 min, all on one line's approach to the merge, where at 92% the most was 20 s-2 min. Open; notes/T-2185.md. notes/T-2199.md.
- T-2200 [P3] Named crossings (Anita 2026-11-03: "ok we can do named crossings. perhaps have them hidden so as to not clog if u haen't made them"): a hand-made list of 20-30 real straits, base tunnels and great lines (built or proposed), each marked when a running line uses it, hidden until then; after T-2150 (sea crossing). research/late-game-spending.md B. Open.
- T-2201 [P3] Maglev (Anita 2026-11-03: "oh yeah we could do maglev lol. can make it super expensive. would be good for being competitive with planes"): a third track kind, 500 km/h, its own track, a high price. Design first; it only pays if long trips ride rail (T-1504). research/late-game-spending.md C. Open.
Anita 2026-11-03 early afternoon (verbatim in notes/todo-2026-11-03g.md):
Passing tracks (T-2228, archived; built as T-2231 to T-2233; notes/T-2228.md):
- T-2234 [P3] Passing loops between stations (T-2228 option B), with single track's meets (T-1967). Only if Anita wants it after playing passing tracks. notes/T-2228.md 4. Includes passing tracks at single-track stations (the bit does nothing there now, T-2231).
- T-2239 [P3] Passing tracks: one phase for both directions of an overtake pair (T-2231 picks each direction's own best offset, so in one direction the drawn trains may not line up with the holds). Done when both directions' holds and drawn trains agree on the 180 km test line at 10/10. notes/T-2228.md 8. Open.
- T-2240 [P3] The express's last minute: on a timetabled overtake pair, the shared-track queue (58 s) and terminus queue (8 s) still sit on top of the overtake plan (T-2231's 180 km line at 10/10: +1.2 min over free, 8 s of it overtakes). Decide whether pairs the timetable already orders should skip that queue. notes/T-2228.md 8. Open.
- T-2241 [P3] Three-speed lines (local, rapid, express) with passing tracks: T-2231's plans are pairwise and a local's waits for several faster lines are added up. Measure on a three-line test before changing anything. notes/T-2228.md 8. Open.
- T-2216 [P3] Islands I4: market zone pairs on two parts or landmasses solved with rail as the only ground mode (plane where it has one) in place of "fixed, never solved" (1.45M longer trips and 631,000 commutes a day today); market zones take their part; closes T-1396 under "car ferries ignored". Own worktree (intercity); T-2214 is done. Open. notes/T-2216.md.
- T-2256 [P3] Islands I2b (from T-2214): the land-parts rule where local demand still lacks it: a selection's far end by cell, and the regional layer's commutes and outings across zone lines (`intercity.rs`, `cross.rs`). Done when Sado's far-end bubbles and Salamis's zone-line commutes show no car or walk off the island. notes/T-2214.md. Open. notes/T-2256.md.
- T-1864 [P3] Barrier solves (3 workers) keep clean market clusters, which T-1607's split marks all dirty (T-1850 stage 3). Measure how often it happens first. notes/T-1850.md. Open.
Anita's playtest of buses and the round, 2026-10-26 (evening). "buses are cool!"
- T-1481 [P3] Marey diagrams (Anita: nothing built yet makes them interesting, but might be nice; 2026-11-04: with a fuller scheduler, "including other lines that come onto the same track", T-2393). Idea. Open.
Anita's answers on T-1411 and T-1412's ideas, 2026-10-26 (afternoon). Not wanted: riders before you pay ("people can just build it and find out"), growth near good stations (confusing), flights or freight you run.
- T-1519 [P3] Income groups in the local layers (Anita's call): three groups as the market's (T-1431) would run the local mode choice three times, about 2.4x the local solve time on New York (72% of it is the mode choice). notes/T-1431.md "Round 3". Open.
Anita's TODO.md short term, filed 2026-10-27 (morning). Her priorities: memory per zone and worker high; rest of the world after the finer demand calibration.
Anita's TODO.md short term, filed 2026-10-28 (afternoon).
- T-1681 [P3] Doubling up the real save: 122 of 1,093 double-track pieces are refused with PlatformCrossing (another track crossing at the same level where the pair's twin platform goes), found by T-1678's scan. Check whether that is right or the twin station could go elsewhere. Open.
Anita's TODO.md short term, filed 2026-10-27 (night).
- T-1736 [P3] After 40 minutes of edits on all of Japan the last whole solve differs from a fresh solve of its network in one zone (Okayama: 1 line, 13 stations, about 2e-9 of their figures; T-1734's run `jp-steady1`; again 2026-11-04 after its rebase, `artifacts/trainworld/solveend/jp-steady`). Find which state carries over (a level refine's periods, a market kept across networks) and make it equal to the bit. Open. notes/T-1736.md.
- T-1764 [P2] Touch rule, rule 2 left (rule 1 done 2026-10-29, see ARCHIVE): a line with at most 3 stops in a zone's view (one of its own plus pads) no longer touches it on trains an hour or run times. Open. notes/T-1764.md.
- T-1763 [P3] After T-1644, the overlay at zoom 15 on all of Japan still draws every station box (9,529, three passes, about 1.1 ms) and train (about 0.8 ms) whatever is on screen; cull both to the view with a spatial index (about 1.5 ms a frame; also the 18-19% GPU with the clock running and the map still). notes/T-1644.md. Open.
- T-1760 [P3] A new express shows about a quarter of its market riders until the whole solve lands on all of Japan (a 623 km copy: 26% in play, against 72-85% at free flow on T-1650's network; China's HSR express 64%). Open. notes/T-1760.md.
- T-1444 [P3] A game mode with a city's real network run as a rival to win riders from (Anita: a new game mode). After T-1305. Open.
Anita's playtest of buses along roads, 2026-10-26 (night), Shenzhen, Bus 13.
- T-1502 [P3] Bus money by region on a road route: `regionMoney.ts` shares a route's road km by the chords between stops, rougher on a winding route; share by the route's own points (notes/T-1465.md). Open.
- T-1467 [P3] Buses stage B3 if wanted after playing (notes/T-1445.md): a bus lane upgrade (ferries are now T-2177's). Buses are settled (Anita 2026-10-27): only if she asks. Open.
- T-1463 [P3] Rename "train share" and the like now that bus-only trips count in it (Anita: later). Open.
- T-1446 [P3] Share a game by link, read only (Anita: yes). Open.
- T-1447 [P3] Copy and paste blueprints between saves (Anita: yes). Open.
Anita's TODO.md "short term", filed 2026-10-26 (afternoon).
- T-1270 [P3] Terrain, stage 4 (stages 1-3 built, ARCHIVE): optional, not recommended first, a 120 km/h cap on mountain-class track at -1 and up. Open; notes/T-1270.md.
- T-993 [P3] noritetsu's account delete question (`noritetsu/dist/account.js`, "Delete your account? Local saves will stay.") should say trainsworld's leaderboard entries go too, since the account is shared (T-972); its privacy page could mention trainsworld. Open.
- T-1791 [P3] Line groups, stage 4 only (stages 1-3 landed, ARCHIVE): boardings in the group view; left out of stage 2 as a later stage if wanted. Open; notes/T-1791.md.

## Track and trains
- T-2063 [P3] Watch the T-1966 waits at a crossing junction move on screen. Neither New York save has a crossing pair of junction moves (the real save's lines share no node's moves; the shared save has none), and `checks/t1966-cross.mjs ... test` builds a Y with one pair but too few trains meet there to see a wait. Open; notes/T-1966.md. notes/T-2063.md.
- T-2164 [P3] After T-2064 the real save has no flat crossing, so `checks/t1966-cross.mjs` and `t1966-pics.mjs` find no waits on it: give them a save with drawn crossings that trains meet at (like the check's `test` Y). Open; notes/T-2064.md.
- T-2165 [P3] Mark `data/work/realnet/jpn7_real.save` (44 flat crossings, after India's run frees data/) and `artifacts/trainworld/t737-japan-trains.save` (60) all `sep` with `notes/T-2064-mark-sep.mjs`, as T-2064 did New York, and rerun the T-2093 ridership figures on jpn7. Open; notes/T-2064.md.
- T-1967 [P3] Opposing trains on single track are drawn through each other between passing points (T-1965's guard covers one direction of a track only). Open.
- T-1794 [P3] T-1787 leftovers: on shared-track New York a headway edit on its busiest line still takes 150-400 ms (two or three settles of ~15 rounds), and Paris placements p95 237 ms; options in notes/T-1779.md "T-1787". Open; notes/T-1407.md "Left". notes/T-1794.md.
- T-1816 [P3] Demand's `transfers` question names the zone in view, so a player who pans to a zone outside the solve and then edits probably gets no transfer pulses until they look back (found in T-1814, untested): ask by the edited network's zones instead. Open; notes/T-1810.md "T-1814".
- T-1842 [P3] `TrackApi.past_day` (the panels' past 24 hours) takes ~40 ms on shared New York, on every account change and 4 times a second while the clock runs: keep each line's tally in `Boarding` and redo only the lines whose account changed (found in T-1815). Open; notes/T-1815.md.
- T-1808 [P3] `pipeline/calib.py run nyc` fails on its `--areas` file: data/work/realnet/cell_county.u8 has 61,427 values against the pack's 54,218 cells (found in T-1801); rebuild it from the current New York pack. Open.
- T-1793 [P3] The drawn platform crowd overstates riders where transfer pulses (T-554) are large: at Jamaica (line 40's terminal, 08:34) pulses plus the steady rate make about 9,750 an hour against 7,297 arrivals, so the platform shows about 1,958 while the train leaves with 1,501. Found in T-1785. Open; artifacts/trainworld/t1785/.
- T-1778 [P3] T-1777's leftovers: the tools' snap under a still pointer is kept in local units, so after a draw-anchor move the ring and ghost sit in the wrong place until the pointer moves; `map.project` placers (bus drawing, touch bar) are not hidden on the globe's far side. Open; notes/T-1777.md "Left".
- T-1776 [P3] Train sounds when zoomed in on trains (Anita's TODO): very light pass-by and acceleration sounds, slightly varied like the track placement sounds, a soft "sheuuu" as one passes, pitch rising a little as it accelerates, shaped by the game's playback speed. Open.
- T-1773 [P3] Shared real networks on their real levels: each piece's level from the graph's tunnel and bridge flags (-1/-2 in tunnel, +1 on a viaduct or over water), ramps of 200 m a level where there is room, else the neighbour's level; today every piece is on level -1 with every edge `sep` (T-1305 stage 3 left it out). Check New York first. Open; notes/T-1305.md.
- T-658 [P3] Double track drawn as its two tracks when zoomed far in (Anita 2026-10-19: "somewhat important"; all four stages built 2026-10-19, notes/T-658.md "as built"). Left: a node drag's preview of single track still shows two tracks close in (`map/nodeDrag.ts`); then archive.
- T-691 [P3] A station tried on edges 997 and 1130 of the real save (an overlapping pair, 688/1075) is refused with a track-drawing reason ("The curve is too tight... would run on top of existing track"): find where the worker's station trial produces it and give the station's own reason. Found in T-671. Open. notes/T-691.md.
- T-1072 [P3] Changing a line's stops at an end (a stop added past the last, the first removed) moves the other direction's running trains by the length added or removed: flights are kept as offsets along the run; match them by place on the ground instead. Open; notes/T-1071.md "Limits".
- T-1073 [P3] Undo or redo of a train count takes effect at once, without T-1071's lead, so at a fast clock its departures hop a little; tell the worker the step was a service edit. Open.

## Code health
- T-2359 [P2] Code and documentation cleanup (Anita 2026-11-04). Surveys done (T-2360/61/62); stages below, their order and waits in notes/T-2359.md (A = notes/T-2360.md app, E = T-2361.md engine, D = T-2362.md docs). Open while any stage is.
- T-2575 [P3] T-2512's item 26 step two: the realRoute worker's `cover` message carries the boxes the page needs, so regions.json (1.5 MB) is fetched once, not twice; changes the realRoute protocol and when the file is fetched. notes/T-2512.md. Open.
- T-2573 [P3] `checks/t409-worker-check.mjs` still does not exercise the walker rule: rail pairs carry no walking share (parts[6] = parts[7] exactly) even with stops by the walkable Boston/Providence cells; needs a line serving two stations at both ends of a walkable cross-pack pair. notes/T-2527.md. Open.
- T-2460 [P3] A boat's rated load in the track sim (found in T-2373): `LineSvc::rated()` (sim/src/track/service.rs) calls a bus or boat full at its crush, so a boat's hourly stretch loads (`stretch_hours`) are over 480 while the app, the demand kernel and the docs give a boat 320 at the rated load. Nothing on a ferry panel reads them today; make it 320 if boat crowding is ever shown. Open.
- T-2374 [P3] Cleanup A7: splits of demandClient.ts, clock.worker.ts, demand.worker.ts, renderer.ts (tools.ts optional), one agent per file, after T-2371 and T-2372. Open.
- T-2375 [P3] Cleanup A9: main.ts's debug probes, one URL-parameter module, unused CSS, tsconfig `noUnused*`. Last of the app. Leftovers from T-2369 in notes/T-2375.md. Open.
- T-2377 [P3] Cleanup D5: trainworld/CLAUDE.md cut from ~3,570 to ~2,400 words, orchestrator text to docs/orchestrating.md. Agents decide the reorg (Anita 2026-11-04: "feel free to make decisions among yourselves regarding reorg"); show her the result after. Open.
- T-2378 [P3] Cleanup D6: DECISIONS.md and ARCHIVE.md split by month, the lint capping entry length. At a quiet moment. Open.
- T-2380 [P3] Cleanup E3: retire data/packs/intercity.json (read by nothing, 13.8 MB copied into every deploy) and joint_gravity's branch that writes it. Between joins. Open.
- T-2382 [P3] Cleanup E6: pipeline and pack_gate small items (T-1114 pipeline F19-20, I). Between joins. Open.
- T-2383 [P3] Cleanup E9: sim refactors after T-2381 (T-1114 sim H 20, 21, 23; `pass_window`, `realcorridors.build`). Open.
- T-1114 [P3] A general code cleanup (Anita 2026-10-24): its leftovers are now T-2359's stages, except pipeline G (the nocar layer, changes pack bytes, with T-878 and T-926), which waits on the next data batch. notes/T-1114/.
- T-1183 [P3] After T-1146: take the fixed-rate line path out of the engine (`capacity.rs` `continuous` rounding, `LineData.trains` as an Option, the test-only `Op::Schedule`, `set_line` and `set_schedule` in `wasm.rs`), which about 30 sim tests pin exact trains per hour on, so they need rewriting onto train counts. Open. notes/T-1183.md.

## Waits on a rebuild
Tasks that land only when a pack or country is rebuilt, grouped by that rebuild, so whoever runs it knows what rides along (2026-10-26). The two replan groups are also in notes/T-1189.md "What rides along".

### Europe's next rebuild (T-1189's Europe replan ran without these, T-2302, 2026-11-05)
- T-617 [P3] Rebuild gla, lis and pot on T-610's unit matching. notes/someday.md
- T-772 [P3] Dead-space units holding cities (Valladolid, Agder...); Anita: low priority. notes/someday.md
- T-785 [P3] France's countryside zones on the France module's sources. notes/someday.md
- T-771 [P3] A Spain module from the ministry's phone trip tables and INE's grid. notes/someday.md
- T-2083 [P3] Eurostat's Census 2021 1 km grid as population for every built EU country (GHSL is good there; small gain). research/next-countries-data.md
- T-853 [P3] A tertiary share where UNESCO has none (Kosovo's Prizren). notes/someday.md
- T-783 [P3] Melilla registered as Morocco, so its trips take the border factor. notes/someday.md
- T-765 [P3] Work at home is 0 in archetype D packs; a per-country constant needs Anita. notes/someday.md
- T-810 [P3] Commutes across open EU borders far too damped at 0.05. notes/someday.md
- T-429 [P3] Cross-border long trips in Europe low on land pairs (Paris - Brussels). notes/someday.md
- T-932 [P3] Countries solved in turn leave a held neighbour off its margins. notes/someday.md
Also T-1344's Paris and London items if those metros are rebuilt.

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
- T-985 [P3] Wei and Linquan rows carry solve_region henan though in Hebei and Anhui. notes/someday.md
- T-996 [P3] GHSL population spikes (Dazhou, Benxi); measure against T-1150's bound. notes/someday.md
- T-1232 [P3] `osm_units.enclaves` can leave land in no outline (Ninghe). notes/someday.md
- T-1258 [P3] China's countryside packs hold few OSM medical sites; a fallback. notes/someday.md

### Korea's next rebuild, and Toronto's (tor was only re-solved in T-1239)
- T-564 [P3] Korea's registration as a tier 0 hook (Canada's DA population done in T-1239; its jobs by tract of work not yet). notes/T-555.md
- T-1243 [P3] Korea's resident registration counts by si-gun-gu for the plan rows. notes/someday.md
Also T-1344's Toronto (T-282, T-357) and Seoul (T-504) items, T-1345's Toronto airport (T-283).

### Russia's build (held)
- T-791 [P3] Moscow's jobs far too even (Anita: "definitely noticeable"); Russia is built (T-2295) on tier 0's job proxy, so the fix is a general change to the proxy. notes/someday.md

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
#### Pipeline (between runs: the runner's files are in use while one runs)
- T-612 [P3] Runner: delete a parked row's pack files once its reason is recorded.
- T-766 [P3] world_run records each extract's PBF snapshot time. notes/someday.md
- T-386 [P3] tier0's signals log totals inside the boundary only. notes/T-291.md
- T-943 [P3] market_zones' Python part at a gate's register, 28 of 33 s. notes/someday.md
- T-986 [P3] Region list upkeep for newly built countries. notes/someday.md

### World and countries
- T-1344 [P3] One-city data sources, 9 folded tasks (Paris Filosofi, schools and Flores; London HESA; Toronto StatCan and outline; Seoul KOSIS; Mumbai jobs and curve): take one when cheap or when its pack is rebuilt. notes/T-1344.md
- T-967 [P3] Keep region caches until the join (its first half, fetching the next region's extracts ahead, is T-1980's). notes/someday.md
- T-968 [P3] The join's per-pack loops and the gate's register across processes. notes/someday.md
- T-969 [P3] Gate figures made in the build workers, not the serial gate. notes/someday.md
- T-970 [P3] About 8 s less build cost a row (h3, GHSL reuse, fewer processes). notes/someday.md
- T-592 [P3] Read the next region's extract while this one gates and joins. notes/someday.md
- T-570 [P3] Regional car-free homes from DHS and open census small areas. notes/someday.md
- T-571 [P3] National no-car figures for larger countries on T-562's GDP rule. notes/someday.md
- T-563 [P3] Fit an archetype A planned city's own curve at build time. notes/someday.md
- T-784 [P3] Low-memory devices: lighter settings and a notice on very big networks (one solving worker on a phone or 4 GB device is done, T-2122). notes/someday.md
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
- T-2385 [P3] T-1920's far-end reach for everyday purposes: their samples carry the subzone pair's choice, so a far cell past `pair_walk_max` can still count rail. T-1920 moved 0.04% of far rail riders on New York, so likely low value. notes/T-1920.md. Open.
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
- T-678 [P3] Day and night tint on the base map (T-676 is done). notes/T-676.md
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
- T-1936 [P3] Run profiles with many holds in one hop are slow: `profile::run_with` works the hop out again for each hold, so T-1924's following waits cost the shared New York save ~10% on edits. A single pass over a hop's holds would let the waits sit at every station the slower trains stop at. track-and-service.md 6.2. Open. notes/T-1936.md.
- T-1937 [P3] Departure spacing (T-1925) leaves flat crossings out: a crossing's other-route users carry no place on their run (`at` 0), so lines meeting only at a flat crossing are not spread. Give them their crossing offset, then count them in `spacing::place`.
- T-340 [P3] Quad track's rest: end-node stations, express pair side. notes/someday.md
- T-702 [P3] Station ghost leftovers on overlapping imported lines. notes/someday.md
- T-755 [P3] No room for ramps at Lake Hamana on the traced Tokaido Shinkansen. notes/someday.md
- T-763 [P3] Penn Station-Emeryville 3-3.6 s from rest to preview. notes/someday.md
- T-914 [P3] A snapped route Schiphol - Amsterdam Centraal refused over water. notes/someday.md
- T-430 [P3] A straight-then-arc fallback for too-tight clicks, only if wanted.
