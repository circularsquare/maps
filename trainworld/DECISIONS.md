# Decisions

Newest first. One entry per decision: what, and why. The spec states the result; this keeps the
reasoning and what it replaced.

## 2026-10-20
- **The globe: our own projection from MapLibre's camera, not its shader prelude** (T-033, my call 2026-10-24): our layer draws on its own canvas and context, outside MapLibre's programs, and the prelude's `projectTile` takes tile coordinates; so every vertex shader of ours calls `project()` (`map/gl.ts` `PROJECT_GLSL`), the same sphere, horizon depth and globe-to-flat blend as MapLibre's `_projection_globe.vertex.glsl`, fed from the custom layer's `defaultProjectionData`. On the flat map it is exactly the old `u_matrix * p` (screenshots identical), and from zoom 12 the globe is the flat map, so close-in precision is untouched.
- **Globe close in follows MapLibre: flat from zoom 12, building everywhere** (T-033, my call 2026-10-24): MapLibre's "globe" is a globe to zoom 11 and blends to the flat map by 12; we draw and pick through the same camera, so the tools need no switch to flat and stay on at every zoom (picks and the track tool checked at zoom 5.5-10 off centre). Far side: points behind the horizon clip by MapLibre's depth (a station or train goes whole with its centre), project to NaN for picking and hovers, and DOM marks there hide (`hiddenOnGlobe`).
- **Globe sizes stay the flat map's px at the same zoom** (T-033, my call 2026-10-24): lines, dots, bubbles and trains keep their flat-map size in px (`pxPerUnitAt`), rather than being rescaled by latitude to the globe's true scale; the overview bubbles are px-sized anyway, and a ground-sized bubble far from the view's latitude differs by the Mercator factor as it does on the flat map.
- **The globe setting is Settings > Map, "Flat" or "Globe", flat by default, before Theme** (T-033, my call 2026-10-24); its hover says the globe becomes the flat map zoomed in to a city.
- **The realm shade on the globe is cut into triangles no longer than 1/64 of Mercator's width, with caps over both poles** (T-033, my call 2026-10-24): about 0.3 px from the sphere at world zoom; the caps (the realms never reach a pole) are drawn only on the whole globe, since MapLibre fades its own poles while it blends to flat.
- **Fares never go below $0 at 0 km, so demand and income agree; the free-short-trips curve is no longer possible** (T-1114 sim batch D, Anita's orchestrator call 2026-10-24): with the multi-node fare curve gone from the sim, demand charges a base plus a rate per km on each ride, and a line below $0 at 0 km (say $0 at 5 km, then $0.10 a km) cannot be that. The Money tab holds the dragged or typed node where the line meets $0 at 0 km (`fitOrigin` in `game/fareCurve.ts`; `tidyNodes` raises an older save's first node), and `faresFor` is the same base and rate with no clamp.
- **A map start's "every realm shown" ends at its first station or track** (T-1133, my call 2026-10-24): it is a picking aid, not the player showing each realm, so it does not count as "shown this session"; a map-start game with nothing built shows them all, also after a reload, and with a station shows what any game would, so a reload never changes the look. Kept as a set of every realm (not "everything shows"), so the shade still darkens the countries no zone covers; the cut, 151k triangles, runs in the shade worker (notes/T-1133.md).
- **Uncrowded zoomed out by dot spacing, not by realm size** (T-1133, my call 2026-10-24): Anita asked for small realms to draw no per-city bubbles zoomed out. Crowding comes from how close dots sit, and big realms crowd as much (Germany's 113 zones span 124 px at zoom 2.5, eastern China's 201 about 156 px), while one-zone realms never do; so every overview zoom leaves out an unopened zone's dot within 20 px of a bigger zone's mark (`DOT_GAP`), keeping each area's biggest cities. Opened zones always draw.
- **Current location's start** (T-1132, my call 2026-10-24): the zone whose coarse outline holds the location, else the nearest centre among zones whose box is within 40 km (the view's own reach, `ZONE_REACH_M`), the camera on the location; farther out a map start centred there at zoom 4, so a player in an uncovered country sees what lies near rather than being dropped into a far city. Waits 15 s for an answer; no cached position.
- **Ground track 0.5x, elevated (+1) 1.0x** (T-1068, Anita 2026-10-23): "raise 0 elevation rail cost from 0.3x to 0.5x (to account for the like building knockouts and road crossings that we're not actually computing). also i guess raise +1 elevated rail to 1.0x". `LEVEL_MULT` in `params.rs`; ground is now US$45M a km of double track, +1 US$90M, the same as -1. Only new construction and blueprint quotes change (stations, junctions and crossings priced from the level follow); built track keeps what was paid, and upkeep is per route-km, not by level.
- **"Ready" for the map is the loading card going** (T-1051, my call 2026-10-23): `mapBlank` is the start menu or a game load not yet released (set at the pick and at every whole-game load, cleared where the clock is released, on a failed demand load and on "play without demand"), not the first world snapshot: the realms are known well before the first demand, but Anita wants the end state drawn once. A load can take 15 s or more (Tokyo, the NYC save over a minute), so the basemap alone shows that long; the card is the only thing over it. Dead space (`demandIdle`) is never blank.
- **World view draws no cities and no region circles below zoom 2.5** (T-1046, my call 2026-10-23): `CITY_MARKS_MIN_ZOOM` in `map/intercityView.ts`; at 2.5 a 1,100 px map shows about 140 degrees, a hemisphere, with 10 or so circles, which still reads; the whole world on a 1,400-1,900 px map is zoom 1.6-2.1. Only the unasked overview is hidden: a city the player selected still draws its trips at any zoom.
- **A region circle is an unopened zone's dot, 16 px** (T-1046, my call 2026-10-23): fill from the Unopened zones setting at the bubble opacity and a 1 px map ink outline, as the overview's dots for zones not opened; those are 4-10 px across, opened cities 10 to 70 px, so 16 px is a little bigger than a typical dot without outweighing a real city.
- **The shaded city grey is the outside shade** (T-1048, my call 2026-10-23): the fill over everything outside the focused city (7.5% of #182536 light, 35% black dark); a hidden realm is that plus the `inactive` fill (total 28% and 64%). Half way to it is a total of 18% and 50%, which is just the `inactive` opacity halved (11%, 22.5%).
- **A click opens a hidden realm only when it did nothing else** (T-1049, my call 2026-10-23): select tool, no Shift, not linking stations, inside one of the realm's zone outlines (not sea or dead space, which are shaded too), the world zoom excluded, and the click left no selection (checked after the pick's own handler ran), so a bubble or line drawn over a hidden realm (a selected city's far ends) keeps its click. The shaded land has no hover or pointer cursor of its own: the circle's tag says to click the shaded area.
- **Labels: 130 px between labelled bubbles, 28 x 16 px between label boxes** (T-1050, my call 2026-10-23): larger box gaps alone only moved a label to another side of its bubble (Nagoya's hopped above its bubble), so the cluster stayed; keeping bubbles apart drops the smaller one of a cluster and the next biggest elsewhere takes its slot, still 8 at most.
- **A unit's fixed list is filtered to the registry, not rebuilt** (T-997, my call 2026-10-23): `intercity.solve_units` costs about 6 minutes on China's 1,516 outlines, and a parked pack leaving a group cannot add a neighbour within the cut, so dropping unregistered ids gives the same list.
- **Planned US rows spread blocks from their internal points** (T-1016, my call 2026-10-23): the `usa` module reads only each state's block .dbf (8 MB by range for Georgia), so a block's people go over the cells nearer its internal point than any other block's by GHSL, and its jobs by the jobs proxy, rather than over the block polygon by land as the built packs' spread.py does; the polygons would be about 20 times the download. County totals come out within 0.5% for people and 1.3% for jobs on the test rows.
- **Planned US rows: home end from population, one work-from-home share** (T-1016, my call 2026-10-23): tier 0's home end (population x commuters a resident) and ACS B08301 over the row's counties, where the built packs use LODES RAC and ACS by tract and industry; both would need new tier 0 hooks for a small change (RAC: T-1034).
- **Puerto Rico rows take the `usa` module's outline, people and work from home** (T-1016, my call 2026-10-23): TIGER municipios, 2020 blocks less institutions and ACS B08301 all cover it; jobs (no LODES), the curve and one region stay tier 0's, as for any country.
- **US metros by Europe's rule** (T-1004, my call 2026-10-23): a GHS-FUA commuting area over 1M (54, giving 44 rows beside the built packs), not an urban centre over 1M (25 in the US: low-density cities such as Atlanta, Tampa, St. Louis and Charlotte have centres under 1M and would have been countryside zones).
- **US countryside zones: counties grown within their state to 0.5M, cap 1.5M, no dead space, contiguous** (T-1004, my call 2026-10-23): Anita's floor (China's 1M was for counties already a European zone's size; the median free US county holds 23,000); every county in a zone, as "whole states as zones" asks, with T-1000's cap for the huge western zones. Small states come out one zone each.
- **US rows take a `usa` module, zones included** (T-1004, my call 2026-10-23): census blocks less institutions, LODES jobs and sectors, the US curve 0.8/30 and the 0.23 state factor, TIGER county outlines, on the tier 0 runner. The sources are national, open, already coded for the 13 built US packs they will be solved with, and clearly better than GHSL and the jobs proxy; Europe's zones keep global sources only. Schools, universities and airports stay tier 0 until T-1019.
- **Alaska: one zone of its road and rail belt; the four small territories out** (T-1004, my call 2026-10-23): Anchorage, Mat-Su, Kenai, Fairbanks, Denali, Southeast Fairbanks, Chugach and Copper River (0.66M, the Alaska Railroad's line); the rest of Alaska (0.16M, the Aleutians across the 180th meridian among it) is dead space. Hawaii and Puerto Rico are in; Guam, the Northern Marianas, American Samoa and the US Virgin Islands (each under 0.2M) wait for their own regions.
- **Riverside County is a metro row** (T-1004, my call 2026-10-23): GHSL's Los Angeles area holds San Bernardino but not Riverside, and a 2.4M county cannot be a zone under the 1.5M cap; Imperial County joins it rather than cross into Arizona.
- **US joint commutes by Census division** (T-1004, my call 2026-10-23): eight solve regions, Delaware, Maryland and DC with the Northeast so the 13 built packs solve together; the Northeast leaves its own `intercity.json` solve for the general one (T-1014).
- **The US's metro rows are planned alone and kept by other replans** (T-1015, my call 2026-10-23): `world_plan.py metros --countries US` plans US FUAs against US counties only and writes only US rows (`FUA_COUNTRIES`); a replan of the other metros keeps them, since the plan's `state.pkl` holds no US units and `--fresh` would recompute every country.
- **San Diego and San Francisco are named in the plan** (T-1015, my call 2026-10-23): GHSL's "Tijuana" and "San Jose" are replaced in the plan rows (`METRO_NAMES`) rather than only in zone-names.json, so the pack ids follow the cities (snd, snf), not Tijuana.
- **Across water, a zone joins another region's only under 80% of the floor, as by land** (T-1015, my call 2026-10-23): South Jersey (0.44M) stays a zone of its own instead of joining Delaware's across the bay. It holds for every `contiguous` country's next replan, China's and Taiwan's planned rows included.
- **Geofabrik extracts: a set of states beats a regional group** (T-1015, my call 2026-10-23): `extracts` weighs sibling extracts that group no other sibling against a single covering one, by area, and takes a smaller single extract anywhere in the index that covers the outline: Chicago takes Illinois and Indiana rather than us-midwest, Cincinnati and Louisville no longer the whole US, Puerto Rico its own extract rather than central-america. Recomputed for every other row in the plan, nothing else changes.
- **Puerto Rico's rows take the `usa` module as archetype C** (T-1015, my call 2026-10-23): population and work from home from the module, jobs from tier 0 (no LODES there).
- **Long commutes: cap the job pull at 20x, add back 1%** (T-1000, Anita 2026-10-23): inside a zone a job's pull is at most 20 times what distance alone gives; 1% of each zone's workers commute to jobs 50-150 km away through the between-zone commutes. Built before the US joins; the built countries' joins rerun later in one fix batch.
- **T-1000's constants, as built** (T-1000, my call 2026-10-23): the cap 20 against each pack's own job-weighted geometric mean pull (`kernel::PULL_CAP`), the share 1% (`joint_gravity` `LONG_SHARE`, about Japan's 1.6% crossing a prefecture line), the ring's inner edge 50 km (`LONG_KM`; where a daily commute becomes a trip between towns, and about the 16 km x ln 20 the cap is worth on China's curve), outer 150 km (the existing cut). Each pack's unheld columns share one fill so its net exchange stays nil; the add-back leaves the balanced shares by pair as they were (jobs scaled with homes, not reduced by its arrivals).
- **Ring zones: a planner guard now, regroups later** (T-1005, Anita 2026-10-23): the planner skips a neighbour that would make a zone wrap another (half of B inside A's hull, A holding half of B's zone border); built countryside rings merge under the cap or re-split, in the same later batch; rings round a metro stay. Brugge-Kortrijk and Lausanne-Fribourg-Neuchâtel are not merged (separate cities).
- **One later fix batch** (Anita 2026-10-23): T-1000's joins, T-1005's regroups (Taiwan's Zhubei split, T-1002, among them) and T-903's European merges run together after China and the US.

- **Hidden regions are shaded by the zones' own outlines, not land shapes** (T-1006, orchestrator's call 2026-10-23, on Anita's "without relying on the accuracy of a shapefile"): one shade over the whole world, sea included, with the shown regions' zones cut out along `outlines.json`, so the edge is a zone outline and never a coast; gaps the zones enclose (central Spain, central France) count as shown. The app no longer reads `realms.geojson`; a station's region is the zone whose coarse outline holds it, else the nearest zone centre, and a region's button sits at its zone nearest the mean of its zones' centres (my calls).
- **Xinjiang's joint commutes are solved with no fixed neighbours, and a solve runs up to 300 steps** (T-998, my call 2026-10-23): held at their own factors, any one of the huge packs around it (Golmud alone, or Golmud, Zhangye, Alxa, Shigatse and Lhasa together) kept Xinjiang's column error at 1e-3 to 4e-2 through 500 steps; alone it reaches 9e-9 at step 249 (`intercity.SOLVE_ALONE`). Its commutes with those regions come from their own solves, which hold Xinjiang's packs. The cap went from 150 to 300 since the best step is kept, so a stalled solve only spends time.
- **An OSM relation holding most of another row's unit is not matched to a neighbour** (T-897 wave 4, my call 2026-10-23): `osm_units.foreign` drops it from a unit's several-relation match, since geoBoundaries draws some city districts far smaller than OSM, and the zone built first took the city's centre (Yingkou's Zhanqian into Dashiqiao's zone, Qitaihe's Xinxing into Bin's). The build order no longer decides which zone holds a city.
- **Huge western zones ship with commutes of 50-160 km** (T-897 wave 4, my call 2026-10-23): under the 1M floor Golmud's zone is 680,000 km², Lhasa's 659,000, Wensu's 556,000; the gravity sends commutes between oases far apart (gol 164 km, weu 112, smz 100, aks 91), relaxed per row with reasons as big zones are (Anita, 2026-10-22). Cutting dead space from them waits until performance or play needs it.
- **Leagues and prefectures name zones as autonomous prefectures do; two centres in one zone show the bigger city** (T-897 wave 4, my call 2026-10-23): a league's or prefecture's seat (Xilinhot, Ulanhot, Hotan, Artux, Tacheng) is an alias, never the zone's name; a zone holding two prefecture-level cities' centres shows the bigger with the other an alias (Wuhai with Shizuishan, Tonghua with Baishan, Jixi with Qitaihe, Zhangye with Jiuquan and Jiayuguan, Lhasa with Chamdo, Nyingchi and Shannan), and Karamay, a prefecture-level city, names the zone led by Shihezi.
- **An autonomous prefecture's seat does not rename its zone** (T-897 wave 3, my call 2026-10-23): wave 2's rule (a zone holding a prefecture city's centre shows that city) is for prefecture-level cities; the seats of autonomous prefectures are small county-level cities, so a zone holding one keeps its most populous county's name with the seat as an alias (Dujiangyan with Kangding, Pingwu with Barkam, Tengchong with Mangshi, Lanping with Shangri-La and Lushui, Kaili with Duyun, Panzhou with Xingyi, Wenshan with Mengzi). Which zone holds a centre is read from the built outlines (OSM), not the plan's geoBoundaries units: Yibin's centre is in yib, Guilin's in Lingui's zone, Beihai's and Qinzhou's both in Hepu's (shown Beihai).
- **A sliver neighbour extract is dropped from a Chinese zone's row** (T-897 wave 3, my call 2026-10-23): under about 1% of the outline outside the main extract (jng into Yunnan 0.68%, wao into Shaanxi 0.91%, qib into South-East Asia 1.17% at the Vietnam border), so one zone does not chain whole provinces into one region and one gate; the outline comes from the main extract's OSM admin polygons anyway.
- **Regions are "realm" in code** (T-975, 2026-10-23): realms.json, pipeline/realms.py, packs/realms.geojson and the registry's `realm`, since pipeline/regions.py already exists; players see only the region names. Land: Natural Earth admin 0 cut by geoBoundaries ADM1 (China, Russia) or the Census states (US), parts over 2,000 km from all of a region's zones dropped; the US split at the Census West region, Russia at the Urals by federal district.
- **A clicked region lasts the game in hand** (T-975, 2026-10-23, Anita's answer): it resets at each load unless a station is there; everything shows until the first game's first snapshot, so a loading save never flashes dark. A station's region is the land holding it, else its zone's.
- **Builds run one thread each; other runner steps take all `--jobs`** (T-966, 2026-10-23): a build step uses about 0.91 cores whatever it is given, and a run's builds are idle while its reads, gate or join run.
- **Pack outline boxes cached in `data/work/pack_boxes.json`** (T-963, 2026-10-23), keyed by name, size and mtime, not beside the packs; the coastline read from `data/raw/osmdata/water-polygons-split-4326.fgb`, made from the zip when missing or older, `zip_fid` keeping the zip's order.
- **The browser retry lives in city-browser.mjs** (T-965, 2026-10-23): a timed-out zone gets one try in a fresh page; rows split over the 3 sessions in plan order, in contiguous chunks.
- **A Chinese zone holding a prefecture city's centre takes the city's name** (T-897 wave 2, my call 2026-10-23): when the centre districts sit in a zone led by a county (Hengyang's in Hengshan's zone, Changde's in Hanshou's, Jingmen's in Zhijiang's), the zone shows the city and the county becomes an alias; the county's namesake zone keeps "County" (Hengyang County, Ji'an County). Jiangxi's Fuzhou shows Linchuan, beside Fujian's Fuzhou metro; Taiwan's island zones show Penghu and Kinmen, the names English uses for the islands.
- **Taiwan's national figures from its own statistics** (T-897, my call 2026-10-23): the World Bank has no Taiwan, so `tier0.NATIONAL` holds its employment ratio and share aged 0-14 (DGBAS 2024, the Ministry of the Interior end 2024); hospital beds take the world figure and tertiary students go unscaled, as for any place without a figure.
- **One anita.garden account shared with noritetsu** (T-972, Anita: either is fine, 2026-10-23): sharing needed only a token check and a delete hook on the server; deleting the account from either page deletes both pages' data.
- **No account icon until the server says so** (T-972, 2026-10-23): the game shows sign-in only when `/noritetsu/config` reports `trainworld`, and a localhost page has no account server unless `?acct=local|live`, so smoke and the pre-deploy live page make no failing request.
- **Game ids are per browser slot** (T-972, T-973, 2026-10-23): kept in localStorage, not the save header; a game that turns sandbox sends one notice and the server refuses it for good. Join stays enabled in a sandbox game with "Sandbox games are not listed" under it, since joining is per account.
- **Unopened cities: light grey dots with a dark outline** (T-976, Anita 2026-10-23, replacing plain black): fill #d9d9d9 with a 1 px ink outline, at least 2 px radius so the outline leaves some fill; Black, Maroon and Grey stay as choices. Stored settings below version 8 take the new default once.
- **Which market rows to redo comes from a record of their inputs** (T-929, 2026-10-22): each row's pack hash, place, row hash and centre, plus the market_pack binary and the rule, not the ids a caller passes, so parks, restores and runs of older code are caught; anything unexplained redoes every row.
- **The gate's rates test is built before the first region** (T-931, 2026-10-22): synchronously at the run's start rather than beside the first builds, to stay inside `--jobs`; the first recompile is paid up front.
- **Chinese county names keep "County" when bare would mislead** (T-897, my call 2026-10-22): a one-syllable county shows "Cao County", "He County" (and the pilot's Pei), as English writes them; a county sharing its name with a city or county nearby shows "Puyang County", "Quzhou County" (Hebei's), "Yucheng County" (Henan's). Two real cities of one name both keep it (Suzhou in Jiangsu and Anhui, Taizhou in Jiangsu and Zhejiang, Fuyang in Anhui and Zhejiang); the map tells them apart.
- **A pack with no OSM medical facility spreads its visits by jobs** (T-897, my call 2026-10-22): Qihe and Yi County have none; an empty medical purpose, the last slot, made the kernel drop it and the empty slots before it (3-4 purposes, the gate wants 6). Jobs are tier 0's fallback for missing places, and the draw is relative. Parks: one park off every cell no longer fails a pack (two must, Yongjia had 1 of 17).
- **A loading card that cannot finish gives way to the failure screen** (T-950, my call 2026-10-22): a clock worker failing before its first game and an uncaught error in main.ts's setup fail at once ("The game could not start", Try again alone, since there is no game to play without demand); otherwise a watchdog fails a card that sees nothing move for 3 minutes with no bar, 10 with one (a big pack batch opening on a slow phone sends nothing for minutes). Query answers and the clock's money ticks do not count as movement, and a hidden page counts no quiet. Times in `ui/LoadingScreen.tsx`.
- **No big calibration at the end** (Anita, 2026-10-22): demand is calibrated in parts, task by task, as it has been since T-636; T-551, which waited on a deep calibration, is settled.
- **No incremental join** (T-928, my call on measurements, 2026-10-22): a solve unit holding a new row is still solved whole. Holding far members at their factors, measured on four real cases, was slower on compact units (Jiangsu with xuz new: 378 s against 200 s whole) and, where fast, not the full result (GB with hih new: 155 s against 765 s, zones up to 0.9% off, since the whole solve also repairs drift from earlier solves). Starting from the old factors took as many steps. `joint_gravity --hold-report` stays as a diagnostic.
- **Publishing reads a freeze** (T-928, my call 2026-10-22): hard links into `data/frozen/<stamp>/` of every file a production build copies, plus copies of the four game JSON files, made by the runner after each clean join or by hand when no run is live; about 2 s and no space on NTFS, safe because pipeline writers replace files rather than rewrite them (a size and time manifest catches one that does not). The newest three are kept.
- **One browser session per region gate** (T-928, my call 2026-10-22): all of a region's rows in one sandbox game, each zone's test line removed after its checks, 20 zones a page load; kept per region rather than once a batch, since China's regions are provinces of 30-100 zones and a batch-end check would change when rows park. The fixed waits became waits on the condition, capped at the old times. Same riders and purposes as one zone alone; about 10 s a zone instead of 27-30.
- **The gate's other-packs check compares every pack** (T-928, a bug, 2026-10-22): `other_packs_unchanged` overwrote the registry it filtered by with the first pack's hashes, so it compared one pack; any FAIL it gives now is a real change.
- **Three loading bars at most** (T-939, my call 2026-10-22): the zone in view first, then the loading zones whose centres are nearest the map's centre, re-picked when a move ends. Zones of one load share a fraction, so "most advanced" would tie; nearest the player's eye is what they will look at. Cap `MAX_BARS` in `map/zoneLoading.ts`.
- **A selected bubble: everything solid and 2 px bigger** (T-936, my call 2026-10-22): "selected" is a commuter-bubble or zone-bubble (city) selection; the Opacity slider is ignored for every bubble on screen meanwhile, and the radius gets +2 CSS px (hit tests too, `bubblePad` in `map/hovered.ts`), not +2 px only on the small ones, so sizes keep their order.
- **Driving 10% quicker, owning a car about 30% dearer** (T-938, Anita's ask, my steps 2026-10-22): every local drive speed x 1.1 (16.5 / 44 / 88 km/h, open highway 110) and ownership $2.00 a trip + $0.13 a km (was $1.50 + $0.10). No speed had ever been cut, so +10% rather than undoing a cut; the two pull against each other and ownership wins slightly: ceilings A and B up about a point (Houston A 86.4 to 87.9), real networks +0.4-1.1% rail. notes/T-938.md.
- **Big zones are fine; only small ones merge** (T-903, Anita 2026-10-22): the gate's `area_km2` over 40,000 km² is a warning, no longer a fail (Highlands and Islands is 41,100). Zones too small in people or area are the ones to merge (the people / 0.5M + km² / 10,000 rule).
- **Vestland's drive factor relaxed** (T-903, my call 2026-10-22): `drive_factor_median` 2.08 against the 2.0 cap (fjords make every drive long), as `gate_relax` on its plan row; all its other checks passed.
- **De facto lines are a short explicit list** (T-897, my call 2026-10-22): `world_plan.DE_FACTO` clips geoBoundaries units only along real lines of control and the territories kept apart (China against India, Bhutan, Taiwan, Hong Kong, Macau; India and Pakistan in Kashmir; Crimea and the Golan with the side holding them, Anita), not against every other Natural Earth row, which dropped Åland and moved border villages where the two sources merely draw the line differently.
- **China's zones are contiguous; islets stay out** (T-897, Anita's no-exclave rule, my numbers 2026-10-22): `zone_groups(contiguous=True)` crosses at most 10 km of water to a neighbour; island zones of one region join within 50 km (Penghu, Kinmen); a zone that joins nothing and holds under 10,000 people is dead space (Matsu at GHSL's 7.8k, Green Island, Orchid Island, Xiaoliuqiu). Units left alone inside one metro join it (Jianghai, Lüshunkou, Sanzhi, Jinshan, Shimen).
- **China solves its joint commutes by province** (T-897, my call 2026-10-22): rows carry `solve_region` (Beijing, Tianjin and Hebei together; Shanghai with Jiangsu), and `intercity.solve_units` treats each as a country, neighbours within 150 km held fixed (T-786's way), since one group of about 730 packs passes MAX_GROUP_ZONES and every wave would re-solve all of it.
- **The Pearl River Delta by prefecture; Macau and Hong Kong alone** (T-897, 2026-10-22, Anita's split): Guangzhou, Shenzhen, Dongguan, Foshan, Zhongshan and Jiangmen rows; the planned zhu row held only Macau's units, so it is now Zhuhai, and Macau (mcu) and Hong Kong (hon) are rows of their own countries. Every Chinese row runs on Beijing time, Xinjiang included.
- **Chinese zone names from Wikidata division codes** (T-897, my call 2026-10-22): a zone shows its most populous county's English name without its kind; a zone led by a city district shows the prefecture city's name, once per prefecture (the oldest district), unless a metro row is that city.
- **A drive to a station counts double** (T-916, my call on Anita's ask, 2026-10-22): `ride_drive_f` 2, so 2.66 perceived minutes a minute against 1.33 for driving the whole way. Of the simple levers it is the one that cuts long drives to a station and keeps short ones; a larger start constant (25-40) did the reverse, and the drop-off and its cap on station parking changed nothing in suburbs. From homes about an hour out, riders driving to a station into the core fall from 52% to 38% (New York) and 38% to 25% (London); most of them walk to a station instead. The 80 minute access bound still sees the drive at 1.33: at 2.66 it cut drives off past about 15 minutes, which emptied central Tokyo's drives (its streets are slow) and took smoke's Tokyo line from 160,824 riders to 103,403; with the bound at 1.33 it is 120,471, and New York and London hardly move. notes/T-916.md.
- **Long press on bubbles kept, panning held back while it waits** (T-908, 2026-10-22): it worked when held still, but MapLibre's pan started after 3 px of hand jitter so it looked like a drag. Now the press's moves within 6 px are kept from MapLibre (a window capture listener before its document one), 0.3 s arms it with a ring, past 6 px it is a normal pan. If it still feels unreliable, drop it; Shift+drag stays either way.
- **One controls panel, not hints in panels** (T-908, 2026-10-22): the "?" beside Saves and Settings lists only gestures, modifiers and keys no button shows; it floats beside the dock and closes on Esc or a press outside, so it costs no panel space.
- **Merges wait; Bergen now** (T-903, Anita 2026-10-22): the European merge batch is planned but not run yet; only the dead-unit change runs now (Bergen, Trondheim, Oulu and Inverness as zones). When merging, islands far apart are never joined (Corsica never with the mainland); nearby island groups may be (Greek islands, some of the Canaries): the contiguity rule's 10 km water gap holds this.
- **Small European zones merged by people and area** (T-903, orchestrator's call 2026-10-22): a zone stands when people / 0.5M + km² / 10,000 >= 1, else it joins a touching neighbour in its region (T-897's contiguity rule). Stage 1 (Italy, France, Greece, Portugal: 124 zones to 57) runs when the China runner has a gap, with Bergen, Trondheim, Oulu and Inverness as zones (a dead unit holding 0.4M or more stands alone) and the Border, Steyr and Düsseldorf exclaves fixed. Countries planned from now on, China aside (1M floor, Anita), use the same rule. The floor alone made French zones 26-34k km².
- **Local bubbles' colour smoothed among nearby cells** (T-900, 2026-10-22): a Gaussian kernel over res-9 cells weighted by trips (sigma 500 m, cut at 1,250 m; 350 m left visible steps at subzone edges), chosen over blending the nearest stations' shares because it needs no station data in the bubble worker and also softens the per-zone walk share. Colour only; a selected bubble shows its real colour, so it can shift a little when selected.
- **Zone bubbles are a red dot with a blue dot inside** (T-901, Anita 2026-10-22, replacing T-863's pie rim): blue area = train share; the blue's radius is at least 2 CSS px (capped at 0.6 of the red's) so any train shows, a share under 0.05% rounded up to one step. The pie rim code is gone; `SPLIT_BUBBLES` false still gives the solid mixed disc.
- **Demand bubbles show a new result at once** (T-906, Anita 2026-10-22, reversing T-156): the local bubbles, nearby zones' and the zoomed-out overview draw each result as it lands, as their clicks already did; `map/catchUp.ts` is gone. Money and the riders counters keep their own smoothing (Live feel), untouched.
- **China whole, as planned** (T-897, Anita 2026-10-22): floor 1M (cap 2.5M, about 652 zones); Hong Kong, Macau and Taiwan included as their own countries, never joined to China in names, groups or labels; borders clipped to the de facto line; the empty-land (20 a km²) rule dropped for China, since open areas are handled well now, with dead space cut from very large counties only if performance needs it; Guangzhou split by prefecture; the pilot province is the agents' choice. Backups in `data/work/pre-T###/` go once a later state is checked clean, without asking.
- **Cells under 50 trip ends merge into the nearest busier cell** (T-850b, 2026-10-22): 50 is the largest floor that moved under about 1% of a typical countryside pack's trip ends (0.5-1.6%, about 0.9 km) with no visible change on test lines; 75 and up started moving small stations 4-6%. The busier cell keeps its own index and place rather than moving to the group's centre: fewer trip-end metres moved, never on water, and read_pack's bounds hold. Remerged all 757 packs: cells -31%, `.bin` -18%. notes/T-850.md 5.
- **Islets under 0.05 km² count as sea when building outlines** (T-888, orchestrator's call 2026-10-22): the fifth of T-888's fixes; packs stay byte-identical on aan, vri and pik, the drawn boundary moves within tier 0's 0.0003° simplification. A unit match within about 1% of the 0.5 threshold could in principle flip; none did.
- **A saved game opens fitted to its network** (T-895, 2026-10-22): the save stores no camera (the view lives only in the URL hash, which a menu pick ignores), so the fit is the only view there is; it runs on a start-menu pick, as the first state arrives (before demand), over built stations (all stations if none built), padding 70 px, max zoom 12, no stations: the opened zone, else the world map. A slot opened in the saves panel and an imported file fit too (Anita, 2026-10-22, via `manageSaveSlot` and `importGame`, so `SavesPanel.tsx` is untouched); `?continue=1` and raw `import` messages keep the camera (checks rely on it). The Build tab's Track rows (Design speed, Platform length; what Anita meant by T-893) and the blueprint cost breakdown's rows take the 6 px padding of the zone-trips rows and no dotted divider.
- **The fare curve** (T-883, 2026-10-21): straight between nodes (the most predictable to read off the graph), carried on past each end at that stretch's slope rather than flat, so the default two nodes keep long trips' fares as before and a cap is a level last stretch; nodes clamp between neighbours so it never falls. In demand, route search still charges a base and rate (the first stretch) per ride, since a curve of whole-trip length is not additive along a route; each station pair's cost takes the rest on its strategy's expected km. Income takes each line's trips at their mean length (exact for a straight curve, a little high for a cap on a line mixing short and long trips). Pressing on the graph adds a node, a double-click or the bin button removes one, down to two.
- **Mixed grid on for new packs** (T-852, Anita 2026-10-21): every region run after T-788 wave 4 passes `world_run.py --coarse 1000`; existing countries take it at their next routes rebuild.
- **A 4 km block decays as its four squares** (T-852a, 2026-10-21): a zone pair with a block takes the mean of the 2 km decay between their squares, not the decay at the centres with a 4 km intrazonal distance (which pulls a block's own trips about 30% less), so blocks keep what four evenly loaded squares would; distances (drive, walk, bands) stay centre to centre. A block needs at least two zones of one region. notes/T-852.md 6.
- **Mixed rural grid at 1,000** (T-852, Anita 2026-10-21): a 2x2 block of 2 km zones becomes one 4 km zone when every zone in it makes under 1,000 trip ends a weekday (about 27% fewer zones); new packs first, existing countries at their next routes rebuild (notes/T-852.md). Intercity bubbles' pie rim stays red and blue only, walking in the red (T-863).
- **The choice curve's floor and ceiling** (T-854, demand agent 2026-10-21): 5% of the trips not walking always take rail while it costs at most the car's perceived minutes, fading linearly to none at 3 times the car's, and 5% never do while the car costs at most rail's, mirrored. A ratio bound so a detour line three times worse than driving gets nothing (the farming the design warned of), on rail of rail + car so walkers are not handed to trains; 5% at 3 keeps tests A and B (London A 85.6) and gives slow suburban lines 1.5-1.7 points (notes/T-854.md "Stage 2").
- **Track and station upkeep** (T-867, Anita 2026-10-21): build notes/T-867.md's proposal: track $30k a km a day ($45k high-speed, $60k quad), one station figure $50k a day at every elevation, trains $9 to $4 a train-km (car-km $2 unchanged); a line's profit includes its share of the upkeep.
- **Upkeep follows the track price factor** (T-867, 2026-10-21): single track pays 0.6 times double track's upkeep, as its price; quad is two double-track edges. A line's share weighs each line by its weekday trains (tph times the hours at each level); a region's share counts each edge where its midpoint is, each station in its own region. A load that settles again gives back the upkeep of the hours since the save's clock, as it does running costs.
- **Sparse cells merged, rural zones coarser** (T-850b, T-852, Anita 2026-10-21): "we should probably merge more of these small cells, and use a coarser grid maybe for these super rural areas". T-850b's merge goes ahead (floor 50 trip ends, or higher if measured to cost nothing visible), and a coarser zone grid for very rural areas is to be built once measured.

- **The fare graph's nodes** (T-864, 2026-10-21): the base node sits at 1 km or mile and the rate node at 100, each dragged vertically only; the base keeps the slope and the rate keeps the base. The graph and the figures follow the pointer, but the fare goes to the game once on release, not as colour does with an undo step: fares are outside the undo history and each change restarts demand. The steppers stay under the graph for exact amounts. The y axis is log from $0.10 up to twice the fare at 100 (at least $10), keeping its scale during a drag: a linear axis made the base node move about $0.40 a pixel.
- **Line colours and the Cities totals row** (T-856, T-857, 2026-10-21): the default palette keeps its thirteen colours and gains olive, coral and lavender (each at least 19 CIEDE2000 from every other and 17 from the map's land, water and parks), so a random pick has more to choose from; a new line takes a random colour unused in its city, else a random one of the least used. The Cities list's totals row ("All cities") is always shown, even with one city, and is not sorted with the rows; its split bar weighs every city's trips.
- **Car ownership out of demand** (T-854, Anita 2026-10-21): drop the per-cell car-free share and add parking at the home end from residential density (factor 0.4 of the destination's rate, the note's pick, unless play says otherwise); remove "other" (bus, taxi, bike), so "car" means car and other. No separate rail-leaning class: instead change the soft choice curve itself so that where rail exists some riders always take it and some never do (a floor and ceiling inside the curve). The note's warning stands for the build: a plain floor can be gamed (a useless detour line picks up about 2% of every trip it touches), so bound it, e.g. by how much slower rail is.
- **A line's opening day** (T-848, 2026-10-21): each stop's platform starts filling one headway before the first train that reaches it, never before the opening, since trains enter only at the ends; the account keeps the trains actually drawn, so an opening day shows fewer trains, not the steady pattern.
- **Headways on a line with no trains** (T-849, 2026-10-21): switching to headways keeps the trains a line has (T-752) except at a level with none, which gets its headway's count, so a new line set straight to headways runs; Construct with nothing to build says so (`NothingToConstruct`).
- **Service span and rail between cities, as they are** (Anita 2026-10-21): hourly lines losing about 5% of riders to the 30-minute end of the wait tolerance is "very reasonable, maybe even higher, fine for now"; London - Paris at 58% rail of rail plus air "seems fine" (T-828), so the market's reach rule stays.
- **Riders know each stop's service span** (T-842, 2026-10-21; Anita: a soft tolerance of about 0.5-6 h, shape the agent's call): tolerance for the wait until the next train is log-uniform from 30 minutes to 6 hours (the share who would wait w is ln(6 h / w) / ln 12, median 1.7 h), and `phi` per line, direction, stop and period is the riders' average share, rounded to hundredths. Boarding costs -ln(phi)/0.10 extra perceived minutes (a stop at 0 gets no departures in the search); long trips feel it less through the logit's damping, kept on purpose. The boarding account weights arrivals by the same share, keeping each period's total. Real save: rail trips 10,279,724 to 10,247,553 (-0.3%), sparse lines -8.5%, frequent unchanged; hourly service loses about 5-7% at midday from the 0.5 h end of the ramp. notes/T-842.md.
- **Reversing at a bare track end** (T-797 B3, agent's calls, 2026-10-21): a node with one edge end and no station (a headshunt); trains reverse there without stopping, standing the line's turnaround with their tail 20 m short of the end; the route search turns there at the same 600 s as at a stop, never at a passed station or a junction; capacity is a one-track pooled terminus plus the reversal limiter on its one side (12 an hour in all, 6 each way for one line); a reversing train takes at most half the distance to the next standing place, so close reversals cannot overlap; a station later added on such an end breaks the line until it is made a stop.
- **Loop lines** (T-798 L1+L2, agent's calls, 2026-10-21): closing is offered only when it adds no reversal at the old last stop; a loop's middle for T-835 is half a lap from its first stop (the lead formula gives it unchanged), both directions leaving the first stop together; demand learns a circle from the times it already gets (first and last stop the same station, a positive time at the last); the boarding account runs a circle in two passes, riders aboard at the first stop booked off and on; odd train counts round up to even; a reroute never turns a circle into a balloon.
- **High-speed running** (T-817 b2, b3, 2026-10-21; per-line choice is Anita's): a stretch runs at the lowest of its curve speed, its track's cap (160, 320 on high-speed track, 50 on a crossover) and the line's train top speed (160 ordinary, 320 high speed); a third traction band of 0.3 m/s² above 200 km/h, only for high-speed trains (0 to 320 in 206 s and 11 km). The train type is saved per line (`LHS1`; old saves ordinary); the fleet is kept per type and spares never cross; high-speed cars cost $3.75M and run at $12 a train-km plus $3 a car-km. The route search times high-speed edges at 320 for every line, so ordinary lines may prefer such track too (kept for simplicity).
- **High-speed track as an edge flag** (T-817 b1, agent's calls, 2026-10-21): `EdgeData::hsr` on double track (save flag 64) at 1.5x double track at its level; stations, junctions and flyovers on it cost as on double track. HSR meets ordinary track at plain nodes but joins under a platform only with its own type; it cannot be doubled up to quad; a station on open ground with High speed set lays HSR. Picking High speed sets the design speed to 320 and picking Double or Quad again restores the earlier one (not kept across a reload). Its look is a thin centre line along the stroke, 18% of its width (at least 0.6 px), carried by lines over it too.
- **A station on open ground: the first click is one end of the platform** (T-907, T-917, Anita, 2026-10-22): the platform is the Build tab's set length, pointing from that end towards the pointer, and the second click sets the direction and places it (not a length from the distance between the clicks). Before the first click only the snap ring shows; within 8 px of the first end the last heading is kept. Replaces T-834's middle-then-direction below.
- **A station on open ground is placed in two clicks** (T-834, agent's calls): the first click sets the middle, the piece turns about it towards the pointer, the second places it; within 8 px of the middle it keeps the last heading; before the first click the ghost shows the last heading used (east-west at first); Esc after the first click drops it. T-794's R key and aim at the nearest station are gone.
- **Select tool picks are of one kind** (T-815, T-829; the rule is Anita's, details the agent's): a pick with any blueprint keeps only blueprint, otherwise constructed track and stations; a click takes the piece under the pointer, station first, blueprint before constructed; a junction goes with its track. The selection bar is the multi-selection's inspector: count, length, the seven elevations, Disable, Enable for blueprint, Remove (Delete) and Clear for either. The Build tab's elevation buttons still mean the level for new track.
- **Re-levelling several pieces moves their joints too** (T-815): a node whose track is all in the pick moves with it, so a picked run has no ramps at its joints; others keep their level and the piece ramps. Pieces that cannot change are left and counted ("9 of 12 pieces set to elevation +1. 3 could not change."); one undo covers the batch. Removing a blueprint pick is one undoable edit; a constructed pick asks first and is final.
- **Hidden tab** (T-826, 2026-10-21): "Run in background" is off by default and off keeps the old behaviour, the game pausing cleanly while its tab is hidden (the clock froze on visibilitychange; nothing half-ran), so leaving the tab never costs money unseen. On, the clock stays wall-anchored and the clock worker's own timers settle money and autosave; no worker-driven clock was needed. Hours passed hidden pay their fares evenly within the hour (same hour totals) rather than working out a boarding hour nobody sees.

- **No track or station upkeep for now** (Anita 2026-10-21): research/running-costs.md option C ($0.6M a route-km, $1M a station a year) stays unbuilt; running cost per train-km and car-km stands in for it.
- **High-speed track and trains** (T-817 stage b, Anita 2026-10-21): running above 160 km/h needs high-speed track; high-speed trains are the player's choice per line, not automatic on lines touching that track (the note had proposed automatic).
- **Zone names** (T-825, 2026-10-21): a zone shows one short name, worked out in the app so joins never undo it: a city zone its biggest city, or a metro name only where it is clearly what people call the whole zone (Ruhr for Dortmund's, Kansai for Osaka's, which holds Kyoto and Kobe; Frankfurt and Mannheim keep their city names over Rhine-Main and Rhine-Neckar); a countryside zone its unit's name unless that is a statistical label or a compound, then its biggest city or an established name for the whole (Dodecanese, Cyclades, Teesside). Every other name is a search alias. The list and reasons: notes/T-825.md.
- **Design speed** (T-817 stage a, 2026-10-21): the Build tab's design speed (160, 200, 250, 320 km/h; 160 by default, exactly today's drawing) sizes the curves the drawing tool makes (the tangent arc and lead-ins) at radius ceil((v/3.6)^2 / 1.1): 1,796, 2,806, 4,385, 7,183 m, set on the edge as an explicit radius; no per-edge design speed, saves unchanged. Routes snapped to the real network keep their fitted radii. For high-speed track (stage b, notes/T-817.md) 320 is recommended over 300: the curve rule was calibrated on 320 km/h LGVs and it saves about a minute per 100 km.
- **Reversing partway** (T-797 B1+B2, 2026-10-21): at most 12 reversals an hour per station and arrival side (300 s a train; the orchestrator's call after Anita deferred: a real switchback holds its platform about a turnaround, like a terminus); a line's two directions count as one stream in that limit's delay (as two, a lone line at 11 an hour was charged about 6 minutes of made-up delay). A large amber U-turn icon marks every station where a line reverses, the same size at every zoom and never hidden, since Anita expects most reversals to be accidents. In stop editing a station that adds a reversal is a fourth marker tier (grey, dashed ring) and outranks "keeps the path". A reversing train waits for room before the platform, as at a line's end; a reroute after a deleted edge is accepted only with no more reversals than the old path.
- **Coarse zone outlines: one file in data/packs, 300 m, shared runs kept identical** (T-800, agent's calls, 2026-10-21): `packs/outlines.json` beside the packs (gitignored like them, copied by `vite.config.ts`), not bundled like the registry, since it is 1.7 MB. Simplifying each ring alone would give two neighbours different points along a shared border and break T-799's draw-once rule, so runs two zones hold are simplified once in one direction with their ends kept. A zone whose own boundary loads while a neighbour is still coarse draws the shared border twice, under a pixel apart, until the neighbour loads.
- **A station on open ground aims at the nearest station or loose end** (T-794, agent's calls, 2026-10-21): east-west when there is neither; R turns it 45 degrees, Shift+R back, until the tool is left. The piece is the platform plus 10 m each end at the Build tab's level and track count, so a later route joins without a stub; within snap reach the tool snaps onto track as before, and a flat crossing on the piece is refused (it would sit at the platform's end).
- **Running cost $9 a train-km plus $2 a car-km** (T-807, Anita's pick of option B in research/running-costs.md, 2026-10-21; replaces T-805's flat $3 a car-km): a 9-car train costs $27 a train-km as before, a 4-car one $17, a 12-car one $33, since only about a quarter of a real metro's cost grows with train length; the 100x economy, the hourly settling and the split by route km are unchanged. Upkeep per route-km and per station (option C) stays unbuilt.
- **Prices against a cheap-country metro** (T-805, Anita 2026-10-21; undoes T-467's halving): track base $90M a km and the station table doubled, so a tunnelled km with its station is about $150M as in Korea or Spain; running cost $3 a car-km so thin areas lose money; the start back to $6B (my call: with doubled prices $3B no longer bought the one real first line T-485 aimed at, about $4.2B).
- **Agent worktrees land as one squashed commit, fast-forwarded** (T-792, 2026-10-21): no merge commits and no agent commits; `land` rebases the tree's trainworld/ changes onto master, so history stays one line. Trees are full checkouts (about 1.2 GB) rather than sparse ones, since git 2.29's sparse-checkout in a worktree sets the shared config and could make the main tree sparse.
- **Countryside zones hold at least about 0.5M people** (Anita, 2026-10-20, T-788; replaces T-788's first plan of NUTS-3 units as they are in most countries and 0.3M grown zones in Germany, the Benelux, the UK, Austria, Switzerland and Slovenia, 507 rows): every country's units are grouped within their NUTS-2 (ITL2, ADM1) region until each zone has 500k, then across regions in the country when the region has no neighbour left; under 100k by the longest border as before, from 100k the smallest neighbour that keeps both within 1.5M (the longest border put Sarajevo, Zenica and Mostar in one zone of 1.39M); islands keep their rules (within 150 km under 100k, 10 km above, so the Isle of Wight stays alone), and islands and enclaves that can join nothing stay as they are (Åland, Trebinje, Norway's three zones between dead units). Applied to every row not yet built: wave 1's 184 built or gated rows (Germany 93, the UK 52, Ireland 7, the Benelux 32) keep their 0.1-1.2M sizes; the plan is 360 zone rows. Spain, Italy, France, Portugal and Greece stay as built. notes/T-788.md.
- **The 20 a km² dead-space line is kept as it is for the Nordic and Baltic countries** (T-788): it leaves Vestland (Bergen), Trøndelag, Agder, Pohjois-Pohjanmaa (Oulu), Keski-Suomi, Lõuna-Eesti (Tartu) and most of Latvia, Sweden north of Uppsala and Iceland outside Reykjavik as dead space; Anita accepts dead zones, and a rule for a dead unit's populated part belongs with T-772.
- **Cyprus is NUTS's one unit, the whole island; Moldova's zones include Transnistria; Kosovo's units are NUTS's** (T-788): each is the unit the source draws (GISCO's CY000, geoBoundaries' Moldovan raions, GISCO's XK districts), as for every other country; the game draws no political line inside them.

- **A joint group spanning countries is solved a country at a time, neighbours held fixed** (T-786, closing T-781): each country's packs are balanced with the other countries' packs within the 150 km cut at their current factors, and each pack pair's commutes come from the newest solve covering either end. Chosen over one group with a higher cap (France alone is 1,077 s of the one group's 1,759 s; Germany, Switzerland and Austria would pass 400k zones) and over a split by size (it would cut inside countries, where pack lines take 0.4 rather than the border's 0.05). Measured on Spain, France and Italy: within each country the commutes across pack lines move 0.1-2%; across national borders, 0.015% of commuters, the first solve after a cold neighbour is off by tens of percent until that neighbour is solved again. notes/T-786.md.
- **Parallel runs share files under cross-process locks, and joins wait for a separate `join`** (T-786): a run's regions use about 4 of its cores, so countries run at once with `--no-join` and one `join` follows. Gates overlap each other (each registers only its own new packs), but never a joint solve, which would change packs under a gate's snapshot; making gates exclusive too cost the Portugal run 16 of its 48 minutes waiting. Rows another live run has claimed are left out instead of shared.
- **The sim-copy check warns, it does not stop** (T-775): a copy that is not HEAD's `sim/` is printed and reported, marking files that match the tree as uncommitted changes carried on purpose, since a run is often made to carry exactly such a change.

- **Italy and France covered whole as Spain was** (orchestrator's calls, T-779): 97 and 80 zones; France's overseas départements are not zoned (a European country's units outside lon -32 to 45, lat 27 to 72); zone rows never take a country module, even where one exists (France's), so every countryside zone is on global sources and the zone curve; Alpes-Maritimes stays out with the parked nic (T-769); no metro was cut down or grown. Mountain zones (Liguria, the Alps, the Peloritani, Corsica: drive_factor_median 2.05-2.55) and Trieste (top_zone_jobs_share 0.22, a province that is nearly all city) relax those checks, with reasons on each row. notes/T-779.md.
- **A joint gravity group may hold up to 400,000 zones** (T-779; was 200,000 from T-539): whole-country cover chained Spain, France, Italy and the Benelux-Rhine metros into one group of 298,507 zones, solved in 29 minutes on 9 threads; the runner gives the joint solve every core of `--jobs`. Splitting a covered continent's group is T-781.
- **The market pack's country is the pack's own** (T-779): `market_zones` took the country under the registry box's centre, which put Gorizia's strip in Slovenia and failed the joint group's border check; it now reads the pack header's country first.
- **The runner makes a vanished sim-copy junction again** rather than stopping (T-779): they vanished mid-run twice (T-613, T-779), cause unknown; the same target is linked again and the report says so; a junction pointing elsewhere still stops the run.

- **The gate's stations come from the extract, written per pack at build time** (T-770): a `.stations.fgb` beside each extract's points cache (same GDAL pass, railway=station|halt nodes) and a `data/work/<id>.stations.json` written by a runner build step, since the caches are deleted after each region and the joint re-gate comes after the last. The gate never queries Overpass; packs gated before keep their cached Overpass answers (`data/raw/stations/`, kept). Same stations as Overpass on Lisbon, Porto and Glasgow (111, 115, 161, every one). A reader of the points cache alone does not download an extract read before T-770 just for stations.
- **Run faults stop the run instead of parking rows** (T-764, T-773): a `TW_SIM_DIR` copy without its `data`/`app` junctions, or a gate or re-gate that leaves a row without a result file, stops the run (exit 2) with rows left `built`. A joint re-gate where only new packs FAIL parks just those and solves the group once more without them; when an older member FAILs, the new packs all park as before, since the solve cannot be pinned on one of them.

- **Whole-country cover, Spain first** (Anita, T-719 decisions 1 and 5; the rest the orchestrator's calls, T-723): zones are the plan's units as they are (NUTS-3 provinces in Spain), under 100k merged into a zone of their NUTS-2 region; small islands join their province's largest island, which the rule "an island joins the most populous zone of its region within 150 km" gives for every Spanish island; Ceuta and Melilla are zones of their own; a unit under 20 people a km² stays dead space for now (9 Spanish provinces, 1.46M people, three of them at 16.8-18.9); no Spain module yet (later tailoring); no registry split for Spain alone. Countryside zones borrow T-314's rule over the 38 fitted Japanese zones only (d^-1.4 exp(-d/16 km)) instead of the metros' d^-1.8 exp(-d/80 km). notes/T-723.md.
- **The runner solves each joint group once a batch**, after its last region, not after each region (T-723, T-719 speedup 6): a covered country is one group, and re-solving it after each of 18 regions would repeat a growing solve and re-gate every member each time. A batch stopped before the end leaves its gated rows registered and `built`, and the next run gates them again.
- **Ceuta and Melilla relax top_zone_jobs_share and mean_commute_km** (my call, T-723): towns of 12-19 km² hold a handful of 2 km zones, so one zone has a third of the jobs and commutes are under the gate's 3 km floor by size alone; `gate_relax` on both rows. **Asturias and Tenerife relax drive_factor_median** at 2.05 and 2.08 (the gate fails over 2.0): mountain valleys, and a zone of four volcanic islands; both passed the browser check. Nice (2.51) stays parked (T-769).
- **The 35 rows a runner fault parked were gated again without the browser check** (my call, T-723): each had passed the full gate, browser included, in the same batch; only the joint solve's re-gate had crashed on a missing snapshot.

## 2026-10-19

- **Lille, Bordeaux, Marseille, Malaga and Seville ship despite pop_vs_fua 1.50-2.17** (orchestrator under Anita's leave for sensible defaults, T-613): their plan units are whole départements or provinces holding more than the FUA, which only widens the playable area, and no neighbouring pack overlaps them; `gate_relax` on each row, as for dus.

- **No slot is written over with another game** (Anita, T-743, after New game replaced her serious game): New game, Load a file and a backup's restore each open a new slot, the game being played saved first; nothing asks for confirmation. The panel's New game takes the next free "Game N" with no name field (one click; Rename is on the card). A loaded file's slot is named after the file: a save carries no name of its own. Test pages (`?new=1`, `?synth=1`, `?demandTest=1`) replace the game in the page and own no slot, as they never wrote one. A slot that fails to load on start is left alone and the new game gets its own slot. **Backups** (T-744) ride on the autosave: one landing 5+ wall minutes after the slot's newest backup is also written over the older one (the first at the slot's first write), so they follow changes and cost no extra encoding; no picture, no demand cache.

- **Cells merge only within one walk part of a zone** (T-725), not "a merged cell takes its busiest child's part": the sim counts a zone's parts by its cells' highest `walk_part`, so a part whose cells all merged into another would shrink the count and the pack would not open (the walk table's size no longer matches); and a cell spanning two banks would bring back T-700's blended walk. A res-8 parent merged in two parts of one zone gives each merged cell its busiest child's index (as sparse groups have), since a zone may hold a parent's index once. Costs a few merges in zones split by water (Stockholm: 377 split zones, 34 such parents, 36,019 to 13,228 cells). The per-row gate override Anita asked for is a plan row's `gate_relax` {check: why} (a FAIL counts as a WARN), set for dus.

- **Düsseldorf ships despite pop_vs_fua 1.59** (Anita, T-613): its plan units hold 1.66M against Eurostat's 1.0M FUA, a real difference between admin units and the FUA rather than an error; relax the check for this row (a per-row override, not the threshold for all) and rerun it with the parked rows.

- **Whole countries from now on, Spain first** (Anita, T-719/T-723): the space between metros is covered with zones as Japan was; zones keep provinces where they are provinces and group small units to about 0.3-1.5M people (orchestrator's default from notes/T-719.md); countryside zones borrow a commute curve from Japan's fitted countryside zones (Anita: no strong view). **Access walks are split into 15-minute walking bands** (Anita, T-720), and the trip explorer prices its own trip exactly, cell to cell.

- **Walk bands of 15 minutes, not 10 or 5** (Anita, T-720): 15 leaves a short line's rail 3-7% over one subzone per cell (10 minutes: 0-4%, 5: within 2%) for about 1.5x the subzones and 1.9x New York's commute pass (10 and 5 cost more again); the explorer's own-cell answer covers the trips she inspects. Revisit if sparse networks still look too generous. notes/T-720.md.

- **A change of trains costs a fixed 3 minutes plus the walk between the two stations' centres** (Anita, T-717; reverses T-557's no-penalty rule): riders should prefer the straight ride when it is about as fast. **The regional layer takes T-710's route choice too** (Anita, T-718). **Russia's remaining rows are left off for now** (Anita); Moscow, Saint Petersburg and Voronezh stay. **Crowding stays at 70% of crush**, Richmond to Waterloo to be looked at again after the change penalty (orchestrator's call, Anita deferred).

- **A full platform side's red stays inside the station box, and each track is drawn 20% narrower once well apart** (Anita, T-713). This reverses T-652's stretch of the red bar past the box (it read badly and clashes with double and quad stations' platforms); the tracks narrow from 2.4 m to 1.92 m between 10 and 16 px for 4 m so the riders waiting behind show; trains stay 3 m (Anita: most of the time no train is in the station).
- **A line keeps the note it was given when built** (Anita, T-706), stored with the line; **the turn-back crossovers are drawn** (Anita, T-658 stage d); **the schedule table has no lines between rows** (Anita, T-692).

- **Double track shows as two tracks from about zoom 17, a platform each side of them; in Line + traffic thickness the two tracks draw over the traffic width, which stays behind** (Anita, T-658; the turn-back crossovers still wait on her).
- **Points keep their size** (New York about 350M a day; Anita, T-694).
- **No road-class or time-of-day drive speeds** (Anita, T-701): "start with just distance and maybe spatial traffic but keep the model quite light".
- **Nunito is the default font and the UI defaults to 90%** (Anita, T-703).

- **Everyday trips take one worldwide set of shorter decay scales (W1) and the parking base goes from 5 to 7 minutes** (Anita, T-637): errands 1.3 km, social 1.4, school 1.4, medical 2.0, university 3.5, replacing each country's survey-fitted scale, which matched the survey's mean length but put too few trips under 2 km (New York 23% against NHTS's 50%). The parking base holds her test A at 85%. notes/T-637.md.
- **A line's colour on the map follows a picker drag about every 0.2 s** (Anita, T-672): a middle path between every move (froze the game) and only on release.
- **Destination bubbles catching up over a game day, still while paused, is fine as it is** (Anita, T-674).

- **A headway line's train count is clamped at its track's limit** (Anita, T-679, proposal A of notes/T-679.md). Rounding the count up made a headway typed at the platform's limit (4 minutes) run a little under it, so a train was held out and an alert fired; now the line runs just over the typed headway (4:07, say) with no alert. A typed headway well under the limit still warns.
- **An edit short of cash is refused, with a shake and the train count in red; a headway shorter than the track allows is shown in red** (Anita, T-625, confirming the 2026-10-17 entry below and adding how it looks).
- **No adoption ramp for new lines** (Anita, T-680): the panels' past-24-hours counts are enough. Her complaint was a different one: 25,000 people waiting at once on a new line's platform (notes/T-680.md).
- **Points: a transfer is one trip** (Anita, T-684), or a player could farm points by splitting a ride; points unlock nothing yet. Stars in the Cities list come before drawn stamps (T-685).
- **Sound and faces on by default, only when zoomed in; happy faces too** (Anita, T-686).
- **The real-world rail layer comes from noritetsu's tiles, view only first** (Anita, T-682).

## 2026-10-17

- **Trains already cost money (T-028), so T-631 only made it visible: US$2.5M a 20 m car, a real metro car's US$2-3M (a New York R211 is about 2.6), and construction and car prices are the unscaled ones while fares and running costs run at the 100x economy** (T-631). A car costs about what 0.2 km of grade track does (US$13.5M a km of double), the same ratio as real life. Cars a line stops needing stay in the depot for any line to use, with no refund: it never punishes trying a smaller schedule, and a refund at some fraction would reward buying and selling.
- **A headway line's train count may run 0.5% over the chosen headway before it takes another train** (T-625). A line's round trip moves a few seconds with its own train count, and a train bought for 0.03% would be noise; the headway the player sees is the one they typed.
- **A headway line that needs more cars than the cash allows refuses the edit that made it need them** (T-625), a track extension included, as a hand-set count does. The alternative, running fewer trains than the headway, would break the promise of the mode quietly.

- **A plan unit may be several OSM relations, and what they leave uncovered keeps the plan's polygon** (T-610, notes/T-610.md). One relation per unit took Vale of Glamorgan for Cardiff's "Cardiff and Vale" unit and left Cardiff out, halving the pack; the build now takes the relations (level 8 or coarser, mostly inside the unit) that overlap it best together, fills a gap of a tenth of the unit or more from the plan's own polygon and prints it, rather than failing, since a cut-off relation at an extract edge (Monmouthshire) is common and the plan's polygon is a fair outline there.
- **The proxy jobs have caps: one OSM place at most 0.75% of all jobs, one cell at most 500 x the square root of the jobs in thousands; the excess goes to the population part** (T-583, notes/T-583.md). Thin OSM and GHSL packs (Aba, Aboh, Owerri) put 3-15% of their jobs in one place or cell. A flat share does not fit all sizes, so the cell cap scales with the root of the jobs, as real packs' densest cells do (100-575 times it; the cap sits at 500, the gate at 600); no existing proxy pack (Belfast 283, Seoul 147) reaches it, so none changes.
- **The gate's proxy top-cell check is the densest cell's jobs over the square root of the jobs in thousands, failing over 600, not a 1% share** (T-611). Belfast's centre at 1.28% is below real centres in cities of 0.5-1.7M jobs (Birmingham 1.68%, Liverpool 1.97%, Sheffield 1.31%); 1% was set when the proxy packs were all large cities. The new check still fails the Nigerian packs as they were before the caps (850-2,640).

## 2026-10-16

- **Packs share a joint commute gravity when their outlines come within 50 km, chained, across borders too** (T-539, notes/T-539.md). Any gap from 25 to 200 km keeps today's Northeast and Japan; pairs of packs over 50 km apart carry 0.3-0.5% of the commuters crossing a pack line, while 150 km chains continents (London to Berlin, Japan with Korea). The Northeast keeps its own solve through `intercity.json`.
- **Red on a platform box means the side is full, not that a train left riders behind** (T-559). Anita read red as full and found a platform red but not full: a side goes red when the last train left anyone, however few, while its fill is a share of the room to stand. Full is what the fill, the dot's pie and the hover's red count already measure, so red now follows the fill (100%) and the two cannot disagree. The station inspector's complex map follows. The left-behind count is still in the line inspector and the stations list, but no platform drawing shows it.
- **Station boxes keep their zoom-15 width on screen out to zoom 14; the crowd pie is solid grey or red** (T-553). Boxes are fixed on the ground from zoom 15 (`BOX_WIDTH_ZOOM`), but at 14 that would be 3.7 px with no white left inside its rim, so below 15 a box stays 7.5 px wide on screen (white 3.5 px) and only its length shrinks (40-55 px). A dot's pie is the full crowd grey or over-full red (the box fills' colours), not the old 45% and 40% tints, since the angle carries the share. At 08:30 on the real save the median station is 3% full and only 11 of 1,756 are over half, so most pies are slivers; a square-root scale would show more but is not the measure the hover states.
- **A city with no flows or distance table borrows the one fitted curve that loses least by country** (T-314, notes/T-314.md). `tier0.borrowed_curve` over `pipeline/fitted_curves.json`; today d^-1.8 exp(-d/80 km). Held out a country at a time (the pipeline's case) it beat nearest by population and vehicles, by income group, a world median and a length scaled to size (+2.9 points by country against +3.3 to +3.9; worst Baltimore +14.2). Held out a city at a time nearest by population and vehicles wins (+2.0), only because a Japanese city then borrows a Japanese one; with Japan out it sends Japan to a US curve (+24.8 worst). The top four candidates are within 0.1 point of each other.
- **Reported (road) distance bands stay divided by 1.3, not by the pack's measured drive factor** (T-314). Mumbai's routed factor weighted by its commutes is 1.30-1.45 at the B-28 band edges, and refitting with it moves 1.6% of commuters; `routes.py`'s logged 1.62 is the unweighted median of near zone pairs, inflated by sparse outskirts, and would move 8.6% (mean trip 8.2 km against 9.9). Routes also run after the build, so the measured factor would mean a refit.
- **Crowding colours come from riders' extra wait, not full trains** (T-491, replacing T-472's and T-477's full-train shares; Anita: they read red all the time). Score = the average rider's wait for a train with room (min; each rider left behind waits the gap to the next departure) / sqrt(their average ride on the line, min, at least 5). Grades: yellow 0.1, orange 0.3, red 0.6. So 2 min on a 10 min ride and 4 min on a 40 min ride are both red. On the real save at 08:30 this picks out 4 red and 3 orange lines, where the full-train rule gave 13 red. Red: HBLR (4.0 min on a 7.7 min ride), NJCL (4.9 on 17.7), Ronkonkoma (4.5 on 22.1), the Z (49 min). Orange: Port Washington (1.8 on 13.8). The Q has 8/16 trains full now but riders wait 0.5 min, which is only yellow. The square root and the grades were set by eye on that one save. The Lines tab reads the line's period now (the account's whole period, not the past hour). The dots are per period, so High shows the morning and evening peaks apart.

- **A new game starts with $3B, and running cost is $2.25 a car-km** (T-485, Anita; was $6B and $1.50). Only the two base constants changed (`START_CASH` in the clock worker, `RUN_COST_CAR_KM` in `params.rs`); the line panel, the money ledger, the history charts and the Build and Money hovers all read them. Construction (halved in T-467), train prices and fares stay, so the start buys about the network it did before the halving, and a line's running cost is half again as much against the same fares.
- **The Caps Lock follower never overrides the stored keep placing setting on the first key** (T-487). Nothing reports Caps Lock's state before an event arrives, so the first event only records it (unless it is the Caps Lock key itself); after that each change of state sets the switch, as before. A player who left Caps Lock on and keep placing off, or the reverse, gets what they left.

- **The Lines tab colours a line's train count by its trains in service now that are packed full somewhere on their trip, not by trains leaving full over the past 24 hours** (T-472). The count beside it is the trains its level needs now, adding trains is done per level, and the line's schedule dots (T-477) already grade every level over its day; a 24-hour share would mix peaks with nights and hide a packed peak at about a third. A whole trip counts, not the stretch a train is on, so a line whose every train fills at one busy stretch reads red rather than the few trains on that stretch. Orange from a fifth, red from half (`crowdClass`), the same grades as the dots; tuned by eye, nothing behind them.
- **Schedule crowding dots read the boarding account, not the demand solve's per-period line loads** (T-477). The account runs every train with crush limits and counts riders who could not get on the first train; the solve's loads are averages over a period that never leave anyone behind. Riders who wanted the line but were put off by crowding in demand's own rounds are not counted, so the hover's "left behind" is a floor on unmet demand.

- **Construction prices halved: track at every level, stations, junctions, flyovers and crossings** (T-467, Anita: high against Subway Builder). Done in the two base constants (`BASE_COST_PER_KM` 90 to 45, `STATION_COST_200` halved), so the level multipliers, the quad and single factors, water and every quote follow. Train price, running cost, fares and the $6B start are unchanged, so the start now buys about twice the network; the 100x economy was not touched.
- **Japan's non-city zones: prefectures whole, Hokkaido's subprefectures merged by the 100k rule, which gives 8 there, not the design's "about 6"** (T-420 stage 3). Each under-100k unit joins the neighbour with the longest shared border, smallest first; Nemuro goes to Okhotsk, not Kushiro, by that rule. A zone's curve is its own fit over a fixed grid on the census flows (minutes a zone) rather than one national curve; most fit flatter and shorter than the metros.
- **Japanese zone outlines leave out the Russian-administered southern Kurils and uninhabited islets over 50 km from other land** (Takeshima, the Senkakus) (T-420 stage 3). The census counts no one there and the game models where people are; the metro packs' outlines are unchanged.
- **Hamamatsu is one unit for jobs and flows** (T-420 stage 3): its wards were redrawn in 2024, so the 2020 census's seven and N03 2025's three cannot be matched ward by ward (old Kita-ku split).

## 2026-10-15

- **The past 24 hours are counted exactly in the sim, the whole line block with them** (T-446). `Boarding::past_day` counts the rest of yesterday on yesterday's kind of day plus today so far, so the fullest train and the spans of full trains are exact; subtracting two `today` calls in TS would have got riders right but not those. The whole line block (fares, running cost, full trains) moved to the past 24 hours with the riders, not only the riders, so its figures agree; Yesterday stays as the second choice. A train running over midnight can count as full in both halves (rare).
- **One-off costs are bars cut at the chart's top, running costs a line** (T-452). The chart is scaled to fares and running costs only, so a build day no longer flattens them; a one-off cost larger than the top is cut with a cap and its figure is in the hover. Stacked bars were the alternative; two lines and pale bars keep the chart light.
- **Money history: fares and costs as two lines, net as bars; hours over the past 48** (T-447). Lines read better than paired bars at 30 days in a 300 px dock; costs are all four kinds together (construction days dwarf the rest, by design: the hover splits them). 48 hours shows two whole days and the day-night shape; the hourly ledger is saved, older saves start it empty.
- **The everyday layer assigns every purpose in one walk, its loads identical to the bit, its transfers not** (T-433). Each purpose is a lane doing the kernel's assignment arithmetic in its order, so every load matches assigning the purposes one by one; the transfers count an arrival's riders summed over the lanes before the split onto departures, which differs from adding the purposes one after another by float rounding (median 1.2e-7, at most 4.1e-6 relative a platform pair, New York and Tokyo grids). Bit-exact transfers would need each purpose's arrivals kept apart until the end; they only feed the complex view. The pair loop stays scalar: lanes across its directions (T-036's way) were 15-20% slower, since that loop waits on memory (25 station-pair reads and 25 writes a direction), not arithmetic.
- **The commute pair loop goes SIMD with every result identical to the bit** (T-036). Lanes run across pairs (eight destinations at a time), not across a pair's egress stations, so each logsum, ln, exp and scatter sum keeps its scalar order and padded entries add exactly 0; the alternative, lanes over egress stations with a fixed lane order, would have moved figures in the last bits and the trip explorer's `pair_choice` with them, for no speed gain (most destinations have 1-3 egress stations). WASM SIMD (`+simd128`) is on for both modules: every current browser runs it.
- **Unneeded packs close after 60 s, or at once beyond two spare zones** (T-301). "Needed" is viewed, beside the view, reached by a station (track alone needs no pack) or in the load in hand; the market follows its zones and its station reach. Time alone would close nothing during a quick pan, and a count alone would close a zone just left; together a pan back over the last two zones finds them open and a long pan holds at most the needed zones plus two. Restarting the aux worker past a size (WASM memory never shrinks) was not built: closing already bounds its peak.
- **Startup solves run their crowding rounds in step across workers** (T-327). With the default 2-2-1 period split the one-period worker ran past the round a click asked about, so the click fell through to the final; a round now starts only when every worker has the one before. Costs the startup final about 0.7 s, no figure changes; edits' solves run free. Dropping it (`target` then asks one round further ahead) is the alternative.
- **Mumbai: 2011 census totals grown by district, four talukas scaled as one, half the jobs proxy on population, B-28's no-travel share as work from home, its bands as road distance** (T-292). Carried-forward district growth keeps the island city shrinking and Thane's suburbs growing, which GHSL (+14% everywhere, from projections) does not; the total is within 3% of GHSL's. OSM's lines between Kalyan, Bhiwandi, Ulhasnagar and Ambarnath do not follow the 2011 talukas, but their sum matches. Population share 0.5 (not 0.2) for informal and home-based work, by judgement. B-28's distances are as reported by respondents, so read as travel distance and divided by 1.3; its 20.1% 'no travel' make no trip.
- **Shanghai: OSM districts clipped to land, the 10.2 km mean commute as straight-line, Pudong O&D 0.85** (T-291). geoBoundaries' China ADM2 is 1:4M and dated 2017, so the districts are OSM's, two broken relations rebuilt from their subdistricts. The survey does not say how it measures; CAUPD-Baidu's location-data figure of 9.8 km is close, while dividing by 1.3 would contradict it. Pudong's transfer share is judged below CDG's and Pearson's for want of a figure.
- **Paris: jobs from INSEE's flow table, work from home 0.12** (T-289). Jobs are the 2022 flows' destination totals by commune (one file serves jobs and the curve; workers living abroad left out), 5.99M against the RP 2023 dossier's 6.03M. Work from home is France's Eurostat share (11.1% usually, 24.4% sometimes, judged at 0.6 and 0.2 of their days), though Ile-de-France works at home more than the country.
- **The OSM feature cache reproduces GDAL's hiding of relation members** (T-363). A closed way in a multipolygon relation is dropped from a read that also returns the relation, even with tags of its own (a hospital inside a campus relation); `osm_features.read` keeps that so packs match the old per-city reads exactly. Counting every such way instead is one line in `read`, at the risk of counting an area twice.

## 2026-10-14

- **London stage b: BRES jobs, Census 2021 LSOA population, GIAS schools, its own curve** (T-360). Jobs are BRES 2024 employment by LSOA (8.64M), spread inside each LSOA by the tier 0 proxy; the curve d^-1.85 exp(-d/30 km) is the best district-pair fit to the 2011 WU01EW flows on the pack's own homes and jobs (13.9% misplaced; Toronto's borrowed curve 18.2%); a flatter d^-1.5 exp(-d/20 km) fits the flows' own margins. Universities stay at the national tertiary rate until HESA (T-382).
- **Tier 0 rules after London** (T-360). Jobs proxy blend 0.3 GHSL non-residential x height^2, 0.5 OSM, 0.2 population (the joint best on Toronto and London, a small gain on both); the one-parameter curve fit uses power 1.4, not 1. Neither fixes London's core (a third of its jobs); that limit is recorded, not tuned away.

- **London's tier 0 pack: whole ITL3 units, Toronto's curve, a national work-from-home share** (T-288). Outline: Greater London plus every ITL3 unit touching it, West Sussex (North East) for Gatwick and Medway to close a hole (35 units, 14.45M people), population scaled to Census 2021 district totals. Curve borrowed from Toronto (d^-1.2 exp(-d/20 km)) since tier 0's one-parameter fit was far too flat there (T-287, T-314); T-360 fits London's own. Work from home 0.27 from ONS's hybrid-working survey. Heathrow's O&D share 0.7 by judgement.
- **Tier 0 universities are scaled to the national tertiary rate** (T-288). Population x UNESCO's tertiary enrolment a resident (World Bank SE.TER.ENRL / SP.POP.TOTL), with Wikidata counts and campus areas only spreading it. Without it London had 2.4M students: its counted institutions are dense central buildings, so the median density sized suburban campuses at 50-150k each. Undercounts university cities; HESA replaces it for London (T-360).
- **The demand cache is looked up by the network's hash, and stores lines and stations by place** (T-328). A slot switch sends its network before the slot it came from is announced, so the cache is found by a SHA-256 of the network (any slot's, one kept per slot), and a load renumbers ids, so the hash and the stored figures use each station's and line's place in the snapshot. Packs are matched by hashing their bytes as they download (no extra request), and the zones it covered must equal the new solve's, so an unreached zone or a changed pack never shows a stale result. The fresh final is written back after each final. The viewed zone's weekday bubbles (all purposes) are kept too when they were on show, 4.9 MB of the 7.3 MB for the real New York save, since without them the bubbles would wait for the fresh final (about 20 s) instead of the first estimate as before; clicks are not cached and wait for the workers.
- **A bare zone's longer trips: the aux worker sends its squares with the question** (T-336). The solving workers hold the regional market but not the zone, and the market needs only the selection as 600 m squares and the share of each selected, so the aux worker works those out from its bare copy (`bareSquares`) and the solving workers answer `icSel` from them; no solving worker opens the zone for a click. The workers' bubble-selection answers are now added up in period order (`byMask`) for solved zones too, so the same click gives the same figures to the bit whichever worker answers first.
- **A zone opens bare only once the network's reach is known** (T-325). Until the first network and the footprints arrive, the viewed city opens in every worker as before, so a save's startup never opens a reached city bare and then again. A bare zone that a station later reaches is downloaded again for the solving workers (the page keeps no copy), and the aux worker keeps its bare copy, so its bubbles stay until a shown solve covers it; that copy is not freed (WASM memory never shrinks anyway; T-301 can restart the aux worker).
- **Disabled blueprint is a flag on the piece, an undoable edit, saved with the game** (T-333, Anita: "disable them so that they're not included in the next construction"). It lives on `EdgeData` and `StationData` (`off`), so undo, splitting and joining carry it without id bookkeeping; the save uses spare flag bits, so old saves load and a game with nothing disabled encodes as before. Only Construct all and the quote skip it; a piece's or a line's own Construct builds it and clears the flag, since that is an explicit ask.
- **A drag box needs a tool of its own, key 5, "Select"** (T-333). Plain drag is the map's pan and Shift+drag is MapLibre's zoom box, and Anita wants Shift to add to the selection, so the box could not be a modifier. While the tool is on the left button draws the box and right or middle drag pans (handled in `map/blueprintBox.ts`); Esc clears the pick, then leaves.
- **Blueprint rails narrowed by scaling the whole width to 75%, rails kept at 1 px or more** (T-331, Anita: T-322's rails "maybe overshot just a little bit"). The middle takes what is left, so at zoom 15 it is 4.5 px instead of a line's 6.6, and a line planned along it is drawn that narrower.
- **Blueprint track is two dark blue rails round a light blue middle; dots smaller with a thinner
  rim** (T-322, Anita: blueprint "light blue", "slightly wider", "like two parallel rails"; dots
  "20% smaller", rim and connectors "a little thinner"). T-311's pale grey (the track colour mixed
  towards white) disappeared in the Lines view, where bare track is drawn thin under the lines.
  Blueprint track now has one look in every view: the middle is a line wide, so a line planned
  along it fills it exactly and the rails show either side, and the rails (16% of a line width, at
  least 1 px) are its casing pass, drawn for all segments before any middle, which keeps joints
  and junctions clean without any per-pixel work. Height and Max speed keep their data colour in
  the middle (lightened, as before) since the colour is the point of those views. Dot radius is
  1.32 times T-189's (T-313's 1.65 less a fifth); a dot's rim is 0.75 of a box's, set only for dots
  so boxes and their widths are unchanged; the connector's white is 0.3 of a dot's white radius
  (0.35 before), so it thins with the dots and a little more.
- **The closest-stations list moves from the bubble hover to the bubble's panel** (T-323, Anita:
  "we need to keep tooltips reasonably lean"). A clicked bubble's panel lists its 3 closest
  stations with line chips and walking minutes, as the station inspector does (one shared
  component). Only a selection of one area has it: the centre of several areas says little.

- **A check page (`?new=1`, `?synth=1`, `?demandTest=1`) does not autosave, until a copy is saved**
  (T-320, found in T-308: a `?new=1` page closed or hidden wrote its empty game over the `autosave`
  slot, so an agent's check in a profile with Anita's games could wipe one). Chosen over giving
  every `?new=1` game a fresh slot of its own: that would add a "Saved game" card to the list on
  every check, and the page's active slot is also remembered for the next start, so the next normal
  page would have opened the empty check game. Not autosaving is the smaller change and changes
  nothing a player sees: a `?new=1` game has no active slot (`""`) and reads none, the worker
  takes no lease on a slot, so `autosave()` does nothing, and "Save as a copy" still works (it
  makes a slot, takes its lease, and autosave goes on there). `?synth=1` and `?demandTest=1` load
  the last game as before but take no lease either: `?demandTest=1` feeds the real game riders
  from the hand-made network, which changes its cash and would have been saved. The top bar says
  "Autosave is off in this test page. Save a copy to keep the game." The client also no longer
  remembers a test page's slot as the one to open next. New game and Load a file in such a
  page change the page's game only.

- **Zoomed out every station is its own dot, a complex's dots one shape with connectors** (T-313,
  Anita: "stop growing dots to fit all the platform centers within the complex dot, it looks a bit
  silly"; dots "65% bigger radius"; the connector "should look like its made of the same
  material"). Replaces T-217's one dot per complex (centre of the smallest circle, standing for the
  busiest station) and T-225's thin 2 px connector with a 1 px rim. Dots are merged the way boxes
  are (all dark, then all white); within a distance group the connectors are a minimum spanning
  tree over the dots' centres (shortest total, no long joins past a nearer dot), and a player's
  link between two groups joins the two linked stations' own dots, so a link still reads as the
  pair the player chose. The connector has the dots' rim and white 35% of an ordinary dot's white
  radius either side (chosen by eye: thick enough to read as the same material, thin enough that
  dots still read as dots); its white goes under the dots' white so a crowd tint stays whole.
  Overlapping dots pick by nearest centre, so each dot gets its own station; the tag lists that
  station's lines, no longer the complex's. Station names keep clear of a transfer dot.
- **A place gets a zone's name and load only within reach of it** (T-318, found in T-307: a London
  game was named "Boston" and showed a "Loading data" card). `cityNear` and `zoneAt` pick the
  nearest zone at any distance, which is right for a lookup but wrong for a name. The view now
  belongs to a zone when it is inside its boundary or within 40 km of it (about the market's onset,
  25-50 km, and the ring round a city T-300 would model), and before the boundaries have loaded
  within 120 km of its centre (past the farthest edge of New York's metro). Farther is dead space:
  no name, "–" for the split, nothing loads (not even the market) and the clock is not held; the
  focus stays on the last zone. One helper, `placeAt` (`game/zones.ts`), answers it and
  `viewedZone` (`game/cities.ts`) holds it, for T-309's local time as well. Panning inside the
  Northeast does not change (the same zone under the view); only a gap wider than 40 km (upstate
  New York, Maine) now has no name.
- **Closest stations and walking minutes are worked out on the main thread** (T-312, Anita: "list
  3 closest stations ... and their walking distance in minutes"). The walking barrier the demand
  model routes on lives in the demand workers' wasm, which is busy during a solve and would need a
  wasm rebuild and a new query, for a hover. Instead `game/walks.ts` fetches the pack's `walk_row`
  and `walk_x` (about 250 KB) and runs a TypeScript copy of `WalkGrid::field` and `Field::eff`, so
  the walks are the model's own and cross no river that it does not. Minutes are the routed metres
  x the model's 1.3 detour at 4.8 km/h, a plain walk (not the 15 km/h fast walk to stations).
  Ranked by walk, same-complex and same-name stations left out, a name once; a cut-off station
  shows "no walk" only to fill the list when fewer than three can be walked to within 10 km. The
  1 and 2 km rings while placing are an SVG layer beside the ghost, not in the renderer.

- **The station ghost says yes or no, and blueprint track is solid** (T-311, Anita: "theres no
  indicator for whether its okay to put a station here"; "the dashedness looks quite confusing as
  the lines curve"). The ghost is always the full box at its true length (it was a dot below zoom
  15) and is blue where a click would place the station, red where it would be refused, the
  reason in the tag. The test is the click's own: `TrackApi.preview_station` runs `add_station`'s
  code as a trial and rolls it back (cash included), asked of the clock worker once per change of
  spot, one in flight, last answer kept meanwhile; a UI-side guess would drift from the model.
  Colours fixed, not the theme accent (blue #2f6fe0, red #e03c3c). Blueprint track lost its dashes
  for the same colour mixed 45% towards white, opaque: plain 55% alpha would double up where
  segments overlap at every joint and show beads on curves, which the dashes had hidden. Lines
  that do not run keep their 55% see-through. `dist` in the stroke buffers (network.ts) is now
  unused by the shader.

- **Tier 0's jobs proxy weights GHSL non-residential volume by height^1.5** (T-287). By volume
  alone Toronto's centre got a third of its jobs (suburban warehouses outweigh towers); weighted by
  each cell's mean non-residential height (volume over surface) to the power 1.5, jobs within 3 km
  of King and Bay come to 383k against the census pack's 354k, and jobs misplaced between
  municipalities fall from 7.4% to 4.0%. Blend 50% that, 30% OSM job places, 20% population (set
  before looking). The power is in-sample on Toronto; London (T-288) checks it. Population from
  GHSL GHS-POP rather than WorldPop: one product family for people and buildings.
- **Tier 0's curve: prefer a borrowed fitted curve to a length fitted to one published figure**
  (T-287). Fitting only the length to Toronto's 10.5 km median commute (2016) gave
  d^-1 exp(-d/95 km): 17.5% of commutes misplaced between municipalities, against 15.7% with the
  US curve and 11.3% with Toronto's own, and 22% fewer riders on a test line. `tier0.fit_curve`
  stays for cities with nothing to borrow; a distance distribution (Mumbai's census bands) should
  fit both parameters.
- **Panels count today so far, not a whole-day forecast** (T-293, Anita: "lets not even show the
  whole day forecast cuz it shows stuff in the future. lets only show so far today, and call it
  'today'"). Replaces the line panel's Demand forecast, Trains today and Finances (riders, fares
  and running cost a day; the account's whole day of trains), the station and complex boardings a
  day, and the Lines and Stations tabs' figures. Counts come from the boarding account's
  departures since midnight (`TrackApi.today`), so they are the trains and platforms the player
  sees; fares from each line's hourly fare as the cash is paid. Lists sort by today's counts
  re-sorted each game hour, except the complex list, kept in forecast order so rows do not jump
  in the early morning. The purpose split shows shares only. The map's demand views and the
  station's commuters and inter-region trips lists stay demand's whole-day picture (Anita: they
  are demand's picture, not the panels' counts). The line panel's longest wait went: a so-far
  version needs the wait per departure, which the account keeps only per hour.

- **Inter-region trips are no purpose in the filter, and the name replaces "longer trips" on screen**
  (T-305, Anita). They show under All purposes only; a filter value of "longer" falls back to all
  (`asFilter`, `game/purposes.ts`). Code identifiers and docs/demand.md keep "longer trips". The
  top bar's cash has a fixed width with the $ in a fixed place; a station's hover tag lists its
  lines by their chips' symbols (each once across its complex) in place of "one dot for the
  complex's N stations". A line the player gave no symbol is listed by its name without a leading mode or operator word
  ("subway 6X" is "6X", "lirr Port Washington Branch" "Port Washington Branch"); its chip keeps
  showing the name's first letter. Lines are de-duplicated on the text shown.

## 2026-10-13

- **A bubble selection's everyday far end visits only the selection's samples and answers sparse**
  (T-285, Anita: a click on a destination took up to ~3 s to show its homes). Replaces weighing
  every destination sample of the city per click (420 ms a Chrome worker in New York) and a dense
  answer of three numbers per cell per purpose (14.6 MB a worker, 80-360 ms to add up on the main
  thread). Same f32 sums in the same order, checked to the bit; a click now shows its far end in
  80-170 ms. Clicks during a re-solve still wait for its final round (7.5-11 s on the real New
  York save): answering them from a kept copy of the shown result is a redesign of the query
  state (notes/T-285.md, T-122) and goes to Anita for review first.
- **Money is paid as trains pick up riders, not once an hour** (T-284, Anita: "money update live
  so that it feels like people are actually paying fares"). Replaces whole-hour settling. Each
  hour's fare is paid by a curve from the boarding account's departures that hour (riders each
  train takes), its running cost evenly, settled four times a wall second; each hour still pays
  its whole fare and cost, so day totals are unchanged (checked on the real New York save to
  1e-12). Per-train timing was cheap (one pass over the hour's platform events), so no plain
  per-second accrual was needed except for hours without an account. The cash is shown to the
  dollar, since at $10M steps the movement was invisible.
- **"Stations full" counts riders left by the last train, not the hour's total** (T-284). The
  hour's total made the list jump as each hour turned; the last train's left-behind riders follow
  the trains. Same meaning, read now instead of per clock hour.
- **Walks and drives follow real paths** (T-275, Anita: "can people currently walk across the
  hudson? seems like a kinda prereq"). Replaces the straight line x 1.3 for both. Walks: a 50 m
  barrier raster per pack (its water mask, less OSM's walkable bridges), searched from each
  station in the game because stations are the player's (cached per station position, so an
  edit costs what it did; opening a big save pays about 1 ms a station). Drives: per-pack zone
  factors from OSM roads (Dijkstra offline, `pipeline/routes.py`), road length only; speeds stay
  the density model's, as Anita asked. Chosen over a coarse walking graph of H3 cells (cells exist
  only where people or jobs are, so parks would be holes) and over per-cell reach tables (stations
  are unknown offline). notes/T-275.md.
- **The routed road factor is scaled by 1.3 / 1.210** (T-275). Shortest-distance routes come to
  1.17-1.26x the straight line (commute-weighted; New York 1.210), against the 1.3 the density
  speeds were fitted with on New York's ACS car times; unscaled, New York's commute rail fell from
  19.7% to 18.4%, mostly in the boroughs, not by water. One scale for every pack, New York's, so a
  pack's roads still differ from New York's; a refit of the speeds would be the alternative.
- **Walking transfers between stations stay straight x 1.3** (T-275, the T-161 question). Every
  population must build the same graph to share strategy searches (T-119); none of the real New
  York network's 80 station pairs within 300 m has water between them; complexes are the
  player's links.
- **Residential streets are in the drive graph** (T-275): without them rural places joined the
  roads kilometres away (Springfield's near factors had a median of 1.66).
- **Toronto: the first pack outside the US, its own adapters** (T-277). The Greater Toronto and
  Hamilton Area by whole census divisions (cheaper than the CMA, which cuts divisions; whole
  divisions are docs/zones.md's unit), built by `pipeline/canada.py` from Statistics Canada's 2021
  census with the substitutions in docs/adding-a-city.md "Outside the US". Outside the regional
  market for now (`market: -1` in the registry) until T-276 places it.
- **Toronto's gravity curve is its own fit**, d^-1.2 exp(-d/20 km), on the 2021 census flows
  between its 28 municipalities (5.6% of commutes misplaced against 15.1% at the US curve). The
  game now takes each pack's shipped curve at opening (`DemandApi::new`); before, a pack whose
  curve differed from the US default would have been re-solved with the US curve. The 2021
  flows are a lockdown month's, likely somewhat more local than normal; the pre-2020 flows are on
  StatCan's blocked server.
- **Lockdown census jobs put back by income, not by office density** (T-277). The census counts
  at-home workers at home; each employment income band's CMA ratio of at-home to usual-place
  workers puts their jobs back at work tracts. Downtown probably stays under its pre-pandemic
  count, but today's commuting (24.7% at home) is plausible, and an OSM office reweight has
  nothing to check it against, so it was not added.
- **Pearson's connecting share 0.75, by judgement** (T-277). Canada has no national source like
  DB1B; 0.75 is the share of the US hubs DB1B measures (Dulles 0.74, BWI 0.66). Other Canadian
  airports 1. Marked as judgement in docs/demand-worldwide.md.
- **One work-from-home share for the whole Toronto pack**, the Labour Force Survey's May 2024
  figure for the Toronto CMA (24.7%), Hamilton and Oshawa included (they had less at home in
  2021, so they get slightly too few commuters back; noted).
- **OSM commercial and industrial land counts as one job place per 5,000 m²** when spreading a
  census tract's jobs over its areas (80% job places, 20% population). Judgement; only the split
  inside a tract depends on it.
- **OSM for a Canadian pack comes from the Geofabrik extract, not Overpass** (T-277): the public
  servers answered 504 for the area all afternoon; the extract (shared with T-275's routes.py) is
  read once with GDAL and written in Overpass's shape, so the shared enrich scripts run unchanged.

- **Traffic widths show the whole day** (T-274, Anita: "lets show demand for the whole day rather
  than the current hour"). A stretch's width is its riders over the day shown, both directions,
  against the busiest stretch-day of either kind of day, still T-227's straight 0.5 to 6.6 line
  widths. Replaces the riders in the period on the clock (T-227), which made a switch of day look
  far bigger than the panels' daily figures at peak hours (T-267) and made widths change through
  the day. Platform crowds, dot tints and train fill still show the hour.
- **The US border factor stays one value per pack** (T-171, Anita: "i feel like the border
  penalty is weird tbh"). Fitted on LODES OD in five packs, against per-state-pair factors, a
  fade near the line, a factor beyond 6 km only, a factor only between metros (GHSL-style urban
  centres from the pack's cells), a water term, a water-crossing (toll) term, water plus the
  factor, and none: only per-pair factors tie it (and only help Washington, which has its own
  0.45). A metro-only factor cannot reach LODES in New York (half its cross-state commuting is
  inside one urban centre), and water or toll terms cannot replace it (untolled land borders
  show the same cut, and a penalty strong enough to match crushes in-state flows across the same
  rivers). It was not PATH's problem: most of that was realnet's way of counting operators
  (T-269); the rest is mainly the cheap drive across the Hudson (T-270). The fade near the line
  (T-273) is left to Anita.
- **Riders left behind cost a wait in the next crowding round** (T-188, T-164's phase 2, Anita:
  yes). Each round runs the boarding account's rule per period on the period's own trains (even
  headways, the account's arrival shapes, crush loads, separate queues per line) from every
  population's averaged loads, and adds each boarding's average extra wait (capped at 60 minutes)
  at the platform wait's weight. Inside the solve rather than read back from the clock worker's
  account: feeding the account back needs a second and third solve after every solve, a new game
  would open with no account, and per round it settles with the crowding by the same averaging
  with no extra damping. Per period, not per hour (the account's hours are not mapped). Weekday
  only: the weekend keeps the weekday's route and mode choices (T-260) and so its waits. Real New
  York save: rail -0.1%, evening-peak riders left behind 40.6k to 30.4k, LIRR Hempstead -4.2%,
  no busiest line over 0.8% (notes/T-188.md).
- **The demand views show the clock's day by default** (T-266, Anita: "that should actually go in
  the settings ... either weekday, weekend, or show whatever current day is (default)"). The top
  bar's Weekday / Weekend buttons are gone; Settings has "Demand shown for": Today, Weekdays,
  Weekends. It is remembered with the other display settings now that the default follows the
  clock (T-260 left the buttons unremembered so nobody returned to a silently weekend map; with
  Today as the default a remembered choice is one the player made). No other mark of the day
  shown: the clock says the day, and the split's hover says weekdays or weekends.
- **Traffic widths stay as they are** (T-267, Anita asked whether the white border makes widths
  grow faster than riders). It does not: the coloured stroke is linear in the riders an hour
  (half a line at none, T-227) and the casing is a fixed margin drawn under it. What she saw is
  that the width shows the hour on the clock: at 8:00 a Saturday has under 40% of a Monday's
  riders on the busiest lines while the day's riders in the panels fall only to about 77%; at
  midday a Saturday is the busier. The busiest-stretch scale is the same on both days.
- **Weekends are worked out from the weekday's solve** (T-260, from T-170's brainstorm). Two kinds
  of day, weekday and weekend: Anita asked for a weekday/weekend selector, Saturday and Sunday
  share the main change (work down to about 0.3, leisure up about 1.6 in NHTS 2017 and England's
  NTS), and a third kind can come later without new plumbing (T-262). Game day 1, a new game's
  first, is a Monday. A weekend is each purpose's weekday riders per period times a factor from
  NHTS 2017 (TSA for airports), not a second solve: without crowding a trip's choices depend on its
  period's service only, so it costs no solve, no opening time, and weekday results stay
  bit-identical. The price is that weekends keep the weekday's crowded route and mode choices.
  Weekend timing is the model's weekday shares shifted by NHTS's weekend/weekday period ratios,
  not NHTS's own shares, so the weekday's judgement carries over. The same trains run every day
  (Anita, T-170: no separate weekend schedules for now); a weekend service level would need its
  own solve and is open (T-261). The top bar's Weekday / Weekend buttons choose what the demand
  views show and start on Weekday at each page load, like the purpose filter; the trains, platform
  crowds and fares follow the clock's own day through one boarding account per kind of day. Line
  widths are scaled to the busier day's busiest stretch so a weekend reads as quieter. T-170's
  other ideas became T-261 to T-265.
- **Doc tidy after batches 17-19** (T-259). SPEC 8 is a short summary again (it had grown to 90
  lines by appending; docs/interface.md has the detail); SPEC 9 keeps only the latest calibration
  figures (T-198), earlier checkpoints stay in notes/T-090.md and notes/T-162.md. TASKS.md is
  grouped by area instead of by batch, with performance tasks in one parked section. T-142 drops
  from P2 to P3: its nine-city work is done, and its other-country adapters wait for city
  additions, which are paused. Notes of finished tasks carry an "Archived" line.
- **A split inside a curve fits its halves to the node** (T-258). The new PIs either side of the
  node are on the millimetre grid. A few metres from the node their turn angle is measurably
  off, so the arc's radius overran the short leg and the station was refused. Each now takes the
  radius whose curve ends exactly at the node. When that differs from the arc's by more than 1%,
  which only happens within a few metres of the curve's end, the split moves to the curve's end
  instead. A station is never refused for this, and the track keeps its shape to a centimetre.
  The alternative, dropping the grid for these PIs, would have made saves and undo depend on
  float noise.
- **A platform follows its track's own line, sampled at 13 points** (T-254, Anita: "ideally we
  actually fit the track"). T-189's circular arc through the platform's middle and ends cannot
  bend both ways, so on an S-curve (110 St-Malcolm X Plaza, Wall St, Grand Central's Lexington Av
  platforms) the box was a near-flat rectangle with the track snaking across the blue edges.
  Now the clock worker samples the track along each platform (`PlatformShape.pts`, network.ts
  `stationLine`) and every consumer measures position along and across that polyline
  (`polyFrame`, the same arithmetic in GLSL and in TS) instead of the arc. A polyline rather than
  a smoother curve: the distance to it is exact and cheap in a fragment shader, and straight track
  gives exactly the old box. 13 points, not 9: at 9 the outline showed visible facets on
  34 St-Hudson Yards' curve at zoom 16.5 (25 m stretches turning about 10 degrees); 13 points are
  12-17 m stretches. The points ride to the fragment shader as flat varyings (7 vec4) and to the
  selection outline as texels (10 a shape, so at most 204 selected shapes share an outline, down
  from 512; Penn Station is 24). Past the end of an edge shorter than half the platform the line
  carries straight on, as the arc's chord did. Cost on Midtown at zoom 16, clock at speed 2,
  trains on: GPU about 1.55 ms a frame against 1.17 before, CPU 0.46 ms against 0.40, frame gaps
  16.7 ms either way.
- **Quad track is two double-track edges side by side, and Express copy moves stops to twins**
  (T-255, Anita: "easy express line construction / setup... maybe add quad track drawing"). A
  4-track edge kind would have touched every part of the model (capacity per track, platforms per
  track, which track a line is on, picking, drawing); two ordinary double edges reuse all of it,
  and "which pair a line uses" is just its path. The express pair is 8 m to the left of the drawn
  direction (4 m between neighbouring tracks, like double track; wider would read better at
  zoom 15 but nothing real is laid that way), concentric at corners, and starts and ends at the
  stroke's own end nodes after a 150 m swing-out, so lines change pairs at the ends of each quad
  stretch with no extra junction nodes to design. Mid-stretch crossovers are not drawn (the
  player can end one quad stroke and start another to get a crossing point; a crossover between
  tracks 8 m apart cannot be drawn by hand, the curves are too tight). A station on quad track is
  two stations of one name 8 m apart, one per pair (a complex, so riders change across the
  platform on foot), placed together; that keeps the one-node station model rather than a
  four-track station kind. Express copy is the smallest UI for an express line: one button beside
  Copy line, moving each stop to its twin within 12 m, then straight into stop editing so the
  skips are clicks on the map; no naming step (it is "<line> express"). Placing a station within
  a platform's reach of a T-244 joint now joins the two pieces first (it used to be refused as
  "platform too long", a T-244 regression this found).
- **Capacity tags let tools through; tabs keep their own scroll** (T-224, T-215). With any tool
  on, the tags take no pointer events at all (a CSS class from the tool signal) rather than
  forwarding clicks as T-221 did for the trip tool: the track tool and node drags also need the
  moves, presses and double clicks under a tag, and forwarding each kind would copy the map's own
  handling. The tags' hovers are off meanwhile. Each tab remembers its own scroll for the session
  rather than always opening at its top, so going back to a long list (Lines, Stations) returns
  where you were; the first visit opens at the top, and nothing carries over between tabs.

- **Saved games move to their own panel** (T-251, Anita: "i dont like the saves being at the top
  of the settings ... a save button to open a floating center saves panel"). Replaces T-241's
  placement at the top of Settings. The button is a floppy disk at the right end of the dock's tab
  row, beside the gear, rather than in the top bar: the top bar's narrow layouts are a fixed grid
  with no free cell, and the tab row already holds the other "open a pane" buttons. The panel is a
  native modal dialog like the confirmation's (focus kept in it, Esc, a close button, a click on
  the dimmed page closes it); Open, Load a file and Start a new game close it too, so the loading
  card shows. The cards get a 192 x 128 picture (drawn at 384 x 256 for sharp screens) and the
  station count. The note on how saving works is gone (Anita), and Settings' "Back to 100%".
- **Dock tabs are icons** (T-252, Anita): Build a wrench, Lines a route with four stops, Stations
  a platform under its canopy (a house read as homes), Money a bill with a coin; names in the
  game's hover and as the buttons' accessible names. Fixed-width tabs on the left, saves and
  settings on the right, rather than stretched icon buttons.
- **Panels drop one-time explanations** (T-253, Anita: "most of the info we display should be
  stuff that can change"; CLAUDE.md "Panels show what can change"). Notes on how a figure is
  made, what a purpose covers or what a button will do moved to the hover of the label they
  explain; instructions the map's tool hint already gives were dropped; empty states and
  Planned/Broken/Blueprint lines kept to the state. Every sentence moved or dropped is listed in
  notes/T-253.md. The track, delete and trip tool hints are shorter.
- **A click on a station always selects its complex; the Station / Complex switch is gone**
  (T-249, Anita: "in general we should just pick the complex"; T-246's rows already reach a
  station alone). Complex view no longer reads `display.complexView`, which stays only as the
  demand circles' "One circle per station complex". The switch would only have toggled between
  the complex and the one station already reachable from the rows, so it went (Anita wants minimal
  UI); a station opened alone shows "Complex of N stations" as a link back in its subtitle, and any
  new selection object clears the alone state, so clicking the same station on the map again
  shows the complex too. Search, the stations-full button and the Stations tab list all select a
  station and so now show its complex as well.
- **The complex view lists lines, not stations** (T-250, Anita: "one row per line, not one row per
  line-direction"; "say what line it is in the tooltip"). The real New York save gives every
  service pattern its own station and line ("subway 6" twice, to Parkchester and to Pelham Bay
  Park; "mnr Hudson" twice), and every chip reads S, M or L, so the station rows all had the same
  name and near-identical chips. Rows are now one per line name with its patterns' platforms,
  waiting riders and boardings summed, labelled with the line's name, the hover giving the station
  names and where its trains go; a click opens the busiest of its stations alone. The line list
  under the complex map was the same lines a second time and went. The platform strip's hover
  sums per line name too. The note T-246 put under the complex map went (panels show what can
  change).
- **Platforms are blue edge lines and the track runs through the boxes** (T-236, Anita: "show the
  actual platforms as like little blue lines. and maybe the track on top of the station"; T-226's
  grey hairlines "are confusing"). Inside each box, over the crowd fill: the track and line strokes
  drawn a second time, clipped by a stencil to the stations' white (the overlay canvas now asks
  for a stencil), then each platform's two long edges in blue. The stencil, not a plain second
  pass of the strokes, because a second pass everywhere would double the alpha of blueprint,
  dimmed (stop editing) and far-out faint strokes; the white is marked per pass (1 unselected, 2
  selected), so the track is drawn once inside each. The track is capped at 30% of the box's white
  and has no casing: a 6.6 px line (44 px with traffic thickness) would otherwise fill a 6-16 px
  white and hide the crowd, and the casing would cut the outline at the box ends. Blue edges along
  the rim rather than beside the track: the brief said "along its edge", they show each platform
  of a complex where boxes overlap (what the hairlines tried to do), and the crowd fills the space
  between them; they fade in from zoom 15 to 15.5, since at 15 the white is about 6 px and two
  blue lines plus the track read as hatching. Blue #8085f0, a periwinkle: the palette's blue
  (#1f6fd1) and light blue (#5cc3f0) leave no plain blue far from both, and this is 15 CIEDE2000
  from each and 38 from white (any bluer choice was 6-12 from one of them).
- **The complex view lists its stations first, live** (T-246). The member list moves above the
  complex map, with riders waiting now beside boardings a day (header row "Waiting", "A day").
  A click opens that station alone through a per-selection override (`singleStation`) rather than
  flipping the remembered Station / Complex setting, which is shared with the demand circles and
  would make every later complex open as one station. Platform crowds and trains come from the
  renderer's own hour of the boarding account and trip buffer, so the panel and the map always
  agree; they are drawn as a 5 px platform strip after each station's lines in the complex map
  (the map's grey/red fill from the middle out, each direction's half) and train pills on the
  line's band, standing ones solid, ones stopping within 60 game seconds paler and sliding in from
  the arriving end. Inside the platform rather than past its end, because the map is fitted with
  23 px of margin and a train outside the platform left the drawing. Worked out every 10 game
  seconds (a 150 ms real-time check), so at 60x every train is seen standing (dwells are 20-40 s)
  and approaching, and the panel re-renders only when a shown number or train changes.
- **Zoomed out, a linked complex is a dot per distance group and a connector** (T-225, Anita:
  "yes, connector"). Dots group by the complex rule's distances alone (`dotGroups`), so links never
  make a dot larger; each link between two groups is a thin white line with a dark rim between the
  two dots (drawn with the bridge program, 2 px white and 1 px rims, a fixed width on screen), under
  the dots, selected with them and part of the selection's outline and the bubbles' cut-out. Each
  dot stands for its own group's busiest station, not the whole complex's: the line and delete
  tools then pick a station near the dot clicked, not one up to 1 km away at the other end of a
  link; selection still covers the complex in complex view. The tag says "One dot for 16 of the
  complex's 23 stations".
- **A drawn route is one piece of track per corner** (T-244, Anita: "if we delete it it deletes
  the whole curve"). The cut is at the middle of each leg between two consecutive corners, not at
  the corners themselves. A cut at the curve's middle near a click would have replaced the click's
  PI by two half-curve PIs with set radii (Network::split_plan inside an arc), moved the ramps
  between neighbouring vertices, and made dragging that corner a node drag; at the leg middle the
  point is always on the straight, every click stays a PI of its own piece with the same auto
  radius (the whole-leg rule next to an end gives back the half leg it had), and the geometry, cost
  and run times equal the unsplit edge (Rust tests compare them). So a route of n corners is n
  pieces, one edge for none or one; "roughly one per click" as asked. Legs that change level and
  legs under 40 m are not cut (a ramp is centred between corners, a node sits at a whole level).
  A reshape moves the joint next to a dragged corner along with it, keeping its fraction of the
  way between the corners either side, because otherwise the joint's fixed heading would refuse
  almost every drag of a corner that dragged freely before. Pieces are not merged back, and the
  track inspector's Construct now builds one piece; Construct all and per line are unchanged.
- **Stop editing tiers come from the clock worker** (T-237, Anita: stations past the end of the
  line show as unreachable). The JS test put anything off the edges listing the line in the grey
  "changes the route" tier, so a station past a terminal looked like a detour though adding it
  only extends the path. `TrackApi::stop_tiers` now tries every station with `insert_stop`'s own
  search (hops memoised, a turn-valid reachability pass first) and calls it "route" when the new
  path contains the old one: on it, or past either end. The markers and a click cannot disagree,
  which also corrects connected stations the line could only reach by turning back (now "cannot
  be added", where the old test said "reroute"). 0-8 ms a line on the real New York save; the
  graph estimate shows until the answer lands.
- **Saves are cards at the top of Settings** (T-241, Anita: "do autosaves work? how can we load
  from an autosave? isn't obvious to me"). Autosave already worked (every minute if changed, after
  a paid edit, on hiding or leaving the page; checked end to end), so the change is in how the
  saves read. They stay in the Settings pane, now first in it (gear tab hover "Saved games and
  settings"), rather than a sixth tab: the tab row is full at the dock's narrowest and a saves list
  is visited rarely. Each slot is a card: a picture of its network, name, "Autosaved 2 min ago" on
  the game being played (others "Saved ..."), day, cities, lines and cash, the game being played
  marked Playing, an Open button on every other. The picture is a 192 x 128 PNG drawn in the clock
  worker on an OffscreenCanvas each time the slot is written, stored in the IndexedDB record with
  the counts (about 11 KB for the real New York network), not computed from the save bytes on
  demand: opening a save means decoding and loading the network, too slow for a list. Old slots
  show "No picture yet" until next saved (no migration, CLAUDE.md). Cities are the registry city
  nearest each station, most stations first.
- **The loading bar moves with time** (T-243, Anita: "show a bar filling, not a % number"; "we
  seem to freeze at 60% for like the majority of the loading time"). Measured: on the real New
  York save opening the packs is 30% of the load and the first estimate 64%, and both report all
  at once at their end (every worker finishes opening together; every period lands within a few
  hundred ms), so the old 10/50/40 percentage sat at 32% for 2 s and 60% for 3 s. Each step now
  weighs its expected time (packs by size, the first estimate by line stops, scaled by this
  browser's last startup), and inside a step the bar moves with elapsed time against that,
  slowing near the step's end, until the real completion lands; if an expectation grows the
  remaining work is spread over the rest of the bar so it never stands still or goes back.
  Real sub-progress from inside the solve would need the Rust loop to report (a WASM change); the
  estimate was enough. The percentage moved to the bar's hover. notes/T-243.md.
- **A UI scale setting zooms the panels, not the map** (T-247, Anita: "the fonts are a bit big
  tbh right now for me"). Panel and text size in Settings, 80-120% in 5% steps, default 100%,
  remembered per browser with the display settings. Every size in style.css is px, so the panels
  (dock, top bar, map buttons, search, loading card, tool notes and dialogs) get CSS `zoom` from
  one `--ui` variable rather than a rem rewrite of the whole stylesheet; the map's DOM tags only
  scale their font size, since their positions come from the map and a zoomed tag would move off
  its anchor. The dock's drag handles divide by the scale. A stepper, not a slider: the dock
  resizes under the pointer as the scale changes, which would make a slider jump away.
- **A selected train gets a halo and a ring** (T-242, Anita: "highlight the train in an obvious
  way"). The selected train is drawn last, over the others, inside a halo round its whole consist
  as one pill, so its own colour, fill gauge and red packed edge stay as they are: the selected
  station's dark outline (T-189), a white band, a dark ring of the train's shape 6 px out, and a
  soft glow. The station's outline and glow alone were tried first and barely showed: the glow is
  faint over a coloured line, and at zoom 11-13 the whole train is a few px; the ring reads at
  every zoom. The train is not drawn larger: its size encodes nothing, but a bigger pill would cover its neighbours and the station
  it stands at. A click tries the selected train first, since it is drawn on top.
- **The track model names each train's next trip** (T-245, Anita: keep following a train that
  turns back). `hand_over` already decided which departure a finished train takes; the trip now
  carries that departure (`Trip::next`, sent as a fourth number per trip), and the selection moves
  to it once the clock passes the old trip's end. Chosen over matching "a departure of the other
  run at this trip's end time" in the UI because that guess goes wrong exactly when trains are
  removed: a train leaving service can be ready within the hand-over tolerance of a departure
  another train takes. A train leaving service keeps the selection, as before, with "This train
  finished its trip and left service": deselecting would throw away the panel the player was
  reading. There is no camera follow mode to carry along.
- **"Waiting now" changes when a train leaves the platform** (T-238, Anita: update "the moment a
  train pulls in"). The account boards a train at its departure (the hour's events), and the map's
  platform fill already followed it to the game second; the panel re-rendered once a game minute,
  up to a second late at the slowest speed. It now also sets a timer for the station's next
  departure. The drop stays at departure rather than arrival: the dwell is 30 s by default (half a
  real second at the slowest speed), the account has no arrival time per stop, and at a terminal
  a train can stand for minutes before it boards in the account, so moving the drop to arrival
  would take a new per-stop time in every hour's slice for a half-second difference, and the map's
  fill and the panel would have to move together.

- **Station boxes are a fixed, narrower width; traffic lines may exceed them** (T-239, Anita:
  "a bit too fat still ... the lines can get thicker than the stations, that's not necessarily a
  problem"). A box is 75% of what a plain line and its rim cover at zoom 15, for every station
  in every mode, replacing T-204's cover-the-widest-line rule and T-227's growth for traffic
  widths. Dots follow (no growth either), so a dot and a box stay consistent. A wide line
  running under a station now shows beyond its rim. Stop editing markers lose the check, and
  reachable stations are drawn smaller than stops so the line's own stops stand out.

- **No homes in institutions** (T-234, Anita: "why are there homes on rikers island?"). The
  census counts people where they are held or cared for, so Rikers Island had 3,772 residents and
  Homes bubbles. Every pack's block population is now the 2020 census figure less its
  institutional group quarters (PL 94-171 table P5, per block: adult correctional, juvenile,
  nursing/skilled-nursing, other institutional), from each state's summary file on
  www2.census.gov: a national source every state has, no key (the Census API refused keyless
  requests). All four institutional types go: inmates make no trips, and nursing-home residents
  rarely leave and are not commuters, shoppers or pupils (nursing homes are the larger part, 0.5%
  of all people against 0.2% in prisons and jails). Non-institutional group quarters stay: college dorms
  (students travel, and the university purpose's adult homes need them), military quarters,
  shelters and group homes. LODES RAC also puts job holders on these blocks (Rikers: 676 on
  blocks where 27 people live outside the jails; likely address records and LODES's noise), so on
  a block with an institution the job holders are capped at the residents left; other blocks keep
  LODES as it is (blocks with more RAC than people exist everywhere and are LODES's business). The
  packs were changed in place (`pipeline/group_quarters.py`): `pop` and `commute_home` re-spread
  from the block table exactly as `build_city.py` spreads them, school and university homes
  rebuilt, gravity re-solved, 56 cells left with no people and no jobs dropped (as a build drops
  them); every other array byte for byte. The park draw's nearby-residents term was left for
  T-235: it needs a fresh Overpass fetch, and the public servers answered 504 for ten minutes running. Not
  done: prisons' staff are LODES jobs and stay; the census's household population on the same
  blocks (27 on Rikers, DP noise or staff housing) stays. notes/T-234.md.
- **Stop editing mode** (T-233, Anita: "when editing station stops we should display very
  prominently on the map where the current stops are. and hide most other stuff ... each station
  should have a top level bubble that we can click on map to either add it or remove it (toggle)
  ... stations that are reachable on top, and stations that are not reachable behind and like
  slightly greyer"). Stop editing is the line tool with a line (`stopEditing` in game/ui.ts), entered
  as before from Edit stops, New line or Copy line and left by Done, Enter or Escape. While on: the
  demand canvas, trip canvas, capacity tags and longer-trip labels are hidden by a page class
  (style.css); line widths and station circles by riders are not built (demandViews.ts), so lines
  draw at plain width; other lines are drawn at 30% (one renderer uniform, `dimOthers`). Every
  station gets a round marker (map/stopMarkers.ts): DOM over the map with no pointer events, picked
  by the map's own click and mousemove so the wheel and dragging still work under it. Reachable is
  graph-based on the main thread: **route** = the station's node is on an edge the line's path uses
  (full size, on top), **reroute** = joined to the stops by track but off the route (smaller,
  grey), **cut** = on no track network the stops lie on (smallest, greyest). Reroute stays
  clickable because the existing tool already reroutes: the worker's `insertStop` tries every
  slot and takes the shortest valid route, so the tag says "Click to add a stop and change the
  route". A click on a cut marker only shows why, in its own tag for 4 s. A line with fewer than
  two stops has no route to stay on, so what joins its stops counts as route; with none, every
  station does. Removing a stop is `setStops` without it, as the inspector's red crosses do (the
  route between its neighbours is the worker's, normally the same track). A click on a station's
  box with no marker under it toggles too, where it used to say "already a stop". The panel at the
  map's lower left shows "Editing stops on <line>: N stops" and a Done button. The other lines'
  trains fade to the same 30%, fill gauge included (the train shader takes `u_sel_line` and `u_dim`;
  Anita via the orchestrator: "hide most other stuff"); the edited line's trains stay at full strength.
- **Faint internal borders inside a complex's shape** (T-226, Anita: "bring back the internal
  borders of stations as like quite faint gray lines"). After the whites, each platform's own edge
  is drawn as a 1 px hairline along the middle of where its rim was, multiplied into what is
  drawn (white becomes about #d6d6d6). Multiplying rather than painting grey means the same
  hairline on the outer outline only darkens the dark, so the outline needs no special case; crowd
  fills go over it. Platforms only (bridges and dots have none).
- **Traffic thickness linear, 0.5 to 6.6 line widths** (T-227, Anita: "raise the max width by like
  3x ... the scaling should be linear. the most used line should be a fixed width like 3x what it
  is now and the least used lines should be like 0.5x normal width"). Replaces T-084's square
  root from 0.45 to 2.2 at a fixed 30,000 an hour. "Fixed width for the most used" is taken as
  relative: the network's busiest stretch in its busiest hour of the day (every period, the
  purpose shown) is 6.6 wide, so any city or player network gets the full range, and the widths
  still fall off in quieter periods. Station boxes and dots grow to cover the wider lines, as
  before. Picking and the track tag reach half a line's drawn width where that is over the 7 px
  slop, so a click anywhere on a wide line hits it.
- **Crowds on dots** (T-228, Anita: "it doesn't appear for when we are zoomed out and the station
  is a circle though, we should also highlight the circle slightly"). From zoom 11 to 15 the dot's
  white is tinted: towards the crowd grey by its fullest platform side's share of the room (at
  most 45%), or pale red (40% towards the over-full red) while a train at any of its platforms left
  riders behind; a complex's dot takes the whole complex. A tint rather than a fill, as asked, and
  because a dot is 4-15 px across. Its tag gives the riders waiting at it and the fullest platform.

- **After an edit only a final result replaces a final result** (T-232, Anita: "when i edit a
  line minorly the ridership goes up from 19% to 20.5% then back to 19% as it recalcs"). The
  demand client still receives every round, but a re-solve after an edit publishes only its last
  crowding round: until then every figure (top bar split, line and station riders, bubbles and
  their catch-up targets, the boarding account) keeps the last finished forecast, faded with
  "Updating demand 2/4…". Real NYC save, subway 1 cut to 2 trains: the split read 19.0% through
  the whole update (about 11 s, four steps) and then 18.8%, one change. Estimates are still shown
  where there is nothing final to hold, marked rather than held: at startup and after a
  whole-game load (T-172 starts the clock at the first estimate, and holding them would leave
  the panels empty with the clock running for another 7-8 s on New York), and for a city just
  opened (its controls would otherwise wait for the final round). The top bar then says
  "Estimate, refining 2/4…", with a hover that the figures settle when the forecast is finished;
  the other panels are not marked one by one, since they all show the same solve and the top
  bar's status is the game's one place for it. Cost: a query asked during an update (a station's
  catchment, a bubble selection, a trip) waits for the final round, because the workers drop the
  shown solve's state when the new one starts; keeping it would double their memory.
- **"N stations full"** (T-231, Anita: "a global alert in the top left under the demand buttons
  that says like n stations full"). A station complex counts once when, in the game hour now, a
  platform side holds as many riders as fit (`platformRoom`, the T-219 fill's own room, the whole
  platform's where trains use one side) or a full train has left riders behind this hour (the
  figure the station inspector's "Waiting now" shows, so the two agree). Left behind counts the
  whole hour's account, not only up to the minute, for that agreement. Ordered by riders left
  behind, then waiting; a click selects the next complex's busiest member and moves the map
  there at zoom 15 or closer, where platform crowds are drawn. A red-outlined button, not a
  banner: it is a count to click through, and it hides at zero.
- **Station complexes drawn as one shape** (T-217, Anita: "all the white rectangles overlapping
  seamlessly ... and then one continuous black outline surrounding the whole white shape ... and if
  we manually link, this white shape should include a thin bridge"). Every platform's dark shape
  is drawn, then every platform's white inside its rim, so where platforms overlap the rims inside
  vanish; this works for any overlap without knowing complexes (overlapping drawn boxes are always
  within the complex rule's 50 m anyway). A faded blueprint draws only its rim so its see-through
  white is unchanged. The player's links get a bridge: 3 px of white at zoom 15, fixed on the
  ground like the boxes, between the nearest points of the two platforms' centre lines kept 6 px
  inside each end (from the end the rim would cut across it). Only links get bridges: members the
  50 m rule groups but that do not touch stay apart, as Anita asked only for links. The selection's
  outline and glow are one draw over the union (each pixel takes the nearest selected shape's
  distance, shapes in a small texture), since one glow per platform doubled where they overlap.
  Picking stays per platform; bridges are not picked.
- **A complex is one dot zoomed out** (T-217, Anita via the orchestrator: "a complex should always
  be one dot ... picking selects the busiest member", generally on, no setting). Below zoom 15 the
  complex is drawn by its busiest station (boardings plus alightings a day; its first station
  before demand) as one dot at the centre of the smallest circle round its platforms' middles,
  large enough to cover them all, so the lines run into it. The smallest circle rather than the
  mean: the mean leans to the side with more platforms, and a linked pair with 16 platforms at one
  end got a dot nearly twice as wide as needed. A widely linked complex gets a big dot (a
  complex spread over 600 m is a dot 600 m across, 40-80 px at zoom 13-14); accepted, as it is
  what "covering its members" means. The line tool, delete tool and clicks zoomed out pick the
  busiest station too; the dot's tag names it and says "One dot for the complex's N stations".
- **Packed trains' red edge 50% thicker** (T-218, Anita: "not super noticeable"): 3.9 px (was 2.6;
  the white rim is 1.7). Far out, where a pill is under 8 px wide, a packed train is now almost all
  red, which reads as packed.
- **Platform crowds drawn as the platform filling up** (T-219, Anita: "visibly represent crowding
  on platforms as like people filling up the platform, sort of like how darkened space represents
  people filling up trains"). Each direction's half of the box fills with dark warm grey from the
  middle outwards to its riders' share of the room to stand on that side, red while the last
  train there left anyone behind; a platform with trains on one side only fills as one. Room: 2.5
  m usable width a side (half a 6 m island less the edge strip) at 0.3 m² a rider, the roomy end of
  Fruin's queuing level of service E (0.2-0.3 m², people standing in contact; the TCQSM's platform
  scale), about a train's standing density at crush, so a full platform and a full train mean the
  same crush; 1,250 a side for 150 m. From the middle because a platform, unlike a train, has no back
  to fill from, and a fill growing both ways stays centred on the station's label and node. The side bands (T-164, T-199) and "Crowds as dots" are gone: the bands lay over the
  lines beside the platform and needed their own hover reach, and a dot pattern inside a 4-10 px
  half box did not read; Settings has one checkbox, "How full each platform is". A train standing
  in the box hides the fill's middle at zoom 15 (the train is 9 px of a 12.6 px box); from 16 the
  sides show. Only the drawing uses the room: the account still lets a crowd grow past it.

- **The City tab is gone** (T-220, Anita: "now that we have these city bubbles we can get rid of
  the city panel entirely. its kinda redundant"). What it offered and where each went: (1) the
  focused city's name: the top bar already shows it; (2) its intercity total and the top five
  connections with train/other bars: the city overview's bubbles (zoom 8.5 and below) and the
  city inspector, which lists every place with all-mode and train counts; the bars were dropped,
  since the inspector has the numbers and the overview colours each city by its train share;
  (3) City centre: a "Go to <city>" button in the city inspector (the same jump as a city picked
  in search, which also clears the selection so the city's own bubbles show) and search itself;
  (4) Whole region: dropped, zooming out shows every city as one bubble; (5) Explore a trip: a
  button at the bottom of Settings (Anita's suggestion), key 4 unchanged; (6) its "Calculating
  demand" text: the city inspector has its own. Nothing else depended on the tab: no key opened
  it, no selection flow switched to it. A browser that last had it open starts on Lines. The
  notes/*.mjs checks that clicked it (t113, t116, t116-real, t134, t135, t137) go to cities by
  search instead, and t116's Whole region is a fitBounds.
- **A trip's end at a clicked station is the station's own cell** (T-221, Anita: "we should be
  able to explore a trip by clicking two stations"). With the trip tool on, a click on a station
  takes that station as the end: the trip is asked about at the station's position, which the
  model reads as the nearest cell. The model's subzones are the cells of a zone that share their
  nearest station, so the station's cell is in the subzone whose nearest station is this one: its
  walks are the shortest to it and the trip mostly boards there. The busiest cell of its walk
  catchment was the other option; it can lie nearer another station, so the trip would mostly
  start from that one. No worker change: the query already takes points. The model still spreads
  riders over nearby stations (Jay St-MetroTech to Fulton St boards mostly at Borough Hall, 1 min
  away), which the inspector shows as it does for any trip. A station's inspector has "Trip from
  here" (the tool on, the station as the start), since starting from a selected station is the
  natural way in. A capacity tag sitting on a station passes the trip tool's click through to the
  map (it swallowed it before).
- **Longer-trip bubbles are always drawn** (T-222, Anita: "we always want that on"). The Settings
  check "Bubbles outside the selected city" and its remembered value are gone; the bubbles still
  give way under a local purpose filter, since the regional market has no purposes.
- **Five more default line colours** (T-223, Anita: gray, yellow-green, pink, magenta, light
  blue): gray #74797e, yellow-green #9ccb19, pink #f59dc6, magenta #e62aa6, light blue #5cc3f0,
  after the first eight. CIEDE2000: the closest pair in the thirteen is purple and magenta at 19.1
  (the first eight's closest, orange and yellow, is 20.1); each new one against the old eight at
  least 19.1, against each other at least 21.2. Yellow-green, pink and light blue take dark
  letters on their chips (`inkOn`, contrast about 7); gray 4.4 and magenta 4.0 with white, better
  than the orange's 2.5. Against the mode colours at least 17.9 (yellow-green and walk green). Some
  sit near purpose colours, which only appear in panel bars with their names beside them: pink
  4.9 from medical, gray 9.2 from longer trips, yellow-green 14.9 from airport (the teal already
  sat 5.6 from school). Magenta was kept off the purple (a first try, #b5179e, was 10.5 from it)
  and 15.6 from a selected station's catchment colour. Rejected: a deeper pink with white letters
  (#e8559a), which came within 15 of the magenta.

## 2026-10-12

- **The trip explorer draws rides along the track, and opens from a far-end bubble** (T-208,
  Anita: the path should follow "the rail paths rather than being straight lines between
  stations"; "click a home bubble and then click one of the destinations, it should route between
  them"). A ride is the line's own drawn strokes over the stop-to-stop stretches it rides (stop
  indices carried from the solve's route nodes), and falls back to straight lines only when the
  drawn network's ends no longer sit on the stations (an edit since the solve). Walks stay dashed
  and straight: the model's walks are straight-line distances, so a street path would claim
  more than it knows. From a selection, each side is one cell, the busiest: the selection's
  cell with the most legs at its end, the far bubble's cell with the most of the selection's
  trips (a station's far zone: its cell with the most rail legs at that end, since the riders'
  own cells in the zone are not on the main thread). The bubble's drawn centre was the other
  option, but as a weighted mean it can sit in a park or the river with nobody, giving an
  "outside the city" or unrepresentative answer. Longer-trip (regional) far ends do not open it:
  the explorer reads one city's cells. The inspector keeps a Back button to the selection.
- **Complexes are transfer points in demand, as walk edges between members** (T-161, Anita:
  "complexes and manual complexes should allow transfer"). Replaces "complexes stay
  display-only". A rider leaving a train may walk to any other demand station of its complex
  however far, besides the walks within 300 m; the clock worker sends each constructed station's
  complex with the snapshot (`game/complexes.ts` over constructed stations, links included) and
  each demand API takes them with `set_complexes`. The walk costs what every walking transfer
  costs (centre to centre x 1.3 at 1.2 m/s, weight 1.39, then the boarding's wait and 5-minute
  penalty), with no extra fixed penalty, because the 5 minutes already fall on every change and a
  second constant would make a change between two platforms 120 m apart jump in cost the moment
  they became a complex. Members stay separate demand stations joined by edges rather than one
  merged station: merging would put a complex's catchment at one mean point, make a 1 km link a
  free teleport, and lose the platform-to-platform transfer counts the complex map draws (T-176).
  The 100 m merge and the 300 m walks stay as they were. Measured on the real New York save:
  nothing moves (every one of its 855 complexes is one demand station or a few within 300 m of
  each other, since the merge is single linkage), so it matters for platforms more than 300 m
  apart and for the player's links (notes/T-161.md).
- **The player links stations into a complex by hand; only those links can be undone** (T-197,
  Anita: "be able to create interchanges, linking stations into a complex even if they wouldnt
  normally"). From the station inspector ("Link to another station", then a click on the other
  station on the map); up to 1 km centre to centre (real interchanges by passage are a few hundred
  metres; 1 km keeps a complex a place and its transfer walk under 25 perceived minutes). The
  links are part of the network (`Network::links`), saved as an optional trailing block (`LNK1`)
  written only when there are any, so older saves load and a save without links is unchanged.
  They are a setting outside the undo history, like fares, which keeps them off the track
  model's edit operations; removing a station drops its links. Stations the distance rule groups
  cannot be split: their platforms nearly touch, those within 100 m are one station for demand
  anyway, and an "unlink" list of exceptions would be a second thing to keep in step for no
  case anyone has asked for.
- **School and university are one shown purpose; the model keeps two** (T-192, Anita: "we should
  probably merge school and university demand"). The filter, pie, purpose split, timing charts and
  every tooltip show "School and university" (the longer name over "School", so nobody wonders
  where students went), in school's blue-green; `game/purposes.ts` `membersOf` lists which model
  purposes a shown one counts, and the demand client sums them for the map. The model keeps them
  apart because their homes (children against adults), places and timing differ, and merging
  there would change the solve. University's amber is retired: the closest purpose pair is now
  medical and social at CIEDE2000 22 (airport had been 19 from university). The "(metro public)"
  and "(metro IPEDS)" source tags are gone from names; the inspector's note says what is covered.
- **Complex view is the default** (T-192, Anita). Stored settings saved the old default with
  everything else, so a settings version (2) drops a stored complex view once; Anita's browser
  takes the new default without her switching it.
- **The demand bar has no background** (T-192): its background showed as a white strip in the
  6 px between Destinations and the purpose filter. The buttons carry their own; the loading fade
  is now faded text rather than opacity, so the map does not show through a faded button.
- **One timing chart a purpose, outward up and home down** (T-193, Anita). Replaces the two rows
  (To destination, Return home) with a mirrored chart on one scale of departures an hour, which
  halves its height and puts the morning-out, evening-back shape in one picture. Kept per purpose
  rather than one stacked chart, so a small purpose's timing stays visible. Numbers stay in the
  hovers. Weekend days are left for T-170.
- **Network things get the track tag** (T-194). Trains (riders aboard against seats and packed
  full), stations (demand circle count, riders waiting on each side), crowd bands (as the
  station) and demand circles away from stations share the track tag, its 0.1 s rest and pick
  order, rather than a second tag: one place by the pointer for the network, and the same rules
  for when it shows. Circle counts are recomputed from the demand result rather than read from
  the renderer (another agent's file). Station and train counts were in the inspectors only.

- **The trip explorer reads a trip as a commute, with the model's own numbers** (T-195,
  `sim/src/demand/trip.rs`, docs/demand.md 4.4, docs/interface.md). A new worker query
  (`DemandApi.trip`) recomputes the one subzone pair's choice with `mode_choice`'s expressions
  in the same order on the period's graph, so the chances are the solve's f32 numbers (checked by
  adding a whole origin's pairs back up to the solve's rail trips, 1e-6), rather than a JS copy of
  the formulas that would drift. Choices: **(1) a commute** from a home at the first click to work
  at the second: the commute population is the one whose state the query can read without
  touching the everyday code (another agent's), and it is most trips; everyday purposes use the
  same rail and road costs but their walk share and drive distance come from the sampled cells.
  **(2) The last crowding round's costs**, the graph a period keeps; the published loads are the
  rounds' average, so a trip's chance can differ a little from the average that made the loads.
  **(3) Routes, not a path**: rail is a strategy (station pairs by the access logit, departures
  by frequency), so the answer lists every route carrying at least 0.5% of the trip's rail riders
  with its share, and the panel calls the biggest the route and says how much of the riders it
  carries (17% for Park Slope to Midtown, where riders spread over neighbouring stations).
  **(4) All five periods at once**, each from the worker holding it (a few ms each), so the
  panel's period rows switch at once; the clock's period is shown until one is picked. **(5) Key
  4 from any tab, the tab left as it is**, and a button in the City tab, since it reads demand
  rather than builds; clicks alternate between start and end, and the trip is a selection that
  outlives the tool. **(6) A veil of the land colour at 60% over the map** while a trip is drawn:
  the route rides lines drawn in the same colours, and without the veil it vanished into them.
  Rides are drawn straight between the stations called at, not along the track (the renderer's
  geometry is another agent's, and a stop-to-stop line reads well at trip scale).
- **Each train its own load, each platform its own crowd: the boarding account** (T-164,
  `sim/src/track/boarding.rs`, docs/track-and-service.md 6.6; Anita's middle option from the
  brainstorm, notes/T-164.md). Demand's boardings per line, direction, stop and period arrive at
  the platform over the period; each train lets off the line's share alighting there and takes
  whoever fits up to crush; the rest wait for the next train. Choices left open by the
  brainstorm, taken as the simplest: **(1) the in-period shape** is a fixed shape per period
  scaled so the busiest clock hour is the demand model's busiest-hour factor (T-087) times the
  mean, a tent for the peaks and midday, falling for the evening, highest at both ends at night,
  rather than a smooth curve across the day: it keeps each period's total exact and needs no new
  input; the rate jumps at period boundaries, which nobody sees in a crowd. **(2) Common lines
  keep separate queues**: demand already splits a stop's riders between lines by frequency, so a
  shared queue would mostly redo that; a rider left by a full train does not switch lines.
  **(3) A day at a time, not an hour ahead**: the account is worked out for a whole day of
  departures whenever the service or the riders change (from 03:00 from empty platforms; again
  from a day earlier for a direction still queueing at the end), and each hour's slice is read
  from it, so it is the same whenever the player looks and nothing has to be saved. New riders
  are worked out 3 lines at a time between the clock worker's messages (40-60 ms in all on the
  real NYC save, steps of about 5 ms). **(4) Trains are never over full**: what does not fit
  waits, so the red-rimmed over-full train is gone except for a line the account has no riders
  for yet. **(5) The Average / Busiest hour setting (T-087) is removed**: each train has its
  own load. **(6) Crowds are drawn as a band beside the platform**, one side per direction, in
  steps (under 100, 300, 1,000, 3,000, more), train blue, red while the last train left anyone
  behind; dots in the same band are a setting, as the brainstorm asked for options to show
  Anita. **(7) Left-behind waits do not feed back into demand** (phase 2, Anita's call).

- **Airport ground trips and medical visits are everyday purposes five and six** (T-142,
  `everyday.rs`, `enrich_airports.py`, `enrich_medical.py`). Both from sources that exist for
  every US city, and globally where possible. **Airports** keep their own totals: each airport's
  daily outings are its FAA boardings x its DB1B O&D share x its catchment's share inside the pack
  / 366, and the kernel spreads them over residents and visitors (half population, a quarter jobs,
  a quarter leisure jobs, none within 1.5 km of a terminal) by a 20 km decay normalised per
  airport. Not a per-resident rate: that would give Springfield an airport and ignore that
  Newark and Tweed differ forty-fold. Connecting passengers come off (DB1B, national and free:
  Dulles and BWI connect a quarter to a third, Boston almost none); staff are commuters already.
  Terminals from OSM, so a station has to reach the terminal, not the runways. Trips to an
  airport in another pack (Springfield to Bradley, Washington to BWI) are dropped, as for every
  everyday purpose. **Medical** is a rate per resident, 0.04 outings a day (NHTS 2022: 4.02% of
  2.27 trips a person, 0.091 legs; the model's errands and social rates sit similarly under
  NHTS's), to OSM hospitals, clinics, doctors and dentists sized in visits a year (a hospital
  1,000 per bed, a clinic 10,000, a practice 4,000, a dentist 3,000). Doctors and dentists are in
  because hospitals alone would pull every office visit onto a hospital campus; OSM maps them
  unevenly, and where they are missing their visits go to the nearest mapped place. Untagged
  hospitals are sized by their buildings' floor area (footprint x levels), not their site's area:
  site area put sprawling suburban psychiatric and VA campuses above every Manhattan hospital.
  Colours olive and pink, the two hues left that stay about as far from the others as those are
  from each other (green is kept for events). **Two destination samples a home cell** for both,
  not sixteen: with sixteen a city's everyday opening took 70-90% longer (the cost is per
  sample); with two it takes 5-15% longer natively and the aggregates are unchanged (medical
  walk share within 0.3 points, every big airport within 0.4% of its input).

- **Transfers are counted in the assignment, every round, and drawn line to line** (T-176,
  `kernel.rs` `assign_with_transfers`, `ui/inspector/ComplexMap.tsx`). The assignment now notes
  each rider leaving a train at a station that is not their destination and splits them over that
  station's departures like everyone there, keyed by arriving and leaving route node (platform to
  platform); the loads are untouched (bit-identical natively and in the WASM worker). Counted in
  every round and averaged like the loads rather than on the last round only: it adds about 5 ms
  to a 36 ms commute assignment and is lost in the noise of a 1.0-1.4 s stage, while the last
  round alone would be off by that round's change and missing when an uncrowded round is skipped.
  Sent only when asked, for one complex's route nodes (a few kB), never with every result. The
  complex map joins lines, not platform bundles as T-147's notes planned: the real save's Penn
  Station and Grand Central are three bundles each, so bundle to bundle gives at most three
  curves and hides which lines change; the busiest 8 line pairs between different bundles are
  drawn, changes inside one bundle go in each line's hover. Curves sit over the platforms (under
  them they vanished beneath Penn's commuter bundle). docs/demand.md 4.4, docs/interface.md.

- **Towns and neighbourhoods come from a per-city OSM place list, beside the pack** (T-177, T-065,
  `pipeline/places.py`, `game/placeList.ts`). OSM `place=city|town|borough|suburb|quarter|village|
  neighbourhood|hamlet` inside each pack's boundary, one Overpass request a city (cached in
  `data/raw/`), same-named rows within 2 km merged; 18,397 places over the nine cities, about
  730 KB. A separate `<id>.places.json` rather than a block in the `.bin`: it is not demand input,
  it can be rebuilt in seconds without touching the pack, and it loads only when the search box is
  first focused or the station tool first picked, never at startup. No admin boundaries: towns
  are already `place` points and a containment test would add nothing the search uses. Hamlets
  are kept (much of Long Island, New Jersey and Maryland is tagged that way). Search ranks places
  after stations on equal match (bigger kinds, then population, first) and also matches clipped
  nicknames ("south philly"), the only general rule that finds it, since OSM gives South
  Philadelphia no alt_name. A new station is named from the list when the map is below zoom 13,
  where the basemap's tiles hold only the bigger roads (a nearest loaded street there is not the
  nearest street), or when no street is within 400 m; in use, it takes another place at most
  800 m further, else "North Williamsburg", else "Williamsburg 2".

- **The regional market measures zone distances on the ground** (T-128, `demand/intercity.rs`,
  `pipeline/intercity.py`). Its decay and drive times used straight distances in its equirectangular
  New York pack frame (1.2% short east-west at Philadelphia, 2.7% at Boston). Now the pack names
  its frame (`frame`, format 2) and the kernel measures zone centres by great circle, as the world
  frame's lengths are checked against. The sites stay in the pack frame, so positions, selections
  and the app are untouched. Site lon/lat are not stored as T-128 first proposed: the frame is
  affine in lon/lat, so it gives the zone centres' lon/lat exactly, and storing them would add
  about 5 MB to the download. A pack without `frame` keeps plane distances under the old formula
  name (its block stays current, its numbers unchanged). Not re-tuned, under "intercity demand is
  rough on purpose": the day's trips are unchanged (696.1k), corridors move up to about 3%, market
  rail on the real NYC save +0.07%. docs/demand.md 4.4.

- **Capacity tags move off station names; names never yield** (T-184, `map/capacityMarkers.ts`).
  Tags sat at one fixed spot above right of their resource, which at a busy station is where its
  name is ("+9:17" on Times Sq, "+5:39" on Union Sq). Names are the map's orientation, so they
  are placed first and keep their spots; each tag then takes the first of six spots around its
  resource (above right as before, above left, below right, below left, right, left) that clears
  every shown name by 3 px and the tags already placed (with T-102's spacing), and hides only if
  none does. A corner of the tag stays 5-6 px from the resource, so it still reads as that
  resource's. On the real NYC save in Midtown at zooms 12-15 and through a pan, overlaps went from
  1-5 a view to none, with about the same number of tags shown. Not moving the tag along the
  track (the resource is a point; a tag far down the line reads as another station's), not
  hiding names under tags, and not MapLibre collision (both are DOM layers over the overlay
  canvas, T-032). `StationLabels` exposes its shown names' boxes and tells the tags when they move.

- **A loading card in the middle of the map while the clock waits** (Anita, T-180). T-172 moved
  the startup load into the top bar so nothing blocked play; Anita wants it obvious until the
  clock starts. While a loaded game's clock is held a plain card centred over the map says
  "Loading New York, 42%" and the step; it takes no pointer or key (a press on it pans the map),
  has no animation, and goes when the clock starts. Later loads (a pan to another city) stay in
  the top bar, whose status now ends in an ellipsis in a narrow column instead of running over
  the cash. Not a dimmed overlay or a progress bar: the map stays fully visible and usable, and
  the percentage is the motion.

- **Hover tests skip far segments in local units** (T-180, `map/pick.ts`). Every pointer move
  scanned all 117,701 line and track segments of the real NYC save, projecting each end into two
  new arrays (about 5 ms a scan, several a move; hovering kept the main thread 70% busy). The
  pointer is now taken into local units once and a segment whose grown box misses it is skipped
  before the old arithmetic; the answers are identical (4,000 checked). Not a spatial index: the
  box test makes a scan 0.66 ms, and an index would have to follow every edit.

- **The regional market goes to the demand workers as bytes** (T-180). The main thread no longer
  decodes the 8.2 MB pack or clones its text six times while the player can already pan; each
  solving worker decodes its own copy, the aux worker gets none.

- **One rule for which keys a focused control keeps** (T-183, `ui/keys.ts`). main.ts and the
  search box disagreed: search treated a select as a text field and ignored `/` and Ctrl+K,
  while the game's Space, 1-3 and Esc fired from it, so after picking a purpose "/times sq"
  searched nothing and the space paused. Now both ask `keyIsTheirs`: a text field keeps every
  key; a select, slider, checkbox, radio or colour input keeps only the keys it works with
  (arrows, Home, End, Page Up and Down, Enter, Space, and a letter in a select), and everything
  else reaches the game; a button keeps nothing (Space pauses, as before). Before, a slider or
  checkbox swallowed every game key too. A select picked with the mouse also blurs, so Space
  pauses straight after a pick instead of reopening the list; a keyboard user keeps focus and
  the arrows, which is why the blur is mouse only.

- **Same-named search results in one city are told apart by their own lines** (T-183). Two
  complexes more than 50 m apart can share a name (Times Sq-42 St and the 8 Av A/C/E platforms
  in the real NYC save), and every line's badge in that save is "S", so the chips do not help.
  The row's second line lists the lines only that result serves ("subway A, C, E"), which is
  what a player knows their network by; a compass direction from the others is the fallback when
  a result has no lines of its own (blueprint stations). Results in different cities keep the
  city.

- **New York's schools come from NCES like every other city** (Anita, T-181). The NYC DOE
  building input for the five boroughs (T-140/T-150) is removed from the pipeline, under the "no
  city-specific data" rule. New York's pupils rise from 2,694,630 to 2,825,244: NCES includes
  charter schools (141,775 pupils in the boroughs), which the DOE building report did not carry.
  School outings +4.9%. notes/T-181.md.

- **Washington's border factor is its own fit, 0.49** (Anita, T-181), replacing the US 0.23
  there. DC is a city counted as a state, so a factor fitted on states between New York, New
  Jersey and Pennsylvania cut DC's commuting to Maryland and Virginia to a quarter of LODES (3-4%
  of DC's commuters against 13.5% each); with 0.49 it is 10%, and the between-state share
  matches LODES (25.0% against 24.9%). A city config's `border_factor` overrides
  `BORDER_FACTOR_US` in `build_city.py`. This replaces T-162's "one value for every US pack".

- **Borders outside the US: national borders get a factor, first-level regions none by
  default; inside the Schengen area the national factor is weaker** (Anita, T-181). Values are
  fitted when the first non-US pack is built. The US states keep theirs because LODES measured
  it (T-159, T-162); elsewhere admin-1 lines are not assumed to cut commuting (Tokyo's
  prefectures, city-states such as Berlin or Vienna). Settles T-162's open question.

- **Anita's UI answers** (T-182). A search hit for a complex's shared name selects the member
  with the most lines and leaves the Station / Complex setting alone: it is a remembered setting
  that also controls the demand circles, so a search should not change it (replaces T-148's
  "lands on the complex"). The City pane's City centre and Whole region buttons jump like search
  does, the house rule of no animations; those were the only animated camera moves in the code
  (MapLibre's own gesture easing is untouched). The line inspector's Local trips and Longer trips
  rows are gone: the purpose split under Riders a day (T-165) already shows longer trips, and the
  demand result no longer carries the line's regional boardings separately (closes T-175).

- **No new cities until the general design settles** (Anita, T-179). Nine cities are enough for
  now; more packs would make it harder to change things not yet nailed down (the market, the
  purposes, the gravity). T-179's batch was undone; `docs/adding-a-city.md` keeps the method.

- **The regional market's all-pairs figures are derived offline; zone pairs are built only
  between served zones** (T-174). Opening the market visited every zone pair twice in every
  worker (1.2-1.6 s in Chrome, about 80 MB a worker, both growing with the square of the zones:
  30 cities would not have fitted). Chosen over the T-172 suggestion of shipping the pairs
  themselves in a binary pack: those would be 2.3M pairs now and about 20M at 30 cities to
  download and hold in every worker, and would freeze the pair formula's output into the pack.
  The block holds only O(zones) numbers computed by the game's own code, names its formula and
  is ignored when stale (the game derives it at opening, same result), so the market's design,
  which Anita has left open, can change with a one-line rerun (`pipeline/intercity.py
  --derive`). Only served-zone pairs can carry rail, so building just those changes no number.
  Same outputs to the bit natively, 1.2e-7 at most in the browser (libm). notes/T-174.md.

- **One demand worker warms the everyday code up before the others open a city** (T-173). Five
  workers opening New York's everyday layer at once from the browser's baseline code took 7-8 s
  each in most page loads, against 1.3-1.8 s on optimised code. Worker 0 opens a small synthetic
  everyday layer at startup (150 ms) and the other solving workers get city packs once it is
  ready, so they open on the optimised code it leaves (code is shared by a page's workers). New
  game clock 2.7-3.1 s in 10 of 10 loads. Chosen over holding the others until worker 0 had opened
  New York itself (also fixes it, but 4.2-4.9 s: they open after it). Undo or compare with
  `?perfOff=stagger`. notes/T-174.md.

- **Every registered city gets New York's non-work purposes, from national sources only**
  (T-142, under Anita's "no city-specific data"). Everyday outings from LODES CNS sectors (already
  cached), school attendance from NCES EDGE/CCD public schools, university attendance from IPEDS,
  with New York's constants (rates, decays, attendance factors, timing) unchanged and no per-city
  calibration. The runtime opens the everyday layer for any pack that carries its arrays instead
  of New York alone. Per capita the inputs land near New York's (school 0.13-0.15 pupils a
  resident against 0.12; university 0.035-0.078 students against 0.038, highest in the college
  towns of Springfield and New Haven), which is the check, not a target. New York keeps its DOE
  buildings (Anita's open question).
- **A merged college's students are split over its absorbed campuses** (T-142). IPEDS reports an
  institution that absorbed others in the data year at one point: Connecticut State Community
  College put all twelve former Connecticut community colleges (25.6k students) in Hartford. The
  split uses each campus's own IPEDS enrollment the year before, a rule that applies to any
  merger anywhere in the US, not a Connecticut fix. It also added 5,330 students to New York
  (Housatonic and Norwalk in Fairfield County, which New York had silently lost). Multi-campus
  institutions that have always reported as one (Northern Virginia CC) stay at one point:
  nothing national gives their campus split.
- **The everyday pass searches only towards stations its pairs can reach** (T-142), as the commute
  pass does, instead of towards every station. A city opened beside the one being built in, with
  no pairs near the network, then searches nothing. Each destination's search is independent, so
  no number changes (T-149's worker check: every load bit-identical).

- **The complex map draws per-line riders now and leaves transfer flows for later** (T-147).
  The demand assignment does not count transfers at any level (a rider leaving a train joins the
  station's flow and loses which line they came from), and drawing a split nothing measures is
  not done, so the map shows platforms and each line's riders there, and counting transfers in
  the solve is its own task (T-176, notes/T-147.md: where in `assign`, cost, and the strongest 8
  pairs to draw). Width is riders on the line's trains at the platform, not boardings: it says
  how busy the line is there, which a line through a hub with few boardings still is; boardings
  stay in the list under the map and on hover.

- **Platforms on top of each other are drawn as one bundle in the complex map** (T-147). Parallel
  within 15 degrees and centre lines within 12 m (about one island platform's width). The real
  New York save gives every service its own station, so true positions alone stacked Penn's and
  Grand Central's commuter lines into one unreadable pile; side by side, the way the main map
  draws lines sharing track, every line can be seen and clicked. Positions and headings stay true
  for everything else.

- **Complex view enlarges every member's platform box** (T-163), with a per-station flag in the
  renderer rather than a second draw. Picking takes the enlarged boxes, as for the selected one.

- **Search sits at the map's upper right, opposite the demand selector** (T-148). The top bar is
  full and already folds into rows at narrow widths, and a row in the dock would push the
  inspector down for good; the map's upper right was empty (attribution and tool hints are at
  the bottom). It is a plain 1 px input like the selector's controls, with the results in a list
  under it. When the map is narrower than 680 px it moves under the selector rather than overlap
  it. `/` or Ctrl+K focuses it: neither is used by the game, and they are what search boxes on the
  web use.
- **A search pick jumps, it does not fly** (T-148). The camera moves at once, like every other
  state change in the game (no transitions); the City pane's City centre button still flies.
- **A search hit for a name shared by a complex's platforms lands on the complex** (T-148). One
  result for the name; picking it selects the platform with the most lines and turns complex view
  on (the inspector's own Station / Complex setting), because the player searched for the place,
  not for one of its platforms. With a build tool on, a pick only moves the camera, so a half-drawn
  route or a line's stop editing carries on where the player went. Equal matches list cities
  before lines and stations, so "new" offers New York and New Haven before Newark's stations.
- **No geocoder for neighbourhoods or addresses yet** (T-148). Every free geocoder sends what the
  player types to a third party, which is Anita's call; the basemap's tiles hold neighbourhood
  names only in zoom-14 tiles already loaded, and no addresses. The suggested route is a place
  list per pack from OSM built by the pipeline (T-177, notes/T-148.md).

- **Riders by purpose in the line and station inspectors: a thin bar of every purpose, the
  filtered one ringed** (T-165). Under the map's purpose filter the split keeps all six purposes
  rather than switching to the filtered purpose alone, and rings that purpose's segment in the
  bar and bolds its key entry. Why: the inspector's totals stay all-purpose (T-158 kept train
  fill, crowding and these totals counting everyone), so a split of that total is only coherent
  with every part in it; the ring answers "how much of this line is the purpose the map shows"
  at a glance, which is what a thin line on the map makes the player ask. Other purposes are
  not faded, because the key is the whole palette and an authored colour is never faded. A bar
  with a key rather than the pie: the inspector is narrow and the figure sits under a single row;
  counts are on hover, shares in the key. Zero purposes stay listed (muted text) so the key does
  not change shape from line to line.

- **Startup never blocks input; the clock waits for the first estimate, not the last crowding
  round** (T-172, replacing T-149's loading screen). Anita asked for input blocked until the
  viewed city was fully loaded, then found the wait too long (production: 12-20 s new game,
  32-38 s on the real New York save). Now the map, inspectors and building work once the network
  is drawn (0.3-1.4 s); the top bar shows "Loading New York, 42%" and the mode split "–", and
  only the clock is held, until every period's free-flow estimate has landed for every city the
  game needs (new game typically 4.3-5.4 s, NYC save 7.5-8.2 s). Why the first estimate: after
  any edit the game already plays on the newest published result while crowding rounds refine
  it, so startup now follows the same rule; the rounds move a big network's figures by a few
  percent and land 2.5-8 s later. Why still hold the clock: with no demand at all, fares would be
  zero for the first game minutes. The same applies to a file, slot or new game loaded later.
  The failure screen (Try again, Play without demand) stays. Easy to undo: `READY_ROUND` in
  `workers/demandClient.ts` (3 restores waiting for the last round); the screen is
  `ui/LoadingScreen.tsx`. notes/T-172.md.

- **The regional market's free-flow estimate routes with the distance fare** (T-172, fixing
  T-121). Its round 0 used a graph without fares, so it ran a strategy search of its own: 1.8 s
  natively a period on the real New York save, and in five Chrome workers at once about 7.5 s of
  that save's 10 s first estimate. Now it shares the local populations' search, as every crowding
  round already did. Effect on the real New York save after three rounds: market rail trips 2,375
  to 1,886 a day (-21%), market boardings 7,388 to 5,835; New York commute and everyday rail
  change by under 0.005%. T-121 left this to Anita because the market had been tuned against
  the old figure; decided here because Anita has since said intercity demand is rough on purpose
  (CLAUDE.md, 2026-10-12) and the old estimate was the inconsistent one. Two lines to undo
  (`set_crowding` and `uniform` in `IntercityApi::solve`).

- **Work gravity: a state line multiplies the decay by 0.23; the decay length goes from 29 to 40
  km** (T-162, implementing T-159). Fitted on a coarse grid against LODES OD, k set so the share of
  commutes between states matches, scored on trip lengths and county pairs. One value for all US
  packs because New York's own fit (0.235) and Philadelphia's (0.219) agree within 7%, and Boston
  (0.29) and Providence (0.35) are near; Washington's own fit is 0.49 (DC is a city counted as a
  state), but 0.23 still beats no factor there. The decay shape stays d^-1 (a d^-0.9 shape scored
  1.1 points better out of 23, not worth a second change). The factor is a per-pack value and a
  pack without one has 1 (no effect): outside the US an admin-1 factor would be a guess (Tokyo's
  prefectures, city-states like Berlin), so each country sets its own from its own commuting
  data, and which borders get one is Anita's call. Replaces T-072's county-pair K-factors and
  road travel time: one parameter fixes the NJ overshoot, and road time closed only a tenth.
  notes/T-162.md.

- **One map label per name per station complex** (T-169). A complex's stations that share a name
  (the real NYC save has 14 platforms named Times Sq-42 St/42 St-Port Authority Bus Terminal, in
  two complexes) printed the name once each on top of itself. They now share one label at their
  mean position, turned like the member with the most lines and ranked by the lines at any of
  them. The complex, not distance, decides: same-named stations in different complexes (8 Av and
  7 Av at Times Sq) keep one label each, and differently named members of one complex keep theirs.

- **Varying demand: Anita's answers in the T-170 brainstorm.** No random disruptions. No separate
  weekend schedules for now. Superholidays (Asian ones especially) get special treatment.
  Stadium events matter most as points that make bursts of demand on some days; generated,
  realistic-looking game dates are fine. Event notices only for places the player has built in;
  elsewhere events are silent. Delays, if simulated, are tracked per city. notes/T-170.md.

- **The purpose filter keeps per-purpose loads beside the total; it never re-solves** (T-158).
  The populations were already separate per worker (city commutes, NYC everyday, regional
  market); only the everyday layer mixed its four purposes into one assignment. It now assigns
  them purpose by purpose along the same strategies, reusing one station-pair matrix, and sums
  them into the total crowding reads. On the real NYC network that costs about 65 ms per
  period and stage (assignment 98 ms against 32 ms) and no extra matrix; a matrix per purpose
  would have saved part of that time for about 9 MB more WASM memory per worker. Re-solving per
  purpose was never an option: mode choice and crowding must see everyone together.

- **Purpose filter: a dropdown at the end of the None/Homes/Destinations bar, not in settings**
  (T-158). It is a way of looking at the map, chosen while looking, like Homes/Destinations;
  settings hold things set once. Seven choices: All purposes, Work, School, University,
  Shopping and errands, Social and leisure, Longer trips (the regional market loads lines too, so
  it is a segment). A swatch in the pie's purpose colour (T-143) marks a filtered map; longer
  trips take the pie's neutral grey. Not remembered across page loads, so nobody returns to a
  map that silently leaves riders out.

- **What the purpose filter changes, and what it does not** (T-158). It changes line widths,
  station circles (complex view too), the Homes/Destinations bubbles and a selection's bubbles,
  their hover counts and the track tag, which names the purpose. Train fill and crowding count
  everyone aboard: a train is as full as it is. Drawings with no such purpose hide rather than
  show another segment: longer-trip bubbles show only under All or Longer trips, a station's
  commuter catchment only under All or Work. Scales stay absolute, so a small purpose (school,
  longer trips) draws thin and faint against the same full width; that keeps purposes
  comparable. Inspectors keep all-purpose numbers, except a bubble selection's counts and places,
  which match its bubbles and say so; its purpose pie keeps every purpose.

- **Crowds and refused boarding come from a per-train account, not from individual passengers**
  (Anita, T-164). Demand stays counts per period; each line's departures then board what fits up
  to crush and leave the rest on the platform for the next train. She chose this over simulating
  people, and allowed coarser treatment out of view. Signals stay a separate question (notes/T-164.md).

- **Station complexes: platforms within 50 m, derived, not saved** (T-146). Two stations are one
  complex when their platform centre lines (a straight segment of the platform's length along
  the track heading) come within 50 m, or their centres within 100 m (the distance at which the
  demand kernel already merges stations, so every demand station sits inside one complex and a
  complex's catchment never counts one twice); single linkage, levels ignored. 50 m is about
  where the platform boxes touch on screen at the zooms players build at (zoom 14-15), and
  about a street's width between real platforms 10-15 m wide. On New York's real network it
  gives the complexes a rider knows (Times Sq, Grand Central with Metro-North, Herald Sq with
  PATH 33rd St, Atlantic Av with Atlantic Terminal, Jamaica with Sutphin Blvd). Complexes are
  worked out from the stations' geometry on the main thread whenever stations change: no save
  format change, no worker change, and nothing to keep in step when a station moves. Complex
  view is one remembered display setting, shared by the station inspector's Station/Complex
  switch and the demand circles. It aggregates display only; demand transfers keep their
  100 m merge and 300 m walk (T-161).

- **Add NYC public-school attendance using enrollment, not employee jobs** (T-140). Match DOE
  buildings once to reported enrollment, then estimate homes from child-population weights.
  Destination budgets avoid mistaking staff counts for pupil capacity. The initial 90% attendance
  factor and 2.5 km home-distance decay are provisional; morning arrivals and mostly midday
  returns distinguish school from work. Include actual assignment in purpose/timing inspection,
  loads and fares, with explicit public-building coverage and no inferred student OD claim.

- **Lower the everyday-trip exclusion to 150 m** (Anita, T-141). Replaces the initial 300 m
  threshold. New outings below 150 m represented cell-centre distance are excluded from every
  mode and total; exactly 150 m remains eligible. Other demand parameters stay unchanged.

- **NYC everyday outings and a 300 m exclusion** (Anita, T-066). Shopping/errands use retail/
  service jobs; social/leisure use food/recreation jobs and visits to residents. Source-neutral
  weights allow later international inputs. Trial rates and timing generate partial demand,
  never a forced commute-share target. Outward/return routes are priced separately and join
  shared fares, loads and crowding. New outings under 300 m cell-centre distance disappear from
  all modes and totals without replacement. Bubbles now count local trip legs, with purpose and
  directional timing inspection shipped alongside them; station commute queries remain labelled.

- **Separate same-city commuting from the longer-trip overlay; show metropolitan focus**
  (Anita, T-139). The relative longer-trip scale made 26 trips look larger than 583 commuters.
  Same-city longer-trip discs are now hidden at all zooms, with counts retained in the inspector
  and cross-city discs retained. Every far zone carries its exact market ID. Thin solid city
  boundaries replace dashes; a subtle exterior basemap shade and darker outside/lighter inside
  focus stroke distinguish the active pack. NYC's pack includes northern New Jersey.
- **Trip-purpose shares need their own distance/time context** (Anita, T-066).
  Work journeys are generally longer than everyday journeys. Compare counts and length profiles
  separately; a trip-count share is not a passenger-distance, travel-time or transit-mode share.

- **Ship destination purpose/timing inspection alongside non-work demand** (Anita, T-066).
  Each selected destination bubble explains its work, shopping/errands, social/leisure and later
  school/facility trip counts and shares, with a rough daily distribution for each purpose.
  Use the bubble's actual purpose/period results and clearly labelled mode and direction;
  counts must add up. Planned behavior, implementation details in notes/T-138.md.

- **Reopen non-work local trips, starting with NYC where useful** (Anita, T-066/T-138).
  Distinguish trip purposes and daily timing; do not force a reported commute share while the
  model covers only part of non-work travel. T-138 records sources and a proposed bounded first
  slice. Its numerical rates/weights are trial suggestions, not calibration or shipped behavior.

- **Add Northeast cities in coherent batches of at most five** (Anita, T-137).
  Baltimore, Providence, New Haven, Hartford and Springfield fill the corridor between existing
  metros, using disjoint conventional county extents and the established trusted data pipeline.
  Basic pack and gameplay checks replace independent calibration. City/place metadata now drives
  the regional builder through the shared registry; new IDs append without renumbering old places.

- **Traveller totals determine bubble area; omit the selected city's aggregate** (Anita, T-136).
  Remove additive radius offsets and minimum sizes that inflated small flows. Use the normal
  demand mix: blue at full train share, red at zero, blended in between. A Boston selection
  shows grouped totals only for other places, keeping Boston's local dots at their real positions.
- **Selected longer trips stay visible, with city totals when zoomed out** (Anita, T-135).
  Work selections show incoming origins, home selections outgoing destinations. Their smaller
  counts need their own visible scale rather than commuter-density sizing and a small-count
  cutoff. All positive detailed zones remain; regional zooms show one labeled bubble per market
  with its exact daily total. This changes display only, preserving trip assignments and shares.

## 2026-10-10

- **Add cities through the established sources with broad metro extents** (Anita, T-134).
  Boston and Washington, DC reuse the Census/LODES/ACS/water pipeline. Basic format, totals and
  game-loading checks suffice; a new extent study or independent calibration is unnecessary for
  familiar sources. Both get local commute packs and regional endpoints. Original market IDs
  stay stable, while flow headers and corridor tables scale to the pack's market count.
- **Demand controls stay on the map; picking follows drawing order** (Anita, T-131–133).
  None/Homes/Destinations is always visible at the map's upper left, with the existing commute
  endpoints and saved preferences. Exact metropolitan pack footprints show at regional zooms.
  Visible demand bubbles beat underlying trains/stations; same-layer hits follow the last drawn
  object, and selection/catchment discs retain their parent selection. Explicit selected editing
  handles stay usable. Local non-commute trips are deferred until their scope is bounded.
- **Content first; performance work waits for a reported problem** (Anita). Current performance
  is adequate. Choose new content and gameplay features rather than proactive optimization;
  keep the existing responsive architecture. Screenshots and diagnostic plots belong outside
  the project in `maps/artifacts/trainworld/`, ignored by Git; preserve existing evidence there
  and update producers/references instead of cluttering the source directory (T-130).
- **Regional packs parse into numeric rows, and immutable tables release spare capacity**
  (T-123). Generic JSON trees kept large temporary allocations alive during pair construction,
  inflating every worker's WASM memory high-water mark. Typed rows preserve numeric precision
  and pair order while skipping metadata; unused capacity is released once pairs and lookup
  indices are complete. Keep separate worker ownership and the existing trip model. On the
  full NYC/Philadelphia fixture this saves about 46 MB across the pool, and 236 MB on opening,
  with unchanged demand. notes/T-123.md.
- **World coordinates: a Lambert conformal frame for new games, the old frame kept for old saves**
  (T-125, closing T-053; Anita: grow past two cities, intercity track between regions, correct
  lengths, curves and speeds on a line of several hundred km, existing saves identical). The track
  model stays planar; what changes is the plane. New games use a spherical Lambert conformal conic
  tangent at New York with the old origin and metre units: conformal, so arc radii and curve speeds
  are right in any direction, and its scale is `1 + dlat^2/2` (0.24% at 4 degrees of latitude,
  under 0.4% across 35.8 to 45.8 N at any longitude), against the old frame's direction-dependent
  1.2% at Philadelphia and 2.7% at Boston. Straight edges from New York to Philadelphia, Boston,
  Washington and Pittsburgh come out within 0.02% of the great circle (they were off by 0.2 to
  0.8%). A network records its frame: TWT5 is the world frame, TWT4 and older the legacy one, and
  a legacy-frame network still encodes as TWT4 byte for byte. Existing saves are NOT converted:
  they load in the legacy frame and stay there, because moving their track would change lengths,
  quotes and schedules and she asked for identical geometry, money, fares and ledger (the real
  NYC network's derived geometry, costs and service hash identically before and after, and its
  demand results are equal to the last digit). An explicit, tested conversion exists but is not
  offered in the UI (her call, T-129). Rejected: per-region local frames with a region id per node
  (an edge between regions needs a frame of its own, crossing tests need a common one, the format
  grows a field, and the old saves' Philadelphia track still needs a frame that is 1.2% off);
  lon/lat plus a local metric (rewrites the planar geometry); Web Mercator as the model frame
  (conformal but scale 1.32 here, which would scale every limit and cost). Water masks go
  through lon/lat from each pack's own frame (tabulated on a 2 km lattice, under 0.1 m, because
  the exact version made an edit 3.7x slower); all demand conversions go through lon/lat so packs
  keep their own origins; one registry file (`game/registry.json`) names the world, frames, packs
  and cities for the workers, the build, the save header and the UI. The regional market keeps its
  sites in its pack frame, so its own zone distances still carry the old frame's 1.2% to 2.7%
  (T-128). notes/T-125.md.
- **Longer-trip bubbles are drawn like commuter bubbles, one per far zone** (T-120; Anita: "Elsewhere
  in the New York region" is confusing, starts within the region should be visualised normally, and
  the yellow bubble style is out: size by people, red or blue by train usage, everyone not on a
  train red). Replaces T-118's one pale-plus-solid amber disc per place at a fixed 44 px. Now a disc
  per 6 km zone at the zone's own place, the commuter far end's sizing (biggest 0.6 of a zone, scales
  with the map and the size slider), opacity and mode mix, blue for rail and red for the rest.
  Per-zone sums come from Rust so the workers' answers still add up; the station's per-market rows
  became per-zone rows (f32) and the market sums are derived from them. The map's name tags are gone;
  the inspector keeps its by-place lists, and the own region's row is "Other New York region places".
  notes/T-120.md.
- **The demand solve is made faster without changing its numbers** (T-119, closing T-061). It was
  4.5 s to the free-flow estimate and 17 s to round three on the real network, and the bubbles
  waited for round three because a worker could only answer about a round while it had no later
  one under way. Now: strategy searches only towards stations a trip can end at, shared between
  the populations of a worker; each worker sends the bubbles' sums with every round and a worker
  with no periods answers the rest; the pool is up to 5 workers (one per period) plus that one
  (was at most 3). Rejected: splitting mode choice by origin zone over the pool (it only fixes
  the 5-periods-on-3-workers imbalance that five workers remove, at a reduce step per round);
  wasm SIMD on the pair loop (no gain in Chrome). Left alone, flagged in TASKS (T-121): the
  regional market's free-flow estimate routes without the distance fare, 17% more rail than a
  consistent run. Crowding rounds in which no segment is over its seats are skipped, exactly.
  The cost is about 240 MB more WASM memory (5 + 1 workers) on a 16-thread machine.
  notes/T-119.md.
- **One Northeast network with shared crowding** (T-116). Keep the existing NYC metre frame
  and legacy saves; transform city queries and append transformed water masks. Both city
  commute packs and the regional market share one bounded worker pool. Every crowding stage
  reads one frozen combined flow vector, and publications require matching versions/rounds.
  Navigation changes the local commute share and queries, while network loads/income remain
  global. The fixed plane's roughly 1.2% Philadelphia east/west scale error is acceptable for
  this first region; broader world coordinates remain T-053.
- **Longer trips are shown by place on selections and stations** (T-118, 2026-10-10; Anita: a
  Philadelphia bubble selection panned to New York lost its selection and could not show intercity
  demand). One bubble per place (the market's three: New York, Philadelphia, Trenton and Mercer),
  not per zone, because the market is coarse and a player reads places. Selected local cells map
  to the market through the 600 m squares both are built from, with the selected share of a
  square; no new numbers are made, and the numbers are the market's own pair totals. Station
  riders come from the assignment itself (per-station counts by far market), so they equal its
  entries and exits. Rail per pair is kept per period and averaged over crowding rounds like the
  loads. Bubbles use area for trips, a pale all-modes disc and a solid rail disc, in a colour no
  mode uses. The "Longer trips" check hides bubbles only; the numbers stay in the inspectors.
  notes/T-118.md.
- **Intercity begins as a tunable non-commute market** (T-117). Actual population/job locations
  in NYC, Philadelphia and Mercer form coarse zone-pair trips, including long trips within
  metros, with a smooth geographic onset. The common rail graph gives feeder lines actual
  passengers and applies fares/service/shared crowding. Keep attribution separate from commute
  denominators. Judgement trip rates are explicit; measured OD calibration, park-and-ride,
  air/coach and cross-pack commuting remain future work rather than implied implemented features.
- **Conventional centered confirmations** (Anita, T-113). Replace bottom-left questions
  with a modal Are you sure? dialog, cancel focused first, trapped keyboard focus and blocked
  background input. Keep the existing operation-specific warning and action labels.
- **City pane is for navigation** (Anita, T-114). Remove the diagnostic period rail-share
  chart and People/Jobs totals; the top bar already supplies the goal share. Begin with
  New York navigation and extend it with shared-world cities, rather than inventing metrics.
- **Philadelphia is the first Northeast expansion** (Anita, T-115/T-116/T-117). Build the
  standard 11-county metro pack separately from New York, with no duplicate demand counties.
  Preserve New York's track/save frame in the future shared world; transform local demand
  into each city's pack frame and combine shared-line crowding before feedback. City packs
  alone do not generate intercity trips; that separate layer remains T-117. notes/T-116.md.

- **Line management starts in the inspector** (Anita, T-109 to T-112). New line creates an
  empty zero-train draft immediately, enters stop editing and selects its default name for typing.
  Persist creation-order names, custom multi-character badges and independent full-property
  copies. Put list actions above long lists; sort lines by natural name or daily passengers.
  Restrict red stop-removal controls to Edit stops and confirm line deletion. Infill additions
  test all ordered insertion positions, rather than failing after trying only the end.
  Optional save metadata keeps old saves readable.

- **Short turns share the station they reverse in** (T-052). Count their full turnaround on
  the arrival platform and their crossover against crossing through moves or opposite reversals.
  Use per-move conflicts with the own-move queue subtracted, so independent through directions
  stay independent and tiny reversing service introduces no abrupt extra queue. Single track
  retains one shared platform; adding a passing loop would require explicit geometry and cost.
  Only constructed track contributes active station capacity. notes/T-052.md.

- **Blueprint planning leaves running endpoint geometry alone** (T-108). Constructed lines
  use constructed edge ends to identify dead ends; planned lines still preview planned ends.
  Drawing an unused extension must not shift a running train's stop or change its timing.
  notes/T-108.md.

- **Draw the overlay once per presented map frame** (T-075). Adopt the existing queued-frame
  guard after six alternating stress checks reproduce half as many draws and about half the
  drawing work, with unchanged map frames and zero camera lag. This measures overlay work, not
  a claimed halving of total browser CPU. `perfOff=noDouble` retains the old path for comparison.

- **Edit blueprint corners from their inspector** (T-071). Add inserts midway between adjacent
  control points, then the player drags its square; each corner has level and removal controls.
  Reuse checked shape edits and undo rather than adding a map gesture that conflicts with
  selection or drawing. Removing the last corner yields a straight stretch. notes/T-071.md.

- **M1 spec stays current and readable** (T-010). Correct stale schedule, save, demand and
  backlog claims; keep earlier calibration figures explicitly historical. Move oversized demand,
  track/service and interface sections intact into `docs/`, with linked summaries in SPEC and
  their existing section numbers preserved. No gameplay design changes. notes/T-010.md.

- **Independent browser saves with one writer per slot** (T-068). Named slots retain the
  original autosave and copy the current game without replacing it. Hold a browser lock in the
  worker for each writable slot; blocked tabs can play/export and rescue into a new slot.
  Flush before switching, read after acquiring the new lock, and keep save failures visible.
  Different slots can save concurrently. notes/T-068.md.

- **Conservative circle growth at station complexes** (T-094). Platforms within 100 m keep
  individual dots and selection but cap rider growth at 1.6x radius, versus 2.6x for isolated
  stations. This reduces clutter without moving station geometry or hiding distinct platforms.
  A cached spatial grid keeps nearby-platform detection out of the animation loop.
- **Common-line waiting is destination-specific** (T-069). Reverse strategy searches combine
  useful departures and their expected onward cost; boarding splits by frequency. Pooling every
  line at a station would incorrectly shorten waits for the wrong branch. Keep per-line loads
  for crowding, income and displays; fares remain separate from crowding. notes/T-069.md.

- **Top bar follows the map column's width** (T-106). Use container queries, since resizing
  the dock can narrow the map without changing the browser window. Hide secondary labels before
  moving to two or three rows; preserve cash, time, train share and all playback buttons.
  Full mode shares and day/demand remain in hover titles. notes/T-106.md.

- **Earlier car detail with continuous train width** (T-107, Anita). Show separate cars from
  zoom 15 instead of 15.5. Both representations use the same width, including physical scaling
  at very close zooms; remove the detailed view's smaller minimum that made cars suddenly tiny.
  Preserve actual car lengths and route positions. notes/T-107.md.

- **Whole-line trains are the schedule input** (T-104, Anita). Choose an integer allocation
  across both directions for each demand level; derive headway from the capacity-adjusted round
  trip. Keep allocations through route and congestion changes, with zero stopping new departures.
  New lines start 8/4/2. Existing TWT2/3 rates stay intact until the player edits a schedule;
  TWT4 saves the integer inputs. Use continuous capacity delays on resources serving count
  schedules to avoid rounding-induced oscillation. No frame-time scheduling work. notes/T-104.md.
- **Four track display choices** (T-105, Anita). Combine line colour and traffic width into
  Line and Line with traffic thickness (default), alongside Height and Max speed. Speed colours
  show the geometry's per-section limit, not current train motion; curves and straights can
  differ on one edge. Height/speed use uniform-width bare track while trains keep line colours.
  Migrate the previous width checkbox preference and remember the chosen mode. notes/T-105.md.

- **Close-up trains bend car by car; quieter crowding colours** (T-103, Anita). Switch at zoom
  15.5 from a compact capsule to the line's actual 20 m cars. Each has its own route position,
  chord heading and line-bundle offset; keep one GPU instanced draw and the existing analytic
  trip timing. Clamp the full consist onto its route at terminals. Picking follows every car.
  Reduce the empty tint from 60% white to 18%, and darken the filled core from 62% to 48% of
  line RGB. The crowding gauge spans the whole consist rather than repeating in every car.
  notes/T-103.md.

- **Player fares affect riders** (T-067, Anita). Use the saved base-plus-km curve in demand,
  at a US value of time of $20/hour (3 perceived minutes per dollar, chosen for play). Charge
  the base once even with transfers; charge distance on actual track length, so route searches
  see it too. Keep fare outside the ride-time crowding multiplier. Fare-only edits start a new
  debounced solve; identical prices do not. notes/T-067.md.

- **See busiest-hour train crowding as well as averages** (T-087). Default train gauges to the
  busiest hour, matching the crowding model and the line's fullest-train statistic. Settings
  can show the period average; the inspector keeps both counts visible and uses the chosen
  basis for its load bar. This changes display only, not schedules or demand. notes/T-087.md.

- **Delay tags leave room and disappear at city scale** (T-102, Anita). Share station names'
  zoom-12 cutoff and spacing rules; show larger delays first when tags compete, then higher
  utilisation. Keep resource positions, tooltip explanations and junction clicks. notes/T-102.md.

- **Fewer station names, one UI font, no commuter corner summary** (T-101, Anita). Hide names
  below zoom 12 and reserve more space around them, particularly before zoom 14. Keep transfer
  priority and T-074's shared-layer pan. MapLibre's Helvetica container font was overriding the
  project's font on station labels and other map DOM text; explicitly use Zen Maru Gothic there
  and on form controls. Remove the commuter legend and its unused state: the top bar already
  shows mode shares, and hover/inspectors retain local numbers. notes/T-101.md.
- **All construction controls live in Build; Select has no button** (T-100, Anita). The blueprint
  total, undo/redo and Construct stay visible at the bottom of the Build pane, while settings and
  the grouped cost breakdown scroll. Esc and clicking the active tool still return to selection.
  Costs come from the simulation's pricing intervals, with stations, junctions, flyovers and
  crossings separate. Include the actual new fleet charge: starting lines together changes
  delays and train needs, so quote a reversible preview of Construct all, not individual planned
  train estimates. notes/T-100.md.
- **Station names pan as one compositor layer** (T-074). The outer box clips to the viewport;
  the inner box moves from the same camera matrix as the network. Keep a 256 px margin of
  names ready and re-place at 128 px, after movement ends, or when the projection changes.
  Retains DOM text and overlap priorities while reducing browser CPU during the 300-station
  drag benchmark by 34%. `?perfOff=labelPan` restores individual placement for comparison.
  notes/T-074.md.

## 2026-10-09

- **Capacity markers go under the demand views; their tooltip is a styled tag** (T-099, Anita: the
  markers showed over the bubbles and seemed to have no tooltip). The demand canvas moved to z 2 and
  the markers to z 1, so markers sit over the network and under bubbles and a station's catchment
  always (with no demand view there is nothing above them). The old native `title` only appeared
  after a second's still hover; the new tag appears at once, like the bubbles' count tag, in plain
  words: what is at capacity and how long each train waits at the demand level in force.
- **A network with no running line has an answer: nobody by rail** (T-099). New game after the
  real New York save kept the old bubbles because `cell_modes` (and `flows`) answered empty when
  there were no subzones, which the views read as "the workers moved on". Fixed at the source in
  `DemandApi` (no subzones means zero rail), not by special-casing an empty network in the views.
- **Bubble size slider: 0.5 to 2 times the width, default 1** (T-099, Anita). It scales the drawn
  radius on the GPU and the hit test, not the built bubbles, so dragging it costs nothing but a
  redraw. Rejected scaling `BUBBLE_DENSITY` (a rebuild in the worker per slider step).
- **The grey-to-blue colouring switch is gone** (T-099): Anita kept the RGB mix.

- **Anita after playing batch 7**: the RGB mode mix stays as the bubble colour (not grey to blue by
  train share); "Track" is fine as the tool name; trains hiding busy stations from clicks is fine,
  players can zoom in, so T-086 is dropped. Asked for: capacity markers under the demand views
  and a tooltip on them, a fix for stale bubble colours after New game, a bubble size slider
  (T-099).
- **Commuter bubbles replace the commuter dot map** (T-097, views agent; Anita after playing T-084:
  too many dots, and clusters in circles; she wants Subway Builder's view). Bubbles are the city's
  H3 res-9 cells summed into their parents by zoom (res 9 from zoom 13.6, 8 from 11.8, 7 from 10,
  then 6): the cells already carry H3 ids and a parent is the id with its lower digits set, so no
  H3 library and no new data; each step is 7 times the area, about 1.4 zoom levels, which keeps
  the spacing on screen steady. Rejected a square grid (would not line up with the cells) and
  keeping the dots at a coarser value (still dots, still the circles: those came from spreading a
  dense cell's dots over a disc wider than its share).
- **Bubble area is commuters at one fixed density per end, no minimum** (T-097): 25,000 a km² for
  homes, 80,000 for jobs. The same density at every level keeps a place's total area the same as
  the view aggregates. Jobs get their own because at the homes' scale Midtown's bubbles covered
  half of Manhattan. Rejected sizing per view (sizes would change as the player pans) and
  a minimum size (Anita: nobody, no bubble).
- **Bubble colour is the RGB mix of red car, blue train, green walk, scaled to full brightness**
  (T-097, Anita's latest words). Her first words were "how many take the train is the colour (if
  none, gray)": that is one switch away (`BUBBLE_COLOURING = "train"`, grey to blue by train
  share); asked her which. The top bar's split uses the same three colours.
- **Bubbles, a selection's far end and a station's catchment are drawn over the network** (T-097,
  Anita: "in demand seeing mode we want to see this stuff on top"), on the demand canvas moved
  above the network's; station names stay above. Bubbles at 80% so the lines still show through.
  Station catchment: the station's own cells opaque on top, the far end filled at 55% below them
  (her words), both without a minimum size.
- **Clicks: trains and stations first, then a bubble, then the rest of the network** (T-097). The
  bubbles cover much of the map, and a station must stay clickable for its own view. Rejected
  bubbles first (stations unreachable while the view is on).
- **Several bubbles by holding still 0.4 s and dragging a box** (T-097), Shift and click to add one.
  A box is plain to draw and to read; a lasso was the alternative, more work for little gain at
  bubble scale. A plain drag still pans.
- **A selection hides the other bubbles; a station selection hides all of them** (T-097): the
  far end's bubbles share the colours, and with the city's bubbles under them the picture was
  unreadable. A click on any place picks it anew, Escape or an empty click goes back.
- **The switch is "Commuters on the map: Off, Homes, Jobs" at the top of the settings pane**
  (T-097, Anita: it is a viewing setting, and she disliked "demand by home / by job"). Rejected
  "Where commuters live / work" as button labels (too long for three buttons in the pane); those
  words are the legend's title instead.
- **A selection's far end is summed into bubbles in the bubble worker** (T-097): a jobs selection's
  far end is ~78,000 home cells, 15 ms on the main thread; the worker does it in 8-10 ms off it.

- **The bubbles' "where do they go" comes from the gravity plus the kept station pairs, not a
  second mode choice** (T-098, `flows`). Everyone from the gravity's zone trips; rail is carried
  along the solve's station pairs (as `stationRiders` does); walk and drive are the rest. Totals
  are exact and match `cellModes`; far rail per place is a little blurred. Rejected: keeping rail
  per subzone pair from the solve (millions of pairs a period) or running the mode choice again
  for the selection (it needs each round's station table, and would not match the averaged rounds
  the rest of the views show). The answer leaves out the far cells holding the last 1% of the
  commuters and gives them as cut totals: a job selection's homes otherwise run to every cell in
  the region.
- **Hotkeys 1, 2, 3 and Esc; a delete tool** (T-096, gameplay agent; covers T-088). Anita: "lets
  also have hotkeys esc for exit construction mode (and also exit other modes, if there are any),
  1 for start drawing track, 2 for placing station, 3 for delete (for a blueprint, deletes it. for
  a constructed track, ask for confirmation to delete it and then delete it". Digits rather than
  T-088's letters, as she asked.
  - **Esc undoes one thing per press**: a pending question (Keep), a drag, a half-drawn route,
    then the tool, then the selection. With a route half drawn, the first Esc drops the route and
    keeps the tool on, as before, so a slip does not cost the tool and the level set for it; a
    second Esc leaves.
  - **Each tool button shows its key as a small muted digit** (and in its title); four buttons
    with digits did not fit the dock, so "Draw track" reads "Track" (its title keeps the full name).
  - **A number key opens the Build tab** (from any tab): the level and platform settings the tool
    uses, and which tool is on, are shown there; the inspector does not move with tabs (SPEC 8).
    The key of the tool already on does nothing, unlike clicking its button (which goes back to
    Select), so pressing 1 again mid-route cannot drop the route; Esc is the way out.
  - **The delete tool removes track, stations and flyovers**: a junction without a flyover is not
    a thing of its own, so a click there takes the nearest stretch of track; lines stay in the line
    panel. Constructed things ask in the hint area above the bottom bar (Remove / Keep, Esc =
    Keep), not in a dialog: SPEC 8 has no floating windows. The inspectors' Remove track and
    Remove station ask the same way for constructed things (they only had a tooltip).
- **Station reach is soft: the walk to a station weighs 2.0 a minute for 10 minutes and 8.0 after,
  with no edge the player sees** (T-090; Anita: "passengers should consider how much time it would
  actually take to walk to station ... and how much time it would take to drive"). Replaces the
  2.5 km hard edge of T-078. A station now loses riders where the whole trip gets worse than
  driving. The steep second part keeps a station's draw to about 3-3.5 km: with 4.0 after 10
  minutes a lone Manhattan line drew riders from 6 km and more (driving into Manhattan is slow and
  parking dear), and any bound cheap enough to compute cut that tail visibly. Rejected: a convex
  curve (harder to say in a sentence), a longer easy part (12 minutes: more pairs, rail 0.6 points
  higher, no better fit).
- **Two computational bounds instead of a rule**: stations within 4 km straight line, and a pair of
  places offered rail only if its two walks cost at most 80 perceived minutes together (T-090).
  Set where little rail is left (0.1-0.3% of the real network's rail in their last band; a lone
  Manhattan line loses 4% of its riders to them) and so the recompute stays near what the 2.5 km
  edge cost: subzone pairs 13.8M against 13.6M on the real network, a day 15-25% longer. With the
  4 km reach and no pair bound it was 2.3x the pairs.
- **Driving speed by trip length and density** (T-090): 1.3x the straight line; the first and last
  2 km on local streets (40 km/h in open country), the rest on main roads (80 km/h), both falling
  towards 15 km/h where dense (half way at 5,000 commuters living and working per km²). Replaces 28
  km/h everywhere, which made suburban driving 2-3x too slow and suburban rail 1.5-2x ACS. Now Long
  Island, Westchester and Connecticut are at ACS; the four big boroughs keep slow driving.
  Rejected: speed by trip length alone (city driving too quick, the boroughs' rail fell to 29%
  against ACS's 42%); density only at the ends with fast main roads everywhere (32%). No road
  network (T-072).
- **A subzone's walk to a station is the logsum mean of its cells' perceived walks** at the mode
  choice's scale (T-090), not their mean distance or mean cost: with a convex walk cost the plain
  mean put rail 3.3% under one subzone per cell; this is within 0.2%.
- **Line width shows riders an hour in the period in force, both directions together, by the
  square root** (T-084, views agent): 0.45 of a line width when empty to 2.2 at 30,000 an hour.
  The period in force ties the width to the trains moving on screen and to the train fill; per
  hour makes the five periods comparable. Rejected: the whole day (does not move with the clock),
  the busier direction alone (a flow map shows everyone on the stretch), linear width (the few
  busiest stretches swamp the rest). Lines sharing track are laid side by side again at their own
  widths, so a bundle widens rather than overlapping.
- **Station circles show the day's boardings plus alightings** (T-084), up to 2.6 times at 150,000,
  by the square root. Rejected the period in force: the demand result keeps station riders by day
  only, and a station's size changing five times a day reads as noise; the day matches the station
  inspector and the Stations tab.
- **The train fill is a gauge: half full when the seats are taken, full at crush, a red rim when
  over full** (T-084). Rejected: fill in proportion to crush (a train with every seat taken would
  look a quarter full), and colouring trains by load (a second colour scale on top of line colours).
  Same numbers as the train inspector (T-083).
- **The commuter dot map is switched from a button by the mode split in the top bar, its legend in
  the map's top left corner while it is on** (T-084). The dot map shows the goal number on the
  ground, so its switch sits beside it; the legend is docked like the bar at the bottom left, not a
  window. Rejected: the City tab (the legend would go when the player opens another tab with the
  dots still on) and the settings pane (hidden for the main demand view).
- **Train, walking and driving have fixed colours, and the top bar's mode split uses them**
  (T-084), in `game/palette.ts`: train #d81b60, walking #e8a33d, driving #958c80. The split was the
  theme's accent and two tints; keeping that would give the same three modes two colour sets, one in
  the bar and one on the map. Driving is a warm grey dark enough to read: a lighter grey vanished on
  the faint basemap and the map looked mostly train at 26% train.
- **Dots by the spatial carry along a Hilbert curve, coarser zooms as every 4th dot** (T-084;
  Anita's dot rule). 5 commuters a dot from zoom 11.8 in, then 20, 80, 320 every 1.4 zoom levels;
  each level is the same carry at a larger value, so minorities stay right at every zoom (rail at
  5% on Staten Island within 1%). Dots spread over each cell's disc by a generator seeded with the
  cell; a sunflower pattern per cell was tried and drew each cell as a rosette. Rejected: rounding
  per zone (drops a mode wherever it is under half a dot, Anita's minority bug), and quadrupling the
  value every zoom level (the city view went too sparse).
- **The dots have a canvas of their own, drawn only when the camera or their data changes**
  (T-084), with only the 4,096-dot runs in view drawn. Rejected drawing them in the network overlay,
  which redraws every frame while the game plays (0.3-3 ms of GPU a frame for nothing new), and an
  offscreen texture blitted each frame (more code for what a second canvas gets for free).
- **Dots are rebuilt for a solve's first estimate and again after its crowding rounds, not for each
  round** (T-084): the rounds between barely move the modes and each rebuild uploads 18 MB. The
  demand client's `cellModes` cache is now per solve and round (one line in the demand agent's file)
  so the second ask gets the refined answer.
- **A station's far-end places are named by the nearest station within 3 km, else by distance and
  direction** (T-084). Rejected neighbourhood names from the basemap: only loaded tiles have them,
  so a place's name would change as the player pans.
- **T-038 (heat map of rail share) is covered by the commuter dot map** (T-084): it shows the mode
  split by place at cell grain, which is what the heat map was for.

- **Clicks are points of intersection again** (T-092, gameplay agent; reverses T-079's "the track
  passes through the clicked points" below). Anita: "We switched to using a track placement method
  that runs the line through the points we click. I think this actually feels less natural and we
  should revert, sorry. Reverting will help in that it'll make our placement of a track not
  interfere much with the track we previously placed in previous clicks." Each click is a PI with
  an auto radius (T-023's drawing, its lead-in from existing track included), so a click changes
  nothing before the curve at the previous click; through the clicks, every click refitted the
  headings of its neighbours. `geom::through` is gone.
- **The previous click's curve may still tighten on a click** (T-092): the auto radius gives a PI
  next to an end the whole leg to it, so while drawing, the last curve can use all of the leg to
  the cursor, and halves to its share once the next point is clicked. Giving it only half would
  freeze it at the click, but the finished track would then differ from its preview or old saves
  would change shape; the pre-T-079 rule is kept. It happens only when the new point is less than
  half the previous leg away.
- **T-079 track keeps its clicks until it is reshaped** (T-092): edges drawn through the clicks
  (TWT3, tracks | 8) load with their fitted PIs, so they look as they did, and keep the clicks
  (`EdgeData::thru`, written back while the edge is unchanged); the reshaping squares are those
  clicks, and the first reshape or node drag makes them the edge's PIs with auto radius. Dropping
  the clicks on load was the alternative: those edges' fitted PIs have set radii that use their
  whole legs, so any drag would be refused. New saves stay TWT3; new edges never carry clicks.
- **The radius stepper is back in the track inspector** (T-092; T-079 removed it because radii
  followed from the clicks). With PIs a radius is a free choice again: the corner stays where it
  was clicked and the curve can be tighter or gentler.
- **A set radius a reshape leaves no room for goes back to auto** (T-092): a split (a station or a
  branch on blueprint track) sets the radii beside it so the shape does not change, and those
  refused every later drag of a neighbouring corner ("does not fit"). A radius the edit itself
  sets is still refused when it does not fit.
- **Dragging a blueprint node keeps a junction's or station's heading; a track end's follows its
  edge** (T-093, gameplay agent). Anita: "add a feature where we can drag around previously placed
  nodes that are just blueprints." Every edge end at a node must leave along its heading (SPEC
  6.2), so at a node with two or more ends the edges' first PIs (which hold the heading) move with
  it and the rest of the PIs stay; an edge with none to move gets a lead-in PI as drawing does
  (the next drag moves that one, so they do not pile up). Turning the heading to follow the drag
  was the alternative: it would bend every edge at a junction at once and has no natural rule for
  three or more ends. Movable only while every edge there is blueprint and its station is not
  constructed; a press on a constructed node pans the map as usual unless that node is selected,
  where the drag is caught and the hint says it cannot move.
- **No squares while a node is dragged** (T-093): the preview line and its curve labels show the
  new shape; a selected stretch's squares hide during the drag and come back at their new places.
  The moving node shows as a ring.
- **Running cost US$1.50 a car-km, down from 6** (T-091, Anita: "we can lower running costs a
  lot"). At 6, running costs beat fares even at 100x (the real New York save lost $0.65B a game
  day: fares $1.71B, running $2.36B), because both scale by the same factor. Lowering the price
  keeps one economy factor and one note in the Money tab; the alternative, a separate 30x factor
  for running costs, would leave a real-looking $6 on screen that the game does not charge. Now
  that save is about $1.1B a day ahead; a line run far beyond its riders still loses money.
- **Anita's answers on the batch 5 questions**: all the proposed demand views (T-084); driving is
  the alternative for the whole trip and getting to and from stations stays the fast walk only;
  station reach should be soft, riders comparing walk, ride and a simple drive time (T-090);
  drawing goes back to points of intersection (T-092, reversing T-079); blueprint nodes can be
  dragged (T-093). The T-085 click question (trains hide stations) was not answered; T-086 waits.

- **The fast walk: 15 km/h on a path 1.3x the straight line, reach 2.5 km, weight 2.0; New York's
  rail constant 0** (T-078, demand agent, batch 5), replacing the 1.5 km walk at 1.2 m/s (weight
  1.39), the drive/bus other leg and the constant 1.5. Chosen for game feel: a station's reach is
  a circle the player can see, a few stations wide in the centre, fading towards its edge. With the
  old constant every bike-like setting sent a third of New York by rail (the fast walk takes 10-20
  perceived minutes off each end); 0 puts the real network at 22% against ACS's 20%. Transfers
  between platforms stay a walk on foot. The suburbs stay 1.5-3x ACS (mostly the gravity, T-072),
  accepted: the game may be easier than real life. notes/T-078.md.
- **Demand views are asked for, not pushed** (T-078): per-cell modes (4.9 MB in New York) and a
  station's riders are computed by the workers on request for the solve on screen, typed arrays
  only; each solve's results carry only what the UI shows today. A query about a solve the workers
  have moved past answers null.

- **The track passes through the clicked points by keeping PIs underneath** (T-079, gameplay
  agent, batch 5): the clicks are stored beside the PIs (`EdgeData::thru`, save TWT3) and the PIs
  are fitted through them by biarcs with circle-through-neighbours headings, each arc a PI with its
  radius set. Storing the clicks as the geometry's input was the alternative; it would have meant
  a second fit in every geometry path (splits, saves, derive) and new rounding rules, while PIs
  keep every T-040 guarantee and every old save unchanged. Track from before T-079 shows its curve
  middles as squares and is refitted through them on the first drag.
- **No radius stepper on blueprint track any more** (T-079): with the track through the clicks a
  curve's radius follows from where the points are; the inspector lists the curves read only.
  Replaces T-064's per-PI radius stepper.
- **Junction waiting saved = the junction's whole waiting** (T-080): a flyover removes every
  conflict resource at the junction (SPEC 6.2), so the inspector shows the current waiting per train
  through it, at the demand level in force, and the price from a trial edit.
- **Undo, redo, the blueprint's cost and Construct stay on the map** (T-082): they are needed from
  every tab (planning a line, the inspector), so they are not in the Build tab with the tools.
- **Riders on board are an average train of the period** (T-083): period riders on the segment over
  the trains run in the period, as SPEC 2 says; the busiest hour is not exposed per period to the
  main thread.
- **The economy constant scales running costs in the track model and fares in money.ts** (T-081):
  both read `ECONOMY` (params.rs, sent in `money_params`); the fare and per car-km figures the
  player sees stay real.
- **Trains win a click over stations** (T-085, Anita): a train is drawn over its station.

- **Anita's answers after batch 4** (all Anita):
  - Realism is for rough calibration, not a goal; the game may be easier than real life, and
    country differences in transit use are not ours to capture. SPEC 1 priority 3 rewritten.
    Consequence: T-072 (road travel times, county factors) drops to P3; T-034 becomes a game-feel
    tuning of the fast walk (T-078).
  - One access mode: a fast walk at about bicycle reach, both ends; the drive/bus "other leg" is
    removed and buses are out of scope; driving may come back later. Replaces T-020's other leg.
    T-035 (other leg at the work end) is abandoned. Driving stays as the whole-trip alternative.
  - Faster-than-real trains are fine ("nyc train speeds in real life are an abomination").
  - Construct stays final; no free undo of building.
  - Economy at 100x: fares and running costs per day x100, like Subway Builder.
  - Build tools move into the Build tab; track passes through the clicked points; the
    flat/flying junction toggle was unclear (replaced by "Build a flyover" in the inspector);
    trains should show who is on board.
  - She wants to see demand soon and asked which visual schemes to consider; options proposed in
    chat, her pick pending.
  - Dock width and the tab/inspector divider are draggable and remembered (relaxes "nothing
    drags": still no floating windows). The panel's explanatory notes go; the drawing tooltip
    shows length, average cost multiplier and price instead. "Construct all" becomes "Construct
    blueprints". A train drawn over a station wins the click.

- **Idle frames are the default** (T-073): with no trips this hour the game draws no frames and
  the clock wakes once a game minute; an empty game went from 13% of a core to 0.6%.
- **The gravity's next fix tries road travel time before county factors** (T-072 note): OSM road
  times work in every country and also fix the too-slow suburban car times; ACS county flows exist
  only in the US, so K-factors are fitted to what remains.

- **CPU is measured on Chrome pinned to two cores, as medians of repeated windows, with A/B by
  URL flags** (T-045): `?perfOff=` turns a part off and `?perfTry=` turns on a candidate fix
  (`app/src/perfFlags.ts`), read once at load, no effect without the parameter. The machine is
  shared with other agents, so single runs swung 47-79% before. The fixes found are left for the
  owners of those files (T-073 to T-076). notes/T-045.md.
- **T-034 (access and rail constants) paused before changing anything; T-072 (county-to-county
  flows in the gravity) goes first** (demand realism agent). The real-network comparison showed the
  crow-fly gravity sending New Jersey residents to Manhattan at 1.9-4.3x the ACS rate and keeping
  too few in their own county (39% against 52%; LODES, the decay's target, 43%). Access constants
  fitted on top would absorb a gravity error. notes/T-034.md.

- **T-063 done without per-tile messages** (gameplay agent, batch 4): measured on a 2,000 km
  network, the edit cost was station labels rebuilt and measured every edit and stroke arrays
  built on the main thread, not the size of the message. Fixing those and building the render
  buffers in the clock worker took the main thread from 160-193 ms to 3-6 ms an edit; sending
  only what changed waits for 10,000 km networks (T-070).
- **The app runs three crowding rounds, not one; no crush cap** (T-022). After one round the
  segment that takes the overflow is at its most overloaded (the first round moves half of what
  the penalty pushes, all at once); by the third a player-sized network moves under 1%. An
  overloaded line should show: it is the player's signal to add trains. notes/T-022.md.

- **Two WASM modules, one per kind of worker** (T-051), replacing T-026's one module for both:
  with LTO each module keeps only the code its exports reach, so the demand module has no track
  model and the track module no demand kernel or `serde_json`, and nothing downloads twice
  (157 + 79 kB gzipped against 244 for one module, same source). The clock worker compiles a
  third less, each demand worker two thirds less. Demand solve times unchanged. Cargo feature
  `track-api` (default) gates `TrackApi`; profile `wasm` = release with one codegen unit.
- **Both modules stay at opt-level 3** (T-051): opt-level s would save 17 kB gzipped on the
  track module and z 41 kB, for edits and previews 1.2x and 1.8x slower; SPEC 1 ranks CPU and
  never freezing above download size. notes/T-051.md.
- **The capacity pass is incremental** (T-050), replacing "the capacity pass stays global": an
  edit rebuilds the resources around the edges, nodes and lines it touched (whole sections that
  reach them) and re-solves those plus the ones a rescheduled line uses; only lines owning a
  changed resource get their holds and profiles compared. Tested equal to the global pass on
  random edit sequences. 10,000 km: 0.2-1.2 ms instead of 10-13 ms per edit. notes/T-050.md.
- **Touching a station wakes the capacity pass even if no line stops there** (T-050): a line
  passing through needs its platform resource; before, it appeared at the next full pass.

- **The home end of commutes is LODES RAC (jobs by home block) x (1 - tract work-from-home
  share)** (T-054), replacing population x the same share: the home end of the jobs WAC counts at
  work, per block, so the Bronx no longer sends 23% more commuters than ACS counts. **The decay stays
  d^-1 exp(-d/29 km)**: the refit's optimum moved along T-006's ridge by less than its spread.
  notes/T-054.md.
- **Demand has walking transfers up to 300 m, after a ride only** (T-007; SPEC 6.1 promised
  them). A station node takes one route's track, so two lines crossing at different levels could
  not exchange riders at all; New York's real network depends on such changes. Walk edges start at
  route nodes, so a walk between two stations is never priced as a rail trip.
- **Snapshot stations within 100 m are one demand station** (T-007): two lines' platforms built
  side by side would otherwise each take one of a cell's 6 nearest-station slots and need a walk
  to change. Never two neighbours on one line. Results stay per route node, so the app is unchanged.
- **The real-network test is New York's GTFS network solved natively through `DemandApi`, compared
  per county (ACS B08301), per subway complex and link (MTA OD 2024, nycriders) and per operator**
  (T-007), rather than built by hand in the game: GTFS gives real run times and trains per hour
  for all 109 lines, and a change to any constant can be rerun in 10 s. A track-model save of the
  same network exists for looking at in the game. notes/T-007.md.

- **Side-by-side lines are offset in screen pixels in the stroke shader, ordered per edge by a
  network-wide "forward" direction, branch lines on their side** (T-062). Screen-space offsets keep
  the bundle the same width at every zoom, as metro maps do, and cost nothing on the CPU per
  frame; ordering against one agreed direction is what keeps two lines from swapping where an
  edge happens to be drawn the other way.
- **Trains are bought by the car, network-wide, by whatever edit makes the lines need more; short
  of money the edit is refused** (T-028). A per-line fleet would strand cars when a line is cut;
  one train kind means any spare car fits any line. Refusing matches Construct and keeps every
  running line fully served (no partial service to model). Cars are never sold, as track is never
  refunded, so undo and schedule changes can never be used to make money.
- **One fare curve for the network, base plus per km, transfers paying the base once** (T-028),
  instead of the shell's flat fare per line. SPEC 6.4 asks for a curve by distance; one curve is
  one control in the Money tab and one input for demand later. Demand ignores it for now.
- **Running cost per car-km (US$6), cars at US$2.5M, fares $1.50 + $0.10/km to start** (T-028):
  real-world magnitudes (New York R211 cars, NYCT cost per car-km), so the numbers on screen read
  true. Pace of income against construction is left to Anita (TODO.md).
- **Money settles by the game hour in the clock worker; fare income is computed on the main
  thread from demand's riders** (T-028). The worker owns the cash (construct checks it); riders
  live on the main thread. The worker gets the clock as (time, rate, wall time) on every change
  rather than a message per tick.
- **$6B start kept** (T-028 recheck): one real 20 km tunnelled line with its trains is about
  US$4.2B now, leaving room for a short second line.
- **Saves are built, compressed and written to IndexedDB in the clock worker** (T-029): the main
  thread never serialises. Format TWG1 = JSON header (clock, money, fares, ledger) + the track
  model's TWT2 bytes, gzip. Autosave after anything paid for, else once a wall minute when
  something changed, and on page hide; the last game loads on start (`?new=1` skips it).

- **Constructing, and removing or changing anything constructed, is final: it clears the undo
  history** (T-055). Undo would otherwise rebuild removed track or unbuild paid track for free.
  Blueprint edits, lines and schedules stay undoable. Making a constructed junction flying, or a
  constructed station's platform longer, is paid at once like a construct.
- **A line runs only when every edge and stop it uses is constructed** (T-055); before that it is
  "planned": its stop times show, it loads no capacity, has no trains and is not in demand's
  snapshot. A junction's cost is paid when its third edge is constructed, a flat crossing's when
  its second is.
- **The drawing preview runs in the clock worker as a trial edit** (apply, measure, roll back),
  not on the main thread (T-023). One code path gives the preview every check an edit gets
  (crossings, ports, platforms, water) and it costs 0.4 ms a round trip; the main thread would
  have needed its own copy of the geometry and the 1.1 MB water mask. Replaces "previews the one
  alignment being drawn itself" in SPEC 6.5.
- **New track leaving a node gets an automatic PI on the node's heading** (T-023), placed so the
  first arc starts at the node. The player cannot click a point exactly on the heading, and a
  branch whose first stretch runs straight on the main line overlaps it.
- **Station names from the nearest basemap street within 400 m**, shortened New York style, a
  name in use skipped; else a neighbourhood; else "Station N" (T-024). Renamable.
- **Station names beat basemap labels through an invisible symbol layer of the same names** above
  the basemap (T-032): MapLibre's own collision drops the place names underneath, with no per-frame
  work and no filter edits.
- **The clock worker sends the whole network state after every edit** (T-025): simple, 4.7 ms on a
  small network. Per-tile updates (SPEC 6.5) wait for big networks (T-063).
- **New lines: "Line N", the next unused palette colour, 12/8/4 trains an hour, 30 s dwell, 3 min
  turnaround** (T-024). **A new game starts on day 1 at 07:00** with $6B and an empty city.
- **The track save is format TWT2**: constructed flags on stations and edges (T-055). No saves
  existed yet.

- **Demand runs in a pool of up to three workers, the five periods split between them by index**
  (T-026, T-021): half the logical cores less one, capped at 3. Three land the mode split in two
  period-times (0.72-0.83 s in Chrome) and leave the machine to the browser and everything else;
  five would halve that for five busy cores per edit. Mode choice is not split by origin zone: at
  three workers it saves a sixth and needs a reduce step every round. notes/T-026.md.
- **One WASM module carries the track model and demand** (T-026): the app builds with feature
  `demand`. Both kinds of worker load the same file, so two packages would download the track
  model twice; demand adds 74 kB gzipped. The T-005 benchmark export moved behind `demand-bench`.
- **Each demand worker runs a warm-up solve on a small network after opening the city** (T-026).
  V8 optimises a wasm function only for later calls, and the pair loop is one long call, so the
  first real solve took 3x as long (1.6 s to the split instead of 0.74 s).
- **A day's commutes: a trip to work and a trip home per commuter, shared over the five periods
  70/12/6/4/8% and 2/18/58/16/6%; trips home reuse the morning's station pairs reversed** (T-026).
  Judgement until ACS departure times are fitted; reversing keeps one mode-choice pass per period
  and is right for park-and-ride. Replaces the spike's 45% / 40% shares of a 3-hour peak.
- **Of trips not taken by rail, a share walks by zone distance (84% within a zone, 32% next door,
  6% two zones away)** (T-026), giving 5.9% walking with no rail against ACS's ~6%. The bar needs
  a walking figure and the kernel had none.
- **Trains carry 44 seats and 160 at crush per 20 m car** (T-026), a New York subway car, times
  the line's cars. Replaces the hand-made network's per-kind capacities.
- **The game clock's demand level reads the shared period table** (T-026): `demandAt` in
  `game/clock.ts` had its own hours (high 16-19, low 23-6) that disagreed with the schedules'
  16-20 and 0-6.
- **Riders show blank until the first solve lands, and the bar shows the no-rail split** (T-026),
  rather than the demo's made-up numbers.

- **Basemap labels in Zen Maru Gothic Regular, with Noto Sans merged into the same glyph files**
  (T-056). MapLibre fetches one file per font stack and range, so a static host cannot fall back
  from one font to the next in the client; each file holds Zen Maru's glyphs plus OpenFreeMap's
  Noto Sans Regular for codepoints Zen Maru lacks, so Arabic, Devanagari and the rest still draw.
  Regular only: Zen Maru has no italic, and Positron's bold country and capital names stay quieter
  in Regular. No CJK ranges: MapLibre draws kana, kanji and hangul in the browser and never fetches
  them. Made with Stadia Maps' `build_pbf_glyphs` (Rust, installs with cargo on Windows and
  combines font stacks itself); MapLibre's font-maker is not published on npm. notes/T-056.md.
- **The basemap style is rewritten before MapLibre commits it** (`setStyle` with `transformStyle`
  in `map/basemap.ts`) rather than repainted on `style.load`, so fonts and colours never flash.

- **The game is called anitabuilder** wherever a player sees it, lowercase (Anita: "silly name but
  i think apt and definitely not ai-y"). trainworld stays the internal name (folder, crate,
  packages, serve.py, code), so nothing is renamed on disk.

- **Build as a blueprint, pay on Construct** (Anita). Drawing is free and undoable; clicking
  Construct pays and builds. Settles the "should undo refund in full" question from T-040: undo only
  exists for blueprint edits, so nothing is refunded by undo.
- **Removing constructed track refunds nothing** (Anita). Replaces T-008's half refund.
- **Station names and, if feasible, basemap place names in Zen Maru Gothic** (Anita), via our own
  glyph tiles for the basemap (T-056).

- **Themes are called pink, blue and green** (were cream, sky, matcha; Anita). A remembered old
  name is mapped to the new one.
- **Space bar toggles pause and resumes the speed that was running** (Anita, T-046).
- **Nothing selected leaves the inspector blank**, no "Nothing selected" text (Anita, T-048).
- **Trains: rim blended into the core, heading along the chord between the train's ends, quad
  1 px larger** (T-047). The "wiggle" was the hard rim-to-core edge stepping a pixel at a time
  (MSAA does not smooth shading) and, zoomed in, the heading turning in steps at each 50 m
  sample. Core flicker halved, turning jitter at zoom 14 p95 1.45° -> 0.51°, GPU cost unchanged.
  notes/T-047.md.
- **Flat crossings are derived from geometry, not stored as nodes** (T-040). A same-level
  intersection of two edges becomes a crossing record (edge, chainage, edge, chainage) that costs
  money and loads capacity. As nodes, every drag of a PI across a track would split and re-merge
  both edges and rewrite the line paths through them; derived, saves and paths never change.
  Replaces "nodes: end, junction, flat crossing, station" in SPEC 6.2.
- **Keyframes are shared phase tables** (T-040). One table of constant-acceleration phases
  (t, s, v, a) per line, direction and demand level; a trip is (profile, departure). Replaces
  T-002's one (offset, speed, time) keyframe per train, which cannot represent stops and
  acceleration; T-025 integrates it (format in notes/T-040.md). A block simulator can still emit
  per-train phases in the same shape.
- **Train performance: 1.0 m/s² to 60 km/h, 0.5 m/s² above, braking 0.9 m/s²** (T-040). Typical
  EMU starting acceleration with the fall-off of a power limit, as two constant rates so every
  phase stays closed form; 0.9 is normal service braking (1.1-1.3 is the usual maximum). 1 km stop
  to stop: 67.5 s.
- **A hold is a slow approach when shorter than the cost of stopping there, else a stop** (T-040),
  instead of a fixed 30 s threshold: the realised delay then always equals the computed one.
- **Edits are refused only for problems they add** (T-040), so a save that a newer water mask
  makes partly invalid can still be edited and fixed. **Ground level over water** means within
  half a level (4 m) of the ground, so ramp ends count. A station whose track was deleted stays
  (its lines are broken) rather than blocking the delete; a node where two ends leave the same way
  with nothing behind (a junction whose stem was removed) is allowed.
- **The capacity pass stays global** (T-040): 1.5 ms at 2,000 km, 10 ms at 10,000 km and 400 lines,
  skipped when no line, schedule, crossing or junction changed. Incremental resources are a
  follow-up if large networks need them.

- **Gravity decay f(d) = d^-1 x exp(-d / 29 km)** (T-006), replacing exp(-d / 9 km). Fitted to
  LODES OD 2023 for New York on zone-centre distances: no exponential fits (its best is the old
  9-10 km, 15% of trips in the wrong 2 km band, 100x too few trips at 100 km), the power x
  exponential form misplaces 2.9% and lifts the county-pair log correlation from 0.83 to 0.88.
  The pack's `decay` field gains `"pow_exp"` with `decay_pow` (still format 1; readers refuse a
  decay they do not know, and the Rust reader knows both). notes/T-006.md.
- **The gravity balances on commuters, not residents and jobs** (T-042, from Anita's rule in SPEC
  4.1). Home end: population x (1 - the tract's ACS 2019-2023 work-from-home share, B08301). Work
  end: each LODES sector's jobs x (1 - its industry's work-from-home share among the region's
  residents, B08126), a published split rather than a uniform cut, so Midtown loses more commutes
  than the Bronx. Shipped as `commute_home`/`commute_work` per cell; `pop` and `jobs` stay true
  counts. New York commutes 10.18M -> 8.61M a day. ACS from the table-based summary files, since
  the Census API now wants a key. notes/T-042.md.
- **Preact 11 with @preact/signals for the UI** (T-027), as SPEC 7 proposed; nothing argued
  against it. Signals let the clock re-render one text node a game minute; Preact plus signals
  add about 10 kB gzipped. No Vite plugin, TSX through tsconfig.
- **The main thread keeps a `WorldView` snapshot (the save's inputs plus worker results) and edits
  only through functions that will become clock-worker edit ops** (T-027). Fake data lives in
  `game/demo.ts` alone, so swapping in worker snapshots touches no UI code.
- **Settings sit behind a gear at the end of the tab row and show in the tab area** (T-027,
  T-037): a pop-over would float, and a sixth named tab would crowd 304 px of tabs. The
  inspector stays where it is.
- **Station names are DOM labels above the overlay, not a symbol layer** (T-027): under the
  overlay, track was drawn through the names. Costs collision with basemap place names (T-032).
- **Clock speeds 60x, 240x, 960x game seconds per wall second**, provisional (T-027): normal is
  a game minute a second as in the mock.
- **The network is drawn on an overlay canvas, the only mode; the custom-layer mode is gone**
  (T-015). The overlay's one-frame lag while panning came from our animation-frame tick running
  before MapLibre's frame callback; drawing the overlay from the empty camera layer's render
  callback, in MapLibre's frame with its matrix, removed it (0 frames off against every drag
  frame off by one step, by camera probe and by composited screencast frames). Overlay keeps its
  CPU saving: 17-21% of a core playing against 35% for the custom layer. Trains sit above place
  names, which Anita accepted. Replaces "our own WebGL2 custom layer" in SPEC 7. notes/T-015.md.
- **Home care jobs spread over where people live: accepted** (Anita, T-014).
- **People who work at their own home are left out of commute demand** (Anita): LODES counts
  remote workers at the office, so trips scale down by the ACS work-from-home share (T-042).
- **County-government and school-district head-office lumps: a low-priority task** (T-043; Anita).
- **Spatial grain matters most in city centres; suburbs may be coarser** (Anita). 202k cells and a
  12% slower recompute are fine for now; adaptive cells only if performance needs it (T-044).
- **Blocks are spread over their polygon by land area** (T-013): 50 m pixels aligned with the
  water mask, each weighted by its land share, each to the res-9 cell of its centre; blocks too
  small to catch a pixel centre keep their internal point. Replaces "a block's whole population
  and jobs at its centroid". New York goes from 74,920 to 202,337 cells and recompute costs 12%
  more native; Anita: fine, grain matters most in centres, adaptive grain only if ever needed
  (T-044). Polygons from the per-county TIGER2020PL block zips, not the state TABBLOCK20 zips
  (73 MB instead of 510 MB). notes/T-013.md.
- **LODES head-office lumps: three rules** (T-014; the home-care rule accepted by Anita).
  Northwell's 41,521 jobs at North Shore University Hospital: keep 10,000, spread the rest over
  the population of the six counties where it ran hospitals. JetBlue's 10,300 transport jobs at
  its Long Island City office: keep 1,300, the rest to JFK. Blocks with 1,000+ jobs, 80%+ health
  care and at most 35% earning over $3,333 a month are home care agencies: their health care jobs
  go over the county's population (300,188 jobs in 109 blocks). The earnings split between
  hospitals (60-100% high earners) and agencies (35% or less) is clean, which is what makes the
  last rule safe without naming each employer. County-government and school-district lumps are
  left for T-043. notes/T-014.md.

## 2026-10-08

- **+2 is 1.3x and +3 is 1.8x** (were 1.4 and 2.0), so tall viaducts are less outpriced by deep
  tunnels. Updated in SPEC 6.4 and `sim/src/track/params.rs`. (Anita)
- **The stops list shows time from the start of the line, next to each station**, not the time
  between stops. Corrects the reading in the entry below. (Anita)
- **Station access forgiveness stays open until New York is playable**; Anita finds a third driving
  or bussing to stations high but wants to judge by playing.
- **Water mask from TIGER for M1, 25 m runs per row, in pack format 1** (T-030). TIGER AREAWATER
  plus the sea outside the state polygons is one small zip per county; OSM's sea polygons are a
  906 MB global file with no regional fetch, so they wait for the world pack (T-041). Runs per
  row (u16 bounds, a u32 row index) give a binary-search lookup and 1.11 MB for New York; 25 m
  rather than 50 m (0.67 MB) because track is built along shores. Added as `per: "water"` arrays,
  which every reader already skips, so no format bump. notes/T-030.md.
- **Water narrower than ~50 m does not count** (T-030): a railway crosses creeks and ditches on a
  culvert or a short bridge at grade, and requiring a viaduct with two 200 m ramps for each would
  make suburban track absurd. A 2 x 2-pixel opening of the 25 m mask; never adds water.
- **Anita's answers on the T-008 design and the T-009 mock** (all Anita):
  - Double-track routes drawn as one stroke are the default; single track is a setting. Closes the
    T-008 open question.
  - Track cost = US$90M/km x a level multiplier: +3 2.0, +2 1.4, +1 0.8, 0 0.3, -1 1.0, -2 1.1,
    -3 1.2. Replaces T-008's per-level price table. Only the selected level's multiplier is shown.
  - No track at ground level over water; bridges and tunnels over water cost 2x. Replaces T-008's
    "5x at grade, 1.3-2x otherwise".
  - Scheduling rows are high, medium and low demand under a "Demand" header, always all three; the
    "same all day" shortcut is dropped.
  - The stops list shows running time from the previous stop, plus the round trip; replaces a lone
    round-trip figure.
  - Display settings: track colour by line or by height; show/hide trains, labels and the rest.
  - Inspector shows any selection and is empty when nothing is selected; its position must not
    move between tabs.
  - Themes cream, sky and matcha all kept, dark later; Zen Maru Gothic; no decorative heading
    squares.
  - Overlay canvas for trains is preferred (trains above place names are fine) if its lag while
    panning can be removed; otherwise layer mode (T-015).
  - Transit-share heat map later.
- **The city pack ships the solved zone gravity; pack format 1** (T-019). A zone id per cell, grid
  position and two balancing factors per zone, and a `gravity` header block. Opening New York in
  WASM went from 8 s (solve) to ~7 ms (parse and rebuild). Format 0 dropped; nothing else read it.
  notes/T-019.md.
- **The pipeline solves the gravity with the sim crate's own Rust code** (`pack_gravity` bin)
  rather than a numpy copy (T-019): the game rebuilds trips with the same decay table, so the two
  cannot drift, and T-006's refit will be made once. Cost: building a pack needs Rust installed.
- **Access: a logit over access options and egress stations, rail priced at the logsum**
  (T-020), scale 0.2 per perceived minute (nest ratio 0.25 against the mode logit's 0.05).
  Replaces all-or-nothing best station pair, which made station counts lumpy; the logit halves
  their error against the per-cell reference. notes/T-020.md.
- **The other access leg at the home end: 30 perceived minutes + 2.5 x time at 25 km/h x 1.3, to
  the 4 nearest stations within 20 km** (T-020). Picked so walking stays nearly alone within 1 km
  (20 minutes + 2.0 had 14% of people 500 m from a station driving to one) while long trips can
  park and ride; judgement until fitted. Work end walk only for now.
- **No walk bands in access subzones by default** (T-020): with the logit, zone x nearest station
  is as accurate as the old 5-minute bands were (station counts 5.5% off, segments 2.3%), and the
  other leg's extra subzones fit in the old recompute budget only this way (5.7M pairs against
  26.5M). Replaces 5-minute bands.
- **Train keyframe times are relative to an hourly render epoch** (T-016). The float32 time
  uniform stays under 3600 s; each game hour every train's (offset, time) is advanced to the new
  epoch in float64 from its master keyframe and re-uploaded, speed and line buffers untouched.
  With absolute time, positions were off by 1.5 m (median) at day 30 and 20 m at day 365;
  re-based they stay within 2 cm at any day. An hour keeps the uniform small and the re-base rare
  (~1 µs per train). Replaces "reference time uploaded once". notes/T-018.md.
- **Unfocused but visible: 10 fps with the clock running; hidden: no frames, clock frozen**
  (T-017). Stopping on blur would freeze a map on a second monitor; 10 fps costs 3-9% of a core
  against 17-36% at 60. notes/T-018.md.
- **MapLibre 6: our matrix is the custom layer's `defaultProjectionData.mainMatrix`** (mercator to
  clip, float64), and overlay mode gets it from an empty custom layer, since `map.transform` is no
  longer public (T-018). It is also the camera MapLibre actually drew that frame with.
- **Track geometry is straights and circular arcs, drawn as points of intersection** with an auto
  or player-set radius at each. Clothoids change only a second or two per curve and are invisible
  at map scale but cost Fresnel maths and inexact parallel tracks; splines vary the speed limit
  along every curve and make the minimum radius hard to enforce. Replaces "arc + straight or
  clothoid". notes/T-008.md.
- **Curve speed a_lat = 1.1 m/s², minimum radius 100 m, M1 top speed 160 km/h.** 1.1 matches LGV
  design (7,000 m at 320 km/h) and New York's 140 m curve at 25 mph; 100 m lets a Manhattan corner
  be turned at 38 km/h. The track itself has no speed cap.
- **Levels 8 m apart, max grade 4%, so 200 m of ramp per level**, ramps centred between PIs of
  different levels. 8 m clears one track over another with wires; 4% is the usual metro limit.
  Clearance needs a full level of height difference.
- **Topology: double-track routes drawn as one object, single track as an option, flying junction
  as a toggle**, assumed until Anita decides (open in SPEC 12). Every junction then has a known
  layout, so its delay can be explained in one sentence; individual tracks (NIMBY) mostly add ways
  to build tangles.
- **Lines store their exact path** (edges with a direction), not just stops, so new track never
  reroutes a line silently.
- **Capacity from occupancy times and a Kingman-style delay curve**, `W = 0.2 x mix x s x rho/(1-rho)`
  to 90%, straight above, plus an hour's fluid queue past 100%. Track sections `90 s + v/0.6`
  (33 trains/h at 40 km/h to 15 at 320, matching real metros, commuter lines and the Shinkansen);
  junction crossings and flat crossings 90 s; one line alone has no queueing delay. Delay is a hold
  at the resource's entry in the run-time profile, so trains visibly queue. Computed in the clock
  worker in one pass (frequencies are inputs, so no iteration), not with the demand as SPEC 2.1
  said before.
- **Costs: one worldwide price list per route-km**, at grade 20, viaduct 60-100, tunnel 90-170
  US$M, stations separate; water 5x at grade but 1.3-2x for viaducts and tunnels (a reading of
  "water ~5x" that Anita should confirm, open in SPEC 12). Calibrated to the Transit Costs
  Project's non-US averages. **Starting money $6B**: one real first line.
- **Edits are operations owned by the clock worker** and dirty only touched tiles, crossings,
  resources and lines, plus one step to lines sharing a resource; demand wakes only for service
  changes (stops, frequency, or a station-to-station time moving 5 s or more). Saves hold PIs, not
  geometry: ~170 KB compressed for 10,000 route-km.
- **Line colours are a normal saturated transit palette, player-settable; only the UI is styled.**
  Replaces "pastel line colours" from the earlier look decision. (Anita: "the only things we should
  be making stylistic decisions for are ui".)
- **The 10k-train target is a simulation target, not a rendering one.** Anita: most trains are
  local to a city and not drawn when zoomed out. Spec priority 2 now measures demand recompute per
  city instead of frame time.
- **Move to MapLibre 6** (Anita agreed); T-018, done early in M1.
- **Demand on two levels (from T-005)**: doubly constrained gravity between 2 km zones, by
  distance only, solved once per city; mode choice between access subzones (zone x nearest
  station x 5-minute walk band), ~11 per station. Replaces the spec's cell-to-cell pairs: 5.6-23
  billion pairs, distance truncation still leaves billions, and even served cells only are
  90-180M pairs growing with network coverage. Subzones cost 15-21M pairs, under a second per
  period in WASM, and stay within 3% of exact per-cell access on segment loads. Trip
  distribution does not react to the network (no logsum feedback), so an edit never reruns it.
  notes/T-005.md.
- **Train drawing layout (from T-002)**: each line resampled every 50 m into a float texture,
  one 16-byte keyframe per train (line, offset, speed, reference time) uploaded once, position
  computed in the vertex shader; coordinates relative to a city origin, with the matrix composed
  in float64 on the CPU. Our per-frame JS stays ~0.05 ms at any train count, so train count is
  a GPU question only. Custom layer vs overlay canvas is left open (T-015). notes/T-002.md.
- **New York's boundary is the 2023 MSA plus Fairfield CT, Dutchess and Orange NY** (25 counties,
  21.7M people). The added three each end a commuter rail line a player would expect; Mercer NJ
  and south stay out for Philadelphia. Details in notes/T-003.md.
- **Pack cells take a block's whole population and jobs at its centroid** for now; spreading big
  blocks over their area is T-013.
- **Accounts and shared saves on shonei-server**, as the plan for now (M4). (Anita)
- **Anita installs only rustup + the wasm32 target**; agents install the wasm build tools
  (wasm-pack or wasm-bindgen-cli) themselves in T-001 via cargo.
- **Project docs: CLAUDE.md + SPEC.md + TASKS.md + ARCHIVE.md + DECISIONS.md.** Anita expects the
  project to run long and does not want beads; she asked whether agents can keep docs current and
  archive finished work. Spec is rewritten in place, history lives here.
- **No signals for now; tangles discouraged by junction and segment capacity.** Keeps every train
  analytic, which is what makes 10k trains cheap. Capacity overload adds delay to the lines through
  it, computed in the background. Signals must stay possible later: track is a directed segment
  graph and the renderer/demand consume keyframes, not `position = f(t)`. (Anita)
- **Tiers 0 and 1 only; tier-2 OD becomes a check.** Subway Builder turned out to assign workplaces
  by gravity too, so tier 1 is already at its level. (Anita)
- **Long-distance demand between zones, not city centroids.** Two neighbouring cities must exchange
  far more short trips across their shared edge than long ones between far ends. The local and
  long-distance layers share one distance curve across the city boundary. (Anita)
- **Several long-distance stations per city are normal** (Tokyo, Kansai). The network is one network;
  each trip picks its best station pair. (Anita)
- **One train kind.** A later kind would differ by speed profile and cost (HSR, light rail), never by
  urban vs intercity. Scheduling is per period for every line, with a "same all day" shortcut.
  Replaces the first draft's urban/intercity kinds. (Anita)
- **Towns can become full cities mid-game** once their pack exists, mainly for development. (Anita)
- **Rust → WebAssembly for the simulation and demand core**, TypeScript + MapLibre for the app.
  (Anita agreed)
- **Start in the Northeast US, New York first.** Familiar to Anita; LODES, NHTS long-distance OD and
  nycriders give checks. (Anita)
- **Mode split bar as the goal number; one bubble per city in the world view.** Subway Builder's
  city-wide "80% drive, 16% train" readout is what Anita likes in it. (Anita)
- **Look: few colour tokens, light and soft; no ink-on-paper styling.** Anita found the paper idea too
  statementy; fixed dock, square buttons and pastel line colours stay.
- **Name: trainworld for now.** Anita dislikes it; a better one is welcome.
- **Flows, not agents; trains as functions of time.** Subway Builder's own changelog blames per-group
  passenger work for its lag and freezes; NIMBY Rails simulates individual riders and slows when
  queues grow. See `research/`.

- 2026-10-11 (T-143, Anita feedback): replace the all-city navigation list with the viewed city and its five largest external corridors by all-mode demand. Label the existing corridor totals as both directions. Preserve demand selection while its first result loads; use stable purpose colours in a share pie and departure plots. T-149 separately tracks startup gating and city scheduling.

- 2026-10-11 (T-150, Anita broader school-data request): preserve NYC DOE building totals inside the five boroughs and use official NCES EDGE/CCD 2023-24 public-school locations/enrollment elsewhere in the NYC metro. Spatially exclude the DOE area to avoid duplicates; expand estimated pupil homes across all metro counties. Keep university inputs separate from child-origin/school-day assumptions. National source reuse is implemented; wider city rollout and university attendance remain follow-ups.

- 2026-10-11 (T-153, Anita short-trip feedback): reserve the distance-based walking share before rail and road compete. Attractive rail may win the non-walking remainder, but it must not erase trips that the simple model already considers walkable. Keep the existing 2 km commute-zone grain as an explicit precision limit; everyday purposes continue to use sampled cell distance.

- 2026-10-11 (T-154, Anita compact intercity feedback): halve both detailed and regional longer-trip bubble radii while preserving area proportionality. Use 5 px connection bars and show train/all trips in the city list; keep percentages and expanded counts in tooltips.

- 2026-10-11 (T-144, Anita station-rendering feedback): ordinary stations are track-aligned platform boxes. Rider magnitude is a separate, off-by-default analytical layer of large translucent circles behind the boxes; those circles do not change picking.

- 2026-10-11 (T-145, Anita track-placement feedback): draw the live track placement ghost as a solid, dark-cased stroke. Its body follows Height or Max speed display colours and is white in Line/Traffic modes; invalid geometry stays red. The placed blueprint retains its normal dashed, faded construction state. Preview speed colours come from the track model's fitted geometry rather than main-thread inference.

- 2026-10-12 (T-152): model university attendance as a separate adult-origin purpose. For the first US layer, use one IPEDS 2023 budget per active degree-granting reporting UNITID, count degree-seeking undergraduates and graduates, subtract exclusively online students, and never replicate a system total across child campuses. Treat attendance, adult homes, decay and timing as explicit provisional assumptions.

- 2026-10-12 (T-155): label demand-derived riders, boardings and loads as steady-state forecasts. Animated train movement selects the current segment but does not observe passengers. Any adoption ramp must affect assignment, mode share and revenue together rather than scale only the display.

- 2026-10-12 (Anita): avoid city-specific data. Use sources that cover every city or every country; a one-city source only when it is very easy to get and use (NYC DOE building enrollment probably was not worth it), and never one for a minor city. Recorded in CLAUDE.md "Working here".

- 2026-10-12 (T-157, Anita track tooltip request): resting the pointer on track for 0.4 s shows what the track display draws there, in the demand views' tag. The traffic view states the hovered line's daily forecast on that stop-to-stop stretch with both directions summed, matching how its width is built, rather than a per-direction pair or the width's current-period hourly rate (Anita asked for daily traffic). Counts round to the nearest 100 above 1,000, since they are forecasts. Height states the level the colour shows, by the Build tab's level names. Select tool only: drawing, station and delete tools have their own tags. Bubbles, capacity tags, editing handles, trains, stations and junctions above the track keep it hidden, following the pick order.

- 2026-10-12 (T-149, Anita's viewed-city-first request): the demand pool loads only the packs a game needs: the viewed city (`activeCity`, nearest centre to the camera; the camera starts on New York and saves hold none, so the initial view is unchanged), every city with a built station inside its footprint or within 4.5 km of it (the walk's 4 km bound plus a margin), and the regional market always (the City panel's corridors need it). Any other city has no rail access, so it adds nothing to loads, fares or crowding; a worker check found every line and New York figure bit-identical with one pack or nine. Blueprints do not count, since demand never sees them. Opened packs stay open.
- 2026-10-12 (T-149): readiness is the loaded game's network solved to the last crowding round for every city it needs, not the free-flow estimate. Constructed cities other than the viewed one are part of it, because shared crowding couples them; a solve waits for every needed pack rather than publish without one. At startup a loading screen takes all input and the clock is held until ready; a later file, slot or new game holds the clock only. A pan to an unloaded city (necessarily one without construction) loads it without blocking or holding the clock, since it changes no number elsewhere, then solves again to give it a view.
- 2026-10-12 (T-149): the loading percentage is work done, weighted 10 download / 30 opening / 60 solving (judged from measured startups), never decreasing and capped at 99 until the result is published; it stalls rather than invent progress. On failure at startup the screen says why and offers Try again or Play without demand.

- 2026-10-12 (Anita): intercity demand need not be closely calibrated; its design is early and undecided. Recorded in CLAUDE.md "Working here".

- 2026-10-12 (Anita): track tooltips show after 100 ms of rest, not 400 ms, so they feel responsive (`DWELL_MS` in map/trackHover.ts).

- 2026-10-12 (Anita, answers to the day's questions): no street-address geocoder for now, neighbourhoods and cities are enough (T-177 stays OSM places only); a station complex stays display-only in demand for now (T-161 not yet); the single US border factor's side effects are accepted (Center City to Camden 0.39x, Fairfield to Manhattan 0.25x, LIRR 1.16x); the playable-while-loading gate stays, with a centre-stage loading indicator until the clock starts (T-180); no new cities until the general design settles.

- 2026-10-12 (Anita): only she has saves, so save compatibility is not a constraint until she says otherwise; T-129 (offer converting old saves) abandoned. Opening ramp redesigned (T-156): no ramp in demand, riders use a new line at once; only the commuter bubbles' mode colours catch up over about one game day, as if each person takes one trip a day. This supersedes the T-155 rule that a ramp must move assignment, mode share and revenue together, because there is no ramp in the model now, only in what the bubbles show. Zoomed far out, every city condenses to one longer-trips bubble, the focused city included (T-186).

- 2026-10-12 (T-156): the bubbles' catch-up blends trip legs per cell, old times (1 - f) plus new times f, so a bubble's size moves with its colour and both agree with its hover tag; f is the share of a day's trips departed since the result landed, by the commute's period profile (36/15/32/10/7%, docs/demand.md 4.2), on the game clock. The commute profile stands for all purposes since it is the one the model states for every city; a flat 24 h ramp would move the bubbles as much at 3 am as at 8 am. Hover tags read what the bubbles show; the inspectors, a bubble selection, its far end and the top bar stay the current forecast, since they report the model and blending them would contradict line riders and fares. Results landing while the clock is held, the first result, and purpose or city changes show at once. Redrawn in the bubble worker every 0.01 of f, at most four times a second, and only while catching up. The city overview's bubbles catch up the same way; a selection's longer-trip bubbles do not, matching its inspector.
- 2026-10-12 (T-186): the city overview starts at zoom 8.5 and below, where the metropolitan outlines are fully drawn and neighbouring cities are on screen; at 9.5 (the selection's per-city threshold) New York still fills the map, too soon to lose its own bubbles. Sizes come from the market's corridors (both directions; the City pane's figures), so Homes and Destinations look the same there. The overview's largest bubble is 28 px before the size slider, larger than T-154's 19 px city circles, because it has no local bubbles to stay under. A selected city is drawn at 28 px with its destinations on the same scale, so a destination is never larger than the city it is part of. A new `city` selection with its own inspector; clicking another city's bubble selects that one. Trenton has no registered centre, so it has no bubble but counts in totals and the inspector. With longer-trip bubbles hidden or a local purpose filtered, there is nothing to condense to and the focused city's own bubbles stay. T-139's rule (a selection's longer-trip bubbles omit its own market) is unchanged; T-186 revises only what is drawn with nothing selected.

- 2026-10-12 (Anita): station complexes, including ones the player links by hand (T-197), are transfer points in demand (T-161), replacing "complexes stay display-only". Park demand is weighted by area and by population nearby, not by size alone (T-196).

- 2026-10-12 (Anita): new standing list docs/demand-worldwide.md of demand assumptions to recheck for cities around the world; every new or changed demand rule adds or updates an entry. The state-line border factor stays for the US (Anita unsure state lines matter much; it is on the list). The flatter distance curve (T-198) goes ahead for the US packs. Left-behind riders feeding demand (T-188): yes, later.

- 2026-10-12 (T-189, T-190, T-191, Anita's TODO): stations are dots below zoom 13 and their platforms from 13, where a 150-200 m platform is 10-14 px, about the dot's size, so the switch hardly changes what is seen; one zoom further out platforms would be shorter than the dot. A platform is drawn as the circular arc through its middle and both ends along its track, with flat ends, at its true length; at a dead end it runs from the node along the track, matching where `service.rs` stops trains. Its width is the larger of the true 12 m and the lines through it plus the rim: true width alone would hide the box under the line until about zoom 17. Selection never resizes a station (replacing T-144's 1.35 times and T-163's enlarged complex): a 2 px dark outline and a 6 px soft dark glow, the selected stations drawn after the others; above the demand bubbles they are shown by clearing the bubble canvas over the station and its outline, fading out over the glow, rather than drawing the station a second time on top, so a train standing in it and its crowds stay visible. A click or hover there goes to the station. Platform crowds draw only where platforms do (from 13, was 12). Train pills bend along the circle through their two ends and middle, one quad each (cheaper than segments, the same draw call); close-up cars stay straight. Lines sharing track are drawn over each other, the highest line id on top (T-190, Anita: "a hard problem"), so stations are no longer double width; T-062's slots and offsets are gone from the network buffers, demand layers and picking. Lines are 60% of their former width at every zoom, bare track, casing and the selected extra in step (T-191); trains keep their size so a pill still reads over a thinner line; picking keeps its 7 px reach, which stays wider than any drawn half line.

- 2026-10-12 (T-199, follow-up to T-189/T-191): trains are 75% of their former width, 3.75/6/9 px at zoom 9/12/15 (was 5/8/12), about 1.4 times a plain line from zoom 12 instead of twice; a pill keeps its former minimum length (2.2 times the old width) so it stays as easy to spot and its fill gauge, which runs along the length, keeps its resolution. The white rim stays as it was. Platform crowd bands are lighter: drawn from zoom 14 (platforms still from 13), 1 px plus 1 px a step of crowd, so 2-6 px (was 4-12), 2 px clear of the platform or a standing train (was 3), the train blue at 40% opacity (was 70%). The left-behind red is 90% opacity (was 85%) and one step thicker than the blue would be, so against the softer blue a platform leaving riders behind is the one that stands out. The band hover (T-194) reaches 10 px beside a platform (was 18) and only where bands are drawn (the renderer's `crowdsShown`, from zoom 14).
- 2026-10-12 (T-198, from T-185, Anita "i think so"): the work gravity's curve is d^-0.8 exp(-d/30 km) in every US pack (was d^-1 exp(-d/40 km), T-162), one value for all nine as T-185 recommended rather than a length per pack. It lengthens job-centre residents' commutes by 0.6-1.3 km and moves centre-to-suburb commuting from 0.73-0.87x LODES to 0.88-0.97x in all nine, and lowers trips misplaced in eight; New York is about neutral (16.2% to 16.7%), the one hint that a very centre-heavy city might keep a steeper curve (docs/demand-worldwide.md). The US border factor stays 0.23 (refitted 0.228 New York, 0.207 Philadelphia); Washington's own factor follows its refit at the new curve, 0.49 to 0.45. The curve lives in each pack, but the kernel's `Params` default changes with it, since `city_gravity` uses a pack's gravity only when the two agree (otherwise every pack would be re-solved at opening on the old curve). notes/T-198.md.
- 2026-10-12 (T-209, Anita: "a complex of like 10 stations renders as 2 station bubbles"): no rule changed. The circles already group exactly as `complexes.ts` does, links included (checked on all 855 complexes of the real New York save: one circle each). What looks like one complex drawn twice is two complexes sharing a name: New York's Times Sq-42 St (10 stations) and its 8 Av platforms (4, 290 m away), two Grand Central complexes (16 and 8 stations, 335 m apart), 14 St/6 Av, Bleecker St, Canal St. Merging same-named stations was rejected because a player's stations take nearby place names and two unrelated stations can share one; linking them by hand (T-197) is the way, and makes one circle and one label. The name labels now take the player's links too (they grouped by geometry alone).
- 2026-10-12 (T-210, Anita): every hover outside the map canvas is the game's own tag (`ui/tip.ts`, from `data-tip`), shown at once and following the pointer, never a browser `title` or SVG `<title>`; a stray `title` from markup the game does not write is turned into the tag under the pointer. notes/T-210.md.
- 2026-10-12 (T-211, Anita): search is a magnifying-glass button until opened (click, `/`, Ctrl+K or Ctrl+F); Escape, a pick or leaving the box empty closes it, while text left in it keeps it open. Ctrl+F is taken from the browser's find only when focus is in the game (not another text field), so typing in a slot name still gets the browser's find.
- 2026-10-12 (T-212, Anita): the loading card and the top bar say "Loading data… N%" rather than listing cities; the step line stays, and the top bar's hover names the cities.
- 2026-10-12 (T-213, Anita): "Station demand circles" and "One circle per station complex" move from Settings to the top of the Stations tab, above its list, with their note. The complex setting goes with the circles since it is the circles' grouping (and the inspector's Station / Complex switch, which stays where it is).
- 2026-10-12 (T-206, Anita: "we should only add or remove trains at the ends of lines"): an edit that changes a running line's departures takes effect at the game time it is made. Departures before it keep the old pattern (an era, kept in the clock worker's service only until no train of it can be drawn, never saved), so running trains carry on where they were; a train whose profile changed is drawn from where it was. At each terminal a train that has finished its run and turnaround takes the next departure within that departure's headway (an hour at most) and stands until then; a departure with no train waiting is a train entering service there; a surplus train leaves at the end of its turnaround; a train that had left service never comes back. The same hand-over runs at every change of period, and at any change of pattern a departure closer than half a headway after the old pattern's last is left out, so trains never leave in pairs. Chosen over per-train state stepped by the clock: trips stay a function of the service, its eras and the hour asked for, cheap and alike from any window. A reload re-spaces once (eras are not saved: saves are Anita's only). The boarding account's day starts six hours before a line's latest change while it has eras, so its trains are the ones drawn. notes/T-206.md.
- 2026-10-12 (T-207): trains do fill where a line is short of trains (a player's Manhattan trunk at 14 trains is packed on every peak train); on the real New York network 3% of morning-peak and 8% of evening-peak trains reach crush, because demand's crowding penalty moves riders away once seats run out and demand carries fewer subway riders than the real city. Not changed: loads stay demand's. Changed: a train packed to crush is drawn full with a red edge and its tag says "packed full", since a full train had looked exactly like one at 85% or one with no load data. notes/T-207.md.
- 2026-10-12 (T-204, Anita: station boxes "change proportions as we zoom in and out ... they end up looking stubby. lets have them only ever be drawn at a single scale", and turn into dots "around ~4x more zoomed in"): a platform box's width is fixed on the ground like its length, so both scale with the map and its shape is the same at every zoom. The width is what covers the widest line through it plus the rim at zoom 15 (12.6 px for a plain line, about 23 m at New York's latitude), not the true 12 m, which at zoom 15 is narrower than the line itself; since lines stop widening at zoom 15 the box covers them at every zoom from there. Boxes now start at zoom 15 (was 13), dots below; crowd bands from 15 (were 14). Replaces T-189's "true width, never narrower than the lines on screen", which kept the box a constant 8-13 px wide while its length doubled each zoom. Close in (zoom 17 on) boxes are much wider than the lines and neighbouring platforms overlap more; accepted as the cost of one shape.
- 2026-10-12 (T-205, Anita: "hide trains and stations when zoomed out, lower min line thickness like 40%"): below zoom 11 neither stations nor trains are drawn or picked; at 11 the screen holds about 30 km, a city such as Manhattan with its neighbours, and further out the dots and pills crowd the lines. A selected station or complex is still drawn and clickable, and demand circles stay (they are their own setting). Line widths go to 0.85 px at zoom 9 (40% of T-191's 2.1), linear to the unchanged 3.9 at 12, with bare track, casing and the selected extra scaled the same; a stroke under a device pixel is drawn one pixel wide and fainter in proportion, so far-out lines do not break up. Pick reach stays 7 px.
- 2026-10-12 (T-196, Anita: parks "by size and by population nearby"): OSM parks are 8% of the social/leisure destination draw, from ATUS 2023 (7.6% of US leisure outings end outdoors away from home; NHTS has no place type), taken evenly from the population and leisure-jobs draws so T-066's 50/50 between them stays; parks move outings and add none. A place draws sqrt(area) x D / (D + 2,000), D the residents per km² within 2 km of its edge: the square root is Anita's diminishing return, and a density rather than a count because a count grows with a park's edge (Harriman State Park outdrew Central Park five times). The density term saturates in built-up places so it does not double what the gravity already does for parks with many homes nearby. Nature reserves count as parks only when named as one ("... Park", "Recreation Area"), other named ones at a quarter, unnamed none (New England's land-trust and watershed parcels); playgrounds, dog parks and pitches are left out (inside parks, and mostly walks under the 150 m cut). No season. The exponent, the 2 km and the 2,000/km² are judgement. notes/T-196.md.
- 2026-10-13 (T-229, Anita: changing a line's trains "lags"): the stepper shows the asked-for count at once and clicks during an edit are folded into one edit sent when it lands; a train count sends a service-only state (the renderer keeps its buffers and takes the new run profiles), since rebuilding the whole network's buffers was half the edit's 85 ms on New York. notes/T-229.md.
- 2026-10-13 (T-230, Anita: "we should fix the demand forecast's average train to instead actually show something closer to the per train account"): every train-fullness figure the player sees comes from the boarding account. The line panel drops demand's busiest-hour average train ("Fullest train at high demand") for "Trains today": fullest train, trains leaving full and when, riders left on the platform, longest wait. A train with no account yet has no gauge, rather than the forecast's average in its place; the train inspector's typical train is this hour's trains from the account. The forecast's figure disagreed because the account's riders arrive fastest mid-peak (1.8 times the mean against demand's 1.4): 28 of New York's 93 lines under 100% in the forecast leave riders behind. notes/T-230.md.

- 2026-10-13 (Anita, answers): long-linked complexes get one dot per nearby group plus a thin connector (T-225); traffic width stays relative to the network's busiest stretch (full range for every network, T-227); nursing homes stay out of homes with prisons (T-234); the 1 km manual link limit stays ("a bit long but fine"); the 46/46/8 social and leisure split stays for now (Central Park possibly undercounted, T-240). Lines may be wider than station boxes in traffic mode.
- 2026-10-13 (Anita, after trying T-255): express lines and quad track are on hold, with design decisions still to make; no further work on them (T-257) until she says. An express line keeps its local line's colour. She is not sure Express copy differs enough from Copy line to need its own button (it moves stops onto the twin stations of the express pair); left in place for now.
- 2026-10-13 (Anita): the same trains run every day for now (T-261 closed). The demand views' day moves to Settings with three choices, the clock's current day by default (T-266).
- 2026-10-13 (Anita): traffic widths show the day's riders, not the hour's (T-274). Intercity: option 2 of research/intercity-review.md (retune the market and add commutes between cities), inside a world divided into zones: city zones, non-city zones between cities, unmodelled space (T-276); Trenton/Mercer probably part of New York. Routed walking and driving before tolls (T-275; T-270 tolls later). Toronto is the first non-US city, an exception to the pause on new cities (T-277). Waits stay as they are (half the headway, the first 5 minutes at 1.37 and the rest at 0.4, at boardings and transfers alike); Anita asked whether infrequent service is penalised, and this rule is the answer. notes/ belongs to the agents.
- 2026-10-13 (T-276 agent, docs/zones.md 6): the focus waits until every zone's boundary has loaded (falling back to the nearest centre until then), so it never flips as boundaries arrive one by one and opens a pack for nothing; where two boundaries share a point the registry's first zone wins. Nearby zones open only while Homes or Destinations is on (they exist for their bubbles), never during the startup load or a game load's clock hold, show no loading bar, and a solve never waits for them; once open they stay open like a viewed city's, at about 60-90 MB of WASM memory per solving worker each (Philadelphia 90, New Haven and Springfield together 57). A nearby zone's bubbles skip the catch-up and show the current forecast. A click on a nearby bubble focuses its zone and selects the bubble in one click: the focused zone takes over the nearby zone's worker and bubbles, and the old focus's become a nearby zone's.
- 2026-10-13 (T-276, docs/zones.md): the world is zones. A non-city zone is a pack like a city's (same pipeline, cells, purposes and views, `kind: "zone"` in the registry), one county-sized statistical unit each (US county, Canadian census division, NUTS-3), modelled when it touches two city zones (Worcester, Windham, New London, Litchfield); a coarse second model was rejected because every view, query and loading path would need a twin. How far modelled space reaches around cities is Anita's (open). Mercer County joins New York's pack (Anita's lean): in the New York CSA, LODES 175k jobs both ways with New York's counties against 88k with Philadelphia's; market id 2 keeps its name with no sites.
- 2026-10-13 (T-276): commutes are one doubly constrained gravity over every zone pack (`joint_gravity`), within-pack pairs with the pack's own curve and factor (so the runtime rebuilds them exactly), cross-pack pairs on the great circle cut at 150 km with a factor per region pair taken from the pack holding both regions, else the country's admin-1 default (US 0.23, as T-171 asked: not one global value), national borders 0.05 (US-Canada, judgement). Each pack's homes are scaled to its own jobs, not the union's: scaled to the union the solve forced net commuting through neighbours and barely converged. The cross-pack part (1.12M commuters, 5.9%) travels in the regional layer at commute timing, the local commute's mode choice and the work purpose's weekend factor; pairs under one commuter a day between 6 km zones are dropped and their commuters given to the home zone's kept pairs (172k pairs among the 13 zones, the market pack 8.2 to 13.9 MB).
- 2026-10-13 (T-276): the market's longer trips: 0.107 weekday departures a resident (was 0.015), onset 25-50 km (was 40-80), `max(d, 25)^-2.37` (was `exp(-d/180)/max(d,40)^1.2`), fitted together on NHTS 2017 Northeast weekday ground trips of 50-1,000 km, with an outside world (Kontur people in no zone) in every normaliser so 17% of departures leave the map. The car pays 15 cents a road km and the local parking, both over two occupants. Local outings keep their 40-80 km taper (the overlap from 25 km is under 2% of social outings; airport trips keep their reach). The market's trips between adjacent packs now include short regional trips across their seam, so they far exceed NEC FUTURE's metro-to-metro figures (New York - Philadelphia 380k against 57k); long pairs are right (New York - Boston 53k against 50k).
- 2026-10-14 (Anita): one save for the whole world, at least as the player feels it (T-304); London first; the sampler's extents as proposed (Greater London, Ile-de-France, Berlin and its ring, Shanghai's 16 districts, the Mumbai metropolitan region); non-city zones between cities only (T-300); line and station panels show only counts so far today, labelled "Today", never whole-day forecasts (T-293); T-122's frozen copies rejected (memory grows with open cities), instead compute what is on screen or likely to be looked at first (T-303); pack backups deleted.
- 2026-10-14 (Anita): the one-world design (docs/world.md) is approved. Local time per region as NIMBY Rails does it: the clock shows the local time where the player is looking, and each region's traffic runs on its own local time (T-309). Old saves convert to the world frame on load, correcting anything that moves substantially where that is easy (T-308).
- 2026-10-14 (Anita, T-310): the Money tab's "A day at this schedule" is gone (a whole-day forecast, as T-293 dropped elsewhere), with the code only it used (`income`, the per-day fare and running figures). The top bar cash keeps its fixed width but the digits sit right after the $. A reload does not restore the exact view: the URL hash only picks the nearest city or zone, and the map starts at that registry centre and zoom (nearest centre, not boundary, since it must be known before the boundaries load). The Lines and Stations lists sort by clicking column headers (a second click reverses, an arrow marks the column), not a dropdown; the sort stays a per-page-load setting as before. The grey veil's flash at the end of a pan was `cityBoundaries.ts` emptying the shade source on every change of focused zone before the new shape arrived; it now swaps in one setData from footprints made ready at startup.
- 2026-10-14 (T-307): the draw anchor moves on MapLibre's moveend only (not during a pan), so a long drag rebuilds the worker's buffers once; at 300 km from the anchor Float32 is still centimetre-exact. The grid is 1 degree, so a fresh anchor is at most about 80 km from the view, inside the 100 km reach. New York keeps (-73.98, 40.73) as the anchor until the view leaves it, so it draws and picks exactly as before. The main thread switches its anchor when the state built at it arrives, not when it asks, so nothing is ever converted with one anchor and drawn with another; previews crossing a switch are shifted (a translation in Mercator). The demand views stay on the home anchor for now (T-317): rebuilding the bubble workers on an anchor change is its own piece of work, and their discs are coarse enough that it matters only far from New York.
- 2026-10-14 (T-306): new games are in the Mercator frame of the whole Earth (TWT6, registry `frame.kind` "merc"). Lengths are measured on the ground: each piece of track keeps its ground length (straights as rhumb lines in closed form, arcs by Simpson's rule on 1/k) and chainage is ground metres everywhere, so profiles, capacity, service, boarding and costs kept their arithmetic; metre figures entering the plane are multiplied by k where the arc is (its middle, not the PI: up to a kilometre apart on wide turns, 0.02% of k a km at London). The conic and legacy frames keep k exactly 1, so their results are bit for bit as before. A plane circle is a ground circle only up to the rhumb line's own curvature (a 1,500 m curve at London measures 1,500.4 m on the ground); left as it is, being sub-metre. TWT6 keeps the game's home origin in its origin fields (rather than zero, as docs/world.md first said) so a conversion back to the conic has one. Track is refused within 5 degrees of the poles and across the antimeridian (`OutsideWorld`). `joint_gravity` takes its conic from the market's frame, since the registry's game frame is no longer one. App code comparing world metres with a metre figure goes through `groundScale`/`groundDist` (station complexes, links, nearest station). docs/world.md.
- 2026-10-14 (T-317): the demand views follow the draw anchor, each disc set carrying the anchor it was built at. The overlay draws every set with its own anchor's matrix and the hover finders get the pointer as longitude and latitude and turn it into each set's units (`discAt(discs, pointer, anchor)`), so while an anchor switch is still rebuilding the bubbles, the old ones stay exactly where they were instead of jumping. The bubble workers keep each cell's longitude and latitude, are told the anchor with their cells and again on every change (`sync`, before anything is asked of them), and answer with the anchor their positions are relative to. Disc sizes use k at the disc (`kAtY` of the disc's local y; `MERC_K` stays only for the stress network). So bubbles in New York are as before to a fraction of a pixel (k there varies by 0.3% over the metro, where the constant was New York's centre), and bubbles at Toronto are 4.7% larger on the ground than before (the old constant was 1.319 where k is 1.384); London's will be 18% larger than the constant would have made them.
- 2026-10-14 (T-308): old saves convert to Mercator when they load, in the clock worker and nowhere else: any network that is not TWT6 goes through `TrackApi.convert_save_frame` once, from the autosave, a slot or a file, and the slot is written again in TWT6 straight away. The old bytes are kept under a hidden IndexedDB key (`old-frame-<slot>`, not listed as a save) since a conversion cannot be undone from the new file. Money, fleet, fares and the ledger are copied; the player is told nothing. Nothing was corrected: on the real NYC save no line's run time moves over 0.3%, no line needs another train, no curve is newly refused (the save's 586 geometry refusals are the same set before and after), so a set radius, already read as ground metres, needed no special case. Where a legacy save moves more (Philadelphia track about 1.2%) the new length is the true one. Riders on the real save fall 0.4% (8.84 M to 8.81 M; the demand model re-solves on the new lengths and times). docs/world.md.
- 2026-10-14 (T-309): local time per region. One game clock keeps world time in UTC-5 (registry `clockUtcOffset`), New York's standard time, so saves made before it need no conversion and the Northeast and Toronto run bit-identically (checked on the real New York save: a week of trips, boarding slices and today's counts). Each registry entry has a `utcOffset`, whole hours of standard time with no daylight saving (a seasonal hour would move every period twice a year for nothing the game models). Half hours round down: India's +5:30 is +5, putting the game's 6-10 morning peak on Mumbai's real 6:30-10:30, nearer its late office peak than +6 would. A place is in the nearest registered centre within 120 km, else it takes `round(lon / 15)` until its region is registered (London 0; Paris also 0 until its entry gives +1). A line runs on its first stop's region's local time (simple, and a timetable has one clock); that carries its service periods, its boarding account's hours and kind of day, and its fares and running costs. Demand needs no change: a solve has no clock, so its period q is each region's own. The top bar, the demand views' period and their "Today" follow the local time at the view's centre, switching as the map moves; the line inspector and Lines tab use each line's own. Left for T-321: ledger days, the bubbles' catch-up, market trips across offsets. docs/world.md "Local time".
- 2026-10-14 (Anita): T-303's loading proposal approved as a whole, built in order as T-324 to T-328; trains with no figures yet are striped; a new line shows riders from the first estimate; on reload the last result shows at once and the clock runs on it. Map: station dots 20% smaller with thinner rims and connectors; blueprint track light blue, wider than built lines, like two rails. Closest stations show line chips and leave the bubble hover tag (tags stay lean).
- 2026-10-14 (T-324): "new or changed" means a line whose stop list differs from the shown final result's (a line with new train counts still has figures that fit, so it keeps them under T-232). Its estimate figures are added to the shown result, which keeps its own version and round, so queries, bubbles and the top bar behave exactly as before; stations served only by such lines get the estimate too, a station an existing line serves keeps its final figures. Trains: fill -2 means "loads shown but no figures yet" (striped), -1 still means no gauge (train loads off, line not running, or demand failed). `?perfOff=early` restores the old behaviour for A/B checks.
- 2026-10-14 (T-332): the Lines and Stations tabs list only the viewed place by default, with a toggle to the whole world (one setting for both tabs). A line with no stops and the selected row stay listed when filtered, so a new line or an open selection never vanishes. Near is an ordinary column header, always present; its distance figures and column width appear only while sorted by Near.
- 2026-10-14 (Anita): build the crossover tool (T-338); quad stays express pair beside local pair (simpler to build); an express slowed by a local ahead is wanted but low priority (T-339).
- 2026-10-14 (T-347, Anita): a loaded game keeps its loading card and held clock until the fresh solve's final round, not the first estimate (T-172) and not the cached result (T-328). Her reason: "we're doing it and it's still loading demand and like lagging / low fps / unresponsive", so keep the indicator and the pause until the lag itself is fixed (T-348). The cached result is still drawn behind the card. A new game waits too (final 2 s after the estimate). A pan or an edit is unchanged. Revisit when T-348 lands: `HOLD_ROUND` in workers/demandClient.ts is the one constant.
- 2026-10-14 (T-319): right-hand running everywhere, Britain and Japan included; no running side per region. Anita: per-region running would be messy at junctions and is not worth the effort. A single global toggle at most, if ever.
- 2026-10-14 (T-352): drawn US boundaries (outline, outside shade, reach test) are TIGER/Line counties with their water, out to the state and county lines at sea, not the shore-clipped cartographic file (Anita: the grey hugging Manhattan and Brooklyn is unwanted, a straight maritime line is fine). Boundary files only: packs, water masks and cells still use the shoreline footprint (`build_boundary` returns it), so no number changed. Simplified once nationwide at 0.0002 degrees with shared borders (`pipeline/tiger_counties.py`) to keep zone boundaries disjoint and files small. `outside.py` has not been rerun: it would now count 194 more Kontur hexagons (0.88 M people, mostly water-centred or edge hexagons) as inside zones; rerun it with `intercity.py` at the next market refresh.
- 2026-10-14 (T-351): in a game with no station, picking the track, station or blueprint tool opens the bare viewed zone in every solving worker (no solve for it), so a first line's solve does not wait 2.4 s for New York to open in five workers; it guesses the player builds where they look and costs that CPU only once a tool is picked. A saved game does not wait for its network's reach before opening the viewed zone (that would cost about 1.4 s whenever a station reaches it), so only a new game starts bare in practice.
- 2026-10-14 (T-295): park-and-ride replaces the fast walk with a true 6 km/h walk plus a "ride" (car, bus, bike, lift) to the station, offered only past a short trip length, at the far end too for long trips; car parks invisible and unlimited; New York recalibrated after. Anita: keeping New York's numbers does not matter.
- 2026-10-14 (T-361): demand workers by threads (12+ get 3, 6-11 get 2, fewer 1; 4 GB or less gets 1), a "use more cores" setting for up to 5; the loading card still waits for the final (Anita: there is still some lag); memory cuts wait.
- 2026-10-14 (T-362): one world market for long trips, very far pairs shown but computed coarsely, no driving between landmasses except fixed links from a precompiled source (Anita). Air as an unbuildable third mode, GDP per head^0.7 scaling and the Northeast's long pairs free to rise were the design's picks, taken by the orchestrator when Anita had no view.
- 2026-10-14 (T-370): a drive's middle runs at 100 km/h (US, Canadian and UK motorways, peak and off-peak mixed), each end's density speeds only over its own first 10 km; Anita found 80 km/h pessimistic for highways. Typical real drives fit between the off-peak and peak figures; New York's commuter rail lost riders before ride access (T-371) restores them, and nothing was retuned to compensate.
- 2026-10-15 (T-371): the ride's starting values are p_ride 10, d_drop 10, offered to commutes whose zone centres are 5 km or more apart; the true walk weighs 1.5 a minute, not the usual 2.0, which put the boroughs' rail 9-16 points under ACS. Homes beyond every walk are one ride-only origin per 6 km square, taking their most populous zone's distance decay (one per 2 km zone cost a third more solve time for the same rail). Everyday purposes and the work end walk only for now (T-389). All for T-372 to refit.
- 2026-10-15 (T-389): everyday trips ride on the commute's 5 km rule, measured on each pair's own mean distance rather than zone centres; a 3 km threshold for errands added only 2% more rail (7.69M against 7.54M) for rides that rarely happen in life. The map's walking minutes follow the model's 6 km/h, replacing T-312's 4.8 km/h display pace.
- 2026-10-15 (T-373): the regional layer's rides use T-371's values unchanged, 15 km reach for long trips too (the design's 40 km would serve far more zone pairs); a site's walks and rides nest at the station choice's 0.2 a minute rather than summing at the trip's 0.025, which would have made any extra option worth many minutes. For T-372 to refit.
- 2026-10-15 (T-394): a pack's open reach stays 4.5 km from its outline, not the ride's 15 km: a pack's trips need a station within a walk of its job cells at the work end, which already opens it, and opening it for a station only a ride away changed no figure (identical riders at the station, Worcester rail trips 0). notes/T-394.md.
- 2026-10-15 (T-338): a crossover's two diagonals cross flat, so trains on both diagonals at once wait for each other as at a real scissors (the crossing is in the crossover's price, not charged again); the other track must be 6-15 m away (centre to centre) and parallel within 3 degrees; an end within 8 m of a plain joint uses the joint; deleting a crossover leaves its four nodes as plain joints rather than re-joining the pieces.
- 2026-10-15 (T-326): demand clusters are joined by generous fixed margins (a city 30 km from its cells' box, the market 200 km from a market zone's centre, stations 1 km apart) rather than the kernel's own access search, and the market's reach comes from the registry, so every worker (the auxiliary one holds no market) splits alike; too wide costs time only. Clusters are recomputed for every network, so a later link (T-362) merges them.
- 2026-10-15 (T-362, Anita): a long-trip pair is served only when actual track joins stations within reach of both ends; unserved pairs are fixed offline as a row per zone and never solved; demand clusters merge only through track, never through the market's trips alone.
- 2026-10-15 (T-396): the market joins a demand cluster only through stations within 30 km of a market zone's registry box (its sites are those zones' cells; its rides reach about 17 km), not 200 km from any centre, and only when open. It stays one population, so networks with no track between them that both lie in its reach still share a cluster until the kernel splits its pairs by network (T-409); the near-pair cutoff (250 km) plays no part in this reach. The market opens once a market zone is viewed, beside the view or reached, or a station lies in its reach, and stays open. notes/T-396.md.
- 2026-10-15 (T-404): a place is given a zone only once every boundary whose box lies within 40 km of it is in; until then it falls back to the nearest centre, so a focus never flips as boundaries arrive one at a time. A zone whose boundary fails is left out of the focus and counts as reached by the demand pool, as before for all of them.
- 2026-10-15 (T-124): periods of a solved cluster whose own level's inputs are unchanged keep their results, decided by the populations comparing networks to the bit; dirty periods are not moved to idle workers, since with 3+ workers `q % n` already puts each level's two periods on different workers and with 2 the periods are dealt out level by level (0, 1, 4 and 2, 3) instead, which gave the same 12.4 s with no change to which worker answers which period. notes/T-124.md.
- 2026-10-15 (T-418, Anita): NIMBY-style drawing: one tangent curve per segment through the click, too-tight clicks refused, Shift held to chain, the old mode kept as a setting until quad works in the new one.
- 2026-10-15 (T-397): everyday destination samples per cell and purpose are one per 8 of the cell's trips a day for it, rounded up, 2 to 16 (airports and medical stay at 2). Against sqrt, 2 sqrt, 0.5 sqrt and trips / 4 or 16 on the real track, it kept station boardings and line riders within 5% everywhere with 31% of New York's slots; single zones' arriving trips move more (a quarter of busy zones past 5%), which a station sums away. Revisit if a view shows one zone's or cell's everyday trips. notes/T-397.md.

- 2026-10-15 (T-402): in a solve with the round barrier (startup, a zone just opened) the zone in view's cluster is solved alone through every round before the other clusters, which go together: its final lands in about 5.7 s instead of 32 s on New York plus a Toronto line, at no cost to the whole final (31.0, 31.7 s against 32.3, 31.7 s). The price is that a zone in the second group gets its first figures only after the first group's final (about 30 s instead of 12 s for Toronto while viewing New York). Two groups, not one per cluster, so the rest keep T-327's rounds in step. A game load's card still waits for the whole final. notes/T-402.md.

- 2026-10-15 (T-418): drawing through points is the Build tab's default (a display setting, remembered per browser, so Anita meets it at once and switches to Corners with one click); a click more than 85 degrees off the heading is refused as "turn back on itself" rather than a new issue kind. Revisit with T-422.
- 2026-10-15 (T-423): quad drawn through points ends each pair loose on open ground (no fork where they meet, as Corners makes), so the next segment can carry both on without changing placed track; a quad segment from other track keeps the swing-out and so the curve-then-straight shape, not one arc.

- 2026-10-15 (T-290): Berlin's jobs come from the Länder's Pendlerrechnung (everyone employed, by Gemeinde of work) rather than the BA's insured employees named in the brief: the same atlas gives the flows for the curve, and it counts civil servants and the self-employed. Its flows publish only each Gemeinde's ten largest in and out, so the curve is scored on those pairs. Brandenburg's Geofabrik extract alone, since it includes Berlin. notes/T-290.md.

- 2026-10-15 (T-375): the fixed links joining landmasses come from Wikidata (bridges, tunnels and causeways in use, near the sea, reaching half their length plus 3 km) over Natural Earth 10m land, since no precompiled global dataset of crossings exists; countries from Natural Earth admin 0 rather than geoBoundaries; GDP gaps (Taiwan and others) from Maddison 2023 scaled to the World Bank. notes/T-375.md.

- 2026-10-15 (T-409): the market splits by track network as one part per cluster beside one opened market (`IntercityApi::beside`, the pack's sites, zones and commutes shared by `Rc`), not one market instance per network (about 45 MB a worker each): a part costs only its own network's pairs and periods. The market joins clusters through one node per 6 km zone of its own (within 30 km of the zone's sites), not per registry zone, whose boxes overlap and are up to 110 km across: with registry zones the real save and a Boston line merged through Providence's box. Two networks one 6 km zone reaches still merge, which keeps the split exact (each zone's access lists stay in one part).

- 2026-10-15 (T-420): Greater Tokyo takes the 2020 census 500 m mesh (not 250 m) for population and the 2021 economic census 500 m mesh for jobs, scaled per municipality to the census's workers by place of work; its curve is d^-2.2 exp(-d/80 km), the steep end of a flat fit (London's 1.85 scores within 0.6 points) because only it keeps the census's 33.5% working in their own municipality; work at home 0.23 (census 9.5% at home plus a 2-days-in-5 telework share of MLIT's 37.5%). notes/T-420.md.
- 2026-10-15 (T-420, Anita): Greater Tokyo takes in Ibaraki (Tsukuba and the Joban corridor commute into Tokyo); Keihanshin is Osaka, Kyoto, Hyogo and Nara as research/japan.md proposes.

- 2026-10-15 (T-376): the long-haul tail is its own segment (rate Rt 0.0047 a weekday, own normaliser, rising from 300 to 700 km, d^-0.70), not T-362's k x tail inside the ground's normaliser, which gave sparse places 3x the long trips; a poorer destination is damped by (y_j/y_i)^1 and the border factor is 0.26 x (origin income)^1.5, since destination income alone sent the US, India and Japan abroad 5-30x too often. NHTS weighs 3x the flights pairs (the 0.6 connecting factor is a guess). Japan abroad stays 6x high, left for country tailoring. notes/T-376.md.
- 2026-10-15 (Anita): Suzhou (Kunshan included) is its own pack beside Shanghai; the Pearl River Delta is five packs (Guangzhou-Foshan, Shenzhen-Dongguan, Zhuhai-Zhongshan, Hong Kong, Macau) with Hong Kong's border friction; Los Angeles is trimmed below whole counties when it comes; T-377 is built with T-376's refitted rule; Japanese place names take romaji from a kanji reading dictionary where OSM has no Latin name; cost scales with area, so merging sparse cells (T-044) is wanted.

- 2026-10-15 (T-377): a far pair's trips (places 250 km and more apart) are split over 6 km zones by each end's departures over the normalisers and its attraction, ground and tail apart, at the places' centre distance, so the zones add up to the place pair exactly; a pair with no served path (no station, no track, or no car between landmasses) shows all its trips and none by train, with no mark of its own until air (T-378). The 'Inter-region trips' label stays.
- 2026-10-15 (T-377): every place is its own group of sites for access and rides (not one group per frame), so a place's figures never depend on which other zones' market packs are open; the Northeast's ride zones moved off their shared grid origin, part of the real save's +0.08%.
- 2026-10-15 (T-377): a zone's market pack opens and closes with the zone in the solving workers and the market is recomposed then; a part keeps the composition it was made from while its cluster's key lasts. The registry keeps market as the Northeast joint gravity's place (so joint_gravity's packs are unchanged) and adds place for the world market.
- 2026-10-15 (T-426): Japanese place names without a Latin OSM name take romaji from UniDic's reading of the kanji (fugashi, unidic-lite), long vowels written short (Omiya) so every shown name is ASCII, a final 町 as cho (as most of OSM's name:en has it); about one in four or five is misread, kept over showing kanji. Work at home stays 0.23 with Ibaraki in, though MLIT's telework share is for the four southern prefectures.
- 2026-10-15 (T-378): air is a mode only between far places (centres 250 km and more apart) that both have airports, priced by place (the drives to the best pair of airports, people-weighted over each place's zones), so a pair's car and plane split is fixed whatever the player builds and never solved; near pairs never fly (NHTS Northeast air trips are near 0 under 500 km), which leaves the real save's figures unchanged.
- 2026-10-15 (T-378): on the rail-air share curve the design's door-to-door air (90 min at the airports, a $60 base fare) gave rail 94% of rail and air at 3 hours on the train and 80% at 4; 60 min at the airports and a $50 base give 79% and 46% (the studies: about 70% and half), with 750 km/h + 30 min, 60 min more with no direct flight and 9 cents a km kept. The market's 0.025 logit has no distance scaling, so the curve is steeper than the studies' (2 hours 94%, 5 hours 17%).
- 2026-10-15 (T-378): a zone's airports are OurAirports' large and medium ones with scheduled service within 150 km of its centre and at least 1M FR24 passengers a year (matched by position where FR24 keeps an old code: BER is SXF), so New Haven and Litchfield fly from New York's airports; a direct flight is an FR24 route of 50,000 passengers a year from any of either end's airports. Only these choices leave the flights data.
- 2026-10-15 (T-378): trips with places that have no zone are shown as an Elsewhere row (all modes only, their modes not modelled), sorted among the places by size and counted in the direction's total; the bubbles keep blue for train and red for the rest, the plane in the red.
- 2026-10-15 (Anita): merge sparse cells (T-044) as notes/T-044.md recommends: parents under 1k trip ends a day, in the pipeline, inside 2 km zones. The loading card goes once the city in view has its final; other cities still solving show their loading bar, not white. Planes stay in the bubbles' red share. T-438 and T-429 wait for calibration and country tailoring.
- 2026-10-15 (T-380, Anita): adopt T-380's parking rule (paid parking from jobs per km2 of land with a free-parking share) together with T-372's park-and-ride fit.
- 2026-10-15 (T-434, Anita): each period stops its crowding rounds once a round moves its summed loads by under 1%, at most 3 rounds (notes/T-434.md option c, per period).
- 2026-10-15 (T-044): cells merge at 1,000 weekday trip ends a res-8 parent, the lowest threshold measured: it takes 42% of New York's cells and half of Toronto's (worker -10 MB) for rail +0.06% and 0.3-1.5% of busy stations moving over 5%, where 5k moved 9-16%. A merged centre on water or outside its zone square keeps its group at res 9; a parent cut by a zone edge keeps one index per part rather than inventing ids.
- 2026-10-15 (T-421, Anita): a game load's card and clock go once the zone in view has its final (its cluster's, T-402), not the whole network's; zones still solving show their own loading bar and grey until theirs, so a zone with only estimates never looks finished.
- 2026-10-15 (T-372): congestion charges are a list of charging areas in the sim (`kernel::CORDONS`, Manhattan below 60th St 13.5 min, central London 35.1), applied to the share of a destination zone's jobs inside, not a registry figure per city: a charge belongs to a place, a pack can hold none or several, and a registry figure would need plumbing through the demand client. Half the day's charge per one-way trip, paid by free and paid parkers alike.
- 2026-10-15 (T-372): `p_ride` 15 (from 10), `d_drop` 10 and `w_walk` 1.5 kept, on the real New York network with T-380's parking: home-end ride shares of 53-66% by railroad (50-70% target; 10 gave 66-77%), county error 2.78 points (20 gave 2.66 but NJ Transit 217k against ~250k). `d_drop` 15 changed nothing; `w_walk` 1.25 overshot Manhattan.
- 2026-10-15 (T-372): land for parking density is a zone's res-9 cells x 0.1053 km² (a merged res-8 cell counts 7, which also keeps `Zones::density` and the drive speeds as before T-044's merge), at most the 4 km² square; no floor, since zones under 1 km² of land hold 0.4% of New York's jobs and the 50-minute cap bounds them.
- 2026-10-15 (T-442): a segment turning past a quarter turn is still one tangent arc, stored as equal corners of at most 90 degrees (not two arcs chained, which would change the radius mid-turn), so the express pair can offset each concentric; turns up to 350 degrees, refused within 5 degrees of straight behind, where the circle grows without bound.
- 2026-10-15 (T-422): Corners drawing retired; the route op keeps its `pis` for the API and tests, the app sends none.
- 2026-10-16 (Anita): merge sparse cells harder, at 2,500 weekday trip ends a res-8 parent (from 1,000; small edge stations moving is acceptable), applied in T-453 stage 2's re-merge. T-453 approved: sector jobs where a country publishes them by small area, else jobs combined with OSM places (fallback D). Destinations that respond to the network (T-454) parked.
- 2026-10-16 (T-458): Sao Paulo's jobs, work from home and commute curve come from a household travel survey (Metro OD 2017: workers by zone of their first job, scaled to the 2022 census population; work at home 9.1%; both curve parameters fitted to workers' home-to-job straight-line distances), the general method for any city with such a survey. The 0-2 and 2-4 km bands are fitted as one, since 2 km zones cannot split them (fitting the split gave d^-2.3).
- 2026-10-16 (T-457): Seoul's population is the resident registration count by si-gun-gu (MOIS, November 2020, open CSV) rather than the 2020 census, which KOSIS serves only with a key; registered foreigners (about 4%) are left out. Ongjin-gun's islands are left out of the pack.
- 2026-10-16 (T-459): Lagos's population is GHSL E2020 unscaled (13.8M): no census since 2006, the state's 20M+ not credible, and UN WUP 2018 (14.4M) and Africapolis (15.4M with Ikorodu) bracket it. Its curve is Mumbai's, borrowed (no flows or distance table). OSM land use capped at 1 km2 an area and spread over it (opt-in, T-460).
- 2026-10-16 (T-453, orchestrator): the everyday fallback's errands base takes residents at b = 0.25 beside jobs (leisure 0): it helped in every thin-OSM test and cost London 0.8 points of misplaced attraction at full OSM (notes/T-456.md 5).
- 2026-10-16 (T-453): a sector source's concentration test runs on the pack's 2 km zones, not the source's own units: at LSOAs it failed BRES on noise (21% concentrated), at zones BRES passes (3.5%) and Tokyo's mesh with wholesale left in still fails (11%). Toronto's retail and other services by tract fail it (9.5%, suburban business parks with retail jobs and no shops), so Toronto's errands use the fallback and its leisure the census. A country's source is judged on its best-mapped metro (Japan: Tokyo, 2.5% and 2.6%) and its other packs run only the level test: per pack, 30 of Japan's 45 failed on roadside malls and resort hotels their OSM lacks, which would have made Japan a patchwork of sources.
- 2026-10-16 (Anita): Japan's outlines follow de facto control, so the southern Kurils and Takeshima stay out; the Senkakus (Japanese-administered) are out only as uninhabited islets over 50 km from other land (T-420 stage 3).
- 2026-10-16 (T-471): a drawn segment whose through-the-click arc would be wider than the 1,796 m cap is the capped curve from the start, then a straight to the click, so a far loop or turn back is one round curve; it still passes through the click but arrives on its own heading. Rejected: letting the curve miss the click (breaks T-418's rule) and allowing radii over the cap (no speed gain, and huge loops). Gentle far clicks changed too (curve then straight, was straight, curve, straight), so the shape never jumps as the click crosses the cap.
- 2026-10-16 (Anita): hotel and accommodation jobs stay in the leisure sectors (people travel to and from hotels). Turning back past the 160 km/h cap as one curve then a straight (T-471) is kept. Clicks only reach what is drawn on top: a click never acts on something drawn beneath another clickable thing (T-486).
- 2026-10-16 (T-482): station use circles read the demand solve shown (the demand bar's day), not the boarding account's past 24 hours: only the solve knows whether a boarding rider came from the street or another line (the account has each stop's boardings with no origin). Total is all boardings (origins plus transfers, so the three agree), not boardings plus alightings; the cap halved to 75,000 to keep the circles' size. Every purpose whatever the purpose filter, since the transfer pairs are not kept by purpose. Transfers count at the station the rider boards.
- 2026-10-16 (T-481): "x/y" trains leaving full is out of the trips that reached any stop in the 24 hours (a trip across the window's edge counts on both sides, as its full count does).
- 2026-10-16 (Anita): unselected stations and track stay drawn under the demand bubbles (T-189 stands); only a selected station is cut out on top. Trip-length scaling of mode choice (T-495) is built now and recalibrated later if it feels off.
- 2026-10-16 (T-495): cost damping is the logit scale 0.05 x sqrt(90 / T) past T = 90, T the car's perceived minutes door to door (even across a sea where no car goes), one rule for commutes, everyday and longer trips (the flat 0.025 goes). The car, not the best mode, so the scale never moves with the player's network; a square root (WebTAG's usual damping power) so the lanes stay cheap; 90 so New York's commutes barely move. It puts the rail-air curve near the studies' (93/73/48/28% at 2/3/4/5 h) but leaves Paris - Berlin's car near nothing: that is the long car's cost (T-437), not the scale.
- 2026-10-16 (T-463): Japan's packs cross pack lines at a border factor of 0.4, fitted on the census's commuters between prefectures (level only; the spread over pairs is the same at any factor). Prefecture lines inside a pack take none, as the curves were fitted.
- 2026-10-16 (T-463, T-465): the at-home fix kept each Japan pack's own telework share (MLIT by region) rather than 0.15 everywhere; zones whose grid choice moved on the new margins took it, the seven metros kept their judged curves (each as good or better on the new margins).
- 2026-10-16 (Anita): travel-time weighting of the gravity (T-498) is not built: the prototype gained little (notes/T-498.md). Industry matching of commutes is designed instead (T-500). Cost damping starts at 60 car-minutes, not 90 (T-502). Bubble hovers drop "on average" (the Day choice names the day).
- 2026-10-16 (T-502): the damping starts at 60 car-minutes (Anita's pick); on the real save regional boardings went 68,301 to 151,704 (T-495's 90 had cut them from 259,376) and rail trips a day barely moved (-0.04%), since a flatter choice on 1-3 hour trips gives rail more of the pairs where the car beats it. The average day's hover phrase is "a day", the weekday and weekend phrases stay.
- 2026-10-16 (T-466): in a res-8 parent over 2,500 trip ends, touching children under 1,250 each merge among themselves (half the threshold, the loosest measured: cells -19% over all packs, 0.4-2.9% of busy stations past 5%; 357, where a group could never exceed a merged parent, took only 3-5%). Merged packs carry each zone's res-9 cell count as its land, which T-044's count by index overstated in sparse zones (rail +0.1-0.3%).
- 2026-10-16 (Anita, T-509): calibration fits New York, London, Paris, Seoul and Tokyo jointly plus one small city, holds out Toronto; real networks may be rough (OSM routes, frequencies from tags or defaults by line type; no ODPT account); car ownership (a no-car share per area) is added before the rail constant is fitted; everyday trip rates come from each country's own survey where it has one, a worldwide set otherwise; long trips are fitted later on corridor counts.
- 2026-10-16 (T-511): people in households with no car (per cell, census by tract or LSOA, else cars per household through one curve fitted on US tracts and English LSOAs) choose rail against walking the whole way in the local layers, walking to stations; with no bus their other way is weak, so on New York's real network rail trips rose 44% (Manhattan 65% against ACS 51%) before T-509 refits the rail constant; the regional layer is left without.
- 2026-10-16 (Anita, T-509): calibration aims at plausible, not exact: the game has no stigma against transit (real New York's better-off riders avoid the subway), so a moderate overshoot of measured rail where that stigma is strong (New York) is acceptable and the fit should not drive rail down to match it.
- 2026-10-16 (Anita, T-513): homes without a car get a third choice, "other" (bus, taxi, bike as one mode): the drive's road at 12 km/h door to door plus an 8-minute wait weighed 1.37, no parking, plus one constant `other_asc` (default 10 perceived minutes, the best of 0/10/20/30 on Manhattan, Brooklyn, the Bronx and Queens against ACS; T-509 fits it). Speed and wait are fixed worldwide: city bus operating speeds run about 10-20 km/h with stops (12 for the walk to the stop folded in) and urban headways 5-20 minutes. Other is shown as driving for now.
- 2026-10-16 (Anita, T-519, T-520): a station selection leaves the map as it was (its catchment view showed what a bubble selection shows, in another look); its commuters stay in the inspector. Lines to a selection are on trial and may be undone: the bubble stubs by `LINES_TO_SELECTION` in `map/commuterView.ts`.
- 2026-10-16 (T-517): no resource carries more trains than its capacity: whole-line fleets are cut to fit by progressive filling (each line in proportion to its fleet's rate), and the trains that do not fit wait at the ends, rather than slowing every train through an overloaded stretch (the old fluid queue let 500 trains run 125 an hour through track rated for 24). Track sections' base drops from 90 s to 70 s so 100% is the achievable maximum, not a practical one: 34 trains/h at 80 km/h, 25 at 160, 16 at 320, matching the Victoria line (36), Chuo Rapid (30), Thameslink (24) and Tokaido Shinkansen (17). Station crossovers only delay; legacy fixed-frequency schedules keep the old queue.
- 2026-10-16 (T-509 stage 4): everyday rates are each country's national survey legs by purpose halved (the leg home under the activity it leaves; return-home trips spread where a survey codes them apart), errands with escorting and personal business, business trips out; distance scales match the game's mean outing length to the survey's mean straight-line leg; a country with no open survey takes the surveys' median. Rates went up in the US and Europe (New York 0.5 to 1.0 outings a resident) and down in Japan, Korea and Brazil, and social scales fell from 6 km to 2-3 km; the everyday rail level left over is mode choice, for stage 5/6.
- 2026-10-16 (T-509 stage 4): no job-centre (competing destinations) term in the commute gravity: a decay that flattens for big centres moved west Tokyo to the central wards from 14.6% to 15.0% (census 17.1%) at best and misplaced more trips; the gap is the radial rail lines (T-465).
- 2026-10-16 (T-526): with a build tool on, the track tag sits above the pointer (the tool's own tag keeps below right), the bubbles' tag shows only off the network and the tool's tag, and the station demand circles' tag is left out. "Level" became "elevation" in player text, the Settings display "Height" too; "ground level" (plain English) and "cross on the level" (at grade) stay.
- 2026-10-16 (T-509 stage 6): mode choice fitted jointly on six cities' real networks (New York at half weight, Anita's moderate-overshoot rule; Toronto held out): `asc_rail` 0 -> 0.4, a new everyday rail constant `everyday_asc_rail` 0 -> -1.1 (everyday -0.7 in all), a separate everyday walk curve `everyday_walk_d0_km` 1.7 -> 1.9; `w_walk` 1.5, `other_asc` 10 and `crowd_alpha` 2 stay (the first two did not move, the third is not identified by Seoul Line 9's express split). The fitted six score 7.77 -> 6.05, Toronto 0.36 -> 0.56. The level wants very different values by country (alone: New York -0.9, Tokyo 1.5); a country setting for it is not added. notes/T-509.md section 14.
- 2026-10-16 (T-509 stage 5-6): the calibration objective floors shares at 1% in its logits (near-zero rural areas dominated), scores Paris's census transit share one-sided (it includes the bus, so only an overshoot counts), and adds NHTS 2017's New York rail and walk shares of everyday legs by purpose.
- 2026-10-16 (Anita, T-509): the fitted values stand (asc_rail 0.4, everyday rail -1.1 below commutes, everyday walk centre 1.9 km) although held-out Toronto worsened; Japan gets a country rail constant of +0.5 (Sendai's value; Tokyo commute rail about 35% against the census 52%), the first country tailoring of mode choice.
- 2026-10-16 (T-509 stage 7): a country's rail constant is a sim table by the pack's `country` (`api::COUNTRY_ASC_RAIL`, Japan +0.5), not a header block, so a new pack of that country gets it with no data step; it applies to commutes, everyday trips and the regional layer's commutes between zones (by the home zone's country), which keep the local commute's constant because they are the same commuters; the regional layer's longer trips keep 0 until fitted on corridor counts. notes/T-509.md section 15.
- 2026-10-16 (Anita, T-520, T-524, T-530): the fading rays to a selection stay (bubble stubs coloured by their source and faded by size, and the inter-region lines); they are no longer on trial.
- 2026-10-16 (Anita, T-530): inter-region bubbles shrink somewhat as the map zooms out but never because of what is on screen: radius ∝ sqrt(trips) x 2^(0.35 (zoom - 6.5)), the 0.35 and the zoom-6.5 scales picked by eye to look about as before T-525 at zooms 4-7. The Stations tab lists complexes, not stations.
- 2026-10-16 (T-492): a pack that downloads whole and fails to open is retried once after 10 s, then left out until the page is reloaded, rather than every 60 s like a failed download: a pack that is whole and does not open twice is broken, not caught mid-write. Reuses T-484's flag and path.
- 2026-10-16 (T-534): a station's cross street skips any street within 30° of the track's direction where the street comes nearest. A neighbourhood's size proxy is the distance from its label to the nearest other place of its OSM kind on the same place list (at most 3 km, so about 3 km across at most), with OSM's population at most 200,000 where given; worked out in the app from the place lists as they are, no pipeline field. Areas would be better but only the Geofabrik-read packs have them; Manhattan (a "suburb", 1.7M people, nearest fellow suburb 12 km) fails both tests, Harlem and Williamsburg ("quarters" about 3.4 km from the next) fail the first.
- 2026-10-16 (T-531): money by region uses the Lines and Stations tabs' region, the zone in view behind the same place-or-world toggle (one setting for all three tabs), not the selection: a zone is never a selection kind, and the toggle already names it. Cash, the fleet, the fare curve and the built value stay network-wide.
- 2026-10-16 (T-531): a line's money divides between regions by its stops: fares by boardings (the base) and passenger-km (the per-km part), running cost by route km, a stretch between two stops in different regions half to each, which approximates km in each region without the border's crossing point. Construction by the newly built track's cost at its midpoint's region; new trains by the edited line's route km. Regions keep their own ledgers in the clock worker from the moment they are kept, so a game saved before T-531 shows regions only from its load.
- 2026-10-16 (T-510): a lone sparse cell (under 1,250 trip ends, merged by neither earlier rule, in a busy parent or alone in a sparse one) joins the touching merged cell of its zone with the fewest trip ends, since a light group's centre moves least for the demand it carries; one pass, never joining another joiner; lone cells touching only each other merge among themselves. No trip-end cap, but no grown or new group may reach past 700 m from its centre (about a T-044 merged parent's own spread): a group sheds its farthest joiner until it fits. The cap costs about a tenth of the cells the rule would otherwise save; uncapped, groups reached 1.3 km.
- 2026-10-16 (Anita, T-535): the unattended city pipeline is approved with all eight of its recommendations: GHSL urban centres over 1M outside the US (functional urban areas over 1M in Europe), the UK first, then the rest of Europe, then by size; unresearched countries ship at the global baseline; extents are functional urban areas snapped to whole admin units; a pack holds at most 40M people; zones between cities keep T-300's rule until about 100 packs; hand-made packs are never rebuilt; an agent skims each batch's contact sheet and parked list. Batches may run overnight at up to 12 of 16 cores, and at 5 while she is using the machine.
- 2026-10-16 (T-460): the land-use cap (1 km2 an area, spread on a 0.003 degree grid) is tier 0's default for every build, T-459's values kept: London's and Toronto's 2 km zone jobs came out 0.07 and 0.8 points better misplaced, Lagos byte-identical. A config turns it off with 0; a city's cached rows (osm_jobs_<id>.csv.gz) keep the settings they were made with. notes/T-460.md.
- 2026-10-16 (T-039): the dark theme's basemap is OpenFreeMap Positron repainted in Dark Matter's manner (paint properties only, same tiles and OSM/OpenMapTiles attribution), not OpenFreeMap's own dark style, so switching theme reloads nothing and the hidden railways, fonts and name rules stay one code path. Lines keep their colours and take a near-black casing (#141517) instead of white; station dots stay white with their dark rim, which reads on the dark map. Light or dark is a choice beside the three colours, not a fourth colour, and is remembered per browser.
- 2026-10-16 (T-387): hospital beds where OSM has no tag pool over the US packs where beds are well tagged (Baltimore, Boston, Philadelphia, Springfield, Washington), New York held out for good as the check, since outside the US no pack tags more than a few; beds follow the square root of floor area (no exponent or estimator beat it on the held-out pack, 20.1% misplaced by 2 km zone), and the pack total is the World Bank's national beds per 1,000. One site keeps at most 5% of the pack's beds or 1,500 if more, never over 4,000: room above the 0.3-3.4% the largest site holds in packs over 10M, a small pack's one regional hospital kept, and the gate's 5,000 bound safe where Japan's and Korea's 12.6-12.8 a 1,000 meet few mapped sites. notes/T-387.md.
- 2026-10-16 (T-536): the world plan's unit level is the finest geoBoundaries level (ADM2-4) where the median person's unit holds 20,000 people, weighted by people rather than by units (by units, Brazil's municipalities fail and Sao Paulo would be the state whole); a level need cover only half the country, and a first-level unit it misses a tenth of takes the finest level covering it (Russia's ADM2 has no Moscow).
- 2026-10-16 (T-536): outside Europe a metro's area is its urban centre plus every 2015 FUA the centre holds half the people of, and the FUA around the centre's population-weighted point; Europe's FUAs and the 1M bar both use 2025 GHS-POP, not the FUA file's 2015 figure. A metro's people and every share are counted on the 1 km grid by pixel centre.
- 2026-10-16 (T-536): Geofabrik extracts are the covering extract or greedy set of children with the least total area at each level, then none inside another; this picks county extracts in England (Manchester: Cheshire and Greater Manchester) rather than England whole, so a batch reads more, smaller files.
- 2026-10-16 (T-543): a save thumbnail is drawn on the land colour of the theme the page was in when the slot was saved (the clock worker is told by a "shade" message), not transparent over the current panel, so a picture is always self-consistent; older transparent pictures still show over the current land. A switched-off blueprint in the dark theme is a dim middle with lighter grey rails (a ghost), since a near-white middle would look the same as a selected track there.
- 2026-10-16 (T-537): a planned city's jobs blend follows ILO's modelled vulnerable employment share (World Bank SL.EMP.VULN.ZS, every country), held to 0.2-0.5, since the informal employment indicator T-535 named (SL.ISV.IFRM.ZS) is archived. It is national, agriculture included, so it overstates a city's informal work; at the 0.5 ceiling that does not matter (Nigeria 66%, India 72%), in between it may (China 33%, Brazil 26%).
- 2026-10-16 (T-537): a planned city's outline takes, for each plan unit, the OSM admin relation of any level overlapping it best (at least half their union), compared on land less inland water: on the coastline-clipped shapes alone Lagos Island and Lagos Mainland matched nothing, since geoBoundaries stops at the shore and OSM's Lagos Island runs over the lagoon. The output keeps the coastline-clipped OSM shapes, lagoons in, as nigeria.py did.
- 2026-10-16 (T-550): the cities list is a section of the Money tab, under the ledger, not a fifth dock button (T-532) or a Stations tab section: its money is the region ledger the tab already shows, and a click on a row moves the camera, which changes the region the ledger follows. Its money figure is yesterday's fares less running trains, so a build day does not swamp it (new trains and construction are lines in its hover); before a first full day it is today so far.
- 2026-10-16 (T-552): riders may ride a line the wrong way to board upstream of a stop where full trains leave them behind (Tokyo's gyaku-nori), as the left-behind wait makes worth it; kept as real behaviour rather than barred. It comes earlier than in life where crush is set too low (T-560) and at a terminus, where the back rider is charged a fresh wait though they would stay on the train.
- 2026-10-16 (T-557, Anita): no change penalty. A change of train costs only its wait and any walk between platforms (by distance at the walk weight; one demand station's platforms, side by side or stacked, no walk); `transfer_min` stays as a knob at 0 for calibration runs. No refit: the real New York save's rail trips rise 6.7% (9,777,718 to 10,433,452 a day), left for the deep calibration. This replaces the reasoning in "Complexes are transfer points" (T-161) that the 5 minutes fall on every change.
- 2026-10-16 (T-555): a planned city takes from its country module only the keys that hold for any city in the country (`plan_hooks`); a hand pack's one-city keys stay out and tier 0's rule stands: the fitted curves (London, Paris, Berlin) give way to the borrowed one, hub O&D judgements to 1, Sao Paulo's survey jobs, work from home and curve, Shanghai's and Mumbai's typed-in district tables, Seoul's registration population (each city's renames since 2020 need mapping by hand, Gunwi to Daegu in 2023 say) and Toronto's DA population and tract jobs (canada.main, not tier 0 hooks) are not carried. Japan has no planned rows and no `plan_hooks`.
- 2026-10-16 (T-555): English, French and German planned rows take the census's own polygons for the plan's units as their outline (ONS ITL3 BGC, départements from geo.api's communes, VG250 Kreise) rather than osm_units' OSM match, so the small areas the population and jobs come from nest in it exactly; the plan's NUTS-3 polygons are 1:3M. Not cut along packs: the plan gives each whole unit to one metro.
- 2026-10-16 (T-562): with no open world car register (WHO's counts motorcycles, the World Bank's IRF series ended, OICA's is gone), a country without small areas takes the first of a national register (`nocar.COUNTRIES`), a DHS survey's households with a car from 2015, Eurostat's cars per 1,000 x household size, a DHS survey from 2010, and last households with a car from GDP per head (a logit fitted on those 100 countries, rmse 0.14) or the income group's median; surveys come before Eurostat because the register curve runs low outside rich countries (Albania 19% without a car against its DHS's 63%). Hong Kong, Singapore and Taiwan are typed in, the rule cannot hold for them. Wales, Scotland and Northern Ireland use their census small areas like England.
- 2026-10-16 (T-538): the pack gate FAILs a check only where it compares a pack with its own source (an OSM mask against OSM water, a GHSL-built pack against GHSL, tier 0 proxy jobs against the 1% cell rule); against another source it WARNs, since GHSL undercounts Shizuoka by a third and TIGER's lakes sit off OSM's. Jobs are bounded per cell absolutely (50,000, New York's densest is 48,900) and the 2 km zone at 20% (city centres reach 15%), not the design's 1% and 5%; the school floor is 0.06 outings a resident (Akita, Sorachi).
- 2026-10-16 (Anita, T-534): a neighbourhood counts as small up to 3.5 km from its nearest fellow (was 3 km), so Harlem, Williamsburg and Midtown can name stations; Manhattan (12 km) still cannot.
- 2026-10-16 (Anita, T-558): walking becomes a third mode in the choice as notes/T-558.md proposes (a three-way logit; walk cost 1.5 a minute for 10 minutes then 4; today's walking held by two constants until the deep calibration; car-free homes walk at the same cost; the trip explorer shows walking as one number in a hover).
- 2026-10-16 (Anita, T-552): riding back on a full line is barred ("i dont like it ... have people just wait at the station"): a rider never changes onto the same line, so a full train's riders wait at their station; replaces the T-552 line that kept it as real behaviour.
- 2026-10-16 (Anita, T-551): for the deep calibration, the car carries an ownership cost of about $10 a day (her figure; AAA's $22 counts depreciation and finance, which few drivers weigh per trip); the walk (1.5) and wait (1.37) weights are to be tested there against weights of 1 (her wish to drop them for clarity).
- 2026-10-16 (T-556): the world plan adds a smaller urban centre within 3 km of a metro's centres to its area (the centre only, not its FUA), rather than reaching 10 km past every centre or using a growth-scaled FUA: Lagos's gap was Ikorodu, a separate centre with its own FUA, not growth, and the 10 km reach gave the same Lagos result for three times the growth elsewhere and +2.5% in Europe. Epe (30 km out) stays outside Lagos. A rerun keeps earlier rows' order and ids.
- 2026-10-16 (T-573): tier 0's GHSL height weight stops at 40 m (`HEIGHT_CAP_M`); above it a cell's weight grows with volume alone. The best height cap on Toronto (2 km zone jobs misplaced 18.05 to 17.79%), London unchanged at 26.06%, Berlin's Treptowers cell 62k to 22k jobs. A cap on one cell's share of the term (1%) scored a little better but depends on how much of a region the outline takes. notes/T-572.md.
- 2026-10-16 (T-574): a tier 0 OSM school site holds at most 3,000 pupils and one cell's sites 6,000 together (a campus mapped as several schools, Vile Parle's eight), the excess to the other sites in proportion; what the sites cannot hold goes to school where the children live (53% in Lagos, 63% in Mumbai, where OSM maps a school for every 6-8 thousand pupils), rather than packing every site past any real size. notes/T-572.md.
- 2026-10-16 (T-572): Shanghai's, Mumbai's, Seoul's and Berlin's OSM job rows rebuilt with T-460's cap; the other tier 0 packs keep pre-cap rows until their next rebuild (T-578), since they pass the gate.
- 2026-10-16 (T-558): the walking constants are 2.86 for commutes and 0.35 more for everyday trips, the compromise between New York and London (each city alone wants about 0.1-0.15 higher in New York than in London), each within 0.3 points of its walking of before rather than one city exact. Rail takes from walking its share of the zone pair's (commutes) or the pair's samples' (everyday) walking with no rail, so the walk totals, the views and the solve agree to the trip; an everyday pair's walk is priced at its samples' mean distance. notes/T-558.md.
- 2026-10-16 (Anita, T-539, T-584): where two packs meet, commuting across the pack line is damped by default at Japan's fitted 0.4, replacing T-181's "no factor on admin-1 lines by default" for pack lines: Japan's census sends 1.9x too many commuters across with no factor, pack lines are mostly prefecture or state lines, and the US already damps its state lines (0.23, Washington 0.49).
- 2026-10-16 (Anita, T-585): the platform wait weight goes (1.37 for the first 5 minutes, 0.4 after): a wait counts as plain minutes, half the headway, with no refit; the walk weight (1.5) waits for the deep calibration's test against 1.
- 2026-10-17 (T-540): the runner gates a region's rows after registering them with their market packs (the browser check opens the pack in the game, and the runner's Vite reads the registry once); a pack that parks leaves the registry again and its market place stays in `market.places` unused, since place ids are positions. notes/T-540.md.
- 2026-10-17 (T-540): after a joint gravity solve the group is gated again, and a FAIL restores every member from a backup taken just before and parks the group's new packs, rather than shipping a group nobody checked or leaving an existing pack changed. notes/T-540.md.
- 2026-10-17 (T-580, T-540): a new pack within 50 km of a Northeast pack parks by rule before its gate; moving the Northeast onto the `--packs` solve waits for a row that needs it (US cities are paused). notes/T-540.md.
- 2026-10-17 (T-540): `--jobs` counts cores in all, one city at a time per 3 cores; a step over its time limit (`STEP_LIMIT_S`, build 90 minutes, four times that over 15M people) is stopped and its row parks; a region's downloads and the caches the runner made are deleted once the region is joined, not at the batch's end. notes/T-540.md.
- 2026-10-17 (T-540): osm_units' match leaves out inland water under 0.1 km2 (`WET_MIN_KM2`) and clips the water to each shape's box: Rotterdam's 66,000 ditches stalled it past 20 minutes; Owerri, Aba and Aboh match the same units as before. notes/T-540.md.
- 2026-10-17 (T-585): every priced wait went to plain minutes with the platform wait: the extra wait for a full train, a change's wait (the same boarding cost) and "other"'s fixed 8-minute bus wait (11 perceived before), with `other_asc` not refitted, as the decision says. notes/T-585.md.
- 2026-10-17 (T-587): a commute zone pair walks at its 36 sampled cell pairs' mean walk share (six home and six job cells a zone, pairs within 3 squares), not at one cost of their logsum or mean distance (the logsum let a pair walk like its shortest cell pair: 200% in the short bands); the choice with rail takes the one cost giving that mean. `asc_walk` 2.80 (everyday +0.41, so 3.21 as before) keeps New York's and London's commute walkers together where they were. notes/T-587.md.
- 2026-10-17 (T-540, T-590): until joint_gravity takes a pack's country from somewhere other than its header, the runner parks a new row whose joint group holds a pack with no header country outside the Northeast (tor, lon, par, sao), rather than solve it across a false US border (factor 0.05); Portsmouth was solved so in the pilot and left shipped for T-590 to re-solve. notes/T-540.md.
- 2026-10-17 (T-590, T-591): the guard is gone: joint_gravity reads a pack's country from its header, else the registry's `marketRow.country`, else US. The gate's `joint_borders` check compares only countries (each "CC-..." label of a border pair against the registry's country for its packs, and every expected country present); it does not judge a same-country pair's factor. notes/T-540.md "T-590".
- 2026-10-17 (Anita, T-621): the walk weight stays 1.5 and is not fitted: perceived time is widely used and understood, so the game should explain it better rather than drop it (T-633).
- 2026-10-17 (Anita, T-621, T-620): car ownership is amortized per car trip as a constant plus a rate per km, about $3 for a typical trip and about $10 a day over a driver's trips; a drive to a station pays it on its own shorter distance. One worldwide dollar figure, as fares are. The deep calibration fixes it (not fitted) and refits the rail and walking constants around it.
- 2026-10-17 (Anita, T-632): trips too short for any train to win stay in the demand the player plays against for now; showing them apart may be tried later.
- 2026-10-17 (Anita, T-621, T-636): the demand model aims at fun and sensible over fitting real cities: Japan's rail constant, the commute-against-everyday rail gap (`everyday_asc_rail`) and per-country walking constants go; real cities become loose checks, and her two ceiling tests the targets (A: 140 km/h, free, 3-minute walks, at least 80% of all trips by rail; B: 100 km/h, 3-minute walks, $1.50 + $0.05/km, at least 40%). Short trips may get a steeper choice (higher `beta`, lower `damp_min`, long trips unchanged); the values wait on T-636's measurements.
- 2026-10-17 (Anita, T-636): the new model's values: ownership on ($1.50 a trip + $0.10/km), rail constant 0 everywhere, one walking constant worth 10 perceived minutes (was about 56: it made a 10-minute walk score -41 against the car), the whole-trip walk at a flat 1.5 a minute, parking's base 5 minutes (was 3; ownership's $1.50 already acts as a fixed car cost), the choice twice as sharp (`beta` 0.10, long trips softened past 60 minutes as before). Targets as rail of rail + car: A at least 85%, B at least 50%. The commute-against-everyday rail gap stays as a parameter at 0, a tool that may come back; Japan's constant and country walking constants are not expected back.
- 2026-10-17 (Anita, T-656): an automatic bridge's ramps (200 m) lie wholly on land either side of the water, rather than reaching over it to need only about 100 m of land; a crossing without that room is refused.
- 2026-10-17 (T-624): a station near a joint on constructed track joins the built pieces under it, as it does on blueprint, rather than asking the player to place stations before Construct (Anita left it to the orchestrator); a joint between unlike track (single and double) still refuses.
- 2026-10-17 (Anita, T-586): the trip explorer matches the view it was opened from: its headline is the far bubble's own sums (the selection's far end under the purpose filter and day), and the route detail is one representative trip, the purpose with most of those trips and that purpose's busiest period by default (agent's pick).
- 2026-10-17 (T-621): car ownership is $1.50 a car trip plus $0.10 a km of its road, the fare's shape: NHTS 2017 weekday drivers make 2.9 car trips of 16.6 km, $9.20 a day, and the model's New York and London drivers come to about $10.40 and $8.20 (notes/T-621.md "Stage 2"). The sim defaults stay 0 until the deep calibration's fit is applied.
- 2026-10-17 (Anita, T-636): the new model is the sim's default, final values: `beta` 0.10 (`damp_min` 60), `asc_rail` 0 (Japan's constant and the everyday walking constant removed; `everyday_asc_rail` stays as a parameter at 0), the walking constant `walk_bonus_min` 10 perceived minutes (the logit's `asc_walk` is that times `beta`), the whole-trip walk a flat 1.5 a minute, `park_min` 5, ownership $1.50 a trip + $0.10/km on. Her tests as rail of rail + car: A 86.3% New York, 85.5% London (target 85), B 79.3% and 78.9% (target 50); the real networks carry x3.1 New York's station counts and x2.0 London's (notes/T-636.md Round 2). Market packs keep the old parking base of 3 until rebuilt.

- 2026-10-19 (T-670, T-672): keys N new line, L edit stops, C construct blueprints (E stays the blueprint pick key); during a colour drag the badge and stop dots follow every move and the map line follows by one edit at most every 0.2 s, longer on a slow machine (6 times the last edit's round trip), plus the final edit on release (one lineLook edit costs about 25 ms of main thread, 100 ms at 4x slowdown); the stops list moves to the top only while stops are being edited.

- 2026-10-19 (T-677): the start menu's New game beside saved games starts in a new slot ("Game N", first written once its network is not empty, so an abandoned new game leaves no card) rather than replacing the remembered one; with no saves it uses the original slot. The map and UI are set up under the menu and only the workers wait for the pick; `?continue=1` (the game opened last, no menu) is the switch checks use to reopen a stored game.

- 2026-10-19 (T-684, T-685): points count passenger-km in the region where riders board (each boarding carries its line's average km that period) rather than the journey's origin, which the solve does not keep per km; points are not scaled by the 100x economy; a star is earned from the average day's final train share only, never the weekday or weekend view, so switching the day shown cannot earn it.

- 2026-10-19 (T-692, Anita): Express copy is removed (the button, the `express` flag on `copyLine`, `express_copy` and `twin_station` in sim/src/track/wasm.rs; quad track and twin stations stay). The crowding dot goes by the level's average wait for a train with room, in minutes (`crowdColour(avg.wait)`): green to 2, yellow at 3, orange at 10, red at 15, deep red from 20; the score still grades the hover's worst-period line and `crowdClass`. The green plateau to 2 min is my choice, so that her 2.2 min reads nearly green.

- 2026-10-19 (T-700): a zone's cells split by bank for walking the whole way: a part is the cells its representative reaches by a walk of at most 1.5 x the straight line + 500 m, at most 4 parts a zone (`routes::walk_tables`); a cell centred in a river joins the first part that reaches it, and a subzone takes the part most of its people are in. The thresholds are my choice; Shanghai splits 127 of 1,879 zones.

- 2026-10-19 (T-710, Anita): riders choose station pairs by the time difference relative to the trip: scale 20 / min(25 + 3 x straight km, 60) perceived minutes, in four bands (0-3, 3-7, 7-12, 12+ km), set by judgement against her examples rather than fitted; crowding counts from 70% of crush, not from the seats (she found 80%'s 1.27 x crush at the worst peak segment too much). The regional layer keeps 0.2.

- 2026-10-19 (T-708): Double up adds the express pair as blueprint even beside built track (Construct then pays, as for any blueprint) rather than charging at once; it covers the whole run through plain joints and stations (twins added) up to a junction (where it swings in) or another end (loose end beside); left of the selected piece, else right. My choices; the alternative was one piece at a time, which T-248 found tedious.

- 2026-10-19 (T-715): a load refunds ledger entries after the save's clock (the gap is settled again), with the running cost of days the ledger has dropped worked out on the loaded network at the rate its kept days show; this repairs saves hit by the stale-clock bug (Anita's bodega: -13,123 to +893.6 US$M) but invents no fares for the stretch the money stood still.

- 2026-10-19 (T-717, Anita): a change of trains costs a fixed 2 perceived minutes (3 at first; Anita lowered it to 2 after the measurements, London's Enfield branch going from 1.41 to 1.00 of crush) plus the walk between the two platforms' centres (snapshot stations, so inside a merged demand station too) plus its wait, replacing T-557's none. At a station of several platforms the search adds the expected walk over the pooled departures (by frequency) to the arrival, rather than to each line, so which lines join the pool does not depend on the platform arrived at; the explorer shows each route's own walk.

- 2026-10-19 (T-718, Anita): the regional layer chooses station pairs as the local layers do (T-710's band scale, 0.33 a perceived minute from 12 km) and prices rail at their logsum before the damped mode choice, instead of weighing every pair at the damped mode scale (whose logsum counted 64 near-equal pairs 40-80 minutes cheaper than any one of them); its walk-against-ride nesting takes 0.33 too. The rule lives in the sim: no market pack needs rebuilding.
- 2026-10-19 (T-722): the cash (top bar and Money tab) has four significant figures ("$11.36B"), every other money figure three, negatives red (Anita). Station circle area is proportional to riders with a fixed STATION_FULL of 140,000 riders = the zoom's circle radius, chosen so the real New York save's biggest total (Penn Station's complex, 681,000) is 100 px in radius (200 across) at zoom 13 (Anita: "max at around 200px"); a fixed constant rather than scaling to the biggest circle shown, so a circle means the same riders in every mode and save.

- 2026-10-19 (T-717, Anita): past 60 perceived minutes the station-pair scale softens further, 20 / 60 x sqrt(60 / T) (`rel_soft` 1), so a long trip minds a minute less ("for long trips we want people to be a bit less time sensitive"); T is the car's perceived minutes in the regional layer (the mode choice's damping measure, T-495) and the band's rough rail length in the local ones, whose tables are per band (only the 12 km band moves, 0.33 to 0.29). Regional rail +6-12% on the measured corridors, cities unchanged to 0.1%. The shape is the damping's, chosen by the orchestrator and approved by Anita.

- 2026-10-19 (T-736, T-294): each game hour's money follows the boarding account's in-period arrival shape rather than an even share of its period; fares and points land over a trailing window as long as the network's car-km-weighted one-way run (rides end, a Friday night ride pays on Saturday) and running costs over the same window (trains run on past a timetable change); rides count when they start. One network-wide window keeps regions adding up; whole days keep their totals, a single day can move the minutes that cross midnight.
- 2026-10-19 (T-732): the resting hover's list is for the select tool only (a build tool's tag is about track), and while it is up the hover tags stay put inside the hull of the rest point and the tag with its list, so the pointer can reach an entry; a rest inside that hull does not refresh the list.
- 2026-10-19 (T-728, T-729, T-731; Anita): sandbox is the normal game with cash hidden and nothing refused for money, line money and the Money tab kept; a normal game can become a sandbox once (confirmed), never back. New games suggest New York, London, Paris, Tokyo and Toronto. Cities not opened are grey bubbles, opened by a click without moving the camera; old saves start with the cities their stations reach plus the one in view.
- 2026-10-19 (T-733; Anita): the crossover tool goes, one crossover is placed past each quad station at full cost; single track is hidden from Build but kept in the sim; platform length stays and lines get a train length in cars capped by their shortest platform, platforms never growing on their own.
- 2026-10-19 (T-733, my choices): the automatic crossover's nearest end is 50 m clear of the platform end (`QUAD_CROSSOVER_CLEAR`, so a 200 m platform can grow to 300 m), off the station's first track end, else its second; "no room" is whatever `crossover_op` or the edit's checks refuse, and the stations then go in without it, as they do when the cash covers them but not the crossover. The station ghost does not draw it (the blueprint total and the double-up price include it). Removing a station leaves its crossover as track. The worker keeps the crossover edit and preview for checks; a browser that remembered single track starts on double.
- 2026-10-19 (T-733, my choices): a platform shortened below a line's set train length caps its trains (clamp) but the setting is kept, so lengthening the platform again restores it; stepping the line's train length up to the cap stores 0 (the most that fits, as a new line), so it follows the platforms from then on. The Build tab's "New stations" heading went, the platform row now sits under the track choice with the hover "Platform length for new stations", so it shows without scrolling in an 850 px window at the default and narrowest dock (at 768 px it still scrolls).
- 2026-10-19 (T-646, my choices): "between two stops" is each stretch between neighbouring stops, offered in the stop list while editing (plus one "add all" when several stretches have some) rather than a pick of two stops on the map; "on the track between" is the line's own path, so a quad twin on the other pair or a station on a branch is not added; order is path order; the preview is the list itself, Add or Cancel, one setStops edit.
- 2026-10-19 (T-752; the orchestrator's call, with Anita's leave): switching a line between train counts and headways never buys or sells a train; on track other lines' fixed rates already overfill, the trains its rate cannot use are held out of service (the alerts' "held") rather than trimmed, and such a fleet counts as settled.
- 2026-10-19 (T-727; the orchestrator's call on the design's recommendations, with Anita's leave): snapping follows only the track the real-network layer draws (New York to San Francisco is then the Lake Shore Limited and the Zephyr); the fit's tolerance is 25 m; Settings > Real network stays the off switch, on by default at 0.15 opacity and 0.4 while snapping; a click on a subway follows subways and intercity routes keep to main lines; real tunnels and viaducts are ignored, only water gets a bridge. notes/T-727.md.
- 2026-10-19 (T-728, my choices): sandbox is a flag in the track model (`set_unlimited`): every money check passes, the edit still pays and the cash goes below zero, so the ledgers and line money are untouched. Where the cash was, the top bar says "Sandbox" and the Money tab "Since day 1" (the cash less the $3B start); saves cards of a sandbox show no cash. The Settings switch is under its own "Game" heading, last before Explore a trip.
- 2026-10-19 (T-728, my choices): New game always takes a new slot, also with nothing saved (the original `autosave` slot is no longer used for new games). A new game starts at the first time from day 1 07:00 on the game clock at which it is 07:00 in the start city, so every start opens on its own morning (Tokyo: Tue 2 07:00 local); New York and map starts are unchanged. The world map start is centre 20° E 30° N at zoom 1.8. `?new=1` is a normal New York game with New York opened; `&mode=sandbox` and `&city=<id>` change it.
- 2026-10-19 (T-728, my choices): the search matches a word start of the name (case and accents ignored), biggest by local trips first, eight rows; the country comes from the registry's market row code named by the browser, so no pipeline change was needed. A city found by search that is one of the five lights its button; a second click on a lit button unlights it (back to a map start).
- 2026-10-19 (T-731, my choices): the seven real networks ship as `packs/real/<id>.save` (`realnet.py ship` copies realnet's game files into `data/packs/real/`, fixing Toronto's header, whose `city` was the whole registry entry; the build copies the folder). A real start keeps the file's network, fleet and fares and takes a new game's clock, $3B, ledgers and stars; "Real network" is ticked by default, and its slot is written at once. The files stay in their older frames and are converted on each start, as any old save is.
- 2026-10-19 (T-730, T-741, my choices): hover lightening mixes toward white in both themes: a bubble's colour 30%, the whole line under the pointer 30% (all its strokes, not the piece), bare track that edge, and a station's rim 60% across its whole complex, since its white has nothing to lighten. Only with the select tool; build tools keep their ghost as the one cue, and a train under the pointer lights nothing. Pointing at a resting list entry lights that thing instead of what the pointer rests on.
- 2026-10-19 (T-729, my choices): grey bubbles show whatever the demand buttons say (the default is None, and they are how a city is opened), are drawn over every other disc and take the select tool's click first, and hide while a bubble or city selection shows partner bubbles at the same centres. Over a grey city Homes and Destinations stay usable. A clicked city opens bare wherever the view is, not only when viewed. A station's reach adds zones to the list as soon as the reach is known, which is also how an old save's reached zones are added.
- 2026-10-19 (T-724, my choices): a station's suggested cross street is read from MapLibre's tile manager (a private part, `map.style.tileManagers`), only the tiles within 400 m, each tile's features kept until its data changes; when an upgrade moves it, the public query of every tile is used and works as before, only slower. The pick and snap grids are built in idle time, so the one or two moves after an edit scan everything as before rather than wait for a build.
- 2026-10-19 (T-727d, my choices): routing runs in its own worker with its own track model for the fit, not the clock worker (New York to Emeryville fits in 2.1-2.4 s in WASM, which would stop the clock). The ring comes from the layer's rendered features, so it shows only on drawn track; where no graph file covers a click the tool draws an ordinary segment. A second click before the route is ready waits for it rather than drawing an ordinary segment. Curve labels while drawing show only on screen and never overlapping.
- 2026-10-19 (T-727d, my choices): a route through more than 16 vertices is bridged river by river (`bridge::bridge_long`, corners 4 km either side), water read at a quarter of the usual step since the pieces are checked separately; where a river has no room for ramps (Lake Hamana on the Tokaido Shinkansen) the stretch is raised whole from the last ramp-length leg before it to the first after, rather than refusing the route. Short segments bridge as before.
- 2026-10-19 (T-737): the Japan stress check (checks/t737-japan.mjs) reports a demand worker's failure on the whole country as known (T-756) rather than failing, so it stays a frame-time check until T-756 lands (DEMAND_STRICT=1 fails on it). The stress network's trains an hour come from MLIT passing counts (6% of a day's riders in the peak hour over the seats), a rough rule for a test save only.
- 2026-10-19 (T-648, my choices): a clock state waits on the main thread for its trips (the worker posts them right after it, at most 250 ms), so a frame never draws old trips on new run profiles; this adds the worker's first boarding hour (about 10 ms) before panels show an edit. A stop click is one `insertStop` edit in the worker instead of a query then `setStops`. While a line's account is redone after a stop edit, its past-24-hour counts stay on their stations (a new stop at 0) rather than following the old stop order.
- 2026-10-19 (T-756, my choices): a city's demand layers solve on the stations within `CITY_REACH_M` (30 km) of its cells' box, each line kept from one stop before its first such stop to one after its last (pads, whose crowding-input boardings are the riders aboard); the market's part stays on the whole cluster. A population's crowding rounds now stop when its own loads settle, not the cluster's. Seven cities of Japan: figures within 0.06% (a small city's within 10 trips), boardings 33,177,340 against 33,177,357; notes/T-756.md.
- 2026-10-19 (T-757, my choices): a demand worker that fails after startup is not restarted: a red alert under the demand buttons says "Demand ran out of memory" or "Demand stopped", its hover that a reload starts it again, and the figures shown stay. Restarting the worker means downloading and opening its packs again; left until a network that fails is not just one too big for memory.
- 2026-10-19 (T-760): a dead-end stop's platform middle never passes halfway to the next stop, so two stops closer than their platforms stand at the same point rather than past each other; the demand client's clamp of the fare distance is gone.
- 2026-10-19 (T-761, orchestrator's call and my choices): the regional market on a big cluster keeps its figures exact (option 1 of notes/T-756.md, not a coarser station set): past 256 MB of strategies (`MARKET_STRATEGY_BYTES`) its destinations are searched one at a time, twice, keeping 256 MB of strategies between the sweeps (a bigger keep would shorten the second sweep but the whole country already peaks at 3.4 GB a worker). The commutes' OD and the far-zone table are held sparse on every network. notes/T-761.md.
- 2026-10-20 (T-776, my choices): the registry's only per-zone detail is the world market's rows, and the market needs every zone's row at once (pairs to unopened places and each place's departures elsewhere), so they are one file, app/src/game/market-rows.json, not a file per zone or country; it stays tracked beside registry.json (a hashed asset in a build, so the 8800 snapshot ships its commit's rows rather than data/'s), is downloaded once by the page beside the first market pack and sent with it to the solving workers (fetching it as soon as a zone started to open cost a new game about 0.2 s, notes/T-776.md). A per-country split would need the market to work from partial rows (T-777). The index keeps each zone's country (from its row) for the new game sheet and the joint solve.
- 2026-10-20 (T-712, Anita): the plane's 60 minutes at the airports go back to 90 (T-378's design value, reversing its 60), and flights pay a fare that leans toward rail, since building long-distance rail should feel rewarding. Her follow-up: keep the fare a simple base plus per km for player predictability, no per-route data.
- 2026-10-20 (T-712, my call within that): the fare is $100 + 5 cents a km (was $50 + 9 cents): published averages (DOT Table 6 2025, JAL FY2023, DLR Low Cost Monitor 2026) barely rise with distance, $150 at 1,000 km sits between the US's ~$215 and Japan's and Germany's ~$110, and short flights, where rail competes, cost more than the old rule said. Flights past 1,250 km are a little cheaper than before. notes/T-712.md.
- 2026-10-20 (T-712, orchestrator's call): a car carried over water pays the crossing, one fixed price and time each: the Channel $150 a car and 60 minutes, the Tsugaru Strait (Seikan is rail only) $170 and 240 minutes, shared by two. Toll bridges stay free like toll roads (T-270).
- 2026-10-20 (T-780, my call in Anita's direction): the market's longer-trip zone pairs are dealt in chunks of about 1 trip a day each way, on by default: each zone's trips with each later place kept exactly, the start fixed by the zones' ids so a network always gives the same figures. All of Japan: 12.4M pairs to 3.1M, a worker's solve 1,256-1,288 s to 965 s and 3.1-3.6 GB to 2.2-2.4 GB; the network's rail moved under 0.002% a period, the market's 0.06-0.17%, the worst line 0.25%. `?perfOff=chunks` solves every pair. notes/T-780.md.
- 2026-10-20 (T-780, my call): Anita's literal chunks (a pair's riders dealt to a few station pairs) were built and dropped: a further eighth off the market's time, but station entries moved 6-7% (worst 64-78%), since a busy pair then sends everyone through its top station pairs. The streamed pass keeps 256 MB of strategies (T-767's 768 MB halved the second sweep natively, but in WASM growing the heap for them cost minutes or 600 MB of peak).
- 2026-10-21 (T-801, orchestrator and my calls): grey city bubbles undone at Anita's word: every zone loads for the view on pan and draws its ordinary bubbles; the save keeps `opened` (start zone, then zones reached) but nothing reads it, and `?opened=all` is gone (checks that still pass it are unaffected). With no market result yet the zoomed-out bubbles show local trips alone. Pan cost: solving workers 125 to 311 MB on the Northeast pan, notes/T-801.md.
- 2026-10-21 (T-802, my call): station faces have three moods and no red. The figures behind the old red face were not wrong so much as harsh (the whole gap since the last train left, which is the longest wait and twice the average, doubled again if even one rider was left behind), and a red face with dark brows reads as angry whatever its brows do. Tired (drained yellow, heavy lids) is an average wait of 10 minutes or more (orange on the crowding dot's scale); sad (blue, raised brows, tear) is 20 or more riders left behind by the train before, one face's worth, and wins over tired. The platform's sound is the lower one for either. Real save over eleven game hours: 26% of arrivals red before, 11% tired and 0.7% sad after. notes/T-802.md.
- 2026-10-21 (T-801, orchestrator's call): a zone's market pack opens only when the zone is reached, a station is within 30 km of its box, or it is selected, never for viewing: undoing the grey bubbles with viewing markets grew the solving workers from 125 to 311 MB on the Northeast pan; without them 8 MB. A viewed unreached zone shows no inter-region trips zoomed out. A start on the world map turns Homes on, and the demand buttons work zoomed out over dead space. notes/T-801.md.
- 2026-10-21 (T-795, my call): the Detail setting has four steps shifting the zoom bands by whole levels (1.8 zoom each), not a continuous slider: a level is 7 times the area, so a finer slider would only move where the switch happens. Steps: res 9 from zoom 15.4, 13.6 (the old banding), 11.8 (new default) or 10. Shifting the zoom thresholds rather than the level index lets res 6 still show far out at the default (below zoom 8.2). All levels are built every time, so nothing rebuilds when it changes; the real save at zoom 12 draws 61,577 discs at the default against 30,808 before, a draw under 0.3 ms either way.
- 2026-10-21 (T-804, my call): the Train share ramp blends in OKLab, so pink to green passes through a muted brown-grey in the middle, not a bright orange or yellow (a hue-arc blend would give that); a bubble's position on the ramp is its train share of legs, linear, so a mostly-driving city sits at the pink end. The worker still builds the mode mix and the main thread repaints the discs (about 7 ms on the real save), so a picker drag needs no rebuild. The zone bubbles zoomed out and the zone inspector's mark follow the choice too; the station-by-mode bars and swatches keep the mode colours.
- 2026-10-21 (T-809): long trips between two countries of the EU 27, Norway, Iceland, Switzerland, Liechtenstein and the UK take a flat border factor of 0.5 in place of 0.26 x income^1.5: an open border's savings do not scale with income. Set on London - Paris (about 1.1x published rail and air); land pairs (Paris - Brussels, Amsterdam - Brussels) stay about half to two thirds low, since a factor near 1 would double London - Paris. Commutes keep their own border factors (T-810). notes/T-809.md.
- 2026-10-21 (T-793, my call): the loading bar's solve stage counts rounds, not one expected time: each published round is progress, a round is expected as long as the pace measured over the rounds so far times its weight (1, 1.1, 1.05, 0.55 first estimates, from the real save's round times), and the bar stays under full until the work is done, so a finished load shows about 95%. The remembered solve (LAST_SOLVE) only sets the pace before round 0 lands. notes/T-793.md.
- 2026-10-21 (T-820, my call): the line view's average load is the day's rider-seconds aboard over trips times their end-to-end time times crush capacity, for today's kind of day (the account the crowding dots use), not the past 24 hours: a time-weighted share, about 1% low because dwell at the last stop is not in the trip time. T-813: a line with no stops yet counts in the viewed city for names and colours.
- 2026-10-21 (T-818, my call, after Anita's suggestion): a line's crowding figure (the panel's dots, the Lines tab's train count) averages the wait for a train with room over riders each counted by the fare they pay on the line, base plus per-km times the km a boarder at that stop rides on average (the account lets each stop off a share of whoever is aboard, so that is the km). Not per stretch: riders left behind wait at platforms, and through riders board at the same crowded city stops as locals; the hover names the stop with the most waiting instead. The test London - Paris line's medium dot goes from 27 min (deep red) to 9 min (orange). notes/T-818.md.
- 2026-10-21 (T-819, my call): the traffic view outlines in red the lines on track at capacity for the demand level in force (as the capacity tags), a full section's edges and both tracks through a full flat crossing; full platforms, termini and junction moves keep only their tags, as they are points, not stretches. The outline is a red band 1.5 px each side outside the casing (the casing stays between, so red lines show it too), one extra instanced pass that skips every other segment: about 0.1 ms GPU a frame on the real save.
- 2026-10-21 (T-816, my call): the real-network graph takes undrawn track that fills a gap in the drawn network (a group of ways no line runs over, at most 3 km, meeting drawn track at two nodes or more; not yards, sidings, spurs, industrial, military, tourist or test track), and a snapped route longer than twice the clicks' distance apart plus 20 km is refused rather than placed, since with no turning back a click on the line beside the one meant otherwise goes round a loop hundreds of km long. notes/T-816.md.
- 2026-10-21 (T-833, my picks): under Train share bubbles (now the default) the mode bars and swatches use car red #e61e1e and train blue #1e3ce6 as before, walking yellow #f0c814 (a walker pulls a bubble towards red and the ramp has no green) and "other" orange #f07a14, kept apart from the yellow and the red. "Other" is planes only (the longer-trip tables' plane column): bus, taxi and bike are inside driving in the sim's figures, so they have no bar part to colour. Mode mix keeps green and the muted plane. T-832: the top bar's date is the game's own "Mon 1" (weekday and day number), always shown, centred between the cash and the time; the game has no calendar date. T-831: all points figures use three significant digits, thousands included ("12.3k").
- 2026-10-21 (T-839, Anita): T-816's detour limit on snapped routes is dropped; a route of any length is placed. The player sees a long way round in the ghost before clicking, and a real mountain line can run twice the straight distance. Instead the real track the first click can reach is lit up (T-839b), since telling what is connected is the harder problem. notes/T-839.md.
- 2026-10-21 (T-838, Anita "like 20% shorter"; T-839b, my call): the snap search costs high-speed chains (flag 8) at 0.8 of their length, so London - Paris takes HS1 and the LGV by Lille (493 km) over Amiens (451 km); the length shown and built is the real one. The reach highlight counts every chain the search's moves reach (no turn over 60 degrees, either family), with no distance limit, not a range cut at what a 2 s search would cover: the whole joined graph is 10-25 ms to walk. It is drawn in the layer's own colours at 0.9 over the layer dimmed to 0.25. notes/T-839.md.
- 2026-10-21 (T-835, Anita): a departure belongs to the period in force when the train reaches the line's middle: every period's departures leave both ends half the mean one-way run early (at most 2 h), and trains still run end to end. Chosen over busier periods starting early (+3.6 to +7% car-km). notes/T-835.md.
- 2026-10-21 (T-835, orchestrator's call): the one exception to the midpoint rule: a period followed by no service departs until its own end, not half a run before it, so a line without night service keeps its last evening trains and a peak-only line its last peak trains (+1.0% car-km on the real save); a period after no service still starts half a run early. First-train wait over demand's, real save: +15.4% before, +25.4% with the plain rule, +13.5% with the exception. notes/T-835.md.
- 2026-10-21 (T-843, my call): the Max speed ramp continues past blue with 240 km/h indigo (#5b4fc9) and 320 violet (#a03fc2), not 200 and 320: the design speeds are 160, 200, 250 and 320, and 240 is the even step (80 km/h like the others) that puts 200 halfway from blue to indigo and 250 just past it. The violet sits apart from the line palette's purple (#8a3fb8) by being brighter and pinker, and Max speed draws no lines anyway.

- 2026-10-21 (T-830, T-841, my calls): the Blueprint added notice gives the length and price only when the last preview answers the very request placed, else just "Blueprint added." (an extra preview per click would queue behind the hover previews and could be dropped); the selection bar is capped at the map width minus 520 px and wraps its groups rather than shrinking its buttons further, so it clears the debug panel at a 1400 px viewport; a straight piece whose elevation cannot change (a flat crossing, a node with other track) says so in a notice instead of showing a disabled button.- 2026-10-21 (T-861, my calls): the date is a 365-day calendar with game day 1 as January 1 and no leap day, the weekday staying the day number's; T-862 may want day 1 moved off New Year's Day if holidays play as weekends. The sky is a 24-hour timeline (6 h back, 18 ahead) rather than a picture of the sky now, so coming sunrises, peaks and later holiday marks show; its colours follow the real sun at the view's centre on that date (standard time, so New York's June sunrise is 04:24). The time and date sit above the sky, not on it, to stay legible over every sky colour. Demand band colours red #d64545, yellow #e5b22a, blue #3f7fd4 (brighter in dark).
- 2026-10-21 (T-859, my calls): removing a station a line stops at goes through; "disable the line" is the existing broken state: the station leaves the line's stops and the line is routed through the rest (no new reversal), else left broken (not running, "Broken" notice, fixed with Edit stops), all in one edit, so one undo restores blueprint. The confirmation shows for any removal that changes a line, blueprint included, and also names lines the removed track alone leaves broken; the track model's check no longer refuses a line set in an edit whose path runs over removed track.
- 2026-10-21 (T-860, my call): a station on a loose end joins the drawn route's joints within its platform first (as a station splitting a piece does), so the end dot works when the last piece is shorter than the platform; only loose ends, not a joint in the middle of track, which the pointer's track spots already cover.
- 2026-10-21 (T-862, my calls): holidays from the Python `holidays` package (national public holidays, no substitute days, all 250 countries it has, loaded as a lazy 239 kB chunk rather than only the registry's 44 countries, so a new country needs no rerun); a game year takes the real non-leap year with the same January 1 weekday, so weekday-tied holidays keep their weekday and Easter moves; game day 1 moved from January 1 (T-861) to Monday January 27, the first Monday that is no registered country's holiday, so a new game never opens on one. Marks only for countries with the player's stations; a holiday running as a weekend day is stage 2, T-872 (notes/T-862.md).
- 2026-10-21 (T-870, my calls): an opened other route shows its own effective time (its terms' sum) and the difference from the main route's own sum, not from the main route's shown Effective time, which is the pooled figure with choice of routes taken off; comparing against that would make every other route look much worse than it is. No Rust change: the worker already sends every listed route's legs. The opened route is drawn on the map at full strength on top of the main one.
- 2026-10-21 (T-868, my calls): a train's riders getting off at each of its next four stops are its load arriving there (its own loads in the boarding account) times the account's alighting share for the line, stop and period, ported to TypeScript (`alightShares` in `ui/inspector/TrainStops.tsx`, a copy of `alight_shares` in boarding.rs), so arriving less off plus on is the load leaving. The list counts everyone aboard at each stop, including riders who board on the way, rather than only those aboard now. Minutes to each stop come from the train's own running profile, so a hold on the way shows; the list moves at each game minute and each stop.
- 2026-10-21 (T-866, my calls): the trip explorer between zones reads the market's own unit, the 6 km area pair, and gives the trips' average over every pair between the two ends weighted by their trips (not one representative pair), because the market has no point-to-point trip: a selection near Penn Station boards at its whole 6 km area's stations (Astor Pl in the test). Each pair's choice is recomputed with the solve's expressions for every pair rail can serve, so it does not follow the sampled solve's pair-by-pair rail (T-780) and can differ a little from the bubbles' rail. It shows inter-region trips only; commutes between zones (the bubbles count them) choose by another rule and are left out, said in the hover. Walks to stations show as felt minutes, since the market keeps only access weights. The trip tool's end outside the focused zone picks the far 6 km area nearest it (6 km), else the place (60 km).
- 2026-10-21 (T-858, my calls): the pointers to a selected bubble fade in opacity from the source bubble's edge (opaque) to nothing at the tip, linearly, at constant width, and the zone pointers of a selected city do the same (one shader); a fade alone leaves less ink, so the start alpha went 0.5 to 0.8 (`STUB_ALPHA`, `ZONE_LINE_ALPHA`). Opaque at the source because the pointer is a short stub off that bubble, not a line that reaches the selection.
- 2026-10-21 (T-863, my calls): intercity bubbles are a pie rim (blue train share from 12 o'clock clockwise, the rest red, `MODE_COLOURS` train and drive in every theme and colour scheme; walking counts in the red) round a 35% white fill, the rim 30% of the radius clamped to 2-4.5 px; the selected city keeps its dark outline outside the rim; hover brightens as before. `PIE_BUBBLES` in `map/intercityView.ts` turns it off. No bubble is grey any more (T-801), so the only grey is a bubble nobody travels in, kept as a plain grey disc.
- 2026-10-21 (T-872, my call): a line's holidays come from its first stop's zone's country, the same rule as its local time; the demand views' Today follows the country under the view centre. Running costs stay grouped by local time only, since they do not change with the kind of day. notes/T-862.md.
- 2026-10-21 (T-871, my calls): zones load for viewing (the viewed zone's pack, exact outlines, water masks ahead) only above zoom 7.5, one level out from the zones' own bubbles (8.5); reached zones and their markets load at every zoom, so no demand figure depends on the zoom. The outline layer draws only zones within a view's width of the screen. No extra unloading: T-301 already closes viewed-only zones a minute after they leave. Europe pan long tasks 15-58 s to 0.4-4 s a pass; notes/T-871.md.
- 2026-10-21 (T-865, my calls): Shift+drag draws a bubble box at once (MapLibre's Shift-drag zoom is off while the bubbles are on in the select tool). The second box is told from a new first one by where it starts: with a selection, a box started on a selected bubble adds to it, one started anywhere else takes the far bubbles and opens the explorer; to start a new first group, Esc or an empty click clears the selection first. Esc while drawing drops only the box. A group is named by its bubble count at the zoom on screen and the station nearest its trip-weighted centre within 3 km ("29 areas around 79 St"). Its routes and times are a sampled average, not every cell pair: the busiest cell of the top 3 bubbles a side by trips, each to each (at most 9 asks), weighted by near legs times far trips as if independent (the far end per near cell is not known on the main thread); routes merged by lines and last station, weighted by pair weight x train chance x route share, top 5 kept. notes/T-865.md.
- 2026-10-21 (T-874, my call): commute zone pairs under 0.1 trips a period are dealt their trips in chunks of about 0.1, per origin zone and 16 km block of destination zones, seeded by the zone and block (as the market's, T-780). Native New York weekday 38.9 to 29.9 s (pairs -49%), Paris -18%, browser load and edit -17% and -19%; stations at most 0.1%, home cell rail share 0.05 points. 0.3 would take another 8 points of time for a worst small station of 0.4%. `?perfOff=chunks` solves every pair. notes/T-874.md.
- 2026-10-21 (T-882, my calls): the cute scene keeps the train, a 13 px red car with lit windows running left to right once every 3 game hours, so it moves at every speed and stops with the clock; it reads at 148 px and is the only part not fixed to time. Its movement sets the redraw rate (about 30 draws a second at the top speed, 0.7 ms each); drop it to save that if the bar's cost ever matters. The now line went: the sun or moon marks now.
- 2026-10-21 (T-885, T-881, my calls): place-to-place trips in the explorer are the market's areas weighted (every 6 km area pair between the two places counted whole, `market_trip_between`), not one trip between their centres, since the market decides them pair by pair and the headline should add up. A selected city keeps its city view only at the overview zoom (8.5 and out); zoomed in the local bubbles show under it, so a city then a local bubble or group works. A Shift box holding inter-region bubbles takes those over any local far bubbles in it (a box past the zone's edge is aimed at them); the two models are never added together. A group of far places is each asked and merged (trips added, times trip weighted, top 5 routes). A 6 km area or a group as the first end is not offered: nothing on the map picks one first. Trip tool with the start outside the zone uses the end's cell as destinations; both outside, the nearest zones' places. notes/T-885.md.
- 2026-10-21 (T-884, my calls): the loading bar is the time since the pick against that time plus the time still expected, and each slot's last wait is remembered in the browser (12 slots, new games under "new") and trusted over the step model until the solve shows its pace, then held to 0.7-1.5 times it; ignored when the network's stops changed by over a fifth. A bar that is a little slow and then jumps when the card goes was preferred to one near full and waiting, so late steps are expected to go on (15% of a late round's time, a quarter of the wait past the remembered length). The pick and its game are one wait (no restart to 0). Zone placing answers are kept (for good once the nearby boundaries are in) to end the freeze behind it. notes/T-884.md.
- 2026-10-22 (T-896, Anita): the banner's train no longer runs; it sits at now as the present-time marker, now moved to the middle (12 h back, 12 ahead, which also narrows the holiday hover to 12 h ahead), with the sun or moon kept above it at the top edge, where they do not clash. This replaces T-882's running train and the redraw rate it set.
2026-10-22 (T-894, my calls): a save's picture is one panel per city with stations, the most stations first, at most four (1 full, 2 side by side, 3 as a 60% panel plus two stacked, 4 a 2 x 2 grid), each framed on its own stations, a city with under 8% of the biggest's stations left out (the real save's New Haven, Litchfield and New London fringes would otherwise shrink New York to a quarter), rather than one frame over the whole network or a whole-network inset (at 240 x 160 an inset is too small to say anything). Ranked by stations, not riders, since the worker has stations at save time. City names and per-city station counts are HTML over the picture, not baked in, so they stay crisp, use the player's font and carry a hover; their colours come from the theme the picture was drawn in. The picture grew from 192 x 128 to 240 x 160 in both the saves panel and the start menu (wider start box), PNG kept (about 10-40 KB). Old pictures show as they are until next saved; no migration.
- 2026-10-22 (T-889, my calls): a save's zones and markets open from its built stations before the clock worker has worked its network out, as prefetched zones (never solved unless the network reaches them); a market opened ahead that the network does not want is closed on the network's arrival, with a console warning, rather than trusting the read to match. Market packs wait on the page until none is still downloading, so the workers compose the market once; one stuck download holds the others back, as it already held the solve. `?perfOff=openAhead,batchOpen` restore the old ways. notes/T-889.md.
- 2026-10-22 (T-889, my call): the bar expects one market composition per load (1.5 s, shared among its market packs) plus 20 ms per MB, not 450 ms a pack; a game load's bar job survives the camera passing over dead space before its network arrives, and demand is not idle then. notes/T-889.md.
- 2026-10-22 (T-902, Anita's asks, my calls): clouds are seeded by the local day's number (35% of days none, 45% one, 20% two), at fixed times within the day and heights, so a day's clouds are the same whenever it is shown; the sky's hover keeps only coming holidays (none means no hover) rather than moving them to the band, whose hover is the one line "Currently low demand until 06:00" in the game's 24-hour format.
- 2026-10-22 (T-904, my call): while a newly focused zone's footprint is still loading, the outside shade covers the whole map, the new zone and the old one included, rather than being emptied (the flash) or kept on the old zone's outline (the old zone would read as in focus).
- 2026-10-22 (T-905, my calls): the stations' Riders column under Origins or Transfers shows the map circle's own figure, riders a day from the demand solve shown, since only the solve knows a boarding's origin; under None or Total it stays boardings in the past 24 hours, so a station's transfers can exceed its 24-hour boardings. No split of the 24-hour count by the solve's share (an invented split). The list sorts were never saved (kept while the page is open), so the new lines default needs no migration.
- 2026-10-22 (T-910, T-911, my calls): the explorer's "no train" line names what holds the trip back (too short to drive to a station, no station within a drive, the far end's walk, or the start even by car: the bigger of the two ends' felt minutes), from a new answer slot saying whether the home end could drive; counted shares are dealt to whole trips by largest remainder, under half a trip counted as one ("Under 1 trip", its likeliest mode 100%), route shares dealt among the counted train riders with zero-rider routes hidden; the representative trip's own chances stay unrounded, and hovers keep the model's share.
- 2026-10-22 (T-913, my calls): the track tool snaps only where a real-network graph exists (inside a built region's noritetsu outline or within 0.05 degrees of one, from `packs/realgraph/index.json`), and elsewhere places track as with the switch off; nothing counts as covered until the router answers. Graphs now built for Europe and Canada too (37 regions, 23 MB) and still copied into the build rather than fetched from R2 (T-727b's choice), since all 37 together add about 10 MB and each loads only when reached.
- 2026-10-22 (T-909, coordinator's call on my measurements): trips of about one person are grouped for display only, not in demand: a selection's far end deals its cells under 10 trips a day (as the hover counts, at every zoom level) along a Hilbert curve into bubbles of at least 10, totals exact. Dealing the model's zone pairs at 10 a day moved small stations 5.5% and still left the far end as specks, since it is a spread of 2 km zone trips over every cell (notes/T-909.md).
- 2026-10-22 (T-727e, orchestrator's call): graphs built for every noritetsu region (118, 39.6 MB) and still copied into the build; moving them to R2 beside noritetsu's tiles is a publishing decision left for Anita. Jordan and Mauritius have no drawn track, so no graph and no snapping.
- 2026-10-22 (T-923, my call): when the two clicks' snap candidates share no family (metro under a main-line station, rail at the other end), each click also takes the other's family within 400 m rather than refusing; 400 m covers a big station's platforms from a metro under it (Tianjin 149 m) without reaching a different station.
- 2026-10-22 (T-918, Anita's pick on my measurements): rail against the car is 1 / (1 + (rail / car)^3), symmetric about equal costs, fading to nothing from 3 to 4 times either way; it replaces the logistic at 0.10 a perceived minute, its damping past an hour and T-854's 5% floor and ceiling. Of four longer-tailed shapes it was the only one with visibly longer tails that kept test A (92 against 85; the two-logistic mix passed only at 10-15% of riders, the arctan and a softer logistic failed). Real networks +3 to +5%, New York's commuter rail +36%. Walking keeps its logit; the regional market keeps its damped three-way logit with the plane, while commutes between zones take the curve. notes/T-918.md.
- 2026-10-22 (T-924, my call): the published page finds its data from a `tw-data` meta that tools/deploy.py writes into the website's index.html, not by probing for a local file as religiondots does: no failing request on every published load (smoke counts console errors), and the data version is fixed by the same file that names the hashed scripts. Data goes to R2 under a dated version prefix, immutable; JSON gzipped, .bin and .rgr raw (Range reads; .rgr is inflated by the game).
- 2026-10-22 (T-953, my call): after a lost WebGL context the renderer and the demand canvas compile again and replay the inputs they keep (network, layers, preview, highlight, disc and line data) rather than the page reloading or asking the workers to resend; the kept inputs are references the app already holds, so it costs no memory beyond the demand lines' arrays. notes/T-953.md.
- 2026-10-22 (T-953, my call): every fragment shader that measures px runs at highp (STROKE_FS was the last mediump one); mediump is fp16 on phones and the cost of highp there is negligible for these shaders. PROBE_FS stays mediump, it writes a constant.
- 2026-10-22 (T-951, my call): on a phone held upright the dock goes under the map rather than being made narrower; a side dock cannot be narrow enough to leave the map more than about 40% of 400 px (its eight tabs need about 230 px), while stacked the map and top bar get the whole width. Revisit if she wants the side dock kept.
- 2026-10-23 (T-976, my call): "not built in" means a zone the game has not opened (`openedZones`: the start zone and the zones a station reaches), not a place missing from the registry (every registry zone is playable) and not merely not loaded in this session (zones load for the view on pan). Only the zoomed-out overview draws it: a plain dot of half the bubble's radius, floored to whole CSS px with a floor of 1 px so a small zone stays visible, black by default; setting Unopened cities: Black, Maroon, Grey, lifted on the dark theme where black would vanish. With a zone selected the trip bubbles are unchanged.
- 2026-10-23 (T-978, my call): the basemap's city, town and village names stay but lose the black dot, and are centred on the place instead of sitting above where the dot was (map/basemap.ts).
- 2026-10-23 (T-979, my call): when the trip explorer gets no answer on a final result it asks once more after 3 s (a pack it needs may still be opening), then says "The demand model has no answer for this trip." instead of waiting; while a solve is running it keeps waiting for the next result as before.
- 2026-10-23 (T-979, my call): a trip selection with an end outside the focused zone opens that zone's market pack, as a bubble selection does (tens of MB in each solving worker, closed by the usual sweep); a trip inside one zone does not.
- 2026-10-23 (T-1005, Anita): a zone grown to the floor skips a neighbour whose union would wrap another zone (half of its area in the union's convex hull and half of its zone border held by the union); thresholds 0.5 and 0.5 approved, since at 0.6 and 0.6 the pairs halve. Metros are not in the pool, so a wrap of a metro is left alone. The planner changes only planned rows; built rows are regrouped later (part 2).
- 2026-10-23 (T-1008, my call): the market's corridors reach the main thread as a table of what no network or period changes (a period's trips are the table's times a period factor, as Rust computes them) plus the pairs with rail; if Rust ever makes per-period trips other than a scaled day, the table needs per-period columns again.
- 2026-10-23 (T-1020, my call): a free-drawn track piece reaches 500 km at most (straight line between its ends); past that there is no ghost, the tag says why and the click is refused; real snap routes are not capped. Chosen over ending the run when the view jumps, since players pan mid-run on purpose; 500 km keeps a long intercity piece (Tokyo-Osaka) to one click with a ghost of about 0.15 s.
- 2026-10-23 (T-1020, my call): the station reach table and the trains' line blocks are built on first use instead of in idle time: a few ms each, where waiting for idle meant scanning every station and trip on each move in a big game.
- 2026-10-23 (T-1025, my call): the realm shade is drawn by a custom layer (map/shadeLayer.ts) from triangles cut on the main thread, not a GeoJSON fill: MapLibre tiles round vertices and then misdraw thin or close parts as long stray triangles, and one GeoJSON feature per triangle cost geojson-vt 0.35-2.3 s a change. The custom layer has no query, filter or style expression; the shade needs none.
- 2026-10-23 (T-1038, my call): the placement ghost is purplish blue, `#5b4ee8` (light) and `#9488ff` (dark), on the map's ordinary casing instead of white on ink, at least 3 px wide with 1.5 px of casing a side; distinct from blueprint's light blue, the real network's slate and the metro's mauve. The Select tool's ink highlight band is unchanged.
- 2026-10-23 (T-1039, my call): a preview the client no longer wants (a newer one waits, Esc, a drag ended) is answered `stale` by the clock worker if it has not started or is waiting for water masks; one already being worked out is not interrupted (the track model has no way to stop midway). The router drops a search or fit when the pointer leaves real track, not only when a newer route is asked.
- 2026-10-23 (T-1039, my call): a water mask installed under drawn track reworks only the water spans and prices of edges whose pieces reach into the mask's box, instead of re-deriving the whole network; nothing else an edge derives reads water (`Network::add_water`, tested against a full rebuild).
- 2026-10-23 (T-1039, my call): the tag's spinner shows after 300 ms of a click or the newest ghost waiting on the clock worker, and the real-snap spinner's threshold drops from 500 to 300 ms to match.
- 2026-10-23 (T-1047, my call): a zone the player's track crosses counts as one a station reaches for what the player sees (opened list, realm shown, figures), for built track; blueprint track shows its realms only, as a blueprint station does. Crossed zones open bare in the aux worker, never in the solves: with no station in reach a solve gives them the same figures, and putting them in every worker took a 1,960 km test save's final from 15-19 s to 33 s and each solving worker from 440 to 910 MB (notes/T-1047.md).
- 2026-10-23 (T-1052, my call): T-1026 (the shade cut off the main thread) stays a separate P3: no pan triggers a cut, only a realm show (one 166-334 ms task per click); worth taking if a save with many realms shown loads slowly.
- 2026-10-23 (T-1054, my call): the ghost's outline is black in the light theme and the map's ink (#e6e4e0) in the dark one, since black on the dark ground vanished and the ghost then read as blueprint; minimum 4 px across the rails (was 3 px plus 1.5 px casing a side) with a 1 px outline. The ghost's middle still takes the Elevation and Max speed colours and red when invalid.
- 2026-10-23 (T-1058, my call): the station tool's trip count has its own radius, 1 km (1 mile in US units, matching the rings' unit), apart from any walk reach in demand; the line has no day phrase, as in Anita's wording.
- 2026-10-23 (T-1017, my call): `enrich_medical` drops one facility that lies off every peopled cell instead of failing the build, as `enrich_parks` does for one park (T-897); two or more still fail. Leon's Sunland Center, an institution whose residents the usa module leaves out, stopped the build.
- 2026-10-23 (T-1017, my call): three US pilot gate relaxes, each with its reason in the plan row: Chatham's pack-alone mean commute 27.2 km (LODES OD 18.5 km in the same counties, 18.8 km after the joint solve), Orlando's top cell of 64,639 jobs (LODES puts Walt Disney World on one block), Hernando's test line with no school trip.
- 2026-10-23 (T-1017, my call): US zone names: a merged zone shows its biggest city, or the name the whole goes by where the biggest place is an unincorporated community (Sarasota, not Port Charlotte); a shown name another zone already has or a planned row will have takes the state ("Columbus, GA", "Rome, GA"), as the plan's county names do.
- 2026-10-23 (T-1055, my call): every boarding rides at least one stop, fixed in the strategy search (a station is offered a route node at the cost of riding on) rather than by dropping rail-less strategies afterwards, so a pair whose only "rail" was boarding to walk off has no rail option and its trips go to walking and the car, and riders already on board still leave wherever that is best. It also closes T-200. notes/T-1055.md.
- 2026-10-23 (T-1056, my call): Anita's walking rule measured but not built in the same task: it cannot go in the route search (which does not know the trip's ends) and belongs in the station-pair choice, four places including the commutes' eight-lane loop, a build of its own (T-1059). Effect small: rail -2.6% New York, -3.6% Tokyo, ceilings about -1 point. notes/T-1055.md.
- 2026-10-23 (T-1026, my call): the realm shade's cut moved to a worker; when the shown realms change the circles and cities change at once and the shade swaps in one frame 60-140 ms later, keeping its old shape meanwhile (no fade). The cut is split into pieces by box (0.1 deg margin) and pieces are kept between cuts; a piece reaching the world's edge falls back to one world ring. Zone outlines go to MapLibre by `updateData`, 100 zones a task. notes/T-1006.md "T-1026".
- 2026-10-23 (T-1064, my call): AWS builds run on a fresh one-time spot machine per wave (m7a.8xlarge, Ubuntu, Python 3.9 by uv with the venv's pins), builds only; gate and join stay at home, since the gate needs the whole shipped world, the registry and a browser. Over Jean's one long-lived box per project: a stopped box cannot be spot or resized as spot, so a wave pays on-demand (5.4 times) and its disk bills idle (about $8 a month) while each wave reads other states. Machines get presigned URLs, never credentials or an IAM role. notes/T-1064.md.
- 2026-10-23 (T-1067, my call): waiting floods are fixed on the boarding side, not in demand: a rider whom full trains keep more than 30 minutes past the first train they came for goes another way, and one platform side holds at most 5,000 waiting (a 300 m platform's side 5 m deep at 0.3 m² a rider, about the longest real platforms, Tokyo's and Mumbai's 15-car trains), later arrivals turned away. Neither boards; demand's riders and fares are unchanged (fares: T-1069; demand loading lines past what they carry: T-1070). notes/T-1067.md.
- 2026-10-23 (T-1067, Anita): riders give up after 90 minutes left behind, not 30: "maybe 30min is a bit aggressive for infrequent lines. maybe 1-2hrs? i think platforms absorbing through giveup and cap kinda makes sense." 90 is the middle of her range; the 5,000 cap and the platform-side approach stay. notes/T-1067.md.
- 2026-10-23 (T-1071, my call): a train count change keeps taking effect at the termini (T-206's hand-overs) rather than respacing running trains toward even spacing, so positions stay a pure function of service and clock and fast-forward stays free. The visible jumps were the browser drawing the change late; it now takes effect a moment ahead (0.25 s plus twice the worker's last edit time, at most 2 s of wall time, at the clock's rate) and the renderer swaps it in then.
- 2026-10-23 (T-1071, my call): saves carry the service's eras and shifts after the network bytes (ERA1), so a game saved mid-change loads with its trains where they were; a save with nothing changing over is unchanged.
- 2026-10-23 (T-1059, my call): the walking rule compares straight metres on both sides (access walks' mean distance, the strategy's expected walks between stations and platforms, the trip's distance times the pack's walk factor round water), counts no walk for a drive to a station, and leaves trips between cities out; a trip it leaves with no station pair gets no train rather than a fallback. notes/T-1059.md.
- 2026-10-23 (T-1065, my call): `enrich_medical` fails a build only when over 5% of its facilities (and more than one) lie 500 m from every cell, not on a second miss: new hospitals on a town's empty edge and on bases (San Antonio, El Paso) are real places, while a lookup fault misses most of them. The misses are logged and their visits dropped.
- 2026-10-23 (T-1065, my call): a wave's first AWS machine builds a check row and waits for a "go" put from home after comparing it with a Windows build; once Linux is proven a later machine's go can follow the launch at once. `land.py` refuses only during a gate, a joint solve or another run's claim on the rows, so builds of other rows (T-1078) do not block a landing.
- 2026-10-23 (T-1059, my call): the commutes keep their eight-lane loop; only lanes under a per-origin bound look at their station pairs, taking the refused pairs off after loading, or loading the kept ones alone where more than half is refused (f32 cancellation). Costs about 10% of a solve; a tighter bound could cut it if speed matters later. notes/T-1059.md.
- 2026-10-23 (T-1087, Anita): zones should be areas that mostly commute within themselves, so no ring zones round a city: "if we have a ring area around a city, thats like the opposite of what we're trying to do. we should be thinking kinda like voronoi." Replaces T-1005's "leave metro rings alone"; the planner rule is being designed (notes/T-1087.md), and US waves 3 and 4 wait for it.
- 2026-10-23 (T-1084, Anita): far drive tables coarser (16 km supercells) taken, "coarser long distance car routes makes sense"; the walk part (no walk search from 4 km blocks) dropped: "when theres small towns some people with very short commutes will still want to walk within the town".
- 2026-10-23 (T-1094, my call): the last train length set is kept per game in the save header (`newCars`), not per browser, so a different game starts from its own; "set to the cap" counts as a set length of 0, so a new line then follows its platforms.
- 2026-10-23 (T-1099, my call): the account icon stays in the tab bar as well as at the top of Settings (Anita asked to add it there, and to move only Controls).
- 2026-10-23 (T-1100, my call): the save preview drops the station rings only; grey track under the lines and the station count beside a city's name stay.
- 2026-10-23 (T-1101, my call): bubbles were already off by default, so only the track display default changed (Line); a start on the world map still turns Homes on (T-801), since it is picked from the bubbles. Stored choices stay, which includes Anita's own browser.
- 2026-10-23 (T-1106, Anita): stored settings are kept (no `VERSION` bump for T-1101); a start on the world map keeps turning Homes on; the account icon is only in Settings (and the start menu), not the tab bar. Replaces T-1099's "both".
- 2026-10-23 (T-1102, my call): the selected-bubble pad (T-936) went from 2 px to 1 px, which is both the "min radius" and how much bigger every bubble draws while one is selected.

- 2026-10-23 (T-1097, my call): high-speed weight zoomed out is 1.4x width at zoom 9 and below, 1.25x at 11, none from 13 (HSR_WIDTH in map/renderer.ts), and the light centre line comes in only as a built stroke widens from 3 to 5 px; far out a high-speed line reads as a slightly heavier solid line rather than a paler one.
- 2026-10-23 (T-1098, my call): two countries keep "the same" holiday when it falls on the same local date and has the same name or names counted as one in SAME (game/holidays.ts: Labor Day, Workers' Day, May Day and the like; Christmas; Boxing Day and Saint Stephen's Day; New Year's Day). A name shared on different dates (US Labor Day in September) stays apart. "Built in" is built track or stations; blueprint does not count.
- 2026-10-23 (T-1088, Anita): T-1087's rule is the planner's ("i think im mostly happy with the new rule"): nearest-centre joins, the wrap test counting metro rows and built packs and never overridden (80% rule), and the ring cut. Commuter belts: "we can let metros take neighboring counties but it doesnt have to be a hard rule", so an unbuilt metro takes a neighbouring county sending 25% of its workers into it where the cap and contiguity allow, only where commuting data exists (the US); built metros keep their extents.
- 2026-10-23 (T-1088, my call): two departures from T-1087's dry-run logic. A zone under 80% of the floor whose every join wraps joins anyway (the cut follows), never another state's neighbour: the dry run's cross-region rescue put Kootenai with Montana and Billings with Wyoming. A cut's half whose every join wraps stays as it is, under the floor if need be, since rejoining rebuilt the ring each round; so New Mexico has four small zones (0.23-0.40M) round Albuquerque and El Paso.
- 2026-10-24 (T-1078, my call): Franklin County MO's zone (a ring from St. Louis's edge to Springfield's) shows "Rolla", its central town, not its biggest place Nixa, a Springfield suburb on its edge; the other rings keep the biggest-place rule (Greene County OH's shows Beavercreek). The ring zones go with T-1088's replan.
- 2026-10-24 (T-1084, agent's call): metros keep 8 km far drive supercells; only countryside zones take 16 km. Atlanta saves a minute of seven and its rail moves as much as a countryside row's, where players build most.
- 2026-10-24 (T-1091, orchestrator's call on Anita's approval): option (a), one rule: every commute curve but the US one (LODES, fitted on blocks) is flat inside 2 km; a pack without `decay_flat_km` stays as solved until the fix batch rejoins it.
- 2026-10-24 (T-1111, agent's call): one-off costs are pale red bars with a red cap (expenses, beside the red running-cost line), and the net chart's gains turned blue to match revenue; blue and red are their own tokens (`--revenue`, `--expense`), not the theme accent, so they hold in the pink and green themes.
- 2026-10-24 (T-1112, agent's call): zoom changes only the distance shown; the fare axis refits to the curve's fare at the right edge (times 1.25), so the line always fills the plot at any zoom, where one factor for both axes would push the $1.50 base off the top when zoomed in. The default view fits the second node (200 km for the default curve); the selected-node steppers became one row per node ("1.0 km costs $1.60"), so nothing refers to a selection; a click on an off-plot arrow zooms out to its node; the wheel over the graph zooms rather than scrolling the panel.
- 2026-10-24 (T-1130, agent's call): fare changes do not join the sim's undo history (that needs a WASM interface change, and fares live in the clock worker); `game/fareUndo.ts` keeps their steps on the main thread, valid while no other edit has been sent, and Ctrl+Z, Ctrl+Y and the blueprint's undo and redo buttons go through it, handing over to the sim's in order. Only an undo button by the fare graph (redo by key or the blueprint's button), enabled only when it would undo a fare.
- 2026-10-24 (T-1114, Anita): the start menu keeps "Welcome to anitabuilder!" with its exclamation mark, an exception to the player-text rules.
- 2026-10-24 (T-1110, agent's call): the top bar's World sums the cities with a built station and a demand view, exactly the Cities tab's World row (`game/worldTotals.ts`), not every zone with a view, so the two figures agree; the switch is map zoom below 5 (`WORLD_ZOOM`), whatever lies under the centre, and the hover gives city and station counts.
- 2026-10-24 (T-1108, agent's call): the name cap is 104 px (88 under 760 px of map column), two lines; shown names have a median of 7 letters, 95% up to 15 and 99% up to 21, so almost all fit on one or two lines; only "Sächsische Schweiz-Osterzgebirge" clips (hover has it). T-1113 may lower it.
- 2026-10-24 (T-1109, agent's call): the time and date go beside the sky from 940 px of map column (about a 1250 px window with the dock at its default), the time over the date; narrower, the old stacked clock.
- 2026-10-24 (T-1120, agent's call): an unopened zone's dot is opaque white (the bubbles stay at 80%), outlined in the theme's map ink: black in the light theme, off-white in the dark one, where the dot reads as a plain white disc on the dark basemap. The realm circles are the same, their hover a light grey. An old stored Unopened cities choice is ignored and dropped on the next save.
- 2026-10-24 (T-1127, agent's call): the hover reads "Long track pieces are laid out straight enough for this speed", Anita's meaning in the panels' voice; code names stay `designSpeed`.
- 2026-10-24 (T-1129, agent's call): the demand stopped alert (T-757) borrowed the stations full alert's CSS class; it now carries the red style itself as `.demand-stopped`.
- 2026-10-24 (T-1131, agent's call): in the saved games panel the file buttons come first, then Save as a copy with its name field, then the list; the red callouts about a read-only or failed store stay on top.
- 2026-10-24 (T-1092, agent's call): the local part of errands and social is drawn at 1.0 km (from 1.3 and 1.4), one number for both, as T-1090's scratch test; it keeps the share under 2 km where it was while the 20% central part at 6 km makes the long tail.
- 2026-10-24 (T-1092, agent's call): the central weight is the purpose's own attraction times its attraction per km² of land (power 1), only for errands and social, with no OSM high-order tags (packs do not carry them) and no cap on one zone's pull (the busiest zones came out Midtown, Oxford Street, Ginza and the Opera, 5-6% of a city's errands).
- 2026-10-24 (T-1134, agent's call): all of an inspector's buttons go in one row under its header, including Remove station and Remove track (asked first as before); a station has two rows, trip and link above construct and remove. The junction's "Paid at once" note moved with its button.
- 2026-10-24 (T-1121, agent's call): the purple lines in a station box were T-236's platform edges (moving in beside the tracks once double track parts, T-658c); they carried no state, so they are gone and the crowd grey fills to the rim, and close in from the track's outer edge. The station inspector's complex diagram keeps its own blue edges.
- 2026-10-24 (T-1122, agent's call): the "coo" is read as the getting-off pops (the sound tied to riders getting off; the chime is the line's own note, T-706, and stays). Skipped at a 30% chance; otherwise the group shifted up to 4 semitones either way from 640 Hz, each pop up to 1 more; level riders/80 times the pops' own, between 0.25 and 1.25.
- 2026-10-24 (T-1123, agent's call): a platform's drawn crowd drops when its train stops, to the riders it will leave behind, and grows again once it leaves. A train's stop is its departure less its stand there (dwell and holds from the profile; at a first stop the line's closing stand, the turnaround), never before the train ahead left.
- 2026-10-24 (T-1124, agent's call): riders changing trains walk as one bubble per arriving train and platform they go to, set off as the train stops (T-554's pulses now reach a walk after the train stops, not after it leaves; the account's totals are unchanged), straight between the two platform middles over demand's walk between the station nodes; radius the box's white half width times sqrt(riders/250), 0.35-1.3 of it. Measured +0.3 ms GPU, +0.05 ms CPU a frame at Penn Station (notes/T-1121.md).
- 2026-10-24 (T-1125, agent's call): "usually runs at capacity" is trains, not track: a stretch where at least half the trains leaving it in its busiest clock hour are packed to crush, either direction, from the boarding account. The traffic view encodes riders, so its red edge should say the trains are full; T-819's track-at-capacity edge stays, both now with the body 14% darker, and the hover names which.
- 2026-10-24 (T-1114 bug B1, agent's call): a region's running cost counts a high-speed line at the high-speed rates (US$5 a train-km, US$3 a car-km), as the network's ledger always did, so the regions add up to the network again. A 6 km high-speed line beside an ordinary one in New York: the region's day went from 24.31 to 29.88 US$M, the network's 29.88.
- 2026-10-24 (T-1114 bug B5, agent's call): Backspace while a segment is still being placed does nothing (it used to end the run); after Backspace the real-network snap starts again from the earlier segment's start. The refused-for-cash shake (T-625) plays again: a rule stopping every animation had stopped it.
- 2026-10-24 (T-1018 wave 3, agent's call): `enrich_parks` fails a build only when under 85% of the park draw lands on cells (was 95%): a place with draw has residents within 2 km, so the shortfall is thin zones, not lookup faults; Kauai lost 11% to coast parks 0.5-2 km from peopled cells.
- 2026-10-24 (T-1053, agent's call): a save naming zones the registry no longer has loads, each id taken as its successor from `app/src/game/zone-successors.json` (followed through later removals) or dropped where there is none; the table is written by `pipeline/zone_successors.py`, which diffs HEAD's registry against the working one and takes the zone now under the removed zone's old centre (else the nearest outline within 30 km). A table from a diff, with smoke failing while an id is missing, holds for every way a zone leaves (replan, park, hand edit) without touching the run code; a centre lookup in the app would need the old centres kept and the masks loaded. The header's `cities` check is gone: it listed every registered zone, so any removal shut out every save.
- 2026-10-24 (T-991, agent's call): an inter-city trip with an end whose zones are not loaded shows the straight-line distance between the two places' centres and the car's effective time from the pair's rule (parking at the far place), and nothing else (no trip counts, bar or Train by period); for a 6 km zone of such a place the distance is to the place's centre. The sim's `market_trip_w` fills these from the places' rows, no change to the wasm interface.
- 2026-10-24 (T-734, agent's call): a node drag with the track tool drops a route being drawn only when the route starts at the dragged node (its start is where the node was); a route from elsewhere carries on and its preview is asked again.
- 2026-10-24 (T-971, agent's call): the autosave-off reason stays in the top bar's pink notice, which stays up while it applies; the one-off notice at the map's bottom left is gone.
- 2026-10-24 (T-1014, agent's call): the Northeast's 13 packs keep their market place ids 0-13 with `market` -1, and `data/packs/intercity.json` stays as the T-276 solve left it for the two ignored sim tests that open it (nothing else reads it); `bin/joint_gravity.rs`'s mode without `--packs` (the Northeast's) is left in place, unused. notes/T-1004.md "T-1014".
- 2026-10-24 (T-1113, agent's call): Windham, New London and Litchfield go into Hartford (hfd, now a world plan row built with the usa module, 1.76M), not an eastern Connecticut zone: Hartford's pack is each county's first destination outside it in LODES OD 2023 (19%, 17% and 24% of their workers; 31-44% of those leaving the county), Connecticut has no other countryside zone to join, and Windham with New London (0.38M) would sit under 80% of the floor. Saves map wnd, nlc and lit to hfd. notes/T-1004.md "T-1113".
- 2026-10-24 (T-1140, Anita): a zone holding every county of a US state shows the state's name, its biggest city kept as an alias: Maine (cud), Montana (yel), North Dakota (css), South Dakota (mic), Vermont (cie), Wyoming (lra), in zone-names.json. notes/T-1140.md section 5.
- 2026-10-24 (T-1136, agent's call): the cash is right-aligned in a fixed box sized to "−$37.34B" (8ch), so it always ends the bar's usual gap before the clock, replacing T-310's $ at a fixed place with digits left-aligned (the reserve for "−$99,999,999,999" made the 70-100 px gap); the $ stays right before the digits. Beside the sky, the time and date line has a fixed 300 px box and the sky takes what the date leaves (a few px day to day) rather than a date slot with dead space.
- 2026-10-24 (T-1137, agent's call): the two hint icons are cells joined to the front of their button rows, not buttons (no click); the track row's hover is "Colour the track".
- 2026-10-24 (T-1138, agent's call): settings version 9 drops a stored ramp end only when it still equals the old default (#e61e1e, #1e3ce6), so player-chosen ends stay; the Train share bars and swatches take the new ends too, and the market trip's Train and Car bars now follow the scheme like its Plane bar.
- 2026-10-24 (T-1139, agent's call): the account icon was already at the top of in-game Settings whenever accounts are on (on localhost only after `?acct=live`); a worded Account row (who is signed in, Account settings or Sign in) is added near the foot of Settings so it can be found. Change password, recovery email, sign out everywhere and delete already existed in the panel and on shonei-server.
- 2026-10-24 (T-1142, agent's call): a hidden realm's circle sits only at its point (a zone centre on its land) and leaves the screen with it; with the point off screen there is no circle, and a click on the shaded land still shows the realm (T-1049).
- 2026-10-24 (T-262, agent's call): holidays.json takes the `holidays` package's observed days as days off in every country (China's and Korea's substitute days, Japan's citizens' holidays, the US Friday for a Saturday holiday, which federal offices and many employers take), plus the bank category only in Japan (Dec 31, Jan 2 and 3, named "New Year holidays") and Korea (Workers' Day, a paid day off for most employees); other countries' bank days are bank closures and stay out. Observed days keep the package's names ("Labor Day (observed)").
- 2026-10-24 (T-262, agent's call): China's make-up working weekends are not in the package for the game's years (its weekend workdays stop at 2011, and v0.83 lacks China's 2026 official schedule), so China's long weeks are the package's (Spring Festival 4 to 6 days off, National Day 3 to 5) and no Chinese Saturday works; the weekend workdays the package does give (Hungary's 2026 swaps) run as weekdays. A rule of our own for China's swaps would be calendar data the package lacks, so it waits for Anita.
- 2026-10-24 (T-262, agent's call): game day 1 stays Monday January 27 although the observed days now make it a holiday in China (Spring Festival's substitute day on 2031's calendar) and Taiwan; moving it would shift every save's dates.
- 2026-10-24 (T-262, agent's call): the track model's per-line day list now means "the other kind from the day number's" (a holiday on a weekday, a Friday in Riyadh, a Saturday in Hong Kong), a toggle in `Boarding::kind_at`, so the WASM interface is unchanged. The demand views' average day stays 5 weekdays and 2 weekend days over 7 in every country.
- 2026-10-24 (Anita): old saves in formats TWT2 and TWT3 and frequency-scheduled lines may stop loading (T-1146). One zone merge rule worldwide (T-1140, covering T-903, T-1089, T-1005 part 2 and T-1002), the plain 0.5M floor relaxed for very big regions; no zone rebuilds until she asks. Accounts need no game links and leaderboard rows open no games (T-989, T-990 dropped). New countries only on spare compute: Korea, Canada (cities and corridors), Mexico, Turkey, Vietnam.
- 2026-10-24 (Anita, T-263/T-264): attractions (not "sights") as their own purpose, go ahead; stadiums later and low priority, no separate view; going-home holidays are the main kind, the proposed numbers fine as a start, calibrated on real data where any exists; Saturday apart from Sunday, yes; destinations do not respond to the network (T-454 stays parked); cars a bit faster (T-1147).
- 2026-10-24 (T-1143, agent's call): the fare graph shows $0.15 of fare per km of distance shown over its height (the old default view's $30 at 200 km), so zoom never changes the line's slope on screen; the fare axis starts at $0 unless the base fare would stand above a quarter of the height, and then slides up with it so the base fare sits at a quarter (zoomed in, a $0-based axis would put the whole line off the top). Fare ticks are the multiples of the smallest 1-2-5 step of at least a quarter of the fare shown. The default view is the smallest 1-2-5 distance holding 1.2 times the last node with every node below 90% of the fare shown (200 km for the default curve).
- 2026-10-24 (T-1143, agent's call): a node's drag picks its one direction after 3 graph units of pointer travel (the larger of the two moves) and keeps it to release; the hover arrows are small grey triangles 9-13 units out from the node, four for nodes 2 and 3, two (up and down) for the base fare, and only the chosen pair during a drag. A node's fare stays between its neighbours' (no pushing): raising the base fare past node 2's stops there.
- 2026-10-24 (T-1143, agent's call): the rows read "Base fare $1.50", then "20 km costs $3.50" and "100 km costs $11.50" as before. Old saves' two-node curves and base-and-rate saves become nodes at 0, 20 and 100 km on their line (a line below $0 at 0 km starts at $0); any other curve becomes the line through its first and last node.
- 2026-10-24 (T-1143, agent's call): demand and income agree on a bent curve by demand counting its own extra: the assignment puts each station pair's trips times the curve's extra at the pair's expected km (what mode choice charged) on the route nodes where they first board, and income adds it to the first stretch's base and rate. This replaces T-887's mean-length income. On weekends the extra scales with each route node's boardings rather than purpose by purpose. notes/T-1143.md.
- 2026-10-24 (Anita): the trip explorer keeps drawing walks and drives straight; no road-traced walking routes (T-937's cost, a road graph per pack, not worth it). Demand and fares are to be capped by what a line's trains carry (T-1070, T-1069).
