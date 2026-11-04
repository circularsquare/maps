# trackdepth handoff (live state)

Orchestrator: session e701057c-e5d3-4237-b364-57b42d8b29cd, name maps-20, since 2026-11-02 09:08
Updated: 2026-11-03 (PAUSED by Anita: "will pick this up later once i get readings")

## Paused state (read this first)
- No agents running. Nothing committed in maps (all trackdepth changes are uncommitted
  working-tree edits; Anita commits maps herself).
- Live site anita.garden/trackdepth = round 2 (she pushed 2026-11-02). Rounds 3-4, the
  New Dorp/Elizabeth fixes and the flat lakes are on 8800 only.
- At her next publish: `rclone sync` data/terrain to R2; `rclone copy` tracks, platforms,
  services, water .json (commands under "Owed" below); copy index.html, trains.js, stations3d.js
  to C:\Users\anita\projects\website\trackdepth\; she commits + pushes the website.
- Open question to her: widen the error bars ~20% (held-out 62% in 1 sd vs 68% nominal).
- Waiting on: her barometer CSVs in evidence/baro/. Build the processing (ASOS 1-min weather
  correction) only once she says.
- Later, on hold: stairwells/elevators; drop basemap street names at the next basemap rebuild.

NOTES.md is the project record (model, evidence, terrain, the Cesium page, every round's fixes).
This file is only what's in flight and what's owed.

## The page now
- http://localhost:8800/trackdepth/ = index.html, a CesiumJS page (3D only). index_deck.html is the
  old deck.gl page, unlinked. Plug-ins: trains.js, stations3d.js (attach via window.trackdepth).
- Square EXTENT [-74.26, 40.508, -73.645, 40.975]; the page clips at LIMIT (inset 0.002 deg),
  exposed as trackdepth.LIMIT. Default view centred on LIC.
- Data base: HEAD data/services.json; if absent -> https://data.anita.garden/trackdepth/data/.

## In flight (maps-20's agents; resume with SendMessage by id)
- **Terrain agent** (Sonnet, a0d589746a1ae953a, scratch .../scratchpad/terrain_agent/):
  block-scale smoothing (SoHo bumps mid-block), Central Park The Lake + all park water blue
  (relation merge in fetch_landuse.py), Hudson Yards not blue (bed_water only where connected to
  river water). Rebuilds data/terrain (TERRAIN_CACHE) and data/water.json.
- Model agent (a71d35f18e0257968) LANDED 21:10: GCM levels split at the shared bumper nodes
  (~15 m apart now); Penn west throat is real geometry (no change); canopy rule CANOPY_DZ 1.5 /
  CANOPY_GAP 2.5 (Stillwell fixed; Pelham Bay Park not caught, all tracks covered). Held-out
  68/92. tracks.json + platforms.json re-uploaded to R2 by maps-20 21:15. (It reported still
  having background work; check its notification if it arrives.)

- Model agent LANDED 21:50: PORTAL_TAPER 400 m (tunnel prior blends to the open prior, cover cap
  tapers to 0) fixes the 1's 122 St dip; 1904 IRT plate north of 40.845 (6 rows) moved to
  evidence/unused_profiles_1904_north.csv (2-6 m high vs elevators and lidar). Held-out 68/92.
  191 St held out now 63 vs 54 m (pinned in the full build). tracks.json + platforms.json
  re-uploaded 21:55.
- One-way tracks LANDED (trains agent a4b92f5ae541737f9): both-ways track-km 23.2% -> 3.8%
  (right-hand scoring in fetch_routes.py, "<=>" relations split into paired variants, rush-only
  services run am variants only). services.json (207 variants) re-uploaded; trains.js copied to
  the website folder. Left: ~36 km real single track; LIRR Far Rockaway/Long Beach and NEC
  NJT/Amtrak stretches where the right-hand track isn't connected nearby in OSM.
- 21:25 maps-20: portal openings hidden (holes primitive show:false); index.html re-copied to
  the website folder. Anita: icon is fine; push only after terrain + model land and she decides.

- Terrain LANDED 22:30: smooth_land() (SIGMA_FLAT 5 cells ~55 m on flat ground, back to 1 cell
  between 3% and 12% slope); multipolygon relations were missing from parks/lakes (out tags geom
  drops members): fetch_landuse.py `relations` mode -> raw/osm_{lakes,parks}_rel.json, read via
  osm_areas.load_all; bed_water() keeps only parts joined to water (Hudson Yards cut out).
  Terrain 20 MB. R2 synced (rclone sync for terrain/, water.json copied): rclone check 638
  files, 0 differences. Website page files match the maps copies (hashes). Lakes look blocky at
  close zoom (baked tex resolution ~15 m at z13): possible follow-up, a vector lake layer like
  the water polygon.
- EVERYTHING LANDED; deploy is ready for Anita to commit + push the website repo.
- 2026-11-02 20:25 (clock per rclone; earlier times in this file run ahead): second round ALL LANDED too (below); R2 + website folder current again.
- 22:45 new round (Anita: smoothing OK). In flight:
  - terrain agent a0d589746a1ae953a: piers get a flat deck height (sank under the smoothing);
    water read as dipping (likely the see-through bed): before/after opacity pair
    review_water_*.png in trackdepth/.
  - trains agent a4b92f5ae541737f9: far trains read black -> line colours
    (review_trains_far_*.png), then two sticks per box for elevated stations once boxes are in
    (review_two_sticks.png).
  On landing: R2 re-upload (terrain/ + water.json), copy index.html/trains.js to website.
  - trains agent also (before the sticks): brighten all trains, drop the box outline and the
    ribbon halo; trains stop ~half a train late (7 at 5 Av overhangs the west end) -> whole
    train inside the platform box (review_train_stop.png).
  - TRAINS AGENT LANDED 23:20: halo + outline gone, bodies +12% toward white, far ribbons thin
    before fading; trains.js loads platforms.json and centres stops in the box (1993 stops,
    168 on the OSM point); two sticks per elevated box from z15.4 (318 boxes). trains.js +
    index.html NOT yet copied to the website (waiting on the terrain agent's addWater edit).
  - TERRAIN LANDED: FEATHER 0.5 (bed -1 m from the first water cell), water alpha 150 colour
    (40,84,140), decks flat (raw/decks.json, flatten_decks, DECK_MIN/MAX 1.5-4 m). Terrain +
    water.json synced to R2 (634 files, 0 differences). index.html not yet to the website.
  - MODEL LANDED: near_water (WATER_NEAR_M 60 m; TD_OLD_WET=1 for the old rule) so low open
    cuts aren't "over water"; portal V dips 7 -> 2 (both checked, not real); held-out 68/93.
    tracks.json + platforms.json uploaded to R2.
  - TRAINS LANDED (00:10): trains drawn twice (solid, then in the translucent pass, +45% lift
    underground); router drives only (bends <= 50 deg, carries on through stops); sharp-bend
    variants 35 -> 15, reversals 14 -> 4. services.json uploaded to R2. Trains agent now on the
    LIRR Far Rockaway/Long Beach/Ronkonkoma fallbacks + dropping Far Rockaway duplicates.
  - TRAINS LIRR round LANDED: whole-route candidate choice, duplicates (90% overlap) dropped;
    133 variants, 0 reversals bar PATH Hoboken. services.json uploaded. Trains agent now merging
    Ronkonkoma/Hempstead pairs and checking Port Jefferson (only 2.3 km inside the square).
  - Merges landed (186 variants, both-ways 2.6%), Port Jefferson 32 km; services.json uploaded.
    Edge extension LANDED: regional variants borrow a same-railway donor's onward route to the
    edge (19 short before, 0 now bar real terminals); 184 variants; uploaded to R2. Trains agent
    idle. trains.js still to copy to the website after the basemap agent lands.
  - BASEMAP LANDED: data/basemap z12-16 512px webp, 17,148 tiles 212 MB, OpenFreeMap dark
    recoloured, street names only; Stadia gone; credit "Basemap from OpenFreeMap, OpenMapTiles
    and OpenStreetMap."; rebuild in make_basemap/ (bake.py docstring). On R2 (check: 17,786
    match; only anchors.csv/doc_rows.csv absent, the page doesn't load them);
    index.html/trains.js/stations3d.js copied to the website folder.
  - (was) basemap agent ac65516a213e70219 (Sonnet, scratch .../scratchpad/basemap_agent/): own raster
    tiles from OpenFreeMap vector + MapLibre, no place/neighbourhood labels (Anita: they don't
    rotate), street names kept faint; data/basemap/ (gitignored) replaces Stadia; credits change.
    On landing: upload data/basemap to R2; check land colour vs make_terrain BASEMAP (16,16,20).
  - trains agent again (23:40): trains drawn bright through the translucent ground like the
    tracks; Penn Station zigzag train (path hopping parallel tracks), count before/after.
  - model agent a71d35f18e0257968 again (23:40): LIRR V dip after Hunterspoint Av (portal toward
    Sunnyside, Amtrak step nearby); sweep portals for V dips > 2 m. On landing: tracks.json +
    platforms.json to R2.
  Stairwells/elevators: ON HOLD (Anita 22:50). Later, maybe from other people's station maps if
  their licences allow. Feasible: MTA entrances dataset (type + position), vertical elevator
  shafts; stair paths aren't in open data.

## Round 3 (2026-11-02 evening, after Anita PUSHED the website: anita.garden/trackdepth is live)
Anita: "we're basically done, wont republish until i get some station readings." So: NO R2
uploads and no website copies this round unless she asks; batch them for her next publish.
- Basemap: drop street names too, but ONLY whenever make_basemap/ is rebuilt for another reason
  (her call: "dont need it immediately").
- trains agent ae9842952814e0336 (Sonnet, scratch .../scratchpad/trains2_agent/): darken trains
  back about halfway (above ground most); shuttles as a general case (GS 2 trains one per
  track, Franklin 1, Rockaway Park 2, back-and-forth); cheap train antialiasing (< ~1 ms);
  then jittery right-drag rotation (measure first; per-frame input, fixed pivot, ~100 ms ease).
  LANDED: tone halfway back (5% lift above ground, 12% under); shuttles via "shuttle": n
  (SHUTTLE_TRAINS in fetch_routes.py; services.json patched in place, not re-run); analytic
  edge AA on boxes (_edgeAA, ~+0.3-0.8 ms); rotation eased per frame (ROT_TAU 50 ms; step sd
  0.32 -> 0.08). Left: Cesium render CPU p95 ~20 ms, 17% of frames over 60 Hz.
- model agent a845cd7105fce45db (Opus, scratch .../scratchpad/model3_agent/): NJT track
  diverging near Journal Square/Bergen toward Hoboken; crossing clearance rule (L vs 6 Av at
  14 St only 2 m apart, F/M vs Amtrak at 32 St); split station complexes into one dot per
  station/level so box hovers show their own height (Lex/59); then GCM tail tracks (pairs
  meeting at one dead-end point, unexplained slopes) as a general dead-end-pair split.
  If its tracks.json changes, fetch_routes.py needs a re-run (trains agent or me).
  MODEL LANDED: portal approach grade + parallel_ties (Bergen tunnels track 3 fixed); crossing
  clearance rule (117 -> 20 crossings under gap; TD_NO_CROSS=1 disables; costs held-out 68->64%
  at 1 sd, 92% at 2 sd; kept, my call); Noble 1910 rows RL0011/12; dots per level (674, 22
  split, labels "· upper level"/"lower level"/"middle level"); dead-end bundles cut back (9,
  GCM tails flat ~1%, levels 15 m apart). Bergen St (F/G) fallback box on closed lower level is
  questionable. fetch_routes.py re-run: services.json byte-identical (routing reads raw OSM);
  trains.js timetable merges stops < 100 m apart and starts/ends at a stop within 100 m of a
  piece end (Hudson Line hop at GCT gone).
  ROUND 3 ALL LANDED. Nothing on R2/website (Anita publishes after barometer readings): when she
  does, upload tracks/platforms/services.json + copy index.html, trains.js, stations3d.js.
  Next-round candidates: Grand Central Madison has no station dot of its own (one "Grand Central
  Terminal" point for MNR + LIRR); Bergen St (F/G) box on the closed lower level; basemap street
  names dropped at the next rebuild.
- Old agents (terrain a0d5..., model a71d..., trains a4b9..., basemap ac65...) are done; their
  contexts are near 900k, so start fresh ones.

## Round 4 (2026-11-03): model agent a845cd7105fce45db (Opus, resumed)
Builds checks/scan.py (one count per class, run after every build) and fixes 12 items from
Anita: GCM upper level ~40 m deep (not 30); cut-and-cover minimum depth (Delancey-Essex 1 m);
PATH WTC depth; above-ground crossing clearance (Hell Gate viaduct shown at 0 over the Astoria
el at Ditmars); Botanic Garden 8 m deep (open cut); cross-platform ties (Prospect Park shuttle
vs B/Q); portal missing after Carroll St (F); structure-type audit (A at Subway Island in
Jamaica Bay shown underground); missing station dots (LIRR Long Beach, Port Washington, Far
Rockaway); vertical curvature (103 St northbound local bump); no boxes on platformless levels
(CPW express, Bergen St lower); hover labels leaking OSM line names ("72 St · 8th Av - Fulton
St"); GCM own station dot. Nothing to R2/website.
LANDED: checks/scan.py (12 checks; e.g. structure disagreements 826 -> 159, above-ground
crossings/low bridges 745 -> 75, shallow cover 32 -> 14); GCM upper 37.6 / lower 47.7 m; PATH
WTC ~23 m (RL0013); open-track ground = surface beside the track (viaducts in the DEM); platform
ties (Prospect Park shuttle = B/Q at ~12 m); vertical curve limit 600 m; dots per real platform
level with `routes` bullets. Held-out 62/92, rms z 1.24 (Prospect Park held out is the loss).
My calls: keep the Prospect Park tie; keep the 600 m curve limit. CPW lower boxes are REAL
(stacked local platforms, uptown above downtown) and were kept.
Trains agent LANDED: fetch_routes.py takes LIRR to Port Washington, Long Beach, Far Rockaway,
Atlantic Terminal (yard-tagged terminal ways count when the relation lists them; end snaps
within 150 m of the best; LIRR/MNR carried on 120-700 m to their own station); 550 trains.
Rebuild done: dots carry the new LIRR terminal routes; heights, scan and held-out unchanged.
ROUND 4 ALL LANDED. Nothing on R2/website. For her next publish: upload
tracks/platforms/services.json (+ anchors not needed), copy index.html, trains.js, stations3d.js.
Then (model agent): New Dorp track yellow but station green/"1 m above" (two ground
definitions?) -> dot vs track check; isolated short track near Elizabeth NJ -> orphan-piece check.
LANDED: one ground per point for tracks/dots/boxes (check 13: 284 -> 35); SIR rule skips
freight operators + drop_isolated (< 2 km, no stop within 150 m; check 14: 13 -> 8). Held-out
62/92, rms z 1.20.
Terrain agent a0d589746a1ae953a (resumed): flatten inland lakes to their shoreline level
(Central Park Reservoir shows a jagged sawtooth band). LANDED: lake_labels/flatten_lakes
(LAKE_PCT 20th percentile of the shore ring; 1,901 lakes), lakes out of the smoothing.
data/terrain (heights + tex) changed -> rclone sync terrain/ to R2 at her next publish. Barometer processing: NOT before her rides (Anita: "ill lyk").
Suggested next to Anita (error bars still awaiting her answer): error-bar recalibration (62% in 1 sd vs 68% nominal), barometer
processing ready before her rides, a website thumbnail, a phone check after all the changes.

## Owed when those land (done 22:30)
1. Re-upload to R2 (rclone works; no leading slash in filters under Git Bash):
   `rclone copy C:/Users/anita/projects/maps/trackdepth/data r2:anitamaps/trackdepth/data --include "{tracks,services,platforms,water}.json" --max-depth 1 --s3-no-check-bucket`
   `rclone copy C:/Users/anita/projects/maps/trackdepth/data r2:anitamaps/trackdepth/data --include "terrain/**" --s3-no-check-bucket --transfers 8`
   Old terrain tiles that a rebuild no longer writes stay on R2 (harmless). Spot-check with curl.
2. If index.html / trains.js / stations3d.js changed again, copy them to
   C:\Users\anita\projects\website\trackdepth\ (never a data/ folder there).
3. Report to Anita with every changed string old -> new.

## Deploy state
- R2 anitamaps/trackdepth/data/ has the 20:15 data (tracks/services/platforms/water + terrain).
- Website repo (C:\Users\anita\projects\website): trackdepth/ folder (index.html, trains.js,
  stations3d.js, icon.png, preview.png) + "nyc track depth" lines in pages/projects/maps.md and
  projects.md (no data-thumb yet). PUSHED by Anita 2026-11-02 (round 2 included). Next push is
  hers, after barometer readings.
- Stadia basemap: her account, domain auth for anita.garden; free 200k tiles/month. If traffic
  outgrows it: bake own OpenFreeMap dark tiles to R2 (~200 MB, z12-16).
- Meta: title "nyc track depth", description "NYC Subway track depth map" (her words).

## Open with Anita
- Push now or after the round (recommended after).
- Icon (icon.png): fine, or a cleaner shape?
- Tests she was asked for: cape strength (capeMat 0.5), trains speed/stops, box tints, phone
  layout (fold settings, sharpness at 1.5x), loading bar.

## Barometer rides
- Plan given 2026-11-02 (phyphox Pressure, record continuously, log time+place; ~0.12 hPa/m).
  Calibration: Myrtle-Wyckoff 26 ft and Broadway Junction 43 ft escalators, Morgan Av x3.
  Priority: Lex/63 (both), GCM (both levels), Atlantic Terminal, L and G end-to-end rides,
  182-183 Sts / Concourse, Fulton St J/Z. Dropped: 191 St, 181 St (she accepts elevator figures),
  190 St, 181 St A, 34 St-Penn, Bowling Green, Inwood-207. Her CSVs + log go in evidence/baro/;
  nothing built to process them yet (needs ASOS 1-min KLGA/KJFK/KEWR weather correction).

## Known leftovers (model)
- Held-out 66% / 91% (refit 66/86). Deep-rock tail (Lex/63, 86 St SAS) shallow; Canal St bridge
  level; Pelham Bay Park canopy; Penn Station one 538 m box; NJT out-and-back relations as one
  variant; no lidar outside NYC 2017 cloud (priors in NJ/LI/Westchester).

## Anita's standing preferences
- No AskUserQuestion; Latin characters; she reads every UI string (old -> new in reports).
- Never commit or push the maps or website repo unless she says so in that message.
