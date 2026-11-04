# trackdepth

A vertical profile of every NYC subway (and Staten Island Railway) track, with error bars.
No public depth dataset exists; this pieces one together. Started 2026-11-01.

Served at http://localhost:8800/trackdepth/ (`python serve.py add trackdepth trackdepth ...`).

## Pipeline

    python fetch_osm.py     # raw/osm_tracks.json, raw/osm_stations.json (Overpass)
    python fetch_mta.py     # raw/mta_elevators.csv (94fv-bak7), raw/mta_stations.csv (39hk-dx4f)
    python fetch_dem.py     # raw/dem/*.tif, USGS 3DEP bare earth at 3 m, UTM 18N, NAVD88 (~1 GB)
    python station_levels.py  # evidence/station_levels.csv (+ _unparsed.md), a few seconds
    python build.py         # data/tracks.json for index.html (the page: see "The page" below)
    python make_platforms.py  # data/platforms.json (station boxes), after build.py
    python checks/scan.py   # after every build: one count per class of mistake, worst cases (--all for every one)

`raw/` is downloads only; everything in it can be fetched again.

## Model (build.py, v1)

Each OSM track way is cut at shared vertices and resampled every 25 m. The unknown is rail
height z (m, NAVD88) at each point, solved as one sparse least-squares problem:

- **prior**: ground (3DEP, running median along the track) plus an offset by structure.
  Tunnels use the OSM `level` tag, which NYC mappers have filled in on almost all underground
  track (-1 to -5, roughly how many levels below the street). Offsets per level are guesses
  (`TUNNEL_LEVEL`), as are elevated (+8 m), embankment, cutting and at-grade offsets.
  Under water there is no ground to hang off, so river tunnels get a weak absolute prior
  (-25 m) and bridges over water a very weak one; the approaches pin them.
- **correlated prior error**: the error d = z - prior is an Ornstein-Uhlenbeck process along
  the track (correlation 1500 m): a stretch built deeper than the rule stays deeper.
- **smoothness**: weak grade term (sd 5%), change of grade sd 1% per 25 m. The prior itself
  is diffused along the track over ~150 m (`PRIOR_SMOOTH`), since it jumps where way tags
  change (portals, level -1 meeting -2) and the solution used to copy those steps. Then any
  step steeper than 4% (`GRADE_LIMIT`) has its grade term tightened and the system is solved
  again; one round clears all 130 such steps (max was 15.5%). Held-out coverage unchanged
  (69% / 87%).
- **station ties**: at each MTA station, the tracks of that line on one OSM level are tied to
  one height (sd 0.3 m), since they run along one platform (Grand Central's 4/5 and 6 came out
  7 m apart without it). The same groups give the page's station dots (`stations` in
  tracks.json; one per station, line and platform level since the fifth round of map review fixes, below). Ties made the
  bars overconfident (held-out 60% / 77%); loosening `CURV_SD` to 0.02 got 60% / 80%.
  Widening priors for shared error between parallel tracks (`PARALLEL` 1.7) made it worse.
  The remaining shortfall is the rock-station tail.
- **hills** (`lower_envelope`, `ENVELOPE_SLOPE` 2%): a tunnel's prior hangs off the lowest
  surface reachable along the track at a 2% grade, not the street right above, since a tunnel
  can't climb a hill faster than trains can. Deep rock stretches (Washington Heights) start
  deep; cut-and-cover on flat streets is unchanged. Held-out sweep: off 58% / 77%, 3% 60 / 81,
  2% 63 / 81, 1.2% 60 / 79, 0.7% 55 / 77.
- **stacked tunnels**: tunnel points within 4 m in plan on different OSM levels (two levels
  under one street, and line crossings) must be at least 5 m apart rail to rail, upper on the
  higher OSM level. Active set like the cover cap; 1,300 of 1,396 such pairs were closer than
  that. Elevator anchors now pin only tracks carrying an OSM stop position, so express tracks
  running under a local platform (Central Park West) aren't pinned to it. Held-out after:
  58% / 77%.
- **tunnels stay underground**: one-sided cap, rail at least `MIN_COVER` (4 m) below the
  street, and under water at least `UNDERWATER_CAP` (-12 m NAVD88, sd 4: a guess until
  bathymetry is in). Solved as an active set in the same rounds as the grade limit (points
  above the cap are pulled to it; held points the system puts clearly below are let go).
  River-crossing priors are held fixed while the prior is smoothed, so the banks don't wash
  them out. Found by the B/D under the Harlem River drawn above the water. About 1,950
  tunnel points sit at the cap. Held-out after: 66% / 84%, median signed error -0.1 m (was
  -0.9 m: the model no longer runs shallow).
- **station anchors**: absolute platform heights from `evidence/station_levels.csv`, made by
  `station_levels.py` from every elevator and escalator rise: each machine's Notes ("A to B")
  is an edge in a graph of levels per complex, street ends pinned to the DEM at that machine,
  solved by robust least squares (disagreeing machines get their sd widened). Assumptions
  (bridge 4.0 m over a platform, a default 5.0 +- 2.7 m street-to-mezzanine where no street
  machine has a rise, station-specific readings in `OVERRIDES` / `EXTRA_EDGES`) are written in
  each row's `notes`. Rail = platform - 1.1 m, sd floor 0.75 m, one equation per OSM way (mean
  of its points within 100 m), tracks matched by line name. A line with platforms on two levels
  is pinned only where the nearby OSM tracks have exactly two `level` values. 89 anchors
  (85 station-lines) as of 2026-11-01, up from 25.
- **junctions**: ways that share a vertex share the variable.
- **error bars**: sd over 60 perturb-and-solve samples. They are only as honest as the
  guessed class sds. To do: calibrate against held-out documented depths (scale sds until
  ~68% of documented points fall inside 1 sd).

The OSM `level` tag is an order, not a depth: 191 St (1) is level -2 and about 53 m down.
So in v1 the deep rock tunnels (Washington Heights, 63 St, Second Av, Hudson Yards) are drawn
far too shallow until anchors or documents reach them.

### Document anchors, matching and calibration (2026-11-01)

- **Document anchors** (`build.py: doc_rows, place_doc_rows`): `depths.csv` rows with coordinates and
  measure `platform_below_street` (rail = street - value - 1.1 m), `rail_below_street` or
  `rail_above_street`. Range "90-100" -> midpoint, half-range added to the sd; ">x", "<x", "105/135"
  skipped. sd = 1.5 / 2.5 / 4 m for high / medium / low confidence, plus the half-range. Street =
  DEM ground at the matched points. Rows that land on the same tracks (the three 168 St figures,
  two 190 St rows) are one anchor: best sd, widened by half their spread. Where an elevator anchor
  already pins the same points the row is not an anchor (it stays a check in `validate.py`). 2004
  Second Avenue design rows are dropped where an as-built row exists (72, 86, 96 St). Result:
  89 elevator + 10 document anchors (11 rows).
- **Matching** (shared by `build.py` and `validate.py`): the row's line (names compared after
  stripping dashes, "Line", "Subway", "Shuttle"; `DOC_LINE_WORDS` for York St, 57 St, Grand St and
  the Manhattan Bridge level, where OSM names the line differently), tunnel/cutting for a rail
  below the street, no yard or crossover track, within 100 m. Where the line has several OSM
  `level` values there: a row naming the lower level, "express" or "Manhattan Bridge" gets the
  lowest, "upper", "local" or "mainline" the highest, and a row naming none is not placed. No
  fallback to any track nearby (the old fallback put 125 St Second Av on the Lexington line).
  `validate.py` prints the OSM level each row matched, whether it was an anchor, and lists the
  rows it could not place.
- **Not used yet** (`build.py` prints the counts): 23 rows with a depth measure but no coordinates
  (most river tunnels, 2004 Second Av design south of 96 St, Smith-9 Sts, Woodside); 9 bounds or
  two-figure values ("<=29" x7, "<1", "105/135"); 11 absolute-datum rows (mean high water, river
  surface, sea level) and 64 other measures (excavation, grades, shafts, relative); 6 rows with no
  OSM track of the line within 100 m (Mott Av, 125 St Second Av, Atlantic Av Brighton platform, Canal
  St Broadway, Lex/53 IND, Lex/59 BMT); 3 with several OSM levels and no level named (High St,
  Lex/51 St, South Ferry 2009); 3 superseded 2004 design rows.
- **Hold-out** (`calibrate.py`): all anchors (89 elevator, 10 document groups) split into 2 folds
  by md5 of their id; build on one fold, score the other: (model - anchor) / sqrt(model sd^2 +
  anchor sd^2), anchor sd as above, so a noisy Wikipedia figure does not count against the model.
  `--refit` fits the constants on the other fold only (nested). Elevator rows that cover a
  held-out elevator anchor are not promoted to anchors, so no truth leaks back.
- **Result**: before (guessed constants) held-out coverage 66% inside 1 sd, 81% inside 2 sd, median
  signed error -0.5 m (n=100). After: 68% / 87%, -0.5 m (nested refit, constants fitted on the
  other fold only: 64% / 86%). 2 sd stays under 95% because of a heavy deep tail the level tag
  cannot see: 191 St, 181 St, Hudson Yards, Lex/63, Roosevelt Island, Penn Station and Bowling
  Green are 10-25 m deeper than their neighbours, and the model's own sd there is small because the
  neighbouring anchors pin the line. Scaling every sd by 1.5-2 only reached 85% at 2 sd.
- **Constants changed** (comment in `build.py`): `TUNNEL_LEVEL` means := median anchor depth per
  level, sds := their sd: -1 6.5+-3 -> 7.0+-5; -2 10+-6 -> 10.5+-10; -3 15+-9 -> 15+-6; -4 21+-12 ->
  19+-9. Level -5 has one anchor (Lex/63, 37.9 m): too thin, left at 27+-15. These were fitted on
  the same 100 anchors that the hold-out scores, so 68% / 87% is slightly optimistic (the nested
  run is the fair figure). `CORR` stays 1500 m: 500, 1000 and 2500 were no better (62-68% / 81-84%).
  The elevated prior (sd 4 m) is too wide on station anchors (12 held out, rms z 0.4, 100% inside
  1 sd) but narrowing it to 2 m changed nothing measurable (anchors dominate), left alone.
- **Suspect anchors**: 34 St-Penn Station (Broadway-7 Av) comes out 21.7 m down from the elevator
  chain, where the platform is about 9 m; the chain probably runs through a deeper concourse.
  (Fixed 2026-11-02, with Bowling Green: see "Worst misses".)
  Lex/63 and 190 St: Wikipedia's 140/155 and 140 ft sit 4-12 m deeper than the elevator chains.

## Evidence added 2026-11-02 (all used by build.py)

- `evidence/riverbed.csv` (`fetch_bathy.py`, `make_riverbed.py`): NOAA CUDEM 1/9" riverbed under
  the 750 tunnel points over water. River tunnel prior = bed - 10 m (sd 5), cap = bed - 5 m;
  river approaches follow the bed's envelope at 3.5%.
- `evidence/placed_rows.csv` / `placed_bounds.csv` (`make_placed_rows.py`): 23 published figures
  turned into rail NAVD88 (tide datums stated per row), plus 3 "at least this deep" limits used as
  caps. TD0025+TD0155 is skipped (`SKIP_PLACED`).
- `evidence/structure_heights.csv` (`structure_heights.py`): 12,108 lidar rail samples on els,
  bridges and embankments (3DEP 2017 point cloud, DSM surface + 0.2 m). Samples off the running
  median of their neighbours by > 1.5 m are dropped (wrong deck); template-copied and
  interpolated ones get sd 2. Embankment prior is now +0.3 m (the bare earth includes it).
- `evidence/profiles.csv`: 129 points read off the 1904 IRT plates, Engineering News 1914 (Lex Av,
  two levels) and Brahdy 1934 (Jay-Smith St); 125 land on track.
- Smoothness re-tuned with all of it in: grade sd 0.2, curvature sd 0.04 per 25 m; tightened grade
  terms floored at 0.01 so caps and anchors can still win (without the floor, 30 rounds of
  tightening made the 60th St tunnel rise above the river). Replaced the same day by a one-sided
  limit (see "Worst misses", 4).
- Stacked pairs closer than 300 m along the track are ramps, not stacks, and are left out.
- Held-out: 66% inside 1 sd, 89% inside 2 sd, median signed error 0.0 m.
- Known leftovers: ~200 steps steeper than 4% where lidar and the 4% limit disagree, mostly the
  Manhattan Bridge's Manhattan end and the Williamsburg approach at Essex St, and a few
  SIR samples near Arlington. (Mostly fixed below, "Worst misses".)

## Other rail (2026-11-02)

PATH, LIRR, Metro-North, Amtrak and AirTrain went in on 2026-11-02 (`load_tracks`: main-line
tracks of those operators that touch NYC, no yards or freight; PATH whole). Their published figures
are `evidence/placed_rows_rail.csv` (RL0001-10, `make_placed_rows_rail.py`); `riverbed.csv` now has
the Hudson under PATH and the North River tunnels; the lidar file grew to 14,839 rows with their
els and bridges. `TD_SUBWAY_ONLY=1` builds the network without them: the hold-out was 62% / 86%
either way, so the other rail is not what moved it. The 2004 Second Av design profile points were
moved out of `profiles_docs.csv` to `evidence/unused_profiles_sas2004_design.csv`, because they
dragged the as-built 72, 86 and 96 St anchors shallow.

## Worst misses (2026-11-02, second round)

Hold-out before: 61% inside 1 sd, 86% inside 2 sd, median signed error -0.1 m (nested `--refit`
62% / 82%). With `--nsamples 40` (the default) the error bars are themselves noisy: the same model
scores within about 3 points either way between runs, so only moves bigger than that count.
Without the 129 profile points it was 64% / 91%: the profiles, not the other rail, cost the 5 points
(finding 1 below). Each step, plain hold-out (40 samples):

1. **Profile points on two levels** (`build.py`, profile matching). 24 profile rows say "OSM level
   -1;-2" (the Engineering News 1914 Lexington Av pairs, the 1904 Times Sq shuttle). The regex took
   the first number, so each express-level point also pinned the local tracks, 15-20 m too deep:
   59 St (Lexington, local) came out 15 m down, truth 7.1. Now picked by order as for document rows
   (lower / express -> lowest level, upper / local -> highest; no level named -> both). 59 St local
   15.0 -> 8.1, 59 St express 19.4 -> 22.4 (truth 22.4). 60% / 89%.
2. **Elevator chains** (`station_levels.py`). Three wrong readings, each now with its reason in
   `OVERRIDES` or the code:
   - Bowling Green: the 1977 lower mezzanine is under the tracks (EL710 "plaza ... to uptown
     platform and lower mezzanine", EL711 "lower mezzanine to Brooklyn-bound platform"), so ES340-343
     (23 ft) rise to the platforms. Platform 19.4 -> 5.1 m below the street; model was 5.7.
   - 34 St-Penn (1/2/3): EL618 drops 46'10" from the street to the Penn concourse, and EL216 then
     *rises* 19'7" to the platform. Read downward the platform was 21.7 m deep, below Penn Station's
     own tracks (about -5 m NAVD88 at Dyer Av, per the Gateway profile), which the IRT crosses over.
     Now 8.3 m, sd widened by 2 m because 46'10" to the concourse is itself odd.
   - Embankment stations: the DEM keeps an embankment, so a street machine whose coordinate is on
     it (Sheepshead Bay's EL793/794 carry the station's own point) was pinned 5 m high. The street
     of an Embankment station is now the 10th percentile of the DEM within 150 m (Sheepshead Bay
     6.9 -> 1.9 m, platform 4.8 -> 0.6 m above the track-level DEM; lidar agrees within 0.5 m). Only
     Sheepshead Bay changed. 62% / 90% (rms z 1.91 -> 1.30).
3. **Manhattan Bridge upper roadway** (`structure_heights.py`, `fix_template_upper_deck`). The
   template track is visible on the main span, but on the Manhattan approach the upper roadway
   covers it too: its lidar "rail" steps up 4.5 m within 60 m past the tower and then follows the
   roadway down at 3%, 6.5 m above the rail. Detected as a step against the rail's own trend of more
   than 3.5 m; those samples (and the other three tracks there) become `lidar_deck`: rail at most
   roadway - 6.5 m (the gap measured on this bridge where both decks are seen), used by `build.py`
   as caps, not readings, since the tracks dive for their portal while the roadway ramps to the
   plaza. 52 samples, all on the Manhattan approach; nothing else in the file changed.
4. **Grade limit one-sided** (`build.py`, `solve`). A step steeper than `GRADE_LIMIT` used to have
   its zero-mean grade term tightened (to 1% sd), which pulls every once-steep step toward *level*:
   the ramp from the Manhattan Bridge portal into Canal St was held flat along its whole length and
   sat 2-3 m under the street, above its own cover cap. Now such a step is pulled to the limit with
   the same sign (`GRADE_SD_ACTIVE` 0.5%), as the caps are. Converges in about 9 rounds instead of
   30. With 3 and this: 67% / 89% (rms z 1.17). TD0086 (Canal St, Manhattan Bridge level) model
   2.2 -> 6.9 m.
5. **Bridge approaches** (`BRIDGE_APPROACH_GRADE` 6% within `APPROACH_M` 800 m of a span over water).
   The East River bridge approaches are documented as steeper than 4% (Manhattan Bridge both ends,
   Williamsburg's Manhattan end), and the Williamsburg lidar reads 4-5% straight off the deck. Not
   fitted: 6% is a loose ceiling. With 3-5: 67% / 91% (rms z 1.18).
6. **Prior-error chain cut at portals** (`OU_ACROSS`): the OU terms no longer link points of
   different structure classes, since a viaduct 15 m above the elevated rule of thumb says nothing
   about the tunnel past its portal. No measurable change (62-65% / 88-90%); kept because it is
   right, `TD_OU_ACROSS=1` restores the old chain.

Tried and left out: stacked-tunnel constraints at line crossings (`TD_CROSS_STACK=1`, see
`stacked_pairs`): Lex/63 improved, but crossings pass one line's error to the other and the active
constraint makes both overconfident (Canal St Lexington, Delancey St F, Atlantic Av), 2 sd 89% ->
86%. Point-to-segment stacking (any point within 4 m of another level's segment) pushed Queens Plaza
10 m too deep where OSM draws side-by-side tracks of two levels overlapping. No cover cap for 150 m
past a portal (no change). A 5% limit everywhere instead of 4% (no change once the limit was
one-sided).

Final (`calibrate.py --nsamples 100`): 67% inside 1 sd, 92% inside 2 sd, median signed error
-0.0 m, rms z 1.17 (was 1.94). Nested `--refit`: 66% / 88%. The TUNNEL_LEVEL constants were not
re-fitted this round; nothing new was fitted on the scored anchors (the 6% approach limit, the
150 m embankment radius and the 3.5 m deck step were chosen from the geometry, not swept on the
hold-out). With Bowling Green and Penn fixed, level -1's anchors now have sd 2.1 m against the
prior's 5.0; the nested refit, which uses that, scores no better, so it stays.

What is left, and why:
- Canal St (TD0085/86) and Grand St (TD0092) are still 4-10 m too shallow held out. The bridge
  tracks now fall at the limit from the portal, but the portal itself sits about 20 m up: nothing
  but the elevated prior (ground + 8 m) says where the approach under the roadway comes down, and
  the published depths need it at street level (~12 m) by the portal. In the full build TD0085/86
  are anchors and pin it. The published figures look right: 20 / 40 / 55 ft for the Lexington,
  Nassau and bridge levels is the order OSM has them stacked in (bridge tracks under the Nassau under
  the Lexington), and the 1914 "lower level 40 ft below mean high water" (TD0025+TD0155) agrees with
  the 55 ft bridge platform within 3 m once the approach is not read off the roadway. SKIP_PLACED
  still leaves TD0025+TD0155 out; with the approach fixed, its reason (a 6% grade from the bridge)
  no longer holds, and it could go back in as the bridge level.
- Pelham Bay Park (z -3): the lidar's flat 22.1 m in the station is the platform canopies, 2.5 m
  above the approach el's rail (19.4 m just before the station); the elevator chain (rail 18.4) is
  right. Station-zone lidar has no way to see under a canopy at a terminal; not fixed.
- Lex/63 (-4, -5), 86 St (Second Av), 23 St express (TD0064), 191 St: the deep-rock tail. No rock
  map is in the repo. 191 St's elevator chain and the 1904 IRT profile disagree by 7 m (181 St by
  4 m), both ways round from the elevators, so one of the two sources is off; not settled.

(TD0025+TD0155 went back in afterwards: SKIP_PLACED is empty, reason in `build.py`. Hold-out
67% / 90% at 100 samples.)

## Map review fixes (2026-11-02, third round)

From Anita's look at the map. Hold-out before (TD0025+TD0155 in, 100 samples): 67% / 90%. After
1 and 2: 68% inside 1 sd, 92% inside 2 sd, median signed error -0.0 m, rms z 1.09; nested
`--refit` 68% / 87%. Nothing fitted.

1. **Bump on one J track on the Williamsburg Bridge's Manhattan approach** (Clinton to Columbia St,
   40.7155-40.7162). Track 2 runs under the bridge's higher level past the end of OSM way 180330673
   (the main span), down the approach way 1547317494, where its own lidar read 4-6 m above track 1
   for 250 m; the template correction only covered the listed main-span way. `structure_heights.py`
   (`way_continuations`, `bridge_adjust`): template and target ways now extend through their
   end-to-end continuations of the same `railway:track_ref` (2 ways on). 8 samples changed, all on
   1547317494 (-4.3 m on average); the two J tracks now agree within 0.2 m there. The Manhattan
   Bridge's 160 template copies are unchanged.
2. **Central Park West splitting and rejoining.** The real arrangement is a directional two-level
   line: northbound local and express (tracks 2 and 4) on the upper level, southbound (1 and 3) on
   the lower, from 72 St to 110 St, the side platforms stacked (OSM levels -1 and -2; Wikipedia's
   CPW station pages agree, B1 northbound). OSM draws the lower tracks on top of the upper ones and
   they share nearly every vertex, and a shared vertex was one variable: the two levels were forced
   together at every vertex (stations included, where vertices are dense) and pushed 5 m apart by
   the stacked rule between. `build_points`: a vertex lying inside tunnel ways of two or more OSM
   levels now gets one point per level; ways ending there join their own level's point, and ways
   ending there whose level runs through no other way share one (a level change along a track, as
   track 1 at 110 St). A vertex where ways only meet end to end stays one point. 95 vertices split,
   all on genuinely stacked lines: CPW (33), the 8th Av under St Nicholas Av (18), 145 St (12), W 4 St
   (15), East Side Access at Grand Central (8), the Concourse yard connector (2). CPW now keeps the
   lower level 4.5-5.5 m below the upper the whole way. W 4 St (8th Av) held out: 12.8 -> 10.5 m
   (truth 8.1).
3. **Els that follow the terrain.** Checked, not changed. Every elevated point has a lidar sample
   within 20 m (55% plain readings, 36% station zone, 3% interpolated, 1% template, 1% "unclean"), so
   no el falls back to the ground + 8 m prior. Interpolated and template samples are copied from
   neighbouring samples or the other track, never from the ground. In 225 m windows where the ground
   is bumpy (detrended sd > 0.6 m, 160 windows) the detrended rail wiggles 0.16 m (median) and the
   two are uncorrelated (median r 0.06). Over longer distances the rail does follow the street,
   because NYC els stand at a near-constant clearance over the street they run above: on the J along
   Broadway the rail is 6.6-8.6 m over the DEM, rising and falling with Broadway's grade, read
   straight off the lidar. The few windows with high correlation are real structures (the Astoria el
   over 31 St crossing the Grand Central Pkwy, AirTrain over the Van Wyck). A stiffer curvature on els
   would smooth lidar noise of ~0.15 m only.
4. **Portals** (`find_portals`, `portals` in data/tracks.json): one entry per tunnel end, parallel
   tracks within 30 m and facing the same way merged; `lon`, `lat`, `z` (rail, m NAVD88), `lines`,
   `bearing` (degrees clockwise from north, pointing into the tunnel), `tracks` (how many merged).
   Service tracks are left out, a tunnel run under 60 m gets none (a street over an open cut), and
   an open run under 50 m between tunnel ends is a tag gap, not daylight. 113 portals, 65 of them on
   subway, SIR or PATH lines.

## Station-box fixes (2026-11-02, fourth round)

From the station-boxes agent (data/platforms.json reads the model's rail heights).

1. **86 St (Lexington)** is two-level, not one: local on the upper level, express on the lower
   (Wikipedia's station page and its 48 ft express platform, TD0075; the 1918 opening of the lower
   level with express service). OSM's -1 / -2 there is right, so no level overrides. The model had
   them only 3 m apart because OSM draws the four tracks within about 7 m in plan and the stacked
   rule only held point pairs closer than 4 m. `stacked_pairs` now also holds two levels of the *same
   line* running side by side within `PAR_STACK_R` (8 m) and parallel within ~25 degrees, along the
   segment; different lines are still left to the 4 m point pairs (that is what went wrong at Queens
   Plaza). TD0075 held out: 11.9 -> 14.0 m (truth 15.7).
2. **Queensboro Plaza** (upper deck Manhattan-bound N/W and 7, lower deck Queens-bound): the lower 7
   track's lidar read 20.4 m, the upper deck's height, since it can't be seen from above, so the two
   decks came out 0-4 m apart. `build.py` lidar loop: a direct lidar reading on an el with an el of a
   higher OSM level within `HIDDEN_DECK_R` (4 m) in plan becomes a cap, rail at least
   `HIDDEN_DECK_GAP` (4.5 m) below the surface read; interpolated and template samples are left as
   they are. This also covers W 8 St (Culver under Brighton), Parkchester's two levels, the Corona
   yard flyovers, and the crossings at Broadway Junction. Queensboro Plaza now has the lower deck
   about 4.5 m under the upper.
3. **Missing station dots**: the New Lots Line (Sutter Av-Rutland Rd to New Lots Av) is MTA
   "Eastern Pky" but OSM's "IRT New Lots Line" (added to `LINE_WORDS`); St George's platform tracks
   are `service=yard` in OSM (dots and ties now fall back to the line's yard tracks when a station
   has no other); PATH stations are `station=subway` in OSM and were skipped as NYC subway (now only
   non-PATH `station=subway` is skipped); and `station_groups` deduplicated on Station ID, which W 4
   St, 145 St and Queensboro Plaza share between two lines, so the second line lost its dot and its
   ties (now one row per station and line).
4. **PATH dots tidied**: where only PATH track is in reach, only a PATH station gets a dot (OSM's
   HBLR, NJ Transit and Amtrak stations at Exchange Place, Harsimus Cove, Hoboken Terminal and Newark
   Penn sat beside PATH), stations NYC Transit runs come only from the MTA table (the "34th
   Street-Herald Square" outline beside PATH's 33rd Street), and one name for one railway within
   300 m is one dot. 13 PATH dots, one per station.

## Regional rail to the edge of the square (2026-11-02)

The map's square is now **[-74.26, 40.508, -73.645, 40.975]** (W, S, E, N), about 52 x 52 km,
centred near Court Sq (`fetch_osm.SQUARE`; the same box in fetch_routes.py and make_platforms.py).

- **Tracks**: `fetch_osm.py` downloads the square (now also `railway=monorail`, for AirTrain Newark)
  and cuts every way at its edge (68 ways cut; a way that leaves and comes back keeps its longest
  part). `build.load_tracks` keeps LIRR, Metro-North, Amtrak and NJ Transit main/branch track out to
  the square (no longer NYC + 800 m), both AirTrains, and also any track a passenger train route runs
  on whoever operates it (the Raritan Valley Line runs from Newark on the Lehigh Line, which OSM gives
  to Conrail; 14 km) and operator-less main-line links that join kept track at both ends (the Grand
  Central throat, Harold). Hudson-Bergen and Newark Light Rail and freight-only track stay out.
  9,292 ways, 2,624 km: subway 1,355, LIRR 312, Metro-North 271, NJ Transit 263, Amtrak 238, PATH
  77, SIR 57, AirTrains 36, Conrail-operated passenger track 14.
- **A fresh OSM download moves way ids.** PATH's operator is now "Port Authority of New York and New
  Jersey" (normalised back to "Port Authority Trans-Hudson" in `load_tracks`, so everything keyed on
  it still finds PATH), and 79 ways the evidence named were split or redrawn. The download the
  evidence was written against is kept as `raw/osm_tracks_2026-11-01.json`: a placed row whose ways
  are gone uses the ways now lying along their old geometry (`placed_points`), and a lidar sample whose
  way is gone goes to the nearest open track within 12 m (820 of 826 placed). All 33 placed rows are
  used again.
- **Heights**: the 3 m 3DEP DEM covers NJ, Long Island and Westchester (13 new tiles, no fallback to
  the 15 m fill; ground missing at 0 points). The lidar point cloud is the NYC 2017 one, so new NJ /
  LI / Westchester els and bridges have priors only (Dock Bridge and the other PATH structures in
  Newark were already measured). Two fixes for what that showed: `tunnel=building_passage` is a
  tunnel only with a negative level or layer (Hoboken Terminal's train-shed tracks had been pushed
  14 m down as tunnel; now at grade), and `BRIDGE_OVER_WATER` is (12, 60), next to nothing, so a
  bridge without lidar takes its approaches' height (Portal Bridge had a 7 m hump at 17 m; now 10 m,
  a swing bridge with 23 ft clearance; Lower Hack, HX, Upper Hack, Lyndhurst and the Newark drawbridge
  7-12 m). Checked: Secaucus Junction's NEC upper level 10.3 m over the Main Line's 2.9 m; the M&E Bergen
  Tunnels 8-15 m NAVD88 under 32-36 m of Jersey City Heights; the North River Tunnels from the North
  Bergen portal under 61 m of Palisades to -25 m under the Hudson; Portal North Bridge 18-20 m.
- **Services** (`fetch_routes.py`): one service per NJ Transit line (NEC, North Jersey Coast,
  Raritan Valley, Morris & Essex, Montclair-Boonton, Main, Bergen County, Pascack Valley, Meadowlands),
  colours from OSM's route `colour` (NJ Transit's own) where it is a real colour, `NJT_COLOURS`
  otherwise (OSM's Meadowlands is white); AirTrain Newark as its own service under "AirTrain". A
  relation that lists only some stops (NJ Transit's NEC starts at Secaucus) is routed from its member
  ways' ends inside the square as well. Relations OSM draws both ways in one ("New York <=> Gladstone")
  stay one variant, out and back. 60 services, 189 variants.
- **Station dots**: 652 (497 subway and SIR, 50 LIRR, 43 NJ Transit, 28 Metro-North, 14 AirTrain,
  13 PATH, 7 Amtrak); one railway's station mapped twice within 200 m is one dot ("Pennsylvania
  Station" beside "New York Penn Station", "Broad Street Station" beside "Newark Broad Street").
  Platform boxes: 1,281, 334 of them for PATH and regional rail.

## Grand Central Madison and canopies (2026-11-02)

- **Grand Central Madison**: OSM ends the upper (201-204) and lower (301-304) platform tracks at
  shared bumper nodes at 44 St, so the two levels met at one height there. `build_points` now also
  splits a vertex where only ends meet when every way leaves it on the same side (nothing continues
  through it) and the ways carry two or more levels. The levels now sit about 15 m apart at the
  bumpers and part from the two 63 St tunnel tracks where OSM's ramps do. Penn Station's west throat
  is all one OSM level (-3) with ordinary 4-way switch nodes: the tracks really do funnel into the
  ladder there, so no change.
- **Canopies over el stations**: a station-zone lidar reading (plain, station or interpolated) more
  than `CANOPY_DZ` (1.5 m) above the median reading of its station level's tracks is a platform roof:
  it becomes a cap, rail at least `CANOPY_GAP` (2.5 m) below it. Coney Island-Stillwell Av's two
  westmost tracks (2-8 m high under the shed roof) now sit with the others; 37 samples in all. A
  station whose every track is under its canopy (Pelham Bay Park) has no uncovered track to compare
  with, so it is not caught; its elevator anchor pins it in the full build.

## Portals (2026-11-02)

- **Dip before the 122 St portal (the 1)**: not the cover cap (the last tunnel points sat 8-9 m
  under the street) but the level prior: rail 10.5 m under the street right up to the portal, where
  the street climbs over the mouth, so the line dived 4 m and climbed back onto the Manhattan Valley
  viaduct. Within `PORTAL_TAPER` (400 m) of open track the tunnel prior now blends linearly to the
  open track's prior at the portal, and the cover cap tapers to nothing the same way. Chosen by
  profile (no taper / 150 / 250 / 400 m): the 1 now falls steadily from 116 St (28 m) to the portal
  (26 m) and runs level onto the viaduct; 250 m left the Dyckman St sag.
- **Dyckman St (the 1)**: the hump and dip there came from the 1904 IRT plate, which north of 168 St
  reads 2-6 m above everything else that measures the track (181 and 191 St elevator chains +3.9 and
  +7.0 m, the lidar on Dyckman St station and the 231 St el +4.8 and +6.3 m). Its 6 points north of
  40.845 moved to `evidence/unused_profiles_1904_north.csv`, the reason in each row. The tunnel now
  runs from 191 St (12.5 m) to the portal (12.2 m) with a 1.5-2 m sag between.
- **The East River tunnels' Queens portal (LIRR and Amtrak, Hunterspoint Av)**: a 10 m V. The open
  cut outside the portal floors at -1.5 m on the DEM, and "over water" meant any ground below 1 m,
  so those tracks got the weak over-water prior and were left out of the portal taper; the tunnel
  ran down to -11 m and the cut track followed it. Ground below 1 m now counts as water only within
  `WATER_NEAR_M` (60 m) of the water polygon (`data/water.json`, `near_water`), or where the DEM
  has none (`TD_OLD_WET=1` brings back the old rule). The tracks now meet the cut floor at about
  -2.3 m. Sweep of all 124 portals for a V (the lowest rail within 250 m more than 2 m below both the
  rail 250-320 m into the tunnel and 250-320 m out along the open track): 7 before (Hunterspoint
  three times, the Empire Connection twice at 11th Av, the NEC into Sunnyside, Dyre Av), 2 after,
  both checked by profile and smooth (the Sunnyside NEC descends steadily; Dyre Av sags 1 m in a 1.2
  km tunnel). Hold-out after: 68% / 93% (100 samples), median signed error +0.3 m, rms z 1.07.
- Checked, no dip: the 7 at Hunters Point, the B/Q at Prospect Park, the F at Smith-9 Sts (0.5 m),
  the F/G at Church Av (1.3 m). The remaining "dips" in a scan of all 66 portals are real descents
  (the Manhattan Bridge's Brooklyn approach into DeKalb, the L at Broadway Junction, the Sunnyside
  approaches to the East River tunnels).

## Map review fixes (2026-11-02, fifth round)

From Anita's look at the live page. Sweep scripts were scratch (not kept); counts below.

1. **Bergen Tunnels: one track over the arch** (NJ Transit Morris & Essex into Hoboken, west portal at
   Tonnele Av, 40.7402,-74.0624). Track 3's portal node lands on the portal face in the 3 m DEM (ground
   33.6 m; tracks 1, 2, 4 read the cut floor at 12-13 m). At a node shared with the tunnel the open
   track's prior (sd 2 m) wins over the tunnel's, so that node got prior 34 m and track 3 ran 6 m above
   the others through the portal. `solve`: within `PORTAL_GROUND_M` (75 m) of a tunnel, open track
   (grade, cutting, embankment) takes the lower envelope of the open approach's ground at the grade
   limit (146 points lowered). Track 3 at the portal 18.7 -> 13.3 m (others 12.3-14.6).
   Then `parallel_ties`: tunnel or cutting tracks of one line (same OSM name) and level within
   `PAR_TIE_R` (6 m) and parallel are tied to one height (sd 0.5 m per point, 22,170 ties), except within
   `PAR_TIE_CLEAR` (200 m) of a level change, where one track really dives away. Found by the Empire
   Connection's two tracks drifting 3.5 m apart in one tunnel under 11th Av. Sweep (pairs of parallel
   tracks of one class and level within 8 m, split by more than 3 m): 658 points in 56 places before,
   533 in 51 after; at a portal or in a cut 19 places before, 15 after. What is left there is junction
   ramps (Nostrand Av junction, the 60 St tunnel connection, the Canarsie line at East New York,
   Franklin Av) and open track tagged at grade whose ground really differs (flyovers at Harold, Jamaica,
   Dyre Av). Many mid-tunnel splits of exactly 5.0 m are level-change junctions held by the stacked
   rule (Grand Central, Lexington north of 42 St), left alone. Held out after 1: 68% / 93%, rms z 1.10.
2. **Crossing tunnels too close** (14 St / 6 Av: the L 1.8 m under the 6 Av line and PATH; 32-33 St /
   6 Av: the East River tunnels 1-3 m *above* the 6 Av line, which OSM has over them). `crossing_pairs`:
   two tunnel segments of different ways crossing in plan at more than ~25 degrees, not joined along the
   track within 300 m, keep at least `crossing_gap(lower, upper)` rail to rail: lower roof (box 4.8 m,
   subway tube 5.1, railroad 5.5: the Penn tubes' outside crown, Hewett and Brown 1910) plus upper invert
   (1.1 subway, 1.3 railroad), so box over box 5.9 m, box over a subway tube 6.2, over a railroad 6.6. Tube
   = subway rail more than 18 m down. Order from OSM levels where they differ (794 of 802; pull sd
   0.8 m), else from the first solution (sd 1.5); one-sided, in the active-set rounds with the stacked
   rows; the needed gap sampled +-1.2 m in the error bars. Which side moves comes out of the solve's own
   uncertainties. Crossings (one per pair of lines per 40 m) under their needed gap: 117 of 196 before,
   20 after; under 6 m 118 -> 85; under 3 m 32 -> 5. 14 St / 6 Av now 5.9 m (L -5.6, 6 Av line +0.3);
   at 6 Av the East River tunnels are 6.4-6.6 m under the 6 Av line and PATH. The 20 left are mostly
   flying junctions within 300 m of where the lines join (PATH at Exchange Place, the New Lots line at
   Utica Av, 4th Av / West End at 36 St), which the rule leaves out as ramps.
   - **East River tunnels at 6th Av** (`make_placed_rows_rail.py`, RL0011 / RL0012): Noble 1910 gives
     0.8% (33rd St) and 0.9% (32nd St) descending east from the station, 1.5% from Fifth Av. From RL0001's
     -2.2 at Seventh Av: -4.6 at Sixth Av, -7.0 at Fifth Av (sd 1.8). The model had +0.7 there: the 1.5 km
     prior-error chain carried Penn Station's shallowness east along tunnels that dive. The first try at
     the rule, with 1 m of ground over every tube and a 6.3 m railroad roof, took held-out coverage from
     68% / 93% to 61% / 90%, mostly the Sixth Av subway at 34 St pushed 6 m up by those tunnels.
3. **One dot per station level** (`main`, stations in tracks.json; `make_platforms.py`; page). 59 St on
   Lexington had its local (level -1, rail 5.8 m) and express (level -4, rail -8.6 m) boxes both pointing
   at one dot at the express level, so the upper box said 22 m down. Station dots are now one per
   (station, line, platform level), each with its own z and sd and `level` and `tier` ("upper level",
   "lower level", "middle level" where a station line has more than one). A level is a platform level when
   an OSM stop position within 150 m of the station lies on its tracks (a stop anywhere on the way made
   phantom stations on express levels); levels within 2.5 m in height are one. `make_platforms.py` gives a
   box the dot of its own OSM level (a dot of another level only when its own level has none). On the page
   a box hover shows the box's own rail height and the dot's tier; the profile's fallback station labels
   skip a name already placed at that point. (Lines that differ already had their own dots: Lex/59's
   N/R/W are OSM's Astoria line.)
4. **Grand Central Madison tail tracks** (`separate_dead_ends` in `load_tracks`; `build_points`). OSM draws
   the tail tracks south of the platforms converging in pairs to one bumper node on each level (201 with
   202, 203 with 204, the same on the 300 level), and the upper and lower platform tracks (204 / 304,
   203 / 303) meet end to end at shared nodes at the platforms' south end. The first made V shapes; the
   second pinched the two levels 15 m apart to one height and sloped platform 204 from -19.3 to -10.9 m.
   Now: where several ways end at one vertex, all leave it on the same side and none runs through, each
   way of a level shared with another is cut back to where it is `TAIL_SEP` (3.6 m) from the others (at
   most `TAIL_MAX_CUT`, 80 m; about 50 m at GCM); and a vertex where every level runs through end to end
   on its own gets one point per level (2 vertices, both at GCM). Sweep: 9 such dead-end bundles in the
   square (GCM 2, Grand Central Terminal 4, Jamaica 2, one subway yard at 36-38 St), 16 way ends cut;
   Penn has none.

Hold-out after all four (100 samples): 64% inside 1 sd, 92% inside 2 sd, median signed error +0.2 m,
rms z 1.12 (before this round 68% / 93%, rms z 1.07-1.10; with 1, 3 and 4 but not the crossing rule, 68% /
92%). The crossing rule costs about 4 points at 1 sd, a little over the run-to-run noise of about 3. A
crossing passes one line's error to the other and makes both surer: held out, 34 St-Herald Sq (Sixth Av)
comes out 11.8 m deep (truth 14.8, was 15.4), pushed up by the East River tunnels under it, and Canal St's
Nassau level (TD0085), already 7 m too shallow, goes 2.4 m shallower with sd 0.4. Sampling the needed gap
at +-1.2 m instead of +-0.7 changed nothing measurable. Gains: both Lexington Av/63 St levels (-2.7 ->
-2.1, -2.6 -> -1.8 in z), 57 St-7 Av, 5 Av/53 St, the Times Sq shuttle. `TD_NO_CROSS=1` builds without
the rule. Nothing was fitted on the scored anchors.

## Checks and map review fixes (2026-11-03, sixth round)

`checks/scan.py` runs after every build (see Pipeline): one count per class of mistake Anita found on the
map, with the worst cases (`--all` for every one, `--data <dir>` for a saved build). Before = the fifth
round's data, after = this round's, both counted by the same scan:

| check | what it counts | before | after |
|---|---|---|---|
| 1 anchors | anchors (elevator, document, placed) the model misses by more than max(2 m, 2 sd) | 0 | 0 |
| 2 cover | 'Subway' stations under 7.0 m (mezzanine) / 4.5 m, tunnels under 3.5 m away from portals | 32 | 14 |
| 4 above-ground | open-track crossings without clearance, bridge-tagged track within 2 m of the surface | 745 | 75 |
| 5 open-deep | open track over 3 m under the ground at the track, stations against the MTA Structure column | 166 | 96 |
| 6 platform-tie | OSM platforms whose two sides' tracks are over 1 m apart | 22 | 9 |
| 7 portals | tunnel-to-open changes with no portal | 2 | 0 |
| 8 structure | stretches where OSM tags, lidar and rail vs ground disagree (table by cause) | 826 | 159 |
| 9 dots | OSM stations on passenger track with no dot within 300 m | 3 (+GCM) | 1 |
| 10 curvature | track points on a vertical curve under 600 m radius | 71 | 73 |
| 11 boxes | station boxes on a level with no OSM platform | 7 | 4 |
| 12 labels | 'upper/lower level' where a level has no OSM platform | 2 | 1 |

Two more checks, from her next look (same day):

| check | what it counts | before | after |
|---|---|---|---|
| 13 station-track | dots and boxes whose rail or ground differs by over 1 m from the track under them | 284 | 35 |
| 14 isolated | pieces of track under 2 km joined to nothing, carrying no service | 13 | 8 |

- **13**: New Dorp (SIR, an open cut) hovered "+1 m" over track drawn 5.6 m under the surface. A dot's ground
  was the DEM at the track while open track's `ground` was the surface beside it (item 4 below). Now one
  ground per point (`shown` in `main`: the DEM over tunnels, the surface beside open track) for the tracks
  and the dots, a dot's ground is its track's at the point nearest the dot (the mean over a platform's length
  differed by metres on slopes), and each box has its own `ground` (make_platforms `ring_ground`: its
  tracks' ground along it, as rail_z is), which the box hover uses.
- **14**: the piece by a creek bridge near Elizabeth, NJ was Conrail's "Staten Island Railroad" freight line,
  kept by the SIR rule's name match; that rule now skips freight operators (`FREIGHT_OPS`). `drop_isolated`
  (in `load_tracks`) drops pieces under `ISOLATED_M` (2 km) with no station or stop position within 150 m:
  5 ways (the LIRR's C Secondary stub, three yard leads by 207 St, one more). The 8 left are Grand Central
  Terminal's platform tracks (cut off from their throat by the service filter) and an Atlantic Branch track
  at Jamaica: at stations, kept.

What changed, by her item:

1. **Grand Central Madison**: Wikipedia puts the mezzanine "on a center level between the LIRR's two track
   levels", "more than 140 ft (43 m) deep below Park Avenue". RL0008 (upper) is now the mezzanine less one
   level (rail 37.5 m down, sd 3; it was "a two-level station 100 feet below street level", 30.5 m, which
   can't fit), RL0007 (lower) the mean of the mezzanine plus one level and amNY's 150 ft (47.6 m, sd 3.5).
   GCM has its own dots now, one per level (no OSM station node exists; see 9).
2. **Station floors** (`STATION_FLOOR`): rail at least 7.0 m under the street at an MTA 'Subway' station
   with a mezzanine in station_levels.csv, else 4.6 m, as caps on the station's tunnel points. Delancey
   St-Essex St (J/M/Z) 1.6 -> 5.6 m (the Williamsburg Bridge ramp holds it up).
3. **PATH WTC** (RL0013): "a basement four stories underground ... along the base of the World Trade Center
   bathtub" (Wikipedia); rail 22.4 m under the street, sd 2.5. The loop was about 32 m down; now 23 m.
4. **The Hell Gate viaduct at Ditmars** was never low: rail 37.3 m over the Astoria el's 29.0. The bare-earth
   DEM keeps solid masonry viaducts and embankments, so the shown ground was the viaduct's top.
   `surface_ground`: tracks.json `ground` for open track is now the median DEM 25 and 40 m to each side
   (the street beside a viaduct, cut or embankment); tunnels keep the DEM at the track (the street over
   them). The el prior uses it too. Bridge-tagged stretches within 2 m of the shown ground: 718 -> 41.
   `crossing_pairs` now covers open track: the lower track's clearance (subway 4.4 m, third-rail
   railroad 4.9, catenary 6.7, unelectrified 6.1) plus the upper deck (1.5 m) or invert.
5. **Open cut, at grade, embankment** (`STRUCTURE_PRIOR`): MTA 'Open Cut' stations' rail 5.5 m under the
   surface beside the cut (sd 2), 'At Grade' 0.5 over the ground at the track (sd 1), 'Embankment' 0.3 over
   it (sd 1.5), where no elevator anchor covers the station (291 station tracks). Botanic Garden 8 -> 5.5 m
   down: an open cut, not at grade.
6. **Cross-platform ties**: tracks within 6.5 m of both sides of one OSM platform (one level, one structure
   class) tied to one height, sd 0.3 (1,140 ties). Prospect Park's shuttle and B/Q now share a height.
7. **Portals**: a short open run between tunnel pieces (a tag gap) counts as tunnel before tunnel runs are
   measured (`find_portals`). The F's portal before the Culver Viaduct is back.
8. **Structure audit** (check 8, table by cause in the scan). The systematic cause was the shown ground (4).
   Also: grade or embankment track over water or marsh gets `CAUSEWAY` (rail 2.5 m NAVD88, sd 1.5) instead
   of sea level, and on land the DEM at its own point (the A across Jamaica Bay sat at 0 m).
9. **Station dots**: LIRR terminal tracks tagged service=yard are kept when a passenger route lists them or
   they carry a train stop (Long Beach, Port Washington, Far Rockaway now have dots); a railroad station's
   dot uses its own railway's tracks (GCT no longer takes the LIRR's under it); every stop of a railroad or
   PATH service with no dot of its railway within 300 m gets one per level of that railway's tracks there
   (GCM). Track with no DEM ground takes the nearest point's.
10. **Vertical curves** (`VCURVE_R_MIN` 600 m): a step sharper than a 600 m radius is pulled to the limit
   with its own sign, like the grade limit. 103 St (CPW) track 1: radius 222 -> 429 m. The count is the same
   (71): what is left sits at junctions and level splits where anchors, caps or stacked pairs force the shape
   (145 St's Concourse junction is also over the 4% grade limit), and the active-set rounds now stop at the
   30-round cap with about 10 changes a round.
11. **Boxes**: a platform takes at most one track per side (`make_platforms.match_platform`), so Central
   Park West's lower (downtown) boxes are the local track and its platform only. Those boxes are real:
   "the local tracks are stacked, uptown above downtown ... so both platforms are on the west side, one
   above the other" (Wikipedia, 81 St-Museum of Natural History); the express tracks have no platforms.
   Station dots exist only on levels with an OSM platform (`build_stations`), so no fallback box is made on
   a level without one (Bergen St's closed lower level).
12. **Labels**: dots carry `routes` (services.json refs that run on the dot's tracks and stop within 250 m,
   late-night variants left out, subway ones cut to the MTA's Daytime Routes); the hover shows the bullets
   and the tier, and the line name only where no service is known.

Hold-out after (100 samples): 62% inside 1 sd, 92% inside 2 sd, median signed error +0.2 m, rms z 1.24 (fifth round 64% / 92%, rms z 1.12). Without the last change (cross-platform ties across a cutting/tunnel mix) it was 62% / 93%, rms z 1.12: the rms rise is one unit. Prospect Park (Brighton) held out comes out 3.1 m deep (truth 9.7, sd 1.0, z -5.4): the tie hands it the Franklin shuttle's height, and the shuttle's own cutting prior (DEM at the track - 5 m) is about 6 m too high there. In the full build the elevator anchor holds both at 12 m (8.9 m down). The tie is right (one OSM platform between them); the shuttle prior in that cut is the weak part. Nothing was fitted on the scored anchors.

## 3D terrain (2026-11-02)

    python fetch_bathy.py --fill   # CUDEM for all the water in the extent -> raw/bathy_fill/ (~220 MB)
    python fetch_coast.py          # OSM coastline -> raw/osm_coast.json
    python fetch_landuse.py        # OSM piers, parks, lakes -> raw/osm_{piers,parks,lakes}.json (Overpass is slow)
    python make_water.py           # data/water.json: the water polygon the page draws flat at 0
    python make_terrain.py         # data/terrain/{z}/{x}/{y}.png heights + tex/{z}/{x}/{y}.webp colours

- **Water** is a vector polygon (OSM coastline, minus OSM piers, minus anything the 3 m lidar ground
  shows above 1.2 m inside it: the coastline runs along the esplanade behind Hudson River Park's
  piers). Inside it the terrain is the CUDEM bed pushed to at most -1 m (feathered from the shore's
  land height); land outside it is never below -0.9 m. Water cells with no CUDEM value take the
  nearest known bed fading toward 0 (e-fold 25 cells). The old sawtooth was where `raw/bathy` ended.
- **Tile seams** came from deck.gl's mesher: it pads a 256 px tile to 257 x 257 vertices by copying
  the last column and row, so neighbours never agreed on their shared edge (a crack; black through
  the transparent terrain). Tiles are now sampled with stride 256/255, so pixel 0 and pixel 255 sit
  on the tile's edges and neighbours agree exactly; the Cesium page samples them that way too.
- **Colours** are baked into the tile textures (the mesh has no vertex normals, so lighting would
  facet steep ground): near-black land, soft hillshade, parks dark green, lakes light blue. The page
  drapes them over the basemap at 0.45.
- `TERRAIN_CACHE=<folder> python make_terrain.py` keeps the two slow mosaics (about 20 min) between
  runs; delete the folder when a source changes.

## The page (index.html, CesiumJS, 2026-11-02)

`index.html` is CesiumJS 1.145.0 from jsdelivr, no ion token or ion assets. `index_deck.html` is the
earlier deck.gl + MapLibre page, kept for reference and not linked; `cesium.html` redirects here.

- **Heights.** Everything is drawn at ellipsoid height `GEOID + z x scale` (`heightOf`), with
  `GEOID = -31.74` m (GEOID18 at Times Sq from the NGS API; -31.0 to -32.5 m across the extent), so
  terrain, water, tracks and portals agree. `scene.verticalExaggeration` with
  `verticalExaggerationRelativeHeight = GEOID` stretches the ground about NAVD88 0; everything else
  is rebuilt at the new scale a moment after the slider stops. The flat view (`#flat`) is a top-down
  camera at scale 1.
- **Terrain** is the page's own provider over `data/terrain` (web mercator z9-13, decoded in two
  workers, 129 x 129 meshes, nothing past z13: Cesium upsamples). The globe is translucent (0.45,
  back faces hidden); the camera is kept 40 m above the ground. Cesium draws no skirts on a
  translucent globe, so tile cracks showed black: a near-black opaque backdrop lies 1.5 km below
  everything and fills them. Imagery: Stadia Alidade Smooth Dark (keyless from localhost only; a
  published page needs a key or another basemap) under `data/terrain/tex` at 0.45.
- **Water** is `data/water.json` flat at NAVD88 0, translucent, drawn over the ground.
- **Contours** are a globe material: faint lines every 5 m and 25 m of real height (the material
  undoes the geoid and the scale), ground a few percent darker the lower it lies, none below
  -0.95 m (the riverbed).
- **Tilt-shift** is a post-process stage, always on: blur by |ln(distance / focus)|, focus the ground
  under the screen centre, distances to a level plane through it (the depth buffer is useless with
  the translucent ground), strength rising with pitch and zoom.
- **Controls** are the page's own (MapLibre-like): left-drag grabs the ground, right or ctrl-drag
  rotates and tilts, wheel zooms to the cursor, touch pans, pinches, twists and tilts. Rotating is
  smoothed (2026-11-02): the pointer adds to `rot` (degrees of bearing and pitch still to turn) and a
  `scene.preUpdate` listener (`rotateStep`) takes a share of it each frame, exponential with a 50 ms
  time constant, so the turn follows the clock and not the event rate. Before, every pointermove
  turned the view whole pixels (0.8 deg) at once: frames with no event stood still, frames with two
  jumped, and the drawn step was 57-153% of the steady one (sd 0.32 of it); now 87-115% (sd 0.08).
  The pivot was never the problem (it is `view.lon/lat`, fixed for the drag), nor the trains.
- **Drawn:** tracks as one polyline primitive (per-vertex colours; ramp or, in Line mode, the line
  colour: most full-time services win, ties by bullet order, unlined grey); station dots with a
  stick to the ground and a flat disc there in the dot's colour; portal arches (`portals`, fading in
  from z11.5, not in the flat view); the open line's highlight and the profile cursor.
- **Loading bar:** "Loading tracks…", "Loading stops…", "Loading terrain…" (terrain tiles done),
  "Loading base map…" (Cesium's tile queue), then it fades. The basemap and `tex` are held at alpha 0
  after a load or jump until the view's tiles are in, so their blurry coarse tiles never show.
- **Hash:** `#elev` or `#line`, `#flat`, `#notrains`, `#s=REF.variant` (an open line), and
  `#v=lon,lat,zoom,pitch,bearing` (camera on load, not written back).
- **Plug-ins.** `window.trackdepth` has `viewer`/`widget`, `scene`, `Cesium`, `GEOID`, `data`,
  `services`, `scale`, `heightOf(z)`, `groundAt(lon, lat)`, `onScale(fn)`, `showStations`,
  `requestRender()`; `'trackdepth:ready'` fires once the data is in, `'trackdepth:stations'` when
  "Show stations" changes. The page renders on request, so an animation calls `requestRender()`.
  `trains.js` is a script tag after the main script and attaches on ready ("Show trains" calls
  `trackdepth.trains.setVisible`); `stations3d.js` (`initStationBoxes(viewer, …)`) goes the same way.

## Trains (trains.js, 2026-11-02 follow-ups)

- **Tone.** Box faces, lift toward white, old -> first brightening -> now (halfway back): body lift
  0 -> 12% -> 5% above ground, 12% underground (`mix(0.05, 0.12, v_under)`); roof 28% -> 30% -> 29%;
  sides 0.92/0.8 -> 1.0/0.9 -> 0.96/0.85; ends 0.72 -> 0.86 -> 0.79; underframe 0.45 -> 0.75 -> 0.6;
  `UNDER_BOOST` 0.45 -> 0.40. Mean luma of the train pixels, before/after at one clock: elevated 7
  and J -8 to -10%, underground (Union Sq) -6%, far ribbons (z15) -3%.
- **Shuttles.** `services.json` gives a service `"shuttle": n` (written by `fetch_routes.py`,
  `SHUTTLE_TRAINS`: GS 2, FS 1, H 2). trains.js then runs n trains in all, back and forth and
  reversing at each end, not the line's headway fleet: one polyline per track (variants over the
  same ways are one track), turned into one out-and-back polyline of length 2L with its own
  timetable (`shuttleTimetable`; the head jumps one train length at each end's dwell, which is the
  train reversing in place), trains dealt out over the tracks in turn, a track with none not run.
  42 St: 2 tracks, 1 train each; Franklin Av: 1 train; Rockaway Park: 2 tracks, 1 train each.
- **Stops near each other or the piece's end.** `timetable(..., near = 100)`: stops under 100 m
  apart are one, and a first or last stop within 100 m of the piece's end is where the train starts
  or ends (was 30 m). The Hudson Line has two Grand Central stops 49 m apart and its trains hopped
  from the piece start to the stop. Shuttles keep 30 (their turn-around assumes the end).
- **Terminals (fetch_routes.py, 2026-11-03).** Terminal platform tracks are often service=yard.
  `snap_all` skipped yard ways, so a stop on one snapped to the running track back from it (Long
  Beach 740 m, Far Rockaway 330 m, Port Washington to Plandome, 2 km); it now counts yard ways the
  relation lists, and `yards=True` for a terminal station point. The two ends of a run keep only
  candidates within 150 m of the best one (a far candidate made a shorter run and the search took
  it). A route 120-700 m short of a LIRR or Metro-North station is taken on to it (`term_pts`:
  Hempstead at Atlantic Terminal). A station of the service's railway at either end of the route
  that no stop is near is added as a stop (Port Washington, Long Beach, NJT NEC's New York Penn),
  and a nameless stop takes the name of the station beside it (Long Beach). Matching is by the
  station's own network: a first version took any railway's station and sent PATH Newark and NJT
  RVL on reversals. Only LIRR and Metro-North are taken on.
- **Outline antialiasing.** MSAA 2 left a car's outline three coverage levels, which crawled as it
  moved. See the VS (outline edges grown by 1.5 px) and FS (alpha = share of the pixel inside the
  edge) comments; the opaque pass draws depth first (pixels at least half covered, colour mask off)
  and colour second (blended, no depth write), else the grown rim would write depth and hide the
  track behind it as a dark outline. `trains._edgeAA = false` turns it off for tests.

## Status (2026-11-01)

v1 runs end to end: 4790 OSM ways, 56k points, 25 station anchors. Agents working on:
`evidence/depths.csv` + `evidence/documents.md` (published figures), and
`station_levels.py` -> `evidence/station_levels.csv` (platform heights from the full elevator
and escalator chains in the MTA Notes text, absolute via ground at each machine).

## Sources

Have:
- OSM track geometry with `level`, `layer`, `tunnel`, `bridge`, `embankment`, `cutting`.
- USGS 3DEP bare-earth elevation (NYC 2017 lidar underneath).
- MTA elevator/escalator inventory: `Elevator Rise` (141 machines) and `Escalator Rise`
  (228), mixed units (feet-inches, mm). Notes say what each connects ("mezzanine to
  Manhattan-bound platform"), so chains street -> mezzanine -> platform can be built.
  Includes the deep ones: 191 St 172'9", 181 St (Ft Washington Av) 106'10", 103'5", 162'4".
- MTA stations table: `Structure` (Subway / Elevated / Open Cut / Viaduct / At Grade / Embankment).

Being gathered (evidence/, by a research agent): every published depth figure with a quote
(`evidence/depths.csv`), and documents with profile drawings (`evidence/documents.md`).

Known leads, not yet used:
- 1904 IRT book (Gutenberg 17569): profile plates of the first subway, Brooklyn extension,
  Harlem River tunnel; text depths.
- HathiTrust: Board of Transportation IND contract drawings 1924-34; PSC First District
  annual reports. HathiTrust refuses automated fetches (403): Anita opens these.
- Second Ave Subway FEIS 2004 / SEIS 2018, 7 extension FEIS, East Side Access, 63 St.
- NOAA CUDEM topobathy for riverbed depth over the river tunnels.

## Barometer rides (idea, not started)

A phone barometer resolves ~0.5 m of height (about 6 Pa). Two problems:
1. Train-made pressure: a train in a tight tunnel pushes and pulls air, a few hundred Pa,
   i.e. tens of metres of fake height. Real and large, as Anita has felt. It is worst while
   moving and entering tunnels, and near zero while stopped at a platform with doors open.
2. Weather drift, up to ~100 Pa an hour in a storm: removed with 1-minute pressure from the
   airports (Iowa Environmental Mesonet ASOS 1-min, KLGA / KJFK / KEWR).

So station stops give platform heights cleanly (median over the dwell). Between stations it
is unproven; first test on a stretch with a known shape (the 7 under the East River, or the
191 St elevator, whose rise is 172'9" in the MTA data, as a calibration). Logger: phyphox
(free, iOS and Android), "Pressure" experiment, export CSV.
