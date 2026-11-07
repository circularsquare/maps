# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-06 23:30 (T-011 served; T-092 and T-039 agents running)

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- All landed work is committed; smoke 18/18 and corridors 7/7 on the served map after T-011's
  swap. 8800 serves this folder live (`localhost:8800/isochrone/`), no build step.
- **T-011 is served** (2026-11-06 ~23:15): walking directed from 90 m GLO-90, L3/L10 by crossing
  L1. Rollback: data/tiles_pre_t011/ (`build_tiles.py swap` the other way). Older backups:
  data/tiles_map2020, data/hazard_map2020. Numbers and shots: notes/T-011.md "Stage 2 run and the
  world build". data/tiles is 735 MB now (T-016).
- Also on the page since 2026-11-06: our own surface with OSM roads (T-028, T-084), 100 m town and
  water-country tiles (T-057, T-086), canoe on our mask (T-085), coarse canoe rivers and lake
  pieces (T-082, T-088), corridor count fix (T-090).
- C: ~130 GB free at 22:22 after Anita's cleanup; her floor is ~40 GB.

## In flight
- T-092 done and committed (5469ab95): cfg.access kept at 0.5.
- T-094 done and committed (66ee3fda): ktrips walks directed; walks 1-6% faster, canoe unchanged.
- T-039 done and committed (86b67450): 100 m tiles carry GLO-90 heights, L0.1 walks walk.py's
  slope rule; seam round the start 1.23 -> 1.04; data/fine is 1.48 GB; old fine tiles deleted.
- **T-057 mountain tiles**: the T-039 agent a98f3d32b23d45d3b (Opus, ~350k) resumed 2026-11-07,
  2 cores as `iso-t039`; builds into a new folder and swaps data/fine. Files: tools/fine/, data/fine.
- **T-093 bucket queue**: agent a1b45d0a8f0b37ef0 (Opus), 1 core. Files: js/search.js, notes/T-093.md.
- Idle agents, retire: a3e69e0de8c33f136 (T-011 stage 3), a0fa8e1aa1250534e (T-011 stage 1),
  a615cd2144584c81f (T-090), a5b92d38b9ad284d8, a53d87b2fa263c8db, a2e305ad8c2b66a43,
  a754c1f8965d2be43, a9b36a2c84531c427, a75105d4c451ffaf1.

## Next
1. When T-057 and T-093 report: review, smoke, commit. Then T-091 corridors by directed cost.
2. Open: T-083, T-065, T-073, T-031 (fold into the next river rebuild), T-075, T-007 (fjord and
   river country too fast at L3/L10: Sogndal, Coari 1.3-1.4).
3. Anita has not yet seen T-011 served; shots in the State line's notes.

## Anita's standing asks
- 2026-11-06 14:15: trainworld is the higher priority; keep ~40 GB free on C:. The terrain run
  restarted 14:17 as `launch.py 3 --min-free 40 --ahead 1` (launcher PID 35848); with C: at
  ~17 GB it computes what it holds and then waits for free space. MAP's friction raster deleted.
- 2026-11-06 afternoon: don't raise the shared core cap; isochrone proceeds slowly within its 4
  while she works on other things.
- 2026-11-06: page text needs no sign-off for now (in CLAUDE.md); 90 m downloads go ahead,
  keeping ~20 GB free on C:; tell her if room would need cleanup outside isochrone (not needed:
  peak ~6 GB).
- 2026-11-05: roads go ahead (rotating, she lacks 84 GB free); T-047's 2.8 GB rule approved;
  10,000 BC later (T-012 P3); 90 m terrain after roads.
- Corridor colour = speed along the stretch, width = trips, the red end not dark; hover a line
  for its mode and speed; smooth lines, no staircases.
- Rivers faint; water paler than land; hovers instant; a globe; hex cells by default.
- Carry x2. Survival and upstream canoe trips checked by her and fine (2026-11-05).
- Pause big downloads when she asks; resume at her word.
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
