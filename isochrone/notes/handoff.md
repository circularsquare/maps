# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-06 11:30

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- All landed work is committed; smoke 18/18 since T-090 (df18aa48). 8800 serves this folder live
  (`localhost:8800/isochrone/`), no build step.
- On the page since 2026-11-06: our own surface with OSM roads (T-028, T-084), 100 m town and
  water-country tiles (T-057, T-086), canoe on our mask (T-085), coarse canoe rivers and lake
  pieces (T-082, T-088), corridor count fix (T-090). Backups kept: data/tiles_map2020,
  data/hazard_map2020.
- C: ~43 GB free at 11:23 (it swings; other sessions). Anita: keep ~20 GB free; the terrain
  downloader waits under 22 GB.

## In flight
- **T-011 stage 2, world terrain run** (90 m GLO-90 streamed, directed crossings per L1 cell),
  PID 18900, started 2026-11-06 11:23 with 1 worker (ledger full; claim `iso-terrain`, granted 1).
  Status: `python isochrone/tools/terrain/world.py status` (maps root); log
  data/terrain/world.log; stdout/err data/terrain/world.out/.err. 18,984 tiles, ~70-75 GB
  streamed at ~7 MB/s; compute ~50 core-hours, so it wants 4 workers. Restart with more: stop PID
  18900 only, then `python isochrone/tools/terrain/world.py run --workers N` detached (resumes from
  data/terrain/done.txt), claim again with the new PID. A background watcher in this session
  (scratchpad/orch/watch_terrain.py) fires when the ledger has 3+ free or the run exits.
- **T-011 stage 3**, page side (build_tiles.py reads the planes, d1/e1 planes, directed search,
  L3/L10 by crossing, output to a new folder behind a `?tiles=t011` switch), Opus agent
  a3e69e0de8c33f136.
- **cross.mjs speed-up**, a copy `tools/terrain/cross_fast.mjs` to swap in at the run's next
  restart, Sonnet agent aa3eb522dfaa8baa2.
- Idle agents, retire: a0fa8e1aa1250534e (T-011 stage 1, ~465k), a615cd2144584c81f (T-090),
  a5b92d38b9ad284d8, a53d87b2fa263c8db, a2e305ad8c2b66a43, a754c1f8965d2be43, a9b36a2c84531c427,
  a75105d4c451ffaf1.

## Next
1. When cores free up or cross_fast lands: restart the run with more workers and the fast engine.
2. When the world run ends: stage 3 builds the full folder; swap it in (keep data/tiles as
   t011's before), show Anita before/after (mountains get faster: 0.64-0.83 of today's), then
   re-measure canoe cfg.access (notes/T-082.md), then T-039 and T-057 mountain tiles.
3. Open: T-083, T-065, T-073, T-031 (fold into the next river rebuild), T-075. T-016 publishing
   must weigh L1 growing ~3x (176 -> ~530 MB).

## Anita's standing asks
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
