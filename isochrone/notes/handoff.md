# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-05 13:06

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- First handoff file. The previous orchestrator (2026-10-26) paused at Anita's word and is long
  gone; nothing was in flight. Last isochrone commit 6446db66 (pause point). Smoke 15/15 then.
- 8800 serves this folder live (`localhost:8800/isochrone/`), no build step.
- C: has ~15 GB free (2026-11-05): world-scale OSM work must fetch, reduce and delete region by
  region.
- Cores: isochrone's own cap is 4; the shared ledger was at 11 of 12 when this session started
  (trainworld runs), so agents get 1 each for now.

## In flight
- T-010 stage 1, roads design + Cluj/New York test: agent addfc90633db2c1e0 (Opus), session
  4a6a5ff2, started 13:10. Owns tools/roads/, docs/roads.md, notes/T-010*. Tools built
  (tools/roads/build.py, fetch.py, regions.py); rule comparison running as node PID 24268, log
  notes/T-010/compare.log, ~8 tiles. Done when docs/roads.md and notes/T-010.md exist, stage 2
  filed as T-053, data/raw/osm/ emptied (1.56 GB there now). Settled: add central-america to the
  fetch; keep 100 m road bits for every tile. No world build.
- Page agent aa378ee8ea1276653 (Sonnet), session 4a6a5ff2: T-037 and T-052 landed (7fdc77b6);
  now T-063 water paler, then T-061 instant hovers (sent ~13:40). Owns index.html, js/paint.js,
  js/riverlines.js. Done when both lines are in ARCHIVE.md and t063_*/t061_* screenshots exist.

## Long runs
(none)

## Next
1. T-010 roads, stage 1 (in flight): design + a Cluj / New York test, to Anita for review.
2. Then one region-by-region OSM pass for the 1 km roads (T-010 stage 2) and the 100 m tiles'
   roads (T-057), then T-028 (page on our own surface).
3. Page queue, in this order since all touch index.html / js/paint.js: T-052 (in flight, now
   with corridors yellow to dark red), T-063 water paler (sent), T-061 instant hovers (resume the page agent), T-062
   corridor colour = speed only, width = trips, stretch hover with mode and speed, T-060
   diagonal / hex look test, T-059 globe (MapLibre 5).
4. T-042 land hazard. T-039 (90 m slopes) before T-057's mountain tiles.
5. Small work alongside: T-048, T-031, T-036.

## Anita's standing asks
- 2026-11-05: roads go ahead; T-047's 2.8 GB rule approved; 10,000 BC is for later (T-012 P3);
  rivers faint, "if they show up as routes we take via boat they should be bright".
- Later 2026-11-05: a globe; instant hovers of our own; corridors yellow (most shared) fading to
  dark red; cells stay crisp, no smoothing, try diagonals or a hex look.
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
