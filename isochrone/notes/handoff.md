# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-06 02:50

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- All landed work is committed; smoke 16/16 at the last check (after T-079). 8800 serves this
  folder live (`localhost:8800/isochrone/`), no build step.
- Landed 2026-11-05: globe, hex look, corridors by speed, faint rivers, paler water, instant
  hovers, land danger (T-042/066/067), corridors after windows (T-077), M1 doc tidy (T-023),
  desert wadis (T-079, tools/hotdesert.py shared by canoe and hazard), level-bias study (T-080).
- Roads pass T-053 finished 2026-11-06 01:47: 513 of 513 extracts (chugoku rerun after three
  download timeouts), `build.py reduce` gave 16,183 tiles in data/roads/l1; data/roads 0.7 GB.
  failed.txt lists 13 names, all later done. iso-roads claim gone.
- C: 48 GB free at 01:50 (it swings; other sessions). Ledger cap 15 overnight.
- Idle agents (resume for follow-ups): T-080 level bias a754c1f8965d2be43 (Opus; T-081 if
  Anita says yes), T-079 desert a75105d4c451ffaf1 (Opus), roads addfc90633db2c1e0 (Opus).

## In flight
- T-028 and T-057 landed (7e807cd7): page on our surface with roads; 4,672 fine tiles (0.52 GB),
  partial town tiles need T-086 to show. MAP's tiles kept in data/tiles_map2020/ until Anita
  has compared. Eight new speeds-panel and info strings to show her (notes/T-028.md).
- T-084 (grade cap 10%, hazard on our land) and T-086 (partial town tiles) landed (cb7ac49b);
  smoke 17/17. Backups kept: data/tiles_map2020, data/hazard_map2020. Agents idle:
  a9b36a2c84531c427 (~430k tokens, retire), a2e305ad8c2b66a43.
- T-085 canoe water + river steps on our mask: T-079's agent a75105d4c451ffaf1. Owns
  tools/canoe/, canoe data, docs/canoe.md (uncommitted edits there are its own).
- New "Roads, towns" hover string for Anita: "...slowed by height and by slope up to a 10% grade."

## Next
1. Land T-085; then T-082 (canoe at L10 in river country), which waits for it.
2. T-011 90 m terrain (Anita queued it after roads): asked her whether to start the downloads
   (100-130 GB streamed); T-084 must land first (both touch the terrain side); Anita asked
   for a pause on big downloads once on 2026-11-05 (then said resume). Then T-039, T-057
   mountain tiles.
3. T-081 crossing-time rule for L10/L3 if Anita says yes (asked 2026-11-05, recommended);
   it goes into T-028's coarse averaging function.
4. Open: T-082 canoe at L10, T-083 desert river permanence, T-065, T-073, T-031 (fold into the
   next river rebuild), T-075.

## Anita's standing asks
- 2026-11-05: roads go ahead (rotating, she lacks 84 GB free); T-047's 2.8 GB rule approved;
  10,000 BC later (T-012 P3); 90 m terrain after roads.
- Corridor colour = speed along the stretch, width = trips, the red end not dark; hover a line
  for its mode and speed; smooth lines, no staircases.
- Rivers faint; water paler than land; hovers instant; a globe; hex cells by default.
- Carry x2. Survival and upstream canoe trips checked by her and fine (2026-11-05).
- Pause big downloads when she asks; resume at her word.
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
