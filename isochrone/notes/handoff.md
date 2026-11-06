# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-05 21:40

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- Landed today, all committed, smoke 15/15 each time (last 65fd440e): rivers faint; corridors
  coloured by speed along the stretch (river length, smoothed), width by trips, nearest-line
  hover, lines straightened and smoothed (T-052, T-062, T-064, T-070, T-072); paler water on
  canoe starts (T-063); instant hovers, no scroll bar (T-061, T-069); hex the default cell look,
  `?look=crisp|diag` kept (T-060); river lines in a worker (T-048); globe on MapLibre 5.24.0
  (T-059); upstream 0.4 of the current, floor 2.5 km/h, carry x2 kept (T-071); roads design and
  tools, rule share3 (T-010 stage 1, docs/roads.md).
- 8800 serves this folder live (`localhost:8800/isochrone/`), no build step.
- C: had 52 GB free at 14:52. The roads pass keeps <=2 extracts on disk and stops under 6 GB.
- Cores: the shared ledger is at its 12 cap (trainworld); isochrone agents get 1 each.
- Idle agents (resume for follow-ups): roads addfc90633db2c1e0 (Opus), page a188a1156511192ef
  (Sonnet), T-071 upstream aeeb2ba51b2f7341c (Opus), globe abe9a2322cac44954 (Opus).
- Uncommitted and others': docs/hazard.md, notes/T-042/, research/friction.md (T-042 agent).

## In flight
- T-066 (bf42d32e) and T-067 (535656b3) landed: land danger on the page, T-042 archived.
  New on-page strings shown to Anita (survival on walking hovers from 0.5% risk; info panel
  danger paragraph and credits).
- T-077 landed (bd409831): corridors after windows, faster in windows (L10 not), hazard in
  reparent slack, a stale-node bug fixed, ktrips walking trips.
- T-023 M1 doc tidy landed (c306d1bc).
- T-079 desert wadis (Opus agent a75105d4c451ffaf1, 21:40): owns tools/canoe/, tools/hazard/ (shared mask),
  the canoe river data, docs/canoe.md, notes/T-079.md. Rebuilds river data; ledger iso-t079.
- T-080 done (agent a754c1f8965d2be43 idle, resume for T-081): zooming in makes walks ~19%
  faster L10->L1, not slower; L0.1 is 21% slower than L1 from the MAP vs LC100+OSM gap. Fix
  T-081 (crossing-time rule) asked of Anita; T-082 canoe at L10 filed.
- Roads pass at 21:29: 313 of 513 extracts done, 55 GB free; at this rate it ends ~02:00.
  CHELSA and OSM water stay in data/raw/; data/raw/t042/ (1.5 GB) goes once T-066 has run.
- T-076 landed (8ada9b62). Page agent a188a1156511192ef idle at ~240k tokens.
- T-056 landed (e3730347): corridors re-chosen tree, counting 2-4x slower. T-077 (count after
  windows paint, speed up `reparent`) waits for T-067 to free js/session.js; fresh agent (the
  T-056 one is at ~455k tokens).
- T-036 landed (2f0b985b): dead-end river water. T-074 river water paler: page agent
  a188a1156511192ef (Sonnet), owns js/paint.js, the flag in js/canoe.js.

## Long runs
- T-053 world roads pass: `python -u isochrone/tools/roads/world.py --jobs 1` from the maps root,
  driver PID 25032, started 14:52, ledger claim `iso-roads`. Log `isochrone/data/roads/world.log`
  (stderr world.err; per extract data/roads/logs/). Alive: `Get-Process -Id 25032` and the
  newest log line. Done when data/roads/done.txt has 513 lines; failures in failed.txt. ~7-8 h.
  Restart: same command, then claim again with the new PID (notes/T-053.md). When it ends:
  `build.py reduce`, the GeoTIFF, then T-028 and T-057's .roads; release `iso-roads`.

## Next
1. When T-053 ends: reduce + GeoTIFF, then T-028 (page on our own surface with roads), T-057
   (100 m tiles: towns and water country first).
2. T-011 90 m terrain right after T-053 (Anita, 2026-11-05), then T-039, T-057 mountain tiles.
3. Land T-079 (desert wadis) and review T-080's proposal with Anita.
4. Anita has checked survival and upstream trips (fine, 2026-11-05). T-065 creek-weaving, T-073
   small rivers, T-031 (fold into a river rebuild) still open.

## Anita's standing asks
- 2026-11-05: roads go ahead (rotating, she lacks 84 GB free); T-047's 2.8 GB rule approved;
  10,000 BC later (T-012 P3); 90 m terrain after roads.
- Corridor colour = speed along the stretch, width = trips, the red end not dark; hover a line
  for its mode and speed; smooth lines, no staircases.
- Rivers faint; water paler than land; hovers instant; a globe; hex cells by default.
- Carry x2 (settled 2026-11-05).
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
