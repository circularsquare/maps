# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-06 (morning, after Anita's reply)

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- All landed work is committed (last isochrone code commit 581579e0); smoke 17/17. 8800 serves
  this folder live (`localhost:8800/isochrone/`), no build step.
- On the page since 2026-11-06: our own surface with OSM roads (T-028, T-084), 100 m town and
  water-country tiles (T-057, T-086), canoe on our mask (T-085), coarse canoe rivers and lake
  pieces (T-082, T-088). Backups kept: data/tiles_map2020, data/hazard_map2020.
- C: ~45 GB free at the morning check (it swings; other sessions). isochrone data/ is ~10 GB.
  Anita: keep ~20 GB free; T-011's downloader pauses under 22 GB.
- Ledger was nearly full (trainworld 11 of cap 12) at the morning check.

## In flight
- T-090 canoe corridor bug (Anita's Semnan start: corridors cut off, look shifted north), Opus
  agent a615cd2144584c81f. Files: js/corridors.js, js/route.js, checks/. Will ask before any data
  rebuild.
- T-011 stage 1 (90 m terrain design, pilot, streaming downloader; holds T-089 and T-081), Opus
  agent a0fa8e1aa1250534e. Files: tools/terrain/, notes/T-011.md, data/raw/glo90, data/terrain.
  Reports before any world download; the orchestrator greenlights stage 2 (Anita already said
  yes to the downloads).
- Idle agents, retire: a5b92d38b9ad284d8 (T-087), a53d87b2fa263c8db (T-088, ~435k),
  a2e305ad8c2b66a43 (T-057/T-086), a754c1f8965d2be43, a9b36a2c84531c427, a75105d4c451ffaf1.

## Next
1. Land T-090 (smoke, before/after screenshots, commit).
2. Review T-011's pilot; greenlight the world stream (stage 2), then the rebuild of every level
   by crossing time (stage 3, keep data/tiles as a backup for Anita's before/after), then
   re-measure canoe cfg.access (notes/T-082.md). Then T-039 and T-057 mountain tiles.
3. Open: T-083, T-065, T-073, T-031 (fold into the next river rebuild), T-075.

## Anita's standing asks
- 2026-11-06: page text needs no sign-off for now (in CLAUDE.md); 90 m downloads go ahead,
  keeping ~20 GB free on C:; tell her if room would need cleanup outside isochrone.
- 2026-11-05: roads go ahead (rotating, she lacks 84 GB free); T-047's 2.8 GB rule approved;
  10,000 BC later (T-012 P3); 90 m terrain after roads.
- Corridor colour = speed along the stretch, width = trips, the red end not dark; hover a line
  for its mode and speed; smooth lines, no staircases.
- Rivers faint; water paler than land; hovers instant; a globe; hex cells by default.
- Carry x2. Survival and upstream canoe trips checked by her and fine (2026-11-05).
- Pause big downloads when she asks; resume at her word.
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
