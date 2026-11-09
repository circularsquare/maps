# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-09 (PAUSED by Anita: trainworld's urgent fixes take this week's usage; "will have to come back to isochrones later")

Read isochrone/CLAUDE.md and TASKS.md first. To resume: `/takeover isochrone`, then ask Anita the
questions in notes/for-anita.md before starting T-007 stage 2.

## State at the pause
- Nothing is running: no agents, no builds, no isochrone claims in the cores ledger. All work is
  committed to master. Smoke 18/18 on the served page at the pause. The map's data isn't
  published anywhere yet (T-016).
- `localhost:8800/isochrone/` serves this folder live, no build step. Checks from the maps root:
  `node isochrone/checks/smoke.mjs` (18), `node isochrone/checks/corridors.mjs` (10),
  `node isochrone/checks/levels.mjs`, `node isochrone/checks/fine.mjs`,
  `node isochrone/checks/ktrips.mjs trips 1000` (and `ONLY=rivers`, `ONLY=lakes`). Canoe level
  agreement: notes/T-102/ (`measure.mjs`, `cmp.mjs`, `excess.mjs`); rerun after any canoe data change.
- TASKS.md "Where things stand" is the summary of what the map is now. Detail per area: SPEC.md and
  docs/; each task's numbers: notes/T-###.md; choices: DECISIONS.md (newest first).

## What the last stretch landed (2026-11-06 and 07, all served and committed)
- Walking: 90 m terrain and directed speeds at every level (T-011), the same slope rule at 100 m
  and mountain tiles (T-039, T-057), corridors costed as the search walks (T-091), bucket queue
  (T-093).
- Canoe: lakes' own ERA5 temperature (T-043); river gradient from GMTED everywhere, big-river
  stream-power closure, GIRES desert no-flow (T-031, T-083); up rapids 2.5 km/h from travel
  records (T-073); small-river length factors, Hydrography90m gradient in tropical forest, coarse
  river steps count the main river only (T-100, T-101); `cfg.access` 1.0 (T-102). T-035 closed
  with no code change.
- Look: windows fade into the coarser level (T-014); corridor lines join across windows (T-104,
  T-105). Upkeep: M2 doc tidy (T-078), T-097 to T-099.

## Where to pick up
1. Anita's answers (notes/for-anita.md). The one that unblocks the most is T-007 stage 2.
2. T-007 stage 2: split coarse cells into L1 land pieces. Design, file list and prototype numbers:
   notes/T-007.md (section 7 lists the files; scripts in notes/T-007/). Its agent's context is
   gone: brief a new Opus agent from that note. The prototype ran on js/ as of 677e6c58, before
   T-093's bucket queue, so its search patch must be redone on today's js/search.js; the one move
   mask per cell is what keeps the click as fast (a per-move piece lookup was 30-35% slower).
3. After T-007: T-075 islands (measure first), T-103 bigger canoe speed-ups, T-016 publishing
   encoding (T-007 and T-016 both touch build_tiles.py and js/tiles.js).
4. Any time: T-095 re-time on a quiet machine, T-096 lake temperatures against satellite, T-015
   big screens, T-065 once Anita has looked at upstream Amazon routes.

## Before publishing (T-016)
- Hydrography90m is CC BY-NC; the river step files carry that until Anita decides (for-anita.md).
- GLO-90 heights are inside the 100 m `.land` tiles: its notice and no-liability sentence are in
  the info panel and must travel with any redistribution (research/friction.md).
- Sizes: data/tiles ~750 MB, data/fine 2.15 GB.

## Data worth knowing about (gitignored, on this machine only)
- Rebuilding walking tiles: `python isochrone/tools/build_tiles.py 3 --out <name>` from the maps
  root (claim cores first; ~20 min on 3), then `build_tiles.py swap <backup> --from <name>`, then
  rerun the land hazard's `tiles` stage (docs/hazard.md 6).
- data/terrain/l1/ (1.3 GB) is the 90 m world run's output, 11 h to remake and ~100 GB streamed:
  keep it. data/terrain/elev/ (1.8 GB) holds heights for the 100 m tiles.
- data/raw/rivers_work/ holds the only copies of river build intermediates, including
  `gires.npz` and `merit.npz` (their downloads were deleted; docs/code-map.md says how to refetch).
- Rollbacks still on disk: data/tiles_pre_t011 (243 MB; can go once Anita has judged the mountain
  walks), data/tiles_map2020 and data/hazard_map2020 (old MAP surface, superseded, can go).

## Anita's standing asks
- 2026-11-09: paused, come back later. Don't start isochrone work until she says.
- 2026-11-06: trainworld is the higher priority; keep ~40 GB free on C:; don't raise the shared
  core cap (isochrone's limit is 4 cores in all); page text needs no sign-off for now (in CLAUDE.md).
- 2026-11-05: roads, T-047's 2.8 GB tile rule and 90 m terrain approved and done; 10,000 BC later
  (T-012).
- Corridor colour = speed along the stretch, width = trips, the red end not dark; hover a line
  for its mode and speed; smooth lines, no staircases.
- Rivers faint; water paler than land; hovers instant; a globe; hex cells by default.
- Carry x2. Survival and upstream canoe trips checked by her and fine (2026-11-05).
- Pause big downloads when she asks; resume at her word.
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
