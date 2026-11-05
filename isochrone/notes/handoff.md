# Orchestrator handoff, isochrone

Orchestrator: session 4a6a5ff2-cbc0-46a5-8b33-b531ae60e4e9, name maps-b5, since 2026-11-05 13:06
Updated: 2026-11-05 15:30

Read isochrone/CLAUDE.md and TASKS.md first.

## State
- Landed today (all committed, smoke 15/15 each time): faint rivers and corridor colour by
  speed (T-052, T-062, T-064: speed along the river, smoothed, nearest-line hover), window edge
  strip (T-037), paler water on canoe starts (T-063), instant hovers (T-061), `?look=diag|hex`
  test looks (T-060, open for Anita's pick), river lines in a worker (T-048). Last commit c79d03c2.
- 8800 serves this folder live (`localhost:8800/isochrone/`), no build step.
- C: has ~15 GB free: world-scale OSM work fetches, reduces and deletes region by region.
- Cores: the shared ledger was near its cap all day (trainworld); agents run on 1 core each.
- T-010 stage 1 landed (57892f55): tools/roads/, rule share3, docs/roads.md waiting on Anita's
  review. Roads agent addfc90633db2c1e0 (Opus) is idle; resume it for T-053 (rerun regions.py
  first so central-america is in the list).

## In flight
- T-059 globe: agent abe9a2322cac44954 (Opus), started ~15:30. Owns index.html, js/paint.js,
  js/riverlines.js, js/riverworker.js, js/windows.js / session.js view handling, smoke.mjs. Done
  when T-059 is in ARCHIVE.md and t059_*.png exist.
- Page agent aa378ee8ea1276653 (Sonnet) is idle at ~270k tokens; use a fresh agent for new page work.

## Long runs
(none)

## Next
1. Anita reviews docs/roads.md (asked 2026-11-05 ~16:00); then T-053 (world roads, ~4 h on two
   cores) with T-057's 100 m road bits in the same pass; then T-028 (page on our own surface).
2. Land T-059 globe.
3. Anita's pick for T-060 (crisp / diag / hex); T-065 creek-weaving and T-056 rays after she looks.
4. T-042 land hazard. T-039 (90 m slopes) before T-057's mountain tiles. T-031, T-036 alongside.

## Anita's standing asks
- 2026-11-05: roads go ahead; T-047's 2.8 GB rule approved; 10,000 BC later (T-012 P3).
- Corridor colour = speed along the stretch, width = trips; hover a line for its mode and speed.
- Rivers faint; water paler than land; hovers instant; a globe; cells stay crisp, no smoothing,
  diagonals or hex tried as looks.
- Commits: the orchestrator commits isochrone/ paths as it goes; never push unasked.
