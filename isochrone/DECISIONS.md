# Decisions

Newest first. One entry per decision: what, and why. The spec states the result; this keeps the
reasoning and what it replaced.

## 2026-10-14

- **View refine thresholds and sizes** (T-002): L3 from zoom 5, L1 from zoom 7; windows are the
  view plus a quarter of its size each side (the L3 under an L1 window, its full size each side,
  so it seeds L1 well); caps 3M cells (L3), 2M (L1). L1 seeds from the L3 window, not L10:
  seeds are better and the L3 window is cheap (~30-100 ms at zoom 7+). Measured times in
  docs/resolution.md.
- **Overlays are canvas sources painted in the worker** (T-002), replacing a PNG data URL made
  on the main thread. Canvas sources let the page cut a hole in the coarser layer where a finer
  window is (otherwise coarse colour shows through the window's water) without re-encoding the
  world image; the worker does all the per-pixel work.
- **5 degree tiles at both levels** (600 cells at L1, 200 at L3): divides both grids exactly and
  lines L3 and L1 tiles up. `data/tiles/` (135 MB) is gitignored and rebuilt in 45 s, like the
  raw tif.

- **Docs and tasks follow trainworld's layout** (Anita): CLAUDE.md, SPEC, DECISIONS, TASKS,
  ARCHIVE, TODO, notes, research, docs. Compartmentalised work goes to subagents.
- **Fastest path, survival reported afterwards** (Anita): "the pathfinding should still just take
  fastest path i think and not care about survival". Hazard is a per-cell rate; survival along the
  fastest path is exp(-sum hazard x time).
- **Canoe is the next mode** (Anita: "the most interesting mode of transport to add next").
- **Dynamic resolution: coarse first, refine the view** (Anita: "compute it coarsely at first,
  then wherever we're looking at we refine"). Design in docs/resolution.md. Her suggestion of a
  coarser grid in uniform regions and a finer one in mountains is kept as a later stage: the view
  refine comes first because it also gives the close-up look she asked for.
- **Modes as layers, toggles remove layers or edges**, so each toggle combination does not need
  its own prepared data (orchestrator, so toggles stay cheap as modes are added).
- **The MAP friction surface is the walking base for now**; CC BY 4.0 allows it with credit. Our
  own surface from primary layers is needed for 10,000 BC and finer terrain (research T-004).

## 2026-10-13

- **Walking first, whole world, click anywhere** (Ashley). Static page with the search in a worker,
  no server, so it can be published like the other maps.
- **10 km cells for the first version**: 7.5M cells solve in ~0.3 s in the browser. A cell's speed
  is the mean of its 1 km pixels' friction (time per metre), so crossing time is kept; a cell is
  land when a quarter of its pixels are.
- **A day is 8 hours of walking** in the legend and hovers.
