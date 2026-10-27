# Orchestrator handoff, trainworld

Orchestrator: session ee78dc43-077b-4b8e-9cf9-ac5bb7367793, since 2026-10-26
Updated: 2026-10-27 morning (winding down at Anita's ask)

Read trainworld/CLAUDE.md and TASKS.md first. History of this session: notes/handoff-2026-10-26.md.

## State
- master c5bc407e. 8800 runs 26aed847 (T-1521 realistic fares and everything before), smoke 20/20
  light and dark, real save 12,483,326 rail trips a day. Main tree demand WASM current.
- Nothing published since 2026-10-25b; much has landed since (deploy only on Anita's word).
- Landed 2026-10-26/27: Canada 49/50 (T-1239, T-1377), income groups (T-1431), realistic fares
  (T-1521), access fixes (T-1506, T-1511), superholiday rush days (T-1373), service levels
  (T-1536), buses on roads with editing (T-1465, T-1512-1517), car parks (T-1528), view-first demand
  and estimates (T-1533, T-1538, T-1539, T-1530), edit speed (T-1380-1382, T-1540), Japan OOM fix
  (T-1482), many UI items. ARCHIVE.md has each.
- Uncommitted: TASKS.md (T-1547 to T-1554 filed from TODO.md this morning; commit it), Anita's
  TODO.md (never commit), notes/t1150/run/ (196 MB run output, never commit), notes/T-1546/ (agent's).
- Core cap 10 (Anita is on the machine). `python cores.py list` at maps root.

## In flight (agents of this session; their reports come here)
- T-1543 rough period split over workers: Opus, tree .tw-trees/t1543 (branch tw/t1543). Done when
  ARCHIVE has T-1543 and notes/T-1543.md exists. Land with scratchpad land_wasm.ps1 or by hand:
  commit in tree, merge master in, `npm run wasm` in tree app, typecheck, cargo test
  `--release --features demand`, merge tw/t1543, rebuild 8800, smoke both shades, `npm run wasm`
  in main app, `agent-tree.mjs remove t1543`.
- T-1466 bus ridership vs real buses: Sonnet, analysis only, notes/T-1466.md; tree t1466 to remove.
- T-1546 rail against plane at 4-5 h: Opus, analysis only, notes/T-1546.md (scripts in
  notes/T-1546/). Keep Anita's rulings: one world plane fare, same money weight for every mode,
  0.4x long trips, no car upkeep, 15 km ride reach.

## Next
1. Commit TASKS.md. Land T-1543; relay T-1466 and T-1546 to Anita.
2. High priority (Anita): memory per zone and worker, T-1169 with T-1545 (market's 1.4 GB
   temporaries). Opus.
3. TODO.md items T-1547 to T-1554 (UI batch 1547/1548/1549/1551/1552; 1550/1553/1554). Sonnet
   mostly; T-1554 changes the fare scale's baseline (saves must load with the same fares).
4. Demand calibration before more world: T-1546, T-1542 (US car), T-1508 (big far pairs too few
   trips), T-1374 (superholidays stage 3), T-1520 (income groups' cost).
5. Rest of the world is lower priority than calibration (Anita 2026-10-27).

## Anita's standing asks
- Several agents going and one compute job; use idle cores; batch rebuilds.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Link every image as a clickable workspace-relative markdown link.
- Long jobs claim cores in cores.py; refused steps: change and retry, don't stall.
- Someday tasks are real open tasks.
