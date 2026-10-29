# Orchestrator handoff, trainworld

Orchestrator: session f9a7f88d-0531-4963-9df6-6771326e513f, name maps-45, since 2026-10-28 evening
Updated: 2026-10-29 evening (past 600k: cycle when convenient; older detail is in this file's git history)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master a1037e0e (plus doc commits after). LIVE site still 2026-10-28g (= 1550be70); nothing since is deployed.
  Anita 2026-10-29: "no need to deploy yet". Deploy only on her word (she has said pushing is allowed).
- shonei-server f74bfad deployed and pushed: /trainworld/ping + /trainworld/stats live; stats stay empty until the
  next site deploy (the ping code is efb32d52).
- data/packs/real: all seven shared-track saves shipped 2026-10-29 (no LRT1 block, so a full settle on load).
- World: 1,897 shipped, 18 held, 5 parked (nic lez meo doo lel), 329 planned; registry 1,960.
- 8800: last smoked build bcee8c05 (20/20 both shades). A rebuild at a1037e0e was started (background bdm6vykh6, log
  scratchpad\snap_a103.log): smoke it both shades, then tell Anita.
- 8800 rebuild recipe: `cmd /c <scratchpad>\snapbuild.cmd <explicit master hash>` (scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad),
  then `node trainworld/checks/smoke.mjs http://localhost:8800/trainworld/` and again with SMOKE_SHADE=dark.
  Never pass HEAD (it means .tw-snap's own HEAD).
- Deploy recipe: deploy.py --version <v> --smoked; website add trainsworld/ trainworld/, commit "trainsworld",
  push; gh run watch; deploy.py --verify --page --version <v>; rclone purge the old R2 prefix.
- Core cap 15 (Anita out 2026-10-29 evening, "feel free to use more cores"); 12 while she is up.
- Uncommitted: Anita's TODO.md (never commit), and the in-flight agents' files below.

## In flight
- Tree .tw-trees/t1799b (T-1799 stage 2 Traffic bands + T-1807 station marks span ribbons, built and checked) is
  MID-REBASE onto master: conflicts in app/src/map/renderer.ts and rendererShaders.ts against T-1812's highlight.
  Agent t1799m (Sonnet) resolves and `git add`s. Then: `git -C .tw-trees/t1799b -c core.editor=true rebase
  --continue`, `node trainworld/checks/agent-tree.mjs land t1799b "<sentence>"`, remove the tree, rebuild 8800.
- T-1813 Lines-tab drag (centre groups, between reorders/ungroups, hand order, station reorder) + group pill bullets:
  agent t1813 (Opus), main tree, Vite 6185.
- T-1814 shared Tokyo gets no transfer pulses: agent t1814 (Sonnet), main tree (demandClient.ts), Vite 6177.
- Old tree .tw-trees/t1734 kept for later thought (Anita).

## Next
1. Land t1799b, rebuild 8800, smoke both shades, tell Anita what's on it.
2. T-1799 stage 3 (four-track trunks as one ribbon); T-1815 (preview waits during solves); T-1808 (calib areas file).
3. Smaller open: T-1794, T-1788, T-1806, T-1793, T-1778, T-1773, T-1776, T-1765's remaining zones.
4. Line groups stage 4 (boardings in the group view) only if Anita wants it.
5. Open asks for Anita: the bent group view, the flipped t, 55% selected dimming, Penn's 64 px Traffic dot.

## Anita's calls this session (in DECISIONS.md)
- Attractions: order-of-magnitude is enough; budget model by page views (T-1766); no per-country term.
- Real networks share track, four-track = two double tracks (T-1305); London at 13.6% holds accepted (my call).
- No train slower than walking, ramp to full share at 30% quicker (T-1782); -11% rail trips OK.
- Line groups are only a grouping; names default to the oldest line, editable; copy joins and opens the group view;
  the group view has station names and shared pills at interchanges; passed stations have faded outlines.
- Side-by-side bands 80% width, fold zoomed out, Traffic keeps own widths (T-1799).
- Analytics: own-server heartbeat, no Google (T-1771). Favicon redrawn to her design. Bus size and pause OK.

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Tokens tight this week (to ~11-01): Sonnet for settled work, Opus for design, data correctness,
  unknown bugs; compute-heavy token-light approaches; investigations stop at a written answer.
- Someday tasks are real open tasks.
