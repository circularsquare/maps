# Orchestrator handoff, trainworld

Orchestrator: session f9a7f88d-0531-4963-9df6-6771326e513f, name maps-45, since 2026-10-28 evening
Updated: 2026-10-29 afternoon (rewritten at the 500k cycling point; older detail is in this file's git history)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master d2e83c6b (T-1799 stage 1). LIVE site still 2026-10-28g (= 1550be70); nothing since is deployed.
  Anita 2026-10-29: "no need to deploy yet". Deploy only on her word (she has said pushing is allowed).
- shonei-server f74bfad deployed and pushed (Anita's word): /trainworld/ping + /trainworld/stats live; the game
  only pings from a build that carries efb32d52, so stats stay empty until the next site deploy.
- data/packs/real: all seven shared-track saves shipped 2026-10-29 (nyc par tyo seo tor sdj from
  artifacts/trainworld/t1305-shared/, lon from its london/ folder). They carry no LRT1 block, so they load with a
  full settle (fine, ~2 s on nyc/par).
- World: 1,897 shipped, 18 held, 5 parked (nic lez meo doo lel, none new), 329 planned; registry 1,960.
- 8800 at 9288b07a (all of today incl. T-1799 stage 1, T-1802), smoke 20/20 light and dark, real save 11,388,675.
- 8800 rebuild recipe: `cmd /c <scratchpad>\snapbuild.cmd <explicit master hash>` (scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad),
  then `node trainworld/checks/smoke.mjs http://localhost:8800/trainworld/` and again with SMOKE_SHADE=dark.
  Never pass HEAD (it means .tw-snap's own HEAD).
- Deploy recipe: deploy.py --version <v> --smoked; website add trainsworld/ trainworld/, commit "trainsworld",
  push; gh run watch; deploy.py --verify --page --version <v>; rclone purge the old R2 prefix.
- Core cap: keep 4 free while Anita is up (claims <= 12), 15 overnight. Agents often claim with dead pids; check
  `python cores.py list`.
- Uncommitted: Anita's TODO.md (never commit). Agents' in-progress files listed below.

## In flight
- T-1805 committed 5751e10f. T-1801 landed bcee8c05. 8800 at bcee8c05, smoke 20/20 both shades.
- Anita away 2026-10-29 evening ("feel free to use more cores"): cap 15. T-1799 stage 2 + T-1807 agent t1799s2 (Opus,
  tree .tw-trees/t1799b from bcee8c05, Vite 6189, 3 cores). T-1810 agent t1810 (Opus, main tree, Vite 6191, 2 cores).
  T-1810 filed (transfers handling 450-520 ms after a solve on shared NYC).
- T-1803/1804 committed 641ccef0. From Anita's TODO: T-1811 track tool (run ends on joining track, blueprint node drag
  in track tool, snap hover) agent t1811 (Opus, Vite 6183); T-1812 logo flip, stronger highlight, hand symbol bug, pinned
  panel titles, minimal station panel, agent t1812 (Sonnet, Vite 6187). T-1813 Lines-tab drag rules + group pill bullets
  queued after T-1809.
- T-1809 group view layout v2 (Anita's playtest: bent schematic, key stops only + toggle, crossings/transfers like
  related view, bullets at line ends + floating, order swaps, branch bugs): agent t1809 (Opus), main tree, Vite 6181.
- T-1802 committed 9288b07a. T-1805 placement-lag timing agent t1805 (Opus), main tree, Vite 6175, 2 cores.
- T-1803 no 4x cap for small edits + T-1804 loading bars fixed while zooming: agent t1803 (Sonnet), main tree,
  Vite 6163.
- No trees left besides t1801 (and the old .tw-trees/t1734, kept for later thought per Anita).

## Next
1. 8800 rebuild at current master + smoke both shades; tell Anita.
2. T-1805: an agent times Anita's two laggy placement cases on 8800 (first blueprint segment joining constructed
   rail; a branch off a blueprint) and checks constructing track / placing blueprints trigger no demand recalc.
3. T-1799 stage 2 (Traffic view bands, own widths, default traffic width a tad lower), then stage 3 (four-track
   trunks as one ribbon), T-1807 (station marks as wide as the ribbon). Asked Anita how stage 1 reads.
4. Smaller open: T-1794 (settle leftovers), T-1788 (click-to-ghost under 20 ms), T-1806 (heal_market reach),
   T-1793 (pulse overstated platform crowds), T-1778, T-1773, T-1776 (train sounds), T-1765's remaining zones.
5. Line groups stage 4 (boardings in the group view) only if Anita wants it.

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
