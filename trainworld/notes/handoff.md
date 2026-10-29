# Orchestrator handoff, trainworld

Orchestrator: session f537e44c-15c4-40d9-bcb8-a6d7cafc4ace, name maps-da, since 2026-10-29 night
Updated: 2026-10-29 night

Anita 2026-10-29 night (to maps-da): tokens still low all week, pick the token-efficient method over the
compute-efficient one; keep at least 4 of 16 cores free for the next few hours (so 12 at most, all sessions).
maps-45 wound down and confirmed: everything landed in a945e323, no agents or trees of its own left.
deploy.py now writes the trainsworld page without the GA include: untested, check the page after the next deploy.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master a945e323. 8800 at a945e323, smoke 20/20 both shades (maps-da). Anita: "looks good".
- LIVE site still 2026-10-28g (= 1550be70); nothing since is deployed.
  Anita 2026-10-29: "no need to deploy yet". Deploy only on her word (she has said pushing is allowed).
- shonei-server f74bfad deployed and pushed: /trainworld/ping + /trainworld/stats live; stats stay empty until the
  next site deploy (the ping code is efb32d52).
- data/packs/real: all seven shared-track saves shipped 2026-10-29 (no LRT1 block, so a full settle on load).
- World: 1,897 shipped, 18 held, 5 parked (nic lez meo doo lel), 329 planned; registry 1,960.
- 8800 rebuild recipe: `cmd /c <scratchpad>\snapbuild.cmd <explicit master hash>` (maps-45's scratchpad, still usable:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad),
  then `node trainworld/checks/smoke.mjs http://localhost:8800/trainworld/` and again with SMOKE_SHADE=dark.
  Never pass HEAD (it means .tw-snap's own HEAD).
- Deploy recipe: deploy.py --version <v> --smoked; website add trainsworld/ trainworld/, commit "trainsworld",
  push; gh run watch; deploy.py --verify --page --version <v>; rclone purge the old R2 prefix.
- Core cap 12 while Anita is up; keep 4 free for now (her ask, 2026-10-29 night).
- Uncommitted: Anita's TODO.md (never commit).
- Next free task id: T-1834.
- Old tree .tw-trees/t1734 kept for later thought (Anita). T-1801 had already landed (bcee8c05); archived.
- Website repo: T-1833's GA-off change not committed or pushed (goes with the next deploy, on her word).

## In flight
- Nothing. T-1822 stage 1 (agent t1822b, done) committed; see git log.

## Next
1. Rebuild 8800 at the T-1822 commit if not done, smoke both shades.
2. T-1822 stage 2 (ripple, Opus) after stage 1. T-1827..T-1832 save fixes 4-9 (fix 5 = 30 days on the account,
   fix 9 = hour-of-play warning, both yes from Anita).
3. T-1799 stage 3 (four-track trunks as one ribbon); T-1815 (preview waits during solves); T-1808 (calib areas file).
4. Smaller open: T-1794, T-1788, T-1806, T-1793, T-1778, T-1773, T-1776, T-1765's remaining zones.
5. Line groups stage 4 (boardings in the group view) only if Anita wants it.
6. Open asks for Anita: the bent group view, the flipped t, 55% selected dimming, Penn's 64 px Traffic dot.

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
