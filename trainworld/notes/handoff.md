# Orchestrator handoff, trainworld

Orchestrator: session f537e44c-15c4-40d9-bcb8-a6d7cafc4ace, name maps-da, since 2026-10-29 night
Updated: 2026-10-29 night

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master b6ff3eff (the 2026-10-29 night batch). 8800 at b6ff3eff, smoke 20/20 both shades. Anita: blueprint text OK.
- LIVE site still 2026-10-28g (= 1550be70); nothing since is deployed. Deploy only on Anita's word.
  Website repo: T-1833's GA-off change not committed or pushed; deploy.py writes the page without GA: check after deploy.
- shonei-server 5719c6f deployed by Anita 2026-10-29 (day/week backups; deleting is for good), not pushed. The auto-mode
  classifier refuses deploy.ps1 from me: hand her `cd C:\Users\anita\projects\shonei-server` + `.\deploy\deploy.ps1`.
- 8800 rebuild: `cmd /c C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f537e44c-15c4-40d9-bcb8-a6d7cafc4ace\scratchpad\snapbuild.cmd <explicit master hash>`
  (log to a file), then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD.
- Deploy recipe: deploy.py --version <v> --smoked; website add trainsworld/ trainworld/, commit "trainsworld",
  push; gh run watch; deploy.py --verify --page --version <v>; rclone purge the old R2 prefix.
- World: 1,897 shipped, 18 held, 5 parked (nic lez meo doo lel), 329 planned; registry 1,960.
- Cores: 12 at most while Anita is up (she asked to keep 4 free, 2026-10-29 night).
- Uncommitted: Anita's TODO.md (never commit).
- Next free task id: T-1844.
- Old tree .tw-trees/t1734 kept for later thought (Anita).
- Localhost has no account server until `?acct=live` is opened once per browser (account.ts, deliberate).

## In flight
- T-1799 stage 4 (lines ease out of their side-by-side offset at splits/merges and ribbon ends; Anita's words and
  screenshot in maps-da scratchpad images/2.png): agent t1799c (Sonnet) resumed on the clean tree.
- T-1843 Settings tooltip/label edits + remove Explore a trip button (agent t1843, Sonnet, Vite 6193).

## Next
1. T-1840: every player-facing string in one review doc with flagged lines and proposed rewrites (also fix
   docs/ui-panels.md's old tutorial step words).
2. Doc tidy: stale TASKS lines (T-1782, T-1791, T-1823, T-1824 landed).
3. T-1407 / T-1780 ghost stalls after an edit (setTrainCount holds the worker 320-570 ms, found in T-1815); T-1842.
4. T-925 licence rebuild of nine European zones (asked Anita whether next). T-1831 unreadable-saves card (P3).
5. Waiting on Anita: tutorial card style notes (T-1822 st3), bus design (T-1445), base map palette (T-676).

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list; she says "text acknowledged" or names lines. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs (trainworld/CLAUDE.md).
- Tokens tight this week (to ~11-01): token-efficient over compute-efficient; Sonnet for settled work, Opus for
  design, data correctness, unknown bugs.
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks in the foreground;
  browser loads count their demand workers in core claims.
- Someday tasks are real open tasks.
