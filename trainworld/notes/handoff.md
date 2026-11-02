# Orchestrator handoff, trainworld

Orchestrator: session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, since 2026-10-30 11:22
Updated: 2026-11-02 (clean break: nothing in flight; Anita starts a fresh orchestrator for her long TODO.md list)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master: c7c5785a plus this handoff commit. 8800 at c7c5785a, smoke 20/20 both shades.
- **LIVE (2026-10-31): the build of c7c5785a on R2 data `trainworld/2026-10-30b`**; website cef6eb1, `deploy.py --verify
  --page --version 2026-10-30b` all PASS. A code-only release: `deploy.py --skip-data --version 2026-10-30b --smoked`,
  then website add/commit/push (the auto-mode classifier refused my commit once, allowed it once Anita said so in the
  same message), `gh run watch`, then verify. A data change needs a new `--version`.
- R2 holds only `trainworld/2026-10-30b` (old prefixes purged by Anita, checked 2026-11-02). For later purges Anita runs `rclone purge
  r2:anitamaps/trainworld/<v> --s3-no-check-bucket` (the classifier refuses it for agents; rclone's "Failed to read
  versioning status ... AccessDenied" line is harmless, let it finish). Check with `rclone lsf r2:anitamaps/trainworld/ --dirs-only`.
- 8800 rebuild: a `.cmd` of 8 lines (checkout --detach <hash> in maps/.tw-snap; CARGO_TARGET_DIR=maps/.tw-snap-target;
  CARGO_BUILD_JOBS=2; %USERPROFILE%\.cargo\bin on PATH; cd .tw-snap/trainworld/app; npm run build), run through
  `cmd /c` with output to a log; mine is
  `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\c9a4388f-3ba1-4290-bb0a-d9ad50756061\scratchpad\snapbuild.cmd <hash>`.
  Then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD. Checks that import
  source (t1832-keep) need a dev Vite, not 8800.
- Shared files: when two agents' edits sit in one file, stage by hunk with a small script (HEAD text plus the one
  agent's pieces, `git hash-object -w` + `git update-index --cacheinfo`; examples stage_*.py in the scratchpad above).
- World: 1,925 shipped (Australia 28 of 28), 18 held, 5 parked (nic lez meo doo lel).
- Cores: the cap was raised to 15 on 2026-10-30 evening while Anita was out; check `python cores.py list` and set it
  back to 10 (`python cores.py cap 10`) while she is at the machine.
- data/work/pre-T1854/terrain (58 MB): delete once T-1854's terrain is seen clean in play. data/work/terrain-fine/
  (3.1 GB): T-1854's intermediate, kept for threshold tweaks.
- Next free task id: T-1892. Old tree .tw-trees/t1734 kept (Anita). Localhost needs `?acct=live` once for accounts.

## In flight
Nothing. Anita's TODO.md short-term has a long new list from her 2026-10-31 to 11-01 playtests: file it as a batch
(copy verbatim into notes/todo-2026-11-02.md and the briefs; trainworld/CLAUDE.md).

## Open questions to Anita (asked, unanswered)
- Blueprint row hover wording (T-1890): "Elevated +1: 1.0x", "Mountains +2.5x", "Cost factor 3.0x" to match the tag
  (now "Elevated +1 1", "Mountains + 2.5", "Cost factor × 3"); and "Water +2x" or "Water +2.0x".
- Mountains at grade: kept 3.0x by her additive rule (T-1890 replaced T-1887's 6x).
- Settings label "Lines side by side" vs "Draw lines side by side" (T-1886); Ctrl no-snap leaves the real-line ring on.

## Open problems
- T-1891: "Demand ran out of memory" once on a new NYC game on 8800; not reproducible by her or the agent; the label
  was wrong (any second worker death said "memory"); now "Demand stopped" with the panic in its hover. If it recurs,
  get that hover text or the console line `demand: worker N panicked`. notes/T-1891.md.
- T-1889: riverbanks priced as hills/mountains (East River at 42nd St) from T-1854's classes.

## Next (after her TODO batch)
1. T-1881 hillshade from a surface model shows dense centres as hills.
2. T-1880 buses as feeders (her question). T-1873 muting worm (now unblocked). T-1867 only if she wants it after T-1874.
3. T-1849, T-1863, T-1864 demand speedups; a `?demandWorkers=3` browser run of T-1862's keep.
4. T-1875, T-1876 to T-1879 (Australia follow-ups; T-1879 has a CLAUDE.md line for data/terrain during joins), T-1882.
5. T-1794 train-count settle; Southeast Asia (T-1738) when cores are spare. Waiting on Anita: bus design (T-1445).

## Lessons from this run
- Look at every UI agent's pictures yourself for spacing before reporting done (Anita, 2026-10-30, the Cities World
  row: "i really feel like you guys should be able to check this"). Measure text-to-rule gaps against the row pitch.
- An agent's tooltip text can wrap mid-range in its own picture; read the pictures, not just the strings.

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs; save a batch in notes/todo-<date>.md.
- The tight-usage week (10-26 to ~11-01) is over unless she renews it: tier by her global CLAUDE.md rules.
- Every demand task (loosely perf) has a "done when"; stop once met. Approximations welcome when invisible.
- Buses settled unless she asks. Thanks to agents in my own reply. Someday tasks are real open tasks.
