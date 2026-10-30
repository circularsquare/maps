# Orchestrator handoff, trainworld

Orchestrator: session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, since 2026-10-30 11:22
Updated: 2026-10-30 evening (maps-da wound down; nothing in flight)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master 4db47c92 (T-1854 finer terrain). 8800 at 4db47c92, smoke 20/20 both shades, 1,988 terrain files in its dist,
  real save 11,392,296 rail trips a day (same as before T-1854).
- **Deploy 2026-10-30 is half done** (= ba582afb, with Australia, before T-1854): data on R2 `trainworld/2026-10-30`,
  verified; website files staged in ../website (no GA in the page). The website commit was refused by the auto-mode
  classifier, so Anita runs: `git -C C:\Users\anita\projects\website commit -m "trainsworld"`, `... push`,
  `venv\Scripts\python.exe trainworld\tools\deploy.py --verify --page --version 2026-10-30`, then
  `rclone purge r2:anitamaps/trainworld/2026-10-28g --s3-no-check-bucket` once the page works. Until she pushes, live
  is 2026-10-28g. The next deploy carries T-1854's new terrain files (299 MB, gzip) as well.
- Deploy recipe: deploy.py --dry-run, then --version <v> --smoked; website add trainsworld/ trainworld/, commit, push;
  deploy.py --verify --page --version <v>; rclone purge the old prefix.
- 8800 rebuild: `cmd /c C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\c9a4388f-3ba1-4290-bb0a-d9ad50756061\scratchpad\snapbuild.cmd <explicit master hash>`
  (8 lines: checkout --detach in .tw-snap, CARGO_TARGET_DIR=.tw-snap-target, CARGO_BUILD_JOBS=2, cargo on PATH,
  npm run build in .tw-snap/trainworld/app), log to a file, then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark. Never pass HEAD. Checks that import source (t1832-keep) need a dev Vite, not 8800.
- Shared files: when two agents' edits sit in one file, stage by hunk with a small script (maps-18 scratchpad
  stage_*.py are examples: HEAD text plus the one agent's pieces, `git hash-object -w` + `update-index --cacheinfo`).
- World: 1,925 shipped (Australia 28 of 28 added), 18 held, 5 parked (nic lez meo doo lel).
- Cores: cap raised to 15 while Anita was out (2026-10-30 evening); set back to 10 when she is at the machine.
- data/work/pre-T1854/terrain (58 MB) backup: delete once T-1854 is seen clean in play. data/work/terrain-fine/
  (3.1 GB) is T-1854's intermediate, kept for later threshold tweaks.
- Next free task id: T-1884. Old tree .tw-trees/t1734 kept (Anita). Localhost needs `?acct=live` once for accounts.

## In flight
- T-1883 two-row bar landed (money on row one, a two-row divider before the clock, 1 px row gap). Anita wants it in the deploy: then redeploy as version 2026-10-30b from a new master build (full data,
  since T-1854's terrain files are new), re-stage the website, and purge both 2026-10-28g and 2026-10-30 once live.
- Anita 2026-10-30: dragging as built is fine (no pivot); triangles good; all listed text acknowledged.
Otherwise nothing. Anita at 94% of the week's usage (2026-10-30): no new tasks until the week resets (~11-01/02).

## Waiting on Anita (asked in the final message of 2026-10-30)
- The website commit and push above.
- T-1869: a platform-end drag moves the whole station; pivot/stretch would need a WASM change. Keep?
- T-1871: drop triangles point downhill (toward their colour's level); flip to uphill?
- T-1868: a money/time divider on the two-row bar too (would move money to the clock's row)?
- Text shown for review: "There is already a station here.", "A station cannot be on a junction.", "The track here
  is too short for this platform." (agent's wording), "Ramp down to -2" (was "Tunnel portal down to -2"), the
  Australia credits line, 13 Australian zone names.

## Next (when the week allows)
1. T-1881 hillshade from a surface model shows dense centres as hills (Anita's Tokyo/Nagoya/Shanghai question).
2. T-1880 buses as feeders (Anita's question). T-1873 muting worm. T-1867 only if she still wants it after T-1874.
3. T-1849, T-1863, T-1864 demand speedups; a `?demandWorkers=3` browser run of T-1862's keep.
4. T-1875, T-1876-T-1879 (Australia follow-ups; T-1879 includes a CLAUDE.md line: treat data/terrain like data/packs
   while a gate or join runs), T-1882.
5. T-1794 train-count settle; Southeast Asia (T-1738) when cores are spare. Waiting on Anita: bus design (T-1445).

## Anita's calls today (in DECISIONS/notes)
- T-1859's straight station rule stays. T-1866: uneven sample homes, at most 100 at 30+ trips (built, T-1874).
- Clone of "X clone 1" is "X clone 2". Text review answers applied (notes/T-1840-answers.md).
- Corridors back on per place (T-1848). Deleting is for good. Negative money red; hovers only for what isn't written.

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs; save a batch in notes/todo-<date>.md.
- Tokens tight this week (to ~11-01): Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Every demand task (loosely perf) has a "done when"; stop once met. Approximations welcome when invisible.
- Buses settled unless she asks. Thanks to agents in my own reply. Someday tasks are real open tasks.
