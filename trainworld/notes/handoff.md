# Orchestrator handoff, trainworld

Orchestrator: session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, since 2026-10-30 11:22
Updated: 2026-10-31 (three agents on Anita's 10-31 batch)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master c5a11543 (T-1884/T-1885 phone dock, after T-1883 and T-1854). 8800 at c5a11543, smoke 20/20 both shades, 1,988 terrain files in its dist,
  real save 11,392,296 rail trips a day (same as before T-1854).
- master 695750aa (10-31 batch + T-1884/5). 8800 at it, smoke 20/20 both shades. Website re-staged from it on
  2026-10-30b's data. T-1890 additive surcharges landed (157c2a6a), 8800 and website re-staged from it; waits for
  Anita's commit and push. HOLD it: T-1891 "Demand ran out of memory" on a new NYC game with one line (agent t1891,
  Opus) is a release blocker; told Anita not to push until it is understood. Next free id T-1892.
- **LIVE: 2026-10-30b** (pushed by Anita 2026-10-30 ~18:00; verify --page all PASS after Pages finished; her first
  verify ran before the Pages build ended). R2 still holds 2026-10-28g and 2026-10-30: her purge printed rclone's
  harmless "Failed to read versioning status ... AccessDenied" and the folder was still there after; my purge was
  refused by the classifier (Cloud Storage Mass Delete). She reruns both purges and lets them finish.
- **Deploy 2026-10-30b supersedes 2026-10-30** (= 5f232d9b: T-1854 terrain, T-1883 two-row bar, the Cities World-row
  spacing fix). Data on R2 `trainworld/2026-10-30b` verified; website re-staged (git add done, page names 30b). Anita
  runs the website commit and push, `deploy.py --verify --page --version 2026-10-30b`, then purges BOTH
  `trainworld/2026-10-28g` and `trainworld/2026-10-30`. The lines below about 2026-10-30 are history.
- (history) **Deploy 2026-10-30 was half done** (= ba582afb, with Australia, before T-1854): data on R2 `trainworld/2026-10-30`,
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
- Next free task id: T-1890. 2026-10-31 batch (T-1886 to T-1888) landed; T-1889 filed (riverbanks as hills). Old tree .tw-trees/t1734 kept (Anita). Localhost needs `?acct=live` once for accounts.

## In flight
- 2026-10-31 batch (notes/todo-2026-10-31.md): T-1886 side-by-side setting default off, city dots, line diagram
  (t1886, Opus) DONE uncommitted (label "Lines side by side", noun like its neighbours; ask Anita); T-1887 water tag + mountain price rule (t1887, Opus, sim/src/track); T-1888 holiday hover + yellow
  dot + shift-click headways (t1888, Sonnet) DONE, uncommitted; maps-18 shortened its click lines to "a going-out
  day, Feb 12-13" / "a coming-back day, Feb 17-18" (the holiday name is on the line above; the long form wrapped mid-range). Land all, rebuild 8800, look at the pictures; deploy with T-1884/5 on
  her word (website-only if no data changed: deploy.py --skip-data --version 2026-10-30b).
- T-1884/T-1885 landed (phone dock grab strip, height kept per browser, 58% max; the inspector beside every tab on a
  phone, Settings too). On 8800, not deployed: deploy only on Anita's word (data unchanged, so
  `deploy.py --skip-data --version 2026-10-30b` then a website commit/push by her).
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
