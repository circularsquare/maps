# Orchestrator handoff, trainworld

Orchestrator: session 7c3c98d5-ca3b-4748-b736-bac9826c72c1, name maps-a7, since 2026-11-02
Previous (nothing in flight, no wait needed; it is not in ListAgents): session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, handed over 2026-11-02
Updated: 2026-11-02 (batch filed, first five agents launched)

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

## In flight (maps-a7, started 2026-11-02; her batch is filed as T-1892 to T-1907, verbatim in notes/todo-2026-11-02.md)
Agents (Opus except ui-text; she said 11-02 skew to Opus, Sonnet for routine). Each has a reserved id range for new tasks
(next free id: see the last agent line).
- ui-text T-1892/T-1893 DONE 11-02, uncommitted in the main tree (archived; pictures artifacts/trainworld/t1893/; smoke not
  run; strings shown to Anita 11-02). Plan: commit t1894 + ui-text + t1928 together once t1928 finishes (shared docs,
  DECISIONS, SPEC; t1905's attractions files stay out), then smoke and rebuild 8800.
- Anita 11-02 on prices: round numbers, +1/-1 = 1.0x (sent to t1928: 2.0/1.6/1.0/0.3/1.0/1.5/2.0, base $75M; surcharges
  +0.5/+1.2/+3/+2.5); upkeep 20k/30k; do road crossings (T-1929) to make cheap ground pay in cities.
- t1894 DONE 11-02, uncommitted in the main tree (archived; pictures artifacts/trainworld/t1894/; smoke not run). Commit
  together with ui-text's T-1893 (both edit LineInspector.tsx). Its 3 hover strings shown to Anita 11-02.
- ab4a79199ce67b330 t1895 (Opus), tree .tw-trees/t1895, port 6130: T-1895 placement geometry. ids T-1916-1919.
- a173638cfe272a6eb t1902 (Opus), tree .tw-trees/t1902, port 6140: T-1902 trip explorer. ids T-1920-1923.
- t1900 and t1897 designs DONE 11-02. Anita's answers 11-02: express loses time (T-1924, with T-1925 spread departures,
  my call); passing tracks not yet (T-1927 to Someday); no pie, crowding strip in the dot's hover with green for no delay
  (T-1926, after T-1893/T-1894 land); cost changes "all reversible, fine either way" so all built as proposed, water
  limit 30 km (my call, Hangzhou Bay fits), motorways = 3 crossings (my call).
- ad0bb7a5986007fb7 t1924 (Opus), tree .tw-trees/t1924, port 6160: T-1924 + T-1925. ids T-1936-1939.
- a6b3e64630cf425c7 t1928 (Sonnet), main tree incl. Rust, port 6170: T-1928, T-1930, T-1931. ids T-1940-1943.
  T-1929 (road crossings) waits for T-1895's landing. Next free id: T-1944.
- t1905 DONE 11-02, uncommitted (attractions_json.py share curve s/(1+s/0.05), attractions.json rebuilt; include in the
  round commit after smoke). Proposed T-1932 (attraction staff no social pull), T-1933 (airport jobs on terminals, US
  rebuild), T-1934 (parking cap 50->40, two constants + market packs + ceiling rerun), T-1935 (China attractions x0.20):
  asked Anita 11-02. Her answers 11-02: build T-1932, rescale China's attractions per country (T-1935; per-country
  attraction factors fine, "attractions are bespoke anyways"), no parking change (T-1934 abandoned), rebuilds later
  (T-1933 waits). t1905 resumed for these (a9c1e7af6b8dcf76b).
- Also 11-02 (ui-text follow-up DONE, uncommitted, strings shown): ui-text resumed (profit hover "Today is a weekend" etc.; F2 hides the day/night animation; rolling counts
  labelled "today" but hovers "in the past 24 hours"). t1897 resumed for a brief real-world upkeep-by-level report
  (she read tunnels cost a lot to maintain, viaducts not); the 20k/30k upkeep build stands until she sees it.
  Report in (notes/T-1897.md 4): proposes ground 20k, viaduct 25k, cut and cover 35k, bored 40k; asked her 11-02. First answers her "how did Disneyland get 10x" (city-wide
  share spread over few known attractions?). Done when named cases are within 2x of published (her 11-02 call). ids T-1932-1935.Not yet started (briefs still to write; verbatim items are in the notes file): T-1896 gap highlight, T-1897 costs (proposal to
her on realism), T-1898 blueprint selection, T-1899 stop logic, T-1903 demand-bubble total, T-1904 speed, T-1905 destination
sanity (needs a done-when), T-1906 money rules, T-1907 map drawing. T-1896/1897/1898/1899 touch the same drawing/sim files as
T-1895: wait for its landing or give them their own trees.

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
