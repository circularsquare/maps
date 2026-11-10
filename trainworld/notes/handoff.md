# Orchestrator handoff, trainworld

Orchestrator: session 8df4d960-eb15-4ffc-86ce-52628238f390, name maps-8e [967c07], since 2026-11-09 22:29
Updated: 2026-11-10 ~09:30 (context past 500k: a fresh orchestrator can take over any time)

## Now

**LIVE = 2026-11-09c** (a6f43c1b). Nothing published since; the next release is v0.2 with trams, after Anita's tram playtest (T-2928 balance).
**Master = e9de5abc, pushed up to 6c0e64a1** (push again soon). **8800 = a17b4f38** (everything through the bus floor size undo; missing: topspeed aaa4c1bf, the text commit e9de5abc). Rebuild 8800 after the Levant run (a snapbuild mid-run would read half-written packs), no smoke needed for 8800.
Save level is 13 (LOW1 one-way loops 11, LSP1 spare-time split 12, LSL1 line speed limit 13). Game popular in Japan since 2026-11-10 (~14k new browsers a day, Tokyo top start city).

**Running:**
- levant (Opus, main tree, agent id a24f2e2c4b075cb7e): T-2964 Israel + Palestine BUILD running (12 rows, Gaza `gaz` held; Anita's calls in notes/T-2964.md and DECISIONS). Batch ids, logs, restart in notes/T-2964.md. It messages when the last gate passes: **until then HOLD every tree landing** (gates read main-tree app/ live). Then join (~1 h), terrain, drives, final report; then commit its plan files (world_plan.json/.py, borders.json, realms.json levant, zone-names.json, registry/market-rows/zone-aliases), smoke, snapbuild, push. Backup data/work/pre-T2964/pre (993 MB): delete after a clean smoke.
- terminal2 (Opus, tree .tw-trees/terminal2, Vite 6460, id a01fc200378b2fe17): T-2986 trains use both platforms at a line's end only with the terminal upgrade (sim capacity + terminalBays.ts); measures riders lost on the real save; a free-upgrade migration is Anita's call.
- busbatch (Opus, tree .tw-trees/busbatch, Vite 6480, id adc2573cf2e30dad9): T-2991 one-way arrows (bus both-side arrowheads, train white chevrons), T-2992 bus-bus transfers within ~60-80 m, T-2993 connection highlights while drawing a bus route. Stages report one by one.
- tramfix (Opus, tree .tw-trees/tramfix, Vite 6490, id a1c228a9aefc54669): T-2996 no flicker placing street tram, T-2997 lower tram min radius + snap rounds tight corners, T-2998 green viable roads, T-2999 tram length and upkeep vs real (upkeep: propose only).
Landing: revert any copied registry.json / market-rows.json / zone-aliases.json / realms.json in a tree before `agent-tree.mjs land`; one master smoke after a batch; then commit records, archive lines, snapbuild.

**Open with Anita:**
- Deploy shonei-server 20f84db (account cap 2,000 -> 10,000, password 5,000, alert 5,000; sign-ups 300/h, 10/IP/h). Built, tested, committed, not pushed or deployed. MUST go out before about Nov 12 (1,035 accounts, 400-700 a day). Needs her explicit go.
- Google OAuth "Publishing status" check (https://console.cloud.google.com/auth/audience?project=noritetsu).
- 16 px alert dot pale at DPR 1 (her look).
- Tram playtest (what to test given to her 2026-11-10).
- Decided 2026-11-10: Resend stays free (confirm mails just stop at the cap); R2 billing fine; text adjustments approved and landed; Israel/Palestine calls (Green Line, East Jerusalem in Palestine, settlements empty "questionable but ok", Golan with Israel, Gaza out, IL|PS 0.01 "pretty closed", names "East Jerusalem" and "Palestine"); curve dragging fine as shipped (T-2978 closed); bus floor size back to 0.65 long, full width.
- To discuss later, not filed (her word): satellite map, single track (a Japanese player's ask).

**Filed, open (selected):** T-2988 turnback loops (P3, design later, her words in TASKS), T-2983 bus stop edit stall (P3), T-2984 Gaza pack-pair border factors (needed only if Gaza is built), T-2985 Israel CBS 2024 update, T-2966-T-2968 data P3s, T-2969 Kharga oasis, T-2928 tram balance, T-2929 tram stage 3. Next free id T-3000.
**Due:** purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09 and 11-09b a few days after.

**Earlier this session (detail in ARCHIVE.md, newest first):** North Africa (MA, DZ, TN, EG; 185 rows), trams landed (maps-47), stale-tab fix and keep-previous-build assets, bus/ferry loops and one-way loops, bus stops on the route + crossing stops, rigid buses, train flow at merges and terminals (T-2974 stages 1-3), icons, NZ one realm, wiki 25 pages, per-line top speed, limits audit T-2990.

## How to
- Scripts in trainworld/checks/: `snapbuild.cmd <hash>`, `snapbuild-frozen.cmd <hash> <freeze>` (packs from a freeze while a
  run owns data/packs), `smoke8800.cmd` (8800 light then dark, only before a publish; logs in artifacts/trainworld/
  smoke8800-*.log), `keep_hunks.py`, `mark_taking.py`. Run via PowerShell:
  `cmd /c "trainworld\checks\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`. opencc-js is in the scratch folder
  ...\cd5c271d-f4a2-4f0f-8c6e-5249850f0046\scratchpad\opencc\; the old orchestrator's scratch (...\b35f881f-...\scratchpad)
  has merge3json.py, resolve_*.py, translate/back/review briefs, the worldnext kit.
- Smoke dev: own Vite via `node trainworld/checks/dev-nowatch.mjs <port>` (restart after app or WASM changes), then
  `$env:TW_CORES='4'; node trainworld\checks\smoke.mjs http://localhost:<port>/` (~3 min).
- Publish: notes/T-924.md "Commands"; `venv\Scripts\python.exe trainworld\tools\deploy.py --dry-run --version X`, then
  `--version X --smoked` (~15 min upload), website `git add trainsworld/ trainworld/`, commit, push; Pages deploy can be
  slow (`gh run list --repo circularsquare/circularsquare.github.io`), then `--verify --page --version X`.
- Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`.
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first.

## Lessons from this run
- Never edit docs with PowerShell 5.1 Get-Content/Set-Content: it reads UTF-8 as cp1252 and writes a BOM, garbling every
  non-ASCII character (TASKS.md 2026-11-09 22:55, repaired with scratchpad unmojibake.py). Use the Edit tool.
- deploy.py's rclone (32 transfers) saturated Anita's Wi-Fi upload (16.6 Mbps) and slowed her game's tile and terrain
  downloads on 8800 (2026-11-07). Anita: no need to cap it, just confirm the cause (tokyoload is checking).
- Always Get-Date before writing a time: this session's own estimates ran ~2 h ahead by mid-morning.
- Two sessions can share a name (both maps-b9): message by ref.
- A world run's gates serve the main tree's app/ live: hold app/ writes (not just commits) during gates; commits,
  .tw-snap rebuilds and freeze-based publishes are safe.
- Agents sharing the main tree work well when each brief lists the others' files; commit app work in one smoked batch.
- T-2716: inline(never) on GlobalAlloc methods; a debug-build abort can be an LLVM fold, not corruption.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack skip the rebuild (touch them).
- Agents that leave a background wait re-notify: TaskStop them after their report.

## Anita's standing asks
- **Token-conservative (Anita 2026-11-07 ~13:00: "we're running close to weekly limit. lets be token conservative and
  lean to use sonnets going forward")**: new agents on Sonnet unless a task truly needs Opus; fewer, smaller agents;
  let the running Opus agents (worldfix, sparse, tokyoload) finish rather than restart them.
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
