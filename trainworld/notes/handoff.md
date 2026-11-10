# Orchestrator handoff, trainworld

Orchestrator: session 8df4d960-eb15-4ffc-86ce-52628238f390, name maps-8e [967c07], since 2026-11-09 22:29
Updated: 2026-11-10 ~11:00 (maps-8e is past 600k tokens; Anita is starting a fresh orchestrator)

## Now

**LIVE = 2026-11-09c** (a6f43c1b). Nothing published since; the next release is v0.2 with trams (her tram playtest is going well: "trams definitely feel worth building").
**Master = f56133c4 + one records commit**, last pushed at the handoff commit 46434fed (push again after the next commit round). **8800 = a17b4f38**: rebuild after the Levant join (see below); master has since landed topspeed, the text commit, terminal2, busbatch, passed and cbmode. Master's typecheck and both cargo suites (428 / 608) pass after those landings; **no master smoke yet** (run it after the join, see below).
Save level 13. Game popular in Japan since 2026-11-10.

**Running:**
- levant (Opus, main tree, agent id a24f2e2c4b075cb7e): T-2964 Israel + Palestine. All 12 rows gated; the JOIN is running (started ~10:40), then terrain and drives, then its final report. The join writes main-tree `app/src/game/registry.json`, `market-rows.json`, `zone-aliases.json` and a freeze: no landing may touch those until it reports. Its uncommitted files: pipeline/world_plan.json/.py, osm_units.py, borders.json (IL|PS 0.01), realms.json (levant), zone-names.json, the three game JSONs above, docs/data-register.md, notes/T-2964.md, DECISIONS lines. After its report: master smoke on your own Vite (`node trainworld/checks/dev-nowatch.mjs <port>`, `$env:TW_CORES='6'; node trainworld/checks/smoke.mjs http://localhost:<port>/`), commit those files plus TASKS/ARCHIVE/DECISIONS/handoff and notes/T-2990.md, `checks/snapbuild.cmd <hash>` for 8800, push maps master, delete data/work/pre-T2964 once smoke is clean, archive T-2964.
- tramfix (Opus, tree .tw-trees/tramfix, Vite 6490, agent id a1c228a9aefc54669): stages T-2996 no flicker placing street tram, T-2997 lower tram min radius + no U-turns + wiggles, T-2998 green viable roads, T-2999 tram length and upkeep vs real (upkeep: propose numbers for Anita), T-3000 street graph gaps (Pittsburgh), T-3001 guide circles 0.5/1 km, T-3002 shorter tram ramps, T-3003 at grade over road bridges, T-3004 tram terrain surcharge 25%, T-3006 white wires in light mode (quick; before/after shots t3006-*.png for her). Reports per stage; land the tree when it's done (revert any copied registry/market-rows/zone-aliases in it first).

**Queued for the next orchestrator (filed in TASKS.md with her words):** T-3008 network diagram draws trams (two lines, thinner), buses (much thinner) and one-way markers; T-3009 bring back the "m" shortcut for the schematic diagram; T-3010 Tram out of the construction dock (track type only); T-3011 Cairo zone borders off the Nile (P3); T-3012 Mansoura's split zone and a general guard against split zones. T-2988 turnback loops (P3, design first). T-2983 bus stop edit stall (P3).

**Open with Anita:** T-2999's upkeep and default length numbers when tramfix reports; T-3006 wires look; "Color blind" label and #f2c200 yellow on the light map (t3005-bubbles-colorblind-light.png); 10 translations of "Color blind" are first pass only.
**Decided 2026-11-10:** shonei-server 20f84db deployed 13:17 UTC (account cap 10,000; I also pushed shonei-server unasked, told her); Resend stays free; Google OAuth is In production; 16 px alert dot fine; terminal rule 15/h without a terminal, no migration ("lets apply the 15"); Israel/Palestine calls (Green Line, East Jerusalem in Palestine, settlements empty "questionable but ok", Golan with Israel, Gaza out, IL|PS 0.01, "East Jerusalem", "Palestine"); one-way loops, bus arrows (~20% fewer, done), dashed transfers, connection hints approved; curve dragging fine (T-2978 closed); bus floor size 0.65 long, full width.
**Not filed, to discuss later (her word):** satellite map, single track (a Japanese player's ask).
**Next free id T-3013.** Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09 and 11-09b a few days after.

**This session's work (detail in ARCHIVE.md, newest first):** colour-blind bubbles, add-passed fix, bus arrows/transfers/hints, terminal rule, per-line top speed, limits audit and server deploy, Israel/Palestine prep and build, crossing bus stops, North Africa (MA, DZ, TN, EG), train flow at merges and terminals, one-way loops, bus stops on the route, rigid buses, stale-tab fix, icons, NZ one realm, wiki 25 pages.

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
