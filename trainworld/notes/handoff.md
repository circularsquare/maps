# Orchestrator handoff, trainworld

Orchestrator: session 6f38e27c-5131-4ca5-81d2-160b4694f258, name maps-c4 [3f93f3], since 2026-11-10 09:06
Updated: 2026-11-10 09:06 (clock check: the "~11:00" maps-8e wrote was an estimate; real time 09:06 EST)

## Now

**LIVE = 2026-11-09c** (a6f43c1b). Nothing published since; the next release is v0.2 with trams (her tram playtest is going well: "trams definitely feel worth building").
**Master = f56133c4 + one records commit**, last pushed at the handoff commit 46434fed (push again after the next commit round). **8800 = a17b4f38**: rebuild after the Levant join (see below); master has since landed topspeed, the text commit, terminal2, busbatch, passed and cbmode. Master's typecheck and both cargo suites (428 / 608) pass after those landings; **no master smoke yet** (run it after the join, see below).
Save level 13. Game popular in Japan since 2026-11-10.

**Running:**
- levant (Opus, main tree, agent id a24f2e2c4b075cb7e): T-2964 Israel + Palestine. All 12 rows gated; the JOIN is running (started ~10:40), then terrain and drives, then its final report. The join writes main-tree `app/src/game/registry.json`, `market-rows.json`, `zone-aliases.json` and a freeze: no landing may touch those until it reports. Its uncommitted files: pipeline/world_plan.json/.py, osm_units.py, borders.json (IL|PS 0.01), realms.json (levant), zone-names.json, the three game JSONs above, docs/data-register.md, notes/T-2964.md, DECISIONS lines. After its report: master smoke on your own Vite (`node trainworld/checks/dev-nowatch.mjs <port>`, `$env:TW_CORES='6'; node trainworld/checks/smoke.mjs http://localhost:<port>/`), commit those files plus TASKS/ARCHIVE/DECISIONS/handoff and notes/T-2990.md, `checks/snapbuild.cmd <hash>` for 8800, push maps master, delete data/work/pre-T2964 once smoke is clean, archive T-2964.
- tramfix (Opus, tree .tw-trees/tramfix, Vite 6490, agent id a1c228a9aefc54669): stages T-2996 no flicker placing street tram, T-2997 lower tram min radius + no U-turns + wiggles, T-2998 green viable roads, T-2999 tram length and upkeep vs real (upkeep: propose numbers for Anita), T-3000 street graph gaps (Pittsburgh), T-3001 guide circles 0.5/1 km, T-3002 shorter tram ramps, T-3003 at grade over road bridges, T-3004 tram terrain surcharge 25%, T-3006 white wires in light mode (quick; before/after shots t3006-*.png for her). Reports per stage; land the tree when it's done (revert any copied registry/market-rows/zone-aliases in it first).

**Anita's priority (2026-11-10 09:15):** "finish tram changes, then ill playtest, then we implement any feedback i give from that / any other loose ends, and try to publish trams some time in the next 3-5 hrs or so. not bad if we miss it ... this will be like a real post / i share all the new things". So: tramfix finishes (not stopped early), its tree and uinext land, 8800 for her playtest, her fixes, then publish v0.2 with release notes worth posting.

maps-8e lands levant's round itself (smoke, commit, 8800, push, archive, delete pre-T2964) and tells maps-c4. Tramfix stops at a clean stage and writes its state to notes/T-2996.md; its tree then passes to maps-c4 (brief a fresh agent from that note; land after levant).

**maps-c4's agents:**
- uinext done and landed (59f31536, plus a9b35a8f: diagram wires use the map's `tramWire`); tree removed; T-3008/3009/3010 archived. **8800 = a9b35a8f** (snapbuild-frozen, North Africa freeze 20261110-022552). Master smoke not run since d0601e8d. Buses in the diagram: filed T-3019 P3, "until someone asks" (Anita). Next free id T-3020.
- tramfix2 done, committed af01c29b, archived (T-3014 hint from z11, T-3015 station length per family, T-3017 no empty tags, T-3018 realSnap ghost). **8800 = af01c29b** (everything for trams so far). T-3016 faster ghost P3 for after the release. Bus hint given the same (2a72bd21, Anita "yes same"), not yet on 8800: batch with numfit. T-2988 answers relayed to loopend: one row per pass for now, T-3025 (one row per station + line diagram loop) filed P3; remove the first-stop express balloon in this stage. Her playtest calls: ghost and turns "feel better", white wires "lets keep it this way".
- maps-8e finished (levant e6a82657, T-2964 archived, pre-T2964 deleted); its push was refused, so the push is mine. **8800 = ee808214** (clean snapbuild). Cargo 430/610 pass on d4b98728; **pushed master d4072a7c** (~10:50). T-3013, T-2999 archived.
- selecthint done, committed 34ca5153, T-3026 archived. Resumed ~11:30 (agent id ac0ca328f168d14e7): the key shown in the theme colour like other shortcuts, no brackets. 8800 rebuild to 34ca5153 started ~11:20.
- TODO.md short-term filed 11:30: T-3027 slopes slow trains (P3, design first, not started), T-3028 bus fare multiple + T-3029 price-strict group proposal (agent busfare, Sonnet, Vite 6156, id a94114787799d6caa), T-3030 twstats changes + T-3031 twfeedback translation (agent twtools, Sonnet, id a53098e7b7acf93a3; may commit in shonei-server, never deploy). Next free id T-3032.
- numfit done, committed c94292aa, T-3020 archived (CSS only; billions in the Lines Money column still overflow ~11 px in de/ru/fr, asked). Next 8800 rebuild should include c94292aa.
- numfit follow-up committed 52a8801a (billions fit). **8800 = 52a8801a** (~12:00). twtools done, committed 4f97cddb, archived T-3030/T-3031; shonei-server 46876ae deployed 10:20 (15:20 UTC, Anita "ok yes, please deploy", service active) and pushed. Anita: German change, translate and stats "look good". Toasts 5 -> 10 s (d5e854cd, her ask), not yet on 8800: batch with tramyield/busfare.
- roadgaps (Sonnet, measurement only, Vite 6158, id a1b2faa2a167da820): T-3034 unjoined road ends 1-8 m in 12 cities, true-junction rate per band, proposed JUNCTION_GAP; notes/T-3034.md.
- Anita 10:40: land loopend now ("yea, lets try landing it"); tram ghost faster before the tram release ("its the first thing a new player experiences regarding trams so want it to feel nice", if not really hard): tramspeed told to build its top one or two fixes after profiling, off tools.ts/clock.worker.ts/protocol.ts/world.ts/sim until loopend lands.
- T-3035 (tram detour instead of a sharp turn; her shot t3035-anita-tram-detour.png) given to tramspeed after T-3016 (same code: BACK_DEG 135, flat TURN_M 200 in streetSnap.ts).
- tramspeed (Sonnet, profiling then build, Vite 6159, id aeeb035094a8c4779): T-3016 (raised P2) where the tram ghost's ~0.2 s goes vs the bus path; her idea of one search tree from the fixed start; notes/T-3016.md. Next free id T-3035.
- tramyield done, uncommitted, archived in ARCHIVE.md (capacity.rs, wasm/service.rs, wasm/tram_tests.rs, map/trainGuard.ts, docs/track-and-service.md which also holds busfare's fare edit, DECISIONS). Cargo 430/610 pass. Commit together with busfare's round (shared doc). Its smoke's one i18n FAIL is busfare's half-done strings. Diagram widths: trams 0.675 (52ae0bc5); rail back to 1.0 (d4072a7c, Anita: "we dont actually shift them around, thats why we cant increase width ... leave it at 1.0x"). Needs one more 8800 rebuild, batch with tramfix2's landing.
- splitzone stage 1 done (cause: metro `assign` never checks contiguity; 65 material split zones of 3,982; checks/split_zones.py, notes/T-3012.md). Stage 2 resumed ~10:35: planner fix + dry runs only; the 65-row mend (~2-3 h, 25 countries) waits until after the tram publish (my call, told Anita).
- (Clock: maps-c4's "~11:xx/12:xx" times in this file ran ahead; real time at this edit 10:38.)
- **loopend built T-3021/T-3022 in its tree, ready to land** (both cargo 1052 pass, check t3021-loopback.mjs 12 PASS, smoke 27 in tree; new WASM fns turn_end, open_loop; first-stop click makes only true rings now; loop closing hop capped at +300 s). Land after busfare's round is committed (both touch clock.worker.ts, protocol.ts, world.ts, track-and-service.md), then edit its ARCHIVE entry ("not landed yet"), rebuild main-tree WASM, 8800. Earlier: scope done (notes/T-2988.md; the sim already runs one-way balloons, no click makes one). Now building T-3021 (sim) + T-3022 (app): marker on the stop before the branch, "Click to loop back here". Lands after the tram publish. Revert copied game JSONs before landing. T-3023/T-3024 manual mode filed P3. Asked Anita: one row per pass (my default) vs folded; drop the first-stop express balloon?
- loopend (Opus, read-only scoping; Anita's manual-sequence idea relayed ~10:25: list every stop in order both ways, remove one direction's stop, or reorder into a loop, pathfinding each leg without reversals, Vite 6153, agent id a4d032183e3be9bd2): T-2988 end-of-line loops raised to P2 by Anita ("high return"; her Pittsburgh shot t2988-anita-pittsburgh-loop.png): can it be done by hand, design, cost; notes/T-2988.md.
- splitzone stage 2 done, committed 36442b17 (planner fix; 37 material splits left). **T-3033 rebuild of the 37 rows queued for after the publish** (Sonnet, from notes/T-3012.md; ~4 core-hours, 18 joins). Old line follows.
- splitzone (Sonnet, main tree, read-only on data, agent id a80fce3dffaea95b4): T-3012 diagnosis, `checks/split_zones.py`, a proposed fix with rebuild cost; notes/T-3012.md.

**Queued:** T-3011 Cairo zone borders off the Nile (P3, maybe explained by splitzone). T-2988 turnback loops (P3, design first). T-2983 bus stop edit stall (P3).

**Tramfix landed** in d0601e8d (by maps-8e; typecheck and cargo 430/610 pass, no master smoke yet). Not done, left past the release: T-3002 shorter ramps, T-3003 at grade over road bridges; loose ends: 2 chained refusals showed an empty tag, `realSnap.ts` blanks its ghost on every move (file these after her playtest). State in notes/T-2996.md.
- speedramp done, committed d4b98728 (T-3013 ramp 1 #8a2232 ... 320 #a03fc2; T-2999 street tram 0.6x). **Native cargo tests (both suites) still owed** once levant's join is done; then archive T-3013 and T-2999. My 8800 rebuild of d4b98728 collided with maps-8e's snapbuild of e6a82657 (levant, committed) and failed; asked maps-8e whether its build finished clean. Review folder cleared (Anita OK). Follow-ups sent ~10:15: speedramp (resumed) cools the 40 orange and 60 yellow ("a bit on the red side"); uinext (resumed, main tree now, Vite 6150) diagram trams 0.9x and rail 1.1x at draw time (Anita: "on trams, i meant in the network diagram").
**Open with Anita:** T-3006 wires look; 10 translations of "Color blind" are first pass only (label and yellow approved: "i think it looks okay to me").
**Stale lines to archive after maps-8e's round:** T-2925, T-2926, T-2927 (trams 1a-1c landed in 1fcae36e; TASKS still says taking/in tree tramkind, which is gone).
**Decided 2026-11-10:** shonei-server 20f84db deployed 13:17 UTC (account cap 10,000; I also pushed shonei-server unasked, told her); Resend stays free; Google OAuth is In production; 16 px alert dot fine; terminal rule 15/h without a terminal, no migration ("lets apply the 15"); Israel/Palestine calls (Green Line, East Jerusalem in Palestine, settlements empty "questionable but ok", Golan with Israel, Gaza out, IL|PS 0.01, "East Jerusalem", "Palestine"); one-way loops, bus arrows (~20% fewer, done), dashed transfers, connection hints approved; curve dragging fine (T-2978 closed); bus floor size 0.65 long, full width.
**Not filed, to discuss later (her word):** satellite map, single track (a Japanese player's ask).
Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09 and 11-09b a few days after.

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
