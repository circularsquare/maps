# Orchestrator handoff, trainworld

Orchestrator: session 4a55787f-16a1-4c11-979b-2681f11405c8, name maps-b7 [ddb824], since 2026-11-11 08:39
Updated: 2026-11-11 ~11:00 local (maps-b7 past 500k tokens; ready for a fresh orchestrator; trust git log)

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 hotfix = R2 2026-11-11** (959e1f8f on tw/release-021, worktree `.tw-trees/hotfix`; T-3196 download
  stall + T-3192 Firefox varyings), published 11-11 on her word; website 9695c68; `--verify --page` all PASS.
- **Master = 39c430a1** (pushed to 1bbfad4b). **8800 = 1bbfad4b** (smoke PASSED). 39c430a1 adds T-3166 quad.
- Uncommitted in the main tree, by agent: buildtime (checks/snapbuild.cmd, tools/snap_wasm.py, app/vite.config.ts,
  notes/buildtime.md, docs/orchestrator.md trap line), traincap (sim capacity.rs, capacity/tests.rs, wasm/lines.rs,
  wasm/tests.rs for T-3201), branchhit (tools.ts, ToolBar.tsx, t3209-repro.mjs), mine (tools/r2_purge.py,
  .claude/settings.json allow rule, docs/orchestrator.md "Old R2 versions" line, TASKS lines T-3201..T-3209).
- R2: purging 2026-11-07c and 2026-11-08 via `python trainworld/tools/r2_purge.py 2026-11-07c 2026-11-08 --yes`
  (her OK; started ~10:53, slow, ~20k files each; check `rclone lsf r2:anitamaps/trainworld/ --dirs-only`). Next due:
  11-09, 09b, 09c in a few days. Only through r2_purge.py. Old tw-* check profiles in %TEMP%: deleted 76 (her OK).
- Landing while agents share the main tree: stage with keep_hunks (markers match added or removed lines), `git commit -m`
  without paths, smoke against the rebuilt 8800. Patch files: LF, no BOM (orchestrator.md Traps).

## v0.3 RELEASE (her word 11-11: "good enough state to push this update ... call it 0.3 since flyovers are pretty big
## i think, and introduction of community sharing")
1. Agent release (Sonnet) is turning the unpublished v0.2.2 entry in app/src/ui/updates.ts into v0.3 (keys
   updates.v03*, V0_3_AT), headliners flyovers and community sharing. Show her the lines; she rewords.
2. Code freeze: land what finishes first among T-3209 (P1, branchhit) and T-3201 (P1, traincap); the rest waits for
   v0.3.1. Land the buildtime snapbuild changes too (first real snapbuild with them is the test).
3. Translation batch for all new and stale keys (agent l10nbatch: 3 groups x 3 passes, ~15 min; i18n.mjs clean).
4. Commit, snapbuild, smoke8800.cmd light+dark, `deploy.py --dry-run --version 2026-11-11b` (or the date), set
   V0_3_AT, deploy `--smoked`, website add/commit/push, `--verify --page`. Publish and website push: her word given
   ("push this update"), confirm the moment with her.
5. **Community going public**: today the server runs SHONEI_COMMUNITY=preview (Community shows only on `?community`
   pages, writes from tester accounts). v0.3 "introduction of community sharing" means switching shonei-server to on:
   a server deploy (`.\deploy\deploy.ps1` in shonei-server, alone; then git push). Needs her explicit yes to that deploy,
   timed with the game publish. Check community.ts `available()` and the server's config for the flag first.

## Agents (resume by id via SendMessage; names here are the briefs' names)
- branchhit (Sonnet, 6112): T-3209 flyover leg shifts after a double joins it (first), then T-3204 end dots, T-3206
  quad hover "Quad track (two parallel double tracks)", T-3205 double-onto-single ghost.
- traincap (Sonnet, 6113): T-3201 one lone train gets "single track full" + red headway.
- complexlabel (Sonnet, 6115): T-3203 loop from either open end.
- passing (Opus, 6121): T-3202 two trains meet on single track; diagnose, stop and propose if big. Not for v0.3.
- flyover (Opus, 6111, ~450k tokens): reviewing 8 lanes-parity diffs in the small test games; re-baseline if explained.
- buildtime (Sonnet): snapbuild ideas 1+3 done (154 s on a scratch copy), adding idea 2 (hard-link packs; her OK).
- release (Sonnet): v0.3 changelog.
- Idle: roads, flatbadge, quad, stuck, varyings, ferrypath (done).

## Next after v0.3
- Queued: T-3160 [P3] memory leftovers, the stop list row's ~20 ms re-render (T-3159 follow-up, not filed).
- T-3156 aws guard false positive: do with her around. T-3138 left-hand running needs her design. T-3150 rays parked.
- Next free id **T-3210**.

## Waiting on Anita
- v0.3 changelog lines (once release reports); the community server switch.
- Flyover strings (new, landed): "Advanced", "Flyover target max speed", hover "Attempt to build new flyovers with
  curves wide enough to reach this speed. Faster flyovers take more room."; 80 km/h default (60-degree branch ~840 m).
- Achievement thresholds for countries with no save were guessed (Spain, Britain, Italy, Germany, Indonesia, Mexico,
  Brazil, Australia, Turkey); she raised KR/TW to 10% and CN to 2%. Show them as a list, not the code file.
- `notes/v0.2-post.md` is hers to reword and post. To discuss later (her word, not filed): satellite map.

## Her calls 11-10/11 (for the record; detail in DECISIONS and ARCHIVE)
Single track: two-way by default, margin 240 s; a terminus branch stays two-way with its flat badge. Double track is
lanes in one piece, never deletable by halves. Flyover ramps 4%. Margins: trams ~40/h, terminals 30/h, junction moves
120 s. Community v1 = share, share link, open a copy (on by default), country leaderboards; copies named
"<title> - <name>"; Biggest points-only with the sandbox note. Click standard as notes/T-3149.md "Proposal". Track
widths 0.71x / 1.24x. 11-11: train count cap = running + 1; HSR and tram quad yes (tram quad no street running);
buses and trams on any road in the data; snap to roads (no discount, below grade too); T-3161 option 1; T-3164 text ok;
text copied word for word from her message needs no review (CLAUDE.md); hotfix v0.2.1 done; rebuild speedups 1-3 yes;
R2 purges via the guarded script; "small changes all look fantastic", "target speed / auto flyovers feel really nice";
4-6 min rebuilds are "not so bad", only low-hanging fruit.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- Cores: ledger cap 14.
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
