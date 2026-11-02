# Orchestrator handoff, trainworld

Orchestrator: session 7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb, name maps-b2, since 2026-11-02 13:52
Updated: 2026-11-02 (maps-a7 wound down: nothing of its left uncommitted; t1738's runs are maps-b2's to adopt)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master: 32fa60e4 (build 786). 8800 at build 786, smoke 20/20 both shades (2026-11-02). Everything built from Anita's 11-02 batch
  and the country speedups is landed (ARCHIVE top). No worktrees open.
- 8800 rebuild: my script `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7c3c98d5-ca3b-4748-b736-bac9826c72c1\scratchpad\snapbuild.cmd <hash>`
  through `cmd /c` with output to snapbuild.log beside it; then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark. Never pass HEAD. My dev Vite: `node trainworld/checks/dev-nowatch.mjs 6105` (background task of this session; dies with it).
- **LIVE (2026-10-31): c7c5785a on R2 `trainworld/2026-10-30b`**. Not deployed since; publish only on Anita's word
  (`deploy.py --skip-data --version 2026-10-30b --smoked`, website commit/push on her word, verify).
- Commits: when two agents' edits share a file, I either wait until both report or stage by hunk (scratchpad
  stage_clock.py is an example). Tree landings rebase; a conflict stops in the tree (resolve there, `rebase
  --continue`, run the tests, land again).
- World: 1,925 shipped, 18 held, 5 parked (nic lez meo doo lel). Cores cap 10.
- Disk: ~33 GB free after deleting world_run's old join backups (Anita OK'd 11-02). sim/target/debug (4.3 GB) can go
  between rounds with no main-tree agent building.
- Build numbers (Anita 11-02): the start menu shows "build N" bottom left, N = `git rev-list --count HEAD` of the
  built commit (vite.config.ts `__TW_VERSION__`; "dev" on a dev server). Name "8800 is at build N" in every report
  about what landed.

## In flight (adopted by maps-b2 from maps-a7, 2026-11-02)
- master bd32f2f4 (handoff commit after 32fa60e4 = build 786). maps-a7 left nothing uncommitted; the dirty files in
  `git status` under trainworld/ are t1738's (and TODO.md, Anita's).
- a655261edf2119ea3 t1738 (Opus, a subagent of maps-a7's session; batches 20261102-134624 and -134628): Cambodia then
  Malaysia (+Brunei if simple) on the new playbook, first real use
  (doubles as T-1997: fixes the playbook/kit in place); one join if timing allows; CPU sampler during the joint solve
  for T-1981. Main tree; no other agent running, so it owns data/packs and data/terrain during its gate and join.
  Liveness, batch ids and restart commands go in notes/T-1738.md. ids T-2000-2004. Its report comes back to
  maps-a7's session (forwarded to maps-b2 if that session is still open) and is also written to notes/T-1738.md
  "Report 2026-11-02". If maps-a7's session closes first, the agent dies but its detached runs go on: adopt from the
  notes (world_run.py status, the batch logs) and finish the join and checks with a fresh agent.
- Launched by maps-b2 2026-11-02 ~14:10 (her second 11-02 batch, filed T-2005..T-2008, verbatim notes/todo-2026-11-02b.md;
  all main tree, Vite ports 6141-6143, no commits by agents, no data/ writes):
  a81b1e270ff7ae37a (Opus) T-2007 line-traffic track hover; ad5ab4d34a5e2198d (Opus) T-2006 elevation hover + T-2008
  tangent preview; a8b6c8e802cd68ba8 (Sonnet) T-2005 keep recent cities from memory dropping. Each reports every
  player-facing string old -> new; show them to Anita, check their pictures myself, then commit explicit paths.
  ad5ab4d3 reported 14:2x: T-2006 and T-2008 done in the tree, uncommitted (BuildTab.tsx, format.ts, map/tools.ts,
  docs/ui-building.md, ARCHIVE/TASKS); strings shown to Anita. Filed T-2009 (Quad upkeep doubles).
  a8b6c8e8 reported: T-2005 done in the tree, uncommitted (demandLod.ts, demandClient.ts, checks/t2005-recent-builds.mjs,
  smoke.mjs node list, docs, DECISIONS, ARCHIVE). Last built-in zone kept always, second last under 10 min
  (SECOND_BUILT_KEEP_MS); counts stations placed/moved and stops added, not track; not saved across loads.
  a81b1e27 reported (T-2007). All four (T-2005..2008) committed as 428b047e (code, checks, UI docs, ARCHIVE; the combined
  tree passed typecheck and smoke 21/21 on 6105). Left uncommitted on purpose because t1738's lines share them:
  TASKS.md, DECISIONS.md (my two T-2005/T-2007 lines are in DECISIONS' first hunk). 8800 NOT rebuilt (still build 786):
  wait until t1738's registry/packs writes end, since a build reads data/packs through the junction.
  Launched ~14:20: a0d8d36ca6921b208 (Opus) T-1896 gap highlight, port 6144, ids T-2010/2011; a828cd6fd8cd2fe5e (Opus)
  T-1907 map drawing items, port 6145, ids T-2012/2013; a17ddd1b8644b3a19 (Opus) T-1903 demand-bubble Total, port 6146,
  ids T-2014/2015. T-1904 (demand speed) not started: design first, after the country runs end (CPU).
  Her testing notes (chat, ~14:30; verbatim notes/todo-2026-11-02c.md) filed T-2016..T-2023. Launched: a0bce273efe559462
  (Opus, tree .tw-trees/t2018, port 6147, ids T-2028/2029) T-2018 express stops mid-route (sim; I run smoke in the tree and
  `agent-tree.mjs land t2018 "<sentence>"` then `remove`; it changes the sim so rebuild main WASM after landing);
  ae74e8e3f1c2bccd4 (Opus, port 6148, ids T-2024/2025) T-2016 bullets + T-2020 related-lines hovers + T-2023 "interval";
  af5bc392b935dcab3 (Sonnet, port 6149, ids T-2026/2027) T-2017 ground 0.5x + "crossings", T-2019 box selects whole
  segments, T-2021 fare multiplier 0.01 rounding. T-2022 (loans repay themselves; design, interacts with T-1906) waits
  for af5bc392's money files; then an Opus agent, decide and show after. Anita's answer (14:3x): equal daily
  instalments over a fixed term, independent of current income (in the TASKS line). Skin tone and hair by country
  (T-1907 item 1): go ahead without waiting, but show her the result.
  a17ddd1b reported ~15:00: T-1903 done in the tree, uncommitted (types.ts, ui.ts, commutersProtocol.ts, commuters.worker.ts,
  commuterView.ts, intercityView.ts [shared with T-1907], trip.ts, airportMarks.ts, DemandSelector.tsx, CommutersInspector.tsx,
  docs/ui-demand-views.md, ui-panels.md, DECISIONS/ARCHIVE/TASKS). Strings: button "Total"; hover "132k trips per day to and
  from here"; panel title "Trips to and from here"; "1,200 inter-city trips per day to and from Boston". Doubt: Total uses
  the destinations' density (homes look ~0.75 of Homes' radius). Shown to Anita in my next report. Commit the round when
  T-1907's agent (a828cd6f) has reported, since intercityView.ts is shared.
  a0d8d36c reported 14:31 (the "~15:0x" times above are wrong, clock reads 14:3x; t1738's join of KH,MY,BN started ~14:11 on 10 cores): T-1896 done in the MAIN tree, uncommitted, includes Rust (sim/src/track/wasm/gap.rs, gap_tests.rs,
  wasm.rs; track WASM rebuilt in the main tree) plus app (protocol.ts, clock.worker.ts, map/gapFlash.ts, stopMarkers.ts),
  checks/t1896-gap.mjs, docs. No strings. Commit the main tree (this and T-1903) BEFORE landing .tw-trees/t2018 (a landing
  fails on uncommitted main-tree edits to a file it touches; wasm.rs may overlap). Remaining agents: a828cd6f (T-1907),
  ae74e8e3 (T-2016/20/23), af5bc392 (T-2017/19/21), a0bce273 (T-2018 tree).
  ae74e8e3 reported 14:50: T-2016/T-2020/T-2023 done in the tree, uncommitted (many ui/inspector files, ui/style.css [shared:
  check the staged diff], game/groupLayout.ts, relatedLayout.ts, today.ts, checks/t1168-layout.mjs, docs/ui-inspectors.md).
  Strings shown to Anita in the report. Open doubt: her bullets (N1, 6a, 2M) came from another save, so every row of bullets
  was made even rather than one confirmed cause; she should retest.
  Anita's lost-save question (14:40): found. Her Tokyo/Seoul/Osaka/Fukuoka/Busan save is slot "Tokyo at 6:17 PM" on the LIVE
  site https://anita.garden/ in Chrome Profile 1 (65 lines, 935 stations, day 406, last saved Oct 31 12:54 local), not on
  localhost:8800 (its Tokyo slots have 44 lines, day 384, no Fukuoka or Busan). Copied to
  C:\Users\anita\Documents\trainworld\Tokyo Seoul Fukuoka Busan (from anita.garden).save (open with "Open a file").
  Read from a copy of her IndexedDB (scratchpad), her profile untouched.
  a828cd6f reported 15:00: T-1907 done in the tree, uncommitted (style.css [shared], trackHover.ts [T-2007's file too], pick.ts,
  main.ts, mapView.ts, faceTones.ts, faces.ts, docs, checks/t1792-idle-bullets.mjs, DECISIONS x4). Strings: only the riders row
  in other views. Skin/hair table shown to Anita (faces-by-country.png). Commit the whole UI round when af5bc392 (T-2017/19/21)
  reports; check staged diffs of style.css, trackHover.ts, TASKS/DECISIONS/ARCHIVE (t1738's hunks).
  a0bce273 reported 15:15: T-2018 done and smoked in .tw-trees/t2018 (sim: profile.rs, capacity.rs, params.rs, capacity/tests.rs;
  checks/t2018-trail.mjs; docs/track-and-service.md, code-map.md; DECISIONS/ARCHIVE in main). Cause: T-1924's FOLLOW_WAITS_MAX
  stands (up to 4 per shared stretch) sat in the express's timetable; now one lower speed over the stretch. Ready to LAND, but
  the main tree has af5bc392's uncommitted edits to sim/src/track/params.rs, docs/code-map.md and docs/track-and-service.md
  (the tree touches the same files), so: wait for af5bc392 (T-2017/19/21), commit the whole main tree, THEN
  `agent-tree.mjs land t2018 "<sentence>"`, `remove t2018`, rebuild main WASM, smoke, rebuild 8800.
  af5bc392 reported 15:40: T-2017/2019/2021 done, uncommitted (sim params.rs LEVEL_MULT 0.5 + tests, cost.rs; WASM rebuilt;
  Blueprint.tsx, BuildTab.tsx, trackTagRows.ts, blueprintBox.ts, income.ts, clock.worker.ts, checks t1898/t1928/t1929/t2021,
  SPEC.md, docs). Strings in my report to Anita. Doubt: "Crossings" heading sits near "Flat crossings" in the breakdown.
  Combined main tree passed typecheck and smoke 22/22 on 6105 at ~15:50, but NOT committed: af2463936931f1a65 (Opus,
  T-2031 right-drag rotation + Settings switch, port 6150, ids T-2032/2033) was already editing mapView.ts, SettingsTab.tsx,
  game/ui.ts, docs/ui-map.md, ui-panels.md (shared with T-1903/T-1907). Plan when it reports: smoke, commit the whole main tree
  (explicit paths, not t1738's: market-rows/realms/registry/zone-names json, country_preflight/zone_*.py, adding-a-country,
  data-register, zones.md, T-1738.md, borders.json, world_plan.*, setbuilt.py; not TODO.md), then land t2018, rebuild WASM,
  smoke, rebuild 8800 if t1738's join is done.
  Anita 15:3x: faces outside the table keep the old yellow (no change). Save sync question answered (T-2030 filed).
  Anita: tell her at once if an orchestrator starts on Sonnet (memory feedback_orchestrator_on_opus).
  16:2x: t1738 reported via maps-a7 (Cambodia 17/21 with 4 held, Malaysia 29/29, Brunei 1/1; world 1,972 shipped, 22 held,
  5 parked); its files committed as 8daed22e. 8800 rebuilt at 8daed22e = BUILD 789, smoke 22/22 light and dark (my
  snapbuild.cmd is in my scratchpad now, same as maps-a7's). a780a2f5 reported: T-2022 loans done, uncommitted (game/loan.ts,
  saveFile.ts, clock.worker.ts, protocol.ts, MoneyTab.tsx, MoneyHistory.tsx, checks/t2022-loan-schedule.mjs [not in smoke's
  list yet: add it], t1906 check, SPEC, ui-panels, code-map). 60-day annuity, refusal compares payment with profit.
  af5bc392 resumed 16:3x: back to "road crossings" everywhere (Anita) and tag chips wrap whole, never inside.
  Filed T-2036 (KL origin in Ampang), T-2037 (save grouping, design proposed, waits for Anita).
  af246393 reported 16:4x: T-2031 rotation done, uncommitted (mapView.ts, live.ts, DemandSelector.tsx NorthButton, ui.ts,
  SettingsTab.tsx, HelpPanel.tsx, gradeMarks.ts shader fix, docs). Perf: turned medians +4-7% (noise), p90 tail +22% at z11.5.
  Only af5bc392's road-crossings follow-up is outstanding; then smoke, commit all, land t2018, rebuild 8800.
  Next free task id: T-2038 (T-2028/2029 are a0bce273's, T-2030 on is free). Not yet started from the earlier batch: T-1896, T-1903, T-1904, T-1907.
- Then: Mexico and Turkey (T-1978 chunked gates before Mexico; T-1979 World Bank fetch once). T-1980 + AWS before
  Indonesia/India/Brazil/replans. A global-sources country may go to Sonnet once the playbook has run clean once.
- Not started from her 11-02 batch: T-1896 gap highlight, T-1903 demand-bubble total, T-1904 demand speed, T-1907 map
  drawing. Follow-ups filed: T-1916, 1917, 1920, 1936, 1937, 1944, 1948, 1949, 1966, 1967, 1970, 1988, 1992, 1994,
  1995, 1999 (planner floor exemptions for islands/exclaves before any replan). These UI tasks can run while the
  country run goes, as long as they do not write data/; app work is safe.
- Next free task id: T-2038.

## Open questions to Anita (asked, unanswered)
- (From maps-18, possibly stale since T-1928 reworked the price table): blueprint row hover wording (T-1890);
  check the current strings before asking again.
- Settings label "Lines side by side" vs "Draw lines side by side" (T-1886); Ctrl no-snap leaves the real-line ring on.

## Open problems
- T-1891: "Demand ran out of memory" once on a new NYC game on 8800; not reproducible by her or the agent; the label
  was wrong (any second worker death said "memory"); now "Demand stopped" with the panic in its hover. If it recurs,
  get that hover text or the console line `demand: worker N panicked`. notes/T-1891.md.
- T-1889: riverbanks priced as hills/mountains (East River at 42nd St) from T-1854's classes.

## Next (older list from maps-18; her 11-02 batch and countries come first)
1. T-1881 hillshade from a surface model shows dense centres as hills.
2. T-1880 buses as feeders (her question). T-1873 muting worm (now unblocked). T-1867 only if she wants it after T-1874.
3. T-1849, T-1863, T-1864 demand speedups; a `?demandWorkers=3` browser run of T-1862's keep.
4. T-1875, T-1876 to T-1879 (Australia follow-ups; T-1879 has a CLAUDE.md line for data/terrain during joins), T-1882.
5. T-1794 train-count settle. Waiting on Anita: bus design (T-1445).

## Lessons from this run
- Text reviews (Anita 11-02): never "less" for minus (now in her global style guide); no two-part hovers joined by a
  comma; "in past 24h" for rolling counts; she answers "text acknowledged" or names lines.
- Commit rounds when agents sharing files have all reported; a tree's landing fails on any uncommitted main-tree
  edit to a file it touches, so commit the main tree first.
- A new row or hover that has an existing counterpart (the local trip panel's Effective time for the market's) copies
  its pattern; Anita 11-02: "theres a clear reference here, i think its bad that we didnt use that". Put the reference in
  briefs for any panel work.
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
