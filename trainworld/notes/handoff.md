# Orchestrator handoff, trainworld

Orchestrator: session 2ba1fd73-4c91-45ba-b838-fe8f908d4b21, name maps-18 [8ecfe1], since 2026-11-11 11:10
Updated: 2026-11-11 11:35 (v0.3 code frozen at bea8551a; translation review running; publish next)

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 hotfix = R2 2026-11-11** (959e1f8f on tw/release-021, worktree `.tw-trees/hotfix`), website 9695c68.
- **Master = bea8551a = the v0.3 code freeze** (T-3203, T-3209 + T-3204/5/6, T-3201 landed by maps-b7); origin at
  593cd568 (later commits unpushed). **8800 = 52738393.**
- Uncommitted, maps-18's, for v0.3: ui/AccountPanel.tsx (Privacy · Terms also in the signed-in panel, her ask);
  changelog nerf lines red only on the new figure (her ask: `{figure}` in updates.v03Nerf* + three *Figure keys in
  en.json/context.json, updates.ts `nerfs` is line -> figure key, StartMenu.tsx, style.css `.changelog-list .nerf`);
  the tram line's parenthesis dropped (her word); the translation batch's messages/ and sources/ merges (and maybe a
  fr plural in i18n/glossary.json).
- Uncommitted, NOT for v0.3: passing's T-3202 (trainGuard.ts, clock.worker.ts, protocol.ts, network.ts, sim
  spacing.rs, wasm/service.rs, checks/t3202-single-meet.mjs, notes/T-3202.md). The main tree's WASM includes its
  guard_singles, so a dev smoke on the main tree is not master. Commit v0.3 files with explicit paths around it.
- Website tree (uncommitted): privacy.html and terms.html Community text (notes/T-2887.md drafts), dated 11 November;
  she approved it as is; push with the game publish.
- R2: next purge due 11-09, 09b, 09c in a few days, only through r2_purge.py.

## v0.3 RELEASE: DONE 2026-11-11 ~12:05
**Live = v0.3 = R2 2026-11-11b** (3a442a6e), website 9ca1187, `--verify --page` all PASS, live terms has Community.
**Community is ON** for everyone (shonei.env `SHONEI_COMMUNITY=1`, TESTERS line deleted, backup
`/opt/shonei/shonei.env.bak-2026-11-11`; config says trainworldCommunity true; 3 shares). Rollback: `preview`, restart.
She keeps one card per person in every Community view ("12 cards good").
Record of the release:
READY at 3a442a6e (11:55): translations reviewed and committed (b35d94b9); 9069c182 took T-3202's two network.ts
lines back out of master (they rode in with 091626fd and broke the build); V0_3_AT = 17:15Z (f2a3a263); privacy and
terms Community text now in app/public/*.html too (deploy copies dist over website/trainsworld, so app/public is the
source; 3a442a6e). 8800 = 3a442a6e; smoke8800 light+dark PASSED on 9069c182; changelog checked
(artifacts/trainworld/review/v03-changelog.png). `deploy.py --dry-run --version 2026-11-11b` preflight ok.
Waiting on her yes, then: `deploy.py --version 2026-11-11b --smoked`, website add/commit/push, `--verify --page`,
then community on (5), push maps master.
5. Community on (T-2887.md "Launch steps" 8): on the box set `SHONEI_COMMUNITY=1`, delete the TESTERS line,
   `systemctl restart shonei`, after the website push; an ssh env edit, no deploy.ps1 (the live server already runs
   shonei-server 0e7dd77's routes: checked top/liked/country boards/copy). Check `/noritetsu/config` says
   `trainworldCommunity: true`. Rollback: `preview`, restart.

## Agents (resume by id via SendMessage)
- l10n pass 1 (Opus, done): A ja/zh-Hans/zh-TW/ko a0099a5560521fc8a, B es/pt/ca/fr a30677ac57ee90419, C de/tr/ru
  ab4bf0175c111884b. Scratch `scratchpad/l10n-<g>-p1/<lang>/` (keys.txt from scratchpad/batchkeys.py, doubts.md).
- l10n pass 2 (Sonnet, done): A a1faf102b42f35c31, B a5de61cc67a509d2b, C a8e067691f86b99d8.
- l10n pass 3+4 (Opus, running): A a096c1fd986c51dab, B a1437117ad2afc220 (also fixing tools.streetsTight in zh/es,
  pt gradeMarks.tramYields, fr "sauts-de-mouton" glossary plural), C a9aaeff42365127d7.
- maps-b7's agents all done and landed; maps-b7 has wound down.
- busstop (Sonnet, 6114, a76b5cf1a943ec24c): T-3210 bus stop one way on a two-way stretch + T-3211 "One way stop" hover.
- ubc (Opus, 6116, afbd5ad584e1de35f): T-3213 UBC demand diagnosis, stops at a proposal (notes/T-3213.md).
- Not started: T-3212 single-track stations drawn as a split (waits on T-3202's call, same drawing area).

## Next after v0.3
- **T-3202** (two trains meet on single track; notes/T-3202.md): passing built a draw-only guard, OFF by default,
  uncommitted. Cause: on a symmetric 2-station line both ends depart together. Options: (1) finish draw-only ~2-3 h
  (two trains side by side at a single-track stop, explain a 273 s lag, t1965/t1966/smoke with it on, default on;
  passing's lean); (2) sim timetable meet schedule, changes capacity (2 trains on 2 km 5.1 -> ~6.4/h): her call;
  (3) both. Ask her after the publish; brief a fresh agent from the notes.
- Queued: T-3160 [P3] memory leftovers, the stop list row's ~20 ms re-render (T-3159 follow-up, not filed).
- T-3156 aws guard false positive: do with her around. T-3138 left-hand running needs her design. T-3150 rays parked.
- Next free id **T-3214**.

## Waiting on Anita
- The publish yes (publish + website push + community on) at the moment it is ready.
- T-3202 option (after the publish).
- Flyover strings (landed): "Advanced", "Flyover target max speed", hover "Attempt to build new flyovers with
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
4-6 min rebuilds are "not so bad", only low-hanging fruit. v0.3 notes are her 16 lines (tram parenthesis dropped);
single track is not a nerf; only the nerf figures red; Privacy · Terms in the signed-in panel; privacy/terms text ok.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- Cores: ledger cap 14.
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
