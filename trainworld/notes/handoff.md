# Orchestrator handoff, trainworld

Orchestrator: session 4a55787f-16a1-4c11-979b-2681f11405c8, name maps-b7 [ddb824], since 2026-11-11 08:39
Updated: 2026-11-11 08:50

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 = R2 2026-11-10d** (f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept for hotfixes until
  the next release). R2 purge due: 2026-11-07c and 11-08 from 11-11; 11-09, 09b, 09c a few days later; keep 11-10, 10b-10d.
- **Master = 943805fd** (T-3162/3163, T-3165, T-3172, T-3173, T-3167 landed); master 0766283d (+ T-3186 ferry, T-3193
  icon, T-3129 snap to roads, flatbadge's three); 8800 = 84d97a7d (smoke PASSED); master =
  d3715a6c (+ T-3171), pushed. 8800 goes to the hotfix build next, then back to master. Patch files must be
  written LF without BOM (PowerShell Out-File breaks them; scratchpad\fixpatch.py converts). Landing pattern while the tree is shared: stage with keep_hunks,
  `git commit -m` without paths, smoke against the rebuilt 8800. Overnight work (detail in ARCHIVE.md): single track finished
  (T-3134..T-3137), click standard T-3149, HSR quad T-3126, typed and capped train counts T-3152, complex list T-3154,
  stop click speed T-2766/T-3159, phone memory T-2802 (rest T-3160).
- **Anita playtested 8800 the morning of 11-11**: notes filed by maps-04 as T-3161..T-3173 (top of "## Interface"); her
  screenshots artifacts/trainworld/t3162-anita-branch-hover.png, t3168-anita-flyover-speeds.png, t3174-anita-complex-label.png.
- **v0.2.2 HELD** until the P1s from her test land and she has looked again. Then: translation batch for new and stale keys
  (19+ stale per language; as agent l10nbatch: 3 groups x 3 passes, ~15 min), snapbuild, smoke8800 light+dark,
  `deploy.py --dry-run --version <date>`, set `V0_2_2_AT` in updates.ts, publish and website push on her word. Changelog:
  propose lines (single track joins and way, the click standard, high-speed quad, typed and capped train counts, T-3141
  lone branch = line end, plus this round's) and show her.
- **Server**: live shonei-server 0e7dd77 (community round 3), deployed and pushed 11-11 morning; community still PREVIEW.

## In flight (maps-b7, launched 08:45)
- **HOTFIX v0.2.1 (her go, 11-11)**: agent varyings porting T-3196 + T-3192 into `.tw-trees/hotfix` (uncommitted).
  Then: commit on tw/release-021, `snapbuild.cmd <that hash>` (8800 serves the hotfix build), smoke8800.cmd light+dark,
  `deploy.py --dry-run --version 2026-11-11`, then `--version 2026-11-11 --smoked`, website add/commit/push,
  `--verify --page`; then snapbuild master again. Upload ~700 MB may slow her Wi-Fi: warn her.
- flyover (Opus, worktree `.tw-trees/flyover`, Vite 6111): T-3170 HSR auto-flyovers, T-3169 Advanced row + flyover target
  speed, T-3168 both-sides flyover labels. Ids T-3174..3176. May get T-3166 next (same track bar).
- flatbadge: T-3164, T-3161, T-3189 LANDED 02314c0a, archived 0766283d, worktree removed. Idle.
- roads: T-3129 snap to roads DONE, committed b5dbfee2 (it also carried branchhit's draw.rs end_joins edit). Idle.
- flyover also has T-3188 (auto-flyovers for shallow splits, her min-angle ~15 degrees) after T-3169.
- ferrypath: T-3186 DONE, committed 5c3fd381 (not yet on 8800). Refusals all say "No water path found"; a separate
  "too long" line (legs over 400 km) was left, no id.
- branchhit (Sonnet, main tree, 6112): T-3162 branch hit area = drawn track, T-3163 snap dots 50% bigger. Ids T-3177/78.
- complexlabel: T-3173 (943805fd) and T-3193 (5defa5e3) DONE; idle. (5defa5e3 also carried roads' T-3129 ARCHIVE
  line, since it shared a hunk; T-3129's code is not committed yet.)
- varyings: T-3192 DONE, committed 84d97a7d: Firefox with fingerprinting resistance (LibreWolf, Mullvad, Tor)
  caps varyings at 8 and render sizes at 2048; the station/bridge programs needed 12. Live v0.2.1 affected. 84d97a7d
  also carried flyover's 4 DECISIONS lines and branchhit's T-3171 ARCHIVE line (shared hunks).
- stuck: T-3196 DONE, committed 19b4588f: two clock-worker downloads had no timeout, so a dropped road tile or
  water-mask request froze Construct, edits and the trains' timetable until a reload. Live v0.2.1 affected; fix is
  3 files, ~40 lines, cherry-picks onto tw/release-021 (passesHeld context differs). Hotfix asked of Anita; if yes,
  resume agent stuck to cherry-pick in .tw-trees/hotfix, then smoke and deploy on her word. Players' saves in
  scratchpad\stuck\kanto (delete when done).
- traincap: T-3167 DONE, committed 943805fd (idle; free for T-3160 or the stop row re-render).
- branchhit: T-3162/63 (fa7f29ea), T-3190/91 (7fce0b52), T-3171 (d3715a6c) DONE. Idle. 8800 rebuilding to 7fce0b52 + smoke.

## Next
- Queued: T-3166 HSR/tram double-up to quad (after flyover, same track bar), T-3129 (roads agent), T-3171 [P3], T-3167
  [P3], T-3164 [P3] per flatbadge's answer, T-3160 [P3], the stop list row's ~20 ms re-render (T-3159 follow-up, not filed).
- T-3156 aws guard false positive: do with her around. T-3138 left-hand running needs her design. T-3150 rays parked.
- Next free id **T-3199**.

## Waiting on Anita
(Hotfix: Anita said yes, "please send it", ~12:50 11-11.)
- Achievement thresholds for countries with no save were guessed (Spain, Britain, Italy, Germany, Indonesia, Mexico,
  Brazil, Australia, Turkey); she raised KR/TW to 10% and CN to 2%. Show them as a list, not the code file.
- `notes/v0.2-post.md` is hers to reword and post. To discuss later (her word, not filed): satellite map.

## Her calls 11-10/11 (for the record; detail in DECISIONS and ARCHIVE)
Single track: two-way by default, margin 240 s; a terminus branch stays two-way with its flat badge. Double track is
lanes in one piece, never deletable by halves; question every "double = one object" shortcut. Flyover ramps 4%. Dead ends
and through stations turn alike; hints say only "build a terminal" (30 tph). Margins: trams ~40/h, terminals 30/h,
junction moves 120 s. Community v1 = share, share link, open a copy (on by default), country leaderboards; copies named
"<title> - <name>"; Biggest points-only with the sandbox note. City circles only for whole loops. Catalan added. Click
standard as notes/T-3149.md "Proposal". Track widths 0.71x / 1.24x. One-way stops shown by the chevron only.
11-11: train count cap = running + 1 (one held train shows why), counts over it lowered; HSR and tram double-up to quad
yes (trams lose street running there); buses and trams on any road in the data, trains following roads may refuse more;
snap to roads approved, not high priority (no discount, centre line); same-named complex stations by chips is fine; drop
"Run trains the way the joined track gives". All strings from the 11-11 morning list acknowledged; "Can't loop back
here" stays (hover only offers the right click where a loop is possible). TODO short-term 11-11: bus-only roads question
sent to roads (T-3172); snap below grade added to T-3129; long ferry routes = T-3186.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- **No v0.2.2 publish until she has tested 8800** ("the last few times around there were things that i caught").
- Cores: ledger cap 14.
- **Usage**: after the free reset, Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not
  tokens.
- Watch whether agents use wiki-first reading well.
