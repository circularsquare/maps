# Orchestrator handoff, trainworld

Orchestrator: session 8be9971b-5363-40be-bba3-d8c1eaa4ca85, name maps-04 [6bfb1b], since 2026-11-10 21:40
Updated: 2026-11-11 ~04:00 (clock times in this file are rough; trust git log)

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 = R2 2026-11-10d** (f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept for hotfixes until
  the next release). R2 purge due: 2026-11-07c and 11-08 from 11-11; 11-09, 09b, 09c a few days later; keep 11-10, 10b-10d.
- **Master = f6ed4814, pushed; 8800 = f6ed4814** (adds T-3159, T-2773, T-3129 notes after 1df60242) (the night's batch: T-3137 C3+C4 single track done,
  T-3149 clicks, T-3157/3158 look, T-3147/3131 bus stops, T-3151/3152/3154 TODO items, T-3153/3126 track mode, T-2766
  stop clicks; tsc clean, cargo 501/681, smoke 27 PASS on dev).
- **v0.2.2 HELD** until she has tested 8800 (single track is now fully built). Then: translation batch for new and stale
  keys (19 stale per language at 04:00; as agent l10nbatch: 3 groups x 3 passes, ~15 min), snapbuild, smoke8800 light+dark,
  `deploy.py --dry-run --version <date>`, set `V0_2_2_AT` in updates.ts, publish and website push on her word. Changelog:
  propose lines for single track joins (side picked by the pointer, way from the joins), T-3141, the click standard,
  high-speed quad, typed train counts; show her.
- **Server**: live shonei-server 80e3692 (community round 2, preview, tester Anita). 0e7dd77 (titles, profanity screen,
  Biggest points-only) committed, not pushed or deployed: deploy once she is up (approved: "yes we can deploy in morning
  when ready").

## In flight (maps-04's agents, main tree)
- phonemem DONE, committed: T-2802 (WASM heap step 8 MB on small devices, outlines.json parsed once on the page and copied to workers; first line 691->600 MB, peak 948->825 on an emulated Pixel 7); rest filed T-3160 [P3]. Was:
  file T-3160.
- riderdip DONE, committed f6ed4814 with stopfast: T-2773 verified closed by the 11-07 hold + T-2793; check
  t2773-pending-dip.mjs rewritten; her Japan save is C:\Users\anita\Downloads\trainsworld-tokyo-day-414.save.
- roadsnap DONE: T-3129 design in notes/T-3129.md (one road snap with profiles; rail on the centre line, motorway..tertiary; curves slowed not refused; crossing rule under 20 deg; stages ~4.5 h). Her questions: discount along roads (no, lean), centre line or beside (centre, lean); drafts "Snap to roads", "Hold Ctrl to draw by hand", "The roads are too tight here. Add a point on the way."
- stopfast DONE, committed f6ed4814: T-3159 marker guessed at the click (2-5 ms), demand layer skips rows-only worlds; list
  row not faster (the inspector re-render, ~20 ms; a possible next step).

## Next
Anita 01:15: "lets do all the self contained tasks. if you find yourself finishing with those too, feel free to decide on
more to tackle yourself". Candidates: T-3138 left-hand running (her TODO has a design idea: side switcher; ask first), T-3156 aws guard false positive.
T-3150 rays parked (P3) until she reproduces it. Next free id **T-3161**.

## Waiting on Anita (all in the morning message's For you)
- Server deploy 0e7dd77 (approved for the morning).
- Strings: "Sandbox games aren't counted", "Title", "Please pick a different title."; "Stations" (complex list header);
  lanes4's "Set by the track it joins", "Use automatic way", "Run trains the way the joined track gives" (suggest dropping
  the last); clicks' 5 help strings ("Right click or hold a route point", "Hold a route point", "Loop the route there, or
  delete the last point", "Click a route point", "Delete it, or save the route at the last one").
- Calls: train counts already over the held-out limit, clamp down (my lean) or leave (built); same-named complex stations
  told apart only by chips; rail right click on a stop with no loop keeps "Can't loop back here" or goes silent; allow
  doubling up an existing high-speed double into quad (lean yes).
- Achievement thresholds for countries with no save were guessed (Spain, Britain, Italy, Germany, Indonesia, Mexico,
  Brazil, Australia, Turkey); she raised KR/TW to 10% and CN to 2%. Show them as a list, not the code file.
- `notes/v0.2-post.md` is hers to reword and post. To discuss later (her word, not filed): satellite map.

## Her calls 11-10/11 (for the record; detail in DECISIONS and ARCHIVE)
Single track: two-way by default, margin 240 s; a terminus branch stays two-way with its flat badge. Double track is
lanes in one piece, never deletable by halves; question every "double = one object" shortcut. Flyover ramps 4%. Dead ends
and through stations turn alike; hints say only "build a terminal" (30 tph). Margins: trams ~40/h, terminals 30/h,
junction moves 120 s. Community v1 = share, share link, open a copy (on by default), country leaderboards; copies named
"<title> - <name>"; Biggest points-only with the sandbox note. City circles only for whole loops. Catalan added. Click
standard as notes/T-3149.md "Proposal". Train count capped at what runs; typed up to 1000. Track widths 0.71x / 1.24x.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- **No v0.2.2 publish until she has tested 8800** ("the last few times around there were things that i caught").
- Cores: Anita went to sleep 01:25 11-11 ("feel free to use high cores"): ledger cap 15; back to 14 (or her word) when she is up.
- **Usage**: after the free reset, Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not
  tokens.
- Watch whether agents use wiki-first reading well.
