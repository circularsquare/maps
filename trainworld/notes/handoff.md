# Orchestrator handoff, trainworld

Orchestrator: session 8be9971b-5363-40be-bba3-d8c1eaa4ca85, name maps-04 [6bfb1b], since 2026-11-10 21:40
Updated: 2026-11-11 morning (maps-04 past 500k tokens; ready for a fresh orchestrator; clock times are rough, trust git log)

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 = R2 2026-11-10d** (f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept for hotfixes until
  the next release). R2 purge due: 2026-11-07c and 11-08 from 11-11; 11-09, 09b, 09c a few days later; keep 11-10, 10b-10d.
- **Master = 8800 = f28a04b9, pushed.** The main tree is clean except Anita's TODO.md. No agents running, no worktrees but
  hotfix. Overnight work (detail in ARCHIVE.md): single track finished (T-3134..T-3137: lanes inside a piece, joins picked
  by the pointer, way inferred, inspector switch), click standard T-3149, look fixes T-3157/3158/3136, bus/ferry half dots
  and out-and-back stops T-3142/3147/3131, TODO items T-3151/3152/3154/3153, HSR quad T-3126, stop click speed
  T-2766/T-3159, rider dip verified T-2773, phone memory T-2802 (rest T-3160), rules cleanup.
- **Anita is testing 8800 now** (morning 11-11). Expect playtest notes.
- **v0.2.2 HELD** until her test passes and the strings below are settled. Then: translation batch for new and stale keys
  (19+ stale per language; as agent l10nbatch: 3 groups x 3 passes, ~15 min), snapbuild, smoke8800 light+dark,
  `deploy.py --dry-run --version <date>`, set `V0_2_2_AT` in updates.ts, publish and website push on her word. Changelog:
  propose lines (single track joins and way, the click standard, high-speed quad, typed and capped train counts, T-3141
  lone branch = line end) and show her.
- **Server**: live shonei-server 0e7dd77 (community round 3: titles, profanity screen, Biggest points-only), deployed and
  pushed 11-11 morning on her word; community still PREVIEW, tester Anita.

## Next
- Her playtest notes first. Then answers below unblock: the train-count clamp, HSR double-up to quad, T-3129 snap to roads
  (design notes/T-3129.md, ~4.5 h in 3 stages), string tweaks.
- Open without her: T-3156 aws guard false positive (a safety hook; do with her around), T-3160 [P3] memory leftovers,
  the stop list row's ~20 ms re-render (T-3159 follow-up, not filed). T-3138 left-hand running needs her design (her TODO
  has a side-switcher idea). T-3150 rays parked until she reproduces it. Next free id **T-3161**.

## Waiting on Anita (asked in the morning summary)
- Strings: "Sandbox games aren't counted", "Title", "Please pick a different title."; "Stations" (complex list header);
  "Set by the track it joins", "Use automatic way", "Run trains the way the joined track gives" (suggested dropping the
  last); help panel "Right click or hold a route point", "Hold a route point", "Loop the route there, or delete the last
  point", "Click a route point", "Delete it, or save the route at the last one"; snap-to-roads drafts "Snap to roads",
  "Hold Ctrl to draw by hand", "The roads are too tight here. Add a point on the way."
- Calls: train counts already over the held-out limit, clamp down (lean) or leave (built, T-752 reason); allow doubling an
  existing high-speed double into quad (lean yes); rail right click on a stop with no loop: keep "Can't loop back here" or
  silent; same-named complex stations told apart only by chips; snap to roads go-ahead (no discount, centre line).
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
One-way stops shown by the chevron only (no hover row).

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- **No v0.2.2 publish until she has tested 8800** ("the last few times around there were things that i caught").
- Cores: ledger cap 14.
- **Usage**: after the free reset, Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not
  tokens.
- Watch whether agents use wiki-first reading well.
