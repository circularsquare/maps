# Orchestrator handoff, trainworld

Orchestrator: session 4a55787f-16a1-4c11-979b-2681f11405c8, name maps-b7 [ddb824], since 2026-11-11 08:39
Updated: 2026-11-11 08:50

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 = R2 2026-11-10d** (f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept for hotfixes until
  the next release). R2 purge due: 2026-11-07c and 11-08 from 11-11; 11-09, 09b, 09c a few days later; keep 11-10, 10b-10d.
- **Master = 8800 = f28a04b9** (plus handoff/TASKS commits). Overnight work (detail in ARCHIVE.md): single track finished
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
- flyover (Opus, worktree `.tw-trees/flyover`, Vite 6111): T-3170 HSR auto-flyovers, T-3169 Advanced row + flyover target
  speed, T-3168 both-sides flyover labels. Ids T-3174..3176. May get T-3166 next (same track bar).
- flatbadge (Opus, read-only, 6114): T-3161 design options for her in notes/T-3161.md, T-3164 feasibility. Ids T-3180/81.
- branchhit (Sonnet, main tree, 6112): T-3162 branch hit area = drawn track, T-3163 snap dots 50% bigger. Ids T-3177/78.
- traincap (Sonnet, main tree, 6113): T-3165 cap = running + 1, stored counts clamped on load. Id T-3179.
- complexlabel (Sonnet, main tree, 6115): T-3173 labels + Lines column, drop the "Run trains the way..." hover. Id T-3182.
- roads (Sonnet, main tree, 6116): T-3172 buses and trams on any road, answer her "easily changeable?". Ids T-3183..85.
  Gets T-3129 snap to roads next (approved, design notes/T-3129.md).

## Next
- Queued: T-3166 HSR/tram double-up to quad (after flyover, same track bar), T-3129 (roads agent), T-3171 [P3], T-3167
  [P3], T-3164 [P3] per flatbadge's answer, T-3160 [P3], the stop list row's ~20 ms re-render (T-3159 follow-up, not filed).
- T-3156 aws guard false positive: do with her around. T-3138 left-hand running needs her design. T-3150 rays parked.
- Next free id **T-3186**.

## Waiting on Anita
- Strings not yet answered: "Sandbox games aren't counted", "Title", "Please pick a different title."; "Stations" (complex
  list header); "Set by the track it joins", "Use automatic way"; help panel "Right click or hold a route point", "Hold a
  route point", "Loop the route there, or delete the last point", "Click a route point", "Delete it, or save the route at
  the last one"; snap-to-roads drafts "Snap to roads", "Hold Ctrl to draw by hand", "The roads are too tight here. Add a
  point on the way."
- Rail right click on a stop with no loop: keep "Can't loop back here" or silent.
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
"Run trains the way the joined track gives".

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- **No v0.2.2 publish until she has tested 8800** ("the last few times around there were things that i caught").
- Cores: ledger cap 14.
- **Usage**: after the free reset, Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not
  tokens.
- Watch whether agents use wiki-first reading well.
