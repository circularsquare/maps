# Orchestrator handoff, trainworld

Orchestrator: session 8be9971b-5363-40be-bba3-d8c1eaa4ca85, name maps-04 [6bfb1b], since 2026-11-10 21:40
Updated: 2026-11-11 01:00 (earlier clock times ran ~40 min ahead; trust git log)

Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- **Live = v0.2.1 = R2 2026-11-10d** (f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept for hotfixes until
  the next release). R2 purge due: 2026-11-07c and 11-08 from 11-11; 11-09, 09b, 09c a few days later; keep 11-10, 10b-10d.
- **Master = 7bda06d3, pushed; 8800 = 7bda06d3** (lanes C1+C2 ed4f66fa and the hover follow-up; cargo 498/678, smoke 27 PASS).
- **v0.2.2 HELD** until single track is fully fixed and she has tested 8800. Then: translation batch for new and stale keys
  (as agent l10nbatch: 3 groups x 3 passes, ~15 min), snapbuild, smoke8800 light+dark, `deploy.py --dry-run --version <date>`,
  set `V0_2_2_AT` in updates.ts, publish and website push on her word. Changelog: maybe a line for T-3141 and for single
  track joins once C3 lands.
- **Server**: live shonei-server 80e3692 (community round 2, preview, tester Anita). 0e7dd77 (titles, profanity screen,
  Biggest points-only) committed, not pushed or deployed; the game side is on master, so titles do nothing until it is.
- Uncommitted: rulescleanup's doc changes (CLAUDE.md files, docs/orchestrator.md, world-run.md, zones.md, ui-panels.md
  "Live feel", wiki/RULES.md, notes/for-anita.md + its archive copy, TASKS T-3156, takeover/handoff skills). Commit them.

## In flight (maps-04's agents, main tree)
- lanes3 (Opus, aa68773ad4e68d9e2, Vite 6197): T-3137 C3, the app side of single joins (`set_route_join`, ends from
  `join_edges()`, ghost chevrons/badge; check t3137-single-join.mjs). Then C4 (inspector switch shows the way).
- clicks (Sonnet, ab9dfd00711fbdab7, Vite 6199): T-3149, the accepted click remapping (notes/T-3149.md "Proposal") and a
  touch button for every important right click.
- rays (Opus, a50b3a5f155fbf579, Vite 6198): T-3150 rays glitch after deleting a parallel line; may file T-3155.
- todo3 (Sonnet, a87f149b37ea7d959, Vite 6200): T-3151 orange issues button, T-3152 typed train count, T-3154 complex lists
  its stations with terminal buttons.
- lookfix (Sonnet, ace9c2afad233d9c3, Vite 6201): T-3157 height colours blend across ramps, T-3158 uneven single-leg widths.
- busstops2 (Sonnet, a365400196914902e, Vite 6202): T-3147 one-way rings in the bus line view and stop hover, T-3131 hand
  stops on both passes of an out-and-back route.

## Next
Anita 01:15: "lets do all the self contained tasks. if you find yourself finishing with those too, feel free to decide on
more to tackle yourself". After lanes3: T-3153 delete mode independent of track mode (tools.ts), T-3126 HSR quad, tram
street trio T-2997/2998/3000. Any time: T-2766 stop click wait on big networks (Opus), T-2417 diagram stretch, a TASKS.md
stale sweep (406 open lines). Not started: T-3129 snap to road (design first), T-3138 [P3] left-hand running (app only).
Next free id **T-3159**. maps-c4's scratchpad (`...\6f38e27c-5131-4ca5-81d2-160b4694f258\scratchpad\`) has
`archive_lines.py` (TASKS lines to ARCHIVE), `cargo2.sh`, `pages_verify.sh`.

## Waiting on Anita
- Server deploy 0e7dd77: approved for the morning ("yes we can deploy in morning when ready"); do it once she is up.
- Strings shown, answer pending: "Sandbox games aren't counted", "Title", "Please pick a different title."
- Achievement thresholds for countries with no save were guessed (Spain, Britain, Italy, Germany, Indonesia, Mexico,
  Brazil, Australia, Turkey); she raised KR/TW to 10% and CN to 2%. Show them as a list, not the code file.
- Translations of the v0.2 strings are first pass only. `notes/v0.2-post.md` is hers to reword and post.
- To discuss later (her word, not filed): satellite map, single track.

## Her calls 11-10/11 (for the record; detail in DECISIONS and ARCHIVE)
Single track: two-way by default, margin 240 s; a terminus branch stays two-way with its flat badge. Double track is
lanes in one piece, never deletable by halves; question every "double = one object" shortcut. Flyover ramps 4%. Dead ends
and through stations turn alike; hints say only "build a terminal" (30 tph). Margins: trams ~40/h, terminals 30/h,
junction moves 120 s. Community v1 = share, share link, open a copy (on by default), country leaderboards; copies named
"<title> - <name>"; Biggest points-only with the sandbox note. City circles only for whole loops. Catalan added.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- **No v0.2.2 publish until she has tested 8800** ("the last few times around there were things that i caught").
- **Keep 4 cores free** until she says or ~03:00 11-11: ledger cap 12, back to 14 after.
- **Usage**: after the free reset, Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not
  tokens.
- Watch whether agents use wiki-first reading well.
