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
- lanes3 DONE 02:30 (C3 built, uncommitted: tools.ts join picking, clock.worker routeArgs, network.ts joinedGhost/legShifts, renderer oneWayGhost, stateRows join; check t3137-single-join.mjs 32; fixed t3117 reading nonexistent fields). Was: T-3137 C3, the app side of single joins (`set_route_join`, ends from
  `join_edges()`, ghost chevrons/badge; check t3137-single-join.mjs). Then C4 (inspector switch shows the way).
- clicks (Sonnet, ab9dfd00711fbdab7, Vite 6199): T-3149, the accepted click remapping (notes/T-3149.md "Proposal") and a
  touch button for every important right click.
- rays DONE 02:00: T-3150 NOT reproduced in ~2 h (130 fuzz edits); suspects strokePost.ts patch mode + staged state, strokeLod; v0.2.1 same code. Needs the player's save: ask Anita. notes/T-3150.md, fuzz2.mjs in scratchpad/rays/.
- todo3 (Sonnet, a87f149b37ea7d959, Vite 6200): T-3151 orange issues button, T-3152 typed train count, T-3154 complex lists
  its stations with terminal buttons.
- lookfix (Sonnet, ace9c2afad233d9c3, Vite 6201): T-3157 height colours blend across ramps, T-3158 uneven single-leg widths.
- todo3 DONE 01:30, uncommitted: T-3151/3152/3154 (style.css .trouble-btn --warn, schedule.tsx typed count cap 200, StationInspector ComplexSummary + StationUpgrades UpgradeButton, ui-inspectors.md, ui-panels.md, checks/t3151-todo.mjs, string "Terminals"). Its smoke: t145-check FAIL (height alignment), likely lookfix or lanes3 WIP; asked lookfix. Commit with the next clean smoke.
- lookfix DONE 01:50, uncommitted: T-3157 `levelCode()` packs both ends' heights, shader blends; T-3158 lone line on single track keeps full width (network.ts 3 lines, beside lanes3's WIP); checks t3157-look.mjs, t145-check.mjs updated; smoke 27. Commit with lanes3 (network.ts shared) after one smoke.
- busstops2 (Sonnet, a365400196914902e, Vite 6202): T-3147 one-way rings in the bus line view and stop hover, T-3131 hand
  stops on both passes of an out-and-back route.

- lanes4 (Sonnet, aa2a25c5e22f52e5e, Vite 6203): T-3137 C4 inspector way switch (Auto / one way / pinned both = 3), joins follow a set way.
- trackmode (Sonnet, a74efe91a5894b55b, Vite 6204): T-3153 delete mode independent of track mode, T-3126 HSR quad.
- stopwait (Opus, ac519524ec0605a30, Vite 6205): T-2766 stop clicks without waiting for the whole state (NYC save); may file T-3159.
- todo3 DONE 02:45 (follow-ups): complex view = one row per station (chips under it, terminal button); train cap = floor(round trip / shortest headway), refuse + shake past it, typed above clamps; did NOT clamp stored counts down (T-752 held trains, would sell bought trains): ask Anita. String "Stations" (replaces "Terminals"); removed per-line row strings.
- busstops2 DONE 03:00: T-3147 bus rings half filled + hover chevron (busLayer.ts, halfDot.ts headingAtPlane sign fix also flips T-3142 ferry/bus marker halves N-S), T-3131 hand stops on both passes (bus.rs near_passes 25 m); check t3147-bus-oneway.mjs.
- clicks DONE 03:10: T-3149 (busDraw.ts, ferryDraw.ts lone-point delete, double click on empty ground only, ferry hold; 5 help strings reworded; check t3149-clicks.mjs). Rail right click on last stop keeps "Can't loop back here" (asked Anita).
- lanes4 DONE 03:30: T-3137 C4 (oneway.rs set_edge_oneway moves joins, edge_infer; ClockEdge setWay/autoWay; TrackInspector switch shows the way, off = 3 or 0, "Use automatic way" button; check t3137-way-switch.mjs). T-3137 + T-3135 archived. Strings to show: "Set by the track it joins", "Use automatic way", "Run trains the way the joined track gives".
- trackmode DONE 03:45: T-3153 snap() family filter only in the track tool, issues.ts onHsr; T-3126 HSR quad (draw.rs route_kind_of, ui.ts); check t3153-trackmode.mjs; cargo 501/681. Doubling up an existing HSR double still refused (could ask).
- COMMIT PLAN (03:10): tsc currently fails in lanes4's WIP (game/oneWay.ts), so no partial commit. Start no new main-tree agents; when lanes4, trackmode and stopwait report, tsc + one smoke, commit everything, snapbuild, push.

## Next
Anita 01:15: "lets do all the self contained tasks. if you find yourself finishing with those too, feel free to decide on
more to tackle yourself". After lanes3: T-3153 delete mode independent of track mode (tools.ts), T-3126 HSR quad. (Tram street trio T-2997/2998/3000 turned out done in d0601e8d; tasksweep closed 9 lines, notes/tasksweep.md.)
Any time: T-2766 stop click wait on big networks (Opus; after lanes3, clock worker files). T-2417 waits on Anita (fix landed).
Not started: T-3129 snap to road (design first), T-3138 [P3] left-hand running (app only).
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
- Cores: Anita went to sleep 01:25 11-11 ("feel free to use high cores"): ledger cap 15; back to 14 (or her word) when she is up.
- **Usage**: after the free reset, Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not
  tokens.
- Watch whether agents use wiki-first reading well.
