# Orchestrator handoff, trainworld

Orchestrator: session cd5c271d-f4a2-4f0f-8c6e-5249850f0046, name maps-b9 [971c94], since 2026-11-07 00:05
Updated: 2026-11-07 07:05 (Get-Date). The old maps-b9 [3e1b28] has finished (worldnext done); its tab may close.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **8ca54120** = 8800 **build 1349** (07:00; dev smoke 26/26, 8800 light/dark 26/26). Real save 12,521,264 rail
  trips. Has the 15 new countries (T-2695/6/7: AR CL CO PE UY EC, ZA LS SZ, AE SA QA BH KW OM; 237 rows; frozen
  data/frozen/20261107-060902), and tonight's work below. Backups pre-T2695 deleted.
- Cap 16 overnight (Anita ~00:25: "feel free to use all coores"); **set back to 10 when she is up** (`python cores.py cap 10`).
- LIVE = 821. Publish only on Anita's word. The upkeep tree still needs `agent-tree.mjs remove upkeep --force`.

## Landed overnight (maps-b9 [971c94]), all on master
- T-2630 checks pin Chrome to their core claim (checks/cores.mjs; TW_CORES=n, else a ledger claim).
- T-2703/T-2705/T-2704/T-2710 failed save, saves-list, demand-worker and clock-worker loads retry or stop with a reason.
- T-2682 edit note settles; T-2631 smoke fails on a refused market pack (26 checks; agent-tree warns on game json drift);
  T-2691 "Nn" = MapLibre AJAXError of a failed basemap tile, now a readable warning; T-2637 sheet-water check; T-2597 no
  change needed.
- T-2588/T-2718 NA save's demand cache hits on reload (saveNetKey without the eras tail; read-ahead drops stray zones).
- T-2699/T-2700/T-2716 (f9142979): sim heap = dlmalloc crate over its own grow-in-big-steps system side (T-2716: the
  --release abort was an LLVM fold of std dlmalloc's inlined size check, no corruption found); boarding step p50 on all
  Japan 311 -> ~4.2 ms; previews 2x not 4x pricing. T-2707/T-2722 boarding memory 1,056 -> 257 MB. All byte-identical.
- T-2681/T-2720 registry city names in ja/zh everywhere (app/src/i18n/place-names.json; rerun
  `node trainworld/tools/place_names.mjs --opencc <scratch>/opencc/node_modules/opencc-js` after any join; done after
  T-2695's, nothing new).
- T-2625 continent diagram condensation (NA networks 5.6% -> 13.9% of the sheet). Follow-up T-2724 waits on Anita's look.
- research/post-framing.md (Anita's ask, how to describe the optimizations in a post).

## In flight ([971c94]'s agents; SendMessage by id)
- **worldfix** ae45801a1a16dbcc8 (Opus, main tree): T-2702 coastal units missing from shipped countries (BR, MX, check
  all) + T-2712 Colombia GHSL blobs (DANE admin_totals; tiq pis lve held). Rows + one join; it messages before its
  join: **then hold pack writers**. Claims <= 8 cores. May file T-2725, T-2726. Commit its files when it reports.
- pairs finished T-2721 (committed after 8ca54120): app's marketRule.ts kept T-279's ferry (30 min, 35 km/h, $50)
  after T-2424 moved the sim to 90 min, 30 km/h, $80; check passes. Not on 8800 yet (rebuild with worldfix's batch).
- My Vite 6195 (task b8p351viu) for smokes.
- Next unassigned id **T-2728**.

## Noticed by capview, not filed yet (worth small tasks)
- A transfer-fare edit (setFareTransfer 0 -> 0.5) right after a stop-drop's long update never started a solve within 60 s.
- linesPending cleared on any demand update end (T-1733): T-2658 replaced the blanket clear; check it is gone.

## Waiting on Anita
- Two strings (shown 2026-11-07): clockClient.storageUnreadable "Could not read your saved game from browser storage
  ({why}). Nothing was written over it."; startMenu.savesUnreadable "Could not read your saved games from browser
  storage. Reload the page to try again."
- Which name she posts under ("trainsworld" or "anitabuilder") for research/post-framing.md.
- A look at artifacts/trainworld/review/diagram-na-condense.png (T-2625) before T-2724.
- T-2620 (keep old rates on track >5 km from stations): maps-93's question.
- T-2667 OSM attractions: yes, after release work.
- Strings shown, no answer = fine.

## Next
1. When worldfix reports: smoke, commit its files by path, rebuild 8800, smoke light/dark, delete its backups.
2. When pairs reports: land if a check fix; take a sim drive-time change to Anita first.
3. Localization left: pass 5 screenshots (checks/l10n-shots.mjs) per language; T-2643 swept files then their
   translation; T-2672 (l10nfix's leftovers); ja/zh for the new storage strings; OFFERED on Anita's release word.
   l10nfix's and ridersnow's pictures still to show her (artifacts/trainworld/l10nfix/, ridersnow/).
4. On Anita's fares answer: build T-2669/T-2670 (+ the multiples display) per notes/T-2668.md.
5. Queued: T-2713, T-2714 (census totals for the new countries), T-2698 (start cameras; registry.json, not during a
   join), T-2627, T-2638 (later), T-2593, T-2592, T-2701 (big: design to Anita first), T-2715 (leave unless seen).

## How to
- My scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\cd5c271d-f4a2-4f0f-8c6e-5249850f0046\scratchpad:
  snapbuild.cmd <hash> (PowerShell: `cmd /c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`),
  smoke8800.cmd (light+dark, TW_CORES=6, logs smoke-light/dark.log), opencc\ (opencc-js). The old orchestrator's scratch
  (...\b35f881f-...\scratchpad) has merge3json.py, resolve_*.py, translate/back/review briefs, the worldnext kit.
- Smoke dev: own Vite via `node trainworld/checks/dev-nowatch.mjs <port>` (restart after app or WASM changes), then
  `$env:TW_CORES='4'; node trainworld\checks\smoke.mjs http://localhost:<port>/` (~3.5 min).
- Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`. On a rebase conflict
  resolve in the tree, `git add`, `git -c core.editor=true rebase --continue`, land again. Commit the main tree's
  uncommitted edits in files the landing touches first.
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first.
- Publish: see trainworld/docs (freeze.py, deploy.py); push only on Anita's word.

## Lessons from this run
- Two sessions can share a name (both maps-b9): message by ref.
- Agents sharing the main tree work well when each brief lists the others' uncommitted files; commit app work in one
  smoked batch once the agents editing app/ have reported.
- A "false" debug-build abort can hide in the allocator: T-2716's fold. inline(never) on GlobalAlloc methods.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack then skip the rebuild (touch them).
- maps-93's handoff times ran ~4 h ahead of the machine clock: always Get-Date.
- A smoke that times out on page.goto at 30 s is the busy machine, not the change: rerun with 4 cores.
- Agents that leave a background wait re-notify: TaskStop them after their report.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
