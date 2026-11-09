# Orchestrator handoff, trainworld

Orchestrator: session 3415d22e-522e-4ce0-ad47-92ca99b058bf, name maps-18 [b94cb3], since 2026-11-09 10:32
Previous: session a759be5a-7ca4-4099-ab0a-427565802ead, name maps-8a [827003], handed over 2026-11-09 10:32, finishing in-flight work
Updated: 2026-11-09 10:32

## State, 2026-11-09 10:31
**LIVE = 2026-11-09 (78ca3b4a), website 84912f4.** R2 holds 2026-11-07c, 11-08, 11-09: purge 07c and 11-08 from about 2026-11-11.
Master and 8800 = 4df8d347 (pushed; 8800 had no smoke, it was a build for Anita to look at). The next release needs one dev smoke, then 8800 light and dark (checks/smoke8800.cmd), then Anita's test and word.
Server shonei-server fc17e49 deployed (not pushed). Feedback threads ON; inbox localhost:8800/twfeedback/, key in her Documents/twfeedback-admin-key.txt. Analytics localhost:8800/twstats/ live.

**Next release so far, on master:** T-2793 fares fix, Discord icon (feedback row, periwinkle), 44 currencies with region default (VERSION 14), autosave wording, feedback report details, phone note once ever, tap for tooltips, no keyboard hints on phones.
**UNCOMMITTED in the main tree, ship with it (typecheck passes, no smoke yet):** the feedback panel T-2853/T-2865 (ui/FeedbackPanel.tsx, game/account.ts, ui/Dock.tsx, ui/AccountPanel.tsx, ui/style.css, i18n/*, app/public/privacy.html, notes/T-2853.md; T-2866 follow-ups); full currency prefix (A$ in the top bar's hover and the Money panel's Cash row: game/currency.ts, game/format.ts, ui/TopBar.tsx, ui/MoneyTab.tsx, ui/MoneyText.tsx); trainworld/tools/feedback-inbox/ (new, untracked); TASKS, ARCHIVE. Never TODO.md.
Update the changelog (ui/updates.ts) for that release and bump its date at publish.

**Running (maps-8a's children, report to it; if taking over, ask ListAgents or check these trees):**
- firefox2 a8e65c686982d91b6 (Opus, tree .tw-trees/firefox2, Vite 6130-39, T-2867): 32-bit browser out-of-memory crash, a Firefox player on 16 GB; assume 32-bit (Anita); detect it, fewer workers, then no intercity trips; replace "Play without demand" on the memory notice with "Play without intercity trips". Land after a look; WASM unchanged expected.
- calcspeed a57a2118ee6b06677 (Opus, tree .tw-trees/calcspeed, Vite 6175-79, T-2830): measure and plan faster trip calculation; report goes to Anita, she decides. No app changes.
- builtday abf45850c45f6397e (Opus, tree .tw-trees/builtday, Vite 6190-94, T-2871): record the build day of track and stations in the save (no UI). Save format and maybe WASM change: land, rebuild main WASM. T-2872 = playback later.

**Waiting on Anita:** T-2873 bus backtracking, start it? (reading the code: legs and stop snaps are chosen one at a time; fix is choosing stops' roads across the line with a reversing penalty; asked her for a bad spot). iPhone test of the tutorial (T-2794, Android fine).
**Queued, no decision needed:** T-2870 basemap loading indicator (small); T-2866 translation review of the feedback strings; T-2843 report's state machines A (done in T-2848), B demand health, C "who may write the save slot" (next, the one that could lose saves); T-2766, T-2757; T-2377 CLAUDE.md cut + moving the "User-facing text" guide out of her global CLAUDE.md + trimming the memory index (she approved all three); T-2854 wiki, design first; T-2851 more languages, order after pricing T-2841's tokens (checks/token_usage.py).
Later, her word first: Electron/desktop version (she'll bring it up); Discord is up (https://discord.gg/RemhwVcZFw).
Decided today: cores cap 14 (even while she plays); orchestrator pushes maps master a couple of times a session, publishing and website pushes still wait for her word; 8800 smoke only before a publish; Sonnet for known-file briefs. Cloud saves stay on the server until the 4 GB alert fires.
10:33: maps-8a handed over the landing; it starts nothing new and forwards firefox2, calcspeed and builtday's reports here. The uncommitted batch is maps-18's to smoke and commit. T-2873 bus routes: agent busroads (Opus, tree .tw-trees/busroads, Vite 6140-49); Anita: no reversing penalty.
Next free id T-2876 (T-2874 and T-2875 reserved for busroads).

## How to
- Scripts in trainworld/checks/: `snapbuild.cmd <hash>`, `snapbuild-frozen.cmd <hash> <freeze>` (packs from a freeze while a
  run owns data/packs), `smoke8800.cmd` (8800 light then dark, only before a publish; logs in artifacts/trainworld/
  smoke8800-*.log), `keep_hunks.py`, `mark_taking.py`. Run via PowerShell:
  `cmd /c "trainworld\checks\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`. opencc-js is in the scratch folder
  ...\cd5c271d-f4a2-4f0f-8c6e-5249850f0046\scratchpad\opencc\; the old orchestrator's scratch (...\b35f881f-...\scratchpad)
  has merge3json.py, resolve_*.py, translate/back/review briefs, the worldnext kit.
- Smoke dev: own Vite via `node trainworld/checks/dev-nowatch.mjs <port>` (restart after app or WASM changes), then
  `$env:TW_CORES='4'; node trainworld\checks\smoke.mjs http://localhost:<port>/` (~3 min).
- Publish: notes/T-924.md "Commands"; `venv\Scripts\python.exe trainworld\tools\deploy.py --dry-run --version X`, then
  `--version X --smoked` (~15 min upload), website `git add trainsworld/ trainworld/`, commit, push; Pages deploy can be
  slow (`gh run list --repo circularsquare/circularsquare.github.io`), then `--verify --page --version X`.
- Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`.
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first.

## Lessons from this run
- deploy.py's rclone (32 transfers) saturated Anita's Wi-Fi upload (16.6 Mbps) and slowed her game's tile and terrain
  downloads on 8800 (2026-11-07). Anita: no need to cap it, just confirm the cause (tokyoload is checking).
- Always Get-Date before writing a time: this session's own estimates ran ~2 h ahead by mid-morning.
- Two sessions can share a name (both maps-b9): message by ref.
- A world run's gates serve the main tree's app/ live: hold app/ writes (not just commits) during gates; commits,
  .tw-snap rebuilds and freeze-based publishes are safe.
- Agents sharing the main tree work well when each brief lists the others' files; commit app work in one smoked batch.
- T-2716: inline(never) on GlobalAlloc methods; a debug-build abort can be an LLVM fold, not corruption.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack skip the rebuild (touch them).
- Agents that leave a background wait re-notify: TaskStop them after their report.

## Anita's standing asks
- **Token-conservative (Anita 2026-11-07 ~13:00: "we're running close to weekly limit. lets be token conservative and
  lean to use sonnets going forward")**: new agents on Sonnet unless a task truly needs Opus; fewer, smaller agents;
  let the running Opus agents (worldfix, sparse, tokyoload) finish rather than restart them.
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
