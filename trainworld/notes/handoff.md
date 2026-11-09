# Orchestrator handoff, trainworld

Orchestrator: session 3415d22e-522e-4ce0-ad47-92ca99b058bf, name maps-18 [b94cb3], since 2026-11-09 10:32
Updated: 2026-11-09 11:03

## State, 2026-11-09 10:31
**LIVE = 2026-11-09b (0956310e, v0.1.1), website 1eb4a62, verified 11:3x.** R2 holds 2026-11-07c, 11-08, 11-09, 11-09b: purge 07c and 11-08 from about 2026-11-11, 11-09 a few days later. website's pages/projects/abouttrainsworld.md has someone's uncommitted edit: not this release's, left alone. Running besides marketwindow: busroads (T-2882, tree .tw-trees/busroads2).
Master and 8800 = 8497aa0f (not pushed; 8800 had no smoke, it was a build for Anita to look at). The next release needs one dev smoke, then 8800 light and dark (checks/smoke8800.cmd), then Anita's test and word.
Server shonei-server fc17e49 deployed (not pushed). Feedback threads ON; inbox localhost:8800/twfeedback/, key in her Documents/twfeedback-admin-key.txt. Analytics localhost:8800/twstats/ live.

**Next release so far, on master:** T-2793 fares fix, Discord icon (feedback row, periwinkle), 44 currencies with region default (VERSION 14), autosave wording, feedback report details, phone note once ever, tap for tooltips, no keyboard hints on phones.
**Committed as 8497aa0f after a dev smoke (27 pass), 8800 rebuild following, ship with it:** the feedback panel T-2853/T-2865 (ui/FeedbackPanel.tsx, game/account.ts, ui/Dock.tsx, ui/AccountPanel.tsx, ui/style.css, i18n/*, app/public/privacy.html, notes/T-2853.md; T-2866 follow-ups); full currency prefix (A$ in the top bar's hover and the Money panel's Cash row: game/currency.ts, game/format.ts, ui/TopBar.tsx, ui/MoneyTab.tsx, ui/MoneyText.tsx); trainworld/tools/feedback-inbox/ (new, untracked); TASKS, ARCHIVE. Never TODO.md.
Update the changelog (ui/updates.ts) for that release and bump its date at publish.

**Running:**
- marketwindow a1c707d708af4b2d4 (Opus, tree .tw-trees/marketwindow, Vite 6160-69, T-2878): share the market window's tables (A2), bit-exact; the only trip speed item she approved. T-2830 archived; T-2879 two rounds then three is an idea for later.

**Release 2026-11-09b in progress (11:13, Anita wants it out before her meeting about 12:00), changelog v0.1.1 (at 16:40Z in updates.ts, adjust if it goes out much later):** master smoke passed 27; smoke8800 passed light and dark on 628edf30 (13 each); 8800 = 0956310e (v0.1.1 wording only, no new smoke). She said go (buses "stupendous"). `deploy.py --version 2026-11-09b --smoked` running from 11:21, log artifacts/trainworld/deploy-2026-11-09b.log; then website commit and push, Pages, `--verify --page --version 2026-11-09b`. Feedback emails off on the server (SHONEI_FEEDBACK_MAIL unset; shonei-server 025ddd2): she checks the inbox. T-2878 is not in this release. shonei-server deployed 11:12 with her approval: /trainworld/stats answers 401 without the key.
**Anita out from about 12:00, 2026-11-09, asked for while out:** busstops (T-2885, Opus, tree .tw-trees/busstops, Vite 6150-59), community design done (T-2887, notes/T-2887.md): 4 questions asked of her, waiting; T-2878, T-2883, T-2885 (save level 7), T-2886, T-2888, T-2882, Portuguese (T-2889, 5.9M weighted tokens), hex terrain T-2890 on master c96f53c4 (smoke 27), 8800 = c96f53c4 and master pushed, none published, next release candidate. Nothing running. Asked Anita: stop spacing's speed gain, community's 4 questions, the other 4 languages. Next if she wants more: T-2843 C save slot. Then languages T-2889 (pt, ru, de, fr, ko) and hex terrain T-2890 if spare resources (hex: "just like show hex cells rather than square grid cells for what is considered hills or mountains when building track").
**Stopping point asked for by Anita (she wants to restart VS Code for Fable 5.2), 11-09 evening:** master 342cea70 smoke 27, 8800 being rebuilt to it, shonei-server c7e5885 deployed. Running only: worldterrain (AWS rebuild then per-block files, T-2901; R2 upload approved, goes with the game changes after). Russian held until after the restart, then de, fr, ko. Nothing published; next release carries everything since v0.1.1 plus T-2907 redirects and a changelog entry. Community T-2887 parked. Airports T-2896 queued under Waits on a rebuild. Next free id T-2911.
**Languages next (T-2889, Anita said yes, afternoon 11-09):** Turkish, then ru, de, fr, ko, one at a time, starting once purposepick lands so no string goes stale.
**Queued, no decision needed:** T-2870 basemap loading indicator (small); T-2866 translation review of the feedback strings; T-2843 report's state machines A (done in T-2848), B demand health, C "who may write the save slot" (next, the one that could lose saves); T-2766, T-2757; T-2377 CLAUDE.md cut + moving the "User-facing text" guide out of her global CLAUDE.md + trimming the memory index (she approved all three); T-2854 wiki, design first; T-2851 more languages, order after pricing T-2841's tokens (checks/token_usage.py).
Later, her word first: Electron/desktop version (she'll bring it up); Discord is up (https://discord.gg/RemhwVcZFw).
Decided today: cores cap 14 (even while she plays); orchestrator pushes maps master a couple of times a session, publishing and website pushes still wait for her word; 8800 smoke only before a publish; Sonnet for known-file briefs. Cloud saves stay on the server until the 4 GB alert fires.
10:33: maps-8a handed over the landing; it starts nothing new and forwards firefox2, calcspeed and builtday's reports here. The uncommitted batch is maps-18's to smoke and commit. T-2873 bus routes: agent busroads (Opus, tree .tw-trees/busroads, Vite 6140-49); Anita: no reversing penalty.
Next free id T-2881 (T-2880 reserved for marketwindow; T-2874 and T-2875 reserved for busroads). Master and 8800 go to 403016ca (feedback text, reply dot hover, basemap tag). Agent twstats (Sonnet, T-2876). T-2877 English station names: queued, design first. T-2867 landed as 26869c8d (not yet in 8800; run a master smoke before the next commit round); its 3 new strings had translation pass 1 only.

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
