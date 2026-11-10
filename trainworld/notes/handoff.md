# Orchestrator handoff, trainworld

Orchestrator: session d455ee97-22bd-4dd5-a487-a2a4e1449db1, name maps-15 [bf4395], since 2026-11-10 13:48
Updated: 2026-11-10 14:45

## Now

**LIVE = 2026-11-10c** (built from 33a8ccec; website 4b39208 pushed with Anita's word 14:40; Pages verified 14:48, all PASS). Before
it: 10b (97c11df7), v0.2 = 2026-11-10. Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09, 11-09b,
11-09c a few days after; keep 11-10, 10b, 10c.
**Master = daac649f** (pushed 52fa18bf; push again soon). **8800 = 2cd98bb6** (exact mode, tram/transit words; not smoked).
Since 10c on master: T-3064, T-3067, T-3068 (tram/bus words, Fleet, "transit"), T-3074 (picker row), T-3023/3024 exact mode.
Website: Anita's `pages/projects/abouttrainsworld.md` edit is uncommitted and hers. The website push needs her word each
time (the classifier refuses it otherwise; she said yes for 10c).

**Next publish = v0.2.1** (Anita 15:00: "lets push next hotfix as v2.1", read as v0.2.1; R2 version 2026-11-10d): master +
T-3078 once fixed. Needs an `updates.ts` entry "v0.2.1" with notes (strings to Anita), smoke8800 light+dark, deploy, then
the website push (she has said yes to this one).
- oomfall (Sonnet, main tree, Vite 6198): T-3072 fall back to fewer workers on out-of-memory + refit memoryCheck estimate.

**Running:**
- hsrplat (Sonnet, id a571dac7d31b93d0b, main tree, Vite 6196): T-3078 [P1] stations over 4 cars refused on HSR. Lead:
  `tools.ts` `tramEdgeSet` filters `e.kind` truthy, HSR is TrackKind 1. Anita's rule: accept or refuse by the track kind
  at the spot, not the tool setting.
- lightload (Sonnet, id ac88d88c5e02c9f62, main tree, Vite 6197): T-3073 "Load without inter-city trips" at the memory
  warning (Anita approved).
- rebuild (Sonnet, id a2b5ad8262d71e795, tree `.tw-trees/rebuild`): T-3059 done; resumed to rebase onto master, add the
  T-3051 hint counting, re-check. Then I land it (`agent-tree.mjs land rebuild`, wasm rebuild comes with it).

**Open with Anita now:** a standalone (desktop) version "for performance": answered 15:00 that a plain wrapper gains
little; only worth it with shared memory between workers or native threads (weeks). She decides whether to scope it.

**Queued:** T-3070 non-home-based trips design agent once this round is done (Anita: "can wait then start a design
agent"). T-3063 bus loops partway along a route (manualstop landed; can start). T-3033 split zones rebuild + T-3077
precomputed inter-city figure (one data batch). T-3060 rebuild preview after T-3059. T-3069 trams on real rail + street-snap
icon. P3s: T-3042, T-3036, T-3027, T-3002/T-3003, T-3019, T-3011.
Next free id **T-3079**.

**Open with Anita:**
- National achievement thresholds for countries without a save were guessed (Spain, Britain, Italy, Germany, Indonesia,
  Mexico, Brazil, Australia, Turkey); she raised KR/TW to 10% and CN to 2%. `NATIONS` in app/src/game/achievements.ts.
- Translations of the day's new strings (hidden achievements, v0.2 notes, hint, bus fare) are first pass only.
- The post `notes/v0.2-post.md` is hers to reword and post; its v0.1.1 / quiet-update boundary is inferred.
- Wiki-first: trainworld/CLAUDE.md now sends agents to wiki/index.md first (her OK); watch whether agents use it well.
- Not filed, to discuss later (her word): satellite map, single track.

**Today's work (detail in ARCHIVE.md, newest first):** v0.2 notes, looped-line view and its follow-ups (T-3025,
T-3039-3041), hidden achievements and player shares (T-3037/3038), road gaps 3 m at ends (T-3034), faster tram ghost and
sharp turns (T-3016/3035), end-of-line loop (T-3021/3022), bus fare multiple (T-3028), trams yield at crossings (T-3032),
split zones planner fix (T-3012), twstats/twfeedback (T-3030/3031), delete hint (T-3026), long number words (T-3020),
speed colour ramp and tram upkeep (T-3013/2999), tram playtest fixes (T-3014-3018), M key, dock, diagram trams
(T-3008-3010), Israel/Palestine (maps-8e), tramfix (maps-8e).

## How to
- Scripts in trainworld/checks/: `snapbuild.cmd <hash>`, `snapbuild-frozen.cmd <hash> <freeze>` (packs from a freeze while a
  run owns data/packs), `smoke8800.cmd` (8800 light then dark, only before a publish; logs in artifacts/trainworld/
  smoke8800-*.log), `keep_hunks.py`, `mark_taking.py`. Run via PowerShell **from the maps root** (Set-Location first; a
  PowerShell call from another folder fails with "cannot find the path"):
  `cmd /c "trainworld\checks\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`. Never run two snapbuilds at once
  (they share .tw-snap; one collided 2026-11-10).
- maps-c4's scratch (`...\6f38e27c-5131-4ca5-81d2-160b4694f258\scratchpad\`) has `archive_lines.py` (move TASKS lines to
  ARCHIVE with a given entry), `cargo2.sh` (both cargo suites), `pages_verify.sh`.
- Smoke dev: own Vite via `node trainworld/checks/dev-nowatch.mjs <port>` (restart after app or WASM changes), then
  `$env:TW_CORES='4'; node trainworld\checks\smoke.mjs http://localhost:<port>/` (~3 min).
- Publish: notes/T-924.md "Commands"; `venv\Scripts\python.exe trainworld\tools\deploy.py --dry-run --version X`, then
  `--version X --smoked` (~15 min upload), website `git add trainsworld/ trainworld/`, commit, push; Pages deploy can be
  slow (`gh run list --repo circularsquare/circularsquare.github.io`), then `--verify --page --version X`. The in-game
  updates list is `app/src/ui/updates.ts` (set the entry's `at` at publish).
- shonei-server deploy: `.\deploy\deploy.ps1` from that repo, alone in its own PowerShell call (combined with a push it
  was refused); then `git push` separately. Only with Anita's word.
- Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`.
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first. Check `git diff` of shared docs before
  committing: an agent's in-progress paragraph rode along once (docs/ui-building.md, bf762960).

## Lessons from this run
- Never edit docs with PowerShell 5.1 Get-Content/Set-Content: it reads UTF-8 as cp1252 and writes a BOM, garbling every
  non-ASCII character. Use the Edit tool.
- deploy.py's rclone (32 transfers) can saturate Anita's Wi-Fi upload and slow her game on 8800.
- Always `date` before writing a time: both orchestrators' estimates ran 1-2 h ahead on 2026-11-10.
- Two sessions can share a name: message by ref.
- A world run's gates serve the main tree's app/ live: hold app/ writes during gates.
- Agents sharing the main tree work well when each brief lists the others' files; resume the same agent for follow-ups.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack skip the rebuild (touch them).
- A trial that changes Rust constants with nothing visible goes into a notes/ patch, not the tree (T-3029).
- An agent's estimate of a change (T-3034 +32 joins) can miss what the code really touches: recount before landing.

## Anita's standing asks
- **Usage this week (Anita 2026-11-10 11:50):** "at this rate we are gonna run out of usage fairly early in the week. thats
  okay, its good to front load work this week so that we can get this important update out some time today. after we get
  v0.2 out though we can return to leaning slightly to use sonnets, and prioritize the token-cheap compute-expensive approach
  if we have a choice. the remaining work is also just not super urgent i think." v0.2 is out: Sonnet by default, fewer
  agents, prefer approaches that spend compute rather than tokens.
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
