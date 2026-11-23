# trainworld — notes for agents

You are the owner and long-term maintainer of this codebase, and its design is up to you. Leave the code better
than you found it, fix the cause rather than the case in front of you, and keep it simple enough for the next agent
to change. What the game is, and the calls under "What waits for her word", stay Anita's.

Read this, then in `SPEC.md` the top and sections 1 and 2 (the game's two names, what **open** means, the
priorities, the core performance idea), the first lines of section 11 (what a save holds), and the sections your
task touches. Grep `TASKS.md` for your task id rather than reading it. Everything else on demand: for how a
mechanic works, the wiki first (`wiki/index.md`, then `wiki/read/<page>.md`, which has the figures filled in and
ends with the doc section and code that own the rule), then the docs. Where code lives: `docs/code-map.md`. The
orchestrator also reads `docs/orchestrator.md`. Where rules came from: `notes/rule-history.md`.

## Talking to Anita

- **Reports go to the orchestrator**, about 15 lines (each stays in its context for the session): what changed and
  where, check results, strings, anything for Anita marked as hers. Detail goes in notes/. She doesn't read code:
  quote the value or list, or name the place on 8800.
- **Estimates in hours of real agent time**, checked against similar past tasks; never "agent-days". Our guesses run
  pessimistic: don't call something impossible by a time. No deadlines unless she gives one; aim for the next release.
- **A time saving as a percent** of the whole wait it belongs to, seconds after ("about 10% off an edit's wait, 11 of
  105 s"); seconds alone don't say much (her 11-23).

## What waits for her word

Publishing the game and the website push; every server deploy (never overnight); adding a complicating factor to
demand; a choice that changes what the game is; a change to what players can build or how fast or far things go,
even when a player asked or a rail fact backs it (a request alone is no reason); anything the spec marks **open**; a
big task's design before it is built. Everything else you decide, record in DECISIONS.md if someone might revisit
it, and tell her after: rail facts, sensible defaults, visible calls, zone and region names. A slip gets a fix, not a
new standing process.

## Player text

- **Voice**: her global CLAUDE.md "User-facing text", the wiki included. "Trips" for what people want to take
  (demand); "rides" and "riders" only for what happens on the player's system. "Manually", never "by hand".
  "Segment" for a stretch of track, never "run". American "turn around", never "turn round" or "round" for "around".
- **Speed and waiting**, game text and changelog alike, one phrase per wait: a game or city "loads"; the map catching
  up to an edit "shows up" ("deleting a station shows up faster"); the numbers updating after a change are
  "calculating trips" ("trips are calculated 10x faster"), never "settle", "demand" or "rides"; memory "uses N MB
  less memory".
- Watch for an object or colour as the subject ("grey fills") and dropped words ("from street zooms"). A gesture with
  no visible control gets its instruction in a tag ("Click to close the loop").
- **Review**: list every in-game string you add or change, old -> new, exactly as shown. Landing doesn't wait; she
  answers "text acknowledged" or names lines. Propose wording; never ask her to write it. Not reviewed: the wiki,
  zone/region/realm names, docs, commit messages, and text copied word for word from her message or from a string
  already in the game (still listed, marked as reused). Changelog items are not asked about at landing: she reviews
  them on the board. A publish still waits for her OK on new player text (the `publish` skill).
- **Changelog**: a change a player would notice (a new control, a changed number, a fixed bug they'd have met) gets
  one item before you report; when in doubt, none. List it with your strings.
  `venv\Scripts\python.exe trainworld/tools/changelog.py add "<line>" --area <area> [--fix] --task T-#### --by <you>`
  - One plain line, lowercase: past tense for the change ("added ...", "fixed ..."), present for how things are now
    ("the game uses 500 MB less memory").
  - Area: track, demand, scheduling, fares, schematic (the map and schematic), community (accounts and sharing),
    performance; other if none fits. A fix takes its area plus `--fix`; `bug` only when no area fits.
  - If the unpublished changelog (`changelog.py list`) already has an item for what you changed, the one that added
    it included, reword it (`changelog.py reword <id> "<line>"`) instead of adding one.
  - No item for internals, tools and docs; a change too minor or fiddly to describe for a player; a change that is
    only a tooltip, an alert's wording or diagnostic text (that text is the whole change, and players can read it).

## Files

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the game is and how it works, **as of now**, in summary | agents, rewritten in place |
| `docs/` | The spec in detail, one file per area (indexes: `interface.md`, `demand.md`); `adding-a-country.md`, `world-run.md`, `data-register.md`, `code-map.md`, `code-tour.md` (the parts and how data moves) | agents |
| `wiki/` | Pages for a curious player; how to edit: `wiki/RULES.md` | agents |
| `TASKS.md` | Open work, by priority | agents |
| `LINES.md` | Anita's lines: the conceptual units tasks group into, by area | agents |
| `DECISIONS.md`, `ARCHIVE.md` | Dated choices and why; finished tasks, newest first. Older: `*-to-2026-10-19.md` | agents, add only |
| `notes/` | `T-###.md` task notes, `handoff.md` (orchestrator: plan, her words), `asks.json` (orchestrator, only through `tools/asks.py`) | agents |
| `checks/` | Standing checks, `smoke.mjs` first | agents |
| `research/` | Background reports with sources | agents |
| `TODO.md` | Anita's own list | Anita only; agents never edit it |
| `docs/demand-plain.md` | Demand in plain words, for Anita | whoever changes a demand rule it mentions |

## Doc rules

- **The spec says what is true now.** Rewrite the affected text; never append "Update:" or leave a section a later
  one contradicts. A section past ~150 lines moves to `docs/<area>.md`. A small UI tweak changes a doc only if a
  rule changed. A rule a wiki page states changes that page too.
- **Touch only the section your change touches.** Read sections, not whole files.
- **Docs hold current rules**: the rule and its deciding numbers. Measurements and how a figure was reached go in
  notes/; don't write down what the code says better.
- **Comments** only for a why the code doesn't show (a non-obvious choice, a unit, an invariant); never what the code
  does, never history.
- **DECISIONS.md and ARCHIVE.md are grepped, never read whole** (the `-to-2026-10-19` files too). A DECISIONS line:
  date, one or two sentences, task id. An ARCHIVE entry: id, title, date, outcome, notes link.
- **Tasks**: written to her and very short; leave out what you're unsure of. Examples: notes/T-3394.md.
  - Head line `- T-012 [P2] {L:slug} Title`: a 2-10 word title, no full stop or colon; `{L:<slug>}` from `LINES.md`
    right after the priority, plus `{waits:Anita}` or `{after:<task id or phrase>}` when it waits.
  - Fields, indented, one line each, in order: `  - Need from Anita:` (exactly when the head has `{waits:Anita}`:
    one sentence she can act on, with the recommended answer); `  - Context:` where things stand and why;
    `  - Goal:` what done is, with the deciding number (left out when the title says it; never "these items are
    done"); `  - Other:` optional, the notes link last.
  - Priorities: P1 now, P2 this milestone, P3 later, P4 lowest (set by her on the board). Someday tasks are P4 and
    don't count as open.
  - Use only the ids your brief gives you. A new task joins an existing line or adds one.
  - `venv\Scripts\python.exe trainworld/tools/tasks.py start <ids> --agent <name>` before starting, `done <id>
    --outcome "..."` when done (it also sets the line's `done:`), `check` for malformed lines.
  - Rewrite a task's fields, never append to the head line. When the main work is done but the task stays open, rewrite
    its title, Context and Goal to the part that's left.
- **Done** means it runs, the spec matches, any decision is recorded, and the task is archived. Then grep TASKS.md for
  the id, close a parent whose last stage this was, and name any other open task your work also did. New work found
  on the way is a new task.
- **Shared files**: re-read right before editing, small edits, never rewrite TASKS, DECISIONS or ARCHIVE wholesale.
  In a tree, TASKS.md, DECISIONS.md, ARCHIVE.md and LINES.md are never edited (a hook refuses): edit the main tree's,
  and run tasks.py from the maps root. Changelog items made in a tree are `<tree>-1`, `<tree>-2` until landing
  numbers them. Never edit docs with PowerShell 5.1 Get-Content/Set-Content (it garbles non-ASCII): use Edit.
- **notes/** is the agents': move, merge, delete. Check outputs and screenshots go in `../artifacts/trainworld/`. A
  notes link that doesn't resolve is in `notes/archive/<band>/`.
- **Data register**: a task that uses, finds or rules out a source updates that country's cell in
  `docs/data-register.md`.

## Working as an agent

- **Before starting**, `git status` and the "taking" marks; leave other agents' files alone.
- **Your own Vite**: `node trainworld/checks/dev-nowatch.mjs <port>` from the maps root (a free port in 6100-6199),
  restarted after app edits. 8800 is Anita's; no long sweeps against it. If your brief gives you a worktree, run
  everything from it.
- **Git worktrees** only through `checks/agent-tree.mjs` (new, remove): a tree links the shared `app/node_modules` and
  `data`, and `git worktree remove --force` or deleting the folder deletes through those links for everyone.
- **Playing the game**: probe with `node trainworld/checks/play.mjs` (how-to at its top). A standing check makes the
  same calls through `checks/lib/play.mjs`, which uses the real toolbar, pointer and panels. Build fixtures by the
  player's path where it matters; a new read goes in `checks/lib/reads.mjs`.
- **The real save** is `T-007-real-nyc.save`. `?new=1` never writes a slot.
- **Typecheck** with `npm run check --prefix app`. No full builds, no commits: landing is the orchestrator's commit
  (a brief that touches shonei-server may say to commit there; pushing and deploying stay the orchestrator's).
- **Rust**: keep the crate compiling between edits. After a sim/ change, `python trainworld/tools/agent_cargo.py
  test`; build your tree's WASM with `agent_cargo.py wasm`, not `npm run wasm` (the shipped build, for landing).
  Your tests link differently from the shipped build (DECISIONS 2026-11-18). Add a function to the WASM interface in
  Rust before any app code calls it.
- **CPU**: the cap is the ledger's (`python cores.py list`); a claim waits at the cap, never goes over it (`cores.py
  --help`). Playwright checks claim for themselves (`checks/cores.mjs`); claim by hand for the pipeline, Rust and
  scripts that launch their own browser. A page's demand workers take about a core each: cap them to your claim.
- **Data runs, packs, joins, AWS**: read the rules at the top of `docs/world-run.md` first.
- **Shell**: pin processes from PowerShell or a `.cmd`, never `cmd /c start ...
  /b` through Git Bash. Nothing may open a window on her screen: a process you start from code gets `windowsHide:
  true` (Node) or `creationflags=subprocess.CREATE_NO_WINDOW` (Python).
- **Advisor** (when your session has one): only for a big design decision nobody has made (her 11-23). Not when
  stuck, not to confirm a cause, not for routine steps, wording, small UI, or a choice already going to Anita. Say in
  your report whether you called it and what changed.
- **A refused permission check**: make the step safer and retry; don't wait for Anita.
- **Stop at a clean point** if the work outgrows one context: finish the stage, file the rest, hand back.
- **Waiting**: no `sleep` or polling loops of your own: `node trainworld/checks/wait.mjs` for a log or build (at most
  170 s a call; wait again if it times out), `play.mjs do 'wait.final()'` for the game. Never end your turn while
  something you need is running (a watch on a background process doesn't hold the turn open). Work that needs no
  result from you may run in the background, with at most one light check every 4-5 minutes. Final checks (smoke)
  write a log: wait on it and read its result before you report.
- **Testing, kept lean**: the checks near your change and one game run that shows it working (UI work: judge your own
  screenshots before reporting); breadth is the batch checker's. Smoke only when your brief asks or the change touches
  sim, saves, demand or money. A standing check only for a bug that really happened and could quietly come back, else
  a scratch script; first grep `checks/INDEX.md` for a check that already proves it.

## Game principles (Anita's)

- **The whole world**, a low-fidelity global build (tier 0) everywhere first; tailoring country by country later,
  popular countries first.
- **Demand is simple on purpose**: sensible and fun over fitting real cities. Targets are her two ceiling tests (rail
  share of rail + car: A at least 85%, B at least 50%; notes/T-636.md). Propose any complicating factor to her first.
  A demand rule change updates `docs/demand-worldwide.md`.
- **Every demand task has a "done when"**: numbers that end it, plus "would a player see something obviously wrong?".
  Once met, stop. Performance tasks get a looser one.
- **General sources**: global datasets or a method any census can feed; a one-city source only if very easy. Local
  demand factors (by country, region or city kind) are allowed where a worldwide rule misses a real gap and published
  figures show it; a place that already looks right (the US intercity) keeps the worldwide rule. Look outward for how
  other games and map makers solve it; write what you find in notes or research/.
- **Performance**: quantify impact, do the big ones first. An approximation is welcome when a player couldn't tell;
  say what the lie is in DECISIONS.md. Speed is measured on real saves only (hand-built saves are for correctness). A
  speedup that shifts riders measures the saving and the shift on a big real-world save (the real NYC save: total
  rail trips, the lines that move most) and names the network shape that would show the shift worst; she decides
  with those numbers (her 11-23). Speedups that keep every figure the same land as usual. Every speed, memory or
  loading change reports all of `node trainworld/tools/perf_bench.mjs`'s numbers (CPU, memory, bytes, solver time),
  not only the one it targets: one save for a small expected change, all three (nyc, tokyo, belgrade) for a risky one
  or a release; worse by more than 10% on any goes to her as a trade-off. How to read it: notes/T-4025.md.
- **Simple prices**: fares and costs the player reasons about follow a simple rule (a base plus per km).
- **Buses are settled**; no more bus tuning unless she asks.
- **Saves are hers**: keeping old ones loading is a nicety, no real migration effort.
- **UI principles** live in `docs/ui-panels.md` "Look" and "Hover tags".

## Build and run

`app/` (Vite, TypeScript, MapLibre, Preact), `sim/` (Rust to WASM), `pipeline/` (Python city packs,
`..\venv\Scripts\python.exe`). Build orders: `docs/code-map.md`.
