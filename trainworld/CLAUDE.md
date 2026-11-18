# trainworld — notes for agents

Read this, then `SPEC.md`. Grep `TASKS.md` for your task id rather than reading it. Everything else on demand:
for how a mechanic works, the wiki first (`wiki/index.md`, then `wiki/read/<page>.md`, which has the figures filled
in and ends with the doc section and code that own the rule), then the docs. Where code lives: `docs/code-map.md`.
The orchestrator also reads `docs/orchestrator.md`.

## Talking to Anita

- **Reports go to the orchestrator** and are short, about 15 lines (each report stays in the orchestrator's context
  for the rest of its session): what changed and where, check results, strings, anything for Anita (marked as hers).
  Detail goes in notes/.
- **She doesn't read the code.** Quote the value or list in the message, or name the place on 8800, never a file
  to open.
- **Estimates in hours of real agent time**, checked against similar past tasks (T-3137 C1, guessed at days, took
  about 45 minutes). Never "agent-days". Don't call something impossible by a time: our guesses run pessimistic, and
  work often lands sooner. No deadlines unless she gives one; aim for the next release and see (Anita, 2026-11-17).
- **Ask when the answer is needed next**, not ahead. Test items only once the change is on 8800.

## What waits for her word

Publishing the game and the website push; every server deploy (never overnight); adding a complicating factor to
demand; a choice that changes what the game is; a change to what players can build or how fast or far things go,
even when a player asked or a rail fact backs it (a request alone is no reason: 350 km/h high speed, Anita
2026-11-12); anything the spec marks **open**; a big task's design before it is built. Everything else you decide, record in DECISIONS.md if someone might revisit it, and tell her after: rail
facts, sensible defaults, visible calls, zone and region names. A slip gets a fix, not a new standing process.

## Player text

- **Voice**: the "User-facing text" section of her global CLAUDE.md, the wiki included. "Trips" for what people want
  to take (demand); "rides" and "riders" only for what happens on the player's system.
- **Speed and waiting** (Anita, 2026-11-17), in game text and changelog alike, one phrase per wait: opening a game or
  city "loads"; the map catching up to an edit "shows up" ("deleting a station shows up faster"); the numbers
  updating after a change are "calculating trips" ("trips are calculated 10x faster"), never "settle", "demand" or
  "rides" for that wait; memory "uses N MB less memory".
- **Review**: list every in-game string you add or change, old -> new, exactly as shown. Landing doesn't wait; she
  answers "text acknowledged" or names lines. Propose wording; never ask her to write it.
- **Changelog**: a change a player would notice (a new control, a changed number, a fixed bug they'd have met) gets
  one item before you report: `venv\Scripts\python.exe trainworld/tools/changelog.py add "<one plain line, lowercase;
  past tense for the change, present for how things are now: "added ...", "fixed ...", but "the game uses 500 MB
  less" (Anita, 2026-11-17)>" --area <area> [--fix] --task T-#### --by <you>` (area: track, demand, scheduling, fares, schematic for
  the map and schematic, community for accounts and sharing, performance; other if none fits; a bug fix takes its area
  plus `--fix`, and `bug` only when no area fits); list it with your strings. If the thing you changed is new in, or
  already listed in, the unpublished changelog (`changelog.py list`), reword that item (`changelog.py reword <id>
  "<line>"`) instead of adding one. Internals, tools and docs get none, and nor does a change too minor or too fiddly to describe
  for a player (Anita, 2026-11-17: undo after a load's cap lowering). Anita rewords
  or hides items on the board's Changelog view.
- **Not reviewed**: the wiki, zone/region/realm names, docs, commit messages, and text copied word for word from her
  own message (Anita, 2026-11-11); still list it in the report.
- American wording: "turn around", never the British "turn round" or "round" for "around" (Anita, 2026-11-12: "its a
  pet peeve of mine").
- Watch for an object or colour as the subject ("grey fills"), dropped words ("from street zooms"), and hovers that
  say what the player can see. A gesture with no visible control gets its instruction in a tag ("Click to close the
  loop"); ordinary controls get none.

## Files

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the game is and how it works, **as of now**, in summary | agents, rewritten in place |
| `docs/` | The spec in detail, one file per area (indexes: `interface.md`, `demand.md`); `adding-a-country.md`, `world-run.md`, `data-register.md`, `code-map.md` | agents |
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
  rule changed; the code is the reference for details. A rule a wiki page states changes that page too.
- **Touch only the section your change touches.** Read sections, not whole files.
- **Docs hold current rules**: the rule and its deciding numbers. Measurements and how a figure was reached go in
  notes/. Don't write down what the code says better and will change soon.
- **Comments** only for a why the code doesn't show (a non-obvious choice, a unit, an invariant). Never what the code
  does, never history.
- **DECISIONS.md and ARCHIVE.md are grepped, never read whole** (the `-to-2026-10-19` files too). A DECISIONS line is
  for a choice someone might revisit: date, one or two sentences, task id. An ARCHIVE entry: id, title, date,
  outcome, notes link.
- **Tasks**: a head line `- T-012 [P2] {L:slug} Title` (2-10 words, no full stop, no colon), then indented
  one-line fields in this order: `  - Need from Anita:` (exactly when the head has `{waits:Anita}`: one sentence she can
  act on, with the recommended answer if there is one), `  - Context:` (where things stand and why), `  - Goal:` (what
  done is, with the deciding number), `  - Other:` (optional; the notes link last). Very short, written to her; leave
  out what you're unsure of (Anita 2026-11-16). Examples: notes/T-3394.md. P1 now, P2 this milestone, P3 later, P4 lowest (set by her on the board);
  Someday tasks are P4 and don't count as open. Use only the ids your brief gives you. Mark a task taken
  before starting and archive it when done with `venv\Scripts\python.exe trainworld/tools/tasks.py start <ids> --agent <name>`
  / `done <id> --outcome "..."` (agents may use both; `check` lists malformed lines); update a task by rewriting its fields, never by
  appending to the head line.
  Every open task carries `{L:<slug>}` from `LINES.md` right after its priority, plus `{waits:Anita}` or
  `{after:<task id or phrase>}` when it waits; a new task joins an existing line or adds one; set a line's `done:`
  when its last task closes.
- **Done** means it runs, the spec matches, any decision is recorded, and the line has moved to the top of
  ARCHIVE.md. Then grep TASKS.md for the id, close a parent whose last stage this was, and
  name any other open task your work also did. New work found on the way is a new task.
- **Shared files**: re-read right before editing, small edits, never rewrite TASKS, DECISIONS or ARCHIVE wholesale.
  In a tree, TASKS.md, DECISIONS.md, ARCHIVE.md and LINES.md are never edited (a hook refuses): edit the main tree's,
  and run tasks.py from the maps root. Changelog items made in a tree are `<tree>-1`, `<tree>-2` until landing
  numbers them.
- **notes/** is the agents': move, merge, delete. Check outputs go in `../artifacts/trainworld/`. A notes link that
  doesn't resolve is in `notes/archive/<band>/`.
- **Data register**: a task that uses, finds or rules out a source updates that country's cell in
  `docs/data-register.md`.

## Working as an agent

- **Before starting**, `git status` and the "taking" marks; leave other agents' files alone.
- **Your own Vite**: `node trainworld/checks/dev-nowatch.mjs <port>` from the maps root (a free port in 6100-6199),
  restarted after app edits. 8800 is Anita's; no long sweeps against it. If your brief gives you a worktree, run
  everything from it.
- **Headless Chrome** through Playwright with its own `--user-data-dir`; stop only PIDs you started.
- **Playing the game in a check**: `checks/lib/play.mjs` (how-to at its top) opens a city or a save, picks tools on
  the real toolbar, draws track and places stations by pointer at lon/lat points (exact), presses controls by name,
  and reads panels and money as shown. Build fixtures through it rather than edits where the player's path matters;
  a new read goes in `checks/lib/reads.mjs`.
- **The real save** is `T-007-real-nyc.save`. `?new=1` never writes a slot.
- **Typecheck** with `npm run check --prefix app`; `npm run wasm --prefix app` when Rust changed. No full builds, no
  commits (a brief that touches shonei-server may say to commit there; pushing and deploying stay the orchestrator's).
- **Rust**: `CARGO_BUILD_JOBS=3`, keep the crate compiling between edits. After a sim/ change run
  `cargo test --release` and `cargo test --release --features demand --lib` from trainworld/sim. Add a function to
  the WASM interface in Rust before any app code calls it.
- **CPU**: the cap is the ledger's (`python cores.py list`). Playwright checks claim for themselves
  (`checks/cores.mjs`); claim by hand for the pipeline, Rust and scripts that launch their own browser. A page's
  demand workers take about a core each: cap them to your claim.
- **Data runs, packs, joins, AWS**: read the rules at the top of `docs/world-run.md` first.
- **Shell**: scripts in a file in your scratchpad. Pin processes from PowerShell or a `.cmd`, never
  `cmd /c start ... /b` through Git Bash (it pops dialogs on her screen).
- **Screenshots** go in `../artifacts/trainworld/`. For UI work, judge your own screenshots before reporting.
- **Advisor** (when your session has one): call it for a call nobody has made, when stuck (the same failure twice or
  about 30 minutes without progress) or before giving up, and before claiming a cause; about once a task. Not for
  routine steps, wording or small UI. Say in your report whether you called it and what changed.
- **A refused permission check**: make the step safer and retry; don't wait for Anita.
- **Stop at a clean point** if the work outgrows one context: finish the stage, file the rest, hand back.
- **Waiting**: a build or test whose result you need runs inside one foreground call with a long time limit (up to
  10 minutes; split anything longer), and you never end your turn while it is running (a watch on a background
  process does not hold the turn open). Work that needs no result from you may run in the background, with at most
  one light check every 4-5 minutes. Final checks (smoke) run in the foreground.
- **Testing, kept lean** (Anita, 2026-11-17: "a bit too test heavy"): run the checks near your change, not every
  check. Smoke only when your brief asks or the change touches sim, saves, demand or money; the orchestrator runs one
  smoke per landed batch. Add a standing check only for a bug that really happened and could quietly come back;
  otherwise prove the change with a scratch script and leave it in your scratch folder. Before writing a scratch
  script, grep `checks/INDEX.md` (made by `tools/checks_index.py`) and reuse a check that already proves it.

## Game principles (Anita's)

- **The whole world**, a low-fidelity global build (tier 0) everywhere first; tailoring country by country later,
  popular countries first.
- **Demand is simple on purpose**: sensible and fun over fitting real cities. Targets are her two ceiling tests (rail
  share of rail + car: A at least 85%, B at least 50%; notes/T-636.md). Propose any complicating factor to her first.
  A demand rule change updates `docs/demand-worldwide.md`.
- **Every demand task has a "done when"**: numbers that end it, plus "would a player see something obviously wrong?".
  Once met, stop. Performance tasks get a looser one.
- **General sources**: global datasets or a method any census can feed; a one-city source only if very easy.
  Local demand factors (by country, region or city kind) are allowed again where a worldwide rule misses a real gap
  and published figures show it; a place that already looks right (the US intercity) keeps the worldwide rule
  (Anita, 2026-11-11). Look
  outward for how other games and map makers solve it; write what you find in notes or research/.
- **Performance**: quantify impact, do the big ones first. An approximation is welcome when a player couldn't tell;
  say what the lie is in DECISIONS.md.
- **Simple prices**: fares and costs the player reasons about follow a simple rule (a base plus per km).
- **Buses are settled**; no more bus tuning unless she asks.
- **Saves are hers**: keeping old ones loading is a nicety, no real migration effort.
- **UI principles** live in `docs/ui-panels.md` "Look" and "Hover tags".

## Build and run

`app/` (Vite, TypeScript, MapLibre, Preact), `sim/` (Rust to WASM), `pipeline/` (Python city packs,
`..\venv\Scripts\python.exe`). Build orders: `docs/code-map.md`.
