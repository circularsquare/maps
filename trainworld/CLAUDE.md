# trainworld — notes for agents

Read this, then `SPEC.md`. Grep `TASKS.md` for your task id rather than reading it. Everything else on demand:
for how a mechanic works, the wiki first (`wiki/index.md`, then `wiki/read/<page>.md`, which has the figures filled
in and ends with the doc section and code that own the rule), then the docs. Where code lives: `docs/code-map.md`.
The orchestrator also reads `docs/orchestrator.md`.

## Talking to Anita

- **Reports go to the orchestrator** and are short: what changed and where, check results, anything for Anita
  (marked as hers). Detail goes in notes/.
- **She doesn't read the code.** Quote the value or list in the message, or name the place on 8800, never a file
  to open.
- **Estimates in hours of real agent time**, checked against similar past tasks (T-3137 C1, guessed at days, took
  about 45 minutes). Never "agent-days".
- **Ask when the answer is needed next**, not ahead. Test items only once the change is on 8800.

## What waits for her word

Publishing the game and the website push; every server deploy (never overnight); adding a complicating factor to
demand; a choice that changes what the game is; anything the spec marks **open**; a big task's design before it is
built. Everything else you decide, record in DECISIONS.md if someone might revisit it, and tell her after: rail
facts, sensible defaults, visible calls, zone and region names. A slip gets a fix, not a new standing process.

## Player text

- **Voice**: the "User-facing text" section of her global CLAUDE.md, the wiki included. "Trips" for what people want
  to take (demand); "rides" and "riders" only for what happens on the player's system.
- **Review**: list every in-game string you add or change, old -> new, exactly as shown. Landing doesn't wait; she
  answers "text acknowledged" or names lines. Propose wording; never ask her to write it.
- **Not reviewed**: the wiki, zone/region/realm names, docs, commit messages, and text copied word for word from her
  own message (Anita, 2026-11-11); still list it in the report.
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
| `DECISIONS.md`, `ARCHIVE.md` | Dated choices and why; finished tasks, newest first. Older: `*-to-2026-10-19.md` | agents, add only |
| `notes/` | `T-###.md` task notes, `handoff.md` (orchestrator), `for-anita.md` | agents |
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
- **Tasks**: `- T-012 [P2] short title. State, owner if taken, notes link.` P1 now, P2 this milestone, P3 later;
  "Someday" tasks are real open tasks. Use only the ids your brief gives you. Add `taking, <date>` before starting.
- **Done** means it runs, the spec matches, any decision is recorded, and the line has moved to the top of
  ARCHIVE.md. Then grep TASKS.md and notes/for-anita.md for the id, close a parent whose last stage this was, and
  name any other open task your work also did. New work found on the way is a new task.
- **Shared files**: re-read right before editing, small edits, never rewrite TASKS, DECISIONS or ARCHIVE wholesale.
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
- **A refused permission check**: make the step safer and retry; don't wait for Anita.
- **Stop at a clean point** if the work outgrows one context: finish the stage, file the rest, hand back.
- **Waiting**: under about 4 minutes, one foreground call; longer, in the background with a wait on the process
  ending and at most one light check every 4-5 minutes. Final checks (smoke) run in the foreground.

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
