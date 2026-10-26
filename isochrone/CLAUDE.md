# isochrone — notes for agents

Read this, then `SPEC.md`, then `TASKS.md`. Everything else on demand; where code lives is
`docs/code-map.md`.

An interactive world travel-time map: pick a start, choose which ways of travelling and which
infrastructure exist, see how long it takes to reach everywhere and how likely you are to survive
the trip. Started by Ashley (Anita's girlfriend) on 2026-10-13 from Anita's idea; Anita leads it,
with an orchestrator session running agents from `TASKS.md` (since 2026-10-26).

## What each file is for

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the map is and how it works, **as of now**, in summary | agents, rewritten in place |
| `docs/` | The spec in detail, one file per area (`resolution.md`), and `code-map.md` | agents |
| `TASKS.md` | Open agent work, by milestone and priority | agents |
| `DECISIONS.md` | Dated choices someone might revisit, and why | agents, add only |
| `ARCHIVE.md` | Finished and abandoned tasks, newest first | agents, add only |
| `notes/T-###.md` | Working notes for a task too big for one line; analysis scripts a note cites | the task's agent |
| `notes/for-anita.md` | Open questions for Anita that outlive a session | orchestrator |
| `checks/` | Standing checks, `smoke.mjs` first | agents |
| `research/` | Background reports with sources, one file per topic | agents |
| `TODO.md` | Anita's own list: read it and file new items as tasks | Anita only; agents never edit it |

## Doc rules

- **The spec says what is true now.** When a decision changes, rewrite the affected text to state
  the new design; never append "Update:" paragraphs or leave a section a later one contradicts.
  A spec section past ~150 lines moves to `docs/<area>.md`, leaving a summary and link.
- **Touch only the section your change touches**, in the spec, docs and code-map alike.
- **Docs hold current rules only**: the rule and the numbers that decide it, in the present tense.
  Measurements, how a figure was reached and before/after tables go in notes/.
- **Comments**: most code needs none. A comment says why, only where the why is not evident from
  the code (a non-obvious choice, a unit, an invariant), with a task id if the reasoning lives
  elsewhere. Never what the code does, never history: that is DECISIONS.md.
- **DECISIONS.md and ARCHIVE.md are searched, never read whole**: grep them for a task id or word.
- **A DECISIONS line** only for a choice someone might reasonably revisit: date, one or two
  sentences, the task id. Not for what the spec already says.
- **An ARCHIVE entry** is at most two lines (about 40 words): id, title, date, the outcome only,
  and a notes/ link for detail.
- **Tasks.** One line each in `TASKS.md`:
  `- T-012 [P2] short title. State, owner if taken, link to notes/T-012.md if any.`
  Ids are never reused (next after the highest in TASKS.md and ARCHIVE.md). With several agents
  at once, the orchestrator hands out ids: a brief names the ids its agent may file, or the agent
  asks. P1 = blocking or now, P2 = this milestone, P3 = later. Tasks under "Someday" are real open
  tasks at a lower priority. Add `taking, <date>` before starting. More than ~3 lines of context
  goes in `notes/T-###.md`.
- **A task is done when** the work runs, the spec matches it, any decision is in DECISIONS.md, and
  its line has moved from TASKS.md to the top of ARCHIVE.md. Abandoned tasks move the same way,
  saying why. New work found on the way goes in TASKS.md; do not silently widen the task.
- **Each milestone ends with a doc tidy task**: spec against code, prune TASKS.md, fold finished
  notes into the spec or archive, flag anything stale.
- **Shared files.** Several agents work here at once: re-read a doc right before editing it, make
  small targeted edits, never rewrite TASKS.md, DECISIONS.md or ARCHIVE.md wholesale.
- **notes/ is the agents'**: move, merge or delete freely. One-off scripts go once their task is
  done; scripts worth keeping go to `tools/` or `checks/`.
- **Every source gets a licence check**, recorded in its research file. Project knowledge lives
  here, not in Claude's memory.

## Agent working rules

Briefs carry only the task; these hold for every agent.

- **Before starting**, check `git status` and the "taking" marks in TASKS.md; several agents work at
  once, each owning named files, so leave others' alone.
- **The page** is served live from this folder at `localhost:8800/isochrone/` by the maps-wide
  `serve.py` (no build step). Never start another server. Headless checks may use 8800; add `?v=N`
  to bust the cache.
- **Headless Chrome**: `node isochrone/checks/smoke.mjs` once it exists; otherwise copy its setup.
  Launch with `--headless=new --enable-unsafe-swiftshader`, a private debugging port and a
  `--user-data-dir` in your own scratchpad folder. Stop only the PIDs you started; never kill
  browsers or node broadly.
- **Python**: plain `python` (C:\Python39 has rasterio, numpy, scipy; the maps venv does not have
  rasterio). Set `OMP_NUM_THREADS` before importing numpy.
- **CPU**: at most 4 of 16 cores in all, every isochrone agent together, day and night (Anita,
  2026-10-26: other projects run at the same time). Long jobs claim their cores in the shared
  ledger first: `python cores.py claim <n> <name> --pid <pid>` at the maps root (see maps/CLAUDE.md).
- **Downloads** go under `data/raw/<source>/` (gitignored). Big raw files go once the layer built
  from them is checked, without asking. A download behind bot protection or a login: stop and give
  the orchestrator the URL for Anita. Never register accounts; never put her name or email in a
  request.
- **Long runs** (over ~4 minutes) start detached with their log in a file; write the command, log
  path and how to check it is alive into the task's notes before waiting. Wait once on the process
  ending, never by polling.
- **Bash**: one simple command per call (no `&&`, pipes or loops); scripts go in a file in your
  scratchpad folder, not heredocs.
- **Screenshots and diagnostic plots** go in `../artifacts/isochrone/` (`maps/artifacts/`,
  gitignored), never in this folder.
- **On-map and panel text** follows the "User-facing text" section of Anita's global CLAUDE.md.
- **Stop at a clean point** if the work outgrows one context: finish the stage you are on, write
  the rest as tasks, hand back.
- **Reports to the orchestrator are short**: what changed and where, check results, anything for
  Anita. Detail goes in notes/.
- **No commits.** Subagents never commit or push.
- **open** in the spec means Anita decides. Otherwise decide, record why, and carry on. Ask only
  where the choice changes what the map is.

## Before handing back a round (orchestrator)

`node isochrone/checks/smoke.mjs` from the maps root. Then commit (Anita, 2026-10-26): the
orchestrator commits `isochrone/` work as it goes, explicit paths only (never `-a` or `git add .`),
one plain sentence, staged diff checked right before committing since agents edit shared files.
Subagents never commit. **Pushing needs Anita's word.**

## Build and run

- `index.html` + `worker.js` (a module worker over `js/`): the page. Plain JS, MapLibre GL 4.7.1 from unpkg, no build step.
  The worker owns the grids and runs the searches; the main thread only paints and handles input.
- `tools/build_grid.py` makes the coarse world grid (seconds); `tools/build_tiles.py` the L3 and
  L1 tiles (~3 min). Details: `docs/code-map.md`.

## Working here (Anita's decisions)

- **The motivating question**: "if you just have a canoe in 10,000 BC, how far can you get?"
  Milestones are ordered to answer it (TASKS.md).
- **Prefer global sources.** A source covering one country is worth it only as a check or if
  trivially easy. Gated data is a last resort: look for an open mirror first.
- **Licences.** The MAP friction surfaces are CC BY 4.0: credit them in the info panel. Every
  credit a source asks for goes in the info panel when the source is used.
- **Time and danger both count** (2026-10-26): the search minimises time + K x hazard on every
  mode; survival along the chosen path is shown.
- **The canoe is always carried overland**, never left behind (2026-10-26).
- **Task size**: split big tasks into stages, each its own id and report. A big task's design goes
  to Anita before it is built; small, quick, isolated work needs no review. Treat subagents well
  by substance: a brief they can act on (goal, decisions made, scope, what done looks like), leave
  to stop or push back, problems relayed without blame, a short thanks when done.
- **UI** follows the maps house style (riders/nycriders): bare title, dark panels, few words,
  sentence-case labels, no transitions, every drawn number gets a hover. Panels show what can
  change; no explanatory or one-time text.
- **Asks for Anita** go in the "For you" block of the orchestrator's final message, never in
  `TODO.md`. Ones that outlive the session go in `notes/for-anita.md`.
