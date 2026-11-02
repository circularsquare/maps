# Adding a country

The procedure for a country new to the game: whole-country cover from the world plan, a low-fidelity
first pass on global sources, run by `pipeline/world_run.py` (T-1975, notes/T-1975.md has the pitfalls
it is built from). Fix this file wherever the next country finds it wrong.

Reference, read only the section you need: what the runner does, its flags, locks, gate, join and faults,
the world plan's rule and what a plan row's config is, `docs/world-run.md`; the zone rules, `docs/zones.md` 2;
a hand-made pack or a country module, the inputs table and the worked examples,
`docs/adding-a-city-by-hand.md`; gate thresholds, notes/T-538.md. Rules that apply: CLAUDE.md "Working
here" (general sources only, zones at least about 0.5M, no calibration beyond the gate) and "Agent working
rules".

Every command runs from the maps root; Python is `venv\Scripts\python.exe -I`. Logs, pictures and check
output go in `artifacts/trainworld/`. `<kit>` is your own scratchpad folder; `<tw>` is `trainworld`;
`<tc>` is `<tw>\tools\country`, the standing scripts every country uses, each with a `--help`:

| Script | Does |
|---|---|
| `setup_sim.py <kit> [make\|check\|refresh\|remove]` | HEAD's `sim/` in `<kit>\simroot\sim`, with `data` and `app` junctions to this tree's |
| `backup.py <task> <stage> <CC,CC\|all> [--dry]` | the shared files and the countries' packs, to `data/work/pre-<task>/<stage>/` |
| `restore.py <task> <stage> [--go]` | puts a backup back (lists without `--go`) |
| `setbuilt.py <ids> [<check> "<reason>"] [--dry]` | parked rows back to built, optionally relaxing one gate check |
| `stale.py [--cmd <file>]` | market packs naming a place with no pack; writes the .cmd that rewrites them |
| `start_hidden.ps1 <cmd>` | starts a .cmd detached, no window; prints its PID and its children |
| `waitany.ps1 <pid,pid>` | waits for any of the PIDs to exit |

## 0. Start (10 minutes)

- Take the task line (`taking, <date>`), start `notes/T-###.md` from the template at the end.
- Read: this file, then the country's row in data-register.md (grep the country's name in every table;
  never read the register whole). Not other countries' notes: their lessons are in this file.
- `git status`; `world_run.py status` (no run claims your country's rows; no join holds `packs`).

Done when: the task is taken and the notes file exists.

## 1. Sources (10 minutes; one hour at most)

Take the first that holds; never take a source from memory.

1. **Global sources only (archetype D)**: the default. No country module; zones and metros on tier 0.
2. **A population hook**, only if all hold: the national census (2015 or later, not a lockdown year for
   anything about work) gives people by small area (about 5 km² or finer in towns) with polygons, from an
   open endpoint with no account; a hook of the same shape exists to copy (`canada.py`'s dissemination
   areas, `australia.py`'s SA1s); it is written and tried in an hour. Try it on one metro and one zone in
   a scratch tree (docs/world-run.md "A scratch tree") and read its census-against-GHSL line before the run.
3. Everything else (jobs by small area, commute flows, travel surveys, schools): not in the first pass.
   Record what you found in the register (N, G or ?) and file a P3 task for anything usable.

An official server that refuses scripts: look for a mirror for 15 minutes, then put the URL in the notes
for Anita and go on with D.

Done when: the country has a row in every register table (or is added to the shared D row) with its
archetype.

## 2. Plan dry run (20-30 minutes)

    <tw>\pipeline\world_run.py step world_plan zones --countries XX --out <kit>\xx.json > artifacts\trainworld\t###-dry.log 2>&1

Pick the zone rule from what the dry run shows (`world_plan.ZONE_RULES`, docs/zones.md 2):

| The dry run shows | Rule |
|---|---|
| zones of whole towns and their countryside | the default (floor 0.5M, cap 1.5M) |
| most of the land empty, towns of 50,000+ inside dead space or zones over 100,000 km² | `reach` as Canada and Australia |
| plan units finer than counties (zones of 100+ units, zones named after a village) | `group_level` as Korea |
| islands | `contiguous` (10 km of water at most between units) |

A rule change means a second dry run. Read the merge log (`zones.<ISO>.merges` in the dry plan).

Done when: the dry plan's rows look like the country's towns and countryside.

## 3. Preflight (on the dry plan; fix before any build)

Each item checked and ticked in the notes. Items 1-5 and 8 in one call, from the maps root:

    venv\Scripts\python.exe trainworld\checks\country_preflight.py XX <kit>\xx.json [--png <kit>\xx.png]

It prints a FLAG line for each fault, NOTE lines for the run, and the picture's path (default
`artifacts\trainworld\preflight\<ISO>.png`: the rows, flagged ones in red, far parts circled, other countries'
packs in grey); exit 1 when anything is flagged. Open the picture, fix every FLAG, dry run again.

1. **Origins** (FLAG origin): none of the country's zones over 25 km from the unit it leads with. Over: the
   row's origin goes to its town (`checks/zone_new_origins.py <ids>`).
2. **Scattered units** (FLAG scattered): no row with towns (1,000 people) in parts over 100 km from all its
   others. The planner already makes such a unit dead space when it is not its zone's most populous
   (`world_plan.far_towns`, T-1984: "Unincorporated NT"); far islets with next to no one are a NOTE. Fix:
   the far unit to dead space.
3. **Lone tiny zones** (FLAG lone): a zone under 50,000 people parks on its own values. The planner makes
   them dead space in every country (`world_plan.ZONE_ALONE_MIN`, T-1984); a FLAG means a country figure
   in `ALONE_MIN` let one through.
4. **Cut cities** (FLAG cut city): the main urban centre of a GHS-FUA over 1M split between rows, two each
   holding a fifth of it. The planner joins such zones over the cap (`world_plan.FUA_WHOLE`, T-1984:
   Adelaide); a FLAG left is a metro row against a zone, or a suburb in another FUA (Sydney's Campbelltown
   was): one zone or the metro takes it, patched under the plan lock as T-1857's `patch.py` did; record
   in DECISIONS.md.
5. **Borders** (NOTE borders): the built countries within 50 km. The default factor is the national 0.05
   (`pipeline/borders.json`); a dense crossing (Singapore and Johor Bahru) needs a pair factor.
6. **Machine**: smoke 20 of 20 on your own Vite (`checks/smoke.mjs`; a FAIL here is a stale WASM or
   another agent's edit, and every browser check would park); your Vite's and the run's ports free; disk
   free at least 15 GB plus twice the biggest extract; nobody else's run, join or terrain job writing
   `data/packs` or `data/terrain`; cores claimed (`cores.py claim`).
7. **Sim copy**: `<tc>\setup_sim.py <kit>` (HEAD's sim/ in `<kit>\simroot\sim`, with `data` and `app`
   junctions); `setup_sim.py <kit> check` OK. It prints `TW_SIM_DIR` and `TW_CARGO_TARGET_DIR`: both go in
   every .cmd. `setup_sim.py <kit> refresh` after sim/ is committed again.
8. **Reused ids** (NOTE reused id): the runner clears a planned row's old per-id caches itself, `data/terrain/<id>.terrain`
   included (T-1743, T-1986); only a cache it does not list (`world_run.ROW_CACHES`) needs deleting by hand.

Done when: all eight ticked.

## 4. Write the plan, back up (15 minutes)

    <tw>\pipeline\world_run.py step world_plan zones --countries XX > artifacts\trainworld\t###-zones.log 2>&1

The live rows must match the dry run's (ids may differ if another country took some). Apply item 1-4
patches if they were made on the dry plan. `world_run.py plan --countries XX --limit 300` lists the
runner's regions and extracts. Then `<tc>\backup.py T### pre XX,<neighbours>` (`--dry` counts first).

Done when: the plan holds the rows and the backup exists.

## 5. Run, detached, one wait

Write `<kit>\run.cmd`:

    @echo off
    cd /d C:\Users\anita\projects\maps
    set PYTHONIOENCODING=utf-8
    set TW_SIM_DIR=<kit>\simroot\sim
    set TW_CARGO_TARGET_DIR=<kit>\simroot\sim\target
    set PATH=%USERPROFILE%\.cargo\bin;%PATH%
    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --countries XX --limit 300 --jobs N --port 63xx --no-join > artifacts\trainworld\t###-run.log 2>&1

with N the cores granted. Regions over 100 rows: `set TW_QUIET_S=3600` in the .cmd. Start it with
`powershell -File <tc>\start_hidden.ps1 <kit>\run.cmd`; it prints the cmd's PID and, three seconds
later, its children (the python PID is one of them; `cores.py claim ... --pid` takes it).

**Before waiting**, write in the notes: batch id (the log's first lines), log path, cmd and python PIDs,
liveness (`Get-Process -Id <pid>`, `world_run.py status`), restart (the same command: built rows are taken
up, not rebuilt), stop (`<tw>\data\work\world_run\STOP`).

**Wait** with one background command, `powershell -File <tc>\waitany.ps1 <python pid>`
(Bash with `run_in_background`), then end your turn or do this task's desk work (register row, names
draft). Never poll: no reading the log, no `status`, no foreground sleeps while it runs. Each wake reloads
your whole context.

Several countries at once: one run each with `--no-join`, its own port and log; one join for all.

## 6. Parks: one relax run

Read the batch report and `world_run.py status`. For each parked row:

| Reason | Do |
|---|---|
| `drive_factor_median` over 3.0 on a zone (it WARNs from 2.0 and fails only above 3.0; a metro fails above 2.0, T-1985) | open the row's routes: hills, gorges or islands (Yuanyang, Yunnan, 3.13), relax naming the ground; else a routing fault, leave parked and file a task |
| `water_points_in_water` at 0.9 or more, the dry points in seasonal wetland | relax, naming the wetland |
| `top_zone_jobs_share` up to 0.3 on a zone under 0.2M | relax |
| `mean_commute_km` on a zone over 20,000 km² | relax |
| `pop_vs_plan` on a row with a census hook | relax with the census figure |
| `browser` (few riders, a purpose missing) | open `world_run/<batch>/gate/<id>-line.png`. The line leaves the town or crosses a border: move the origin (section 3 item 1), never relax. Otherwise relax with what the picture shows |
| `register failed ... invalid commute between zones` | `<tc>\stale.py --cmd <kit>\fix.cmd`, start the .cmd as in 5 and wait, then `stale.py` again: 0 |
| a build step error | an old cache (ZeroDivisionError in `osm_signal_line`): delete it. "Failed to open dataset": memory, rerun with fewer jobs. Else a code fault: leave parked, file a task |
| anything else | leave parked, file a task with the reason |

Relax with `<tc>\setbuilt.py <ids> <check> "<reason>"` (it sets the rows built; `--dry` first shows
what it would change), all at once, then one rerun (`run --rows <ids> --no-join`, detached, one wait as in
5). A row that parks again stays parked with a task.

Done when: every row is gated, dead space, or parked with a filed task.

## 7. Join (once)

Only while no other run, join or terrain job writes `data/packs` or `data/terrain`. Claim at least 6
cores (a join on 2 timed out). `<tc>\backup.py T### prejoin XX,<neighbours>`, then a `join.cmd` as the
.cmd in 5:

    <tw>\pipeline\world_run.py join --countries XX[,YY] --jobs N > artifacts\trainworld\t###-join.log 2>&1

Liveness, restart (the same command) and the wait as in 5. The join writes outlines, realms, aliases and a
freeze itself. A unit that fails restores itself; `<tc>\restore.py T### prejoin --go` undoes the whole join.

Done when: `world_run.py status` shows the country's gated rows shipped.

## 8. After (done when every line holds)

- `world_run.py status`: the country's rows shipped, dead space, or parked with a task each.
- `<tc>\stale.py`: 0 market packs name a freed place.
- `checks\zone_origins.py`: none of the country's rows over 25 km.
- Load check on the biggest metro and one rural zone: `ROWS=<kit>\rows.json node <tw>\checks\city-browser.mjs`
  (rows from `world_run/<batch>/gate/<id>.row.json`) on your Vite: PASS, and the `-line.png` opened: the
  line runs through the town.
- Smoke 20 of 20 on your Vite; the real save's rail trips within 0.1% of the figure in your brief, or the
  difference explained.
- Names: zones named after a rural unit that hold a town take the town's name in
  `app/src/game/zone-names.json`, the unit's name as an alias; the country's line in
  `app/src/game/realms.json`.
- Docs: the register row (packs, archetype, extract dates); DECISIONS.md for relaxations of a new kind and
  hand patches; demand-worldwide.md only if a rule changed; ARCHIVE line; task off TASKS.md.
- Clean: the backups deleted, the kit's sim copy removed with `<tc>\setup_sim.py <kit> remove` (it takes
  the `data` and `app` junctions off with rmdir before the copy), cores released.

## 9. When to stop and hand back

- Your context is past about 200k and a wait of an hour or more is next: write the state in the notes
  (batch, PIDs, log, restart, next step) and hand back; a fresh agent takes the next stage.
- A fault in shared pipeline code needing more than a small fix, or a third of the rows parked on one
  reason: stop the run (STOP file), file the task, hand back.
- A join unit's solve passing two hours, or a choice that changes what the game is: hand back with the
  question. Rail facts, zone shapes and names are yours to decide.

## 10. Retrospective (in the notes, at most 10 lines)

Parts in order with wall time; runs and joins (how many, why more than one each); parks by reason; what
the preflight missed, as a proposed new line for this file; anything not worth its cost. Tokens and wakes
are the orchestrator's (`checks/token_usage.py`), added to notes/city-costs.md.

## Notes template

    # T-###: <country> (<date>)
    Agent <name>. Kit: <full scratchpad path>.
    ## Sources        archetype, hook if any, what was found or ruled out
    ## Plan           rule, rows (metros, zones, dead space), patches and why, preflight ticks
    ## Runs           per run: .cmd, batch, log, PIDs, liveness, restart; result (gated, parked with reasons)
    ## Join           the same; shipped
    ## Checks         the section 8 lines with their figures
    ## Retrospective
