# Handoff — Finland pilot

Updated 2026-09-10 (America/New_York). Start here together with `AGENTS.md` and `README.md`; this file is the durable state for a fresh task and does not require the previous conversation.

## Current state

- Project: `C:/Users/anita/projects/maps/traincounts` (inside the parent `maps` repository).
- Static MapLibre app in `dist/`, with a dark OpenFreeMap base matching the Japanrail reference direction. The user has seen and liked the first preview.
- Finland only: 454 corridors, 217 boarding/alighting stations, 16 dashed straight-line geometry fallbacks.
- Snapshot: September 14–20, 2026. Fintraffic feed publication `2026-09-10T01:52:22.837Z`.
- Daily scheduled physical-train identifiers: 1238, 1244, 1226, 1221, 1238, 963, 938. Weekly average: 1152.6 unique scheduled trains/day across the feed. This is not the sum of segment counts; a train traverses many segments.
- Width shows both-direction corridor counts; color is the dominant service category over the full week. Green includes intercity and regional services.
- Week, weekday, weekend and individual-day views; thickness slider and legend; selected-corridor daily/directional/operator details; source and methodology dialog.
- No deployment configured. Initial local preview used `http://127.0.0.1:8766/`. Do not assume a server survives a new session.
- At handoff, all project files are **untracked in the parent repository**; no commit was made. A fresh task using this same local directory can read them. A new worktree from the committed branch will not contain the project until it has been checkpointed into Git or explicitly transferred. Inspect current status rather than assuming this remains unchanged.

## Run and verify

From the project directory:

```powershell
python -m http.server 8766 --bind 127.0.0.1 --directory dist
python -m unittest discover -s scripts
python scripts/validate_finland.py
node --check dist/app.js
```

Run the server separately; it is a persistent process. Reuse it if already serving this project, otherwise start it. Validation needs `data/raw/finland.zip`, which is present in the original local checkout and ignored by Git.

To reproduce the current snapshot without overwriting the downloaded input:

```powershell
python scripts/build_finland.py --start 2026-09-14 --days 7
```

`--download` fetches the current feed and overwrites that raw file. Preserve the original if you need to reproduce the initial week; future publications are not the same snapshot.

## Verification already performed

- Three regression tests pass: calendar additions/removals, loop shape endpoint matching, and combined platform/pass-through/bidirectional fixture.
- Independent reconciliation matches all 454 saved corridor counts to raw stop-time/calendar records on all seven dates. Direction sums, operator/category totals and coordinate bounds also checked.
- JavaScript syntax check passed; local root returned HTTP 200; preview handed to the user.
- No automated browser interaction or visual QA was performed. The user's positive preview feedback is not exhaustive browser validation.
- Reconciliation proves consistency with the feed, not completeness of national coverage or correctness of source geometry.

## Known gaps and next choices

1. Sixteen geometry fallbacks remain. One is the busy Hiekkaharju–Havukoski corridor (639 trains/day); geometry improvement matters even though its count comes from the timetable.
2. Shared corridors with differing intermediate-point sequences may remain separate. Audit network alignment before treating this as a finished physical network inventory.
3. The frontend hardcodes Finland, bounds, labels and the `data/fi/` path. The second-country task must introduce dataset selection and per-dataset metadata/bounds while retaining Finland.
4. The current independent validator is Finland-specific; other countries need their own source checks. Frequency-based service is deliberately rejected by the current processor. Deduplication uses Finland train numbers, and coupled services are not otherwise reconciled.
5. Metro, tram, freight and replacement buses are excluded from this pilot. The longer-term passenger-rail scope will expand system by system.
6. No second country was selected or researched to completion. Finland's clean official feed made it a good first source. Amtrak's inspected ArcGIS GTFS directory redirected to sign-in; this does not prove no usable public Amtrak feed exists.

## Suggested next-task prompt

> Work in C:/Users/anita/projects/maps/traincounts. Read AGENTS.md, HANDOFF.md and README.md. Add [COUNTRY OR SYSTEM] using verifiable schedule data. Preserve Finland and the existing map style. Verify source coverage and geometry before choosing the adapter; implement and validate counts, expose the dataset in the map, and update the handoff notes. Clearly label any missing systems or approximate geometry. Keep the result local.

For the second country, Sol is the prior agent's suggested starting model because source discovery and shared-code extraction are still open-ended. Terra may suit later additions with a demonstrated adapter pattern. This is a workflow suggestion, not a requirement or a benchmark result. Switching models does not change the project's acceptance criteria.
