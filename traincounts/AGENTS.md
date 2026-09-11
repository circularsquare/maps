# Traincounts: instructions for future agents

Read `README.md` and `HANDOFF.md` before changing this project. The user's current request takes precedence over these defaults.

## Product intent

Build a world map of scheduled passenger train frequency, country by country and system by system. Preserve the existing MapLibre working surface and Japanrail-inspired dark OpenFreeMap base, bright outlined railway corridors, compact controls and clickable segment details. Thickness represents average scheduled train traversals per day in both directions, not passengers, seats or observed train movements. Keep the square-root width scale and its legend consistent unless the user asks to change it.

## Working conventions

- This directory is a subfolder of an existing maps Git repository. Inspect status before editing; preserve unrelated projects and user changes. Do not initialize a nested repository or overwrite another agent's changes.
- The current app is authored directly in `dist/`; it is not disposable build output. No package installation or frontend framework is required. Use Python 3.9+ standard library for the existing processor.
- Work locally by default. Hosting has not been configured. A country addition does not require deploying or redesigning the app.
- Prefer a bounded country/operator adapter and extract shared logic only when demonstrated by the second source. Do not assume the Finland adapter is a generic GTFS importer.
- Keep one active writer on shared frontend and processing code. If independently working in worktrees, confirm the initial project files are actually present; untracked files do not form a committed baseline.

## Country addition workflow

1. Verify an actual downloadable schedule source, publisher, reuse license, timezone, coverage and feed dates. Record these in project documentation and generated metadata. Do not claim national completeness from one operator's feed. Report blocked sources honestly.
2. Inspect calendars, exceptions, route types, trip identity, shapes and intermediate pass-by coverage. Establish whether rail replacement buses, trams, metro, heritage, cross-border and coupled services are included. The long-term scope includes passenger rail broadly; exclusions must be explicit for each pilot.
3. Keep the existing Finland snapshot and its attribution available. Add data under a separate country/system namespace. With the second country, remove Finland-only frontend assumptions and introduce an explicit dataset selector or registry; do not silently replace Finland.
4. State the analysis window. Prefer matching windows for comparisons when feeds support them; otherwise show differing dates clearly. Do not silently refresh Finland just to add another country.
5. Count active service dates after both exception additions and removals. Include zero-service days for each segment in the denominator. Handle times beyond 24:00 by local service date. Expand frequency-based service explicitly when present.
6. Normalize platform IDs and deduplicate physical services using source-specific evidence. Finland's train-number prefix rule is not portable. Never sum duplicate operator or cross-border feeds blindly.
7. Put express and stopping trains on shared physical corridors before aggregation. Consecutive boarding stops alone do not establish the shared railway path. Feeds without pass-by points need appropriate geometry/network routing. Do not conceal unresolved routing with plausible-looking lines.
8. Mark approximate or missing geometry, retain provenance and expose coverage gaps. Do not call raw-GTFS reconciliation an independent validation of the publisher's completeness or the physical route.

## Validation and handoff

- Run `python -m unittest discover -s scripts` after changing counting logic.
- Run `python scripts/validate_finland.py` when the cached original Finland feed is present and Finland processing/data changes. It independently reconciles the saved output with the raw feed; it must match the recorded checksum.
- Run `node --check dist/app.js` after frontend JavaScript changes. Inspect local asset references and serve `dist/` for the preview. Do not claim browser interaction testing unless actually performed.
- For a new adapter, add small meaningful fixtures for source-specific calendar, direction, identity and routing edge cases. Reconcile per-day output with source records and manually spot-check a few busy, rural and express corridors where public schedules are available.
- Preserve source URLs, license, publication timestamp, download checksum, analysis dates, metric, scope and geometry quality in output metadata. Keep raw feeds out of Git; retain the original raw snapshot locally when available.
- Before finishing, update `HANDOFF.md` with what changed, exact commands, validation performed, known gaps and a concrete next step. Keep `README.md` accurate. Report limitations without overstating completion.
