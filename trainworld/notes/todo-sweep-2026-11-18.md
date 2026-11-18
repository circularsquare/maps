# TODO.md sweep, 2026-11-18

Agent todosweep, read-only. Each TODO.md item checked against TASKS.md (all sections, Someday included), ARCHIVE.md, DECISIONS.md, notes/ and git log.
Status: Done / Filed / Closed (archived with a reason) / Neither. "Unsure" is said where it applies.
Note: notes/T-3390.md records that the "for anita" section and most suggestion lines were deliberately never filed ("that section is hers unless she points at an item"), so many Neither rows below are expected.

## Short term

| Item | Status | Evidence / id | Proposal |
|---|---|---|---|
| wanan: build track/stations as one big update | Filed | T-3444, T-3365, T-3348 (LINES update:track-stations) | |
| scheduling/fares as a separate medium update | Filed | T-3346, T-3542 (LINES update:scheduling-fares); fares line done 2026-11-17 (T-3567) | |
| sign in / account + community buttons on new game and landing | Done | T-3401 (sign in on new game sheet), T-3222 (Community button on start menu) | |
| profit map mode and single track / flyover splits | Done | T-3403 | |
| intercity trip explorer: effective time breakdown hover | Done | T-3404 (not seen rendered on an intercity trip) | |
| select-tool rebuild options in the track panel, hint on Rebuild hover | Done | T-3408 | |
| make single/double/quad obvious when building and viewing | Filed | T-3365 (waits on you) | |
| ...same, for bus / tram / train / hsr | Neither | nearest T-3365 (track counts only) | see list, item 6 |
| schematic with one city: does far building hurt? schematic of chosen cities | Filed | T-3409 | |
| viktini: tram passes through or reverses at a one-way station | Filed | T-3079 (folded T-3410, waypoints) | |
| double track: maybe model everything as single | Filed | T-3444, T-3365 | |
| split single track station: 2 stops in one complex on one line | Filed | T-2920 (P4), T-3079, T-3444; T-3672 is the side-platform bug (P1). Context at TASKS line ~530 asks for it. | raise T-2920 from P4: players hit this now |
| set certain tracks as available for bypassing | Filed (unsure) | T-3444 (parallel tracks, per-track roles) | |
| what is "too close to end of track" for; frustrating | Neither | text is `issues.splitPoint` ("5 m of track on each side"); nearest T-2268, T-1970 (station on a ramp) | see list, item 3 |
| 4 px gap between issues panel and its toggle | Done | T-3414 | |
| waypoints for trains | Filed | T-3079 (folded T-3384) | |
| isochrone map mode | Filed | T-3418 | |
| waiting room for ferries | Done | T-3419 | |

## For anita

| Item | Status | Evidence / id | Proposal |
|---|---|---|---|
| line subgroups, infinitely | Filed | T-3423 | |
| missing zones like Jurong: where, how prominent | Filed | T-3375 (empty units beside a metro, worldwide count first), T-3287 notes | |
| fix industrial areas' demand too high | Filed | T-3376 | |
| -5 to +5 track levels (advanced mode, colours) | Neither | only -3..+3 exist (T-3420, T-3593 mention +3 to -3 buttons); no task | see list, item 4 |
| mod support | Filed | T-3377 | |
| minor single/quad improvements, AA/BB, clear up confusion | Filed | T-3348, T-3335, T-3365, T-3444 | |
| better schematic | Filed | T-3345 (T-3360, T-3358, T-3453) | |
| timetables, better scheduling | Filed | T-3346 | |
| mini scheduling tier (morning / midday / afternoon) | Filed | T-3542 (4 or 5 periods) | |
| fares | Done | fares line done 2026-11-17; T-3567 fare multiple per line, T-1801 transfer fare, T-3581 | |
| open source (next week?) | Filed | T-3523 (public repo the real one, after v0.4.0); T-2913 export done | |
| better coverage | Filed | T-3245, T-3620 (Caribbean, Bolivia, Paraguay); T-3399 Central America and Venezuela done | |
| bezier track, toggleable (listed twice) | Neither | no hits in TASKS, ARCHIVE or notes | see list, item 7 |
| line improvements as a general interface for station improvements | Neither | T-2692 (done) added three buttons at the foot of the line view only | see list, item 10 |
| worker sharing memory optimisation | Filed | T-3373 | |
| more powerful fare controls | Done (unsure) | T-3567 per-line multiple 0x to 3x; blocks and route-specific fares are in the suggestions section | |
| buses faster / BRT, boarding penalty, BRT upgrade, bus build cost | Filed | T-3378 (design first; "buses are settled" unless she asks) | |
| ferries faster | Filed | T-3435 (first pass shipped, 45 km/h) | |
| "join the leaderboards" at top, leave at bottom | Neither | `leaderboard.leave` string exists; no task | see list, item 1 |
| placename languages | Filed | T-2586 | |
| community: friend request and message people | Neither | no task | see list, item 15 |
| view-only map of someone else's save, option to full load | Filed | T-1446 (link opens read only); the light load and "load fully" part is not in its goal | add to T-1446's goal |
| like button for the currently opened community game | Neither | likes exist (T-3132); no button on an opened share | see list, item 5 |
| no leaderboard from someone else's save | Done | T-3115, DECISIONS 2026-11-10: a copy plays as normal but sends no figures; T-3413 (lineage) filed for the rest | |
| ylik: sandbox save uploaded shows as normal in Community | Neither | T-3144 handles the Sandbox tag; no report-specific task | see list, item 9 |
| wiki | Filed | T-3447, T-2854 | |
| "only intercity" mode, each city with feeder | Neither | T-3630's intercity setting is different | see list, item 14 |
| language list: polish, arabic, italian, malay, dutch, thai | Filed | T-3398 | |
| better errand modelling | Neither | demand change, needs her word first | see list, item 16 |
| general brainstorm with two claudes | Done | T-1412 (22 ideas, picks filed as T-1430 to T-1450), T-2074 research; her two-claude variant not run | |
| leaderboard map view | Neither | no task | see list, item 12 |
| offline version when standalone, pick cities to download | Neither | no task | see list, item 13 |
| aa/bb quad track, split 2 into 4, platform assignment | Filed | T-3348 (T-3363 folded in) | |
| detailed load panel on crowding bubble | Filed | T-3456 | |
| loading focused city demand >3 min on big asia save | Neither (unsure) | nearest T-3256, T-3373, T-2624; T-2524 profiled an earlier load | see list, item 8 |
| money exploit: transfer fare set to 1000 | Neither | T-3657 is about free fares and leaderboard points, not this | see list, item 2 |
| traffic view: tram/metro width normalised by passengers | Neither (unsure) | T-3580 made single and double equal; per-mode widths not covered | see list, item 17 |

## Other player suggestions, not sure / deferred

| Item | Status | Evidence / id | Proposal |
|---|---|---|---|
| notifications for money and ridership up/down as a per-line list | Neither | no task | see list, item 11 |
| settings getting cluttered, searchable advanced settings | Filed | T-3681 (skipped as asked) | |
| venezuela | Done | T-3399 (91 of 95 rows); last four T-3517 | |
| airports too low (Wuhan, Istanbul, Tianfu 40k vs 150k) | Filed | T-3245 item 1 (airport trips across zone lines), T-3501, T-3502, T-3485 notes | |
| Istanbul airport: merge the enclave | Done | T-3488 | |
| French player: saves empty, stuck at 27 January | Filed | T-3395 (v0.4.0 keeps the last undamaged copy; triage notes) | |
| ctrl while snap to road stops snapping | Done | T-3608 | |
| overall travel by train / hsr / bus / tram | Neither | no task | see list, item 6b |
| leaderboard flags wrong (Sierra Leone shows Antarctica) | Neither | T-3269 added flags; no fix filed, not reproduced | see list, item 5b |
| sandbox demand scaling, border damping controls | Neither | no task | P4, design call yes |
| island platforms, multi-level stations, 5+ tracks with chosen roles | Filed | T-3444 | |
| stationsworld ?? | Neither (unclear) | not understood; ask her | |
| station logo / paint it | Neither | no task | P4, design call yes |
| detailed schedule settings, rush hour scheduling | Filed | T-3346, T-3542 | |
| custom drawable platforms (tapered) | Neither | maybe part of T-3444 | P4, design call yes |
| building demolition cost | Neither | no task | P4, design call yes |
| downloadable (not on web) | Neither | same as the offline item above | see list, item 13 |
| service suspended in severe weather | Neither | no task | P4, design call yes |
| make a custom fake city | Neither | no task | P4, design call yes |
| real world network showing just station names | Done | T-3570 (Show stations), T-3571 (names from nearby real station) | |
| advertisement | Neither | no task | P4, design call yes |
| loans | Done | T-2022 | |
| slopes slow trains | Filed | T-3027 | |
| generate a timetable, Japanese style, other lines on same track, line groups | Filed | T-3346 | |
| advanced fares by blocks, route-specific fares | Neither | T-3567 covers only a per-line multiple | P4, design call yes |
| what is limited express; do we support it | Done | answered in the feedback triage: per-line fare multiple (T-3567) plus a stopping pattern | reply to players only |
| multiplayer (sponsored compute, private server, websockets) | Neither | no task | P4, design call yes |
| better view of roads while placing track | Done | T-3129 (Snap to roads tints the roads rail may follow) | |
| quad stations when mousing over parallel tracks | Filed | T-3348 (T-3363 folded in) | |
| region redraws: Hokkaido into 4, Manila smaller, Muzaffarpur split | Neither | nearest T-1878 (planner never splits far-apart pieces), T-2111 notes | see list, item 18 |
| region redraws: Istanbul / exclave | Done | T-3488 | |
| region redraws: central Lisbon missing data | Done | T-617 (Lisbon rebuilt) | |

## Long term / deferred

| Item | Status | Evidence / id | Proposal |
|---|---|---|---|
| mode balance (commuter rail low, metro and tram high, bus low) | Filed (partly) | T-3378 covers buses and trams; commuter rail and metro have no task | see list, item 19 |
| improve trip computation time | Filed | solve-speed line: T-3513, T-2624, T-2880, T-3373 and others | |
| start in a city with an existing network to compete against | Filed | T-1444 | |
| more ideas to get a sense of progress | Filed (unsure) | nearest T-2872 (timeline), T-2150 and T-2200 (achievements) | |
| timeline / construction playback | Filed | T-2872 (Someday-like, after Anita asks) | |
| campaign mode, unlock cities, game-mode selector, easy/medium/hard starts | Neither | no task | P4, design call yes |
| campaign: flights, freight you run | Closed | Anita 2026-10-26 "not wanted: flights or freight you run" (TASKS line ~834) | |
| more dynamic demand like weather | Neither | demand complication needs her word | P4, design call yes |
| eventually car pathing | Neither | deliberately deferred ("too expensive") | P4, design call yes |
| settings: icons to find things | Neither | T-3681 is the search box | P4, design call no |

## Neither, in priority order

Priority guess weighs easiness as well as impact (her words). "Design" says whether she must decide first.

| # | Proposed task title | Pri | Design call |
|---|---|---|---|
| 1 | Put "join the leaderboards" at the top | P2 | no (copy of the leave button, 15 min) |
| 2 | Cap transfer fare against the 1000-dollar exploit | P2 | yes (what cap, or tie to points like T-3657) |
| 3 | Explain or relax the too-close-to-end error | P2 | yes (what 5 m guards; show the limit on the ghost, or shorten it) |
| 4 | Track levels -5 to +5 in advanced mode | P3 | yes (colours are hers: maroon, dark magenta, periwinkle, light purple, light violet) |
| 5 | Like button on an opened community game | P3 | no (her sketch: top right, under the day/night control) |
| 5b | Fix leaderboard flags (Sierra Leone shows Antarctica) | P2 | no (needs a repro first) |
| 6 | Make bus, tram, train and hsr obvious while building | P3 | yes (what to show) |
| 6b | Travel by mode overview in city and world view | P3 | yes |
| 7 | Bezier track placement, toggleable | P3 | yes (big) |
| 8 | Find why focused city loads take 3+ minutes on big saves | P2 | no (investigation, size first) |
| 9 | Check a sandbox save uploaded as normal in Community | P3 | no (needs the save) |
| 10 | Station improvements as one general interface | P3 | yes |
| 11 | Per-line money and ridership change list as a setting | P3 | yes |
| 12 | Leaderboard map view | P4 | yes |
| 13 | Offline and downloadable build with chosen cities | P4 | yes |
| 14 | Only-intercity game mode with feeders per city | P4 | yes |
| 15 | Friend requests and messages in Community | P4 | yes |
| 16 | Better errand modelling | P4 | yes (demand rule, needs her word) |
| 17 | Normalise traffic view widths across modes | P3 | no (unsure it is wanted over T-3580) |
| 18 | Redraw Hokkaido, Manila and Muzaffarpur zones | P3 | no (decision is the agent's) |
| 19 | Commuter rail and metro demand balance check | P3 | yes |
| 20 | Others, all P4 and design yes: sandbox demand scaling and border damping, station logo, tapered platforms, demolition cost, weather suspensions, fake custom city, advertising, fares by blocks, multiplayer, campaign mode, dynamic demand, car pathing, settings icons (design no) | P4 | yes |

Also: add the light load and "load fully" option to T-1446's goal, and raise T-2920 (two stops on one line in a split station) from P4.
