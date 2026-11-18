# Lines

Anita's view of the work: each line is one conceptual unit, and every open task in TASKS.md belongs to exactly one
line through its `{L:<slug>}` tag. Areas are the board's rows. Format and the board's rules: notes/T-3389.md
"Format (decided)". A new task joins an existing line or adds one here; set `done:` when a line's last task closes.

## Track and stations {area:track}

- quad-track | AA/BB quad track | Splitting a double into a four-track stretch and back, and choosing which pair a line stops on. Stage 1 (per-direction stops) landed 2026-11-16; the rest is paused until the track rework (T-3444) is discussed. | update:track-stations
- track-kinds | What single, double and quad mean | One clear picture of single, double and quad track and where crossovers come from, for players and for us. Her answers are recorded and it is parked until she reopens it. | update:track-stations
- single-track | Single and double track fixes | Bugs and rough edges on single and one-way track, and on doubling up, mostly from player reports. | update:track-stations
- stops | Stops and stop order | Which stations a line stops at and in what order: stopping twice at one station, reordering, waypoints. | update:track-stations
- station-placement | Station placement | Placing stations: room on ramps, snapping to a good spot, clearer refusals, the name a new station shows. | update:track-stations
- track-drawing | Track drawing tool | Drawing and dragging track: curves, ramps, bridges, snapping, and copying blueprints. | update:track-stations
- rebuild-track | Rebuild built track | Changing built track's level or kind in place. Rebuilding landed; the preview before you pay is left. | update:track-stations
- trams | Trams | Trams after their release: snapping to real light rail, shorter ramps, crossing water on a road bridge.
- buses | Buses, BRT and ferries | Bus balance against trams and a possible BRT upgrade, long distance buses, and small bus and ferry fixes.
- track-speed | Track speed, slopes and maglev | Ideas that change how fast trains go on track: slopes slowing trains, a speed cap in mountains, maglev.
- terrain | Terrain | The terrain under track prices: worldwide blocks, hexagon cells, water depth.
- real-networks | Real networks | The real city networks a game can start with: missing lines, slow loads, and modes built on them.

## Scheduling and fares {area:schedule}

- timetables | Timetables | Timetables and better scheduling: the schedule panel, weekend and rush hour service, trains meeting on single track. | update:scheduling-fares
- passing-tracks | Passing and overtakes | Expresses passing locals: passing loops, holds that line up both ways, three-speed lines. | update:scheduling-fares
- trains-drawn | Trains drawn sensibly | Trains on the map should never pass through each other or run far behind their timetable. | update:scheduling-fares
- line-rates | Line rates | How a line's trains an hour settle when lines share track, and the bugs where that drifts.
- fares | Fares | Fare levels and fare controls, starting with high-speed lines losing money. | update:scheduling-fares | done:2026-11-17
- money | Money history | Money over time: per-line history, bus money by region, and how much of it a save keeps.

## Demand and world data {area:demand}

- airports | Airport demand | How many trips airports make and where their jobs sit; players say several airports are too low.
- attractions | Attractions and beaches | Where attractions come from and filling the gaps outside Europe, Japan, Korea and the US, beaches included.
- wealth-split | Wealth split | Rich and poor riders choosing differently, so some take the express and some the local.
- jobs-places | Jobs, schools and hospitals | Where people work and study, by country: census job sources, industrial areas, campuses, hospitals.
- census-totals | Census population | Putting each country's people on its own census totals instead of the satellite estimate.
- region-redraws | Region redraws | Region shapes and names: split or odd regions, regions in pieces, empty units, coasts and islands.
- borders | Trips across borders | How many people cross each border, twin towns and one-way flows.
- new-countries | More countries | Countries not built yet: Myanmar, Laos and Timor-Leste are what is left of Southeast Asia.
- world-rebuild | Next world rebuild | Fixes that wait for the next rebuild of every pack, gathered so they all ride along.
- islands | Islands and ferries in trips | Trips between islands and across water: who drives, who flies, which ferries carry cars.
- car-free | Car-free homes | How many homes have no car, by country and region. | done:2026-11-16
- route-choice | Walking and route choice | How riders pick between walking, changing lines and the train, including reports of odd walks.
- ridership-checks | Checks against real ridership | Comparing the game's riders with real counts in Japan, New York and elsewhere.
- demand-ideas | Demand model ideas | Bigger changes to how trips are made, most parked or waiting for her call.

## Schematic {area:schematic}

- schematic | Better schematic | The schematic map as a whole: layout on big maps, city centres, buses and ferries, settings.
- schematic-rivers | Schematic rivers and water | Rivers drawn as one course on the right side of stations, and sea where it really is. Paused by her.
- schematic-layout | Schematic layout fixes | Smaller layout faults: crossings and parallel lines lost, stations on the wrong side, names at busy stations.
- schematic-speed | Schematic speed | How long the schematic takes to draw on big maps like all of Japan.

## Community and accounts {area:community}

- community | Community | Sharing games with other players: the community saves, view only links.
- feedback | Feedback | The in-game feedback form and replies: screenshots, phone layout.
- twstats | Play stats | The twstats page: who is playing, where, and for how long.
- open-source | Open source and mods | The public trainsworld repo, contributors' pull requests, and mod support.
- accounts-server | Accounts and server | Accounts, security, server backups and the website.
- achievements | Achievements | Achievements and other marks of progress: sea crossings, named crossings, city stamps.

## Interface {area:interface}

- translations | Translations | The game in more languages, and place names in the player's language.
- help-tutorial | Help and tutorial | Help panels and tutorial hovers that explain demand and the game's words.
- map-look | Map look | How the map looks: day and night, station faces, zones not loaded yet, sounds, small drawing faults.
- world-map | World map | The zoomed-out world map: start cameras, labels, the real rail layer at far zooms.
- small-fixes | Small interface fixes | Panel and hover fixes too small for a line of their own.

## Performance {area:performance}

- memory | Memory | Memory use of big games, sharing zone data between demand workers, phones and 32-bit browsers.
- edit-speed | Edit speed | How long an edit takes on big networks like London, all of Japan and North America.
- solve-speed | Demand solve speed | How fast the demand answer comes back after an edit or a click.
- solve-consistency | Same answer every time | The demand answer should not depend on load order or history.
- loading | Loading and download size | Load times, download sizes and loading bars.
- frame-rate | Smooth frames | Hitches while zooming and panning, and frames when little moves.

## Ops and code health {area:ops}

- board | Anita's board | This board, and her answers reaching the orchestrator. | done:2026-11-16
- cleanup | Code and docs cleanup | Splitting big files, removing old paths, shorter agent docs.
- checks | Checks and agent tools | Standing checks that fail or test the wrong thing, and agent tool fixes.
- world-tooling | World build tooling | The world run and join: speed, failures, preflight checks.
- wiki | Wiki | The player wiki that agents read instead of the docs.
