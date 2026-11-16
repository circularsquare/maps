# A tour of the trainworld code

What the code is made of, where each part lives, and how a city's raw data becomes the trains on a player's screen. Written for reading, not for finding a line of code. Counts are from 2026-11-16 (task T-3452).

![The parts and how data moves between them](artifacts/trainworld/review/code-tour.png)

## The short version

- There are **four big parts**: a Python **pipeline** that turns world data into one "pack" per zone, a Rust **simulation** (the track model and the demand model), the **browser app** written in TypeScript, and a small part of **her Go server** for accounts and shared games.
- **The Rust code is used three ways.** It is compiled twice for the browser (once for the clock worker, once for the demand workers), and also run natively on her machine by the pipeline, so the pipeline and the game do the same sums with the same code.
- **The browser splits the work across threads.** The main thread draws and handles clicks. A "clock worker" owns the player's network (track, lines, trains, money). One to three "demand workers" work out who rides. They pass snapshots to each other; none of them waits on another.
- **Data flows one way**: raw files, then packs on her disk, then R2 when published, then the browser fetches only the zones near the player's stations.
- **Everything together is about 320,000 lines**, tests and checks included. The biggest single parts are the Rust track model (49,000), the checks (46,000) and the map drawing (39,000).

## The parts

### 1. The pipeline: building zone packs (Python)

**What it does.** Turns world data into what the game needs for each place: how many people live and work in each small hexagon cell, how far they travel, where the schools and airports are, which roads and water are in the way, and trip counts between nearby zones. It runs offline, on her machine or on rented AWS machines for big waves.

**How it goes, in order.** First the *world plan* decides which places on Earth get a zone and what each zone holds (from the GHSL population grid). Then the *world run* builds each planned zone: downloads the OSM extract, counts people and jobs per cell, adds schools, universities, hospitals, parks and airports, solves the commute sums, adds walking and driving routes, and merges sparse cells. A *gate* checks each pack before it ships. A *joint solve* then works out trips between zones that are near each other, and the *market* step does the long trips (between cities, by car, plane or ferry). Country modules (France, Germany, India, Korea, Mexico and others) swap in a country's census numbers where they are better than the worldwide guess.

**A second job.** The same folder builds the **real networks** that sandbox starts open with: a city's actual metro and rail lines, routed onto the game's track model from GTFS timetables and OSM.

- Where: `pipeline/`, 124 files, 40,800 lines. Biggest: world_plan.py (3,400), tier0.py (the worldwide recipe, 1,800), world_run.py (1,700).
- Talks to: the Rust native tools (it calls them for the heavy sums), the raw downloads in `data/raw` (28 GB), its working files in `data/work` (50 GB), and writes the packs.

### 2. The packs, and where they live

A **pack** is one zone's data: a small JSON header and binary arrays, plus its outline, its place names and its row for the long-trip market. There are 4,045 zones, five files each, 5.7 GB in `data/packs`. Beside them: the real networks for sandbox starts and the real-rail routing graphs (44 MB) the track tool snaps to.

- **Locally**, the game reads packs from beside the page.
- **Published**, they go to Cloudflare R2, one folder per version, and the published page names its folder. noritetsu's rail map tiles (the "Real network" layer) also come from R2.

### 3. The simulation (Rust, `sim/`)

One Rust codebase, 86,900 lines, of which about 26,400 are tests. It has three halves:

**The track model** (`sim/src/track`, 49,200 lines). Everything about the player's network: the shape of track (curves, ramps, bridges, tunnels, crossings), what it costs, stations and buildings, lines and their stops, timetables, how fast trains run along each line, how many trains fit on shared track and how much junctions delay them, buses and ferries, how full each train and platform is through the day, and the save format. Edits go through it with undo. It runs inside the **clock worker** in the browser.

**The demand model** (`sim/src/demand`, 32,900 lines). Who travels where, and how. It reads the packs, works out commutes and everyday trips (shopping, school, visits, airports), the long trips between cities, and then for each trip whether it goes by train, car or on foot, which lines it rides, and what happens when trains are full. It runs inside the **demand workers**. Plain-words version of the rules: docs/demand-plain.md.

**Native tools** (`sim/src/bin`, 3,600 lines). Small programs built from the same code and run on her machine: the pipeline's commute and market sums, calibration against real cities, timing benches.

Plus one small extra: the schematic's heavy maths (1,000 lines) is also in Rust, for speed.

### 4. The browser app (TypeScript, `app/src`)

116,100 lines across 437 files. Built by Vite into the page; the Rust code comes in as two WebAssembly files.

**The main thread** (about 94,000 lines). What the player sees and clicks:

- **Map** (`map/`, 110 files, 39,200 lines): MapLibre draws the basemap; everything of ours (track, stations, trains, crowds, the demand views, crossing marks, labels, sounds) is drawn by a WebGL renderer of our own on a canvas on top. The build tools live here too: the track, station, bus and ferry tools, with the ferry's water routing in `ferry/`.
- **Panels** (`ui/`, 117 files, 21,300 lines): the dock and its tabs, the line and station inspectors (the biggest group, 8,100 lines), the start menu, saves, the account, leaderboards, the tutorial. Written with Preact.
- **Game state** (`game/`, 141 files, 19,900 lines): the latest picture the workers sent, and the small rules the panels share: the clock and holidays, zone names and realms, station complexes, line groups, the trip explorer. Also the hand-edited and generated lists (zone names, realms, start difficulty).
- **Schematic** (`diagram/`, 17 files, 10,400 lines): the network diagram (key M), laid out in its own worker.
- **Translations** (`i18n/`): 2,688 English strings with 12 other languages beside them.
- `main.ts` (870 lines) starts everything up.

**Workers** (`workers/`, 40 files, 21,900 lines). The glue between the page and the Rust code:

- **Clock worker** (clock.worker.ts, 4,900 lines, with the track model inside): owns the network, applies edits, answers the track tool's previews, keeps the money and the clock, saves and autosaves, and talks to the account for cloud saves.
- **Demand pool** (demandClient.ts, 5,000 lines, which runs on the main thread, and demand.worker.ts, 2,600, inside each worker): decides how many workers a device gets (1 to 3), which zone packs each opens, splits the network into parts that share no rider, and collects the answers.
- **Smaller workers**: the real-rail route finder for the track tool, the schematic's layout, ferry water, demand sums, map shading, gzip for saves.
- **Saves** (saveFile.ts, 900 lines): the save format and the browser's own storage (IndexedDB): save slots, backups, and the last demand result kept beside each save.

### 5. The server (Go, in `shonei-server`)

Her own server, shared with noritetsu and her other projects. The trainworld part is 8,100 lines in 11 files (plus 5,900 lines of tests): sign-in (the noritetsu accounts), games kept on the account with backups, leaderboards and achievements, community schematics, feedback threads with replies, and play-time counts. The game works fully without it; signed out, saves stay in the browser.

### 6. Building and publishing

`npm run build` compiles the Rust twice, bundles the TypeScript, and copies the packs in. The orchestrator rebuilds her copy on 8800 from a snapshot (snapbuild). Publishing runs tools/deploy.py: packs to R2, the page to the website repo.

### 7. Around the game

- **Checks** (`checks/`, 320 scripts, 45,700 lines; 42 more retired in `checks/old`): the smoke test, browser checks that click through the game, timing checks, the pack gate, and the linters.
- **The board** (`board/`, 3,200 lines, twboard on 8800): this page. Tasks, notes, the changelog, and her inbox to the orchestrator.
- **Tools** (`tools/`, 7,800 lines): publishing, the changelog, the AWS scripts (with a guard that refuses raw AWS changes), feedback, server checks, country census readers.
- **Docs and notes**: SPEC.md, `docs/` (32 files, 277,000 words), the wiki for players, and a note per task in `notes/`.

### Sizes at a glance

| Part | Where | Lines |
|---|---|---|
| Track model (Rust) | sim/src/track | 49,200 |
| Checks | checks/ | 45,700 |
| Pipeline (Python) | pipeline/ | 40,800 |
| Map drawing and tools | app/src/map | 39,200 |
| Demand model (Rust) | sim/src/demand | 32,900 |
| Worker glue | app/src/workers | 21,900 |
| Panels | app/src/ui | 21,300 |
| Game state | app/src/game | 19,900 |
| Schematic | app/src/diagram | 10,400 |
| Server, trainworld part (Go) | shonei-server | 8,100 + 5,900 tests |
| Tools | tools/ | 7,800 |
| Native Rust tools | sim/src/bin | 3,600 |
| The board | board/ | 3,200 |

Lines are lines in files, blank lines and comments included. The Rust figures include their tests (26,400 lines in all).

## Three everyday moments

### Opening a save

1. The page draws the map and the start menu, and nothing else loads yet.
2. The player picks a save. The page starts the demand workers first, then the clock worker.
3. The clock worker reads the save from the browser's storage; when she is signed in, the copy on the account is checked too. A save holds only what the player built and chose: track, stations, lines, timetables, fares, money and the clock. Population, trips and riders are never saved; they are worked out again from the packs.
4. The Rust track model rebuilds the network from those bytes. Line rates were saved too, so it does not have to settle them again.
5. The clock worker sends the page a picture of the network and this hour's train runs. The renderer draws them, and the trains start moving, each one placed by the clock along its line.
6. At the same time it sends the network to the demand side. The page works out which zones the stations can reach, fetches those packs (from beside the page, or from R2), and hands a copy to each demand worker.
7. If the last demand result for this exact network was kept beside the save, the figures show at once. Otherwise the loading card holds the clock until the first full answer lands.

### Drawing a track

1. The player picks the track tool. That tool lives on the main thread, in the map part.
2. As the pointer moves, the tool asks the clock worker for a preview (a "ghost"). The Rust track model works out the curve, the price, and what the track crosses (roads, water, other track). Only one preview is worked on at a time; a newer one replaces any that has not started. Near real rail, the real-rail route finder (its own worker) suggests a path that follows it.
3. A click sends an edit. The track model applies it (with undo), takes the money, and the clock worker sends back only the rows that changed. The renderer redraws only the changed strokes.
4. Plain track that no line uses yet changes nobody's trip, so demand does not run again. The worker even waits until the pointer rests before sending its full picture, so a long drag stays smooth.

### Demand working itself out again

1. An edit that changes service (a new line, new stops, more trains an hour) makes the clock worker send a fresh network snapshot.
2. The demand pool on the main thread splits the network into parts that share no rider, and sends only the parts the edit touched. Each demand worker owns some of the day's five periods.
3. In each worker the Rust demand model: takes the zone's trips (commutes from the packs, everyday trips, long trips between cities), works out the walk to stations, searches the network, picks train, car or foot for each trip, puts riders on lines, and then runs a few crowding rounds, in which full trains leave some riders behind and they choose again.
4. Early rounds can show for lines that are new; everywhere else the old figures stay until the new final answer lands, so numbers do not flicker.
5. A small sums worker adds the answers up for the views; the panels and the map read them. The riders also go back to the clock worker, which works out how full each train and platform is through the hour and what the fares bring in.

## Odd things

Things that look silly, doubled, oversized or tangled. None of these were changed.

1. **Two giant files at the centre.** demandClient.ts (4,993 lines) and clock.worker.ts (4,882) are the two biggest files in the whole project. The clock worker file does the network, previews, money, autosave, cloud saves and save pictures in one place, and 79 app files talk to it. The demand pool's 5,000 lines run on the main thread, not in a worker.
2. **The same job written twice, kept in step by hand or by a test.**
   - The schematic's sheet maths is in Rust (diagram.rs, 1,000 lines) and also in JavaScript (sheetLines.ts), meant to match "bit for bit"; the JavaScript one is kept as the fallback.
   - The real-rail route finder is in the game (realRoute.worker.ts, 658 lines) and copied into the pipeline (realroute.py, 229 lines, "the track tool's router ported").
   - Car speeds on local streets are written down twice in Rust (once for buses, once for demand), with a test to keep them equal.
3. **A pipeline step whose output nothing reads.** nocar.py adds each cell's households without a car to every pack, and the docs still say to rerun it after rebuilds, but demand stopped reading it at T-854.
4. **Checks named after tasks.** 283 of the 320 standing checks are named after the task that made them (t3137-lanes-parity, and so on), with 42 more in checks/old. At 45,700 lines they are bigger than the pipeline, and it is hard to tell which ones still guard something.
5. **Code kept in notes.** notes/ holds 351 scripts (26,400 lines), mostly in the archive: one-off measurement scripts that stayed with their task note.
6. **Docs longer than the code they describe.** docs/ is 277,000 words, about three long novels. The code map alone is 24,000 words of long sentences full of task numbers, which makes it slow to scan for "where does X live". TASKS.md is 1,432 lines.
7. **Ten scripts for real networks.** Building a city's real network for sandbox starts takes ten Python scripts (realnet, realcorridors, reallevels, realshared, realauto, realnet_n02 and four more), 5,400 lines, about an eighth of the pipeline for one feature.
8. **One very long pipeline file.** world_plan.py is 3,437 lines in one file, the biggest Python file.
9. **Test files bigger than the code.** The track model's main test file (wasm/tests.rs) is 3,693 lines, the biggest Rust file; the intercity tests are 2,671.
10. **Accounts named after another project.** Trainworld sign-in goes through the server's /noritetsu/ routes, since the accounts are noritetsu's. It works, but anyone reading the server sees "noritetsu" for every trainworld login.
11. **Small leftovers.** mock/ is a design mock last touched 2026-10-09 that loads an old MapLibre (5.24) from the web while the game is on 6.13. Market place numbers 8 and 11-13 in the registry have been unused since T-1113.

## How the counts were made

Lines counted in every file git tracks (TypeScript, Rust, Python, JavaScript, CSS, HTML, scripts), tests included, notes/ counted apart. Pack and disk sizes measured on her machine. The server figures are the trainworld and leaderboard files in shonei-server.
