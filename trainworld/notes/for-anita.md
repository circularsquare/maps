# For Anita

## Open now (2026-11-04, maps-c9)

Answer "text acknowledged" per block, or name the lines to change.

### Tutorial (T-2263, done, on 8800 after the next build)
Shot: artifacts/trainworld/review/tutorial-card.png
- Step, new: "Hover the bar at the top to see how people travel"
- Step, new: "Show Traffic on the map"
- Step: "Reach 10% train share in New York" -> "Reach 2% train share in New York"
- Note on the bar step: "Green is walking, red is cars and other ways, blue will be your trains."
- Note on Show Homes: "Each bubble is the trips from homes there. Red ones mostly drive, green ones mostly walk, blue ones take your trains."
- Note on Click a red bubble: "Green bubbles still take trains once there are some. People there walk when it's dense, or a car and its parking cost too much." (I'd look at "once there are some")
- Note on Hover the dot by High: "Over 100% is crowded. To carry more, add cars under Train length."
- Note on Add trains: "If only a short stretch is crowded, more trains can cost more than they earn."
- Note on Show Traffic: "Lines get thicker with more riders. Where they're crowded, build more lines."
- Road chip, tutorial only: "85 crossings +$12.0M" -> "85 road crossings +$12.0M"
- Crowding dot hover, tutorial only, new line: "How full the trains get in the busiest hour, on the line's fullest stretch. Over 100% is crowded."
- Crowding dot hover, tutorial only, when at most half the stretches are past 100%: "Only a short stretch is crowded. More trains would run mostly empty, and may cost more than they earn."

### Track and station bars, mode frame (T-2267 + T-2270, design only)
Note: notes/T-2267.md. Shots: artifacts/trainworld/review/track-bar.png, mode-colours.png.
Questions: build it as T-2282 to T-2284? the optional cross "Stop (Esc)", the bars' 3 px top edge in the mode
colour, and Shift+1-7 re-levelling (T-2285)? keep the mode colours though they sit only 11-15 dE from the game's
own? (I'd say yes to all.)
- Track type: "Double / Quad / High speed" -> "Regular / High speed"
- Build tab Terrain button and its hover "Show terrain. Steep terrain costs more to build on." -> gone
- New descriptors: "Placing track", "Placing stations", "Deleting", "Selecting track and stations", "Exploring a trip", "New passage from Bedford Av"
- Kept descriptors, gaining a colour square: "New bus line: 3 points", "Route of Bus 3: 3 points", "New ferry line: 2 ports", "Editing stops on Z: 22 stops", "New station building: 4 corners", "Station building outline: 4 corners"
- Optional cross, hover: "Stop (Esc)"
- Level hover (T-2271's): "Viaduct (Shift+3)" / "1.0x cost, $75M per km" / "Upkeep $20k per km per day", 1x on light yellow

### Fares (T-2266, done)
- Optional hover on the fare controls, not built: "Riders pay the old fares until demand has settled". Want it?

## From 2026-10-26 (may be stale)

What we could work on next, from the tidied TASKS.md (2026-10-26). Tell me what to keep, cut or
move up; the agent that gets your answer updates TASKS.md and this file.

## Built overnight 2026-10-26 (to land and try on 8800)

T-1168 related lines (all three stages), T-1286 and T-1300 fares, T-988 cloud saves stage 3,
T-1270 stage 3 terrain tint, T-1287 waiting room, T-1284 crowding marks, T-1174 worker crash,
T-1207 to T-1210 and T-1348 slow-computer work, T-1283 service edits, T-1285 ocean as water,
T-1144 Kwai Chung. Items below that name these are done.

## Candidates

New countries
- **T-1239 Canada**: Montreal, Vancouver, Calgary, Edmonton and about 44 zones along the corridors, with Toronto re-solved among them. An agent day plus a few hours of builds, after the walk repair running now (T-1261).
- **Mexico, Turkey, Vietnam**: next in the queue after Canada. Not filed yet; each is a design look, then about a day of builds.
- **T-1144 Kwai Chung**: the Hong Kong district has no cells or zone. Hours. Under way now.

Demand feel
- **T-264 Superholidays**: Chinese New Year, Golden Week, Obon, Chuseok and Thanksgiving as long-distance days with a direction, trips to touristy places, and telling the player what to expect. You marked it a priority. Days.
- **T-1300 Fare what-if**: the fare panel shows what a fare change would do to riders and money a day. About a day.
- **T-1286 Fares for inter-city riders**: a country's fare scale moves fare income now but not how many inter-city riders choose rail. About a day.

Big networks
- **T-1223 Level of detail for demand**: zones you are not looking at solved coarsely, so a network across all of China fits in memory. Days.
- **T-1207 to T-1210 Slow computers**: smoother panning, and clicks, edits and demand results that stop freezing the page for half a second at a time. About a day each. Panning, demand results and clicks (T-1207, T-1208, T-1210) are under way now.
- **T-1028 to T-1030 Track edits on big networks**: placing track on a Japan-sized network sends only what changed, so a click costs tens of ms, not hundreds. Days.
- **T-1174 Demand worker crash**: a demand worker panicked once while the China network was built. Hours to a day, if it reproduces.

Interface
- **T-1168 Related lines diagram**: the line panel draws the lines sharing track with this one (expresses, branches, crossings). Your design from 2026-10-24. Days, in three stages.
- **T-988 Cloud saves, the last stage**: backups listed, a question on sign-out, Add on local-only cards. About a day.
- **T-1270 Terrain tint while building**: the four ground classes shown faintly while a build tool is on. About a day.
- **T-1287 Waiting room drawing**: a small building beside the platform with the waiting crowd inside. Hours to a day.

## Moved to Someday that you might want back

- T-1305 real networks sharing track (London Bridge is a tangle until then; you said later)
- T-750 add a city's real network to a sandbox game already running
- T-685 drawn city stamps at 10% rail share (the stars are done)
- T-651 unopened zones look fully red when zoomed out
- T-632 walking feels too strong to play against (you said try later)
- T-438 a giant metro takes most of a smaller neighbour's 80-300 km trips
- T-263 stadium events
- T-803 more station faces by time of day
- T-678 day and night tint on the base map
- T-784 lighter settings for low-memory phones

## Your calls

- **T-1189** (2026-10-26: not yet) replan the US, Puerto Rico, Europe and Taiwan by the compact-zone rule (China's is done)? It also does the Zhubei split, the wrap and ring fixes and Italy's small zones. A few days of agent time and builds; its first step (T-1307, carrying the gate relaxations over) is under way. You held it in case something else makes us rebuild again.
- **T-279** build routed drives for inter-city trips? Measured: the flat 1.25 rule is fine on average but off by up to 2x on single corridors (mountain roads, inlets). Days. I would wait.

## Still waiting on you

- **T-676**: the base map palette (screenshots in artifacts/trainworld/t676/).
- **T-1275**: the answer to "could a transatlantic line ever pay" (no) is in notes/T-1277.md, if you want to read it.
- **T-988**: the rewritten privacy page (`app/public/privacy.html`), if you have not read it yet.
