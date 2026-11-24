## 2026-11-18, before v0.4.2

Reviewed: the 44 shown items among c001-c053. Nine were already hidden before this pass and not re-judged (c049, c027, c021, c017, c014, c010, c009, c004, c003).

Hidden in this pass:
- c051: the clear button in search boxes shows whenever there is text in them (tiny polish)
- c037: fixed old copies of the clock piling up in the top bar (rare, page-translation oddity)
- c030: moved the join control to the top of the leaderboards (minor layout move)
- c028: fixed community previews showing "Could not load this network" when switching tabs quickly (rare race)
- c020: fixed the new game city pick showing 0 intercity trips before they are calculated (minor display)
- c011: fixed the start menu covering the out of memory message after a game failed to load (rare failure path)

Reworded: none.

## 2026-11-19, v0.4.3

Ids restarted at c001 after the v0.4.2 publish, so the lists above are a different numbering. Reviewed: all 29 shown items c001-c029. Items marked (edited) by Anita were kept without re-judging.

Hidden:
- c028: stuck trip calculation restarts once on load (failure path)
- c024: crash at startup when the browser drops graphics (rare)
- c023: settings memory reading too high with several workers (diagnostic)
- c014: renamed station showing old name at some zooms (minor)
- c012: wrong fare on a bent fare curve (bug in a feature new in this release)
- c010: unused flat junctions and crossings no longer show a mark (tiny)
- c007: second tab unable to go back to the main menu (rare)
- c004: track ending just short of a bare track end refused (minor)
- c003: Copy shown instead of Sandbox on a kept copy (minor)

Reworded:
- c022: "fixed long games using more memory every game day; past about 2 years, money history older than 4 months is kept by month, so a 50 year game uses 1 GB less memory" -> "fixed long games using more memory every game day, so a 50 year game uses 1 GB less memory"

Areas: all items have one.
Proposed highlights: c021, c011, c015, c020, c005, c013

## 2026-11-20, before next publish

Ids restarted again; this is a new numbering. Reviewed: all shown items c001-c027 (c001, c002, c003, c007, c009, c010, c014, c016, c018, c025 are Anita's edits, kept without judging; c005 and c006 were already hidden).

Hidden:
- c023: calculating trips bar restarting during a long calculation (minor progress bar glitch)
- c015: panning the schematic so far that nothing showed (rare)
- c013: trip routes and gap flash off place on very large screens (rare)
- c012: schematic drawing indicator going away early (minor)

Kept (judged): c027, c026, c024, c022, c021, c020, c019, c017, c011, c008, c004.

Reworded: none.

## 2026-11-20, second pass (c028-c058)

Reviewed: shown items c028-c058 not covered above (Anita's "(edited)" items kept, no merging needed; c053, c047, c046, c035, c031 were already hidden).

Hidden:
- c054: bus tool shows each stretch's speed (tooltip-like detail)
- c045: default timetable for new long ferry lines (fiddly)
- c043: undo not restoring other lines' trains (rare edge)
- c038: station ghost turning red briefly (minor)
- c029: trains bought by adding a stop not counted in a region in the money panel (minor)
- c056: train lines accepting ferry ports as stops (obscure)

Reworded (reword marks them edited):
- c058: "added a BRT upgrade for bus lines: bus lanes so its buses run at least 35 km/h, paid for only where the route is in town" -> "added a BRT upgrade for bus lines: bus lanes make its buses run at least 35 km/h, paid for only where the route is in town"
- c057: "... on trunk roads and motorways at most 1.4x their street speed" -> "... on trunk roads and motorways in town at most 1.4x their street speed"

Sorted by area (track, demand, scheduling, fares, schematic, community, other), features before fixes, hidden last.
Highlights: c037 (tram joins rail), c051 (plain buses), c014 (timetable zoom), c010 (schematic cities list), c022 (new countries), c048 (small and large ferries), c058 (BRT).


## 2026-11-24, before v0.4.6 (agent clreview)

Ids restarted after the v0.4.5 publish. Reviewed: all of c001-c088 (c056, c032, c021, c015, c013, c012, c007 were already hidden; Anita's "(edited)" items were kept unless folded into a merge).

Hidden:
- c070, c076, c078, c081, c083, c084, c085: single edit-speed lines, folded into c080
- c086: folded into c064 (same bus-merge speedup, T-4158 builds on T-4094)
- c068: folded into c035 (station tool)
- c018: folded into c017 (brt lanes)
- c031: overlaps c019 (memory on big networks)
- c029 (credits folded), c028 (lines list money on narrow panel), c040 (hover staying up), c067 (labels over tutorial card): tiny
- c025 (simulation stopped message), c042 (out-of-memory load retry), c049 (bubbles after graphics loss), c057 (startup after graphics crash), c071 (edits failing after model stop): rare failure paths
- c054 (community search misses), c077 (curved-track station over water), c079 (track end almost in line): minor or obscure

Reworded (old -> new):
- c080: "edits on a big world show up faster, placing or deleting a station and deleting a line most of all" -> "edits on a big world show up at the click: placing or deleting a station, deleting track or a line, constructing, undoing, and changing a line's stops, colour or settings"
- c064: "bus-heavy games calculate trips 20-40% faster and use 20-40% less memory" -> "bus-heavy games calculate trips 20-40% faster and use 20-40% less memory, and about 20% faster still on belgrade (16 min to 13 min)"
- c035: "the station tool picks train station, tram stop, BRT station or port; BRT stations snap to bus routes, ports no longer need a ferry line, and stations can go on open ground again with two clicks" -> same plus "and its Auto choice picks from what is under the pointer" and "show trips within 600 m"
- c017: "a brt lane takes 100 buses an hour each way, and lines sharing a full lane run fewer buses" -> "a brt lane takes 100 buses an hour each way, lines on the same road share its lanes, paid for once with the upkeep split, and lines sharing a full lane run fewer buses"
- c027: "added the Discord and leaderboards links to Community, and report is now a flag icon that asks which problem before sending" -> "added the leaderboards link to Community, and report is now ..." (the discord link was removed by c056)

Sorted by area, features before fixes, hidden last. 58 shown items.
Highlights: c080 (edit speed), c064 (bus-heavy speedup), c088 (reversal capacity), c035 (station tool), c043 (busways), c022 (brt stations), c051 (plain buses and expresses).

## 2026-11-24, second pass (agent clreview2)

Reviewed: c089, c090, c091, c092, c019 (reworded by its task). Kept: c089, c091, c019.

Hidden:
- c090: big games use 10-60 MB less memory (overlaps c019 and c063, memory items)
- c092: riders on new lines changing again after trips were calculated (rare edge)

Merged: none. Reworded: none. Sorted: c091 among track fixes, c089 after c065 in performance, hidden last.
Highlights: unchanged (7).

## 2026-11-24, third pass (agent clreview3)

Reviewed: c093-c105 (c094 was already hidden; c019 and c089-c092 were done in the second pass).

Kept: c093, c095, c096, c097, c102, c103, c104, c105.

Hidden:
- c098: folded into c104 (track ghost fix)
- c099: folded into c104 (track ghost fix)
- c100: loading time left jumping, too minor
- c101: grouping, ending a line at a stop and bus/ferry route edits showing up at once; same edit-speed theme as c080

Merged: c098 + c099 + c104 into c104. c105 stays separate (it is about edits acting on the old map, not the ghost).

Reworded (old -> new):
- c104: "fixed the track ghost wiggling while drawing and vanishing for a moment after a click" -> "fixed the track ghost wiggling, jumping or pointing off while drawing, after a click or after zooming"
- c105: "fixed quick edits right after another acting on the old map, like deleting the wrong track" -> "fixed edits made right after another acting on the old map, like deleting the wrong track"
- c097: "big games no longer recalculate all their trips now and then to free memory" -> "big games no longer recalculate all their trips now and then" (memory claim dropped, overall memory went up)

Sorted by area (track, demand, scheduling, fares, schematic, community, performance, other), features before fixes with highlights first, hidden last. 55 shown items.
Highlights: unchanged (7): c043, c035, c051, c088, c080, c048, c072. (c022 and c064 still carry a highlight flag but are hidden.)
