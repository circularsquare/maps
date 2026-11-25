# T-4267 measuring: how long until a click is confirmed, phase by phase (agent editmeasure, 2026-11-25)

**Summary for Anita**
- One click on a quiet game is confirmed on your Asia save (the game knows it went through and the page has heard) in 0.05-0.16 s for a station off any line, a track delete, a passageway, a new line or a deleted line, and 0.15-0.6 s for a station on a line's track.
- Stations clicked 0.3 s apart: each later one is confirmed 2.2-4.7 s after its press. It waits for the click before it, then behind that click's whole map rebuild, which a message from the page (income, rides, a panel's figures) makes run in one block of 0.4-2.5 s.
- Inside one edit, the checks that can refuse it (does it fit, does it cross, its price, the cash) take 5-15 ms on your save (twice about 40). The other 20-480 ms work out the trains: the service on shared track, the weekend timetable, train counts, departures and timings.
- A confirm needs at least: the edit put on the track, its geometry and crossing checks, its price and the cash. These read only what the click touched, apart from two price sums and the cash count of about 1 ms each. The station ghost already runs exactly this as its check, 8-15 ms on your save.
- One check is not local: the cash for extra trains on headway lines that share the track, which needs the train counts worked out. The ghost skips it today and the click is refused for money instead; a fast confirm would keep that lie or check the cash right after.
- So a confirm within 100 ms is possible on your save: answer after those checks, then work out the trains. Nothing is built that way yet; it needs the order of work changed, and the previous click's rebuild kept from blocking the next click.
- After the confirm, the map and panels catching up still take what they take now (0.4-2.7 s for one click), and trains on lines far along shared track really do change.
- Found on the way: a station on built track is charged a tiny re-pricing of the split track ($0.001 to $220), so it counts as paid and its answer waits for its rows (0.2-0.5 s), and the next click waits with it.

## How it was measured

- Her save asianita-day568 (242 lines, 2,698 stations, 8,746 edges, 9,215 nodes), my tree (master 4b5088233, with T-4149 stage 2, T-4249 and the T-4262 stage in), my Vite on 6163, "Use more cores" off (the setting's default; the check harness gives the page 3 demand workers on its 4 pinned cores), clock paused, trips final before the first edit. Each edit made the player's way (station tool presses on the track, the delete tool and its question, the Lines tab's New line, the line panel's Delete line); the passageway through the join button's own call (`links.ts` `unlink`). Scratch driver and analysis: artifacts/trainworld/t4267-measure/ (measure.mjs, analyze.mjs, both runs' events).
- Instrumentation in the tree only, taken out after (artifacts/trainworld/t4267-measure/instrumentation.patch): a timer per phase in the track model (read after each apply), the worker's stamps around each edit and every task it ran (messages, follow-up pieces, timers), and the page's stamps (post, arrivals, takes, frames, long tasks). Worker and page stamps are absolute ms; between the two threads they agree to about 25 ms.
- Run 1: 3 other cores claimed at the start and the end. Run 2: 3 at the first edit, 6 at the last, so its later edits may be noisy. Run 1's presses were each awaited, which spaced the click run 0.5-1.5 s apart; run 2 sends each press at its own time (0.3 s apart), as the edit_bench row now does. One sample per edit per run.
- Both runs use a DEV build (Vite dev, needed for the worker's laps) and the tree's agent WASM (no LTO). What she plays is faster in both; by how much was not measured. Every edit_bench figure is taken the same way, so they compare.

## Every edit at a glance (ms from the press, run 1 / run 2)

"Confirm" is the worker's word (`applied`, accepted) or its refusal arriving in the page. "Answer" is the `edited` message the tool waits for before it lets the next click go and shows its toast.

| edit | confirm (press to the word in the page) | answer (the tool lets go) | built look drawn | the apply (worker) | of it, the checks |
|---|---|---|---|---|---|
| first station on a line's track | 583 / 498 | 880 / 879 | 2048 / 2695 | 489 / 421 | 9 / 6 |
| second station 0.3 s after it | 2065 / 3641 | 2081 / 3653 | 3394 / 5316 | 435 / 531 | 12 / 9 |
| station on a line's track, alone | 155 / 154 | 288 / 320 | 880 / 1173 | 137 / 137 | 10 / 4 |
| station off any line | 59 / 50 | 66 / 56 | 451 / 385 | 39 / 35 | 9 / 5 |
| delete built track on a line (from Remove) | 137 / 159 | 2593 / 2182 | 2592 / 2180 | 63 / 27 | 42 / 6 |
| remove a passageway | - / 137 | - / 661 | - / 660 | - / 18 | - / 6 |
| new line from the Lines tab | 101 / 79 | 242 / 226 | - (nothing drawn) | 93 / 70 | 14 / 15 |
| delete a line on shared track | 53 / 46 | 526 / 703 | 500 / 694 | 29 / 24 | 5 / 6 |

The first station's apply is 3x the lone one's: its edge carries 3 lines (Hikari among them) whose service, capacity and settle reach about 150 lines' holds (449 line updates over 3 service passes); the lone one's edge carries 1.

## Per edit

Rows under 1 ms in both runs are left out. "Scales with the save": the phase's own loops run over every line, edge, node, train or resource of the save (section "Where each phase's time grows" has the loops); "no" means it reads only what the edit touched. "Needed for the confirm": whether the phase decides accepted or refused.

### A first station on a line's track (edge 1818, 5.0 km, 3 lines)

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 36 / 16 | no: waits on earlier clicks | no |
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | 27 / 4 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 9 / 6 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 8 / 6 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 1 / 1 | yes (light) | no |
| Apply: service rebuilt (`update_scoped`: lines, capacity, holds) | 153 / 78 | yes (loops) and the touched lines' whole length | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 139 / 211 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: departures placed (weekdays and weekends) | 44 / 14 | yes (weekends copy every line) | no |
| Apply: retime | 106 / 71 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | 5 / 14 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 5 / 4 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 19 / 16 | no | no |
| Word and map patch (worker) | 29 / 55 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| Page takes the word and patch (`pendingWord`, `takePatch`) | 12 / 14 | yes (the tools' map: every edge's points copied, every record re-read) | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 583 / 498 |  |  |
| Answer in the page (the tool lets go, the toast) | 880 / 879 |  |  |
| Follow-up: rows state (worker) | 325 / 497 | yes | no |
| Follow-up: drawn network (worker) | 439 / 1014 | yes | no |
| Page takes the state | 73 / 73 | yes | no |
| Renderer `setNetwork` | 49 / 52 | yes | no |
| Built look drawn (from the press) | 2048 / 2695 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 9 / 6 | yes | no |

### A second station 0.3 s after it, on the same edge, while the first's follow-up runs

What the worker ran before it: run 1 whole follow-up 913 inside: income 913, stationName 347, chargeShares 199; run 2 whole follow-up 1774 inside: transfers 2147, chargeShares 284 (ms). The page held it 0.4-0.6 s first: the station tool keeps a click while the one before is out (`toolWait.ts` 63-67, `clicksOut`), and the first station's answer waited for its rows (the re-pricing remainder below).

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 405 / 637 | no: waits on earlier clicks | no |
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | 1211 / 2370 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 12 / 9 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 7 / 39 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 1 / 1 | yes (light) | no |
| Apply: service rebuilt (`update_scoped`: lines, capacity, holds) | 115 / 36 | yes (loops) and the touched lines' whole length | no |
| Apply: weekend service copied (`mirror_weekend`) | 87 / 121 | yes | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 6 / 30 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: departures placed (weekdays and weekends) | 28 / 120 | yes (weekends copy every line) | no |
| Apply: retime | 134 / 150 | yes (keys of every cluster) and the bundle | no |
| Apply: undo record (`line_rates_all` twice, `had`, `keys`) | 2 / 1 | yes (every line) | no (undo) |
| Apply: after (`note_changes`, fleet) | 2 / 2 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 2 / 2 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 40 / 21 | no | no |
| Word and map patch (worker) | 14 / 100 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| Page takes the word and patch (`pendingWord`, `takePatch`) | 16 / 12 | yes (the tools' map: every edge's points copied, every record re-read) | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 2065 / 3641 |  |  |
| Answer in the page (the tool lets go, the toast) | 2081 / 3653 |  |  |
| Follow-up: rows state (worker) | 266 / 495 | yes | no |
| Follow-up: drawn network (worker) | 743 / 699 | yes | no |
| Page takes the state | 85 / 100 | yes | no |
| Renderer `setNetwork` | 25 / 35 | yes | no |
| Built look drawn (from the press) | 3394 / 5316 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 3 / 6 | yes | no |

### A station on a line's track, alone (edge 3212, 5.9 km, 1 line)

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 9 / 8 | no: waits on earlier clicks | no |
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | 0 / 1 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 10 / 4 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 6 / 13 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 1 / 1 | yes (light) | no |
| Apply: service rebuilt (`update_scoped`: lines, capacity, holds) | 25 / 25 | yes (loops) and the touched lines' whole length | no |
| Apply: weekend service copied (`mirror_weekend`) | 21 / 21 | yes | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 4 / 4 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: departures placed (weekdays and weekends) | 16 / 21 | yes (weekends copy every line) | no |
| Apply: retime | 34 / 35 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | 2 / 1 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 1 / 1 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 17 / 10 | no | no |
| Word and map patch (worker) | 7 / 7 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| Page takes the word and patch (`pendingWord`, `takePatch`) | 9 / 10 | yes (the tools' map: every edge's points copied, every record re-read) | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 155 / 154 |  |  |
| Answer in the page (the tool lets go, the toast) | 288 / 320 |  |  |
| Follow-up: rows state (worker) | 118 / 171 | yes | no |
| Follow-up: drawn network (worker) | 259 / 333 | yes | no |
| Page takes the state | 43 / 48 | yes | no |
| Renderer `setNetwork` | 24 / 23 | yes | no |
| Built look drawn (from the press) | 880 / 1173 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 3 / 99 | yes | no |

### Four stations 0.3 s apart on a line's track (edge 5128, 7.9 km, 3 lines; the click run)

| click | page holds it | worker busy before it starts (what ran) | its apply (checks / the rest) | confirm | answer |
|---|---|---|---|---|---|
| 1, run 1 | 11 | 0 | 376 (11 / 366) | 395 | 775 |
| 1, run 2 | 10 | 123 | 192 (14 / 178) | 332 | 595 |
| 2, run 1 | 15 | 2513 (whole follow-up 2514 inside: query:today 2595, riders x6 71, regionShares 68, previewStation 42, chargeShares 27) | 115 (7 / 108) | 2651 | 2859 |
| 2, run 2 | 309 | 1682 (whole follow-up 1591 inside: income 1591, previewStation 101, chargeShares 28) | 154 (8 / 147) | 2155 | 3262 |
| 3, run 1 | 2337 | 1353 (whole follow-up 1218 inside: income 1218, release 112, riders x3 58) | 50 (6 / 43) | 3748 | 3873 |
| 3, run 2 | 2963 | 926 (whole follow-up 371 inside: query:blueprintQuote 372, riders x9 296, previewStation 232, regionShares 26, chargeShares 26) | 86 (6 / 79) | 3980 | 4204 |
| 4, run 1 | 2394 | 519 (whole follow-up 450 inside: regionShares 504, chargeShares 39) | 54 (6 / 48) | 2973 | 2984 |
| 4, run 2 | 3936 | 635 (whole follow-up 544 inside: transfers 611, chargeShares 46, regionShares 34) | 81 (37 / 45) | 4656 | 4665 |

The edit_bench row on the same save (run 3, NOISY: 6 other cores claimed; edge 1482, 3 lines): confirm 1,370 / 3,019 / 6,591 / 8,763 ms, answer 1,393 / 4,014 / 6,661 / 8,775 ms.

### A station off any line (on blueprint track drawn just before, no line on it)

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 10 / 10 | no: waits on earlier clicks | no |
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | 5 / 1 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 9 / 5 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 4 / 10 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 1 / 2 | yes (light) | no |
| Apply: weekend service copied (`mirror_weekend`) | 1 / 1 | yes | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 4 / 3 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: retime | 2 / 2 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | 1 / 1 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 2 / 2 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 14 / 10 | no | no |
| Word and map patch (worker) | 5 / 3 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| Page takes the word and patch (`pendingWord`, `takePatch`) | 5 / 4 | yes (the tools' map: every edge's points copied, every record re-read) | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 59 / 50 |  |  |
| Answer in the page (the tool lets go, the toast) | 66 / 56 |  |  |
| Follow-up: rows state (worker) | 156 / 88 | yes | no |
| Follow-up: drawn network (worker) | 100 / 138 | yes | no |
| Page takes the state | 35 / 39 | yes | no |
| Renderer `setNetwork` | 7 / 8 | yes | no |
| Built look drawn (from the press) | 451 / 385 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 3 / 4 | yes | no |

### Deleting a piece of built track on a line (edge 4210, 3.0 km, 3 lines; the press, its question, then Remove)

The question showed 159 / 68 ms after the press (its `removal` query, a light trial, answered 40 / 45 ms after the press). The rows below count from the Remove click. The answer waits for the whole state: a line (Hikari) stops running, so no patch is sent (`clockPatch.ts` `movedPaths`, "starts or stops running") and the tool waits for the network (1.5-1.9 s to build here).

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 59 / 126 | no: waits on earlier clicks | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 42 / 6 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 4 / 5 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 2 / 1 | yes (light) | no |
| Apply: service rebuilt (`update_scoped`: lines, capacity, holds) | 1 / 1 | yes (loops) and the touched lines' whole length | no |
| Apply: weekend service copied (`mirror_weekend`) | 2 / 4 | yes | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 4 / 4 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: departures placed (weekdays and weekends) | 1 / 1 | yes (weekends copy every line) | no |
| Apply: retime | 2 / 2 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | 2 / 1 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 3 / 3 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 1 / 0 | no | no |
| Word and map patch (worker) | 14 / 6 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| Page takes the word and patch (`pendingWord`, `takePatch`) | 27 / 30 | yes (the tools' map: every edge's points copied, every record re-read) | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 137 / 159 |  |  |
| Answer in the page (the tool lets go, the toast) | 2593 / 2182 |  |  |
| Follow-up: rows state (worker) | 232 / 117 | yes | no |
| Follow-up: drawn network (worker) | 1891 / 1516 | yes | no |
| Page takes the state | 126 / 152 | yes | no |
| Renderer `setNetwork` | 51 / 70 | yes | no |
| Built look drawn (from the press) | 2592 / 2180 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 6 / 13 | yes | no |

### Removing a passageway (stations 8626 and 8852)

From the call (no press). The answer waits for the state: a `separate` is not a map click, so it is not answered at its patch (`clock.worker.ts` 1077).

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | - / 114 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | - / 6 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | - / 4 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | - / 1 | yes (light) | no |
| Apply: retime | - / 5 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | - / 1 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | - / 1 | yes | no |
| Word and map patch (worker) | - / 4 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| **Confirm in the page** (press to `applied`, or the refusal) | - / 137 |  |  |
| Answer in the page (the tool lets go, the toast) | - / 661 |  |  |
| Follow-up: rows state (worker) | - / 195 | yes | no |
| Follow-up: drawn network (worker) | - / 230 | yes | no |
| Page takes the state | - / 37 | yes | no |
| Renderer `setNetwork` | - / 8 | yes | no |
| Built look drawn (from the press) | - / 660 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | - / 4 | yes | no |

### A new line from the Lines tab (three commits: the line, its headways, its weekend share)

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 6 / 9 | no: waits on earlier clicks | no |
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | 2 / 0 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 14 / 15 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 13 / 12 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 3 / 4 | yes (light) | no |
| Apply: service rebuilt (`update_scoped`: lines, capacity, holds) | 1 / 0 | yes (loops) and the touched lines' whole length | no |
| Apply: weekend service copied (`mirror_weekend`) | 3 / 3 | yes | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 46 / 22 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: retime | 5 / 5 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | 4 / 4 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 5 / 6 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 1 / 1 | no | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 101 / 79 |  |  |
| Answer in the page (the tool lets go, the toast) | 242 / 226 |  |  |
| Follow-up: rows state (worker) | 173 / 155 | yes | no |
| Page takes the state | 10 / 11 | yes | no |
| Built look drawn (from the press) | - / - |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 74 / 6 | yes | no |

### Deleting a line on shared track (Hikari clone 1, 14 stops, 486 path pieces shared)

No patch for a line gone (`movedPaths`), so the answer waits for the state.

| phase | ms, run 1 / run 2 | scales with the save | needed for the confirm |
|---|---|---|---|
| Page holds the click (the tool waits while an earlier click is out) | 17 / 17 | no: waits on earlier clicks | no |
| Worker busy before it starts (earlier follow-up pieces, whole follow-ups other messages force) | 3 / 2 | yes | no |
| Apply: the checks that can refuse (raw edit, geometry and crossings, cost sums, cash) | 5 / 6 | mostly no (the two cost sums walk the save, about 1 ms each) | yes |
| Apply: running trains noted (`before_edit`) | 11 / 5 | yes | no |
| Apply: lines to rebuild (`lines_touched`) | 1 / 1 | yes (light) | no |
| Apply: weekend service copied (`mirror_weekend`) | 1 / 1 | yes | no |
| Apply: settle (sharing closure, rounds, weekend settle) | 4 / 5 | yes (every line and resource each round) | only for the cash of headway lines |
| Apply: departures placed (weekdays and weekends) | 1 / 1 | yes (weekends copy every line) | no |
| Apply: retime | 1 / 2 | yes (keys of every cluster) and the bundle | no |
| Apply: after (`note_changes`, fleet) | 1 / 2 | yes | no |
| Apply: the rest (demand view, sharing, resources, glue in Rust) | 2 / 1 | yes | no |
| Worker JS around the apply (op, names, kind headways) | 2 / 1 | no | no |
| Word and map patch (worker) | 3 / 3 | mostly no; a moved path reads every line's info | no (lets the next click go) |
| Page takes the word and patch (`pendingWord`, `takePatch`) | 5 / 4 | yes (the tools' map: every edge's points copied, every record re-read) | no |
| **Confirm in the page** (press to `applied`, or the refusal) | 53 / 46 |  |  |
| Answer in the page (the tool lets go, the toast) | 526 / 703 |  |  |
| Follow-up: rows state (worker) | 82 / 153 | yes | no |
| Follow-up: drawn network (worker) | 135 / 217 | yes | no |
| Page takes the state | 34 / 48 | yes | no |
| Renderer `setNetwork` | 7 / 6 | yes | no |
| Built look drawn (from the press) | 500 / 694 |  |  |
| Crash copy (`keep`, a whole `api.save()`) | 20 / 8 | yes | no |

## Where each phase's time grows (master's lines)

- **The checks** (`apply_and_derive`, world.rs 2424: `apply_raw`, `take_touch`, `derive` of the touched records and crossings near them; `check`, 2667: `validate` of the touched edges and nodes, `check_line` of the lines the edit set, every line only when a one-track station is touched, 2672-2673): only what the edit touched, 2-35 ms. The two cost sums around them (`total_and_built_cost`, net/costs.rs 204-253: every edge, node and crossing) and the cash (`extra` = `trains_to_buy` -> `cars_needed`, wasm.rs 1010 and 274-290: `weekend_own` and every line) walk the save, about 1 ms each here.
- **`before_edit`** (service.rs 1642: every line's lattice and every train running now): the save, 4-39 ms.
- **`lines_touched`** (world.rs 2334: every line's stops when a station is touched, 2339-2345; `bus::lines_touched`, bus.rs 356: every bus line's route; `back_lines`, net/paths.rs 185: every edge's way and every line's path): the save, 1-14 ms.
- **The service rebuilt** (`update_scoped`, service.rs 1170: `build_line` of each touched line over its whole length, 1219; `Capacity::update`, capacity.rs 1336: loops over every edge 1351, node 1362 and resource 1391, then rebuilds the resources along the touched lines' whole paths; holds and profiles for every line owning a rebuilt resource): the save's loops plus the touched lines and everything sharing their resources, 1-160 ms.
- **The weekend copy** (`mirror_weekend`, world.rs 1190: `weekend_own` over every line, then `update_scoped` again on the weekend service): the same shape, 1-198 ms.
- **The settle** (`sharing_closure`, world.rs 731-772: every resource each round until it stops growing; `settle_counts`, 1301: every line with trains, 239 here, each round, `bottlenecks` over every resource 1351, `settle_kept.contains` a list search per line and level 1405, `cap.fill` over every resource per level 1454, `update_scoped` of all 239 lines 1458; `auto_counts`, 1055, with `line_limits` over every resource; `settle_weekend`, 1213, all of it again for weekends): the save every round, 3-211 ms. Its only part a confirm reads is the train counts of headway lines, for the cash.
- **Departures** (`place_departures`, world.rs 581, for the lines the edit changed; `place_weekend`, 594-648, copies every line's weekend phases twice and places the edited ones; `pin_twin_phases`, 2540, every twin): the save plus the edited lines, 1-120 ms.
- **`retime`** (timing.rs 226: `clusters` and `cluster_key` of every cluster and a lattice per clustered line, then only the clusters whose key moved are worked out): the save for the keys, the shared bundle for the work, 2-150 ms.
- **The undo record** (`line_rates_all` twice, world.rs 2014; `had` 2088 and `keys` 2096): every line, 0.5-2 ms.
- **After the commit** (`note_changes`, service.rs 1703: every line's era test, the noted lines' trains; `cars_needed` again for the fleet): the save, 1-80 ms.
- **The word and the patch** (worker): `wordAfter` (`stationWord` scans every line's path for a track delete), `mapPatch` (`touchedRows` of the touched ids; `movedPaths` reads `lines_info` of every line when a path moved, clockPatch.ts 60): mostly the edit, 3-100 ms (the 100 a station on Hikari's shared track).
- **The page takes the word** (`takePatch`: `clickMap.ts` 67-89 copies every drawn edge's points into a new map; `layPatch`, clockClient.ts 1238-1246: `patchRows` over every edge and node row and `toWorld` over every record): the save, 4-30 ms.
- **The follow-up** (clockMain.ts `stateSteps` 553): rows (`buildState`, every edge and node; its samples 250-520 ms of it after a station on shared track) 82-711 ms; the drawn network 100-1,891 ms; the page's take 10-152 ms; `setNetwork` 6-70 ms. The crash copy (`keep`, a whole `api.save()`, clockPanic.ts 41): 3-20 ms, twice 74-99 ms.
- **What a later click waits behind.** In the page, the click is held while the one before is out (`toolWait.ts` 63-67) until that one's answer. In the worker, any message but a preview, the delete question or an edit that goes ahead runs the waiting follow-up whole first (clock.worker.ts 731-734). Seen ahead of later clicks: demand's `transfers` (611-2,147 ms) and `riders`, the page's `income` (913-1,591) and `regionShares` (504), the panel's `today` query (2,595) and `blueprintQuote` (372), each a whole follow-up in one block; a station's name (`stationName`) goes ahead and sends a rows state (347 ms); `chargeShares` 15-284 ms of its own (the money state).

## What can refuse an edit

1. The tool before it sends: the station ghost's check (`preview_station` -> `attempt` -> `trial_light`, world.rs 1910, wasm/draw.rs 1014-1033): the edit applied without its service, checked, priced, the cash counted with the trains as they are, rolled back. 8-15 ms a check in the worker on her save (a few at 40-230 ms when other work ran in the same task). The delete tool's `removal` query is the same trial.
2. The model's own refusals before the commit (`refuse` in the TrackApi methods: no such edge, station track, node in use, a node that is already a station): lookups.
3. `apply_raw`'s errors (world.rs 2692): the records themselves.
4. `check` (world.rs 2667): geometry, crossings, water and levels of the touched edges and nodes (`validate`), and the lines the edit set (`check_line`, port stops).
5. Money (world.rs 1740-1746): the construction charge (built value after minus before) plus the cars to buy beyond the fleet, more than the cash: refused, and a second commit (`restore`) takes the edit back.
6. Edit kinds with their own: a stop edit that cannot insert, drop or order (`INSERT_REFUSED` and its cousins), an undo or redo short of cash.

Only part of 5 reads anything the edit's consequences make: the cars to buy read every line's trains after the settle (`cars_needed` reads `svc.lines[i].trains`). For a line on train counts the player set them and an edit does not move them. For a headway line on the edit's track, or sharing track with it, the settle's `auto_counts` (world.rs 1055) and the service's `fleet` (service.rs 1287) can raise them. That is the one check that is not bounded by what the click touched; today's ghost check already skips it (DECISIONS, T-4214 stage A's lie).

For the edits measured: a blueprint station can be refused by 1-4 and by the headway-line cash; a track delete by 2 (station track) and 4; a passageway by 3; a new line by 5 (its own trains, read from its own round trip); a line delete in practice by none.

## Found on the way

- **A free station charged a re-pricing remainder.** A blueprint station on built track splits a built edge; the two halves are priced again, and the built value moves by $0.001 to $220 (charges 1.4e-9, 7.4e-8, 3.7e-7, 2.2e-4 US$M, the same on both runs), over the 1e-9 the commit counts as rounding (world.rs 2323). The edit then counts as paid, so it is not answered at its patch (clock.worker.ts 1077); its answer comes with the rows state its name sends (0.2-0.5 s later) and the next click waits for it. The 2nd station of the pair and the 4th of the click run charged 0 and were answered at once. A bigger rounding bound, or pricing a split's halves as the whole, would answer every free station at its patch.
- **The answer waits for the whole state where there is no patch**: a delete that stops a line running (2.2-2.6 s here), a line deleted (0.5-0.7 s), a passageway (0.66 s; not a map click). Each is accepted in the page at 0.05-0.16 s.
- **The crash copy is small**: `api.save()` 3-20 ms (twice 74-99), so T-4267's stage K would rarely be needed.

## The edit_bench row

`node trainworld/tools/edit_bench.mjs --only clickrun` (in the default set): on up to `--runs` built edges 3-8 km long on running rail lines (those on the most lines first, never the stop actions' line), four stations at 0.2, 0.4, 0.6 and 0.8 of the edge, each press sent at its own time 0.3 s apart (a raw CDP session, so a busy page does not space them out). Rows: "click run: confirm, first click", "click run: confirm, later clicks" (each later click a sample), "click run: answer, later clicks". Tested once on her save (run 3 above, NOISY).

## Budget

Three runs on her Asia save (two of the driver, one edit_bench invocation with `--only clickrun`, NOISY). Game checks on New York to build the driver and the row: five short runs, about 7 minutes. Typecheck clean; the tree WASM built once with the instrumentation (`agent_cargo.py wasm`). No perf_bench, no smoke. Advisor not called (a measuring stage, no design decision).
