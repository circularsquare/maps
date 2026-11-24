# check10 batch check, 0.4.6 (2026-11-24)

Tree: .tw-trees/waitclick (master ef9fb81f4 + uncommitted waiting-clicks change). Vite on 6175, stopped at the end. Probes in the scratchpad check10 folder (probe1-13.mjs). About 17 minutes of game checks, 0 benchmarks.

## 1. Waiting clicks (T-4214): PASS with one FAIL
- t4214-waiting-clicks.mjs: all three sequences PASS (track then delete, track then station, station then delete past it keeps the station).
- Delete, undo, redo: PASS held (undo and redo wait for the click; track gone) and unheld (undo puts it back, redo removes it).
- Two stations on one track: PASS unheld and with states held (both on the track, degree 2, +2 stations).
- Held click then Escape: PASS (dropped, track drawn before stays). Then tool switch: PASS dropped. Then pan away: PASS dropped, waitingClicks 0, clicksOut 0, nothing stuck.
- Fares, borrow, repay then a station click: PASS, clicks 0 waiting after each (no stall).
- Load another game while a click is held: PASS. Tokyo save imported with a delete click held: click dropped, counts equal the control import (22 lines, 130 stations, 448 edges), no page errors.
- Burst (3 stations clicked quickly, then delete): with states delayed 0 and 150 ms the final network equals the click-by-click run (same 6 edges, same stations). No page errors.
- FAIL (design edge, with delayed states 600 ms): a station click that is waiting is dropped when the player switches to the delete tool. Sequence: station tool, three clicks, delete tool, click. The third station (the one still waiting) never lands: network has 2 stations less piece than the click-by-click run. Silent, the player made the click. Cause: tryClicks drops the queue when h.tool !== tool.peek(). Same code as the by-design "tool change drops a held click", but it also drops a click the player already made and expects to land. Suggest carrying the queued click out (or waiting for the queue to drain) before the tool changes, or keeping the dropped click's tool for it. Not seen with 0 and 150 ms delays.
- Not tried: 5-6 click mixed burst with every tool (only the one above), touch confirm clicks.

## 2. T-4210 ghost: PASS
- t1811-track-tool fresh: 7 of 7 PASS.
- Multi-click runs of 4 points: unheld and with states delayed 400 ms, with a zoom mid-run, edges +3 each, run ended cleanly (start null, chain 0); a run started from a station with states delayed 300 ms: +6 edges, no errors.
- Not judged: that the ghost visibly follows the real curve (no frame-by-frame look); screenshots in scratchpad check10 (ghost-*.png) only.

## 3. Left untested last round
- T-4194: PASS on numbers. Small NYC game, line of 3 stops by headway 600 s: 2 trains; add stop B (round trip 993 s to 1156 s): 2 trains; remove it: back to 2 and 993 s; add D, headway 420 s: 3 trains; remove D: 3. All match ceil(round trip / headway). No page errors. t2018-trail (express behind a local): FAIL by the check, 4 of 109 standing frames at s=7661 m with no local ahead (ahead null), timetable stands 0, no console errors. The check reports "local":0 built, which may mean the check's own setup did not build the local. Not compared with master, so not known whether new. Needs a look by someone who knows the check.
- T-4202 stage 1 (keepSet): PASS. Empty NYC game running, tokyo-day60 save imported into it: 886307263 trips/day, 2861307 rail, 22 lines, equal to a fresh load of the same save. No page errors. A zone at 0 was not read separately; totals and line count equal.
- T-4204: PASS on the text. statusTag.capped renders "Time capped at {cap} while calculating trips" and topBar.speedCapped "{speed} (time capped at {cap} while calculating trips)" in the running page through t(). The capped state could not be reached in a game within the time (forcing clock.capped did not change the speed button's tip), so the hovers were not seen drawn.

## Skipped
- Real NYC save (slow loads); benchmarks and smoke (brief).
- The capped hover on screen (see above), express-through-local beyond t2018, touch clicks.
