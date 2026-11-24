# Batch check 2026-11-24, agent checkclick

Game: Vite 6178 on master (80ab55dc3 plus the uncommitted tree), real NYC save (play.mjs, headless), Asia via t4170 check. Probe scripts: scratchpad checkclick/*.js. Shots: scratchpad checkclick/standin-1.png (ghost with answers held 2.5 s), standin-3.png; t4170 --shots in artifacts/trainworld/t4170/.

## Results

- T-4145 (base, undo at its click) PASS. t4170-one-path nyc and asia both PASS: station, copy, Construct drawn at the click then nothing left (entries 0, ahead 0 after the final state); delete then undo drawn at its click (drawn 3 pieces); undo+redo at once OK; 4 (nyc) / 6 (asia) entries at most at once.
- T-4147 PASS. Copy line (free in NYC, 1 line ahead at click, gone after), colour change shown at the click (ahead.colours has the new rgb while states are held back 300 ms, line colour ends right), lineFast/lineLook burst, new line through the real tool (2.4 s incl. stop clicks), removeLine + undo. No entries left.
- T-4146 PASS. Station placed by the tool, station deleted, platform length, rename answered by rows alone: covered by t4170; station tool click in open water right after a copyLine: no station, no bad notice, no stale entry or ghost left.
- T-4148 PASS. On a blueprint edge: edgesTracks, edgesLevel, pieceLimit each drawn/entered at the click and cleared (0 entries) after; refused edgesLevel 99 and edgesTracks on a missing edge returned ok:false and left nothing; removePieces hid the edge at the click, undo brought it back. Construct: see money.
- T-4170 PASS. One entry per edit (doubled 0, swept 0, no shared seq), refused edits dropped in their answer with one notice, undo after a long pan (anchor moved) hides nothing (miss "the draw anchor moved"), clock running + 3 edits + 3 undo/redo: after 35 s no entries, no pending, nothing ahead (backstop never needed).
- T-4143 PASS. Track tool, answers to previews delayed 2.5 s: after the first click, moving the pointer draws a straight stand-in ghost (standin-1.png). Second click placed the edge in 0.5 s; ghost gone after Escape (standin-3.png).
- T-4162 PASS. en.json: issues.nodeHeading = "Track alignment error". sumTripsNear called with fake legs: 0 -> "0 trips within 1km", 1 -> "1 trip within 1km", 2..49 exact, 49.6 -> "50 trips", 51 exact, 1234 -> "1,200"; bus rows say "1 trip within 1,000 m". Live hover in NYC showed only large counts (no spot under 50 found), so the under-50 path was checked by calling the function. I did not trigger the alignment refusal itself in play.

## Money (Construct, station, copied line)

NYC real save: Construct of 2 stations plus a 0.7 km blueprint: cash 6000 -> 5820 (180 once; the notice and the cash agree). A second Construct (2 stations, 56 M track quote): shown balance 5820 -> 5704 at +32 ms -> 5524 at +73 ms -> 5524 final (charged 295.9 once, no double dip, notice "Constructed for $296M."). A third: 5315.6 -> 5212.8 -> 4852.8 final (462.8, the stray earlier blueprints included). The balance is NOT fully at the click: it moves in two steps within about 75 ms, the first a partial quote. Copied line in NYC costs 0 (balance unchanged). Final always equals the shown figure.

## Notes / odd things (not failures)

- Construct's balance at the click is a two-step guess (first step about a third of the total), then the final. Looks like a flicker of 2 frames; harmless but "the balance at the click" is not one figure.
- Undo right after Construct returns ok:false (probably by design, construction is not reversible).
- lineLook with colour not a hex string logs "an edit's guess part failed" (caught, console.error): my probe's wrong op, not a bug.
- play.mjs probes reading renderer.ahead must re-read it each time (the object is replaced); a held reference reads stale.

## Skipped

- Edits while the whole real-save trips were still calculating (I waited for the final before most probes; the t4170 runs begin right after load, so partly covered).
- No cash / no room refusal by a real broke state (cash is 5000+); refusals tried: unknown line, unknown station, missing edge, level 99, station in water.
- Station on a station by the tool (the tool placed a new blueprint station next to South Ferry instead of refusing).
- Touch input, Asia save beyond t4170, T-4147 insert_path preview for an added stop on a long line (new line + toggleStop by the real tool not timed).
- No smoke (Anita's rule).
