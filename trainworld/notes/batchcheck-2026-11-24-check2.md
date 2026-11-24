# Batch check 2, 2026-11-24 (agent check2)

Master at 6b68241cc, WASM built 02:29 (after the last commit). Vite on 6184, stopped. No code edited, no smoke, no commits.

## Results

| change | verdict |
|---|---|
| T-4133 re-place only changed lines' trains | PASS (black box; the Rust fuzz check is the builder's) |
| T-3939 sparse boarding account | PASS on what a game shows (NYC real save loads, same figures); memory figure not measured |
| T-4181 show-at-click fixes | PASS (t4170-one-path.mjs, NYC, --shots: no FAIL, 0 exit) |
| T-4185 one undo per Copy line / New line | PASS |
| T-4178 reversal limits | PASS |

## Evidence

- Real NYC save (trainworld/T-007-real-nyc.save) loaded with play.mjs: after wait.final, 109 lines, railTripsPerDay 12,672,465 (matches the bench count in the brief), tripsPerDay 899,564,070.
- t4170-one-path.mjs on NYC with --shots (log C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/5cd3af11-2ab8-4788-8daf-1956104ead51/scratchpad/check2/t4170-nyc.log): no FAIL line. Covers: cash at the click = charge for Construct all quoted (464.43 vs 464.43), Rebuild by level (593.94 shown, 593.94 taken; single-track rebuild 0/0), a free edit untouched, nothing shown for a Construct the cash cannot pay, copy colour drawn at its word, first station in a new game drawn at its click, no stale ahead textures. Screenshots: C:\Users\anita\projects\maps\artifacts\trainworld\t4170\nyc-construct-click.png (and nyc-copy-click.png). Not run on Asia.
- Undo/redo (real NYC save, wm.edit ops, clock stopped): A rename line, Copy line, rename another line, then undo x3 / redo x3 / undo x3: each undo took back one player action (rename, copy whole, rename) in reverse order; redos mirrored it; copy undone as ONE step (110 -> 109 lines). B New line then Copy line: undo x2, redo x2, undo x2 gave 111/110/109 and back. C two copies: same. D copy, rename the copy, undo x2: name reverted, then the copy removed. E copy, undo, new rename, redo: redo refused (correct, history cut). All ended at the original 109 lines and names.
- Same path by player in sandbox NYC (act.newLine + stop clicks, clock running, Ctrl+Z / Ctrl+Shift+Z): line removed in one undo after the stop and name steps are undone (stop clicks and the rename are their own steps, as designed); second undo refused (construct is final). Redo walks back the same way.
- T-4133 black box (real NYC, clock running at speed step 1, 24 lines, lineTrips queries over a fixed absolute window): rename of another line changed no line's trips; +2 trains on line X changed only X's future trips (others identical, past identical); two undos returned every line's future trips to the baseline. Money with the clock stopped: a setTrainCount charged nothing, undo/redo/undo left cash exactly unchanged.
- T-4178 by the player's path (sandbox NYC, Staten Island; track X-T-J-M plus branch J-BR with the track tool, stations with the station tool, Construct, line BR,T,M from New line, 12 trains asked):
  - default (160 km/h, no terminals): panel says "Merge near Forest Hill Rd is full", at most 12.55 trains/h, 5 of 12 run, round trip 23m29s, turnLimits empty, no reversal icon. Matches the stem-limits-first claim.
  - speed limit 80 km/h, terminals at both ends, 14 asked: panel "Turning at Grant City is full", limit 15 trains/h, turnLimits {line: Grant City -> 15}, icon shown; hover text "Grant City / Reversing here limits Switchback to 15 trains per hour" (new string, as specified). Screenshot: C:\\Users\\anita\\projects\\maps\\artifacts\\trainworld\\check2-switchback-limiting-icon.png (may be under artifacts/trainworld/ by name).
  - line-end turnaround unchanged: a plain two-stop line X-M alone, 20 asked: 15 trains/h at the ends; with terminals at both ends 24.99/h (track-limited, 30 turn cap not binding).
  - a second line ending at the same node as the reversal takes turn capacity from it (Switchback then limited to 3.99/h at Grant City with End asking 20): expected sharing, not a fault.

## Observations (not failures)

- Vite log showed three console.warn lines during t4170 on NYC: "T-4145: the network after line N's word draws other edges than the word (48 drawn, 49 in the word)" for lines 1, 109, 110, all at one moment (copy line in the check). The check passes; I did not trace it. A fix agent may want to look at whether the copy's word includes one edge the network does not draw.
- A "clock worker: TypeError ... reading 'r'" appeared once: caused by my own probe sending a non-existent op (deleteLine), not by the changes.
- Reversal tip with n = 1 would read "1 trains per hour" (plural not handled); only at a limit of exactly 1.0-1.9.

## Skipped

- Asia save runs (t4170 on asia, t4153-net): no time budget used; NYC only.
- T-3939 memory saving (30 MB less) not measured; only that the real save loads and trips match.
- T-4133 train positions were compared through lineTrips (schedule), not a per-train position read; no edit of shared track that changes another line's profile was tried by hand (the builder's TW_TRAINS_CHECK fuzz covers it).
- Unaffordable edit by hand (covered by the standing check's blueprint cases only).
- Stops edit with autoFast joined undo (T-4185 note says no check): not tried.
- No smoke (rule of the night).
