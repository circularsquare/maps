# Batch check 2026-11-24, check4 (T-4075, T-4051)

Master 82588e6e1, own Vite 6121, cores from the ledger (2-3 per game).

## T-4075 PASS
- t4075-on-track.mjs (Amsterdam save): 916 rail / 85 bus placements, worst 0.21 m rail, 0.14 m bus, GPU vs CPU 0.005 m. PASS.
- t3250-return-trains.mjs: PASS all.
- Real NYC save (play.mjs, my probe over every trip for 10 game minutes, every 30 s): 13,125 placements, worst 0.26 m, 0 over 1 m; GPU vs CPU worst 0.012 m of 6,610; 15,212 bent intervals, 24,750 points (same as the builder's note).
- Edit while running: removeEdge on line geo 10 edge 326 -> bent 15,024, worst still 0.26 m, 0 over 1 m; undo -> bent 15,212 / 24,750 again, 0 over 1 m. New 2 km line (across) -> 0 over 1 m, GPU 0.012 m.
- Frame rate headless on NYC with the clock running: 56.7 fps at zoom 12.5, 56.0 at 16.5 (machine busy).
- Click on trains (14 trains, zoom 17): 9 selected the train (same line), 5 selected a station (trains standing in a station). Zoom 15: same.
- Screenshot at a bent interval, zoom 17.5 (curve.png in scratchpad check4): cars bend along the curve, no gaps.
- Reload: NYC load itself builds the same bend counts.

## T-4051 PASS (with a caveat on total trips)
Save: artifacts/trainworld/saves/na-day-491.save, 2 solving workers (cap full; the builder used 5).
- Edit right after load, Medellin 2 km line: edit to final 7 s + 25 s = about 32 s (builder's v0.4.5: 117-133 s). Total trips 1,731,969,494/day, same as the builder's note. Rail 30,733,181.
- Two zones in quick succession (Managua, Bogota in one edits file, final waited once): 45 s from the first edit to the final. Rail 30,854,985.
- Edit in San Jose (Costa Rica) then undo straight away, before the zone opened: final came back in 3 s, no stall, rail trips 30,854,985 (same as before the edit). The edits harness's undo refused after 2 steps; stations of the San Jose line remain (2,252 vs 2,249), lines back to 177.
- Reload of the edited game (downloaded via Saved games, Download as a file): rail 30,854,985, equal. Total trips 1,776,871,587 on reload against 1,775,472,523 live (0.08% apart). The live state had the undone San Jose stations; I did not run a v0.4.5 arm to say whether the gap is old behaviour.

## Together
Real NYC save: railTripsPerDay 12,672,465 (play.mjs wait.final). PASS.

## Skipped
Tight-corner terminals visually; zoom 12 pick; old-build baseline for the 0.08% total; fps baseline against the old build; one-core NA run.
