# batchcheck14 (2026-11-22, master 5edb2e77, own Vite 6141; scratch: scratchpad\batchcheck14)

## T-4041 (arrival-first line choice, street trams)
- (a) PASS. Real save T-007-real-nyc.save: load 17.6 s, demand final round 3 in about 2 min on 4 cores (v3), 12,660,073 rail
  trips/day (notes: 12,882,296 average day on master before, so -1.7%, as T-4041 measured). Boardings 23,539,563 vs
  alightings 23,539,608 (diff 44, 2e-6); lines' riders sum = boardings. No page errors in any call.
- (b) PASS. Sandbox NYC, T-4019's Local (15 stops) + Express (4), scenario.sh from arrivefirst2. At 15/30 min: pairs 0-5/0-10/0-14
  Express 35/45/51%, Local 65/55/49; riders Local 34,056, Express 1,214 (rail 34,929). At 30/30: Express 63/75/82%; Local 26,195,
  Express 2,271. Matches T-4041's "now" table exactly; local never near zero.
- (c) PASS. Sandbox: add line, delete Express, undo: each recalculated to a final, Express back with 2,434 riders as before.
  Real save: add (across), removeLine 0, undo, undo: four solves to final (~95 s each), boardings = alightings (23,548,451 vs
  23,548,496), no errors. (Net 110 lines at the end: the across line is several undo steps, one undo did not remove it; not a defect.)
- (d) SKIPPED in game: T-4041's own notes say a street tram stop cannot be placed by act.placeStation; only the sim tests cover it
  (`a_train_trip_pays_its_way_in_and_out`, tram/bus table 52%/66% vs before).
- (e) Real save final about 2 min (4 cores) against smoke's 412 s on 3 cores: nothing slow, no stuck bar seen.

## T-1354 (corridor book per composition) - UNRESOLVED, possible FAIL
New NYC sandbox, a PHL line at (-75.2,39.95) and a PVD line at (-71.45,41.82), 3 stops each, same trains.
- PHL then PVD (separate edits, and also both in one edit batch): rail 75,950; PHL 63,996, PVD 11,954; trips 845,085,378.
- PVD then PHL (separate, or one batch): rail 80,068; PHL 68,341, PVD 11,728; trips 845,081,766.
- Each order repeated: identical to itself (deterministic). So the final differs with the order the lines (hence zones/stations) are
  made, PHL riders +6.8%. Not shown to be pack-open order: line/station numbering also changes with the order, and I had no
  pre-T-1354 build to compare. Worth a look by whoever owns T-1354 (a control with the same creation order but a different
  zone-open order was not built).

## T-4040 / load bar
- t2552-load-bar.mjs PASS (all, incl. "a model's 30% is shown at 5-10%", zone bar moves with the solve, bare zone full once open).
- In game (new Tokyo and NYC loads, aria-valuenow sampled every 100 ms): bar rises 0 -> 1 -> 28 -> 42 (jump 2 -> 28 at about 7.5 s
  when the solve starts), monotonic, then the card goes at a real final with the last sample at 36-42%. The bar is capped
  at shownShare(0.99) = 91.6%, so on a normal final it never reaches 100 before it disappears (only a cached-result end shows it
  full, T-2574). Possibly meant; flagged against "ends full".
- Zone-opening bar and T-4040's Fare multiple row for HSR/ferry/bus/BRT: not tried (time).
