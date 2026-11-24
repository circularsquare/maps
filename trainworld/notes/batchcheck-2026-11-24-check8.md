# Batch check8, 2026-11-24 (T-4182 and T-4097 stage), tree .tw-trees/check8 at 095480256

## T-4182 PASS
- Head (tree WASM, Vite 6157) rail trips a day: real NYC 12,672,465 (equal to the figure before); Montana 6,849; 13-zone Tokyo stand-in 16,262,490 (all equal to the figures before).
- Per-line riders against the base build (perfbench/dist/sl-base, built before T-4182): Montana 4 lines identical (1509/1408/509/3422); stand-in all 142 named lines identical (a map by name, so the three same-named lines count once). NYC per-line not compared against base (total only).
- Trip explorer, Missoula (msu) to Spokane / Idaho Falls (bvl) / Billings (yeo): text identical before and after a reload (6k, 2k, 13k trips a day; intercity 4k, 1k, 4k; commutes, everyday, drive, plane figures). One early read of Spokane after reload was cut mid-render; a second read was identical.
- Not measured: load time (notes/T-4182.md has it).

## T-4097 stage PASS (one gap)
- a. Montana: took stop 3 out of Bitterroot Flathead by the real UI (Edit stops, press): stops 0,1,2,4,5,6, rail trips 6,849 to 6,579, line riders 1509 to 1239. Exported, reloaded: same stops, status, trains, riders per line, 6,579.
- b. Stand-in: took stop 11 out of Takasaki line (21 to 20 stops): rail trips 16,262,490 to 16,255,074, 14 lines moved. Fresh load of the exported edited save: every line's riders and stops identical, 16,255,074. GAP: I did not confirm that line crosses a zone line (the dock lists only 5 player lines, all with 0 riders; Joban north edit, 47 to 46 stops, moved nothing). Intercity corridors are per market pair, so a stop edit in one zone does not touch them.
- c. Burst on Helena Butte (4 presses 120 ms apart, 3 to 2, 1, 0, 1 stops): no page errors, notes "1 stop" and "Now click the next station", clock not held, demand reached final (6,339). A refusal could not be produced (the editor allows 0 stops); not tested.

Scratch: C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/3a819365-cd3a-4850-a112-6396d17c97b4/scratchpad/check8/
