# docs/ui-panels.md: history and measurements

Moved out of the doc in the T-940 tidy (2026-10-22), since docs hold current rules only. Grep by
task id.

- T-009: the mock (`mock/`) was the approved starting point for the look; its 3 font sizes and 2
  weights are the ceiling the doc still cites.
- T-882, T-896: the top bar's sky scene costs about 0.7 ms a draw, about 7 draws a second at the
  top speed (DECISIONS.md, T-882, has the earlier 30 draws a second of the moving-car version).
- T-902: the scene's bottom strip had grass until T-902; now a track with sleepers only.
- T-832: the train share label lost its rides count in T-832 (the hover has it).
- T-703: the Day choice (Average, Weekday, Weekend) was on the map's demand bar until T-703 moved it
  into Settings. Stored settings from before T-703 (stored version below 3) lose their saved font
  and panel size once, so they get Nunito and 90%.
- T-565: a Money history day from a save made before T-565 has no train rides.
- T-894: save pictures stored before T-894 (one frame, no city names) show as stored until the slot
  is next saved.
- T-550, T-749, T-532: the Cities list started as the last section of the Money tab (T-550), there
  because a row's profit is the tab's region ledger and a click on a row changes the region the
  table above follows; a fifth dock tab button was ruled out then by T-532. T-749 gave it its own
  dock tab.
