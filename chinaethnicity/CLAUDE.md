# chinaethnicity: notes for agents

A public dot map of China's 2020 census by nationality, from each province's own county
table. Read these before changing anything:

- `NOTES.md` is the record: why each step is the way it is. Its last section, **Handoff**, is
  the current state and the list of open tweaks.
- `provinces.md` is what is drawn and the queue for the other 15 provinces, in order, with
  URLs and leads.
- `COMMANDS.txt` is the build order.

Rules that are Anita's or that the build depends on:

- **Measured first, fallback second, and the map says which.** 16 provinces are published
  2020 county counts (`parse.py`, `join.py`). The other 15 are estimates from `fallback.py`,
  the method Anita approved on 2026-09-14: the 2000 census county pattern scaled to 2020
  totals, per prefecture where the 2020 table is published by prefecture and per province
  otherwise. The viewer hatches them and says so on hover and in the About panel. Anything
  beyond that method (fitting to modern county populations, splitting Chongqing's "other
  nationalities" column), ask.
- **`parse.py`'s checks must pass**: prefectures sum to their counties in every column, total
  = male + female, and each province matches the national table. Never loosen them to get a
  province through; a failure means the parse is wrong.
- **`colors.csv` is hand-edited.** `palette.py` refuses to overwrite it; do not work around
  that. After a colour change re-run `scatter.py` only (legend.json carries the colours).
- **Read `data/work/join_report.txt` after any `join.py` run.** Every `!!` line is a real
  problem; a new province usually needs a few entries in `OVERRIDES`, each with its reason.
- **Read `data/work/fallback_report.txt` after any `fallback.py` run.** The script stops on a
  county polygon that no 2000 row reaches; add it to `CARVED` with the counties it was carved
  from, which must border it. The report prints each province's large scaling factors and how
  far its drawn county totals sit from ASPECT.
- Who is drawn, and how finely, is her call (the county grain was her choice, and so was placing
  Sansha on the Paracels only). Raise such questions; do not settle them quietly.
- Publishing follows the religiondots pattern (pmtiles to the `anitamaps` R2 bucket, page to
  the website repo) and has not been done. Do not commit or push without her say-so.
- Serve with `npx serve`, never `python -m http.server` (PMTiles needs range requests).
  `tiles.py` cannot replace the archive while a server holds it open.
