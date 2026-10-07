# Cuba (cu): record

Drawn 2026-10-05 (session edd42a8c-carib). No language source: all 10,055,968 people on Spanish
across 168 municipalities, rows `derived`, on ONEI's 2023 municipal populations.

Files: `sources/cu_geo.py` (counts and placement), `taxonomy/cu2023.py`, `taxonomy/tree.d/cu.txt`
(bare repeats), `countries/cu.py`, `data/normalized/cu.csv`, `data/geo/cu/cu_hexes.gpkg`
(languagedots' own re-keyed copy of religiondots' hexes).

## Source

- No Cuban census has asked language (scout 2026-10-05: 2012 asked skin colour; the 2022 census
  is still postponed). Spanish is the first language of practically everyone. The only other
  first-language community of any history is Haitian Creole in the east (descendants of the
  1910s-30s cane migration); no count exists, and it is said in `note_public`.
- Population: ONEI's 2023 municipal figures as religiondots carries them
  (`cu_municipalities.csv`, `pop2023`), 10,055,968. religiondots uses ONEI's end-2024 province
  count (9,748,007), which is lower after emigration; 2023 is used because it is the latest
  municipal figure on disk. Say so if the newer level matters more than the finer grain.

## Placement

Kontur 2023 is badly off in eastern Cuba (religiondots' `cu_grid.py`: Granma's raw Kontur is
0.17 of its ONEI share). Municipal counts pin each municipality's people, so Kontur only places
them within a municipality. `cu_geo.py` assigns each of religiondots' 62,557 province hexes to a
municipality by representative point; 694 whose point fell outside a municipality of their own
province went to the nearest one in that province. Asserted: 168 municipalities in csv and
polygons, joined 1:1 on `adm2_pcode`; every municipality gets hexes; every hex stays in its
province. religiondots also lowers two false Kontur cap blocks in Havana; this layer uses raw
Kontur `pop`, so inside those two Havana municipalities the blocks may draw dots a little
denser than real. scatter.py raised no cap-block stop.

## Calls someone might reverse

- Municipal 2023 over provincial 2024 (finer grain, older level).
- Haitian Creole in the east not drawn (no count).

## Scatter

10,055 dots over 4,540 hexes; water.py left 71 near-sea units unclipped.
