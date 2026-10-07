# Turks and Caicos Islands (tc): record

Drawn 2026-10-05 (session edd42a8c-mono5). No census language question. Built as the Bahamas
(`sources/bs.md`): 2012 census country of citizenship, national, applied to each of the seven
inhabited islands' populations; 31,458 people, 6 nodes, every row `derived`. New placement layer:
Kontur hexes keyed to islands. 28 dots, 1 ring.

Files: `sources/tc_census.py`, `taxonomy/tc2012.py`, `taxonomy/tree.d/tc.txt`, `countries/tc.py`,
`data/normalized/tc.csv`, `data/geo/tc/tc_hexes.gpkg`, `data/raw/tc/` (two sheet CSVs).

## Sources

Department of Statistics, gov.tc/stats/statistics/social/5-population; each table is a public
Google Sheet, exported as CSV (no login):
- Population by Country of Citizenship 1990-2012: TCI 12,239, Haiti 10,928, Dominican Republic
  1,541, USA 874, Bahamas 551, Canada 423, England 384, Other 4,518; total 31,458.
- Population by Island 1960-2012: seven islands summing to 31,458.
No citizenship-by-island or birthplace table is published; the 2012 census report itself was
not found online. visittci.com puts Jamaicans at 8% in 2012 (secondary, not used).

Checks (asserted): citizenship rows and island rows each sum to 31,458.

## Geography

No religiondots layer; geoBoundaries has no TCA ADM1. Kontur TC (419 hexes, 46,067 people, a
2023 level 1.46x the 2012 census) keyed to islands by a Voronoi split of seven settlement points.
Kontur/census per island normalised 0.63 (North Caicos) to 2.23 (Parrot Cay, where Pine Cay's
hexes join it); all within the factor 3 band; only placement within an island is borrowed.

## Calls someone might reverse

- **Other (14.4%) on `other`.** The table pools it; Jamaicans are probably the largest part,
  but no count is published, so nothing is guessed into Jamaican Creole.
- TCI citizens all on Turks and Caicos Creole, including naturalised Haitians and their
  children; white or English-dominant belongers are not separated.
- US, Canadian and British nationals on English (bb/ag/bm convention).
- One national mix on every island.
