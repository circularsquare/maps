# Dominica (dm): record

Drawn 2026-10-05 (session edd42a8c-mono5). No census language question. Built as Haiti
(`sources/ht.md`), with two groups the census's own figures place: 2011 census, 71,293 people,
3 units (Wesley, Marigot/Concord, rest of the country), 3 nodes, every row `derived`. Placed on
religiondots' Kontur hexes re-keyed into those three units. 70 dots, no rings.

Files: `sources/dm_census.py`, `taxonomy/dm2011.py`, `taxonomy/tree.d/dm.txt`, `countries/dm.py`,
`data/normalized/dm.csv`, `data/geo/dm/dm_hexes.gpkg`, `data/raw/dm/dm_census_2011.pdf`.

## Source

2011 Population and Housing Census, Preliminary Results (Central Statistical Office, September
2011; stats.gov.dm/wp-content/uploads/2019/06/Population_and_Housing_Census_2011.pdf). No
language, ethnicity or birthplace table; it gives the total (71,293), Table 8 non-institutional
population by town/village (Wesley 1,362, Marigot/Concord 2,411) and, in the Review, the
Haitian-born count (1,054). The script asserts each figure as printed and that rows sum to the
total. No survey with a language item was found (none of Afrobarometer, WVS or DHS covers
Dominica).

## The model

- **Kokoy for Wesley and Marigot/Concord** (3,773): the English-lexicon creole of the two
  Methodist villages settled by Antiguan and Montserratian labourers; Glottolog has it in
  Antiguan and Barbudan Creole (anti1245), the `antiguan` node from `bb.txt`/`ag.txt`.
- **Haitian Creole for the Haitian-born** (1,054), on the rest of the country by population
  (no area breakdown).
- **Kweyol (Antillean Creole) for everyone else** (66,466, 93.2%).

## Placement

religiondots' `dm_hexes.gpkg` (one unit, read-only) re-keyed: hexes with centroid within 1.5 km
of each village go to it. Wikipedia's minute-rounded coordinates put Marigot 1.8 km inland, so
each point was moved to the Kontur built-up cluster beside it. Check: Kontur in the circles
1,087 (Wesley, census 1,362) and 1,591 (Marigot/Concord, 2,411), both within the asserted factor 2.

## Calls someone might reverse

- **Everyone outside the two villages on Kweyol.** Language shift to English among the young,
  above all in Roseau, is widely described but uncounted; no share is guessed. Said in
  `note_public`. This overstates Kweyol more than Haiti's model overstates Haitian Creole.
- Concord (an inland village counted with Marigot) drawn as Kokoy with it.
- Kalinago Territory (2,145) on Kweyol: Island Carib is extinct.
- White Dominicans and other immigrants: no count in the preliminary report.

## Room for improvement

The final 2011 census tables (ethnicity, birthplace by parish) are not online; they would place
immigrants. Any language-use survey would split Kweyol from English.
