# St Kitts and Nevis (kn): record

Drawn 2026-10-05 (session edd42a8c-mono5). No census language question. Built as Barbados and
Antigua (`sources/bb.md`, `ag.md`): 2011 census country of birth by island, 46,983 people (212
not stated left out), 2 islands, every row `derived`. Placed on religiondots' Kontur hexes
(one unit there, read-only) re-keyed by island. 42 dots, 156 rings.

Files: `sources/kn_census.py`, `taxonomy/kn2011.py`, `taxonomy/tree.d/kn.txt` (origin_mix block
generated), `countries/kn.py`, `data/normalized/kn.csv`, `data/geo/kn/kn_hexes.gpkg`,
`data/raw/kn/` (two saved HTML tables).

## Sources (Department of Statistics, stats.gov.kn, HTML tables)

- "Foreign Born Population by Country of Birth (2011)": St Kitts 5,330, Nevis 3,049, total
  8,379 incl. 212 not stated; every country named.
- "Number of Households and Population by Parish and Island 2001 to 2011": St Kitts 34,918,
  Nevis 12,277, total 47,195 (religiondots' figure).
- Native-born per island = island population - foreign-born table total.

Checks (asserted): St Kitts + Nevis = total per row; countries + not stated = the printed total
per island; islands sum to 47,195; Kontur's island shares within 1.5x of the census's (0.96,
1.12).

## Mapping

- Native-born (38,816) on `creole.english_based.antiguan`: Glottolog's anti1245 is the Leeward
  creole and covers St Kitts and Nevis (`bb.md`). `how` says so, since the legend label reads
  "Antiguan and Barbudan Creole".
- Montserrat, Anguilla, Antigua -> the same node; BVI, St Croix, St Thomas, USVI -> Virgin
  Islands Creole; Guyana, Trinidad, Grenada -> their creoles; St Lucia -> Antillean Creole;
  Turks and Caicos -> its creole (bs.txt); Puerto Rico -> Spanish.
- US-, Canada-, UK-, Bermuda-, Cayman- and Belize-born on English (bb/ag/bm convention).
- Everything else through `origin_mix.mix(iso, "kn")` (St Martin on SX's mix, St Eustatius on
  BQ's, Curacao and Netherlands Antilles on CW's).

## Calls someone might reverse

- All native-born on the creole; white Kittitians and Nevisians (no ethnicity table reached)
  and English-dominant speakers are not separated.
- US-born (1,682) on English, though some may be US-born of Puerto Rican or other origin.
- Island split at 17.217 N, between Nevis's northernmost hex (17.209) and St Kitts's
  southeast peninsula (17.225).
