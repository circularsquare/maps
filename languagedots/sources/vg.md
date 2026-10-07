# British Virgin Islands (vg): record

Drawn 2026-10-05 (session edd42a8c-mono5). No census language question. Built as Barbados and
St Kitts (`sources/bb.md`, `kn.md`): 2010 census grouped country of birth by island, 27,992
people (62 not stated left out), 4 islands, 12 nodes, every row `derived`. Placed on
religiondots' Kontur hexes for the four islands (read-only). 22 dots, 3 rings.

Files: `sources/vg_census.py`, `taxonomy/vg2010.py`, `taxonomy/tree.d/vg.txt`, `countries/vg.py`,
`data/normalized/vg.csv`. The report PDF is religiondots' `data/raw/terr/VGB-2016-09-08.pdf`,
read only.

## Source

2010 Population and Housing Census Report, Table 84 "Grouped Country of Birth by Island" (PDF
p. 72): 22 grouped birthplaces x 7 islands. Cooper Island, Great Camanoe and Yachts (50 people)
folded into Tortola as religiondots does. Checks (asserted): each row's islands sum to its
total; island columns sum to the Total row; 28,054 overall; four rows agree with Table 83.

## Mapping (`taxonomy/vg2010.py`)

- BVI-born (10,975) and USVI-born (1,481) on Virgin Islands Creole (Glottolog virg1240 covers
  both territories).
- Caribbean birthplaces on their creoles (bb conventions; St Kitts on the Leeward creole node
  `antiguan`); Dominican Republic and Puerto Rico on Spanish; US and UK on English.
- Pooled "Other Caribbean", "Overseas Territories", Europe, Latin America, Asia, Pacific,
  Middle East, Other Countries on `other` (1,738, 6.2%); Africa on `africa_other`.

## Calls someone might reverse

- All BVI-born on the creole; English-dominant and white Virgin Islanders not separated (Table
  75 has ethnicity by island but no cross with birthplace).
- US-born (1,537) on English, though many are children of islanders or of USVI families.
- "Overseas Territories" (Anguilla, Montserrat, Bermuda...) on `other` rather than guessed onto
  the Leeward creole.

## Moved from countries/vg.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- The other 61% come from more than a hundred countries and are drawn on the language of where they were born: Guyanese, Vincentian and Jamaican Creole are the largest, then Spanish for those born in the Dominican Republic and Puerto Rico.
