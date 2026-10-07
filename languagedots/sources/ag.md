# Antigua and Barbuda (ag): record

Drawn 2026-10-05 (session edd42a8c-mono3). No language question: 2011 census country of birth,
national, 83,477 people (1,341 not stated left out), 12 nodes, every row `derived`. Built as
Barbados (`sources/bb.md`). Placed on religiondots' Kontur hexes for one national unit
(read-only). 79 dots, 5 rings.

Files: `sources/ag_census.py`, `taxonomy/ag2011.py`, `taxonomy/tree.d/ag.txt`, `countries/ag.py`,
`data/normalized/ag.csv`, `data/raw/ag/ag_2011_country_of_birth.pdf`.

## Source

2011 Population and Housing Census, Q58 country of birth, a Redatam WebServer output the
Statistics Division's server produced on 2024-12-15 (redatam.org/redatg/tempo/46442/~tmp_4644201.pdf;
found through Wikipedia's citation, fetched from the Wayback Machine). redatam.org's 2025
relaunch took the Antigua WebServer down (`/redatg/`, `/binatg/` all 404), so birthplace by
parish, which it could cross, is not reachable now. If the Statistics Division republishes it,
parish grain is the next step (Wikipedia's "Demographics of <parish>" pages carry such tables
from the same server, some by major division).

The 20 rows sum to 84,818, two more than the printed Total (84,816); asserted as printed.

## Mapping (`taxonomy/ag2011.py`)

Antigua and Barbuda, Montserrat, St Kitts and Nevis → Antiguan and Barbudan Creole (Glottolog
anti1245 covers the Leeward creole; node from `bb.txt`, hand-coloured here 0.55 0.12 160, kept
apart from Jamaican Creole's green). Guyana, Jamaica, St Vincent, Trinidad → their creoles;
Dominica, St Lucia → Antillean Creole; Dominican Republic → Spanish; USA, Canada, UK → English;
USVI → Virgin Islands Creole (virg1240, new); Syria → Levantine Arabic; Africa → `africa_other`;
the four "Other ..." pools → `other`.

## Calls

- National grain: one unit, so every language follows population.
- White Antiguans (English) not separable: no ethnicity table was reached; drawn as Antiguan
  Creole.
- US-born (2,608) on English, though many are children of returning Antiguans; the same call as
  Barbados.
