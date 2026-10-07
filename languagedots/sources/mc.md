# Monaco (mc): record

Drawn 2026-10-05 (session edd42a8c-last). 38,857 residents (IMSEE census, 31 December 2025),
one unit, ~76 nodes, rows `derived`. 30 dots, 51 rings.

Files: `sources/mc_census.py`, `taxonomy/mc2025.py` (identity), `taxonomy/tree.d/mc.txt`,
`countries/mc.py`, `data/normalized/mc.csv`, `data/raw/mc/imsee_recensement_2025.pdf`,
`data/geo/mc/mc_hexes.gpkg` (9 Kontur hexes; Kontur MC downloaded into `data/geo/kontur/`).

## Source

IMSEE "Recensement de la population 2025" (May 2026), Tableau 4: the 30 commonest nationalities
with counts and plurinational shares; total 38,857. imsee.mc returns 403 to curl even with a
browser UA and Wayback has no copy; WebFetch's fetch got the PDF and it was copied into raw.

## How (Anita's 2026-10-05 rule: national language + immigrants by citizenship)

- Monegasques (9,333, counted once) on French. Monegasque (Ligurian) is a school language with
  almost no native speakers; no count exists, not drawn.
- Foreign nationalities: Tableau 4 counts plurinationals in every community, so each is weighted
  count x (1 - pluri/2), then `origin_mix.mix(iso, "mc")` (Italy's home mix brings Neapolitan,
  Sicilian etc. as on Italy's own map). Weighted top 29: 27,288.
- The remaining 2,236 (114 smaller nationalities) on `other`.
- Drawn: French 46%, Italian and Italy's regional languages ~19%, English 9.5%.
- Placement: Kontur MC (9 hexes) under one box-shaped unit; Kontur's total is 1.58x the census
  (it counts commuters' or neighbours' density), only the weights matter.

## Calls someone might reverse

- Monegasques all French; French citizens on France's home mix rather than French only.
- The half-weight for plurinationals is a convention, not a figure.
