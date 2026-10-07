# Samoa (ws): record

Drawn 2026-10-05 (session edd42a8c-mono3). No language question: all 204,339 Samoan citizens on
Samoan, 25 traditional districts, rows `derived`; the 1,218 non-citizens not drawn. Haiti model
(`sources/ht.md`). Placed on religiondots' Kontur hexes (read-only). 204 dots.

Files: `sources/ws_census.py`, `taxonomy/ws2021.py`, `taxonomy/tree.d/ws.txt` (bare repeats),
`countries/ws.py`, `data/normalized/ws.csv`.

## Sources and why

- 2021 census workbook (religiondots' `data/raw/ws/CensusTablesEXCELFiles.xlsx`, 49 sheets): no
  language or ethnicity table (coverage sweep 2026-10-03; rechecked: only Table 8a citizenship
  and 8b non-citizens' reason for staying). No survey with a language item and district codes
  was searched for: with Samoan near-universal it could not change the map.
- Village populations from religiondots' 2021 religion table (no not-stated cell, so it sums to
  each village), folded to religiondots' 25 traditional districts by its `ws_lookup.csv`.
- Table 8a gives non-citizens (1,218) by census district, folded the same way; the script asserts
  Table 8a's district totals equal the folded village totals unit by unit.

## Calls

- Non-citizens not drawn, rather than guessed onto a language: Table 8b says why they stay
  (work, study, mission...), not where they are from. Said in `gap` and `note_public`.
- English-speaking households (part-European families in Apia) drawn as Samoan; no count exists.
- Tokelauan and other Pacific minorities among citizens have no count; drawn as Samoan.
