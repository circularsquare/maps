# San Marino (sm): record

Drawn 2026-10-05 (session edd42a8c-last). 34,045 residents (register, 31 December 2024), one
unit, 3 nodes, rows `derived`. 32 dots, 1 ring.

Files: `sources/sm_census.py`, `taxonomy/sm2024.py`, `taxonomy/tree.d/sm.txt` (bare repeats),
`countries/sm.py`, `data/normalized/sm.csv`, `data/raw/sm/bollettino_202503.pdf`,
`data/geo/sm/sm_hexes.gpkg` (108 Kontur hexes; Kontur/register 1.06). Kontur SM downloaded into
`data/geo/kontur/`.

## Sources

- Bollettino di Statistica I trimestre 2025 (statistica.sm), Tavola 1.8, residents December
  2024: Sammarinesi 28,204, Italiani 5,030, Altri 811 (asserted to sum to 34,045). No census or
  register asks language; "Altri" has no published breakdown.
- Romagnol: no Sammarinese count. Foresti's 1998 survey (Treccani, "La Repubblica di San
  Marino", 2020): ~70% grew up with dialect present, alone or with Italian; less than a third
  Italian only; dialect concentrated among the old and called endangered.

## How

Romagnol = 17.6% of Sammarinese and Italian citizens: Rimini province's rate on Italy's map
(it.csv ITH59: Romagnol / (Romagnol + Italian), from ISTAT 2024's Emilia-Romagna
dialect-in-the-family share, "both" counting half). The rest of those two citizenships on
Italian; "Altri" on `other`. National grain (castello populations exist in Tavola 1.7, but no
castello polygons are on hand and it would move a few dots).

## Calls someone might reverse

- The borrowed Rimini rate (could be higher in San Marino's older, more rural castelli, per
  Foresti; Pesaro-Urbino's rate on Italy's map is 31.5%).
- Italian residents given the same rate as Sammarinese.
