# Vatican City (va): record

Drawn 2026-10-05 (session edd42a8c-last). 882 inhabitants (31 December 2024), one unit, 3 nodes,
rows `derived`. No dots (every language under 1,000), 3 rings.

Files: `sources/va_pop.py`, `taxonomy/va2024.py`, `taxonomy/tree.d/va.txt` (bare repeats),
`countries/va.py`, `data/normalized/va.csv`, `data/geo/va/va_hexes.gpkg` (3 Kontur hexes; Kontur
VA downloaded into `data/geo/kontur/`; its 22,112 people are Rome spilling over, only the
weights matter).

## Source

vaticanstate.va, "Population" (English page; the Italian URL now 404s): 673 citizens, 458 living
inside the walls including 120 Swiss Guards; 882 inhabitants in all. No nationality or language
breakdown.

## How

The 120 Swiss Guards on Switzerland's home mix (`origin_mix`: German 84, French 27, Italian 9);
the other 762 on Italian, the state's working language. Clergy residents' own first languages
are not counted anywhere; said in `note_public`.

## Calls someone might reverse

- Everyone but the Guard on Italian (the clergy are international).
