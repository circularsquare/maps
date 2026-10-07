# Marshall Islands (mh): record

Drawn 2026-10-05 (session edd42a8c-last). 42,418 people (2021 census), national, 2 nodes, rows
`derived`. 41 dots. Placed on religiondots' `mh_hexes.gpkg` (one unit `MH`, read-only).

Files: `sources/mh_census.py`, `taxonomy/mh2021.py`, `taxonomy/tree.d/mh.txt` (bare repeats),
`countries/mh.py`, `data/normalized/mh.csv`, `data/raw/mh/rmi_census_2021.pdf`.

## Source

RMI 2021 Census of Population and Housing, Analytical Report
(infomarshallislands.com, PDF), Table 3.3 p. 15: of 36,808 people aged 5+, 96.0% speak
Marshallese, 23.9% speak other languages (several allowed). Table 2.1: 42,418 enumerated.
Table 3.4: 93.2% citizens. The report prints no breakdown of the other languages and no
ethnicity table by group (only "95.6% Marshallese" in the summary).

## How

Marshallese 40,721 (96.0% of 42,418, the 5+ rate applied to all ages); 1,697 who do not speak
Marshallese on `other`. English as an additional language is not drawn (AGENT_BRIEF section 2,
learned second languages). National only: Table 3.3 gives Majuro 94.7%, Kwajalein 97.2%, rural
98.3%, but religiondots' layer is one unit and the difference is a handful of dots.

## Calls someone might reverse

- The 4% non-speakers drawn as language not named rather than guessed (Filipino, Gilbertese,
  Chinese and English are all plausible; no table says).
- National grain, though the report has Majuro / Kwajalein / rural shares.

## Room for improvement

The census microdata or a fuller tabulation (SPC PDH.Stat) naming the other languages and the
ethnicity of non-citizens.
