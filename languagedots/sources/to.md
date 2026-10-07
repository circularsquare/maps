# Tonga (to): record

Drawn 2026-10-05 (session edd42a8c-mono3). **Tonga has a home-language question** (the queue said
"unknown"): 2021 census ID7.7, three answers. 99,408 people, Tongan 98.9%, `other` 1.1%;
division shares placed on 156 villages, rows `derived`. Placed on religiondots' Kontur hexes
(read-only). 99 dots.

Files: `sources/to_census.py`, `taxonomy/to2021.py`, `taxonomy/tree.d/to.txt` (bare repeats),
`countries/to.py`, `data/normalized/to.csv`, `data/raw/to/census_report_vol1_2021.pdf`.

## Source

Tonga Statistics Department, *Tonga 2021 Census of Population and Housing, Volume 1: Basic
Tables* (tongastats.gov.to/download/272/census-report-and-factsheet/7647/census-report-vol1-2021.pdf).
Table G 48 (pp.155-158): population 5+ by language use at home x age x sex x division. TONGA:
89,254 = Tongan only 75,828 + Tongan and other 12,420 + Tongan not used 1,006. Only by division
(Tongatapu, Vava'u, Ha'apai, 'Eua, Ongo Niua). The questionnaire is on p.350.

Population base: religiondots' parse of the same census's religion table by village (99,408,
every category incl. refusals; the census's headline total is 100,179, the difference being
whatever religiondots' table excludes; its record has the detail). Division from its note column.

## Calls

- "Tongan and other language(s)" on Tongan, not shared with an unnamed remainder: the other
  language is unrecorded and these are, in nearly every case, Tongans who also use English.
- "Tongan not used at home" on `other`: the language is unrecorded (no nationality table by
  division either).
- Under-5s take their division's shares.
- Niuafo'ou language (Glottolog niua1240) not drawn: Ongo Niua answered 982 Tongan only, 57
  Tongan and other, 1 not Tongan, so the census gives it no room.

## Moved from countries/to.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- The census does not record which other language, so the 1,006 people who do not speak Tongan at home are drawn as other languages; most are expatriates and foreign-born residents in Nuku'alofa.
- Niuafo'ou has a language of its own, related to Wallisian, but nearly everyone in the Niuas division answered Tongan only, so it is not drawn.
