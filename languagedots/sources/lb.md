# Lebanon (lb): record

Drawn 2026-10-05 (session edd42a8c-arab). Lebanese residents per caza (OCHA 2026) at their
mohafaza's pooled survey language shares; Syrians and Palestinians by OCHA's caza counts on
Levantine Arabic: 5,209,087 people, 26 cazas, 8 nodes. Lebanese rows `modelled`, refugee rows
`derived`. Placed on religiondots' Kontur 400 m hexes. 5,205 dots.

Files: `sources/lb_build.py`, `sources/ab_firstlang.py`, `taxonomy/lb2026.py` (identity),
`taxonomy/tree.d/lb.txt` (all borrowed), `countries/lb.py`, `data/normalized/lb.csv`. Reads
religiondots' OCHA workbook, WVS 7 .dta and `data/normalized/lb.csv`, read-only.

## 1. Population base

religiondots' base (its `sources/lb.py`): OCHA Lebanon 2026 LRP population package, Lebanese
3,864,296 (CAS 2018-19 LFHLCS), Syrians 1,120,000, Palestinian refugees 224,791, migrants
164,097, by caza. religiondots draws the Lebanese where the register files them (ask 049);
here they are drawn where OCHA says they live, since language follows residence. Migrants are
the gap, as religiondots.

## 2. Surveys (one respondent, one vote, per pre-2014 mohafaza)

| round | item | n | used |
|---|---|---:|---|
| Arab Barometer II 2011 | first language | 1,387 | yes, except South |
| Arab Barometer III 2013 | first language | 1,200 | yes |
| Arab Barometer IV 2016 | first language | 1,500 | yes |
| WVS 7 2018 | Q272 language at home (`N_REGION_WVS`) | 1,200 | yes |
| Arab Barometer VII 2021-22 | ethnic group | 2,399 | check: Armenian 41 of 2,379 (1.7%) |

AB II's South: 11 of 179 answered English, against 0 of 408 in the South in the other rounds:
a code slip, so that round's South is left out (Iraq's WVS 4 Kirkuk rule). WVS 7 interviews all
in Arabic.

Pooled Armenian: Beirut 2.7%, Mount Lebanon 2.2%, North 0.1% (one answer), elsewhere 0. AB VII's
ethnic Armenian: Beirut 4.8%, Mount Lebanon 3.0%: the language answers sit a little below the
ethnic ones, as expected from a few Armenians answering Arabic first or interview-language bias.

## 3. Mohafaza to caza

Inside each mohafaza: Armenian by the caza's share of the 2022 register's Armenian Orthodox +
Armenian Catholic voters (religiondots' carried register; Beirut 49,848, El Meten 31,242, Zahle
10,029 of 101,307 scaled), every other non-Arabic answer by resident Lebanese, Arabic the rest.
Only placement inside the surveyed mohafaza moves (AGENT_BRIEF section 4).

## 4. Result

Levantine Arabic 5,133,291 (98.5%), Armenian 42,475, French 17,856, English 7,788, Kurdish 2,785,
Spanish 2,213, Syriac 1,601, Ukrainian 1,078. The register lists 104,000 Armenian-sect voters
aged 21+, so Armenian is likely undercounted 2-3x; said in `note_public`.

## 5. Calls someone might reverse

- Residence (OCHA) rather than religiondots' registration base for the Lebanese.
- Armenian placed by the Armenian sects' register inside a mohafaza.
- AB II South dropped; foreign first-language answers (French, English, Spanish, Ukrainian) kept
  as answered.
- Syrians all on Levantine Arabic (no count of Syrian Kurds in Lebanon); Armenian not split into
  Western Armenian.
- Kurdish from 3 answers; Lebanese Kurds (Beirut's Mhallamiye and Kurmanji speakers) mostly
  missed.

## 6. Room for improvement

- An Armenian-language survey or the community's own school figures.
- UNHCR's registered Syrians by caza with their governorate of origin (Hasakah, Aleppo) would
  separate Kurdish speakers.
- Migrant workers by nationality (ILO, IOM) would let the gap be drawn.

## Terms

OCHA package: HDX, CC BY. Arab Barometer, WVS: free, citation requested. Register via
religiondots. Kontur CC BY 4.0.
