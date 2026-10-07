# Jamaica (jm): record

Drawn 2026-10-05 (session edd42a8c-carib). Jamaican Creole 82.9% and English 17.1%, national
shares from the Jamaican Language Unit's Language Competence Survey 2006, on the 2011 census
parish populations: 2,678,981 people over 14 parishes, every row `modelled`. Placed on
religiondots' 400 m grid (read-only).

Files: `sources/jm_jlu.py`, `taxonomy/jm2006.py`, `taxonomy/tree.d/jm.txt` (bare repeats),
`countries/jm.py`, `data/normalized/jm.csv`, `data/raw/jm/` (the two JLU PDFs below).

## 1. Sources

- No census language question (2011 asks ethnic origin and religion; the 2022 questionnaire was
  not found online; scout 2026-10-05).
- **JLU Language Competence Survey of Jamaica (LCS), fielded 2006**, *Data Analysis* report
  (UWI Mona, September 2007,
  mona.uwi.edu/dllp/sites/default/files/dllp/The Language Competence Survey of Jamaica - Data
  Analysis.pdf). 1,000 adults 18+, quota-stratified by region (west 400 / east 600), urban/rural
  (500/500), age and sex. Interviewers (UWI students) ran a scenario, a prompt and a debrief and
  recorded which languages the respondent actually produced. Table 4: English only 171 (17.1%),
  Jamaican only 365 (36.5%), both 464 (46.4%). Table 5: west 13.5% English only, east 19.5%.
  Table 6: urban 20.6%, rural 13.6%.
- **JLU Language Attitude Survey 2005** summary (same site): self-declared, 78.6% speak both,
  10.9% English only, 10.5% Jamaican only.

## 2. The decision

Neither survey asks first language. Jamaican is acquired at home by the great majority and
English through schooling (the standard description, and the JLU's own premise), so a bilingual
is read as Jamaican-first. The English-only group is read as English-first. That gives 82.9 /
17.1 from the LCS. The LCS's English-only share probably overstates English as a first language
(respondents who would not speak Patwa to student interviewers count as English-only; the report
shows the share moves with the interviewers' gender and the language they opened in), while the
LAS's 10.9% is self-report under the same stigma. The LCS is used because it measured behaviour
and is the one with published breakdowns; `note_public` gives both figures.

**Not drawn by region.** The LCS regions (west/east) are not defined by parish in the report, and
the 2005 round used west/central instead, so mapping them onto parishes would be a guess.
Urban/rural would need parish urban shares; not done. Every parish gets the national share.
**Flag for the supervisor**: English is a learned language for most Jamaicans but a real first
language for an English-dominant middle class, so drawing it at 17.1% rather than folding it
into Jamaican (§2's learned-language rule) is a judgement call made here, not a rule applied.

Immigrants: foreign-born are about 1% (2011), mostly from English-speaking countries; not drawn
separately.

## 3. Mapping

`Jamaican (Patwa)` on `creole.english_based.jamaican` (already used by us, ca, uk, fr, pt);
`English` on English. No colour changes.

## 4. Population and placement

2011 census parish totals as religiondots normalised them (all religion rows incl. not stated,
summed; 2,678,981; religiondots notes 4,124 people STATIN left out of parish tables, not
recoverable). Grid units are the same `JAM_GEO1_nn` ids, so no join beyond the 14-unit assert.
The 2022 census parish totals would be newer if someone fetches them.

## 5. Scatter

2,678 dots over 2,062 cells.
