# Belize: 2022 census, languages spoken, district

**Drawn** 2026-10-05 (agent d9e44929-bz). 368,209 people on 6 districts, 11 nodes, 363 dots at
1:1000, 3 rings. Every row `derived` (a multi-answer question, spec §3.6).

## Source

- Statistical Institute of Belize (SIB), 2022 Population and Housing Census (census day 12 May
  2022). `Census2022_GeneralCharacteristics.xlsx`, sheet `Languages_Spoken`, "Table 7: Population
  Four Years and Older by Languages Spoken and District: 2022", a plain link on
  https://sib.org.bz/census/2022-census/. Same workbook religiondots read for religion (Table 9).
- The question, Individual Questionnaire 1.6 (`2022CensusIndQuestionnaire.pdf` on the same
  page), asked of everyone 4 and over: "Which language(s) do you/does N speak well enough to
  conduct a conversation? [MULTIPLE RESPONSES ALLOWED]". Precoded: Creole, English, Spanish,
  Garifuna, German, Maya Yucatec, Maya Ketchi, Maya Mopan, Chinese, Hindi, Other (specify),
  Cannot speak, DK/NS. There is no other language question (no mother tongue, no home language),
  so no single-answer table exists for 2022.
- `CensusKeyFindingsReport_2022.pdf` (same page), Tables 3.8 and 3.9: the same figures as shares,
  plus a 4+ population per district. Used as the second-table check.
- Figures are SIB's undercount-adjusted counts and are fractional (religiondots/sources/bz.md).
- Licence: none stated; an official statistics office's published tables, open and keyless.
- Not used: 2010 census on REDATAM (`redatam.sib.org.bz`, which the coverage sweep flagged as a
  route to finer units or combinations). Tried 2026-10-05: the portal URL now 302-redirects to an
  unrelated page (`/report-bees`). 2010 asked the same multi-answer question anyway (the Key
  Findings Report's Figure 3.3 compares the two). 2022 microdata are not public.

## Checks (`sources/bz_census.py`, all asserted, all pass)

1. Table 7's title, the seven column groups (Total plus the six districts, north to south) and
   the Total/Male/Female triple under each, and the twelve row labels in order.
2. The six districts sum to the national column on all 11 rows with no suppressed cell
   (relative 1e-6). One Total cell is printed `<10`: Hindi in Toledo, recovered as national less
   the other five districts, 3.517, asserted under 10.
3. Male + Female == Total on all 82 cells where both sexes are printed (0 failures): the check
   on the read.
4. The 4+ population: all ages (Table 9 Total row) less under-4s, where under-4s per district
   are Table 6's `Less than 1` plus `1-4` times the national share of ages 1-3 in 1-4 (Table 4,
   single ages, national only, 0.7306). Nationally that is 368,924.4; the Key Findings Report
   prints 368,924.
5. Key Findings Table 3.9's 84 printed shares (12 labels x Belize and 6 districts) against
   Table 7 over Table 3.9's own 4+ row: worst difference 0.051 pp, at one-decimal printing.

## The 4+ population, and why the report's row is not the one drawn

Table 7 has no universe row, and §3.6 needs one per district. Two candidates:

| district | all ages | under 4 (age tables) | 4+ (age tables) | 4+ (Key Findings 3.9) | diff |
|---|---|---|---|---|---|
| Corozal | 45,310 | 3,156 | 42,154 | 42,103 | -51 |
| Orange Walk | 54,152 | 3,874 | 50,278 | 50,219 | -59 |
| Belize | 113,630 | 7,238 | 106,392 | 106,522 | +130 |
| Cayo | 99,105 | 7,441 | 91,664 | 91,664 | 0 |
| Stann Creek | 48,162 | 3,600 | 44,562 | 42,971 | -1,591 |
| Toledo | 37,124 | 3,249 | 33,875 | 35,446 | +1,571 |

Four districts agree within 130 people (the residue of the 1-3/1-4 split being national). The
report's Stann Creek figure would leave 5,191 under-4s in a district whose Table 6 has 4,597
under-5s, which is impossible, and Toledo's is too high by about the same amount: it looks like
~1,580 people tabulated in the wrong one of the two southern districts in the report's
denominator. **Drawn on the age-table figure.** The shares inside each district come from Table
7 either way; the choice moves only how many people Stann Creek and Toledo draw (±1,580). Both
rows are in `data/normalized/bz.csv`.

## How it is drawn (`countries/bz.py`)

Per district, P = 4+ population less `Cannot Speak`; each language's count = its mentions x P /
the district's total mentions (spec §3.6). 721,403 mentions from 368,208 people, 1.959 each;
1.80 in Corozal up to 2.07 in Stann Creek. National result: English 38.7%, Spanish 28.0%, Belize
Kriol 24.8%, Q'eqchi' 3.1%, Mopan 1.9%, German 1.6%, Garifuna 1.0%, other 0.4%, Yucatec 0.3%,
Chinese 0.2%, Hindi 0.1%.

- **DK/NS is not published** (the questionnaire has it; Table 7 does not print it). It is inside
  P and shared like everyone else. Its size is unknown; religion's DK/NS in the same census was
  1.0%.
- **Gap:** 28,559 under-4s (7.2%), not asked, and 716 `Cannot Speak`.
- **English is overstated as a first language**, by design of the question: it is the official
  and school language and 75.5% named it. Under §3.6 it takes a share of everyone who named it.
  Said in `note_public`. No second source gives first language for Belize; none corroborates or
  corrects the split.

## Mapping calls (`taxonomy/bz2022.py`, `taxonomy/tree.d/bz.txt`)

- `Speaks Creole` -> new leaf `creole.english_based.belize_kriol`, "Belize Kriol" (Glottolog
  beli1260, Belize Kriol English). Colour hand-picked 0.62 0.14 128 (#6d952c), a darker olive
  green, apart from Garifuna's light mint (#75d78d) on the Stann Creek coast, English's pale blue
  and Spanish's ochre.
- Q'eqchi', Mopan, Maya (Yucatec), Garifuna, German, Hindi, English, Spanish: existing nodes.
- `Speaks German` -> German, though most are Mennonites whose everyday speech is Plautdietsch
  (Glottolog has a Belize Plautdietsch dialect, beli1263): the census says German, as Paraguay's.
- `Speaks Chinese` -> `sinotibetan.sinitic` (drawn unwashed as "Chinese"), as other countries.
- `Speaks Other` -> `other`. 1,676 of its 2,475 mentions are in Orange Walk, the Mennonite
  district, so it may be largely Plautdietsch or Dutch written in; nothing published says so.
- Colours not retuned: German (#359bd9) and Mopan (#688fe8) are both blues and both present in
  Cayo, but at 3.8% and 2.0% of mentions there and different enough in hue; left.

## Geography

religiondots' placement layer for Belize, read-only: `data/geo/bz/bz_hexes.gpkg`, Kontur 400 m
hexes keyed to COD-AB district pcodes (BZ01 Belize ... BZ06 Toledo), 5,095 hexes, joined and
checked there (religiondots/sources/bz_geo.py). SIB prints districts north to south and COD-AB
codes them alphabetically, so `sources/bz_census.py` maps name -> pcode explicitly, the same
table religiondots uses. Placement inside a district is plain population: no village-level
language or ethnicity figure is published (SIB publishes population by village, not
ethnicity), so Q'eqchi' in Toledo is spread over Punta Gorda as well as the Maya villages.

## Re-running

```
python sources/bz_census.py --fetch    # xlsx 69 KB + pdf 2.7 MB into data/raw/bz/
python taxonomy/build.py
python tools/check_country.py bz
python scatter.py --country bz
```
