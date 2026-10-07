# Mauritius: the record

Drawn 2026-10-04 (agent d9e44929-mu). 1,233,097 people, the whole enumerated resident
population, on 182 wards and village council areas (183 census rows), 12 drawn categories,
1,227 dots and 4 single-dot languages.

## Source

Statistics Mauritius, **2022 Housing and Population Census, Volume II (Demography), Table D9**,
"Resident population by geographical location and language usually spoken at home", report pages
165-170 (PDF pages 175-180). The same 4.0 MB PDF religiondots reads for religion (Table D6):

    https://statsmauritius.govmu.org/Documents/Census_and_Surveys/Census2022/HPC_TR_Vol2_Demography_Yr22.pdf

Fetched by `python sources/mu_hpc.py --fetch` into `data/raw/mu/`. Free to download and cite.
The site is SharePoint; its .aspx pages 404 or redirect-loop, the /Documents/ tree is open
(religiondots `sources/mu.md` §1).

**The question.** Individual questionnaire, "Language(s) usually spoken at home", all residents,
up to two languages. Volume II has three language tables:

| table | level | what |
|---|---|---|
| D7 | island (3 units) | language of forefathers (ancestral language), 56 labels incl. pairs |
| D8 | island (3 units) | home language by sex, 55 labels: 25 single languages and 30 pairs |
| **D9** | **ward / VCA (183 rows)** | **home language, 12 columns, one per person** |

D9 is drawn. D7 is an identity item (Creole 569,874 ancestral against 968,952 at home) and is not
the language question this map draws. The coverage sweep's lead (district, ~10 languages) was
too coarse: the table is at the finest civil geography Mauritius has.

## How D9 files a two-language home

D9's footnote: "Includes the language mentioned together with its combination with other
languages." Read against D8, the rule is **the first-named language of the pair takes the
person**: every "Bhojpuri & X" is in Bhojpuri (including Bhojpuri & Creole, 63,101), every
"Creole & X" in Creole (Creole & French 36,762, Creole & Other 22,907...), "English & X" in
English, "French & X" in French, "Hindi & X" in Hindi. Marathi, Tamil, Telugu and Urdu columns are
single-language homes only. Everything else (Malagasy 3,172, Sinhala 463, Other Mixed European
705, Other European 192, Bengali 110, Other Mixed Indian 135, Other 314, Not stated 246, and
smaller) is "Other & Not stated". `check_d8()` rebuilds D9's republic row and its Rodrigues row
from D8's 55 labels with this rule, exactly, on all 12 columns.

**Drawn as published, all measured** (call below). Combinations are 12.8% of the population.
What §3.6's half-person split would give nationally, from D8:

| | drawn (D9) | half-split | single-language homes only |
|---|---|---|---|
| Creole | 1,045,558 | about 1,038,800 | 968,952 |
| Bhojpuri | 106,583 | about 68,200 | 29,827 |
| French | 30,039 | about 49,500 | 29,840 |
| English | 10,616 | about 11,400 | 7,029 |
| Hindi | 16,730 | about 22,500 | 16,684 |

(Half-split: each pair's people shared half to each named language; "Other"/"Oriental"/"Other
European" partners go to other languages.)

## Categories (taxonomy/mu2022.py)

```
1,045,558  84.79%  Creole              -> morisyen (ca.txt's leaf)
  106,583   8.64%  Bhojpuri            -> bhojpuri
   30,039   2.44%  French              -> french
   16,730   1.36%  Hindi               -> hindi
   14,483   1.17%  Bangla              -> bengali
   10,616   0.86%  English             -> english
    5,917   0.48%  Other & Not stated  -> other
    1,035   0.08%  Tamil               -> tamil
      997   0.08%  Chinese languages   -> sinitic (Chinese, language not named)
      413   0.03%  Urdu                -> urdu
      364   0.03%  Marathi             -> marathi
      362   0.03%  Telugu              -> telugu
```

No new nodes. Glottolog: Morisyen mori1278, with Rodrigues Creole (rodr1234) a dialect of it, so
the census's one "Creole" label is one node on both islands; Mauritian Bhojpuri (maur1239) is a
dialect of Bhojpuri (bhoj1244). `tree.d/mu.txt` repeats the needed non-tree.txt nodes identically.

## Checks (numbers from the run)

1. 209 rows parsed, every row's 12 columns sum to its Total.
2. The 183 unit rows sum to the republic (1,233,097) on all 13 columns.
3. **Second table, same census, per unit:** the 183 unit names join to religiondots' Table D6 rows
   183 of 183 both ways (whitespace-insensitive; D9 prints "Grand Baie VCA- East" where D6 has
   "VCA-East"), and every unit's D9 total equals its D6 total.
4. D8 rebuilds D9's republic and Rodrigues rows exactly (above).
5. check_country: ok, 1,233,097 drawn, 182 units, 12 languages.

Parse notes beyond religiondots' (`religiondots/sources/mu.py` has the row-shape traps): D9's stub
indentation is not a fixed ladder. Units are at x0 66.6 under districts with no urban/rural split
(pages 175-177) and at 59.0 under districts with one (178-180); districts at 44.6 everywhere. A
unit is "deeper than 57". The four towns printed at the unit depth beside their own wards (Beau
Bassin/Rose Hill, Curepipe, Quatre Bornes, Vacoas/Phoenix) are found and set aside as in D6.

## Geography

religiondots' `data/geo/mu/` read-only: 1,870 Kontur hexes over 182 units (OSM ward and VCA
polygons, every pairing checked spatially there), every unit with positive hex population. The
unit rows carry religiondots' D6 geo_ids, so its `mu_lookup.csv` maps them unchanged; Vacoas-
Phoenix Ward 5 and Ward 6-West share one polygon and are summed. Plain `pop_weight`, not
religiondots' Kenya-style weighter (its fallback is for slivers with no hex, and every unit here
has hexes). The scatter's grid-floor warning (median 8 cells per unit) is religiondots' registered
`warn` for mu (`religiondots/sources/geo_checks.csv`), inherited.

## Calls someone might reverse

- **D9's first-named allocation drawn as published, not §3.6's half-split.** The brief ranks a
  single-answer table from the office above sharing; D9 is one, though the single answer is the
  office's rule and not the respondent's. Splitting per unit would need D8's national pair mix
  applied in every ward (combinations are island-level only), which is a model. Cost of the
  choice: Bhojpuri is drawn at 106,583 where a half-split gives about 68,200. note_public says so
  with the numbers. To reverse: in `countries/mu.py`, move half of each column's pair share (D8
  ratios) to the partner language as `derived`.
- **"Other & Not stated" drawn on `other`**, though 246 of its 5,917 gave no answer: the pool
  has no split anywhere, as with religiondots' religion cell for Mauritius.
- **Bangla on `bengali`**; D8's separate "Bengali" (110) stays in D9's Other.
- **Morisyen hand-coloured** (0.80 0.12 205, `#44d4e2`) in `tree.d/mu.txt`; it was generated
  pale grey-cyan from ca.txt's bare line. OKLab distance to its Mauritian neighbours: Bhojpuri
  0.35, Hindi 0.27, Bangla 0.23, French 0.18, English 0.16, Tamil 0.11 (Tamil is 1,035 people here).

## Gap

Agalega (a few hundred residents), which the published tables leave out (religiondots
`sources/mu.md` §10). Nothing else: the universe is the whole enumerated resident population.

## Figures cut from note_public (2026-10-06)

- Table D8: Hindi alone 16,684 people, 84% men; Bangla 14,483, 81% men (foreign contract
  workforce rather than Mauritian families).
- Chinese languages, national only (D8): Mandarin 406, Chinese 369, Hakka 60, Cantonese 17,
  other 145.
- Creole includes 36,762 people in homes speaking Creole and French.

## Not done

- D7 (ancestral language) not drawn.
- A finer split of "Chinese languages" or "Other & Not stated" does not exist below island level.
- The 2011 census (IPUMS LANGMU) was not looked at; 2022 is current.
