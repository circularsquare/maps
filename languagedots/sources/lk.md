# Sri Lanka: 2024 census ethnic groups, drawn as languages, with 2012 retention

Built 2026-10-05, session `edd42a8c-lk`. Scripts: `sources/lk_census.py` (both tables, the
split), `taxonomy/lk2024.py` (labels -> nodes), `taxonomy/tree.d/lk.txt` (one new node),
`countries/lk.py`. `data/` is gitignored; this is the record.

**This is a proxy.** Sri Lanka's census asks no home-language question. Built under Anita's
2026-10-05 ruling for ethnicity-only countries (AGENT_BRIEF §2): every row `tier=derived`, said in
`how` and `note_public`, retention checked first.

## 1. Sources

| source | what it has | use |
|---|---|---|
| **CPH 2024, `GN_Level_Population_by_Ethnic_Group.xlsx`** (DCS, `.../CPH2024/GNLevel/GN_Level_Population_by_Ethnic_Group`) | 10 ethnic groups + Other on all 14,003 GN divisions, 21,781,800 people | the counts |
| **CPH 2012 district reports, Tables A30 and A32** (`.../PopHouSat/CPH2011/Pages/Activities/Reports/District/<District>/A30.pdf`; Kandy's are `Table%20A30.pdf`; Moneragala's folder is `Monaragala`) | population 10+ by ethnic group: ability to speak Sinhala, Tamil, English (A30) and each pair and all three (A32); all sectors, both sexes; 25 districts | retention |
| CPH 2024 Tables A40-A43 | ability to speak/read Sinhala, Tamil, English, 10+ | national and by sector only, not by ethnicity or district (coverage sweep); not used |
| APiCS survey 66 (Sri Lankan Malay) | "approximately 40,000 ethnic Malays ... not all ethnic Malays are fluent speakers"; 30,000-40,000 speakers; shift to Sinhala among under-50s | the only Malay retention evidence; no share |

The 2024 workbook has the same layout and disclosure rule as the religion workbook religiondots
uses: `-` is zero, and any group under 10 people in a GN division is moved into `Other`.

## 2. Method

A30 + A32 give, by inclusion-exclusion, the exact number of each group in each district who speak
each combination of Sinhala (S), Tamil (T), English (E). The A32 column order is asserted by the
partition: every combination must come out non-negative, and all 125 group-district rows do.
Heritage language: Sinhalese S; Sri Lanka Tamil, Indian Tamil, Moor T; Burgher E.

- Speaks the heritage language (alone or with others) -> heritage language.
- Does not: speaks S or T -> that one (S+E -> S for a Tamil-heritage group, T+E -> T for
  Sinhalese, S+T for Burghers -> whichever the district's whole population speaks more); English
  only -> English; none of the three -> heritage language.
- The 2012 district shares (aged 10+) are applied to every age in the 2024 GN counts, integer
  split by largest remainder so each GN's groups still sum exactly. Rows under 200 people aged 10+
  use the group's national pooled shares (11 rows: Burghers in 9 districts, Indian Tamils in
  Hambantota and Polonnaruwa). Shares used are in `data/normalized/lk_retention.csv`.
- Malay -> Sri Lanka Malay, all of them: no source gives a retention share.
- Sri Lanka Chetty (1,753), Bharatha (553) -> Tamil: Tamil-speaking communities; the 2012 ability
  table filed them under Other, so there is no row to correct them by.
- Veddahs (1,287) -> the GN division's majority of Sinhalese vs Tamil+Moor: the Vedda language
  has few speakers, inland Veddahs speak Sinhala and coastal ones Tamil.
- Other (59,489) -> `other`. It is mostly the disclosure rule's sweepings of every group, not a
  community, so no language can be assigned.

## 3. Checks and numbers

- 14,003 GN rows; groups sum to each GN's total; GNs sum to the national line, 21,781,800, and
  each group to its national figure. Languages drawn sum to 21,781,800.
- 2012 national pooled ability (10+): Sinhalese 99.95% speak Sinhala; Sri Lanka Tamil 98.10%
  Tamil, 1.88% Sinhala only; Indian Tamil 99.31% Tamil; Moor 98.63% Tamil, 1.29% Sinhala only;
  Burgher 72.77% English, 17.57% Sinhala, 9.67% Tamil.
- Colombo Moors hand-checked from the PDFs: S 177,367, T 188,758, E 120,681, S&T 168,324,
  S&E 114,220, T&E 113,558, all 107,934 of 199,222 -> 9,043 (4.54%) Sinhala without Tamil,
  matching the script.
- Drawn: Sinhala 16,230,506 (74.5%), Tamil 5,448,835 (25.0%), other 59,489, Sri Lanka Malay
  22,838, English 20,132.
- Moved off heritage: SL Tamil -> Sinhala 59,815 (Badulla 17,797, Gampaha 9,703, Colombo 8,280,
  Puttalam 4,908); Moor -> Sinhala 28,169 (Colombo 12,935); Sinhalese -> Tamil 5,793 (Puttalam,
  Batticaloa); Burgher -> Sinhala 4,045, -> Tamil 3,265; Indian Tamil -> Sinhala 2,930.
- Badulla stands out: 22% of its 23,644 "Sri Lanka Tamil" aged 10+ could not speak Tamil in 2012
  (A30: Sinhala 85.3%, Tamil 77.7%). Taken as printed; it may be a self-description shift
  (people of Tamil descent living as Sinhala speakers) rather than an error.
- scatter: 21,779 dots at 1:1000 over 12,274 polygons; 2,800 people under one dot per language.

## 4. Calls someone might reverse

- **Ability, not home language.** Bilinguals stay on the heritage language, so shift is
  undercounted. The main case is Moors in the south and west who speak Sinhala at home but can
  speak Tamil; the literature says "a few" Moors have Sinhala as first language and gives no
  number. Moving bilinguals instead would move 58.7% of Moors (2012 national S&T share), which is
  far too many.
- **Burghers on English.** 73% speak English, and Burgher households are conventionally
  English-speaking, but many now use Sinhala at home; nothing measures it. Batticaloa's
  Portuguese-creole-speaking Burghers (Glottolog mala1544) are not separable and are drawn as
  English or Tamil by ability.
- **Malays all on Sri Lanka Malay.** APiCS's 30,000-40,000 speakers against ~40,000 ethnic Malays
  (2012) suggests high retention, but the 2024 count is 22,838 and urban shift is documented.
- **2012 shares on 2024 counts**, district level, aged 10+ applied to all ages.
- **Placement**: religiondots' GN polygons have no population column, so dots are spread evenly
  inside each GN division (religiondots does the same). GN divisions average 1,550 people.

## 5. Room for improvement

A home-language or mother-tongue table would fix all of §4. The 2024 census asked language
ability (A40-A43) but published it nationally and by sector only; the same table by ethnic group
and DS division would give 2024 retention at 30x finer grain. A survey asking Moors' and Malays'
home language by district (none found) would settle the Sinhala-speaking Moor and Malay shift
questions.

## 6. Files

`data/raw/lk/GN_Level_Population_by_Ethnic_Group.xlsx`, `data/raw/lk/cph2012_lang/<District>_A30.pdf`
and `_A32.pdf` (50 files), `data/normalized/lk.csv`, `data/normalized/lk_retention.csv`.
