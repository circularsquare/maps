# Vanuatu: 2020 census, first language learnt to speak, 66 area councils

Drawn 2026-10-04 (session d9e44929-vu). `sources/vu_census.py`, `taxonomy/vu2020.py`,
`taxonomy/tree.d/vu.txt`, `countries/vu.py`. Ask 008 (the not-asked remainder), ruled
2026-10-05: draw them as Bislama, derived.

**269,281 people aged three and over, 4 nodes, 66 units (4,100 on average), 267 dots.** Of them
239,833 were asked (measured) and 29,448 were not and are drawn on Bislama (`derived`). Vanuatu
languages 73.6%, Bislama 23.8%, English 1.8%, French 0.7%.

## 0. Ruling on ask 008 (2026-10-05)

Anita: "ok", draw the not-asked as Bislama. `DRAW_NOT_ASKED = True` in `countries/vu.py`;
re-scattered 2026-10-05 by session d9e44929-rulings (`check_country` ok; 267 dots, 0 rings; 2,281
people, 0.85%, under one dot per language nationally). `how` now ends "everyone else aged three
and over drawn as Bislama"; `gap` drops the 29,448; `note_public` says they are drawn as Bislama
and that in the two towns Bislama is somewhat overcounted and English and French undercounted.
Sections 2 and 6 below describe the build before the ruling (237 dots, the 29,448 undrawn).

## 1. The table

*2020 National Population and Housing Census, Basic Tables Volume 1* (VNSO), **Table 6.16**,
"Population 3 years and over living in private households with first language learnt to speak by
sex and region", PDF pages 188-191 (printed 178-181). The volume is religiondots' cached copy,
`../religiondots/data/raw/vu/vu_2020_basic_tables_vol1.pdf` (read-only; `--fetch` downloads it
into `data/raw/vu/` only if that copy is gone). URL:
`https://vnso.gov.vu/images/Public_Documents/Census_Surveys/Census/2020/Basic_Tables/2020NPHC_Volume_1_-_Version_2.pdf`.

Answers: English, French, Bislama, Indigenous (Vernacular), Not stated, each by sex (Not stated
printed as one column headed "Male"; it is the total). Rows: nation, urban/rural, Port Vila,
Luganville, six provinces, 64 rural area councils: the same hierarchy as Table 3.5, which
religiondots draws. The coverage sweep's lead was right on table, level and categories.

## 2. Who was asked: not everyone

The questionnaire (Volume 1 appendix, PDF p374): E9 "Can XYZ speak an indigenous (vernacular)
language?" (yes easily / yes with difficulty / no), then E10 "What is the first language XYZ
learned to SPEAK?" with the condition `speak_language!=3`. Volume 2 (Analytical Report, p65) says
the same in words. So Table 6.16's universe is people aged 3+ who can speak an indigenous language.

Table 6.17 (numeracy, same universe, everyone asked) gives the full 3+ population per unit:
269,287 nationally against 6.16's 239,839. **29,448 (10.9%) were not asked.** By unit: Luganville
32.3%, East Malo 28.2%, Port Vila 22.4%, Central Malekula 20.6%, Eratap 19.6%, South East Santo
19.1%, Malorua 18.3%. vu.csv carries them per unit as `Not asked (speaks no indigenous
language)` = 6.17 Total - 6.16 Total. **Not drawn**; `gap` and `note_public` say so. Drawing them
on Bislama as `derived` (the spec 3.5 pattern) is behind `DRAW_NOT_ASKED` in `countries/vu.py`,
off, and is ask 008. My reason for leaving it off: 3.5 was ruled for indigenous-only censuses,
this is a different skip, and the complement's first language is a mix of Bislama, English and
French that nothing here splits.

## 3. Two printing faults in Volume 1, both corrected and asserted

- **Table 6.16 has no Aneityum row.** Every other table prints it last under TAFEA (3.5, 6.17);
  6.16 stops at Futuna, but its TAFEA row still includes Aneityum. Recovered as TAFEA minus the
  ten printed councils, column by column: 1,261 asked (English 28, French 6, Bislama 174,
  Indigenous 1,052), against 1,338 aged 3+ in 6.17 and 1,484 of all ages in 3.5. Every
  recovered cell is non-negative; its Male + Female miss its Total by at most 9 (a difference of
  twelve independently rounded rows).
- **Table 6.17 swaps Port Vila and Luganville.** It prints Port Vila 15,978 and Luganville 44,856
  aged 3+; Table 3.5 has Port Vila at 48,461 of all ages and Luganville at 17,407, and 6.16 has
  34,802 Port Vila residents asked. Swapped back; the file stops if the printed rows ever stop
  looking swapped.

## 4. Checks (`python sources/vu_census.py`, all asserted unless said)

- 66 units = 64 councils + 2 towns (TORBA 7, SANMA 9, PENAMA 10, MALAMPA 10, SHEFA 17, TAFEA 11),
  and their names equal religiondots' `vu_lookup.csv` geo_ids both ways.
- 6.16 answers + Not stated = printed Total on 52 of 75 rows; the other 23 miss by 1. Male +
  Female = Total everywhere within 2 (78 cells off). VNSO rounds each cell on its own;
  religiondots met the same in Table 3.5 (`../religiondots/sources/vu.md`).
- Port Vila + Luganville = URBAN, URBAN + RURAL = VANUATU, provinces = RURAL, each province's
  councils = the province: within 2 in 6.16 and within 3 in 6.17, every column.
- 6.16 Total (asked) <= 6.17 Total (aged 3+) <= Table 3.5 Total (all ages) on all 66 units; the
  3+ share of all ages runs 0.888 (North Tanna) to 0.946 (Merelava).
- **Not asserted: Volume 2 does not agree.** Its p65 gives, of those asked, 84.8% raised in an
  indigenous language, 12.4% Bislama, 2.0% English, 0.8% French; Table 6.16 gives 82.6%, 14.5%,
  2.1%, 0.8%. Every Volume 2 figure is more vernacular than the table (urban 69.5% vs 64.4%,
  Port Vila Bislama 25.0% vs 30.2%, Luganville 30.7% vs 35.0%). Too large and too one-sided for
  rounding, and Volume 1 is "Version 2", so Volume 2 was perhaps written from an earlier cut or
  on a different universe; it does not say. Volume 1's own sums all hold, so the table is drawn.

## 5. Mapping and tree

- English, French, Bislama on the existing nodes (Bislama is au.txt's, under English-based
  creoles).
- **Indigenous (Vernacular), 82.6%** is every indigenous language in one answer, on a new group
  node `austronesian.oceanic.vanuatu`, "Vanuatu languages", drawn washed out as "language not
  named" (spec 3.2). Glottolog: 118 spoken indigenous languages with VU in their countries, all
  Oceanic (ocea1241), in three branches: North and Central Vanuatu 106, Southern Melanesian 9,
  Central Pacific 3 (the Polynesian outliers Emae, Futuna-Aniwa, Mele-Fila). No Glottolog
  subgroup below Oceanic holds just them, so the node is areal, like `australian`; it contains
  exactly what the census filed there. Colour: Oceanic's own (0.70 0.13 225).
- Not stated (3) is not drawn.
- Colour check: in the towns, the washed blue of the group sits beside Bislama's olive
  (#a6a355), English's pale blue (#dbe6f2) and French's lavender. Bislama, the one that matters,
  stands apart; English is 2% and was left.

## 6. Geography

religiondots' placement layer as is: `../religiondots/data/geo/vu/vu_hexes.gpkg`, 4,946 Kontur
hexes over the same 66 units, joined through its `vu_lookup.csv` (name -> COD-AB pcode; it
checked the join against OCHA's ADM1 for all 64 councils). `pop_weight`. Scatter: 237 dots at
1:1000, 0 rings; 2,829 people (1.18%) are under one dot per language nationally. water.py reports
32 units losing over 95% to the sea and leaves them unclipped; religiondots shares that cache, so
it is the same there and not a language-map issue.

## 7. Leads not followed

- The 2009 census asked everyone "language mostly spoken at home" (nationally 63% local language,
  34% Bislama, 2% English, 1% French). Its Basic Tables Vol 1
  (`https://moet.gov.vu/docs/vnso/National%20Population%20and%20Housing%20Census%20Basic%20Tables%20Report%20-%20Vol%201_2009.pdf`)
  and Analytical Vol 2 (education.gov.vu) would show how Bislama-heavy the non-vernacular town
  population is, the evidence ask 008 needs. moet.gov.vu timed out from here (2026-10-04); not
  retried, no Wayback attempt yet.
- No source names the indigenous languages by place. Language-area maps (Francois et al. 2015)
  could split the vernacular answer geographically, but that would be a model, not a count.
