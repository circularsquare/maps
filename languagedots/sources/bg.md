# Bulgaria: Census 2021, mother tongue

Built 2026-10-04 (session d9e44929-bg). Rebuild:

```
python sources/bg_census.py [--fetch]   -> data/normalized/bg.csv
python taxonomy/build.py
python tools/check_country.py bg
python scatter.py --country bg
```

Drawn: 5,842,873 people on 265 obshtini (municipalities), 4 labels on 4 nodes, 5,840 dots, no
rings. Not drawn: 676,916 (10.4%), the gap: 616,681 `Непоказан` (added from administrative
registers, never asked), 49,602 `Не желая да отговоря` (do not wish to answer), 10,633
`Не мога да определя` (cannot determine).

## 1. The table

National Statistical Institute (NSI), Census 2021, reference date 7 September 2021. The results
page https://www.nsi.bg/statistical-data/151/1349 lists nine workbooks; no login, browser
User-Agent (TLS verification is switched off in the fetch, as religiondots' bg.py does).

| file | what | use |
|---|---|---|
| `Census2021_Ethnocultural characteristics_BG.xlsx` (79 KB, `file/download/d6bebe...`), sheet 3 | mother tongue by country, 2 NUTS1, 6 NUTS2, 28 oblasti, 265 obshtini | drawn |
| same workbook, sheets 2 and 4 | ethnic group and religion by obshtina | checks |
| `file/24016/Census2021-ethnos.pdf` (press release, 15 pp.) | national figures in the text (p. 6-7), Table 2 mother tongue by oblast (p. 14) | checks |

This is the same workbook religiondots drew religion from (its sheet 4); this project fetches its
own copy into `data/raw/bg/`.

**The question**: майчин език, mother tongue, defined in the press release as "the first language
learned at home in early childhood". One answer. Voluntary at the 1992, 2001, 2011 and 2021
censuses. Columns: Bulgarian, Turkish, Romani, `Друг` (other), cannot determine, do not wish to
answer, `Непоказан` (footnote 1: people added from administrative registers for whom the
registers held no information on this).

**Finer categories**: none. The press release prints the same three languages plus other,
nationally and by oblast; none of the other eight workbooks has a language table. The census's
own site (statistical-data/151/489) and the 2011 portal were not searched for a breakdown of
`Друг`; a national-only breakdown could not be drawn on the map anyway.

**Finer geography**: none found. Sheet 3 is obshtina and nothing below; the workbook's sheet 1
lists ЕКАТТЕ (settlement code) in its legend but holds national rows only.

## 2. Reading the sheet

Column 0 is NSI's code, column 1 the Bulgarian name. Levels by code shape: `BG` country, `BG3`
NUTS1, `BG31` NUTS2, three letters an oblast (`VID`), three letters and two digits an obshtina
(`VID09`). Rows without a code are the notes and legend at the foot. `-` is zero. No `..`
(suppressed) cell appears; the parser stops if one does. Labels are written to `bg.csv` verbatim,
footnote digit included (`Непоказан1`).

## 3. Checks (all pass)

- level counts 1, 2, 6, 28, 265.
- national total 6,519,789; the national row equals the press release's text in all seven
  categories (5,037,607 / 514,386 / 227,974 / 62,906 / 10,633 / 49,602 / 616,681).
- the seven categories partition all 302 rows.
- obshtini sum to their oblast (by code prefix) in every column; each of the NUTS1, NUTS2, oblast
  and obshtina levels sums to the country in every column. Exact; no rounding.
- **press release Table 2**, parsed from the PDF: all 28 oblast rows and the national row equal
  the workbook in all 8 columns. A second printing of the same counts, not an independent source.
- **same census, other sheets**: every obshtina's total equals the ethnicity sheet's and the
  religion sheet's; `Непоказан` equals the religion sheet's `Непоказано` in every obshtina (the
  same 616,681 register-added people). The ethnicity sheet's `Непоказана` is smaller, 467,678:
  registers held an ethnicity for 149,003 people they held no language for.
- **join**: all 265 obshtina codes match religiondots' `data/normalized/bg.csv` both ways with
  identical totals. religiondots' `bg_geo.py` already asserts the same 265 codes equal GISCO LAU
  2021's `LAU_ID` and the grid's `unit`; `check_country.py` finds the 265 units in the placement
  layer.
- **mother tongue against ethnic group, by obshtina** (a sanity check, not an identity): Turkish
  514,386 vs Turkish ethnicity 508,378, r = 0.996; Romani 227,974 vs Roma ethnicity 266,720,
  r = 0.967; Bulgarian r = 0.9999. Turkish mother tongue most exceeds Turkish ethnicity in
  Dulovo (+1,366), Razgrad (+1,109), Dobrichka (+960), Glavinitsa (+848), Novi Pazar (+830), the
  Ludogorie, where Turkish-speaking Roma live. Romani falls furthest short of Roma ethnicity in
  Sofia (-3,118), Botevgrad, Pernik, Byala Slatina and Dulovo. The press release's cross-tab
  agrees: of the Roma ethnic group, 82.8% gave Romani, 11.3% Bulgarian, 5.4% Turkish.

## 4. Mapping (taxonomy/bg2021.py)

| label | count | node |
|---|---:|---|
| Български | 5,037,607 | `indoeuropean.slavic.south.bulgarian` |
| Турски | 514,386 | `turkic.turkish` |
| Ромски | 227,974 | `indoeuropean.indoaryan.romani.romani` (variety not stated) |
| Друг | 62,906 | `other` |
| Не мога да определя, Не желая да отговоря, Непоказан1 | 676,916 | not drawn (gap) |

No new tree nodes, so no `tree.d/bg.txt`.

`Друг` sits on `other`, not on a regional remainder: it mixes long-settled minorities (Armenian,
Aromanian, Vlach Romanian, Russian, Greek) with recent migrants, and its largest count is Sofia's
17,131. The census gives no way to tell the two apart. Its largest shares are on the Black Sea
coast: Suvorovo 7.6%, Nesebar 6.4%, Byala 6.1%, Avren 4.9%, Balchik 4.3%. Nothing published says
which languages those are.

**Pomaks**: Bulgarian-speaking Muslims in the western Rhodopes (Smolyan, Blagoevgrad's Gotse
Delchev area) answer Bulgarian and are drawn as Bulgarian. The census has no column that would
separate them and a language map should not.

## 5. Geography

religiondots' `data/geo/bg/bg_grid_1km.gpkg`, read only: NSI's own Census 2021 1 km population
grid (`POPGRID2021_1000M`), keyed to obshtina, 21,959 cells, `pop` 6,461,591 (99.1% of the
census; NSI's metadata gives the 0.9% shortfall as records with no point location). Dots inside an
obshtina follow cell population; all 544 drawn (unit, language) rows placed on population, none on
equal shares. The grid's licence allows mapping products but not showing individual cell values,
which this does not. Kontur is not involved, so no cap block.

## 6. Numbers from the build

- 265 units, 1,038 (unit, node) rows, 5,842,873 people, 5,840 dots at 1:1000.
- 2,873 people (0.05%) under one dot per language nationally, no dot.
- Gap by obshtina: median 7.0%, range 0.6% to 17.9% (Sofia, Stolichna, of which 17.0% is
  register-added). Next: Ihtiman 16.9%, Straldzha 16.2%, Avren 15.5%, Nesebar 15.1%.
- Turkish outnumbers Bulgarian in 33 obshtini; the highest shares of those answering are
  Chernoochene 97.6%, Venets 92.7%, Ruen 87.2%, Hitrino 86.9%, Momchilgrad 86.3%.
- Romani's highest shares: Maglizh 34.6%, Tvarditsa 32.5%, Ihtiman 28.5%, Kotel 26.1%.

## 7. Colours

No change. Bulgarian `#00978a` (cz.txt, ua.txt), Turkish `#d476cd` (cz.txt, us.txt), Romani
`#9e68ae` (generated under cz.txt's Romani group). OKLab distances: Bulgarian-Turkish 0.27,
Bulgarian-Romani 0.21, Turkish-Romani 0.11, Bulgarian-other 0.12. Turkish and Romani share
villages in the Ludogorie and are the closest pair; 0.11 is the margin Serbia's agent accepted,
both nodes are used by several countries already, so they were left alone. If they read too close
on the map, Romani is the one to move (lighter or bluer), and every Romani country should be
looked at when it is.

## 8. Wording

- `how`: "census, 2021, mother tongue".
- `grain`: 265 obshtini, 24,600 people on average (6,519,789 / 265).
- `gap`: the three undrawn columns, with counts.
- `note_public`: what was asked, the three named languages plus other, where Turkish leads, the
  Pomaks drawn as Bulgarian, Roma who gave another mother tongue, the register-added people and
  Sofia's gap, the 1 km grid.
