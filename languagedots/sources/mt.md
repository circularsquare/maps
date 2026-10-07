# Malta: Census of Population and Housing 2021, main language spoken from early childhood

Built 2026-10-05 (session edd42a8c-mt). Rebuild:

```
python sources/mt_census.py [--fetch]   -> data/normalized/mt.csv
python taxonomy/build.py
python tools/check_country.py mt
python scatter.py --country mt
```

Drawn: 497,265 people aged 5 and over, 7 answers on 7 nodes, 494 dots, 0 rings. Maltese
citizens (386,091) by 68 localities; non-Maltese residents (111,174) by 6 districts, placed among
each district's localities by where its non-Maltese residents live. All rows `measured`.

## 1. The table

NSO Malta, Census of Population and Housing 2021, Final Report **Volume 3** ("health, education,
labour and languages"), `https://nso.gov.mt/wp-content/uploads/volume3-Census-of-Population-2021.pdf`,
117 pages, 6,930,701 bytes, `%%EOF` present. `nso.gov.mt` answers 403 to curl with a browser
User-Agent (religiondots met the same wall for Volume 1, which Anita downloaded by hand). The
Wayback Machine's capture of 2026-08-11 serves the file (`.../web/20260811082947id_/<url>`), and
`--fetch` takes it from there.

The question: **main language spoken from early childhood**, asked of everyone aged 5 and over.
Seven answers are printed everywhere: Maltese, English, Italian, German, French, Arabic, Other.
No not-stated column, and every row's answers sum to its total. The volume has no definition of
the question in its glossary (p.116 defines only literacy) and does not say how non-response was
handled.

| table | page | content | use |
|---|---|---|---|
| 3.1 | 53 | nation x citizenship x sex | check |
| 3.3 | 55 | 6 districts x Maltese / non-Maltese | **non-Maltese drawn from here** |
| 3.6 | 58-60 | **Maltese citizens only**, 68 localities | **Maltese citizens drawn from here** |
| Vol. 1, 2.1 | 116-118 | population by citizenship (all ages) x 68 localities | placement of non-Maltese |

The coverage sweep's lead ("locality breakdown for Maltese nationals; foreign residents may be
tabulated separately") was right: no table gives non-Maltese residents' language below district.
Volume 1 is religiondots' copy, read in place, never written.

## 2. Checks (all asserted in sources/mt_census.py)

- Every row in Tables 3.1, 3.3 and 3.6 sums across the seven answers to its printed total.
- Table 3.6: the localities sum to their district in every column; its six district rows equal
  Table 3.3's Maltese rows; its national row equals Table 3.1's Maltese row (386,091).
- Table 3.3: Maltese + non-Maltese = district total for all 6 districts; districts sum to the
  national row, which equals Table 3.1 (497,265).
- Vol. 1 Table 2.1: sexes sum, Maltese + non-Maltese = total, localities sum to districts.
- Each locality's Maltese citizens aged 5+ (3.6) are no more than its Maltese citizens of all
  ages (2.1). Nationally 386,091 / 404,113 = 0.955 for Maltese citizens and 111,174 / 115,449 =
  0.963 for non-Maltese, so the under-5 share is similar in both groups.
- Under-5s: 519,562 - 497,265 = **22,297 (4.29%)**, the whole gap.
- check_country.py's national figures equal Table 3.1 in every answer.

Corroboration from Vol. 1 Table 2.4 (non-Maltese by main citizenship): 13,838 Italian citizens
against 12,360 non-Maltese Italian speakers aged 5+; 3,311 Libyans and 2,861 Syrians against
7,461 non-Maltese Arabic speakers (Arabic is also other nationalities' language). 10,614 British
citizens against 20,312 non-Maltese English speakers: the rest are presumably Irish and others
who named English. German (10,172 non-Maltese) has no citizenship column to compare against
(Germans and Austrians are inside "Other EU", 22,443).

## 3. Calls

- **Two grains in one country.** Maltese citizens at locality, non-Maltese at district. The
  district counts are the census's; only their position inside the district is borrowed, which
  the brief (§4.4) allows without an ask. Each district's non-Maltese answers are split across
  its localities in proportion to the locality's non-Maltese residents of all ages (Vol. 1
  Table 2.1; the 5+ split by locality is not published). Every language gets the same split:
  no table says which foreign residents live where by citizenship below district, so Italian
  and Arabic speakers are spread alike. Inside a locality, dots follow the Kontur population of
  religiondots' cut hexes, as for Maltese citizens. `note_public` says the non-Maltese dots
  show where foreign residents live, not which language each locality's foreign residents
  speak.
- **"Other" (57,818, 11.6%) on `other`.** 55,541 of it is non-Maltese, half of all non-Maltese
  residents. The citizenship table suggests much of it is Serbian, Bulgarian, Indian, Filipino,
  Nepalese and Albanian citizens' languages, but nothing names the languages, so it is the
  bare `other`. Malta has no indigenous language besides Maltese, so there is no indigenous
  remainder to keep apart.
- **Maltese node and colour.** `afroasiatic.maltese` already existed (fi.txt, uk.txt; Glottolog
  malt1254, Afroasiatic). Generated, it was steel blue (#71b6d1), between English's pale blue
  and German's blue, the languages most mixed with it on the ground. Hand-picked in
  `taxonomy/tree.d/mt.txt` to `0.60 0.13 150` (#3b9555), a darker green in Afroasiatic's part of
  the wheel, apart from English, German and Arabic's pale mint. Its nearest colour among nodes
  drawn in Australia, Canada and the UK, where Maltese also appears, is Malayalam's (OKLab
  distance 0.039).
- **Geography.** religiondots' `mt_hexes.gpkg` (Kontur hexes cut to the 68 GISCO LAU localities,
  sources/mt.md §5 there). Its unit ids are the census's locality names with ASCII hyphens, the
  same names Volume 3 prints (U+2010 and NBSPs folded; "San Pawl il-Baħar" prints with a small
  i, matched case-blind). countries/mt.py asserts the 68 names equal religiondots'
  `mt_lookup.csv` both ways.

## 4. Figures used in note_public

Maltese 351,356 / 386,091 = 91.0% of citizens; English 29,930 = 7.8%; Is-Swieqi 2,818 / 7,480 =
37.7%; Tas-Sliema 2,519 / 9,727 = 25.9%; San Ġiljan 1,458 / 5,687 = 25.6%; non-Maltese 111,174 /
497,265 = 22.4%; their "Other" 55,541 / 111,174 = 50.0%.
