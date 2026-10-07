# Tajikistan: 2010 census native language, a nationality model at five regions

Drawn 2026-10-05 by session `d9e44929-tj`. Code: `sources/tj_census.py`, `taxonomy/tj2010.py`,
`taxonomy/tree.d/tj.txt`, `countries/tj.py`. Raw: `data/raw/tj/census2010_vol3.pdf` (copied from
religiondots' raw folder; the script downloads it from the Wayback Machine if neither copy exists).
Ask 010 (Pamiri languages) is open; nothing waits on it.

## Source

| | |
|---|---|
| office | Agency on Statistics under the President of the Republic of Tajikistan, `www.stat.tj` |
| table | Population and Housing Census 2010, Volume III, *Natsionalny sostav, vladenie yazykami i grazhdanstvo naseleniya Respubliki Tadzhikistan* (2012), pp.12-56: population by sex, nationality and native language; pp.108-115: seven nationalities by region |
| URL | `web.archive.org/web/20131014054442id_/http://www.stat.tj/ru/img/526b8592e834fcaaccec26a22965ea2b_1355501132.pdf` (537 pages, text layer, 13,682,802 bytes); linked from ru.wikipedia's 2010 census article |
| question | native language (*rodnoy yazyk*), self-declared; children's by their parents (methodology, p.5) |
| licence | the Agency's site footer says CC BY 4.0 (coverage sweep) |
| 2020 census | asked the same; Volume III is still an empty heading on `stat.tj/ru/perepis-naseleniya-i-zhilishhnogo-fonda-2020/` (checked 2026-10-05). Only nationality shares are out (Tajik 86.1, Uzbek 11.3, Kyrgyz 0.4, Russian 0.3, other 1.9; religiondots `sources/tj.md` §7). |

**What the table is.** For each of 92 nationality rows (90 nationalities, "other nationalities"
and "nationality not stated"), how many name as native language (a) their own nationality's
language, or another nationality's: (b) Tajik, (c) Russian, (d) other languages. Printed for the
whole country, then by sex, then urban and rural. **No region or district anywhere in the volume
has native language.** Nationality by region is printed only for Tajiks, Uzbeks, Russians, Kyrgyz,
Turkmens, Tatars and Kazakhs (pp.108-115), five regions, nothing below.

## Checks (all in `sources/tj_census.py`, all pass)

- Every row's four columns sum to its total, in the national, urban and rural blocks (92 rows each).
- The rows sum to the printed total row (7,564,502 = 7,498,938 own + 27,809 Tajik + 7,851 Russian +
  29,904 other) in the urban and rural blocks exactly. The national block misses by a 2-person
  Tajik/Russian swap: its Chuvan row prints (4: 0 own, 2 Tajik, 0 Russian, 2 other) where urban plus
  rural give (4: 0, 0, 2, 2). **Urban plus rural is used**; it equals the national block in every
  other cell and its columns hit the printed total row exactly.
- Each row's total equals the nationality table, pp.7-11 (religiondots' transcription).
- The seven regional groups sum to their national rows; the five regions' 2010 populations
  (religiondots' `tj_lookup.csv`, read-only, which Volume I's 2010 column confirms) sum to
  7,564,502.
- The model reproduces every region's 2010 population and every national cell to under 0.5.
- note_public's figures are asserted against the build.

## The model (as Azerbaijan, ask 006, ruled yes)

Placement only; the national counts are the census's. Each region's `other` column is its 2010
population less the seven printed groups.

1. The seven groups: their own 2010 count per region.
2. Nationalities of non-Muslim heritage (Ukrainians, Germans, Koreans, Armenians, Chinese, the
   "other" and "not stated" rows ...; religiondots' list): on the Russians' regional distribution,
   as religiondots places them.
3. Everyone else (the Uzbek tribal groups, Arabs, Afghans, Lyuli, Turks, Uyghurs, Azerbaijanis,
   Bashkirs ...; 141,345 people): the rest of each region's `other` column, pro rata. That rest is
   Khatlon 121,214, Districts of Republican Subordination 16,130, Dushanbe 2,239, Sughd 1,702,
   Gorno-Badakhshan 60. Asserted non-negative.
4. Each nationality's people in a region split over native languages by its national split.

Every row `modelled`. What it cannot see: whether Uzbeks in Dushanbe name Tajik more often than
Uzbeks in Sughd (likely), and where inside the `other` column each small group lives (the Lakai
and Kongrat are known to be in Khatlon's Vakhsh valley, which the model agrees with only because
Khatlon holds 86% of the column).

**Vintage.** The dots are the 2010 population (7,564,502), not the 2020 census's 9,657,005:
scaling 2010 shares onto 2020 regional totals would be a proxy on the counts, and the 2020 shares
that are out (Uzbek 11.3%) are national only.

## Result

National: Tajik 6,384,880 (84.4%), Uzbek 903,211 (11.9%), Lakai 60,392, Kyrgyz 59,437, Russian
40,598, Kongrat 37,831, other languages 29,904, Turkmen 13,653, Durmen 7,565, Katagan 7,552, Arabic
4,089, Tatar 3,996, Yuz 3,798, Afghan 2,320. By region: Sughd Tajik 84.2%, Uzbek 14.4%; Khatlon
Tajik 82.0%, Uzbek 12.6%, Lakai 1.9%, Kongrat 1.2%; Districts Tajik 85.2%, Uzbek 11.4%, Kyrgyz
2.0%; Dushanbe Tajik 89.5%, Uzbek 6.5%, Russian 2.7%; Gorno-Badakhshan Tajik 94.2%, Kyrgyz 5.2%.
72 nodes, 7,551 dots at 1:1,000, 58 rings.

Of interest: 21,999 Uzbeks (2.4%) named Tajik; 13,581 Tajiks named "other languages"; 1,983 of the
2,334 Lyuli named Tajik; all 5,267 Barlos named "other languages".

## Calls someone might reverse

- **Uzbek tribal dialects as nodes** (Lakai, Kongrat, Durmen, Katagan, Yuz, Ming, Kesamir, Semiz;
  117,609 people). The census keeps the tribes apart as nationalities, and the Barlos (all 5,267
  under "other languages", none "own") show the dictionary had a separate language for the others
  and not for the Barlos. Reversing: map them to `turkic.uzbek` in `tj2010.py` and drop the
  fragment's nodes.
- **Afghans' own language on `other.afghan`**, not Pashto: Afghan nationals (Dari, Pashto) and the
  Parya of the Hisor valley (Indo-Aryan, Glottolog pary1242), who call themselves Afghan, share
  the census row.
- **Arabs' own language on `afroasiatic.arabic`**, not a new Tajiki Arabic node (taji1248): the
  census says only "the Arabs' language".
- **Lyuli's own language on Romani** (338), following Kyrgyzstan's Roma.
- **Non-Muslim-heritage groups on the Russians' distribution**, borrowed from religiondots.
- **2010 people, not 2020.**

## Not drawn, and why

- **Pamiri languages** (ask 010): the census counts Pamiris as Tajiks and their languages as Tajik.
  Kept as the census has it; `gap` and `note_public` say so.
- **"Other languages"** (29,904) stays on `other`: probably Uzbek for the Barlos and part of the
  Lakai, possibly Pamiri for some Tajiks, but unnamed.

## What reopens it

- The 2020 census's Volume III, even national only: it replaces the 2010 table. If it prints
  native language by region or district, the model goes.
- `nada.stat.tj` microdata, if it ever answers (religiondots `sources/tj.md` §6).
