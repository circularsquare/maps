# Thailand: 2010 census, language usually spoken in the household, by changwat

Session d9e44929-th2, 2026-10-05 (the first session, d9e44929-th, was stopped before it wrote
anything to the project; it had only probed the Wayback Machine for 2000-census reports and found
the whole-kingdom 2010 report, all in its scratch folder). Drawn: 65,835,645 people on 76 changwat,
39 nodes since 2026-10-06, when session 5d7dac7e-th split the census's Thai into Central Thai,
Isan, Northern Thai and Southern Thai by World Values Survey shares ("The Thai varieties" below).

```
python sources/th_census.py
python sources/th_wvs.py [--fetch]
python taxonomy/build.py
python tools/check_country.py th
python scatter.py --country th
```

## Source

National Statistical Office (NSO), *The 2010 Population and Housing Census*, complete reports
(รายงานผลสมบูรณ์), published 2012: one PDF per changwat plus the whole kingdom, listed at
https://www.nso.go.th/nsoweb/main/summano/aE (year 2553, type "สมบูรณ์"; the form is a plain GET,
`?year=957&type=3292`). Each provincial report's Table 6 (Table 7 in the kingdom's) is
"Population by usual languages spoken at home, sex and area". The 78 file names are in
`sources/th_census.py`; only the table pages (and one front page) are kept, in `data/raw/th/`,
about 130 KB each against 6 MB for a whole report.

The question (questionnaire item G, in the kingdom report's appendix) is asked once per
**household**: "Language usually spoken in this household: Thai only / Thai and other / Other
language only; specify language". Every member counts under the household's answer. The table
gives the three-way split, then 40 "other language" rows that add up to Thai-and-other plus
other-only (short by 81,519 in the kingdom: households whose other language is not in any row).

The figures are **estimates**: the detailed questions went to a sample of households, and the
appendix gives the estimators. They are rounded to the person, so identities hold to a few people.

Kingdom: 65,981,659 people; Thai only 59,866,190; Thai and other 4,214,001; other only 1,901,468.
Largest other languages: local Malay/Yawi 1,467,369; "local language" 958,251; Burmese 827,713;
Tai Khün/Thai Loei/Lao Loei 787,696; Karen 441,114; English 323,778; other indigenous and
hill-tribe 318,012; Khmer 180,533; Lao Khrang 177,453; Hmong 149,090; Chinese 111,866.

Leads that were not this: the coverage sweep had "province (provincial PDF reports, assumed)",
right. The 2000 census also asked language; its provincial reports survive only in the Wayback
Machine (`web.nso.go.th/pop2000/finalrep/`), which was rate-limiting the first session. 2020 was
register-based and asked no language. The 2010 tables exist only at changwat; nothing finer
crosses language with amphoe.

Terms: NSO publications, free to download; reused with attribution, as religiondots does for the
same census's religion figures.

## The tables are untidy, and how the reader copes (`python sources/th_census.py`)

Typeset province by province, the 77 reports differ: some print only their non-zero rows
(Phuket, Kalasin, Maha Sarakham, Samut Songkhram), one drops a row (Uttaradit, Polish), some
print the subtotal of the language rows on the "Other languages" heading, some wrap a label and
put its figures on the first line, some on both, Phangnga sets some numbers without commas,
Surat Thani has two overlaid text layers, and the text order wanders. So the table is read by
**position on the page**: words are rotated into the page's visual frame, columns are found by
clustering the figures' right edges, rows by their height, and each row is matched to the
expected label by its English or Thai text. Every quirk met is logged in the run's output.

## Checks

- Each row's total equals male + female or municipal + non-municipal (to 2). Where a report
  misprints another cell (Chiang Rai's municipal Thaikueng, Saraburi's male Arab and English,
  Sing Buri, Kalasin), the total agrees with the other identity and is kept.
- Each table: Thai only + Thai and other + other only = total (to 3). The language rows never
  exceed Thai and other + other only.
- The file-to-province pairing: each table's total against the census total religiondots read
  for the same unit from NSO's provincial indicator sheets, a different document. 74 of 76 agree
  to 12 people. **Rayong's and Ratchaburi's language tables are short of their own populations
  by 50,163 and 44,987**: the reports print it so. Those 95,150 people are in `gap`.
- **Bueng Kan** (created March 2011) has its own report, but **Nong Khai's report is the whole
  pre-2011 changwat** (821,526, which is also religiondots' figure for unit 443), so Bueng Kan's
  362,754 lie inside it and are not added. Its Karen, Mon and Hmong rows exceed Nong Khai's by a
  few dozen to a hundred people; separate processing, left alone.
- **The 76 provinces against the kingdom table, row by row.** Total, Thai only and the two other
  splits differ by Rayong and Ratchaburi's shortfall (Total -95,155, of it Thai only -90,815).
  Every language row is within 2%: the largest gaps are Khmer -1,602 of 180,533 and Burmese -915
  of 827,713; 31 of the 40 rows are within 40 people.

## Repairs, each found by the kingdom check and confirmed by it

- **Phitsanulok and Chumphon print their first block of language rows one line high**, the first
  figure on the "Other languages" heading (263 and 79, where the subtotal would be tens of
  thousands). Read as printed, Chumphon had 32,397 Chinese and 11 Burmese, and the kingdom check
  was off by +32,963 Chinese and -34,186 Burmese; Phitsanulok had 5,872 Pattani Malay and 7,965
  Lao Khrang, and the kingdom was off by +5,847 Malay, -5,696 hill-tribe, +6,610 Lao Khrang and
  -7,836 Hmong. Moved down a row (Karen to Malaysia take the figures of the line above), all of
  those come within a few hundred, and Korean and Japanese come right too. Shifting the other
  reports that print a heading figure (Chiang Mai, Nakhon Si Thammarat, Yala) makes the check
  worse, so they are left as printed.
- **Narathiwat's English row repeats its subtotal**, 569,967 (the same nine figures as the
  heading). Replaced by the kingdom's English less the other 75 provinces, column by column:
  1,076.
- **Uttaradit's Vietnamese total is printed 7,204 beside eight zeros**; its language rows then
  exceed its other-language households by exactly 7,204. Taken as 0.

## How it is drawn (`countries/th.py`, spec §3.6)

A household answering "Thai and another language" names two languages, so its people are shared
half and half (§3.6's exact split for two answers). Per changwat:

- Thai = Thai only (measured) + half of Thai and other (derived);
- each other language = its row × (other only + half of Thai and other) / (Thai and other +
  other only), derived. The scale is exact in total but shared over the rows in proportion,
  because the table does not cross the three-way split with the languages.

People whose other language has no row take their share of the scale with them: 50,859 not
drawn. Drawn 65,835,645 (59,775,375 measured, 6,060,270 derived). Placement: religiondots' 76
changwat and its Kontur hexes (`th_hexes.gpkg`, `unit` = its code), read-only; the unit codes
are religiondots', paired by name (its `th_lookup.csv` Thai names, with NSO's listing typos
ศรีษะเกษ, ประจวบครีขันธ์ and the short อยุธยา mapped by hand) and then by population.

## The Thai varieties (session 5d7dac7e-th, 2026-10-06)

Anita, 2026-10-06: "i think we should maybe also try to do isan somehow", her go-ahead for a
proxy that changes the counts. The census's Thai in each changwat is now split into Central
Thai, Isan, Northern Thai and Southern Thai; every other row, and each changwat's total, is
unchanged.

**Source: the World Values Survey**, whose Thai card names all four varieties
(`sources/th_wvs.py`, raw pages `data/raw/th/wvs_*.html`, output `data/normalized/th_wvs.csv`):

| wave | year | n | question | geography | Esan | Northern | Southern | Central |
|---|---|---:|---|---|---:|---:|---:|---:|
| 7 | 2018 | 1,500 | Q272 language at home | 49 changwat (ISO 3166-2) | 482 | 134 | 132 | 632 |
| 6 | 2013 | 1,200 | V247 | Bangkok + 4 NSO regions | 403 | 81 | 106 | 560 |
| 5 | 2007 | 1,534 | V222 | 4 NSO regions (+8 "East") | 494 | 106 | 175 | 641 |

(IHSN catalogue frequencies, unweighted: catalog.ihsn.org/catalog/12307, 9027, 8955.) Fetched
from the WVS online analysis tool by `ir_wvs.py`'s route; Thailand's ids are in the script.
**The tool weights Thailand's tables**, so counts are percent x N as floats (weighted totals
agree with IHSN's unweighted within 2%, Malay excepted: 76 against 65).

**The split** (`th_wvs.py` docstring has the formula): per changwat, the share of each variety
among respondents who named a Thai variety, wave 7's own respondents plus 30 interviews' worth
of its zone's pooled share. Zones are the official six regions (North 9, Northeast 20, West 5,
East 7, South 14, Central 22 with Bangkok and the lower north). Bangkok adds wave 6's Bangkok
column (214 Thai-variety answers in all). 28 changwat had no wave 7 interviews and take their
zone's share. Zone priors: North 90.5% Northern / 6.1% Central / 3.4% Isan; Northeast 98.3% Isan;
West 100% Central; East 89.0% Central / 11.0% Isan; South 87.6% Southern / 11.3% Central;
Central 97.3% Central / 1.8% Isan. Rows `modelled`.

Two single-cluster answers handled by hand (wave 7 samples a sampling point or two per
changwat):
- **Suphan Buri's "Northern Thai; Lanna" (74% of 52) is left out of the split.** Suphan Buri has
  no Lanna community; its non-Central Tai speakers are Lao Khrang, Lao Song (Tai Dam) and Lao
  Wiang, and the card had no Lao box. The census already counts Lao Khrang there in its own row.
- **Prachuap Khiri Khan's 12 interviews (71% Esan, 0% Central)** are kept for Prachuap (22% Isan
  after shrinking) but left out of the West's prior, where they would have put 16% Isan on
  unsampled Tak and Kanchanaburi.

**Drawn** (65,835,645, unchanged): Central Thai 31.0M (47.1%), Isan 18.9M (28.8%), Southern Thai
7.0M (10.6%), Northern Thai 5.0M (7.6%).

**Checks.**
- Against the Royal Thai Government's 2011 report to CERD (CERD/C/THA/1-3, from Mahidol
  University's Ethnolinguistic Maps of Thailand, as quoted on Wikipedia's "Languages of
  Thailand"): Central 20.0M, Isan 15.2M, Northern 6.0M, Southern 4.5M; Wikipedia's L1 shares
  40/33/11/9%. Ours sit between the two for Isan, under both for Northern Thai, over both for
  Southern Thai and Central Thai. Southern is high because the WVS South answered Southern Thai
  at 88% of its Thai speakers and the South's Thai population is about 8M.
- Waves 5 and 6 by NSO region against wave 7 pooled the same way (share of Thai-variety
  answers, central/isan/northern/southern):

  | region | w5 2007 | w6 2013 | w7 2018 |
  |---|---|---|---|
  | North (17 changwat) | .34/.08/.58/.00 | .54/.10/.36/.00 | .57/.02/.42/.00 |
  | Central (w5 incl. Bangkok) | .94/.05/.01/.00 | .95/.03/.02/.00 | .83/.07/.09/.00 (Suphan Buri's "Northern" included) |
  | Northeast | .03/.96/.00/.01 | .08/.92/.00/.00 | .02/.98/.00/.00 |
  | South | .08/.01/.00/.91 | .26/.01/.00/.72 | .11/.01/.01/.88 |
  | Bangkok | | .96/.02/.00/.02 | 1.00/.00/.00/.00 |

  The Northeast agrees across all three. The South's wave 6 is the outlier (14% "No answer"
  there). The North varies with how many lower-north points each wave drew; the split here
  handles that by zone.

**Naming and colours.** Isan is `kradai.isan`, "Isan (Northeastern Thai)": Glottolog's
nort2741 "Northeastern Thai" is a language beside Lao (laoo1244) in Lao-Phutai, but its speakers
are Thai citizens who call it Isan (phasa Isan), so the node is a Kra-Dai leaf of its own, not a
child of Lao, and the label leads with the name readers know. Northern Thai reuses Laos's
`kradai.northern_thai | Northern Thai (Tai Yuan, Nhuan)` (same label, required); "Kham Mueang"
would be a better gloss for Thailand but the label is shared. Southern Thai is
`kradai.southern_thai`, "Southern Thai (Pak Tai)". Colours (tree.d/th.txt): Isan orange
#f99532, apart from Thai's light green #a9cd63 and Lao's olive #9e9f1e across the Mekong;
Northern Thai green #54a859; Southern Thai gold #c2a200; the Khün/Loei box moved from amber to
cream #fece96 because amber sat on Isan's orange in Loei.

## Calls

- **The Thai split uses the WVS** (above): a small clustered survey, so a changwat's split is
  an estimate, said in `how` and `note_public`. 28 changwat take their region's shares.
- **Bangkok is drawn 98% Central Thai, 1% Isan.** Both waves that sampled Bangkok found almost
  no Isan spoken at home (0 of 111 in 2018, 2 of 106 in 2013), though Bangkok has a large
  population born in the Northeast. Not adjusted: no source gives Isan retention among
  Bangkok's migrants. Said in `note_public`.
- **Tak (West) is drawn all Central Thai** (unsampled; the West's sampled changwat answered
  only Central Thai). Parts of Tak speak Northern Thai. Uttaradit (North) takes the North's 90%
  Northern Thai.
- Thai Korat falls in Central Thai (Glottolog and Ethnologue put it there); Nakhon
  Ratchasima's wave 7 respondents answered 12% Central, drawn 8% after shrinking.
- **"Local language" (ภาษาถิ่น) is not guessed.** 552,323 in Surin, 180,247 in Buri Ram and
  131,006 in Si Sa Ket: the Northern Khmer and Kuy country (the census's "Khmer" row there is
  tiny; it is mostly Cambodian Khmer in Bangkok and the east). It could be read as Northern
  Khmer by place (§3.3), but the census never names it, Kuy speakers are in the same districts,
  and Nan's 29,915 are something else again. It and "other indigenous and hill-tribe languages"
  go on a new areal root, `seasia_other`, as Indonesia's unnamed regional languages do; the
  public note says where most of it is.
- **"Tai Khün, Thai Loei or Lao Loei" is one leaf**, labelled with the census's three names.
  364,423 are in Loei (the Loei speech), most of the rest in Lamphun, Chiang Mai, Chiang Rai and
  Mae Hong Son (Khün), but 60,459 are in Kalasin and 48,706 in Mukdahan, where neither is
  spoken. Splitting by province would name those.
- **Karen sits on the Karen group node** (washed, "Karen, language not named"), as Australia's
  and Canada's "Karen" do: the answer covers S'gaw, Pwo, Pa-O and Kayah.
- **"Malay/yawi" is Pattani Malay**, a new leaf beside Malay, Satun's Kedah-type Malay included.
- **"India/Hindi" is Hindi**; "Mexican" and "Cuban" are Spanish.

## Not done

- No district grain: none published.
- Regional Thai varieties, room for improvement: the WVS gives a few dozen interviews per
  changwat. Searched 2026-10-06 and not usable: MICS 2005-06 (mother tongue of household head
  coded only Thai / other), Alexander and McCargo 2014 (122 students), NSO surveys (none found
  asking the variety). The WVS microdata files would allow an exact unweighted count, but sit
  behind a form; the online tool's weighted tables were used. Mahidol's Ethnolinguistic Maps of
  Thailand (Premsrirat 2004) may give provincial figures for the varieties; not traced to a
  table.
- The national-only remainder: none; every row is by changwat.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Isan is in the Lao group (Glottolog's Northeastern Thai, a sister of Lao in Lao-Thai), so Laos's Lao continues across the Mekong. Thailand itself counts Isan as Thai; this is the call most open to reversal. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
