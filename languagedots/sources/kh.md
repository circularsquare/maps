# Cambodia: the record

Drawn 2026-10-05 (agent d9e44929-kh). 15,552,211 people, the whole census population, as ONE
national unit with 30 languages; placed inside the country by province-level census figures.
15,539 dots and 11 single-dot languages.

## Source

National Institute of Statistics (NIS), Ministry of Planning, **General Population Census of
Cambodia 2019**, reference date 3 March 2019. Mother tongue was asked of everyone (Form B,
column 9, "Native language", with codes 01-09 for Khmer, Vietnamese, Chinese, Lao, Thai, French,
English, Korean, Japanese, 10-28 for 19 minority languages and 29 "native language other").
There is no ethnicity question; NIS's "ethnic minority" is this answer.

| table | where | level | what |
|---|---|---|---|
| **2.7.1** | National Report on Final Census Results, p.25 (PDF p.53) | country | 7 rows: Khmer, Vietnam, Chinese, Lao, Thai, Other, Minority Languages |
| **2.3** | Series Thematic Report on Ethnic Minorities in Cambodia (Sept 2022), p.7 (PDF p.31) | country | 23 named minority languages + Other, 2008 and 2019 |
| **2.2** | same report, p.6 (PDF p.30) | **province** | all minority-language speakers together, 2008 and 2019 |
| 2.1.1 | final report p.15 | province | population (read by religiondots, used read-only) |
| 2.5.1 | final report p.24 | province | religion, as percentages (religiondots' counts, read-only) |

    https://nis.gov.kh/wp-content/uploads/2025/09/Final-General-Population-Census-2019-English.pdf
    https://www.nis.gov.kh/nis/Census2019/Ethnic%20Minorities.pdf

`python sources/kh_gpcc.py --fetch`. Both open, no wall; the final report is byte-identical to
religiondots' copy (26,586,966 bytes). The NIS site moved to WordPress in 2025; the old
`/nis/Census2019/` tree still serves files but its index is 403.

## Why national: what was searched for a province table

No table crosses mother tongue with any geography. Checked:

* **2019 final report** (English, 304 pp.): Table 2.7.1 only. **The Khmer edition** (380 pp.,
  `Final-General-Population-Census-2019-Khmer.pdf`) has more pages but the same national table
  (its p.61).
* **2019 Ethnic Minorities report**: Table 2.2 gives the minority total per province, Table 2.3
  each language nationally, Table 2.4 religion by region. No language by province.
* **CamStat** (`camstat.nis.gov.kh`, a .Stat Suite; SDMX at
  `nsiws-stable-camstat-live.officialstatistics.org/rest`): 96 dataflows, none with a language
  or mother-tongue code.
* **2008 census final report** (`GPC2008_Report_ENG.pdf`, 308 scanned pages, no text layer):
  Section 3 "Select Province Tables" numbers PT 01-27 and **skips PT 08**, between marital
  status (PT 07) and religion (PT 09), where a mother-tongue table would sit. Mother tongue is
  national only (its Tables 2.12, 2.13; stat.go.jp also hosts them as `tbl2-12.pdf`,
  `tbl2-13.pdf`). The 2008 provincial reports had mother-tongue Tables 4-1 and 4-2 per province
  (stat.go.jp's `pc08kc_10.pdf`, the Kampong Cham brief, cites them), but only Kampong Cham's
  and Tbong Khmum's re-issues are online, without those tables.
* **1998 census report**: no mother-tongue table at all.
* **CIPS 2013** (`ci_fn02.pdf`): national Table 3.12 only, and a sample survey.
* **Priority tables** (2008, `rp5_ant2.pdf`): A3/A3A "Population by Mother Tongue" existed for
  every level down to commune, on the CD-ROM / REDATAM product; not online. No live REDATAM
  server for Cambodia was found (the old celade.cepal.org path 404s).
* **Microdata**: `nada.nis.gov.kh` is alive and lists the 2008 census but downloads need an
  account; IPUMS has 1998-2019 with district geography (account dead). Gated, skipped.
* The Wayback Machine was down during this session ("Temporarily Offline"), so the old NIS
  `/nis/census2008/` tree was not walked.

## The national counts (all `measured` except one row)

Table 2.7.1's five non-minority rows, Table 2.3's 24 rows, and one derived row:

* **The two tables split the same 466,251 people differently.** 2.7.1's Minority Languages
  (448,282) + Other (17,969) = 466,251. Table 2.3's total is 455,610, including its own Other
  (7,413, the form's code 29). Its 23 named languages sum to 448,197, 85 under 2.7.1's Minority
  Languages. So 2.7.1's Other holds 2.3's unnamed minority remainder (less 85) plus the foreign
  languages. Drawn: Table 2.3 as printed, and **Other foreign language = 466,251 - 455,610 =
  10,641**, `derived`. The drawn total is 15,552,211 exactly.
* **Table 2.7.1's sex columns do not add up**: Male + Female exceeds Total on every row (Khmer
  by 2,812, Minority Languages by 85, the rest by 1-18), and each sex column sums to more than its
  own total. The Total column sums to the census population exactly and is the one used.
* Table 2.3: Male + Female = Total on all 25 rows in both years; rows sum to 389,424 (2008) and
  455,610 (2019).
* Table 2.2: Male + Female = Total on all 32 rows; 25 provinces and urban + rural sum to the
  total in both years; the four regions miss the 2019 total by 5 (NIS's own).
* **Cross-table: Table 2.2's total equals Table 2.3's in both years** (389,424 and 455,610).
* Provinces join to religiondots' KH-01..KH-25 **by print position and by name** (romanisation
  fold), which agree on all 25; every province's minority total is under its population.

## Categories (taxonomy/kh2019.py, taxonomy/tree.d/kh.txt)

Every printed label is a node. Glottolog checks are in the mapping's docstring. Calls:

* **Chamic**: Cham, Jarai (Charai), Rade (Rodae). New group, Austronesian.
* **Bahnaric**: Tampuan, Brao (Prov), Kreung, Kavet, Lun (Lorn), Khleung, Bunong (Punorng),
  Stieng, Kraol, Mel, Khaonh. Kreung, Kavet and Lun are Brao dialects in Glottolog; they sit
  beside Brao, not under it, so Brao stays a leaf (a group would wash its dots out). Khleung is
  not in Glottolog; filed with the Brao groups on Cambodian sources' account, less certain.
  Mel and Khaonh are one Glottolog language (Mel-Khaonh); NIS prints two answers, so two nodes.
* **Katuic**: Kuy. **Pearic**: Pear (Por), Suoy, Sa'och. **Mon** (Morn) on the existing node.
* **Thmon, Ro-ong, Ka-chrook, Kanh-Chok** (1,164, 573, 266, 16): no Glottolog match; placed
  directly under Austroasiatic, which is all that is claimed.
* **Other minority language** (7,413) on `seasia_other`, Thailand's indigenous remainder root;
  **Other foreign language** (10,641) on `other`. Kept apart (AGENT_BRIEF §3).
* Chinese on `sinitic` (as every country); Vietnam is Vietnamese.

Colours hand-picked in the fragment for the neighbours on the ground: Cham teal against Khmer's
pale pink along the Mekong; in Ratanak Kiri, Tampuan deep violet, Kreung pale lavender, Brao
blue-violet, Jarai sky blue; Bunong rose in Mondul Kiri; Kuy red in Preah Vihear; Pearic a warm
orange group. Not yet looked at on the map.

## Placement (countries/kh.py): one unit, placed by province

Counts are one unit, `KH`. The placement layer is religiondots' Kontur 400 m hexes for the 25
provinces (77,453 hexes), read-only; `_place_unit` keeps the province as `prov` and returns `KH`.
Inside each province every language follows Kontur's population. Between provinces, per
language (AGENT_BRIEF §4.4: moves people only inside the unit they were counted in):

| language | province weight |
|---|---|
| Khmer | population (Table 2.1.1) less minority speakers (Table 2.2) |
| Vietnamese, Chinese, Lao, Thai, other foreign | population: nothing more specific is published |
| Cham | the province's Muslims (Table 2.5.1), capped at its minority speakers |
| the other 23 minority languages | Table 2.2's minority speakers less the Cham share, raked (IPF) to each language's national total from a seed |

The seed for the 16 minority languages with a Glottolog point is nearness to it: the province's
population-weighted mean of exp(-d / 25 km) over its hexes, mixed 95:5 with an even share
(`KERNEL_KM`, `LAMBDA`, chosen by looking at the result, not fitted). Seven get an even seed:
Khleung, Thmon, Ro-ong, Ka-chrook, Kanh-Chok, Mon and the minority Other. Scaling by the census
populations also removes Kontur's between-province bias (Pailin is 4.6x over in Kontur,
religiondots `sources/kh.md`).

What it gives (people, rounded): Bunong 26,900 in Mondul Kiri, 4,700 Kratie, 3,300 Ratanak Kiri;
Tampuan 34,700 and Jarai 25,500 in Ratanak Kiri; Kreung 18,700 Ratanak Kiri and 1,300 Stung
Treng; Kavet 5,100 Ratanak Kiri, 2,100 Stung Treng; Kuy 10,600 Preah Vihear, 1,800 Kampong
Thom, 1,500 Siem Reap, 600 Stung Treng and Oddar Meanchey; Stieng and Kraol about 4,000 each in
Kratie; Suoy 590 Kampong Speu. Cham follows Muslims: Tbong Khmum 90,000, Kampong Chhnang 30,100,
Kratie 24,700, Phnom Penh 22,900, Kampong Cham 20,700. Print the whole table with
`_KhWeighter(place).table()`.

Known weaknesses: in the ten provinces whose Muslims outnumber their minority speakers
(Battambang, Kampong Chhnang, Kampot, Koh Kong, Phnom Penh, Pursat, Preah Sihanouk, Kep, Pailin,
Tbong Khmum) every minority speaker is drawn as Cham, so Pursat's Pear and Kampot's
Sa'och are drawn elsewhere (Sa'och lands in Kampong Speu). Vietnamese and Chinese are spread with
the population, though both are mostly urban and riverine; no province figure for either, or
for the foreign-born, is published for 2019.

**Rings ignore the weighter.** scatter.py puts a sub-dot language's single mark on a random hex
of its unit, and here the unit is the whole country: Khleung, Rade, Khaonh, Kanh-Chok, Mon,
Sa'och, Ka-chrook, Ro-ong, Mel, Pear and Suoy (11 rings) land anywhere in Cambodia. Asked of
the supervisor (scatter.py is not an agent's to edit).

## Checks and figures from the scatter

15,552,211 people; 13,211 (0.08%) under one dot per language and not drawn as dots; 15,539
dots over 12,029 hexes; 11 rings. One Kontur cap block, already registered by religiondots as a
real core. `tools/check_country.py kh`: ok.

## Text on the map

`how` "census, 2019, mother tongue". `grain` says national, placed by province. `gap`: the
census leaves out Cambodians working abroad (its own footnote; several hundred thousand in
Thailand alone). `note_public` says the placement is an estimate and how it is made.

## Not done

* A province or district mother-tongue table. The 2008 provincial reports (Tables 4-1, 4-2)
  would give province x language for 2008; worth asking NIS or a library for, or a Wayback walk
  of `nis.gov.kh/nis/census2008/` when the archive is back up.
* Microdata (NADA, IPUMS), gated.
