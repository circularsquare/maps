# Brunei (bn): record

Drawn 2026-10-06 (session `5d7dac7e-bn`). The census asked home language (BPP 2021 item E27) but
DEPS never tabulated it (appendix below). Built instead under the two 2026-10-05 rulings: the
ethnicity route (AGENT_BRIEF §2, tier D, with a retention check) and published speaker
estimates placed by homeland (ask 019). 440,715 people (the whole enumerated population,
temporary residents included), 4 districts, 67 nodes. Brunei Malay rows `derived`, every other
row `modelled`. 412 dots at 1:1000, 46 rings.

```
python sources/bn_build.py
python sources/origin_mix.py --fragment bn
python taxonomy/build.py
python tools/check_country.py bn
python scatter.py --country bn
```

Files: `sources/bn_build.py`, `taxonomy/bn2021.py` (identity), `taxonomy/tree.d/bn.txt`,
`countries/bn.py`, `data/normalized/bn.csv`. Base table read from religiondots' pinned workbook
`religiondots/data/raw/bn/bpp2021_excel_table_A-C.xls` (read-only).

## 1. The base: Table A3, race by district

BPP 2021 report, Annex A, Table A3 "Population by Race, District and Sex", persons. Three races
only; nothing finer (no puak, no mukim) is published.

| district | Malays | Chinese | Others | total | temporary residents (A1) |
|---|---:|---:|---:|---:|---:|
| Brunei Muara | 222,772 | 28,522 | 67,236 | 318,530 | 63,560 |
| Belait | 33,245 | 11,068 | 21,218 | 65,531 | 12,386 |
| Tutong | 35,283 | 2,318 | 9,609 | 47,210 | 4,452 |
| Temburong | 5,716 | 224 | 3,504 | 9,444 | 814 |
| total | 297,016 | 42,132 | 101,567 | 440,715 | 81,212 |

Checks (`bn_build.py::check_workbook`): A3 and A1 in the workbook equal the transcription; races
sum to each district and to 440,715 (religiondots' `sources/bn.py` already reconciles A1 to A4,
A12, C1 and UNSD). Every district's output rows sum to its A3 total (asserted).

The census's "Malay" holds the seven puak jati (Brunei Malay, Kedayan, Tutong, Dusun, Bisaya,
Belait, Murut). Others less temporary residents ("settled Others", 20,355) is about what the
Iban figure needs (15,800), which suggests temporary residents are nearly all in Others, so
foreign Indonesians and Malaysians are probably not counted as Malay. Not stated by DEPS.

## 2. The split

| race | language | node | figure | where | source |
|---|---|---|---:|---|---|
| Malays | Tutong | `north_borneo.tutong` | 17,000 | by the Tutong people's district shares: Tutong 64.7, Brunei-Muara 24.5, Belait 10.5, Temburong 0.2 | Ethnologue 18th ed. (2006 figure) via Wikipedia "Tutong language"; shares from Wikipedia "Tutong people" (16,958 people) |
| Malays | Kedayan | `malayic.kedayan` | 30,000 | Brunei-Muara, Tutong, Belait by their Malays (22,943 / 3,634 / 3,424) | Dewan Bahasa dan Pustaka Brunei 2006, via Wikipedia "Kedayan" |
| Malays | Brunei Dusun (Bisaya inside) | `north_borneo.brunei_dusun` | 10,000 | 2/3 Tutong, 1/3 Belait | low end of Wikipedia "Dusun people (Brunei)" 10,000-20,000; places from UBD *Pronunciation of Dusun* (majority Tutong, some Belait) |
| Malays | Belait | `north_borneo.belait` | 200 | 150 Belait, 50 Tutong | Wikipedia "Belait language": "fewer than 200" now; Kuala Balai, Labi, Kiudang |
| Malays | Murut (Lun Bawang) | `north_borneo.lundayeh` | 600 | Temburong | Joshua Project people group 15053, Brunei |
| Malays | Brunei Malay | `malayic.brunei_malay` | the rest, 239,215 | all four | Ethnologue's 266,000 for Brunei Malay (Wikipedia "Languages of Brunei") is 11% higher; it likely counts some Kedayan |
| Chinese | English 16%, Mandarin 30% | | 6,741 + 12,639 | one national mix | Asia Harvest, "Brunei Chinese" (people-groups.asiaharvest.org/Brunei/Brunei-Chinese.pdf) |
| Chinese | the other 54%: Min Nan, Cantonese, Min Dong, Hakka | | 10,666 / 4,977 / 4,710 / 2,399 | one national mix | Joshua Project Brunei (Ethnologue-based): 12,000 / 5,600 / 5,300 / 2,700 |
| Others | Iban | `malayic.iban` | 15,800 | Belait 8,713, Tutong 5,087, Temburong 2,000 | Omniglot, "Iban language and alphabet": "about 15,800 speakers ... in Belait, Tutong and Temburong"; split by settled Others, Temburong held at 2,000 (UBD IAS working paper 65, "less than 2,000"; the plain split gave 2,548) |
| Others | foreign residents | origin_mix home mixes | the rest, 85,767 | each district's Others less its Iban | shares: Bangladesh 26,000, Indonesia 25,000, Philippines 23,000 (2024 private-sector work-pass figures as reported online; the Minister of Home Affairs, Nov 2022 Labour Department census of 60,293 work passes, named the same three as largest), India 10,000 (Government of India via Wikipedia "Indians in Brunei"), Nepal 5,822 and UK 2,238 (UN DESA International Migrant Stock 2020) |

National result: Brunei Malay 54.3%, Kedayan 6.8%, Bengali 5.7%, Tutong 3.9%, Iban 3.6%, Mandarin
2.9%, Min Nan 2.4%, Dusun 2.3%, English 2.0%, Javanese 1.9%, Tagalog 1.7%, Cebuano 1.4%.
Tutong District: Brunei Malay 30%, Tutong 23%, Dusun 14%, Iban 11%, Kedayan 8%.

## 3. Retention check (tier D)

- **Tutong, Kedayan, Belait, Iban**: the figures used are speaker figures, not ethnic counts, so
  retention is already in them as far as their sources measured it. Tutong's 17,000 speakers
  (2006) equal the 16,958 Tutong people on Wikipedia, so that source saw no shift; vitality 2.5
  of 6 (Noor Azam and Siti Ajeerah 2016, via Deterding 2020) says otherwise. No share exists.
- **Dusun**: only ethnic figures exist (Minority Rights Group 2018: 6.3% of the population, about
  27,800 now; Wikipedia 10,000-20,000; Ethnologue's 42,000 for Brunei Bisaya). Vitality 2 of 6,
  and the young "often have poor competence" (Fatimah and Najib 2015, via UBD). The low end,
  10,000, is drawn as the speaker figure; no published retention share.
- **Murut**: 600 is Joshua Project's people figure. Coluzzi (2010, Oceanic Linguistics 49(1))
  surveyed the Murut and Iban of Temburong and found Murut relatively healthy; not opened.
- **Chinese**: Asia Harvest's English 16% and Mandarin 30% is the retention step: those are drawn
  off the dialects. "Use of these dialects is declining ... younger generations are increasingly
  raised to speak English" (Wikipedia "Ethnic Chinese in Brunei").
- **Foreign residents**: drawn on their home mix, no retention; nearly all are temporary.

## 4. Calls someone might reverse

- The whole route: race plus estimates (rulings of 2026-10-05). No figure below the three races
  is a count.
- Dusun at 10,000, the low end of an ethnic range, standing in for speakers.
- Kedayan spread over three districts by their Malays: no district figure exists, and Temburong
  gets none. Inside Brunei-Muara it is placed by population, so it lands in the town as much as in
  its home villages (Sengkurong, Jerudong, Pengkalan Batu); placement only.
- Chinese with one national mix; Joshua Project's 10,000 Min Bei left out (no other source names
  Min Bei speakers in Brunei; every one names Hokkien as the largest group). Hainanese and
  Teochew are named in sources but have no figure, so they are inside Min Nan and Cantonese by
  default.
- Foreign nationality shares: the 2024 work-pass figures are the weakest source here (a business
  blog repeating them); UN DESA 2020's Brunei table (Malaysia 52,628, Indonesia 6,717, Bangladesh
  1,343) is an extrapolation from 1991 and contradicts the minister's ranking, so it is used only
  for Nepal and the UK. Thailand is left out (UN DESA's 15,674 is from the same extrapolation).
- India on its home mix (Hindi first), though Wikipedia says Tamils are the majority of Indians in
  Brunei; no share given, and India is 11% of the foreign pool.
- English is drawn only for Chinese (16%) and the British. Malays who speak English at home are
  not split out; no figure exists.

## 5. Room for improvement

The census's own E27 table (home language: Malay, English, Chinese, Arabic, Others), by district,
would replace the Chinese mix, the English share and the foreign split, though it would put all
seven puak on Malay. A puak-by-district table from any census would fix the Malay split. A Labour Department table of work passes by
nationality and district would fix the foreign pool.

## 6. Geography

religiondots' layer, read-only: `religiondots/data/geo/bn/bn_hexes.gpkg`, Kontur 400 m hexes keyed
to the four geoBoundaries districts (religiondots' `sources/bn.md` §5). Units join by name
directly (`Brunei Muara`, `Belait`, `Tutong`, `Temburong`); `check_country.py` says ok. Placement
by Kontur population inside each district.

## Terms

DEPS tables: public files (via religiondots). Wikipedia text CC BY-SA; Ethnologue and DBP figures
cited as Wikipedia carries them. Joshua Project: public profiles (its shares are not comparable
across rows; only Brunei counts used, as estimates). Asia Harvest: public PDF. Omniglot: public
page. UN DESA migrant stock: public. Kontur CC BY 4.0.

## Appendix: the 2026-10-05 search for the home-language table (session `edd42a8c-bn`)

Parked at A then; ask 016 closed (no emails). Kept as the record of what was checked.

### The question exists, the table does not

| census | item | answers |
|---|---|---|
| BPP 2021 | **E27** "Bahasa pertuturan utama yang awda gunakan di rumah / Language mainly spoken at home", one answer | 1 Malay, 2 English, 3 Chinese, 4 Arabic, 5 Others (please specify) |
| BPP 2011 | **D18** "Bahasa pertuturan utama / Language mainly spoken" | same five-code layout (form p.10, `data/raw/bn/Questionnaire_BPP-2011.pdf`) |

E27 read off religiondots' cached form (`religiondots/data/raw/bn/bpp2021_questionnaire.pdf`, p.12).
Five codes only: Brunei Malay, Kedayan, Dusun, Iban, Murut, Tutong and Belait speakers can only
answer Malay or Others, so even a published table would draw Brunei mostly on one node.

### What was searched (nothing tabulates E27 or D18)

| publication | where | language content |
|---|---|---|
| BPP 2021 *Demographic, Household and Housing Characteristics* report, 94 pp | Wayback `deps.mofe.gov.bn/.../DOS/POP/2021/RPT.pdf`, capture 20240624/20240712 (40.0 MB, `%%EOF`; the 2023-06 and 2023-12 captures are fragments) | none (race, religion, marital, birthplace only) |
| BPP 2021 Annex A-C workbook (A1-A12, B1-B6, C1-C10) | religiondots' cached `bpp2021_excel_table_A-C.xls` | none |
| *Report of the BPP 2021 Education Characteristics*, 21 pp | `deps.gov.bn/wp-content/uploads/2025/10/Report-of-the-BPP-2021-Education-Characteristics.pdf` (kept in `data/raw/bn/`) | literacy by language (E26, read and write, several allowed), national: 368,110 literate aged 10+; one language only 76,729 (Malay 62,064, English 6,221, Chinese 3,205, other 5,239); several 291,381 (Malay+English 207,124; +Chinese 22,311; +Arabic 20,031; other 41,915). Same for 2011 |
| *Insights into the BPP 2021*: Health; ICT and Digital Technology | `deps.gov.bn/wp-content/uploads/2026/05/` | none |
| KBPP 2016 (mid-term survey) report and annexes A-E | Wayback `.../DOS/KBPP/finalreport2016/` | none |
| BPP 2011 *Demographic Characteristics* (10 pp summary) and concepts | deps.gov.bn and Wayback `.../DOS/BPP2011/` | none |
| Brunei Darussalam Statistical Yearbook 2021 | `deps.gov.bn/wp-content/uploads/2025/12/BDSYB-2021.pdf` | literacy by language only (p.139) |
| DEPS WordPress media API, searches language, bahasa, spoken, pertuturan, home, Malay, census, BPP, 2021, 2011, table | `deps.gov.bn/wp-json/wp/v2/media?search=` | only the literacy-by-language time series xlsx (kept) |
| UNSD DYB table 27 (language) | `data.un.org` DownloadHandler | dead: serves the UN Data Commons app (same as 2026-10-03) |

Not tried: the 1991 and 2001 census reports (not on either DEPS site or in the Wayback library
listing); IPUMS (no Brunei sample, and the account is blocked anyway); a data request to DEPS
(needs Anita, ask 016).

### Calls (2026-10-05, superseded by the build above)

- **Literacy by language is not drawn.** It asks what people can read and write, several answers
  allowed, national only; Malay+English is 71% of the multilingual. That is a skill question, and
  AGENT_BRIEF §2 rules learned languages out.
- **Race by district is not used as a proxy** (Malay / Chinese / Others, Table A3). It would set
  the counts, so it is Anita's to allow; offered in ask 016 and advised against.
- **Parked, not dropped,** so the queue shows it is waiting on the ask rather than free.

### If the table arrives

Reuse religiondots' district layer exactly as `countries/np.py` does (`RD_GEO`, units are the four
districts; religiondots' `sources/bn_geo.py` also has a mukim lookup witnessed against census C1,
39 mukims, if DEPS sends mukims). Nodes: Malay, English, Chinese (group node unless DEPS splits it,
since the form does not), Arabic, Others on `other`.

### Files

`data/raw/bn/`: `Report-of-the-BPP-2021-Education-Characteristics.pdf` (sha1 c208328192cf...),
`Literacy-Rate-of-the-Population-Aged-9-and-Above-by-Language.xlsx` (b9fc76cce4f5...),
`Questionnaire_BPP-2011.pdf` (db3a8cf2e508...). The other downloads above were deleted after
reading; their URLs are in the table.
