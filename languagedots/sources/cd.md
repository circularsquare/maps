# DR Congo (cd)

Drawn 2026-10-05 (session edd42a8c-cd), from the Enquête 1-2-3 ethnic model alone. **Since
2026-10-09 (session 32a047f0) the national languages come from MICS-Palu 2017-18 microdata**
(§0): 117,808,872 people, 164 territoires and cities, 98 nodes, every row `modelled`, 117,761
dots at 1:1000, no rings. Sections 1-7 describe the ethnic model, which now only splits MICS's
"another language"; §4's city shifts are superseded.

```
python sources/cd_e123.py --fetch     # copies the USCB workbook from religiondots; checks; cd.csv
python sources/cd_mics.py             # MICS HC1B by province and stratum on top of cd.csv; cd_mics.csv
python taxonomy/build.py
python tools/check_country.py cd
python scatter.py --country cd
```

## 0. MICS-Palu 2017-18 (2026-10-09, `sources/cd_mics.py`, `taxonomy/cd2017.py`)

Anita made a UNICEF MICS account and downloaded the DRC MICS6 SPSS files to
`data/raw/cd/mics_2017/` (gitignored: research use, no redistribution; the readme asks that copies
of reports and publications go to the INS and UNICEF Kinshasa). 20,810 households, 20,792
interviewed, 103,422 members, 26 provinces (HH7), 25-30 clusters each (4-30 urban). The report
tabulates no language.

**Items.** Every language item codes French / Kikongo / Lingala / Swahili / Tshiluba / another
language. HC1B (head's mother tongue), HH16 (household respondent's), HH15 (interview language),
WM14 and MWM14 (women's and men's own, 15-49), FL7 (child 7-14's main home language). HC2
(ethnicity) is only Bantu / Nilotic / Sudanic / Pygmy and is not used.

| check | figure |
|---|---|
| HC1B = HH16, households | 92.9% |
| heads "another language" whose respondent also said another language | 9,956 of 10,834 (though 4,076 interviews were in Lingala, 2,994 in Swahili) |
| HC1B vs HH16 per province, national languages | all within 10 points; widest Ituri Swahili 29 / 38, Kinshasa Lingala 46.5 / 54 |
| male heads who answered both HC1B and MWM14 (same person) | 93.6% agree; of "another language" in HC1B, 6.0% name a national language in their own interview |
| female heads, same test | 87.2% agree; 13.3% |
| sons / daughters 15-49 of "another language" heads naming a national language | 13.8% / 18.8% |

**Choice: HC1B**, read as every member's (hl.sav members x hhweight), MICS's standard item and the
one Iraq used. HH16 tracks it closely and does not slide to the interview language as Iraq's
did. The individual interviews do slide: the same woman who said "another language" for herself
as household respondent names a national language 13% of the time in her own interview (men 6%),
so WM14 and MWM14 are not drawn. What is left of the sons' and daughters' 14-19% after that drift
is a real generational lean, strongest in Kinshasa (heads 46.5% Lingala, men 15-49 62%; daughters
of Kikongo-speaking heads 80 of 180 Lingala): the head's answer leans old, said in
`note_public`.

**How it enters: MICS replaces the ethnic model for what it names.** The ethnic model said
Lingala 7%, Swahili 3% (city shifts only); MICS's mother-tongue item says 16% and 20%, so as
retention or a check it would leave most of the gap. MICS is representative by province x
urban/rural:
- The 18 cities that are units of their own (`VILLES`: Kinshasa, Matadi, Boma, Bandundu, Kikwit,
  Mbandaka, Zongo, Gbadolite, Kisangani, Goma, Beni, Butembo, Bukavu, Kindu, Lubumbashi, Likasi,
  Mbuji-Mayi, Kananga) take their province's urban-stratum shares.
- The province's other territoires share the remainder (province shares x COD-PS minus the
  cities), fitted across territoires (IPF) with the ethnic model as seed where the ethnic groups
  account for a category: "another language" by the non-national groups' share, Tshiluba by
  Luba-Kasai heads, Kikongo by Kongo heads (Kongo Central). Lingala, Swahili, French: by population.
- Inside each territoire, "another language" is split among its non-national ethnic groups in
  the Enquête 1-2-3's proportions; Tshiluba is Luba-Kasai; Kikongo is the Kongo varieties in
  Kinshasa and Kongo Central (ethnic Kongo heads at least half of MICS's Kikongo; asserted), and
  elsewhere the Kongo varieties up to the territoire's ethnic Kongo share, the rest Kikongo ya
  leta (Kituba, cg.txt's node): Kwilu 14.7%, Kwango 10.4%.
- Counts: shares x each territoire's whole COD-PS 2024 population, largest remainder. The old
  build left out 2.3% of heads with no ethnic group; MICS answered for all but 16 households, so
  the total rises from 115,474,204 to 117,808,872.
- MICS's urban weights do not match COD-PS's cities (Kasai-Oriental 41% urban in MICS, Mbuji-Mayi
  63% of COD-PS; Sud-Kivu 53% against Bukavu's 14%). Where the cities alone exceed a province's
  total the remainder is clipped at 0 (Kasai-Oriental Swahili, French, Lingala; under 1.5% of the
  rest of the province; printed). A density-based urban split from the hexes was tried and
  dropped: COD-PS calibration makes rural Haut-Lomami "urban" and Likasi rural.

| % | Lingala | Swahili | Tshiluba | Kikongo (Kongo + Kituba) | French | other |
|---|---:|---:|---:|---:|---:|---:|
| **DR Congo**, before (115.5M) | 7.1 | 3.1 | 8.9 | 7.9 | 0 | 72.8 |
| **DR Congo**, after (117.8M) | 14.6 | 19.6 | 11.0 | 9.8 + 1.3 | 0.6 | 43.1 |
| Kinshasa | 58.0 → 46.5 | 0 → 4.3 | 4.4 → 10.2 | 11.3 → 28.8 | 0 → 2.3 | 26.3 → 7.9 |
| Lubumbashi | 0 → 1.6 | 72.0 → 61.1 | 7.3 → 12.6 | 0.3 → 0.4 | 0 → 1.1 | 20.4 → 23.3 |
| Bukavu | 0 → 1.0 | 72.0 → 64.0 | 0 | 0 | 0 → 0.3 | 28.0 → 34.7 |
| Goma | 0 → 3.4 | 72.0 → 80.6 | 0.9 → 0 | 0.2 → 0 | 0 → 0.7 | 26.9 → 15.3 |
| Kisangani | 0 → 75.5 | 0 → 18.2 | 1.4 → 0 | 0.7 → 0 | 0 → 4.7 | 97.8 → 1.6 |
| Mbuji-Mayi | 0 → 0.5 | 0 → 3.2 | 57.0 → 92.1 | 0.3 → 0.1 | 0 → 2.3 | 42.7 → 1.9 |
| Kananga | 0 | 0 → 0.9 | 86.7 → 74.1 | 0 | 0 → 0.3 | 13.3 → 24.7 |
| Nord-Kivu | 0 → 1.2 | 6.7 → 71.4 | 0.2 → 0 | 0.9 → 0 | 0 → 0.5 | 92.1 → 27.0 |
| Sud-Kivu | 0 → 0.7 | 9.8 → 41.0 | 0.1 → 0 | 0 | 0 → 0.2 | 90.1 → 58.1 |
| Haut-Katanga | 0 → 1.0 | 33.1 → 42.2 | 5.7 → 8.1 | 0.2 | 0 → 0.7 | 61.1 → 47.8 |
| Maniema | 0 | 0 → 83.4 | 0.1 → 0 | 0 | 0 → 0.2 | 99.8 → 16.4 |
| Tshopo | 0 → 63.3 | 0 → 22.3 | 1.0 → 0.1 | 0.8 → 0 | 0 → 2.1 | 98.3 → 12.1 |
| Bas-Uele | 0 → 83.1 | 0 → 0.8 | 0.1 → 0 | 0.3 → 0 | 0 → 0.5 | 99.6 → 15.6 |
| Équateur | 9.9 → 30.4 | 0 → 0.2 | 0 → 0.1 | 0.1 → 0.9 | 0 → 0.2 | 90.1 → 68.2 |
| Kwilu | 0 → 2.6 | 0 → 0.1 | 0.4 → 0.7 | 0.1 → 14.8 | 0 → 0.3 | 99.5 → 81.5 |
| Kongo Central | 0 → 6.1 | 0 → 0.7 | 0.5 → 0.1 | 91.9 → 91.9 | 0 | 7.6 → 1.2 |

Local languages shrink where
the lingua francas grow: Nande 5.04M to 1.54M, Kinyarwanda 2.33M to 0.75M, Shi 2.97M to 1.96M,
Luba-Katanga 6.00M to 4.69M, Budza 1.06M to 0.36M, "other African" 9.75M to 5.06M; Luba-Kasai
10.32M to 12.98M, Kongo varieties 9.06M to 11.51M, Kituba 0 to 1.48M, Lingala 8.43M to 17.20M,
Swahili 3.60M to 23.14M.

**Weak points.**
- **Nord-Kivu** reads 71% Swahili (rural 66%), and every item agrees (HH16 70, WM14 72, MWM14 84,
  children's home language 87), so it is drawn. MICS cannot say where in the province: Lubero
  and Beni (Nande) take the same Swahili share as Masisi and Nyiragongo, and Butembo and Beni the
  province's town figure (81% Swahili), set mostly by Goma. Nande's fall from 5.0M to 1.5M rests
  on that.
- **Clusters.** 25-30 per province: a group holding 1% of a province in one place is missed
  entirely about three times in four ((0.99)^28). It matters for lingua-franca enclaves, e.g.
  Lingala in eastern garrison and mining towns, which MICS puts at 0-1.2% in the Kivus, Maniema
  and Katanga.
- The head's answer leans old (above). Children's own first language is not asked; FL7 (home
  language, 7-14) is the use reading ask 018 keeps out, and it runs higher still (Kinshasa 72%
  Lingala, 22% French).
- Placement inside a territoire is by population for every language (cities are units, so
  Lubumbashi's Swahili stays in Lubumbashi, but Kolwezi's is spread over Mutshatsha).

## 1. What exists, and what was not used

No census since 1984. Nothing open prints a language table below the whole country.

| Source | What it has | Why not drawn |
|---|---|---|
| **USCB "Tribe and Religion" sheet** (HDX, CC BY), Enquête 1-2-3 2005 + 2012 | 31,755 household heads by 103 named ethnic groups + Other + No data, for 26 provinces and 164 districts | **Drawn** (§2) |
| CLEAR Global `democratic-republic-of-the-congo-languages` (HDX, CC BY-SA, 2025) and TWB `drc-languages` (2019; its PDF says CC BY-NC-SA) | CAID 2016 administrative reports' "languages spoken" per territoire, 115 territoires; several languages per person, lingua francas at 80-100% (Lingala 80% of Aketi); CLEAR rescales to sum 1. TWB: "methodology unclear, differs across areas, all confidence low". Glottocode slips (Tamazola Mixtec, Lander River Warlpiri, Ibaloi). | Multi-answer, administrative estimate, the lingua franca problem at its worst. Used as the second-source check (§3). Files in `data/raw/cd/`. |
| MICS 2010 (World Bank microdata catalog 1313, variable HC1B) | "Quelle est la langue parlée principalement par le chef de ménage et dans le ménage?": Swahili 25.3%, Lingala 18.1%, Tshiluba 11.8%, Kikongo 9.4%, French 2.0%, English 0.2%, other 33.1% (11,393 households, unweighted) | Open frequencies are national only; microdata at mics.unicef.org behind a UNICEF account. A free self-serve account is allowed but needs an email; not created. The national check (§4). |
| MICS-Palu 2017-18 | asks HC1A/HC1B; the report tabulates neither by province | **Drawn since 2026-10-09** (§0), microdata through Anita's UNICEF account |
| DHS 2013-14, 2023-24 | language of interview / native language in recode files | DHS registration is off |
| Afrobarometer, WVS | DR Congo not covered | |
| QUIBB-RDC (Ministère du Plan, 2024 PDF) | no language or ethnicity table (searched) | |

## 2. The source drawn

The U.S. Census Bureau's DR Congo workbook (`democratic-republic-of-the-congo_uscb_202103.xlsx`,
the one religiondots draws religion from), sheet "Tribe and Religion": household heads of each
ethnic group summed over the INS's Enquête 1-2-3 rounds of 2005 and 2012. **Rule: ethnicity
only (AGENT_BRIEF §2), each group read as its language**, laid on OCHA's COD-PS 2024 territoire
populations (religiondots' `cd_territoires.csv`, read-only), every row `modelled`.

- **Unit: territoire**, not province. Ethnic geography in DR Congo is local, the survey sampled
  every district it visited (40 to 2,758 heads, median 174), and religiondots' hexes already carry
  `territoire` with a COD-PS-calibrated `pop`. `place_unit` is `territoire`.
- **16 districts not sampled** (Kiri, Bolobo, Yumbi, Moanda, Kimvula, Bomongo, Ingende, Befale,
  Sakania, Nyunzu, Nyiragongo, Niangara, Opala, Bafwasende, Shabunda, Idjwi) take their
  province's shares, sampled districts weighted by COD-PS 2024 population.
- **Join**: USCB's NSO_CODE is COD-AB's admin2 pcode for 163 of 164; Kasongo-Lunda is CD3107 in
  USCB and CD3106 in COD-AB, joined and asserted on the name.
- **No data** (731 heads, 2.3%) not drawn, the `gap`. Shares are over heads with a group.
- **Household heads, not people**: everyone in a household is drawn in the head's group, as
  religiondots' cd does. No household sizes to test it.
- **Labels**: the survey's own field names (USCB's Data Dictionary keeps them), not USCB's ISO
  renamings ("Twa" became Plains Bira; "Mbunda" took Angola's code).

## 3. Checks (sources/cd_e123.py, all pass)

- 105 columns, all distinct original names; 1 / 26 / 164 rows; every sampled row sums exactly to
  its sample size; districts sum exactly to provinces and country per column.
- 164 districts = 164 territoires one to one; every district in the province religiondots' hexes
  put it in.
- **Second source, CLEAR/CAID 2016**: per-territoire Spearman between CAID's "can speak" share and
  the survey's ethnic share, 24 languages both name: median **+0.46** (Hunde +0.83, Yombe +0.80,
  Bushoong +0.76, Lendu +0.68, Shi +0.66, Zande +0.66; Fuliiru -0.45 and Hemba -0.12 over 8-11
  territoires). Positive for most, but weak: no permutation baseline, and the territoires
  compared are only those where either source names the language.

## 4. Lingua franca against first language (ask 018), and the Kinshasa switch

**Superseded 2026-10-09** (§0): MICS-Palu 2017-18 measures the head's mother tongue per province
and in its towns, so `CITY_SHIFT` is gone. Against the borrowed shares below, MICS gives Kinshasa
46.5% Lingala (58% drawn here), Mbandaka 67% (58%), Bukavu 64% Swahili (72%), Goma 81%,
Lubumbashi 61% (72%), Likasi 61%, and Kisangani 76% Lingala, 18% Swahili (nothing drawn here).
Kept as the record of what was tried.

Reading ethnicity as language is the first-language reading that ask 018 leans to (and that the
rest of the map draws). It cannot show a national language anyone speaks as a first language
outside their group, and DR Congo's four are exactly that. Two figures bear on it:

- **MICS 2010** (the use reading, "parlée principalement ... dans le ménage"): Swahili 25.3%,
  Lingala 18.1%, Tshiluba 11.8%, Kikongo 9.4%. Drawn here: Lingala 7.1%, Tshiluba (Luba-Kasai)
  9.4%, the Kongo varieties 7.9%, Swahili 0. **Tshiluba and Kikongo agree** (each group's own
  language, so the readings coincide); **Swahili and Lingala are the gap**: 36 points of
  households that mainly use a lingua franca, as Tanzania's R7-R9 rounds were.
- **Kinshasa**: 58% of 500 third-year secondary pupils in Ngaliema commune declared Lingala their
  mother tongue "regardless of their ancestral language" (87% speak it); Mavita Tseki, Kalokola
  Yangonde, Kamasukako Buka, Mukala Bobo and Olomwene Omo, *IJSSMR* 9(2),
  https://ijssmr.org/vol-9-issue-2/language-use-among-3rd-year-literary-students-in-ngaliema-kinshasa-a-sociolinguistic-and-psychopedagogical-study-of-multilingualism-in-the-congolese-school-context/ .
  Applied to all of Kinshasa: 58% drawn Lingala, every ethnic share scaled by 0.42. It is one
  commune and one school year, and older Kinois born elsewhere will name their group's language
  more often: a lean, said in `note_public`.
- **Switch**: `CITY_SHIFT` in `countries/cd.py` (empty it to draw every city by ethnic group).

### 4a. Five more cities (2026-10-07, fix-cd, Anita asked for a stab)

**Measure used: first language, not household use** (ask 018's ruling). MICS 2010's HC1B is the
use reading and is not drawn. The two figures used are the closest to first language found.

**Second measured figure, Bukavu Swahili**: Timothy Wilt, *Bukavu Swahili: a sociolinguistic
study of language change*, Michigan State PhD 1988 (open full text,
https://d.lib.msu.edu/etd/21180 , OCR at `/etd/21180/FULL_TEXT/download`), Table 0.1, "Mother
tongue language(s) according to age", from the question "which language do you use with your
parents", 84 subjects, stratified non-random sample, fieldwork c. 1986:

| born | n | ethnic only | ethnic + Sw | Sw only | Sw + Fr | ethnic + Sw + Fr |
|---|---|---|---|---|---|---|
| before 1950 | 28 | 86% | 7% | 7% | 0 | 0 |
| 1950-69 | 27 | 7% | 44% | 48% | 0 | 0 |
| 1970-75 | 29 | 3% | 14% | 72% | 7% | 3% |

**72% drawn**: the youngest cohort's "Swahili only", the narrowest reading (no ethnic language
with parents at all). Counting Swahili-French (79%) or sharing the mixed answers (84%) would be
higher; nearly everyone in Bukavu today was born after 1970, and Wilt says the shift was still
running, so 72% leans low. It is a use-with-parents question, which for people then aged 11-16
is close to first language; Wilt notes it "does not necessarily imply that they learned the
Swahili from their parents".

**Borrowed, where a study saw children speak the lingua franca among themselves** (shares
borrowed, never invented; each city takes the measured figure of its own lingua franca):

| unit | city | drawn | evidence that the city is like the measured one |
|---|---|---|---|
| CD6101 | Goma | 72% Swahili | Translators without Borders, *Missing the mark* (Goma, Feb 2019, 216 respondents): 97% speak Congolese Swahili as their main language at home; the report says city youth often no longer speak their group's language |
| CD7101 | Lubumbashi | 72% Swahili | ACCELERE! 1 sociolinguistic mapping (USAID / SIL LEAD, Gibson 2018, pdf.usaid.gov PA00TRB3, via Wayback): urban Haut-Katanga, 82% of children's play groups in Swahili, the rest French; every urban first-grader speaks local Swahili well |
| CD7106 | Likasi | 72% Swahili | same, urban Haut-Katanga |
| CD4101 | Mbandaka | 58% Lingala | same study: 53 of 54 play groups in Lingala, the other French; all children speak it well |

Effect: Swahili (Congo Swahili, `swahili_congo`, zm.txt's node) 0 to 3,595,599 (3.1%); Lingala
8,232,244 (7.1%) to 8,427,462 (7.3%). Combined 7.1% to 10.4%, against MICS's 43% household use.
Most of that gap is the use/first-language difference the ruling keeps; the rest is below.

**Not moved, and why**:
- **Gemena** (CD4202) and **Kolwezi** (inside Mutshatsha, CD7202): the city is not its own unit;
  ACCELERE! found rural Gemena children 2% Lingala, so a territoire-wide share would be wrong.
- **Kisangani** (CD5101): Swahili and Lingala both, by neighbourhood (Nassenstein 2018, Kisangani
  Swahili); no split exists, and an unsourced split is not drawn.
- **Kindu, Uvira, Kalemie, Bunia, Butembo, Beni**: no observation of children's language found.
- **The countryside**: ACCELERE! saw rural Haut-Katanga children play 84% in Swahili and rural
  Lualaba 65%, but play language is not first language and no rural mother-tongue figure exists.
- **Kikongo ya leta (Kituba)** in Kikwit, Bandundu, Matadi: also a lingua franca drawn as ethnic
  Kongo/Yaka/Mbala etc.; out of this pass's scope (Swahili and Lingala), no figure looked for.

**Searched, nothing tabulated by province** (2026-10-07):
- MICS 2010 final report (Wayback copy of childinfo.org `MICS-RDC_2010_Final_Report_FR.pdf`,
  scanned): Table HH.3 gives head's sex, province, residence, size, education, religion; no
  language. HC1B is never tabulated.
- DHS 2013-14 (FR300) and 2023-24 (FR393, open PDFs): both record language of interview, and
  2023-24 the respondent's "langue maternelle" (codes French, Kikongo, Lingala, Swahili,
  Tshiluba, other) on the cover sheet, but no table prints either. Recode files gated.
- Afrobarometer R4-R9 (religiondots' .sav files): DR Congo in no round (R9 has Congo-Brazzaville).
- Lubumbashi: Ngoie Kyungu Kiboko, *Le français à Lubumbashi* (Nice 2015 thesis, HAL behind a
  bot wall, not read); Kalunga-type ELA 2009 article (erudit) has no figures.
- Goma/Bukavu: nothing newer than Wilt with a first-language share found.

**Room for improvement**: DHS 2023-24's "langue maternelle" item, 26 provinces, would replace all
of this with a measured first-language share per province (DHS registration is off).

## 5. Mapping calls (taxonomy/cd2012.py)

- **"Luba" alone is Luba-Kasai everywhere**, Katanga included. The Kiluba heartland answered "Luba
  Shaba" (Manono 208, Kamina 151, Kabongo 127, Kaniama 122 heads) and almost never "Luba" (0, 0,
  2, 18); Katanga's "Luba" heads are in Lubumbashi (114), Likasi (76) and Mutshatsha (53), where
  Kasai Luba settled. A first build that split "Luba" by province drew Luba-Katanga at 7.5M; now
  6.4M (5.6%).
- **Merged**, each one language: Lulua + the Luba-Kasai clans (Bakwa Kalonji, Bakwa Dishi, Bakwa
  Mulumba, Bakwanga) on Luba-Kasai; Kanioka + Bena-Kanioka on Kanyok; Leele + Selele (Basilele)
  on Lele; Tshokwo + Tshoko on Chokwe (Tshoko 61% in Kasai/Tshikapa, a Chokwe area).
- **Kongo varieties kept apart** (Ndibu, Manyanga, Ntandu, Mbata, Lemfu, Besi Ngombe, Bakongo du
  Sud-Est, Mboma, Yombe) under a new group `kongo_dialects`, beside Angola's `kongo` leaf.
- **"Hutu (Ruzizi)"** is 94% in Nord-Kivu, not the Ruzizi plain: Kinyarwanda.
- **"Mbunda"** (87% Kwilu) is the Mbuun, not Angola's Mbunda. **"Bangando"** (92% Tshuapa) is
  Glottolog's Ngando of DR Congo. **"Ngbaka (Gwakamabo)"** (Sud-Ubangi 63%) is Ngbaka Minagende,
  cf.txt's node. **"Bango"** reuses cf.txt's Babango. **"Nyari"** is Nyali (Bantu, in Ituri).
- **Twa** (70 heads; Kasai 43%, Mai-Ndombe 30%, Équateur 23%) speak their neighbours' languages,
  no single one: `africa_other`. **Other** (3,237 heads, 10.2%; no breakdown): `africa_other`,
  nearly all Congolese groups the survey did not list.
- Tere, Kundu, Benye Nonda: no Glottolog entry under these names; Bantu leaves where they live.

## 6. Colours

Bantu's 45 generated slots are full: leaving the ~60 new leaves to the generator moved 34 other
countries' colours (Kinyarwanda, Luba-Kasai, Mongo, Ganda...). Every new leaf is hand-coloured at
chroma 0.20 or 0.08 (the parent's is 0.14), which cannot land within 0.04 of a generated slot;
checked: no existing node's colour changes. Neighbours (≥2% of a province) checked pairwise in
OKLab. Congo Swahili (2026-10-07) keeps zm.txt's generated #ffa86d, the same as the Zone J
group colour (only drawn for unnamed Zone J, which cd has none of); nearest neighbour on the
ground is Luba-Kasai in Lubumbashi at 0.048, Bemba 0.070. Left for the colour pass. **Since
MICS (2026-10-09) this matters more**: Swahili is now 19.6% of the country and meets Luba-Kasai
across Haut-Katanga, Lualaba and Lomami, and Kusu (#ff9800) in Maniema. Hand-colouring it would
free a generated slot and move other countries' Bantu colours (as Luba-Katanga did), so not
done here. Kituba (Kwilu, Kwango) keeps cg.txt's pale green, far from Yaka, Mbala and Pende. **Left for Anita's colour pass**: Luba-Katanga (generated, used by ca/fi/pl) sits 0.022
from Zambia's Tabwa (Tanganyika) and 0.043 from Luba-Kasai; hand-colouring it freed its slot and
moved eight other countries' Bantu colours, so it was not touched.

## 7. Room for improvement

- ~~MICS 2010 or 2017-18 microdata~~: 2017-18's HC1B drawn since 2026-10-09 (§0). MICS 2010
  (use reading, "parlée principalement") is not wanted under ask 018.
- **DHS 2023-24 recode** (registration): native language per respondent, 26 provinces, and a
  second sample to set against MICS's Nord-Kivu (71% Swahili).
- Anything below the province for the national languages (territoire or health-zone language
  shares) would replace the population spread of Lingala and Swahili inside each province.
- The person-level Enquête 1-2-3 files (University of Antwerp registration) would give ethnicity
  for every household member, not only heads, and any language item the rounds carried.

## 8. Terms

USCB workbook CC BY (HDX). CLEAR Global CC BY-SA, used only as a check, not republished.
MICS-Palu 2017-18 microdata: UNICEF's terms, research use, no redistribution of the files; only
province-level aggregates are written (`data/normalized/cd_mics.csv`), and the readme asks that
copies of publications based on it go to the INS and UNICEF Kinshasa.
COD-PS 2024 and COD-AB via religiondots, CC BY-IGO; Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

The Kongo varieties (Yombe, Ndibu, Manyanga, Ntandu, Mbata, Lemfu, Besi Ngombe, Kongo of the south-east bank, Mboma) now sit directly in a Kongo group with Angola's and Congo's 'Kongo' (that leaf is 'Kongo (variety not given)'); the old 'Kongo varieties (H.10)' node is empty. Nande is in a Konzo-Nande group with Uganda's Konzo, Kinyarwanda in Rwanda-Rundi, Lugbara in a Lugbara group with Uganda's Aringa. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
