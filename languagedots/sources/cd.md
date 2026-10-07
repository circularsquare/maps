# DR Congo (cd)

Drawn 2026-10-05 (session edd42a8c-cd). 115,474,196 people, 164 territoires and cities, 95 nodes,
every row `modelled`. 115,428 dots at 1:1000, no rings.

```
python sources/cd_e123.py --fetch     # copies the USCB workbook from religiondots; checks; cd.csv
python taxonomy/build.py
python tools/check_country.py cd
python scatter.py --country cd
```

## 1. What exists, and what was not used

No census since 1984. Nothing open prints a language table below the whole country.

| Source | What it has | Why not drawn |
|---|---|---|
| **USCB "Tribe and Religion" sheet** (HDX, CC BY), Enquête 1-2-3 2005 + 2012 | 31,755 household heads by 103 named ethnic groups + Other + No data, for 26 provinces and 164 districts | **Drawn** (§2) |
| CLEAR Global `democratic-republic-of-the-congo-languages` (HDX, CC BY-SA, 2025) and TWB `drc-languages` (2019; its PDF says CC BY-NC-SA) | CAID 2016 administrative reports' "languages spoken" per territoire, 115 territoires; several languages per person, lingua francas at 80-100% (Lingala 80% of Aketi); CLEAR rescales to sum 1. TWB: "methodology unclear, differs across areas, all confidence low". Glottocode slips (Tamazola Mixtec, Lander River Warlpiri, Ibaloi). | Multi-answer, administrative estimate, the lingua franca problem at its worst. Used as the second-source check (§3). Files in `data/raw/cd/`. |
| MICS 2010 (World Bank microdata catalog 1313, variable HC1B) | "Quelle est la langue parlée principalement par le chef de ménage et dans le ménage?": Swahili 25.3%, Lingala 18.1%, Tshiluba 11.8%, Kikongo 9.4%, French 2.0%, English 0.2%, other 33.1% (11,393 households, unweighted) | Open frequencies are national only; microdata at mics.unicef.org behind a UNICEF account. A free self-serve account is allowed but needs an email; not created. The national check (§4). |
| MICS-Palu 2017-18 | asks HC1A/HC1B; the report tabulates neither by province | same account |
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
- **Switch**: `KIN_LINGALA` in `countries/cd.py` (0.58; 0 draws Kinshasa by ethnic group alone).

**Not moved, for want of any figure**: Swahili as first language in Lubumbashi, Likasi, Kolwezi,
Kisangani, Kindu, Goma, Bukavu; Lingala in Mbandaka, Kisangani and the Équateur river towns.
Searched (2026-10-05) for a city or province mother-tongue share and found none beyond Kinshasa.

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
OKLab. **Left for Anita's colour pass**: Luba-Katanga (generated, used by ca/fi/pl) sits 0.022
from Zambia's Tabwa (Tanganyika) and 0.043 from Luba-Kasai; hand-colouring it freed its slot and
moved eight other countries' Bantu colours, so it was not touched.

## 7. Room for improvement

- **MICS 2010 or 2017-18 microdata** (UNICEF account): household language by province crossed
  with nothing else would give the use reading per province; 2017-18's HC1B per province would
  give a first-language reading for the national languages, the retention this build lacks.
- **DHS 2023-24 recode** (registration): native language per respondent, 26 provinces.
- The person-level Enquête 1-2-3 files (University of Antwerp registration) would give ethnicity
  for every household member, not only heads, and any language item the rounds carried.

## 8. Terms

USCB workbook CC BY (HDX). CLEAR Global CC BY-SA, used only as a check, not republished.
COD-PS 2024 and COD-AB via religiondots, CC BY-IGO; Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

The Kongo varieties (Yombe, Ndibu, Manyanga, Ntandu, Mbata, Lemfu, Besi Ngombe, Kongo of the south-east bank, Mboma) now sit directly in a Kongo group with Angola's and Congo's 'Kongo' (that leaf is 'Kongo (variety not given)'); the old 'Kongo varieties (H.10)' node is empty. Nande is in a Konzo-Nande group with Uganda's Konzo, Kinyarwanda in Rwanda-Rundi, Lugbara in a Lugbara group with Uganda's Aringa. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
