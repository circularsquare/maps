# Somalia (so): record

Drawn 2026-10-05 (session edd42a8c-mono2). REACH Joint Multi-Cluster Needs Assessment (JMCNA)
2021, household main language, weighted shares per region times COD-PS 2026 region populations;
every row `modelled`. 19,442,160 people on 18 regions (Somaliland's five included): Standard
Somali 61.0%, Maay 20.2%, Benaadir Somali 18.1%, Mushungulu 0.26%, others under 0.2%. Placed on
religiondots' Kontur hexes (read-only).

Files: `sources/so_jmcna.py`, `taxonomy/so2021.py`, `taxonomy/tree.d/so.txt`, `countries/so.py`,
`data/normalized/so.csv`; raw in `data/raw/so/` (the REACH workbook, 64 MB; its extract
`jmcna2021_language.csv`; CLEAR Global's admin1/admin2 CSVs, the lead).

## Source

- No census since 1975; PESS 2014 has no language item (queue scout). No Afrobarometer round.
- Lead: CLEAR Global's HDX dataset `somalia-languages` (CC BY-SA), whose admin1 shares cite the
  JMCNA 2021 dataset. Went to the source instead: REACH repository,
  `REACH_SOM2101_Final-Dataset_JMCNA_Somalia_01112021.xlsx`, open download. Sheet `Clean_Data`,
  11,349 households; question `main_language`, "What is the main language your household speaks
  at home?", one answer; `weights` column (mean 1.0) used. Phone survey, 30 May - 18 Aug 2021,
  quota sample of phone-owning households with network coverage, IDP and non-IDP strata;
  REACH calls results indicative. The 2025 MSNAs are HDX-restricted.
- CLEAR Global maps the "Somali Sign Language" answer to Kenya-Somali Sign Language and shows it
  at 15% of Galgaduud; that is the response error below, not a finding.

## Calls

- **Maay** drawn as its own language (Glottolog maay1238 beside Somali soma1255). Bay 95.8%,
  Bakool 88.0%, Lower Shabelle 41.8%, Banadir 25.5% (displacement from Bay), Middle Shabelle
  15.9%, Gedo 11.8%. Answers 2,416. The 2.75M (2020) speaker figure on Wikipedia is lower than
  our 3.9M; ours scales a phone sample by 2026 populations, and Banadir's IDP-heavy sample counts.
- **Benaadir Somali** is a Somali dialect in Glottolog (bena1268) but a separate answer in the
  survey, so a sibling node beside Somali (a child would wash Somali out as a group). Banadir
  48%, Middle Shabelle 47%, Lower Shabelle 39%, Hiraan 37%.
- **"Somali Sign Language"** (256 households, 2.7% weighted; Galgaduud 15%, Hiraan 8%, Mudug 6%)
  dropped as a response error: no deaf population is that large, it clusters where Standard
  Somali dominates, and the Somali wording "calaamadaha luuqada soomaaliga" reads close to
  "Somali". Not guessed onto Somali; regions renormalised over valid answers; in `gap`.
- Don't know / prefer not (0.5%) not drawn. Other-specify: "biyo maal" (a Dir clan, 11) on
  Somali, "oromo" (1) on Oromo, 9 blank on `other`.
- **Middle Juba** has no respondent (Jilib's IDP sample of 45 was cleaned out): drawn on Lower
  Juba's shares, its note says so.
- English and Italian drawn as measured (home language, not ability; 6 households).
- Bantu (Mushungulu 21 households) is likely undercounted by a phone survey; said in
  `note_public`. No retention or Bantu source used.

## Population and checks

religiondots' `so_lookup.csv`: COD-PS 2026 HRP regions, 19,442,160, asserted. Region labels
asserted to map; every label asserted to have a mapping; each region rounded by largest
remainder to its population exactly. `check_country.py so`: ok.

## Room for improvement

District grain is possible (74 districts sampled, ~150 households each) with a district
placement layer; the 2025 MSNA (restricted) would update it.

## Scatter

19,438 dots over 5,093 hexes.

## Moved from countries/so.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- Maay, spoken by the Digil and Mirifle clans and treated by linguists as a separate language from Somali, is 96% of Bay and 88% of Bakool, and about a quarter of Mogadishu.
- The survey reached households with a phone and network coverage, so remote and al-Shabaab-held areas are under-covered, and Bantu languages such as Mushungulu, which most Somali Bantu have given up for Maay or Somali, may be undercounted.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Somali is now a group holding Somali (its own leaf), Benaadir (a Glottolog dialect of Somali) and Maay (its own language in Glottolog, commonly called a Somali variety), so Ethiopia's and Kenya's Somali continue across the border. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
