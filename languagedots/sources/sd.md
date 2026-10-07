# Sudan: speaker estimates placed by homeland (ask 019)

Redrawn 2026-10-05 (session edd42a8c-est) on ask 019's ruling: published speaker estimates by
homeland, rows `modelled`. First drawn the same day from surveys (session edd42a8c-sd, 98.4%
Arabic); that build is kept only as a check (section 4). 46,934,433 people (COD-PS 2022), 18
states, 75 nodes, every row `modelled`. 46,901 dots at 1:1000. Sudanese Arabic 82.2%, other
languages 17.8% (8.36M).

```
python sources/sd_estimate.py        # -> data/normalized/sd.csv
python taxonomy/build.py
python tools/check_country.py sd
python scatter.py --country sd
python sources/sd_afro.py            # the survey check -> data/raw/sd/sd_survey_by_state.csv
```

Files: `sources/sd_estimate.py`, `taxonomy/sd2026.py` (identity), `taxonomy/tree.d/sd.txt`,
`countries/sd.py`. `taxonomy/sd2022.py` (the survey mapping) deleted; `sources/sd_afro.py` now
writes to `data/raw/sd/` so it cannot overwrite the map's CSV.

## 1. What exists

- **Census.** Only 1955-56 asked a language (51% Arabic over the whole of Sudan with the South;
  about 70% for what is now Sudan). Volumes search-only on HathiTrust (record 001886402); no
  province table online. 1973-2008 asked none.
- **Surveys** (Afrobarometer R5-R9, Arab Barometer II-VII): every interview in Arabic or English,
  98% Arabic answers, no Nuba language at all (section 4; the interview-language table is in
  `ask/019-sd.md` and `sources/sd_afro.py`'s docstring). CLEAR Global's layer is Afrobarometer
  R9. MICS 2014 needs an account.
- **Estimates used**: Ethnologue via Wikipedia's infoboxes (read 2026-10-05) and Joshua Project's
  people-group table (`data/raw/pg/joshuaproject_pgic.csv`, already on disk from pg).

## 2. How the counts are made (`sources/sd_estimate.py`)

**Figure, one rule.** Ethnologue's Sudan figure where Wikipedia carries one dated 2019 or later
for a language spoken (almost) only in Sudan: Beja 2,550,000 ("In 2022 there were 2,550,000
Beja speakers in Sudan"), Masalit 980,000 (2022-24), Fur 790,000 (2004-2023), Nyimang 170,000,
Gaam 110,000, Midob 93,000, Moro 79,000 (all 2022), Dongolawi 35,000 (2023). Everything else:
Joshua Project's Sudan groups summed by primary language, times COD-PS 2022 / JP's Sudan total
(46,934,433 / 52,825,000 = 0.8885). JP already puts 40 Arabised groups (6.6M people: "Zaghawa,
Arabized", "Kadugli, Arabized"...) on Sudanese Arabic, so its language column is close to a
speaker estimate. Koalib (100,000, 2009) and Katcha-Kadugli-Miri (75,000, 2004) have only
older Ethnologue figures, so JP's 152,000 and 102,000 are used.

**Ethnologue against JP (scaled)**, the check: Beja 2.55M / 1.96M (JP files Beni Amer under
Tigre, 432k), Masalit 980k / 494k, Fur 790k / 1.28M, Nyimang 170k / 183k, Gaam 110k / 121k,
Midob 93k / 97k, Moro 79k / 84k, Dongolawi 35k / 83k. Close for the Nuba and Blue Nile
languages; far apart for Fur and Masalit (both directions), which is the map's least sure part.
Sudanese Arabic: Ethnologue L1 41M worldwide (2022-24); drawn 38.6M in Sudan.

**Left out**: JP's foreign groups (Egyptian 865k, Moroccan, Syrian, Algerian, Yemeni Arabs,
Amhara, Oromo, Tigrinya, Kunama, Me'en, Swahili, Chinese), "Deaf"; Tigre (Beni Amer counted in
Ethnologue's Beja; Eritrean Tigre foreign); Kanuri (JP's 461,000 "Kanuri, Yerwa" at one East
Darfur point: no speaker source, no survey answer). All stay on Sudanese Arabic.

**Placement.** One-point languages to the state holding JP's point (nearest hex centroid), with
two fixes: Midob to North Darfur (JP's point a degree east of Jebel Midob), Katcha-Kadugli-Miri
to South Kordofan (Kadugli; the point sits on the line). Spread languages by fixed split
(`SPLIT`, reasons in the code): Beja Red Sea 45 / Kassala 45 / River Nile 5 / Gedaref 5;
Nobiin Northern 50 / Khartoum 30 / Kassala (New Halfa) 20; Fur Central 45 / South 25 / North 25
/ West 5; Masalit West 60 / South 25 / Central 15; Zaghawa North 80 / South 20; Dar Fur Daju
South Darfur 50 / West Kordofan 50; Fulfulde South Darfur 40 / Blue Nile 30 / Sennar 15 / White
Nile 15; Hausa a fifth each in Khartoum, Gezira, Sennar, Gedaref, Blue Nile.

**80% cap.** South Kordofan's projection (1,198,313) is smaller than its languages' 1.36M, and
West Darfur came out 94% non-Arabic. No state is drawn above 80%; the excess (397,052 and
157,286) goes to Khartoum at the state's own mix. Khartoum is then 8.5% non-Arabic.

**As drawn**, non-Arabic share: Red Sea 69%, West Darfur and South Kordofan 80% (capped),
Central Darfur 61%, Blue Nile 61%, Kassala 45%, Northern 32%, North and South Darfur 16%,
West Kordofan 10%, Khartoum 8.5%, River Nile 7.5%, Gedaref 6%, the rest under 3%; East Darfur
0% (its JP groups are all Arabic or Kanuri).

## 3. Calls someone might reverse

- Ethnologue over JP for Fur (790k vs 1.28M) and Masalit (980k vs 494k).
- The 80% cap and Khartoum as the overflow (uncited; 554k people moved).
- Every `SPLIT` fraction (judged from where the literature names each group; no state table).
- Kanuri and Tigre left out; foreigners on Arabic.
- Kordofanian as one group under Niger-Congo, Kadu under Nilo-Saharan; Moro on au's
  `nigercongo.moro` outside the Kordofanian group so the language has one node.
- Inside a state everything is placed by population (Nobiin in Kassala is not kept to New
  Halfa, Beja not kept out of it).

## 4. The survey check

Pooled Afrobarometer R6-R9 and Arab Barometer III (`sources/sd_afro.py`), located respondents'
Arabic share against the drawn one, central states: Khartoum 99.8% / 91.5%, Gezira 99.3 / 99.6,
White Nile 98.8 / 98.7, Sennar 99.9 / 97.2, River Nile 99.0 / 92.5, North Kordofan 99.3 / 99.7,
Gedaref 100 / 93.8. Agreement where the estimate places no homeland; the gaps are Khartoum
(Nobiin and the Nuba and Darfuri overflow), River Nile and Gedaref (Beja), Sennar (Hausa,
Fellata), all groups an Arabic interview misses. On the periphery the survey cannot check:
Red Sea 9% Beja surveyed, 69% drawn.

## 5. Room for improvement

The 1955-56 census's province tables (the only measured geography), a state-level source for
the Darfur languages, MICS 2014 Sudan (needs a UNICEF account), zones inside states (New Halfa,
Jebel Marra, the Nuba hills of West Kordofan).

## Terms

Ethnologue figures as quoted on Wikipedia (CC BY-SA). Joshua Project dataset: free download.
COD-PS: OCHA, CC BY-IGO. Afrobarometer, Arab Barometer: free, citation requested. Glottolog CC
BY. Kontur CC BY 4.0.
