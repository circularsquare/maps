# Pakistan

Drawn: 463 units, 246,500,586 people, 25 language nodes. Four provinces and Islamabad from the
2023 census's Table 11 (measured), at tehsil grain since 2026-10-07 (443 units from 591 tehsils,
§9), with Khyber Pakhtunkhwa's OTHERS split by MICS 2019 since 2026-10-09 (§0); Gilgit-Baltistan
and Azad Jammu & Kashmir modelled at district grain (added 2026-10-06, session `5d7dac7e-gb`, on
Anita's "would be nice to get gilgit baltistan if possible").

| part | source | people | tier |
|---|---|---|---|
| Punjab, Sindh, KP, Balochistan, Islamabad (591 tehsils, 136 districts) | Census 2023 Table 11, mother tongue (`sources/pk_t11.py`) | 239,498,938 | measured |
| KP's census OTHERS, named | Table 11 OTHERS x KP MICS 2019 district shares (`sources/pk_mics.py`) | 959,151 | modelled |
| Gilgit-Baltistan (10 districts) | census 2023 GB totals x GB MICS 2016-17 weighted district shares, raked (`sources/pk_north.py`) | 1,709,030 | modelled |
| Azad Kashmir (10 districts) | AJK Statistical Year Book 2025 Table 15.31 x census 2023 population | 4,333,467 | modelled |

## 0. MICS microdata (2026-10-09, session `32a047f0`, `sources/pk_mics.py`)

Anita downloaded two MICS rounds from mics.unicef.org with her UNICEF account; the SPSS files are
in `data/raw/pk/mics_gb2016/` and `data/raw/pk/mics_kp2019/` (gitignored; research use, no
redistribution; the GB readme asks that copies of publications go to GB P&DD and UNICEF
Pakistan). `pk_mics.py` reads them in place and writes only shares and split counts:
`data/normalized/pk_mics_gb2016.csv` (read by `pk_north.py`) and `pk_kp_mics.csv` (read by
`countries/pk.py`). Run order: `pk_t11.py`, `pk_mics.py`, `pk_north.py`.

### 0.1 Gilgit-Baltistan, MICS5 2016-17: same survey as before, now weighted

6,460 households, 6,213 interviewed, 46,276 members, 323 clusters (30-36 per district); HH7 is
the census's ten districts ("Baltistan" = Skardu, "Diamir" = Diamer). Item HC1B, mother tongue
of the household head, read as every member's (hl.sav x hhweight). HC1C (language usually
spoken at home) agrees in 96.6% of households and drifts to Urdu (5 Urdu heads, 40 Urdu homes),
so HC1B stays, as before, and it is the census's question.

**Check: the microdata is the Pamir Times table.** The unweighted HC1B household counts reproduce
the article's district table in all 60 cells, with Urdu folded into its "other languages"
(asserted in `pk_mics.py` against `pk_north.MICS17`). So the change is the weighting and the
switch from households to persons; the method in §3 is otherwise unchanged. Biggest moves in the
seed, unweighted households -> weighted persons (points): Nagar Shina 23.9 -> 18.3 and
Burushaski 76.0 -> 81.4, Hunza Burushaski 65.5 -> 70.1, Skardu Shina 20.3 -> 24.4, Gilgit Shina
75.5 -> 79.3, Ghizer Khowar 29.9 -> 33.5. The seed now lands closer to the census's GB totals
(Shina 49.3 against 50.2, before 48.2; Balti 29.4 against 29.9).

**"Other" cannot be split further.** The brief hoped the microdata would; it cannot. MICS5 GB
codes only Urdu, Shina, Balti, Burushaski, Khowar, Wakhi and other, and the files hold no other
language item (their `language` and `ethnicity` variables are recodes of HC1C to Shina / Balti /
Burushaski / other). Urdu is now seeded from its own answer as well as its part of "other".

**MICS 2024-25 is kept** for dividing the census's GB "Others" (17.74%) among Burushaski,
Khowar, Wakhi and a remainder. Its report gives households, so its shares are now turned into
persons with 2016-17's persons per household for each language: Shina 48.0 -> 48.3, Balti
29.2 -> 28.9, Burushaski 12.3 -> 12.1, Khowar 5.2 -> 5.5, Wakhi 1.0 -> 0.9, Other 4.2 -> 4.2.

**The two rounds agree.** GB-wide, weighted households, 2016-17 against 2024-25: Shina 46.6 /
48.0, Balti 30.4 / 29.2, Burushaski 12.6 / 12.3, Khowar 4.6 / 5.2, Wakhi 1.3 / 1.0, other
(with Urdu) 4.5 / 4.2. Nothing moved by more than 1.4 points in eight years, which is also the
check that the 2016-17 district pattern can stand for 2023.

Clusters: a group confined to 2% of a district's people is missed by all 30 clusters 55% of the
time, 5% 21% of the time. Domaaki (Hunza, Nagar) and scattered Gojri are below that, which is
why they stay in the remainder.

Raked shares (%), before -> after where the change is 0.3 points or more:

| district | Shina | Balti | Burushaski | Khowar | Wakhi | Pashto | Kohistani | Urdu | other |
|---|---|---|---|---|---|---|---|---|---|
| Astore | 99.8 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1 | 0.1 |
| Diamer | 87.6 -> 88.0 | 0.1 | 0.1 | 0 | 0 | 2.1 | 4.4 | 0.8 | 4.5 |
| Ghanche | 0 | 99.9 | 0 | 0 | 0 | 0 | 0 | 0.1 | 0 |
| Ghizer | 44.5 -> 41.3 | 0 | 17.4 -> 17.7 | 33.0 -> 35.6 | 0.7 | 1.3 | 0 | 0.5 | 2.8 |
| Gilgit | 80.3 -> 83.3 | 0.9 | 11.4 -> 8.3 | 2.9 | 0.5 | 1.1 | 0 | 0.6 | 2.3 |
| Hunza | 13.9 -> 12.2 | 0 | 69.0 -> 72.1 | 0.2 | 16.8 -> 15.6 | 0 | 0 | 0 | 0 |
| Kharmang | 13.8 -> 13.0 | 83.2 | 0 | 0 | 0 | 0.8 -> 1.1 | 0 | 0.4 | 1.6 -> 2.3 |
| Nagar | 27.8 -> 22.1 | 0.5 | 72.1 -> 77.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| Shigar | 0.2 | 99.6 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1 |
| Skardu | 21.5 -> 22.1 | 76.7 -> 77.0 | 0.2 | 0 | 0 | 0.2 | 0 | 0.1 | 0.8 -> 0.4 |

GB totals: Burushaski 181,043 -> 178,268, Khowar 76,540 -> 80,950, Wakhi 14,719 -> 13,206; Shina,
Balti, Pashto, Kohistani and Urdu are the census's and do not move. 30,666 people (1.8% of GB)
change language.

### 0.2 Khyber Pakhtunkhwa, MICS6 2019: names the census's OTHERS

23,740 households, 23,501 interviewed, 178,631 members, 1,187 clusters (30-65 per district);
HH7 is the 32 districts of 2019, the merged tribal districts included.

**Not a replacement.** The census asked everyone, at tehsil grain; MICS is a sample at district
grain. Where both name a language they roughly agree (Saraiki in Dera Ismail Khan 70% against
66%, Hindko in Kohat 11.5 against 12.4; Hindko in Haripur and Mansehra differs by 8-9 points,
eight years and a sample apart). MICS is used only where the census has nothing: OTHERS, 1,136,990 people in KP,
which until now was all drawn on `other`. Nothing in the build split it before this (the brief
guessed `pk_north*.py` might handle Chitral; it did not: Upper Chitral was 99.8% `other`).

**Items.** HC1B, the head's language, codes Pashto, Hindko, Saraiki, Urdu, English,
"Kohistani/Gujri" (one code) and other. HH16, the respondent's native language, adds
Khowar (Chitrali) and Shina. Inside HC1B-"other" households HH16 is Khowar 764, Shina 189, other
48 of 1,046 (Chitral's 807 are 95% Khowar). HH16 against HH15 (interview language) shows no slide
of the Iraq kind: 639 Hindko respondents interviewed in Urdu still answered Hindko, 419 Khowar
respondents were interviewed in "other". But HH16 does lose some Kohistani/Gujari: 95
Kohistani/Gujari-headed households answer HH16 Pashto (Swat, Shangla, Upper Dir, Kohistan),
against 35 the other way. So **HC1B, with its "other" split by HH16** into Khowar / Shina / still
other, as Iraq was done; weighted persons per district.

**Split, per MICS district** (census districts crosswalked to it, asserted both ways; Chitral and
Kohistan are one district each in MICS, two and three in the census): with census population P,
OTHERS O and KOHIOSTANI K, and MICS shares Khowar k, Kohistani/Gujari g and still-other o:
Khowar = kP, Kohistani/Gujari the census did not already name = max(0, gP - K), other = oP. If
these add to more than O they are scaled down to O; otherwise they are taken whole and the rest
of O stays OTHERS. Each tehsil's OTHERS is split in its MICS district's proportions, largest
remainder; tehsil totals are unchanged (asserted). The census's named counts are never touched.

**What the Kohistani/Gujari part is called depends on place** (spec §3, the Pahari rule). In
Hazara (Mansehra, Batagram, Abbottabad, Haripur, Torghar) it is **Gujari**: the census already
names Kohistani, and Gujari is the one other language of that code spoken there (the Kaghan
valley's Gujars; Balakot tehsil alone has 66,750 OTHERS). Elsewhere it goes on the Indo-Aryan
group node, drawn as "language not named": Swat's OTHERS is 86% in Behrain tehsil (141,119),
and Upper Dir's largest share in Sharingal (Dir Kohistan), where Torwali and Gawri, also called
Kohistani, live beside Gujars; the census's KOHIOSTANI caught only part of them (Behrain 32,154).
Nothing measured separates Torwali, Gawri and Gujari there.

**Then split by place, from knowledge** (same day, on Anita's word relayed by the supervisor: she
prefers splitting to lumping and is happy for us to decide from knowledge where no data separates
languages). The Indo-Aryan part is named by tehsil (`BY_PLACE` in `pk_mics.py`), using Glottolog
points and tehsil geography:

| tehsil | named as | people | why |
|---|---|---:|---|
| Behrain (Swat Kohistan) | Torwali 50%, Gawri 50% | 69,638 / 69,637 | Torwali (torw1241, Bahrain, Chail) and Gawri (kala1373, Kalam, Utror, Ushu); an even split, see below |
| Sharingal (Dir Kohistan: Kumrat, Thal, Kalkot) | Gawri | 20,932 | Dir Kohistani is Gawri; Kalkoti (kalk1245, one village) not split out, since nearness to its point would give it most of the tehsil |
| Bisham (Shangla) | Kohistani | 6,334 | on the Indus beside Lower Kohistan (Pattan): Indus Kohistani |
| Upper Chitral (Mastuj) | Indo-Aryan, unnamed | 5,031 | few Gujars, no Kohistani language; nothing to name it by |
| every other tehsil (lower Swat, Buner, Dir, Malakand, Shangla's Alpuri and Chakisar, Lower Chitral, Peshawar, Mardan...) | Gujari | 110,388 | the Gujars are the people of this code living there |

**Behrain's shares.** First drawn by nearness: each of Behrain's 227 Kontur hexes went to the
nearer Glottolog point and its people were summed, giving Torwali 85.8%, Gawri 14.2%. That
flatters Torwali, because Bahrain and Chail are lower and denser than Kalam. The supervisor then
asked for Joshua Project's speaker estimates by ISO code (trw, gwc), as `id_papua.py` and
`pg.md` use them, with Sharingal's Gawri subtracted. That file
(`data/raw/pg/joshuaproject_pgic.csv`, downloaded 2026-10-05, 773 Pakistan rows) has **no
Pakistan row for either code**. It files Swat and Dir Kohistan under "Pashtun Kohistani"
(142,000) and "Pashtun Dir Kohistan" (137,000), primary language Northern Pashto, and its only
Gawri row is Afghanistan's "Garwi, Kohistani" (2,200). Nothing else on disk gives speaker
numbers, so the supervisor's earlier fallback is used: **an even split**, floored so Gawri never
falls below the 14.2% that nearness alone gives the Kalam end (the floor does not bind).
Placement inside Behrain is by population for every language, as in every unit: nearness set
only the first shares and never placed dots. Putting Torwali in the south and Gawri in the north
would mean splitting Behrain into two units in `pk_hexes.gpkg`. New nodes
`dardic.torwali` and `dardic.gawri` in `tree.d/pk.txt`, hand-coloured (mid green, deep teal).

| MICS district (clusters) | census OTHERS | census Kohistani % | MICS Khowar % | MICS Koh./Gujari % | MICS other % | drawn now |
|---|---:|---:|---:|---:|---:|---|
| Chitral (45) | 474,149 (92.4%) | 0.3 | 82.2 | 2.7 | 6.0 | Khowar 422,088; Indo-Aryan 12,243; other 39,818 |
| Mansehra (40) | 206,893 (11.6%) | 2.5 | 0 | 19.7 | 0 | Gujari 206,893 |
| Swat (40) | 163,758 (6.1%) | 1.5 | 0 | 8.7 | 0.1 | Indo-Aryan 161,618; other 2,140 |
| Batagram (30) | 65,712 (11.9%) | 2.1 | 0 | 9.5 | 1.9 | Gujari 40,877; other 24,835 |
| Upper Dir (34) | 39,968 (3.7%) | 5.1 | 0 | 8.1 | 0 | Indo-Aryan 31,960; other 8,008 |
| Shangla (34) | 30,438 (3.4%) | 1.0 | 0 | 5.2 | 0 | Indo-Aryan 30,438 |
| Abbottabad (45) | 22,545 (1.6%) | 0.8 | 0 | 1.2 | 0.1 | Gujari 4,446; other 18,099 |
| Buner (32) | 18,239 (1.8%) | 0.1 | 0 | 1.8 | 0 | Indo-Aryan 17,507; other 732 |
| South Waziristan (30) | 16,406 (1.8%) | 0 | 0 | 0 | 0 | other 16,406 |
| Kohistan (50) | 15,365 (1.7%) | 88.5 | 0 | 71.3 | 0.3 | other 15,365 |

(The "drawn now" column shows the first split; the Indo-Aryan figures in it are named by place
below.) KP's OTHERS, 1,136,990 -> Khowar 422,088, Gujari 365,491 (255,103 in Hazara, 110,388
named by place), Torwali 69,638, Gawri 90,569, Kohistani 6,334, Indo-Aryan unnamed 5,031, other
177,839. Chitral is now 82.2% Khowar (Upper Chitral's 194,851 OTHERS -> 173,457 Khowar).
National: Khowar 503,038 (KP and GB), Gujari 1,169,929 (KP and AJK), `other` 3.41M -> 2.45M.

Left on `other`, unnamed by anything: South Waziristan's 16,406, nearly all in Ladha tehsil
(16,349), which holds Kaniguram, where Ormuri is spoken; Peshawar's 21,047 (Persian/Dari and
others; MICS's "other" there is 0.6%); Kohistan's (MICS finds no Gujari beyond the census's
Kohistani); and Chitral's 39,818 (Kalasha the census names apart; Wakhi, Yidgha, Palula, Dameli
and Persian are plausible, none named).

**Disagreements, left as the census has them.** In Kohistan MICS finds 17% Shina (HC1B "other"
-> HH16 Shina) against the census's 6.5%, and 71% Kohistani/Gujari against the census's 88%
Kohistani: Kohistani Shina speakers likely told the census "Kohistani". Mansehra's MICS
Kohistani/Gujari (19.7%) is above the census's Kohistani + OTHERS (14.1%); Upper Dir's (8.1%) is
a little below (8.8%). The census's own counts win everywhere it names a language.

## 1. Provinces and Islamabad: Table 11

`sources/pk_t11.py` reads PBS's Table 11 from the CRAN package PakPC2023 (its docstring has the
route) and writes `data/normalized/pk.csv` by tehsil, keyed by religiondots' Table 9 tehsil ids
(§9). Fourteen named languages plus OTHERS
(3.34M: Khowar, Burushaski, Wakhi, Gujari, Persian and more, on `other`; in KP split by MICS
since 2026-10-09, §0.2). `taxonomy/pk2023.py`.

## 2. What exists for Gilgit-Baltistan and Azad Kashmir (searched 2026-10-06)

- **PBS Table 11** covers only the four provinces and Islamabad; religiondots' `sources/pk.md`
  §10.3 lists 33 probed `table_*_{gb,ajk,...}` URLs, all 404.
- **GB at a Glance 2025** (GB P&DD Statistical & Research Cell, `pnd.gog.pk/pages/downloads`,
  saved as `data/raw/pk_north/gb_at_glance_2025.pdf`), p.9, citing "Census 2023, Pakistan Bureau
  of Statistics": **mother tongue for GB as a whole**, Shina 50.21, Balti 29.94, Pushto 0.86,
  Kohistani 0.86, Urdu 0.38, Others 17.74 (%). p.4: 2017 and 2023 census population by the ten
  districts (male, female), adding to the printed GB row (891,558 + 817,472 = 1,709,030;
  religiondots' 1,709,049 is a different print of the same total, 19 apart, likely transgender).
  No district split of mother tongue was found. The ten 2024 district brochures that might hold
  one are cut-off Wayback captures (religiondots §10.3).
- **GB MICS 2024-25** survey findings report (same downloads page,
  `gb_mics_2024_25_sfr.pdf`), Table SR.3.1, p.39: households by language of household head
  (HC1B asks the head's *mother tongue*), GB-wide: Shina 48.0, Balti 29.2, Brushaski 12.3, Khowar
  5.2, Wakhi 1.0, Other 4.2. Every other table uses language only as a row variable, never
  crossed with district.
- **GB MICS 2016-17** district x language: sampled households by language of household head for
  each of the ten districts, printed in Shah Zaman, "Treading the Sacred Linguistic Landscape of
  Gilgit-Baltistan", Pamir Times, 2023-12-23 (`pamirtimes_2023-12-23_linguistic_landscape.html`).
  6,213 households, 585-708 per district. Checks: each row adds to its printed total, the
  columns add to the printed totals row, and the districts add to the article's first table's
  three divisions (Baltistan 2,607, Diamer 1,190, Gilgit 2,416). MICS microdata (UNICEF) needs
  a registration, so the article's unweighted counts were used until 2026-10-09; since then the
  weighted microdata is (§0.1), which reproduces this table exactly.
- **AJK Statistical Year Book 2025** (`religiondots/data/raw/pk2023/`, read in place), Table
  15.31 "Languages Spoken in AJ&K", pdf p.226: percent by district, source "Kashmir Liberation
  Cell, Muzaffarabad". Whole percents, an administrative estimate, not a census count. No census
  mother-tongue table for AJK was found (the yearbook prints the 2023 census's religion, not its
  language).

## 3. Gilgit-Baltistan method

(As first built, 2026-10-06. Since 2026-10-09 the seed is the weighted microdata and MICS
2024-25 is turned into persons, §0.1; the figures and table below are the first build's.)

Seed: MICS 2016-17 district shares times census 2023 district population. Targets: the census's
GB-wide shares, with its Others (17.74%) divided among Burushaski, Khowar, Wakhi and a remainder
in MICS 2024-25's proportions (12.3 : 5.2 : 1.0 : 2.1, the 2.1 being MICS "Other" 4.2 less the
census's separately named Pashto, Kohistani and Urdu). MICS 2016-17's "Other languages" seed is
split among Pashto, Urdu and the remainder in those target proportions; **Kohistani is seeded in
Diamer only** (the one GB district on the Kohistan border; Kohistani Shina of Darel and Tangir).
Then IPF to the district populations and the GB targets; integer counts by largest remainder.

The seed already lands near the census, which is the check that the two sources describe the same
place: Shina 48.2 seed against 50.2 census, Balti 30.1 against 29.9, Burushaski 11.4 against
10.6, Khowar 4.1 against 4.5, Wakhi 1.3 against 0.9.

Raked shares (%), what the map draws:

| district | Shina | Balti | Burushaski | Khowar | Wakhi | Pashto | Kohistani | Urdu | other |
|---|---|---|---|---|---|---|---|---|---|
| Astore | 99.7 | 0 | 0 | 0 | 0 | 0.1 | 0 | 0 | 0.2 |
| Diamer | 87.6 | 0.2 | 0.1 | 0 | 0 | 2.2 | 4.4 | 1.0 | 4.6 |
| Ghanche | 0 | 99.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ghizer | 44.5 | 0.2 | 17.4 | 33.0 | 0.6 | 1.2 | 0 | 0.5 | 2.6 |
| Gilgit | 80.3 | 1.1 | 11.4 | 3.0 | 0.8 | 1.0 | 0 | 0.4 | 2.0 |
| Hunza | 13.9 | 0 | 69.0 | 0.2 | 16.8 | 0 | 0 | 0 | 0 |
| Kharmang | 13.8 | 83.5 | 0 | 0 | 0 | 0.8 | 0 | 0.3 | 1.6 |
| Nagar | 27.8 | 0.2 | 72.1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Shigar | 0.5 | 99.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1 |
| Skardu | 21.5 | 76.7 | 0.3 | 0.2 | 0 | 0.4 | 0 | 0.2 | 0.8 |

GB's remainder (1.8%) goes on `other`: Domaaki, Gojri, Kashmiri and whatever else MICS filed as
other, none of them named. Domaaki (Hunza and Nagar, a few hundred speakers) cannot be drawn.

## 4. Azad Kashmir method

Percent x the district's 2023 census population (religiondots' `pk.csv` AJK rows, which are the
yearbook's Table 15.24 and add to 4,333,467), largest remainder per district. Every row prints
100% except **Mirpur, 10 + 85 + 2 = 97**, scaled up. Labels, as the cells print them: Kashmiri,
Gojri, Pahari (with the variety named in seven districts), Shina (Neelum 5%), Kundal Shahi
(Neelum 2%), Dogri (Bhimber 30%, printed in the Shina column), Punjabi (Bhimber 35%, in the
Others column), Others (Sudhnoti, Kotli, Mirpur).

AJK totals drawn: Pahari-Pothwari 2,981,612 (69%), Gojri 804,438, Kashmiri 205,687, Punjabi
153,648, Dogri 131,698, other 40,878, Shina 11,076, Kundal Shahi 4,430.

Doubts, kept as printed: Kundal Shahi's 2% of Neelum is 4,430 people where linguists report a few
hundred speakers; Bhimber's 30% Dogri is high for the Pakistani side. The table is the only
district source and is drawn as the government prints it; `note_public` says it is an estimate.

## 5. Calls

- **In `pk`, not separate entries.** religiondots draws AJK inside `pk` (its `countries/pk.py`)
  and Natural Earth's Pakistan outline holds both areas, so not_drawn.py's hatching for them
  goes once `pk` covers them.
- **All Pahari varieties on one node, `pahari_pothwari`.** Table 15.31 is one Pahari column with
  the local variety written beside the figure; Glottolog files Chibhali, Punchhi, Pothwari, Mirpur
  Panjabi and Pahari as dialects of Pahari Potwari (paha1251); India's J&K Pahari is on the same
  node (in2011.py), so the colour runs across the Line of Control. Splitting would need five
  near-identical nodes from a whole-percent estimate.
- **Khowar under Dardic** as conventionally grouped (Glottolog: directly Indo-Aryan).
  **Burushaski** under `isolate`. **Kundal Shahi** under Dardic (Glottolog Shinaic).
- **GB's census totals win over MICS** where they differ; MICS only splits the census.
- **Colours** (`taxonomy/tree.d/pk.txt`): Burushaski hand-picked purple (#9f50ca), Khowar dark
  blue-teal (#116d8a), against Shina teal #33a6a0, Balti blue #488acb, Wakhi salmon #d08a7b.

## 6. Geography (`sources/pk_north_geo.py` -> `data/geo/pk/pk_district_hexes.gpkg`)

Since 2026-10-07 this is the district stage; `sources/pk_tehsil_geo.py` re-keys it to tehsils
and writes the place layer `pk_hexes.gpkg` (§9). religiondots' pk2023 layer (146 districts, AJK included) copied, with GB appended: COD-AB v01's 14
GB districts folded to the census's 10 (Darel, Tangir into Diamer, COD's "Diamir"; Gupis-Yasin
into Ghizer; Rondu into Skardu), asserted both ways; Kontur 2023-11 hexes by centroid. None of
GB's 5,839 hexes was already in religiondots' layer.

Kontur under-reads GB: 1,150,007 against 1,709,030 (0.67), Gilgit 0.44 and Diamer 0.54, the rest
0.60-1.10; log correlation 0.92 against a best shuffle of 0.88 (ten units, a weak control). It only
moves dots inside a district, so the counts are unaffected.

**Line of Control.** India's 1:1,000 dots inside GB 2 and inside AJK 14; China's 0. But 105 of
Pakistan's AJK dots (of 4,333) fall inside India's placement polygons near Poonch and Rajouri
(lat 32.75-34.03): India's layer overruns the LoC a little there, onto AJK hexes that religiondots
already drew. It is India's layer's edge, not a gap; not changed here.

## 7. Room for improvement

- PBS's district Table 11 for GB and AJK, if it is ever printed (or the GB 2024 district
  brochures, if a full copy turns up), would replace both models with measured counts.
- ~~GB MICS 2016-17 microdata would give weighted district shares and could split "other"
  further~~: done 2026-10-09 (§0.1); weighted shares yes, "other" no (no finer item). The GB
  MICS 2024-25 microdata, if released, would be the more recent seed; it has the same codes.
- KP: a source separating Torwali, Gawri and Gujari in Swat and Dir (MICS codes Kohistani and
  Gujari as one answer), and Ormuri in Ladha, would name the rest of KP's OTHERS.
- A boundary layer of the 2023 tehsils (PBS's census maps, or the provinces' own notifications
  traced) would split the 71 grouped units of §9, most of them in Balochistan, Lahore, Peshawar
  and Karachi.

## 8. Cut from note_public (2026-10-06 text sweep)

- Pahari varieties as Table 15.31 names them: Punchi in Poonch and Sudhnoti, Mirpuri in Mirpur
  and Bhimber, Chibali in Haveli, Dhundi-Khairali in Bagh; all on `pahari_pothwari`, the node
  India's J&K Pahari uses (§5).
- The yearbook gives the AJK table's source as the Kashmir Liberation Cell (§2).

## 9. Tehsil grain (2026-10-07, session `fix-pk`)

Until now Table 11's 591 tehsils were summed to 136 districts, which blurred the Punjabi/Saraiki
line, Pashto/Hindko in Hazara, Sindhi/Balochi/Brahui and Karachi. Measured on the table itself:
drawing each tehsil at its district's mix puts **18.9M** people under a language other than
their tehsil's own mix; at the new units it is **6.1M** (scratch `fix-pk/gain.py`, half the L1
distance between tehsil and unit shares, times population).

**Counts.** `pk_t11.py` writes tehsil rows with religiondots' Table 9 tehsil ids. The pairing
uses two keys, both asserted: the tehsil's total population (unique within its district in all
136) and its folded name. All 591 pair 1:1, so every tehsil total of Table 11 equals Table 9's
(a separate transcription, from PBS's PDFs), as do the 136 district totals. Table 1's
populations are not used as a check: Table 1 counts people "counted by head only" and is higher.

**Polygons** (`sources/pk_tehsil_geo.py`). No layer has the 2023 tehsils. COD-AB v01 ADM3 has 521
in the four provinces and Islamabad, each inside exactly one census district (religiondots'
districts are COD tehsils dissolved). Karachi's COD ADM3 is the 2001 towns, so Karachi uses
OSM's 21 admin_level=7 towns (2023 layout, ODbL), fetched one relation at a time from
polygons.openstreetmap.fr (Overpass gave 504/500 from three endpoints), clipped to religiondots'
Karachi; each town's district is asserted against the subarea members of OSM's district
relations. Names pair inside each district (fold, substring or ratio 0.75, global greedy), 498 of
591, 7 through `ALIAS` (Bori = Loralai, Golarchi = Shaheed Fazil Rahu, Mirwah = Thari Mirwah,
"AI" = Allai, Malakand's two sub-divisions as religiondots pairs them, the two de-excluded
tribal areas). 83 districts pair off one to one.

**Placement.** religiondots' district hexes are re-keyed, each to the polygon of its own district
its centroid falls in (18 hexes, 2,806 people, to the nearest). GB and AJK are untouched.

**Grouping.** A tehsil is drawn alone when its polygon's Kontur/census is within 1.5x of its
district's ratio and, in districts that gained tehsils since COD, the polygon's area is within 2x
of Table 1's printed area. Everything else in the district is one unit, which takes further
pairs (best first) until its own ratio is within 1.5x. Result: **372 tehsils alone, 51 groups,
20 whole districts** (443 units). Why the area test: Lahore City is 214 km2 in the census and
670 in COD, the old parent, and its Kontur ratio happened to pass; Lahore is now one unit.
Why the band applies to one-to-one districts too: 35 pairs there were out of band, among them
Kohat's Gumbat (0.08) beside Lachi (1.79), Hyderabad's City (1.97) / Latifabad (0.31) /
Qasimabad (0.37), Karachi's Orangi (0.24) and Baldia (0.24) beside Mauripur (3.30). Kontur's
error and a moved line look the same here, so they are grouped.

**Checks** (printed by the script): Kontur/census per unit, normalised, p10 0.66, median 1.02,
p90 1.25; the 372 single-tehsil units' log correlation with census r = 0.978, against a best of
0.831 over 500 shuffles of census population **within each district** (the null a right district
with wrong tehsils would pass). Units never cross a district line (asserted).

Whole districts: 20, of which four have one tehsil anyway (Barkhan, Sherani, Islamabad, Upper
Chitral). The rest are Lahore, Karachi South, Malir, Quetta, Tank, Kolai-Palas, and ten in
Balochistan where sub-tehsils were carved after 2022 (Sohbatpur, Duki, Surab, Zhob...). Peshawar
is two units (Hassan Khel alone; the six tehsils that replaced COD's four towns together).
`data/geo/pk/pk_tehsil_report.txt` lists every unit.

Calls: Gadap, Bin Qasim and the rest of Malir are one unit (OSM's three towns against six
sub-divisions). The band and area thresholds were set before reading the results; the switch to
applying the band everywhere was made after the first run, on the list above.
