# Nigeria: MICS 2021 and Afrobarometer R4-R9, by state, placed by LGA

**Since 2026-10-09 (session 32a047f0) the nine languages MICS names are drawn from MICS6 2021
microdata** (§0); the Afrobarometer build below now writes `data/normalized/ng_afro.csv`, kept as
the comparison, and still supplies the split of MICS's "other language", English, Pidgin's spread
and the placement inside states.

```
python sources/ng_afro.py           # ng_afro.csv (comparison) and ng_lga.csv (placement)
python sources/ng_mics.py           # ng.csv, the drawn counts
python tools/check_country.py ng
python scatter.py --country ng
```

## 0. MICS6 2021 (`sources/ng_mics.py`, 2026-10-09)

Anita made a UNICEF MICS account and downloaded Nigeria's MICS6 2021 (with the National
Immunization Coverage Survey; NBS and UNICEF); the SPSS files are in `data/raw/ng/mics_2021/`
(gitignored: research use, no redistribution). 41,532 households, 39,632 interviewed, 207,496
members, 2,079 clusters, all 37 states (38 to 78 clusters, 745 to 1,477 households a state). The
published tables are national only, which is why §1 had it under "not used".

**Item: HC1B, language of the household head**, read as every member's (members x hhweight).
Nine named answers plus "other language": Hausa, Igbo, Yoruba, Fulani, Kanuri, Ijaw, Tiv,
Ibibio, Edo. Three items were compared household by household:

| | HC1B head | HH16 respondent | WM14 women 15-49 |
|---|---:|---:|---:|
| Hausa | 28.15 | 30.67 | 27.53 |
| Yoruba | 16.41 | 16.97 | 17.70 |
| Igbo | 14.23 | 14.11 | 15.29 |
| Fulani | 7.17 | 5.44 | 4.98 |
| Tiv | 2.23 | 2.15 | 2.38 |
| Kanuri | 2.14 | 1.98 | 1.69 |
| Ibibio / Ijaw / Edo | 1.91 / 1.33 / 1.24 | 1.91 / 1.35 / 1.19 | 1.94 / 1.36 / 1.29 |
| other | 25.20 | 24.22 | 25.84 |

(% of persons nationally, the survey's own weights.) HC1B and HH16 differ in 2,609 households.
In 24,072 the respondent is the head, so both describe one person, and they still differ in
1,419, following the interview language: heads interviewed in Hausa give HH16 Hausa though their
language is Kanuri 60 times in 335, Fulani 299 in 1,315; heads interviewed in Yoruba give HH16
Yoruba though theirs is Igbo (23 of 56), Hausa (18 of 54), Fulani (20 of 82), Tiv (8 of 21), Edo
(9 of 16); in English interviews the two agree almost always (52 of 4,446 non-Hausa heads say
Hausa, 21 of 4,361 say Yoruba). HH16 slides to the interview language as Iraq's did, so HC1B is
drawn. HC1B equals the head's ethnic group (HC2) in 96% of households, but is answered as
language: 246 Edo and 215 Ijaw heads by ethnicity give "other language" (Esan, Yekhee, Kalabari,
Okrika...).

**Fulfulde is the call most worth reversing.** HC1B is the high reading (7.2% against HH16 5.4%,
WM14 5.0%): Bauchi 29.7 / 18.6 / 19.3, Gombe 49.8 / 36.6 / 34.0, Kano 17.2 / 12.8 / 12.5, Kebbi
13.8 / 8.4 / 8.9. Of the 1,545 heads who answered both items themselves with HC1B Fulani, 322 gave
HH16 Hausa, nearly all in Hausa interviews; some of those may really be first-language Hausa
speakers of Fulani descent. Drawn at HC1B for one rule across the nine; `note_public` says it
may run high. The Afrobarometer had Fula at 3.7%: its Kano Fulani are 1.7%, while MICS finds
Fulani households in 34 of Kano's 50 clusters, eight of them entirely Fulani (rural settlements
the Afrobarometer's sample seems to miss).

**How MICS enters, per state.**
1. The nine named shares as measured. Each is the Afrobarometer answer of the same name only,
   checked state by state: Rivers Ijaw MICS 3.0% against the Afrobarometer's Ijaw 3.6% (its
   Kalabari, Okrika, Nembe and Ibani are another 26%); Akwa Ibom Ibibio 54 against 68 (Anaang and
   Efik 25 more); Cross River Ibibio 12 against 9 (Efik 40); Delta Igbo 19 against 25 (Ika and
   Ukwuani 13 more); Delta Edo 1.7 against 1.9 (Urhobo and Isoko 39). Across the 37 states the
   two surveys agree on where each is (Pearson r: Hausa 0.97, Yoruba 1.00, Igbo 1.00, Ijaw,
   Ibibio, Edo, Tiv 0.99, Kanuri 0.91, Fulani 0.80).
2. MICS's "other language" split across the Afrobarometer's other answers in the state, with 8
   pseudo-respondents on "Other Nigerian language" (`K_OTHER`): weight = (respondents naming it
   + 8 if it is the unnamed remainder) / (all other-answer respondents + 8). Where the
   Afrobarometer has many such respondents (Rivers 359, Benue 198, Plateau 179, Kogi 176) its
   split stands; where it has a handful against a real MICS share (Oyo 3 respondents against
   6.9%, Ekiti 3 against 12.1%, Ondo 13 against 10.0%) most of the share is drawn unnamed rather
   than multiplying three people. The southwest's "other" households are spread over a third of
   the clusters, mostly rural, mostly Christian, with an ethnic group that is also "other": read
   as migrants of other groups, not Yoruba naming a dialect.
3. English (Afrobarometer R7 mother tongue, shrunk as before) and Pidgin (Ethnologue's 4.7M,
   spread by the R7-R9 Pidgin-at-home answers) keep their levels, since HC1B offers neither;
   everything else scaled to what is left.
4. §2a's "Hausa outside Hausaland" step is dropped: HC1B is itself a mother-tongue item, with
   ~1,000 households a state against R7's 16-128 respondents. Hausa rises where that step had cut
   it (FCT 4.7 -> 20.2%, Plateau 5.1 -> 12.9%, Yobe 20 -> 33%, Taraba 16 -> 27%), and all three
   MICS items agree there (FCT HC1B 21.5, HH16 22.0, WM14 18.4).

**Weights.** hhweight varies up to 200-fold inside a state. In Ebonyi three urban clusters carry
87% of the state's weighted people (about 4 effective clusters of 46; Anambra 7, Bayelsa 11, Imo
11), and Ebonyi's 7.4% Hausa is three households in two of them (7 of its 10 Hausa-coded heads
give Igbo as ethnic group and respondent's language). Each household weight is capped at 5 times
its state's median. That moves a named share by more than a point in four states: Ebonyi Hausa
7.4 -> 2.6%, Igbo 92.6 -> 97.3; Bayelsa Ijaw 61.8 -> 71.3%; Kaduna and Kwara about a point. In
Ebonyi and Bayelsa it moves MICS towards the Afrobarometer (Igbo 97.5, Ijaw 72.6). The cost:
Bayelsa's urban share by weight falls from 29 to 9%. Every state then has 22+ effective clusters.

**Zeros.** A group living in enclaves holding a share p of a state is missed by every sampled
cluster with probability (1 - p)^clusters: in the smallest state (Sokoto, 38 clusters) an
enclave of 2% is missed 46% of the time, 5% 14%; at the median 56 clusters, 32% and 6%. Two cells
where the Afrobarometer had 1%+ and MICS none: Benue Kanuri 6.9% (an Afrobarometer oddity, not
Kanuri country) and Bayelsa Edo 1.3%. Drawn as MICS says.

**Before (Afrobarometer build) -> after, % of each state's people:**

| state | |
|---|---|
| Kano | Hausa 94.8 -> 77.4, Fulfulde 1.7 -> 17.0, Kanuri 0.5 -> 1.6 |
| Bauchi | Hausa 68.5 -> 45.3, Fulfulde 14.8 -> 29.3, Kanuri 1.3 -> 6.3, Zaar 4.3 -> 6.6 |
| Jigawa | Hausa 92.6 -> 72.5, Fulfulde 2.1 -> 18.2, Kanuri 1.4 -> 7.0 |
| Gombe | Fulfulde 38.2 -> 48.7, Hausa 13.3 -> 17.7, Tangale 16.1 -> 10.0, Waja 13.2 -> 8.2 |
| Borno | Kanuri 38.2 -> 36.3, Bura-Pabir 17.7 -> 15.9, Hausa 14.0 -> 15.7, Fulfulde 5.2 -> 7.8 |
| Yobe | Hausa 20.4 -> 32.6, Kanuri 8.5 -> 21.5, Fulfulde 17.7 -> 20.4, Karekare 28.3 -> 11.3, Bade 15.8 -> 6.3 |
| Adamawa | Fulfulde 39.7 -> 16.9, Hausa 11.5 -> 17.2, Kamwe 7.4 -> 10.2, unnamed 2.3 -> 10.0 |
| Benue | Tiv 36.4 -> 63.3, Idoma 32.7 -> 17.8, Igede 6.4 -> 3.5, Kanuri 6.7 -> 0 |
| Plateau | Tarok 20.6 -> 18.6, Berom 14.8 -> 13.3, Hausa 5.1 -> 12.9 |
| FCT | Hausa 4.7 -> 20.2, Yoruba 15.7 -> 8.3, Igbo 16.9 -> 6.4, Gbagyi 9.1 -> 9.0 |
| Lagos | Yoruba 67.5 -> 55.7, Igbo 14.6 -> 22.2 |
| Oyo / Ekiti | Yoruba 94.3 -> 85.1 / 95.1 -> 80.9; unnamed 0 -> 5.0 / 0.1 -> 8.9 |
| Delta | Urhobo 30.0 -> 32.6, Igbo 23.7 -> 18.0, Ijaw 12.4 -> 8.7 |
| Akwa Ibom | Ibibio 58.3 -> 46.7, Anaang 12.6 -> 16.6, Efik 8.9 -> 11.7 |
| Rivers | within half a point everywhere |

Nationally: Hausa 29.6 -> 27.7%, Yoruba 18.3 -> 15.8%, Igbo 13.7 -> 13.4%, Fulfulde 3.7 -> 7.1%,
other Nigerian (unnamed) 1.8 -> 3.8%, Tiv 1.2 -> 2.1%, Kanuri 1.7 -> 2.1%, Ibibio 1.7 -> 1.7%,
Nupe 1.8 -> 1.6%, Ijaw 1.3 -> 1.3%, Edo 1.3 -> 1.1%, Idoma 1.4 -> 0.9%; English 1.8% and Pidgin
2.2% unchanged. 119 answers, as before; total unchanged (216,798,930, COD-PS 2022).

Against REACH's household surveys (language used, so more Hausa expected): Yobe Kanuri 21.5% vs
18 (was 8.5), Hausa 33 vs 53 (was 20); Borno Kanuri 36 vs 44; Adamawa Fulfulde 17 vs 24 (was 40).

Calls someone might reverse: HC1B rather than HH16 (Fulfulde 7.1% vs ~5.4%); the 5x weight
trim; K_OTHER = 8; dropping the Hausa-outside-Hausaland step; English and Pidgin taken pro rata
from every language rather than from MICS's "other" alone.

Drawn 2026-10-05 (session edd42a8c-ng). 216,798,930 people (COD-PS 2022), 37 states, 119
nodes from 120 answers, every row `modelled`. 216,747 dots at 1:1000, no rings. Inside each
state, each language's dots lean towards the LGAs where the survey's own respondents named it.

```
python sources/ng_afro.py --fetch   # Nigeria's rows from religiondots' six merged .sav files
python sources/ng_place.py          # religiondots' hexes + each hex's LGA
python taxonomy/build.py
python tools/check_country.py ng
python scatter.py --country ng
```

## 1. What exists, and why this source

- **No census asks.** Nigeria's last census with an ethnicity or language question was 1963;
  1973 was annulled, 1991 and 2006 asked neither, 2023 has not been held
  (`religiondots/sources/ng.md` §1). The USCB HDX geodatabase carries none.
- **Afrobarometer, rounds 4-9 (2008-2022), 11,923 respondents**, asks every respondent their
  home language, with region (state) codes in every round and LGA labels in four. Open: the
  merged files are plain downloads, the use policy asks only for a citation. religiondots
  already holds all six (`religiondots/data/raw/afrobarometer/`, read-only here) and draws
  Nigeria's religion from the same pool. Question: R4-R6 "Which language is your home
  language?" (variable label "Language of respondent"), R7-R9 "Language spoken in home". One
  answer; a card of Nigeria's main languages plus "Other (specify)", whose free text is in the
  release.
- **CLEAR Global's HDX `nigeria-languages`** (CC BY-SA) was the lead. Its admin1 file is
  Afrobarometer R9 alone (about 43 respondents a state) in 31 states and REACH's household
  surveys in Adamawa, Borno, Yobe (MSNA 2021) and Katsina, Sokoto, Zamfara (2022); its admin2
  is REACH, those six states only. Used as a check (§4), not drawn: pooling six rounds gives
  seven times the sample in every state, and one source for all 37.
- **Not used:** DHS (language of interview only, and its microdata needs an institutional
  registration); MICS 2021 (ethnicity of household head, national tables; used since
  2026-10-09 from its microdata, §0); GHS-Panel/LSMS
  (needs a World Bank account; not tried, since the Afrobarometer covers every state).
- **Population**: COD-PS 2022 state totals (NPC/UNFPA projection off the disputed 2006 census),
  religiondots' `ng_lookup.csv`, so both maps stand on the same base.

## 2. How the counts are made (`sources/ng_afro.py`)

Each state's count of a language = the weighted share of that state's pooled respondents who
named it, times the state's COD-PS population. Weights are each round's within-country weight
(`Withinwt`/`withinwt`, `withinwt_hh` from R8; each asserted to average 1). 8 non-answers
dropped. Respondents per state: 118 (Yobe) to 940 (Lagos), median 280. Round 6 has no
Adamawa, Borno or Yobe (the insurgency).

**English (the call most worth reversing).** English was not on the R4 and R6 cards (0
answers), 29 in R5, then 166, 199 and 120 in R7-R9. Nearly every English answer came with an
English-language interview (153 of 166, 180 of 199, 116 of 120), 40% from rural respondents.
As drawn, that would put 47% of Cross River, 34% of the FCT and 21 million Nigerians on English
as a first language. So, following Anita's El Salvador and Ecuador ruling on learned second
languages, **the 514 English answers are drawn on the language of the respondent's own ethnic
group** (the survey's ethnicity question, coded label or free text): 510 moved (Igbo 145,
Yoruba 71, Efik 27, Tiv 20, Hausa 20, other Nigerian 28, ...), 4 who gave no group stay on
English (122,000 people). Flagged to the supervisor: English is a real first language for
some urban Nigerians, and this call is Anita's to reverse.

**Nigerian Pidgin is drawn as measured**, a real first language in the Delta and Rivers. Its
answers are 1, 8, 6 in R4-R6 and 23, 45, 39 in R7-R9, so each state's Pidgin share comes from
R7-R9 only (63-940 respondents a state, median 112), and every other answer's share from all
six rounds among the rest, scaled to what is left. 3.9 million, 1.8%; Bayelsa 10%, Rivers 10%,
Akwa Ibom 8.5%.

**R4's combined label.** R4 coded 172 respondents "Ijaw/Kalabari/Okirika/Andoni/Ogoni/Nembe";
53 have a free-text answer (Okirika 34, Andoni 12, Kalabari 7), read from it. The other 119
(Bayelsa 81, Rivers 22, Delta 10, Lagos 4, Osun 1, Adamawa 1) are shared across the six in
proportion to those answers' weighted counts in the same state elsewhere in the pool; Osun and
Adamawa had none, so the national split.

**Free text.** Every "Other (specify)" answer (838 rows, 345 spellings) is in `VERBATIM`: a
spelling, dialect or town is put on its language ("Agbor" Ika, "Auchi" Yekhee, "Kwale"
Ukwuani, "Calabar" Efik, "Effon" Yoruba, "Mupun" Mwaghavul); a place holding several
languages (Ogoja, Obubra, Ikom), a former state ("Bendel"), or nothing identifiable goes on
"Other Nigerian language". A language named in free text by one respondent only (3 of them)
also goes there. Two coded labels whose free text says otherwise are read from the text:
"Okpe" in Edo (11 wrote "Okpella", drawn as Okpela; 3 in Delta wrote Okpe). Two coded labels
are one language under two names and are merged: Babur and Bura (Bura-Pabir), Kataf and
Kataf (Atyap) (Tyap); "Yakhor" (16, all Cross River) is Yakurr, not Yekhee.

**Superseded 2026-10-05 for English, Pidgin and Hausa: see §2a.** The English and Pidgin moves
above still run first, so their respondents' own languages carry the rest of each state.

As drawn, nationally: Hausa 29.7%, Yoruba 18.5%, Igbo 14.3%, Fula 3.7%, Ibibio 1.9%, other
Nigerian languages 1.8%, English 1.8%, Nupe 1.8%, Kanuri 1.7%, Ijaw 1.4%, Idoma 1.4%, Edo 1.4%,
Tiv 1.3%, Igala 1.2%, Efik 1.2%, Urhobo 1.0%. Pidgin 0. (Since 2026-10-06, §2b: Hausa 29.6%,
Yoruba 18.3%, Igbo 13.7%, Fula 3.7%, Pidgin 2.2%; 216,739 dots.)

## 2a. Lingua francas at R7's mother tongue (2026-10-05, ask 018)

Anita's ruling: lingua francas are drawn at Afrobarometer R7's separate **mother tongue**
question (Q2A), which R7 asks beside "language spoken in home" (Q2B). `state_shares` in
`sources/ng_afro.py`; the R7 reader and the shrink rule are `r7_mother` and `shrink` in
`sources/wafr_afro.py`. R7 has 16-128 respondents a state, so each state's own Q2A share is
shrunk towards a prior by 50 respondents: (Q2A answers + 50 x prior) / (respondents + 50).

- **English**: Q2A 1.86% nationally (28 respondents, 25 in English interviews); prior the
  national share. Drawn 3.9M (1.8%): Cross River 13%, Adamawa 5%, Lagos 5%, Akwa Ibom 4%, about
  1% elsewhere. Before: 122k (0.06%), every answer moved to the ethnic language.
- **Pidgin**: no R7 respondent names it as mother tongue (0 of 1,600; the 41 who speak it at
  home give Igbo, Edo, Ijaw, Esan, "other"). Drawn 0, its 122 pooled answers moved to the
  respondent's ethnic group's language like English's (8 new group spellings in ETH_VERBATIM).
  Before: 3.9M (1.8%), Bayelsa 10%, Rivers 10%, Akwa Ibom 8.5%. **Superseded 2026-10-06, §2b.**
- **Hausa outside Hausaland** (superseded 2026-10-09: MICS's head's-language share is drawn
  instead, §0; the step still runs in `ng_afro.csv`) (29 states where R7's Q2A Hausa share is under 50%): Q2A share,
  shrunk towards the pooled share x 0.37, the Q2A/Q2B ratio of those states' R7 respondents.
  Hausaland's 8 states (Bauchi, Jigawa, Kaduna, Kano, Katsina, Kebbi, Sokoto, Zamfara) keep
  the pooled share. Hausa 32.8% -> 29.7% nationally: Borno 30 -> 14%, Adamawa 35 -> 11%, Gombe
  40 -> 13%, Yobe 37 -> 20%, Niger 40 -> 23%, FCT 17 -> 5%, Nasarawa 18 -> 8%, Plateau 13 -> 5%.
  This widens the gap to REACH's household surveys (Borno 14 vs 28%, Yobe 20 vs 53%), which
  ask the language used, not the first.
- Every other answer is scaled to what is left (Fula 3.2 -> 3.7%, Kanuri 1.5 -> 1.7%).

## 2b. Nigerian Pidgin from a published first-language estimate (2026-10-06)

Anita, 2026-10-06: "ok we can add nigerian and cameroonian pidgin". R7's mother-tongue question
drew it at 0, though it is the first language of many in the southern cities. Drawn by the ask
019 route (a cited estimate placed by a stated rule), rows still `modelled`.

- **Level: 4.7 million** (`PIDGIN_L1` in `sources/ng_afro.py`), Ethnologue 26th ed. (2023), L1
  users 4.7M (2020 figure), via Wikipedia's infobox; ethnologue.com itself returns 403. Applied to
  the 2022 base unchanged. Other figures seen: Faraclas (APiCS survey chapter 17) speaks of
  "millions" of first-language speakers "in and around ... Warri and Sapele", without a number;
  Ihemere (2007, *A Tri-Generational Study of Language Choice and Shift in Port Harcourt*) found
  Ikwerre families shifting to Pidgin in the youngest generation, the Port Harcourt pattern.
- **Across states**: each state's weighted share of R7-R9 respondents answering Pidgin at home,
  read before those answers are moved to the ethnic language (`pidgin_pattern`), times its
  population, scaled by 1.193 so the total is 4.7M. The home answers stand in for where L1
  speakers live; the ratio of L1 to home use is assumed the same everywhere.
  Drawn: Bayelsa 12.2%, Rivers 11.5%, Akwa Ibom 10.1%, Abia 8.7%, Edo 6.9%, Imo 6.5%, Delta
  5.3%, FCT 4.8%, Benue 3.4%, Cross River 3.2%, Taraba 3.1%, Lagos 2.6%; none in the states with
  no Pidgin answer (most of the north). 2.2% nationally.
- **Taken from the others proportionally**: in each state every other answer is scaled to the
  1 - Pidgin left (the same step as English).
- **Inside a state, in the cities** (`_pidgin_share` in `countries/ng.py`): Pidgin's count goes
  on the hexes of 5,000+ people per km² as a flat share of their people, at most 50%; anything
  over that spreads flat over the state's other hexes. Every other language's placement weight
  is multiplied by (1 - Pidgin's share of the hex), so each hex still draws its own people. 5,000
  is mine: at 1,500 (GHSL's urban-centre density) Kontur's hexes hold 50-90% of every southern
  state's people and the rule would not concentrate anything.
- **Check, as drawn** (Pidgin dots within ~16 km of the centre): Yenagoa 15%, Uyo 13%, Aba 12%,
  Benin City 11%, Warri 9%, Port Harcourt 9%, Lagos 3%, Kano 0. 4,699 dots.
- **Weak point**: the survey puts Delta (Warri, Sapele, the heartland in Faraclas) below Abia
  and Imo, since its Delta respondents are mostly rural Urhobo and Isoko. Left as the survey
  says rather than hand-weighting a state.
- **Ghana, checked only lightly**: Ghanaian Pidgin English has about 2,000 L1 speakers (Ethnologue
  2011, via Wikipedia) and APiCS's survey chapter calls it a language with no L1 speakers. Two
  dots at most; not drawn.

## 3. Placement inside states (`sources/ng_place.py`, `countries/ng.py`)

religiondots' Kontur 400 m hexes (635,561), each given the COD-AB LGA its centroid falls in (14
put on the nearest LGA of their own state; all 774 hit; population religiondots' to the
person). Inside a state a language's dots go to each hex by Kontur population times that
language's share in the hex's LGA from the survey's own respondents: in a sampled LGA,
(respondents naming it + 8 x the state share) / (respondents + 8), 8 being one enumeration
area; an unsampled LGA borrows the inverse-square-distance mean of the 3 nearest sampled LGAs
of its state; a language no LGA-labelled respondent in the state named goes on population
(166 of 580 state-language rows). A placement weight only; the state counts do not move.

- **LGA labels.** R4 (`DISTRICT`), R6 and R9 (`LOCATION.LEVEL.1`); R5 and R8 have none.
  6,547 respondents in 427 LGAs. 42 spellings joined to COD-AB by hand (`LGA_ALIAS`, each
  read: "Obio/Akpor" is COD-AB's "Obia/Akpor", "Atisbo" its "Atigbo", "AMAC" Abuja
  Municipal), 5 left unplaced as ambiguous (Obioma Ngwa, Uquo Ibeno(Esit Eket, Akoko, Tai
  Eleme, R6's Gboko under Niger).
- **R7's LGA labels are shifted by one state and are not used.** 336 of its 1,600 respondents
  carry an LGA of the alphabetically previous state (Ebonyi respondents in Delta's Warri South,
  Katsina's in Kano's Rano and Wudil) and answered as their REGION, not their LGA, says (all 8
  "Warri South" said Igbo). A trap for anyone using that column.
- **Check, as placed** (share of the language's dots in the state, against the LGA's share of
  population): Berom in Barikin Ladi 26% (pop 6%), Jos South 22%, Riyom 13%; Tarok Langtang
  North 17% (5%); Ngas Pankshin 24% (6%); Tyap Zango Kataf 23% (4%); Kanuri Jere, Bama,
  Maiduguri; Bura-Pabir Biu 16% (4%), Hawul 12% (3%); Kalabari Asari-Toru 24% (4%); Ogoni
  Gokana 39% (5%), Khana; Nupe Edu 28% (8%), Pategi 18% (5%); Idoma Oturkpo; Mumuye Zing 19%
  (6%); Ebira Okene 24% (12%). One oddity: Jju lands 30% in Lere (northern Kaduna), from R4/R6
  respondents there; Jju's home is Kaura and Zango Kataf.

## 4. Checks

| check | result |
|---|---|
| extract | 2,324 / 2,400 / 2,400 / 1,600 / 1,599 / 1,600 Nigerian respondents; question and ethnicity variable labels asserted per round; weights average 1.000 |
| states | every REGION spelling (case, hyphens, Nassarawa, three FCTs) -> the 37 COD-AB states, both ways |
| drawn total | 216,798,930 = COD-PS 2022 |
| split-half, R4-R6 vs R7-R9, r across 37 states | Hausa +0.956, Yoruba +0.984, Igbo +0.996, Ibibio +0.993, Ijaw +0.999, Idoma +0.988, Tiv +0.986, Nupe +0.996, Kanuri +0.938, Igala +0.982, Urhobo +0.969, Efik +0.816, Fula +0.788, other Nigerian +0.234 |
| CLEAR / REACH (big languages, drawn vs REACH) | Adamawa Hausa 35/37%, Fula 31/24%; Borno Hausa 30/28%, Kanuri 31/44%; Katsina Hausa 91/96%; Sokoto 89/97%; Zamfara 91/98%; **Yobe Hausa 37/53%, Kanuri 7/18%** |
| CLEAR R9-only states | far off where R9 alone kept English (Imo Igbo 93 vs 60) or sampled differently (Niger Hausa 40 vs 73); expected, R9 is a seventh of this pool |

Yobe is the weakest state: 118 respondents in five rounds, Karekare 23% and Bade 13% where
REACH has Kanuri far higher. Drawn as the pool says.

## 5. Mapping and tree (`taxonomy/ng2022.py`, `taxonomy/tree.d/ng.txt`)

Every family and branch checked against Glottolog; the fragment's header lists the glottocodes.
New groups under Niger-Congo: Ijoid, Cross River, Plateau, Kainji, Jukunoid, Bantoid (not
Bantu), each with a colour. Volta-Niger members flat, as us.txt has Yoruba and Igbo. Waja and
Tula under Adamawa although Glottolog files them under Gur. "Gwoza" (coded in R5, an LGA of
several Chadic languages) on `afroasiatic.chadic`, drawn as unnamed Chadic. Poland's
`ibibio_efik` left apart from the new Ibibio and Efik.

**Colours.** Hausa, Edoid and Edo were generated and Nigeria's new siblings took their slots,
moving them and 16 small Mande and Gur nodes in Burkina Faso and Côte d'Ivoire; Hausa, Edoid
and Edo are pinned in ng.txt to their old colours, and Busa and Bariba hand-picked off the Mande
and Gur grids, so no node outside Nigeria moved (diffed against a build without ng.txt). About 30
Nigerian nodes hand-picked for neighbours. Closest remaining pairs at 1%+ in one state (OKLab):
Gwandara/Igbo 0.041 (Nasarawa), Edo/Igala 0.042 (FCT), Gade/Igala 0.043, Fula/Yoruba 0.048
(Kwara; both colours belong to other countries).

## 6. Calls someone might reverse

- English and Hausa outside Hausaland at R7's mother-tongue share (§2a, Anita's ruling);
  the 50-respondent shrink and the 50% Hausaland line are mine.
- Pidgin at Ethnologue's 4.7M, spread by R7-R9 home answers and into hexes of 5,000+/km² with a
  50% cap (§2b); the density line and the cap are mine.
- Pooling 2008-2022 rather than taking the latest round (R9 has 43 respondents a state).
- A free-text language named by one respondent on "Other Nigerian language".
- Adult survey shares applied to whole state populations, children included.

## 7. Room for improvement

A census language table would replace all of this. Short of that: the NDHS household
microdata (language of respondent, 40,000 households, state-representative) needs an
institutional DHS registration; the GHS-Panel (LSMS) needs a free World Bank account and was
not tried; REACH's MSNA microdata could sharpen the six north-eastern and north-western states.

## Terms

Afrobarometer data: free download, citation requested ("Afrobarometer Data, Nigeria, Rounds
4-9, 2008-2022, available at http://www.afrobarometer.org"). CLEAR Global CC BY-SA 4.0, used
only as a check. COD-AB and COD-PS (OCHA) CC BY-IGO; Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Yoruba is in a 'Yoruba and Ede' group with Benin's Nago, Idaasha and the other Ede languages. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
