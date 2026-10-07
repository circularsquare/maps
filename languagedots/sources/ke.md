# Kenya: Afrobarometer R4 and R6-R9 home language, by county

Drawn 2026-10-05 (session edd42a8c-ke). 47,213,282 people (KNBS 2019 county totals), 47
counties, 33 nodes, every row `modelled`. 47,197 dots at 1:1000, no rings. Placed on Kontur
population inside each county (religiondots' layer, read-only).

```
python sources/ke_afro.py --fetch   # Kenya's rows from religiondots' merged .sav files
python taxonomy/build.py
python tools/check_country.py ke
python scatter.py --country ke
```

## 1. Two routes, and why the survey

- **(a) Census ethnicity under the ethnicity rule.** The 2019 KPHC has no language question.
  Ethnicity is in Volume IV Table 2.31 (pp. 423-424 printed, PDF index 435-436; religiondots
  holds the PDF, `religiondots/data/raw/ke/kphc2019_volume4.pdf`) **for the nation only**:
  ~45 groups and sub-groups, no county or sub-county table anywhere in the four volumes
  (religiondots' and the coverage sweep's reading, confirmed: the only page carrying the
  47,564,296 total is Table 2.31). KNBS stopped publishing ethnicity below national after
  1989. So route (a) is one unit for 47.6 million people, plus a retention guess for every
  group.
- **(b) Afrobarometer home language by county.** Five rounds carry a county (or a district
  inside one), every county sampled: 9,900 respondents, 28 (Lamu) to 1,032 (Nairobi), median
  184. Chosen: 47 units against 1, and the language is asked rather than inferred.
- The census table is kept as a check (§4): it agrees with the drawn languages to within 1.2
  points for every group over 2%.

## 2. How the counts are made (`sources/ke_afro.py`)

Each county's count of a language = the weighted share of that county's pooled respondents who
named it, times the county's KNBS 2019 population (the conventional household population,
Volume IV Table 2.30's universe, religiondots' `data/normalized/ke.csv`: 47,213,282 against a
census 47,564,296; the 351,014 in institutions, travelling or sleeping outdoors are outside
it). Weights: each round's within-country weight, asserted to average 1. 17 non-answers
dropped.

**Rounds and counties.** R4 2008 (`DISTRICT`, 1999-2009 districts, each joined to the 2010
county holding it in `DISTRICT_COUNTY`: Thika and Kiambu to Kiambu, Buret to Kericho, Trans
Mara to Narok, Mt Elgon to Bungoma, Teso to Busia...), R6 2014 and R7 2016
(`LOCATION.LEVEL.1`, county), R8 2019 and R9 2021 (`REGION`, county). **R5 (2011) has only
the eight old provinces and is not used** (2,399 respondents). Nigeria's R7 location column
was shifted by one state; here every R4/R6/R7 county is asserted to lie in the province its
REGION names, and all do. R9 also carries 294 ward names (8 respondents each), not used.

**Swahili and English (the call most worth reversing).** The question changed between R6 and
R7: R4-R6 "Which language is your home language?", R7-R9 "Language spoken in home". Swahili
was on the card throughout:

| round | 4 (2008) | 5 (2011) | 6 (2014) | 7 (2016) | 8 (2019) | 9 (2021) |
|---|---|---|---|---|---|---|
| Swahili | 1.4% | 0.7% | 3.6% | 28.8% | 27.8% | 27.1% |
| English | 0.3% | 0.4% | 0.5% | 1.1% | 4.4% | 3.4% |

Under the R4/R6 wording 97-98% of Kikuyu, Luo and Kalenjin respondents name their group's
language; under R7-R9's, 64-79%, the rest mostly Swahili. Nothing on the ground moved eightfold
in two years, so the jump is the wording: R4-R6 reads as a first language, R7-R9 as the language
the household uses, often Swahili in a mixed or town household. R7-R9 Swahili is not an
interview artefact (a third of it came in English-language interviews), and it has a real
geography: Mombasa 74%, Trans Nzoia 63%, Nairobi 46-60%, Kajiado 53%, Nakuru 47%, against
3-6% in Makueni, Kitui, Homa Bay, Bomet.

So, for a first-language map: **Swahili's and English's share in each county comes from R4 and
R6** (`SE_ROUNDS`; 8 to 336 respondents a county, median 64); in R7-R9, a Swahili or English
answer from a respondent who names a Kenyan ethnic group is drawn on that group's language
(1,882 of 2,003: Luhya 418, Kikuyu 386, Luo 197, Kalenjin 193, Kamba 167, Mijikenda 147); 121
who named no group with a language of its own ("other", national identity only, Swahili,
missing) are left out of the shares; every other answer's share comes from all five rounds
among the rest, scaled to what Swahili and English leave. As drawn: Swahili 1.27 million
(2.7%), English 237,000 (0.5%); Nairobi Swahili 4.0%, Mombasa 7.6%.

Flagged to the supervisor: Swahili as a first language is real in Nairobi and Mombasa, and for
the children of the R7-R9 Swahili households (counted here at their parents' answer) it may
often be the first language. The adult R4/R6 answer is the conservative reading. Setting
`SE_ROUNDS = [7, 8, 9]` reverses it in one line and puts Swahili near 28% nationally.

**Combined card labels.** "Meru/Embu" (617 answers) and "Masai/Samburu" (270) each name two
languages. Split by the respondent's county: Embu county -> Embu; Meru, Tharaka-Nithi, Isiolo
-> Meru; Narok, Kajiado -> Maasai; Samburu, Marsabit, Isiolo -> Samburu (527 and 234 answers);
anywhere else shared by the census's national ethnic counts (Meru 83 : Embu 17, Maasai 78 :
Samburu 22). Tharaka-Nithi's Chuka and Tharaka speakers who answered "Meru/Embu" are drawn as
Meru: the survey cannot tell them apart.

**Clusters.** The card's Luhya, Kalenjin and Mijikenda each name a cluster (Glottolog Luyia,
Kalenjin, Mijikenda subgroups). Drawn as one language each; the few free-text answers naming a
variety (Maragoli, Bukusu, Bunyore, Marachi, Tiriki, Samia, Wanga, Idakho; Digo, Duruma,
Giriama, Kauma, Chonyi; Nandi, Kipsigis, Tugen) are counted in the cluster. R8's coded
"Giriama" (21) likewise goes to Mijikenda. Pokot, Sabaot and Okiek, which the survey names
apart, are drawn apart.

**Free text** (`VERBATIM`): Gabra and Boran to Borana (Glottolog files Gabra as a Borana
dialect); Wardei to Orma (the census files Wardei under Orma); Ajuran to Somali (a Somali clan);
"Kirinyaga" to Kikuyu. South Asian answers without a language ("Indian", "Hindu", "Asian
South") and Punjabi on `other`. One-off or unidentifiable answers (Bokom, Shelshel, Munyaya,
Watta, Malakote, Kisagalla, Nyarwanda, Nyasa, and Nubi, named once) on "Other African
language", 90,000 people.

As drawn, nationally: Kikuyu 17.6%, Luhya 13.2%, Kalenjin 11.4%, Luo 11.2%, Kamba 10.2%, Gusii
5.8%, Somali 5.6%, Mijikenda 4.9%, Meru 4.6%, Swahili 2.7%, Turkana 2.4%, Maasai 2.4%, Pokot
1.3%, Embu 1.3%, Borana 1.0%. 18 more under 1%. (Before §2a; Swahili is now 1.6%.)

## 2a. Swahili and English at R7's mother tongue (2026-10-05, session edd42a8c-r7e)

Anita's ruling on ask 018: lingua francas are drawn at Afrobarometer R7's separate **mother
tongue** question (Q2A, asked beside R7's "language spoken in home"). `SE_SOURCE = "r7_mother"`
in `sources/ke_afro.py` now takes Swahili's and English's share from R7 Q2A (1,582 answers,
cached in `data/raw/ke/ab_ke_r7_mother.csv`) **by old province**: 31 Swahili and 7 English
answers are too few for 47 counties, and the province's share is given to each of its counties.
Everything else is unchanged (the other languages still pool five rounds, R7-R9's Swahili and
English home answers still go on the ethnic group's language). `SE_SOURCE = "r46"` puts back
the R4/R6 county shares.

| | before (R4/R6, by county) | after (R7 Q2A, by province) |
|---|---|---|
| Swahili, national | 2.7% (1.27M) | 1.58% (745,000) |
| English, national | 0.5% (237,000) | 0.29% (138,000) |
| Swahili, Nairobi | 4.0% | 4.8% |
| Swahili, Mombasa (Coast) | 7.6% | 3.8% |

R7 Q2A by province, Swahili: Nairobi 4.8%, Coast 3.8%, Rift Valley 2.3%, Central 0.9%, Western
0.7%, Eastern, Nyanza and North Eastern 0. The census check (§4) now has every group over 2%
within 0.7 points of its ethnic share. Swahili as drawn (1.58%) still sits above the census's
Swahili ethnic group (0.12%), as it should for a language many non-Swahili people now have
first.

## 3. Placement

religiondots' `ke_hexes.gpkg` (Kontur 2023, 230,139 hexes over the 47 counties; its unit is
the KNBS county code via `ke_lookup.csv`). Population weight only: nothing here says where in a
county each language's speakers live. 5 Kontur cap blocks, already registered as real cores.

## 4. Checks

| check | result |
|---|---|
| extract | 1,104 / 2,397 / 1,599 / 2,400 / 2,400 respondents (R4, R6-R9); question and ethnicity variable labels asserted; weights average 1.000 |
| location | every label a county; every R4/R6/R7 county in its REGION's province; all 47 sampled |
| drawn total | 47,213,282 = KNBS Table 2.30 universe |
| split-half, R4+R6 vs R7-R9, r across the 39 counties with 40+ respondents in each half | Kikuyu +0.98, Luhya +0.98, Kalenjin +0.98, Luo +0.995, Kamba +0.99, Gusii +0.99, Meru +0.99, Mijikenda +0.998, Somali +0.99, Turkana +0.997, Maasai +0.96. Across all counties Somali is +0.86: Isiolo, Marsabit, Tana River swing between Somali and Borana as one round's eight interviews land in one village or another |
| census ethnicity, national (Table 2.31) | drawn language - ethnic share: Kikuyu +0.5, Luhya -1.2, Kalenjin (less Pokot, Sabaot, Okiek) +0.4, Luo +0.6, Kamba +0.4, Somali -0.3, Gusii +0.1, Mijikenda -0.3, Meru +0.4, Maasai -0.1, Turkana +0.3, Pokot -0.3 points |

The census check doubles as the retention check the ethnicity rule asks for: the big groups'
languages are drawn at their ethnic shares, so near-full retention, as the R4/R6 cross-tab
(97-98%) says. Where the drawn language falls well short of the ethnic group, the survey says
the group mostly speaks another: Suba 0.05% against 0.33% ethnic (the Suba largely speak Luo),
Sabaot 0.11% against 0.62% (most answered Kalenjin), Teso 0.45% against 0.88%, Pokomo, Bajuni.
Embu's 1.27% matches Embu and Mbeere together (1.26%).

## 5. Mapping and tree (`taxonomy/ke2022.py`, `taxonomy/tree.d/ke.txt`)

Families and branches checked against Glottolog; the fragment's header lists the glottocodes.
New leaves: 11 Bantu (Luhya, Gusii, Mijikenda, Meru, Embu, Taita, Kuria, Suba, Pokomo, Bajuni,
Sheng), 7 Nilotic (Kalenjin, Pokot, Sabaot, Okiek, Samburu, Turkana, Teso), 3 Lowland East
Cushitic (Orma, Garre, Rendille). Borana on the existing `oromo`. Sheng has no Glottolog entry
and sits under Bantu beside Swahili (4 respondents). Bajuni, a Swahili dialect in Glottolog, is
a sibling of Swahili, not a child, so Swahili stays a leaf.

**Colours.** Every Kenyan node hand-picked off its parent's generated grid, Gikuyu and Kamba
pinned to their old colours: a build without ke.txt against one with it shows no node outside
Kenya moved.

## 6. Calls someone might reverse

- Swahili and English from R7's mother-tongue question by province (§2a, Anita's ruling);
  R7-R9's home answers on the ethnic group's language (§2). `SE_SOURCE` reverses it.
- Pooling 2008-2022; R5 left out (provinces only).
- Meru/Embu and Masai/Samburu split by county.
- Luhya, Kalenjin, Mijikenda as single languages.
- Adult survey shares applied to whole county populations, children included.

## 7. Room for improvement

A census language question would replace all of this; Kenya has never asked one. Short of
that: a county ethnicity table (KNBS holds it; not released since 1989), the KDHS
(language of interview only, registration needed), or IPUMS's 2009/2019 samples (no
ethnicity variable for Kenya, per the coverage sweep). Inside counties, R9's ward names and
R4's districts could lean each language's dots towards where its respondents were, as Nigeria
does with LGAs; it needs ward polygons and was not done.

## Terms

Afrobarometer data: free download, citation requested ("Afrobarometer Data, Kenya, Rounds 4,
6-9, 2008-2022, available at http://www.afrobarometer.org"). KNBS 2019 KPHC: free publication.
Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Kalenjin is now a group: the census's 'Kalenjin' draws on 'Kalenjin (variety not given)', beside Pokot, Sabaot and Uganda's Kupsabiny. Turkana and Teso are in an Ateker group; Somali is a group with Somalia's Benaadir and Maay. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
