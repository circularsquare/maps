# Cameroon: Afrobarometer R5-R9 home language, by unit, placed by department

Drawn 2026-10-05 (session edd42a8c-cm). 29,442,318 people (COD-PS 2025), 12 units (ten regions,
Yaoundé/Mfoundi and Douala/Wouri apart), 95 answers on 95 nodes, every row `modelled`. 29,396
dots at 1:1000, no rings. Inside each unit, each language's dots lean towards the departments
where the survey's own respondents named it.

```
python sources/cm_afro.py --fetch   # Cameroon's rows from religiondots' five merged .sav files
python sources/cm_place.py          # religiondots' hexes + each hex's department
python taxonomy/build.py
python tools/check_country.py cm
python scatter.py --country cm
```

## 1. What exists, and why this source

- **No census asks.** The 2005 RGPH (3rd) published no language or ethnicity table; religiondots
  (`religiondots/sources/cm.md`) found religion only nationally. The 4th RGPH was taken in 2026,
  nothing out yet.
- **CLEAR Global `cameroon-languages` (HDX)**: the coverage sweep found it labelled "2005 census"
  but holding data for only 39 departments, French or English as the main language, and ~75
  mostly empty local-language columns: official-language data, not first language. Not used.
- **WVS**: Cameroon is not in it. **DHS 2011, 2018**: language of interview only in the open
  reports; microdata registration is off.
- **Afrobarometer, rounds 5-9 (2013-2022), 5,984 respondents** (1,182 to 1,202 a round), the
  same pool religiondots draws Cameroon's religion from (`religiondots/data/raw/afrobarometer/`,
  read-only). Region in every round, department (`LOCATION.LEVEL.1`) in R6, R7 and R9. Cameroon
  is not in R4. Free download, citation requested.
- **Population**: religiondots' `cm_lookup.csv` (COD-PS 2025, BUCREP projection), so both maps
  stand on the same base.

## 2. How the counts are made (`sources/cm_afro.py`)

Each unit's count of a language = the weighted share of that unit's pooled respondents who named
it, times the unit's COD-PS population. Weights `withinwt` / `withinwt_hh`, asserted to average
1. Region labels decoded by name (religiondots' NORM; R8's codes are shifted against the other
rounds); every R6, R7, R9 department lies in the unit its region label names (asserted, 3,584
respondents). Respondents per unit 208 (Sud) to 1,030 (Extrême-Nord), median 492.

**Lingua francas (ask 018; the call most worth reversing).** The wording changes between R6 and
R7 ("home language" -> "language spoken in home") and the answers move with it, weighted % of
all respondents:

| | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|
| French | 16.6 | 15.6 | 30.4 | 37.9 | 49.2 |
| English | 0.9 | 0.5 | 7.2 | 11.2 | 16.4 |
| Cameroonian Pidgin | 0.9 | 0.5 | 6.2 | 3.8 | 0.3 |
| Fulfulde | 13.4 | 10.8 | 21.0 | 13.4 | 11.6 |

In R9 95% of Nord-Ouest and 100% of Sud-Ouest answered English, which is the language used, not
a first language. So, as Tanzania and Kenya, the four lingua francas' shares per unit come from
**`LF_ROUNDS = [5, 6]`** and every other language's from all five rounds among the non-lingua-
franca answers, scaled to what the four leave. `[5, 6, 7, 8, 9]` draws everything as answered.
That reading drew French 16.9% (Douala 57%, Littoral 35%, Yaoundé 26%, Sud 21%), Fulfulde 13.0%
(Nord 41%, Adamaoua 40%, Extrême-Nord 27%), English 0.6%, Pidgin 0.6% (Littoral 9%). **Replaced
2026-10-05 by R7's mother-tongue question (§2a).**

### 2a. Lingua francas at R7's mother tongue (2026-10-05, ask 018)

Anita's ruling: lingua francas are drawn at Afrobarometer R7's separate **mother tongue**
question (Q2A). `LF_ROUNDS = "R7Q2A"`, `r7_targets` in `sources/cm_afro.py` (reader and shrink
in `sources/wafr_afro.py`). Each unit's Q2A share (40-216 R7 respondents a unit) is shrunk by 50
respondents towards the national Q2A share (French, English, Pidgin) or, for Fulfulde, towards
the unit's R5-R6 share x the national Q2A / R5-R6 ratio. Departments keep their R5-R6 pattern,
scaled to the unit's new level (placement only). Every other answer is scaled to what is left.

| | before (R5-R6) | after (R7 Q2A) |
|---|---|---|
| French | 4.99M, 16.9% (Douala 57%, Yaoundé 26%) | 566k, 1.9% (Extrême-Nord 5%, Douala 0.5%, Yaoundé 1.8%) |
| Fulfulde | 3.84M, 13.0% (Nord 41%, Adamaoua 40%, Extrême-Nord 27%) | 3.30M, 11.2% (Adamaoua 35%, Extrême-Nord 30%, Nord 25%) |
| English | 182k, 0.6% | 339k, 1.2% (Nord-Ouest 4%, Yaoundé 3%) |
| Pidgin | 184k, 0.6% (Littoral 9%) | 237k, 0.8% (Sud-Ouest 3%, Nord-Ouest 3%) |

All 20 French mother-tongue answers came in French interviews. Douala's French falls from 57% to
0.5% (none of Wouri's 136 R7 respondents named it): Njeck's 32% of Yaoundé teenagers with French
only is now far above the map, as is CEFAN's ~5% for Pidgin.

Second sources, both weak: Echu (2004, *Linguistik Online* 18) cites Njeck (1992): 32% of
Yaoundé's 10-17 year olds spoke no Cameroonian language, French being their only one; drawn
26% French in Yaoundé, of all ages. Leclerc's CEFAN page (axl.cefan.ulaval.ca/afrique/cameroun.htm)
puts Pidgin as a first language at about 5% of Cameroonians, unsourced; drawn 0.6%, so Pidgin is
probably under-drawn here. In Yaoundé R5 gave French 46% and R6 8% (the R6 interviewers took
more Ewondo and Eton answers); pooled 26%.

**Pidgin's row above is superseded by §2b (2026-10-06).**

### 2b. Cameroon Pidgin from a published first-language estimate (2026-10-06)

Anita, 2026-10-06: "ok we can add nigerian and cameroonian pidgin". R7's mother tongue drew it
at 0.8%, far under both published figures. Drawn by the ask 019 route (a cited estimate placed by
a stated rule), rows still `modelled`.

- **Level: 5% of the population**, 1,472,116 people (`PIDGIN_L1_SHARE` in `sources/cm_afro.py`).
  Neba, Chibaka and Atindogbé (2006), "Cameroon Pidgin English (CPE) as a tool for empowerment
  and national development", *African Study Monographs* 27(2): 39-61: about 5% of Cameroonians
  native speakers (about 70% speak it in some form). Leclerc's CEFAN page gives the same 5%,
  unsourced. 2006's share applied to the 2025 base.
- **Across units**: each unit's pooled R5-R9 weighted Pidgin-at-home share times its population,
  scaled by 2.26 to the 5%. Drawn: Nord-Ouest 28.6%, Sud-Ouest 27.9%, Littoral 10.8%, Nord 1.4%,
  Douala 1.2%, Adamaoua, Sud, Ouest and Yaoundé under 0.5%; none elsewhere. The 2.26 means L1 is
  drawn above the home answers, which the survey's wording swings (R5-R6 had no Pidgin answer in
  either Anglophone region, R7 37-40%) make hard to read; the 5% is what sets the level.
- **Taken from the others proportionally**: the unit's other answers scaled to 1 - Pidgin, as
  for the other lingua francas.
- **Inside a unit, in the towns** (`_pidgin_share` in `countries/cm.py`): the same rule as
  Nigeria's (sources/ng.md §2b): hexes of 5,000+ people per km² get a flat Pidgin share up to
  50%, the rest spread flat over the unit's other hexes, every other language's weight times (1 -
  that share). The department shares of the other languages are renormalised without Pidgin's
  department share first, so Pidgin is not taken out twice.
- **Check, as drawn** (Pidgin dots within ~16 km): Buea 38%, Limbe 38%, Kumba 36%, Bamenda 31%,
  Douala 1%, Yaoundé 0. 1,472 dots.
- **Weak point**: Douala. Its Anglophone quarters (Bonabéri, New Bell) are known for Pidgin, but
  the survey heard it at home from only 3 of Wouri's respondents, so Douala gets 1.2%.

A side effect: in Nord-Ouest and Sud-Ouest most R7-R9 respondents answered English or Pidgin,
so the local-language mix there rests mostly on R5-R6 (about 200 respondents each).

**Card labels.** The Bamileke languages are on the card under their chief towns: Bandjoun is
drawn as Ghomala', Bafang Fe'fe', Dschang Yemba, Bagangté Medumba, Bangwa Ngwe; "Mbouda" kept as
its own leaf (the town's chiefdoms speak three languages). Merged as one language under two names:
Bulu/Bula, Moudan/Moudang (Mundang), Guider/Guidar, Banso/Lamnso', the three Fulfulde spellings.
"Fong" (R5-R6, all 13 respondents ethnic Beti) read as Fang. **Split by place**: "Yamba" is
Yamba in Nord-Ouest and Yemba elsewhere (Menoua, ethnic Bamileke); "Mboum" is Limbum in
Nord-Ouest (Donga-Mantung, ethnic Wimbum) and Mbum elsewhere.

**Free text** (about 600 answers, 330 spellings, all in `VERBATIM`): a Bamileke or Grassfields
chiefdom is read as its language (Baham, Bafoussam, Bamendjou, Batié Ghomala'; Haut-Nkam
chiefdoms Fe'fe'; Batcham Ngiemboon; Babadjou Ngombale; Batibo Moghamo; Bali Nyonga Mungaka;
Balikumbat and Bali Gashu Mubako). 14 chiefdoms not placed on one language go on the Bamileke
group, Bamenda/Lebialem/Widikum on Grassfields, Sawa/Mbamois on Bantu. "Plusieurs langues" (3)
dropped as a non-answer. 13 languages named by one respondent, and unidentifiable words, go on
"Other Cameroonian language" (`africa_other`), as are R5-R6's "Others" with blank text (77).

## 3. Checks

| check | result |
|---|---|
| extract | 1,200 / 1,182 / 1,202 / 1,200 / 1,200; question and ethnicity labels asserted per round; weights average 1.000 |
| units | every REGION label -> one of 12 units; R6/R7/R9 departments all in their unit |
| drawn total | 29,442,318 = COD-PS 2025 |
| split-half, R5-R6 vs R7-R9, r across 12 units | Ewondo +0.97, Mafa +0.995, Bamileke +0.84, Bamun +0.82, Eton +0.91, Tupuri +0.88, Basaa +0.86, Bulu +0.99, Gbaya +0.99, Mundang +0.99, Kapsiki +0.996, Lamnso' +0.996; weaker: Ghomala' +0.48, Fe'fe' +0.75, Medumba -0.08, Hausa +0.25 |
| Massa | +0.675, not asserted: Extrême-Nord share 1, 11, 3, 24, 1% by round (which Mayo-Danay areas were sampled) |
| departments | 3,579 respondents in 55 of 58 departments; Boyo Kom 70%, Bui Lamnso' 67%, Donga-Mantung Limbum 41%, Menchum Esimbi 56%, Mayo-Tsanaga Mafa 51%, Mayo-Danay Massa 31% Tupuri 26%, Logone-et-Chari Kotoko 24% |
| colours | no node outside Cameroon moved (diffed against a build without cm.txt) |

## 4. Calls someone might reverse

- Lingua francas from R7's mother tongue (`LF_ROUNDS = "R7Q2A"`, §2a; Anita's ruling); the
  50-respondent shrink and Fulfulde's regional prior are mine.
- Pidgin at 5% (Neba et al. 2006), spread by the pooled home answers, which puts it at 28% of
  Nord-Ouest and Sud-Ouest and 1% of Douala; into hexes of 5,000+/km², capped at 50% (§2b).
- Bakundu drawn 396,000 (1.3%), all from 20 R5-R6 respondents in Meme; far above any estimate of
  its speakers. Left as the survey says, said in `note_public`.
- Bamileke chief-town labels read as languages; Yamba and Mboum split by region.
- Adult survey shares applied to whole populations; pooling 2013-2022.
- Placement: department shares from R6, R7, R9 shrunk with K = 8 (Nigeria's), 3 unsampled
  departments borrow from the nearest 3.

## 5. Tree and colours

`taxonomy/tree.d/cm.txt` lists the Glottolog codes. New groups: Grassfields (under Bantoid) with
Bamileke inside it. New Bantu, Chadic, Adamawa and other Bantoid leaves are hand-coloured off the
generated grids (cd.md §6's method), Grassfields members partly generated, partly hand-picked
for NW neighbours. Closest remaining neighbour pairs (OKLab, ≥3% in one department): Ghomala'
/Ngombale 0.058 (Bamboutos), Limbum/Meta' 0.059 (Fako), Mbouda/Yemba 0.059 (Mifi).

## 6. Room for improvement

A census language table (the 2026 RGPH, if it asks) would replace all of this. Short of that:
DHS 2018 microdata (native language, registration); the ELAN/OIF or Ethnologue L1 figures for
French in the cities; REACH's Far North and NW/SW assessments on HDX for those regions.

## Terms

Afrobarometer: free download, citation requested ("Afrobarometer Data, Cameroon, Rounds 5-9,
2013-2022, available at http://www.afrobarometer.org"). COD-AB, COD-PS (OCHA) CC BY-IGO; Kontur
CC BY 4.0; Glottolog CC BY.
