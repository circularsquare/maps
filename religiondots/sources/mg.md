# Madagascar — religion by region from the pooled Afrobarometer, on the 2018 census

**Drawn 2026-09-14** (session `d743fc47-mg`), through `COMMANDS.txt` step 9; the supervisor runs
the build tail. 22 units, 12 categories, 25,674,196 people, every row `modelled`. Drawn on
Anita's Nigeria ruling (`ask/answered/010-ng`) and her Madagascar ruling of 2026-09-14 night
("check whether the Afrobarometer card offers a separate ancestral or traditional answer, then
apply the draft no-religion procedure"). No ask filed; §5 is the place a reviewer should look.

- `sources/mg_geo.py` -> `data/geo/mg/mg_regions.gpkg`, `mg_lookup.csv`, `mg_districts.csv`
  (COD-AB of 2026-08-13, ADM1 and ADM2, 24 regions dissolved to 22; RGPH-3 Tome 1 Tableau 6,
  re-read from the PDF on every run)
- `sources/mg_grid.py` -> `data/geo/mg/mg_hexes.gpkg` (Kontur 400 m, 307,762 hexes)
- `sources/mg.py` -> `data/normalized/mg.csv` (the shared Afrobarometer `.sav` files)
- `taxonomy/mg2018.py`; one new node, `other.mg` (`taxonomy/branches.py`); `countries/mg.py`
- `kontur_cap.csv`: seven Madagascar rows (§8)

```
python sources/mg_geo.py --fetch
python sources/mg_grid.py --fetch
python sources/afrobarometer.py --fetch   # only if data/raw/afrobarometer/ is empty
python sources/mg.py
```

## 1. What Madagascar publishes: no religion count

| source | what it has | date checked |
|---|---|---|
| RGPH-2 1993, RGPH-3 2018 household forms | **no religion item** (the scout read both; the 2018 form's fonts shift character codes, so a text search finds nothing, `SEXE` included) | §11aq |
| INSTAT, whole-domain Wayback CDX (28,503 URLs) | no religion file; `instat.mg` itself is a Cloudflare challenge | §11aq |
| UNSD Demographic Yearbook (oracle) | **ABSENT** | 2026-09-14 |
| EDSMD-IV 2008-09 (DHS FR236), EDSMD-V 2021 (DHS FR376) | ask religion with separate traditional and no-religion boxes; the reports print national shares of respondents aged 15-49 only (Tableau 3.2, PDF p.63; Tableau 3.1, PDF p.82); microdata DHS-gated. **Used as the witness** (§6) | 2026-09-14 |
| MICS 2012, South (World Bank microdata catalog 3484) | household head's religion, `Chretien`, `Autres`, `Sans religion` (52.1% of 3,130 heads), no traditional box; the south only. Not used | 2026-09-14 |
| MICS 2018 | not checked for religion: the UNICEF report links returned a 4,570-byte bot page and the guessed S3 path a 403 | 2026-09-14 |
| COD-PS Madagascar (`cod-ps-mdg`) | labelled 2018 but, by its HDX caveat, the 2009 BNGRC population grown forward; **not used** | 2026-09-14 |

## 2. The construction

    row margin      region populations     RGPH-3 2018, Tableau 6          EXACT
    the composition each unit's own mix    Afrobarometer R5, R6, R7, R9    measured, n=4,788
    the national level                     neither                         computed

Nothing is fitted to a column margin (`sources/ng.md` §3).

### The units: 22, from a file with 24

The census and every round before 2022 have 22 regions. COD-AB's 2026 edition has 24: Vatovavy
(MG26) and Fitovinany (MG27) apart, and Ambatosoa (MG34: Mananara-Avaratra, Maroantsetra). MG27 is
dissolved into MG26 and MG34 into Analanjirofo (MG32). The witness is the census's own table of
urban communes by region (Tome 1, PDF p.46), which lists Maroantsetra and Mananara-Avaratra under
Analanjirofo and Manakara Atsimo, Mananjary, Ifanadiana, Ikongo and Vohipeno under Vatovavy
Fitovinany; `mg_geo.py::check_dissolve` reads it. The older 2018-10-31 COD-AB zip returns 404 on
HDX. The name join is 22 of 22 with one spelling difference (`Amoron I Mania`). Tableau 6's urban
and rural columns close on every row.

### Round 4 is left out, and Madagascar is not in round 8

Round 4 (June-July 2008) is cut by the six old provinces. It carries `DISTRICT` for all 1,326
respondents, 85 districts, which would place them in 21 of the 22 regions; **no Betsiboka district
was sampled**, and `cab.stability` refuses an empty (round, unit) cell. Tanzania's round 5 is the
precedent. Round 8's merged file has 34 countries and no Madagascar.

### Round 9 re-cuts two regions and moves the codes

Round 9 labels `Vatovavy` and `Fitovinany` apart, `Matsiatra Ambony` for Haute Matsiatra, and
`DIANA`; codes 433-435 name different regions than in rounds 5-7. Decoded by label only. Round 6
spells `Vatovavy Fitonany`.

### The checks

- **Labels against locations.** Rounds 6 and 7 carry a district in `LOCATION.LEVEL.1`: 1,134 of
  1,198 and 1,134 of 1,197 respondents' districts match a COD-AB name, and **0** disagree with the
  region label. Round 9 carries the old province there: 1,196 of 1,196 agree.
- **Held-out, per round**: R5 +0.984, R6 +0.982, R7 +0.979, R9 +0.991; pooled +0.991; 0 of 20,000
  pairings reach any of them. Thinnest unit Ihorombe, n=64; median n=183.
- **Quota**: 6 of 6 round pairs compared, most extreme 6 vs 7 with 9 of 145 cells identical,
  Bonferroni p = 1.
- **`Christian only`**, weighted share by round: R4 0.8%, R5 0.3%, R6 2.3%, R7 0.6%, R9 1.6%.
  Asserted under 5 points across the drawn rounds (`CONLY_SWING_MAX`).
- **The card** (`report_card()`, value labels): `None`, `Traditional/ethnic religion`, `Atheist`,
  `Agnostic`, `Calvinist` and `Lutheran` are on every drawn round's card.

## 3. The churches, and why they are kept apart here

Grouped, weighted, by round (%):

| | R4 | R5 | R6 | R7 | R9 |
|---|---:|---:|---:|---:|---:|
| Roman Catholic | 38.5 | 37.9 | 40.5 | 38.4 | 35.1 |
| Calvinist (FJKM) | 23.5 | 19.7 | 23.8 | 20.7 | 23.8 |
| Lutheran | 13.9 | 14.6 | 12.9 | 12.7 | 14.1 |
| Traditional/ethnic religion | 1.3 | 8.5 | 4.5 | 1.5 | 1.3 |
| None (with atheist, agnostic) | 7.0 | 8.2 | 4.0 | 13.0 | 12.7 |

The playbook's rule is to fold denominations unless `Christian only` is steady and an outside
witness holds their level. Both hold (§6). R4's card calls the box `Calviniste (FJKM)`; rounds 5-9
call it `Calvinist` at the same level.

## 4. The split-half, and what it placed

`cab.stability`, median over the 3 halvings of 4 rounds, per-round permutation null, chi-square veto:

| answer | n | share | median rho | null 95th | p | chi2 p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Roman Catholic | 1,800 | 37.59% | +0.548 | +0.295 | 0.0005 | 1e-40 | own geography |
| Calvinist (FJKM) | 1,083 | 22.62% | +0.826 | +0.311 | 0.0005 | 4e-72 | own geography |
| Lutheran | 654 | 13.66% | +0.797 | +0.294 | 0.0005 | 1e-92 | own geography |
| None or traditional | 636 | 13.28% | +0.682 | +0.297 | 0.0005 | 4e-133 | own geography |
| Other Christian | 169 | 3.53% | +0.385 | +0.305 | 0.0190 | 6e-10 | own geography |
| Other | 153 | 3.20% | +0.242 | +0.320 | 0.1124 | 2e-04 | fails |
| Seventh Day Adventist | 73 | 1.52% | +0.548 | +0.310 | 0.0015 | 6e-10 | own geography |
| Pentecostal | 67 | 1.40% | -0.070 | +0.295 | 0.6362 | 0.06 | fails |
| Muslim | 63 | 1.32% | +0.574 | +0.336 | 0.0020 | 3e-57 | own geography |
| Anglican | 62 | 1.29% | +0.217 | +0.327 | 0.1524 | 2e-27 | fails |
| Jehovah's Witness | 28 | 0.58% | +0.333 | +0.319 | 0.0435 | 2e-03 | passes, **under the 1% floor**, not placed |

Tested apart before pairing: None +0.857 (chi2 7e-97), Traditional +0.246 (fails, chi2 2e-61).

- **Standouts**: none (Anglican's top unit, Atsinanana, in 1 of 3 halvings).
- **Small-category rule**: worst residual multiple 1.68x (Anglican and Jehovah's Witness in
  Bongolava), under 2x, so the tail is the **residual**.
- **Level** (Norway): the pool on census weights against rounds 7 and 9 alone. Largest gaps:
  Catholic +1.48, None or traditional -1.10, Muslim -0.65 points. All under 3.5; no rescale.

## 5. None and traditional: one box, split at one ratio

**The card check Anita asked for**: every drawn round offers `Traditional/ethnic religion` as its
own answer, beside `None`, `Atheist` and `Agnostic`. So under step 2 of the draft procedure `None`
is `unaffiliated`, and that is how it is drawn.

**But the two answers trade places between rounds.** Per unit, rounds 5-6 against rounds 7 and 9
(% of respondents, traditional / none):

| region | traditional | none | both |
|---|---|---|---|
| Melaky | 45 -> 0 | 0 -> 12 | 45 -> 12 |
| Atsimo Atsinanana | 32 -> 0 | 1 -> 17 | 33 -> 17 |
| Atsinanana | 12 -> 0 | 2 -> 14 | 14 -> 14 |
| Betsiboka | 12 -> 0 | 19 -> 13 | 31 -> 13 |
| Sofia | 23 -> 10 | 22 -> 32 | 45 -> 42 |
| Androy | 7 -> 0 | 1 -> 55 | 8 -> 55 |
| Anosy | 2 -> 0 | 6 -> 42 | 8 -> 42 |
| Ihorombe | 2 -> 0 | 2 -> 41 | 4 -> 41 |
| the five central highland regions and Analamanga | 0 -> 0 | 0 -> 0 | 0 -> 0 |

The first four rows are the same people changing box, which is `Christian only`'s trap arriving
between two other answers, so the answer is the playbook's: **pool to a level the probing cannot
move.** None, atheist, agnostic and traditional are tested and placed as one category, then split
at rounds 7 and 9's national ratio, **0.905 None**. Both DHS surveys offer the two separately and put
None at 0.896 (2008-09 women), 0.922 (men), 0.912 (2021 women) and 0.958 (men) of the pair; asserted
inside that range. The pool's own ratio (0.72) is the one the swap corrupts.

Said plainly, because it cuts against the construction: **pooling does not make the regional
ranking steadier.** Early against late, the pair ranks the units at +0.57 and None alone at +0.59.
Androy, Anosy and Ihorombe (the last three rows) do not swap at all; they go from almost nobody to
40-55% with no traditional answers in between, on 16-32 respondents a round, which is a handful of
sampled villages. What pooling buys is that traditional religion's early answers are not thrown
away and no level rescale is needed.

**The alternative, built first and dropped** (the same day): None and traditional as separate
categories. None passes (+0.857), traditional fails (+0.246); None's pool was 10.16% against
13.69% in rounds 7 and 9, a 3.53-point gap, so §3.4 scaled None x1.3475 and traditional x0.3633 to
the late level; the tail went flat (Anglican 3.01x in Melaky). It drew None 13.7% and traditional
1.4% flat everywhere, which discarded Melaky's and Atsimo Atsinanana's early traditional answers,
and it rested on Norway's factor-or-shift test, which has no power here (ratio CV 1.96 against
difference CV 1.64). National totals of the two builds differ by about a point.

**The level is the soft part.** The DHS finds no religion at **20.6% of women and 25.0% of men**
aged 15-49 in 2021 (19.9% and 24.7% in 2008-09); this map draws 12.7%. The Afrobarometer is adults
18 and over, and neither survey says what these people practise. Anita's ruling left the reading of
Madagascar's None to the card check, and the card check is step 2; the Laos and Mozambique outcomes
were not used.

## 6. The DHS witness

| | Catholic | FJKM+FLM+Anglican | ratio | Muslim | traditional | none |
|---|---:|---:|---:|---:|---:|---:|
| EDSMD-V 2021, women 15-49 | 32.5% | 34.6% | 0.94 | 1.3% | 2.0% | 20.6% |
| EDSMD-V 2021, men 15-49 | 31.8% | 32.5% | 0.98 | 1.8% | 1.1% | 25.0% |
| EDSMD-IV 2008-09, women 15-49 | 35.7% | 35.6% (`Protestante/FLM`) | 1.00 | 0.7% | 2.3% | 19.9% |
| EDSMD-IV 2008-09, men 15-49 | 34.1% | 33.6% | 1.01 | 0.9% | 2.1% | 24.7% |
| **this map, as drawn** | **37.8%** | **36.4%** | **1.04** | 1.3% | 1.3% | 12.7% |

`witness_eds()` asserts the drawn ratio inside the DHS range widened by 10% (0.85-1.12). The DHS
2021 also has `Autre chrétien` at 8.5% and 7.1%, against 3.7% `Other Christian` plus 3.1% for the
Adventist, Pentecostal and Jehovah's Witness boxes here. **Nothing outside the Afrobarometer
separates the FJKM from the Lutherans.**

## 7. As drawn

Catholic 37.85%, FJKM 21.37%, Lutheran 13.91%, None 12.69%, Other Christian 3.66%, Other 3.34%,
Adventist 1.45%, Pentecostal 1.43%, traditional 1.33%, Muslim 1.30%, Anglican 1.15%, Jehovah's
Witness 0.53%.

- Catholic: Haute Matsiatra 60.1%, Bongolava 50.4%, Ihorombe 49.0%; Melaky and Boeny 18.8-18.9%.
- FJKM: Itasy 39.9%, Analamanga 39.0%, Alaotra Mangoro 36.5%; Androy 1.5%, Ihorombe 2.2%.
- Lutheran: Menabe 37.3%, Atsimo Atsinanana 35.2%, Vakinankaratra 30.1%, Anosy 29.7%.
- Muslim: Diana 14.7%, Melaky 7.8%, Boeny 4.9%; **zero** in ten regions where no pooled respondent
  was Muslim.
- None: Sofia 39.4%, Androy 27.3%, Sava 27.0%; with traditional, **zero** in Itasy, Bongolava,
  Vakinankaratra, Amoron'i Mania and Haute Matsiatra.
- Adventist zero in six regions (no respondents), Other Christian zero in Atsimo Atsinanana.

## 8. Placement: Kontur and its density cap

Kontur's MG extract (2023-11-01) is 1.177x the census over the 22 units, 0.87x (Haute Matsiatra)
to 1.69x (Melaky) per unit; 788 hexes (0.38% of people) fall outside every unit. Seven blocks
reach the cap (`python kontur_cap.py mg`), measured by distance to COD-AB's admin capitals, the 3 km
ring median and the share of everything within 10 km (scratch script, reasoning in each row):

- **capped**: 10 km south-east of Mahajanga (4 hexes, 121,866 people, ring median 17/km2); across
  the bay from Antsiranana near Ramena (1 hex, 34,072, ring 720/km2); 11 km south-east of Toliara
  (1 hex, 35,692, ring 963/km2).
- **real**: Antananarivo's agglomeration (82 hexes, 33.0% of Analamanga against Renivohitra's 35.2%
  of the region in Tome 1 Tableau 4); Ambovombe (0.4 km from the town; capping would lower it to
  its rural ring's 342/km2).
- **unreviewed**: two small blocks on Antananarivo's edge (2.9% and 1.3% of Analamanga), under 5%
  of the unit.

## 9. The §14 read

The queue row carries no §14 note. Regions average 1.2 million people; the survey's Muslim
communities are drawn at region level only (Diana, Melaky, Boeny). Nothing here was raised.

## 10. What would improve it

- **EDSMD-V 2021 microdata** (DHS registration): religion (`v130`) by region for about 27,000
  respondents, a second geography and a level for None. The brief treats DHS as walled.
- **MICS 2018**: whether it asks religion is unchecked (bot wall on the UNICEF links).
- **Afrobarometer round 10**, when merged.
- **Round 4 by district** for 21 regions, if `cab.stability` learns to tolerate one empty cell.

## 11. Review, 2026-09-15 (`d743fc47-rev5`)

Read the mapping, `mg.csv`, the note and the raw `.sav` files (scratch script, not kept).
`check_md` clean, `built_countries --check` OK, `check_rollup mg` 0 orphaned (every row
modelled). One screenshot: dots over the whole island, none in the sea. No ask, nothing rebuilt.

- **Mapping: agreed.** `christianity.united` (Uniting Church, United Church of Zambia, Church of
  South India) is not named in the FJKM `REVIEW` line. The FJKM still belongs at
  `christianity.reformed`: its Congregational and Reformed parents both sit under `reformed` in
  this tree, and the card calls the box `Calvinist`.
- **One sentence in `note_public` gives drawn figures as what people answered.** "It is the answer
  of a fifth or more of the people interviewed in most regions outside the central highlands,
  39.4% of Sofia and 27.3% of Androy." Those are the drawn figures, which include 90.5% of each
  region's traditional answers. Pooled over the four rounds, 27.4% of Sofia's respondents said
  none and 16.2% said traditional. By answers, 6 of the 16 regions outside the highlands reach 20%
  (Androy 26.6, Sofia 27.4, Anosy 24.7, Menabe 24.6, Ihorombe 23.0, Atsimo Andrefana 21.8); as
  drawn, 11 do. Suggested: "No religion is drawn at 12.7%, and at a fifth or more of the people in
  most regions outside the central highlands: 39.4% of Sofia and 27.3% of Androy."
- **Smaller wording point.** "Anglicans, Pentecostals, Jehovah's Witnesses and other religions are
  too few to place" is not quite the reason. `Other` (3.34%) and Anglican (1.15%) are both larger
  than the Muslims (1.30%), who are placed. They are spread because their pattern does not repeat
  between halves of the survey; only Jehovah's Witnesses are under the 1% floor.
- **Where the national ratio moves traditional religion.** In rounds 7 and 9 traditional answers
  come from six regions only: Sofia 9.9%, Menabe 8.4, Sava 7.4, Diana 4.6, Boeny 2.1, Atsimo
  Andrefana 1.0, and 0 in the other sixteen. Split at 90.5/9.5 everywhere, traditional is drawn at
  4.1% in Sofia and at 2.3-2.9% in Androy, Anosy, Ihorombe, Melaky and Atsimo Atsinanana, where the
  late rounds found none. That flattens the geography but does not bias the national level (both
  DHS rounds back the ratio), and the note describes the method. Left as built; if traditional
  religion is ever drawn by region, the late rounds' north-west and west is the pattern to test.
- **The highland zeros are real in the file.** Itasy, Bongolava, Vakinankaratra, Amoron'i Mania,
  Haute Matsiatra and Analamanga have 3 `None` answers (all in Analamanga) and no traditional answer
  among 1,854 pooled respondents. `Other` is 4-7% there (80 answers), more than in the regions with
  many `None` answers. There is no specify text, so whether that is revival churches or something
  else cannot be told. Recorded only.
- §14: nothing to add to §9.
