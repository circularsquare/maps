# Tanzania — a regional pattern from the pooled Afrobarometer, on the 2022 census

**Drawn 2026-09-14** (session `f95259a4-tz`). 30 units, 5 categories, 61,741,120 people, every
row `modelled`. Drawn on Anita's Nigeria ruling (`ask/answered/010-ng`, §9cv): the regional
picture from a pooled survey, with a short public note on why the state does not count. No ask
filed; §5 below is the one place a reviewer should look twice.

- `sources/tz_geo.py` -> `data/geo/tz/tz_regions.gpkg`, `tz_lookup.csv`, `tz_districts.csv`
  (COD-AB 2018 ADM1 and ADM2; NBS 2022 census Table 1, re-read from the PDF on every run)
- `sources/tz_grid.py` -> `data/geo/tz/tz_hexes.gpkg` (Kontur 400 m, 505,415 hexes)
- `sources/tz.py` -> `data/normalized/tz.csv` (the six shared Afrobarometer `.sav` files; the
  GFS CSV `sources/jp_gfs.py` already fetched, as a witness only)
- `taxonomy/tz2022.py` -> the mapping; one new node, `other.tz`; `countries.py` `"tz"`
- `sources/afrobarometer.py` gained an optional `extra=` on `load()` to carry the district column

```
python sources/tz_geo.py --fetch
python sources/tz_grid.py --fetch
python sources/ng.py --fetch          # only if data/raw/afrobarometer/ is empty
python sources/tz.py
```

## 1. What Tanzania publishes: nothing since 1967

| census | religion |
|---|---|
| 1967 | asked. Mainland Tanzania 32% Christian, 30% Muslim, 37% traditional religions (as reported by C.K. Omari, *Religion and Society in Tanzania*, Tanzanian Affairs, July 1983) |
| 1978 | **removed**: *"in the 1978 census the religious question was removed from the questionnaire for reasons known to those concerned"* (Omari 1983). No public reason, then or since. |
| 1988, 2002, 2012 | not asked |
| 2022 | not asked. The *Initial Results* report (116 pp, with the questionnaire) was searched for `religio`, `Muslim`, `Christian` and `dini`: two hits, a thank-you to religious leaders and an employer-type box ("NGO, religious organisation"). |

Tanzania is **ABSENT from the UNSD oracle** (`python tools/oracle.py Tanzania`, 2026-09-14).
Kieran Halloran (Berkley Center, Georgetown, 2013-04-16) confirms *"the Tanzanian government has
not taken a census that included religion since 1967"*. The reason commonly repeated online
(Nyerere's national-unity policy) was looked for and **not found in any source that could be
opened**: ConstitutionNet (Maoulidi 2014) does not discuss the census, and the State
Department's 2023 religious freedom report and *Social Analysis* 64(1) both returned HTTP 403 to
the fetcher. So `note_public` says the question was dropped without a public reason, which is
what the one source that addresses it says.

## 2. Routes, and where each ended

| route | date | outcome |
|---|---|---|
| UNSD Demographic Yearbook (oracle) | 2026-09-14 | absent |
| NBS 2022 PHC *Initial Results* | 2026-09-14 | population by region only (Table 1, PDF p. 17, used as the row margin); no religion anywhere |
| COD-PS Tanzania (`cod-ps-tza`) | 2026-09-14 | a 2020 projection off 2012 that HDX says does not reflect Songwe; **not used**, the office's 2022 count replaces it |
| DHS (TDHS-MIS) recode microdata | | registration-walled, §11ag; the last resort the brief names, not pursued |
| IPUMS | | blocked (`[[reference_ipums_account]]`) |
| **Afrobarometer R4-R9** | 2026-09-14 | open, on disk; **used**, R4 and R6-R9 |
| **Global Flourishing Study wave 1** (OSF `vrejf`, CC BY) | 2026-09-14 | 9,075 Tanzanians, Feb-Aug 2023, `REGION1` = regions (codebook sheet `gfs_sample_data_variables`); **a witness, not drawn** (§4) |

## 3. The construction

    row margin      region populations     NBS 2022 census, Table 1        EXACT
    the composition each unit's own mix    Afrobarometer R4, R6-R9 pooled  measured, n=10,745
    the national level                     neither                         computed

Nothing is fitted to a column margin (`sources/ng.md` §3 has why that matters).

### The units: 30, not 31

Mbeya and Songwe are one unit (`TZ12`). Songwe was cut from Mbeya in January 2016; the survey
labels it separately only from round 8, and rounds 4 and 6 list Songwe's districts (Mbozi, Momba)
under Mbeya. Everything else is the 31 census regions as COD-AB draws them; the name join is 31 of
31 with one spelling difference (`Dar-es-salaam`).

### Round 5 is left out, round 4 is placed by district

Geita, Katavi, Njombe and Simiyu were created in March 2012 from Mwanza, Kagera, Shinyanga,
Iringa and Rukwa. Rounds 4 (June-July 2008) and 5 (May-June 2012) both use the 26 old regions.

- **Round 4 has a `DISTRICT` column.** Every respondent is placed by district, matched to COD-AB
  2018's district names and asserted to land inside one of its old region's 2012 successors:
  1,192 placed, 136 of them in a region created after the round. **Magu's 16 are dropped**,
  because Busega District was split off Magu in 2012 and went to Simiyu (Wikipedia, *Busega
  District*). Arumeru is aliased to Meru.
- **Round 5 has nothing below the region**, so its respondents in the five split regions cannot be
  placed. Rather than measure nine units on fewer rounds than the other twenty-one, the round is
  out: **2,396 respondents**. Carrying it in the 21 unchanged units would add about 1,650, and
  would need a split-half that tolerates empty (round, unit) cells.

### Round 8 has codes and no labels

The R8 merged file's `REGION` label set (`labels7`) has no entries for 740-770, so `ab.load`
prints *0 REGION labels* for Tanzania. The codes mean one unit in every labelled round (asserted,
which is also what proves `Mrwara` = Mtwara and `Unfuja Kusini` = Kusini Unguja), so round 8 is
decoded by code and checked twice: its unweighted sample share per unit against rounds 7 and 9 (r
= +0.989) and its weighted share against the census (+0.959, 0 of 20,000 pairings reach it). R8
already carries Songwe as code 770.

### The checks

- **Labels against districts.** Rounds 6, 7 and 9 carry `LOCATION.LEVEL.1`: 2,358, 2,366 and 2,378
  respondents' districts match a COD-AB name, and **0** disagree with the region label.
- **Held-out, per round** (Denmark's rule): R4 +0.952, R6 +0.942, R7 +0.944, R8 +0.959, R9 +0.896,
  each against 20,000 pairings with none reaching it. Pooled +0.947. Kusini Unguja is the thinnest
  unit at n=104; Zanzibar is 10.0% of the pool and 3.1% of the census, which the per-unit
  composition removes from the national figure.
- **Quota** (`cab.assert_not_quota`, waves passed explicitly): **10 of 10** pairs compared, most
  extreme 7 vs 9 with 2 of 56 cells identical, Bonferroni p = 1.
- **Split-half** (`cab.stability`: median over all 10 halvings of 5 rounds, per-round permutation
  null, chi-square veto):

  | answer | n | share | median rho | null 95th | p | chi2 p | verdict |
  |---|---:|---:|---:|---:|---:|---:|---|
  | Christian | 6,310 | 58.72% | +0.923 | +0.238 | 0.0005 | 0 | own geography |
  | Muslim | 3,916 | 36.44% | +0.967 | +0.232 | 0.0005 | 0 | own geography |
  | None | 414 | 3.85% | +0.852 | +0.252 | 0.0005 | 6e-208 | own geography |
  | Other | 87 | 0.81% | +0.523 | +0.262 | 0.0010 | 9e-20 | passes, **under the 1% floor**, not placed |
  | Traditional/ethnic religion | 18 | 0.17% | +0.139 | +0.292 | 0.28 | 3e-13 | fails |

- **Standouts** (Honduras): none to test; the two failing categories are under the floor.
- **Small-category rule** (spec §12, 2x): under the residual, Traditional would be drawn at
  **4.03x** its national share in Mbeya and Songwe, where none of 606 respondents gave it, so the
  tail is **flat**: Traditional 0.20% and Other 0.93% everywhere, carried shares scaled to fill.
- **Level** (Norway; ask 015's question): the pool is not stale here.

  | | Christian | Muslim | None |
  |---|---:|---:|---:|
  | by round, survey weighting | 61.0-64.7% | 29.2-33.0% | 3.5-5.3% |
  | **as drawn** (pool, recomposed on the census) | **64.13%** | **29.50%** | **5.23%** |
  | R8-R9 alone, recomposed the same way | 63.32% | 30.29% | 6.02% |
  | the pool's own weighting | 62.89% | 31.46% | 4.51% |

  Every carried category is within 0.81 points of the recent rounds; no §3.4 rescale. Asserted
  under 3.5 points.

## 4. The witness: the Global Flourishing Study, 2023

An independent sample, 9,068 answered in 29 units (South Pemba has no wave-1 respondents; the
codebook has no Songwe, so its Mbeya is read as both). Decode: weighted share against the census
r = +0.926, 0 of 20,000. **It orders the units like the Afrobarometer: Christian Spearman +0.928,
Muslim +0.941, permutation p 0.0002 for both** (asserted at +0.80 and 0.01). Its own weighted
national shares are 62.3% Christian, 35.2% Muslim, 2.4% none; recomposed on the census over its 29
units, 64.6% and 32.6%. The largest disagreement is **Morogoro**, where the GFS is 24.7 points more
Muslim than the pool (drawn at 23.9%); that is one unit on 527 pooled respondents and is printed
on every build.

Not drawn, for three reasons: it is one wave, so it cannot run the round split-half; its card has
no box between Christianity and the world religions, so it adds nothing below the level already
drawn; and mixing two instruments' levels is §3.1a's problem for no gain, since the two agree.

## 5. Mainland and Zanzibar, and the §14 read

**Considered and not asked.** Zanzibar is semi-autonomous, 1,889,773 people, and is drawn as its
five regions at 97.2-98.9% Muslim. The candidate §14 question is Zanzibar's small Christian
minority: rule 2 says a persecuted group is drawn no finer than the state publishes, and Tanzania
publishes nothing. Nigeria is the precedent that decides it. Anita's ruling drew Christians in the
twelve Sharia states at state level; the rule-2 refusal in Nigeria was the Shia, a group whose
movement the state proscribed. Zanzibar's Christians are the first kind, not the second. What the
map shows is regional (Mjini Magharibi 1.7% Christian, three regions with none found), which is
coarse and common knowledge. The Zanzibar units are smaller than Nigeria's states (196,000 to
893,000 people), and that is the thing a reviewer might weigh differently.

Nothing else about the union question changes the construction: the 2022 census is one table for
both parts, and the survey's Zanzibar oversample is absorbed by composing per unit.

## 6. What is deliberately not drawn

- **Denominations and Muslim traditions.** `Christian only` runs 6.4% to 21.3% across the drawn
  rounds (3.6% in R5); **round 7's card is a different card** (Methodist 7.1% there and 0.0-0.2%
  elsewhere, Anglican 0.5% against about 4.5%, and `Tanzania Assemblies of God`, `Pentecoste`,
  `Evangelical Assemblies of God` only there). `Sunni only` runs 0.4% to 4.7%.
- **`None` is not §9dn's `unknown`.** Mozambique's and Laos's boxes took traditional religion by
  their own wording; this card offers `Traditional/ethnic religion` separately in every drawn
  round (`report_card()` reads the value labels). None is 24.0% of Shinyanga and 23.5% of Simiyu.
- **Zero cells** (§3.5): no Christians in Kaskazini Unguja (n=152), Kusini Unguja (104) and
  Kaskazini Pemba (160); no `None` in seven mainland and five Zanzibar units.

## 7. What would improve it

1. **TDHS-MIS 2022 recode microdata** (`v130` by region): same regions, far larger sample. Walled
   (§11ag), Anita's.
2. **Round 5 in the 21 unchanged units**, if the split-half is taught to skip empty cells.
3. **Afrobarometer round 10** when released; the decode here asserts `expect_rounds`.

## Placement: Kontur's density cap, 2026-09-14

One block at the cap: Dar es Salaam, 177 hexes, 36 at the cap, 4,447,234 people, 55.0% of the
unit's placement weight, 1.9 km from the city centre. Registered **`real`** in `kontur_cap.csv`:
it is the inner city, and Kontur's excess over the census (1.50x for the region) is city-wide
rather than a displaced block. Kontur carries 1.086x the census nationally; 5,401 hexes (0.82%)
fall outside every unit and are dropped.

## Review, 2026-09-14 (session `f95259a4-tzrev`)

Tested from the raw `.sav` files, the GFS CSV, the hex layer and COD-AB, not from the sections above.
Scratch scripts only; nothing rebuilt.

- **The shared `afrobarometer.py` change is clean.** With `extra=None` it adds no columns. Nigeria
  and Liberia rebuilt into scratch with the current module are byte-identical to
  `data/normalized/ng.csv` (46,691 bytes) and `lr.csv` (24,707 bytes).
- **Round 8's code decode holds on checks that read no names.** By raw code, round 8's share of
  the sample, share urban and mean weight track rounds 6, 7 and 9 at +0.967 to +0.989, as closely
  as two labelled rounds track each other (R7 against R9, +0.961 to +0.982). The only code that is
  100% urban is 746, Dar es Salaam; the five with the lowest mean weight (the Zanzibar oversample)
  are 761 to 765. Weighted share against the census is +0.959; shifting the code order along the
  list gives at best +0.337, and 0 of 20,000 random pairings reach it. Nothing here can rule out a
  swap of two similar-sized rural regions (2 of 29 adjacent swaps tie on population), and nothing
  points to one: the code range is the same in rounds 6, 7 and 9, and 770 is Songwe in both 8 and 9.
- **Round 4 by district is right.** Chato (24), Bukombe (8) and Geita (24) go to Geita, Mpanda (24)
  to Katavi, Njombe (16) and Makete (8) to Njombe, Bariadi and Meatu (16 each) to Simiyu: each a
  2008 district that moved whole. Round 4 lists Chato separately, so Biharamulo is not a second
  Magu. Round 5 really has nothing below the region; its only area columns are yes/no facility
  items (`EA_SVC_*`, `EA_FAC_*`).
- **Songwe did not need merging.** The district columns place it the way round 4's split regions
  were placed: Mbozi (R4), Mbozi and Momba (R6), Mbozi, Momba and Tunduma (R7), with Chunya
  ambiguous before Songwe District was cut from it in 2016 (16 in R4, 8 in R6), as Magu is. A
  31-unit build would have Songwe at about n=255 and Mbeya at about 327, both above Katavi's 120.
  Not wrong as drawn; one more item for §7.
- **Dropping round 5 is not a systematic bias, but it moves two units away from the witness.**
  Across the 21 unchanged units round 5 is the most Muslim round on average (+3.6 points against
  the other rounds), about as far out as round 7 is the other way (-3.4); adding it moves their
  census-recomposed Muslim share from 39.9% to 41.0%. But Morogoro and Ruvuma each move 7 points
  toward Islam with it, and in both cases toward the GFS: Morogoro 23.9% drawn, about 31% with
  round 5, GFS 48.7%; Ruvuma 41.4%, 48.4%, GFS 47.5%. Morogoro's six rounds run 60, 62, 31, 13, 19
  and 15% Muslim, which is which enumeration areas were drawn, not change, so more rounds is the
  only remedy this survey has. Nigeria carried round 6 with three states unsampled; that precedent
  argues for §7 item 2. The "about 1,650" in §3 is 1,756.
- **Dar es Salaam's `real` holds.** Kontur's density falls away from the city centre with no
  plateau-and-cliff: 17,900/km2 within 2 km (the business district), 22,000 to 23,800 at 2-8 km,
  15,400 at 8-10, 9,400 at 10-13, 1,600 at 20-30. Tashkent's block was 15.6 km out with 2,432/km2
  at the centre and Conakry's 24 km out; this is neither. Kontur's excess is also even across
  COD-AB 2018's three districts: Ilala 1.52x, Kinondoni with Ubungo 1.43x, Temeke with Kigamboni
  1.57x, against NBS's 2022 council counts (Ilala 1,649,912; Kinondoni 982,328; Ubungo 1,086,912;
  Temeke 1,346,674; Kigamboni 317,902). Those counts are not on disk; they sum exactly to Table 1's
  5,383,728.
- **`note_public` figures match `tz.csv`**, Rukwa narrowly (94.27% against "more than 94%"). Two
  sentences read wrongly and were fixed, not refreshed: "That last count found..." came straight
  after "the one in 2022" and now reads "The 1967 count found..."; "No religion has a geography of
  its own here" read as "no religion does" and now reads "Having no religion is regional too".
- Checks clean (`check_md`, `built_countries --check`, `check_rollup tz`); one screenshot at the
  country's bounds looks right.

### §5, for Anita to weigh

The facts in §5 check out (Zanzibar's regions 97.2 to 98.9% Muslim, 195,873 to 893,169 people).
It leaves out three things.

1. **The Nigeria precedent is an outcome, not a ruling on this point.** Ask 010 asked whether
   Nigeria's state balance could be published at all. Neither it nor the ruling discusses
   Christians in the twelve Sharia states as a rule-2 group; rule 2 was applied to the Shia only,
   and the ruling accepted the build as a whole.
2. **Zanzibar's Christians have been attacked, though not by the state.** Churches were burned in
   2012 during the Uamsho unrest, a Catholic priest was shot on Christmas Day 2012, and Father
   Evarist Mushi was shot dead outside Zanzibar Town in February 2013. §5's test (a group the state
   proscribed) is a fair reading of rule 2, and northern Nigeria's Christians have a similar
   history, so the precedent still points the same way. The record should say so rather than
   leave it out.
3. **The drawn groups are much smaller than Nigeria's.** Mjini Magharibi 14,848 Christians (12 of
   512 respondents) and Kusini Pemba 1,839 (one respondent of 144, about two dots at 1,000 people
   per dot; §5 does not mention it). The smallest Christian population drawn in any Sharia state is
   Jigawa's 135,856, and the smallest of those states, Yobe, has 3.65 million people. The three
   zero regions are not evidence of none: no Christian among 104 to 160 interviews is consistent
   with up to 2 to 3%.

What limits the exposure: dots inside a region follow Kontur's population, not where Christians
live, so nothing finer than the region is shown. If she wants it coarser, one option is a single
Christian share across Zanzibar's five regions (13 Christians among 1,072 interviews), which
removes both the one-respondent figure and the zeros. §2 also does not list Zanzibar's own
statistics office (OCGS), the other government rule 2's "the state" could mean; not checked here.
