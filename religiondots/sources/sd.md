# Sudan (`sd`)

Session `cb8b206e-sd`, 2026-09-15, under a supervisor. Reopened from `queue.md`'s closed row on
Anita's priority of 2026-09-15 (the biggest holes first) and her Maghreb and Mauritania rulings
(`ask/RULINGS.md`, 2026-09-15 and 2026-09-16). Code: `sources/sd.py`, `sources/sd_geo.py`,
`sources/sd_grid.py`, `taxonomy/sd2022.py`, `countries/sd.py`. `sources.md` §sd-2026-09-15. Ask
041 (§14).

## 0. Outcome

Drawn at 18 states, **46,934,433 people** (the Central Bureau of Statistics' projection for 2022,
COD-PS), at **one national non-Muslim share and mix in every state**. Every row `modelled`.
Foreigners and refugees are not in the surveys and are named in `gap` (2.855%). Abyei is not drawn.

| | people | share |
|---|---:|---:|
| Muslim (`islam`) | 46,612,128 | 99.313% |
| Christian (`christianity`) | 236,716 | 0.504% |
| No religion (`unaffiliated`) | 85,589 | 0.182% |

## 1. Sources

- **Afrobarometer** merged rounds, `data/raw/afrobarometer/`: R5 (1,199, February 2013), R6
  (1,200, June 2015), R7 (1,200, July-August 2018), R8 (1,800, February-April 2021), R9 (1,200,
  November-December 2022). Sudan is not in R4. `REGION` is 15 pre-2012 states in R5 and six
  macro-regions in R6-R9; `LOCATION.LEVEL.1` is the 15 old states in R6, town names in R7, empty
  in R8 and the 18 states in R9.
- **Arab Barometer**, `data/raw/arabbarometer/`: the files offer Sudan waves II (1,538), III (1,200,
  29 April to 29 May 2013), V (1,758) and VII (2,353). `Q1` is 13 states in II, 15 in III and all
  18 in V and VII. V carries `Q1012A` (one sect item for everyone); VII `Q1012A_MUSLIM`,
  `Q1012A_CHRISTIAN` and `Q1012B` ethnicity.
- **Arab Barometer Wave VII Technical Report** (`arabbarometer.org/wp-content/uploads/
  AB7_Technical_Report.pdf`, p.12, opened): Sudan, 30 January to 11 April 2022; "Citizens aged 18
  and above"; excluded "Institutionalized populations; West Kordofan and North, which were included
  to Greater Kordofan provinces after the census" (but VII has 104 West Kordofan and 64 Northern
  respondents, so the line is read as a frame note, not checked further); frame "2018 census of
  the Sudanese Central Bureau of Statistics" (no 2018 census was held; COD-PS says it was deferred);
  36 strata, 18 states, 297 PSUs, response rate 74%.
- **COD-AB `cod-ab-sdn` v03** (OCHA ROSEA, CC BY-IGO; boundaries created 2013-08-13, valid
  2020-08-31, reviewed 2025-10-30), `data/raw/sd/sdn_admin_boundaries.geojson.zip`: `sdn_admin1`
  is 19 features, SD01-SD18 and `Abyei PCA` SD19.
- **COD-PS `cod-ps-sdn`**, `data/raw/sd/sdn_admpop_2022.xlsx` (HDX resource dated 2022-11-22,
  CC BY): 18 states, 46,934,433, reference year 2022, source CBS, baseline the 2008 census, cohort
  component; "does not include the region of Abyei". `sdn_admpop_2025.xlsx` and
  `sdn_2025_technical_accompanying_note.pdf` are beside it, read for the method only.
- **Kontur population `SD` 20231101**, `data/raw/sd/`; **GeoNames `SD`** (CC BY 4.0).
- **Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), Sudan
  2020: Muslims 98.856%, Christians 0.488%, unaffiliated 0.563%, other religions 0.092%; 2010
  Christians 0.172%. South Sudan 2020: Christians 60.5%, other religions 32.8%, Muslims 6.2%.
- **UN DESA, *International Migrant Stock 2024*** (`data/raw/mr/undesa_pd_2024_...xlsx`, the copy
  Mauritania's build fetched), Sudan as destination: 1,379,147 at mid-2020, 2,397,113 at mid-2024;
  type of data `B R` (foreign-born, with UNHCR's refugees added); 1990-2005 include South Sudan.
  Mid-2020 by origin: South Sudan 867,593, Eritrea 220,960, Chad 103,065, Ethiopia 70,173, CAR
  25,962, Nigeria 20,508, Egypt 18,566.
- **UNHCR Refugee Data Finder API**, `api.unhcr.org/population/v1/population/?year=2022&coa=SDN&
  coo_all=true&cf_type=ISO` (read 2026-09-15): end 2022, refugees South Sudan 796,831, Eritrea
  119,410 (plus 15,304 asylum seekers), Syria 93,478, Ethiopia 55,455 (plus 15,522), CAR 24,363,
  Chad 4,654, Yemen 1,941.
- **UNHCR, *Sudan Country Refugee Response Plan, January-December 2022*** (`data.unhcr.org/en/
  documents/download/91010`, opened): end-2021 estimate 1,141,313 (South Sudanese 801,014, Eritrean
  135,356, Syrian 93,498, Ethiopian 71,339, CAR 32,057, Chadian 4,612, Yemeni 1,983); South Sudanese
  "about 30 percent" in 11 camps in White Nile (9) and East Darfur (2), "an estimated 113,000" in
  Khartoum's open areas; Eritreans in nine camps in Kassala and Gedaref, Kassala town and Khartoum;
  Syrians and Yemenis mostly in Khartoum; CAR refugees in South and Central Darfur. Per-state
  figures are drawn as map classes only, with no numbers in the text layer.

## 2. Rulings and §14

- 2026-09-15 (priority): Sudan is one of the holes to fill first.
- 2026-09-15 and 2026-09-16 (Maghreb, Mauritania): a near-uniformly Muslim country is drawn with the
  best evidence for where its non-Muslims are; foreigners drawn where a state count by province
  and a nationality mix allow, otherwise sized in `gap` (ask 033).
- **§14, flagged in ask 041, not settled here.** Sudan has been at war since April 2023, with mass
  killing along ethnic lines in Darfur and fighting in South Kordofan and Blue Nile, where the
  Nuba Mountains hold many of Sudan's Christians; apostasy was a capital crime until 2020. As
  built, nothing is placed below the nation: the Christian and no-religion shares are the same in
  every state, and the Kordofan excess the late rounds show is not drawn (§4). The ask asks
  whether to publish Sudan at all while the war goes on, and whether a later refugee layer by state
  (camps are public) would be acceptable.
- Not checked: whether either survey reached the parts of South Kordofan and Blue Nile the SPLM-North
  holds. The technical reports for Afrobarometer R5-R9 and Arab Barometer V were not opened.

## 3. The pool

**The card** (`ab.card`). II's codes are muslim, christian, unspecific answer, khaki, jewish, don't
know, declined; III's are Muslim, Christian, Other, Jewish (Yemen only), don't know, refuse: neither
has a box for having no religion, so both are out (4 and 2 Christians would be lost). V has
`Atheist`, VII `No religion`. Every Afrobarometer round offers `None`, `Atheist` and `Agnostic`.

**Grouping** (`AFRO_GROUP`, raising on a new answer). Muslim: `Muslim only`, `Sunni only`, Shia 2,
Ismaeli 2, Tijaniya 6, Qadiriya 1, Mouridiya 1. Christian: `Christian only` 25 (6 short form, 19
long), Presbyterian 3, Seventh Day Adventist 2, Anglican, Church of Christ, an independent church,
Jehovah's Witness, Lutheran, Methodist, Mormon, Orthodox, one each. No religion: `None`, `Atheist`,
`Agnostic`.

**Four drops, each asserted.**

1. Refusals and don't-knows: Afrobarometer 8, Arab Barometer 3.
2. **Contradictions** (Algeria's rule): in wave V, six of seven atheists gave a Muslim branch on the
   follow-up (`Just a Muslim` 5, `Sunni` 1; the seventh refused), and so did one Christian
   (Khartoum) and the one Jewish answer (River Nile). No Muslim named a church. Wave V's six
   Catholic and two Orthodox follow-ups all belong to Christians.
3. **Round 8's `None` in Darfur.** R8 has 32 `None` answers; 31 are in Darfur (30 rural), 7.2% of
   its 429 Darfur interviews, weighted 2.15% of the round against 0.00-0.56% in every other round.
   Darfur gave `None` once in R5, never in R6-R7, once in R9, and once in the Arab Barometer (V,
   North Darfur, follow-up refused): 3 in about 2,400 other interviews. R8 has no location below
   the region, so the block cannot be placed further. Read as a recording artefact (a code for "no
   particular sect" is the likeliest), dropped, not recoded to Muslim. Kept, and not dropped: R7's
   5 `None` in consecutive rural interviews at Suakin (Red Sea), one sampling point; the East
   returns `None` in R7, R9 and wave V, so it is not the same shape. Trap added to
   `playbooks/afrobarometer.md`.
4. **Single answers with no community to place**: Bahá'í (R9, Sennar), Jewish (R9, North Kordofan)
   and traditional religion (R7, Central), one each.

**Pool as drawn:** 10,657 answers (Afrobarometer 6,557, Arab Barometer 4,100), **86 not Muslim: 62
Christian, 24 no religion.** Weighted non-Muslim share by wave: AF5 0.297%, AF6 0.643%, AF7
1.052%, AF8 0.717%, AF9 0.963%, AB V 0.614%, AB VII 0.612%. Christian: 0.172, 0.643, 0.488, 0.607,
0.407, 0.589, 0.521%. Pooled **0.6867%**, mix Christian 73.4%, no religion 26.6%. The late half
(2021-2022) alone: 0.7261%, Christian 0.5237%. Pew 2020: Christian 0.488% (the pool reads 0.504%),
unaffiliated 0.563% (0.182%).

**States.** Arab Barometer V and VII and Afrobarometer R9 name all 18 (`STATE`, names folded to
letters; `The Island` is Al Jazirah, `North` is Northern state where a list also has River Nile).
Weighted respondents per state against COD-PS 2022, Spearman: V +0.988, VII +0.986, R9 +0.934.
Held-out r = +0.999 over 18 states, best of 20,000 pairings +0.931. Quota test, V against VII: 12
free cells, none identical, p = 1. R6's old-state location column agrees with its region labels in
every row (asserted).

## 4. What stands apart: nothing in both halves

Six regions every round can be read at (Khartoum; Central = Al Jazirah, Sennar, White Nile, Blue
Nile; East = Red Sea, Kassala, Gedaref; North = Northern, River Nile; Kordofan = North, South and
West; Darfur = five states). Halves: early AF5, AF6, AF7, AB V (2013-2019); late AF8, AF9, AB VII
(2021-2022). Exact within each wave; bar 0.05/6 = 0.0083.

| region | answers | non-Muslim | Christian | early | late |
|---|---:|---:|---:|---|---|
| Central | 2,618 | 15 | 13 | 10 vs 10.23, P 0.594 | 5 vs 10.88, P 0.992 |
| Darfur | 2,531 | 11 | 8 | 8 vs 9.40, P 0.756 | 3 vs 11.01, P 1.000 |
| East | 1,533 | 15 | 5 | 10 vs 5.98, P 0.066 | 5 vs 6.42, P 0.790 |
| Khartoum | 1,933 | 24 | 15 | 9 vs 7.73, P 0.364 | 15 vs 7.98, P 0.0084 |
| Kordofan | 1,396 | 17 | 17 | 4 vs 5.30, P 0.795 | 13 vs 5.85, P 0.0037 |
| North | 646 | 4 | 4 | 0 vs 2.36, P 1.000 | 4 vs 2.86, P 0.320 |

**Kordofan passes the late half and not the early one**, on R8's 10 Christians in 232 Kordofan
interviews, which have no state; Khartoum misses the late bar by 0.0001. Neither is drawn. Every
Kordofan non-Muslim answer is Christian, which is the direction the Nuba Mountains would give; a
next round that repeats it would pass.

**Towns.** Afrobarometer 26 urban of 58 non-Muslims against 20.53 expected, P 0.086; Arab
Barometer 9 of 28 against 10.76, P 0.810. Morocco's bar needs both under 0.05. No urban split.

## 5. Units and population

`sd_geo.py`. COD-AB admin1 with SD19 Abyei dropped: no projection in COD-PS, sampled by neither
survey, status between Sudan and South Sudan not settled. **The Halaib triangle** (Red Sea state
north of 22 N, 18,042 km2) is clipped on spec §14.18 (de facto Egypt); Northern state's Wadi Halfa
salient (to 22.225 N) stays. Kontur's `SD` extract already stops at 22 N there (0 hexes beyond).
Joined on p-code, every English name equal on both sides.

**Why the 2022 edition.** It is the last before the war: the 2024 and 2025 editions fold in
displacement since April 2023 (2025 note, p.3: "more than 12 million people have fled their homes";
Khartoum 8.94 million in 2022, 5.41 million in the 2025 file's 2024 row; West Darfur 1.14 million
and 0.37 million), and every interview in the pool is from 2013 to 2022. It is a projection from
the 2008 census, whose post-enumeration check, if any, is unrecorded (the workbook's own metadata).

## 6. Placement

`sd_grid.py`. Kontur `SD` 234,490 hexes, 48,134,387 people; 1,376 centroids outside every state
(103,814), 350 snapped within 2 km, 1,026 dropped (44,902, 0.093%, Abyei and the extract's
overrun). **Kontur over the projection 1.025.** Join witness: Spearman over 18 states +0.864, 0 of
20,000 shuffles reach it (best +0.816). Per state over the national ratio: Gedaref 0.69, Central
Darfur 0.70, West Darfur 0.76 to North Darfur 1.47, East Darfur 1.49, Red Sea 1.67. Placement is
inside each state only, so these move no count. Seat check: 16 GeoNames seats of 50,000 or more, no
hole (Khartoum 0.32 of GeoNames within 5 km and 0.98 within 10).

**Kontur's cap: 26 blocks, all judged false.** Every block sits on a small town or none. Where
GeoNames gives the town a population the block holds 2.7 to 14.9 times it: Kabkabiya 633,115
against 42,376, Buram 453,732 against 65,473, Kas 286,238 against 55,255, Sinkat 108,748 against
35,617, Jubayt, Aroma, Rahad al Bardi. The rest have no town with a population within 10 km
(Derudeb 209,728, Saraf Omra 222,984, Haya 146,308, and ten more in the Red Sea hills). No city
Kontur draws densely reaches the cap: Khartoum, Port Sudan, Nyala and El Obeid have no block. 23
are `capped` in `kontur_cap.csv`. **Three are `unreviewed`** (Hillat Ashat 92,082, a Timerein
block 72,035, Hillat Adam 36,029, all Red Sea state): no populated hex lies within 3 km, so `capped`
has no ring to set a ceiling and stops the scatter. **Two capped Timerein blocks (72,153 and 36,082)
are not lowered either**: their only ring hexes are the unreviewed Timerein block's, at the cap, so
the ceiling came out at 46,199/km2 (the scatter prints it). So five blocks, about 308,000 people and
10.8% of Red Sea state's Kontur weight, are drawn as Kontur has them, in the desert south-west of
Tokar, until the cap method handles an isolated block (the supervisor's). The other 21 blocks are
lowered, from 633,115 to 3,914 at Kabkabiya.

**Isolated blocks, the rule (session `cb8b206e-fixes2`, 2026-09-15, written before any number
below was computed).** A block judged false whose 3 km ring holds no populated hex outside every
dense block (none at all, or only another block's hexes) gets the registry status `isolated`:
each of its hexes is lowered to the median density of the populated hexes of its own state that
lie in no dense block. Placement is proportional inside each state, so this spreads the block's
excess over the state's other populated hexes in proportion to Kontur, and no count moves.
`kontur_cap.apply` refuses `isolated` on a block that does have a populated ring outside the
blocks, so the status cannot stand in for `capped`. The five blocks qualify only if Somalia's
test for a real town also fails (`sources/so.md` §5: a GeoNames place within 5 km holding a
third of the block); the registry rows already say the nearest GeoNames places give no
population, and the check is rerun on the file.

**Result.** Rerun on `sd_hexes.gpkg` with `kontur_cap.find_blocks`, for every registry row: each of
the five has no populated hex within 3 km outside a dense block, and no GeoNames populated place
within 5 km holds a third of its people, so Somalia's test calls none of them real. Red Sea state
has 3,191 populated hexes outside the blocks, at a median of 9.9/km2, so the five went from
308,381 people to 69 (Hillat Ashat 23, the two-hex Timerein blocks 15 each, the one-hex Timerein
block and Hillat Adam 8 each). At 1:1,000 the 291 dots within 3 km of the five are gone; the
state's dot count cannot change, so they moved to its other populated hexes. Both editions
rescattered.

Found and not changed, both for the supervisor:

- **Somalia's town test would call three of the 21 `capped` blocks real**: Jubayt (89,182 against
  GeoNames' 30,856, now lowered to 552), Aroma (35,645 against 12,708, now 497) and Rahad al Bardi
  (66,166 against 24,768, now 1,608), each 2.7 to 2.9 times its town, under that test's bar of 3.
  Somalia's rule also needs the block to hold at most 1.5 times the districts it touches, which
  Sudan's state table cannot check. So the towns may be drawn too thin rather than too thick.
- **Tokar's `capped` block keeps 11,771 people.** 2 of its 4 ring hexes are in another block, so
  the ring's median is 15,053/km2; it gained dots at 1:1,000 (55 to 65 within 3 km) as the
  isolated blocks' weight spread over the state. Making `capped` ignore other blocks' hexes would
  move ceilings in every country with neighbouring blocks, a shared rule, so it is not done here.

## 7. Foreigners and refugees: in `gap`

Both surveys sample citizens (Arab Barometer's report; Afrobarometer's standard frame). The
projection carries forward the people the 2008 census counted and has no refugee component, so
most refugees who arrived since 2013 are outside it, and foreigners counted in 2008 are inside it
at the citizens' shares. **`gap` and `gap_share`** take UN DESA's 1,379,147 (mid-2020, `B R`) on
the projection plus that figure: 1,379,147 / 48,313,580 = **0.02855**. It is an upper bound on who
is left out, by the part already inside the projection, which nothing measures.

**Not drawn, and the route to drawing them.** The ruling's construction needs a count by state and
a nationality mix. UNHCR has both: its *Population Dashboard: Overview of Refugees and
Asylum-seekers in Sudan* was published for 28 February 2023 and 31 March 2023 (1,144,675 refugees
and asylum seekers before the war, per ReliefWeb's summary; 24 camps: 10 in the east, 10 in White
Nile, 2 in East Darfur, 1 in Blue Nile, 1 in Central Darfur). ReliefWeb returned an empty 202 to
curl and 403 to WebFetch; the UNHCR portal search page was saved and its document ids were not
extracted. With it, South Sudanese (Pew South Sudan: 60.5% Christian, 32.8% other religions),
Eritreans and Ethiopians would be the largest non-Muslim population in Sudan, larger than the
Christians the surveys find among citizens. Not started, on the context rule (brief §4); REOPEN.

## 8. REOPEN

- **Refugees by state** (§7), through `taxonomy/origin_religion.py` as `sources/mr.py` does; §14 in
  ask 041 first.
- **Kordofan's Christians** (§4), if a later Afrobarometer or Arab Barometer round repeats the late
  half.
- **The survey technical reports** for Afrobarometer R5-R9 and Arab Barometer V: exclusions, and
  whether SPLM-North-held areas were sampled.
- **MICS 2014 and the 2010 household health survey** have no religion item (§11aq); a post-war
  census, if one is held.

## 9. Calls someone might reverse

1. One national share and mix; Kordofan's late-half excess not drawn.
2. Round 8's 31 Darfur `None` answers dropped as a recording artefact; R7's 5 at Suakin kept.
3. Arab Barometer II and III out on the card.
4. Eight wave V contradictions dropped (Algeria's rule).
5. COD-PS 2022 (pre-war) as the base, not the 2025 edition.
6. Abyei not drawn; Halaib clipped to Egypt.
7. Foreigners and refugees in `gap` on UN DESA mid-2020, not drawn from UNHCR by state.
8. Five ring-less cap blocks `isolated`, lowered to Red Sea state's median density outside the
   blocks (§6), not to a wider ring.

## 10. Review, 2026-09-15 (session `cb8b206e-rev8`)

Full pass. `check_md` clean; `check_rollup sd` all modelled, nothing orphaned; `built_countries
--check` has sd in both editions. The mapping (bare `islam`, bare `christianity`, `unaffiliated`)
follows Libya, Algeria and DR Congo; no new node. One screenshot of the country and one of the Red
Sea hills: dots follow the Nile, Al Jazirah, Kordofan and Darfur, none in the sea, the desert
empty; the five blocks left at the cap show as small clumps near Timerein and Tokar, nothing that
reads as a town that is not there.

Two points for whoever next edits `note_public`. Neither is an ask.

1. **Pew is not an outside check on this pool.** Pew's Sudan 2010 row is Christians 0.1719% and
   unaffiliated 0.1256%, together 0.2975%. That is Afrobarometer round 5's weighted Christian share
   (0.172%) and non-Muslim share (0.297%) from §3, so Pew's Sudan series is built on the
   Afrobarometer, and its 2020 figure shares inputs with this map. The note's "close to Pew
   Research Center's 2020 estimate" reads as a second source agreeing when it is mostly the same
   one. Either say Pew draws on the same surveys, or drop the comparison.
2. **The level is likely biased low, and the note does not say so.** Morocco's note says
   "Moroccans who have become Christian or left Islam may not say so to an interviewer". Sudan's
   case is stronger: apostasy carried the death penalty until 2020, and four of the seven waves
   (AF5, AF6, AF7, AB V) were fielded before that. The open coverage question (§2, §8) points the
   same way. Afrobarometer R9's technical information form (*Summary of results*, p.4,
   `afrobarometer.org/wp-content/uploads/2024/03/Summary-of-results-Sudan-Afrobarometer-R9-4march24.pdf`,
   opened) names no excluded area: citizens 18 and over, frame "Sudan population census 2008,
   projected in 2020", stratified by region and urban-rural, EA substitution 14 of 150 (9.3%).
   The World Bank catalog pages for R8 (`microdata.worldbank.org/index.php/catalog/5822`) and R9
   (`.../catalog/6752`) say "National coverage" and leave exclusions to the technical report. If
   the SPLM-North-held parts of South Kordofan and Blue Nile were not reached, the one share was
   measured away from where §4 puts Kordofan's Christians. Suggest a sentence like Morocco's,
   and a line in `gap` or the note once a technical report settles coverage.

**Done, 2026-09-15 (session `cb8b206e-fixes2`).** Both points are in `note_public`. The Pew
comparison now says Pew's figures are not a separate check, because its 2010 estimate matches the
Afrobarometer's 2013 round, and the no-religion comparison (Pew 0.56% against 0.18%) is dropped
with it. Added: Sudanese who have become Christian or left Islam may not say so to an interviewer,
since leaving Islam carried the death penalty until 2020 and four of the seven rounds came before
that (AF5, AF6, AF7, AB V). The coverage line waits for a technical report, as before.
