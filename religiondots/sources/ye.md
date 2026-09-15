# Yemen — Arab Barometer wave V (2018-2019), with wave III (2013) as the witness

Built 2026-09-14 by session `d743fc47-ye`. The fourth country drawn from the Arab Barometer
after Egypt, Jordan and Iraq, and built on Iraq's shape (`sources/iq.md`). `sources.md`
§ye-2026-09-14 is the short record.

**34,803,293 people over 21 of Yemen's 22 governorates**, five source categories on three nodes,
every row `modelled`. Socotra (75,725) is not drawn.

## 1. The state route

There is none to close. Yemen's last census was 2004 and published no religion; the Central
Statistical Organization's projections carry population only, and Yemen is absent from the UNSD
Demographic Yearbook's religion table (`tools/oracle.py`). The population denominator is the
humanitarian Population Task Force's 2025 estimate (§6).

## 2. What is drawn

| source category | node | share as drawn | people |
|---|---|---:|---:|
| Sunni | `islam.sunni` | 58.01% | 20,189,443 |
| Just a Muslim | `islam` | 23.77% | 8,271,463 |
| Zaydi | `islam.shia` | 17.71% | 6,163,130 |
| Muslim, denomination not given | `islam` | 0.41% | 142,578 |
| Muslim, other denomination | `islam` | 0.11% | 36,679 |

`taxonomy/ye2019.py` argues each in `REVIEW`. No new node: a `islam.shia.zaydi` child would be a
legend row only Yemen uses, which AGENT_BRIEF §3 sends to Anita, and `islam.shia` is exact.

Zaydi share as drawn, by governorate: Sa'dah 62.5, Amran 51.3, Dhamar 39.4, Raymah 36.5, Hajjah
34.6, Sana'a City 28.1, Al Jawf 24.9, Sana'a 22.0, Al Mahwit 16.3, Ibb 12.2, Al Hudaydah 11.6,
Ma'rib 1.5, Hadramawt 1.0, and none in Al Bayda, Aden, Ad Dali', Abyan, Al Maharah, Lahj,
Shabwah and Ta'iz.

## 3. The files, and why one wave is drawn

| wave | fielded | Yemenis | religion answer | sect item |
|---|---|---:|---|---|
| II | 2011 | 1,200 | empty | none |
| III | Nov-Dec 2013 | 1,200 | 1,198 Muslim, 2 Other | card: Sunni 971, Shia 214, Refuse 13 |
| V | 2018-2019 | 2,400 | 2,399 Muslim, 1 Atheist | logged: Shafi'i 815, Sunni 616, Just a Muslim 545, code 14 `Alawi` 410, refused 11, other 2, don't know 1 |

Wave II is not offered by `ab.wave_coverage` (no answered `q1012`). Waves IV, VI, VII and VIII
did not field Yemen.

**Wave V is drawn; wave III is not pooled.** Wave III's card is Sunni or Shia and nothing else,
so its Muslims had no box for naming no branch. Pooled beside wave V's `Just a Muslim`, the
undifferentiated share would be half-measured and the branch shares inflated by forced choices,
which is Iraq's card rule (Iraq omitted II and III). Wave III is used as the independent
replication in §5.

**Wave V's sect item is interviewer-logged.** The questionnaire (`ABV_SourceQuestionnaire_ENG`,
p. 50) reads `Q1012a ... What is your religious denomination? [INTERVIEWER: DO NOT READ, LOG
ANSWER]` over one cross-country precode list: Maronite, Orthodox, Catholic, Armenian, Sunni,
Shia, Hanbali, Shafi'i, Ja'fari, Druze, Ahmadiyya, Mozabite, Just a Muslim, Alawi, Other. There is
no Zaydi code.

## 4. Four readings of the file

1. **Code 14 `Alawi` is Zaydi.** Its 410 respondents are 34.0% of the nine highland governorates
   (Sa'dah, Amran, Hajjah, Dhamar, Sana'a, Sana'a City, Al Jawf, Al Mahwit, Raymah), 10.6% of Ibb,
   Al Hudaydah and Ma'rib, and **1 in 950** across the nine southern and eastern ones, with 1 of
   120 in Hadramawt, which rules out the Ba 'Alawi sayyids of the Wadi. Nobody in Yemen is logged
   under code 6 `Shia`. Wave III's `Shia` on a different card lands in the same places (37.3%,
   7.7%, 0.93%; Sa'dah 39 of 40). `sources/ye.py::zaydi_geography` asserts both waves.
2. **`Shafi'i` is folded into Sunni.** `branches.py` has `islam.sunni.shafii`, and the split
   between the two boxes is a logging habit: Lahj 100 Sunni and 0 Shafi'i, Al Maharah 40 and 0,
   against Aden 9 and 57 and Ibb 4 and 139.
3. **`refused`, `don't know` and `other` on the sect item stay Muslim**, composed to `Muslim,
   denomination not given` and `Muslim, other denomination` on `islam`, as Iraq.
4. **The one atheist is dropped**: one interview would draw about 14,500 people at the national
   rate. Egypt and Iraq dropped theirs.

## 5. What replaces the split-half

Two waves on two cards cannot give §14.16's early-against-late split, and `queue.md` asked for the
replacement to be decided before building. Two tests, both run on every build:

**Replication across fieldworks** (`ye.py::replication`). The Zaydi (wave III: Shia) share of the
Muslims who name a branch, per governorate, wave III against wave V: **Spearman +0.844** over 21,
against `spearman_null`'s exact bar **+0.3701** (exact p 2.1e-06); a 20,000-draw unit permutation,
which handles the seven tied zeros in each wave, gives p 5e-05; chi-square across governorates
p=2e-81 (III) and 9.3e-135 (V). Leave-one-out runs +0.818 (without Sa'dah) to +0.886 (without
Ma'rib) against +0.3805 at 20. Decides Sunni and Zaydi.

**PSU split-half inside wave V** (`lits.stability`, 240 PSUs of ten, each inside one governorate):
median Spearman over 400 random PSU halves against a 400-draw PSU regrouping null, with the
chi-square and the one-PSU cap as vetoes.

| answer | national (wave V weighted) | median rho | null 95th | p | verdict |
|---|---:|---:|---:|---:|---|
| Sunni | 56.86% | +0.947 | +0.250 | 0.0025 | own geography |
| Just a Muslim | 24.42% | +0.846 | +0.256 | 0.0025 | own geography |
| Zaydi | 18.19% | +0.930 | +0.265 | 0.0025 | own geography |
| Muslim, denomination not given | 0.43% | +0.459 | +0.313 | 0.0150 | under the 1% floor |
| Muslim, other denomination | 0.10% | -0.050 | +1.000 | 1.0000 | under the floor |

**The two sub-floor answers are drawn at their governorate shares anyway** (`ye.py::SAME_NODE`).
`ab.build` spreads a tail only into the room the carried answers leave, and 15 of 21 governorates
returned nothing but the three carried answers. Both sit on `islam` with `Just a Muslim`, so at the
node the map is identical to folding them in; asserted against the mapping.

**What neither test can see.** In eight governorates nobody was logged `Just a Muslim`: Al Bayda
0/70, Ta'iz 0/260, Al Jawf 0/60, Shabwah 0/60, Lahj 0/100, Ma'rib 0/40, Al Maharah 0/40, Ad Dali'
0/60. That reads as a team's logging habit, and a PSU split inside a governorate replicates a
habit perfectly. Wave V's interviewer column `E2001B` exists and is blank for all 2,400 Yemenis,
so it cannot be tested. The note says it.

**Lebanon's quota check** ran on the one pair (wave III's `Shia` read as the same box): 8
comparable units, 4 free cells, 0 identical, p=1.

## 6. Level against outside figures

Of wave V's branch-namers **24.2%** are Zaydi (weighted); as drawn, 23.4%. The State Department's
*2023 Report on International Religious Freedom: Yemen*, read directly: *"the U.S. government
estimates 65 percent of the population is Sunni and 35 percent Zaydi"*, and *"ACLED estimated in
2022 that 55 percent of Muslims is Shafi'i Sunni and 45 percent is Zaydi Shia"*. Neither is a
count. The survey reads well under both. `Just a Muslim` is highest in Sana'a (46.8%), Sana'a City
(43.9%) and Hajjah (36.2%), so the likeliest reading is highland Zaydis on the plain answer, and
the drawn Zaydi share is a floor. `ZAYDI_OF_NAMED_BAND` (10-50%) guards a re-release only.

## 7. Geography and population

`sources/ye_geo.py`, `sources/ye_grid.py`.

- **Boundaries: COD-AB** `yem_admin1.shp`, 22 governorates, `YE11`-`YE32`, CSO-provided, HCT
  approved October 2019, reviewed December 2024. CC BY-IGO.
- **Population: the Population Task Force's 2025 estimate** (CSO, UNFPA, IOM, OCHA),
  `yem_population_projection_2025_final.xlsx`, 333 districts, **34,879,018**: the CSO's 2025
  projection from the 2004 census, with IDPs updated from DTM round 39 (106 government districts),
  CCCM and UNHCR (190 de facto authority and 37 shared). CC BY. The CSO's unadjusted projection is
  in the same file and in `ye_lookup.csv`. Ma'rib is 1,771,017 against the CSO's 397,624 because
  1,639,142 of its people are displaced; everywhere else the two are within 0.90-1.12.
- **Witnesses on the join**: the 22 p-codes agree; every governorate's district sum equals the
  methodology note's printed Results table to the person (read off the PDF); all 335 COD district
  polygons fall inside the governorate their p-code names; the sparsest are Al Maharah and
  Hadramawt and the densest Sana'a City and Aden; **Kontur's per-governorate totals against the
  Task Force's, Spearman +0.870, 0 of 5,000 random pairings** (the two share no input).
- **The survey decode**: wave III on its code, wave V on its names (§8); held-out r = +0.960
  (V) and +0.938 (III) against the 2025 populations, 0 of 20,000 pairings. Ma'rib is sampled at
  0.23x its 2025 share, which is the displacement.
- **Placement: Kontur 400 m hexes** (2023-11), 76,350 keyed to governorate, ratio 0.984 to the
  Task Force nationally, 0.79-1.22 per governorate apart from Ma'rib (0.23x, the camps).
- **Kontur's density cap**: 69 blocks, all written to `kontur_cap.csv` (§9).

## 8. Two traps in the governorate labels

- **Wave III labels two governorates `Sana'a`**: code 10503 (Amanat al-Asimah, the capital, 110
  respondents) and 10513 (Sana'a governorate, 60). A name join merges the capital into the
  governorate with every total intact. Wave III is decoded on its code, which is the CSO order
  (10500 + n is `YE{n+10}`); the names disagree on exactly the 110 at 10503, and wave II's own
  labels for the same 21 codes name 10503 `Amanat al Asimah` and decode to the same order.
  `ye.py::decode_iii`.
- **Wave V's code is its own order**, 220001-220021, with the former YAR first and the six former
  PDRY governorates last; decoded by name, 220016-220021 are exactly those six, asserted.

## 9. Kontur's density cap, reviewed

`python kontur_cap.py ye` found 69 blocks at 46,200/km2. Each was paired with the nearest GeoNames
place with a population (download.geonames.org `YE.zip`, CC BY) and with the Task Force population
of the district its peak sits in.

- **real, 8**: Sana'a, Ta'iz, Al Hudaydah, Ibb, Sa'dah town and three blocks in Aden (Ash Shaykh
  Othman and Al Mansura, Crater, At Tawahi). Each within 1 km of the city's point except the Aden
  and Sana'a blocks (12.5 and 3.5 km), where Kontur is at the cap at the city point too, and each
  holding at most 1.4x the city's figure.
- **capped, 7**: Ar Rujum, Melhan and Hufash (Al Mahwit), Mazhar and Kusmah (Raymah), with no
  settlement GeoNames gives a population within 5 km and 0.47-0.98x their district's people on
  2-4 hexes; and the blocks at Al Mahwit town (6.9x the town) and Yarim (6.3x).
- **unreviewed, 54**: three at 5% or more that sit on real district towns GeoNames gives no
  population (Lawdar, Ja'ar, Al Qa'idah), and 51 under 5% of their governorate, not acted on as
  the registry's other countries. Most of those 51 are single hexes of about 31,000 people on
  highland villages, which look false; they were left to the existing rule rather than judged.

**Capping lowers a block to its 3 km ring's median, which in the highlands is tens of people per
km2.** The seven capped blocks keep 24 to 954 people of their 45,144 to 208,908, so the real town of
Yarim (about 33,000) and Al Mahwit town now draw almost nothing, their governorates' dots moving to
the rest of Ibb and Al Mahwit. The shares are uniform inside each governorate, so no religion moves
with them; a later review could mark those two `real` at a town-sized ceiling if `kontur_cap.py`
grows one.

**GeoNames is thin for Yemen**: 9,134 places with a population, and district towns such as Lawdar
and Ja'ar are not among them. A second list (OSM `place=town` with `population`) is where a later
review would start.

## 10. §14

`ask/answered/021-jo` confirmed Iraq's Sunni and Shia geography at governorate, and the ask's own
text says the ruling covers Yemen; Anita: *"leave as built fine"*. So no ask was filed. What is
drawn with a geography is the Zaydi highlands against the Shafi'i south, west and east, which is
the most described religious fact about Yemen. The groups §14.4 rule 2 is about (Bahá'ís, whom the
Houthis have detained; Christians; the last known Jew, in Houthi detention since 2016, per the
State Department report) are in neither wave and are drawn nowhere.

## 11. Terms

Arab Barometer: free for analysis, no redistribution clause (§11af, `sources/iq.md` §9). COD-AB:
CC BY-IGO. Task Force estimate: CC BY, *"for United Nations humanitarian program cycle planning
purposes only"* is its caveat, not a use restriction. Kontur: CC BY. GeoNames: CC BY, used only in
the review.

## 12. On disk

```
data/raw/ye/yem_admin_boundaries.shp.zip             COD-AB, 5.3 MB
data/raw/ye/yem_admin_boundaries.xlsx                COD-AB gazetteer
data/raw/ye/yem_population_projection_2025_final.xlsx Task Force 2025, by district
data/raw/ye/yem_population_methodology_2025_en.pdf   its methodology note
data/raw/ye/kontur_population_YE_20231101.gpkg(.gz)  the placement grid
data/geo/ye/ye_governorates.gpkg                     22 polygons
data/geo/ye/ye_lookup.csv                            22 rows: pop, CSO, IDPs, under 15, area
data/geo/ye/ye_hexes.gpkg                            76,350 hexes
data/normalized/ye.csv                               105 rows, 34,803,293 people
```

## 13. Shared code changed

`sources/arabbarometer.py::load` gained `raw=` (an undecoded design column: wave V `psu`, wave III
`bid`) and `blank_weights={wave: reason}` (`_fill_blank_weights`: a named wave's blank weights take
their PSU's mean; stops on a stale entry). **32 wave V Yemenis have no weight**, every one with no
recorded gender, 29 in five PSUs of Hajjah and Dhamar, 22 logged Zaydi; dropping them would take a
tenth of both governorates' interviews out of the Zaydi column. Egypt, Jordan and Iraq were rebuilt
after the change and `eg.csv`, `jo.csv` and `iq.csv` are byte-identical (md5).

## 14. Review, 2026-09-15 (`d743fc47-rev6`)

Read the mapping, `ye.csv`, the entry, ask 021, the `arabbarometer.py` diff and the wave V `.sav`
(scratch script, not kept). `check_md` clean, `built_countries --check` OK, `check_rollup ye` 0
orphaned (every row modelled). One screenshot: dots over the highlands, the Tihama, Aden, the
Hadramawt coast and Wadi and a dense block at Ma'rib, nothing on Socotra, nothing in the sea.
Sunni, Zaydi and plain Islam are all greens, so the highland split is hard to see at the default
view, as in Iraq. No ask, nothing rebuilt.

- **One figure in `note_public` is off.** *"8.27 million people are on the plain Islam colour"* is
  `Just a Muslim` alone. `Muslim, denomination not given` (142,578) and `Muslim, other
  denomination` (36,679) are on `islam` too, so the colour holds **8,450,720**, 8.45 million.
  Not edited. Every other figure recomputes off `ye.csv`: 58.0% and 17.7%, Sa'dah 62.5, Amran
  51.3, Dhamar 39.4, Hajjah 34.6, Sana'a City 28.1, 23.4% of branch-namers, Just a Muslim 46.8,
  43.9 and 36.2, Ma'rib 1,771,017, both lists of eight governorates. (Trivial: `REVIEW["Zaydi"]`
  says 0.12% of the southern and eastern nine where §4 and the note say 1 in 950.)
- **The blank-weight fill holds up against the raw file.** The 32 answered every other item
  normally (age, education, work, income, the Q101 economy item); only `Q1002` gender is empty.
  20 of them are housewives against 35% of Yemen's sample, so most are probably women whose
  gender was not keyed. In their ten PSUs the weighted respondents are 24% code 14 and the 32 are
  69% (22), so they matter: dropped, Hajjah's unweighted Zaydi share goes 33.8% to 30.3% and
  Dhamar's 37.3% to 32.4%. Dropping them would push the map further under the outside estimates
  the note already calls a floor, so filling is the less biased choice. Agreed.
- **Shared code:** `_fill_blank_weights` runs only for a wave named in `blank_weights`, and the
  `w` column now takes the same `wv` the assertion checked, which equals the old expression
  whenever nothing is filled. Nothing else moves.
- **Mapping: agreed**, and it matches Iraq box for box (Shafi'i to Sunni, `Alawi` to Shia, Just a
  Muslim and the refusals on `islam`). No new node; not minting `islam.shia.zaydi` is right.
- **Gap: honest.** Socotra's 0.22% is hand-written correctly (in no table). The biased part is the
  eight governorates with no `Just a Muslim` (Sunni up there, Zaydi down in the highlands), and the
  note says both.
- **§14: agreed.** Ask 021's ruling text names Yemen. No group §14.4 rule 2 is about is placed.
