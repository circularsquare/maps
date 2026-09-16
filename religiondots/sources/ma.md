# Morocco (`ma`)

Session `d743fc47-ma`, 2026-09-15, under a supervisor. Built on Anita's Maghreb rulings
(`ask/RULINGS.md`, 2026-09-15 and 2026-09-16) and the scouting in `sources.md`
§maghreb-2026-09-16. Code: `sources/ma.py`, `sources/ma_geo.py`, `sources/ma_grid.py`,
`taxonomy/ma2024.py`, `countries/ma.py`. Ask filed: `ask/031` (Western Sahara).

## 0. Outcome

Drawn at 73 units (69 provinces and prefectures, 4 Western Sahara units), 36,828,330 people
(RGPH 2024). Every row `modelled`.

| | people | share of Morocco |
|---|---:|---:|
| Moroccans, Muslim (`islam`) | 36,560,447 | 99.27% |
| Moroccans, Christian (`christianity`) | 78,753 | 0.214% |
| Moroccans, no religion (`unaffiliated`) | 37,010 | 0.100% |
| Moroccans, other (`other.ma`, new node) | 3,968 | 0.011% |
| Foreign residents, 20 nodes | 148,152 | 0.402% |

Foreign residents as drawn: Muslim 78,286 (52.8%), Catholic 30,383, Protestant 17,427, no religion
16,665, Orthodox 1,445, Hindu 1,055, the rest under 1,000 each. Non-Muslims as drawn, both halves:
about 190,000, 0.51% of the country.

## 1. Sources

- **Arab Barometer**, waves V (2,400 Moroccans, fielded Oct-Dec 2018), VI-1 (1,002), VI-2 (1,005),
  VI-3 (1,201), VII (2,404, Mar-Apr 2022) and VIII (2,411, Dec 2023-Jan 2024),
  `data/raw/arabbarometer/`. `Q1012` religion, `Q1` region, `WT`, `Q13` urban/rural (V, VII, VIII),
  `PSU` (V, VII, VIII), `Q1012A_MUSLIM`/`Q1012A_CHRISTIAN` follow-ups (VII, VIII). The technical
  reports for V, VII and VIII (`ABV_Methods_Report-1.pdf`, `AB7_Technical_Report.pdf`,
  `AB8_Technical_Report_Nov_2024.pdf`, opened 2026-09-15) give the target population as **citizens
  aged 18 and above**, the frame as the 2014 census, 12 regions by urban and rural.
- **Afrobarometer** R5-R9 (5,981 answers), `data/raw/afrobarometer/`, a witness only (§4).
- **HCP, *Population légale ... RGPH 2024***, Excel (`hcp.ma/file/242341/`, 166,693 bytes, saved as
  `data/raw/ma/hcp_population_legale_2024.xlsx`). Moroccans, foreigners, population and households
  for the nation, 12 regions, 75 provinces, every commune, and the urban and rural part of each
  region and province. Footnote on the starred communes (Lagouira, Aghouinite, Zoug): figures
  gathered from local administration because of seasonal mobility.
- **HCP, *Les résidents étrangers au Maroc, analyse issue du RGPH de 2024*** (November 2025,
  `hcp.ma/file/246070/`, 2,040,089 bytes, `data/raw/ma/hcp_residents_etrangers_2024.pdf`). Regions of
  nationality (p.4) and the leading nationalities within each (annex tables 1-6).
- **COD-AB `cod-ab-mar` v01** ADM2, 69 features (HCP lineage); **COD-AB `cod-ab-esh` v01** ADM1, 4
  features (GADM lineage). Natural Earth 10m disputed areas (on disk) for B19, B28, B60, B61.
- **Kontur population `MA` and `EH` 20231101**; **GeoNames `MA` and `EH`** (CC BY 4.0).
- **Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), per
  nationality and regional totals for the foreign half, and Morocco 2020 as an outside level:
  Muslims 99.68%, unaffiliated 0.131%, other 0.102%, Christians 0.085%, Jews 0.006%.

## 2. The rulings applied

- 2026-09-15: draw the 99.x% Muslim countries with the best evidence for where non-Muslims are.
- 2026-09-16: foreigners on the country-wide nationality mix where the non-Muslim foreigners are
  small or predictable (§7); no IOM DTM; no presence rings for Jews or Ibadis.
- §14: Moroccan converts to Christianity face social and legal pressure (proselytism is an offence).
  Nothing finer than urban and rural within a province is drawn, and no region stands apart, so the
  map places no Christian community; not escalated. Western Sahara is a different §14 question and
  is `ask/031`.

## 3. The survey: pool, card, labels

**The card** (`ma.py::card`). III and IV offer no box for having no religion; V offers `Atheist`,
VI-1 to VIII `No religion`. 15 of the 36 non-Muslim answers in V-VIII are that box. III and IV also
sample the 16 regions abolished in 2015, so both are in `OMIT`. II has no Morocco; I no region.

**Spellings.** `refused` (V) and `Refused` (VI-1) merge into `Refused to answer` (25 dropped).
VI-3's card has `Something else: SPECIFY_______` and no plain `Other`, so it is recoded to `Other`
(one answer). V's `Atheist` has no Moroccan answer, so there is nothing to recode.

**Geography.** 17 `Q1` labels fold to the 12 regions (`SPELLINGS`); 16 respondents in VI-2 and VI-3
answered `Don't know` or `Refused` for a region, all Muslim, and leave the geography. The code
witness: `Q1` minus 130000 (V, VII, VIII) or 13000 (VI) is HCP's region code in every row. Held-out:
r = +0.997 between respondents and RGPH 2024 Moroccans over 12 regions, best of 20,000 pairings
+0.932 (Dakhla-Oued Ed-Dahab sampled at 2.19x, the southern oversample). Quota test p = 1.

**Follow-ups.** No non-Muslim answer carries a Muslim follow-up and no Jewish answer is in the pool,
so Algeria's contradiction check has nothing to drop.

**Pool as drawn:** 10,423 answered, 10,398 after refusals, 10,382 with a region; 36 non-Muslim
(Christian 19, no religion 15, other 2). `NOTE`, asserted.

**By wave** (weighted non-Muslim share): V 0.00%, VI-1 0.44%, VI-2 0.28%, VI-3 0.06%, VII 0.25%,
VIII 0.82%. Wave VIII holds 12 of the 19 Christians, 5 of them in the southern oversample
(Laâyoune-Sakia El Hamra 3, Guelmim-Oued Noun 2), all `Just a Christian`. Waves V-VII: 7 Christians
in 7,976 answers. Pooled all the same, as Algeria.

## 4. What stands apart: no region, but towns

**Regions** (`standouts`): each region with two or more non-Muslims against the rest, in both halves
(V to VI-3, VII and VIII), Bonferroni over 8 (bar P < 0.0063). None: the lowest is
Laâyoune-Sakia El Hamra at 4 against 1.3 in VII-VIII (P 0.037) and 0 in the first half.

**Urban and rural** (`urban_test`). Bar written before the test: P < 0.05, exact within wave, in the
Arab Barometer waves carrying `Q13` AND in the Afrobarometer (`URBRUR`), and the largest PSU under
half the urban non-Muslims.

| | urban non-Muslims | expected | P |
|---|---:|---:|---:|
| Arab Barometer V, VII, VIII | 23 of 27 | 17.4 | 0.016 |
| Afrobarometer R5-R9 | 16 of 18 | 11.2 | 0.014 |

Largest PSU: 2 of 23. Weighted in the Arab Barometer: urban 0.463%, rural 0.185%, ratio 2.50.
Wave VI has no urban and rural stratum (`Q13A` in VI-2 and VI-3 is a self-reported "big city,
village or rural area" and is not used).

**Drawn.** The pooled level (0.326%, all six parts) is split at that ratio over RGPH 2024's
Moroccans, 62.6% urban: **0.421% of urban and 0.168% of rural Moroccans** in every province. The
non-Muslims are at the pooled weighted mix everywhere: Christian 65.8%, no religion 30.9%, other 3.3%.

**Afrobarometer as a level witness.** 18 non-Muslims in 5,981 (None 6, Agnostic 4, Atheist 3,
Christian 3 with a Calvinist, Jewish 1), weighted by round R5 0.40%, R6 0.18%, R7 0.11%, R8 0.05%,
R9 0.70%. The level agrees with the Arab Barometer; the mix does not (Christians 3 of 18 against 19
of 36). The drawn mix is the Arab Barometer's, as Algeria's.

## 5. Units and population

`ma_geo.py`. HCP's 75 provinces add to its 12 regions and urban plus rural to totals, for Moroccans
and foreigners separately (asserted). Two `Préfecture d'arrondissement` rows (Hay-Hassani,
Aïn-Chock) in the commune section are sub-prefectures of Casablanca and not provinces.

- **COD-AB Morocco's 69 provinces pair 1:1 by name** with HCP's 69 north of 27°40'N, six spellings
  aliased (Fquih Ben Saleh, El Kelaat Es Sraghna, Rhamna, Mohammedia, Tangier Assilah, Taroudant).
- **COD-AB Morocco leaves out all of Tarfaya province**, including Tarfaya town (27.94°N) and
  Akhfennir (28.09°N), which lie in neither COD file. So Laâyoune and Tarfaya are one unit: ESH
  `Laayoune` plus the strip's hexes (§6). Tarfaya's Daoura and El Hagounia are inside ESH `Laayoune`
  by GeoNames.
- **Oued Ed-Dahab and Aousserd are one unit** (ESH `Oued el Dahab`; ESH has no Aousserd line).
- **Al Mahbass** (Assa-Zag, 19,139 people in 170 households) is inside ESH `Es Semara` by GeoNames
  (B19), so it moves there. Assa and Zag are exactly Assa-Zag's urban population, so Al Mahbass is
  rural (asserted). Labouirat, Touizgui and the two Aouint communes matched no GeoNames point and
  stay in COD's Assa Zag.
- **Western Sahara**, 690,132 people in the four ESH units, cut to Natural Earth's B19 (`Admin. by
  Morocco`); B28, east of the berm, is left out. `ask/031`.
- **Ceuta and Melilla** cut out of COD's polygons (7.1 and 0.9 km2 overlap).

## 6. Placement

`ma_grid.py`. Kontur `MA` (37.5M) and `EH` (0.5M), 2,956 hexes in both kept once; 37,487,537
placed against 36,828,330 counted (ratio 1.018); rank join rho +0.970 against a best shuffle of
+0.360.

- No unit: Tarfaya strip 237 hexes, 14,324 people, kept for Laâyoune (reaches GeoNames' Tarfaya and
  Akhfennir, asserted); east of the berm 888 hexes, 9,890, dropped; Ceuta and Melilla 20 hexes,
  22,737, dropped; hexes Algeria's own layer draws 56, 1,175, dropped; within 1 km of a unit 597
  hexes, 175,373, snapped (the Atlantic coast); 58 hexes, 23,595 people, dropped otherwise (not
  located).
- **Three towns Kontur has lost**, found by the rule (unit under 0.5 of its census share, then the
  main municipality under half of HCP's commune count within 5 km): Smara 4,258 against 56,607
  (0.07), Tan-Tan 9,094 against 76,134 (0.12), Assa 5,124 against 15,601 (0.32). A 3 km disc on each
  GeoNames point takes the shortfall (53,362, 68,403, 10,756). After: every unit 0.59 (Nouaceur) to
  1.26 (Chefchaouen), and all 22 GeoNames seats of 50,000+ hold a tenth or more within 5 km.
- **The berm garrisons.** HCP counts large collective households in communes named for places on or
  beyond the berm: Al Mahbass 19,139 in 170 households, Touizgui 6,787 in 177, Tichla 6,735 in 41,
  Tifariti 5,728 in 38, Haouza 5,197 in 78, Amgala 4,102 in 83. Kontur has none of them; their dots
  go to the unit's towns.
- `kontur_cap.py ma`: 14 blocks. 13 `real` (city cores: Tangier 2, Casablanca 2, Tétouan, Marrakech
  2, Rabat, Salé 2, Fès 2, Tiflet); 1 `capped`, 2 hexes and 53,921 people 17 km east-south-east of
  Laâyoune's centre in open desert.

## 7. Foreign residents

**The small-or-predictable test** (Anita, 2026-09-16). About 70,000 of the 148,152 foreigners are
non-Muslim as modelled, against about 120,000 non-Muslim Moroccans as drawn, so they are not small.
They are placed by the census itself, per commune, and 95% of foreigners live in towns (study p.8);
what is not published is whether the mix differs between places. The error is bounded by each
unit's foreign count (0.4% of the country; the largest shares are Rabat prefecture 2.8% and Dakhla
2.9%), and the two large non-Muslim groups (French, Ivorians) live in the same large cities as the
Muslim ones (Senegalese). Judged to hold; not filed.

**The mix.** Regions of nationality: sub-Saharan Africa 59.9, Europe 20.3, Middle East outside the
Maghreb 7.3, Maghreb 6.0, Asia 4.1, North America 1.8; within each, the annex tables (Senegal 30.8,
Côte d'Ivoire 28.9, Guinea 8.0, Mali 4.1, Congo 3.9; France 68.2, Spain 7.0, Italy 4.0, Belgium 3.2,
Germany 2.7; Syria 41.4, Egypt 13.6, Saudi Arabia 9.9, Palestine 7.6, Iraq 6.7; Mauritania 31.7,
Algeria 31.4, Tunisia 28.9, Libya 7.9; China 24.8, Philippines 21.5, Türkiye 15.0, India 11.7, South
Korea 5.9; United States 75.6, Canada 24.4). Cameroon's 1.9% of all (text p.5) comes out of "other
sub-Saharan". Seven shares the text prints are checked against the tables (asserted within 0.1).
Each region's "other" takes Pew's regional total; the 0.7% outside every region is spread over the
rest. Muslim branches are folded to `islam` (`taxonomy/ma2024.py` REVIEW).

## 8. Not drawn, and REOPEN

- **Jews.** No Jewish answer in the Arab Barometer pool; 1 in the Afrobarometer. Community figures
  (about 3,000, Casablanca about 1,000) are claims not opened, and the ruling is no rings.
- **Ibadis and Sufi orders.** Afrobarometer: Qadiriya 2, Tijaniya 1 (R8).
- **Foreigners' nationality by province.** The 2004 study lists province tables by nationality group
  not in its PDF (`hcp.ma/file/231897/`); HCP's 2024 dashboard has no nationality. Not searched further.
- **The Christian level.** Pew's 0.085% for the whole country is below the Moroccan Christians drawn
  here alone (0.215% of Moroccans). No independent count was opened.

## 9. Calls someone might reverse

1. Pool V-VIII, including wave VIII's 12 Christians; without VIII, Christians are about 0.09%.
2. The urban and rural split (bar pre-registered, passed in two surveys); without it, one national
   share of 0.326%.
3. The Arab Barometer's mix (65.8% Christian) over the Afrobarometer's (3 of 18).
4. Western Sahara drawn inside Morocco (`ask/031`).
5. The foreign layer on a national mix, and Muslim branches folded to `islam`.
6. Discs on Smara, Tan-Tan and Assa; the Laâyoune desert block capped.

## 10. Review, 2026-09-15 (`d743fc47-rev14`, full pass)

`check_md` clean, `built_countries --check` ok, `check_rollup ma` all `modelled` (36,828,330), none
orphaned. Screenshot at Morocco's bbox: dots follow the northern and Atlantic cities and the Souss,
Laâyoune and Dakhla carry thin dots west of the berm, nothing east of it or in the sea.

- **Figures: agreed.** Off `ma.csv` and `ma_foreign.csv`: 36,680,178 Moroccans, 119,731 non-Muslim
  (0.326%), mix 65.8/30.9/3.3, each province between 0.178% and 0.421%; 148,152 foreigners, 78,286
  on `islam` and 49,529 on Christian nodes; 690,132 in the four EH units.
- **Mapping, `other.ma`, the foreign Muslim fold: agreed**, as `dz` and `tn`.
- **Western Sahara** is built as ask 031 describes it. Not reopened.
- **Wave VIII's weight leaves the southern oversample in place.** Weighted sample share over RGPH 2024
  population share (Moroccans, `ma_pieces.csv`), Guelmim-Oued Noun, Laâyoune-Sakia El Hamra,
  Dakhla-Oued Ed-Dahab: wave VIII 3.54, 4.04, 6.07 (unweighted 3.54, 4.04, 6.12); wave VII 1.05,
  0.88, 0.78, so VII's `WT` does correct it; wave V 2.18, 1.78, 1.23. Five of wave VIII's 12
  Christians are in those three regions, one at `w` 3.54, the largest weight on any non-Muslim in the
  wave. Post-stratifying each wave to the 2024 regions takes wave VIII from 0.819% to 0.649% and the
  pooled level from 0.326% to 0.287% (waves by sample size). So the drawn level leans high by about an
  eighth, about 14,000 people, one way, and the lean falls on Christians. Small against §9's call 1
  (without VIII, Christians are about 0.09%), but it is a bias rather than noise. Not rebuilt; whether
  to post-stratify is the builder's call.
- **`note_public`, not edited:**
  - It leaves out half of the Maghreb brief (`queue.md`, "Maghreb reopened", survey shares:
    "`note_public` says no census asks, and that converts may not say so to an interviewer"). §2
    records the pressure on converts, and Algeria's note carries the point. One clause after the Pew
    sentence would do.
  - "Jews and Ibadis are not drawn": nothing records an Ibadi answer from Morocco (§8 has Qadiriya 2
    and Tijaniya 1; `queue.md` puts the region's Ibadis in the M'zab, Djerba and the Nafusa), and the
    sentence goes on to explain only Jews. It reads as carried over from Tunisia's note; "Jews are not
    drawn" would be exact.
