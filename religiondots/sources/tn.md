# Tunisia (`tn`)

Session `d743fc47-tn`, 2026-09-15, under a supervisor. Built on Anita's Maghreb rulings
(`ask/RULINGS.md`, 2026-09-15 and 2026-09-16) and the scouting in `sources.md`
§maghreb-2026-09-16. Code: `sources/tn.py`, `sources/tn_geo.py`, `sources/tn_grid.py`,
`taxonomy/tn2024.py`, `countries/tn.py`. No ask filed.

## 0. Outcome

Drawn at 24 governorates, 11,972,169 people (RGPH 2024, 6 November 2024), at **one national share
and mix in every governorate**, because nothing finer passes. Every row `modelled`.

| | people | share |
|---|---:|---:|
| Muslim (`islam`) | 11,898,062 | 99.381% |
| No religion (`unaffiliated`) | 30,149 | 0.252% |
| Other (`other.tn`, new node) | 28,004 | 0.234% |
| Christian (`christianity`) | 15,954 | 0.133% |

## 1. Sources

- **Arab Barometer**, waves V (2,400 Tunisians), VI-1 (1,005), VI-2 (1,002), VI-3 (1,200), VII
  (2,400) and VIII (2,406), `data/raw/arabbarometer/`. `Q1012` religion, `Q1` governorate, `WT`,
  `Q13` urban/rural (V, VII, VIII), `PSU` (V, VII, VIII), `Q1012A` / `Q1012A_MUSLIM` /
  `Q1012A_CHRISTIAN` follow-ups. Wave VI carries no PSU. No wave's public file carries an
  interviewer or team column for Tunisia (checked in V, VII, VIII; V's `E2001B` is one value).
- **Afrobarometer** R5-R9 (5,959 answers), `data/raw/afrobarometer/`, a witness only (§4). R5 and
  R6 code the 24 governorates; R7 to R9 code 7 regions (Great Tunis, North East, North West,
  Centre East, Centre West, South East, South West).
- **INS, *RGPH 2024, Bilan Démographique*** (May 2025,
  `ins.tn/sites/default/files-ftp3/files/2025-05/Bilan%20D%C3%A9mographique_0.pdf`, 5,037,895
  bytes, `data/raw/tn/ins_rgph2024_bilan_demographique.pdf`). p.15: population by governorate at
  the 1994, 2004, 2014 and 2024 censuses with district subtotals; p.7: foreign population by census;
  p.21: Tunis's 2024 density. A second copy on INS's event page (`Bilan_Démographique.pdf`,
  6.73 MB) was not compared.
- **COD-AB `cod-ab-tun` v01** (OCHA, source INS, valid 2022-11-15, reviewed 2024-12-19),
  `data/raw/tn/tun_admin_boundaries.geojson.zip`. ADM2 is the 24 governorates; ADM1 is the six
  old economic regions; ADM3 264 delegations; ADM4 2,084 imadas.
- **Kontur population `TN` 20231101**; **GeoNames `TN`** (CC BY 4.0).
- **Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), Tunisia
  2020 as an outside level: Muslims 99.304%, unaffiliated 0.441%, Christians 0.247%, Jews 0.008%.
- Opened and not used: INS's 2024 thematic workbooks (`Indicateurs de la Migration_Final.xlsx`,
  `Migration Externe_1.xlsx`, `Pourcentage Etat Matrimonial par Milieu et Sexe_Final.xlsx`,
  `Nombre de ménages par Secteurs.xlsx`, from
  `ins.tn/enquetes/recensement-general-de-la-population-et-de-lhabitat-2024-analyses-thematiques`),
  §8. The *Présentation RGPH 2024* PDF has no text layer.

## 2. The rulings applied

- 2026-09-15: draw the 99.x% Muslim countries with the best evidence for where non-Muslims are.
- 2026-09-16: foreigners only where the state publishes a count by province (§8: it does not); no
  IOM DTM; no presence rings for Jews or Ibadis.
- §14: Tunisians who leave Islam or convert face social pressure. No governorate and no urban or
  rural stratum is drawn apart, so the map places no community; not escalated.

## 3. The survey: pool, card, labels

**The card** (`tn.py::card`). Waves II, III and IV offer no box for having no religion; V offers
`Atheist`, VI-1 to VIII `No religion`. 28 of the 65 non-Muslim answers in V-VIII are that box, so
II-IV are in `OMIT`. III and IV also record no non-Muslim among 2,399 Tunisians; II has 4 Christian
and 2 Jewish answers in 1,196. The wave IV blank weight the scouting found is therefore not read.
Wave I has no subnational column.

**Spellings.** `refused` (V) and `Refused` (VI-1) merge into `Refused to answer` (38 dropped in
all); V's `other` into `Other`; VI-3's `Something else: SPECIFY_______` into `Other` (3); V's
`Atheist` into `No religion` (7).

**Follow-ups.** No non-Muslim answer carries a Muslim follow-up (`contradictions`): V's are `don't
know`, `other` or `refused`; the two Christians asked in VII and VIII give `Catholic` and `Just a
Christian`. No Jewish answer is in the pool; V's card offers `Jewish` and nobody chose it, and
VI to VIII have no Jewish box.

**Geography, and wave VI's two `Jendouba`s.** V, VII and VIII label all 24 governorates, and
`Q1` minus 210000 is the position in INS's official order in every row. VI-1, VI-2 and VI-3 code
21001-21024 in another order and label both **21009 and 21010 `Jendouba`**; there is no `Kef`
label. Decoded by name, Le Kef would merge into Jendouba with every total intact. VI is decoded on
the code (`VI_CODES`), every other label agreeing, and 21009 is Le Kef: it holds 23, 23 and 28
respondents against 21010's 37, 37 and 44, a ratio of 0.63 against 0.64 for Le Kef over Jendouba
in V, VII and VIII (1.58 if swapped); VI's per-governorate allocation ranks with the other waves'
at Spearman +0.986 (`decode`, asserted). Two VI respondents refused a governorate (both Muslim)
and leave the geography. Trap added to `playbooks/arabbarometer.md`.

**Held-out:** r = +0.996 between respondents and RGPH 2024 over 24 governorates, best of 20,000
pairings +0.885. Quota test p = 1.

**Pool as drawn:** 10,413 answered, 10,375 after refusals, 10,373 with a governorate; 65
non-Muslim (no religion 28, other 25, Christian 12). `NOTE`, asserted.

**By wave** (weighted non-Muslim share): V 0.55%, VI-1 0.85%, VI-2 1.01%, VI-3 0.71%, VII 0.83%,
VIII 0.17%. Pooled 0.619%.

## 4. What stands apart: nothing

- **Governorates** (`standouts`): each of the 12 with two or more non-Muslims against the rest,
  in both halves (V to VI-3; VII and VIII), Bonferroni bar P < 0.0042. None. Tunis 8 against 4.3
  in the first half (P 0.058) and 1 against 2.2 in the second; Le Kef 0 then 4 (P 0.001);
  Siliana 0 then 3; Sousse 1 then 5 (P 0.010), which is §5's cluster.
- **Greater Tunis** (`greater_tunis`; Tunis, Ariana, Ben Arous, Manouba), the queue row's "slight
  lean to Tunis": 18 against 16.2 expected, P 0.35; halves P 0.093 and 0.94; weighted 0.614%
  against 0.621%. The Afrobarometer reads 12 against 6.5 (P 0.015), on a region its R7-R9 frame
  names directly. The rule is both halves of the Arab Barometer and the Afrobarometer; it fails.
- **Towns** (`urban_test`, Morocco's bar from `sources/ma.py`, unchanged): Arab Barometer V, VII,
  VIII 25 urban of 35 against 23.8 expected, P 0.41 (weighted urban 0.558%, rural 0.432%);
  Afrobarometer 21 of 26 against 17.7, P 0.12. Fails in both.

These were measured in a probe before `tn.py` was written, so none is pre-registered here; each
outcome is asserted, so a re-release that changes one stops the build.

**Afrobarometer as a level witness.** 26 non-Muslims in 5,959, weighted 0.41%; by round R5 0.00%,
R6 0.24%, R7 0.69%, R8 0.64%, R9 0.49%. Atheist 6, None 5, Agnostic 3, Hindu 3, Other 2, Jewish
2, traditional 2, Adventist 1, Christian 1, Bahá'í 1. The level agrees with the Arab Barometer
within the noise; the Afrobarometer's card has no plain Other and splits no religion three ways.

## 5. Wave VII's `Other`

14 answers, **10 in Le Kef (3), Siliana (2) and Sousse (5)**, in PSUs 139, 140, 143, 145, 147,
150, 156, 157, 158 and 164; those three governorates hold 1 non-Muslim in 580 interviews in waves
V to VI-3, and 242 of VII's 2,366 respondents. The PSUs are numbered consecutively by governorate,
so the run is consistent with one fieldwork team, but no file carries an interviewer column to
test it (the Uzbek `Other (vol.)` refusal rested on an interviewer id, `sources/uz.md`).
**Kept as recorded**: nothing is placed by it, and dropping answers on a pattern alone would be a
new rule. Without the ten the share is 0.519% and the mix Christian 25.7%, no religion 48.6%, other
25.7% (`vii_cluster` prints it every build).

## 6. Units and population

`tn_geo.py`. The key is INS's governorate code: COD's `adm2_pcode` is `TN` plus it. Witnesses:
the authored table covers COD's 24 p-codes and p.15's 24 rows, with COD's `adm2_ref_name` under
each p-code the expected one; each district subtotal on p.15 equals its governorates in 1994,
2004, 2014 and 2024, and the Total equals the districts except in **2014, where it is 278 above
them** (10,982,754 printed, the official figure, against 10,982,476; pinned); COD's ADM1 parent is
the economic region the INS code's first digit names; Tunis is 263 km2 on COD against 288 km2 from
p.21's density (0.91x). The rank witness is in §7.

Drawn unscaled on the 2024 count, which is within a year of every wave but VIII's end.

## 7. Placement

`tn_grid.py`. Kontur `TN` 80,282 hexes, 12,467,391 people; 764 hex centroids outside every
governorate (128,696 people, the coast and the Djerba causeway), 737 snapped within 1 km, 27 hexes
and 454 people dropped. Kontur over the census 1.041. **Join witness:** Spearman between Kontur
and the census over 24 governorates +0.988, 0 of 20,000 shuffles reach it (best +0.806). Per
governorate over the national ratio: Kébili 0.90 to Manouba 1.22 (Ariana 1.21); Kontur puts more
of Greater Tunis in its outer governorates than the census. GeoNames seats of 50,000+ (18): the
lowest within 5 km are Kairouan 0.49 and Tunis 0.57, no hole under 0.10. `kontur_cap.py tn`: no
block at the cap.

## 8. Foreign residents: no layer

INS counts **66,349 foreign residents in 2024 (0.55%)** and 53,490 in 2014 (Bilan p.7), nationally
only. The 2024 migration workbook tabulates **non-Tunisian arrivals from abroad between November
2019 and November 2024**, 14,553, by governorate of residence (`imm_gouv_sexe`: Tunis 37.3%,
Ariana 18.2%, Sousse 11.3%, Nabeul 5.9%, Médenine 5.8%, Sfax 5.3%) and by country of provenance
(`immi_sexe_pays`: other African countries 33.7%, other Arab countries 15.2%, Libya 13.2%, Algeria
9.1%, France 8.0%). That is a five-year flow of 22% of the stock, by provenance rather than
nationality, so it is not the count by province the ruling asks for. **Not drawn**; the foreigners
stay inside each governorate's census count and take the survey's shares, which the note says.

## 9. Not drawn, and REOPEN

- **Foreigners by governorate.** `dataportal.ins.tn`'s query tool was not explored; RGPH 2014 V5
  tables 3.2.5-3.2.7 (arrivals 2009-14) per the scouting. A stock by governorate, with a nationality
  mix, would allow Morocco's construction.
- **Jews.** No Jewish answer in the pool (II's two are out with the wave); Afrobarometer 2 of 5,959.
  Djerba's community (about 1,000 to 1,300, a community claim not opened) and the ruling is no rings.
- **Ibadis.** 7 Ibadi answers on the follow-up in VII and VIII, 4 of them in Médenine, Djerba's
  governorate; no magnitude, every Muslim on `islam`.
- **Urban and rural.** INS's 2024 workbooks give households by imada and milieu
  (`Nombre de ménages par Secteurs.xlsx`), which would carry an urban split or a placement check if a
  later wave passed the urban bar; not needed now.

## 10. Calls someone might reverse

1. One national share everywhere; nothing passes at governorate, Greater Tunis or urban.
2. Pool V-VIII, including wave VII's ten clustered `Other` answers (without them 0.52%).
3. II-IV out on the card.
4. Wave VI's 21009 decoded as Le Kef on sample size.
5. No foreigner layer; the 66,349 foreigners take the survey's shares.

## 11. Review, 2026-09-15 (cb8b206e-rev1)

Full pass. `check_md` clean, `built_countries --check` ok, rollup clean (all modelled).

- **Figures.** Every `note_public` figure recomputes off `tn.csv`: 11,972,169, 0.619%
  non-Muslim, mix 40.7 / 37.8 / 21.5, grain 498,840. Every governorate sits at 0.6187-0.6192%.
- **Mapping and node.** Same box-to-node mapping as `dz2022` and `ma2024`. `other.tn` is a
  per-country residual like `other.dz` and `other.ma`, so agreed.
- **§14.** No ask, agreed: nothing is placed below the national level.
- **Screenshot.** Clean: dots on the coast, the Sahel, Sfax and the southern oases, none in the
  sea or on the erg.
- **Soft, not edited: the note's Pew sentence.** It sets Pew's 0.25% Christian against 0.13%
  drawn and puts the difference down to what people will tell an interviewer. But Pew counts
  everyone living in Tunisia, and call 5 draws the 66,349 foreign residents at the Tunisian
  survey's 0.62%.
  - §8's 2019-24 arrivals are 33.7% from other African countries and 8.0% from France. If the
    stock looks anything like that, the foreigners alone could hold something like 10,000
    Christians, the same order as the 15,954 drawn. That is a guess from a flow, not a count.
  - The error is one-way. It understates Christians most in Tunis, Ariana and Sousse, where 67%
    of the arrivals live.
  - Not rebuilt: nothing gives a stock by nationality.
  - Suggested wording: the note says part of Pew's gap is foreign residents, as Libya's does.
- **For the non-national `gap_share` sweep.** `runlog.md` 637 names `tn`, but Tunisia's
  foreigners are drawn, inside each governorate's census count, so they are not a hole. The
  sweep should leave `tn` without a `gap_share` unless the construction changes.
- **Applied 2026-09-15 (cb8b206e-gap, ask 033 sweep).** The note's Pew sentence now says Pew covers
  everyone living in Tunisia, and that part of the difference is the 66,349 foreign residents drawn
  at the Tunisians' shares, part what people will tell an interviewer. No `gap_share`, as above.
