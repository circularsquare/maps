# Togo — the 2022 census's national religion table on six units, patterned by the Afrobarometer

**Drawn 2026-09-15** (session `d743fc47-tg`). 6 units, 13 nodes, 8,095,498 people (7,823,457 drawn),
every row `modelled`. Liberia's construction (`sources/lr.md` §3): an IPF of the survey's pattern to
two exact margins from the same census. No ask filed. §5 lists the calls worth a second look.

- `sources/tg_geo.py` -> `data/geo/tg/tg_units.gpkg`, `tg_lookup.csv`, `tg_hexes.gpkg` (45,088 hexes),
  `tg_prefectures.csv` (COD-AB v02; Livret 01 Tableaux 2 and 4, re-read from the PDF; Kontur 2023-11)
- `sources/tg.py` -> `data/normalized/tg.csv` (the shared Afrobarometer `.sav` files, rounds 5-9; UNSD
  table 28 through `tools/oracle.py`, asserted against the transcription)
- `taxonomy/tg2022.py`; one new node, `other.tg`; `countries/tg.py`
- imports, does not copy: `lr.ipf`, `lr.round_within_rows`, `cm.key`, `cm.gkey`, `cab.stability`

```
python sources/tg_geo.py --fetch
python sources/tg.py                 # Afrobarometer: python sources/afrobarometer.py --fetch
```

## 1. What Togo publishes

| census | religion |
|---|---|
| 2010 (RGPH-4) | asked; national only (§11w read the volumes page by page; *Nuptialité* Tableau 2.4 has it for men 12+) |
| **2022 (RGPH-5)** | asked. **UNSD Demographic Yearbook table 28** holds 15 rows by urban and rural (`python tools/oracle.py Togo`). INSEED's own releases are three population booklets with no religion (§11w, §11aq); the thematic analyses are unpublished. |

UNSD's rows sum to **8,058,172**, 37,326 short of the census's 8,095,498 (Livret 01 Tableau 1), 23,099
urban and 14,228 rural. They carry no `Total` row, and the urban and rural rows miss their own totals
by one person each for Catholic, Assembly of God (short) and Baptist (over). Nothing says what the
37,326 are (collective households are the obvious guess, unchecked).

## 2. Routes

| route | date | outcome |
|---|---|---|
| UNSD oracle | 2026-09-15 | 15 rows, national, urban, rural; **the column margin** |
| INSEED Download Monitor (`/wp-json/wp/v2/dlm_download?search=`) | 2026-09-15 | RGPH-5 Livrets 01-03 and the results leaflet; no religion. Livret 01 is **the row margin** |
| RGPH-4 *Résultats définitifs* (download 2456) | 2026-09-15 | 65 pages with no text layer; not read, deleted |
| COD-PS Togo (`cod-ps-tgo`) | 2026-09-15 | a 2021 projection from 2010; **not used** (the religion margin is 2022, `lr.md` §3's reason) |
| COD-AB Togo v02 (`cod-ab-tgo`) | 2026-09-15 | 5 regions, 40 prefectures, 373 cantons; used |
| Afrobarometer R5-R9 | 2026-09-15 | 5,987 answers, 6 units; **the pattern**. Togo is not in R4 |
| EHCVM 2021-22 (`s01q14`, every member) | §11aq | World Bank login; not pursued |

## 3. The units and their populations

The survey samples the commune of Lomé as its own stratum in every round (5 labels), with
Arrondissements I-V in the location column of rounds 6, 7 and 9, and samples the rest of Golfe
inside Maritime. So six units.

**The commune of Lomé is not a 2022 census unit.** Grand Lomé is 13 communes since loi 2017-08.
Golfe 1-5 are the old city's cantons (Bè, Amoutivé, Aflao-Gakli; Tableau 10's quartiers are Bè,
Tokoin, Nyékonakpoè, Hédzranawoé, Totsi, Agbalépédogan); Golfe 6 is Baguida and Golfe 7
Aflao-Sagbado. **Lomé = Golfe 1-5 = 866,307.** Maritime without it = 3,534,991 - 866,307 = 2,668,684.

COD-AB draws the old commune (104.6 km²) and leaves two canton pieces named `Aflao Gakli` (11.0 km²)
and `Amoutive` (23.6 km²) in Golfe, on the commune's northern edge. Kontur (people inside each; it
counts nothing here) settles which polygon goes with 866,307:

| polygon | Kontur | against Golfe 1-5 |
|---|---:|---:|
| COD-AB `Lome Commune` alone | 778,807 | 0.90x |
| **plus Aflao Gakli and Amoutive** | **969,214** | **1.12x** |
| national | 9,129,266 / 8,095,498 | 1.13x |

So the drawn Lomé is the three pieces (asserted within 5% of the national ratio). All six units run
1.10x (Savanes) to 1.15x (Plateaux). The dissolved regions rebuild COD's region areas within 0.5%.
For contrast, COD-PS's 2021 projection puts Lomé Commune at 1,100,387 against the 2022 count's
866,307: the old core barely grew and the suburbs took the growth.

**A small mismatch left in:** the survey's Maritime|Golfe sampling points may include places inside
the two canton pieces, which the map puts in Lomé. They are dense suburbs next to the city.

## 4. The checks (`sources/tg.py` output, 2026-09-15)

- **Card**: Catholic, `Christian only`, Evangelical, Pentecostal, Presbyterian, Baptist, Muslim only,
  None, Traditional, Other are value labels in all five rounds. `Assembly of God` is a label in every
  round, but Togolese chose it only in R5 (46) and R6 (103, spelled `Assemblies of God`).
- **Locations**: the prefecture column agrees with REGION for every respondent in R6 and R9, and for
  all but one R7 sampling point (8 people labelled Centrale, prefecture Cinkassé in Savanes). Centrale
  and Savanes have 128 and 112 in R7 against about 120 in other rounds, which leans to Savanes; the
  weights are per sampling point and cannot decide. Kept under REGION, the weighting stratum. Asserted.
- **`ab.held_out` is not called.** At 6 units, 14 of 719 other orderings reach r = +0.943 and the
  function stops; its own printout says under 7 units it cannot carry a join. Lomé is sampled at
  1.40x its 2022 share (a 2010 frame) and Savanes at 0.73x. The labels are names and the location
  check is the witness.
- **Quota**: 10 of 10 round pairs; most extreme 6 vs 9, 2 of 63 cells, p = 1 after Bonferroni.
- **Split-half** (`cab.stability`, 6 units, 10 halvings, 2,000-draw per-round null, chi-square veto):

  | group | n | median rho | null 95th | p | verdict |
  |---|---:|---:|---:|---:|---|
  | Catholic | 1,664 | +0.371 | +0.544 | 0.14 | fails |
  | Muslim | 847 | +0.886 | +0.543 | 0.0005 | **carried** |
  | Traditional | 656 | +0.943 | +0.571 | 0.0010 | **carried** |
  | None (with Atheist) | 431 | +0.600 | +0.544 | 0.043 | **carried**, the weakest |
  | Evangelical or Assembly of God | 625 | +0.514 | +0.571 | 0.08 | fails |
  | Pentecostal | 454 | +0.886 | +0.571 | 0.0010 | **carried** |
  | Presbyterian | 274 | +0.943 | +0.600 | 0.0005 | **carried** |
  | Baptist | 168 | +0.429 | +0.571 | 0.12 | fails |
  | Christian only | 596 | +0.029 | +0.543 | 0.48 | fails (spread, not a row) |
  | Other Christian | 125 | +0.229 | +0.571 | 0.29 | fails |
  | Methodist / Jehovah's Witness | 44 / 55 | +0.649 / +0.812 | | 0.03 / 0.002 | pass, under the 1% floor |
  | Adventist / Other | 17 / 31 | | | | fail |

  Evangelical alone and Pentecostal alone, tested as a pair, give +0.114 each: the Evangelical box
  has no pattern of its own.
- **`Christian only`** runs 5.4, 7.0, 16.5, 15.2, 7.2% by round, and is 12.2% (Savanes) to 17.8%
  (Kara) of Christians by unit. A carried church's seed is divided by one minus its unit's share.
- **Traditional and None** move against each other nationally (traditional 11.3% to 8.1%, none 4.2% to
  10.4%), but not unit by unit in step: Kara's traditional rises from R5-R6 to R8-R9 while Savanes's
  falls. Both pass alone and the census fixes both levels, so they are placed apart.
- **Urban witness** (UNSD's urban row against the survey's urban share, each as a multiple of its
  whole): Animist 0.29x / Traditional 0.33x, No Religion 0.68x / None 0.72x, Pentecostal 0.83x / 0.92x,
  Catholic 1.34x / 1.21x. Asserted: Animist and Traditional are the most rural in both. Other
  Christians is the census's most urban row (1.46x), which the survey's small churches (1.02x) do not
  match.
- **Survey against census, nationally**: Catholic 1.31x, Muslim 0.75x, Traditional 0.59x, None 0.69x,
  Pentecostal 1.15x, Presbyterian 1.32x. The fit corrects all of them.
- **IPF**: 9 passes, every unit its census count, rounding drift at most 2 per row.

## 5. Calls someone might reverse

1. **Lomé's polygon includes the two canton pieces** (§3). To reverse: drop `LOME_CANTONS` in
   `tg_geo.py`; the Kontur assertion then fails, by design.
2. **`Assembly of God` would be patterned by `Evangelical` plus the R5-R6 AoG box.** It fails, so it
   moves nothing today. If a later round makes it pass, re-read this before carrying it.
3. **Catholics are seeded flat**, so they are drawn by what the placed religions leave: Lomé 27.0%,
   Centrale 16.7%, Savanes 17.4%. The survey's own pooled shares put Savanes at 29.4% and second, and
   Dapaong is a strong Catholic diocese; the survey's order does not repeat (+0.371), so the map does
   not use it. `note_public` says Catholics are not placed. This is the one to look at twice.
4. **`held_out` replaced by the location check** at 6 units (§4).
5. The R7 Cinkassé sampling point stays in Centrale (8 people).

## 6. As drawn (share of people with a stated religion, %)

| unit | Catholic | Muslim | Animist | None | AoG | Pentecostal | EEPT | Other Chr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Centrale | 17.2 | **54.2** | 3.8 | 5.7 | 7.0 | 1.3 | 0.3 | 5.5 |
| Savanes | 17.9 | 25.3 | **30.5** | 4.2 | 7.3 | 3.7 | 0.2 | 5.8 |
| Kara | 18.8 | 24.2 | 18.1 | **16.4** | 7.7 | 2.5 | 0.7 | 6.1 |
| Plateaux | 23.5 | 13.7 | 9.9 | 8.2 | 9.7 | **11.2** | **9.3** | 7.6 |
| Lomé | **28.3** | 12.3 | 5.5 | 11.2 | 11.6 | 8.6 | 5.2 | 9.1 |
| Maritime | 22.2 | 9.7 | 24.2 | 11.2 | 9.1 | 6.7 | 3.2 | 7.2 |
| **Togo** | 21.6 | 19.2 | 17.5 | 9.7 | 8.9 | 6.4 | 3.6 | 7.0 |

(`note_public` quotes shares of each unit's whole population, which include the undrawn 3.4%.)

## 7. The §14 read

Considered and not asked. Jihadist attacks have reached Savanes (Kpendjal and Tône) since 2021.
Savanes is one unit of 1,143,520 people, the survey's own ceiling, and dots inside it follow Kontur,
not where any group lives. Anita drew Burkina Faso at 45 provinces and Mali at 20 régions because the
units are big (ask 018), and kept Niger, Cameroon and Senegal free with the grain left to the builder.

## 8. What would improve it

1. **INSEED's RGPH-5 thematic analyses**, if a socio-cultural volume prints religion by region or
   prefecture. That replaces the survey pattern outright. The 39 prefectures would be the prize.
2. **EHCVM 2021-22** asks every member's religion with region and milieu (World Bank login, Anita's).
3. **A Catholic pattern**: the Annuario Pontificio's diocesan counts (Lomé, Aného, Kpalimé, Atakpamé,
   Sokodé, Kara, Dapaong) would place Catholics as Belarus's build does (`sources/by.md`). Not searched.
4. **Afrobarometer round 10**: `expect_rounds` will stop the build.

## Placement

`kontur_cap.py tg`: no stops. Water: 65 of 45,088 hexes clipped (0.06% of their area); 4 hexes
that lost over 95% to the sea left whole. 1:1,000: 7,816 dots on 4,629 hexes, 28 carried by the
Hilbert order, 0 rings. 1:10,000: 775 dots, 11 (unit, node) rows carried into other units, 0 rings.
Steps 10-12 are the supervisor's.

## 9. Review, 2026-09-15 (`d743fc47-rev7`)

Full pass. Read the entry, `taxonomy/tg2022.py`, `tg.csv`, `other.tg`'s node note and the playbook.
`check_md` clean, `built_countries --check` OK, `check_rollup tg` 0 orphaned (every row modelled).
One screenshot: dots over the whole country, faded as modelled rows are, densest in Lomé, nothing in
the sea. No ask, nothing rebuilt, no edit to the entry or the mapping.

- **Figures match** `tg.csv`: Centrale 419,431 Muslims of 795,529 is 52.7%; Savanes 198,414 Catholics
  of 1,143,520 is 17.4%; Kara's Catholics 18.2%.
- **Mapping: agreed.** Assembly of God on `pentecostal.trinitarian` with four precedents, `Other
  Christians` on `christianity.other` as `bj2013`, `other.tg` as the usual `other.<cc>` residual.
- **The location check in place of `ab.held_out`: agreed.** The labels are names, R6 and R9 agree
  with the prefecture column for every respondent, and the pattern is taken per unit, so Lomé's
  1.40x sampling does not reach the fit.
- **§5.3, the Catholic call: the witness in §8.3 has now been looked up, and it does not back the
  survey's Savanes.** The spec §12 Latvia rule asks for a witness when a residual reverses a survey
  share. The Annuario Pontificio's diocesan counts, as catholic-hierarchy.org tabulates them (read
  2026-09-15): **Dapaong**, whose 8,534 km2 is Savanes, is 6.1-9.4% Catholic in every edition from
  ap2000 to ap2020, 12.6% in ap2022 and **11.9% in ap2024** (122,488 of 1,028,282). Its ap2005 row
  (147,716, 24.7%) sits between 25,233 and 41,746 and is a typo; Wikipedia's infobox quotes that row.
  In ap2024 Sokodé is 12.0% and Kara 21.9% (gcatholic.org), Lomé 39.8% (Wikipedia, 2023), and Togo
  27.9% (gcatholic.org). On the church's count Savanes is about 0.43x the national rate; the map
  draws it at 0.83x, and the survey's pooled 29.4% is about 1.04x. So if the dioceses have the
  order right, the map errs high in Savanes, not low, and the survey's second place is the outlier.
  It is a different basis (baptised Catholics, diocesan population estimates) and the dioceses do not
  follow the Plateaux-Maritime line, so it is a witness to order and not a pattern to draw. No
  change; `note_public`'s "not because the survey found them there" stays true. Not looked up:
  Aného, Atakpamé and Kpalimé in a recent edition (Wikipedia has only 2004-2006).
- **One wording point, left alone.** The note's national figures are shares of people with a stated
  religion and its unit figures are shares of the whole population (§6's footnote), so Centrale's
  Muslims are 52.7% in the note and 54.2% in §6. Under 1.5 points and neither is wrong.
- **§14: agreed with §7.** Nothing drawn locates a group below a region of a million people, and dots
  inside a region follow Kontur.
