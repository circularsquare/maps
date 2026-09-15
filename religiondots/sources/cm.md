# Cameroon — religion in 12 units from the pooled Afrobarometer, on COD-PS 2025

**Drawn 2026-09-14** (session `d743fc47-cm`). 12 units, 7 categories, 29,442,318 people, every row
`modelled`. The construction of `sources/tz.py`, on Anita's Nigeria ruling (`ask/answered/010-ng`).
No ask filed. §4 is the call a reviewer should look at twice.

- `sources/cm_geo.py` -> `data/geo/cm/cm_regions.gpkg`, `cm_lookup.csv`, `cm_hexes.gpkg` (106,388
  hexes), `cm_departments.csv` (COD-AB v01 ADM1 and ADM2; COD-PS 2025's region and metropolis tables;
  Kontur 2023-11)
- `sources/cm.py` -> `data/normalized/cm.csv` (the shared Afrobarometer `.sav` files, rounds 5-9;
  the 2005 census volume as a witness, re-read from the PDF on every run)
- `taxonomy/cm2025.py`; one new node, `other.cm`; `countries/cm.py`
- `sources/afrobarometer.py` gained `round_within_rows` and `compose(df, nat, units, cats, carried)`,
  lifted from `tz.py`; `tz.py` and `ng.py` keep their own copies

```
python sources/cm_geo.py --fetch
python sources/cm.py --fetch          # the census PDF; Afrobarometer: python sources/afrobarometer.py --fetch
```

## 1. What Cameroon publishes

| census | religion |
|---|---|
| 2005 (3e RGPH, BUCREP) | asked (Q12, NADA `nada.stat.cm` catalog 58, per §11aq). Printed nationally only: *Volume II Tome 01, État et structures de la population*, **Tableau 5.8** (printed p.97, PDF p.124), by sex and urban/rural: Catholique 38.4, Orthodoxe 0.5, Protestant 26.3, Autres chrétiens 4.0, Musulman 20.9, Animiste 5.6, Autres religions 1.0, Libre penseur 3.2. Tableau A.37 is sex ratios. One paragraph (printed p.101, PDF p.128) gives the leading religion's share per region. The microdata is licensed (NADA catalog 89). |
| 2026 (4e RGPH) | enumerated from 24 April 2026 (§11aq); nothing published. **Reopen when it is.** |

Cameroon is **absent from the UNSD oracle** (`python tools/oracle.py Cameroon`, 2026-09-14). The
volume was read from CEPED's IREDA inventory
(`ireda.ceped.org/inventaire/ressources/cmr-2005-rec_TOME2.1_etat_structure.pdf`, 190 pages, complete
`%%EOF`). The office-side sweep (BUCREP's download categories and CDX) is §11aq's; nothing further was
searched here.

Two things in the volume that were noticed and not resolved. Tableau 5.8 has eight lines where §11aq's
reading of the Q12 form lists seven codes (the table adds `Orthodoxe` and says `Libre penseur` for the
form's `Sans religion`); the form was not re-read. And the p.101 paragraph says animists are found
"surtout" in Ouest (4.6%) and Extrême-Nord (4.1%), which cannot both be the highest regions if the
national figure is 5.6%; that sentence is not used.

## 2. Routes

| route | date | outcome |
|---|---|---|
| UNSD Demographic Yearbook (oracle) | 2026-09-14 | absent |
| 2005 census, Vol. II Tome 01 | 2026-09-14 | national table and the p.101 paragraph; **a witness, not drawn** (twenty years older than the populations, §6's IPF) |
| 2005 microdata (NADA catalog 89) | §11aq | licensed; not pursued |
| DHS 2018 recode (`v130` by region) | | registration-walled (§11ag); not pursued |
| **Afrobarometer R5-R9** | 2026-09-14 | open, on disk; **used**. Cameroon is not in R4. |
| **COD-PS Cameroon 2025** (`cod-ps-cmr`) | 2026-09-14 | BUCREP's projection from 2005 (cohort component, zero internal migration), with metropolis rows for Douala and Yaoundé; **used** |

## 3. The construction

    row margin      unit populations        COD-PS 2025 (BUCREP projection)   EXACT
    the composition each unit's own mix     Afrobarometer R5-R9 pooled         measured, n=5,949
    the national level                      neither                            computed

### Twelve units

Every round samples Yaoundé and Douala as strata of their own, under labels that change every round:

| round | Yaoundé | Douala | REGION codes |
|---|---|---|---|
| R5 | `Yaounde` | `Douala` | 1220, 1221 |
| R6 | `Centre-YaoundÃ©` (LATIN1 mojibake) | `Littoral-Douala` | 1220, 1221 |
| R7 | `Mfoundi` | `Wouri` | 1220, 1221 |
| R8 | `Mfoundi` | `Wouri` | **1223, 1227: every code shifted against R6, R7, R9** |
| R9 | `Yaounde` | `Douala` | 1220, 1221 |

So the decode is by label, never by code. **Checked against the survey's own department column**
(`LOCATION.LEVEL.1`, rounds 6, 7 and 9): 1,173, 1,194 and 1,200 respondents, every department matched
to COD-AB (four aliases: `KAKEY`, `MKAM`, `KOUPE ET MANENGOUBA`, `NGO KETUNDJIA`), **0 disagree** with
the unit from REGION, and the 811 city respondents are all in Mfoundi or Wouri. Mfoundi (289 km²) and
Wouri (976 km²) are cut out of Centre and Littoral as whole COD-AB departments; the dissolved regions
match COD's region polygons within 0.5%.

Populations: COD-PS 2025 regions, with `Ville de Yaoundé` (3,762,931) and `Ville de Douala`
(3,816,532) subtracted from their regions. Kontur against COD-PS per unit: **0.87x (Extrême-Nord, Est)
to 1.13x (Sud)**, nationally 0.973; banded at 0.75-1.30 in `cm_geo.py`. 1.49% of Kontur's extract
falls outside every unit (Nigeria, Chad, the lake) and is dropped.

### The checks

- **Cards**: None, Traditional/ethnic religion, Other, Christian only, Roman Catholic, Presbyterian,
  Baptist and Lutheran are value labels on every pooled round.
- **Held-out** (per round and pooled): R5 +0.976, R6 +0.939, R7 +0.985, R8 +0.988, R9 +0.954, pooled
  +0.989; 0 of 20,000 pairings reach any of them. Littoral outside Douala is the fullest pooled (1.34x).
- **Quota**: 10 of 10 round pairs compared; most extreme 6 vs 8, 2 of 45 cells identical, Bonferroni
  p = 1.
- **Split-half** (`cab.stability`, 12 units, median over 10 halvings):

  | answer | n | share | median rho | null 95th | p | chi2 p | verdict |
  |---|---:|---:|---:|---:|---:|---:|---|
  | Christian (all but the two churches) | 3,763 | 63.25% | +0.899 | +0.385 | 0.0005 | 2e-93 | own geography |
  | Presbyterian | 553 | 9.30% | +0.860 | +0.385 | 0.0005 | 1e-92 | own geography |
  | Baptist | 206 | 3.46% | +0.800 | +0.397 | 0.0005 | 2e-38 | own geography |
  | Muslim | 1,126 | 18.93% | +0.801 | +0.390 | 0.0005 | 3e-261 | own geography |
  | None | 212 | 3.56% | +0.839 | +0.376 | 0.0005 | 2e-10 | own geography |
  | Other | 62 | 1.04% | +0.633 | +0.385 | 0.0015 | 1e-21 | passes; **not placed** (below) |
  | Traditional/ethnic religion | 27 | 0.45% | +0.255 | +0.417 | 0.17 | 5e-06 | fails |

  With the churches tested apart (scratch run): Catholic +0.829, Lutheran +0.609, Evangelical +0.696,
  Pentecostal +0.601, `Christian only` +0.357 (null +0.341) all pass; Adventist +0.130 fails.
- **`Other` is not placed.** All 58 `Other` answers are in rounds 5-7 (7, 28, 23) and none in 8 or 9,
  with the box on both cards; Anglican, Methodist and Coptic also vanish from round 8, so the later
  teams probably coded other answers away. Asserted in `cm.py::main`.
- **Small-category rule**: under the residual, Traditional is at most 1.05x its national share
  (Littoral) and Other 1.71x (Ouest) in a unit where nobody gave it; under 2x, so the **tail is the
  residual**, not flat.
- **Level** (Norway):

  | | Christian | Presbyterian | Baptist | Muslim | None |
  |---|---:|---:|---:|---:|---:|
  | **as drawn** (pool, recomposed on COD-PS) | **61.84%** | **9.00%** | **3.54%** | **20.08%** | **3.82%** |
  | R8-R9 alone, recomposed the same way | 60.61% | 8.27% | 3.55% | 21.20% | 5.53% |

  Largest gap None, 1.71 points; under 3.5, no §3.4 rescale. The grouped Christian share does swing by
  round in the survey's own weighting (57.1-66.6% for the bare category, R7 and R8 low), and
  recomposing per unit absorbs most of it.

## 4. Which churches are drawn (the call to look at)

Weighted share of all respondents by round:

| | R5 | R6 | R7 | R8 | R9 | range |
|---|---:|---:|---:|---:|---:|---:|
| Roman Catholic | 40.5 | 41.4 | 29.5 | 26.3 | 36.8 | **15.1** |
| `Christian only` | 6.7 | 6.0 | 10.4 | 12.2 | 13.5 | 7.5 |
| Presbyterian | 8.6 | 10.9 | 9.8 | 8.8 | 8.7 | **2.2** |
| Baptist | 3.4 | 4.1 | 3.2 | 3.9 | 3.3 | **0.9** |
| Lutheran | 2.2 | 2.4 | 2.3 | 2.1 | 2.3 | **0.3** |
| Evangelical | 7.8 | 8.9 | 4.2 | 6.5 | 2.7 | 6.2 |
| Pentecostal | 5.4 | 2.0 | 5.1 | 5.6 | 5.4 | 3.6 |

The playbook asks for an outside witness to a church's level. The census gives one at family level:
**Protestants 26.3% in 2005**, and the survey's Protestant bodies come to about 27% pooled without
touching `Christian only`, while its Catholics (35%) are 3-4 points short of the census's 38.4%. So the
catch-all takes mostly from the Catholic answer, and the churches that hold their level while it
doubles are measured, as floors.

**The second test, from the reviewer.** `Christian only` is not even across units: as a share of
Christians it is Ouest 7%, Mfoundi 8%, Nord-Ouest 8%, Centre 9%, Sud-Ouest 9%, Littoral, Sud and Est
11%, Wouri 14%, Extrême-Nord 18%, **Nord 29%, Adamaoua 35%**; national 13.0%. A church living where a
third of Christians name none is drawn short exactly there. Averaged over each church's own
respondents: **Presbyterian 10.7%, Baptist 12.5%, Lutheran 22.6%**. Presbyterian and Baptist are drawn;
Lutheran is folded into `christianity`. Baptist is close to the line and is the first to fold if the
data moves (`cm.py::unnamed_where_they_live` asserts both sides).

**A second opinion was asked** (a reviewer agent pointed at the playbook, `afrobarometer.py`, the
sibling modules and the scratch outputs). It answered one Christian node, as Nigeria, Tanzania and
Liberia, and would accept Presbyterian if detail were wanted; its argument against was the uneven
catch-all above. I kept its test and not its conclusion: Presbyterians and Baptists live where 8-11% of
Christians name no church, and dropping two level-stable, split-half-passing churches from a country
whose Protestant geography (Anglophone Presbyterians and Baptists, the Sud's Presbyterians) is the
main thing the survey adds seemed the larger loss. **To reverse it**: `CHURCHES = []` in `cm.py`, the
two `GROUP` lines back to `Christian`, `CARRIES`, and the two MAP lines in `cm2025.py`; rerun `cm.py`
and scatter. No other country is affected.

It also flagged: `Evangelical` in Cameroon is most likely the Reformed Église Évangélique du Cameroun
(Ouest 18.5%), not `christianity.evangelical` (kept inside `christianity`, noted in REVIEW);
`Orthodox` clusters by round and unit (10 of round 6's 20 in Extrême-Nord, 13 of round 9's 21 in Nord),
which looks like one team's coding; it stays Christian as the card's box says, at 1% of respondents.

## 5. The survey against the 2005 census

National: census 69.2% Christian, 20.9% Muslim, 5.6% animist, 3.2% free-thinker, 1.0% other; drawn
74.4%, 20.1%, 0.55%, 3.8%, 1.2%. **The animist gap is the instrument**: the card offers traditional
religion as a peer of Christian and Muslim, so it counts only people who give it as their one
religion; `note_public` says so. By region (survey merged back to ten regions by COD-PS population):

| | census 2005 | survey |
|---|---:|---:|
| Catholic, Centre | 65.4 | 53.3 |
| Catholic, Littoral | 52.0 | 42.2 |
| Catholic, Est | 42.4 | 34.9 |
| Catholic, Sud-Ouest | 41.3 | 32.0 |
| Catholic, Ouest | 34.7 | 41.9 |
| Protestant, Nord-Ouest | 49.3 | 46.9 |
| Protestant, Sud | 49.1 | 38.5 |
| Muslim, Adamaoua | 71.5 | 63.1 |
| Muslim, Extrême-Nord | 42.7 | 38.4 |
| Muslim, Nord | 40.7 | 41.6 |

Asserted: the three northern regions are the survey's three most Muslim, Centre its most Catholic, and
Nord-Ouest and Sud are Protestant-led. **Not asserted, and disagreeing**: in Sud-Ouest the census had
Catholics leading at 41.3% and the survey has Protestant bodies ahead (Presbyterian 22.5, Pentecostal
12.7, Baptist 10.2 against Catholic 32.0); Pentecostal growth in seventeen years would do it, and
nothing here can tell.

## 6. The §14 read

§11aq's note: Boko Haram in Extrême-Nord. **Considered and not asked.** The ruling of 2026-09-14 kept
Cameroon free and left the grain to the builder; Chad (ask 017) and Burkina Faso and Mali (ask 018)
were drawn at their published grain because the units are big. Extrême-Nord is one unit of 5.5 million
people, the survey's own ceiling, and dots inside it follow Kontur's population, not where any group
lives. The Nord-Ouest and Sud-Ouest conflict since 2016 is a linguistic and political one; the two
churches drawn there are those regions' majority Protestant churches, at region grain.

## 7. What is deliberately not drawn

- Catholics, Evangelicals, Pentecostals, Lutherans and Adventists as churches (§4).
- Sunni and the Tijaniyya: `Sunni only` is 53 people, 1.4-2.0% in round 7 and 0.2% in round 9.
- `Other`'s own geography (§3).
- **Zero cells** (§3.5): no Baptist among Adamaoua's 309; `note_public` says so.

## 8. What would improve it

1. **The 4th RGPH (2026)**, if BUCREP prints religion by region. This build should then be replaced.
2. **DHS 2018 recode** (`v130` by region, larger sample, women and men 15-49); walled, Anita's.
3. **Afrobarometer round 10** when released; `expect_rounds` will stop the build.

## Placement

`kontur_cap.py cm`: no stops. Water: 282 of 106,388 hexes clipped. 29,439 dots at 1:1,000 on 14,978
hexes, 0 rings; 35 dots carried by the Hilbert order.

## Review, 2026-09-15 (`d743fc47-rev5`)

Read the mapping, `cm.csv`, the note and the raw `.sav` files (scratch scripts, not kept).
`check_md` clean, `built_countries --check` OK, `check_rollup cm` 0 orphaned (every row modelled).
One screenshot: dots over the whole country, the north green and the south yellow, nothing in the
sea. No ask, nothing rebuilt.

- **Every figure in `note_public` matches `cm.csv`** (74.4% Christian, 20.1% Muslim, 3.8% none,
  0.55% traditional; Adamaoua 63.1% Muslim, Est 17.3% the highest outside the north; the
  Presbyterian and Baptist shares).
- **Round 8's label decode holds up against the survey's own ethnic and home-language columns**
  (Q81, Q2), which fills the gap left by `check_locations` covering rounds 6, 7 and 9 only: 37 of
  round 8's 64 Adamaoua respondents speak Fulfulde at home, and its Sud Christians are Beti speaking
  Bulu or Ewondo.
- **Sud's 10.8% Muslim (95,233 people) is mostly one round.** Muslim share of Sud's respondents
  by round: 0, 0, 6.1, **42.1**, 0. Round 8's 48 Sud interviews include 9 Bamoun and 7 Hausa
  Muslims, speaking Bamoun or Hausa at home, spread over several sampled areas, urban and rural.
  These are real communities, not a decode error, and they are not one cell, so `CELL_CAP` does not
  apply. But they set Sud's drawn share at roughly ten times what the other four rounds give.
  Left as built, because pooling is the construction. If Cameroon is rebuilt, or a rule for one
  round carrying a unit's share is written, this is the unit to test. (Est's round 8, 35.8%, is
  spread over both kinds of area and sits closer to its other rounds.)
- **Mapping: agreed**, including folding `Evangelical` into `christianity` and filing both
  Presbyterian churches at `christianity.reformed.presbyterian`. Round 8's 5 Beti Lutherans in Sud
  are folded in with the rest of the Lutherans and change nothing.
- **Shared helpers**: the `sources/afrobarometer.py` diff is additions only, so no drawn country
  changes. `round_within_rows` now exists in `afrobarometer.py`, `tz.py`, `ng.py` and `lr.py`,
  and `mg.py` imports it from `tz`; a tidy-up for later.
- **§14: agreed** with §6. Asks 017 and 018 were decided on the same reasoning (big units).
- `queue.csv`'s grain still says `10 regions`; the build is 12 units. Not edited, since other
  builders are writing that file.
