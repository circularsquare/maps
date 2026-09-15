# Venezuela — LAPOP AmericasBarometer single-country files, waves 2010–2016/17

Built 2026-09-14 by session `d743fc47-ve`, from `sources.md` §11ap's scouting. Write-up in
`sources.md` §ve-2026-09-14. Code: `sources/ve.py` (build), `sources/ve_geo.py` (boundaries and the
2011 count), `sources/ve_grid.py` (Kontur hexes), `taxonomy/ve2016.py` (mapping), `countries/ve.py`.

## 1. Why a survey

No Venezuelan census asks religion. §11ac read INE's own REDATAM base `CPV2011` (67 person
variables, none religion) and its 24-page person metadata; §11ap read the ENCOVI 2016, 2021 and 2024
questionnaires (no affiliation item) and found INE's ANDA with no microdata. LAPOP's free grand merge
drops Venezuela after 2008, and LAPOP's data directory still offers each wave through 2016/17 as a
Free Tier single-country file, the route taken here. There is no Venezuelan wave after 2016/17.

## 2. The files

| file | what | from |
|---|---|---|
| `data/raw/ve/lapop_ve_<year>.dta`, 2010-2016 | LAPOP single-country Stata files, ids 1582 (2010), 1970, 2020, 2195 (2016/17) | `vanderbilt.edu/center-for-global-democracy/data/directory/?lp_download=<id>`, fetched with a plain GET on 2026-09-14; the same site usage agreement as the grand merge and Bolivia's files |
| `data/raw/ve/ven_admin_boundaries.shp.zip` | COD-AB v01: 25 federal entities, 336 municipalities | HDX `cod-ab-ven` |
| `data/raw/ve/ven_admpop_adm1_2011_v2.csv` | COD-PS: the 2011 census by entity, 24 rows | HDX `cod-ps-ven` |
| `data/raw/ve/kontur_population_VE_20231101.gpkg.gz` | placement | Kontur |
| (not saved) | INE, *XIV Censo Nacional de Población y Vivienda, Resultados Total Nacional*, May 2014, Cuadro 2.2 p. 13 | `ine.gob.ve/wp-content/uploads/2024/09/Censo-Nacional-2011.pdf`, transcribed into `ve_geo.py::INE_2011` |

## 3. The population is the 2011 census

INE counted 27,227,930 people (reference date 30 October 2011). COD-PS is that count: all 24 of its
rows equal Cuadro 2.2, asserted, and it leaves out the Dependencias Federales (2,155 people), which
the table gives. COD-PS calls itself `Distrito Federal` for the Distrito Capital and `Vargas` for La
Guaira; the codes are the same and the names are checked beside them.

No newer base was taken. There has been no count since, a projection from this census cannot see the
emigration of the late 2010s, and the pooled rounds are 2010-2016/17 anyway. COD-AB's eastern bound
is -59.741, Venezuela's own border, so the Guayana Esequiba is not in it (asserted, spec §14.18).

## 4. The state labels, checked without trusting them

`prov` is **alphabetical over the 21 states of 2010** (1601 Anzoátegui to 1621 Zulia) and **a
different order over 17 states in 2012-2016** (1601 Distrito Capital, 1602 Miranda, ..., 1617
Zulia). A code join across waves would misplace most of the sample. Each wave's labels are joined to
COD-AB by name (`vargas` -> La Guaira), then checked in `ve.py::decode_wave`:

- **`municipio` names** against COD-AB's 336 municipalities: the states each prov code's names can
  belong to intersect in the labelled state, 60 municipalities a wave. Seven labels needed an alias,
  each a renaming or a spelling: Heres (Ciudad Bolívar) is now Angostura del Orinoco, Guaicaipuro is
  Bolivariano Guaicaipuro, San Carlos is the seat of Ezequiel Zamora (Cojedes), Aragua's Zamora is
  Ezequiel Zamora, `LANDER/ TOMAS LANDER`, `FERNADEZ FEO`, `TURISTICO DIEGO BAUTISTA URBANEJA`. An
  alias adds states to the set and never settles one.
- **One-municipality states** whose municipality name other states share are settled by
  elimination, since the decode is one to one: the Distrito Capital (Libertador) and Cojedes in 2010,
  Falcón (Miranda) in 2012-2016. In 2012-2016 the Distrito Capital's Libertador is still left beside
  unsampled Monagas's, and `tamano` settles it: all 78 to 80 respondents are in the national capital's
  metropolitan area.
- **2012, 2014 and 2016 share one municipality numbering** (1601 -> 1600001): all 60 codes sit under
  the same state in all three. 2016's code 1600028 has no label and is 2012's Sifontes, in Bolívar.
- **Sample against the 2011 census**, reported: r=+0.993 in 2010 over 21 states and +0.922, +0.922,
  +0.913 over 17 after; none of 20,000 random pairings reaches any. From 2012 the allocation is by
  region, not by state (Falcón 0.32x its population share, Cojedes 1.77x to 2.09x).

## 5. Answer codes by wave

Weighted %, post-stratified (§6):

| code | 2010 | 2012 | 2014 | 2016/17 |
|---|---|---|---|---|
| 1 Católico | 78.11 | 79.91 | 73.67 | 67.45 |
| 5 Evangélica y Pentecostal | 6.07 | 8.99 | 10.31 | 13.00 |
| 4 Ninguna (creyente) | 9.55 | 7.00 | 8.61 | 6.70 |
| 2 Protestante tradicional | 4.22 | 1.44 | 5.18 | 2.15 |
| 12 Testigos de Jehová | 1.33 | 1.24 | 0.47 | 3.35 |
| 77 Otro | 0.00 | 0.00 | 0.00 | 3.49 |
| 7 Religiones Tradicionales | 0.06 | 0.46 | 0.79 | 1.79 |
| 3 Orientales | 0.07 | 0.24 | 0.30 | 1.21 |
| 11 Agnóstico o ateo | 0.46 | 0.52 | 0.67 | 0.50 |
| 6 Mormones | 0.14 | 0.20 | 0.00 | 0.37 |

5,894 respondents answered (1,492, 1,464, 1,475, 1,463). Jewish (10) was offered in 2010 and 2012
and nobody chose it. `lapop.CARD_ABSENT` holds (no `Otro` before 2016).

**`lapop.wave_flags` flags 2016/17 twice**: code 3 at 19 respondents against 4.0 expected and code 7
at 26 against 6.0. Judged harmless and kept (`ve.py::WAVE_FLAGS`): Honduras 2016's shifted codes
took respondents out of the evangelical box, and here the Christian answers keep their course (the
Catholic fall and the evangelical rise were already under way), code 7 rises in every wave, and the
19 and 26 sit in 18 and 24 sampling cells across 9 and 7 states. The cost is that both small answers'
pooled shares are inflated by one wave, which the mapping says.

## 6. Weights

No wave has a design weight: 2010 has no `wt` column and 2012-2016's `wt` is 1 for everyone, in a
design that allocates by region and leaves states out. `ve.py::poststratify` (Bolivia's) scales every
(wave, state) to the state's 2011 census share of 1,500, so each wave counts equally inside a state.

## 7. Which answers carry their own geography

`bo.py::stability` (median Spearman over the three halvings of four waves, per-wave permutation
null, chi-square veto, 50% largest-cell cap, which-unit-tops-both-halves), **on 20,000 draws**. At
the 17 states sampled in every wave:

| answer | n | share | median rho | p | chi-square p | verdict |
|---|---|---|---|---|---|---|
| Católico | 4,273 | 74.77% | +0.343 | 0.0524 | 4e-30 | fails |
| Evangélica y Pentecostal | 567 | 9.53% | +0.547 | 0.0021 | 2e-19 | state share |
| Ninguna (creyente) | 460 | 7.98% | +0.296 | 0.0788 | 4e-10 | fails |
| Protestante tradicional | 188 | 3.33% | +0.093 | 0.34 | 2e-13 | fails |
| Testigos de Jehová | 92 | 1.58% | +0.043 | 0.43 | 4e-07 | fails |
| Otro | 50 | 0.87% | undefined (one wave) | | | no test |
| Religiones Tradicionales | 44 | 0.78% | +0.581 | 0.0034 | 2e-08 | state share |
| Agnóstico o ateo | 32 | 0.55% | +0.268 | 0.12 | 0.02 | fails |
| Orientales | 29 | 0.45% | +0.372 | 0.0424 | 0.25 | refused (units do not differ) |
| Mormones | 11 | 0.16% | -0.105 | 0.66 | 0.97 | fails |

At 2010's six design regions (capital; centro-occidental; los llanos; occidental, which is the
Andes; oriental; zuliana), which nest the 17: Católico +0.657 p=0.0438, evangelical +0.943 p=0.0008,
Testigos +0.771 p=0.0205 and Tradicionales +0.955 p=0.0006 pass. So **Catholic and Witnesses are
drawn on region shares**. 2012-2016's eight regions move Falcón, Cojedes and Bolívar between regions,
so they nest no pool of all four waves and were not used.

**Why 20,000 draws.** On bo.py's 2,000, Catholic at the states came out p=0.0500 on seed 0, a pass
by 25 millionths, and 0.0535, 0.0570, 0.0450, 0.0575, 0.0560 on seeds 1-5: the Monte Carlo error at
p=0.05 on 2,000 draws is about 0.005, so the verdict was the seed's. On 20,000 it is 0.0524. The
regional Catholic p was 0.0385-0.0465 on all six seeds. Nothing else changed verdict across seeds.

The two Protestant boxes pass together (+0.529) but their state shares are not opposed (Spearman
+0.27), so they are kept as the card offers them (Colombia and Bolivia did the same). No failing
answer has a unit on top of both halves in 95% of halvings. Largest (wave, cell) share of any placed
answer: 7%.

## 8. Composition, and the reversal it draws

**One-round states** (`co.py::region_fallback`, copied, on 2010's regions, over the two state-level
answers): capital region 2 of 2 closer than the country (mean error 3.5 against 8.3 points), so La
Guaira takes the Distrito Capital's and Miranda's shares; los llanos 2 of 2 (1.8 against 6.8), so
Apure and Barinas take Guárico's and Portuguesa's; oriental 1 of 3, so Monagas takes the national
rate for evangelicals and traditional religions. All four take their region's Catholic and Witness
shares, since those are drawn by region everywhere.

**The tail** is each state's remainder at national proportions; it is 8.4% to 21.0% of a state
against 13.3% nationally. The 2x rule does not fire (worst: traditional Protestant in Guárico at
1.58x, where the survey found none).

**The residual reverses two answers against the survey's own state shares** (spec §12, Latvia):

| answer | Spearman, drawn against measured | the worst states, drawn / measured |
|---|---|---|
| Ninguna (creyente) | +0.13 | Falcón 11.7 / 1.3 (95 interviews); Anzoátegui 10.1 / 3.2; Sucre 12.1 / 6.8; Bolívar 6.2 / 13.0 |
| Protestante tradicional | -0.28 | Bolívar 2.5 / 9.8; Portuguesa 5.1 / 11.2; Guárico 5.2 / 0.0 |
| Otro | +0.37 | |
| Orientales | -0.43 | (29 respondents) |

The mechanism: a state whose own Catholic share is above its region's (Falcón 89.9% on its own
interviews, 74.8% as drawn for zuliana) is left a large remainder, and the remainder is filled at
national proportions. With Catholic at the state, as the 2,000-draw run had it, Ninguna came out
+0.80 and traditional Protestant +0.14. **The rule is that a reversal asks for a witness and
licenses no override; Venezuela has no witness** (no census, and the other open instruments are
national or eight regions), so it is built as the rule says and recorded here and in the mapping.
The standing alternative, if someone rules on it: draw Catholic at the state after all, or put the
failing answers flat at national shares with the carried shares scaled (`do.py`'s construction),
which removes the reversal and flattens Bolívar's 9.8% traditional Protestant to 3.4%.

## 9. Mapping calls (`taxonomy/ve2016.py`)

- **`Religiones Tradicionales` -> `other.ve`, not `indigenous` and not an `afrodiasporic` node.**
  27 of 44 respondents are in the Distrito Capital and Miranda, none in Zulia, and the label names no
  tradition; a Caracas answer to it could be Santería, María Lionza or espiritismo. Colombia's call.
- `other.ve` added to `taxonomy/branches.py`, `build_tree.py` run (703 nodes).
- Otherwise as `bo2023.py`.

## 10. What was drawn

26,422,160 people on 21 states; Amazonas, Delta Amacuro, Nueva Esparta and the Dependencias
Federales (805,770, 2.96%) blank. As drawn: Católico 74.03%, Evangélica 9.76%, Ninguna (creyente)
8.41%, Protestante tradicional 3.44%, Testigos 1.49%, Otro 0.89%, Tradicionales 0.78%, agnostic or
atheist 0.56%, Orientales 0.46%, Mormones 0.19%.

| state | n | evangelical | traditional | Catholic (region) | Witnesses (region) |
|---|---|---|---|---|---|
| Bolívar | 344 | 20.4% | 0.0% | 68.1% | 1.1% |
| Portuguesa | 257 | 16.2% | 0.0% | 62.9% | 0.2% |
| Apure, Barinas (llanos pool) | 34, 42 | 15.7% | 0.3% | 62.9% | 0.2% |
| Guárico | 147 | 15.1% | 0.7% | 62.9% | 0.2% |
| Anzoátegui | 444 | 13.1% | 0.9% | 68.1% | 1.1% |
| Lara | 414 | 11.7% | 0.0% | 76.0% | 1.8% |
| Zulia | 841 | 11.5% | 0.0% | 74.8% | 3.2% |
| Sucre | 191 | 10.7% | 0.0% | 68.1% | 1.1% |
| Monagas (national) | 48 | 9.5% | 0.8% | 68.1% | 1.1% |
| Mérida | 264 | 9.2% | 0.0% | 81.1% | 1.3% |
| Cojedes | 129 | 8.3% | 0.6% | 76.0% | 1.8% |
| Trujillo | 199 | 8.3% | 0.0% | 81.1% | 1.3% |
| Yaracuy | 133 | 8.0% | 0.7% | 76.0% | 1.8% |
| Carabobo | 622 | 7.9% | 1.1% | 76.0% | 1.8% |
| Aragua | 368 | 7.4% | 0.9% | 76.0% | 1.8% |
| Táchira | 225 | 7.3% | 0.0% | 81.1% | 1.3% |
| Miranda | 723 | 5.0% | 2.1% | 79.0% | 0.6% |
| La Guaira (capital pool) | 24 | 4.1% | 2.7% | 79.0% | 0.6% |
| Distrito Capital | 350 | 2.9% | 3.5% | 79.0% | 0.6% |
| Falcón | 95 | 2.7% | 0.0% | 74.8% | 3.2% |

## 11. Kontur

182,656 hexes on the 21 drawn states (the four never-sampled entities are left out of the layer; they
hold 5,123 hexes). Kontur's 2023 grid is 1.015x the 2011 census over the drawn states, 0.80x in the
Distrito Capital to 1.23x in Cojedes. **Six blocks reached the cap** and are in `kontur_cap.csv`:
Petare (eastern Caracas) and northern Maracay **real**, inside continuous dense city; south of El
Tigre, north-east of San Felipe (Yaracuy) and beside Duaca (Lara) **capped**, each a lone spike with
a 3 km ring median of 16-697/km²; and a lone hex at the cap on Cubagua island (29,067 people, nothing
populated within 10 km), which `kontur_cap.py` could neither cap (no ring) nor call real, and which
went away with Nueva Esparta's hexes.

## 12. Open

- **The non-Christian tail**, as for every LAPOP country (`queue.md`'s refinement list). No Jewish
  respondent in any wave; Eastern religions 0.46% and refused.
- **The reversal in §8**, which only a second instrument at the state can settle. **WVS wave 7
  (2021) is at 22 states** (§11ap) behind the WVS licence form, which is Anita's; it is also the only
  post-2016 reading, and after the emigration.
- **Latinobarómetro 2023**, a direct zip whose Venezuelan `REG` is eight regions (§11ap): a recent
  level (53.9% Catholic, 29.8% evangelical) at a grain coarser than this map. Read its labels first,
  since `REG` changes meaning between waves.
- **Four blank entities** (2.96%), which LAPOP will never supply.

## 13. Review, 2026-09-15 (d743fc47-rev4)

Checks clean: `check_md`, `built_countries --check`, `check_rollup ve`. Screenshot at the entry's
`view`: dots on the coast, the Andes, Maracaibo and Ciudad Guayana; the four never-sampled entities
blank, as `gap` says.

- The mapping matches `bo2023.py` and `co2023.py`, and `other.ve` follows `other.co` and `other.bo`.
  Nothing to change.
- **`Religiones Tradicionales` is now filed two ways across the LAPOP countries**: `indigenous` in
  `gt`, `ec`, `cr`, `pa` and `sv`; `other.<cc>` in `co`, `bo` and `ve`. Venezuela's reason (27 of 44
  in Caracas, none in Zulia) is local and holds. Listed so whoever tidies the LAPOP tail has all
  eight in one place.
- The residual reversal in §8 is the rule applied as written, with the alternative recorded; not
  re-argued. The survey's own state figures for the reversed answers failed the split-half, so the
  note is not hiding a measurement.
- Small: the internal `note` in `countries/ve.py` says "Monagas the national rate". §8,
  `sources/ve.py` and `note_public` agree that Monagas takes its region's Catholic and Witness
  shares and the national rate only for the state-level answers. Left for the builder.
