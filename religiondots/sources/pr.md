# Puerto Rico (`pr`)

Session `f95259a4-pr`, 2026-09-14. Scouted in `sources.md` §11ap (verdict then: "not open", the
WVS licence form). Anita downloaded the file; no form was needed.

**Drawn: 6 regions, 3,203,295 people, every row `modelled`.** World Values Survey wave 7, Puerto
Rico 2018, `Q289`, laid on the Census Bureau's Vintage 2024 municipio estimates.

| file | what |
|---|---|
| `sources/pr.py` | the survey: load, geography witnesses, split-half, construction, `data/normalized/pr.csv` |
| `sources/pr_geo.py` | six regions dissolved from the 78 municipios; populations; `pr_regions.gpkg`, `pr_lookup.csv`, `pr_municipios.csv` |
| `sources/pr_grid.py` | Kontur 400 m hexes keyed to region, `pr_hexes.gpkg` |
| `taxonomy/pr2018.py` | the mapping, with `REVIEW` |

## 1. Source

- `data/raw/pr/F00013157-WVS_Wave_7_Puerto_Rico_Csv_v5.1.zip`, one CSV, 1,127 rows, 404 columns,
  semicolon-delimited, UTF-8 with a BOM. Dataset version `6-0-0 (2024-04-15)`, DOI
  `doi.org/10.14281/18241.20`.
- Fieldwork 16 March to 27 October 2018, face to face, PAPI, adults 18+. Universidad del Sagrado
  Corazón (Javier Hernández Acosta) with the Instituto de Estadísticas de Puerto Rico.
- **The survey team's report** is the key document: *Encuesta Mundial de Valores para Puerto Rico
  2018*, Instituto de Estadísticas, 17 June 2019,
  `https://estadisticas.pr/files/Publicaciones/Encuesta_Mundial_de_Valores_para_Puerto%20Rico_20190617.pdf`
  (122 pp; the host's TLS chain is incomplete, so `curl -k`). It gives the design (p.16), a map of
  the regions (p.16), the questionnaire (pp.97-122) and Tabla 80, religion nationally (p.78).
- **The WVS-7 codebook** (`F00011055-WVS-7-Codebook-Variables-report`, mirrored at
  `gabors-data-analysis.com/ai-course/data/VWS/codebook.pdf`) has the code lists: annex
  `N_REGION_ISO` p.235-236, `N_REGION_WVS` p.257, `Q289CS` p.393-395.

**Design (report p.16):** 1,127 interviews "distributed proportionally across Puerto Rico by region
and socioeconomic level"; three municipios drawn at random in each of six regions; block groups
within each municipio classed low, middle and high, two drawn per class; Kish selection in the
household. The file has **no weights** (`W_WEIGHT` = 1 for everyone, `S018` and `PWGHT` constant),
**no PSU** (`I_PSU` is 0 or -4) and **no interviewer**. `D_INTERVIEW` is 630, 07, then a serial
1-1228, and does not carry the municipio (serial ranges and dates overlap across municipios).

## 2. The trap: a trailing semicolon shifts every column by one

The header has 404 names; every data row has 405 fields, the last empty. pandas' default reads the
first field as the index and pairs each remaining value with the header one place to its left. The
result is plausible everywhere: `A_YEAR` holds 630, `N_REGION_ISO` holds six region codes,
`N_REGION_WVS` holds eighteen town codes, `Q289` holds eight-digit codes. The supervisor's peek
read it correctly and this session's first read did not. `pr.py` asserts the trailing delimiter on
every row, reads with `index_col=False`, and asserts `doi`, `A_YEAR`, `B_COUNTRY` and
`D_INTERVIEW` hold their own values. Spec §12.

## 3. Geography

**`N_REGION_WVS` is the region** (codebook p.257): 630001 Norte 172, 630002 Sur 148, 630003 Oeste
157, 630004 Este 194, 630006 Centro 188, 630007 Metropolitana 268. No 630005.

**`N_REGION_ISO` is the municipio**, 18 of them (codebook p.235): 630000 plus the municipio's place
in the alphabetical list of 78, which `pr_geo.py` asserts for all 18. (US FIPS codes are not that
arithmetic past Florida, 72054.) Three per region:

| region | municipios sampled (interviews) | municipios in the region |
|---|---|---|
| Norte | Barceloneta 45, Toa Baja 65, Vega Baja 62 | 12 |
| Sur | Guayama 74, Peñuelas 31, Yauco 44 | 12 |
| Oeste | Hormigueros 73, Moca 52, San Germán 32 | 16 |
| Este | Canóvanas 59, Juncos 61, Río Grande 74 | 16 + Vieques, Culebra |
| Centro | Cayey 52, Naranjito 55, Corozal 80 | 14 |
| Metropolitana | Cataño 68, San Juan 151, Trujillo Alto 49 | 6 |

**The regions are defined only by the report's p.16 map**, which colours all 78 municipios. Read
municipio by municipio into `pr_geo.py`'s `REGION_OF` from a 2.5x render and two 8x crops (the
embedded image is about 2 px per point, so the crops are enlargements; colour, not label, is what
was read). The border municipios worth a second look if anyone doubts it: Quebradillas (Norte),
Maricao, Sabana Grande and Guánica (Oeste), Lares and Aguas Buenas (Centro), **Cidra (Este, not
Centro)**, Carolina (Metropolitana), Loíza and Maunabo (Este), Villalba and Patillas (Sur).

**Vieques and Culebra are not on the map.** Assigned to Este (Fajardo and Ceiba's region, where the
ferries run), 8,506 people in 2024. This build's call.

**Witnesses** (`pr.py`, `pr_geo.py`):
- the report's interview count for each of the six regions and eighteen municipios equals the file.
  The six region totals are all different, so a relabelling of regions cannot pass; the municipio
  totals tie once (Moca and Cayey, 52), which a swap of those two would survive only if both code
  columns were swapped together;
- `N_REGION_WVS` agrees with the municipio's region for 1,126 of 1,127. The exception is one
  Guayama interview coded Centro; the report's counts (Sur 149, Centro 187) side with the
  municipio. Asserted as exactly that one;
- name-free, in `pr_geo.py`: Mayagüez (westernmost centroid) is in Oeste, Ceiba (easternmost on the
  main island) in Este, Norte's centroid north of Sur's, Metropolitana the densest (1,705/km²).

**Sample share against adults does not pin the join, and says something about the design.** The
report says the allocation was proportional. Against 2020 census adults (P3):

| region | sample | adults | ratio |
|---|---|---|---|
| Norte | 15.3% | 15.9% | 0.96 |
| Sur | 13.2% | 12.9% | 1.02 |
| Oeste | 13.9% | 15.2% | 0.92 |
| Este | 17.2% | 18.7% | 0.92 |
| **Centro** | **16.6%** | **10.8%** | **1.54** |
| Metropolitana | 23.8% | 26.5% | 0.90 |

r = +0.855; 37 of the 720 orderings reach it. Centro's 187 interviews are also the report's figure,
so the over-sampling is the team's, not a label error, and with four regions of nearly equal size
the permutation has no power anyway. `pr.py` asserts the pattern (Centro above 1.3, the rest within
15%) so a changed file is noticed. It moves no region's shares, which are within-region; nationally
the build weights by population.

## 4. Population

- **Drawn:** Census Bureau Vintage 2024, `PRM-EST2024-POP`, July 1 2024, 3,203,295, per municipio,
  summed to region. `www2.census.gov/programs-surveys/popest/tables/2020-2024/municipios/totals/`.
- **Adults for the held-out check:** 2020 census redistricting file `pr2020.pl.zip` (P1 total,
  P3 18+), 3,285,874 and 2,724,903. The estimates base equals the census count in every municipio.
- The Census data API now refuses requests without a key (2026-09-14: an HTML "Missing Key" page
  behind a 302). The static files above need none.
- Boundaries: the shared `data/geo/cb_2020_us_county_500k.zip`, state 72, joined on GEOID. Names in
  its DBF read as mojibake without an encoding, so names come from the census and estimates files.

## 5. Religion and mapping

`Q289` on the Puerto Rican card (report p.121): `0 No pertenece, 1 Católico, 2 Protestante,
3 Ortodoxo, 4 Judío, 5 Musulmanes, 6 Hindú, 7 Budista, 8 Otros (escribir)`. No `Q289CS`; `Q289CS9`
is one to one with `Q289` (asserted). Frequencies equal Tabla 80 exactly.

| answer | n | % answering | node |
|---|---|---|---|
| Católico | 554 | 49.60 | `christianity.catholic.latin` |
| Otros | 227 | 20.32 | `christianity` (the root; `christianity.other` until §11) |
| No pertenece | 227 | 20.32 | `unaffiliated` |
| Protestante | 103 | 9.22 | `christianity.protestant` |
| Budista | 5 | 0.45 | `buddhism` |
| Hindú | 1 | 0.09 | `hinduism` |
| no answer (-2) | 10 | | excluded, `gap=` |

**`Otros` is a write-in**, and the archive harmonised all 227 to 80000000 `Other Christian; nfd`;
the text is not in the file. Mapped to the `christianity` root (built on `christianity.other`,
moved after the §10 review, §11), not `christianity.evangelical`, though most are very likely
evangelical and Pentecostal: `taxonomy/pr2018.py`'s `REVIEW` has the argument (the node notes of
`christianity.other` and `christianity.evangelical`; Chile's coded Christian write-ins and Peru's
`Cristiano` on the root; Pew 2014's 33% Protestant against Protestante plus Otros at 29.5% here).

## 6. Which categories carry geography

§14.16's split-half on the sampling unit available, the municipio. Each region's three municipios
split one against two, every distinct halving of the country (3 x 6^5 = 23,328), statistic the
median Spearman over the six regions. **Null: the 18 municipios dealt into random groups of three**,
400 draws. Municipios nest in regions rather than crossing them the way ESS rounds cross units, so
Belgium's within-round label shuffle has no analogue; regrouping is the permutation that destroys
region structure. The formula bar on six units, +0.877, is printed and not used. Vetoes: chi-square
over regions at 0.05, no municipio with half of a category's respondents, no municipio-day with half.

| | n | median | null 95th | p | chi² p | top municipio | verdict |
|---|---|---|---|---|---|---|---|
| Católico | 554 | +0.714 | +0.543 | 0.017 | 1e-8 | 0.15 | **own geography** |
| Otros | 227 | +0.771 | +0.486 | 0.005 | 8e-7 | 0.10 | **own geography** |
| Protestante | 103 | +0.029 | +0.543 | 0.464 | 0.045 | 0.17 | fails |
| No pertenece | 227 | +0.143 | +0.543 | 0.357 | 0.26 | 0.12 | fails; regions do not differ |
| Budista | 5 | +1.000 | +0.775 | 0.007 | 0.006 | **0.60** | **refused**, 3 of 5 in San Juan |
| Hindú | 1 | | | | | 1.00 | no test |

Protestante plus Otros as one column, as a diagnostic: +0.771, p 0.010. No failing category has a
standout region (Honduras's test): No pertenece tops both halves in Oeste 12.9% of the time.

**Construction: §9bi's residual.** Católico and Otros keep their measured regional shares; the other
four share each region's remainder at national proportions. Both constructions are printed every
run. The flat alternative (Honduras) would scale the carried shares and move Centro's Catholics from
64.0% to 59.0%; the residual's cost is Budista and Hindú at up to 1.2x national in the regions with
the largest remainders (Honduras rejected the residual at 3x). Protestante and No pertenece vary as
drawn only because the remainder does (Oeste Protestants 9.6% drawn against 5.9% measured).

## 7. Result

As drawn, share of people drawn:

| region | Católico | Otros | No pertenece | Protestante |
|---|---|---|---|---|
| Norte | 46.5 | 25.0 | 19.2 | 8.7 |
| Sur | 53.7 | 21.5 | 16.8 | 7.6 |
| Oeste | 54.2 | 14.4 | 21.2 | 9.6 |
| Este | 30.7 | 33.3 | 24.3 | 11.0 |
| Centro | 64.0 | 11.8 | 16.3 | 7.4 |
| Metropolitana | 50.2 | 16.6 | 22.4 | 10.2 |
| Puerto Rico | 48.52 | 20.87 | 20.68 | 9.38 |

Budista 0.46%, Hindú 0.09%. No answer 0.93% of people, not drawn.

## 8. Kontur

`kontur_population_PR_20231101`, 14,908 hexes, 3,260,315 people; 364 hexes (0.78%) fall outside the
boundary file's coastline and are dropped. Kontur is 1.01x the 2024 estimate, per region 0.97-1.06.
**Density-cap scan (`plateau_scan2.py pr`): peak 9,391/km², nothing at the 46,200 cap, no blocks.**
Nothing to record for the guard.

## 9. Not done, and routes for whoever reopens it

- **Puerto Rico is in WVS waves 3 (1995) and 4 (2001)** as well (IHSN `PRI_1995_WVS-W3_v01_M`,
  `PRI_2001_WVS-W4_v01_M`; the report says both used the same multi-stage design). If either file
  carries a region or municipio, a second round would give a real outside witness for the ordering
  and could license Protestante or No pertenece. Not looked for: they are in the WVS longitudinal
  file, behind the same download route Anita used.
- The report prints no religion table below the national one.
- Pew's 2014 Religion in Latin America survey (Puerto Rico n≈1,500) is behind an account.
- The U.S. Religion Census (ASARB) does not cover Puerto Rico.
- The report's p.16 map is the only region definition found; the WVS documentation page for the
  country was not opened (it is JavaScript-driven).

## 10. Review, 2026-09-14 (`f95259a4-prrev`)

Read from the report, the Planning Board's resolution and the file. Nothing rebuilt, no note edits.

- **The region reading holds.** The p.16 map is three embedded strips, 973x576 px together; at native
  resolution every border municipio in §3 is where `REGION_OF` puts it. **No official six-region
  grouping exists**: the Instituto's own classification page ("Regiones de Puerto Rico") points to the
  Junta de Planificación's resolution JP-2014-309, primera extensión (5 August 2015), which has 11
  áreas funcionales. It supports the Vieques and Culebra call: its Área Funcional de Fajardo is Ceiba,
  Culebra, Fajardo, Luquillo, Río Grande and Vieques, all Este here.
- **The split-half is a fair test and its p-values are stable.** Three seeds of 1,500 regroupings give
  Católico p 0.015-0.017 and Otros 0.005-0.006. A different statistic on the same null, the
  between-region F on municipio shares weighted by respondents (20,000 regroupings), gives the same
  verdict for all six answers: Católico 0.026, Otros 0.007, Budista 0.007, Protestante 0.48, No
  pertenece 0.45, Hindú 0.25.
- **Otros is robust; Católico is thinner.** Dropping any one municipio, Otros's F p stays at or under
  0.023. Católico's goes over 0.05 for four of eighteen (Juncos 0.097, Río Grande 0.079, Corozal
  0.062, Naranjito 0.059). Centro's 64% is Corozal 71% and Naranjito 76%, with Cayey at 40%. The rule
  was applied as written and nothing moves.
- **The construction complies with the rule written since** (spec §12, residual unless it draws a
  category at 2x where the survey found none; Puerto Rico 1.20x). The leftover as one column
  (Protestante, No pertenece, Budista, Hindú) has no geography of its own (F p 0.40, split-half p
  0.28-0.30, chi-square 0.084), so the drawn spread of Protestants and no religion comes from Católico
  and Otros, as the note says. It reverses nothing: drawn against measured is +0.83 for Protestante
  and +0.60 for No pertenece.
- **Centro's over-sampling moves at most 0.21 points.** Carried shares are within-region. The residual
  splits each leftover at unweighted sample-wide ratios, which Centro's 1.54x does enter; weighting
  them by 2020 adults moves no drawn cell by more than 0.21 points (No pertenece in Este).
- **`Otros` should be on the `christianity` root, not `christianity.other`, and not evangelical.**
  `christianity.other`'s node note is "bodies with no branch to belong to, not a residual". The cells
  §5 cites (gh, ke, bb, gy, mu, hu) are census rows adding up named small bodies, which fits that; a
  write-in coded `Other Christian; nfd` names no body. The precedent for that is the root:
  `cl2024.py` sends INE's coded Christian write-ins there, `pe2017.py` sends `Cristiano` there, and
  Canada's `Christian, n.o.s.` is spec §6.6's example of a branch drawn with its own dots.
  Evangelical is ruled out by its own node note (a source that collects `Evangelical` as an answer),
  and every Latin American cell on it names the word. The card (report p.121) is the WVS template with
  its "enmendar la lista" instruction not acted on, so evangelicals and Pentecostals had no box. One
  line in `taxonomy/pr2018.py` and a re-scatter; not done here, noted in `queue.md`.
- **Screenshot:** dots on land and on the people. Protestant, unspecified and Other Christian are near
  the same pale yellow in adjacent legend rows, together 30% of the island; worth a human eye.
- `check_md.py` clean, `built_countries.py --check` OK, `check_rollup.py pr` nothing orphaned.

## 11. Review fix, 2026-09-14 (`f95259a4-pr2`)

Supervisor-decided, from §10. Two changes and one measurement.

- **`Otros` moved from `christianity.other` to the `christianity` root.** One line in
  `taxonomy/pr2018.py`'s `MAP`; its `REVIEW` entry now gives the three reasons: `christianity.other`'s
  note ("bodies with no branch to belong to, not a residual") fits the census rows of named small
  bodies mapped there in gh, ke, bb, gy, mu and hu, not a write-in coded only `Other Christian; nfd`;
  the root is where the map already puts a Christian answer that names no church (`cl2024.py`'s coded
  write-ins, `pe2017.py`'s `Cristiano`, Canada's `Christian, n.o.s.` in spec §6.6); and
  `christianity.evangelical`'s own note keeps it for a source that collects `Evangelical` as an answer.
  Nothing in `pr.csv` changes, since nodes resolve at run time.
- **Checked.** `check_mapping.py pr`: 662,366 people on `christianity`, nothing unmapped, every target
  a node. Re-scattered both editions: 3,170 and 314 dots, as before. The exact test re-runs
  `scatter.py`'s allocation (Kontur cap, water clip, Hilbert key per unit, per-node carry) with the old
  and the new mapping: **dots per (unit, node, tier) are identical once `christianity.other` is read as
  `christianity`, at both editions.** A point-in-hex join of the dot files agrees with the allocation
  exactly for the old dots and differs by one dot between neighbouring regions twice for the new ones
  (Este/Centro Catholics at 1:1,000, Norte/Metropolitana `christianity` at 1:10,000). That is 4-dp
  coordinate rounding putting a border dot in the next region's hex, not a moved count; `scatter.py`
  asserts each row places exactly its allocation. Scripts kept in the session scratchpad
  (`pr_alloc_check.py`, `pr_dot_snapshot.py`), not the repo.
- **A trap on the way.** `coverage.py` run straight after the re-scatter failed with `pr
  christianity.other draws dots but is not in the country's coverage`. Its `verify` reads the node
  list from `counts.json`, which only `tiles.py` rewrites, so after a remap of a drawn country step 9
  reports the old node until the build tail has run. Not a real gap.
- **`note_public`: cut "from 64% of the central mountain region".** It rested on two of Centro's three
  municipios (Corozal 71%, Naranjito 76%, Cayey 40%; §10). The sentence now reads "Catholics are about
  half of Puerto Rico and 31% of the east, where a third of people gave some other Christian answer";
  the rest of the note is unchanged. `check_md.py` clean.
- **Build tail** under the lock (`build_tail.py --id f95259a4-pr2`, 149 countries, exit 0). After it,
  `coverage.py` ok and `built_countries.py --check` OK.
- **Palette: measured, nothing changed** (the palette is Anita's to tune). §10 saw `Protestant,
  unspecified` and `Other Christian` as near-identical pale yellows. The remap does not change the
  colour: in the all-religions view `Other Christian` was #f8dc4f, hsl(50,92,64), and the root's
  `Christianity, unspecified` row is the same #f8dc4f. Against `Protestant, unspecified` #f8dd81,
  hsl(46,90,74), 662 dots against 297 at 1:1,000: **CIE76 21.5, CIEDE2000 6.0**. The two differ
  mostly in lightness (64 against 74), and CIE76 flatters saturated colours, so 6.0 is the figure
  that matches the eye. Neither repo checker names the pair: `check_overview.py` holds pairs inside
  one family to dE 12 and pairs across families to 25, and this is a same-family pair at 21.5;
  `check_palette.py` measures families, and Puerto Rico's four are all at least 25 apart. The rest of
  the view, CIE76 then CIEDE2000: `Christianity, unspecified` / Catholic #eca832 25.3, 16.2; Catholic /
  Protestant 29.4, 15.7; Buddhism / Hinduism 27.4, 20.2 (14 and 2 dots). With Christianity selected
  Protestant turns blue (#5476fc), and the only close pair left is `Christianity, unspecified` /
  Latin Catholic at 25.3, 16.2. CIEDE2000 is a scratch implementation checked against the 12 published
  test pairs of Sharma, Wu and Dalal (2005), all equal to 4 dp.
