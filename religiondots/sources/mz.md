# Mozambique — INE, IV RGPH 2017 by province, and III RGPH 2007 by district

Built 2026-09-14 at province. Upgraded 2026-09-15 (`cb8b206e-mz`) to 121 districts as they were
in 2007, in nine provinces, with Cabo Delgado and Manica still whole: §7. `sources/mz.py`,
`sources/mz_2007.py`, `sources/mz_geo.py`, `sources/mz_grid.py`, `taxonomy/mz2017.py`,
`countries/mz.py`.

**26,899,105 people, 123 units (121 districts as of 2007, 2 provinces), 8 columns (7 drawn),
97.49% drawn.** The value of the country is one column: `Zione/Sião`, the Zionist churches,
4,199,083 people, 15.6%. Sections 1 to 6 are the 2017 province build and still hold, except
§1's "nothing finer than province is published", which §7 reverses for 2007.

## 1. The route, and why it is the Wayback Machine

INE released the 2017 definitive results in 2019 as a set of xlsx tables per province on its
old Plone site, `http://www.ine.gov.mz/iv-rgph-2017/<province>/`. Quadro 11, *População por
religião, segundo área de residência, idade e sexo*, is in every set. That site was replaced
by a Liferay portal in 2022 and every `/iv-rgph-2017/` path now 404s.

What was tried on the live host, 2026-09-14:

| route | result |
|---|---|
| `http://www.ine.gov.mz/...` | connection refused on port 80 |
| `https://www.ine.gov.mz/...` | TLS verification fails: the server does not send its intermediate certificate. With verification off it answers normally. Not a wall. |
| `/censo-2017/-/document_library/pfpz/view/<folder>` | 200. One folder per province (Niassa 44373, Cabo Delgado 44346, Nampula 44370, Zambézia 44385, Tete 44382, Manica 44358, Sofala 44379, Inhambane 44352, Gaza 44349, Maputo Província 44364, Maputo Cidade 44361, Moçambique 44367) and `Territoriais (Distritais)` 44394. The page HTML carries no document list. |
| `/o/headless-delivery/v1.0/document-folders/<id>/documents` | 403 `{ }` |
| `/api/jsonws/dlapp/get-file-entries/...` | 403 `{}` |
| `/documents/d/guest/<slug>` guesses for Quadro 11 | 404 |
| PX-Web `http://41.94.86.11/Censo2017/pxweb/pt-PT/`, linked from the portal | connection refused |
| Wayback CDX for `41.94.86.11/Censo2017*` | 503 three times; not settled |

The Wayback CDX query that found the files (it 503s in bursts; retry):

```
http://web.archive.org/cdx/search/cdx?url=ine.gov.mz/iv-rgph-2017/*&output=txt&limit=3000
    &fl=original,statuscode,mimetype,timestamp&collapse=urlkey&filter=original:.*religi.*
```

It lists Quadro 11 and Quadro 13 (somatic type × religion) for all eleven provinces and the
national set, captured 2019-11-14 as `200 application/vnd.openxmlformats...sheet`.
`sources/mz.py` fetches Quadro 11 with the `id_` flag and checks the zip magic.

**Nothing finer than province is published.** `LISTA DE QUADROS.xlsx` on `mozdata.ine.gov.mz`
(catalog 24) lists only Quadros 3, 6, 7 and 8 by district (sources.md §11w). The 2021
provincial report for Nampula (on `sina.gov.mz`, 68 pages) has religion as one chart, province
level, 2007 against 2017. The national *Indicadores Sócio-Demográficos* (2022) has it
national and urban/rural. The upgrade path is INE's offer of custom tabulations to any
administrative level (mozdata abstract, sources.md §11d) or the licensed microdata; both need
a person and neither was attempted.

## 2. The numbers are INE's printed ones, and UNSD's are not

Three witnesses to the national row:

| | Quadro 11 national xlsx | Brochura, p. 42 | UNSD table 28 |
|---|---:|---:|---:|
| Total | 26,899,105 | 26,899,105 | 26,899,105 |
| Católica | 7,313,576 | 7,313,576 | 7,344,788 |
| Anglicana | 457,716 | 457,716 | 461,714 |
| Islâmica | 5,094,024 | 5,094,024 | 5,131,177 |
| Zione/Sião | 4,199,108 | 4,199,108 | 4,389,294 |
| Evangélica/Pentecostal | 4,124,710 | 4,124,710 | 4,481,176 |
| Sem religião | 3,737,354 | 3,737,354 | 3,625,355 |
| Outra | 1,297,856 | 1,297,856 | 1,148,051 |
| Desconhecida | 674,761 | 674,761 | 317,550 |

The xlsx and INE's printed brochure agree to the person. UNSD has half the unknowns and the
difference lands mostly on the two big Protestant cells, which looks like an imputation done
after publication. §11w's "Saio/Zione 16.3%" came from UNSD's figure; INE's own is 15.6%.

The Nampula 2021 provincial report's 2017 chart (37.8 / 39.7 / 2.1 / 6.2 / 0.8 / 1.7 / 10.3
/ 1.5) matches Nampula's Quadro 11 shares to the decimal, so the provincial files are also
what INE's later provincial reporting uses.

**The provinces against the national table.** Totals agree to the person; categories are
within 150 (Outra +150, Desconhecida -81, Católica -29, Zione/Sião -25, the rest under 8).
`SLACK = 200` in `mz.py`. Provincial files are drawn.

**Layout traps.** A provincial Quadro 11 has two `TOTAL` columns (all, and all less
`DESCONHEC.`); the national one has one, so every religion column is one place further right
in the provincial files. Columns are matched on header text. Maputo Cidade's table has no
residence split and a different title wording. Gaza's file name is truncated on INE's server.

## 3. Mapping calls (taxonomy/mz2017.py REVIEW has the full reasoning)

- `Zione/Sião` -> `christianity.africaninstituted`, bare parent.
- `Evangélica/Pentecostal` -> `christianity.evangelical`, on the Latin American `Evangélica y
  Pentecostal` precedent (cr, gt, ec, sv, pa) and Angola's `Evangélica`.
- **`Sem religião` -> `unaffiliated`, Anita's ruling of the night of 2026-09-14.** The
  questionnaire (Brochura p. 210, question P11) words the box `Sem religião (ateu, animista,
  agnóstico,...)`, so traditional religion has no cell of its own and sits inside this one. Tete
  32.8%, Manica 25.0%, Sofala 24.9%, Niassa 0.75%. It was `unaffiliated` at the first build,
  `unknown` from Anita's evening ruling (sources.md §9dn), and `unaffiliated` again once the draft
  "no religion" procedure in `WORKFLOW_PLAN.md` was applied to §6: the surveys that offer both
  answers put the box at about 93-98% no religion, over the procedure's 80% bar. Both surveys
  measure what people call themselves, not practice.
- `Outra` -> `other.mz`, new per-source residual node.
- `Desconhecida` excluded, 674,680, 2.51%, `gap_share=0.0251` confirmed both ways by
  `tools/gap_share.py`.

## 4. Geography and placement

COD-AB Mozambique (`cod-ab-moz`, shapefile, `valid_on` 2025-01-01), ADM1, 11 polygons. COD's
`adm1_pcode` MZ01..MZ11 is in INE's own province order, so the census `geo_id` and the pcode
coincide; the join is still by folded name and asserted both ways. Two names differ after
folding: COD `Maputo` = INE `Maputo Província`, COD `Cidade de Maputo` = INE `Maputo Cidade`.

Placement is Kontur 400 m hexagons (`kontur_population_MZ_20231101`), 303,270 kept, 1,974
dropped with centroids outside every province (273,183 people, 0.80%). Kontur's total is
33.8M against the 2017 census's 26.9M, ratio 1.257, which is six years of growth; it is only
a within-province weight.

## 5. What would reopen it

- The PX-Web database at `41.94.86.11/Censo2017` if it comes back or turns up in the Wayback
  Machine; PX-Web cubes are often by district when the published tables are not.
- INE's custom tabulation offer, by district (154 units in 2017).
- The 2007 census (III RGPH) district volumes: **drawn 2026-09-15, §7.** What is left of that
  route: Cabo Delgado's and Manica's volumes, which the Wayback Machine never captured (a browser
  search of the live portal, or an email to INE), and Cabo Delgado's district tier would be a §14
  question for Anita before drawing it (§7.6); and Zambézia's 2017 Quadro 3, which would give its
  2007 districts 2017 populations in place of 2007 proportions.

## 6. How `Sem religião` splits, from national sources (researched 2026-09-14, drawn the same night)

Anita, 2026-09-14: *"are we able to find any national level source for roughly how many of these
people are indigenous religion and how many are no religion?"* **For Mozambique the answer runs the
opposite way from Laos** (`sources/la.md` §10): the surveys that offer both answers find almost
nobody naming traditional religion, and a no-religion share that follows the census box province
by province. Anita ruled on it the same night: the box is drawn as `unaffiliated` (§3).

**No census splits it.** INE's 2007 census used the same categories as 2017, `Sem religião` and
`Outra` included (*Estatísticas de Cultura, Província de Maputo 2021*, Gráfico 4, p.10), so there
was no animist box then either. 1997 is unconfirmed (the Wayback Machine was offline). Pew 2012's
`folk religions 7.4%` is *"based on 2007 Census, adjusted to account for underrepresented religious
groups"* (*The Global Religious Landscape*, p.76), which reads animism into `Outra`; not a
measurement.

**The surveys that offer both answers:**

| source | year | base | traditional | no religion | n |
|---|---|---|---:|---:|---:|
| Afrobarometer R4-R9 pooled | 2008-2023 | adults 18+, weighted | **0.17%** (21) | none 7.04%, atheist 0.53%, agnostic 0.04% | 10,467 |
| Afrobarometer R8 + R9 | 2021-2023 | same | **0.27%** (6) | none 12.56%, atheist 0.17%, agnostic 0.09% | 2,204 |
| Pew, *Tolerance and Tension*, Q32 | 2009 | adults 18+ | **1%** | 13% (atheist, agnostic or nothing in particular) | 1,500 |

Afrobarometer is the six merged Mozambique files already in `data/raw/afrobarometer`, religion item
Q90, Q98A, Q98 or Q95 by round, each round's own within-country weight, recomputed 2026-09-14 from
scratch scripts. Traditional by round, R4 to R9: 0.12, 0.25, 0.07, 0.14, 0.23, 0.30%; none, atheist
and agnostic together: 8.1, 4.3, 6.2, 7.2, 13.0, 12.6%. Pew is *Tolerance and Tension: Islam and
Christianity in Sub-Saharan Africa* (April 2010), table `Religious Affiliation`, printed p.20:
Christian 63, Muslim 23, Traditional African religions 1, Unaffiliated 13, other or no answer 1. Its
show card offered *"ancestral, tribal, animist, or other traditional African religion"* on the same
list as *"atheist; agnostic; something else; nothing in particular"*.

**The province pattern is the census's.** Pooled Afrobarometer `None` is highest in Tete (16.4%),
Sofala (14.6%) and Manica (12.3%) and lowest in Niassa (0.6%), the census box's own order (Tete
32.8%, Manica 25.0%, Sofala 24.9%, Niassa 0.75%). Traditional is under 0.6% in every province. Of
the free-text `Other` answers in R5-R9, one names a *curandeiro* and nearly all the rest are churches.

**What it implies, and its limit.** On Afrobarometer, traditional religion is about 2% of
traditional plus unaffiliated, roughly 80,000 of the 3.74 million; on Pew 2009 it is about 7%,
roughly 270,000, and whole-number rounding allows anywhere from about 140,000 to 400,000. **Both
measure what people call themselves.** Someone who keeps ancestral practice and answers *none* is
counted as none, and that is exactly the person the census's `animista` gloss was written for. So the
surveys bound traditional religion as an identity and say nothing about practice. Both cover adults
only.

**The modelled figure is the outlier.** The World Religion Database (via ARDA,
`thearda.com/world-religion/national-profiles?u=156c`, re-read 2026-09-14) has ethnic religionists
at **26.06%**, agnostics 0.36% and atheists 0.08%. No method is given, and WRD counts practice
rather than self-description, so it is not comparable with the census box or the surveys. Pew 2025's
row (unaffiliated 18.9% in 2010, 14.3% in 2020) was read only through Our World in Data's
`religious-composition` grapher, because pewresearch.org returned 403; not re-read here.

Checked, no split: DHS 2003, DHS 2011, IMASIDA 2015 and IDS 2022-23 (Quadro 3.1 in each has the same
`Sem religião` box, 7.0% of women and 12.5% of men aged 15-49 in 2022-23, FR389 p.55); INE's
*Indicadores Sócio-Demográficos* 2022 (p.31, 13.5%); the US State Department's 2023 report (ecoi.net
copy: 14% unaffiliated, citing INE, and indigenous belief *"a category not included in government
census figures"*). Not checked: MICS 2008 (report not found), Afrobarometer R3 (2005, not on disk),
IPUMS's religion codes for 2007 and 2017 (the table did not load), UNSD table 28's download.

## 7. The district upgrade: 2007 districts fitted to 2017 totals (2026-09-15, `cb8b206e-mz`)

**What is drawn now.** 121 districts as they were in 2007, in nine provinces, and Cabo Delgado and
Manica whole: 123 units, 219,000 people each on average. The 2007 census supplies each district's
mix of answers; the 2017 census supplies every province's total per answer and, where it was
published, every district's population. Every fitted row is `derived` with roll NOWHERE, so
`inferred dots: not shown` empties the nine provinces and keeps Cabo Delgado and Manica.
26,222 dots at 1:1,000 and 2,618 at 1:10,000, on 853 (unit, node) rows.

### 7.1 The sources

**2007: INE, *III RGPH 2007: Indicadores Sócio-Demográficos Distritais*, one volume per province**
(© 2010, printed 2012), Quadro 8.1, printed as `Quadro 8.` in Inhambane, Gaza, Maputo Província and
Maputo Cidade: *Distribuição percentual da população por religião segundo distritos*. The same eight
answers as 2017's Quadro 11, one decimal, with each district's population `N`. Nine volumes from the
Wayback Machine, sizes pinned in `sources/mz_2007.py` `VOLUMES` (the 2015 Plone tree
`.../relatorio-de-indicadores-distritais-2007/<slug>/at_download/file`; Cidade de Maputo only on the
2019 tree). Table pages: Niassa 20, Nampula 21, Zambézia 22, Tete 20, Sofala 18, Inhambane 19, Gaza
20, Maputo Província 21, Maputo Cidade 15-16. Cabo Delgado and Manica: not captured (the scout's
search, not repeated here).

**2017: Quadro 3 per province**, *População por idade, segundo área de residência, distrito e sexo*,
Wayback 2019 captures, sizes pinned in `Q3`. Each province's T O T A L equals its Quadro 11 total.
**Zambézia's 2017 set has no district table at all**: a CDX prefix query for `iv-rgph-2017/zambezia/`
(2026-09-15) lists Quadros 1, 2, 4, 5 and 9 to 59 and no 3, 6, 7 or 8, and its 2018 provincial
yearbook (`anuario-zambezia-2018.xls`, Wayback 20191115164611) has no census population by district.
**Tete's Quadro 3 districts sum to 2,551,824 against its own 2,551,826** (summed by hand from the
xlsx as well), pinned as `Q3_SLACK` and put on the two largest groups.

### 7.2 The 2007 tables' checks, all in `check_volume`

| province | districts | N | worst N-weighted share against the Total row |
|---|---:|---:|---:|
| Niassa | 16 | 1,170,783 | 0.066 |
| Nampula | 21 | 3,985,613 | 0.051 |
| Zambézia | 17 | 3,849,455 | 0.049 |
| Tete | 13 | 1,783,967 | 0.046 |
| Sofala | 13 | 1,642,920 | 0.056 |
| Inhambane | 14 | 1,271,818 | 0.059 |
| Gaza | 12 | 1,228,514 | 0.049, against the corrected row |
| Maputo Província | 8 | 1,205,709 | 0.055 |
| Maputo Cidade | 7 | 1,094,628 | 0.053 |

Every row's eight shares close to 100.0 within 0.3; every volume's district N sum to its Total row
to the person; the caption names the province; the district names and order are `DISTRICTS`'; and
the parsed table's digest is pinned. The nine N total 17,233,407; not checked against a national
2007 figure (Cabo Delgado's 2007 population is 1,606,568 in INE's 2008 *Estatísticas do Distrito de
Ancuabe*; Manica's was not read).

**N printed off its row** in five volumes, so the reader pairs the kth count with the kth row and the
two sums above prove it: Tete's Changara, Chifunde, Moatize (between name and shares), Mutarara,
Tsangano and Zumbo; Sofala's Chemba, Cheringoma, Chibabava, Marromeu, Muanza and Nhamatanda;
Inhambane's Morrumbene, Panda and Vilanculos; Gaza's Chicualacuala, Chigubo and Distrito de Xai-Xai;
Maputo Cidade's DM 4 to 7.

**Gaza's printed Total row is its districts' eight numbers in the wrong cells.** Printed: Católica
37.5, Anglicana 15.4, Islâmica 15.8, Zione/Sião 19.8, Evangélica/Pentecostal 7.0, Sem religião 0.9,
Outra 3.0, Desconhecida 0.6. N-weighted districts: 15.38, 3.01, 0.89, 37.55, 15.77, 19.82, 6.96, 0.63.
No district is over 22% Católica. The Total row is used only by this check.

**One template, nine volumes.** Nampula's credits page says Niassa and Maputo Província; Zambézia's
commentary above the table is Tete's. Only the caption is trusted for the province.

### 7.3 The vintage construction, and why

Three were on the table: (a) draw 2007 as printed; (b) scale each 2007 district's count of each
answer so the province matches 2017 (one margin); (c) IPF to two 2017 margins, answers per province
and population per district (Switzerland's, `countries/ch.py`; ask 003 allows a mixed vintage when
the method is sound). **Chose (c).** (a) would draw Tete at 38.8% no religion where 2017 counted
32.8%, beside Cabo Delgado and Manica at 2017. (b) holds each district's share of its province's
people at 2007: Marracuene 7.0% of Maputo Província against 11.5% in 2017, Cidade de Tete 8.7% of
Tete against 12.0%, Matola 55.7% against 54.1%.

**What (c) assumes, and where it shows.** It keeps 2007's pattern of which districts lean which way,
and moves every district of a province together when the province's mix changed. The largest moves,
2007 share to fitted, in points:

| district | answer | 2007 | fitted | the province, 2007 to 2017 |
|---|---|---:|---:|---|
| Matutuine | Evangélica/Pentecostal | 25.4 | 47.1 | Maputo Província 16.9 to 34.4 |
| Marracuene | Evangélica/Pentecostal | 20.5 | 39.8 | the same |
| Cidade da Matola | Evangélica/Pentecostal | 17.6 | 35.1 | the same |
| Massangena | Sem religião | 48.2 | 31.3 | Gaza 19.8 to 12.5 |
| Chigubo | Sem religião | 49.3 | 33.4 | the same |
| Lugela | Outra | 55.6 | 40.5 | Zambézia 14.6 to 9.6 |

If Maputo Província's evangelical growth was concentrated in Matola, the map spreads it over the
province, and nothing published can say. Zambézia's unknowns went from 0.8% to 5.8%, which the fit
also spreads (Lugela 10.5%); they are not drawn.

**Zambézia keeps 2007's population proportions** between its districts, because no 2017 district
table was captured: its seventeen districts are one row group, so only the answer margins bind.

### 7.4 The 2007 districts on today's boundaries

COD-AB (2025) has 161 districts; these nine provinces had 121 in 2007. `DISTRICTS` lists each 2007
district's successors, and a 2007 district is the union of their administrative posts (COD-AB adm3),
dissolved in `sources/mz_geo.py::build_units`. The posts tile every province (area ratio 1.0000 in all
nine). Districts made since 2007 from a raised post: Larde (from Moma) and Liupo (Mogincual); Luabo
(Chinde), Mulevala (Ile), Mocubela (Maganja da Costa), Molumbo (Milange) and Derre (Morrumbala);
Marara (Changara) and Doa (Mutarara); Mapai (Chicualacuala). Renamed: Distrito de Lichinga to
Chimbonila, Nampula-Rapale to Rapale, the old Distrito de Xai-Xai to Limpopo. The pcodes are checked
against COD-AB's names, with the twelve that differ pinned in `NAME_DIFFERS`.

**Six posts changed district, `POST_MOVES`.** The 2017/2007 growth witness is each group's growth
over its province's:

| post | in 2017 | in 2007 | evidence |
|---|---|---|---|
| Anchilo | Cidade de Nampula | Nampula-Rapale | Rapale alone 0.82 against the province's 1.38; with the city 1.37 |
| Meponda, Lussanhando | Cidade de Lichinga | Distrito de Lichinga | Chimbonila alone 0.76 against 1.46, the city alone 1.70; together 1.33 |
| Mazucane, Nguzene | Chongoene | Mandlacaze | COD-AB lists them under Chongoene with Chongoene post; Mandlakazi alone 0.83 against 1.13, old Xai-Xai alone 1.29; together 1.09 |
| Maquival | Quelimane | Nicoadala | Portuguese Wikipedia's *Nicoadala (distrito)*, quoted in the code; Kontur over 2007 N, over Zambézia's, Nicoadala 0.55 and Quelimane 1.26 without the move, 0.90 and 0.84 with it |

Where posts moved, the old districts are one row group in the fit and keep their 2007 proportions to
each other: Cidade de Lichinga with Distrito de Lichinga, Cidade de Nampula with Nampula-Rapale,
Mandlacaze with Distrito de Xai-Xai. With the moves in, every group reads 0.71 to 1.63 of its
province (Marracuene 1.63 and Cidade de Tete 1.38 are real suburban growth; `GROWTH_BAND` is
0.65 to 2.0). **Not checked:** a small post that moved without pushing a district out of the band; no
2007 list of posts per district was found (INE's 2008 *Estatísticas do Distrito* print district
totals only; Ancuabe's read).

**Kontur against the fitted 2017 populations** (`sources/mz_grid.py::kontur_witness`): 116 of 123
units within 0.67 to 1.5 of the country's ratio. Outside: Boane 0.53, Chinde 0.61, Mutarara 1.53,
Chifunde 1.61, Macanga 1.73, Moatize 1.88, Lago 2.05. All but Chinde have a 2017 district count, so
those are Kontur's error; Chinde is on 2007 proportions and could be either. `kontur_cap.csv`'s one
Mozambique row (Cuamba, `real`) still names unit `MZ01`: the registry matches on coordinates, so the
label is stale and nothing reads it.

### 7.5 Cabo Delgado, Manica and §14

Both are drawn whole from 2017's Quadro 11, `measured`, because neither 2007 volume was captured.
**Cabo Delgado is also a §14 case**: an armed Islamist insurgency since 2017, and a district map of
Muslim and Christian shares there is the shape raised for Burkina Faso and Mali under ask 018, which
ruled only on those two at their published grain. Per this session's brief it is left at province
and no ask was filed. If its volume turns up, raise its district tier with Anita before drawing it.
Manica has no safety flag; its volume is only missing.

### 7.6 Calls someone might reverse

- Fitting to 2017 (c) rather than drawing 2007 as printed.
- Zambézia's districts on 2007 population proportions.
- The six post moves, Maquival's especially, which rests on Wikipedia and Kontur.
- A province-wide change spread evenly over its districts (Maputo Província's evangelicals).

### 7.7 Review, 2026-09-15 (`cb8b206e-rev7`)

- **Checks.** `check_md.py` clean; `built_countries.py --check` OK; `check_rollup.py mz` reports
  22,151,691 people (84.5%) gone under `inferred dots: not shown`. That is §7's NOWHERE doing what it
  says, and Switzerland's outcome, so nothing to change.
- **Every figure in `note_public` matches `mz.csv` and `mz_districts.csv`**: Zione/Sião 4,199,083
  (15.6%), 73.7% of it in the six provinces from Manica south; Funhalouro 57.4% and Mabote 54.6%
  (51.7% and 50.1% printed in 2007); Changara 62.1% no religion fitted, against 71.4% printed; Lago
  38.7% Anglican fitted, against 34.0%.
- **One note sentence fixed.** It said Niassa's districts "along the lake and the Tanzanian border,
  Meumbe, Mavago and N'gauma among them" are over 95% Muslim. Measured on `mz_units.gpkg`: Muembe
  touches no border, N'gauma is on the Malawi land border, and Lago, the lake district, is 50.7%
  Muslim. Mavago, Muembe and N'gauma are the only three over 95% after the fit (Mecula 94.8%,
  Distrito de Lichinga 94.0%, Sanga 91.8%). It now reads "Mavago on the Tanzanian border, Muembe and
  N'gauma are each over 95%", with COD-AB's spelling Muembe in place of the 2007 volume's Meumbe. It
  reaches the page on the next build tail, since `tiles.py` writes the note.
- **§14 beyond Cabo Delgado: ask 037.** Insurgents have attacked Mecula in Niassa (from late 2021)
  and Eráti and Memba in Nampula (September 2022, including the Chipene Catholic mission), and all
  three are drawn at district: Mecula 20,888 people, 94.8% Muslim and 3.0% Catholic; Memba 328,460,
  65.5% and 27.9%; Eráti 387,713, 42.3% and 50.8%. §7.5 did not mention these. Left as built; the ask
  recommends keeping it, because Memba and Eráti are the size of Burkina Faso's provinces under ask
  018 and Mecula's Catholics are under one dot.
- **Screenshot at 1:1,000.** Dots in all eleven provinces, none in the sea or the lake, green in the
  north and Zione yellow in the south. Nothing looks off.
