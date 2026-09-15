# Mozambique — INE, IV RGPH 2017, religion by province

Built 2026-09-14. `sources/mz.py`, `sources/mz_geo.py`, `sources/mz_grid.py`,
`taxonomy/mz2017.py`, `countries.py` entry `mz`.

**26,899,105 people, 11 provinces, 8 columns (7 drawn), 97.49% drawn.** The value of the
country is one column: `Zione/Sião`, the Zionist churches, 4,199,083 people, 15.6%.

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
- The 2007 census (III RGPH) had provincial and district publications too; not checked.

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
