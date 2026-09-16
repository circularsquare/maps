# Guinea — RGPH 2014, religion by région administrative

**Drawn 2026-09-14.** 8 régions, 5 categories, 10,503,134 people in ordinary households,
every row `measured`.

- `sources/gn.py` -> `data/normalized/gn.csv` (the PDF is `data/raw/gn/RGPH3_etat_structure.pdf`)
- `sources/gn_geo.py` -> `data/geo/gn/gn_regions.gpkg`, `gn_hexes.gpkg`, `gn_lookup.csv`
  (COD-AB ADM1 + Kontur 400 m)
- `taxonomy/gn2014.py` -> the mapping; `countries.py` `"gn"` -> the wiring;
  `taxonomy/branches.py` `other.gn` is the one new node
- sources.md **§9dh** is the write-up; it corrects **§11w**'s Guinea row.

```
python sources/gn.py     --fetch
python sources/gn_geo.py --fetch
```

## 1. What INS publishes

| release | religion | tier |
|---|---|---|
| **RGPH 2014, *État et structure de la population*** (INS, 122 pp) | **Tableau 5.10, région x religion, % to one decimal**; Tableau 5.09, urban/rural x sex; Tableau 5.11, 1996 and 2014 by residence | **8 régions** |
| the other 17 RGPH 2014 thematic reports | religion only as a breakdown of other topics (fertility, marriage, children, the elderly), never crossed with prefecture | national / urban-rural |
| RGPH 1996, *État de la population* (scanned) | Tableau 5-3, religion by sex and residence; the table list has nothing finer | national |
| INS *Annuaire statistique* 2014, 2021, 2024 | the national line (1983, 1996, 2014) | national |
| UNSD Demographic Yearbook table 28 | Guinea 2014, six categories in counts, total/urban/rural | national |
| RGPH-4 preliminary results (INS, 2026) | none seen; a new census is under way | not checked further |

**§11w recorded "full RGPH-3 thematic series, no religion volume".** There is no volume about
religion; the structure volume has the région table in chapter 5. A volume's title is not its
table list.

All eighteen RGPH-3 PDFs the Wayback CDX lists for `stat-guinee.org` were downloaded and every
page naming a religion was checked for prefecture names; none prints religion below the
région. `microdata.insguinee.org` (NADA) holds DHS 2018, a malaria survey and a livestock
survey, no census. `rgph.insguinee.org` is an indicator database with no religion theme. So
eight régions is the ceiling in both censuses that are online.
Re-checked 2026-09-15 (sources.md §scout-2026-09-15-africa-upgrades): the 2015 and 2019 *Annuaires
statistiques* print religion nationally and by région only (2019 Tableau 15.3 repeats 5.10), plus church
buildings by préfecture; the census microdata exists only in IPUMS (1983, 1996, 2014).

## 2. The construction

    shares        Tableau 5.10, région x religion, one decimal       PDF p88
    denominators  Tableau 2.07, resident population by région        PDF p34, repeated in the annex p104
    magnitudes    UNSD table 28, Guinea 2014, national counts         tools/oracle.py

Share x région population, then each religion rescaled so the eight sum to UNSD's count. The
rescale factors are 0.994 (no religion), 0.998 (Muslim), 1.002 (Christian), 1.008 (animist)
and 0.989 (other), which is one-decimal rounding and nothing else.

**The religion universe is ordinary households.** UNSD's `Unknown` is 20,129 nationally,
5,750 urban and 14,379 rural; Tableaux 2.04 and 2.05 give collective households (barracks,
hotels, boarding schools, orphanages) as exactly those three numbers. The report's
percentages are shares of the 10,503,132 in ordinary households: `gn.py` reproduces all
fifteen national, urban and rural cells of Tableau 5.09 from UNSD's counts over those
denominators. Collective households are not published by région, so the shares go onto
resident population and the rescale spreads the 0.19% across régions pro rata; every région
comes out at 0.997 to 0.999 of its resident count.

## 3. The checks (all in `sources/gn.py`)

| check | result |
|---|---|
| Tableau 5.10 parsed off the page = the transcription | 8 régions + Ensemble, identical |
| every row sums to 100 | within 0.10 pp |
| région populations | appear on p34 and again in the annex; men + women = total; sum 10,523,261 |
| population-weighted région shares vs the printed Ensemble row | 2.42/2.4, 89.12/89.1, 6.75/6.8, 1.58/1.6, 0.12/0.1 |
| UNSD `Unknown` = collective households | national, urban and rural, to the person |
| Tableau 5.09 from UNSD's counts | all 15 cells reproduced to the printed decimal |
| COD-AB name join | 8/8, three differ only by accents or the apostrophe in N'Zérékoré |
| Kontur 2023 vs census 2014 | 1.35x overall, 1.22x (Mamou) to 1.54x (Kindia) |

**The questionnaire.** The 2014 household form (IPUMS enumeration materials,
`enum_form_gn2014a.pdf`, p3) asks P11 with codes 0 sans religion, 1 musulmane, 2 chrétienne,
3 animiste, 4 autre religion. That is Tableau 5.10's column order and count exactly. No
denomination was ever asked.

## 4. What the table shows

Seven régions are 89% Muslim or more (Labé and Mamou 99.4%). **N'Zérékoré is 46.7% Muslim,
28.1% Christian, 10.4% animist and 14.2% no religion**, and holds 15.0% of the people, 62.5%
of the Christians, 98.8% of the animists, 88.1% of those with no religion and 72.8% of
`Autres religions`.

`Sans religion` at 14.2% in the one région where traditional religion is also strong is
probably partly traditional practice. The form has an `Animiste` box, so this is less clear
than Côte d'Ivoire's case, and the cell is drawn as printed.

## 5. §14 was considered and no ask was filed

Forest Guinea has had deadly clashes along ethnic lines that are also religious ones
(around N'Zérékoré in 2013, and again around the 2020 referendum). Not escalated: the tier is eight régions of 1.3 million people
on average, coarser than Egypt's governorates (`ask/answered/001-eg`); the table is the
statistics office's own publication, on its live website since at least 2020; and nothing
here locates a group more finely than INS already has. A prefecture or sous-préfecture table
would be a different question, and none exists.

## 6. Gotchas

- **Tableau 5.08 on the page before is labelled "population résidente de plus de 15 ans" and
  its Effectif row sums to 9,439,468**, which is neither the 15+ population nor the resident
  total. It is the language table and is not used; do not take région populations from it.
- **N'Zérékoré's men are printed `76 2 301`** in Tableau 2.07, with a stray space. The build
  checks the région totals as text and transcribes the men, rather than parsing that column.
- **The report spells the région `N'Zérékoré` in tables and `Nzérékoré` in its own chart.**
  COD-AB has `Nzerekore`. The join folds out everything but letters.
- The 1996 volumes on `stat-guinee.org` are scanned; a text search of them finds nothing and
  proves nothing. Read their table lists as images.

## 7. Terms

INS Guinée's reports are public PDFs on its own site with no licence text. UNSD Demographic
Yearbook data is public. The IPUMS questionnaire PDF is a public enumeration-materials page
and was read, not redistributed.

## Placement: Conakry's block at Kontur's density cap, capped 2026-09-14 (session `f95259a4-kontur`)

Kontur limits every hex to 46,200 people/km², and a block of hexes at that limit is either a real
dense core or a false concentration (spec §12, "KONTUR'S DENSITY CAP"). **Conakry had a block of
53 hexes, 15 of them at the limit, holding 936,877 people and 44.3% of the region's placement
weight**, at the northeast edge, 24 km from Kaloum. Kontur is thin in the older districts: within
1 km of Kaloum it peaks at 6,732/km², within 1 km of Ratoma at 5,489/km². One more hex at the
limit, 4 km north of the block (28,845 people, 1.4%), is the same thing.

Both are `capped` in `kontur_cap.csv`, and `scatter.py` now lowers each hex to the median density
of the populated hexes within 3 km: 6,970/km² for the block, which leaves it 230,529 people and
16.3% of the weight, and 10,269/km² for the single hex, which leaves it 6,411. The ring's median
is high here because the block's own ramp of suburbs is in the ring, so the northeast edge is
still drawn about as dense as the old centre.

Counts did not move: dots per node are identical before and after, 10,500 at 1:1,000 and 1,048
at 1:10,000.
