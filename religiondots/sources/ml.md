# Mali: RGPH5 2022, religion by région

**Drawn 2026-09-14** (session `d743fc47-ml`). 20 régions, 7 categories, 21,347,586 residents of
ordinary households, every row `measured` (printed shares on printed populations, raked to
printed national counts). New node `other.ml`. `gap` 4.68%.

- `sources/ml.py` -> `data/normalized/ml.csv` (the three PDFs are in `data/raw/ml/`)
- `sources/ml_geo.py` -> `data/geo/ml/ml_regions.gpkg`, `ml_hexes.gpkg`, `ml_lookup.csv`
  (COD-AB v03 + Kontur 400 m)
- `taxonomy/ml2022.py` -> the mapping; `countries/ml.py` -> the entry; `taxonomy/branches.py`
  `other.ml` is the one new node
- sources.md **§ml-2026-09-14** is the short write-up; **§11aq** has the scouting.
- Drawn at the published grain by Anita's ruling on ask 018 (`ask/answered/018-bf-...`).

```
python sources/ml.py     --fetch
python sources/ml_geo.py --fetch
```

## 1. What INSTAT publishes

Both reports are live on `instat-mali.org/laravel-filemanager/files/shares/rgph/` and on the NADA,
`microdata.instat.ml/index.php/catalog/95` (downloads 708 and 722). `instat.gov.ml` does not resolve.

| release | religion | tier |
|---|---|---|
| **RGPH5, *Rapport d'analyse, Thème: État et structure de la population*** (168 pp) | **Tableau 6.13, région x 7 answers, one decimal, with populations** (PDF p132); 6.12 national by milieu and sex; 6.14 2009 against 2022, national | 20 régions |
| **RGPH5, *Caractéristiques culturelles de la population*** (70 pp) | **Tableau 2.03, région x 5 groups, two decimals, Christians undivided** (PDF p31); 2.01 national counts by sex; 2.02 milieu; 2.04 sedentary and nomad; **annex A01, national counts for all 7 codes and `Non Déclaré`** (PDF p52) | 20 régions |
| UNSD Demographic Yearbook table 28 | Mali absent | |
| RGPH 2009 | national by sex only (§11aq) | national |

Nothing below région was found: the cultural volume's annexes A01-A07 are national, and §11aq found no
cercle table. **Not opened:** the *Résultats globaux* report (NADA download 586) and the census
microdata's access terms on catalog 95; a cercle table would be finer than ask 018 ruled on anyway.
Both checked 2026-09-15 (sources.md §scout-2026-09-15-africa-upgrades): the *Résultats globaux* has no
religion table, and the microdata needs a login; the catalogue lists no cercle or commune volume.

## 2. The tables and their checks (all in `sources/ml.py`)

| check | result |
|---|---|
| files | three digests pinned; page counts; text on every page but the blank covers, which are pinned too |
| parsed off the page = the transcription | 6.13, 2.03, 2.9, 2.3, 1.2, 1.1, 2.01, A01, 6.12's Ensemble column |
| one population per région | 6.13 = 2.03 = 2.9 on all 20; the régions sum to 21,347,586, one under the printed 21,347,587 (2.3: 22,395,488 against 22,395,489) |
| rows close | 6.13: every row exactly 100.0; 2.03: within 0.000 |
| 6.13 against 2.03, cell by cell | Muslim within 0.06, animist within 0.05, the three Christian cells within 0.08 of 2.03's one; 10 `Sans religion`, `Autre religion` or Muslim cells beyond rounding, all within 0.15 |
| 2.03 population-weighted | reproduces its national row within 0.006 |
| **A01 -> 2.01** | 2.01's counts are A01's with the 48,746 `Non Déclaré` spread in proportion, every cell within 0.8 people, in all three sex columns |
| 2.1 | ordinary 21,347,587 + collective 105,416 + homeless 1,151 + not enumerated 941,335 = 22,395,489 |
| the form | P10 prints codes 1-6 in the tables' order; Tableau 1.01 adds 7 |

**Tableau 6.13's rows are forced to 100.** All 21 sum to exactly 100.0, and the cells that disagree
with 2.03 beyond rounding are the ones where a row would be closed: Koulikoro `Autre religion` 0.5
against 0.36, Sikasso 0.3 against 0.23, Ségou 0.1 against 0.02; `Sans religion` in Mopti, Nioro,
Dioïla, Bougouni, Koutiala and Douentza (0.1 against 0.00); Kidal's Muslim 99.7 against 99.64. So
6.13 is used only for how each région's Christians divide.

**The construction.** Stage 1: 2.03's shares times 2.9's populations, raked to 2.9's région totals and
to A01's counts with non-response prorated (which are 2.01's). Factors: Muslim x0.99994, Christian
x1.00087, animist x1.00154, no religion x1.00343, other x1.01197. Stage 2: each région's Christians
split in 6.13's Catholic : Protestant : other proportions and raked to A01's three: Catholic x0.9854,
Protestant x1.0043, other Christian x1.2618. 6.13's one-decimal Catholic cells weighted by population
come to 295,926 against A01's 291,618 (Bamako's 1.5 carries 4.2 million people), hence the 1.5%. The
drawn shares are within 0.107 pp of 6.13 (worst San Catholic, 9.49 against 9.6) and the Christian
totals within 0.008 pp of 2.03.

**The national margins are rounded to the régions' sum, 21,347,586**, not the printed 21,347,587, or
the rake's two margins disagree by one person and it never converges.

## 3. The form, the universe, and the gap

**P10** of the ordinary-household questionnaire (NADA catalog 95, download 585,
`IMP_RGPH5_Questionnaire_Menag_Ordinaire_VF_SMAP_20MARS2021.pdf`, p2, read 2026-09-14): *"Quelle est
la religion de [NOM] ?"* 1 Musulman, 2 Catholique, 3 Protestant, 4 Autre religion chrétienne, 5
Animiste, 6 Sans religion. That is the printed (paper) form. Tableau 1.01 of the cultural volume
lists **7 Autre religion**; the tablets were used in the secure south and paper in the north, Mopti,
the Macina and Niono cercles and Banamba and Nara (structure volume PDF p44). The tablet form was
not found on the NADA, so how a paper answer outside codes 1-6 was coded is not known.

- Animism has its own code beside `Sans religion`, so the no-religion box is the `separate` case
  (`tools/check_no_religion.py` passes it).
- **No code for no answer, and 48,746 `Non Déclaré` in A01 anyway (0.23%).** Every share table in
  both volumes spreads them over the seven answers in proportion; the dots do the same. Spec §12:
  record a proration, never undo it. It is not in `gap`: it is inside the drawn counts.
- **Omission adjustment.** Counts were raised by the post-enumeration survey's coefficients: Kayes
  1.051, Koulikoro 1.052, Sikasso 1.068, Ségou 1.029, Bamako 1.034, and 1.068 everywhere on paper
  forms (Tableau 1.4, PDF p47). Recorded, not undone. It is why the regional counts do not sum
  exactly.
- **The universe** is residents of ordinary households, 21,347,587 of the population de droit
  22,395,489 (Tableau 2.1). **`gap` is 4.68%**: 941,335 (4.20%) in areas not enumerated for
  insecurity, estimated with GRID3 from building footprints and roads (PDF p44-45), and 106,567
  (0.48%) in collective households (105,416) or homeless (1,151). Hand-written:
  `tools/gap_share.py` sees no excluded column, which is right.

**Tableau 1.2 and Tableaux 2.3 minus 2.9 disagree about Douentza and Bandiagara.** 2.3 minus 2.9
leaves 22,960 people in Douentza and 148,637 in Bandiagara outside ordinary households; 1.2 puts
the unenumerated at 154,979 in Douentza and 986 in Bandiagara. Every other région's residual is
non-negative, the two régions together come to +15,632, and the national residual is exactly the
collective and homeless population. Which table has the two régions right is not resolved; it moves
no drawn count, only where `note_public` could say the unenumerated are, so the note says "the old
Mopti région".

The two volumes also split urban from rural differently: 6.12 has 6,745,306 urban residents and 2.02
6,698,015. Not used.

## 4. The mapping calls (`taxonomy/ml2022.py` REVIEW has the figures)

- `Autre religion chrétienne` -> `christianity.other`, as `bj2013.py` and `ci2021.py` map the same
  box, though the node's description is for bodies with no branch. 17 dots; nine régions print 0.1
  or 0.2 and eleven 0.0, so its région pattern is mostly rounding (raked x1.26).
- `Autre religion` -> new `other.ml`. The report does not say what it holds.
- `Animiste` -> `indigenous.african`, a floor. The 2009 census printed 2.0% (6.14) against 0.7%.
- `Protestant` -> `christianity.protestant`, one code for every Protestant, evangelical and
  Pentecostal church.

## 5. What the table shows

National: Muslim 96.45%, Catholic 1.37%, Protestant 0.82%, animist 0.65%, no religion 0.50%, other
0.13%, other Christian 0.08%.

- **San** is the exception: 70.1% Muslim, 9.5% Catholic, 7.3% Protestant, 8.3% animist, 3.7% no
  religion, 0.9% other. With 3.8% of the population it holds 26.5% of the Catholics, 33.6% of the
  Protestants and 48.9% of the animists.
- **Koutiala** 4.3% Christian, 3.1% animist; **Bandiagara** 6.0% Christian, 0.04% animist.
- **No religion**: San 3.66%, Koulikoro 1.45%, Sikasso 1.04%; Koulikoro holds 30.6%.
- 14 of 20 régions are over 98% Muslim.

## 6. Boundaries and placement

**COD-AB Mali v03** (`cod-ab-mli`, IGM Mali, boundaries created 2025-02-20, valid 2025-09-04):
20 régions and 160 cercles, the census's tier. geoBoundaries MLI ADM1 (2021) is the 9 units of before
the reform and is used only as a witness.

- **Name join** 20/20 on `adm1_name1` (the French names), no alias.
- **Witness 1, cercles**: every région holds the number of COD-AB cercles Tableau 1.1 prints (Bamako,
  printed `-`, is one).
- **Witness 2, parent**: each région's largest overlap with geoBoundaries ADM1 is the old région it
  was cut from, and at least 75% of it; lowest Ségou 89.8%, Bougouni 91.7%, Koutiala 95.7%, the rest
  from the two files' lines not agreeing. geoBoundaries spells `Koulikouro`.
- **Bamako is bigger than the old district.** COD-AB's is 733 km2, the pre-2023 district in
  geoBoundaries 246 km2 and 100% inside it. The census's Bamako is 4,227,569 people in seven
  arrondissements (Tableau 1.1), not the old six communes, and Kontur puts 4.81 million in the 733
  km2 (1.14x), so the larger district is the one counted. The report does not describe the new
  boundary.
- **Witness 2b**: Nioro and Kita (both from Kayes, 6 cercles, 678,061 and 681,671 people) and Sikasso
  and Koutiala pass witnesses 1 and 2 swapped; the pre-2023 cercle of each name falls in its
  namesake région.
- **Witness 3, population**: Kontur summed per région against Tableau 2.3, log-log correlation
  0.923 against a maximum of 0.802 over 2,000 shuffles.

**Kontur 2023-11**: 141,299 hexes, 23,373,673 people. 524 centroids fell outside every région on the
border (75,292 people); 475 snapped within 2 km, 49 dropped (1,256 people, 0.005%). Kontur / census
is **1.044x** nationally, and per région from **0.27x (Ménaka), 0.60x (Taoudenni), 0.78x
(Koulikoro)** to **1.47x (Bandiagara), 1.65x (Kidal), 2.52x (Douentza)**. The two ends are where the
census counted least directly (Tableau 1.2 models 93,653 of Ménaka's 318,876, and Douentza is the
région §3's disagreement is about) and where the nomads live, whom a building-footprint model
misses. Kontur is a within-région weight, so no count moves; in Ménaka the dots go to the few
settled places Kontur sees.

`python kontur_cap.py ml`: 141,250 cells, two blocks at the cap, both in Bamako: the right bank
south-east of the centre (77 hexes, 1,421,941 people, 29.6% of the district) and the left bank
north-east (44 hexes, 769,859, 16.0%). Both registered `real` in `kontur_cap.csv`: the unit is the
district, whose Kontur total is 1.14x the census, so the excess is city-wide, not a block piled up
off-centre. Median région area is about 29,000 km2; the grid floor does not bind.

## 7. §14

The Christian and animist minorities are concentrated in San, Bandiagara and Koutiala, and central
Mali has had identity massacres (§11aq). **Anita ruled on ask 018** to draw Mali at its 20 régions
everywhere, because the units are big. Nothing here is finer than INSTAT's own table.

## 8. Gotchas

- Tableau 6.13's rows are closed on their last cells; take cells from 2.03 where it has them.
- The non-response is visible only in the *other* volume's annex (A01); no religion chapter in
  either volume mentions it.
- A rake whose margins differ by one person oscillates forever; round one margin to the other's sum.
- The text layer puts one cell per line, except Tableau 2.1 and 6.12's `Effectifs` row, which put
  several numbers on one line.
- The form's page also prints `17 = Autre parent du CM`: a search for code 7 needs a lookbehind.
- The cultural volume lists the régions under their pre-2023 parents (Kayes, Kita, Nioro, ...), the
  structure volume in law order.
- `fetch_checks.check_pdf_doc` fails a PDF with blank cover pages; `ml.py` pins which pages are empty.

## 9. Terms

INSTAT's reports and questionnaire are public downloads (no login). COD-AB Mali is CC BY-IGO
(HDX `cod-ab-mli`); geoBoundaries and Kontur Population are CC BY 4.0.

## 10. Review, 2026-09-15 (session `d743fc47-rev3`)

A second reader, working from `countries/ml.py`, `taxonomy/ml2022.py`, `data/normalized/ml.csv`
and the other francophone census mappings. `check_md`, `built_countries --check` and
`check_rollup ml` are clean (21,347,586 people, all `measured`). Every figure in `note_public`
recomputes from `ml.csv`: San 70.06% Muslim, 9.49% Catholic, 7.25% Protestant, 8.34% animist,
3.66% no religion; San has 3.82% of the people, 26.5% of the Catholics, 33.6% of the Protestants
and 48.9% of the animists; 14 régions over 98% Muslim; Koutiala 4.32% Christian, Bandiagara 6.00%.
The mapping matches `bf2006.py`, `td2009.py`, `bj2013.py` and `ci2021.py` box for box, `other.ml`
follows the per-country residual convention, `gap` carries its figure, and §14 is settled by ask
018. Screenshot: dots inside Mali only, dense in the south, a line along the Niger bend, sparse in
the north. Nothing rebuilt or changed.

- One word to consider: `note_public` puts San "in the south". Sikasso, Bougouni and Koutiala are
  further south, and San, south-east of Ségou, is usually described as central Mali. "in the
  south-centre" or "east of Ségou" would be exact. Left for the builder.
