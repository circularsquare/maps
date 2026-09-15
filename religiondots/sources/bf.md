# Burkina Faso — RGPH 2006, religion by province

**Drawn 2026-09-14** (session `d743fc47-bf`). 45 provinces, 6 categories, 14,017,262 residents,
every row `measured`, counts exact. New node `other.bf`. No `gap`.

- `sources/bf.py` -> `data/normalized/bf.csv` (the PDF is `data/raw/bf/Theme2-Etat_et_structure_de_la_population.pdf`)
- `sources/bf_geo.py` -> `data/geo/bf/bf_provinces.gpkg`, `bf_hexes.gpkg`, `bf_lookup.csv`
  (geoBoundaries ADM2 + Kontur 400 m)
- `taxonomy/bf2006.py` -> the mapping; `countries/bf.py` -> the entry; `taxonomy/branches.py`
  `other.bf` is the one new node
- sources.md **§bf-2026-09-14** is the short write-up; **§11aq** has the scouting.
- Drawn at the published grain by Anita's ruling on ask 018 (`ask/answered/018-bf-...`).

```
python sources/bf.py     --fetch
python sources/bf_geo.py --fetch
```

## 1. What INSD publishes

| release | religion | tier |
|---|---|---|
| **RGPH 2006, *Thème 2: État et structure de la population*** (181 pp) | **Tableau A5.6, province x religion, counts** (PDF p164-165); A5.5 région x sex; A5.4 national x milieu x sex; A5.7 age; Tableau 5.3 région shares (p95) | **45 provinces** |
| RGPH 2006, the 52 `principaux_tableaux` PDFs (`insd.bf/contenu/enquetes_recensements/rgph-bf/principaux_tableaux/`, Wayback 2021-08-24) | **not opened**; untitled; could carry a commune cut | ? |
| UNSD Demographic Yearbook table 28 | Burkina Faso 2006, six categories, total, urban, rural | national |
| **5e RGPH 2019, *Volume des tableaux statistiques*** (INSD, June 2024, 314 pp, `insd.bf/sites/default/files/2024-06/`, live) | **Tableau I.22, région x religion, shares, one decimal** (PDF p42); I.21 national by milieu and sex | 13 régions |
| 5e RGPH 2019, the thirteen regional monographs | Graphique 3.2, religion by province, one decimal, chart labels (§11aq checked two) | 45 provinces |

**The 2006 file is not on insd.bf's current site.** Three Wayback captures on three retired paths
(`documents/publications/insd/publications/resultats_enquetes/RGPH2006/`, 2019-02-03;
`contenu/enquetes_recensements/rgph-bf/themes_en_demographie/`, 2021-08-24; `fr/IMG/pdf/`,
2010-11-13) all have SHA-1 `TEQWCJXT64ODTIJH47FGJFD4KECGRVTU`, 2,348,972 bytes, 181 pages, a text
layer on every page. `fetch()` tries them in that order against the digest.

## 2. The table and its checks (all in `sources/bf.py`)

Tableau A5.6 is drawn as printed: no share is multiplied by anything.

| check | result |
|---|---|
| file | digest pinned; 181 pages; text on every page |
| A5.6 parsed off the page = the transcription | all 46 rows (45 + Total), digit for digit |
| rows and columns close | every province over 6 answers; every column over 45 provinces to the Total row, 14,017,262 |
| **A5.6 summed by région = A5.5** | all 7 columns, all 13 régions, which also proves the province -> région assignment |
| **A 3.1 bis (population by sex, area, density)** | same 45 province totals; M + F = T on 46 rows; every density = total / area to rounding |
| A5.4 national | the same six counts |
| **UNSD table 28** | all six columns equal to the person |
| Tableau 1.3 | 0,00% religion non-déclarés in all nine milieu x sex cells |

The transcription was generated off the page by solving each row's cell boundaries against its
own printed total (the numbers use a space as the thousands separator, so `2 622 730 55` is
ambiguous until the row sum settles it). Every row had exactly one solution.

The 45 printed areas sum to 272,968 km2 against the printed national 272,967; not asserted.

## 3. The form, the universe, and why there is no gap

Column **P15** of the household form (IPUMS International's copy, `enum_form_bf2006a.pdf` p11;
the enumerator manual `enum_instruct_bf2006a.pdf` p24; both read 2026-09-14): *"Quelle est la
religion de (Nom) ? Encerclez le code correspondant"*, **1 Animiste, 2 Musulman, 3 Catholique,
4 Protestant, 5 Autre, 6 Sans religion.** So:

- the table's columns are the form's codes in the form's order;
- animism has its own box, so `Sans religion` is the `separate` case and goes to `unaffiliated`
  (`tools/check_no_religion.py` passes it on the `Animiste` sibling);
- **there is no code for no answer**, and Tableau 1.3 prints 0.00% for religion (0.53% for age,
  1.53% for language). The universe is every resident, 14,017,262, the headline figure. Nothing
  is left out, so no `gap` and no `gap_share` (`tools/gap_share.py`: nothing excluded).
- Children under six were given their mother's religion, or that of the person caring for them
  if she was not in the household (p37 and the manual).

## 4. `Autre` probably holds some blank answers

A form without a non-response code and a report with 0.00% non-response have put the blanks
somewhere. Tableau A5.7's `ND` row (age not recorded, 74,487 people) is **18.7% `Autre`**
(13,919), against 0.57% of the population, and 0.2% to 0.6% in every age group. Records missing
one item usually miss others. The 2019 census puts `Autre` at 0.2% (Tableau I.21). How much of
2006's 79,485 is blank, nothing printed says.

**Call: `Autre` -> `other.bf`**, not `unknown`. The code is a religion answer, the blank share
is unmeasured, and `unknown` would relabel whatever real other religions it holds; it is 79
dots. Reversing it is a one-line change in `taxonomy/bf2006.py`. The `other.bf` description on
the legend says the same.

## 5. What the table shows

National: Muslim 60.53%, Catholic 19.01%, animist 15.34%, Protestant 4.17%, other 0.57%, no
religion 0.38%.

- **Muslim**: Oudalan 98.0%, Loroum 97.0%, Séno 96.8%, Soum 96.3%; the Sahel and Nord.
- **Animist**: Poni 74.8%, Noumbiel 72.5%, Bougouriba 61.8%, Tapoa 57.2%; Tapoa holds 9.1% of
  the national count. The Sud-Ouest région is 64.9%.
- **Catholic**: Sanguié 44.4%, Ioba 40.3%, Boulkiemdé 37.6%, Kadiogo 36.2%; Kadiogo holds 23.5%.
- **Protestant**: Gnagna 18.7%, Nahouri 12.0%, Tapoa 9.8%; Kadiogo holds 17.8%.
- **No religion**: Kompienga 3.5%, Sanguié 2.0%, Tapoa 1.9%.

## 6. The 2019 census, and why 2006 is drawn

Tableau I.22 of the 2019 *Volume des tableaux statistiques* (read 2026-09-14, PDF p42), 18,171,751
residents: animist **9.0%**, Muslim 63.8%, Catholic 20.1%, Protestant 6.2%, other 0.2%, no
religion 0.7%. By région, Sud-Ouest animist 64.9% -> 48.1%, Est 30.7% -> 20.3%, Centre-Nord
23.6% -> 13.8%; Protestants in the Est 11.2% -> 21.0%. That is a large move in thirteen years,
and the note says so.

2006 is drawn because it is exact counts at 45 provinces with every resident enumerated. The
2019 province figures exist only as one-decimal chart labels in thirteen monographs, and nine
communes were not enumerated and 28 only partly (Soum, Sanmatenga, Bam, Yagha; Volume 2 p376),
which are the insurgency provinces. **An upgrade path, not taken:** read Graphique 3.2 in all
thirteen monographs, weight by province populations, rake to Tableau I.22's régions, and say
what the unenumerated communes do. A later session's call.

## 7. Boundaries and placement

**geoBoundaries `gbOpen/BFA/ADM2`** (commit 9469f09, source World Bank, `boundaryYearRepresented`
2017, CC BY 4.0): 45 provinces. **COD-AB is unusable here**: `cod-ab-bfa` v03 (valid 2025-08-01)
is the July 2025 reform's 17 régions and 47 provinces, and HDX keeps no older version.

- **Name join** 45/45, one alias: census `Komandjoari` = geoBoundaries `Komonjdjari` (A 3.1 bis
  spells it `Komandjari`). A5.6 also prints `Bale` in its second block; written `Balé`.
- **Witness 1, area** against A 3.1 bis: polygon / census area 0.994 to 1.031, worst Léraba
  (3,112 against 3,019 km2).
- **Witness 2, région**: every province's representative point lies in the geoBoundaries ADM1
  région that the census's own A5.5 sums put it in. 45 province pairs are within 4% of each
  other's area; 41 of them are in different régions, so only 4 pairs rest on the name alone.

**Kontur 2023-11**: 136,973 hexes. 685 centroids (70,206 people, 0.30%) fell outside every
province along the border; 472 snapped within 2 km, 213 dropped (32,430 people, 0.14%).
Kontur / census is **1.66x** overall, against 1.30x from the 2006 to the 2019 census, and runs
from **0.93x (Sourou), 0.96x (Nayala, Séno), 0.98x (Soum, Kossi)** to **2.15x (Tapoa), 2.29x
(Ziro), 2.35x (Komandjoari), 2.93x (Comoé), 3.23x (Kompienga)**. The low end is the north and
northwest, which fits displacement since 2019, and the high end the south and east. Kontur is a
within-province weight, so no count moves; it sets the shape inside each province to 2023's.

`python kontur_cap.py bf`: 136,760 cells, no stops. Median province area is about 5,600 km2, so
thousands of cells a unit; the grid floor does not bind.

## 8. §14

Insurgents in the Sahel attack Christians and communities by identity (§11aq, and the USCIRF
Sahel update cited in ask 017). The scout raised it; **Anita ruled on ask 018** to draw at the
published 45 provinces everywhere, because the units are big. Nothing here is finer than INSD's
own table, and the counts predate the insurgency by a decade.

## 9. Gotchas

- **Space-separated thousands** make a row of numbers ambiguous as text: `2 622 730 55` is 2,622,
  730, 55 or 2,622,730, 55. Solve against the row total (every row here has one solution) and
  compare digit strings, not tokens.
- A5.6 prints its 45 provinces in two blocks, the 30 of 1985 then the 15 cut in 1996, and spells
  `Bale` in the second.
- A5.5's `Masculin` block continues onto p164 above A5.6; bound A5.6 by its caption.
- The printed national area is one km2 off the sum of the provinces.

## 10. Terms

INSD's reports are public PDFs, reached through the Wayback Machine. The IPUMS enumeration form
and manual are open documentation (no microdata was requested). UNSD Demographic Yearbook data is
public. geoBoundaries and Kontur Population are CC BY 4.0.

## 11. Review, 2026-09-14 (`d743fc47-rev1`)

Read from `data/normalized/bf.csv`, `taxonomy/bf2006.py` and `countries/bf.py` before this file.
Nothing to rebuild, no ask. One soft wording point, recorded and not edited.

- **`note_public` recomputes off the CSV**, every figure: Oudalan 98.0%; Loroum, Séno and Soum
  over 96%; Poni 74.8%, Noumbiel 72.5%, Tapoa 57.2% animist; Sanguié 44.4% and Kadiogo 36.2%
  Catholic; Gnagna 18.7% Protestant; 15.34% animist nationally. The `other.bf` legend note's 21%
  is Kadiogo's 21.2%.
- **Soft: the Catholic sentence leaves out the second-highest province.** "Catholics are most
  concentrated in the centre-west and around the capital" fits Sanguié, Boulkiemdé and Kadiogo,
  but Ioba, 40.3% and second of 45, is in the Sud-Ouest, the région the paragraph before calls
  animist (Ioba is 50.5% animist too). Kouritenga (Centre-Est, 35.1%) and Bazèga (Centre-Sud,
  34.5%) come next. Something like "in the centre, and in Ioba in the southwest" would be exact.
  Left for whoever next edits the note.
- **The mapping follows precedent**, the same French boxes as Chad (`sources/td.md` §11), and
  `other.bf` is a spec §3.11 per-source residual like `other.gn` and `other.cg`. `Autre` on
  `other.bf` rather than `unknown` is a fair call at 79 dots, and the A5.7 evidence that some of
  it is blank answers is on the legend note itself.
- **No `gap`, correctly**, since every resident is in A5.6. The distortion is the vintage, and it
  is biased rather than merely old: animism fell to 9.0% by 2019, so the drawn animist share is
  too high everywhere and by most where animism is largest (Sud-Ouest 64.9% against 48.1%).
  `note_public` states exactly that with the figures, and the 2019 alternative (§6) is missing
  communes in the insurgency provinces, so it has its own bias. The builder's call, disclosed.
- §14 is settled by ask 018.
- **Screenshot at 1:1,000:** dots on land only; the north green, the southwest and Tapoa purple,
  Ouagadougou and Bobo-Dioulasso dense. Nothing looks off.
