# Bosnia and Herzegovina: BHAS, Popis 2013, mother tongue

`sources/ba_census.py` -> `data/normalized/ba.csv`. Mapping `taxonomy/ba2013.py`, new nodes
`taxonomy/tree.d/ba.txt`, entry `countries/ba.py`. Placement: religiondots' hexes, read only.

**3,523,672 of 3,531,159 people drawn (99.8%) on 142 municipalities, 16 nodes.** Two small
workbooks, no wall, an exact partition, a second table agreeing in every unit, and a join to
religiondots' units that matches both ways with equal totals.

## 1. The table, and why not the one the sweep found

The coverage sweep pointed at `Popis2013prvoIzdanje.pdf`, the preliminary first edition, and
religiondots' record (`../religiondots/sources/ba.md` §9) pointed at table 5.3 of the final
results PDF. Both fold mother tongue to five answers: Bosnian, Croatian, Serbian, Other, Unknown.

BHAS's book index, https://popis.gov.ba/popis2013/knjige.php, lists a finer one. The second
results book, *Etnička/nacionalna pripadnost, vjeroispovijest i maternji jezik* (Knjiga 2), has
**table 6.1, population by mother tongue and sex, by municipalities/cities**, as xlsx:
https://popis.gov.ba/popis2013/doc/Knjiga2/BOS/K2_T6-1_B.xlsx. Sixteen named answers, Other and
Unknown, at country, entity, Federation canton and municipality, each as total, male, female.
Only the total rows are read.

| answer (Bosnian / English) | national | share |
|---|---|---|
| Bosanski / Bosnian | 1,866,585 | 52.86% |
| Srpski / Serbian | 1,086,027 | 30.76% |
| Hrvatski / Croatian | 515,481 | 14.60% |
| Srpsko-Hrvatski / Serbo-Croatian | 27,299 | 0.77% |
| Ostali / Other | 10,649 | 0.30% |
| Nepoznato / Unknown | 7,487 | 0.21% |
| Romski / Romani | 5,766 | 0.16% |
| Albanski / Albanian | 2,420 | 0.07% |
| Bosansko-Hrvatsko-Srpski | 1,897 | 0.05% |
| Turski / Turkish | 1,233 | 0.03% |
| Hrvatsko-Srpski / Croato-Serbian | 1,195 | 0.03% |
| Bošnjački / Bosniak | 1,167 | 0.03% |
| Ukrajinski / Ukrainian | 1,081 | 0.03% |
| Bosansko-Srpsko-Hrvatski | 890 | 0.03% |
| Bosansko-Hrvatski | 714 | 0.02% |
| Bosanskohercegovački | 636 | 0.02% |
| Njemački / German | 632 | 0.02% |

The same book's national table 6 and table 4 (mother tongue by ethnicity) split nothing further;
Other is never broken down at any level.

**Brčko District** is a level-1 (entity) row in 6.1 with no municipality row beneath it. It is one
unit, so the script lifts it into the municipality cover: 141 + 1 = 142, as in religiondots.

## 2. Checks, all exact (asserted in `check()`)

* National total 3,531,159, BHAS's published figure.
* The 17 categories partition every one of the 156 total rows.
* The three entity rows, and the 142 municipalities, each sum to the national row in all 18
  columns; each of the 10 Federation cantons equals the sum of its municipalities.
* **A second table of the same census agrees per unit.** The first results book's table 5
  (https://popis.gov.ba/popis2013/doc/RezultatiPopisa/BOS/FR_T5_B.xlsx, the five-answer version)
  has the same Bosnian, Croatian, Serbian and Unknown in all 142 units, and its Other equals
  table 6.1's thirteen smaller named answers plus its Other, in every unit.
* **The join.** The census publishes no codes; `geo_id` is religiondots' fold of the name
  (`religiondots/sources/ba.py` `fold`, copied), which its hex layer carries as `unit`. The PDF
  religiondots read spells `BANjA LUKA` and `FOČA - F BiH`, the workbook `BANJA LUKA` and
  `FOČA - FBiH`; the fold drops case and spaces so both land on one key. All 142 keys match
  religiondots' `ba.csv` both ways, and every unit's total equals its religion-table total.

## 3. Calls

* **Bosnian, Serbian, Croatian as three leaves**, as rs2022 and hr2021. Glottolog files the three
  standards (bosn1245, serb1264, croa1245) as dialects of Serbian-Croatian-Bosnian (sout1528). The
  answer follows nationality: Knjiga 2 table 4 (https://popis.gov.ba/popis2013/doc/Knjiga2/BOS/K2_T4_B.xlsx)
  gives 1,762,577 of 1,769,592 Bosniaks (99.6%) Bosnian, 1,066,593 of 1,086,733 Serbs (98.1%)
  Serbian, 511,007 of 544,780 Croats (93.8%) Croatian. Croats are the least uniform: 17,088 gave
  Bosnian and 7,504 Serbo-Croatian. `note_public` says this.
* **Every compound and alternative name its own leaf.** Serbo-Croatian and Croato-Serbian on
  the existing leaves (cz/fi/us, hr). Five new: Bosnian-Croatian-Serbian, Bosnian-Serbian-Croatian,
  Bosnian-Croatian, Bosniak (Bošnjački), Bosnian-Herzegovinian. The two orders of the three-part
  name are not merged, for the reason hr.txt keeps Croato-Serbian apart: BHAS counted them apart
  and the order is what people wrote. Together they are 5,304 people; the compound answers peak in Novo Sarajevo, Centar Sarajevo and
  Tuzla, Bosniak nowhere above 0.3%.
* **Bosniak is not merged into Bosnian.** Bošnjački is the name Serb and Croat usage gives the
  language, as against Bosniaks' own "Bosanski"; the census prints it apart.
* **Other (10,649) on `other`.** BHAS never names Slovenian, Macedonian or Montenegrin as a
  language, though table 4 has them as ethnicities, so those answers are inside Other with
  migrants' languages; no indigenous remainder can be told apart. Highest in central Sarajevo
  (1.3%) and Prnjavor (1.1%, next to its Ukrainians; plausibly Polish or Czech, Prnjavor's other
  Austro-Hungarian settlers, but nothing published says so).
* **Unknown (7,487, 0.21%) is the gap.** One outlier: Fojnica 772 (6.2%), the same municipality
  that is 33x the national rate for "no answer" in the religion table; an enumeration artefact.
* **Vintage 2013**, the last census (the one before was 1991). The language geography this
  draws is the post-1995 one; religiondots' §5 argues the same for religion and it holds here.
* **The RS dispute** (RZS rejects BHAS's residence rule and publishes lower entity figures) is
  about who was counted, not the language question. BHAS's figures drawn; said in `note_public`.

## 4. Geography and placement

Religiondots' `data/geo/ba/ba_grid_400m.gpkg` (read only): 40,733 Kontur 400 m hexes over the
142 units, with `pop`. Its record (`../religiondots/sources/ba_geo.md`) covers the four repairs
geoBoundaries ADM3 needed (a polygon named Republika Srpska that is Višegrad, two Novi Grads,
Kupres's halves labelled backwards, Kupra na Uni). Placement is plain population weight inside
each municipality; no language-specific proxy exists at a finer grain.

Scatter: 3,516 dots at 1:1000, 4 rings, 7,672 people (0.22%) below one dot per language.

## 5. What it shows

* 80 of 142 municipalities are over 90% one of Bosnian, Serbian, Croatian (Bosnian 39, Serbian
  31, Croatian 10). Brčko is the three-way unit: 43.5% Bosnian, 34.7% Serbian, 20.1% Croatian.
* Serbo-Croatian is an urban, mixed-town answer: Vareš 2.9%, Novo Sarajevo 2.6%, Centar
  Sarajevo 2.5%, Banja Luka 2.3% (4,277 people, the largest count).
* Ukrainian is Prnjavor (446, 1.2%) and Laktaši; Turkish is Ilidža (658, 1.0%) and Hadžići;
  Romani peaks in Vukosavlje (2.0%) and Kakanj; Albanian is Sarajevo and Mostar.

## 6. Left over

* Colour: generated Bosnian sits 0.122 (OKLab) from Croatian and 0.133 from Serbian; tree.d/ba.txt
  records the hand-pick to try (a green, 0.68 0.16 135) if Mostar or the central Bosnian valley
  reads as one colour.
* Nothing finer exists: no newer census, and Other is unsplit everywhere BHAS publishes.
