# Moldova — BNS, Recensământul Populaţiei şi al Locuinţelor 2024

Ingested and drawn 2026-09-08. Rebuild with `python sources/md.py --fetch` and then
`python sources/md_geo.py --fetch`.

14 religious affiliations at **UAT** level — oraş/municipiu and sat/comună, the first tier
of local government — 901 units for 2,409,207 people, about 2,700 a unit and a median of
1,251. Finer than Romania's communes next door, with a shorter category list. The polygons
come from the same office, keyed by the same code, and carry the same population, so there
is no name join anywhere in this country.

---

## 1. The files

Two, both open, both plain HTTPS with no login and no bot wall.

```
https://statistica.gov.md/files/files/ComPresa/Recensamant/2024/Ro/Anexa_Caracteristici_Etnoculturale_RPL2024.xlsx
https://gis.statistica.md/server/rest/services/Hosted/comune_p_distrib_2024_view/FeatureServer/0/query?where=1=1&outFields=*&returnGeometry=true&outSR=4326&f=geojson
```

The workbook is the ethnocultural annexe to the final 2024 results, 1,310,944 bytes, 42
sheets numbered `5.1` to `5.41`. Four carry religion:

| sheet | what | used as |
|---|---|---|
| `5.26` | religion × urban/rural, **2024 beside 2014** | not read; the 2014 comparison in §2 |
| `5.29` | religion × development region and **raion/municipiu** | `raion`, a cross-check level |
| `5.31` | religion × **oraş (municipiu) and sat (comună)** | `uat`, the drawn level |
| `5.35` | religion × ethnicity, national only | not read |

The results index is at
`https://statistica.gov.md/ro/rezultatele-finale-ale-recensamantului-populatiei-si-locuintelor-2024-10118.html`,
where the twelve annexes are named by subject, so the religion one is found by opening
`Caracteristicile etnoculturale ale populației` and reading its `Cuprins` sheet.

## 2. Why 2024 and not 2014, and what that costs

The 2014 census is the obvious alternative and it is the worse one on every axis but one.
Its file is on disk here as `data/raw/md/Caracteristici_populatie_Comune_RPL_2014_rom_rus_eng.xls`
because it was fetched first.

| | 2014 | 2024 |
|---|---|---|
| religion published at | **raion**, 35 units (sheet `2.1.`) | **UAT**, 901 units |
| enumerated population | 2,804,801 | 2,409,207 |
| coverage | BNS estimates 2,998,235 persons *should* have been enumerated, so about one in fifteen was never reached | the census's own universe, `populaţia cu reşedinţă obişnuită` |
| did not answer the religion question | 193,042, **6.88%** | 18,103, **0.75%** |
| named categories | 14 | 14 |

The commune-level sheet of the 2014 release, `1`, carries sex and age groups and **no
religion**, so 2014 offers nothing at the finer tier.

**The one thing 2014 has that 2024 does not is two category names.** 2014 counted
`Iudaism` (584 people) and `Evanghelică de Confesiune Augustană (Luterană)` (2,291); the
2024 list has neither, and both are inside `Alte religii`. Those figures are recorded here
and nowhere else in the project, because nothing draws from them: they are 2014 numbers and
`data/normalized/md.csv` is entirely 2024. The consequence for the map is that **no Jewish
dot is drawn in Moldova**, and that is a real loss in Bessarabia rather than a rounding one.

2014 also split `Creștină Evanghelic Baptistă` from `Creștină după Evanghelie`, which 2024
keeps, and had no `Fără religie` or `Liber cugetător`, which 2024 adds. So the 2024 list is
not simply the 2014 list minus two; it is shorter at the religious end and longer at the
secular one.

## 3. Reconciliation

Exact, and there is nothing to be careful about. Nothing is suppressed, nothing is rounded,
and `-` is stated in the sheet's own key as `magnitudine zero`.

`python sources/md.py` asserts, and prints, all of:

* the 901 leaves sum to the published national figure in **every one of the 16 columns**;
* each of the **35** raion rows equals the sum of its own UATs, in every column;
* sheets `5.29` and `5.31` carry identical raion figures, which is a check that the
  workbook is internally consistent rather than one sheet being a different vintage;
* the five Chişinău sectors sum to **567,038**, exactly the `or. Chişinău` row that is
  dropped to avoid double counting.

## 4. The row structure is three tiers deep and nothing in a row says which tier it is

Sheet `5.31` interleaves raion totals with UATs, and inside municipiul Chişinău there is a
third row — `or. Chişinău, din care pe sectoare` — which is the sum of the five sector rows
below it. Read as delivered, Chişinău is counted three times.

The rule used is structural rather than positional, because the sheet may be reordered:

* a code ending `00000` is a raion or municipiu (35 of them), dropped;
* `0101000` is the Chişinău city row, dropped by code, keeping its five sectors;
* everything else is a leaf, and there are 901.

## 5. The geography, which nearly went badly

Three of the usual boundary sources stop at the raion and would have forced a 35-unit map:

* geoBoundaries has `MDA` at ADM0 and ADM1 only;
* the HDX COD-AB `cod-ab-mda` geojson bundle contains `mda_admin0` and `mda_admin1` only;
* Kontur's `kontur_boundaries_MD_20230628.gpkg` has 287 units at its finest level.

A name join to OpenStreetMap was built, got to 898 of 901, and was **thrown away** when
BNS's own GIS server turned up. It is worth recording what it cost and what it found,
because the same shape will come up in another country:

* CUATM has two parallel numberings, a 7-digit `cod statistic` and a 4-digit `cod unic`,
  and below the raion **neither is derivable from the other** — Drepcăuţi is `1422000` and
  `1426`, because the unique code numbers component villages that the statistical code does
  not. The census publishes the statistical code; OSM tags `ref:cuatm:codunic`, on 802 of
  982 relations.
* Six names collide inside their own raion, every one of them a town and a village of the
  same name (`or. Dondușeni` beside the village Dondușeni). This is exactly
  [[reference_name_join_wrong_neighbour]]'s failure mode and it is invisible to any totals
  check.
* Sărata-Răzeşi, a real UAT of 509 people in Leova, **is not in OpenStreetMap at any
  admin_level**, and OSM's other Leova relations already tile the raion, so its ground is
  inside a neighbour's polygon.

**The layer actually used is `comune_p_distrib_2024_view` on `gis.statistica.md`**, 897
polygons keyed by `code_com`, which is the CUATM statistical code with the trailing `00`
dropped. Its code set *is* the census table's: 896 of the 897 are exactly the 896
non-sector UATs of `md.csv`, and the 897th is `01010`, oraşul Chişinău.

**And it carries `p_distrib`, BNS's own 2024 population for the polygon, which equals the
total computed here from table 5.31 to the person for all 896 units.** `01010`'s 567,038
equals the sum of the five census sectors. A code join can still be a join to the wrong
vintage of the boundaries; agreement on a published population as well is what rules that
out, and it is a stronger statement than any of the drawn countries that join by name can
make.

BNS's parallel ArcGIS Online organisation publishes the same tier as `lau2_2024`
(`services-eu1.arcgis.com/Pqa2pBN0HSNd7QdR`), 982 polygons, which is this layer plus
Transnistria, Bender and the four named right-bank exclusions. It is deliberately not used:
`comune_p_distrib` is already the census's universe, so *who is missing* is answered by BNS
rather than by a filter written here. The same server also has
`localitate_p_distrib_2024_view`, 1,529 **points** carrying the full 7-digit code, which is
the only geometry below UAT level anywhere and is not needed for religion.

### The five Chişinău sectors

BNS draws oraşul Chişinău as one polygon and counts it as five. Botanica, Buiucani, Centru,
Ciocana and Rîşcani are 567,038 people, 23.5% of the country, and one polygon for them
would have been the second worst capital case on this map after Bucharest. The sectors are
taken from OpenStreetMap's five `admin_level=7` relations and **clipped to `01010`**; the
clip matters, because OSM's sectors are sectors of the *municipality* and reach out over
eighteen suburban towns and communes that the census counts separately. After clipping they
cover 99.8% of the city and do not overlap each other, both asserted.

## 6. Who is not drawn

Footnote 1 of tables 5.29 and 5.31, verbatim in `sources/md.py`'s `EXCLUDED_NOTE`: the
census covers only the UATs actually enumerated, and does **not** include the
administrative-territorial units on the left bank of the Nistru, municipiul Bender
(including Proteagailovca), comuna Chiţcani (including Mereneşti and Zahorna), the villages
Cremenciug and Gîsca of raionul Căuşeni, comuna Corjova (including Mahala) of raionul
Dubăsari, and the village Roghi in comuna Molovata Nouă, raionul Dubăsari.

That is Transnistria plus the right-bank places administered from Tiraspol. **No `gap_share`
is set**, because this census publishes no population for them and a figure from anywhere
else would be a different instrument sized against this one; the same call `ge` makes for
Abkhazia. Note that some left-bank villages of raionul Dubăsari *are* Moldova-administered
and *were* enumerated, so the drawn area is not simply the right bank: the polygons are
30,323 km², 89.6% of Moldova's 33,846.

## 7. What is drawn

2,391,104 of 2,409,207 people, on 12 nodes. `Nu au declarat religia`, 18,103 people, is the
only category off the tree.

Three things here are worth a reader's attention and all three come from the fine grain:

* **Briceni** is 80.5% Orthodox where Şoldăneşti is 98.7%, and the difference is Jehovah's
  Witnesses (3,603, 7.7% of the raion) and Pentecostals (1,979, 4.2%). In the village of
  Caracuşenii Vechi the Witnesses are 822 of 2,629 people.
* **Cunicea**, in Floreşti, has 982 Old Believers and **Pocrovca**, in Dondușeni, is 921
  out of 940 residents; the two together are nearly half of the 4,053 the whole country has.
  (Corrected by the review pass, §9: the original read as though Pocrovca were the largest.)
* The two Orthodox jurisdictions, the Metropolis of Chişinău under Moscow and the Metropolis
  of Bessarabia under Bucharest, are **one column**. That is the live religious question in
  Moldova and this source cannot answer it.

## 8. Not done

* **Sheet 5.35, religion × ethnicity, national only.** Would say whether the Gagauz and the
  Bulgarians of the south differ from the Moldovan Orthodox majority. National-only, so it
  cannot be drawn, but it would be a good check on the `Alte religii` cell.
* **The 2014 raion table is not ingested**, so nothing here shows change over the decade.
  It would need its own `source_id` and a second normalized file; the categories do not line
  up (§2), so it is not a straight comparison.
* **Nothing splits the Orthodox column.** The two metropolises publish parish lists; a
  congregations-basis layer would be a different `basis` from this one and would not mix
  (spec §3.1).

## 9. Review pass, 2026-09-08

Second look by a reviewer, not the builder. Everything below is checked against
`data/normalized/md.csv` directly rather than against §3-§7's account of it.

**Reader-facing figures all reconcile except one.** Orthodox 94.268% (drawn as 94.3%),
non-response 0.751% (0.75%), `Alte religii` 4,720, Old Believers 4,053, Briceni 80.46%
Orthodox with Witnesses at 7.68% and Pentecostals at 4.22%, Şoldăneşti 98.74%, Caracuşenii
Vechi 822 of 2,629, mean UAT 2,674 people. Briceni really is the least Orthodox raion and
Şoldăneşti really is the most, so the pairing in `note_public` is a fair one.

**The Pocrovca claim was the exception and has been corrected in `countries.py` and above.**
Pocrovca's 921 Old Believers are 22.7% of the national 4,053, so "nearly a quarter" was
true of it, but **Cunicea, in Floreşti, has 982**, which is both more people and a larger
share (24.2%). Naming Pocrovca alone read as naming the largest concentration and it is the
second. The note now says nearly half of them are in two villages, which is 1,903 of 4,053,
47.0%, and keeps Pocrovca's 921-of-940 saturation, which is the genuinely striking fact and
which Cunicea (982 of 1,656) does not match.

**The join is exact and there are no unjoined units.** 901 census UATs against
`comune_p_distrib_2024_view`'s 897 polygons is not a shortfall of four: the layer is the
896 non-sector UATs plus `01010` oraşul Chişinău, and `01010` is replaced by the five
sectors, so 896 + 5 = 901 both ways. Checked in `data/geo/md/md_uat.gpkg`: 901 polygons,
the unit set matches `md.csv` exactly in both directions, `md_uat_lookup.csv` matches both,
and **no polygon with a `0101` prefix is drawn**, so oraşul Chişinău is not on the map
beside its own sectors.

**Chişinău does not double-count.** The 23 UATs whose `raion` note says Mun. Chişinău sum to
720,128, which is the raion row to the person; the five sectors are five of those 23 and sum
to 567,038, 23.54% of the country, matching the `countries.py` note. Geometrically the five
OSM sectors clipped to `01010` overlap the 18 suburban communes by **11 m² in total across
13 pairs**, which is float slivers off the clip and not territory. Sector area 121.9 km²,
whole layer 30,308 km² in UTM 35N, both right for right-bank Moldova.

**The 2014 Judaism and Lutheran loss is stated in three reader-facing places**, not just
here in §2: the `note_public` closes on it, and `other.md`'s node description in
`branches.py` carries it too, so a reader who opens the legend row meets it as well.

**One mapping call recorded rather than changed**: `Penticostală` to
`christianity.pentecostal.trinitarian` on a bare category name, which is the one place
`md2024.py` reverses the principle it applies to `Islam` and `Adventistă` two entries away.
Reason and precedent are now in that module's `REVIEW` dict. Substance is very likely right
and nothing was altered.
