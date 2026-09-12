# Benin — INStaD, RGPH-4 (2013), the twelve departmental *Principaux indicateurs*, Tableau 8

Wired 2026-09-07. 10,008,749 people, 77 communes, 10 drawn categories, 9,887,923 drawn.

| | |
|---|---|
| source | Institut National de la Statistique et de la Démographie, **RGPH-4 (2013)**, twelve departmental *Principaux indicateurs* booklets, **Tableau 8: ETHNIE ET RELIGION PAR COMMUNE**, with commune populations from **Tableau 2** of the same booklets |
| basis | `self_id`, whole census population |
| geography | **77 communes** — ~130,000 people each |
| categories | **10**, all drawn |
| drawn | **9,887,923 people, 98.79%** — the gap is 1.21% non-response, computed rather than published |
| licence | INStaD publication, free to download and cite |

**The first Vodun mapping on this project, and probably the first anywhere at commune
resolution.** RGPH-4 gives `Vodoun` a cell of its own, separate from `Autres
traditionnelles`, and it also gives one to `Chrétien céleste` — the Celestial Church of
Christ, an African Instituted Church founded in Porto-Novo in 1947. Those two cells are the
whole reason to draw Benin: everywhere else in Africa the traditional religions arrive as
one undifferentiated box and the African churches arrive inside `Other Christian`.

---

## 1. The table is percentages, and sources.md §11p was wrong about what that costs

§11p recorded Benin as buildable but expensive: *"the tables are percentages to one decimal,
not counts, so commune totals have to come from the Résultats définitifs and be multiplied —
which is a §3.4-shaped join and loses precision at the small end."*

**Half of that is right and the expensive half is not.** The percentages are real. But
**Tableau 2 of the same booklet prints the population of every commune in it**, so no join
to a second document is needed and each of the twelve booklets is self-contained. The
§3.4-shaped cross-vintage join never happens; the arithmetic is a published share times a
published total from the same page range of the same PDF.

**And it is not a §3.4 case at all, which matters for §7a's tier.** Nothing is carried from
a coarser level and nothing is fitted: every person here was counted by INStaD, in 2013, in
the commune they are drawn in. What the percentage costs is **precision**, and the bound is
computable rather than vague — one decimal on a share is ±0.05% of the unit, which is ±34
people in a 68,000-person commune and ±340 in Cotonou. That is smaller than the disclosure
noise Germany's census carries and it is not a confidence claim, so the rows are `measured`.

**The general form, for whoever meets the next percentage-only source:** ask whether the
same publication also prints the denominator, before assuming a rescale. A report designed
to be read by a prefect usually does — it is the first table in the booklet.

## 2. Four checks, and the useful one is a different file

| check | what it would catch |
|---|---|
| the 12 departments sum to 10,008,749; the 77 communes sum to it too | a misparsed population, a missing or duplicated column |
| **each booklet's department column against `Doc_principaux_indicateurs.xlsx`, at full float precision — 120 equations** | **a whole department's column read into the wrong place** |
| the communes sum to their department on all 10 categories, in the band the rounding allows | a transposed or shifted column inside one booklet |
| Cotonou's 13 arrondissements sum to Cotonou on all 10 | the Littoral parse, which has a different shape from the other eleven |

The second is the one worth copying. INStaD publishes the **same shares at full float
precision** for the nation and all twelve departments in a workbook on its *other* site —
`instad.bj`, a Joomla install, under `resultats finaux/Resultats globaux/` — while the
booklets live on `rgph5.instad.bj`, a WordPress install. Two sites, two formats, one
release. Every rounded departmental figure in the twelve PDFs is checked against it and all
120 agree.

**Every band is computed from the rounding, never chosen to pass.** For a parent and its
children the tolerance is `0.0005 × (Σ children's populations + parent's population)`,
because that is exactly what ±0.05% on each printed share allows.

## 3. Three misspellings by the office, all load-bearing

Not a complaint — these are keys, and repairing them breaks the parse (§12).

* **`Principaux` is spelled four ways across the twelve filenames**: `Princiapux-indicateurs-Mono_Final.pdf`, `Principaunx-indicateurs-Plateau_Final.pdf`, `Principaux-idnicateurs-Couffo_Final.pdf`, `Principaux-idndicateurs-Oueme_Final.pdf`. They are transcribed verbatim in `BOOKLETS`.
* **The Ouémé booklet titles its table `PAR COMMMUNE`**, with three Ms. The title regex tolerates it. A regex written from the other eleven finds eleven booklets and reports the twelfth as missing the table entirely — which reads as "INStaD did not publish religion for the Ouémé", a department of 1.1 million people.
* **The workbook's row label is `Autres Réligion`.** Its ten labels differ from the booklets' ten in four places, so `XLSX_LABELS` maps between them explicitly rather than folding.

## 4. The parse, and the two traps in it

Read as text lines, Tableau 8 returns the first value of a row on its own line and the rest
run together, so column identity is destroyed. The module works on **word coordinates**:
rows are clustered by vertical centre, data rows are the ones carrying a full complement of
percentage-shaped tokens, and each column's x-centre is the median over those rows. The
header is then rebuilt by giving **every header word to its nearest column**, which absorbs
the wrapped names (`Abomey-` above `Calavi`, `Akpro-` above `Missérété`, `Torri-` above
`Bossito`) with no special case at all.

**Trap 1 — the table's own title number is a digit in the header band.** `Tableau 2 :` puts
a bare `2` at x=115. That is 190 points from most departments' first column and **14 points
from the Littoral's and the Plateau's**, so reading the population row without cutting the
title off gives **Cotonou 2,679,012 people instead of 679,012** and the Plateau 2,622,372
instead of 622,372 — in two booklets of twelve, with the other ten correct. Both wrong
figures are plausible, both parse as integers, and the only thing that sees it is the
national sum. *The general rule: when a table's caption is in the same coordinate space as
its data, the caption's own numbering is data-shaped.*

**Trap 2 — a fixed y-binning splits a row.** A label and its figures do not share a
baseline; the label is a point smaller and sits slightly lower. Binning on the top edge put
the label of seven of Atacora's ten religion rows in a different bin from their values, and
the table came back looking as though INStaD had changed its category list. Clustering on
the vertical centre with a tolerance fixes it.

**`(*)` means "under 0.1%", not "missing"**, and the booklets say so in a footnote. **No
religion cell in any of the twelve uses it** — it appears in Tableau 2's age rows and in
Tableau 8 only inside the footnote — but the parser handles it and `check()` counts it,
because dropping the token would leave the row one value short and shift every remaining
figure one column left, in silence. Malawi's Likoma trap with a whole row's blast radius.

## 5. Cotonou was offered thirteen times finer and is drawn once

**The Littoral booklet publishes Tableau 8 `PAR ARRONDISSEMENT`** — religion for each of
Cotonou's 13 arrondissements, 679,012 people, 6.8% of Benin — where the other eleven publish
`PAR COMMUNE`. That is §12's capital-city case exactly: *if the office publishes religion for
city districts, use them and let them REPLACE the parent.*

**It is not used, and the reason is that no boundary layer for it could be verified.**

* **COD-AB Benin has no ADM3 at all.** The bundle holds admin0, admin1, admin2, capitals, lines and points. It stops at the 77 communes.
* **geoBoundaries BEN ADM3 has 546 arrondissements**, from OpenStreetMap via a uMap, ODbL. Its thirteen Cotonou polygons are all present and unambiguously numbered 1er to 13ème.
* **They do not agree with COD about where Cotonou is.** Union of the thirteen: 81.65 km²; COD's Cotonou: 80.58 km². The areas agree to 1.3% and the **IoU is 0.729** — 13.3 km² sticks out and 12.2 km² is uncovered. Clipping to COD's outline leaves the 2ème arrondissement with 15.7% of itself.
* **And the independent check fails.** Census population per arrondissement against Kontur's, unclipped, gives a ratio band of **0.64× to 1.98×** around a median of 1.44 and a share correlation of **r = 0.81**. The project's standard (§12, North Macedonia and Serbia) is a factor of two around a tight median. Clipped it is worse: 0.19× to 1.98×, r = 0.66.

A band that wide cannot distinguish "OSM drew the internal boundaries approximately" from
"the numbering does not correspond", and §12's costliest failure is a confident wrong
pairing. **So Cotonou stays one polygon.** `sources/bj.py` parses the thirteen anyway,
checks that they sum to Cotonou's own row on all ten categories, and writes them into
`bj.csv` at `geo_level=arrondissement` — so if an authoritative Cotonou arrondissement layer
ever appears, this is a lookup and not a re-parse.

**Two things generalise.** The first is that §12's capital rule has a precondition nobody had
written down: *use the city districts if the office publishes them* assumes a boundary layer
exists at that tier, and Hungary found one for Budapest in geoBoundaries ADM2 while Croatia
did not for Zagreb. Benin is the third case and the first where the layer **exists and is not
good enough** — which is a different answer from both. The second is that the deciding
evidence was a quantity neither side determines. Name agreement was total; it said nothing.

## 6. The join: five names disagree and not one of them is an accent

COD's ADM2 is the census's commune tier exactly — 77 polygons, split 6/9/8/8/6/6/4/1/6/9/5/9
across the twelve departments, which is the split the booklets print. Both are asserted
before the join.

INStaD and COD romanise Benin's languages differently, and the differences are consonants:

| INStaD | COD | what differs |
|---|---|---|
| `Boukoumbé` | `Boukombe` | `ou` for `o` |
| `Cobly` | `Kobli` | `C` for `K`, `y` for `i` |
| `Torri-Bossito` | `Tori-Bossito` | a doubled `r` |
| `Akpro-Missérété` | `Akpo-Misserete` | an `r` that is simply absent |
| `Dassa` | `Dassa-Zoume` | the census abbreviates Dassa-Zoumè |

No alias table is written. Three tools in order, each refusing rather than guessing: an
exact fold takes 72; a **transliteration fold applied only inside one department** takes
three; **elimination** takes the last two, each being the only name left on either side of
its department. Two leftovers anywhere would stop the run.

**The join is scoped by DEPARTMENT NAME and never by p-code**, which is what leaves the
p-code free to test the result afterwards.

### 6a. A matching order on 71 of 77 is not an order match

This module was first written to assert that INStaD's printed column order reproduces COD's
`adm2_pcode` order — both are alphabetical within a department, so the p-code would have
been a free key and, in Malawi, exactly that check caught nothing because there was nothing
to catch. Here it fails, and the failure is instructive.

| department | INStaD prints | COD numbers | why |
|---|---|---|---|
| Atacora | Cobly, Kérou | Kérou, Kobli | COD sorts under **its own** spelling: `Kobli` follows `Kérou` where `Cobly` precedes it |
| Zou | Zagnanado, Za-Kpota | Za-Kpota, Zagnanado | the hyphen sorts before a letter for one of them and is ignored by the other |
| Donga | Copargo, Djougou | Djougou, Copargo | no explanation — COD's Donga is simply not alphabetical |

**Six rows in 77 is exactly the size of discrepancy a stale vintage also produces**, so an
assertion kept and then loosened until it passed would have stopped detecting anything.
What is asserted instead is that **no commune's rank moves by more than one place** — a
transposition of neighbours survives it, a shifted block does not — and the `geo_id` is
minted as `BJ12-07` rather than `BJ1207` so that it cannot be mistaken for a p-code.

*The rule: two orderings that agree 92% of the time are not the same ordering, and a key
built on the assumption that they are will be wrong exactly where two names are close.*

## 7. `Aucune` in the Atacora, which is the best evidence this map has for §11b's rule

sources.md §11b says every African census offering `Traditionalist` as a box exclusive of
the Christian and Muslim boxes undercounts it, by an unknown amount. That has been asserted
here for Ghana, Kenya and Malawi from general knowledge. **Benin supplies the first direct
evidence, and it comes from the `no religion` column.**

`Aucune` is 5.8% nationally. Its geography is not a national-irreligion geography:

```
Toucountouna   45.2%      Atacora
Kérou          26.9%      Atacora
Cobly          20.4%      Atacora
Tanguiéta      18.9%      Atacora
Matéri         18.8%      Atacora
Natitingou     17.3%      Atacora
...
Cotonou         2.8%      the largest city in the country
Aguégués        0.3%
```

The department is **19.0% against 2.9% in the Couffo and 2.8% in the Littoral** — and the
Atacora is the poorest and least urban department in Benin, while Cotonou, the one place a
secularising population would appear, is near the bottom. The Atacora is also where `Autres
traditionnelles` peaks, at 18.0% departmental and 54.1% in Boukoumbé. **The same communes
lead on both answers.**

The straightforward reading is that a traditional practice with no congregation, no weekly
assembly and no name on the form is being reported by some respondents as *no religion*.
That is §11b's claim, visible in a published table for the first time.

**It is not corrected.** Moving people between two published cells would invent a magnitude
(§14.4), and there is no basis for a split. It is drawn as INStaD published it and said out
loud in the country's public note and in `taxonomy/bj2013.py`.

## 8. What the map shows

**Two traditional religions, not one, and they do not overlap.** Vodun is the south-west and
`Autres traditionnelles` is the north-west, and between them they make Benin the only
country here where indigenous religion is a majority answer anywhere.

| | Vodoun | Autres traditionnelles |
|---|---|---|
| Couffo | **56.5%** | 1.4% |
| Mono | 33.1% | 1.2% |
| Zou | 20.1% | 1.9% |
| **Atacora** | 6.3% | **18.0%** |
| Alibori | 0.5% | 1.8% |

**The Vodun heartland is Adja country, not Fon country.** Djakotomey 69.1%, Toviklin 66.1%,
Lalo 56.2%, Aplahoué 55.3%, Klouékanmè 50.5% — five communes of the Couffo, and the top five
in Benin. Abomey, capital of the kingdom of Dahomey and the name in every history of the
religion, is **23.5%**; Agbangnizoun beside it is 39.2%; Ouidah's department is 12.1%. The
royal and Atlantic-trade sites are not where the census finds the most Vodun, and a reader
who knows the history will look in the wrong place first.

**The Celestial Church of Christ is 6.8% of a country and is still centred on its founding.**
Sô-Ava 30.2%, Akpro-Missérété 25.8%, Bonou 24.4%, Avrankou 23.7%, Zè 22.7%, Dangbo 22.1% —
the Ouémé valley and the lagoons around Porto-Novo, where Samuel Oschoffa founded it in 1947.
Under 0.2% across the north. No other source on this map counts a single African Instituted
Church anywhere near that size; Kenya's node pools five bodies to reach 7.0%.

**Islam is the north and the boundary is one of the sharpest on the map.** Karimama 95.4%,
Malanville 94.4%, Ségbana 92.3%, Kalalé 91.2% — the Niger valley and the Alibori — against
0.3% in Djakotomey: a **three-hundred-fold range across 77 communes**. Benin's south has an
older and quite different Muslim population, the Yoruba communities of Porto-Novo, and the
census cannot separate them.

**Catholicism is urban, coastal and Collines.** Cotonou 51.2%, Abomey-Calavi 49.5%, then
Bantè 49.4% and Glazoué 47.8% inland, where the Société des Missions Africaines worked from
the 1890s. 1.2% in Karimama.

## 9. Ethics (§14)

**Nothing here needs a §14 conversation, and the reason is worth stating rather than
assumed.** §11p flagged that Tableau 8 carries **ethnicity beside religion**, so §14.5's rule
about deriving religion from ethnicity is in view. It is not engaged: the religion rows are
read and the ethnicity rows are not read at all. Nothing on this map is derived from a
Beninese ethnic category.

The state published this table itself, at this geography, and no drawn group is drawn finer
than the office drew it. There is no persecuted category in the list — Vodun is a recognised
religion with a national holiday, which is the opposite of the §14.2 case — and the one cell
that could carry a sensitivity, `Aucune` in the Atacora, is documented in §7 as probably
*under*-reporting traditional practice rather than exposing anybody.

The 2013 census is twelve years old at the time of writing, which is the oldest African
source here and is stated in §10 as a limitation rather than repaired.

## 10. Not done

* **RGPH-5 has no results.** `rgph5.instad.bj` shows the cartographic phase running through 2024 and publishes no tabulations. RGPH-4 is the current census for this purpose. **Twelve years is the real limitation of this country**, and Benin grew from 10.0M to roughly 14M over them; the shares are what is drawn, and the level is 2013's.
* **Cotonou's 13 arrondissements** — §5. Parsed, checked, in `bj.csv`, undrawn for want of a verifiable boundary layer.
* **Arrondissement level for the whole country does not exist in this release.** Benin has 546 arrondissements and geoBoundaries has polygons for all of them; INStaD publishes religion at that tier for the Littoral only.
* **THE 1992 CENSUS PUBLISHES RELIGION BY VILLAGE, AND IT IS A REAL ASSET NOBODY HAS SPENT.** `rgph5.instad.bj` hosts **`92-Population-par-sexe-la-religion-et-par-village.xls`**, 2.8 MB, a genuine OLE2 workbook: RGPH-2 (1992), religion by sex, nested department → sous-préfecture → arrondissement → village, for all 4,915,555 people counted that year. **The `92-` prefix is the year, not a table number** — the national total is 1992's, and reading it as a recent file would put a thirty-three-year-old Benin on the map.
  Its category list is **eight** and shaped differently: `TRADITIONNELLE` is one cell at 35.1% with no Vodun split, `AUTRES_CHRETIENS` exists, `Chrétien céleste` does not, and there is an explicit `ND` at 0.66%. So it cannot be joined to 2013 as a §3.4 structure source — the categories are not the same categories.
  What it *could* do is **placement**: 1992 village-level shares as a within-commune weight for 2013 commune counts, which is §14.4's "refine placement only" and the shape of §8.4. That was not built. Twenty-one years and a changed category list are a lot to ask of a weight, and the honest version needs its own validation against something independent, which nothing here supplies. Recorded so that it is a decision rather than an oversight.
* **The ethnicity half of Tableau 8** is parsed past and never read (§9).
* **No branch for Islam and none for the Protestants beyond the Methodists.** RGPH-4 gives none, and Benin's Sunni Maliki / Tijaniyya composition would be an inference.
