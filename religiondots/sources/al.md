# Albania — INSTAT, Population and Housing Census 2023

Drawn from **table 1.13 `Popullsia banuese sipas besimit fetar dhe gjinisë`**, ten categories
on the twelve qarqe, 2,402,113 people. Boundaries and the join proof come from INSTAT's own
ArcGIS Online organisation. Built 2026-09-08.

---

## 1. The host was never down, and three separate things make a sweep think otherwise

`queue.md` carried Albania as **"`instat.gov.al` did not resolve on 2026-09-06, -07 or -08 —
connection layer, not a 403"**, and `sources.md` §11k and §11o both recorded the same. On
2026-09-08, with `User-Agent: Mozilla/5.0`, `https://www.instat.gov.al/` answers **200 with a
119 KB page**, and every path under it that this build needs answers too. Nothing about the
office changed in between. This is now the fourth or fifth country on this map to come back
from a recorded connection failure; §12's retry rule is worth more than it sounds.

Three further things sit between a sweep and this country's data, and each one on its own is
enough to make an office look like it publishes nothing:

* **The database is on a non-standard port and only the census microsite links it.**
  `databaza.instat.gov.al` on 443 goes nowhere, which is what earlier sweeps tested; the live
  system is `https://databaza.instat.gov.al:8083/pxweb/en/DST/`, a **PxWeb 21.1**, and the
  only place on the site that names it is `/en/census-2023/`. This is Egypt's port-8080 shape
  exactly.
* **The English database is EMPTY under Census 2023 and the Albanian one is not.**
  `/pxweb/en/DST/START__Census2023__Census_Bashki/` returns 200, resolves the breadcrumb to
  *Municipality level data*, and lists **zero** tables. `/pxweb/sq/DST/` on the same path
  lists **24**. The English tree exists as folders with nothing in them, so a reader who
  stays in English concludes the 2023 census was never loaded.
* **PxWeb's own search index does not cover the 2023 branch.** Searching the database for
  `religion` returns two hits, both 2011; searching it for `fetare` returns five, all 2011 or
  earlier; searching it for `2023` returns three population projections. The Census 2023
  tables are reachable only by walking the tree.

**PxWeb's REST API is not exposed here.** `/api/v1/<lang>/DST/...` is an IIS 404 at every
spelling tried and `/pxweb/api/v1/...` returns PxWeb's own 500 page, so tables are read
through the UI or, as here, from the published XLS.

---

## 2. Religion stops at the qark, and that was established three ways

The interesting question for this country was never whether INSTAT publishes religion; it is
how finely. The 2023 census is published at **four** levels and the answer differs by level.

| PxWeb folder | tier | tables | religion? |
|---|---|---|---|
| `START__Census2023__Cens` | national | 41 | **yes** — `CE15` |
| `START__Census2023__Census_Qark` | 12 qarqe | 39 | **yes** — `CENS_13` |
| `START__Census2023__Census_Bashki` | 61 bashki | 24 | no |
| `START__Census2023__Census_NjesiAdm` | 373 njësi administrative | 8 | no |

The bashki set is age, marital status, citizenship, fertility, birthplace, return migration,
education, childcare, disability, household composition, family nuclei, amenities and
dwellings. The njësi set is seven tables of the same kind. **Neither carries religion,
ethnicity or language**, all three of which are in the qark set as tables 1.12, 1.13 and 1.14.

Two independent confirmations, because a single missing folder is not evidence:

* **The published XLS set has the same shape.** The census theme page serves 33 files whose
  names end `_qarqe.xls`, of which `tab_1_13_...besimit-fetar..._qarqe.xls` is this one. There
  is no bashki religion file in the series.
* **The 2011 prefecture booklets, which DO have a per-commune section, stop before religion.**
  Each of the twelve 2011 volumes (`/media/3059/1__berat.pdf` … `/media/3070/12__vlore.pdf`)
  has a whole second half of tables *sipas bashkisë/komunës*: 2.1.1 to 2.5.6, population, age,
  marital status, education, disability, households, family nuclei, amenities, buildings,
  water, toilets, heating. Religion is table **1.1.13**, in the prefecture-level half, and has
  no section-2 counterpart. This is the check the brief's "ask for a publication per unit of
  the tier below" rule asks for, and here it comes back negative.

**The 2011 public microdata does not rescue it either.** `/media/1548/censusi_i_popullsis__
dhe_banesave_2011_-_mikrodata.rar` is an open download, no login, 973 KB, holding `ind.sav`
and `hh.sav`. `ind.sav` has 44 variables including `IND_40 Besimi fetar` — and its only
geography is `Id_00n Prefektura` plus urban/rural. Twelve units again.

So 12 qarqe, 200,176 people each, is the ceiling for the full category set. That is coarser
than Armenia's 11 marzes (266,612 each) on units and finer on people, and spec §3.9b already
withdrew the unit-count floor that §11d invoked against this country.

---

## 3. The thing that IS finer, and why it is not drawn

**INSTAT owns 739 items on ArcGIS Online**, at
`services7.arcgis.com/E9FE1JuiACmTPbPv/arcgis/rest/services`, one hosted view per published
indicator per geography per census. Found through `arcgis.com/sharing/rest/search?q=owner:INSTAT`;
it is not linked from the statistics site's data pages and it is not on Albania's national
open-data portal. Sixteen of those items are religion:

| tier | units | 2011 | 2023 |
|---|---|---|---|
| `prefecture_p_{muslim,bekta,cathol,orthod}_YYYY_view` | 12 | yes | yes |
| `municipality_p_{muslim,bekta,cathol,orthod}_2011_view` | 61 | yes | — |
| `administrativeunit_p_{muslim,bekta,cathol,orthod}_2011_view` | **373** | yes | — |

Each carries one percentage field of the total resident population, on the polygon, with the
office's own codes. **The 373-unit layers are thirty times finer than what is drawn here.**
They are not used, and the reasons are cumulative rather than one:

1. **Four categories out of ten.** Muslim, Bektashi, Catholic and Orthodox were 75.6% of the
   2011 census (1,587,608 + 58,628 + 280,921 + 188,992 of 2,800,138, UNSD table 28). The other
   24.4% is evangelicals, other faiths, believers of no denomination, atheists, refusals and
   unknowns, all in one lump, none of them separable. Drawing those four would put a quarter
   of Albania into a single undifferentiated hole, on a map whose §3.5 machinery is built to
   name what is missing.
2. **It loses the two categories that make Albania's form unusual.** `Besimtarë të pacilësuar`
   and `Ateist` are separate boxes here and are 13.8% and 3.6% of the 2023 census; neither has
   an administrative-unit layer at all.
3. **It is the 2011 census, not the 2023 one.** Albania's internal migration over those twelve
   years was very large; Tirana qark alone is 758,513 of 2,402,113 in 2023.

**If the trade is ever worth taking, this is what it looks like**: 373 units, four colours,
about 75% of the country drawn and 25% grey, from a twelve-year-old census. Anita's call, not
an agent's, and it is a real option rather than a hypothetical one, which is why the layer
names are written out above.

### 3a. Reviewer's pricing of that option, 2026-09-08 — it costs more than "25% grey"

Added by the review pass, at the section above's invitation. Nothing here is built and nothing
is decided; the point is to make the trade legible before Anita spends a decision on it. Every
figure below was read straight off the ArcGIS layers.

**The layers do carry what §3 says.** All four `administrativeunit_p_*_2011_view` return exactly
373 features, keyed on a clean four-character `ID_ADMUNIT` that nests inside `CODE_MUNICIPALITY`
and `CODE_PREFECTURE`, with `DATA_REF = 2011` and one percentage field of the total resident
population. `administrativeunit_p_distrib_2011_view` sits beside them carrying **absolute** 2011
population per unit, summing to 2,800,138, so shares convert to counts without a second source.
Population-weighting the four layers by it reproduces **75.57%**, which is UNSD table 28's figure
to two places. So the option is buildable and §3's arithmetic is right.

**But 75% is a national average that describes almost nowhere.** Per unit:

| the four categories cover | units | share of the 2011 population living there |
|---|---:|---:|
| under 50% | 28 of 373 | 3.3% |
| under 60% | 59 | 13.1% |
| under 70% | 99 | 37.2% |
| under 75% | 136 | **49.3%** |

Median unit 82.0%, mean 78.0%. The worst are Tunjë in Elbasan at **7.4%** covered, Shëngjergj in
Tiranë at 9.7%, Sult at 13.9%, Vendreshë at 16.1%, Barmash at 19.2%. Half of Albania would live
in a unit that is more than a quarter grey, and twenty-eight units would be more grey than drawn.

**And the grey is structured, not scattered.** Population-weighted coverage by prefecture:

| grey | prefectures |
|---:|---|
| 41.1% | Vlorë |
| 34.7% | Fier |
| 33.5% | Gjirokastër |
| 33.0% | Berat |
| 28.9% | Elbasan |
| 24.7% | Tiranë |
| 7.5% to 21.6% | Korçë, Durrës, Kukës, Dibër, Lezhë, **Shkodër 7.5%** |

That is the shape that should decide this. The four categories are nearly the whole answer in the
Catholic and Sunni north and about two thirds of it across the Bektashi south, so the finer map
would be sharpest exactly where Albania is least interesting and blankest exactly where it is
most. It is §3.5's lean argument again, moved to the tier where a reader would see it.

**The vintage cost, priced rather than asserted.** The same four categories exist for both
censuses at the twelve qarqe, so the drift is measurable directly. Mean per-qark total-variation
shift 2011 to 2023 on those four: **8.40 points**, and **22.73 points in Gjirokastër**. Bektashi
alone, share of the qark:

| qark | 2011 | 2023 |
|---|---:|---:|
| Gjirokastër | 8.48% | **21.04%** |
| Vlorë | 1.08% | 6.32% |
| Fier | 1.01% | 5.88% |
| Durrës | 1.60% | 5.32% |
| Tiranë | 2.66% | 4.95% |

**136 of the 373 units read 0.00% Bektashi in 2011.** So the finer file does not merely age the
picture, it disagrees with the drawn map about the one category Albania is here for, by a factor
of two and a half in the qark the note leads with. Whether that is real change, a different
question, or 2011's boycott is not separable from this distance.

**Is a hybrid possible? Buildable, but I do not think it is defensible.** The tempting shape is
2023's ten categories at twelve qarqe for the magnitudes, with the 2011 373-unit shares used only
to place dots inside a qark, which §14's rule 1 permits as refining placement rather than
estimating a magnitude. Three things stop it:

1. **Only four of the ten have a layer.** Inside one qark, Muslim, Bektashi, Catholic and Orthodox
   would be placed on evidence and evangelicals, other, believers of no denomination and atheists
   on population alone. Two placement resolutions in one country, and `grain` can state one.
2. **The only test of whether 2011's within-qark pattern is a good 2023 prior is the qark-level
   series above, and it fails it.** A within-unit pattern is a safe prior when the unit-level value
   is stable; Gjirokastër's tripled.
3. **It would need a per-qark rescale** of the 2011 shares onto the 2023 totals, which stacks a
   modelled step on a stale one and would have to draw desaturated under §7 anyway.

**So the recommendation, for whatever a reviewer's is worth: if the finer file is ever wanted, want
it as a standalone 2011 edition and not as a refinement of the 2023 one.** As a 2011 edition it is
honest, and the grey is at least contemporaneous with the colours. As a hybrid it would quietly
assert that Albania's Bektashi live where they lived in 2011, which is the one thing the two
censuses disagree about.

---

## 4. The join, and why it is a proof rather than an assertion

`sources/al_geo.py` takes the twelve prefecture polygons from
`prefecture_p_distrib_2023_view`, whose value field `P_DISTRIB` is **the qark's 2023 census
population as an absolute number**, and checks it against the printed `Gjithsej Total` of that
qark's worksheet in table 1.13.

* All twelve agree **to the person**: Berat 140,956, Dibër 107,178, Durrës 226,863, Elbasan
  232,580, Fier 240,377, Gjirokastër 60,013, Korçë 173,091, Kukës 61,998, Lezhë 99,384,
  Shkodër 154,479, Tiranë 758,513, Vlorë 146,681.
* All twelve values are **distinct**, which is what makes it a test: a code swapped between
  two qarqe fails on the population instead of passing quietly
  (`[[reference_name_join_wrong_neighbour]]`). The script asserts the distinctness rather than
  relying on it.
* Then the four 2023 religion-share layers are checked against `count / total` computed from
  `data/normalized/al.csv`. Worst of the 48 comparisons: **7.1e-15 percentage points**, which
  is floating point and not data.

No name matching happens anywhere in this country.

---

## 5. The 15.8% who are not drawn, and what is known about them

`Preferoj të mos përgjigjem` 244,331 and `Nuk disponohet` 134,451, together **378,782 people,
15.77%** of the census. INSTAT keeps the two apart and so does the normalised file. Both are
excluded under §3.5.

**It is not spread evenly and that is the caveat that matters for reading the map.** Berat
loses 35.36% of its population to the two cells, Lezhë 4.52%. The eight drawn categories cover
64.6% of Berat and 95.5% of Lezhë, so a comparison of composition between two qarqe is partly
a comparison of response.

Albania's religion question has a contested recent history. The 2011 round drew organised
campaigns from several directions, including calls to boycott the ethnicity and religion
questions and disputes over enumerator conduct, and the 2011 census recorded 386,024 refusals
and 68,022 not-stated on the same question (UNSD table 28). The 2023 refusal cell is smaller
in absolute terms and the `not available` cell is twice as large. Nothing here tries to model
either of them.

---

## 6. Bektashi

115,644 people, 4.81% of Albania, 9.50% of its Muslims. `taxonomy/branches.py` carries the
argument for the new `islam.bektashi` node and `taxonomy/al2023.py` the mapping.

The distribution is the reason the country is worth having. Share of the qark's whole
population, all twelve, from `data/normalized/al.csv`:

| qark | Bektashi | share of qark | share of its Muslims |
|---|---:|---:|---:|
| Gjirokastër | 12,627 | **21.04%** | 59.37% |
| Berat | 11,422 | 8.10% | 21.61% |
| Vlorë | 9,263 | 6.31% | 16.51% |
| Dibër | 6,317 | 5.89% | 8.59% |
| Fier | 14,130 | 5.88% | 13.10% |
| Durrës | 12,070 | 5.32% | 8.88% |
| Tiranë | 37,515 | 4.95% | 9.22% |
| Korçë | 7,910 | 4.57% | 8.53% |
| Elbasan | 3,186 | 1.37% | 2.52% |
| Lezhë | 576 | 0.58% | 3.20% |
| Kukës | 319 | 0.51% | 0.60% |
| Shkodër | 309 | **0.20%** | 0.42% |

Gjirokastër is the only qark in the country where Bektashis outnumber other Muslims.

---

## 7. Files

```
sources/al.py        table 1.13 -> data/normalized/al.csv   (12 qarqe x 10 categories)
sources/al_geo.py    INSTAT ArcGIS -> data/geo/al/al_prefectures.gpkg  + the join proof
sources/al_grid.py   Kontur 400 m -> data/geo/al/al_hexes.gpkg          (22,289 hexes)
taxonomy/al2023.py   the mapping; EXCLUDED is the two non-response cells
```

**The `.xls` INSTAT serves is SpreadsheetML 2003 XML, not a BIFF workbook.** `xlrd` fails with
*"Expected BOF record; found b'<?xml ve'"* and `pandas.read_excel` cannot pick an engine. It is
parsed as XML. The row labels live inside an `<ss:Data>` element with an html40 namespace and
`<B>`/`<I>` children holding the Albanian and English halves, so `.text` on the Data element is
`None` and a parser that reads it sees a clean table of numbers **with no row names at all** —
which does not raise, and would silently take the categories in printed order.

**The whole file reconciles to UNSD table 28 to the person**, on all ten categories and the
total, and `sources/al.py` refuses to write anything if it does not.

---

## 8. Review pass, 2026-09-08

A second agent read the mapping, the normalised file and the note without starting from the
sections above. Everything it checked held except one thing, which is §8.2.

### 8.1 The reader-facing figures all reproduce, and so does the join

Every figure in `note_public` and in §6 above was recomputed from `data/normalized/al.csv`:
115,644 Bektashi and 4.81%, 9.50% of Muslims, 332,155 believers of no denomination and 13.83%,
85,311 atheists, 378,782 not drawn and 15.77% against `gap_share=0.1577`, 2,023,331 drawn.
The four superlatives are the class that usually fails and all four are right: Bektashi min
0.200% Shkodër and max 21.040% Gjirokastër, no denomination 1.868% Lezhë to 21.729% Vlorë,
atheism 0.350% Shkodër to 8.299% Vlorë, non-response 4.524% Lezhë to 35.361% Berat, and
Gjirokastër really is the only qark where Bektashis outnumber other Muslims. "Second largest
answer of any kind" holds against all ten cells and not only the eight drawn.

**The join was verified a third way, independently of §4's method.** §4 pairs `CODE_PREFECTURE`
to the worksheet total, which is a real proof of the attribute join but cannot see a polygon
whose `CODE_PREFECTURE` was wrong in INSTAT's own layer. So each polygon in
`al_prefectures.gpkg` was tested against its qark capital's coordinates, taken from outside this
repo: **all twelve capitals fall inside their own polygon**, the union bounds are
19.264-21.057 E, 39.645-42.661 N, and six adjacency assertions pass (Shkodër touches Kukës and
not Vlorë, Kukës does not touch Gjirokastër). The geography is right, not only the arithmetic.

The mapping is consistent with precedent on the two calls most likely to drift. A standalone
`Atheist` box goes to `secular` in au, br, cz, ie, in, lc, md, nz, ro and uk, and to
`unaffiliated` only where the box is merged with "no religion" (cy, hr, ke, sb); Albania's
`Ateist` is standalone, so `secular` is right. `unchurched` is now fourteen countries.

### 8.2 The gap leans, and the note now says which way

§3.5 asks for the direction and not only the size, and §5 gave the size. Run across the twelve
qarqe, against each drawn category's share of the **responding** population:

| category | Pearson r | Spearman | permutation p, 20k |
|---|---:|---:|---:|
| Believers without denomination | **+0.684** | +0.552 | **0.009** |
| Other religion or faith | +0.629 | +0.315 | 0.022 |
| Christian - Catholicism | **-0.542** | -0.413 | 0.079 |
| Atheists | +0.523 | +0.455 | 0.075 |
| Muslim - Bektashism | +0.496 | +0.650 | 0.112 |
| Christian - Orthodoxy | +0.184 | +0.224 | 0.567 |
| Christian - Evangelists | +0.082 | +0.161 | 0.765 |
| Muslim | +0.028 | +0.161 | 0.930 |

Taking the two excluded cells apart is sharper still: it is the **refusal** cell that carries the
signal, not the coverage one. `Preferoj të mos përgjigjem` against the no-denomination share is
r = +0.816, against Catholic r = -0.621; `Nuk disponohet` is +0.289 and -0.243. People declined
the question where the answer would have been "I believe and belong to nothing".

**The magnitude, which is deliberately NOT in the note** (§3.5: correcting is inventing a
magnitude). If the missing in each qark had answered like the responders of that qark, the drawn
national composition would move Catholic **-0.745 pp**, no denomination +0.301, Bektashi +0.171,
Muslim +0.129, atheist +0.092, everything else under 0.05. Total variation 0.74 pp, and a
permutation test on that whole-composition statistic comes back p = 0.42. So: the correlation is
real and the direction is clear, the effect on any single national share is under a point, and
the honest reading is that the lean matters for comparing two prefectures rather than for reading
the national figures. `note_public` gained one sentence naming the direction and the correlation,
in the form South Africa's note uses, and states that no correction is made.

### 8.3 Two edits the review made, both small

* **`note_public` and `taxonomy/branches.py` said the Bektashi world headquarters "has stood in
  Tirana since" 1925**, which the same note contradicts eleven lines later by saying Albania
  closed every tekke in 1967. The order's headquarters moved to Tirana following Turkey's 1925
  ban and was itself shut from 1967 to 1990. Both now read "moved to Tirana after", which keeps
  the fact and drops the continuity claim. No figure changed; `check_md.py` clean,
  `build_tree.py` re-run, `tiles.py --refresh-meta` run.
* Nothing else was touched. The note is at the house median for length (2,485 characters against
  a median of 2,232) and for structure (four bold topic sentences against a median of four), and
  carries no em dash.

### 8.4 `islam.bektashi` is placed right, and the Alevi complex now sits three ways

The placement under `islam` rather than under `alevism` is correct and `branches.py` argues it
properly: INSTAT prints the row as a subdivision of `Mysliman`, the order is one of Albania's
four recognised communities, and spec §2 files containment as it holds for people rather than as
a classification would prefer. No already-drawn country is made inconsistent by it. Kosovo's
note already says its Bektashi tekkes are inside the undivided `islam` cell, and North
Macedonia's says the same, so both roll up under the new node rather than against it. Greece
puts its Bektashi and Alevi Pomaks in `islam.sunni` and says so, which is a stated limitation of
an unsplittable ESS category rather than a contradiction.

**Worth one human eye, but not Albania's to fix.** The Alevi-Bektashi complex is now filed three
ways across three countries, each following its own source's word: `islam.bektashi` here,
`alevism` at the root for the UK's `Other religion: Alevi` write-in, and `islam.shia` for
Bulgaria's `Мюсюлманско шиитско`, whose 29,470 people `bg2021.py` itself identifies as Alevi and
Kazalbash. The `alevism` node's own text says the Shi'a placement is one "many Alevis reject",
which is in tension with Bulgaria's row and was so before Albania existed. Recorded here because
this is where someone will next be reading about Bektashis, not because anything should change.
