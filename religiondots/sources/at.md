# Austria — Volkszählung 2001, Tabelle 4 "Bevölkerung nach Religion"

**Drawn 2026-09-08.** 8,032,926 people, ten religion categories, 2,380 drawn units
(2,357 Gemeinden + Vienna's 23 Gemeindebezirke). 97.99% drawn; the gap is the 2.00% who did
not state a religion, plus one Gemeinde of 272 people that has no polygon.

---

## 1. Where the data is

Nine per-Bundesland volumes plus a national one, all open, all PDF:

```
https://www.statistik.at/fileadmin/publications/Volkszaehlung_2001__Hauptergebnisse_I_-_<Land>.pdf
```

`<Land>` is `Burgenland`, `Kaernten`, `Niederoesterreich`, `Oberoesterreich`, `Salzburg`,
`Steiermark`, `Tirol`, `Vorarlberg`, `Wien`, `OEsterreich`. 3–9 MB each, ~50 MB in total.

**`www.statistik.at` omits a TLS intermediate.** Every ordinary client fails with *"unable to
get local issuer certificate"*, which reads as a dead host and is not one. Third host here to
do this after `stat.gov.pl` and Ghana's StatsBank; `sources/at.py` turns verification off for
this download exactly as those two do.

**Tabelle 4 is the religion table and its tier depends on the volume.** In the eight Länder
volumes it is by Gemeinde. In the **Wien** volume it is by **Zählbezirk** (245 of them), with
the 23 Gemeindebezirke as the tier above. In the **Österreich** volume it stops at Politischer
Bezirk, so the national volume is a cross-check and not a source.

The Excel editions of these tables shipped on a **CD-ROM** with the print run and are not
online. One exception, and it is worth having:

```
https://vorarlberg.at/documents/302033/472657/Volksz%C3%A4hlung+2001+-+Bev%C3%B6lkerung+nach+Religion.xls/736910d9-36a3-51b2-e5cd-8c9c2395091f?t=1616166395157
```

The Land of Vorarlberg republishes its own Tabelle 4 as a 48 KB `.xls`, 105 rows, all 96
Gemeinden. `sources/at.py` reads it and checks the PDF parse against it: **101 rows, 1,111
figures, every one an equality.** No other Bundesland does this.

## 2. The trap that would have cost a day

**The printed column numbers are out of order, and the volumes disagree with each other.**

In the eight Länder volumes the header numbers read, left to right:

```
1   2   3   5   4   6   7   8   9   10   11
```

so **Orthodox is printed fourth and numbered 5**, and **Evangelisch is printed fifth and
numbered 4**. In the Wien volume the same eleven columns are numbered `1 2 3 4 5 6 7 8 9 10
11`, in print order.

A parser that keys on the printed number therefore swaps Orthodoxy and Protestantism in eight
volumes of nine. Nothing catches it: both are plausible sizes (159k and 376k), the partition
still sums, the Bezirk totals still agree, and the national figures still reconcile — because
the swap is consistent within each volume. It would have shown up only as Austria looking
oddly Orthodox.

`sources/at.py` identifies columns from the header **labels** by x-position and asserts the
resulting order against a written-out canonical list. Two independent things confirm the
anomaly belongs to the source and not to the extraction: the Vorarlberg `.xls` prints the same
`1 2 3 5 4 6 …` in its own header row, and the volume's prose gives three figures that pin the
label order (Vorarlberg 274,000 römisch-katholisch = 78.0%, and "höchster Anteil islamischer
Bevölkerung in Vorarlberg", both of which the parse reproduces).

**This is [[reference_pdf_table_geometry]] from the other end.** The usual advice is to render
the page before blaming the parser. Here the page is fine, the parser is fine, and the *source*
is internally inconsistent in a way that only a cross-volume comparison reveals.

## 3. The other things that bite

- **Tabelle 15 shares Tabelle 4's header.** It is the same eleven religion columns cross-tabbed
  by age and citizenship instead of by place, and in the Burgenland volume its stub carries
  figures that read as a Kennziffer. Geometry alone cannot separate them, so `at.py` bounds
  Tabelle 4 by its own printed title. **Continuation pages repeat that title**, so the table
  ends at the next page carrying a *different* number; bounding on "the next title of any
  kind" silently returns two pages of a fifty-page table and every check downstream still
  passes on the subset. That happened here and read as a clean build.
- **The fourteen Statutarstädte are printed once, at the Bezirk tier, and never as a Gemeinde.**
  Eisenstadt, Rust, Klagenfurt, Villach, Krems, St. Pölten, Waidhofen an der Ybbs, Wiener
  Neustadt, Linz, Steyr, Wels, Salzburg, Graz, Innsbruck. Left alone that drops **1,044,429
  people, 13.0% of Austria and every one of its large cities**, out of the drawn layer with no
  total disagreeing, because at the Bezirk tier they are all present and correct. Their
  Gemeinde code is the Bezirk code plus `01`, which GISCO's 2001 commune file confirms for all
  fourteen.
- **`-` is an in-band zero**, in 7,259 cells. Dropping it shifts a row left by one column.
- **The volumes' text layer loses every umlaut** to a broken font encoding (PyMuPDF returns
  U+FFFD). It does not matter, because the join is on the Kennziffer and the names come from
  GISCO, but it does mean header matching has to use ASCII-safe fragments.

## 4. Boundaries — GISCO Communes 2001

Austria has merged Gemeinden hard since the census. **Styria alone went from 542 to 287 in the
reform of 2015**, and Carinthia, Burgenland and Upper Austria have all moved. Joining a 2001
table to a current boundary file loses a third of one Bundesland and mis-seats the rest, which
is §8.1's Connecticut trap in a much larger size.

Eurostat's GISCO publishes a **Communes 2001** layer, which is the Gebietsstand of the census
to the day:

```
https://gisco-services.ec.europa.eu/distribution/v2/communes/shp/COMM_RG_01M_2001_4326.shp.zip   (80 MB)
https://gisco-services.ec.europa.eu/distribution/v2/communes/csv/COMM_AT_2001.csv                 (8 MB, attributes)
```

So **no crosswalk is written and none is needed.** 2,358 Austrian communes; the join is on
`NSI_CODE`, which is the Topographische Kennziffer the census prints in its Vorspalte.
2,357 of 2,358 match. Names are used only to *check* the join and agree on 2,344 of 2,357
(99.4%) after folding.

Two things about that file:

- **It lies about its own encoding.** The `.cpg` says UTF-8 and the `.dbf` is not, so a default
  read dies on some commune elsewhere in Europe before ever reaching Austria. Read it as
  latin-1 for the geometry and take names from the separate attribute CSV, which really is
  UTF-8.
- **One commune is several features**, split by surface cover, so the layer must be dissolved
  on the Kennziffer or a Gemeinde is drawn several times.

**Stallehr (80125), 272 people**, has no polygon in that layer at all. Every one of its
neighbours does, so it is an omission rather than a vintage disagreement. Its people are
dropped rather than folded into a neighbour (§3.5) and are named in `gap`. The assertion is
pinned to that one code, so a second missing unit fails the build.

**Vienna is drawn as its 23 Gemeindebezirke**, not as the single commune GISCO has for it.
`data.wien.gv.at`'s `BEZIRKSGRENZEOGD` layer carries `STATAUSTRIA_BEZ_CODE` — Statistik
Austria's own Bezirk Kennziffer — so the join is an integer equality with no name matching.
The 23 cover 1.0003 of GISCO's Wien polygon by area.

**The 245 Zählbezirke are parsed, kept in `at.csv`, and not drawn.** Vienna's current OGD
Zählbezirk layer has 250, so the vintages do not correspond and pairing them needs a crosswalk
nobody publishes. Benin's rule. If a 2001-vintage Zählbezirk layer turns up, Vienna gets six
times finer for nothing.

## 5. Placement — Kontur

`kontur_population_AT_20231101.gpkg.gz`, 6 MB. Austria is half Alps and its Gemeinden do not
know it: Sölden is 466 km², Neustift im Stubaital 250, and the Hohe Tauern fringe the same
shape, all polygons whose people live along one valley floor. It also removes the lakes, which
here sit *inside* the Gemeinden (Neusiedler See, Attersee, Wörthersee).

**The grid is 2023 and the census is 2001, and that is fine for exactly one reason**: Kontur is
a within-unit weight only. How many dots a Gemeinde gets is the census's answer, so the 22-year
gap cannot move a dot between Gemeinden; it can only place a dot inside one according to where
people live now. At a median unit of 35 km² that is a small claim and a better one than
"spread evenly over the polygon".

Which is why **the ratio band is reported and not asserted here** — a Gemeinde that grew by
half since 2001 reads 1.5 and is not a join error. The band comes out median 1.04, quartiles
0.94–1.16, which is tighter than the vintage gap entitles it to be. The check that carries the
join is the correlation: **r = 0.9792 against a best of 0.0603 over 500 shuffles.** Zimbabwe is
this the other way round.

**Rattenberg (AT70521) is 0.10 km², the smallest town in Austria, and no 400 m hex has its
centroid inside it.** A unit absent from the placement layer has no geometry to draw into and
its people are dropped, not spread, so it is given its own polygon as its placement geometry —
Mauritius's rule, and [[reference_kontur_resolution_floor]] from the other side. The assertion
allows at most five such units.

## 6. The 31 categories, and what was done with them

UNSD's Demographic Yearbook carries **31 named bodies** for Austria 2001 and this is why the
country sat high in `queue.md`. **They are national and do not exist at Gemeinde.** Every
subnational table in the 2001 publications carries ten, and Tabelle 15 — the Bundesland
cross-tab, where more detail would live if it lived anywhere — carries the same ten cross-tabbed
by age instead. §3.9 in its usual shape.

They are used as a **check** instead, and it is much the sharpest one here. Six of the 31
reproduce a drawn column exactly (Roman Catholic 5,915,421, Protestant 376,150, Islam 338,988,
No Religion 963,263, Not Specified 160,662, Jewish 8,140). The other 25 decompose the remaining
four columns **to the person, with no row used twice and no remainder**:

| column | = |
|---|---|
| Griechisch-katholisch 1,853 | Greek Oriental 1,089 + Catholic 764 |
| Orthodox 179,472 | Orthodox 159,115 + Greek Orthodox 18,533 + Armenian Apostolic 1,824 |
| Andere christliche 69,227 | Jehovah Witness 23,206 + Old Catholic 14,621 + 11 more |
| Andere nichtchristliche 19,750 | Buddhist 10,402 + Hindu 3,629 + Sikh 2,794 + 4 more |

Four exact sums over 25 independently published figures would not survive a single misread
column, so this doubles as the strongest parse check in the file.

**Two of UNSD's English labels do not mean what they say.** `Greek Oriental` (1,089) and
`Catholic` (764) sum to Griechisch-katholisch and belong to nothing else — 1,853 has no other
decomposition among the 25. `Greek Oriental` is a translation of *griechisch-orientalisch*, the
old Austrian term for the Orthodox, and it lands in the Catholic column anyway. Which is the
reason nothing here maps a UNSD row by its name.

### The decision: the two `Andere` cells are NOT split

The decomposition is proved, so the split *could* be applied the way Germany's ESS residual is.
It is not, and the reason is that **Austria publishes those figures at one geography: the
country.** Germany's split buys nine Bundesländer of variation and India's buys states;
Austria's would buy none. All 2,380 units would receive an identical internal mix, so the map
would gain a dozen legend rows and not one spatial fact, and 88,977 measured people would
become derived to pay for it. §14's first rule is not to estimate a magnitude at a finer
resolution than the source publishes it, and a national constant pushed to Gemeinde is that in
its purest form.

What the reader gets instead is `note_public` naming the contents and their national counts.
**The decomposition is banked in `sources/at.py`'s `CROSSWALK` and asserted on every run**, so
reversing this call is ten minutes rather than a day.

## 7. The vintage, which is the only real objection to this country

**2001 is the last census that asked, and there is no post-2001 religion figure for any
Austrian place.** Statistik Austria's own wording: since the move to a register-based census,
*"dieses Merkmal im Rahmen des Zensus nicht mehr erhoben"* wird. This was checked rather than
taken on trust:

- **`data.statistik.gv.at`** — the OGD catalogue is 1,069,292 bytes and 469 dataset ids, with
  **zero** matches for `relig`, `konfession` or `bekenntnis`. The census datasets there are
  population totals and age/sex/citizenship. (This confirms §11d.)
- **`data.gv.at`** — its real search API is
  `https://www.data.gv.at/api/hub/search/search?q=…&filters=dataset` (the `/katalog/api/3/…`
  CKAN route 404s because the site is an SPA). 16 religion hits, none census-related.
- **STATcube** — guest login needs no registration but is capped at **10,000 cells** and
  explicitly excludes API access. Austria at Gemeinde × religion is ~26,000 cells, so the
  table is out of reach for a guest even if it is exposed.
- **Wayback CDX on statistik.at** — three archived religion spreadsheets, all Bundesland or
  national.

The office's current figures come from **extra questions on the Mikrozensus labour force
survey**, Q1–Q4 2021, published at Bundesland only, in
`fileadmin/pages/439/neu__Religion_2021_Bundesland.ods`. Six categories, with a stated floor:
values under 6,000 people are *"sehr stark zufallsbehaftet"* and under 3,000 *"statistisch
nicht interpretierbar"*.

### Why the country is drawn at 2001 as published, rather than rescaled to 2021

§3.4 would license taking the 2001 Gemeinde structure and fitting it inside 2021 Bundesland
totals. That was considered and **rejected**, on four grounds:

1. **The 2021 source is a sample survey, not a total.** §3.4's Brazilian case fits an old
   structure inside a newer *census*. Fitting 8.03M measured people to an LFS supplement whose
   own publisher marks cells under 3,000 as uninterpretable would convert the whole country to
   `modelled` on the strength of that supplement. Several of the 9 × 6 cells it would supply
   are at or under that floor (Burgenland's *Andere Religion* is 2,134).
2. **The category sets do not crosswalk.** 2021 has no `Unbekannt` at all, and its residual
   *Christentum* cell is 4.29% against the 0.86% of the 2001 column it would pair with — a
   ~5× mismatch of exactly the kind that produced Brazil's `Outras religiosidades` at 626×.
3. **It assumes the within-Bundesland geography of each religion is unchanged since 2001**,
   which is least defensible for the two categories that moved most: Islam and Orthodoxy are
   immigration-driven and city-concentrated, and Vienna's housing has changed.
4. **Church membership registers are not an alternative**, because the census is `self_id` and
   a register is a `roll`; §3.1 forbids mixing the two bases.

So Austria goes in at its own year, with the year on every figure, and `note_public` states the
drift in numbers (73.6% → 55.2% Catholic, 12.0% → 22.4% none, 4.2% → 8.3% Muslim, 2.2% → 4.9%
Orthodox). That is §3.4's own fallback for the case where there is no recent *total* to rescale
to, and it is what India (2011) and Russia (2012) already do. **The confidence tier is not
reduced for age alone** — an old census is a measurement.

## 8. The non-response, and which way it leans

160,662 people, 2.00%, in `Unbekannt`. Not the same answer as `Ohne Bekenntnis`, which 963,263
gave and which is drawn.

§3.5's lean check, over the 2,381 units: the `Unbekannt` share correlates **+0.43** with the
no-religion share, **+0.41** with the Orthodox share, **+0.35** with the other-non-Christian
share and **−0.39** with the Roman Catholic share. It runs 4.24% in Vienna against 0.57% in
Burgenland. So the excluded are drawn disproportionately from the least religious and most
minority places, and **every share this map draws for Austria is slightly more Catholic than
Austria was.**

## 9. Unspent

- `fileadmin/pages/402/Religion.ods` sheet A2 — national 2001 religion × citizenship with
  **~60 denominations**, which is deeper than UNSD's 31. National only. It would refine the
  §6 crosswalk if the split is ever wanted, and it splits by citizenship, which the ten-column
  table does not.
- **Vienna at Zählbezirk**, if a 2001-vintage boundary layer appears (§4).
- The Wien volume's Tabelle 5 is Umgangssprache by Gemeindebezirk. Not religion, and §14.5
  territory.

## 10. Review pass, 2026-09-08

Second read against `data/normalized/at.csv` rather than against §1-9. Everything in §6-8
reconciles: the ten national shares, the 2,381 drawn units, the 8,032,926 universe, the
160,662 + 272 that makes the drawn 7,871,992, and §8's three correlations (+0.427, +0.408,
−0.392 recomputed unweighted over the drawn units). The Bundesland figures in `note_public`
are right to the decimal — Vorarlberg 8.36% Muslim, Burgenland 13.26% and Kärnten 10.32%
Protestant against Vorarlberg 2.23%, Vienna 49.16% Catholic and 25.65% with no religion,
Unbekannt 4.24% in Vienna against 0.57% in Burgenland. The Jewish claim holds exactly:
Innere Stadt 3.34% and Leopoldstadt 3.11% are the only two units in the country above 1%,
the next is Döbling at 0.78%. Rudolfsheim-Fünfhaus is 40.22% Catholic, 14.69% Muslim and
11.37% Orthodox as stated. Screenshot glance: dots sit inside the border, follow the Inn and
Drau valleys and the Vienna basin, and the legend totals match the CSV column for column.

**Two things in `note_public` that a reader can catch, both illustrative rather than
structural.** Neither was changed here; they are the author's to take or leave.

1. *"Inside it the range is wider still, from 40.2% Catholic in Rudolfsheim-Fünfhaus to the
   outer western districts"* has no closing figure, so the comparison dangles. The top of the
   Vienna range is **Hietzing at 57.74%**, then Döbling 55.73% and Liesing 55.23%, so "outer
   western districts" is a fair description of that end; only the number is missing.

2. *"The largest Muslim share in Austria was not in Vienna. Lustenau in Vorarlberg was 15.7%,
   ahead of every Viennese district"* is true as literally written, but the bolded lead makes
   a reader take Lustenau for the national maximum and it is fifth. Ahead of it are
   **Waldegg 18.82%** (2,062 people), Felixdorf 17.02% (4,288), Fulpmes 16.38% (3,895) and
   Wimpassing im Schwarzatale 16.03% (1,915) — the Steinfeld mill villages south of Wiener
   Neustadt, plus one Tyrolean toolmaking town. Lustenau's 19,709 people make it the largest
   Muslim share in any Austrian town of real size, which is the defensible version of the
   sentence and is the more interesting fact anyway.

**Spec citations in `taxonomy/at2001.py` point at the wrong sections twice.** The decisions
are both right; only the numbers are wrong, and a later session copying the citation would
propagate it. `EXCLUDED["Unbekannt"]` calls conflating a refusal with a report of no religion
"the §3.1 error"; §3.1 governs the `basis` field and both cells are `self_id`, so the
authority is §3.5. `REVIEW["Evangelisch"]` says "§3.1 forbids adding" a split the source does
not make; §3.1 does not forbid splits, §3.4 expressly permits them, and the rule that actually
refuses an Evangelisch → Lutheran split is §14's first. The docstring's paraphrase of that
first rule as "not to estimate a magnitude at a finer resolution than the source publishes it"
is also loose — rule 1 is about estimating a magnitude at all; §3.10, on assuming a branch's
internal composition is constant across fine units, is the sharper cite for the `Andere` call.

**Checked and found fine, so nobody need look again:** `other.at` is one of 90 per-country
residual nodes and collapses into the single "Other" overview row, so it adds no legend row;
Orthodox → the parent rather than `.canonical` matches `de2022` and `hu2022` and is
composition-driven, not an inconsistency with `ch2000`; Catholic → `.latin` is safe because
the census prints Griechisch-katholisch separately; and excluding the non-response cell is
what 39 of the 50 modules that face the question do.
