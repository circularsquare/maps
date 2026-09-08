# Germany — Destatis, Zensus 2022 (Sonderauswertung Religionszugehörigkeit)

Ingested and drawn 2026-09-04. Rebuild with `python sources/de.py --fetch`.

**The finest geography on the map attached to the coarsest categories on the map, and the
two facts have the same cause.** Germany does not ask about religion; it reads it off the
population register, which knows only what church tax requires it to know. Everything
below follows from that.

---

## 1. The file

One XLSX, 839,787 bytes, published 4 Jul 2024.

```
https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Zensus2022/
Publikationen/Downloads-Publikationen/Sonderauswertungen/
bevoelkerung_religionszugehoerigkeit_je_gemeinde.xlsx?__blob=publicationFile&v=3
```

Saved as `data/raw/de/religion_je_gemeinde.xlsx`. Licence **dl-de/by-2-0**.

### `?__blob=publicationFile` is load-bearing

Without it destatis returns **HTTP 200 with 71,651 bytes of HTML** — the landing page for
the download, `<title>` and all. `sources.md` §5a again, and the fifth distinct disguise
the project has met. `de.py` asserts size, then `zipfile.is_zipfile`, then that the
`Religion` sheet is present, before parsing anything.

### The sheet

`Religion`, 10,802 rows: two title lines, three header lines, one `Bund` row, 10,786
`Gemeinde` rows, and ten trailer rows carrying the Zeichenerklärung and footnotes. Ten
columns — a name, an AGS, a Regionalebene, the population, and then **three count columns
each followed by its percentage twin**.

| category (verbatim) | count | share |
|---|---|---|
| Römisch-katholische Kirche (öffentlich-rechtlich) | 20,746,959 | 25.1% |
| Evangelische Kirche (öffentlich-rechtlich) | 19,127,360 | 23.1% |
| Sonstige, keine, ohne Angabe | 42,845,220 | 51.8% |
| *Einwohnerzahl* (universe total) | 82,719,540 | |

## 2. NOBODY WAS ASKED, and this is the whole story

**Zensus 2022 carries no religion question.** The figures come out of the *Melderegister*,
which records membership of a public-law religious society because it determines church-tax
liability. So `basis` is **`roll`** and not `self_id` (spec §3.1) — an institution's
records, comparable with ASARB and not with any census that asks a person.

Destatis states the limit itself, in `Datensatzbeschreibung_Religion_Gitterzellen.xlsx`:

> Unter „Sonstige, keine, ohne Angabe“ werden alle Personen zusammengefasst, die einer
> anderen öffentlich-rechtlichen Religionsgesellschaft angehören als der
> römisch-katholischen bzw. der evangelischen Kirche. **Für diese anderen
> öffentlich-rechtlichen Religionsgesellschaften liegen nur in sehr begrenztem Umfang
> Einträge im Melderegister vor, die die entsprechenden Zugehörigkeiten nicht zuverlässig
> statistisch abbilden können, weshalb auf den Nachweis verzichtet werden muss.**

So the third category is three different things at once — in another public-law body, in
no body, or no register entry — and its composition is a property of the register, not of
the people in it. Germany's ~4M Muslims, ~2M Orthodox Christians, the Jewish communities,
the Freikirchen and the Alt-Katholiken are all inside it and cannot be separated. It maps
to `unrecorded`, a node added for this source; `taxonomy/de2022.py` has the argument.

Two smaller definitional facts, both from the same sheet: *Römisch-katholisch* **excludes**
the Alt-Katholiken, and *Evangelisch* is the **EKD** — "der Zusammenschluss der zwanzig
selbständigen lutherischen, reformierten und unierten Landeskirchen", i.e. three of the
tree's families in one number.

### Why Zensus 2011 does not rescue it

2011 *did* ask, with two questions, and the result is worse than it looks. Source:
Otto Püschel, "Religion und Glauben im Blickpunkt des Zensus 2011", *Statistische
Monatshefte Niedersachsen* 8/2014, pp. 395–402.

- **Frage 7** (mandatory): membership of a public-law religious society — Römisch-katholische
  Kirche, Evangelische Kirche, Evangelische Freikirchen, Orthodoxe Kirchen, Jüdische
  Gemeinden, Sonstige, or none.
- **Frage 8** (voluntary): Christentum, Judentum, Sunnitischer / Schiitischer / Alevitischer
  Islam, Buddhismus, Hinduismus, Sonstige, or none.

Three problems, and they compound:

1. **Frage 8 was the only voluntary question on the form** and most people skipped it.
   Destatis' conclusion: *"können mit den Ergebnissen des Zensus 2011 keine zuverlässigen
   Angaben zum Glauben bzw. zum Anteil der Weltreligionen gemacht werden."*
2. **Frage 8 was conditional.** Anyone who named a public-law body in Frage 7 was routed
   straight past it and *"sollte bzw. konnte sogar in Frage 8 keine Angabe mehr"* make. It
   was put only to people who had just said they belonged to no public-law body — which is
   the Northern Ireland trap of spec §3.1 exactly, in a different country.
3. **The richer Frage 7 breakdown came from the ~10% household sample**, extrapolated, and
   small groups fall below reliability: Jewish communities were reportable in 3 of
   Niedersachsen's 46 Kreise, the rest marked `/`.

And the decisive one for this project: **at Gemeinde level 2011 had the same three
categories as 2022**, because Gemeinde figures are counted register data —

> Die Kartendarstellung auf Gemeindeebene basiert hingegen auf ausgezählten Registerdaten.
> Dabei war lediglich eine Unterscheidung zwischen der Zugehörigkeit zur
> Römisch-katholischen Kirche, zur Evangelischen Kirche oder zu Sonstigen möglich.

7 categories at Kreis from a 10% sample, or 3 at Gemeinde. It buys nothing at fine
geography, and it still cannot see Muslims, because Islam only ever appeared in Frage 8.
The Zensus 2011 portal has since been retired and redirects to the 2022 landing page.

## 3. Reconciliation

Everything asserted, all of it passing:

| check | result |
|---|---|
| Gemeinden | 10,786 |
| every `geo_id` is the 12-digit Regionalschlüssel | yes |
| national row vs the four published figures | exact |
| national pop − Σ Gemeinde pop | **8,258**, and that is *Deutsche im Ausland* |
| Σ categories − Σ population, Gemeinde level | **−174** over 32,358 cells |
| the same on the national row | −1 |

The −174 is **Cell-Key perturbation**, not error: the method moves category cells and
leaves the Einwohnerzahl alone, and the sheet says so. The band is computed from the
method rather than chosen — the perturbations are per-cell, bounded and roughly
independent, so the sum over *n* cells grows like √*n*; `de.py` allows 5√*n* ≈ 899 and
observes 174.

**Suppression is almost absent.** 178 cells of 32,358 are the true-zero dash `–` (109
Catholic, 61 Protestant, 8 Sonstige) and nothing at all is withheld. 796 cells (2.5%)
carry a parenthesised share meaning *Aussagewert eingeschränkt*; the count is still given
and is used.

### The percentage-twin check, and what it found

Every count column is followed by its share, so `de.py` recomputes the share and compares.
A flat tolerance was wrong here: **75 cells disagree by more than 0.6pp, and every one of
them is in a Gemeinde of between 9 and 122 people.** Two documented effects, both of which
scale in *people* rather than points:

- perturbation of a few people is double-digit percentage points in a village of 18;
- where the perturbed count would give an implausible share, destatis **adjusts the share
  and leaves the count** — *"nimmt das Geheimhaltungsverfahren ... eine Anpassung des
  Anteils vor"*. **Ammeldingen an der Our** has population 18 and 20 Catholics, published
  as 100.0%.

So the check converts the disagreement back into people and bounds it there: worst case
**3.46 people**, in Emmelbaum (population 72). A percentage column misread as a count
would fail this by hundreds of thousands in every large city, which is the point of it.

## 4. Traps, for the playbook

- **HTTP 200 with an HTML landing page** unless the destatis blob parameter is present.
- **Counts are text in some cells and numbers in others, in one column of one sheet.** An
  `isinstance(v, (int, float))` filter drops 2,228,001 people and every remaining total
  still looks plausible. This cost a reconciliation pass to find.
- **A share column that is not count ÷ population, on purpose.** Assert the residual in
  the units the disclosure method works in, not in the units the table is printed in.

## 5. What is not used, and is worth using

`Religion.zip` (27,682,902 bytes,
`https://www.destatis.de/static/DE/zensus/gitterdaten/Religion.zip`) carries **the same
three categories on INSPIRE grids at 10km, 1km and 100m** — 3,821 / 210,555 / **3,088,036**
populated cells. Geometry is derivable from the cell id (ETRS89-LAEA, EPSG:3035) with no
boundary file at all.

It is no help on categories and it is the answer to Germany's placement problem. See
`sources/de_geo.md` §5.

## 6. SPLITTING THE 42.8M BUCKET — scouted 2026-09-08, BUILT 2026-09-07 as §9. Read §9 first

**Route A is built and Route B is not.** `sources/de_ess.py` splits the residual by
Bundesland; the citizenship grid that would place those dots better is still unbuilt. The
plan below is kept as written because §9 departs from it in three places and the reasons
only make sense against the original. **Four of its numbers are wrong** — five ESS rounds
are usable rather than three, the pooled Muslim share is 4.13% and not 5.00%, Hamburg's
n is 268 and not 134, and Bremen's is 72 rather than 43. §9 has the corrected figures.

## 6 (as scouted). A plan for whoever picks it up

Anita asked whether anything could get granularity on the religions that are **not** Catholic
or Protestant. Two sources were verified live; neither is wired. This section is the outline,
not the build.

**The target.** `Sonstige, keine, ohne Angabe` is 42,845,220 people, 51.8% of Germany, and it
holds the non-religious, Muslims, Orthodox, Jews and the free churches in one cell. Germany
currently draws **no Muslim, Orthodox, Jewish or free-church dots at all.**

### Route A — the magnitude: ESS `rlgdnade`, at 16 Bundesländer

Germany is on the roster of ESS countries with a **country-specific denomination variable**
(Italy is not — see `sources/it.md` §2). `rlgdnade` gives 13 categories. Usable rounds are
**8, 9 and 11**; round 10 returns `E201VariableNotFound`. ~7,600 pooled respondents,
`region` = NUTS 1 = the 16 Länder.

| | pooled share |
|---|---|
| no religion (`Not applicable`) | 44.10% |
| Roman Catholic | 22.90% |
| Protestant, EKD excl. free churches | 22.05% |
| **Muslim** | **5.00%** |
| evangelical free church | 1.42% |
| **Eastern Orthodox** | 1.11% |
| other / unspecified Christian | 1.84% |
| Eastern religions · other non-Christian | 0.68% · 0.34% |
| **Jewish** | 0.10% |

**THE CROSS-CHECK IS WHAT LICENSES THE SPLIT.** ESS says 22.90% Catholic and 22.05% EKD; the
register counts **25.1% and 23.1%**. Two independent instruments within ~2 points on the two
categories both can see — so its split of the third is worth something. ESS's non-C/P total
is 55.05% against the register's 51.8%, close enough that the bucket is the same bucket.

The operation is spec §3.4's permitted SPLIT, as Greece does for Thrace and Spain for UCIDE:
**keep each Gemeinde's `Sonstige` total exactly as the register counts it**, and split it by
its Bundesland's ESS composition of the non-Catholic/non-Protestant part. Catholic, Protestant
and the unit totals are untouched.

Real spread, and it is the two things everyone knows about Germany:

| | Muslim | no religion |
|---|---|---|
| Hamburg · Berlin | 11.2% · 11.0% | 49.3% · 59.7% |
| Nordrhein-Westfalen | 7.3% | 34.8% |
| Bayern | 2.1% | 36.2% |
| Sachsen-Anhalt · Sachsen | 1.8% · **0.7%** | **74.9%** · 72.8% |

### Route B — the placement: citizenship on the same grid the religion data uses

`https://www.destatis.de/static/DE/zensus/gitterdaten/Staatsangehoerigkeit_nach_ausgewaehlten_Laendern.zip`
— 27,225,007 bytes, **100m / 1km / 10km INSPIRE grids**, identical geometry to `Religion.zip`,
so `de_grid.py` already knows how to read it. Thirteen countries:

> Türkei · Bosnien-Herzegowina · Griechenland · Italien · Kasachstan · Kroatien · Niederlande
> · Österreich · Polen · Rumänien · Russische Föderation · Ukraine · Deutschland

Turkey and Bosnia are the Muslim signal; Romania, Russia, Ukraine and Greece the Orthodox one.
**Placement only, never a magnitude** (§8.2) — the same move `_ItWeighter` makes in
`countries.py` for Italy, France and Spain, but at 1km rather than at a municipality.
Suppression: cells under 3 are `–`.

### The four caveats, and none of them is small

1. **ESS undercounts Muslims by about a third.** 5.00% implies ~3.9M against BAMF's
   *Muslimisches Leben in Deutschland* estimate of ~5.5M. Surveys reach migrant populations
   badly and the gap runs the usual direction. State it; do not tune it.
2. **The citizenship grid misses the countries that matter most after Turkey** — no Syria
   (~900k), Afghanistan, Iraq, Iran or Morocco. And citizenship is not origin: Germany's ~3M
   people of Turkish background are mostly German citizens, so the column sees under half.
3. **Bremen has n=43 and Hamburg n=134.** `sources/it.py`'s `N_FLOOR` lesson applies directly
   — Hamburg's 11.2% is exactly the shape of claim that floor exists to stop.
4. **IT PUTS MODELLED DOTS INTO THE ONLY FULLY-MEASURED COUNTRY ON THE MAP.** Germany's tier
   is `measured` throughout and uniquely. The three register categories stay measured; only
   the split becomes `modelled`, so §7's inferred-dots toggle would empty part of Germany
   where today it empties none of it. That is a change to what this country *is* here, and
   it is a §14 call rather than a technical one.

### Order of work

1. `sources/de.py` — pool ESS 8/9/11 on `rlgdnade` × `region`, apply `N_FLOOR`, split each
   Gemeinde's `Sonstige` by its Land's composition. New `taxonomy/de2022.py` targets for the
   13 categories (the two register ones already exist).
2. `sources/de_grid.py` — read the citizenship zip onto the existing 1km cells; a weighter in
   `countries.py` on the `_ItWeighter` pattern, blending per node.
3. `note_public` has to say the split is Land-level while the geography is Gemeinde-level, and
   that the Muslim figure is a known undercount.

### The one part that would NOT be modelled, and is worth doing first

**Zentralwohlfahrtsstelle der Juden in Deutschland (ZWST)** publishes membership per Jewish
community — roughly 105 communities with locations. That is a `roll` at real coordinates: a
§4.4 **sites** layer rather than a split, needing neither ESS nor a grid, and it is the only
route here that adds a real count rather than an estimate. Checked 2026-09-07 — see §7.

## 7. ZWST — BUILT 2026-09-07. No machine-readable release, and it does not matter

Rebuild with `python sources/de_zwst.py --fetch`. Writes `data/normalized/de_zwst.csv`:
**96 Gemeinden, 87,934 members, basis `roll`, year 2025.** Drawn since 2026-09-07 —
Germany is four nodes now, and `judaism` is the first thing on its map that the Zensus
did not put there.

**There is no Excel or CSV anywhere on the site. There is a clean text layer in the PDFs, a
complete per-community table inside it, and a published total to reconcile against, which is
worth more than a spreadsheet with no check in it.**

`https://zwst.org/de/publikationen/statistik` carries a Kurzversion and a Langversion per
year for **2007–2025**, plus excerpts for 1990–2000 and a 1955–1985 series. Current file:

```
https://zwst.org/sites/default/files/2026-07/ZWST-Mitgliederstatistik-2025-web.pdf
```

2,020,546 bytes, 68 pages, `%%EOF` present, text layer throughout — no OCR. Pages 11–66 give
**every community by name with its membership, split by sex and by twelve age bands**, and
each association's communities are followed by a subtotal page.

### It reconciles exactly

`de_zwst.py` reads **105 communities summing to 87,934**, against the 87,934 published on
page 5. **Residual zero.** Thirteen association
subtotals reconcile individually on the way — IRG Baden 5,028 from 11 communities, Nordrhein
14,934 from 8 — so the check is thirteen checks and not one. 105 is also the community count
ZWST states for itself, reached here without being told it.

### Three layouts and three traps, and the second one is the dangerous one

1. **The column order flips between pages.** `G 469` on p12, `2.238 G` on p27, and the Bayern
   pages are **transposed outright** — age bands down the left, communities across. A parser
   anchored on "the number after G" reads the 0-3 age band as the total on the p27 pages, and
   **Hamburg's 2,238 members come out as 7** — small enough to pass for a real small community.
2. **`M + W == G` DOES NOT CATCH IT, and it is the check anyone would reach for.** The table is
   male/female/total in *every* column, so the age bands satisfy it too: Hamburg's wrong reading
   is 3 + 4 == 7 and validates cleanly. **The invariant that discriminates is the other one —
   the total equals the sum of its own twelve age bands.** That is false for the misreading,
   true for the real column, and true on all three layouts, so it picks the reading and
   validates the row in one step. §3 of `sources.md` again: an arithmetic check that the wrong
   answer also passes is not a check.
3. **The thousands separator is used in the total column and omitted in the age-band columns of
   the same table** — Düsseldorf is `6.371` and its 71-80 band is `1232`. A `\d{1,3}(\.\d{3})*`
   pattern silently drops every large community and the total still looks plausible.

### What it is worth, and the one thing it cannot do

**The basis matches, which is the whole reason this is the cheap route.** Germany is `roll`
because the Melderegister is an institution's records (§2); ZWST is an institution's records
too. Nothing here mixes bases, no ESS is involved, and Germany's `measured` tier survives
intact — unlike §6, which spends it.

These people are already inside `unrecorded`. Jewish communities are public-law bodies in most
Länder, so the 87,934 sit in the 42.8M bucket now and drawing them **carves a counted number
out of it** rather than modelling a share of it.

**§3.6 is the binding limit and it bites harder here than usual.** A roll counts the
institution's location, not the member's, and these are regional catchments, not parishes:
Düsseldorf's 6,371 covers much of the lower Rhine. The grain claim has to be the community's
seat, and it should be said out loud. Two associations also run communities in the same city —
Hannover, Göttingen, Hameln, Wolfsburg, Kiel and Lübeck each appear twice — which is not
double counting and simply adds.

**And it is a floor, not a population.** 87,934 is affiliated membership; the unaffiliated and
much of the post-2022 Ukrainian arrival are outside it. That is the same shape of undercount the
register already has, so it is consistent rather than a new problem — but §3.5 says it gets
marked, not filled.

### The AGS join is hand-authored, and the fuzzy version would have been wrong

`SEATS` in `de_zwst.py` is 102 names written out by hand. **72 of the 105 match a Gemeinde
name exactly and one of those 72 is wrong**: `Weiden` is a village of 84 people in
Rheinland-Pfalz as well as a city of 42,047 in the Oberpfalz, and the exact match takes the
village. A community of 178 in a Gemeinde of 84 is the only sign, which is why
`MAX_SHARE_OF_SEAT` is asserted rather than eyeballed — the real worst case is Straubing at
1.74%, so the check has two orders of magnitude of room and still catches a mis-join.

The rest are the association's own house style rather than place names: abbreviations
(`Mönchengladb.`, `Herford-Detm.`), compounds (`Rheinpfalz/Speyer`, `Kiel u. Region`,
`Göttingen/Südn.`), and association-level rows whose seat is a fact about the body and not
about its name — `Württemberg` is Stuttgart, `Thüringen` is Erfurt, `IKG München` is Munich.
Writing the 102 out took one pass; **twelve of the codes were wrong on the first attempt and
the checker caught all twelve**, three of them by pointing at a real but different town
(Salzkotten for Paderborn, Bad Bramstedt for Bad Segeberg, Bad Münder for Bad Pyrmont).

Two associations run communities in the same city — Hannover, Göttingen, Hameln, Wolfsburg,
Kiel and Lübeck each appear twice, orthodox and liberal. Not double counting; both are real,
the seats coincide, and the counts add. 105 communities land on **96 Gemeinden**.

**One row is not a place.** `IRG Baden` 528 is the association's own directly-registered
members — people in Baden who belong to the Landesverband rather than to any of its ten local
communities. Putting all 528 in Karlsruhe would assert a concentration that does not exist, so
they are spread over the association's communities in proportion to them (largest-remainder,
so it sums back exactly). 0.6% of the layer, and it is the only modelled step in the file.

### The wiring, and the one line of it that matters

`countries.py::_de_add_zwst()` does the join, and **the whole of it is that the subtraction
comes first.** These people are inside `Sonstige` today, so a `judaism` row added without
taking them back out of `unrecorded` would give Germany 87,934 people it does not have.
Verified on the way through: the three Zensus categories plus judaism sum to 82,711,108,
which is what de.csv's three summed to before — residual zero.

```
  87,934 ZWST members moved from `unrecorded` to `judaism` in 96 Gemeinden
  32,276 (unit, node) rows, 4 nodes, 10,786 units
  82,709 dots, of which 87 are judaism
```

The taxonomy target lives in `taxonomy/de2022.py` rather than in a `de_zwst2025.py`, because
`taxonomy/registry.py` discovers one module per country and a second vintage for `de` would
need an `OVERRIDE` entry naming which one is drawn — when the honest answer is both, one per
source. `judaism` and not a movement under it: ZWST's members are largely Einheitsgemeinden,
single communities spanning orthodox and liberal on purpose, and the statistic names no
movement anywhere.

**Placement is the one place this is weaker than the rest of Germany, and it is labelled.**
destatis publishes no Jewish grid, so `_DeGridWeighter` places these dots on the `Sonstige`
column they were carved out of. That is a far better locator inside a city than total
population — `son` is where the register has no church for people, which is where a community
whose members are largely post-Soviet immigrants actually is — but it is a **proxy**, and
`de_grid.md` §2's claim that Germany's placement is *measured* stays true of the three Zensus
categories and is not true of this one. The run reports the two counts separately for that
reason.

### What is left

The year mismatch is worth a thought rather than a shrug: the register data is 2022 and this
is 2025, and ZWST lost about 1,300 members a year over that span. Subtracting a 2025 count out
of a 2022 bucket is off by roughly 3,000 nationally, which is 0.007% of the bucket — the 2022
edition is on the same page if exactness is ever preferred to currency.

Licence is unstated. ZWST is a private association and the PDF carries no reuse grant, so ask
before anything ships commercially.

**And the rest of §6 is untouched by this.** Germany still draws no Muslim, Orthodox or
free-church dots, which is most of the 42.8M bucket. This route worked because a second
*register* existed; ESS is a survey and would spend Germany's `measured` tier, which is the
call §6 leaves open. *(Made 2026-09-07 — see §9.)*

## 8. What was checked and ruled out, 2026-09-07

Written down so it is not re-derived. None of these is a verdict; each is a record of what
was tried (`spec.md` §12).

- **REMID and fowid publish national totals only.** They are the best index of *which* German
  bodies publish anything, and they carry no per-place figure, so nothing on them is drawable.
- **OSM has no Landeskirche boundaries.** Overpass returns 23 `boundary=religious_administration`
  relations for all of Germany and exactly **two** are Protestant — Baden, and a parish called
  Eisenberg. 19 are Catholic dioceses, which buys nothing because Catholics are already one
  node. So the Lutheran/Reformed/United split of the 19.1M EKD (argued in `taxonomy/de2022.py`)
  needs a hand-built Kreis→Landeskirche table, and the Landeskirche borders follow 1815–1866
  state lines that cut through modern Kreise. EKD publishes per-Gliedkirche membership that
  would check such a table twenty times over — `Kirchenmitgliederzahlen Stand 31.12.2024`,
  e.g. Bayern 2,025,552 and Anhalt 24,180.

  **NOT DOING IT — Anita's call, 2026-09-07. And the pitch above was overstated, which is
  the part worth keeping.** It was argued here as *reading a documented territorial fact
  rather than fitting a model*. That is true of eighteen Landeskirchen and **false of
  exactly the confession the split exists to show**, per EKD's own footnote 3:

  > *"Die Evangelisch-reformierte Kirche ist **keine Territorialkirche**. Sie befindet sich
  > schwerpunktmäßig auf dem Gebiet der Evangelisch-lutherischen Landeskirche Hannovers.
  > **Beiden Kirchen können Bevölkerungszahlen nicht direkt zugeordnet werden.**"*

  The Reformed church is a **diaspora, not a territory** — 151,083 members in 141
  congregations, concentrated in Ostfriesland and the Grafschaft Bentheim but scattered as
  far as Bavaria, and sitting *inside* Lutheran Hannover's borders. A territorial join
  cannot see it, and EKD says outright that population cannot be assigned to it. The
  *united* Landeskirchen are internally mixed too, by construction: the 1817 Prussian Union
  merged administration, and many Rhineland and Westphalian congregations stayed Lutheran or
  Reformed inside it.

  So the split would draw **which church body governs a place, not what confession its
  people hold**. For a `roll` basis that is a defensible claim — the register records
  membership of a body — but it is a weaker and different claim than "here is where
  Germany's Lutherans are", and the two were blurred when this was first written up.

  It is also worth less than it looked. Reformed is **~282,000 of 19.1M, about 1.5%**
  (ERK 151,083 + Lippe 130,705, and Lippe is itself majority Reformed with a Lutheran
  minority). What actually remains is Lutheran-versus-United — the Prussian Union line —
  which is one boundary rather than three colours, and does not justify a 400-row hand-built
  table. **`taxonomy/de2022.py`'s original refusal was better founded than this section
  first gave it credit for.**
- **Zensus 2011 at Gemeinde level exists and adds no categories.** destatis publishes
  `religionszugehoerigkeit_zenus2022_und_zensus2011_bundesland.xlsx` (1,459,220 bytes) which,
  despite the filename, carries **both years at all 10,786 Gemeinden on 2022 boundaries** — and
  the 2011 sheet has the same three columns. §2 was right from the literature; this is the file
  confirming it. What it is instead is a clean eleven-year change layer: 30.0 / 29.1 / 40.9 in
  2011 against 25.1 / 23.1 / 51.8 in 2022, municipality by municipality. Unused; the map draws
  one snapshot.

## 9. THE RESIDUAL IS SPLIT — BUILT 2026-09-07. `sources/de_ess.py`

Rebuild with `python sources/de_ess.py --fetch`. Writes `data/normalized/de_ess.csv`:
16 Bundesländer × 6 nodes, shares of the residual. **Germany draws nine nodes and is 90.3%
measured.**

```
  8,040,156 people (10.8% of Germany) split out of `unrecorded` into 6 religions
  islam 3,471,376 · christianity 1,505,755 · evangelical 1,174,182
  other.de 909,221 · orthodox 868,151 · protestant +110,000
  de: 82,707 dots over 9 nodes
```

### Five rounds, not three

§6 reported rounds 8, 9 and 11 usable. **Rounds 6 and 7 carry `rlgdnade` and `region` too**,
so the pool is **13,643 respondents rather than ~7,600**. That is the difference between
Hamburg resting on 268 people rather than 134, and it moves Bremen from unusable-and-drawn to
unusable-and-caught. Rounds 5 and 10 raise `E201VariableNotFound` — round 5 has no `rlgdnade`,
round 10 is the COVID round with a reduced German variable set.

### The two checks, and neither was spent

| | ESS | independent | |
|---|---|---|---|
| Catholic | 24.70% | 25.1% register | −0.40pp |
| Protestant (EKD) | 23.88% | 23.1% register | +0.78pp |
| **Jewish** | **0.103%** | **0.106% — ZWST's COUNT** | **held out** |

The first two are the licence: two independent instruments land within a point on the two
categories both can see, so the survey's account of the third is worth something. **The third
is the interesting one and it was free.** ZWST counts 87,934 Jews at community seats (§7);
ESS, which has 12 Jewish respondents in 13,643, puts them at 0.103% of Germany. Nothing was
fitted to it — it is the closest thing to a §14.10-condition-5 check this build has, and it
says ESS's small-category estimates are not collapsing.

### THE NON-RELIGIOUS ARE NOT DRAWN, and that is the design decision

France maps ESS's `Not applicable` — everyone who said they belong to no religion — straight
to `unaffiliated` (`fr2024.py`). **Germany does not**, and the two reasons are both specific
to Germany having a register:

1. **`unrecorded` is a MEASURED cell and France has no equivalent.** destatis genuinely
   counted 42.8M people into it. Replacing that with a modelled `unaffiliated` swaps a count
   for an estimate and calls the result more detailed. Carving the *religions* out and leaving
   the rest keeps the measured cell — smaller, still measured, still honest about being a
   register artefact rather than a statement about belief.
2. **§14.12 exactly.** *An ancestry-shaped cell models well and an attitude-shaped one badly.*
   Islam, Orthodoxy and the Freikirchen in Germany are close to functions of descent. Non-belief
   is a behaviour inside every group at once — and in Germany it is also the East/West divide,
   which no survey composition applied uniformly within a Land can see. Drawing the religions
   and not the irreligion is that rule applied rather than restated.

So ~19% of the residual becomes named religions and ~81% stays put. **Germany goes from 100%
measured to 90.3%**, not to 48%.

### What the toggle does, and a correction

**`modelled` dots are REMOVED by `inferred dots: not shown`, not rolled up.** §7a-i's roll-up
applies to `derived` only — `index.html`: *"`t` is 1 for derived and 2 for modelled, and only
1 rolls — modelled was never counted at any level and layerFilter removes it."* An earlier
reading of §7a-i in this file's §6 was right to worry and a later note claiming otherwise was
wrong. What makes it survivable is §9's own design: only 9.7% of Germany is modelled, so
turning inferred dots off returns the country to very nearly the map it is today rather than
emptying half of it.

`modelled` and not `derived` is §7b's test — *the tiers are about whether anybody was counted*
— and nobody counted Germany's Muslims at any level.

### Three sampling zeros became structural zeros, and that is the method's own shape

A Land with 250–450 pooled respondents will often have **none at all** in a category whose
true share is half a percent — and the split then asserts *nobody*, not *few*:

```
  christianity.evangelical   absent from Saarland
  christianity.orthodox      absent from Sachsen-Anhalt
  other.de                   absent from Mecklenburg-Vorpommern
```

Sachsen-Anhalt has Orthodox Christians; ESS drew none of them. `N_FLOOR` does not catch this
because it tests the Land's *total* sample, which is fine — it is the individual cell that is
empty. It is left as it is, because the alternative is smoothing a zero towards the national
figure, which invents a number for a place the survey says nothing about. Worth knowing before
reading a small category off any one eastern Land.

**The per-Land Muslim shares are also not uniformly undercounted.** Berlin 9.2%, Hamburg 8.4%,
Nordrhein-Westfalen 6.4% are close to independent estimates; **Bayern at 1.9% and
Baden-Württemberg at 4.4% are visibly low**, both being Länder with large Turkish-origin
populations. So the national undercount described below is not a constant that could be
divided out even if doing so were permitted.

### Placement, and the honest version of it

The six new nodes have no grid of their own, so they are placed on **`son`**, the `Sonstige`
column they were carved out of. Inside a city that is a far better locator than total
population — `son` is most of Neukölln and a tenth of a Bavarian village — but it is a proxy,
and the run counts it separately: **16,932 rows on the religion's own grid counts, 4,550 on
`son`, 2 on cell population.** Route B, the 1km citizenship grid, is the fix and is still
unbuilt.

### The Muslim undercount is stated, not corrected

4.13% pooled is ~3.5M against BAMF's ~5.5M. Four mechanisms push the same way and none is
corrected here:

- **ESS samples residents 15+** and Germany's Muslim population is markedly younger than
  average, so no 15+ survey can report the all-ages share the register is measured on;
- **interviewing is in German**, which sheds exactly the recent arrivals most likely to be
  Muslim;
- **the household frame misses collective accommodation**;
- **non-response correlates with migration background**, in the usual direction.

**Rescaling to BAMF is refused, and for two reasons rather than one.** It would fit the
coefficient to the only independent check the model has, which §14.10's second condition
forbids and §14.12 calls spending the check. And — the part that makes it actively wrong
rather than merely undisciplined — **the residual is a fixed total, so every Muslim added has
to be taken from the non-religious**, and there is no evidence for that transfer at all. The
undercount is in `note_public` in plain words instead, per §14.12's rule that a modelled
country names its weakest drawn cell rather than declaring itself modelled once at the top.

### What is left

Nothing in §9 itself. Route B is §10.

## 10. ROUTE B — the citizenship grid as a placement signal. BUILT 2026-09-07

`sources/de_grid.py` now reads a second destatis grid,
`Staatsangehoerigkeit_nach_ausgewaehlten_Laendern.zip` (27,225,007 bytes), whose 1km CSV
carries **the same `GITTER_ID_1km` cells as the religion grid**, so the join is on the cell
id and needs no geometry at all. `de_grid_1km.gpkg` gains two columns:

```
  isl_ctz   = Tuerkei + Bosn_u_Herzegowina                          1,498,990
  orth_ctz  = Griechenland + Rumaenien + Russ_Foederation + Ukraine 1,918,207
```

**Placement only, never a magnitude** (§8.2). Every Gemeinde's counts are unchanged; only
where inside it the dots land moves.

### Why `son` was not good enough, in one number

`Sonstige` is half of Germany and contains every secular German, so as a locator it barely
discriminates. In Berlin, **the top 10% of cells hold 35% of `son` and 63% of `isl_ctz`**.
Placing Muslim dots on `son` spread them across the whole city; the citizenship signal puts
them in Neukölln, Kreuzberg and Wedding, which is where they are.

### The blend ratio is computed, not chosen

Citizenship is **not** origin — most of Germany's Turkish-descended population holds German
passports and is invisible to this column — so placing every Muslim dot on it would assert
that naturalised families live exactly where non-naturalised ones do, and would put nobody in
the many cells where the community is entirely German-citizen.

So `_DeGridWeighter._sharpen` does `_ItWeighter`'s move (Italy, §8.4a) with the ratio taken
from the unit's own arithmetic: if the signal accounts for **C** people out of the node's
**N** in that Gemeinde, the citizenship vector gets `min(1, C/N)` of the weight and `son`
keeps the rest. **No constant is picked anywhere**, so there is nothing here to tune.

### Islam blends. Orthodox does not, and that is a finding rather than a bug

```
  islam                 1,414 rows at 39% citizenship, CAPPED in  6% of them
  christianity.orthodox   624 rows at 97% citizenship, CAPPED in 88% of them
```

For Islam the blend does real work. **For Orthodox it degenerates to pure citizenship**,
because the measured citizen count exceeds the modelled magnitude nearly everywhere —
1,918,207 against 868,151 nationally, and in **7,954 of 10,786 Gemeinden**.

That is worth reading the other way round. The citizenship grid is *measured*, so where it
exceeds a modelled count the model is provably low there, and this is **an independent
indication that ESS's 1.05% Orthodox is a worse undercount than its 4.13% Muslim** — one that
owes nothing to BAMF and so does not spend §9's check. For Islam the same comparison is
reassuring rather than damning: the measured count exceeds the model in only **366** Gemeinden
and by **56,548 people**, 1.6% of the modelled total.

Two reasons both contribute and neither is separable here: ESS undercounts Orthodox, and not
every Romanian, Russian or Ukrainian passport-holder is Orthodox. Scaling each column by its
country's published Orthodox share was considered and refused — it would still cap in ~85% of
Gemeinden, and adding coefficients to a signal used only for *relative* weight inside a unit
buys precision the output cannot express.

**The degeneracy is printed on every run rather than left in a file**, because a node placed
97% on citizenship is making a stronger claim about location than the 39% one beside it.

### What Route B still does not fix

§6's caveat 2, unchanged: the thirteen published countries include **no Syria, Afghanistan,
Iraq, Iran or Morocco** — the largest recent Muslim origins — so the signal is blind to them
and those dots keep the `son` placement by default. `Kasachstan` is deliberately unused:
Kazakh citizens in Germany are heavily Russlanddeutsche, so the column is neither a Muslim nor
an Orthodox signal, which is §14.12's ancestry warning in the one place here where the
ancestry is genuinely mixed.
