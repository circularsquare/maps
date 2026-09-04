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
