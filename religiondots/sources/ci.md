# Côte d'Ivoire — RGPH 2021, religion by région administrative

**Drawn 2026-09-07.** `sources/ci.py` -> `data/normalized/ci.csv`.
Boundaries and placement: `sources/ci_geo.md`. Mapping: `taxonomy/ci2021.py`.

| | |
|---|---|
| source | RGPH 2021, *Rapport thématique tome 1 — État et structure de la population* |
| publisher | Agence Nationale de la Statistique (ANStat) |
| route | the **Wayback Machine's** copy of ANStat's own URL — the live host is walled |
| drawn tier | **33 units** — 31 régions + the autonomous districts of Abidjan and Yamoussoukro |
| people | **29,276,660** in ordinary households; 28,630,628 drawn |
| coverage | **97.79%** (`ND` excluded per §3.5) |
| categories | **9 drawn** (8 published + the évangélique split, §10) |
| basis | self-identification |

## 1. The office was never the route, and the archive was

§11w recorded Côte d'Ivoire as *"the most valuable African lead still open"* and closed it on
the host. Both hosts are still shut:

* **`ins.ci` is a parked cPanel page.** The root serves a `defaultwebpage.cgi` redirect and
  **every document path 404s** — `/documents/rgph/ABIDJAN.pdf`,
  `/documents/RGPH2014_principaux_indicateurs.pdf`, all of it. The site is not deployed.
* **`anstat.ci` is Cloudflare, including its static assets.** `/assets/publications/files/*.pdf`
  returns 403 exactly as the app does, so §9aa's *"a 403 on a directory is not a 403 on its
  contents"* does **not** rescue this one. That rule has a limit and this is it.

**The Wayback Machine has the file.** §11f used the CDX API to read a *dead* host; here it
reads a *live but walled* one, which is the more generally useful case:

```
https://web.archive.org/web/<timestamp>id_/
  https://www.anstat.ci/assets/publications/files/rgpg_tom1.pdf
```

The CDX also enumerated what ANStat has ever served — 28 distinct downloadable files — which
is how the tome was found at all. `ins.ci`'s CDX separately turned up a complete set of
**per-région RGPH-2014 booklets**; they were downloaded and **contain no religion table**, so
that lead is closed rather than untried.

## 2. THE ARCHIVE TRUNCATES, AND THE OBVIOUS CAPTURE IS THE BROKEN ONE

The finding that generalises furthest here.

`ABIDJAN.pdf`'s first-listed capture delivers **exactly 1,048,576 bytes — 2^20 — with no
`%%EOF` trailer.** PyMuPDF opens it anyway and reports 18 pages. Nothing raises. A build that
took the first capture would parse a truncated document and find "no religion table on these
pages", which reads as a fact about the publisher.

The fix is to ask the CDX for **every** capture with its stored `length` and take the largest:

| file | first capture | largest capture |
|---|---:|---:|
| `ABIDJAN.pdf` | 911,111 (delivers 1 MiB, no trailer) | 1,756,591 (delivers 1,980,153, intact) |
| `PORO.pdf` | 894,398 | 1,510,365 |
| `rgpg_tom1.pdf` | — | 25,116,283 (delivers 34,204,439, intact) |

`ci.py`'s `fetch()` does that, and **raises on a body of exactly 2^20** and on a missing
trailer. memory/`reference_pdf_truncated_at_source` in a new disguise: there the damage was
the publisher's, here it is the archive's, and in both cases the length looks plausible.

## 3. The oracle was two censuses stale

UNSD table 28 (§11r) lists Côte d'Ivoire at **2014**, 22.7M people, 11 categories. What is
drawn is the **RGPH 2021** — 29.4M people. §9k's Chile finding repeated: *the lead was stale
in the useful direction*, and the oracle's own caveat about voluntary reporting cuts both
ways. **Check the office's current release before trusting the oracle's vintage.**

## 4. Three tables in one volume — §3.4's move

The religion table is percentages, so magnitudes have to come from somewhere else. Everything
needed is inside the same 151-page PDF.

| | table | page | content |
|---|---|---:|---|
| shares | **Tableau 4.6** | 87-88 | religion by district/région, **percentages to 1 dp** |
| magnitudes | **Tableau 4.1** | 81 | the same 9 categories nationally, **in counts** |
| denominators | **annex** | 132-133 | *Région administrative / Population totale*, in counts |

The build multiplies each région's percentage by its population, then **rescales each
category so the 33 units sum to its published national count**. That removes the one-decimal
rounding from every national figure and confines it to the within-country distribution, where
nothing can help it. The rescale factors are the honest report of how much rounding there was:

```
Catholique                    x0.99553      Musulmane            x0.99622
Méthodiste/Protestant         x0.99449      Animiste             x0.99718
Harriste                      x1.02511      Autres religions     x0.97954
Autres religions chrétiennes  x0.99584      Sans religion        x0.99676
                                            ND                   x0.99128
```

**The two extremes are the two smallest categories**, which is exactly where 1 dp hurts:
`Harriste` at 0.48% nationally needed +2.5% and `Autres religions` at 0.18% needed −2.0%.
(`Autres religions chrétiennes` is subsequently split in two — see §10 — so its factor above
is superseded by the two the split produces.)

## 5. Two nested tiers in one column, and a duplicated name

**Serbia's §9p in a third country.** Tableau 4.6's first column interleaves the **14
districts** with **their régions** and marks neither. Summing the column double-counts the
country.

**The separator is not the layout — it is the annex.** A row is drawable iff its name appears
in the population table, which lists régions and the two autonomous districts and no other
district. That lands on exactly 33, and it is self-checking in a way a hardcoded district
list is not.

**And `Lacs` is printed twice**, with different figures. The second is **Lagunes** —
identifiable because its children in the table are Agnéby-Tiassa, Grands-Ponts and La Mé,
which are Lagunes' three régions, while the real Lacs district's (Bélier, Iffou, Moronou,
N'Zi) sit under the first. Both are districts so neither is drawn and the collision costs
nothing here — but it would sink a build that took the district tier. `check()` asserts the
count is still two (§9p: a duplicated place name is a finding, not a typo).

## 6. THE SAME DOCUMENT USES TWO THOUSANDS SEPARATORS

**Tableau 4.1 groups its digits with U+2009 THIN SPACE** — `5 784 899` — while the annex on
p132 uses an ordinary U+0020. A regex written against one matches **nothing** on the other,
and the symptom is not an encoding error: it is an empty result that reads as *"that table is
not on this page"*. The first run of `ci.py` reported `9 missing` national categories and
`parsed national rows: []` for exactly this reason.

Every line is folded through a translation table of 13 Unicode space variants before any
pattern is tried. **Worth doing by default on any PDF parse** — the cost is one `str.translate`
and the failure it prevents is silent.

## 7. Two other tables share the annex pages

The annex parse initially returned 38 rows summing to 58.3 million, about twice the country.
Two other tables sit on those pages: an UPPERCASE one with a different universe (its own total
is 11,225,273), and an **urban/rural** one contributing `Abidjan ville`, `Autres villes`,
`Ensemble urbain` and `Rural` — each of which appears **twice with different numbers**.

A `{name: value}` dict silently keeps whichever came last. None of the four is a région, so
none reaches the drawn set — but that is luck rather than design, so the parse keeps
`{name: [values]}` and `check()` **asserts no drawn unit is ambiguous** rather than trusting
it. That check is the point; the filter is not.

## 8. The checks

`ci.py` prints all of these on every run.

1. **The volume is 151 pages** — a page-index build breaks loudly if the release changes.
2. **`Lacs` appears exactly twice.**
3. **33 of 46 Tableau 4.6 rows appear in the annex** — the district/région separator.
4. **No drawn unit is ambiguous in the annex** (§7 above), with the 3 aggregate names that
   *are* ambiguous printed for the record.
5. **Every row's 9 categories sum to its printed Total** within 0.45 pp — the bound is
   9 cells × 0.05, derived rather than fitted. Observed worst: 0.40.
6. **`Ensemble Chrétien` equals its four parts** on every row within 0.20 pp. An internal
   identity the table volunteers, and the best check here because the publisher did not have
   to print it.
7. **The 33 drawn units sum to 29,389,152**, the RGPH 2021 resident population.
8. **Tableau 4.1 gives a national count for all 9 categories**, and they sum to 29,276,659
   against the published ordinary-household figure of 29,276,660 — one person out.

## 9. What it shows

* **The Harrist Church, and nowhere else on this map.** 140,482 people. 2.4% in La Mé, 1.7%
  in Grands-Ponts, 1.6% in Agnéby-Tiassa; **0.0% in seven northern régions**. That is Harris's
  1913-15 itinerary, legible in a 2021 census. See `taxonomy/ci2021.py`.
* **A north/south line rather than a gradient.** Islam 95.7% in Folon and 92.3% in
  Kabadougou, against 13.9% in N'Zi. Christianity does the inverse.
* **Traditional religion in one place**: Bounkani 24.7% against 2.2% nationally.
* **`Sans religion` is 12.6% and inverted** — 29.8% in Tonkpi, 3.6% in Abidjan.

## 10. The évangélique split — Anita's call, 2026-09-07

Tome 1's `Autres religions chrétiennes` is **6,004,781 people, 20.51%** — the second largest
religious group in the country — and that volume never divides it. The **Résultats Globaux
Définitifs RGPH 2021** does, in one sentence of prose above its Tableau 6:

> *"Parmi les chrétiens, on compte 17% de catholiques, de protestants/méthodistes (2,3%), de
> harristes (0,5%) et de 20% d'autres chrétiens, **composés principalement des évangéliques
> (18,6%)**."*

Those percentages are of the **total population** — they reproduce that publication's own `%`
column exactly (Catholique 17.0, Protestante/Méthodiste 2.3, Harriste 0.5, Autres chrétiens
20.0). So **18.6% is 5,445,459 people**, and évangéliques are about 91% of the cell.

**What is established and what is not.** The national magnitude is the source's. The
*geography* is nobody's: no ANStat publication gives évangéliques by région. So the split is
applied at one national ratio to every unit, and both halves inherit the residual's shape
exactly. Evangelicals in Côte d'Ivoire are very likely more southern and more urban than a
flat 90.7% of every région's cell, so **the per-région shares are wrong in a spatially
correlated way while the national total is right.** Every row from the split is
`tier="derived"` in `countries.py` and can never ring (§3.10); `note_public` says the size is
measured and the pattern is not.

**How it is implemented.** `emit()` gives `Évangélique` and the remainder the *same*
per-région raw value and lets the existing per-category rescale divide them into their two
national totals. The uniform assumption is therefore a consequence of machinery that was
already there rather than special-case arithmetic — and it makes the two halves' geographies
identical by construction, which is what the source supports.

### 10a. The two publications disagree, and the split is arranged so it does not matter

| | `autres chrétiens` | `autres religions` | sum |
|---|---:|---:|---:|
| Tome 1, Tableau 4.1 | 6,004,781 | 53,051 | **6,057,832** |
| Résultats Globaux, Tableau 6 | 5,845,573 | 212,259 | **6,057,832** |

**159,208 people move across the Christian / non-Christian line between two ANStat
publications of the same census, and the sum is identical to the person.** Every other row in
the two tables matches exactly.

Tome 1 is used, because it is the one with the régional table. Subtracting the stated
évangélique count from tome 1's *larger* cell leaves the disputed 159,208 in the
**remainder** — which is where they belong if the Résultats Globaux is right that they are
not Christian. **So the évangélique figure is unaffected by the disagreement either way**,
and the uncertainty lands in the cell that is already a residual.

### 10b. What is still lost

After the split, **559,322 people (1.91%)** remain in `christianity.other`. The tome's
methodology names four modalities the census collected and no table prints — *Evangélique,
**Céleste**, **Bouddhiste**, **Témoin de Jehova*** — and with the évangéliques lifted out,
the other three are the bulk of what is left. **The Celestial Church of Christ is in there**,
and `bj2013.py` draws that same church by name in every one of Benin's 77 communes. Bouddhiste
is in there too, despite not being Christian under any reading.

### 10c. And the 1998 comparison, which is the best thing in the second publication

Tableau 6 carries both censuses in counts:

| | 1998 | 2021 |
|---|---:|---:|
| Autres chrétiens | 470,495 (3.1%) | 5,845,573 (20.0%) |
| Animiste | 1,827,675 (11.9%) | 629,938 (2.2%) |
| **Harriste** | **197,515 (1.3%)** | **140,482 (0.5%)** |
| Musulmans | 5,931,958 (38.6%) | 12,453,840 (42.5%) |
| Catholique | 2,976,023 (19.4%) | 4,984,388 (17.0%) |

**Other Christians went up twelvefold and animists collapsed**, which is one movement seen
from both ends. And **the Harrist church declined in absolute numbers** — 197,515 to 140,482,
down a quarter — while the country nearly doubled. That is the only body in the table that
shrank in people rather than share.

## 11. There is NO finer geography, and this is now settled

The Résultats Globaux Définitifs was checked end to end for the two things that would beat
tome 1's 33 régions. Its full contents:

* **Tableaux 1–19**, of which **only Tableau 6 concerns religion, and it is national.**
* **Annex tables 17–19** are population, households and age groups by district/région — no
  religion.
* **Cartes 1–3** are administrative regions, population density and regional weight — no
  religion.

So **33 régions is ANStat's published ceiling**, and Côte d'Ivoire's 113 départements and 510
sous-préfectures have no religion table in either publication. The remaining lead is the
dedicated RGPH-2021 website the `anstat.ci/publication` index advertises but does not link;
it is not `rgph.ci`, `rgph2021.ci` or `recensement.ci`, none of which resolves.

**The file is a 68-page SCAN with no text layer** — every page is a single full-page image,
so `page.get_text()` returns the empty string on all 68 and a text search reports zero hits
for `religion`. That is not evidence of absence; it is evidence of a scan. Render the pages
and read them, or OCR.

## 11. Access notes

* No key, no account, no wall on the archive. One CDX query plus one ~34 MB GET.
* `ins.ci` 404s everything; `anstat.ci` 403s everything including `/assets/*.pdf`.
* ANStat's publication index is `https://www.anstat.ci/publication`; individual items are
  `/publication-details/<64-hex-hash>`, which are not guessable. The archived index names a
  **"RESULTATS GLOBAUX DEFINITIFS RGPH 2021"** release and a dedicated RGPH-2021 website that
  is not `rgph.ci`, `rgph2021.ci` or `recensement.ci` — none of which resolves. **Both are
  open leads for a browser that Cloudflare lets through**, and either could carry a
  département-level table or a split of §10's cell.
