# Bosnia and Herzegovina — BHAS, Census 2013

`sources/ba.py` -> `data/normalized/ba.csv`. Boundaries and placement: `sources/ba_geo.md`.
Taxonomy: `taxonomy/ba2013.py`.

**3,531,159 people on 142 municipalities, eight categories, 98.9% drawn.** One PDF, one
GET, an exact partition in both directions, and no reconciliation left over. The whole cost
of this country was the parse and the boundary repair; the acquisition was a guessed URL.

---

## 1. The route, and what §11c got wrong about it

`sources.md` §11c filed Bosnia as *"reachable, not quick"*:

> `popis.gov.ba` is a React SPA with no data endpoint found; the 2013 final results are
> **12–19 MB PDF books** at `popis.gov.ba/popis2013/doc/`. Not in GISCO, so the boundaries
> are also unsolved.

**Every clause of that is true and none of it was the obstacle.** What it missed is that
the two facts explain each other. Running §12's grep-the-bundle rule here returns **zero
`/api` routes across all 125 KB** of `1.52b2fd01.chunk.js` and `main.alen.chunk.js` — and
that is the *correct* result rather than a failed search, because there is nothing behind
the SPA to serve. The books are static files. The SPA is a menu.

**The one thing that looked like a wall is a statement about a directory.**
`https://popis.gov.ba/popis2013/doc/` returns **403**, which reads as "this material is
protected". `https://popis.gov.ba/popis2013/doc/RezultatiPopisa_BS.pdf` returns **200 and
12,462,115 bytes**. Directory listing is off; the files are open. A 403 on a directory is
not a 403 on its contents, and this is the general lesson worth carrying out of the
country — *test a file path before believing a directory's status code*, because index
suppression is the default on a great many servers and looks identical to access control.

Two books are live at that prefix:

| file | bytes | what it is |
|---|---|---|
| `RezultatiPopisa_BS.pdf` | 12,462,115 | **Final Results, June 2016** — 268 pages, the tables. This is the one used. |
| `Popis2013prvoIzdanje.pdf` | 19,679,892 | *Prvo izdanje*, the preliminary first edition. Superseded. |

## 2. The table

**Table 4.3, `Stanovništvo prema izjašnjavanju o vjeroispovijesti i spolu, nivo općine u
BiH`** — *Population by religion and sex, Municipality Level in BiH*. Printed pages 70–81,
which are PDF pages 72–83. **Twelve pages.**

§11b's rule is that the real predictor of cost is *how many pages the religion table
occupies*, not what platform serves it. Twelve pages of clean two-column bilingual table
parses in an afternoon, and did.

The book publishes the same variable at three levels — 4.1 entity, 4.2 canton, 4.3
municipality — and every other variable the same way. Only 4.3 is read.

Eight categories plus the unit's own total:

| Bosnian | English | national | share |
|---|---|---|---|
| `Islamska` | Islamic | 1,790,454 | 50.70% |
| `Pravoslavna` | Orthodox | 1,085,760 | 30.75% |
| `Katolička` | Catholic | 536,333 | 15.19% |
| `Ostali` | Other | 40,655 | 1.15% |
| `Nisu se izjasnili` | Did not declare | 32,700 | 0.93% |
| `Ateist` | Atheist | 27,853 | 0.79% |
| `Agnostik` | Agnostic | 10,816 | 0.31% |
| `Bez odgovora` | No answer | 6,588 | 0.19% |

## 3. The reconciliation, which is as clean as this project has had

Both directions, exactly, with `check()` asserting rather than reporting:

* **Every municipality's eight categories sum to its own published total.** 142 of 142,
  discrepancy zero.
* **The 142 municipalities sum to BHAS's own national row, category by category.** Eight of
  eight, discrepancy zero.

No suppression at any level, no rounding, no proration. BHAS publishes single people —
Berkovići has one person in `Nisu se izjasnili` and three in `Ostali`. Contrast Guyana
(§9r), where the office prorated its non-response away and only a footnote said so.

## 4. Two parse traps, and the second one passes every per-row check

**A municipality name can wrap onto a second line, and the wrap is marked only by a
trailing space.** `BOSANSKA ` / `KRUPA`, `KUPRES - F ` / `BiH`. A parser reading one line
as one name gets `KRUPA`, which then fails to join to any polygon and presents as a
*boundary* problem — sending you to geoBoundaries to look for a municipality that was never
missing. Names are accumulated until the `Uk./Tot.` marker instead.

**Tables 4.1 and 4.2 are the same shape as 4.3 and sit immediately before it.** Identical
headers, identical `Uk./Tot.` / `M/M` / `Ž/F` three-row structure per unit. A generous page
window picks up all three, yielding 156 "units" that sum to **5,583,946 against a country
of 3,531,159** — and **it still passes a per-row partition check**, because an entity row
and a canton row are each internally consistent. Only the national total catches it.

That is the finding worth generalising: **a check that only looks at rows cannot detect a
table that contains the right rows at the wrong level.** The page window is pinned in
`ba.py` rather than searched for, the printed caption of the first page is asserted, and
the national total is asserted against BHAS's own.

**The sex rows are dropped explicitly.** Each unit publishes `Uk./Tot.`, `M/M` and `Ž/F`;
summing all three doubles the country. They are skipped by name, not by falling off the end
of a pattern.

## 5. THE VINTAGE, AND THE TEST IT HAS TO PASS

**2013 is the most recent census. BiH has run none since, and the previous one was 1991.**
That gap is the country's politics rather than its statistics: a census here allocates
power between three constituent peoples, and running one is contentious enough that
twenty-two years passed between the last two.

So this is a twelve-year-old count, and §11j set the test for exactly this case when it
declined to draw the Central African Republic:

> **The test is not how old the data is. It is whether the drawn categories were themselves
> the target of something that moved them since.**

**Bosnia passes it, and the reason is a matter of sequence.** The violence that rearranged
Bosnian religious geography — the ethnic cleansing of 1992–95 — happened *before* this
census, not after it. The 2013 count is a record of the post-war distribution, not a
pre-war one that no longer exists. CAR's file fails because its 2003 enumeration predates
the 2013 anti-balaka campaigns; Bosnia's is the same situation with the order reversed,
which is the whole difference.

What twelve years does cost is **magnitude, not shape**. BiH has emigrated heavily since
2013 — `ba_geo.py`'s independent check measures Kontur's 2023 surface at a **median 0.94×**
the 2013 census count, and that shortfall is real rather than an artefact. Read the country
as Ethiopia's is read (§9u): the shape of Bosnian religion, not its current size.

## 6. The dispute over the results, which is not about religion

Republika Srpska's statistical institute (RZS) **rejected BHAS's 2013 results** and
publishes its own lower figures for the entity. The disagreement is about **residence** —
specifically how to treat people enumerated in BiH who have lived abroad for years — and
RS's position is that BHAS counted too many of them, which inflates the Bosniak return in
particular.

Two reasons this does not change what is drawn:

1. **It is a dispute about the denominator, not about the religion question.** Nobody
   disputes the religious composition of the people who were counted.
2. **BHAS's figures are the ones the state, Eurostat and the EU use**, and there is no
   competing municipality-level religion table to prefer instead.

Said in `note_public` rather than resolved, per §3.5's discipline of marking rather than
filling.

## 7. What the source is worth

**It is the only place on this map where Islam, Orthodoxy and Catholicism meet at
municipality resolution in one country**, and 50.7% Muslim makes it the second most Muslim
country in Europe here after Kosovo.

What the map shows, and none of it is visible in the national numbers:

* **Almost complete separation.** 84 of 142 municipalities are over 90% one religion.
  Bužim 99.7% Muslim, Posušje 99.8% Catholic, Ribnik 99.5% Orthodox. This is what the war
  produced, and the census is the measurement of it.
* **Srebrenica returns 55% Muslim and 45% Orthodox**, against roughly 73% Muslim in 1991.
* **Brčko is the only genuinely three-way unit in the country** — 44/35/21 Muslim /
  Orthodox / Catholic — and is the district placed under international arbitration
  precisely because it could not be assigned to either entity. The map makes the reason
  legible.
* **The central Bosnian valley towns are the other exception**: Vitez, Busovača, Kiseljak,
  Novi Travnik, Jajce and Žepče all sit near half Muslim and half Catholic, which is the
  1993 Croat–Bosniak war's front line still visible thirty years later.
* **Mostar splits 50% Catholic / 46% Muslim** across the Neretva.
* **Irreligion is Sarajevo and is tiny** — 1.10% nationally, the lowest of any European
  country drawn here, against 8.60% in Centar Sarajevo.

## 8. What it cannot show

* **No denominational depth at all.** Three religious boxes for a country whose Catholics
  are pastored by a Franciscan province with a 700-year continuity and whose Muslims have
  a Sufi tekke tradition. `taxonomy/ba2013.py` argues this is a fact about Bosnia rather
  than a defect in the form — religion and nationality are near-substitutes here, and a
  longer list would have been answered as though it were short.
* **No 'no religion' box.** The only irreligious answers are the *positions* `Ateist` and
  `Agnostik`, so the 1.10% is a floor rather than a measurement and is not comparable with
  the no-religion shares elsewhere on this map.
* **`Ostali` has a sharp peak nobody can explain from this source.** Velika Kladuša is
  **7.42%** `Ostali` — 2,998 people, six times the national rate, in a 96%-Muslim
  municipality — with Tuzla second at 3.51% and nothing else above 3.3%. By §9r's rule a
  residual with a sharp geography is a missing category rather than a mixture, but BHAS
  publishes no composition of the cell at any geography, so it is named as an open question
  in `branches.py` and not guessed at (§14.4).
* **`Bez odgovora` has one absurd unit.** Fojnica is 6.24% — 771 of 12,356 — against 0.19%
  nationally and 2.69% in the next highest. That is 33× the national rate in one
  municipality and is an enumeration artefact, of the kind §3.8 predicts and no note
  repairs. Not drawn either way.

## 9. What is left

* **The 2013 book has ethnicity (table 3.3) and mother tongue (5.3) at the same geography**,
  in the same file, already downloaded. Not needed to draw the country and worth knowing it
  is there — it would let the religion/nationality substitution be measured rather than
  asserted.
* **No newer census exists** and none is scheduled. Nothing to re-base onto.
