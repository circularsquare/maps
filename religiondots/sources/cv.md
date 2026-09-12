# Cabo Verde — RGPH-2021, the twenty-two municipal workbooks

Instituto Nacional de Estatística, Praia. Drawn: **14 nodes on 22 concelhos, 351,183 people,
99.63% of the census's own universe and 71.49% of the country**. `sources/cv.py`,
`sources/cv_geo.py`, `sources/cv_grid.py`, `taxonomy/cv2021.py`, `taxonomy/branches.py`
(`other.cv`).

The country is worth drawing for one cell. **`Racionalismo Cristão` is 6,129 people, 1.74%**
of everyone the census asked, the fourth largest named religion in Cabo Verde, and this is
the only country on earth where Christian Rationalism is a mass movement rather than a
handful of centres.

---

## 1. The queue priced it national-only, and the tier below was one API call away

`sources.md` §11w (2026-09-07) ranked Cabo Verde third of Africa's undrawn countries on the
oracle's category depth and closed the row as **"open office, not chased past the catch-all"**.
That was a statement about `tools/oracle.py`, whose rows are national and urban/rural, and
not about INE, which publishes religion for every one of the 22 concelhos.

**`ine.cv` is an Angular single-page app and every page it serves is the same 10 KB shell.**
`WebFetch` on `ine.cv/ine_censos_quadros_category/censo-2021/` returns the word "INE" and
nothing else, and every path on the host answers 200 with that shell, so a probe cannot tell
a real page from a 404. [[reference_spa_hidden_apis]]: the bundle names the API.

    GET https://ine.cv/main.<hash>.js                       1.4 MB
      -> apiUrl: "https://bdmi.ine.cv/site_deploy_api/api"

`bdmi.ine.cv` itself redirects to `Account/LogOn`, which reads as a wall and is not one: the
site API under it needs no key and no session. Four calls get the workbooks.

    /api/Census                              -> Censo 2010 (code 1), Censo 2021 (code 2)
    /api/Census/2                            -> the four content categories of Censo 2021:
                                                  3  Quadros por Concelho 2021      <- these
                                                  4  Publicações gerais 2021
                                                  5  Publicações Temáticas 2021
                                                  6  Agregados e População por Zonas e Lugares
    /api/Census/content/paginated/3          -> 23 workbooks: 22 concelhos + "Cabo Verde"
    /api/GenericFile/GetFile
        ?codePublication=<id>&publicationType=censo
                                             -> the .xlsx, under /site_deploy_api/Uploads/

`sources/cv.py`'s `fetch()` is those four calls. The whole set is 4.8 MB and takes seconds.

**The category listing is the thing worth carrying forward.** The description INE attaches to
its own `População e Censo` statistic says it outright — *"São volumes estatísticos sobre Cabo
Verde, **cada um dos municípios**, e Zonas e Lugares"* — and that sentence is in the API
response for the 2000 and 2010 censuses, three lines above a 2021 category that does the same
thing. Asking an office whether it publishes a volume **per unit of the tier below the one you
expect** is the highest-yield question in this file.

### What the other three categories are, so nobody re-checks them

* **Publicações gerais 2021** (24 items) is a *Cabo Verde em número* leaflet per concelho plus
  the preliminary results. Headline figures, no religion table.
* **Publicações Temáticas 2021** (11 items) is the analytical series: population structure,
  education, migration, fertility, mortality, disability, children, the elderly, youth and the
  labour market, housing, economic characteristics. **There is no sociocultural volume for
  2021**, which is where religion sat in the 2010 series (*Análise das características
  socioculturais da população*). So the workbooks are not a summary of a fuller publication;
  for 2021 they are the publication.
* **Agregados e População por Zonas e Lugares 2021** (9 items, one per inhabited island)
  reaches the *lugar*, which is finer than the concelho, and carries households and population
  and nothing else. That is the answer to "is there anything below the concelho": there is a
  geography and there is no religion on it. The 32 freguesias have COD-AB boundaries and no
  religion table either.

## 2. The universe is the population aged 15 and over, and the oracle's sixteenth row is the children

`tools/oracle.py "Cabo Verde"` prints **16 categories summing exactly to 491,233**, which is
the whole resident population, and the largest after Catholic is `Unknown` at **138,739**.
That is not a non-response. Tabela 1 of the same workbook prints the 0-4, 5-9 and 10-14 bands
as 45,540 + 46,619 + 46,580 = **138,739**, and 491,233 − 352,494 is the same figure to the
person. INE simply did not ask anyone under 15, and forwarded the residual to New York as
`Unknown`.

So the published list is **fifteen**, not sixteen, and `sources/cv.py` asserts all three
identities rather than describing them. The queue row's "16 categories" was one too many, and
the mistake is worth naming because it is the same shape in either direction: **a category
count off the oracle counts rows, and one of the rows may not be a category.**

The country's two holes are therefore different kinds in spec §10.4a's sense:

| | people | share of Cabo Verde | kind |
|---|---:|---:|---|
| under 15, never asked | 138,739 | 28.242% | 2, hand-written from Tabela 1 |
| asked, did not answer | 1,311 | 0.267% | 1, a column in `cv.csv` |
| **not drawn** | **140,050** | **28.510%** | `gap_share` |

`tools/gap_share.py` computes the kind-1 part as 0.37% *of the 15+ universe* and reports the
authored 28.51% as larger, which is the healthy direction.

## 3. The §3.5 lean check, and it is the age cut that matters

Both holes were tested against every drawn category's share, across the 22 concelhos, with a
leave-one-out on each (22 units is few enough that one outlier can manufacture a lean).

**Hole A, the 1,311 non-responses**, leans towards the less Catholic end: r = **+0.49** with
`Sem religião` and **+0.59** with `Racionalismo Cristão`, **−0.29** with Catholic. Both
survive dropping the most influential unit (+0.39 to +0.54, and +0.53 to +0.73). The
direction is real and the magnitude is not: 1,311 people is 0.37% of the table, so excluding
them moves no share by a tenth of a point.

**Hole B, the 138,739 children, is 28% of the country and its direction is measurable**,
because INE publishes the age bands per concelho in the same workbooks. The child share runs
from **33.31% in Santa Cruz**, 32.45% in Santa Catarina do Fogo and 31.31% in Santa Catarina
de Santiago down to **22.29% in Paul**, 23.26% in Ribeira Brava and 23.36% in São Vicente.
Against the drawn shares:

| category | r | leave-one-out range |
|---|---:|---|
| Racionalismo Cristão | **−0.62** | −0.69 to −0.55 |
| Sem religião | −0.41 | −0.48 to −0.27 |
| Universal do Reino de Deus | −0.27 | −0.37 to −0.20 |
| Católica | +0.11 | −0.03 to +0.27 |
| Jesus Cristo dos Santos dos Últimos Dias | +0.33 | +0.16 to +0.40 |
| Nova Apastólica | +0.34 | +0.19 to +0.39 |
| Adventista | **+0.51** | +0.44 to +0.54 |

So the concelhos that lose most people to the age cut are the young rural ones on Santiago
and Fogo, and the one that loses least is São Vicente. **An adults-only map is therefore
tilted slightly towards São Vicente's secular and Racionalismo Cristão shares and away from
rural Santiago's Catholics and Fogo's Adventists.** The two strongest results survive
leave-one-out comfortably, so this is not São Vicente on its own. Nothing corrects for it;
`note_public` says which way it leans, per spec §3.5.

**One caveat that belongs with the number.** This correlates a concelho's *child* share with
its *adult* religion composition. It says which places the age cut removes people from. It
does not say what those children would have answered, and nothing published does.

## 4. The join, and the three pairs of concelhos that share a name

    Ribeira Grande  (Santo Antão)   vs  Ribeira Grande de Santiago
    Santa Catarina  (Santiago)      vs  Santa Catarina do Fogo
    Tarrafal        (Santiago)      vs  Tarrafal de São Nicolau

COD-AB prints the short name for the first of each pair. **So does the table title inside each
workbook** — *Tabela 1 - População residente no concelho de Ribeira Grande*, with nothing to
say which island. A name join built on the sheet would pair all three with a coin flip, and
[[reference_name_join_wrong_neighbour]] is that the country still sums to 491,233 either way:
no total, no category and no row count can see it.

The pairing lives in `sources/cv.py`'s `PCODE`, keyed on INE's own content id from the API
listing, whose titles *are* unambiguous (*Ribeira Grande de Santo Antão*, *Santa Catarina de
Santiago*, *Tarrafal de Santiago*). `sources/cv_geo.py` proves it three ways:

1. **19 of 22 names fold to a COD name exactly**, and the 3 that do not are precisely the
   ambiguous ones. Asserted, so a COD rename stops the file rather than moving the ambiguity.
2. **Populations rank together**, Spearman **0.9650** over 22 units against COD-PS 2022.
3. **Each ambiguous swap is tested and must fail.** Ribeira Grande's true pairing is −1.5%
   and its swap is +75% and +92%; Santa Catarina's swap is +635% and +913%; Tarrafal's is
   +227% and +239%.

**COD-PS 2022 is 16% above the 2021 census nationally** (569,523 against 491,233) and Boa
Vista is +72%, because it projects the 2010 census forward and the tourist islands outran it.
It is used for rank only. The magnitude check is `sources/cv_grid.py`'s: Kontur 2023 against
the census over 22 units gives **r = 0.9638**, and none of 2,000 random pairings reaches
0.67.

## 5. Placement, and why the strays are snapped

`sources/cv_grid.py` writes 2,156 Kontur 400 m hexes. **123 of them, 23,699 people, have a
centroid just outside every concelho polygon**, all within 700 m, and all are snapped to the
nearest rather than dropped: [[reference_archipelago_grid_snap]]. Cabo Verde is ten islands
and all of it is coast, so the loss is directional; dropping those cells walks every island's
dots inland, which here means uphill onto ground nobody lives on. Nothing remains unplaced.

The units need the weighting for the same reason. Santa Catarina do Fogo *is* the caldera of
Pico do Fogo; Porto Novo is 558 km² of Santo Antão with its people in the port and one
ribeira; Boa Vista and Maio are dune.

## 6. The mapping, and the two calls worth reading

Full reasoning in `taxonomy/cv2021.py`'s `REVIEW`, which has nine entries. The two that
could have gone the other way:

**`Igreja do Nazareno / Protestante` → `christianity.holiness.nazarene`.** The slash could
make this a mixed cell belonging on `christianity.protestant`. Three things say it is a gloss:
every other slash in INE's list joins two names for one body (`Islâmica / Muçulmano`,
`... dos Últimos Dias / Mórmons`); **INE's own English transcription of this table, forwarded
to the UN Demographic Yearbook, calls the row `Church of Nazarene` and nothing else**; and the
cell's geography is the mission's history, 8.10% in Brava and 3.90% in Mosteiros against 0.07%
in São Lourenço dos Órgãos. The Nazarene mission reached Brava and Fogo in 1901 with emigrants
returning from New England, and *protestante* is what Cape Verdeans call that church.

**`Racionalismo Cristão` → `spiritualism`, not a node of its own.** Christian Rationalism was
founded in Santos in 1910 by Luís de Mattos out of a Kardecist group and then explicitly
against Kardec on mediumship, so it is a sibling of `spiritualism.kardecist` rather than a
child of it, in the same relation `br2010.py` uses for Brazil's `Espírita` and
`Espiritualista`. A `spiritualism.rationalism` node was considered and not taken: it would be
a legend row no other country uses, which `AGENT_BRIEF` §3 says goes to Anita, and
`spiritualism` is already drawn for Antigua, Australia, Brazil and Canada, so nothing is
hidden. **If Anita would rather see it named, this is a one-line change** and the REVIEW entry
says so.

Its geography is the movement's own account of itself. 3,988 of the 6,129 are in São Vicente
(**6.86%** of that concelho's adults), then Tarrafal de São Nicolau 4.41%, Boa Vista 3.54%,
Paul 3.17% and Sal 2.38%; it draws nobody at all in São Salvador do Mundo or Santa Catarina do
Fogo. São Vicente, São Nicolau, Boa Vista, Santo Antão and Sal is the island list the centres'
own histories give, and Santiago is absent from both.

## 7. What the map shows, in one paragraph

Catholicism is 72.49% and the concelhos run from **97.53% in São Salvador do Mundo** to
**42.98% in Santa Catarina do Fogo**. The bottom of that range is two unrelated things. São
Vicente, the northern port island, is 46.36% Catholic and **38.20% no religion**, against
0.77% in São Salvador do Mundo — a fiftyfold spread on `unaffiliated` inside half a million
people. Santa Catarina do Fogo, 3,204 adults in the volcano's caldera, is 42.98% Catholic
because the missions got there instead: 13.42% Adventist, 11.52% New Apostolic and 7.65%
Latter-day Saint, each the highest figure in the country. Islam is 1.31% and is Boa Vista
6.63% and Sal 4.37%, the two resort islands, then Praia 1.88% — West African labour migration
of the last twenty years, and 3,668 men to 948 women nationally. And **23 people** answered
Jewish, across seven concelhos (Praia 7, São Vicente 6, Sal 4, Boa Vista 2, Brava 2, Maio 1,
Santa Catarina de Santiago 1), the smallest answer this census printed. It is not the
smallest on the map — Czechia, the Philippines and Poland each print a body with **one**
adherent — and a cross-country superlative was written here and withdrawn on checking, which
is the class of claim `AGENT_BRIEF` says keeps failing.

## 8. Files

    data/raw/cv/119.xlsx … 148.xlsx     the 22 concelho workbooks, ~200 KB each
    data/raw/cv/137.xlsx                "Cabo Verde - CORRIGIDO", the national one
    data/raw/cv/_index.json             content id -> title, filename, URL
    data/raw/cv/cpv_admin_boundaries.shp.zip     COD-AB, HDX
    data/raw/cv/cpv_admpop_adm1_2022B.csv        COD-PS, HDX
    data/raw/cv/kontur_population_CV_20231101.gpkg
    data/normalized/cv.csv              330 rows, 22 concelhos x 15 categories
    data/geo/cv/cv_concelhos.gpkg       22 polygons, with island and census population
    data/geo/cv/cv_lookup.csv           the same, without geometry
    data/geo/cv/cv_hexes.gpkg           2,156 Kontur hexes

---

## 9. Review, 2026-09-08 — read back off the workbooks, not off this file

A second pass re-parsed all 22 workbooks independently of `sources/cv.py` and reproduced
every figure in §2 through §7, in `countries.py`'s `note_public`, and in `taxonomy/cv2021.py`'s
REVIEW entries. Nothing needed changing. Four things are worth adding.

### 9.1 The age cut is proved per concelho, not only nationally

§2 closes 138,739 two ways on the national workbook. The stronger fact is that
`sources/cv.py`'s `check()` closes it **on every one of the 22 units separately**: each
concelho's own Tabela 1 population minus its own 0-4, 5-9 and 10-14 bands equals the total
its own religion table prints. Twenty-two independent identities, all exact. A residual that
was partly non-response would not do that in twenty-two places at once, so `Unknown` is the
under-15 population and `gap`, `gap_share`, `basis` and `note_public` all describe it
correctly as an age cut.

### 9.2 How large the lean actually is, in points

§3 gives the direction and the leave-one-out ranges, which all reproduce to the second
decimal: Racionalismo Cristão −0.621 (LOO −0.691 to −0.552), Adventista +0.506 (+0.435 to
+0.538), Sem religião −0.413, Universal −0.274, Nova Apastólica +0.342, LDS +0.330, Católica
+0.108 (−0.031 to +0.266). Católica's range does cross zero, so it fails leave-one-out on
its own terms and §3 is right to keep it out of the two it calls strong.

What §3 does not give is a magnitude, and it is small. Reweighting each concelho by its
**whole** population instead of its 15+ population — that is, assuming a concelho's children
would answer like its adults, which is the only part of the bias anything published can
address — moves no national share by more than a third of a point:

| | drawn | reweighted | shift |
|---|---:|---:|---:|
| Católica | 72.757% | 73.095% | **+0.34** |
| Sem religião | 15.608% | 15.285% | **−0.32** |
| Racionalismo Cristão | 1.745% | 1.675% | −0.07 |
| Adventista | 1.887% | 1.924% | +0.04 |

Every other category moves by less than 0.02. So `note_public`'s *"tilts a little"* is
right, and the direction it names is right including for Catholics, whose per-unit
correlation is the weak one: the geographic reweighting understates them by the largest
margin of any category, because the high-child concelhos are Santa Cruz (85.8% Catholic),
Santa Catarina de Santiago (89.8%), São Domingos (94.1%) and São Miguel (92.6%). The
correlation is noisy only because São Salvador do Mundo and São Lourenço dos Órgãos, the two
most Catholic concelhos of all, sit at the middle of the child-share range.

### 9.3 The swap assertion fires, and one of its three arms gives the wrong diagnosis

`sources/cv_geo.py`'s `check_swaps` was tested by handing it a **swapped census** — which is
what a wrong `PCODE` produces, and what witness 1 cannot see, since the names stay put. All
three swaps raise, so the guard is not vacuous. Witness 2 catches all three as well
(Spearman falls 0.9650 to 0.9424, 0.6002 and 0.7595 against a 0.95 floor), so the join is
guarded twice over.

**But the Ribeira Grande arm raises the wrong error.** A CV09/CV10 swap leaves the two
`TRUE_BAND` residuals at 0.65 and 0.56, inside the 0.80 band, so that check does not fire;
what fires is `SWAP_FLOOR`, whose message reads *"swapping them is NOT distinguishable — this
witness has stopped working and the join needs another one. Do not delete it."* That sends a
future session hunting for a new witness when the actual fault is `PCODE`. The other two
pairs raise the correct *"check PCODE"* message, because their swapped residuals (2.32/1.99
and 1.22/1.18) clear the band easily. Left as found; the guard works, only its explanation of
itself is wrong for one of the three pairs, and the fix is a wording or an ordering change
somebody should make deliberately rather than a reviewer in passing.

### 9.4 `Racionalismo Cristão` on `spiritualism` is right, and nothing else moves

Agreed with §6, and the precedent is closer than §6 claims. `br2010.py` maps `Espírita` to
`spiritualism.kardecist` and `Espiritualista` — IBGE's own catch-all for spiritualists who
are not Kardecists, 61,736 people — to `spiritualism`. Christian Rationalism is inside that
Brazilian cell, so filing Cabo Verde's on `spiritualism` puts it exactly where Brazil's are
already. No drawn country becomes inconsistent.

One wording to be aware of if this is ever revisited: `ph2020.py`'s REVIEW calls
`spiritualism` *"the Anglo-American node"*, as its reason for putting the Union Espiritista
Cristiana on the Kardecist child. That description is not what the corpus does — `br2010.py`
and `mx2020.py` both put a Latin American cell on the parent — but it is the one sentence a
future reader could hold against this call. Separately, `au2021.py` and `nz2023.py` map a
category called `Rationalism` to `secular`; that is Anglophone freethought and a different
body, not an inconsistency.

### 9.5 One correction to §7

§7's *"Czechia, the Philippines and Poland each print a body with **one** adherent"* is right
about Poland and wrong about the other two. Smallest national figure in each normalised file:
Poland 1 (`Polski Kościół Dialogu`), Australia 3 (`Pentecostal City Life Church`), New
Zealand 3 (`Commonwealth Covenant Church`), Czechia 6 (`Společenství buddhismu v České
republice`), Paraguay 7, Armenia 9 — and the Philippines 1,074, which is nowhere near it. The
**withdrawal was right**: 23 is not the smallest body on the map, four countries print
smaller. Only the list of examples is wrong, and `note_public` does not carry it — it says
*"the smallest answer the Cape Verdean census printed"*, which is correct and correctly
scoped.

### 9.6 The rest of the pass

`check_md.py` clean, `built_countries.py --check` clean, `check_rollup.py cv` shows 351,183
measured and nothing derived or orphaned, `gap_share.py --check` reports the authored 28.51%
as larger than the 0.37% it can prove from rows, which is the healthy direction. Screenshot
at the country's own view: dots on all ten islands, none in the sea, concentrated on Praia,
Mindelo and the settled coasts, fourteen legend rows, and the `not drawn` panel reading as an
age cut. Nothing looked off.
