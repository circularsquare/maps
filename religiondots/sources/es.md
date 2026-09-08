# Spain — `sources/es.py`, `sources/es_geo.py`, `taxonomy/es2026.py`, `taxonomy/es_origin.py`

Drawn 2026-09-06. **52 provinces, 41 nodes, 48,984,549 people, 98.36% of the country.**
No census has ever asked, and this is still one of the better-measured countries on the map.

| | |
|---|---|
| counting geography | **provincia, 52 units, 940,000 people each** — finer per person than Kenya's counties (§9o) and half Russia's federal subjects |
| placement | 8,131 municipios, GISCO LAU 2021, weighted by municipal population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents |
| tier | **`modelled` throughout.** Neither half is a count of anybody |
| vintage | CIS 2023–26 pooled, INE population 1 July 2026, INE nationality mix 2022, Pew 2020, UCIDE 2023 |

---

## 1. The finding the whole country rests on: CIS does not sample foreigners

The CIS *barómetro* asks `RELIGION` every month, publishes the microdata free within 15 days,
and carries `PROV`. Pooling three years gives **464,524 respondents across all 52 provinces**,
about 8,900 each — the largest survey behind any country on this map by a wide margin.

The trap is the universe, and it is one variable away from invisible. `NACIONALIDAD` in a
barómetro takes exactly two values:

    1  La nacionalidad española          3,863 of 4,024 in estudio 3567
    2  La nacionalidad española y otra     161

**There is no third value.** Spain's 7.4 million foreign nationals — 15% of the population —
are not undersampled, they are outside the sampling frame. Everything CIS reports is a share
of Spanish citizens.

That single fact reconciles two numbers that look irreconcilable and are quoted against each
other constantly:

| | |
|---|---|
| CIS `creyente de otra religión` | 3.2–4.6% — **of citizens**, so 1.9M people |
| UCIDE's Muslim estimate | ~5% **of residents**, so 2.5M people |
| UCIDE's Spanish-CITIZEN Muslims | **1,085,593 — which fits inside CIS's cell** |

So Spain is drawn as two populations that partition the country exactly, neither a subset of
the other, and the sum is INE's own figure. **The general lesson, which will recur: before
believing a survey's minority cell, read its universe definition, not its response rate.**

---

## 2. CIS reuses the variable name for a different question, and nothing warns you

**This is the one that would have shipped a wrong map.** Of the 170 studies downloaded, 122
carry a `RELIGION` variable. Two of them — **estudios 3462 and 3506** — carry a *different
question under the same name*:

| code | the question this project maps | the other one |
|---|---|---|
| 1 | Católico/a practicante | **Sin religión/no profesa ninguna religión** |
| 2 | Católico/a no practicante | Católico/a |
| 3 | Creyente de otra religión | Creyente de otra religión |

Code 1 means *practising Catholic* in one and *no religion* in the other. Pooling them moves
several hundred thousand irreligious Spaniards into the Catholic column, and **every check
that a normal ingest runs still passes**: the totals sum, all 52 provinces are present, the
row counts are right, and the national share drifts by about a point — which looks like
sampling noise. Nothing is malformed.

`taxonomy/es2026.py` therefore checks the **whole value-label signature** rather than the
codes or a label prefix. It has to be the whole signature, because a first-word check passes
on `Creyente de otra religión` for the wrong study too.

Three more variants exist and are cosmetic, which is why the check has to distinguish them
from the real one:

- code 4 is `Agnóstico/a` in 11 studies and `Agnóstico/a (no niegan la existencia de Dios
  pero tampoco la descartan)` in 111;
- code 6 likewise gains and loses `(niegan la existencia de Dios)`;
- **estudio 3434 alone** shortens code 3 to `De otra religión` **and** numbers N.C. as **7**
  instead of 9.

**What generalises: a source that reuses a variable NAME across incompatible questions cannot
be caught by validating the data, only by validating the metadata.** Version the whole
codelist, not the codes.

---

## 3. The site is walled and the data is not

`cis.es` is behind BunkerWeb. A plain fetch of the catalogue, of any study page, or of the
barómetro methodology page returns a **15,054-byte "Bot Detection" challenge** with a 30-second
meta-refresh — from `curl` with a browser user-agent, and from the WebFetch service too.

The document paths are wide open. Microdata is at:

    https://www.cis.es/documents/d/guest/MD<estudio>-zip     newer studies
    https://www.cis.es/documents/d/guest/MD<estudio>         older studies

**Two naming conventions, both live, and no way to tell which from outside** — `MD3567` is
410 Gone and `MD3567-zip` is 200; `MD3435-zip` is 410 and `MD3435` is 200. `sources/es.py`
tries both. The `-zip` form is Liferay rendering the filename `MD3567.zip` with the dot as a
dash, which is the same trick behind the report PDFs (`es3567mar_a-pdf`).

This is **§9s's KOSIS wall inverted**: there the metadata was open and the data walled. The
generalisation is the same either way — *probe the document paths separately from the pages
that link to them.*

Two file layouts inside the zip, also both live: older studies carry `<n>_num.csv` and
`<n>_etiq.csv` at the top level, newer ones nest both inside `<n>_csv.zip`. Every study also
carries a `.sav`, which is not read — there is no `pyreadstat` wheel for this interpreter and
the CSV is the same matrix.

**The R package `opencis` is what cracked the URL scheme**, indirectly: it scrapes the study
page for a `documents/<id>/<id>/MD<n>.zip` link, which is a *third* form that now 404s — but
seeing the shape was enough to guess `/documents/d/guest/`. Reading a client library's source
is a cheap way to learn a walled API's URL grammar.

---

## 4. INE: three products, and the detailed one stopped in 2022

| what | where | note |
|---|---|---|
| foreign residents by **province x 121 nationalities** | `jaxi/files/_px/es/px/t20/e245/p08/l0/03005.px` | 2.8 MB PC-Axis, keyless. **Series ends 2022** |
| population by province x Spanish/foreign, **current** | wstempus table `82104` | keyless JSON, 1 July 2026 |
| municipal population | GISCO LAU 2021, already on disk | placement weight only |

The detailed-nationality series is the old *Padrón continuo* and it stops when the *Estadística
Continua de Población* took over; ECP publishes province x nationality only as an
**agrupación de países** (EU / rest of Europe / Africa / …), which is far too coarse to carry
a religion. So spec §3.4 applies — **structure from the detailed source, totals from the
recent one** — and each province's 2022 nationality vector is rescaled to its current foreign
total.

**What that costs, stated rather than hidden:** the rescale is uniform, so nationalities that
grew faster than the average since 2022 are understated and those that shrank are overstated.
The largest single case is **Ukrainians**, whose Spanish population roughly doubled after
February 2022 — the 2022 figure is as of 1 January, five weeks before the invasion — so
Spain's Ukrainian Orthodox are the most understated group on this map's Spain.

Two path notes worth keeping:

- `.../p08/l0_ant/03005.px` is the same table one series older, ending **2020**. The `l0` /
  `l0_ant` pair is easy to get backwards and there is nothing in either file that says which
  is current except the period axis.
- The `wstempus` series name for `82104` is **`province . AGE . nationality . sex . measure
  . unit`**, not province-nationality-age-sex. Reading it in the wrong order silently
  produces an empty join rather than an error.

**And a normalisation trap that cost a province.** INE writes the same name three ways across
its own products — `Illes Balears` in the px axis, `Balears, Illes` in the ECP series names,
`Rioja (La)` against `Rioja, La` — so the join key is a **sorted token set**, not a string.
The moment that changed, an alias table keyed on raw lowercase (`"la coruña"`) stopped
matching and the UCIDE parse silently fell to 51/52 provinces and disabled itself. *An alias
table has to go through the same normaliser as everything else.*

---

## 5. UCIDE, and the check that makes it usable

`ucide.org/wp-content/uploads/2024/02/estademograf23.pdf` — 18 pages, free, annual, data to
31 December. Its central table is **52 provinces x (extranjeros, españoles, totales)**, and
the `españoles` column is the only published subnational count of Spanish-citizen Muslims
that exists.

The layout is nasty: the autonomous-community subtotal is interleaved into the **first**
province row of each community, so a first row carries seven numbers and every other row
four. The parser walks the token stream, keys on the province name and takes the 4th or 3rd
number accordingly.

**The check that makes that trustworthy is the report's own prose.** UCIDE states
`Total de hispanomusulmanes 1.085.593` in a sentence, on a different page from the table. The
52 parsed provinces sum to **1,085,593 exactly**. A parser that grabbed a neighbouring column
would miss it by hundreds of thousands. *When a PDF states its own total in words, that is a
stronger test than any internal consistency check on the table.*

**Its method, and why §14.9 permits it.** UCIDE applies per-nationality Muslim shares to the
*padrón* across 29 OIC nationalities — naming the low ones (Nigeria 50%, Guinea-Bissau 43%,
Ivory Coast 39%, Cameroon 21%, Togo 14%) — and adds a reconstructed Spanish-Muslim stock
built from naturalisations (570,606 over 55 years), Ceuta and Melilla, converts and
descendants. Under §14.5's old wording the five mixed nationalities disqualified the table;
Anita withdrew that ban on 2026-09-06 (spec §14.9) and it is now a preference.

**Four provinces are capped and they are the informative ones.** UCIDE's Spanish-Muslim count
exceeds CIS's *whole* other-religion cell in **Almería (49,920 > 31,675), Teruel (3,869 >
3,211), Ceuta (31,163 > 21,351) and Melilla (33,462 > 22,006)**. The split takes the smaller
of the two, so those four are drawn LESS Muslim than UCIDE would have them. That is the right
direction for a split — spec §3.1 lets an outside source divide a category, never enlarge it —
and the disagreement is worth reading as evidence about both sources: CIS's Ceuta sample is a
few hundred people over three years, and UCIDE has an interest in the number being large.

---

## 6. The foreign half: Pew for the family, hand-authored for the split

`taxonomy/es_origin.py`. Pew's *Religious Composition by Country, 2010-2020* (June 2025) is
one zip, no key, 201 countries, and gives **seven families**: Christians, Muslims,
Religiously unaffiliated, Buddhists, Hindus, Jews, Other.

**It does not divide Christianity, and for Spain that division is the entire question** — it
is what separates 630,000 Romanians and 111,000 Ukrainians from 315,000 Colombians and
212,000 Venezuelans. So the Christian split is hand-authored per origin country, two
significant figures, with the documented migrant skews corrected by name (Ukraine's Greek
Catholics at 0.12 rather than 0.09 because Spain's Ukrainians are western; India's Sikhs at
0.85 of Pew's "Other" because Spain's Indians are heavily Punjabi).

**INE's 121 nationality leaves partition its published foreign total to +0.000%** once
`APÁTRIDAS` is treated as a leaf rather than an aggregate — it is the only uppercase row that
is not a regional subtotal, and excluding it by an uppercase rule costs exactly 3,631 people
and breaks the partition test that would otherwise catch a real error.

**The cross-check.** The foreign half produces **1.82M Muslims**; UCIDE's implied foreign
Muslim figure is ~1.4M. 30% apart, in the expected direction — Pew's Morocco is 99.7% Muslim
where UCIDE discounts, and the uniform rescale to 2026 inflates the large 2022 nationalities.
Reported, not tuned away.

---

## 7. Boundaries cost nothing, and the counting geography is derivable from them

GISCO LAU 2021 carries Spain's **8,131 municipios** with INE's own five-digit code as
`LAU_ID`, and **the first two digits are the province**. So there is no join at all between
the placement layer and the counting layer — none of spec §8.1's three failure modes can
occur, because there is nothing to match. Portugal (§9v) was the first country where GISCO
cost nothing; Spain is the first where it also supplies the counting geography for free.

The one thing that would break quietly is zero-padding: Álava is `01059`, and a read that
drops the leading zero puts it in province `10`, Cáceres, at the other end of the country.
`es_geo.py` forces a 5-character string and asserts the province set is exactly `01`…`52`.

---

## 8. What Spain is worth drawing for

- **51.0% Catholic and 36.6% irreligious**, with 18% of citizens practising against 37% not.
  The `secular` share (atheist + agnostic, 24.2%) is behind only Czechia and Estonia here.
- **A southwest-to-northeast Catholic gradient**: Jaén 68.9%, Badajoz 65.8%, Ciudad Real
  65.8% against Girona 40.5% and Barcelona 41.0%. The Basque provinces are the most atheist
  and agnostic in Spain, around 31%.
- **Melilla 37.9% and Ceuta 32.3% Muslim** — the only places in the European Union that look
  like that — and then a belt that is agricultural rather than urban: **Almería 16.7%, Lleida
  13.8%, Girona 13.7%, Tarragona 12.3%, Murcia 10.3%**, with Madrid below the national
  average.
- **Romanian Orthodoxy as Spain's third-largest religious body**, and in provinces nobody
  associates with immigration: **Castellón 7.8%, Cuenca 5.8%, Lleida 5.2%, Guadalajara 5.1%.**
- **Protestantism as two unrelated things in the same provinces** — Alicante 3.7% and Málaga
  3.4% is northern European retirement plus Latin American evangelical churches.

## 9. What is left undone

1. **Placement is population, not religion.** Almería's Muslims spread over the whole
   province instead of the El Ejido greenhouse belt. The fix is the **Observatorio del
   Pluralismo Religioso's directory** — 7,756 geocoded non-Catholic places of worship over
   1,378 municipios in 18 confessions, downloadable as XLS. It is a §4.4 location layer and
   using it as a placement weight is a different claim from the one the counts support, so it
   is the biggest outstanding upgrade and it needs a decision, not just work.
2. **One cell holds every non-Catholic religion among citizens.** CIS has no follow-up
   question and no barómetro in the pool has one. **BREC** (Fundación Pluralismo y
   Convivencia, n=4,742, first wave November 2024, biennial) asks about forty religion items
   and would split it nationally if its microdata is released — that is the single item that
   would most improve Spain.
3. **The practising/non-practising split is thrown away** at the taxonomy step and only
   reported in prose. It is the strongest signal CIS carries that no other source here has.
4. **The Ukrainian understatement** in §4, which one newer INE nationality release would fix.

## 10. Placement by the right population — 2026-09-08, and the country where it pays least

§9as's finding, applied here the same day as France and Italy: a country drawn from a citizen
half and a nationality-derived foreign half needs **two placement weights**, because placing
both by total municipal population scatters the foreign half in proportion to where *citizens*
live. Spain looked like the strongest case for it on paper — **CIS does not sample foreigners
at all**, so the foreign half is not a correction to the survey but the other 15% of the
country, a larger share than Italy's.

The source is INE table **33571**, `Poblacion por sexo, municipios, nacionalidad
(espanol/extranjero) y edad`, keyed on INE's own five-digit municipal code — the one GISCO
carries verbatim as `LAU_ID`, so §7's "no join at all" holds here too. **8,131 of 8,131
municipios matched, none missing either way.** 41,932,488 Spanish and 5,542,932 foreign,
11.68%.

**And the effect is small, which is the finding.** Measured on DEGURBA, which nothing in the
build reads:

| | in cities (DEGURBA 1) | |
|---|---|---|
| **France** | foreign 64.5% vs citizens 36.1% | **28-point gap** |
| **Italy** | foreign 42.5% vs citizens 34.4% | 8-point gap |
| **Spain** | foreign **55.3%** vs citizens **53.5%** | **1.8-point gap** |

Spain is already 53.8% urban — far more than France's 38.1% — and its foreign population is
not concentrated in the big cities the way France's is. It is on the Mediterranean coast, in
the Almerian greenhouses, in the Balearics and the Canaries, and in Madrid and Barcelona, and
those pull in opposite directions on an urban/rural axis. So the drawn minorities barely
move: Sunni Islam 53.4% against a 53.8% baseline, Romanian Orthodox 52.1%, Ukrainian Orthodox
61.9%.

**The change is still right and is kept** — Barcelona is 22.0% foreign and Madrid 15.6%, and
a Moroccan or Romanian dot now sits where those people are rather than where Spaniards are —
but it buys Spain a fraction of what it buys France. **The size of the prize is the gap
between where the two populations live**, and that is worth measuring before assuming a
country needs the work: three countries, the same fix, and a fifteenfold difference in what
it was worth.

One caveat on vintage. INE 33571's latest period is **1 January 2022** and `es.py`'s
magnitudes are INE's 1 July 2026 population, so the weight is four years older than the
counts. That is fine for a weight and would not be for a count (§8.2, and §7's own warning
about `POP_2021`), but it means the weight understates the newest arrivals — the same
Ukrainian understatement §4 already records, from the other end.
