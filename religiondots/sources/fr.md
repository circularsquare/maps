# France — `sources/fr.py`, `sources/fr_geo.py`, `taxonomy/fr2024.py`, `taxonomy/origin_religion.py`

Drawn 2026-09-07. **26 units, 48 nodes, 67,252,447 people, 99.30% of the country.**
France has never asked, and is barred by law from asking.

| | |
|---|---|
| counting geography | **NUTS 2** — 21 anciennes régions + the 5 overseas régions, 2.59M people each |
| placement | 34,606 communes, GISCO LAU 2021, weighted by commune population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents; `estimate` for the overseas régions |
| tier | **`modelled` throughout** — nothing here is a count of anybody |
| vintage | ESS 2010–2024 pooled (7 rounds), census 2021, Pew 2020 |
| authored cells | **none** |
| not drawn | Corsica, 347,585 people, 0.51% |

**Three instruments, not two.** The build landed on 2026-09-07 with 21 units and 96.43%, leaving
Corsica and the five overseas régions undrawn rather than borrowing a metropolitan composition
for Martinique. Anita asked the obvious next question — *"could we maybe try to find corsica or
martinique from some other data source?"* — and the answer was already on disk. See §10.

---

## 1. This is the country spec §14.3 was written against

`spec.md` §14.3 names France, by name, as the worked example of the move this project does not
make: *"France has no count at all. Estimating religion there would mean inventing the
magnitude as well as the location, most plausibly from surnames, origin or nationality."*
`sources.md` §11l reached the same verdict on 2026-09-06 and closed with *"France is not a gap
in the sweep; it is the boundary the sweep is drawn against."*

**Both were correct about every route known when they were written, and neither is about this
one.** ESS asks French residents which religion or denomination they belong to. It has asked in
seven rounds since 2010 and it publishes the answer by région. So the citizen half — 92.7% of
the people drawn — is a self-identification magnitude from a survey that asked, the same basis
as Russia's Arena, Georgia's Caucasus Barometer or Greece's own citizen half. No surname model,
no origin inference, and **no microdata at all**: the ESS API cross-tabulates server-side, which
matters here more than anywhere, because §14.3's legal paragraph draws its line precisely at
*"building the same model from individual-level microdata"*.

**§11l did not consider ESS, and the reason is chronology.** Its France section reviews TeO2,
the CEF's diocesan returns and IFOP, all national or access-controlled. The ESS API was found
later the same day, by §9z, while looking for something else in Greece. `spec.md` §14.11 is the
amendment; §14.10, decided by Anita on 2026-09-07, is what permits the foreign half.

**The one thing §14.3 keeps, and it is the operative rule:** *never model at a finer resolution,
or a stronger claim, than the source publishes its magnitude at.* That is what caps this country
at 21 régions and what keeps the foreign half off the 101 départements it is available at.

## 2. §11l closed France on a geography ESS does not use — the finding

§11l's people-per-unit table, which is the right first filter and was applied correctly:

| unit | count | people per unit | §11l's verdict |
|---|---|---|---|
| France **département** | 96 | 713k | passes |
| France **région** | 13 | 5.26M | **fails badly** |

**ESS uses neither.** It carries the *anciennes régions* — the 22 that existed before the 2016
merger — as NUTS-2010 `FR10`/`FR21`…`FR82` in rounds 5–7 and NUTS-2016 `FR10`/`FRB0`…`FRL0` in
rounds 8–11. The 2016 revision did not merge them out of NUTS; it **relettered** them and made
the 13 new régions their parents. The recode is 1:1 and is `FR10_TO_16`.

That is **21 units at 3.10M people each**, not 13 at 5.26M. Still the coarsest counting geography
here — Russia's federal subjects are 1.82M — and spec §3.9b settled that a country's geography is
not a gate.

*A scouting note that rejects a country on a number should say where the number came from.* §11l's
13 was the administrative fact about France, not a fact about the source, and it was never checked
against ESS's codelist. It closed the country for a day short of a fortnight.

## 3. Which rounds exist, and what pooling costs

France is in all eleven rounds. **Rounds 1–4 have no `region` variable** (`E201VariableNotFound`),
the same wall Greece hit, so **seven** are usable against Greece's three: 12,678 pooled citizen
respondents over 21 régions, about 600 each.

**The category list does not collapse the way Greece's does.** Every French round offers and uses
at least ten denominations, so nothing blinks out of existence the way `Islam` does in Greek round
11. What pooling costs instead is time — rounds 5–11 span 2010 to 2024 — and the drift is smaller
than that span suggests:

| round | year | n | none | Catholic | Islam | Protestant |
|---|---|---|---|---|---|---|
| 5 | 2010 | 1,649 | 52.6% | 39.3% | 4.6% | 1.3% |
| 6 | 2012 | 1,863 | 53.8% | 37.4% | 4.5% | 1.9% |
| 7 | 2014 | 1,802 | 52.3% | 39.2% | 5.1% | 1.5% |
| 8 | 2016 | 1,966 | 49.1% | 42.3% | 4.9% | 1.9% |
| 9 | 2018 | 1,867 | 49.9% | 40.4% | 4.4% | 2.3% |
| 10 | 2020–22 | 1,835 | 50.8% | 38.6% | 6.4% | 1.7% |
| 11 | 2023–24 | 1,670 | 53.5% | 36.2% | 6.8% | 1.5% |

**"No religion" is flat across the whole fourteen years** with no trend at all, which is the fact
that makes a seven-round pool defensible and is itself worth noticing about France. Catholicism
falls about a point every five years. **Islam among citizens rises 4.6% → 6.8%**, which is
naturalisation and age structure rather than conversion, so the pooled 5.25% understates the
present by roughly a point. That is in `note_public`.

**Reading trap, and it is Greece's in a quieter form.** `region` must be read as a codeList
*value*, never as a label. Greece switched from Latin to Greek labels mid-series; France keeps the
same twenty-one régions but codes them in two NUTS vintages and has drifted in punctuation anyway
— `Nord - Pas-de-Calais` against `Nord-Pas de Calais`, `Centre` against `Centre — Val de Loire`.
Pooling on labels would split six régions in two, halve each, and raise no error anywhere.

## 4. `ctzcntr`, and the foreign half

ESS samples non-citizens and reaches them badly, so `ctzcntr = Yes` is what stops the two halves
double-counting — Greece's §9z argument unchanged. France's non-citizen share in the sample runs
4.6–7.1% across the seven rounds against a true 7.3%, which is better coverage than Greece's
(3.0% against 7.2% in round 10) and still not good enough to use.

Eurostat **`cens_21ctz_r3`** gives the other half, and France's numbers are the best any country
has produced on it:

| | |
|---|---|
| total | 67,439,568 |
| `NAT` | 62,528,810 |
| `FOR` | 4,910,758 (7.28%) |
| named citizenships | **201, covering 100.00% of `FOR`** |

Greece's named list covered 99.84% and needed the unnamed-remainder rescale; France's is exact, so
that step is a no-op here. It is kept because the next country's will not be.

Portugal (547k), Algeria (546k) and Morocco (494k) are a third of the foreign population between
them, then Italy, Turkey and Tunisia. **No single origin decides the country the way Albania
decides Greece's Muslim total** — the largest, Portugal, is 11% and is Catholic — so France needs
no equivalent of §9z's Albanian coefficient judgement.

**The foreign half is available at 101 départements and is drawn at 21 régions.** `countries.py`'s
Greece note gives the reason and it is sharper here: the citizen half is a survey at 21 units and
this half is a model, so drawing the model five times finer would make the most-inferred part of
France also the most precise-looking part of it.

## 5. No cell is authored, and that is the difference from Greece

Greece needed two hand-written rows: the Thracian minority, whom a Greek-language sample reaches
essentially none of, and Mount Athos, which no sample will ever contain. **France needs neither,
and the reason is that its survey is not blind in the way Greece's is.** ESS finds, unaided:

- **Alsace at 10.6% Protestant** among citizens against 1.75% nationally — 8.9pp on a cell whose
  own binomial SE is 1.55pp at n=396, so **5.7 SE**, or 4.7 at a design effect of 1.5.
- **Île-de-France at 16.4% Muslim and 1.7% Jewish**, both the highest in the country by a wide
  margin.
- **Poitou-Charentes at 63.8% no religion** against Alsace's 38.0%.

Where it *is* weak — the banlieues, where a general-population survey under-reaches recent
immigrant populations everywhere — **there is no published régional figure to substitute**. Greece
had the Council of Europe's ECRI count of a recognised minority to reach for; France has nothing
comparable, and inventing a coefficient to fill the gap is spec §14.4's first prohibition. So the
under-reach is stated in `note_public` rather than corrected.

## 6. The cross-check, per spec §14.10's fifth condition

| | |
|---|---|
| Pew's own country estimate for France, 2020 | **9.10% Muslim** |
| this build, bottom-up | **8.04%** |

Looser than Greece's 5.08 against 5.12, and it should be read as looser: the categories are eight
times larger, ESS is a self-identification survey and Pew's figure is a composite estimate, and a
survey always finds more "no religion" than a composite does. Across all seven Pew families:

| family | this build | Pew 2020 |
|---|---|---|
| Christians | 41.8% | 46.5% |
| Muslims | 8.0% | 9.1% |
| Religiously unaffiliated | 48.8% | 42.7% |
| Buddhists | 0.13% | 0.71% |
| Jews | 0.53% | 0.69% |

Nothing is tuned to close any of these. **The Buddhist row is not sampling noise and is diagnosed
in §8 below.**

## 7. What is drawn, and what it says

**48.8% report no religion, which is the largest node France puts on the map and is larger than
Catholicism.** Catholicism is 38.0%, Islam 8.1%, Protestantism 2.2%.

**Alsace is the exception to everything.** 10.4% Protestant against 2.2% nationally — the Lutheran
and Reformed churches of the one part of France where the 1801 Concordat never lapsed, where the
state still pays clergy of four recognised cults and religion is taught in public schools. It is
also **10.0% Muslim**, so the most Protestant région in France is very nearly its second most
Muslim one, and it has the lowest "no religion" share in the country at 38.0%.

**Islam's geography is the cities and the industrial north-east**: Île-de-France 16.4%, PACA
11.6%, Alsace 10.0%, Franche-Comté 9.4%, Rhône-Alpes 8.9% — Paris, Marseille, Strasbourg, Sochaux,
Lyon, which is to say where the car plants and the ports were — against **1.8% in Poitou-Charentes**
and 2.4% in Bretagne.

**The déchristianisé west and centre survive in the data.** Poitou-Charentes 63.8% no religion,
Picardie 58.0%, Centre-Val de Loire 57.9%, against Alsace 38.0% and Lorraine 42.7%. That is
recognisably the map Gabriel Le Bras and Fernand Boulard drew from Mass attendance in the 1940s
and 1950s, sixty years and one collapse in practice later, from a completely different instrument.

**France's Jews are 0.53% of the country and 1.7% of Île-de-France** — the largest Jewish
population in Europe, and after Alsace's Protestants the sharpest regional concentration here.

## 8. What is left undone, and what the sources cannot do

**Nothing reaches `secular`, and in France that is worse than anywhere it has been noted.** ESS
asks which denomination you belong to and never offers atheist or agnostic, so all 48.8% land in
`unaffiliated`. France has the largest self-declared atheist population in Western Europe and this
map cannot draw one dot of it. gr2024.py and ge2014.py make the same call for the same reason with
far less at stake.

**The Buddhists are the residual's fault, and it is the biggest single loss.** France has the
largest Buddhist population in Europe — the Vietnamese, Cambodian and Lao communities who arrived
after 1975, plus a large convert following — and ESS offers no Buddhist box: `Eastern religions`
and `Other Non-Christian religions` both land in `other.fr`. So the map draws 0.13% against Pew's
0.71%, and **almost every Buddhist it does draw holds a foreign passport**, because the foreign
half can name them and the citizen half cannot. `other.fr` is 0.64% of France and that gap is most
of it.

**The counting unit is the biggest thing wrong and it is not fixable from here.** Île-de-France is
12.3M people drawn as one composition, so its 16.4% Muslim share spreads over the whole région in
proportion to where anyone lives. **This map can say Île-de-France is 16% Muslim and can say
nothing whatever about Seine-Saint-Denis**, which is the thing a reader would most want from it.
No source on this map supplies that, and §14.3's rule is what stops the obvious substitutes.

**0.51% of France is not drawn: Corsica.** The overseas régions were in this list until §10 below
took them out of it. Corsica stays, and §10 records what was tried.

**The Alevis are drawn Sunni on the citizen half.** 100,000–150,000 people of Turkish origin with
their own cemevi network and a live dispute about whether they are Muslims at all; ESS does not ask
and no French source counts them. `origin_religion.py` does split Turkish nationals into sunni,
alevism and shia, so France's `alevism` node is populated by passport-holders only and is an
undercount by construction. gr2024.py records the same shape for the Bektashi Pomaks.

**Protestantism is one node holding two different objects** — the historic Lutheran and Reformed
churches of the Wars of Religion and the Concordat, and the evangelical and Pentecostal sector that
is the fastest-growing religious movement in the country and is largely Caribbean, African and
Roma. The survey cannot separate them. **The geography does, partly**: Alsace's 10.4% is almost
entirely the first and Île-de-France's 3.0% is substantially the second, and nothing in the data
says so.

**The leads that were checked and are not used.** TeO2 (INED/INSEE 2019–20) is a real
religion-and-parents' religion survey whose microdata needs authorisation from the Comité du secret
statistique and would yield *région* — coarser than what is drawn here — so it is not worth the
application. The CEF's diocesan returns cover 94 metropolitan dioceses and 10,326 parishes and are
published as national totals only; the per-diocese figures exist and are not released, and are the
one outstanding lead that would beat this build. IFOP polls religion often and nationally.
`sources.md` §11l has all three.

## 9. What generalises

1. **ESS's `region` variable can be a vintage older than the country's current geography**, and it
   is worth reading the codelist before judging a country's resolution. France's administrative
   answer is 13 régions; its ESS answer is 21.
2. **Two of the four §11l countries are now drawn** — Greece and France — and both were closed on
   claims that were true about the sources §11l checked and false about ESS. Spain is drawn on CIS
   and §9z already recommends moving its foreign half onto `cens_21ctz_r3`. **Italy is the
   remainder and was checked while writing this**, because point 1 says to look rather than assume:

   | round | n | `region` level |
   |---|---|---|
   | 6 (2012) | 959 | **NUTS 2 — 19 regioni** |
   | 8 (2016) | 2,630 | **NUTS 2 — 20 regioni** |
   | 9, 10, 11 | 2,744 / 2,643 / 2,866 | NUTS 1 — 5 macro-regions only |

   **Italy went the other way from France: it had the fine geography and gave it up.** Only rounds
   6 and 8 carry regioni, pooling to ~3,589 respondents over 20 units — about 180 each, against
   France's 600 and Greece's 600 — and the recent rounds, which are the large ones, are five units
   of 12M people. So Italy on ESS is buildable and thin, and it is a *worse* map than the 8x1000
   route §11l identified as *"the single highest-value outstanding lead in Western Europe"*, which
   costs an email. **Check ESS before writing a country off; do not assume ESS wins once you have.**
3. **One alphanumeric value in a column protects every leading zero in it.** Greece lost 644 LAU
   codes to Excel; France has identical exposure in départements 01–09 and loses none, because
   Corsica's codes are `2A001` and `2B033` and pandas therefore reads the column as text.
   `fr_geo.py` guards it anyway — a vintage that drops Corsica puts the trap straight back.

## 10. The overseas régions, found in a file that was already open — 2026-09-07

The first build left 2.28M people undrawn. Anita asked whether Corsica or Martinique could come
from somewhere else, and **the answer for the five overseas régions was in `data/raw/fr/pew.zip`,
which `origin_religion.py` had been reading all along for the foreign half.**

Pew's *Religious Composition by Country* publishes **French Guiana, Guadeloupe, Martinique,
Mayotte and Reunion as separate countries**, with their own rows, populations and seven-family
breakdowns. Nobody had looked, because the file is thought of as "the origin-country composition
table" rather than as a source of countries in its own right.

### Why it is legitimate here and would not be in most places

**Each DOM is exactly one NUTS 2 unit.** `FRY1` *is* Guadeloupe; `FRY4` *is* La Réunion. So a Pew
country row is a unit row, and there is **no downscaling at all**. spec §14.3's rule — never model
at a finer resolution than the source publishes its magnitude at — is satisfied by identity rather
than by argument, which is the cleanest this map ever gets to be about a modelled figure. The same
Pew row used for metropolitan France would be worthless, because there it would have to be spread
over 21 units.

`estimate` is §3.1's own basis for exactly this (*"a compiler's judgement | Pew, WRD/WCD, ARDA"*),
and each DOM unit is built on that basis **alone**: the foreign half is deliberately not run over
them, because Pew's estimate already covers every resident whatever passport they hold, and adding
it would double-count. `_foreign_half` filters on `NUTS2` for that reason.

### The magnitudes were checked before they were used

| unit | | census 2021 | Pew 2020 | ratio |
|---|---|---|---|---|
| `FRY1` | Guadeloupe | 415,792 | 407,394 | 0.980 |
| `FRY2` | Martinique | 360,748 | 356,614 | 0.989 |
| `FRY3` | Guyane | 286,617 | 289,056 | 1.009 |
| `FRY4` | La Réunion | 871,156 | 861,446 | 0.989 |
| `FRY5` | Mayotte | **no row** | 284,370 | — |

**Two independent counts of the same four populations agreeing to within 2%** is confirmation of
the magnitude from a source that is not the census, and `fr.py` asserts the 0.85–1.15 band rather
than reporting it. Per spec §3.4 the shares are Pew's and the totals are the census's for those
four; **Mayotte has no census row at all**, so it is Pew's on both — and France's one
overwhelmingly Muslim département is on this map only because a second source counted it.

`fr.py` still asserts that Mayotte's census row stays empty, but now for the opposite reason to
before: a row appearing would let the foreign half reach the unit and double-count against Pew's
estimate. That is a decision for a person, not a rescale.

### What they add

| | none | Christian | Muslim | Hindu | other |
|---|---|---|---|---|---|
| Mayotte | 0.2% | 0.5% | **98.8%** | — | 0.5% |
| La Réunion | 2.1% | 87.5% | 4.2% | **4.5%** | 1.6% |
| Guyane | 3.4% | 84.2% | 0.9% | 1.6% | **9.2%** |
| Guadeloupe | 2.5% | 95.9% | 0.4% | 0.5% | 0.6% |
| Martinique | 2.7% | 96.0% | 0.2% | 0.2% | 0.8% |
| *metropolitan France* | *48.8%* | *41.8%* | *8.1%* | *0.1%* | *0.7%* |

**The last row is the point.** Metropolitan France is the most irreligious large country on this
map and its own overseas régions are among the most religious places on it — 2.5% reporting no
religion in Guadeloupe against 48.8% in the Hexagone. That contrast is the single largest thing
these five units add, and it exists inside one state and one legal order.

La Réunion's 4.5% Hindu and 4.2% Muslim are the Malbar, descended from indentured Tamil labourers
brought after abolition in 1848, and the Zarabe, Gujarati Sunni traders who arrived from the
1870s. Guyane's 9.2% *Other religions* is the Businenge (Maroon) communities of the Maroni and the
Kalina, Wayana, Teko, Wayampi and Palikur; see `branches.py`'s `other.fr`.

### What it costs, and it is the category depth

Pew publishes **seven families**, so the five overseas units are drawn at seven where metropolitan
France is drawn at nine denominations. **Their Christianity is one undivided colour.** Guadeloupe
and Martinique are ~96% Christian and overwhelmingly Catholic in every account of them; the
Seventh-day Adventists are strong in both and the evangelical sector has grown steadily since the
1970s — and Pew publishes no split, so this map draws none. §14.4's first rule is the one that
never moves, and `fr2024.py`'s `PEW_REVIEW` holds the depth calls one by one.

The asymmetry inside that mapping is deliberate and is the rule worth carrying: **map each source
category as deep as its own NAME warrants.** Pew's `Christians` is a family that is genuinely
mixed in these territories, so it goes to the `christianity` root. Pew's `Muslims` is the same
category ESS calls `Islam`, which already goes to `islam.sunni` for metropolitan France — and the
case is stronger overseas, since Mahorais Islam is Shafi'i Sunni almost without exception. Sending
one to a bare `islam` while the other went to `islam.sunni` would put 281,000 Mahorais in a
different legend row from the Muslims of Marseille as an artefact of two ingest decisions.

### Corsica: looked for, and still not drawn

**ESS carries 709 variables and exactly one geography.** `geographicVariables` returns `cntry`
alone; the full list adds `region`, `regunit` (which NUTS level `region` is) and `domicil`
(urban/rural). There is no French geography in ESS that reaches Corsica, in any round. That is now
established rather than inferred from the codelist.

**Pew publishes the five DOM because they are territories with their own ISO codes. Corsica has
none, because it is metropolitan France.** What is left:

- **A general-population survey.** Corsica is 0.51% of France, so a 2,000-person national sample
  holds about ten Corsicans. That is not a composition.
- **The Annuario Pontificio's diocese of Ajaccio**, which is exactly coterminous with the region —
  geographically perfect, and a **`roll`** against a `self_id` map. §3.1 forbids the addition, and
  it would draw Corsica far more Catholic than the mainland purely as an artefact of the
  instrument. This is the same trap §11l's *avvalentesi* lead is in Italy.
- **Borrowing the metropolitan average**, which draws the national mean and tells the reader
  nothing true.

So Corsica stays undrawn at 0.51%, §6.12's wash marks it, and `note_public` says so. *The general
point: a territory with an ISO code is a candidate for a compiler's country table and a region
without one is not, whatever its population — which is a fact about how the compilers are
organised and not about the place.*

### A pre-existing viewer bug this made visible — FIXED 2026-09-07, Anita chose the fix

Standing over Mayotte or La Réunion, the country picker says **"Auto (all countries)"** instead of
France, even though the dots are drawn correctly and the legend is one click away.

The cause is exact and is not in this country's data. `countryAt(55.53, -21.12)` returns **`fr`** —
`country_shapes.geojson` knows Réunion is France — but Auto then gates on `viewFill` and
`viewOverlap`, and **both read `META[cc].view`, the declared OPENING box**, which for France is
metropolitan (`[-5.4, 42.2, 8.4, 51.2]`). `viewOverlap('fr')` over Réunion is **0**, so the branch
that would set the country never fires.

**This predates France and affects four countries**, all of which draw territory far outside their
opening view: Portugal over the Azores (`view` is mainland, `bbox` reaches −31.2), the United
States over Hawaii and Alaska (`view` is the lower 48), and Mauritius over Rodrigues — which §9af
chose deliberately, because a box holding both is thirteen parts ocean. France just makes it
matter more, because its overseas régions are 2.2M people rather than Rodrigues's 43,000.

**Anita's call was fix 1: let the shapes win where they have already answered.** The `viewOverlap`
term is dropped at the two call sites that pass `at` — the country `countryAt` has placed the
camera inside — and left at the other four, which take `country`, `cc` or `best` and have no such
confirmation. Two lines in `index.html`, plus the reasoning beside them.

The rejected alternative was **falling back from `view` to `bbox`** in `viewFill`/`viewOverlap`.
`counts.json` does carry a `bbox` per country and France's is `[-63.1, -21.4, 55.8, 51.1]`, but
those boxes are enormous: the US one runs from Hawaii to Maine, so it would score "the middle of
the screen is over the United States" across most of the Pacific and could take the legend off
Mexico. It fixes the islands by making the boxes mean nothing.

**What the gate was actually carrying, checked rather than assumed — the answer is nothing.**
`viewOverlap`'s own note says *"Position only: scale is viewFill's question, and a view can score 1
here at any zoom"*, so hemisphere range is `viewFill`'s job; border drift is `stillHolds` and
`contested`, in the branch below. Measured after the change:

| view | expected | got |
|---|---|---|
| Réunion, Mayotte, Martinique, Guadeloupe, Guyane | `fr` | **`fr`** |
| Hawaii → `us`, Azores and Madeira → `pt`, Rodrigues → `mu` | fixed too | **fixed** |
| Paris, metropolitan France, Berlin, Warsaw | unchanged | **unchanged** |
| Europe-wide and hemisphere, centre pixel in France | must release | **releases** — `viewFill` |
| mid-Atlantic | nothing | **nothing** |
| Kehl, arrived cold from Madrid | not France | **`null`** |

**The border band is unchanged and is not this term.** Drifting from Strasbourg to Kehl keeps
`fr` for about 4 km past the Rhine and hands over to `de` by Offenburg; the same walk from
Karlsruhe keeps `de` into Strasbourg. Symmetric, and it is `stillHolds`' anti-flicker band doing
exactly what its comment says. Arriving at Kehl cold selects nothing, so the change did not make
France greedy at a border.

### One thing left, and it is not Auto — `country_shapes.geojson` is 220 m coarse

`countryAt` returns **null** at Basse-Terre town and at Ponta Delgada, both of which are on a
shoreline. `SIMPLIFY = 0.002` degrees is about 220 m, so the stored outline can sit that far
inland of the true coast and a point taken from a waterfront town centre falls outside it.

It is pre-existing, it is invisible on large countries, and **it bites small islands hardest** —
which is now four countries' worth of territory rather than a curiosity. Pointe-à-Pitre, Le Moule,
Angra do Heroísmo and Funchal all resolve correctly, so the islands are named; it is specific
coastal points that are not. Not changed: `SIMPLIFY` is shared by every country and drives the
file's size, which the viewer loads on every page view. **Recorded rather than fixed.**

## 11. Placement by the right population — 2026-09-08, and France is where it pays most

§9as found this in Italy and it applies here with more force than anywhere else on the map.
Every country with a citizen half and a nationality-derived foreign half — Greece, Spain,
France, Italy — was placing **both** halves by total commune population, so the foreign half
was scattered in proportion to where *citizens* live. France's counting units are 26 régions
of 2.6M people, the coarsest counting geography here, so the placement weight is doing more
of the work of making the country look like a country than in any other build.

INSEE's RP 2021 detailed table **`TD_NAT1`** is population by sex, age and nationality at
`CODGEO` — the same five-character commune code GISCO carries as `LAU_ID`, so there is no
crosswalk and none of §8.1's failure modes apply. `INATC` is the condensed nationality
indicator: 1 Français, 2 Étrangers. 4.5 MB zipped. `fr_geo.py` puts `french` and `foreign`
on the placement layer and `countries.py` blends them per (unit, node) in the proportion that
région's own counts give. **62,357,103 French and 5,050,949 foreign, 7.49%.**

**The gap it closes is enormous, and much larger than Italy's.** Measured on DEGURBA, which
nothing in the build reads:

| | in cities (DEGURBA 1) |
|---|---|
| foreign nationals | **64.5%** |
| French nationals | **36.1%** |
| *(total population)* | *38.1%* |

Italy's equivalent gap is 42.5% against 34.4% — eight points. **France's is twenty-eight.**
The drawn dots follow:

| | in cities | |
|---|---|---|
| Buddhism | 71.1% | |
| Romanian Orthodox | 71.1% | |
| Judaism | 63.3% | |
| Sunni Islam | **55.0%** | against a 38.1% baseline |
| other.fr | 53.1% | |
| Protestant | 46.9% | |
| Latin Catholic | 36.8% | citizen half, correctly below |
| unaffiliated | 35.7% | citizen half |

**§8.2 is satisfied and nothing about a magnitude moved** — every région's totals are still
ESS's and Eurostat's; only the within-région scatter changed. It does not fix §8's real
complaint about this country, which is that Île-de-France is one composition over 12.2M
people: the map still cannot tell you anything about Seine-Saint-Denis. What it can now do
is put Île-de-France's Muslim dots in Île-de-France's *cities* rather than across its wheat.

**Two things about the file.** `NIVGEO` is `COM` or `ARM` — the municipal arrondissements of
Paris, Lyon and Marseille are counted **both** in their own `ARM` rows and inside their
commune's `COM` row, so summing the column double-counts 2.7M people in exactly the three
cities the map most wants right. And 53 communes have no `TD_NAT1` row and keep the
population weight; they are mostly in the overseas régions, which are drawn from Pew as whole
units anyway and whose residents are overwhelmingly French nationals, so nothing there
changes.

## 12. The foreign half moves to NUTS 3 — 2026-09-08, and it reverses §8's biggest complaint

§8 named one thing as *"the biggest thing wrong and it is not fixable from here"*: the map
could say Île-de-France is 16% Muslim and **nothing whatever about Seine-Saint-Denis**. It was
fixable from here. The foreign half was always available at NUTS 3 in the same Eurostat table
the NUTS 2 figures come from — the note declining it is quoted in §11 above — and it was
declined on Greece's rule that mixing puts the sharper geography on the half with the weaker
claim to it.

**Italy (§9as) showed that rule has an unstated premise: that the fine half is the small
half.** France's foreign half is 4.74M people and carries most of what a religion map of this
country exists to show. So France now counts:

| | level | units |
|---|---|---|
| foreign residents | **NUTS 3** | 94 départements |
| the five overseas régions | NUTS 3 by identity | each DOM is one NUTS 3 unit |
| French citizens | NUTS 2 | 21 anciennes régions, spread down by each département's own citizen count |

99 counted units at about 680,000 people each, against 26 at 2.59M.

### What it buys, in the place §8 named

| département | people | Muslim | no religion | Catholic |
|---|---|---|---|---|
| **Seine-Saint-Denis** | 1,666,102 | **21.5%** | 37.0% | 30.1% |
| Val-d'Oise | 1,254,414 | 16.9% | 40.6% | 33.2% |
| Val-de-Marne | 1,412,945 | 16.6% | 40.6% | 33.0% |
| Hauts-de-Seine | 1,632,377 | 15.8% | 41.9% | 33.3% |
| Essonne | 1,311,428 | 15.5% | 41.3% | 33.9% |
| Paris | 2,129,331 | 14.9% | 42.4% | 33.4% |
| Yvelines | 1,453,706 | 14.9% | 42.4% | 34.0% |
| **Seine-et-Marne** | 1,435,448 | **13.9%** | 42.6% | 34.4% |

All eight used to draw **16.3%**. Seine-Saint-Denis is now the most Muslim unit in
metropolitan France, which is what every other account of the country also says.

**And the drawn spread is NARROWER than the real one, which has to be said.** The citizen
half is still the région's composition — ESS carries nothing below it — so roughly 92% of
each département's people take Île-de-France's average and only the foreign half varies. On
the foreign half alone the spread is 12.9% against 3.7%, three and a half times; drawn
together it is 21.5 against 13.9, about one and a half. **The map understates the difference
between these places and cannot do otherwise from these sources.** Every citizen row's `note`
says which région its composition came from.

**Nothing about the basis or the magnitudes changed** — the national totals are identical to
the NUTS 2 build, 67,252,447 drawn and 99.30%. Only the resolution moved, and Eurostat
publishes at that resolution, so §14.3's *never model at a finer resolution than the source
publishes* is satisfied by the source rather than by an argument. That distinction is the
whole justification: this is not a finer model, it is the same model read at the grain its
input already had.

### Below the département there is nothing, and the obvious substitute is a trap

Anita asked whether arrondissements could be reached — *"something that would tell us like
which arrondissements have muslims"*. They cannot, and the near miss is instructive.

**Paris is one département**, so all twenty arrondissements share a composition even at
NUTS 3. INSEE's commune-level tables do carry them — `NIVGEO = ARM`, 45 arrondissements
across Paris, Lyon and Marseille — but only as French vs foreigner, never nationality by
country.

**And using that foreign share as a within-Paris weight would make the map confidently
wrong.** The twenty arrondissements sit in an 11.4–16.8% band, and the ranking is:

> 18e **16.8%** · 19e **16.2%** · **16e 16.1%** · … · 9e 11.4%

The 16th — Passy and Auteuil, the wealthiest arrondissement in Paris — is statistically tied
with the 18th. One is European, American and Gulf nationals; the other is Maghrebi and West
African. A Muslim dot placed by foreign share lands in Passy as readily as in Barbès.

**The general form: within a city, nationality COMPOSITION varies far more than nationality
SHARE.** The weight that works across a country stops working at the scale where populations
interleave, and it fails silently, because the resulting map looks more precise rather than
less. Reaching arrondissements honestly would need nationality by country below the
département, which INSEE's open tables do not publish and its detailed files coarsen for
disclosure control — §14.3's wall again, one level down.
