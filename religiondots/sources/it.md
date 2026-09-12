# Italy — `sources/it.py`, `sources/it_geo.py`, `taxonomy/it2024.py`, `taxonomy/origin_religion.py`

Drawn 2026-09-08. **107 province, 48 nodes, 58,930,857 people, 99.83% of the country.**
Italy has never asked, and ISTAT does not collect religion at all.

| | |
|---|---|
| counting geography | **NUTS 3 — 107 province, 553,000 people each**, the finest in Europe on this map |
| citizen composition | NUTS 2 (20 regioni) for Catholic and unaffiliated; NUTS 1 (5 ripartizioni) for everything else |
| placement | 7,903 comuni, GISCO LAU 2021, weighted by comune population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents |
| tier | **`modelled` throughout** — nothing here is a count of anybody |
| vintage | ESS 2012–2023 pooled (5 rounds), census 2021, Pew 2020 |
| authored cells | **none** |
| not drawn | 99,276 people, 0.17% — survey refusals, per §3.5 |

**The first country on this map drawn at three resolutions at once**, and the call was
Anita's. §3 below is the argument.

---

## 1. ISTAT does not collect religion, and that closes more than it looks like

Every other country here that lacks a religion census lacks a religion *question*. Italy
lacks the whole category: **religious affiliation is treated as sensitive data and ISTAT
does not collect it in any instrument** — not the census, not the permanent census, not the
multiscopo. That is worth stating precisely because it kills three leads that otherwise look
like Spain's CIS:

- **Aspetti della vita quotidiana** is the obvious candidate — about 50,000 individuals a
  year, free public microdata, published by region *and* by tipo di comune. What it measures
  is **`pratica religiosa`**, attendance frequency. Not affiliation.
- **The second-generation survey** looked better still: 68,127 students across 821 comuni,
  which would have been the largest fine-geography sample in the country. No religion
  question, for the same reason.
- **The 8x1000** is the one that survived longest and §5 has it.

So there is no Italian CIS-equivalent and there cannot be one. Spain works because CIS asks
`RELIGION` every month and publishes microdata in fifteen days; Italy's statistical office is
structurally barred from the question. **ESS is not the best available Italian source — it is
the only one**, and everything below follows from that.

## 2. ESS gave Italy a geography and then took it back

`sources.md` §11l closed Italy on the *otto per mille* and fr.md §9 checked ESS while
writing France up. Both were right about what they looked at. The full picture:

| round | year | citizen n | `region` level |
|---|---|---|---|
| 1, 2 | 2002, 2004 | — | **no `region` variable at all** (E201VariableNotFound) |
| 6 | 2012 | 916 | **NUTS 2 — 19 regioni** |
| 8 | 2016 | 2,452 | **NUTS 2 — 20 regioni** |
| 9 | 2018 | 2,557 | NUTS 1 — 5 ripartizioni |
| 10 | 2020 | 2,474 | NUTS 1 |
| 11 | 2023 | 2,632 | NUTS 1 |

**Italy went the opposite way from France.** France was re-lettered between NUTS vintages and
kept its 21 anciennes régions throughout, so `fr.py` maps `FR10`→`FR10`, `FR21`→`FRB0` and
pools seven rounds at one level. Italy's later files simply do not contain the finer level.
It is not a coding drift that can be undone.

**`regunit` is the variable that says which, and `it.py` asserts it rather than trusting the
code lengths.** It prints `NUTS level 2` for rounds 6 and 8 and `NUTS level 1` for 9–11. A
future release that moves Italy again — in either direction — will fail the assertion instead
of being pooled as though nothing had changed.

**And there is no `rlgdnait`.** Rounds 6–11 carry about twenty country-specific denomination
variables — `rlgdnade`, `rlgdnach`, `rlgdnanl`, `rlgdnbat`, `rlgdnagr`, `rlgdnapl` — giving
Germany its Landeskirchen and the Netherlands its whole Protestant taxonomy. Italy has none,
in any round. Both the geography *and* the categories are capped, and `taxonomy/it2024.py`
is short because of it.

## 3. Three resolutions, and why Greece's answer was wrong for Italy

Greece and France both hit this asymmetry and resolved it the same way — **throw the finer
level away.** `gr_geo.py` says it outright: the foreign half is available at NUTS 3 and is
aggregated up to NUTS 2 because *"that extra level is used for placement rather than for
counting, which is the only honest thing to do with a resolution that only half the data
has"*. `fr.py` repeats it: *"mixing would put the sharper geography on the half with the
weaker claim to it."*

**That reasoning is sound and its premise does not hold in Italy.** It assumes the fine half
is the small half. Compare:

| | foreign residents | citizen minorities |
|---|---|---|
| Greece | 0.76M, 7.2% | the recognised Thracian minority, ~110,000 |
| **Italy** | **5.03M, 8.5%** | **1.38M, 2.55% of citizens** |

**Roughly four fifths of the people this map exists to show in Italy are in the half that has
107 province.** Flattening to five units would have cost the country its best data in order
to protect its worst. So Italy keeps all three levels:

- **foreign residents** — NUTS 3, 107 province, measured citizenship counts
- **Catholic / unaffiliated** — NUTS 2, 20 regioni, rounds 6+8
- **every other category** — NUTS 1, 5 ripartizioni, rounds 9+10+11

**The split is by CATEGORY and not by half**, which is the part worth understanding. At
NUTS 2 the median regione holds **three** minority respondents and the thinnest holds one;
at NUTS 1 the median is 35. Catholic and unaffiliated have 2,418 and 862 respondents pooled
and survive being cut twenty ways. Nothing else does.

**How three levels become one column.** `scatter.py` takes a single `unit`, so counting
happens at NUTS 3 and the coarse rows are spread down proportional to each province's own
**citizen** population. This invents nothing: the dots would be placed by population inside
the coarse unit regardless (§8.2), so disaggregating first is spatially a no-op. It buys one
real thing — the citizen half is spread by *citizen* rather than total population, and
Italy's foreign share runs from 1.6% in Carbonia to 20.6% in Prato. Every citizen row's
`note` names the level its composition came from.

### 3a. The sample floor, which was added by disbelieving the output

The first honest build made **South Tyrol the least Catholic region in Italy** — 48.3% no
religion, 42.6% Catholic — on **29 pooled respondents.** Trento had 23. Every other account
of South Tyrol has it among the most observant places in the country.

spec §3.9b withdrew the floor on how many *units* a country may have and says nothing about
how few *people* may stand behind one. `N_FLOOR = 100` closes that: below it a regione's
Catholic/unaffiliated ratio comes from its ripartizione instead. At n = 100 the standard
error on the Catholic share is about 4.3 points; at 23 it is near 9.

Ten of twenty-one regioni fall back — Valle d'Aosta (33), Liguria (86), Abruzzo (57), Molise
(0), Basilicata (92), Sardegna (97), Bolzano (29), Trento (23), Friuli-Venezia Giulia (63),
Umbria (67). **That is 14.3% of the population, not half the map**, because the ten are the
small ones, and the population-weighted figure is the one `it.py` prints. South Tyrol now
draws 64.2% Catholic and 26.8% no religion.

**Molise is in no ESS round at all** — not round 6, not round 8 — so it was already taking
this path. 294,000 people, and the same shape as France's Corsica except that here the
surrounding NUTS 1 unit is sampled and can carry it honestly, so nothing goes undrawn.

`N_FLOOR` is a knob and moving it is a §14 decision rather than a tuning.

## 4. What the map says

**66.6% Catholic, 24.5% no religion, 3.8% Muslim, 2.7% Orthodox.**

**The least religious places are Tuscany and Emilia-Romagna, and the survey finds the *zone
rosse* unaided.** Tuscany reports 43.2% no religion and Emilia-Romagna 37.4%, against 11.1%
in Sicily and 15.7% in Puglia. Sicily is 83.4% Catholic; Tuscany is 47.0%. That is very
nearly the map of the post-war anticlerical and communist heartland, sixty years later, out
of a survey that asks nothing about politics.

**Islam runs the opposite way to Catholicism and follows the work.** Emilia-Romagna and
Lombardy are both 6.0%, Liguria 5.4%, against 1.3% in Sardinia. At province level: Piacenza
7.1%, Imperia 7.1%, Brescia 7.0%, Bergamo 6.7%, Modena 6.7% — the Via Emilia food-processing
belt and the Lombard engineering valleys — against 0.75% in Oristano.

**Italy's largest single non-Catholic node is Romanian Orthodox**, 1.05M people, more than
every Protestant, Jewish, Buddhist and Hindu community here combined. Orthodoxy peaks in
Lazio at 4.5% and Viterbo at 5.0%.

**Prato is 20.6% foreign**, the highest of any province; Parma, Piacenza and Milan follow at
about 15%.

## 5. The cross-checks, and the two that fail

CESNUR's *Le religioni in Italia* is read by nothing in the build, which is what makes it a
check rather than a restatement.

| | drawn | CESNUR | ratio |
|---|---|---|---|
| Muslim citizens | 491,070 | 417,900 | **1.18×** |
| Protestant + other Christian | 365,801 | 378,000 | **0.97×** |
| Orthodox citizens | 211,588 | ~400,000 | 0.53× |
| Jews | 63,863 | ~24,000 (UCEI registered) | **2.66×** |
| non-Catholic citizens, all | 2.55% | 4.2% | 0.61× |

**The Protestant row is the most useful thing in the table, because it diagnoses rather than
scores.** Taken alone the Protestant cell is absurd — 70,571 people against 313,000
Pentecostals plus ~65,000 historic Protestants. Add `Other Christian denomination` (295,230)
and the total lands within 3% of CESNUR. **Italian Pentecostals and Jehovah's Witnesses do
not tick *Protestant* on a survey form**, and the two cells are only meaningful together.
Nothing is moved between them — that would be inventing a magnitude to fix a label (§14.4) —
but `taxonomy/it2024.py` says so on both nodes.

**The Orthodox shortfall is in the split rather than the magnitude.** Orthodox across both
halves is ~1.6M against CESNUR's ~1.5M residents, which is fine; it is the citizen/foreigner
boundary the two sources draw differently, and CESNUR's "Orthodox citizens" includes
naturalised Romanians that `ctzcntr` and the census both place elsewhere.

**The Jewish cell is simply wrong.** Nine weighted respondents, 2.66× the registered
community. It is drawn rather than dropped, per §3.9b — dropping a real community to protect
an estimate is worse than drawing it with the error named — and the error is named in
`note_public`, `it2024.py` and here.

**And the divergence to flag is `unaffiliated`: 24.4% drawn against Pew's 13.3%.** That is
an instrument difference, not an error. ESS asks whether you belong to *a particular religion
or denomination*, which collects far more "no" than a question asking what your religion is;
the same gap puts France at 48.8%. Nothing is adjusted to close it.

## 6. Two bugs, and the first one is the transferable one

### 6a. A NUTS 3 code does not have to end in a digit

The first build drew **97 province and 52.4M people**, reported **99.83% coverage**, and
balanced perfectly in every internal check. The regex was `IT[A-Z]\d{2}`.

**Lombardia has twelve province and NUTS ran out of numerals**, so Mantova is `ITC4A`, Lodi
`ITC4B`, **Milano `ITC4C`** and Monza e Brianza `ITC4D`. Sardegna and Sicilia do the same.
Ten province and 6.6 million people vanished, including the largest one in the country.

**Every check passed because the coverage figure was computed against the truncated total.**
A percentage taken from the same filter that lost the rows cannot detect that rows were lost,
and it will report a *higher* number, not a lower one, because the missing people are missing
from the denominator too. This is a general shape and it is worth carrying: **a coverage
ratio is only a check if its denominator comes from a different file.**

`it.py` now asserts the province count and the national population against
`sources/it_geo.py`'s independently-built placement layer.

### 6b. The zero-padding trap, wider than Greece's

Excel stores the comune code as a number, so the workbook writes Agliè's `001001` as `1001`.
Italy's codes come out at **three different lengths** (4, 5 and 6) against the shapefile's
uniform 6. `zfill(6)` takes the join from **370 matched to 7,903**. Greece lost 644 rows to
this and it looked like a workbook with rows missing; Italy would lose 7,533 and it would
look like the wrong country. France escaped it only because Corsica's `2A001` forces the
column to be read as text.

## 7. The otto per mille is not used, and why that took two reversals

Recorded in full at `sources.md` §11l and §11l-ii; the short version, because it is the
question anyone coming to Italy asks first.

Italy has a church-tax analogue: every IRPEF filer assigns 0.8% of income tax to the State or
one of **thirteen named confessions**. 42,026,960 taxpayers, 16,893,963 expressed choices,
two exact partitions, no suppression — better arithmetic than most census tables here.

§11l concluded it was national-only and one email away from being a `roll`-basis country at
comune level, and called it *"the single highest-value outstanding lead in Western Europe"*.
§11l-ii reversed that twice in one day: the **regional detail is published** (behind a second
application with a near-identical name, reachable only by an `onclick`), and **the measure
fails as religion data**:

| | choices, tax year 2022 | actual |
|---|---|---|
| Chiesa Evangelica Valdese | **497,013** | ~22,000–25,000 members |
| Arcidiocesi Ortodossa | **39,359** | ~1.5M residents |

Twenty times over on one, forty times under on the other — a spending vote, not an
affiliation count. Soka Gakkai, the Jewish Communities and the Buddhist Union are all within
about 2× of their real numbers, so the bias is not uniform; it varies by a factor of ~800
across categories for explicable, category-specific reasons, which cannot be corrected
without already knowing the answer.

**Comune-level counts exist and are held by the Agenzia delle Entrate, not MEF**, which told
the Corte dei Conti it has no duty to publish them but supplies them to confessions who ask.
A FOIA (`accesso civico generalizzato`, d.lgs. 33/2013 art. 5 c.2) would reach them. It is
not worth sending: the best outcome is comune-level precision on a measure this map cannot
draw as religion, and at most a placement weight for the Catholic/State split.

## 8. Placement by the right population — 2026-09-08, and the question that produced it

Anita, on the finished map: *"it does look a bit weird to see hinduism spread like
population-proportionally throughout the italian countryside when in reality i imagine it's
much more urbanized."*

**The obvious fix was checked first and is useless.** ESS carries `domicil` — big city /
suburbs / town or small city / country village / farm — so a religion-by-settlement table for
citizens is one more break variable on the query already being run. It gives:

| | village | big city | national |
|---|---|---|---|
| Roman Catholic | 76.6% | 70.9% | 74.1% |
| No religion | 20.9% | 24.9% | 23.1% |
| Islam | **1.10%** | **1.10%** | 0.91% |
| Eastern religions | 0.18% | 0.80% | 0.26% |

Islam has **no urban gradient at all** and "Eastern religions" has a 4.4× one on about twenty
respondents. Nothing there can move a dot honestly.

**And it was the wrong instrument anyway, because Italy's Hindus are not citizens.** 166,000
people, essentially all in the foreign half. What was actually wrong is that **every dot in
the country was placed by TOTAL comune population**, so a province's foreign residents were
scattered in proportion to where *Italians* live.

**ISTAT publishes resident population by comune and citizenship** (`demo.istat.it`, RCS, 1
January 2025), so `it_geo.py` now puts `ital` and `foreign` on the placement layer and
`countries.py`'s `_ItWeighter` places each half by its own population. A node drawing from
both — `islam.sunni` has 491,000 citizens and 1.65M foreign residents — gets the two vectors
blended in that province's own proportion, computed from the same CSVs the counts come from.
**No magnitude changes; only the within-province scatter.** §8.2's own instruction.

Measured against **DEGURBA**, Eurostat's degree-of-urbanisation class, which nothing in the
build reads: foreign residents are **42.5% in cities against Italians' 34.4%**, and the drawn
dots now follow —

| | in DEGURBA 1 (cities) |
|---|---|
| Shia Islam, Buddhism | 50.0% |
| Eastern-rite Catholic | 45.5% |
| Ukrainian Orthodox | 43.4% |
| Protestant | 42.3% |
| Romanian Orthodox | 40.2% |
| Sunni Islam | 38.9% |
| *(national population)* | *35.2%* |
| Latin Catholic | 34.3% |

### And Hinduism barely moved, which is the finding

It sits at **36.4%**, a point above the national share. That is not the fix failing. Italy's
Hindus, by provincia:

> **Roma 18,703 · Brescia 12,203 · Latina 10,291 · Bergamo 8,408 · Milano 8,105 · Mantova
> 7,877 · Verona 6,081 · Vicenza 5,329 · Cremona 5,283**

Rome and Milan, and then **the Po valley dairy belt and the Agro Pontino** — Brescia,
Bergamo, Mantova, Cremona, Verona, Vicenza, Reggio Emilia, Parma, and Latina. That is the
Punjabi agricultural labour force of the Lombard and Emilian dairies and the Pontine market
gardens, and it is genuinely rural. **The intuition that a South Asian religion must be urban
is right in most countries and wrong in this one**, and the map was closer to correct than it
looked — for the wrong reason, since it was drawing rural Hindus by accident rather than
because it knew where they were.

**What this does not do**, recorded per §2.4: RCS is comune × *individual* citizenship and
would place Indians and Chinese by name rather than "foreigners" collectively. That needs
ISTAT's numeric country code mapped to ISO-2, and **ISTAT's own code list is a dead link** —
`Elenco-codici-e-denominazioni-unita-territoriali-estere.zip` is published on the
classification page and 404s. It buys less than it sounds at 107 province, because Prato and
Latina are each their own provincia and the composition already carries them.

**The trap in the file**, which cost a build: RCS stacks **four territorial levels in one
column**, distinguished only by code width — `1` Nord-ovest, `01` Piemonte, `001` Torino,
`001001` Agliè. Summing blind gives 294.7M people, five times Italy. Worse, `zfill(6)` turns
the aggregates into plausible comune codes (`000001`) that then fail the join *quietly*, as
117 unmatched rows rather than a wrong total. Filter on the raw width before padding, and
assert the national sum.

## 9. What generalises

1. **Check whether the statistical office collects the variable at all, before checking
   whether any survey asks it.** Italy's three most promising leads — a 50,000-person annual
   multiscopo, a 68,000-student survey, a national census — all fail for one reason, and
   knowing the reason closes them in one step instead of three.
2. **A coverage percentage computed from the filter that lost the rows is not a check.** It
   moves the wrong way when rows go missing. §6a.
3. **The rule "never mix resolutions" has a premise: that the fine half is the small half.**
   Greece and France were right to flatten and Italy is right not to, and the thing that
   decides it is not the resolution gap but which side of it the map's subject lives on.
4. **A survey category can be wrong about its label and right about its magnitude.** Italy's
   Protestant cell is out by 5× and its Protestant-plus-other-Christian total is out by 3%.
   Checking a node against an outside count is worth doing at more than one level of the
   tree, because the level where it agrees tells you what the respondents actually heard.
