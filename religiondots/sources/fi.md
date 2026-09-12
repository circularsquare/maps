# Finland — `sources/fi.py`, `sources/fi_geo.py`, `taxonomy/fi2024.py`

Drawn 2026-09-08. **19 units, 54 nodes, 5,511,092 people, 99.59% of the country.**
The register that counts everybody has no geography; the survey that has geography counts a
different thing.

| | |
|---|---|
| counting geography | **NUTS 3, the 19 maakunnat** — 291,000 people each |
| placement | 310 kunnat, GISCO LAU 2021, weighted by municipal population |
| basis | self-identification (survey) for citizens; nationality-derived for residents |
| tier | **`modelled` throughout** |
| vintage | ESS rounds 5-11 (2010-2023) pooled, census 2021, Pew 2020 |

---

## 1. Why this is not built from the register, which is the whole country

Finland records religious-community membership for every resident in the population
information system, kept by the Digital and Population Data Services Agency, and Statistics
Finland publishes it as StatFin **`vaerak/11rx.px`** — *Belonging to a religious community by
age and sex*, **25 named religious communities**, 1990 to 2025, exact to the person. (§11c and
§11k both say 26; the variable has 26 values and one is the TOTAL row.)

**Its dimensions are community, sex, age group and year. There is no region dimension.** That
was §11c's finding and §11k's closure, and it is confirmed here directly from the table's own
metadata rather than repeated.

So Finland is Germany's shape (§3.9a) with Germany's answer withheld: a state register that
covers everybody exactly, could answer at any geography, and is published flat. Germany's
Zensus 2022 gives the same kind of number for 10,786 Gemeinden. Finland gives it for one unit.

**The two quantities are not the same and the gap is large enough to be the point of the
map.** `sources/fi.py` fetches the register purely so this comparison is reproducible from the
repo, and prints it at the end of every build:

| | survey (this map) | register, 31.12.2024 | |
|---|---:|---:|---|
| Evangelical Lutheran | 45.00% | 62.24% | **-17.25 points** |
| no religion / no community | 46.94% | 34.88% | +12.06 |
| Orthodox | 1.86% | 1.03% | +0.83 |
| Islam | 1.67% | 0.48% | +1.20 |

The first row is the famous one: about a million people are on the church's rolls and do not
say they belong to it, because resigning requires filing a form and not resigning requires
nothing. **The last two rows go the other way and are the more useful half of the table**, and
they are why "the register is simply better" is wrong. The register can only see membership of
a *registered congregation*: most Finnish Muslims never join one, and an Orthodox resident of
Russian or Estonian background need not belong to a Finnish parish. On those two groups the
register undercounts and the survey plus the census's own origin counts do better.

## 2. The sweep, because §11k's closure deserved a real test

§11k's instruction was *"do not re-scout the Nordics without a specific new release to point
at"*, and that is about the register tier and is correct about it. The re-test was not a hunt
for a new release; it was a check that the closure had looked everywhere.

**All twelve top-level databases at `pxdata.stat.fi` were walked in full on 2026-09-08**, both
levels, against nine needles: `uskonn`, `uskonto`, `kirkko`, `kirkkoo`, `seurakun`, `kirkoll`,
`hiippa`, `luteril`, `ortodoks`.

| database | nodes | religion hits |
|---|---:|---|
| `StatFin` | 1,532 | **`vaerak/11rx.px` only** |
| `StatFin_Passiivi` | ~3,800 | parish employees' salaries (`yskp`, 6 tables); *"read religious or devotional books"* (`vpa`, 2 tables) |
| the other ten | ~300 | none |

`Kuntien_avainluvut` (municipal key figures), `Postinumeroalueittainen_avoin_tieto` (Paavo,
postcode areas) and `Kokeelliset_tilastot` (experimental statistics) were all checked
specifically, because a Norway-style *"share belonging to the church"* key figure would have
lived in one of them. **Finland does not publish one.** Norway does (§11k, table 12025); that
is the difference between the two countries and it is worth recording, because it means
Norway has a route Finland does not.

The Digital and Population Data Services Agency, which actually holds the register, publishes
no subnational religion either; statistical dissemination of the register is Statistics
Finland's job and Statistics Finland has published one flat table.

`kirkontilastot.fi`, the Evangelical Lutheran Church's own statistics service, publishes
membership by parish, deanery and diocese through embedded Tableau views with no download or
API found. **That is a live lead and it is a different tier** — a church's own roll, one
denomination, joined to municipalities through a parish correspondence that does not exist as
a file. It was not pursued, and §11k's objection to the Norwegian version of this move
(*"joining them would be modelling"*) applies.

**So the register tier closes, on what the register measures rather than on reachability, and
it closes harder than before.** What was never tested is the survey tier.

## 3. The ESS route, which nobody had looked at

Finland is in every ESS round. Rounds 1-4 have no `region` variable, the wall Greece, France
and Italy all hit, so **seven of eleven are usable** — and unlike Italy (§9bp), where `region`
drops from NUTS 2 to NUTS 1 partway through, Finland is **NUTS 3 in all seven**.

| round | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| respondents | 1,878 | 2,197 | 2,087 | 1,925 | 1,755 | 1,577 | 1,563 |
| Finnish citizens | 1,842 | 2,150 | 2,039 | 1,892 | 1,724 | 1,558 | 1,536 |

**Those are unweighted counts of people, and that distinction cost a correction here.**
`pspwght` is a post-stratification weight whose subtotal over citizens is **12,718** against
**12,741** people, so a figure lifted off the weighted frame and printed as "respondents" is
quietly 23 short. `sources/fi.py` now prints both, takes every people-count off the unweighted
pass, and asserts `N_CITIZENS` so a reissued ESS round fails the build rather than silently
falsifying `note_public`.

**12,741 citizens interviewed over 19 maakunnat**, a median of 455 each, from 64 in Åland to
3,406 in Helsinki-Uusimaa. Refusals are 0.28% of citizen respondents, which is small enough
that the undrawn remainder is 0.41% of the country and carries no `gap` row, following the
other four ESS countries.

§11ai listed the deferred Nordic ESS route for Norway, Sweden, Denmark, the Netherlands,
Belgium, Latvia, Ukraine and Luxembourg. **Finland is not on that list**, and the reason is
visible in the section itself: §11k had closed Finland, so it was not among the countries
anyone thought needed a route.

### `rlgdnafi`, and the reason it is worth chasing

`rlgdnfi` **does not exist in any round** and raises `E201VariableNotFound`. The variable is
`rlgdnafi`. §11ai's warning was that `a`/`b`-suffixed revisions silently drop later rounds
when you pool on the bare name; Finland is the loud version, where the bare name drops
everything and errors.

It is worth the five minutes because the fallback is much worse. `rlgdnm`, the harmonised
variable, offers seven families and would put 45% of Finland on **`Protestant`**. `rlgdnafi`
names fourteen bodies, and the ones that matter here are the Evangelical Lutheran Church, the
Orthodox, Pentecostals, the Free Church, Adventists, Jehovah's Witnesses and Mormons, all
separately. The category list shrinks with the sample in the usual way (14 codes in rounds 5
and 6, 11 by round 10), which is why seven rounds are pooled.

## 4. Three traps, one of which would have been silent

**`table.path` is a list of INDICES into `values`, not a list of code values.** This is the
expensive one. Read as codes, every category whose numeric code is not also a valid array
index returns zero — for Finland that is everything from code 10 upward, which is **Islam,
`Other Protestant`, `Other Christian`, `Eastern religions` and `Other Non-Christian`**. The
map would have been drawn with no Muslims at all, no error anywhere, and a national total
that still looked about right because the missing five categories are 2% of the country.
`sources/gr.py` indexes correctly and never said why.

**Three NUTS vintages in seven rounds**, which is worse than Greece's two:

| | round 5 | rounds 6-10 | round 11 |
|---|---|---|---|
| vintage | NUTS 2006 | NUTS 2013 | NUTS 2021 |
| Kainuu | `FI134` | `FI1D4` | `FI1D8` |
| Pohjois-Pohjanmaa | `FI1A2` | `FI1D6` | `FI1D9` |

Pooling on the raw code splits those regions three ways and every piece comes out undersized
with no error. **A recode table is a name join wearing a code's clothes**
(`[[reference_name_join_wrong_neighbour]]`), so `_check_recode()` requires every pair to carry
the same ESS label on both sides and refuses the build otherwise.

**It earned itself on the first run**, refusing `FI181 Uusimaa -> FI1B1 Helsinki-Uusimaa`.
That is genuine: Itä-Uusimaa (`FI182`) was abolished on 1 January 2011 and absorbed into
Uusimaa, and NUTS then renamed the enlarged region Helsinki-Uusimaa. Both old codes are
exempted by name with that reason. Territorially `FI181 + FI182 = FI1B1` exactly, so pooling
loses nobody; what it loses is the ability to separate the eastern fringe from Helsinki in
round 5 alone.

**`11rx.px` is the table id and `statfin_vaerak_pxt_11rx.px` is not.** The second is what the
PxWeb web UI shows in its own URL, and it 400s against the API. The listing's `id` field is
the only thing that works. Also: the API rate-limits at about 30 requests per 10 seconds and
answers 429 with no `Retry-After` in the body, which reads like a dead host if you are walking
it in parallel. One request every 0.45s with backoff walks the whole system in a few minutes.

### The one thing that failed at the end

`coverage.py` keeps a hand-maintained list of the countries that have a foreign half (`es`,
`gr`, `fr`, `it`), because their coverage is the union of the mapping's targets and the ~40
nodes `origin_religion.py` can emit. **Finland was not added to it and failed the first
coverage run**, with nine nodes drawing dots that `fi2024.py` cannot express: `islam.shia`,
`buddhism`, `hinduism`, `alevism`, bare `islam`, and the Greek, Romanian, Ukrainian and
Bulgarian Orthodox churches. That list's own comment records Italy having been missed the same
way and staying invisible for weeks, because coverage only sees nodes that DRAW. Finland's
foreign residents are concentrated enough to round up, so it surfaced immediately. Fixed, and
`tiles.py --refresh-meta` is enough to push it (14 covered nodes -> 54).

## 5. The boundaries cost nothing, and the placement has one named limit

The GISCO LAU 2021 bundle, on disk since Poland (§9e), carries **all 310 Finnish
municipalities** with their NUTS 3 code and their 2021 population. The join is 310 of 310 in
both directions, the 19 NUTS 3 codes are exactly the 19 ESS regions after the recode, and the
population sums to 5,525,292. **No download at all.** §9v said Portugal was the case where
§9e's promise about the GISCO file is true without qualification; Finland is another one.

**The placement is a population weight and not a religion one.** Helsinki-Uusimaa is drawn as
one composition over 1.69M people, so its 3.12% Muslim share spreads across the region in
proportion to where anyone lives; eastern Helsinki and Vantaa against the western commuter
belt is invisible.

**The named improvement**: Statistics Finland publishes foreign citizens by municipality
(`vaerak/11rq`, which §11c saw while finding that religion is not published that way). Wiring
it in would let Finland use the Italy weighter (`_ItWeighter`), placing the citizen half by
citizen population and the foreign half by foreign population, which is worth real accuracy in
Helsinki-Uusimaa (8.96% foreign) and Åland (11.76%) against Etelä-Pohjanmaa (1.82%). Not done;
it is one PxWeb call and a wiring change.

### The foreign half, in one table

The 2021 census's own count, largest first, of the citizenships behind the foreign half
(200 named citizenships cover 99.98% of the 271,091 foreign residents):

| | | | |
|---|---:|---|---:|
| Estonia | 50,866 | Afghanistan | 7,059 |
| Russia | 28,866 | Syria | 6,915 |
| Iraq | 14,708 | Viet Nam | 6,630 |
| China | 10,458 | Somalia | 6,460 |
| Sweden | 8,041 | Ukraine | 5,837 |
| Thailand | 7,851 | Türkiye | 5,679 |
| India | 7,237 | United Kingdom | 4,847 |

So Finland's Muslim dots outside the survey are mostly **Iraqi, Afghan, Syrian and Somali**,
in that order, and its non-Finnish Orthodox are mostly Estonian, Russian and Ukrainian. Getting
that order right matters because note_public names them.

## 6. What the map says, and what it cannot

The reading that survives the sample: **no religion is the largest single answer at 46.94%,
just ahead of Lutheran at 45.00%**, and the two together are 92% of the country.

The geography is not the north-south gradient one expects. As a share of each maakunta's own
drawn population, Lutheran runs from **36.75% in Helsinki-Uusimaa to 59.54% in
Etelä-Pohjanmaa**, and unaffiliated from **32.89% in Etelä-Pohjanmaa to 53.06% in Lappi**. The
axis is the Ostrobothnian revivalist west coast against the capital and the far north.

**Pohjois-Karjala comes out 6.22% Orthodox against 1.86% nationally**, the highest of the
nineteen. That is the part of Karelia that stayed Finnish in 1944 and it is a real historical
geography recovered from about 400 respondents. It is the strongest single piece of evidence
that this instrument is working, because nothing told it to expect that.

**What it cannot do:**

- **Laestadianism.** Conservative Laestadianism is a revival movement inside the national
  church whose strongholds are Pohjois-Pohjanmaa and Lappi, and it is **not drawable from this
  instrument at all** — its members are members of the national church and answer code 1 like
  every other Lutheran. `christianity.lutheran.laestadian` exists on the tree and Finland puts
  nothing on it. Pohjois-Pohjanmaa comes out 46.16% Lutheran against 45.00% nationally, which
  is to say the survey sees no trace of it. Drawing it from the movement's own membership
  estimates would be inventing a geography, and §3.5's rule says name the hole instead.
- **`secular`.** ESS offers no atheist or agnostic option, so all 2.53M non-belongers land on
  `unaffiliated` and Finland puts nothing on `secular`. Greece and Georgia do the same.
- **Shia Islam among citizens.** ESS has one Islam code. The foreign half splits Iraq and Iran
  by their own Pew compositions, so Finland's Shia dots come entirely from foreign residents
  and the citizen Shia are inside `islam.sunni`. `taxonomy/fi2024.py` says so.
- **Anything below one percent** should be read as "some, here" rather than as a number. Jewish
  and Mormon are three pooled respondents each.

## 7. Files

| | |
|---|---|
| `sources/fi.py --fetch` | ESS rounds 5-11 (weighted and unweighted), Eurostat `cens_21ctz_r3`, Pew 2020, StatFin `11rx` |
| `sources/fi.py` | -> `data/normalized/fi.csv` (167 rows) and `fi_foreign.csv` (841 rows) |
| `sources/fi_geo.py` | -> `data/geo/fi/fi_lau.gpkg`, 310 kunnat with `unit` and `pop`. No download |
| `taxonomy/fi2024.py` | 15 source categories -> 14 nodes, with `REVIEW` on seven of them |

---

## 8. Second look, 2026-09-08 — a review pass, read from the files rather than from §1-7

`check_md.py`, `built_countries.py --check`, `check_rollup.py fi` (5,511,092 all `modelled`,
no derived, no orphans) and `review_dump.py fi` are clean. Screenshot clean: dots inside the
border, the southern triangle and the Helsinki-Tampere-Turku cities dense, Lappi sparse,
Åland present, nothing in the sea, Sweden and Norway correctly blank.

### Every reader-facing figure recomputes, and the two the builder's verifier caught stayed caught

Recomputed from `data/normalized/fi.csv` + `fi_foreign.csv` + `data/raw/fi/statfin_11rx.json`
without reading `fi.py`'s prints. **Sixteen of sixteen.** Register Lutheran 62.2445%, register
Orthodox 1.0271%, register Islam 0.4779%, register no-community 34.8788%, survey Lutheran
44.9977%, unaffiliated 46.9433%, sum 91.94%, Orthodox 1.8574%, Islam 1.6738%,
Etelä-Pohjanmaa 59.5420% and it IS the Lutheran maximum, Helsinki-Uusimaa 36.7504% and it IS
the minimum, Lappi 53.0578%, Etelä-Pohjanmaa 32.8909%, Pohjois-Karjala 6.2154% and it IS the
Orthodox maximum, Helsinki-Uusimaa 3.1196% and it IS the Islam maximum, 271,091 foreign,
99.5898% coverage, 12,741 unweighted citizen respondents, median 455 / max 3,406 / min 64,
Pohjois-Karjala 396 respondents. The register variable has 26 index entries with `SSS`
present, so **25 named communities is right and §11c/§11k's 26 was the TOTAL row**, exactly
as §1 says. `grain` at 291,000 checks: 5,533,793 / 19 = 291,252.

### One more weighted-subtotal-as-a-count, in §3 of this file

§3 says *"Refusals are 0.28% of citizen respondents"*. **0.28% is the weighted share (36.00 of
12,718 = 0.2831%); the respondents are 33 of 12,741 = 0.2590%.** Same class as the correction
§3 itself records, one sentence later. `fi.py` is not wrong — `answered` is a share and taking
it off the weighted pass is the right call — only this sentence's *"of citizen respondents"*
is. The downstream 0.41% is fine either way: 22,701 undrawn of 5,533,793 is 0.4102%.

### The `path` fix is right, and the assertion it deserves is on the wrong axis

Confirmed from the raw JSON, not from the note. `ess_r11_n.json`'s `rlgdnafi` `codeList` is
thirteen entries whose `value`s run `1..7, 10..14, 6666`, and `table[0]["path"]` is
`[0, 0, 0, 0]` — **0-based positions, and position 7 is `Islam` at code `10`**. `_ess_table`'s
`labels[n][j]` is correct, and using the returned `variableValues` order rather than the
requested `breakVariables` order is the safer of the two.

**But the guard is on `region` and not on the denomination.** `_check_recode()` asserts the
recoded region set is exactly the 19 maakunnat, which a shifted index would break. The
denomination axis has only `set(pool["cat"]) - set(fi2024.MAP)`, which is **one-directional**:
it catches a category that appeared and cannot catch one that vanished, which is precisely the
no-Muslims failure. The symmetric assertion is free and holds exactly today — the pooled
category set over seven rounds is all fifteen `MAP` keys, verified — so
`assert set(pool["cat"]) == set(fi2024.MAP)` would fail loudly on the regression the docstring
describes. Not applied by the review: `fi.py` is the builder's and this is a behaviour change,
not a typo.

**Applied 2026-09-08.** `_citizen_shares()` now runs the check both ways: the existing
`set(pool) - set(MAP)` for a category that appeared, and a new `set(MAP) - set(pool)` for one
that vanished, each with its own message. `python sources/fi.py` was re-run to confirm it
passes on today's raw files, and it does: 167 rows, 15 source categories, and
`data/normalized/fi.csv` and `fi_foreign.csv` came out byte-identical to the shipped ones, so
the assertion is free in the literal sense. The comment above it says what to do if ESS ever
retires a category for real, which is to drop it from `MAP` in the same commit rather than to
weaken the check.

### `coverage.py`'s foreign list is now complete, checked both ways

Five countries write a foreign half and all five are listed. Enumerated from
`data/normalized/*_foreign.csv` (`es, fi, fr, gr, it`) and again from every reader of one in
`countries.py` (lines 4895, 4982, 5074, 5277, 6336, plus `_foreign_share`, which is generic
over `cc` but is called only by `_it_foreign_share`). **No sixth country is missing.**

### Two decimal places, which is Finland's own convention and is the thing the note argues against

Italy's ESS-drawn `note_public` prints one decimal; Greece's prints none. **Finland prints two,
everywhere.** That is worth naming because of what this particular note is for: `45.00%` set
beside `62.24%` gives a pooled sample estimate and an exact register count the same four
significant figures, and a trailing `00` on a survey number is the strongest available visual
claim to register precision — in the sentence whose whole job is to say the two are different
kinds of number. The 95% half-widths behind those decimals, unweighted on the citizen half:
±0.9 points nationally, ±1.6 in Helsinki-Uusimaa, ±4.5 in Etelä-Pohjanmaa, ±4.6 in Lappi,
±2.3 on Pohjois-Karjala's Orthodox share, ±12 on Åland.

### Two of the superlatives cannot be separated, and one of them flips when the weights come off

- **"the most unaffiliated is Lappi at 53.06%"** — Lappi 53.06% against Helsinki-Uusimaa
  51.90% in the drawn data, and **unweighted on the survey alone the order reverses**:
  Helsinki-Uusimaa 53.73% (n=3,406, CI 52.05-55.40) against Lappi 52.81% (n=462, CI
  48.26-57.37). A superlative that depends on the weighting is not a finding about Finland.
- **"the most Lutheran maakunta is Etelä-Pohjanmaa"** survives weighting but is not separable:
  unweighted Etelä-Pohjanmaa 61.76% (57.29-66.22), Pohjois-Savo 60.24% (56.51-63.98),
  Keski-Pohjanmaa 59.62% (51.92-67.32), Kainuu 59.46% (53.00-65.92). Four units inside each
  other's intervals. The **west-coast-against-capital axis** the sentence is really making is
  solid; the ranking inside the west coast is not.
- **"Pohjois-Karjala comes out 6.22% Orthodox, the highest of the nineteen" is sound** and is
  the note's flagship claim: 5.56% (3.30-7.81) on the survey alone against Pohjois-Savo's
  2.43% (1.25-3.60), non-overlapping. It really is recovered, and it really is the maximum.

### `45.00%` is the map's blended number and the sentence attributes it to ESS

*"The European Social Survey asks people whether they consider themselves as belonging to any
particular religion, and gets **45.00%**"* — **ESS alone gets 47.56%** among Finnish citizens.
45.00% is that survey diluted by the nationality-derived foreign half, which contributes almost
no Lutherans. The comparison against the register is still the right one (both are shares of
all residents); it is the attribution to one instrument that is off by 2.6 points. Same shape
one paragraph down: Pohjois-Karjala's 6.22% Orthodox includes the foreign half's Russians and
Estonians, so *"recovered from about 400 respondents"* is describing a number the 400
respondents did not produce on their own — they produced 5.56%.

### §14.16's split-half, applied to an ESS country for the first time — and Lutheran fails it

The project's settled view on survey sampling is **not** a sample-size floor. `lapop.py`'s
`stability()`, on spec §14.16: *"Size is eligibility; stability is evidence."* Eligibility is
1% of the country; the bar is `1.96/sqrt(n_units-1)`, here **+0.462** on 19 units. Finland
runs neither test. Run on the pooled rounds, early half 5-7 against late half 8-11:

| category | national (citizen, weighted) | n | spearman | verdict |
|---|---:|---:|---:|---|
| No religion | 48.01% | 5,929 | **+0.65** | own geography |
| Evangelical Lutheran | 47.56% | 6,250 | **+0.44** | **fails the bar** |
| Eastern Orthodox | 1.02% | 129 | **+0.59** | own geography |
| the other twelve | 0.93% and below | 3-112 | | under the 1% eligibility floor |

**And the check has the power to mean something, which is the first thing to establish**
([[reference_check_needs_power]]). Simulating a world where each maakunta's Lutheran share is
fixed at its pooled estimate and only Finland's actual per-unit sample sizes vary: median
Spearman **+0.72**, 5th percentile **+0.50**, and it clears the bar **97%** of the time. The
observed +0.44 sits below that 5th percentile, so this is a real failure to replicate and not
a thin-sample artefact. (No religion 98%, Orthodox 94% — both passes are meaningful. Islam
would have had 53% and no power, so its exclusion by the floor costs nothing.)

The likely cause is not error. **The pooling window is thirteen years of fast secularisation.**
The register's own Lutheran share over exactly rounds 5 to 11, pulled from `11rx.px` for
2010-2024 rather than remembered: **78.27, 77.30, 76.43, 75.32, 73.85, 72.98, 71.97, 70.94,
69.83, 68.72, 67.76, 66.59, 65.15, 63.57, 62.24** — sixteen points, monotone, inside the window
this map averages over. If the cities emptied faster than Ostrobothnia the ranking *should*
move, which is §14.16's China sentence exactly: *"a rank correlation is a stability test, not a
reality test"*, and a low value is a failure to demonstrate signal rather than a demonstration
of noise. The chi-square agrees the units differ overwhelmingly (Lutheran p=3.6e-31, no
religion p=1.9e-32, Orthodox p=6.8e-18). §14.16's treatment of that combination is to draw it
with the weakness named in `note_public`; Finland's note names sample thinness but not the
instability.

### The register comparison is the substance, and the headline gap is measured across a decade

Survey Lutheran by round (weighted, citizens) against the register in that round's fieldwork
year, which is the like-for-like the pooled table cannot show:

| round | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fieldwork | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2023 |
| survey | 56.25% | 45.18% | 44.47% | 49.63% | 50.18% | 44.35% | 42.39% |
| register | 78.27% | 76.43% | 73.85% | 71.97% | 69.83% | 66.59% | 63.57% |
| gap | -22.0 | -31.3 | -29.4 | -22.3 | -19.7 | -22.2 | -21.2 |

**Two things fall out and both are good for the note's argument.** First, the two instruments
fall in parallel — the register loses 14.7 points across the window and the survey 13.9 — so
the disagreement is a stable offset of roughly twenty-two points rather than drift, which is
much better evidence for the *"never filed the form"* explanation than a single-year snapshot
is. That is worth having.

Second, **`note_public`'s -17.25 points is the smallest version of the gap available**, because
it sets a survey pooled over 2010-2023 against the register at the end of 2024 and the register
moved nine points in between. Built on round 11 alone the map would draw roughly 40% Lutheran
against that year's register 63.57%, so the like-for-like divergence is about 23 points, not
17. `fi.py`'s own docstring says *"in round 11 those are 63% and 43%"* and is the more honest
statement of the two; the reader-facing number is the conservative one. **Not a defect** — the
pooled figure is what the map actually draws and 62.24% is the current register, so both
numbers are the right ones to print — but the sentence reads as one measurement against
another taken at the same time, and they are fourteen years apart at one end.

**And the survey does not fall smoothly, which is probably what breaks the split-half.**
45.18 to 49.63 to 44.35 between adjacent rounds is five- and six-point swings on n≈1,800, where
sampling error is about ±2.3. Something round-specific is moving the Finnish sample; ESS design
and mode changed more than once across rounds 5-11 and I have not checked which round did what,
so that is a lead and not a finding. It does mean the pooled composition is an average over an
instrument that was not constant, and the rank instability at maakunta level is the same
wobble seen through nineteen small samples.

**Twelve of the fifteen categories are under the eligibility floor and are drawn on their own
unit shares anyway.** Finland's entire Jewish population is placed on three respondents in two
maakunnat and its Mormons on three in three. `note_public` does say to read anything under one
percent as "some, here", which is the honest disclosure; what is missing is what §14.16 does
about it, which is to put the national rate in each unit's residual instead.

**This is not Finland's defect alone and should not be fixed as one.** Greece, France, Italy
and Spain all pool a survey and none of the four runs a stability test either — `grep` for
`spearman|split-half|stability` across `sources/gr.py`, `it.py`, `fr.py`, `es.py` returns
nothing. The whole ESS family inherited the omission, and Finland is the country where it
became visible because it is the first one whose largest category fails. **Applying §14.16 to
the five ESS countries is a rule everyone shares and belongs to Anita**, so it is written here
rather than filed as a fifth ask.

### Checked against precedent and found consistent, so nothing to say about it

`year=2024` on a citizen half pooled from 2010-2023 matches `gr.py`, `it.py` and `fr.py` to the
line. No `gap`/`gap_share` matches all four other survey countries. `other.fi` is the standard
per-country bucket beside `other.es/gr/fr/it`, not a new legend row. The eight `REVIEW` entries
are argued and land where the named precedents land. Finland has no `view`, and neither do ten
other countries.

### One inconsistency across three files, none of it reader-facing

`rlgdnm`'s Protestant share is *"43%"* in `fi.py`'s docstring and in `fi2024.py`'s, and *"45%"*
in `countries.py`'s internal `note` and in §3 here. `fi2024.py` also says the register is
*"62.9%"* where everything else says 62.24%. The true figures: 48.96% of citizens would land on
`rlgdnm`'s Protestant, and the register is 62.2445%.
