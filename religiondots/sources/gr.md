# Greece — `sources/gr.py`, `sources/gr_geo.py`, `taxonomy/gr2024.py`, `taxonomy/origin_religion.py`

Drawn 2026-09-06. **14 units, 48 nodes, 10,461,956 people, 99.80% of the country.**
No census has asked since 1951.

| | |
|---|---|
| counting geography | **NUTS 2, 13 regions + Mount Athos** — 750,000 people each, finer per person than Kenya's counties (§9o) |
| placement | 6,137 LAUs, GISCO 2021, weighted by LAU population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents; two authored cells |
| tier | **`modelled` throughout** |
| vintage | ESS 2010/2020–22/2023–24 pooled, census 2021, Pew 2020 |

---

## 1. The ESS API is open, and this is the most reusable thing here

`ess.sikt.no` is a single-page app that answers **the same 1,070-byte shell for every path** —
§11d's compare-the-404s test, positive. Its `/env.js` names the backend in one line:

    VITE_API_URL='https://api.nsd.no/graphql'

**That endpoint answers anonymous introspection and anonymous queries.** No account, no key,
no registration — which matters, because the portal's own download flow
(`ESSQuery.startDownloadJob`) *does* require a login. Better still,
`analysis.frequencyTabulationByVariables` **computes the cross-tabulation server-side**, so a
country's religion-by-region table is one HTTP request and no microdata ever moves.

Four things cost an hour between them and are worth writing down:

1. **`breakVariables` takes variable NAMES, not the UUIDs the search returns.** Passing an id
   gives `E201VariableNotFound`, which reads like the variable is absent.
2. **Its GraphQL type is `[String!]!`.** A query variable declared `[String]` or `[String!]`
   is rejected by the type checker with a message about position, not about the value.
3. **`region` across all 30 countries at once trips `E204TooManyCategoriesInVariable`.** The
   way round is `byVariables: ["cntry"]`, which tabulates *within* each country.
   **`subsetJson` returns a bare 422 in every documented shape** — object, list, dict,
   `{"filters": …}` — and the app itself passes a structured `subset` field that the deployed
   schema's `AnalysisDatafile` does not declare.
4. **The JS bundle is where the working call shape is written down**
   (`breakVariables: B.map(I => I.name[t.value])`). §12's grep-the-bundle rule, applied to
   find an argument convention rather than an endpoint.

Datafile ids are found through `search.searchDatafiles`, whose `searchTerms` is also a list;
`ESS10` overflows the API's 10 MB response cap and needs a narrower term.

## 2. Which rounds exist, and what they can and cannot see

Greece is in ESS rounds 1, 2, 4, 5, 10 and 11. **Rounds 1, 2 and 4 have no `region` variable
at all** (`E201VariableNotFound`), so three are usable: 2,511 + 2,708 + 2,666 = **7,885 Greek
citizens over 13 regions, about 600 each** — the same order as Russia's federal subjects.

**The category list shrinks as the sample does, and that is a property of surveys rather than
of Greece.** Round 5 uses eight denominations, round 10 uses seven, round 11 uses four — and
`Islam` is not among round 11's, not because Greece stopped having Muslims but because 2,757
respondents reached none.

**Two reading traps, both silent:**

- **ESS labels the Greek regions in Latin in round 5 (`Anatoliki Makedonia & Thraki`) and in
  Greek in rounds 10–11 (`Aνατολική Μακεδονία, Θράκη`).** Pooling on labels rather than
  codeList *values* splits all thirteen regions in two and every one of them comes out
  half-sized, with no error anywhere.
- **The NUTS code changed too.** Round 5 is NUTS-2006 `GR11`…`GR43`; rounds 10–11 are
  NUTS-2016 `EL30`…`EL65`. The recode is 1:1 and is in `GR_TO_EL`.

## 3. `ctzcntr` is what makes two halves possible

**Unlike Spain's CIS, ESS does sample non-citizens** — its target population is everyone
resident regardless of nationality. So the two-half design would double-count without a
filter, and `ctzcntr` (are you a citizen of this country) is in all three rounds.

It also shows *why* the second half is needed at all:

| round | non-citizen respondents | share of sample | true foreign share |
|---|---|---|---|
| 5 (2010) | 203 of 2,713 | 7.5% | ~7% |
| 10 (2020–22) | 83 of 2,800 | 3.0% | 7.2% |
| 11 (2023–24) | 87 of 2,757 | 3.2% | 7.2% |

**Coverage of foreigners halved between 2010 and 2020 and has not recovered.** Restricting to
citizens removes the overlap and that undercount in one move.

## 4. The census table, which is EU-wide and better than Spain's source

Eurostat **`cens_21ctz_r3`** — *Population by country of citizenship, age groups and NUTS 3
region*, 2021 census round, published 21 May 2025, keyless JSON:

    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/cens_21ctz_r3
        ?format=JSON&lang=EN&age=TOTAL&sex=T

For Greece: **10,482,482 people** against ELSTAT's own census total of 10,482,487 — five
people apart — split **9,716,914 citizens / 758,597 foreign**, with **200 named citizenships
covering 99.84%** of the foreign total. The unnamed remainder (`STLS`, `RNC`, `UNK`) is spread
across the named citizenships of its own region rather than dropped, so each region's foreign
total is the census's own.

**Both halves come out of this one table**, because it publishes `NAT` and `FOR` next to the
named countries — so the denominator of the citizen half and the numerator of the foreign half
are rows of the same file and cannot drift apart. Spain's equivalent needed two INE products
three years apart and a uniform rescale (§9y); **Spain should be moved onto this table.**

**Two failure modes, one of them expensive:**

- **The `geo` dimension holds every NUTS level in one column** — `EL`, `EL3`, `EL30`, `EL301`
  are all rows — so a prefix filter counts the same people four times. Greece first came out
  at **41.9 million**. The filter has to be `EL(\d{3}|ZZZ)`.
- **Eurostat needs a current `certifi`.** Its chain is GlobalSign Atlas R46 → GlobalSign Root
  R46, and the installed bundle was **2020.12.05**, predating that root — which fails as
  `unable to get local issuer certificate` on curl, urllib *and* certifi at once, i.e. exactly
  like §9h's signature for a server omitting its intermediate. **It is not one.**
  `pip install -U certifi` fixed it. `sources/ru.py`'s note about Rosstat was retested with the
  same updated bundle and still fails — that one is the genuine article. **The two are
  indistinguishable from the error message and are not the same thing; check the trust
  store's date first.**

## 5. The two authored cells

### The Muslim minority of Western Thrace

**ESS cannot see it.** Among Greek citizens it finds nine weighted Muslim respondents in round
5 and **none at all** in rounds 10 and 11. The minority is 100,000–120,000 people — roughly
29% of Western Thrace — and they are Turkish- and Pomak-speaking, which a Greek-language
national sample does not reach.

The magnitude used is the **Council of Europe's ECRI figure quoted in the US State
Department's International Religious Freedom reports**, midpoint 110,000. It is applied as
spec §3.1's permitted **split**: Anatoliki Makedonia–Thraki's citizen total is unchanged, its
composition is not, and everything else in the region scales down to make room. §14.9 permits
the basis.

**The argument for doing it is what the map says otherwise.** The foreign half alone puts
Muslims in all thirteen regions and puts *fewest* of them in Thrace — 1.8%, the lowest in the
country — because Thrace's Muslims are citizens rather than immigrants. Left uncorrected the
map would state that the historically Muslim region of Greece is its least Muslim one, which
is a sharper falsehood than the silence it replaces.

**With the correction, Anatoliki Makedonia–Thraki is 22.4% Muslim** and every other region is
between 2.2% and 5.8%.

### Mount Athos

NUTS **`ELZZZ`** — "extra-regio", the code for territory belonging to no region — is one LAU
of 1,811 people. It is an autonomous Orthodox monastic republic and only Orthodox monks may
reside there. It is kept as its own counting unit rather than folded into Kentriki Makedonia,
and drawn Orthodox.

**It also caught a modelling error worth recording.** The census counts 125 "foreign
residents" on Athos — the Russian, Serbian, Romanian and Bulgarian houses — and running the
origin model over them put **17.6 Muslims and 13 irreligious people on the Holy Mountain**.
That is what a nationality model does when applied one unit past where it means anything, and
the fix is to exclude the unit from the model rather than to adjust its coefficients.

## 6. The cross-check that validates the whole construction

| | |
|---|---|
| Pew's own country estimate for Greece, 2020 | **5.12% Muslim** |
| this build, bottom-up | **5.08%** |

Two routes sharing no input except Pew's method: one is Pew's Greece row, the other is
Eurostat citizenship counts × Pew *origin* compositions + one minority figure. Nothing was
tuned.

**And it is what settles the coefficient the country turns on.** Albania is **374,917 of
758,597** foreign residents — 49% of them — so one number decides Greece's Muslim total. Pew
puts Albania at 59% Muslim; the literature on Albanian migrants in Greece (Hatziprokopiou and
others) documents Orthodox baptism and name-changing as integration strategies and a
disproportionately southern, Orthodox origin, so the true share for *this stream* is certainly
lower. **It is left at Pew's national figure anyway**, because excluding Albanians gives 2.31%
— less than half Pew's independent estimate — and §14.9's standard is *documented rather than
fitted*. No source publishes a Greece-specific Albanian coefficient, and the cross-check says
the undocumented adjustment would have been the error rather than the correction.

## 7. What Greece is worth drawing for

- **83.9% Greek Orthodox**, and 7.0% who say they belong to no religion — the second largest
  answer, concentrated in Peloponnisos (12.1%), Thessalia (11.8%), Attiki (9.7%) and Ipeiros
  (9.4%) rather than spread evenly. Dytiki Makedonia is the most Orthodox at 95.2%.
- **Anatoliki Makedonia–Thraki at 22.4% Muslim**, against 2.2%–5.8% everywhere else.
- **Notio Aigaio at 10.5% Catholic against 1.1% nationally** — Syros, Tinos and Naxos, whose
  Latin-rite communities have been continuous since the Venetian period, and which the survey
  picks up on its own without any help. It rests on a few dozen respondents.
- **A whole immigrant Orthodoxy that no account of Greece mentions**: Albanian, Romanian,
  Bulgarian, Georgian, Ukrainian and Russian national churches, 155,000 people between them.
- **Mount Athos as a unit of its own.**

## 8. What is left undone

1. **Nothing reaches `secular`.** ESS asks which denomination you belong to and never offers
   atheist or agnostic, so all of Greek irreligion lands on `unaffiliated` and the country is
   dark for `secular` — correctly, per §6.12, but it is a real hole against Spain and Czechia.
2. **The Thracian minority is placed across the whole region.** It is concentrated in Rodopi
   and Xanthi, two of five regional units, and LAU-population weighting spreads it evenly, so
   Kavala and Drama draw Muslim dots they should not. The foreign half *is* available at NUTS
   3; the minority is not, and mixing the two would put the sharper geography on the weaker
   claim.
3. **The Bektashi and Alevi Pomaks are not separable.** No source counts them, so they are
   inside `islam.sunni` and gr2024.py says so rather than rolling them in silently.
4. **Rounds 1, 2 and 4 are unused** for want of a `region` variable. Their country-specific
   region variables may exist under another name; ~7,000 more respondents, from 2002–2008.

## 9. The split-half as a report, 2026-09-14 (ask 012, report only): nothing carries

Anita ruled on `ask/answered/012-be-five-ess-countries-were-drawn-without-the-sp.md` on
2026-09-14: run the test Belgium (§9cy) and Sweden (§9cz) draw with on the five ESS countries
drawn before it, print the tables, move no dots. `python tools/ess_split_half.py gr` reproduces
this section. It imports `sources/stability.py`'s statistic (`median_rho` and `wave_null`;
`be.py::_median_rho` until 2026-09-14) with be.py's alpha (0.05), draw count (2,000) and seed (0),
and adds Sweden's spatial chi-square at 0.05 (`stability.chi2_p`) as the second
requirement. Nothing in `gr.py`, `gr2024.py`, `countries.py` or the built outputs was changed.

The pool is the build's own: Greek citizens, rounds 5, 10 and 11, `rlgdnm` with Refusal, Don't
know and No answer dropped, recoded to the 13 NUTS 2016 regions through `gr.GR_TO_EL`. 7,935
unweighted respondents (7,874 weighted). Three rounds give three splits, each one round against
the other two, with a null that permutes the region labels per round. **Both tests need
unweighted counts and `gr.py` only ever fetched the weighted pass**, so the unweighted one was
fetched with the same break variables as `data/raw/gr/ess_r<N>_n.json`; `gr.py` does not read
it. Mount Athos and the Thracian minority are authored and outside the test. `national` is the
weighted share of citizens who answered, before the Thrace split and without the foreign half,
which is why Orthodox reads 91.50% here and 83.9% in §7.

| category | n | national | median rho | null 95th | p | chi² p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Eastern Orthodox | 7,239 | 91.50% | -0.038 | +0.374 | 0.5757 | 1.9e-29 | national rate |
| Not applicable (no religion) | 573 | 6.95% | -0.066 | +0.396 | 0.6182 | 2.1e-26 | national rate |
| Other Christian denomination | 54 | 0.67% | +0.197 | +0.435 | 0.2389 | 2.2e-07 | national rate |
| Roman Catholic | 33 | 0.42% | +0.538 | +0.568 | 0.0625 | 9.5e-113 | national rate |
| Other Non-Christian religions | 25 | 0.32% | +0.303 | +0.444 | 0.1279 | 1.1e-20 | national rate |
| Islam | 7 | 0.11% | | | | 2.1e-08 | no test possible |
| Eastern religions | 3 | 0.01% | -0.123 | +0.736 | 1.0000 | 2.0e-01 | national rate |
| Protestant | 1 | 0.01% | | | | 5.8e-04 | no test possible |

If the test were applied, all eight categories would move to the national rate, 100% of the
citizens ESS draws: with nothing carrying, each region's residual is the whole region, so every
region would get Greece's national citizen composition, and what still varied across the country
would be the Thracian minority and the foreign half. Islam cannot be tested at all, because it
has no respondent in round 10 or 11 and so every split has an empty half; Protestant has one
respondent.

**The recode was checked first, because a wrong one would look exactly like this.** Round 5
carries NUTS 2006 codes with Latin labels and rounds 10 and 11 carry NUTS 2016 codes with Greek
labels, and a region mapped onto its neighbour would destroy every rank correlation while
leaving every chi-square large. It is not that: all thirteen `GR_TO_EL` pairs name the same
region in both scripts (`GR14 Thessalia` and `EL61 Θεσσαλία`, `GR21 Ipeiros` and `EL54 Ήπειρος`,
and so on down the list).

**The regions differ inside each round, and not the same way from one round to the next.**
Unweighted per-round shares (%) for the regions §7 and `note_public` name, and the ones that
swing most:

| region | respondents r5 / r10 / r11 | no religion | Orthodox | Catholic |
|---|---|---|---|---|
| Attiki | 799 / 947 / 904 | 11.9 / 9.2 / 10.7 | 87.1 / 86.2 / 88.8 | |
| Peloponnisos | 133 / 127 / 138 | 19.5 / 11.0 / 7.2 | 80.5 / 89.0 / 92.0 | |
| Thessalia | 188 / 190 / 191 | 1.1 / **32.1** / 0.5 | 98.9 / **67.4** / 99.5 | |
| Ipeiros | 105 / 78 / 75 | 7.6 / 17.9 / 2.7 | 90.5 / 80.8 / 97.3 | |
| Ionia Nisia | 62 / 47 / 52 | 0.0 / 19.1 / 0.0 | 100.0 / 80.9 / 98.1 | |
| Voreio Aigaio | 51 / 42 / 42 | 0.0 / 0.0 / 21.4 | 100.0 / 97.6 / 76.2 | |
| Dytiki Makedonia | 86 / 66 / 72 | 1.2 / 4.5 / 0.0 | 96.5 / 95.5 / 100.0 | |
| Notio Aigaio | 69 / 93 / 55 | 0.0 / 5.4 / 5.5 | 75.4 / 88.2 / 94.5 | 24.6 / 6.5 / 0.0 |

Four readings:

1. **Thessalia's no-religion share is one round.** 61 of its 64 no-religion respondents are from
   round 10, where 32.1% of 190 people said they belong to no religion; rounds 5 and 11 found 1.1%
   and 0.5% of about as many. Sampling error on 190 people at the pooled 11% is about 2.3 points,
   so that is far outside it, and it is the shape of a few sampling points per region per round
   rather than of a region. This API returns no PSU, so that last part is inference. Ionia Nisia
   (all 9 from round 10) and Voreio Aigaio (all 9 from round 11) are the same shape. **Attiki is
   the one that recurs** (11.9, 9.2, 10.7), and Peloponnisos falls steadily from 19.5 to 7.2.
2. **The Cycladic Catholics are real, and this pool cannot replicate them.** Notio Aigaio is 24.6%
   Catholic in round 5, 6.5% in round 10 and none of 55 in round 11, so 17 of its 23 Catholic
   respondents are from 2010. Syros, Tinos and Naxos have their Latin-rite communities whatever
   the survey finds, so this failure (p = 0.0625, 8 of 13 regions empty) is a failure to
   demonstrate a true geography, which is exactly what §9cy says a failure means. It is the case
   that makes ask 013 a real trade and not a cleanup.
3. **Every chi-square but one passes, and the chi-square cannot be the gate here.** The chi-square
   treats the 7,935 as independent draws; a round's regional subsample drawn from a few sampling
   points is not independent, so the test sees that the thirteen cells differ and cannot see
   whether they will differ the same way next round. Sweden added the chi-square to refuse small
   categories the rank test let through. Greece is the other direction, the rank test refusing
   large categories the chi-square lets through, and between them they are why the rule needs
   both.
4. **Three splits, each one round against two, is the thinnest version of this test that runs.**
   Sweden's §2 showed that three halvings can disagree, and a median of three is only a little
   better than one. So this is weak evidence. It is weak evidence pointing one way, with Orthodox
   and no religion both near zero rather than just under the bar.

**`note_public` makes three regional claims that rest on these categories**: no religion
"concentrated in Attiki, Peloponnisos and Thessalia", Dytiki Makedonia "the most Orthodox", and
Notio Aigaio "10% Catholic". Attiki recurs, Peloponnisos is a trend, Thessalia is one round,
Dytiki Makedonia is 6th, 4th and 1st by Orthodox share in the three rounds, and the Cyclades are
2010. Filed as **ask 013**, the one ask from this pass; the other four countries' tables are in
their own files and none of them is alarming. Nothing was changed.
