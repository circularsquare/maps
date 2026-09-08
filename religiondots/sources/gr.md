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
