# Liberia — 2022 census religion, given a county geography by the Afrobarometer

**Drawn 2026-09-08.** 15 counties, 5 categories, 5,250,187 people, every row `modelled`.

- `sources/lr_geo.py` -> `data/geo/lr/lr_counties.gpkg`, `lr_lookup.csv` (COD-AB ADM1 +
  census populations)
- `sources/lr_grid.py` -> `data/geo/lr/lr_hexes.gpkg` (Kontur 400 m, keyed to county)
- `sources/afrobarometer.py` -> `data/raw/afrobarometer/*.sav` (six merged rounds, ~280 MB)
- `sources/lr.py` -> `data/normalized/lr.csv`
- `taxonomy/lr2022.py` -> the mapping; `countries.py` `"lr"` -> the wiring
- sources.md **§9cl** is the write-up, **§11ai** assesses the Afrobarometer for the rest of
  Africa, and **§11w** is the sweep whose Liberia row this closes.

```
python sources/lr_geo.py  --fetch
python sources/lr_grid.py --fetch
python sources/lr.py      --fetch      # once; the six .sav files are shared with every
python sources/lr.py                   # other Afrobarometer country
```

## 1. What LISGIS publishes, and it is one number

**Liberia has asked religion in every modern census and has never published it below the
national line.** That is the finding, and everything else here follows from it.

| release | what it has | tier |
|---|---|---|
| **2022 PHC, *Final Results*** (LISGIS, 2023) | Table A13, five categories, national, by sex | national |
| 2022 PHC, 15 **thematic reports** (Aug 2024) | *Population Size, Distribution and Structure* repeats the national five, against 2008 | national |
| 2008 PHC, *Final Report* (NPHC 2008) | table 4.3, religion by age and sex | national |
| 2008 analytical monographs (May 2012) | *Population Size and Composition* §4.4, religion by urban/rural | national + urban/rural |
| **2011 Census Atlas** | Fig 9-58, a county map of the **urban** population by religion, no table | county, qualitative, urban only |
| UNSD Demographic Yearbook table 28 | the 2008 national row, 5 categories | national |

The 2022 report's other appendix tables give residence, household size, sex of head,
education, literacy and employment **by county**. Religion is the one that stops at Liberia.

**The five categories are real and not an artefact.** The queue priced Liberia at five on the
oracle's 2008 row, and the 2022 census is five as well, defined in the report's own §2.20:
*Christians: all Christian denomination churches; Islam: all Islamic denominations;
Traditional Religion: includes worship of deities and ancestors; Other: consists of religions
other than those captured above (examples include Eckankar, Baha'i, Shintoism); No Religion.*
This is not Cabo Verde's case, where a sixteenth row turned out to be the under-15s: nothing
here is a mis-read row. Liberia's census card has five boxes.

## 2. What was searched, and where each route ended

`lisgis.gov.lr` **now serves a one-page placeholder** — *"A new lisgis.gov.lr is on its way"*,
IIS/ASP.NET, no catalogue — and every deep path from the old PHP tree 404s, including the
report URLs Google still indexes. The old files are in the Wayback Machine and were taken from
there. **The predecessor domain `lisgis.net` is worse than dead: it is a redirect loop behind
Cloudflare**, and its `pg_img/` tree (the 2008 census final report, the 2011 atlas, the 2012
monographs, fifteen county maps) exists only as archived copies.

A CDX prefix sweep of both hosts, filtered to documents, returns 280 files for `lisgis.gov.lr`
and 171 for `lisgis.net`. Every census-shaped one was opened.

| route | outcome |
|---|---|
| 2022 *Final Results*, all 92 pages | religion on pp. 35, 60, 78; national only |
| 15 thematic reports (Aug 2024) | **no religion volume**; *Population Size* has the national chart and nothing else |
| 2008 *Final Report*, 352 pages | one religion table, national by age and sex |
| 2008 monographs (`Population size 210512.pdf`, `Administrative report final 210512.pdf`) | religion by urban/rural; the administrative report mentions it once, about escorting enumerators |
| 2011 Census Atlas (86 pp, also on **tile.loc.gov** as `gdcebookspublic/…/2019667184.pdf`) | one religion figure, Fig 9-58, urban, county, qualitative |
| the fifteen per-county PDFs on `lisgis.net` (`Bomi.pdf`, `Bong.pdf`, …) | **they are maps, not reports** — the tier-below-the-tier trick does not apply here |
| IHSN catalog 4325 (2008 census) | five documents, all questionnaires and manuals; `Population_by_County.pdf` has no religion; access routed to the Ministry of State |
| **`microdata.lisgislr.org`** (LISGIS's own NADA) | 17 studies, and its variable search finds religion in **DHS 1986/2007/2013/2019-20 and MIS 2008/2011/2016 only** — HIES 2014-15, HIES 2016 and the 2024 agriculture censuses have no religion variable. **Every download is behind a free account**, so it is not used; the 2008 census entry is an IPUMS pointer, and IPUMS is `[[reference_ipums_account]]`. It returns **406 to a bare curl and 200 to a browser UA** ([[reference_overpass_user_agent]]). |
| USCB per-country geodatabases on HDX | 34 datasets, **Liberia is not one of them** |
| `liberia.opendataforafrica.org` (Knoema) | 403 to WebFetch and to curl with a browser UA; not pursued further, since the census route was already exhausted |

**So the only open sub-national religion data for Liberia is a survey**, and the Afrobarometer
is the one that needs no account.

## 3. The construction

    row margin      county populations       2022 census Table A4       EXACT
    column margin   the five religion totals 2022 census Table A13      EXACT
    the interaction the county pattern       Afrobarometer R4-R9        measured, n=7,163

Both census margins sum to **5,250,187**, so the 15 x 5 table is fitted to them by IPF and the
survey supplies only the interaction. No magnitude is invented (spec §14.4 rule 1). It is still
`modelled` in §7's sense and not `derived`, because §7b's rule is whether anybody was counted
and nobody counted the cell.

**The three categories that do not carry a geography are seeded flat**, not out of a residual.
`ab.build`'s residual scheme refuses Grand Cape Mount, whose 240 pooled respondents are all
Christian or Muslim and whose residual is exactly zero; with the column margins supplied by the
census there is no need for a residual at all.

## 4. The checks

| check | result |
|---|---|
| **decode** (respondent share per county vs census population) | **r = +0.996** over 15, and **0 of 20,000** random pairings reach it |
| **split-half** (§14.16), bar +0.524 on 15 units | Christian **+0.789**, Muslim **+0.756**; No religion **+0.114** FAILS; Traditional 0.45% and Other 0.46% under the 1% floor |
| **the atlas** — the one published county statement | the survey makes exactly one county Muslim-majority and it is **Grand Cape Mount**, which is what LISGIS wrote in 2011 |
| **survey vs census national margins** | Christian 1.04x, Traditional 0.93x, **Muslim 0.81x**, **No religion 0.48x**, Other 7.09x |
| county totals after rounding | every one exactly its census population; per-category drift +2, +3, -2, -2, -1 people |

The margin ratios are the argument for the fit rather than an objection to it: drawn on the
survey's own levels Liberia would be 9.7% Muslim against a census that counted 11.98%.

## 5. THE DENOMINATIONS ARE IN THE FILE AND ARE NOT DRAWN

This is the one arguable call here and it is worth reversing deliberately if it is reversed.

Afrobarometer's card names twenty Christian bodies that Liberians actually chose, and pooled
over six rounds they read:

    Pentecostal 9.6%   Methodist 7.0%   Lutheran 5.8%   Baptist 5.3%   Roman Catholic 4.3%
    Evangelical 1.9%   Presbyterian 0.9%   Seventh Day Adventist 0.9%   Jehovah's Witness 0.7%
    Church of Christ 0.6%   Independent 0.5%   Anglican 0.4%   Orthodox 0.3%   + seven more

**No Liberian census has ever published any of that**, and it is exactly the kind of split this
map exists for. It is not drawn because the share who answer `Christian only` instead of naming
a denomination is set by the fieldwork:

    R4 2008-12  28.0%    R5 2012-06  44.4%    R6 2015-05  23.1%
    R7 2018-06  67.4%    R8 2020-10  72.0%    R9 2022-08  61.6%

A 48.9-point range, not monotone in time, in a country whose Christian total moved by four
points over the same fourteen years. A pooled denominational share is therefore a measurement
of which rounds are in the pool. Grouping up to the census's five removes the effect entirely,
because a Methodist and a `Christian only` are both Christian in every round.

`sources/lr.py` **asserts the swing is still there on every build** (`by_round.max() -
by_round.min() >= 0.15`), so a future Afrobarometer release with consistent probing fails the
build and forces the decision to be re-taken rather than leaving this in place by inertia.

**What would settle it**: any Liberian source that gives a denominational total, even
nationally. The Liberian Council of Churches and the individual bodies publish membership
figures; none was found in a form that could be cited, and none was chased hard, because the
national margin alone would not fix the round-mix problem in the county pattern.

## 6. Gotchas

- **`REGION` is 33 strings for 15 counties.** `Rivercess`/`River Cess`, `Bassa` for Grand
  Bassa, `Cape Mount` for Grand Cape Mount, and round 9 upper-cases everything.
  `[[reference_pooled_survey_labels]]`, and `NORM` in `sources/lr.py` is the fix.
- **The religion variable moves name every other round**: Q90, Q98A, Q98A, **Q98**, Q98A, Q95.
  Round 7 has a `Q98A` and it is not religion, so a fallback lookup would read the wrong
  column silently. `afrobarometer.py` names it per round and asserts the variable LABEL.
- **The weight changes name and meaning at round 8**: `withinwt` becomes `withinwt_ea` and
  `withinwt_hh`, and `Combinwt` is the cross-country one that must not be used. Asserted by
  requiring the weight to sum to the respondent count.
- **Round 6's .sav is not valid UTF-8** and `pyreadstat.read_sav` raises on it; `encoding=
  "LATIN1"` reads it.
- **`Jehovah's Witness` arrives with two apostrophes** across rounds, which is the one
  `assert_one_wording` clash Liberia has. Resolved by the grouping, not folded silently.
- **COD-AB spells the county `Rivercess` and LISGIS spells it `River Cess`.** The map uses the
  census's spelling. COD's pcodes are alphabetical by COD's spelling, so `River Gee` is LR13
  and `Rivercess` LR14; the pcodes are not used as a key.
- **River Cess draws zero Muslims**, because none of its 142 pooled respondents was one. §3.5
  drops rather than invents; the 95% upper bound on that sample is about 2%.
- **The 2022 report's Figure 3.5.2 contradicts its own Table A13 and must not be quoted.** It
  plots the 1984/2008/2022 trend and puts traditional religion at 18%, 2.3% and 3.2%. Table
  A13 and Figure 3.5.1, in the same report, say 0.5% for 2022 (25,445 people), and UNSD's 2008
  row says 0.58% (20,134). Two of the figure's three columns are wrong, so its 1984 column is
  not usable either. Nothing in this build cites it. The same page's prose has a second slip:
  it says Christians were 87% in 2008, where the 2008 census's own table is 85.6%.

## 7. Terms

Afrobarometer's merged rounds are linked as plain files on `afrobarometer.org` with no form and
no account. Its data usage and access policy gates only early access and the geocoded extracts,
and asks for the citation *"Afrobarometer Data, [Country(ies)], [Round(s)], [Year(s)],
available at http://www.afrobarometer.org"*. Read before anything was downloaded (§11ac).

The census figures are LISGIS's own published tables, transcribed from the *Final Results* PDF;
`CENSUS_2022` in `lr_geo.py` and `CENSUS_RELIGION` in `lr.py` are the transcriptions and each
is asserted against the report's own national total.

## 8. §14 was considered and no ask was filed

Liberia's civil wars had a communal dimension and Mandingo Muslims were targeted in them, so
the question is worth asking rather than skipping. It was not escalated, for three reasons
taken together, and they are recorded here so the call can be reversed rather than re-derived:

1. **The tier is coarse.** Fifteen counties, 350,000 people on average, is the geography
   LISGIS itself reports everything else on, and it is coarser than the governorates Anita
   ruled acceptable for Egypt (`ask/answered/001-eg`).
2. **The state publishes the national figure itself and has since 1984**, and it is the state
   supplying both margins here. This is not a case of mapping a minority finer than its own
   government will.
3. **The pattern is already published by the same office.** The 2011 Census Atlas names Grand
   Cape Mount as the Muslim-majority county in print. Nothing on this map locates a group that
   LISGIS has not located first.

What would change the answer is a finer tier: the 136 districts would be a different question,
and nothing here can reach them anyway.

## 9. Review, 2026-09-08

A second pass, run against the primary material rather than against the sections above. The
country holds up; the three findings are all in the shared module or in one sentence of the
note, and two of them generalise beyond Liberia.

**The IPF does what §3 says it does, checked independently.** IPF preserves the seed's
cross-product ratios exactly, so the test is not a correlation but an identity: over the
fourteen counties with any Muslims, the drawn Muslim:Christian odds divided by the survey's
own run **1.39724 to 1.39743**, one constant to five figures with the variation being nothing
but the largest-remainder rounding. So the drawn geography is the survey's, moved by one
national factor per religion and by nothing else; Spearman between the survey's county Muslim
share and the drawn one is **+0.9964**. The margins are met — every county exactly its census
population, the five religion totals within three people of Table A13 from rounding inside
rows. The decode reproduces at **r = +0.9960**, 0 of 20,000 pairings. The split-half
reproduces: Christian +0.789, Muslim +0.756, No religion +0.114 against the +0.524 bar.

**One thing the split-half table does not show, worth knowing rather than acting on.**
Traditional African Religion is excluded by the 1% eligibility floor and never reaches the
test; run anyway it scores **+0.752**, well over the bar. The floor is §11ad's and applying it
first is right, but it means Liberia draws flat a category that does replicate. Nothing to
change: at 0.45% the floor exists precisely because that instrument was measured failing below
it, and 25,445 people over fifteen counties would be a thin claim either way.

**A CATEGORY SEEDED AT THE NATIONAL RATE IS NOT DRAWN FLAT, AND THE `note_public` SAID IT
WAS.** The note read *"Irreligion is drawn flat, at 2.56% in every county"*. In
`data/normalized/lr.csv` it runs **2.013%** in Grand Cape Mount to **2.737%** in Nimba. The
cause is the fit, not a bug: the IPF solution is `f[u,c] = a_u * b_c * seed[u,c]`, and the row
factor `a_u` varies with each county's mix of the two categories that DO carry a geography, so
a flat seed emerges as a mirror of them — irreligion correlates with the Muslim share at
**-0.98** across the fifteen counties. The build's own by-county table prints the 2.0-2.7%
spread, so this was visible and read past. The sentence was corrected to *at the national
rate, near 2.56% everywhere*, with the residue named; the mechanism went into spec §12,
because §12 now recommends this construction to every country whose office publishes religion
nationally and nothing finer.

**The probing swing is not a Liberian fact, and Botswana is worse.** Recomputed over all six
rounds and every country in them: the `Christian only` share swings more than fifteen points
in **ten of the twenty-eight countries with three or more rounds**, median swing 10.2 points,
Botswana **65.7** points against Liberia's 48.9. It is not a round effect that a per-round
factor could remove — between R7 and R8 Zambia falls 23.1% to 3.7% while Kenya rises 24.8% to
30.7% — so it is fieldwork rather than release, and §11ai's rule is right for the continent
and not only here. The table is in §11ai and in `sources/afrobarometer.py`'s docstring.

**The shared module lost whole rounds for three countries and would not have said so.**
`ab.load()` selected on one exact country name, and `COUNTRY` is a per-round label set exactly
as `REGION` is: **Côte d'Ivoire** arrives as `Cote d’Ivoire` / `Cote d'Ivoire` / `Côte
d'Ivoire`, **Eswatini** as `Swaziland` / `eSwatini` / `Eswatini`, **Cabo Verde** as `Cape
Verde` before R7. Asking for the current name got three rounds of six with no error and no
missing unit, which given the paragraph above is the worst possible way to lose respondents:
`Cabo Verde` returned 3,587 from R7-R9 instead of 7,216 from all six. Fixed in the module with
`COUNTRY_ALIASES`, a `fold()` match, a printed note for any round that returns nothing and a
hard failure when a round holds a label differing only by an accent. Liberia is unaffected and
rebuilds byte-identical.

Small: `sources/lr.py` and `sources/afrobarometer.py` both pointed at §11ag, which is the
Gemini lead triage; the Afrobarometer assessment is **§11ai**, as this file already said.
