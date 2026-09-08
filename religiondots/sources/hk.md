# Hong Kong — 2021 census ethnicity by district, and one published survey

Built 2026-09-08, spec §14.24, as **mainland China's sibling and in the same two layers**.
`sources/hk.py` rebuilds `data/normalized/hk.csv` and `hk_survey.csv`; `sources/hk_geo.py`
writes the 18 district polygons and `sources/hk_grid.py` the Kontur hexes. `data/` is
gitignored, so this file is the record.

| | |
|---|---|
| structure | 2021 Population Census, ethnicity by District Council district (C&SD `DC_21C.CSV` and the *Thematic Report: Ethnic Minorities*) |
| religion | *Hong Kong Political Culture Survey 2021*, Cai and Hung, published as Table 1 of *Religion and Trust in Hong Kong*, The China Quarterly **257** (2024), 609–628 |
| geography | **18 District Council districts**, mean 412,000 people |
| basis | `ethnicity_derived` for three migrant nationalities; `self_id` for everyone else |
| tier | **`derived` 64.8%, `modelled` 35.2%; nothing is `measured`** |
| licence | C&SD publications are Crown/HKSAR government works, free to reproduce with acknowledgement; the China Quarterly paper is open access |
| drawn | **7,411,948 of a census 7,413,070**; the 1,125 not drawn are the marine population, which belongs to no district |

## 1. Hong Kong has never asked either, and that is the whole reason this is hard

The 2021 Population Census publishes the **46 topics** it covered and religion is not among
them; nor was it in 2016, 2011 or any round before. Hong Kong does not appear in UNSD table
28. So this is China's problem a second time and it gets China's answer: an ethnic derivation
where the category and a religion are the same object (spec §14.5), and a self-identification
survey carved out of the grey that remains (§14.16).

`taxonomy/hk2021.py` argues each category and is the file to read before touching anything
here.

## 2. What is deliberately NOT drawn, and this is the most important section

**`gov.hk`'s *Hong Kong: The Facts — Religion* is where every published figure about religion
in Hong Kong comes from, and it must not be used.** It gives over 1 million Buddhists, over 1
million Taoists, 1,040,000 Protestants, 390,000 Catholics, 300,000 Muslims, 100,000 Hindus and
15,000 Sikhs. **Every one of those is the religious body's own estimate of itself** — the sheet
says so explicitly for Islam (*"according to the Incorporated Trustees of the Islamic Community
Fund of Hong Kong"*) and for Sikhism (*"The Khalsa Diwan estimates"*).

That is not a basis this map has anywhere, and spec §3.1 forbids mixing it with the two that
are here. But the decisive thing is that **it fails the only external checks available, and
fails them in both directions**:

| | gov.hk | what can be checked against it |
|---|---|---|
| Protestants | **1,040,000** (Jan 2026) | the *same office* said **480,000** in July 2022, quoted from SAR government statistics in the US State Department's 2023 IRF report; and Hong Kong's own churches counted themselves in the 2024 Hong Kong Church Survey and found **255,091 congregants, 197,935 at weekly worship, down 26% in five years** |
| Muslims | 300,000 | the census's ethnic counts give ~148,000 and the survey ~176,000 |
| Hindus | 100,000 | the census's *upper bound* is ~72,000 and the survey says ~44,000 |

**The government's Protestant figure more than doubled over exactly the period in which the
only measurement of it fell by a quarter.** And for Islam and Hinduism, a census of 7.4 million
and a survey of 3,740 with no lineage in common agree with each other to within about 20% and
disagree with the official number by a factor of two. When two independent sources agree and a
third does not, the third is the one to leave out.

## 3. The three sources, all open and none gated

| # | what | where |
|---|---|---|
| 1 | **`DC_21C.CSV`** — exact counts per district: total population, and ethnicity as Chinese / Filipino / Indonesian / White / Others | `census2021.gov.hk/doc/DC_21C.zip`, 20 KB, linked from data.gov.hk |
| 2 | **Thematic Report: Ethnic Minorities** — Table 3.1 (each ethnicity's territory total) and Table 8.1 (each ethnicity's % distribution over the 18 districts) | `census2021.gov.hk/doc/pub/21c-ethnic-minorities.pdf`, 6 MB |
| 3 | **Hong Kong Political Culture Survey 2021** — 3,744 respondents, eight religion categories | China Quarterly 257 (2024) Table 1, open access |
| geo | **18 district boundaries**, WGS84 GeoJSON, one file, 18 features | `had.gov.hk/psi/hong-kong-administrative-boundaries/hksar_18_district_boundary.json` |

**No microdata is needed and none was sought.** The survey's published table *is* the
tabulation, which is how Guatemala's and El Salvador's LAPOP shares are used (§11ad).

### The survey's design, which is why its small cells are not trusted for geography

Two-stage cluster sample: **72 of Hong Kong's 452 electoral districts** drawn at random, then
52 residents in each, quota-matched on the 2016 census's sex, age, education, income and
housing type. 30-minute tablet interviews, May–September 2021, **aged 16 and over**. The
authors say themselves that *"the number of followers of Islam, Hinduism, Sikhism and other
religions was limited"* — 89, 22 and 2 respondents.

Its eight categories sum to **exactly 100.00%** and to exactly 3,740 respondents, so the table
is a partition both ways and needs no residual of its own. `hk.py` asserts both.

## 4. The two layers, and the arithmetic

### Layer one: the ethnic derivation, at 18 districts

Three nationalities carry a fractional share of one religion; the remainder of each goes to
`unknown`, and **both halves are `modelled`** — §14.9's shape, exactly as China's six Yunnan
border peoples are handled.

| nationality | 2021 census | → | share | drawn |
|---|---:|---|---:|---:|
| Indonesian | 142,065 | `islam` | 87.51% | 124,321 |
| Pakistani | 24,385 | `islam` | 96.47% | 23,524 |
| Filipino | 201,291 | `christianity.catholic.latin` | 78.88% | 158,778 |

**The coefficients are this map's own countries and not a new source.** Indonesia, Pakistan
and the Philippines are each drawn here from their own censuses, so Hong Kong's Indonesians
get the Muslim share of Indonesia *as this project already draws it*. It cannot drift away from
the rest of the map, because it **is** the rest of the map, and it inherits every later
correction to the source country. Recompute any of them with
`countries.COUNTRIES[cc]["counts"]()` grouped on node.

**That fixes the provenance of the number and does nothing for whether it applies, which is
the part that matters** — Anita, 2026-09-08: *"i think doing diaspora coefficients from maps
own drawn countries is probably pretty bad for large origin countries cuz the people migrating
are probably skewed in some way."* Exactly so, and it is §14.12: migration selects on region,
class and ethnicity, which are the axes religion varies on. spec §14.24 turns that into three
conditions, at least one of which has to hold before a national share may be laid over a
migrant stream. All three hold here, which is why these rows are drawn and the Indian and
Nepalese ones are not:

| | condition | |
|---|---|---|
| Pakistani | **the origin share is near 1**, so no selection moves it much | 96.47% |
| Indonesian, Filipino | **the direction of selection is known and stated** | both are floors; see §6 and `hk2021.py`'s REVIEW |
| Islam overall | **there is an independent check on the result** | the survey, within 20% |

### Layer two: the survey, for the territory, carved out of `unknown` only

| answer | share | respondents | → |
|---|---:|---:|---|
| Buddhism | 13.72% | 513 | `buddhism.mahayana` |
| Protestant | 9.11% | 341 | `christianity.protestant` |
| Catholic | 4.28% | 160 | `christianity.catholic.latin` |
| Taoism | 4.04% | 151 | `daoism` |
| Islam | 2.38% | 89 | **not drawn — see below** |
| Hinduism | 0.59% | 22 | `hinduism` |
| Sikhism | 0.05% | 2 | `sikhism` |
| No religion | 65.83% | 2,462 | stays `unknown` |

**Only the `unknown` residual is carved**, which is `_cn_counts`'s arithmetic exactly and for
the same reason: applying the survey's shares to the whole population would count an
Indonesian domestic worker as Muslim once from the census and again from the survey. The
derived population is 4.1% of Hong Kong, so the cost of carving the residual alone is small
and runs the other way.

### Islam comes from the census and not from the survey

China's rule for the same category, for the same reason. 89 Muslim respondents in a
72-cluster design carry no geography; the census counts 142,065 Indonesians and 24,385
Pakistanis **exactly and by district**. The two agree on the magnitude to within 16% —
147,845 against 176,431 — which is what makes either of them believable, and only one can say
where. Its share is left inside the residual rather than reallocated over the other six,
which understates them by about 0.8 percentage points between them.

## 5. The checks, and none of them is a tolerance

**Table 8.1 is parsed out of a PDF** whose rows print ten numbers *before* the district's
name, so a slipped column would be silent. Two independent things have to hold:

- **Its Filipino and Indonesian columns must reproduce `DC_21C.CSV`'s exact counts.** Those two
  ethnicities are in both sources, so the percentages are recomputable. **Worst disagreement
  over 36 comparisons: 0.05 pp.**
- **Its South Asian `Overall` column must equal the weighted mean of its own four South Asian
  columns**, weighted by Table 3.1's totals. **Worst over 18 districts: 0.07 pp.**

Also asserted: every district's five ethnicity columns sum to its population **to the person**;
the seven named groups fit inside `DC_21C`'s lumped `Others` in every district; the survey's
eight shares sum to 100.00 and its counts to 3,740; and HAD's district letter codes match the
census's independently (`hk_geo.py` fails if either side ever renumbers).

**The grid correlates with the census at r = 0.87 over 18 districts, band 0.60–1.61, and that
is a pass.** Kontur is built from building footprints and Hong Kong is the hardest place on
earth for that: it reads a 40-storey housing estate much as it reads a village. So it
undercounts the vertical districts (Wong Tai Sin 0.60×, Sham Shui Po 0.75×) and overcounts the
spread-out ones (North 1.61×). **A scrambled join has no such pattern** — it pairs a large
district with a small one and the correlation collapses. And this check cannot affect the
output anyway: dots per district come from the census, and Kontur only decides which street
inside a district they land on.

## 5a. Nothing here is `measured`, so `inferred dots: not shown` empties the country

64.8% `derived`, 35.2% `modelled`, **0 measured** — and turning on `inferred dots: hidden`
removes all of Hong Kong, exactly as it removes all of China. §14.6 calls that the honest test
of a country whose census never asked, and it is worth performing here for the same reason.

**The `unknown` rows are `derived` rather than `measured`, and that is a judgement call worth
knowing about.** China's are `derived` because of §3.4: its county figures are a 2000 count
carried onto 2010 provincial totals, so the *placement* is inferred even though the people are
counted. **Hong Kong has no such carry** — the district populations and the ethnicity are both
2021, straight from one census. So an argument exists that a grey Hong Kong dot is a counted
person in a counted place, claims nothing, and should be `measured`; under it, the toggle would
leave Hong Kong grey while emptying China, which would truthfully say *these people are counted
where they are and those are placed by a 25-year-old structure*.

It is left at `derived` for now, which under-claims rather than over-claims, and because a
`measured` row that measures **population** rather than **religion** would be the first on this
map. Anita's call if it is ever wanted; `tools/check_rollup.py hk` prints the 4.8M as orphaned
and asks the question directly.

## 6. What to distrust

**The survey is one wave, so §14.10's split-half test cannot be run.** Guatemala and China both
decide which categories may carry a geography by testing rank stability across waves. Here
there is nothing to test — but there is also **nothing to defend**, because the survey layer is
territory-wide and makes no geographic claim at all. The test exists to police a between-unit
ordering; with one unit there is no ordering.

**Sikhism rests on 2 respondents and Hinduism on 22.** They are drawn anyway, at territory
grain, which is Guatemala's rule (§9bi): nobody is deleted, and only the claim to know *where*
they are is withdrawn. Read them as *this community exists and is small*.

**The Indians and Nepalese are counted and left grey, and that is the largest deliberate
absence here.** 42,569 and 29,701 people, and both are §14.12 cases rather than oversights:
Hong Kong's Indian community is disproportionately Sindhi Hindu and Punjabi Sikh rather than a
cross-section of India, and its Nepalese are the families of Gurkha soldiers, recruited from
hill peoples (Gurung, Magar, Rai, Limbu) far more Buddhist and Kirat than Nepal's 81% Hindu
average. **The selection runs along exactly the axis the coefficient would need to be stable
on.** Thai is the close call and is argued in `hk2021.py`'s REVIEW.

**The survey is 16+ and is applied to the whole population**, which assumes children's
households resemble the adults'. CGSS (18+) is used the same way for the mainland.

**The Indonesian coefficient is a floor.** Hong Kong's Indonesian residents are ~93% domestic
workers recruited overwhelmingly from Central and East Java, which is more Muslim than
Indonesia as a whole, so the true share is probably above 95%. It is left at the documented
national figure rather than adjusted upward, because §14.12's lesson is that the adjustment
which feels more careful is usually the error.

## 6a. Hong Kong is where Daoism gets drawn at all

**287 of the 302 Daoism dots on this entire map are Hong Kong's.** The other fifteen are
diaspora counts in Australia, Canada, the United Kingdom and New Zealand, whose censuses ask a
religion question with a Daoist box on it. **Mainland China draws none**: CGSS *does* ask, and
80 respondents in 32,495 said Daoism, which §14.16 judged far too thin to place, so China is
unlit for a religion it certainly contains.

So a reader who selects Daoism sees Hong Kong and a scatter of migrant communities, and that is
an honest picture of *where the question has been asked* rather than of where Daoists are. It
rests on 151 respondents here, which is thin in absolute terms and much the best available.

## 7. The two findings worth carrying

**The derived layer's geography is an employment geography, not an enclave geography, and that
was not the expected answer.** The Muslim share runs 3.00% (Wan Chai) to 1.51% (Kwun Tong) and
the Catholic share 11.33% to 4.95%, with both **highest in the wealthiest districts on Hong
Kong Island**. There is no Muslim quarter in Hong Kong on this map because most of the people
concerned are live-in domestic workers, so what the census's ethnicity column locates is the
households that employ them. Yau Tsim Mong holds 42.3% of Hong Kong's Nepalese and 18.8% of
its Indians but only 4.9% of its Pakistanis; the enclave geography that does exist belongs to
the nationalities this map refuses to derive from.

**Hong Kong measures China's §14.22 gap in one instrument, which the mainland cannot.** Of the
65.83% reporting no religious affiliation, the same table records that **2,097 respondents —
56.07% of the whole sample — practise folk religion anyway**. So most of Hong Kong's "no
religion" is people who tend graves, burn incense, draw fortune sticks and pick auspicious
dates, and will not call any of it a religion. China has to borrow that finding from Pew and
from the 2007 Spiritual Life Study; here one survey asked both questions of the same people.
It is why nothing in Hong Kong is drawn on `chinesefolk`: **what that number measures is
practice, and §3.1 forbids mixing it with the naming layer beside it.** The 1988 and 1995
surveys the same table prints did offer folk religion as an affiliation and found 23.0% and
15.3%; the 2021 instrument dropped the answer box.

## 8. THE BIGGEST WEAKNESS IS SPATIAL, AND IT IS THE THING TO FIX NEXT

Anita, 2026-09-08, on the finished country: *"hong kong being 1 spatial component is pretty
bad, so we should mark that theres definitely room for improvement here if future agents want
to pick it up."* **She is right.** Hong Kong is registered at 18 districts and almost nothing
varies across them:

| layer | share of each district | varies by district? |
|---|---:|---|
| the survey (Buddhism, Protestant, Catholic, Daoism, Hinduism, Sikhism) | 31.8% | **no — identical everywhere** |
| the ethnic derivation (Islam, part of Catholicism) | 4.1% | yes, but only 1.51%–3.00% Muslim across all 18 |
| `unknown` | 65.4% | only as the two above move |

So a reader moving around Hong Kong sees the same mixture everywhere and the 18 districts do
almost no work. §3.9b removed the granularity floor so a coarse country is drawn rather than
skipped, but **coarse is a fact to state, not a resting place**, and this is the least
spatially informative country of its size on the map. **Read Hong Kong as a national pie chart
with a population-weighted scatter**, which is what it is.

### Four routes, in order of what they would buy

1. **A survey with district-level religion** — the one that matters, because it would give the
   31.8% a geography instead of a constant. The Hong Kong Panel Study of Social Dynamics
   (HKPSSD, HKU) and the Asian Barometer both carry religion and finer location, and both are
   behind an application, so §11b's rule applies: attempt only once the open routes are done.
2. **The 2024 Hong Kong Church Survey's district tables** — 1,318 congregations and their
   attendance by district, which would give Protestantism a real geography on a `congregations`
   basis. The report is a paid publication and only the summary figures used in §2 are public;
   worth an email to the Hong Kong Church Renewal Movement.
3. **The Catholic Diocese of Hong Kong's parish statistics**, the same shape for Catholicism.
4. **452 District Council constituency areas**, with boundaries and population on data.gov.hk.
   Not used because the ethnicity detail the derived layer needs exists only at the 18
   districts, so adopting them would mix grains inside one layer. Useful only alongside (1).

## 9. What else is open

- **A folk-religion affiliation figure for 2021.** The category exists in the 1988 and 1995
  surveys printed beside this one and would be drawable on the same basis. Whether to carry
  15.3% forward across thirty years is Anita's call and the answer is probably no.
- **Macau and Taiwan are the same gap.** `cn_geo.py`'s `NOT_MAINLAND` excludes all three and
  none was drawn before today. **Taiwan is much the larger prize**: it has a religion-carrying
  social survey with county geography and a Ministry of the Interior register of religious
  bodies, and its census does not ask either.
- **Thai → Theravada** is the one refused derivation that is a close call, argued in
  `taxonomy/hk2021.py`'s REVIEW. It is 12 dots.
