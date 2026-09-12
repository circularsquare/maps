# Singapore — Census of Population 2020

Built 2026-09-08. Drawn: 9 categories on 31 planning-area units, 3,459,094 people.

Files: `sources/sg.py`, `sources/sg_geo.py`, `taxonomy/sg2020.py`, `countries.py["sg"]`.
There is no `sg_grid.py`; see §3.

---

## 1. The source, and it is completely open

**SingStat TableBuilder table CT/17592**, republished on data.gov.sg as dataset
`d_a58564fbed922609a0f79af96069dd9b`, *Resident Population Aged 15 Years and Over by Planning
Area of Residence and Religion (Census of Population 2020)*. Singapore Open Data Licence, free
for commercial use, no key, no login, no bot wall. 2,136 bytes.

**The download URL is signed and expires**, so there is nothing stable to hard-code. The
pattern is a two-step: `GET
https://api-open.data.gov.sg/v1/public/api/datasets/<id>/poll-download` returns JSON carrying
a presigned S3 URL, which you then fetch. Worth knowing for any future Singapore work, and
worth knowing that **the two dataset families answer that poll differently**: a *table*
dataset returns `data.status = "DOWNLOAD_SUCCESS"` alongside `data.url`, and a *geospatial*
dataset returns `data.url` with no `status` key at all. Code that asserts on `status` works
for the religion table and throws `KeyError` on the boundary file.

The printed report, *Census of Population 2020, Statistical Release 1: Demographic
Characteristics, Education, Language and Religion*, is at
`https://www.singstat.gov.sg/files/467e88fc-0c55-453c-885f-268b89731904.pdf` (6.1 MB, 247
pages), cached as `data/raw/sg/cop2020sr1.pdf`. **The obvious guessable URL
`/-/media/files/publications/cop2020/sr1/cop2020sr1.pdf` returns a 404 body with HTTP 404 and
a 164 KB length**, which is large enough to look like a successful download if you only check
that bytes arrived. The real path is a UUID and has to be read off the publication page.

**The religion-by-planning-area table is not in the report.** The report's statistical tables
give religion by age, sex, ethnicity, residential status, place of birth, qualification and
marital status, but not by geography; the planning-area cut exists only in TableBuilder and on
data.gov.sg. So the PDF is not an alternative route to this data, and is used here only as an
independent check on the national totals.

## 2. What the categories are

Nine, and they partition the total exactly:

| category | count | share | node |
|---|---|---|---|
| Buddhism | 1,074,159 | 31.1% | `buddhism` |
| No Religion | 692,528 | 20.0% | `unaffiliated` |
| Islam | 539,251 | 15.6% | `islam` |
| (Christianity) Other Christians | 411,674 | 11.9% | `christianity` |
| Taoism | 303,960 | 8.8% | `chinesefolk` |
| (Christianity) Catholic | 242,681 | 7.0% | `christianity.catholic.latin` |
| Hinduism | 172,963 | 5.0% | `hinduism` |
| Sikhism | 12,051 | 0.35% | `sikhism` |
| Other Religions | 9,827 | 0.28% | `other.sg` |

Nine is middling for this map: the Philippines maps 129 source categories and Vietnam 28, so
"the deepest census in Southeast Asia" would be wrong and was written and removed during this
build. What is unusual is the *shape*. Singapore gives Sikhism its own cell at twelve thousand
people while leaving 411,674 non-Catholic Christians undivided, which is a statement about
which distinctions the office thinks matter.

The two arguable calls are argued at length in `taxonomy/sg2020.py`'s `REVIEW` and summarised
in §6 below. A second agent was asked both questions cold, pointed at the files rather than at
the reasoning, and reached the same two answers.

## 3. Geography, and why there is no Kontur here

The counts are on **URA Master Plan 2019 planning areas**, which the dataset's own footnote
names. Boundaries: data.gov.sg `d_4765db0e87b9c86336792efe8a1f7a66` (planning areas, 55) and
`d_8594ae9ff96d0c708bc2af633048edfb` (subzones, 332), both "No Sea" editions, both clean
GeoJSON with proper attribute fields rather than the HTML-table-in-a-Description property that
some URA layers on data.gov.sg still carry.

**The placement layer is the 332 subzones weighted by census resident population, not the
Kontur grid, and Singapore is the one country on this map where that is strictly better rather
than a preference.** SingStat publishes *Resident Population by Planning Area/Subzone of
Residence, Ethnic Group and Sex* (`d_e7ae90176a68945837ad67892b898466`), which is the same
census, the same office and the same universe-defining word, *residents*, as the religion
table it weights.

A footprint-derived global surface would be wrong here in a direction you can name in advance.
**Kontur counts everybody physically present, and 1.64 million people in Singapore are
non-residents the religion table does not cover**, many in worker dormitories. The resident
population of Tuas is **70** and of Sungei Kadut **750**; a presence-based weight would put
resident dots in both. This is the one case found so far where
[[reference_kontur_resolution_floor]]'s "finer than the counting tier" is satisfied and the
grid is still the wrong instrument, because the resolution was never the problem, the
*universe* was.

What it costs: the population table is all ages, the religion table is 15+. Across the 31
units the 15+ share runs **0.756 to 0.885** against a national 0.855, so a young planning area
is weighted a little high. That moves dots *within* a unit by a couple of per cent and never
between units. The band has the right shape, which is the reassuring part: the two lowest are
Punggol (0.776) and Sengkang (0.826), the newest towns and the ones full of young families,
and the highest are Ang Mo Kio and Pasir Ris (0.885), mature estates. A scrambled join has no
such pattern.

## 4. The universe, and it is the biggest caveat on this country

The question is asked of **residents aged 15 and over**. Both exclusions are large:

| | people | of total population |
|---|---|---|
| total population, June 2020 | 5,685,800 | 100% |
| residents (citizens + PRs) | 4,044,210 | 71.1% |
| **residents aged 15+, the religion universe** | **3,459,093** | **60.8%** |
| residents under 15 | 585,117 | 10.3% |
| non-residents | 1,641,590 | 28.9% |

**Neither is scaled up**, which is Chile's rule (`sources/cl.md` 3) applied twice.

* *The under-15s.* Religiosity moves sharply with age in Singapore: 24.2% of 15-24s report no
  religion against 15.2% of the over-55s (report Chart 5.5). A flat scale-up would be an
  assertion about children the census declined to make.
* *The non-residents.* Work permit and S Pass holders, employment pass holders, dependants and
  foreign students. **No SingStat table anywhere gives their religion** — the religion tables
  are all headed "Resident Population", and the residential-status cut
  (`d_bf205c3175b8835fef381d7d0f18715a`) splits residents into citizens and PRs rather than
  reaching outside the resident population. This is not a gap that better searching closes; it
  is a scope decision the census made.

There is **no "not stated" cell** in this table, so nobody is hiding in a refusal residual
either. The nine categories are the whole of the 3,459,093.

Was a Hong Kong style derived layer for the non-residents considered? Yes, and rejected. HK
works because the census counts Indonesians, Filipinos and Pakistanis *by district*. Singapore
publishes non-resident numbers by pass type and not by nationality, and nothing by planning
area, so there is no geography to hang a coefficient on. Stated so nobody re-derives the
question: it is not that the coefficient would be arguable, it is that there is no denominator.

## 5. The four checks, and the one that caught something

1. **The download against the printed report, exactly.** All ten national figures (nine
   categories plus the total) must equal Table 51's `Total` row in Statistical Release 1. They
   do, to the person. These are two different production runs of one census, and the same ten
   figures appear a third time in UNSD Demographic Yearbook table 28.
2. **The nine partition each row's own total.** Worst residual over 32 rows: **-2** (Jurong
   West).
3. **The 31 units sum to the national row.** Worst column **+4** (Islam, on 539,251). This is
   SingStat's random rounding for confidentiality; the report says "Figures may not add up to
   the totals due to rounding". Asserted with a tolerance of 10, never as equality.
4. **15+ against all-ages, over 31 units: r = 0.99857.** Both sides are the same census, so
   unlike the Kontur correlations elsewhere on this map this one is allowed to be tight, and
   the assert is `r > 0.99` with the share band inside 0.60-1.00.

**`-` is an in-band sentinel meaning "nil or negligible"**, the report's own notation. It
occurs once, in Downtown Core's Sikhism cell, and that single cell is why data.gov.sg's
metadata types the entire Sikhism column as `Text` while every other category is `Numeric`.
Both scripts raise on any other non-numeric token rather than coercing it (Sri Lanka's rule).

### The one that actually caught something: `Changi- Total`

The subzone population CSV is a **flat list with positional nesting**: a planning area appears
as a header row reading `<name> - Total` and its subzones follow it, unindented and unmarked.
The parent is carried by position.

**`Changi- Total` is printed with no space before the hyphen.** It is the only one of the 55
like that. The obvious `label.endswith(" - Total")` misses it, so Changi is never opened as a
planning area and its three subzones (Changi Airport, Changi Point, Changi West) are silently
attributed to the **previous** header, which is Central Water Catchment. The symptom is
Central Water Catchment holding 3,700 people when its own total row says nil, on an
uninhabited reservoir catchment, and **nothing else in the file complains**: every other
planning area still reconciles, the grand total is untouched, and 54 planning areas parse
where there should be 55.

Caught by asserting the planning-area count against URA's 55 rather than trusting the parse.
Matched now on `^(.*?)\s*-\s*Total$`. This is [[reference_name_join_wrong_neighbour]] in its
positional form: the wrong *neighbour* is the row above rather than the same-named town
elsewhere, and it is invisible to every totals check for the same reason.

## 6. The two mapping calls

Both are in `taxonomy/sg2020.py`'s `REVIEW` with the full argument. In short:

* **`Taoism` goes to `chinesefolk`, not `daoism`.** The source footnotes it, under every
  religion table in the release: **"'Taoism' includes Chinese Traditional Beliefs."** That
  makes the cell a declared combination the office will not separate, which is spec §3.3's
  test. **Hong Kong's Taoism answer goes to `daoism`** (`hk2021.py`), so the two neighbours
  draw different colours, and that is deliberate: HK's is a *survey* option offered beside
  Buddhism with no gloss, where a respondent chose the word unaided, and Singapore's is a
  census cell the office has told you is broader than the word. Following the printed label
  over the printed footnote is precisely the error [[reference_census_questionnaire]] exists
  to catch. Closest precedent: `mu2022.py`'s `Buddhist/Chinese`.
* **`Other Christians` goes to the `christianity` ROOT, not `christianity.protestant`.**
  411,674 people, 11.9% of the counted population, and the largest single call here.
  `christianity.protestant` "holds the ANSWER, not the category" per its own note, and nobody
  in Singapore answered *Protestant*; it would also assert Protestantism of the Orthodox
  minority inside the cell. `christianity.other` fails the other way: its note says it is for
  bodies with no branch to belong to and explicitly **not** for a residual, and this is
  nothing but a residual. Precedent is exact: **`lk2024.py`** faces the identical
  Catholic/not-Catholic binary from Sri Lanka's DCS, sends it to the root, and rejects
  `christianity.protestant` in those same two words. Australia, Chile, Fiji, France, Germany
  and Switzerland all sit at the root too.

The cost of the second one is real and is named in `note_public`: **the map cannot show that
Singapore's Protestants outnumber its Catholics by nearly two to one**, because the census
never used the word.

## 7. The `Others` unit

The table names 30 planning areas and puts the remaining 25 in one row of **25,756** people.
Those 25 are Boon Lay, Central Water Catchment, Changi, Changi Bay, Lim Chu Kang, Mandai,
Marina East, Marina South, Museum, Newton, North-Eastern Islands, Orchard, Paya Lebar,
Pioneer, Rochor, Seletar, Simpang, Singapore River, Southern Islands, Straits View, Sungei
Kadut, Tengah, Tuas, Western Islands and Western Water Catchment. Several have a resident
population of zero.

**It is drawn, not dropped**, on the union of exactly those 25 polygons, and it is `measured`
— the count *is* measured, at a unit that happens to be disjoint. The identification is
checked rather than assumed: the census's own population table puts **34,050 residents of all
ages** in those same 25 areas, against 25,756 aged 15+, a ratio of 0.756 against a national
0.855, which is the right size for a set of areas skewed old.

**The cost is specific and worth stating: Rochor is in there, and Rochor contains both Little
India and Kampong Glam** (verified against the subzone layer, which has `LITTLE INDIA` and
`KAMPONG GLAM` as Rochor subzones). The two districts a reader would go to looking for Hindu
and Muslim Singapore are drawn with one averaged mixture spread over an area that also takes
in Tuas and the Southern Islands. Anyone tempted to read a neighbourhood off this map should
start there.

## 8. What the map shows

* **It is flatter than almost anything else here, and that is policy.** Since 1989 the Ethnic
  Integration Policy has capped each ethnic group's share in every public housing block and
  neighbourhood, and about eight in ten residents live in that housing. Malays are 98.8%
  Muslim by this census, so an ethnic rule is in practice a religious one: the most Muslim
  planning area is **Woodlands at 28.1%** against a national 15.6%, a ratio of 1.8. For
  comparison, Ghana has districts between 95% and 99% Muslim.
* **The variation that survives is housing tenure, not ethnicity.** Christians are drawn from
  all three main ethnic groups, so the quota never constrained them, and Christianity has much
  the widest range of anything here: **44.1% of Bukit Timah against 11.5% of Woodlands**,
  twelve kilometres apart. Bukit Timah, Tanglin and River Valley are the private-housing
  districts; they are also where `no religion` runs near 30% against a national 20.0%, and
  where Taoism falls to between 2% and 5% against 8.8% (River Valley 2.08%, Tanglin 3.22%,
  Bukit Timah 4.79%; the fourth-lowest of the 31 is Downtown Core at 4.01%, which is not a
  private-housing district, so this trio is not the bottom three).
* **Hougang has the highest Catholic share of its own Christians**, 43.5% against a national
  37.1%, with Ang Mo Kio and Bedok next. The Church of the Nativity of the Blessed Virgin Mary
  at Hougang dates from 1853 and served a Teochew Catholic village; that is the obvious
  reading and this file is not claiming it is a demonstrated one.
* **Sikhs are present everywhere and concentrated nowhere.** 12,051 people, and the planning
  areas run from 0.19% to 0.71%. For a group that size, on this map, that is unusual.
* **`other.sg` is the wealthy central belt**: 1.13% in the `Others` areas, 0.96% River Valley,
  0.88% Marine Parade, 0.78% Tanglin, against 0.15% in Ang Mo Kio.

## 9. Rebuilding

```
python sources/sg.py --fetch        # 2 KB, seconds
python sources/sg_geo.py --fetch    # 3.2 MB, seconds
python tools/check_mapping.py sg
python scatter.py --country sg
python scatter.py --country sg --dot-value 10000
python tools/build_tail.py --id <sid>
```

Nothing here takes more than a few seconds. Singapore is one of the cheapest countries on the
map to rebuild from scratch.

## 10. Review, 2026-09-08

A second-perspective pass (`rd-review`), read from the PDF and the raw CSV rather than from
sections 1-9. `check_md.py`, `built_countries.py --check`, `check_rollup.py sg` and
`check_overview.py --focus sg` were all clean, the last one at a loosened `--near 22`, so the
Catholic and Hinduism swatches that look close on screen are not close by measurement.

**Confirmed against the primary material, not taken from this file.** The Taoism gloss is real
and is printed under every religion table: `'Taoism' includes Chinese Traditional Beliefs`, on
report pages 12, 48 and 215-226 of `cop2020sr1.pdf`. So `chinesefolk` over `daoism` is the
source's own instruction rather than an interpretation, and it is the call
`[[reference_census_questionnaire]]` exists to protect. Every other country on the map that
carries a bare `Taoism` label sends it to `daoism` (au, ca, cz, hk, nz, pl, uk) and Singapore is
the only one that does not; that divergence is correct here, and worth leaving visible. The
`Other Christians` to the `christianity` root call matches about thirty existing files including
`lk2024.py` exactly as section 6 claims. `Among the Malays, 98.8 per cent were Muslims` is on
report page 50; `20.0 per cent` with no religion against `17.0` in 2010 is on page 10; the
15-24 figure of `24.2 per cent` is on page 52; resident population `4,044,210` and total
`5.69 million` both check. The planning-area and subzone joins are asserted as set comparisons
in `sg_geo.py` and not eyeballed, which is what `[[reference_name_join_wrong_neighbour]]` asks
for.

**One figure is wrong and is in two places.** Section 8 and `countries.py`'s `note_public` both
say that in Bukit Timah, Tanglin and River Valley `Taoism falls to 2-3%`. River Valley is 2.08%
and Tanglin 3.22%, but **Bukit Timah is 4.79%**, and it is only the fourth-lowest of the 31
units; Downtown Core at 4.01% sits between them and is not one of the three named. The honest
phrasing for that set is 2 to 5%. Left unedited by the review because it is the builder's prose
in a reader-facing note and that pass was read-mostly. **Corrected in both places 2026-09-08**,
to *"between 2 and 5%"* in `note_public` and to the same with the three figures spelled out in
section 8; the four shares above all reproduce from `data/normalized/sg.csv`.

**The flatness claim is true but is doing more work than the data supports.** Section 8 and the
note open with the map being flatter than almost anything else here `and that is policy`,
crediting the Ethnic Integration Policy. The quota is real and does flatten the resident
distribution. But the 1,641,590 non-residents left out of the universe are the most spatially
concentrated religious population on the island, which this file already knows: section 3 argues
against Kontur precisely because a footprint grid would put dots in the Tuas dormitories. So
part of the flatness is the universe choice and not the housing rule, and the two causes point
the same way, which makes them easy to conflate. Nothing to rebuild; the note would just be
stronger if it said the excluded third is also the concentrated third.

**Fixed here.** `grain` read `the other 25, 112,000 people on average`, two numbers joined by a
comma, which renders in the panel as `25, 112,000` and reads at a glance as 25,112,000. The
comma is now a semicolon. The house shape elsewhere is `unit, N people on average; aside`
(`kz`, `hk`, `cn`), so reordering to `31 planning-area units, 112,000 people on average; 30 are
named and one holds the other 25` would match convention better, if that is wanted.
