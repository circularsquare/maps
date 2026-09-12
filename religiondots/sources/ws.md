# Samoa — Samoa Bureau of Statistics, 2021 Census, Table 2

Built 2026-09-08. `sources/ws.py`, `sources/ws_geo.py`, `sources/ws_grid.py`,
`taxonomy/ws2021.py`. 205,557 people, 26 categories, **read at 339 villages, drawn at 25
districts**, nothing derived and nothing excluded.

§11aa had already verified the data end to end and written *"the blocker is the geography, and
it is the one place the usual answers all fail… worth an hour before committing to it."* This
is that hour, and the answer is yes, at a coarser tier than the data deserves.

---

## 1. The data — one GET, and the cleanest table on the map

`https://www.sbs.gov.ws/wp-content/uploads/2022/12/CensusTablesEXCELFiles.xlsx`, 2.68 MB, 49
sheets. No plugin, no wall, no PDF. `Table 2. Total population by sex, religion and place of
residence, 2021` is 401 × 82.

**It closes in every direction:**

- The 26 categories sum to the printed total at **every one of its 395 place rows**, not just
  nationally. No tolerance anywhere.
- The four tiers nest exactly, **on all 27 columns**: 339 villages → 51 districts → 4 regions →
  the country.
- **There is no `Not stated` column at all.** The 2016 census had one; 2021 does not. The only
  residual is a named `OTHER CHURCHES` at 1.89%. **Samoa is 100% drawn**, which nothing else
  here is.

**The tiers are indentation in column A and nothing else** — §9p/§9af/§9az's pattern, here with
four levels: 0 = Samoa, 4 = region, 8 = district, 12 = village.

## 2. The outside check is on the instrument, not the numbers

UNSD table 28 has Samoa for **2001 and 2016 but not 2021**, so it cannot verify a 2021 figure
the way it verified Tonga's (§9bj). What it can do is check that the question did not change:
`ws.py` asserts that **all 24 of 2016's churches still have a 2021 column**, so none was quietly
merged away. 2021 adds `ASO FITU (SISDAC)` and `AMAZING LOVE CHRISTIAN CHURCH`, and drops
`Not Stated`.

**And the Yearbook identifies three cells the 2021 workbook renders in Samoan**, which is the
only reason they are mappable at all:

| 2021 workbook | UNSD 2016 | what it is |
|---|---|---|
| `POROTESANO` | `Protestant` | Samoan for Protestant; an answer, not a body |
| `PABTISM` | `Baptist` | the workbook's own spelling, kept in `source_category` per §2.4 |
| `BIBLE STUDY` | `Aoga Tusi Paia` | Samoan for *Bible school*; still unidentified as a body |

## 3. The geography, which is the whole problem

**Samoa is the one Pacific country with no COD-AB.** Re-checked 2026-09-08: Tonga, Fiji, the
Solomons, PNG, Vanuatu, Kiribati, FSM and the Marshall Islands all have one on HDX and Samoa
does not. What exists instead:

- **geoBoundaries WSM ADM2, 43 polygons**, from the Pacific Data Hub.
- **GADM 4.1 level 2 is the same 43** — same names, same `Gaga'emauga I (PART)` artefacts. One
  source, not two, so there is no second opinion to be had.
- OSM has 11 admin relations for the whole country.
- `pacificdata.org` serves a **Cloudflare challenge** on its CKAN API and was not fought.
- SBS publishes no geography itself: **849 media files and not one shapefile**, and its own
  census dashboard is a Looker Studio embed. Its `data.sbs.gov.ws` is a **.Stat Suite with an
  SDMX API** (`data-sdmx-disseminate.sbs.gov.ws/rest`) and `microdata.sbs.gov.ws` is a NADA
  catalogue; both are open and neither carries boundaries.

### The 43 are a different cut from the census's 51, not a coarser one

This is the thing that took the hour to see. The census numbers its districts — `Vaimauga 1`
… `Vaimauga 4` — and the polygon layer names them by compass point — `Vaimauga East`,
`Vaimauga West`. Four against two, and **neither nests inside the other**. Matching on the name
stem alone settles **18 of the 51 districts, 23.1% of the population**, and leaves the rest
ambiguous:

    Faasaleleaga 1..5  ->  Faasaleleaga I / II / III / IV      (5 against 4)
    Vaimauga 1..4      ->  Vaimauga East / West                (4 against 2)
    Aana Alofi 1..4    ->  Aana Alofi I / II / III             (4 against 3)
    Safata 1..2        ->  Safata                              (2 against 1)

### What works: both are cuts of the same 25 traditional districts

Every census district and every polygon carries one of **25 traditional district names** as its
stem, so both aggregate onto those 25 exactly, by name, with nothing geocoded and nothing
assumed. `ws_geo.py` folds both sides and **asserts the two stem sets are equal, set against
set** — a stem on one side only would mean a district went undrawn or a polygon got no people,
and it fails rather than warning. 43 polygons and 51 census districts, one set of 25.

**8,222 people per unit.** Coarser than Tonga's 637 and the Solomons' 3,940, finer than
Kazakhstan's 17 regions or Rwanda's 5. §3.9b removed the unit-count floor and
[[feedback_no_granularity_floor]] is the standing instruction: draw it and state the grain.

### The route to 43 was tried and rejected, and it is the next improvement

The 339 villages could be assigned to the 43 polygons by locating each one, and **OSM has 554
Samoan villages**. Matching the census's names against them, **disambiguated by requiring the
census district's stem to agree with the polygon the point falls in**, placed 285 villages and
**85.2% of the population**. Why the rest fail is known and fixable:

- **The census anglicises.** `Lalovaea East` is OSM's `Lalovaea Sasa'e`; `Samata Uta` is
  `Samata-i-Uta` (*uta* = inland, *tai* = seaward).
- **The census qualifies repeated names with their district** — `Vailoa Faleata`,
  `Siufaga Faasaleleaga`, `Fusi Safata`, `Matavai(Asau)` — which is the same duplicate problem
  from the other side, and the qualifier is the answer.
- **Solosolo, Falefa, Faleseela, Falevao and Tuanimato are absent from OSM entirely**, verified
  by refetching the whole country by bbox rather than by area.

**85% is not a basis to draw on**, because a village put in the wrong district moves people
between units and every total still balances — [[reference_name_join_wrong_neighbour]] exactly.
The fold on stems has no such failure mode, which is why it is what shipped.

## 4. Placement — Kontur, and the correlation is not carrying the join

Upolu and Savai'i are volcanic islands with forest and lava-field interiors and their people in
a ring of villages along the coast road; Savai'i is 1,700 km² with an empty middle. Weighting by
area would put Palauli's and Gagaifomauga's dots on Mount Silisili.

- **16% of Kontur's people fall outside every district and are snapped, not dropped** (§9bg's
  rule). Settlement is a coastal ribbon so the loss would be entirely seaward; **100% of the
  strays are within 700 m** and 4 people in the country are dropped.
- **r = 0.986 over 25 units**, all inside a factor of 3, against a best of 0.743 over 2,000
  random pairings. **But 25 units is a weak test** and this is corroboration rather than the
  check that carries the join, which is said in `ws_grid.py`'s docstring so nobody later reads
  it as stronger than it is. Gagaemauga at 1.79× is the loosest, and it is the district whose
  polygons carry the `(PART)` split across the two islands.

## 5. What the census shows

**The Samoan village belongs to one church.** This is the finding, and it is in the data even
though it is not on the map:

| village | people | and |
|---|---:|---|
| Malua | 424 | **424 Congregationalists**, nobody in any other column |
| Amaile | 244 | 99.6% Roman Catholic |
| Mulivai Safata | 399 | 99.5% Roman Catholic |
| Tapueleele | 330 | 99.4% Latter Day Saints |
| Gataivai | 1,017 | 94.4% Methodist |

Under *fa'amatai* the village council decides matters for the village and the church is one of
them, so a village row reads more like one collective answer than six hundred separate ones.
**Drawn at 25 districts none of this is visible**, which is the cost of the geometry and is said
in `note_public` rather than left for the reader to discover.

**Three churches divide the country at the district scale, and that is visible.**

- **The Methodists are Savai'i**: 20.5% of that island against 8–10% everywhere else,
  **62.7% of Satupaitea**, 94.4% of Gataivai.
- **The Catholics are Apia and the two far ends**: 25.3% of the Apia urban area, and 35.4% of
  *both* Falealupo at the western tip of Savai'i and Aleipata Itupa i Lalo at the eastern end of
  Upolu.
- **The Latter Day Saints run opposite the Catholics**: 21.4% on Savai'i against 13.6% in Apia,
  reaching 34.6% of Vaa o Fonoti. At **17.6% nationally Samoa is second only to Tonga's 19.7%**
  among the 67 country-years in UNSD table 28 that count them separately.
- Over all of it the **Congregational Christian Church is the largest body in every region**,
  27.0% nationally.

## 6. Two new nodes, and one of them was predicted

- **`christianity.reformed.congregational.cccs`** — the Congregational Christian Church of
  Samoa, 55,411 people. This **completes a set rather than opening one**: `.cicc` (Cook
  Islands), `.ekt` (Tuvalu) and `.niue` were added with the microstate tier (§9bf), and
  `.ekt`'s own note already said *"Samoa's own CCCS would be a third if Samoa is drawn"*. It is
  the largest of the LMS's Pacific daughters and the mother of the other three: John Williams
  landed at Sapapali'i in 1830 and the Samoan teachers it trained carried the mission on.
- **`christianity.adventist.sisdac`** — `Aso Fitu`, Samoan for *the seventh day*, is SBS's
  column for the Samoa Independent Seventh Day Adventist Church, 1,962 people. **No other census
  anywhere counts it.** The argument against is that at 25 districts it never exceeds 3.0% of a
  unit and shows no geography; the argument for is that merging it into `christianity.adventist`
  loses the only place it is visible, and Ekalesia Niue at 981 people is the precedent for a
  node this size. In REVIEW with both halves written down.

`CONGREGATIONAL CHRISTIAN CHURCH OF JESUS IN SAMOA (EFIS)`, a CCCS breakaway at 482 people,
sits on the **parent** `christianity.reformed.congregational` rather than getting a fourth
sibling: at 0.23% and under 2% of any district it would be a colour nobody could find.

`OTHER CHURCHES` goes to `christianity.other` and **not** to an `other.ws`, which is the
opposite of Vanuatu's call (§9bg). There the same header was defined by volume 2 of the same
census as *"88 different religions"*; here the header says **churches**, and Muslims, Baha'is
and `No religion` all have columns of their own, so the cell is what was left after
twenty-five named Christian answers.

## 7. What is left

- **The 43-unit build**, once the village geocoding is closed. Section 3 says exactly what
  breaks and why; the census's own district qualifiers are most of the answer.
- **The census's own 51 districts** would be better still and need a boundary file nobody has
  found. They look like Samoa's 51 electoral constituencies, and SBS publishes a **district
  profile PDF per constituency** under `/digi/` with the Samoan names (`Safata Sasa'e`,
  `Faasaleleaga Nu.1 i Sasa'e`), which is a naming key for them but not a geometry.
- **`data.sbs.gov.ws` is an unexplored .Stat/SDMX instance** and `microdata.sbs.gov.ws` an
  unexplored NADA catalogue. Neither was needed for this build.
- **The 2016 and 2001 censuses** are in the oracle at national level, so there is a three-census
  series: the CCCS runs 34.8% → 29.0% → 27.0% while the Latter Day Saints run 12.7% → 16.9% →
  17.6%, the same direction as Tonga's.
- **`pacificdata.org`'s Cloudflare wall** is the one route not tried. It is the source of the
  43 polygons, so it may hold more.
