# Israel — CBS, 2022 Census of Population and Housing

Ingested 2026-09-07. `sources/il.py`, `sources/il_geo.py`, `taxonomy/il2022.py`.
Drawn at **statistical area + locality**: 2,968 units, ~3,000 people each.

Summary: the finest geography on this map after the United States, and the only country here
whose religion comes off a **state population register** rather than a question. The
interesting sections are §2 (an API that reads as absent and is not), §3 (the rate limit, and
how it was earned), §4 (a lump category that would have erased Israel's Christians), §5 (an
observance axis that is not a split of Judaism) and §7 (the territorial cut).

---

## 1. What CBS publishes, and at which grain

Four things, and only one of them is both fine and complete:

| source | geography | categories | counts? |
|---|---|---|---|
| Statistical Abstract **ST02-11x** | district; sub-district only partially, via "Thereof:" rows | Jews, Muslims, Christians, Druze, not classified | **yes, exact** |
| `bycode2023.xlsx` (data.gov.il) | 1,485 localities | Jews-and-others / Arabs, plus a **modal** religion label | yes, but 2 categories |
| 2022 census bulk CSV (data.gov.il) | 1,186 localities + 2,193 statistical areas | a **modal** religion label only | no |
| **2022 census per-area dashboards** | every unit, to statistical area | Jews, Muslims, Christians, Druze, Others (+ observance) | **shares, 0.1%** |

The last row is what is drawn. The trade §3.9 describes is unusually stark here: the exact
counts exist only at ~16 units, and the 3,236-unit geography carries shares rounded to a
tenth of a percent. At 1:1,000 people per dot a tenth of a percent of a 3,000-person unit is
three people, so the rounding is immaterial and the geography is worth everything.

**The bulk CSV's `religion` column is a modal label and not a composition** — §12's fifth
failure shape. Abu Sinan is labelled Muslim and its statistical area 1 is labelled Druze.
Using it as a composition would make every unit religiously homogeneous and delete every
minority in the country. It is used here only for the unit list and the population.

## 2. The API reads as absent, and the key is an autocomplete nothing links to

`census.cbs.gov.il` is an Astro shell in front of a Looker instance. Probing it the obvious
way says there is no API:

- `/api/<anything>` returns **HTTP 200 and the same 2,804-byte SPA shell** as any other
  unknown path, so status codes and content types agree that nothing is there.
- The area IDs are opaque: `ID=1fd1aec` works, `ID=3000` — Jerusalem's real CBS code — does
  not, and the neighbours of a working ID all fail, so the space is sparse and cannot be
  enumerated or guessed.

Both are wrong, and the two findings generalise:

1. **A catch-all shell is not evidence of absence** ([[reference_spa_hidden_apis]] again,
   third instance in this project). `/en/api/get-csv?dashboardId=..&ID=..` returns a **zip of
   the dashboard's underlying CSVs**, unauthenticated. It was found because a *search engine*
   had indexed one `get-pdf` URL — the site's own pages never link to `/api/` at all.
2. **The ID mapping is in the site's htmx autocomplete.** The search box carries
   `hx-get="/en/partials/search/area"`, and `GET /he/partials/search/area?search=<term>`
   returns `<button data-id data-search data-type>` for every geographic unit matching the
   term — locality, district, sub-district, quarter, sub-quarter, statistical area. **That
   endpoint is the whole country.** It was found by loading the page in headless Chrome and
   dumping every element carrying an `hx-*` attribute, which took one CDP call after two days
   of the API looking absent.

**IDs are language-independent, so the two halves of the fetch use different editions** —
§12's North Macedonia rule arriving in a new shape. Search runs against `/he/`, because the
Hebrew labels are what the census file's own locality names join on; data is fetched from
`/en/`, because the English category labels are what the taxonomy keys on and because the
two lumps (§4) are distinguishable at a glance in English and not in Hebrew.

**The search caps at 100 results and Jerusalem has 194 statistical areas.** Asking by number
prefix — `"<name> אזור סטטיסטי 1"` — brings back every area whose number starts with 1, so
Jerusalem costs ten queries rather than 194.

## 3. THE RATE LIMIT, WHICH WAS EARNED

The first harvest ran at 0.15 s between requests and a second process was pointed at the same
host to check a label. Within about thirty minutes **every connection to census.cbs.gov.il
was being reset — including the static home page**, so it was an IP-level block and not an
endpoint one. It had not lifted an hour later.

This is §12's KOSIS entry from the other side: there, a wall spread *while being probed*;
here it was created outright. Three things to carry forward:

- **Count the requests before starting.** 3,236 units × (a search + a dashboard) is ~4,600
  requests against a small national statistical office. That is a number worth looking at
  rather than discovering.
- **Never point a second process at a host a harvest is already walking.** The concurrent
  probe is what turned a working run into a block.
- **A cache written only at the end is a cache that never gets written.** The first run kept
  everything in memory and saved every 200 units; it was killed after 30 minutes having saved
  nothing, so the whole thing was lost. `il.py` now writes every 50 units through a temp file
  and `os.replace` ([[reference_wb_truncates]]), reuses one connection for the whole run, and
  ratchets its delay up on every error and never back down.

## 4. `Other religions` is not `Others`, and the difference erases the Christians

The dashboard publishes the full five-way breakdown for large units and, for smaller ones,
**the dominant group plus a lump**:

| unit | what comes back | what the lump actually is |
|---|---|---|
| Nationwide | Jews 72.7 / Muslims 17.8 / Others 6.1 / Christians 1.9 / Druze 1.5 | — |
| Haifa | all five, Druze at 0.2% | — |
| **Nazareth** | **Muslims 73.1 / Other religions 26.9** | essentially all Christian |
| **Shefar'am** | **Muslims 62.9 / Other religions 37.1** | Christians *and* Druze |
| Fassuta | Christians 99.4 / Other religions 0.6 | a small mixed remainder |
| Majdal Shams | Druze 99.9 / Other religions 0.1 | a small mixed remainder |

**Two labels, unrelated meanings, and only one letter of difference in a hurry.** `Others` is
the register's "not classified by religion" and is a real category with a node
(`unrecorded`). `Other religions` is *everything except the dominant group, lumped*, and it
is not a category at all.

Mapped naively to anything — `other.il`, or worse, to the dominant group's neighbour — it
would have deleted most of Israel's Christians from the map, which is §14.2's second risk in
its purest form. It is therefore **deliberately absent from `taxonomy/il2022.py`'s MAP**, so
that an unresolved lump fails `tools/check_mapping.py` loudly, and `countries.py`'s
`_il_counts` raises rather than dropping the rows. The lump is resolved in `il.py` against
its sub-district's published Christian / Druze / Muslim totals from ST02-11x.

*The tell was not in the data.* Every unit's shares sum to 100.0% either way. It was found by
checking six places whose composition is known independently — a Christian village, a Druze
village, a mixed town — which is §12's "read the whole list and check one number you already
know", done on six.

## 5. The observance axis, and why it is not a split of Judaism

The same dashboard publishes **"population by main lifestyle in the household"**: Secular
53.2%, Traditional 24.4%, Religious / Very religious 12.2%, Ultra-religious 6.4%, Mixed 2.4%,
Other 1.4%. This is the only instrument on this map that separates Israeli Jews at all, and
it is worth having: Bene Beraq returns **83.8% ultra-religious**, Jerusalem 23.2%.

**But it is asked of the whole population, not of Jews.** Umm al-Fahm is 99.8% Muslim and
returns Traditional 47.2%, Religious 42.2%. Crossing the two axes per unit would be a model,
and in a mixed unit it would attribute one group's answers to another. So `il.py` applies the
split **only where a unit is at least 85% Jewish** and leaves everyone else on the parent
`judaism`. Israel's statistical areas are segregated enough that this covers most Israeli
Jews without ever applying the split where it would describe somebody else. Those rows are
`modelled` in §7's sense.

**`Masorti` is a false friend and it is the most likely thing here to be got wrong.** In
Israel *masorti* means traditional-but-not-strictly-observant; everywhere else in the Jewish
world "Masorti" is the name of the Conservative movement, which is `judaism.conservative` and
a different object with a different history. Same trap as `animismus` in §12: the string is
shared and the place decides the meaning.

### The segregation on the map is real, and it was checked rather than assumed

Anita, looking at the rendered country: *"these areas do seem to be quite segregated between
haredi religious traditional and secular. is this an artifact of our thresholding?"* It is the
right question to ask of any map built this way, and the answer is measured:

| | |
|---|---|
| median **largest** observance share within a qualifying unit | **86%** |
| units where one category exceeds 60% | 822 of 1,043 (**79%**) |
| units where one category is 90–99% | **482 (46%)** |
| effective number of observance groups per unit (4.00 = even mix) | median **1.32** |

**CBS's own per-unit shares are that lopsided**, so the pattern is in the source. It also
matches the ground: the red clusters land on Bene Beraq, Modi'in Illit and Beitar Illit,
which really are ~85% Haredi.

**Quantisation sharpens it, but only a little, and the amount is worth knowing.** A qualifying
unit holds a median 2,256 Jews — about 2.3 dots at 1:1,000 — and 78% hold fewer than four
dots' worth. Per unit the median number of categories genuinely present (≥5%) is **2** and the
median number that can draw at least one dot is **1**, so a two-category unit does round to
one. That is real exaggeration applied to an already near-pure distribution, not a pattern
invented from a mixed one.

**What IS an artifact is the blue.** `unspecified` is not a fifth observance category: it is
every unit under the threshold, i.e. the religiously mixed cities. It therefore reads as a
spatial pattern of its own while actually meaning "not split here", and it is the single most
misreadable thing in this country's legend.

Four nodes were added to `branches.py` for this — `judaism.haredi`, `.dati`, `.masorti`,
`.hiloni` — in their own LINEAGE group, `By observance`, placed next to `Traditional` so
Haredi sits beside Orthodox in the legend. **A country must use the observance axis or the
movement axis and never both.**

## 6. The register is the basis, and it has no box for irreligion

`basis` is **`roll`**, not `self_id`. Israel's religion is ascribed at registration from
parentage or a state-recognised conversion; nobody was asked. Germany's case (§3.9a), with
the state rather than a church-tax body as the register-holder. Two consequences the map has
to carry rather than hide:

- **Israel draws as ~100% religious**, because a secular Israeli Jew is registered as a Jew.
  That is a fact about the form. The observance axis is the only corrective, which is most of
  why it is worth its complications.
- **`Others` → `unrecorded`, never `unaffiliated`.** ~442,000 people the register holds with
  no religious classification, overwhelmingly immigrants under the Law of Return who are not
  Jewish by halakha, largely from the former Soviet Union. `unrecorded`'s own note already
  said "any register-basis source with the same shape belongs here"; this is the second.

## 7. The territorial cut — the Green Line, with the Golan kept

**Anita's decision, 2026-09-07**, and the reasoning is in `sources/il_geo.py` at length
because it is the kind of thing that gets quietly reversed later.

- **West Bank, Gaza and East Jerusalem are not drawn.** CBS counts 503,732 Israelis in its
  "Judea and Samaria Area" district and about 360,000 Palestinians in East Jerusalem. Both go.
- **East Jerusalem goes in both directions**, which is the part that needed deciding. Cutting
  only the Palestinian neighbourhoods while keeping Gilo, Pisgat Ze'ev, Ramot and Neve
  Ya'akov would draw the settlements and erase the people they were built among. Jerusalem
  therefore draws as a western fragment.
- **The Golan is drawn**, against the same rule and as a stated exception: Israel counts
  those 56,600 people and nobody else does, and 24,900 of them are Druze — about a sixth of
  the Druze on this map.

**The cut is a published geometry.** OCHA's `cod-ab-pse` (`pse_admin0.geojson`) draws the West
Bank *including East Jerusalem* plus Gaza; a unit is dropped when the majority of its area
falls inside it. 267 of 3,235 units go, 29 straddle and are kept.

**geoBoundaries PSE is the wrong file and looks right.** Its ADM0 excludes the annexed
Jerusalem municipality, so Shu'afat, Silwan and the Old City all test as *outside* Palestine
and East Jerusalem would have stayed on the map with nothing to show it had been considered.
Ten hand-checked points separate the two files and the check is permanent in `il_geo.py`, so
a future release of either cannot move the line silently.

## 8. Boundaries: CBS's own, and the `+` is a dissolve instruction

**The tabulation geography is published by the office**, which §12 says to look for first and
which almost never happens. CBS runs an ArcGIS Online organisation (`ISRAEL_CBS_GIS`) whose
`Statistical__Areas_2022` layer is **3,857 polygons keyed by `SEMEL_YISHUV` + `STAT_2022`** —
exactly the codes the census tables carry. No name join, no vintage gap, no correspondence
workbook. It is **EPSG:2039**, not 4326.

**The published unit is a group of polygons and the table says which.** CBS merges small
areas before publishing, so `StatAreaCmb` is `"1022+1023+1024"` where `StatArea` is `1022`.
Every `+` is a dissolve instruction; 193 units need it. Ignoring it leaves those units drawn
at a fraction of their real extent with every count still correct.

**THERE IS NO PLACEMENT GRID, AND THAT IS MEASURED RATHER THAN SKIPPED.** A Kontur grid was
built here first, on the reflex that every country since Kenya has had one. It fails §8.2e's
resolution-floor test worse than Saint Vincent, the country the rule was written for:

| | Israel | Saint Vincent |
|---|---|---|
| median unit area ÷ hex area | **0.59** | single digits |
| units with no hex at all | **42.1%** | 36% — enough to abandon it |
| units smaller than ONE hex | **68.9%** | 43 of 221 |
| per-unit Kontur/census ratio | p10 0.27, median 1.01, p90 **3.49** | p10 0.00, p90 2.68 |

**A ratio below 1 means the grid is coarser than the thing it is refining** — a 0.69 km²
median statistical area against a 1.17 km² hex. For two units in three the "weight" is one
cell covering the whole unit and several neighbours, which is uniform placement with extra
steps applied to whichever unit the hex centre happened to land in. So `place` is the units
layer and `place_weight` is absent; `measure_grid_floor()` runs the test on every build so
the decision re-evaluates rather than rotting.

*Two wrong turns on the way, both recorded because they are the shape of the mistake rather
than the mistake itself.* The first version gave each hex-less unit a representative **point**,
which would have stacked every dot in the unit on one coordinate. The second clipped the
hexes to their units to stop a measured 1.28× overhang spilling dots into neighbouring
neighbourhoods — a real fix, carefully done, to a layer that should not have existed. **The
signal was in the build output the whole time**: "1,250 units have no hex centre" was printed
and read as a fallback statistic instead of as a verdict on the grid.

What uniform costs, stated because §3.9b requires it: the handful of genuinely large units —
Negev Bedouin localities and regional councils, up to 195 km² — get an even wash where the
population really is clustered.

## 9. Two host facts worth keeping

- **data.gov.il serves the same file from three hostnames with three answers.**
  `aws-e.data.gov.il` — the one CKAN advertises — redirects to a **Google OAuth login**;
  `e.data.gov.il` returns HTML; plain **`data.gov.il` returns the file**. Same path, same
  resource id. §12's "a wall is a fact about a host and a path" in its cheapest form: the fix
  was deleting four characters from a hostname.
- **The Statistical Abstract's geographic tables have an `x` suffix.** Chapter 2's tables are
  `st02_NN.xlsx`, except 04, 05, 06 and 11–16 — the ones with geography in them — which are
  `st02_NNx.xlsx`. The plain name returns a **2,056-byte SharePoint fake 200**, not a 404, so
  a size-and-magic check catches it and a `raise_for_status()` does not.

## 10. What the source is worth

- **2,968 units at ~3,000 people**, the finest geography here after the US, and fine enough
  to show the Haredi/secular boundary inside Jerusalem street by street.
- **~153,000 Druze, the largest count of them anywhere on this map**, in two populations the
  register does not distinguish and the geography does: the Galilee and Carmel villages, and
  the four Golan villages whose residents mostly hold permanent residency rather than
  citizenship.
- **The first observance split of Judaism on this map**, and the only instrument here that
  separates Israeli Jews at all.
- **Israel's Christians are one cell** of ~182,000 — Greek Orthodox, Greek Catholic, Latin,
  Maronite, Armenian, Syriac, and the ex-Soviet and migrant Christian population, all
  undivided. CBS's own footnote says the cell holds "Arab Christians and non-Arab Christians"
  and stops. The largest single loss in this file.

## 11. What is left

- **Sunni/Shia is not available and barely applies**: Israel's Muslims are overwhelmingly
  Sunni and the register does not record the distinction.
- **A Christian denominational split would need a different source** — the churches' own
  rolls, which would be `roll` on a different instrument and could not be mixed with these
  figures under §3.1.
- **The vintage.** The dashboard is the 2022 census; ST02-11x reaches 31.12.2024 and shows
  the register totals moving. A §3.4 re-levelling onto 2024 totals is possible and was not
  done, because the 2022 shares are what the fine geography exists for.
