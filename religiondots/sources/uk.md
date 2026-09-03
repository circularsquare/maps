# United Kingdom — four censuses, three agencies, four religion questions

`sources/uk.py` → `data/normalized/uk.csv` (1.83M rows, **162 MB**).

**The UK has no census.** It has four, and treating them as one country's data is the
first available mistake. They differ in date, in question wording, in whether the
question is voluntary, in category depth, in the finest geography published, and — in
Northern Ireland — in *how many religion questions there are*. `uk.py` keeps them apart
with `source_id` and never forms a UK total.

| `source_id` | agency | census day | finest geography | finest categories |
|---|---|---|---|---|
| `uk_ew_census_2021` | ONS | 21 Mar 2021 | **Output Area**, 188,880 | 50 write-in, at MSOA |
| `uk_sc_census_2022` | NRS | 20 Mar 2022 | **Output Area**, 46,363 | 12, at Output Area |
| `uk_ni_census_2021` | NISRA | 21 Mar 2021 | **Data Zone**, 3,780 | 32, at LGD |
| `uk_ni_census_2021_brought_up_in` | NISRA | 21 Mar 2021 | **Data Zone**, 3,780 | 7, at LGD |

`basis` is `self_id` on every row — all four are census self-declaration.

> ⚠ **`data/normalized/` is not in `.gitignore`.** `data/raw/`, `data/geo/` and
> `data/processed/` are; `data/normalized/` is not, so a `git add -A` would try to commit
> a 162 MB `uk.csv`. Not fixed here — `.gitignore` is shared and other agents are working
> in this tree. Worth adding `data/normalized/` to it.

---

## 1. Northern Ireland asks TWO religion questions, and they are not the same variable

This is the thing to get right about NI, and the reason MS-B23/B24 carry their **own
`source_id`** in `uk.csv` rather than sitting alongside MS-B19/B20.

From NISRA's own guidance note (`data/raw/uk/ni_census-2021-guidance-note-religion-question-outputs.pdf`,
22 Sep 2022):

- **Q13, "current religion"** — *"What religion, religious denomination or body do you
  belong to?"* Options: Roman Catholic / Presbyterian Church in Ireland / Church of
  Ireland / Methodist Church in Ireland / Other, write in / None. → tables **MS-B19**
  (8 categories) and **MS-B20** (32 religion categories plus a total).
- **Q14, "religion of upbringing"** — *"What religion, religious denomination or body were
  you **brought up** in?"* — **asked only of respondents who answered "None" to Q13 or
  did not answer Q13 at all.** → tables **MS-B23** (4 categories) and **MS-B24** (7).

So MS-B23 is *not* a second measurement of the same thing; it is Q13's answer with Q14
substituted in wherever Q13 was "None" or blank. MS-B24 makes the substitution visible,
and it is large:

| | belong to | brought up in | published total |
|---|---:|---:|---:|
| Catholic | 805,151 | **+64,600** | 869,753 |
| Protestant and Other Christian | 710,992 | **+116,552** | 827,545 |
| Other religions | 25,519 | +2,994 | 28,514 |
| None | — | — | 177,360 |

**181,000 people — 9.5% of Northern Ireland — appear under a religion in MS-B23 that they
told the census they do not belong to.** NI's familiar 45.7%/43.5% "sectarian balance"
headline is the MS-B23 figure. The MS-B19 figure is 42.3% Catholic against 37.4%
(Presbyterian + Church of Ireland + Methodist + Other Christian), with 17.4% no religion.
Both are true and they answer different questions; the map must pick one, say which, and
never add them.

**Two more asymmetries, both from the guidance note:**

1. **MS-B23/B24 have no "not stated" category and MS-B19/B20 do**, because NISRA ran the
   CANCEIS imputation model on "religion of upbringing" but deliberately **not** on
   "current religion". 27,400 respondents (1.5%) left the religion questions incomplete;
   those records are modelled into MS-B23 and left as `Religion not stated` in MS-B19.
2. A further **58,700 whole-household non-responses (3%)** are modelled into *both*.
   NISRA's enumerated-vs-modelled table is worth reading: among the modelled records
   "No religion" is 26.3% against 17.1% among the enumerated. The imputation is not
   neutral with respect to religion.

`spec.md` §3.1's rule that bases are never mixed has a sibling here: **questions are never
mixed either.** MS-B19 and MS-B23 are the same basis (`self_id`) over the same people and
still cannot be added, because they measure different variables.

---

## 2. England & Wales — spec §3.4's split, and no Christian denominations at all

**TS030** (top-level, 9 categories + total) is published down to **Output Area**;
**TS031** (60 columns: 9 top-level plus 50 write-in sub-categories) stops at **MSOA**. Neither is rescaled to the other in
`uk.py`; both are emitted at their own geography and `reconcile.py` can do the §3.4 split.

The two do not use the same not-stated label — TS030 says `Not answered`, TS031 says
`Religion not stated` — and their England-and-Wales totals differ by 4 people
(59,597,540 vs 59,597,544), which is §5's perturbation.

**The gap TS031 does *not* close, and it is the big one: England and Wales publish no
Christian denominations.** 27,522,672 Christians, 46.2% of the population, and TS031's
detail hangs entirely off `No religion` and `Other religion`. There is no Anglican, no
Catholic, no Methodist, no Baptist row anywhere in the E&W census. Every other part of
these islands has that split — Scotland has three Christian categories, Northern Ireland
has 26, the Republic has fourteen. On the dot map, England and Wales will be one flat Christian
colour while Scotland, NI and Ireland are not, and that is a property of the *source*, not
of English religion.

**TS031 is hierarchical, and `uk.py` keeps the hierarchy in `source_category`.** ONS writes
its columns as `Religion (detailed): Other religion: Pagan`; `uk.py` strips only the
dimension name (`Religion (detailed): `) and keeps `Other religion: Pagan`. So the parent
path is recoverable, and **summing every TS031 row for a unit double counts** — take the
top-level set (`Christian`, `Buddhist`, …, `No religion`, `Other religion`,
`Religion not stated`) or the leaf set, never both.

### TS031, England and Wales — all 60 published columns

Top level: Christian 27,522,672 · No religion 22,162,062 · Muslim 3,868,133 ·
Religion not stated 3,595,589 · Hindu 1,032,775 · Sikh 524,140 · Other religion 348,338 ·
Buddhist 272,508 · Jewish 271,327. Total 59,597,544.

`No religion:` Agnostic 32,114 · Atheist 13,848 · Humanist 10,246 · Free Thinker 305 ·
Realist 76 · No religion 22,105,473.

`Other religion:` **Pagan 73,733** · Other religions 66,016 · Spiritualist 33,134 ·
Spiritual 31,611 · **Alevi 25,672** · **Jain 24,991** · Wicca 12,813 · Mixed Religion
11,402 · Ravidassia 9,572 · Shamanism 7,889 · Rastafarian 5,948 · Satanism 5,054 ·
Heathen 4,721 · Baha'i 4,716 · **Zoroastrian 4,090** · Taoist 3,724 · Druid 2,490 ·
Believe in God 2,414 · Pantheism 2,299 · Own Belief System 2,199 · Scientology 1,859 ·
Shintoism 1,375 · Deist 1,093 · Witchcraft 1,045 · Valmiki 1,034 · Theism 860 ·
Universalist 764 · Reconstructionist 742 · Traditional African Religion 661 ·
Druze 626 · Yazidi 413 · New Age 397 · Eckankar 329 · Vodun 257 · Brahma Kumari 235 ·
Thelemite 227 · Unification Church 202 · Mysticism 145 · Chinese Religion 112 ·
Native American Church 82 · Confucianist 76 · Animism 802 · Occult 490 ·
Church of All Religion 24.

That list is `spec.md` R2's target arriving intact: Alevi, Jain, Zoroastrian, Yazidi,
Druze, Vodun, Ravidassia, Valmiki, Shintoism and Rastafarian all get their own node from
one file. `Church of All Religion` at 24 people is the smallest published religion figure
anywhere in these four censuses and is a ring, not a dot (§4.3).

**Category kinds are mixed here too** (spec §2.3): `Mixed Religion` (11,402) is not a
religion, `Believe in God` and `Own Belief System` are not affiliations, `Spiritual` and
`Spiritualist` are two different published rows and only the second is a movement, and
`Occult`/`Witchcraft`/`Wicca`/`Pagan`/`Heathen`/`Druid` overlap heavily in ordinary usage
while being separate write-ins here. These are `unmapped.csv` decisions, not ingest ones.

---

## 3. Scotland — the denominational split E&W lacks, and Pagan at Output Area

**UV205**, Output Area, 12 categories + `All people`:

| category | count | share |
|---|---:|---:|
| No religion | 2,780,882 | 51.1% |
| Church of Scotland | 1,107,708 | 20.4% |
| Roman Catholic | 723,310 | 13.3% |
| Religion not stated | 334,740 | 6.2% |
| Other Christian | 279,435 | 5.1% |
| Muslim | 120,104 | 2.2% |
| Hindu | 29,961 | 0.6% |
| **Pagan** | **19,239** | 0.4% |
| Buddhist | 15,527 | 0.3% |
| Other religion | 12,421 | 0.2% |
| Sikh | 10,984 | 0.2% |
| Jewish | 5,829 | 0.1% |
| **All people** | **5,440,284** | |

(sums of the Output Area table; NRS's published population is 5,436,600 — see §5)

2022 was the first Scottish census in which most people reported no religion. Two things
are notable for the map: **Church of Scotland / Roman Catholic / Other Christian is a real
Christian split available at the finest geography**, which is where the Glasgow and
Lanarkshire Catholic/Presbyterian geography actually lives; and **Pagan is a tick box on
the Scottish form**, not a write-in, so 19,239 Pagans exist at Output Area resolution.
Scotland does not publish a write-in tail below "Other religion" (12,421), so its
long tail is shallower than England's even though its Christian detail is deeper.

Scottish Output Areas are small — **median 111 people, min 56, max 2,991, none empty**,
against England and Wales' median 306. Both are population-equalised by design, so
spec §8.2's "the geometry already carries the weight" applies to both and no population
grid is needed inside a unit.

---

## 4. Geography level and **vintage** (spec.md §8.1) — one safe case, one trap

| system | level in `uk.csv` | units | boundary set to join |
|---|---|---:|---|
| E&W | `output_area` | 188,880 | **OA 2021** (`OA21CD`) |
| E&W | `msoa` | 7,264 | **MSOA 2021** |
| E&W | `ltla` | 331 | **LAD as at 2021** — see below |
| E&W | `country` | 2 | England, Wales |
| Scotland | `output_area` | 46,363 | **OA 2022** |
| NI | `data_zone` | 3,780 | **DZ2021** (`DZ21`) |
| NI | `lgd` | 11 | **LGD2014** |
| NI | `country` | 1 | Northern Ireland (N92000002) |

**The trap: the E&W `ltla` level is the 2021 vintage, not today's.** Checked against the
file — Allerdale, Carlisle, Copeland, Barrow-in-Furness, Eden, South Lakeland, Harrogate,
Scarborough, Mendip and Sedgemoor are all present; Cumberland, Westmorland and Furness,
North Yorkshire and Somerset are all absent. Those **17 districts were abolished in April
2023** and replaced by 4 unitary authorities, taking England and Wales from 331 local
authority districts to 318. Join TS031's `ltla` rows to a current LAD boundary file and
Cumbria, North Yorkshire and Somerset vanish without an error — spec §8.1's Connecticut,
in England. Check the join in **both** directions and report both sides.

**The safe case, worth naming because it is the exception:** Scotland's 2022 Output Area
codes run `S00135307`–`S00181669`, a contiguous block starting immediately after the last
2011 code (`S00135306`). The two vintages are **disjoint**, so joining 2022 data to 2011
boundaries matches nothing at all and fails loudly. England and Wales' Output Areas do the
opposite — 2021 OAs reuse 2011 codes wherever the area was unchanged, so a wrong-vintage
join *partly succeeds* and only the split and merged areas go missing. That is the
failure mode §8.1 exists for.

Boundary files (**not downloaded — `data/geo/` is shared and another agent owns it**):
ONS Open Geography Portal for OA/MSOA/LAD 2021; NRS "2022 census geography products" for
Scottish Output Areas; NISRA for DZ2021 and LGD2014.

---

## 5. Every one of the four perturbs, and none of them sum

All four agencies apply **cell key perturbation** on top of record swapping, independently
at each geography. Aggregating a fine table therefore does *not* reproduce the published
coarse figure. This is not an ingest bug and must not be "fixed".

| | measured drift |
|---|---|
| **E&W**, sum of 188,880 OAs vs ONS's England-and-Wales row | **+344 on 59.6M = +0.0006%**; worst category `Other religion` +0.028%, `Sikh` −0.018% |
| **E&W**, TS031 total vs TS030 total | +4 |
| **Scotland**, sum of 46,363 OAs vs NRS's 5,436,600 | **+3,684 = +0.068%** — 100× the ONS drift |
| **NI**, sum of 3,780 Data Zones vs 1,903,175 | MS-B19 **+75**, MS-B23 **−17** |
| **NI**, MS-B20/B24 own totals vs 1,903,175 | +1 and −7 |

NRS says so on the face of the file: the UV205 CSV's own footer reads *"Data has been
perturbed"* and *"Cells might not sum to sub totals and totals due to these Statistical
Disclosure Controls."*

`uk.py` therefore reconciles England and Wales against ONS's published figures with a
**0.1% tolerance** rather than exact equality, and prints the drift per category. Compare
with spec §3.6's yardstick for the US roll data: ASARB's overshoot ran at 0.0065% of
population, and these are the same order or smaller — except Scotland's 0.068%, which is
ten times ONS's and worth remembering when a Scottish unit looks slightly wrong.

---

## 6. The religion question is **voluntary** in all three GB/NI systems

| | statutory position | not-stated | rate |
|---|---|---:|---:|
| England & Wales | voluntary since introduced in 2001 (Census (Amendment) Act 1991); the **only** voluntary census question | 3,595,589 | **6.03%** |
| Scotland | voluntary | 334,740 | **6.15%** (NRS quote 6.2%) |
| Northern Ireland | *"taking part in the census is required by law, but … there is no penalty for failing to complete questions on religion"* — NISRA, from the Census Act (NI) 1969 | 30,529 | **1.60%** |
| *(Ireland, for contrast — `sources/ie.md`)* | *no voluntary question; whole form compulsory* | *345,165* | *6.70%* |

**Northern Ireland's non-response is a quarter of everyone else's** despite the same legal
position, which is worth pausing on: 1.6% against 6.0–6.2%. Part of it is the imputation
described in §1 (CANCEIS fills the religion-of-upbringing question but not current
religion, and the whole-household modelling fills both), but the enumerated-only figure is
still 1.6%. In a place where the answer carries the most weight, the fewest people decline
to give it.

Non-response is the second- or third-largest category in England, Wales and Scotland and
must be carried as its own node, never redistributed (spec §3.2). ONS's own guidance is
that percentages are computed on the whole population, not on those who answered, and NRS
says the same explicitly — so the published shares already include the non-responders in
the denominator and no rescaling is needed or wanted.

---

## 7. Exact URLs and how to re-fetch

Nothing here needed a login, an API key, or fought bot protection.

```
# ---- England & Wales: ONS Census 2021 via Nomis bulk downloads ----
# TS030, top-level religion, includes census2021-ts030-oa.csv (188,880 OAs)
curl -sSL -o data/raw/uk/census2021-ts030.zip \
  https://www.nomisweb.co.uk/output/census/2021/census2021-ts030.zip
# TS031, religion (detailed), 60 columns; ctry/rgn/utla/ltla/msoa ONLY
curl -sSL -o data/raw/uk/census2021-ts031.zip \
  https://www.nomisweb.co.uk/output/census/2021/census2021-ts031.zip

# ---- Scotland: NRS Census 2022, all Output Area topic tables (73 MB) ----
# uk.py reads only "UV205 - Religion.csv" out of it; that member is extracted to
# data/raw/uk/sc_UV205_religion_OA.csv.  The media path carries a build hash and
# WILL change; find the current one on
#   https://www.scotlandscensus.gov.uk/documents/2022-output-area-data
curl -sSL -o /tmp/sc_oa.zip \
  "https://www.scotlandscensus.gov.uk/media/zz85kfinmf97whklasd98gfkadft5hj4f_Topic2H_20241120_1747/Census-2022-Output-Area-v1.zip"

# ---- Northern Ireland: NISRA Flexible Table Builder (Cantabular) ----
# The download endpoint is /en/custom/table.csv -- NOT /en/custom/data.csv,
# which 404s.  Variable names are the trap; these two are right:
curl -sSL -o data/raw/uk/ni_MS-B19_religion_DZ21.csv \
  "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=DZ21&v=RELIGION_BELONG_TO_DVO"
curl -sSL -o data/raw/uk/ni_MS-B23_religion_brought_up_in_DZ21.csv \
  "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=DZ21&v=RELIGION_BELONG_TO_OR_BROUGHT_UP_IN_DVO"
# swap v=DZ21 for v=LGD14 or v=SDZ21 for coarser geographies

# ---- Northern Ireland: expanded classifications, LGD level ----
# MS-B20 (33 categories) and MS-B24 (belong-to vs brought-up-in) are inside this
curl -sSL -o data/raw/uk/ni_census-2021-main-statistics-phase-1-all-tables.zip \
  https://www.nisra.gov.uk/system/files/statistics/census-2021-main-statistics-for-northern-ireland-phase-1-all-tables.zip

# ---- NISRA's guidance note on the two religion questions (essential reading) ----
curl -sSL -o data/raw/uk/ni_census-2021-guidance-note-religion-question-outputs.pdf \
  https://www.nisra.gov.uk/system/files/statistics/census-2021-guidance-note-on-use-of-religion-question-outputs.pdf
```

**Finding the NISRA variable names**, since guessing fails: open a table page such as
`https://build.nisra.gov.uk/en/custom/data?d=PEOPLE&v=DZ21&v=RELIGION_BELONG_TO_DVO` and
read the `table.csv` / `table.xlsx` / `table.csv-metadata.json` hrefs out of the HTML.
`v=RELIGION` alone returns `{"message":"Not Found: 404 … variable at position 2 does not
exist"}`.

**NISRA's PxStat portal at `data.nisra.gov.uk` is the same software as Ireland's CSO** —
`https://ws-data.nisra.gov.uk/public/api.restful/PxStat.Data.Cube_API.ReadCollection`
returns 1,170 tables — but its 2021 religion tables (`TS013`, `TS014`) are **Local
Government District only**. The Flexible Table Builder is the only route to Data Zone.

**The ONS "create a custom dataset" API is a dead end for R2**:
`https://api.beta.ons.gov.uk/v1/population-types/UR/dimensions/religion_tb/categorisations`
offers exactly two categorisations, `religion_tb` (10) and `religion_tb_5a` (5). The 58
write-in categories exist only in the published TS031 table.

---

## 8. Licences

All three agencies publish under the **Open Government Licence v3.0**, Crown copyright.
Attribution lines to use:

- England & Wales — *Source: Office for National Statistics licensed under the Open
  Government Licence v.3.0. Census 2021 © Crown copyright.* (Nomis is ONS's own
  redistribution channel and carries the same terms.)
- Scotland — *© Crown copyright 2024. Scotland's Census 2022, National Records of
  Scotland.* The string `Crown copyright 2024` is in the UV205 file's own footer.
- Northern Ireland — *Source: NISRA, Census 2021. © Crown copyright, Open Government
  Licence v3.0.*

OGL v3.0 permits commercial reuse and adaptation with attribution, so all four datasets
can ship.

---

## 9. What `uk.py` writes, and the one rule that keeps it safe

**No two tables share a `(source_id, geo_level)` pair.** That is deliberate: it means a
consumer who groups by those two columns can never silently mix a coarse table with a fine
one, or MS-B19 with MS-B23.

| `source_id` | `geo_level` | table | units | categories |
|---|---|---|---:|---:|
| `uk_ew_census_2021` | `output_area` | TS030 | 188,880 | 10 |
| `uk_ew_census_2021` | `msoa` | TS031 | 7,264 | 60 |
| `uk_ew_census_2021` | `ltla` | TS031 | 331 | 60 |
| `uk_ew_census_2021` | `country` | TS031 | 2 | 60 |
| `uk_sc_census_2022` | `output_area` | UV205 | 46,363 | 13 |
| `uk_ni_census_2021` | `data_zone` | MS-B19 | 3,780 | 8 |
| `uk_ni_census_2021` | `lgd` | MS-B20 | 11 | 33 |
| `uk_ni_census_2021` | `country` | MS-B20 | 1 | 33 |
| `uk_ni_census_2021_brought_up_in` | `data_zone` | MS-B23 | 3,780 | 4 |
| `uk_ni_census_2021_brought_up_in` | `lgd` | MS-B24 | 11 | 8 |
| `uk_ni_census_2021_brought_up_in` | `country` | MS-B24 | 1 | 8 |

Consequences of that rule, both intended:

- **MS-B19 is not emitted at LGD**, because MS-B20 occupies that slot and MS-B19's eight
  categories are recoverable from MS-B20's 32 by rolling up the `Christian: ` prefix.
- **TS030 is not emitted above Output Area**, because TS031 occupies MSOA and above and is
  strictly more detailed there.

**Rows whose count is 0 are dropped**, except each unit's own total (`Total: All usual
residents`, `All people`, `All usual residents`), which is always kept so every unit
appears exactly once at its level. Without that the file is roughly three times larger and
almost entirely zeroes — 188,880 output areas × 10 categories is 1.9M rows of which most
say "no Sikhs here". A missing row means zero, not unknown.

**Category strings are verbatim** except that the dimension name ONS prefixes to its
column headers is stripped (`Religion: `, `Religion (detailed): `), NISRA's `[note N]`
markers are removed and its embedded newlines collapsed to single spaces, and leading and
trailing whitespace is trimmed -- ONS ships `Religion (detailed): Muslim ` with a trailing
space, which would break any later join on the category string. Everything after
that — including the `Other religion: ` and `Christian: ` parent paths, and NI's curly
apostrophe in `Jehovah’s Witness` — is exactly as published.

---

## 10. Surprises, collected

1. **181,000 Northern Irish people are counted under a religion they said they do not
   belong to**, in the table that produces NI's best-known statistic (§1).
2. **England and Wales publish no Christian denominations at all** — 27.5M people, one
   category, while Scotland has three and Northern Ireland has 26 (§2).
3. **TS031's write-in tail hangs only off "No religion" and "Other religion"**, so its
   remarkable depth (Alevi, Jain, Yazidi, Vodun, Valmiki) is entirely non-Christian.
4. **Pagan is a printed tick box in Scotland and a write-in in England** — 19,239 Scottish
   Pagans at Output Area against 73,733 English and Welsh ones only at MSOA. Two censuses,
   two levels of resolution, for the same group, 100 miles apart.
5. **Every one of the four censuses perturbs, and Scotland's drift is 100× the ONS's**
   (§5). Exact reconciliation is not available anywhere in the UK.
6. **NI's religion non-response is 1.6% against Great Britain's 6%** (§6), and NISRA
   imputes one religion question but deliberately not the other.
7. **The E&W local-authority tables are on the pre-2023 331-district geography** and will
   silently drop Cumbria, North Yorkshire and Somerset against current boundaries (§4).
8. **Scotland's 2022 Output Area codes are disjoint from 2011's**, which makes a
   wrong-vintage join fail cleanly — the opposite of England's, which fails quietly (§4).
9. **The NISRA download endpoint is `table.csv`, not `data.csv`**, and the variable is
   `RELIGION_BELONG_TO_DVO`, not `RELIGION`. Both wrong guesses return a 404 HTML page
   with a 200-shaped filename (§7).
10. **ONS ships one column header with a trailing space** -- `Religion (detailed): Muslim `
    -- so a verbatim category string quietly fails to match `Muslim` from TS030 (§9).
11. **`Church of All Religion`, 24 people**, is the smallest published religion figure in
    any of these four censuses — and the ONS published it rather than suppressing it.
