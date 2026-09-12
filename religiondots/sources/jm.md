# Jamaica — STATIN, Census 2011, via the U.S. Census Bureau

`sources/jm.py` -> `data/normalized/jm.csv`. Boundaries and placement: `sources/jm_geo.md`.
Taxonomy: `taxonomy/jm2011.py`.

**2,678,981 people on 14 parishes, 19 religion categories, 97.7% drawn.** Two GETs totalling
1.9 MB, an exact partition, a free join, and one caveat that would have shipped a false map.

---

## 1. Why it was not built for a day, and what changed

§11j verified this file end to end on 2026-09-06 — workbook downloaded, religion sheet
parsed, partition summed, `GEO_MATCH` joined, Metadata read — and then **left the country
unbuilt**:

> **Jamaica is a category source on a geography that fails the floor.** … it is **14
> parishes**, below Albania's 12-unit rejection in §11d and far below North Macedonia's 80.
> … **This is a judgement call rather than a technical one, and it is Anita's.**

**Anita's answer, 2026-09-06:** *"i think especially for smaller countries, we dont need that
many regions for it to be a cool plot."* The floor is withdrawn — **spec §3.9b** — and with it
the only thing that was ever stopping this country.

The arithmetic had been on Jamaica's side the whole time. **14 parishes over 2.68M people is
191,356 per unit**, which is *finer* than Georgia's 11 regions at 340,000, and Georgia was
already drawn. §11j compared unit counts against a threshold instead of comparing the country
against the map.

## 2. The route

The U.S. Census Bureau series on HDX (§11h) — the national office's own tabulation,
transcribed, with the boundaries in the same file:

| resource | bytes |
|---|---|
| `jamaica.gdb.zip` | 1,472,454 |
| `jamaica_uscb_202302.xlsx` | 438,450 |

Nine sheets; **`Ethnicity and Religion`** is the one read. 15 rows — one country, 14 parishes —
and 39 columns, of which 8 are ethnicity and 20 are religion (19 categories plus `RLG_RTOTL`).

The Metadata sheet cites STATIN's *General Report Volume 1, Table 3.1*.

## 3. THE FOUR MISSING RELIGIONS — the caveat this country exists to demonstrate

**This is the country §11h's *read the Metadata sheet before the data sheet* rule was written
for**, and it is the sharpest example of it in the series.

STATIN's national religion universe is **2,683,105**. The 19 published columns sum to
**2,678,981**. The Metadata sheet explains the 4,124-person difference:

> *"Baha'i, Hinduism, Islam and Judaism were not included in parish tables by the Statistical
> Institute"* — **269 Bahá'í, 1,836 Hindu, 1,513 Muslim, 506 Jewish**.

**They are absent, not pooled**, and the arithmetic proves it rather than the sentence:

* every parish's 19 cells sum to its own `RLG_RTOTL` **exactly**, all 14 of them;
* the 14 parishes sum to the national row **exactly**, on all 19 categories;
* the gap appears **only** in the national `RLG_RTOTL`.

So the four religions are not hiding inside `Other religion`. They are simply not in the
parish tables.

**The failure mode this creates is the dangerous kind.** A build from the data sheet alone
would draw a Jamaica with **no Muslims, no Hindus, no Bahá'ís and no Jews at all** — and every
reconciliation it ran would pass, because the parish rows are internally perfect. Nothing in
the numbers is wrong; the *universe* is smaller than it looks.

`check()` therefore **asserts the gap is exactly 4,124** rather than tolerating a small
discrepancy. If STATIN or USCB ever re-release with those four included, this build fails
loudly instead of quietly changing what the map claims.

Those people are not drawn (§3.5), and `countries.py` carries a `gap` line so the map says so
while the blank is on screen (§6.12). Jamaica *has* all four communities — the Hindu and
Muslim populations descend from 19th-century Indian indenture — and this source will not
place them.

## 4. What the categories are worth

Nineteen cells. **The best denominational detail in the Americas outside ASARB**, and three
of them exist nowhere else on this map.

| category | people | share |
|---|---|---|
| Non-religious | 572,008 | 21.35% |
| Seventh Day Adventist | 322,228 | 12.03% |
| Pentecostal | 295,195 | 11.02% |
| **Other Church of God** | 246,838 | 9.21% |
| **New Testament Church of God** | 192,086 | 7.17% |
| Baptist | 180,640 | 6.74% |
| Other religion | 169,014 | 6.31% |
| **Church of God in Jamaica** | 129,544 | 4.83% |
| **Church of God of Prophecy** | 121,400 | 4.53% |
| Anglican | 74,891 | 2.79% |
| No Data | 60,326 | 2.25% |
| Roman Catholic | 57,946 | 2.16% |
| United Church | 55,360 | 2.07% |
| Jehovah's Witness | 50,849 | 1.90% |
| Methodist | 43,336 | 1.62% |
| **Revivalist** | 36,296 | 1.35% |
| **Rastafarian** | 29,026 | 1.08% |
| Brethren | 23,647 | 0.88% |
| Moravian | 18,351 | 0.68% |

**The Church of God family is the largest thing in Jamaican religion and nothing else here
counts it apart.** Four cells, **689,868 people, 25.7% of the country** — more than any single
denomination and more than the whole irreligious share. `taxonomy/jm2011.py` splits them
between `christianity.pentecostal` (the two Cleveland-Tennessee lineages) and
`christianity.holiness` (the Anderson lineage and the residual), and flags the two it is less
certain about.

**Rastafari is counted where it began.** `rastafari` has been a root in `branches.py` since
Czechia arrived with **190** of them; Jamaica has **29,026**. Read it as a floor — Rastafari
is a way of life more than a membership, its census enumeration is widely held to undercount,
and 1.08% is far below any cultural estimate. Not corrected (§14.4); said in `note_public`.

**Revivalist got a new node**, `afrodiasporic.revival` — Revival Zion and Pukkumina, out of
the Great Revival of 1860-61. Filed beside Umbanda and Candomblé rather than under
Christianity (spec §3.3), and **the geography supports that**: 3.58% in Saint Thomas against
0.68% in Saint Ann, which is where the Kongo-derived practice concentrated.

## 5. What the map shows

* **Irreligion is the largest single answer at 21.35%, the highest in the Americas here**, and
  it is urban: **34.06% in Kingston**, 26.0% in Saint Andrew, against **11.84% in Manchester**.
* **Adventism is enormous and rural** — 12.03% nationally, **20.81% in Hanover** and 6.44% in
  Kingston, which is the exact inverse of the irreligion pattern.
* **Revival is eastern**: Saint Thomas 3.58%, Portland 2.39%, against 0.68% in Saint Ann.
* **Rastafari is urban**: Kingston 1.54%, Saint Thomas 1.46%, against 0.80% in Manchester.

## 6. Two things left open

* **`Other religion` is 6.31% and its geography is sharp** — **2.92% in Kingston against
  14.18% in Westmoreland**, a fivefold spread with the peak in the west. By §9r's rule a
  residual with a sharp geography is a missing category rather than a mixture, and STATIN
  publishes no composition of the cell at any geography. Named as an open question in
  `branches.py`, not guessed at (§14.4). The same shape as Bosnia's Velika Kladuša peak,
  found the same day.
* **The 2022 census exists and is not the cheap route.** Jamaica enumerated in September 2022
  and STATIN's detailed tables were not located; this file is the 2011 census. §3.4's
  re-basing machinery would apply if they surface.

## 7. What is in the file and not used

The same sheet carries **eight ethnicity cells** (Black, Chinese, Mixed, East Indian, White,
Other, No data) keyed identically, and the workbook has six further tables — age-sex,
population, housing units, households, amenities and poverty. Not needed to draw religion, and
worth knowing they are there.
