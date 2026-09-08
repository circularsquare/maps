# Kiribati — Kiribati NSO, 2015 Population Census, Report Volume 1, Table 6

Built 2026-09-08. `sources/ki.py`, `sources/ki_geo.py`, `sources/ki_grid.py`,
`taxonomy/ki2015.py`. 110,136 people, **24 inhabited islands**, 14 categories, nothing derived
and nothing excluded.

The queue said *"religion is a tabulated census topic and the Census Atlas 2022 maps it;
`nso.gov.ki/download/<id>/` is an open Download Monitor catalogue and nobody has swept it."*
The catalogue turned out to be WP File Download rather than Download Monitor, the Atlas map
turned out to be a raster, and the usable table was in a report neither of those pointed at.

---

## 1. Access — the fifth Pacific office on the same plugin

`nso.gov.ki` is WordPress with an open REST API. Its `/download/<id>/` route 404s on low ids
because it is **WP File Download**, not Download Monitor: the real shape is
`/download/<cat>/<slug>/<fileid>/<file>` ([[reference_wpfd_sweep]]), the fifth Pacific office
here to run it after Fiji, PNG, the Solomon Islands and Tonga. `id=0` on its AJAX route returns
the whole library: **482 files**.

The library is organised **by island** — 23 island categories with 15 files each — which looks
exactly like what this map wants and is a trap; see §3.

## 2. Three candidate sources, and why 2015 wins

| | year | tier | categories | verdict |
|---|---|---|---|---|
| Per-island workbooks | **2005** | **village** | 7–8, varying | finest, 20 years stale |
| Report Vol 1 Table 6 | **2015** | **island** | **14** | **built** |
| 2020 report Table G-3 | 2020 | national | 11 | newest, no geography |

- **The 2020 census publishes religion nationally only.** Its report has `Table G-3: Total
  population of Kiribati by sex and religion` and nothing subnational. The **Census Atlas 2022
  has `Map 18: Religious affiliation by island`** and that page carries **66 characters of
  text** — it is a raster, and no numbers come off it. The Atlas's Table 9 *is* extractable and
  is national.
- **The 2005 island workbooks go down to village** (`Table 7: Population by village, sex and
  religion`), which would be the finest Pacific tier after Tonga. They are twenty years old and
  **their column sets differ island to island** — Beru has an `AOG` column Abaiang does not — so
  a parser has to reconcile per-island headers. Left on the table deliberately.
- **2015 is the newest year published with a geography and has the longest list of the three.**

## 3. The table, and the outside witness

`Table 6: Population by island, sex and religion: 2015`, four pages, 24 inhabited islands at
4,589 people each.

- The 24 islands sum to the printed national row **exactly, on all fifteen columns**.
- Every island's categories sum to its own total.
- **UNSD Demographic Yearbook table 28 reproduces the national row on all fourteen categories,
  to the person** — from Kiribati's own return, not from this PDF. Same external check Tonga got
  (§9bj) and Samoa could not have (§9bk).

### The oracle's labels for this country are wrong and its numbers are right

Worth its own heading because it would have wasted a session. UNSD table 28 renders `KPC` as
**`Kempsville Presbyterian Church`** — a false expansion of the initials, which are the
**Kiribati Protestant Church** — and its **1995 row is mangled outright**: `African Methodist
Episcopal Church` 54.3%, `Arya Samajist` 37.9%, `Bengali` 7.8%, for a country that has none of
the three. It also splits the Witnesses under their Gilbertese name `Te koaua` alone.

`ki.py` therefore pairs the two sources on an **explicit alias table** rather than on the
string, so a disagreement about a NUMBER can never hide behind one about a NAME. All fourteen
numbers agree. **The lesson generalises: a Yearbook label is a lookup somebody applied, and a
lookup can be wrong; the count is the part that was forwarded.**

## 4. Parsing — the only trap is the stacked header

Table 6's rows extract cleanly as `label + 15 figures`, with `-` for zero, so this needed no
column geometry ([[reference_pdf_table_geometry]] was not necessary). The one trap is that the
header is **three stacked lines**, and its fragments (`Seventh Latter Jehova's`) look exactly
like island names to a name-then-Total walker — a first pass captured the national row as an
island called *"Seventh Latter Jehova's"* and doubled the country.

The fix is to **anchor on the header's bottom row and ignore everything above it on each page**,
and to assert that row's exact token string before reading a number, so a re-typeset table fails
instead of silently shifting a column.

## 5. The antimeridian, where the usual check is the wrong check

**Kiribati's own bounding box legitimately spans 351 degrees of longitude.** The Gilberts are at
173 E and Kiritimati at 157 W, 4,000 km apart. Every other country on this map gets a
country-level width assertion because a 180-crossing polygon reprojected carelessly comes out
spanning the globe ([[reference_antimeridian]]) — and here that assertion **fires on correct
data**, so the tempting move is to switch it off, which is precisely how a real tear gets missed
a year later.

**So the check is per polygon instead.** No island may span more than 3 degrees; the widest is
Kiritimati at about 0.7. `ki_grid.py` does the same per hexagon (widest 0.0101 degrees). The
country is then allowed to be as wide as it really is, and a torn island is still unmissable.

Fiji (§9bd) needed the **opposite** treatment, because its *provinces* genuinely cross 180 and
had to be stitched. None of Kiribati's islands does. The two countries are the pair worth
remembering: same ocean, opposite fix.

Consequences elsewhere: the dot bbox `tiles.py` computes runs `[-160.4, 176.8]` and frames
badly, so the entry carries an explicit `view` in Fiji's beyond-180 convention,
`[169.0, -3.5, 203.5, 5.5]`, which is the inhabited extent.

## 6. The join and the placement

COD-AB `cod-ab-kir` ADM2, 36 island polygons, joins **24/24 on the name**, with five
contractions the report uses for the compass-point pairs (`NTarawa`, `STarawa`, `NTabiteuea`,
`STabiteuea`, `Teeraina`). The **twelve unclaimed polygons are all genuinely uninhabited**: five
in the Line Islands (Malden, Starbuck, Millenium/Caroline, Vostok, Flint) and seven in the
Phoenix group, where Kanton and its 20 people are the only settlement.

**39% of Kontur's people fall outside every island and are snapped, not dropped** — the highest
share of any country on this map, and it is not a join problem. An atoll is a strip of land a
few hundred metres wide between ocean and lagoon, so nearly every populated cell is a shoreline
cell. 100% of the strays are within 700 m and **9 people in the country are dropped**.

r = 0.964 over 24 units against a best of 0.72 over 2,000 random pairings.

## 7. What the census shows

**The mission partition of the Gilberts is still almost perfect after 150 years.** The chain runs
Catholic in the north and Protestant in the south and the two ends invert:

| island | Catholic | KPC |
|---|---:|---:|
| Butaritari | **82.5%** | 13.2% |
| Makin | 79.8% | 15.9% |
| Marakei | 76.8% | 15.6% |
| Abaiang | 75.3% | 16.8% |
| … | | |
| Beru | 29.4% | 65.3% |
| Onotoa | 27.1% | 67.1% |
| Tamana | 2.0% | 95.8% |
| Arorae | **1.4%** | **98.0%** |

The Sacred Heart mission worked the northern islands and the Congregational missions the
southern ones, and the share falls almost monotonically down 600 km of chain. **Arorae is 98.0%
Protestant and Butaritari is 82.5% Catholic.**

**South Tabiteuea is 12.0% Bahá'í**, against 2.1% nationally — the sharpest minority
concentration in the country and the sort of thing only an island table shows.

## 8. One new node, and one thing 2015 cannot show

**`christianity.reformed.congregational.kpc`** — the Kiribati Protestant Church, 34,464 people,
**the fifth and last of the Pacific Congregational set** beside `.cccs` (Samoa, §9bk), `.cicc`,
`.ekt` and `.niue`. It is unusual in the set for descending from **two** Congregational missions
rather than one: the American Board from 1857 by way of Hawaiian pastors, and the LMS from 1870
by way of Samoan and Tuvaluan ones.

**The 2014 union is not in it, and the census year is why.** In 2014 the church reconstituted as
the **Kiribati Uniting Church** — a union of Congregationalists, Evangelicals, Anglicans and
Presbyterians — and about ten thousand members, mainly Congregationalists, refused and re-formed
a separate KPC. The **2015 census still prints one cell**; the **2020 census prints KUC at 21%
and KPC at 8%**. This map draws 2015, so it draws them together. Splitting the 2015 cell on
2020's ratio was considered and rejected: §2.6 forbids using a later ratio as a magnitude for an
earlier year, and it would be a modelled split of the country's second-largest body.
`christianity.united` is where the KUC half belongs if a year that separates them is ever drawn.

**`other.ki`** holds two cells, 918 people. 832 is NSO's own `Other`, a real tail after thirteen
named answers. The other 86 are **`Te Ran`, and this is an admitted failure to identify**: it is
a printed cell in 2015 and again in 2020 (89 people), so the office treats it as a body worth
naming, but nothing in the 2015 report, the 2020 report, the Census Atlas or any reachable
secondary source says what it is. It goes to the country's unclassified cell rather than
`christianity.other` because calling it Christian would be a claim nothing supports.

## 9. What is left

- **The 2005 village tables**, if the per-island header reconciliation is worth doing. Religion
  by village on 23 islands would be the second-finest tier in the Pacific after Tonga.
- **The 2020 census at island level exists inside KINSO** — the Atlas mapped it — and is not
  published as numbers. It is the one request that would improve this country most.
- **`Te Ran`.** One line in `ki2015.py` when somebody identifies it.
- The DYB has Kiribati for 1995, 2010 and 2015, and the 2015 report's annex A4 carries a
  **religion series back to 1931**, which nothing here draws.
