# Belize — SIB, 2022 Population and Housing Census, General Characteristics Table 9

Wired 2026-09-07. 397,483 people, 6 districts, 11 drawn categories, **98.96% drawn**.

| | |
|---|---|
| source | Statistical Institute of Belize, **2022 Population and Housing Census**, `Census2022_GeneralCharacteristics.xlsx`, sheet `Religion_by_District` = **Table 9: Population by Religion, District and Sex: 2022** |
| basis | `self_id`, whole census population, all ages |
| geography | **6 districts** — ~66,000 people each, finer than Guyana's 10 regions (75,000) |
| categories | **12** plus the universe total; 11 drawn |
| drawn | **393,349 of 397,483 people, 98.96%** — the gap is `Don't Know/Not Stated` |
| licence | SIB publication, free to download and cite |

**One category is the reason to draw the country.** `Mennonite` is **15,440 people, 3.9% of
Belize**, and no other census anywhere on this map names Mennonites at all —
`christianity.anabaptist.mennonite` has existed since the United States arrived and only
ASARB has ever put anyone in it.

---

## 1. Access — one link, one 406, no wall

```
https://sib.org.bz/wp-content/uploads/Census2022_GeneralCharacteristics.xlsx    69,401 B
```

A plain link on `sib.org.bz/census/2022-census/`. The one thing worth recording:
**`sib.org.bz` answers `406 Not Acceptable` to `User-Agent: Mozilla/5.0` and `200` to a full
browser token.** That is not a bot wall — no challenge page, no cookie, no rate limit — it is
a content-negotiation rule that only inspects the UA string, and the fix is one header. It
cost one failed run and it would read as a block to anyone who stopped there.

The site is WordPress and its `wp-json` search endpoint is open, which is how the census page
was found in the first place (§12's WordPress rule again, after Benin and Zimbabwe).

Everything else on that page is also open and was checked: `Census2022_PopulationCTV.xlsx`
(population by city/town/village), the education, marital-status, housing and agriculture
workbooks, the Key Findings report and the 2022 migration report. **Only the General
Characteristics workbook carries religion.**

## 2. Six districts is the ceiling, and the finer tier does not help

SIB publishes population down to **city, town and village** — `Population_CTV` lists about
fifty places per district — and the `Admin_Area` sheet splits every district into its town
and rural halves. **Religion is published at district and nowhere else.** Spec §3.9b: take
the finest geography a country publishes and say what it therefore cannot show.

What that costs is specific and worth naming, because Belize's religions are sorted by
*settlement* rather than by district. The Mennonite colonies are particular places — Blue
Creek, Shipyard, Spanish Lookout, Little Belize — inside districts whose other 90% is not
Mennonite. Six districts cannot show that, and nothing here pretends to. The Kontur
placement layer at least puts each district's dots where its people are rather than over
empty bush (§6).

## 3. The figures are fractional, and that is SIB's practice

Every cell in this workbook is a real number. The national total is
**`397483.45623886667`**, not 397,483. SIB publishes *undercount-adjusted* census counts
throughout — the `Admin_Area` sheet gives 2010 the same way, `322423.8195952825` — the
adjustment being for coverage measured by its own post-enumeration survey.

They are carried as floats into `bz.csv` and rounded only where dots are made. **Nothing
here reconciles to the integer**, so every identity in `sources/bz.py` is asserted to a
relative tolerance of 1e-6 rather than to zero. All of them pass at that tolerance:

* the 12 categories sum to each district's own Total, on all 7 columns;
* the 6 districts sum to the national column, on all 13 rows;
* **Male + Female == Total on all 91 cells.**

The last is the check on the *read*, not on the census. The sheet lays every group out as
three columns (Total, Male, Female) and only Total is drawn, but all three are read, because
the sex columns are the only thing that would catch a district's block landing one
column-group left or right — every other identity reconciles inside one group whichever
columns were taken. Zimbabwe's panel rule (§9aj) applied to a workbook instead of a PDF.

## 4. `None` is a category name, and pandas deletes it

**Fourth sighting of §12's Philippine trap**, after `ph`, `gy` and `zw`, and Belize is the
worst of the four by share. SIB's no-religion cell is the literal string `None`. Default
`read_csv` parsing turns it into `NaN`; it then fails to resolve in the taxonomy; and every
reader that drops unresolved rows — which is all of them, correctly — removes **exactly 6
rows and 123,372.67 people, 31.04% of Belize**, with no error and no warning.

Measured rather than asserted: a default read of `bz.csv` returns 6 NaN `source_category`
rows summing to 123,372.66875731661.

`keep_default_na=False, na_values=[]` is load-bearing here, and `_bz_counts` **asserts the
category is present** after reading rather than trusting the flags to survive a future edit.
The reason it needs its own assert is that nothing else would notice: `sources/bz.py`
reconciles the workbook, not this read, so every check in the file would still pass and the
map would simply show a devout Belize.

## 5. District order is a trap, and it is why `geo_id` is COD's code

SIB prints its districts **north to south** — Corozal, Orange Walk, Belize, Cayo, Stann
Creek, Toledo — which is Belize's own national district order. **COD-AB codes them
alphabetically**, so `BZ01` is Belize District and not Corozal.

Numbering the table by row position, the way `sources/mw.py` legitimately does, would produce
a `geo_id` that disagrees with the boundary file on **five of six units** — and every total
in `bz.py` would still reconcile, because a total does not care which polygon it is paired
with (§9n's `TMA` lesson). So `bz.py` carries the pcode against the *name*, and
`sources/bz_geo.py` asserts the same mapping from the boundary side.

## 6. Boundaries and placement

See `sources/bz_geo.md`. Short version: COD-AB ADM1 is the census's district tier exactly,
the join is **6/6 both ways with no name variants at all**, and placement is Kontur's 400 m
grid (5,095 hexes) because six districts over 22,966 km² averages **3,828 km²** — five times
Jamaica's — and Cayo and Toledo are mostly uninhabited forest.

## 7. What the map shows

**Belize is Catholic in the north and Pentecostal in the west and south, with a very large
irreligious share everywhere.**

| | Catholic | Pentecostal | Mennonite | None |
|---|---|---|---|---|
| Corozal | **37.9%** | 5.3% | 8.9% | 22.8% |
| Orange Walk | 37.3% | 7.5% | **9.9%** | 24.2% |
| Belize | 33.4% | 5.5% | 0.5% | 30.8% |
| Cayo | 27.2% | **14.9%** | 4.2% | 34.7% |
| Stann Creek | 26.1% | 8.4% | 0.5% | **46.6%** |
| Toledo | 31.7% | 13.3% | 2.8% | 21.7% |

* **The Mennonite geography is the settlement history.** The Kleine Gemeinde and Old Colony
  communities arrived from Mexico and Canada in 1958 under an agreement granting exemption
  from military service and control of their own schools, and they farm the north: 9.9% of
  Orange Walk, 8.9% of Corozal, against 0.5% of Belize District and 0.5% of Stann Creek.
* **Baptists are 12.0% of Toledo** against 0.9% of Orange Walk — by far the most concentrated
  named Christian body, and Toledo is the Maya south.
* **Anglicans are 8.6% of Belize District** and nowhere else above 3.3%, which is the old
  colonial capital.

### The `None` figure is large and is drawn as published

**31.04%, 123,373 people — the highest irreligious share this map draws in the Americas**,
above Jamaica's 21.35%. It is also very unevenly spread: 46.6% in Stann Creek against 21.7%
in Toledo.

This is a large rise on the 2010 census, and **the reason is not in this source.** SIB
publishes no commentary on it, no question-wording note, and no comparison table for the
religion question specifically. Possibilities that cannot be separated here: a genuine
secularising shift, a change in how the question or its prompts were administered between
rounds, or an interviewer/coding effect. **It is drawn as published and flagged rather than
adjusted** (§14.4), `check()` in `sources/bz.py` prints the per-district spread on every run,
and `note_public` states the figure plainly.

## 8. What the question does not ask, which is the real limitation

SIB names **nine Christian bodies and nothing else at all**. There is no cell for Hinduism,
none for Islam, and none for any indigenous or Afro-Caribbean tradition, in a country that
has all of them. Everything else is `Other`, 6.32%.

`Other`'s geography is sharp — **11.9% in Orange Walk against 2.7% in Stann Creek**, a 4.4x
spread — which by §9r's rule means a missing category rather than a mixture. It does not
resolve to one thing and is not guessed at; see `other.bz` in `taxonomy/branches.py`.

**The Garifuna case is the one that shows why nothing is assigned.** Stann Creek is the
Garifuna district and has the *lowest* `Other` in the country. That is consistent with the
well-documented pattern of Garifuna practising *dugu* alongside Catholicism and answering a
census with the church — so a tradition that certainly exists is probably not in the residual
at all, and is instead invisible inside `christianity.catholic`. Naming that possibility is
as far as this source allows anyone to go.

## 9. Ethics (§14)

Nothing here is sensitive in §14.2's sense. Belize's religion question is voluntary and
ordinary, no group in it is persecuted by the state, the geography is coarse (66,000 people
per unit) so no small community is locatable, and the Mennonite communities — the most
identifiable group on this map — are publicly and deliberately visible. The one judgement
call is §8 above: naming what is probably *missing* from the count rather than filling it in.

## 10. Not done

* **The city/town/village tier**, because religion is not published on it (§2).
* **Splitting `Other`**, because SIB publishes no composition (§8). §14.4.
* **A 2010 comparison.** The 2010 census asked religion and the `None` jump (§7) would be
  worth quantifying properly, but `sources/bz.py` reads the 2022 workbook only and §3.4 is
  not attempted here — the district structure is unchanged since 1882, so it would be a
  genuinely cheap addition if anyone wants the change rather than the level.
