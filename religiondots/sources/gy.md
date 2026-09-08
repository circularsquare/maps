# Guyana — Bureau of Statistics, Population and Housing Census 2012

`sources/gy.py` → `data/normalized/gy.csv`. 746,955 people, 10 administrative regions,
13 categories, **100% of the census drawn**.

| | |
|---|---|
| source | *Final 2012 Census Compendium 2: Population Composition*, Table 2.19 |
| url | `https://statisticsguyana.gov.gy/wp-content/uploads/2019/10/Final_2012_Census_Compendium2.pdf` |
| geography | administrative region (10) — 74,700 people each |
| categories | 13 named, no residual non-response |
| basis | `self_id` |
| year | 2012 |
| access | open, no login, no key, no bot wall. One 3.6 MB GET. |
| licence | Guyanese government publication; no explicit licence stated |

The 2022 census exists and has published **only a preliminary report**, which has no religion
table in it. 2012 is the current religion vintage and will be until GBS publishes 2022
composition tables.

## 1. A PDF-only office, and it cost an afternoon anyway

The Bureau of Statistics runs no dissemination platform of any kind — no PxWeb, no SDMX, no
API, no data portal, not even a bad one. Its entire output is a list of PDFs on a WordPress
`/publications/` page. By the rule this project believed in the morning of 2026-09-05 — *the
useful countries are the ones with a dissemination platform rather than a report series* —
Guyana is unusable.

It took an afternoon, because **the religion cross-tabulation is one page**. This is Kenya's
lesson (`sources.md` §11b) collecting a second time and it should now be treated as the rule
rather than the exception: **ask how many pages the table occupies before asking what serves
it.** A one-page table parses in an hour whatever it is wrapped in. Finding it took one
regex over table captions across 66 pages.

The captions are worth listing, because they are how the table was found and they show the
shape of what a small office publishes:

```
p 42  Table 2.17: Distribution of the Population by Religious Affiliation, Guyana: 2002 & 2012
p 46  Table 2.18: Growth and Changes in the Size of Religious ...
p 49  Table 2.19: Distribution of the Population by Religious Affiliation and Administrative Regions
p 49  Table 2.20: Percentage Distribution of the Population by Religious Affiliation and ...
```

2.19 and 2.20 are on the **same page**, and 2.20 is the percentage twin of 2.19 — Croatia's
trap (`spec` §12) in a PDF instead of a spreadsheet. A reader that took the first thirteen
numbers after each label off that page would get percentages for the bottom half of the rows
and nothing would complain. `gy.py` reads by y-coordinate and stops at the 2.19 block.

## 2. The Total column is clipped, and it parses as a number

The rightmost column of Table 2.19 overflows its cell in the PDF's text layer:

| row | text layer | truth |
|---|---|---|
| Anglican | `38,96` | 38,962 |
| Pentecostal | `170,2` | 170,289 |
| Hindu | `185,4` | 185,439 |
| Total | `746,9` | 746,955 |

**`38,96` is a valid number.** Nothing raises, nothing looks wrong, and every figure is off by
one or two digits in a way that is invisible unless you already know the national total. This
is `sources.md` §5a's family — a thing that reads as data and is not — in a new disguise:
not a truncated *file*, a truncated *cell*.

Every row total is therefore recomputed from the ten regions, and **the clipped string is
then asserted to be a prefix of that sum**. That turns the defect into a check: it fails if a
row is misparsed, and it keeps passing if GBS ever re-renders the PDF with a wider column.
The generalisable form: *where a source's own derived figure is unusable, do not simply drop
it — assert the relationship it still has to the figure you computed.*

## 3. The independent check is a second published table

`Table 2.17` on page 42 gives the 2012 national figure for every category, **uncut**, in a
different layout on a different page (its columns are Male/Female/Total for 2002 and again
for 2012). The ten regions sum to it exactly, category by category.

That is the check this project keeps asking for and rarely gets: *a quantity the parse does
not determine*. A misread region column, a transposed row or a shifted label cannot reproduce
thirteen exact national totals. Together with the two arithmetic identities inside 2.19 —

- the 13 categories sum to each region's own `Total` row, all 10 regions;
- the 10 region totals sum to 746,955, the published census population;

— the table is a **perfect partition in both directions**, which is the strongest
reconciliation state any source on this map has been in.

## 4. Non-response was prorated in, and this is the first source that does it

Table 2.19's own note:

> `'363 Religious Affiliation Not Stated' added to '16,331 No-Contact Persons' and
> '7,443 Institution Population' and prorated.`

So **24,137 people — 3.23% of Guyana — were distributed across the thirteen categories in
proportion, before publication.** There is no non-response column, because it was spent.

This matters for `spec` §3.5, whose rule is that non-response is reported and never filled.
Here the filling happened **upstream, in the office, and is not recoverable**: no published
figure separates the prorated share from the answered share, at any geography. So Guyana is
the country where "100% of the census is drawn" and "some of what is drawn is the office's
estimate" are both true, and the second fact lives in `note_public` because nothing on the
map can carry it.

It is not corrected. Undoing a proration means inventing the distribution it replaced
(§14.4). It is worth knowing that the effect is small and non-uniform: no-contact and
institutional population are urban-weighted, so Region 4 carries more of the 24,137 than its
share, and every Guyanese count here is inflated by roughly 3% with a slight urban tilt.

**A new entry for the §12 checklist:** *ask whether the office has already redistributed its
non-response, and read the table's footnote to find out.* Guyana states it plainly in four
lines under the table. A source that does this and does not say so would be undetectable.

## 5. The categories, national

| category | people | share | node |
|---|---:|---:|---|
| Hindu | 185,439 | 24.83% | `hinduism` |
| Pentecostal | 170,289 | 22.80% | `christianity.pentecostal` |
| Other Christians | 155,050 | 20.76% | `christianity.other` |
| Roman Catholic | 52,901 | 7.08% | `christianity.catholic` |
| Muslim | 50,572 | 6.77% | `islam` |
| Seventh Day Adventist | 40,374 | 5.41% | `christianity.adventist` |
| Anglican | 38,962 | 5.22% | `christianity.anglican` |
| None | 23,419 | 3.14% | `unaffiliated` |
| Methodist | 10,106 | 1.35% | `christianity.methodist` |
| Jehovah Witness | 9,602 | 1.29% | `christianity.witnesses` |
| Other | 6,324 | 0.85% | `other.gy` |
| Rastafarian | 3,496 | 0.47% | `rastafari` |
| Bahai | 421 | 0.06% | `bahai` |

**Every category resolved to a node that already existed except the residual**, so Guyana
cost the tree exactly one node (`other.gy`). See `taxonomy/gy2012.py` for the arguable calls.

Two things about this list are unusual. **Rastafari is counted by name** — 3,496 people —
which almost no census anywhere does, and no other source on this map. And **`Other
Christians` is 20.8%**, a fifth of the country left unnamed beside a list that is otherwise
generous; in Guyana that is mostly Baptists, Congregationalists, Lutherans, Moravians and the
Church of the Nazarene, all of which the tree could hold apart. A source naming them is the
biggest single upgrade available for this country.

## 6. `None` is a category name and pandas deletes it

`countries.py`'s `_gy_counts` reads `gy.csv` with `keep_default_na=False, na_values=[""]`. It
did not at first, and **23,419 people vanished with no error anywhere** — the string `None`
became `NaN`, failed to resolve in the taxonomy, and was dropped by the `node.notna()` filter
that correctly drops unmapped rows.

This is `spec` §12's Philippines trap, hit by the second country to have a category with that
name, and it is worth recording that **every check upstream of that line still passed**:
`gy.py`'s reconciliation is exact, `check_mapping.py` reports 746,955 on 13 nodes, and the
loss appears only in the dataframe `countries.py` hands to `scatter.py`. The symptom was a
single number in a debug print being 23,419 short. The rule stands exactly as written and it
now has a second country behind it.

## 7. What the map shows

Guyana is **the indenture map**, and it is unusually legible for ten units because the
regions are cut across the grain of the thing:

- **The coast is Hindu.** East Berbice-Corentyne 42.1%, Essequibo Islands-West Demerara
  37.7%, Mahaica-Berbice 34.1%, Pomeroon-Supenaam 33.2% — the sugar belt, in the order the
  estates were. Muslims sit on the same ground about a third as thick (Essequibo Islands
  11.8%, East Berbice 9.5%) because they came on the same ships to the same plantations.
- **The interior is its photographic negative.** Upper Takutu-Upper Essequibo — the Rupununi
  savannah — is **50.1% Roman Catholic and 0.4% Hindu**; Potaro-Siparuni is 39.8% Catholic.
  That is the Amerindian interior and the missions that worked it. The country's most Hindu
  region and its most Catholic one are 200 km apart with almost nothing between them, because
  almost nobody lives between them.
- **Barima-Waini is 39.9% Pentecostal and 33.8% Catholic** — three quarters of it in two
  churches.
- **Linden is a third Guyana.** Upper Demerara-Berbice, the bauxite region, is 36.0%
  Pentecostal and 14.8% Seventh Day Adventist, and holds both the highest no-religion share
  in the country (7.2%) and the highest Rastafarian share (1.3%). Afro-Guyanese, industrial,
  and almost untouched by the Hindu–Muslim coast 20 km away.
- **Cuyuni-Mazaruni**, the mining interior, is the Anglican and Adventist outlier — 17.5% and
  17.3%, both national highs — and carries 7.0% `Other`, eight times the national rate.

## 8. What the map cannot show, and it is not a small thing

**There is no Amerindian religion in this data, and its absence is a fact about the form.**
The 2012 census offered thirteen boxes and none of them was traditional practice, so the nine
Amerindian nations of Regions 1, 7, 8 and 9 answered one of the Christian categories instead.
The Catholic interior should be read as *the church people gave as their answer*, not as the
whole of what is practised there.

This is Ghana's exclusive-category undercount (`sources.md` §11b) outside Africa, and it
generalises further than that entry implied: **it is not an African phenomenon, it is a
phenomenon of censuses whose religion question was designed around the missionary
denominations.** Not corrected, because correcting it would mean inventing a magnitude
(§14.4). Recorded in `other.gy`'s node note, in `taxonomy/gy2012.py`, and in `note_public`.

## 9. Not done

- **The 2022 census.** Preliminary report only; no religion table published yet. When one
  appears this country should be re-based onto it (§3.4).
- **Anything below region.** GBS publishes religion at region and nowhere finer in this
  compendium. Whether a neighbourhood-level table exists in another volume was not checked
  beyond the four compendia listed on the publications page, none of which carries one.
- **Suriname**, next door, which asks the same question of a similarly mixed population and
  is the obvious companion. `statistics-suriname.org` answers 200 at the root and 404s the
  publication paths tried; see `sources.md`.
