# Malawi — NSO, 2018 Malawi Population and Housing Census, Table E5

Wired 2026-09-06. 17,563,749 people, 32 districts, 10 drawn categories.

| | |
|---|---|
| source | National Statistical Office, 2018 PHC **Main Report, Table E5** (pages 134-138) |
| basis | `self_id`, denomination question, whole census population |
| geography | **32 districts** — ~549,000 people each |
| categories | **10** plus the universe total; all 10 drawn |
| drawn | **17,563,749 people, 100%** — no residual, no `not stated`, no gap |
| licence | NSO publication, free to download and cite |

**The first source on this map to name a single Presbyterian body, and the first African one
to count Anglicans apart.** Malawi asks a *denomination* question rather than a religion
question, and the shape of what that buys and costs is the whole country: eight of the ten
answers are Christian groupings, the other two are Islam and No Religion, and there is no
cell anywhere for Buddhism, Hinduism, Judaism or the Bahá'ís at the drawn geography.

It is also, unusually, **complete**. The ten denominations sum to 17,563,749 on all 36
printed rows, and that figure is the entire 2018 census count. There is no non-response
category, no residual row and no §3.5 gap of any kind — which is true of almost nothing else
here.

---

## 1. The office publishes nothing and the download route is open anyway

`www.nsomalawi.mw` is a Nuxt application. Its census page is client-rendered and the
server-side payload carries **no publication links at all** — the six PDFs in the raw HTML
are strategy documents in the site chrome, and the report list arrives from an API.

That API is `cms.nsomalawi.mw`, and **§12's compare-the-404s test comes back positive on
it**: `/api/nonexistent-path` returns a 404 JSON body while `/api/census` returns **401**.
A live route behind auth, not a missing one — so the listing is closed.

**The DOWNLOAD route beside it is wide open, and the filename is decorative.**

```
https://cms.nsomalawi.mw/api/download/<id>/<anything>.pdf
```

`/api/download/270/a.pdf` and `/api/download/270/2018-Malawi-Population-and-Housing-Census-Main-Report.pdf`
serve the identical bytes. The path segment is ignored; only the id selects the file.

**And the real filename comes back in `Content-Disposition`**, which turns the id space into
a catalogue that can be read with HEAD requests and no bodies at all. Sweeping ids 1-700
enumerated **575 files**, among them:

| id | file |
|---|---|
| **270** | `2018-Malawi-Population-and-Housing-Census-Main-Report.pdf` (8.05 MB) — **the source** |
| 271 | `2018-Population-and-Housing-Census-Preliminary-Report.pdf` |
| 294-325 | `2018-Census-District-Report-<name>.pdf`, one per district |
| 262 | `Population-Projections-2018-2050.pdf` |
| 225 | `2018-Statistical-Yearbook.pdf` |

**The generalisable form.** A closed listing API and an open object route is a common CMS
shape, and the thing that makes it searchable is that the server names the file it is
serving. Before recording an office as "publishes through an app with no endpoints", check
whether *one* download URL is reachable, and if it is, whether its id is a small integer and
its response carries a filename. Both were true here and the whole library fell out of it in
four minutes. (Burkina Faso's INSD has the identical shape at
`insd.bf/fr/file-download/download/public/<id>` — see sources.md §11p.)

## 2. 32 districts is NSO's ceiling, and the district reports are what prove it

This is §3.9's trade, and the office made it. The Main Report is 311 pages and religion
appears on **six of them**: a national table and figure at page 19, and Table E5 at 134-138.
Nothing else in the report crosses religion with anything.

The obvious place for a finer tier is the **32 per-district reports** (ids 294-325, ~1.4 MB
each), which are exactly what a Traditional Authority-level religion table would live in.
Checked: `2018-Census-District-Report-Mzimba.pdf` is 38 pages and **the word "religion" does
not appear in it**, nor does "denomination". The district reports carry population, housing,
agriculture and education, and no religion at all.

So district is the ceiling. COD-AB ships an ADM3 layer of **433 Traditional Authorities** and
there is nothing to put on it.

**32 units for 17.56M people is ~549,000 each**, which is finer per head than Kenya's 47
counties (1.0M) and than Ghana's 16 regions, and coarser than Ghana's 272 districts.

## 3. The category list, and the two places it is not what it looks like

```
Total                                        17,563,749   universe
Catholic                                      3,028,435   17.24%
CCAP                                          2,498,969   14.23%
Seventh Day Adventist/Baptist/Apostolic       1,644,829    9.36%
Anglican                                        410,633    2.34%
Pentecostal                                   1,332,420    7.59%
Other Christian Denominations                 4,666,337   26.57%
Islam                                         2,426,754   13.82%
Traditional                                     186,284    1.06%
Other Denomination                              992,304    5.65%
No Religion                                     376,784    2.15%
```

**`Seventh Day Adventist/Baptist/Apostolic` is three traditions in one cell**, and it is the
one arguable mapping call in the country. The tree holds Adventists, Baptists and the African
Apostolic churches in three different places, and every person in this cell belongs to one of
them — but the census does not say which, and the three are not the same kind of thing at
all. It goes to `christianity.sdabaptistapostolic`, a node added for Malawi that holds the
merge and names what is in it. Splitting it by assumption would invent three numbers;
`christianity.other` would file 1.6 million people as bodies with no branch when the truth is
that they have three. See `taxonomy/mw2018.py` and the node's own note.

**The grouping is not arbitrary and its geography says so.** The cell is 29.1% in Neno, 21.5%
in Thyolo, 21.0% in Mwanza and 17.6% in Chikwawa, against 9.4% nationally and 2.2% in
Machinga: one contiguous block over the Shire highlands and valley, where the Seventh-day
Adventist Malamulo mission has been since 1902 and where the Apostolic churches are
strongest. A merge of three unrelated things would look like noise on the map. This does not.

**`Other Denomination` is partly known and cannot be separated.** Table 3.4 on page 19 of the
same report splits the identical national figure three ways —

```
Buddhism                          5,506
Hinduism                          3,211
Other non-Christian Denomination  983,587
                                  -------
                                  992,304   = Table E5's `Other Denomination`, exactly
```

— and **no table anywhere in the report gives any of the three a geography**. So 8,717
Buddhists and Hindus are drawn inside `other.mw` and there is nothing that would let them
out. This is §3.9's trade made *across two tables of one publication* rather than inside one,
which is a shape worth recognising: the finer category list and the finer geography are both
there, in the same document, and they never meet. `sources/mw.py` reads Table 3.4 anyway,
because it is the only independent reading of that column and its sum is a real check.

## 4. The table is printed sideways, and that is the whole parse

Table E5 is **rotated 90° on the page**. Every text line has `dir == (0, -1)`; each printed
*column* is one area; the eleven figures run down it in the header's order.

**Reading the page the normal way parses without error and transposes the country.** Group
words into horizontal lines, as every other PDF source here does, and each "line" you get
back is one *denomination* across many districts — in an order that changes page to page as
the areas do, with no total anywhere disagreeing. The first probe of this file produced
eleven tidy rows of 25 numbers each and they were columns.

So `sources/mw.py` groups lines by **x**, sorts each column by descending y, and requires
every data column to hold exactly twelve entries: a name and eleven numbers. Title lines,
the stub header and the wrapped fragments of `Seventh Day / Adventist/ / Baptist/ /
Apostolic` all have the wrong count and drop out on their own.

**There are three panels, not one.** Pages 134-138 hold Both sexes / Males / Females one
after another, 36 area columns each, 108 in total. They are identified by the first column's
name (`Malawi`, `Males`, `Females`) rather than by position. Only the first is drawn — and
the other two are read anyway, because **`Males + Females == Both sexes` on all 396 cells is
the check on the transposition**, and it is the only check that would catch a column landing
in the wrong panel.

**`-` is an in-band zero and dropping it shifts a whole column.** One cell has it: Likoma's
`Traditional`. A numeric regex drops the token, the column holds eleven entries instead of
twelve, and every remaining figure moves up one row — Likoma would be recorded with 156
traditionalists and 32 people of other denominations and no No-Religion cell at all, and
**nothing downstream would disagree**, because the column-sum check would be comparing a
shifted column against a shifted total. Sri Lanka's trap (§9j) with a smaller blast radius
and the same shape. The parser maps every dash form to 0 and raises on any other
non-numeric token.

## 5. The four cities are peers of their districts, not parts of them

Mzuzu, Lilongwe, Zomba and Blantyre Cities are printed **beside** Mzimba, Lilongwe, Zomba and
Blantyre, not inside them. That is 2,115,867 people, **12.0% of the country**, and getting it
backwards either double-counts them or loses an eighth of Malawi into the wrong polygons.

Nothing in the table says which it is — the city rows are typographically identical to the
district rows. §12's shape-3 trap exactly, and the only thing that sees it is the parent sum:

* the seven Northern units sum to the `Northern` row on all eleven columns;
* the ten Central and fifteen Southern units likewise;
* the three regions sum to the `Malawi` row on all eleven columns.

All three hold exactly, so the 32 are disjoint. `sources/mw.py` asserts all of it rather than
assuming it, and **COD-AB's ADM2 independently carries the same 32 units in the same 7/10/15
split** (`sources/mw_geo.md`).

## 6. What the map shows

**The mission map of the 1880s, still legible.** CCAP is the 1924 union of three missions and
all three are visible: Livingstonia in the north (Mzuzu City 28.0%, Mzimba 23.9%, Rumphi
22.7%), Nkhoma in the centre (Lilongwe City 23.2%, Dowa 22.4%), and 8.8% across the whole
south, where Blantyre synod is the smallest of the three.

**Likoma is 74.6% Anglican** — the sharpest single-denomination figure of any Malawian
district, on an island of 18 km² where the Universities' Mission to Central Africa put its
cathedral in 1903. Ntchisi (21.5%) and Nkhotakota (15.3%) are the lakeshore stations behind
it, and then it stops dead: Nkhata Bay is fourth at 4.2% and twenty-two districts are under
2%. Those three units hold **33.8% of Malawi's Anglicans on 4.1% of its people**.

**The Muslim south is one block, not a scatter.** Mangochi 72.7%, Machinga 67.0%, Balaka
34.7%, Salima 30.7% — against 13.8% nationally and **0.08% in Chitipa** on the Tanzanian
border, a 900-fold range across 32 units. This is the Yao lakeshore, converted along the
19th-century trade routes from Kilwa, and it sits on the southern lake in one contiguous
piece.

**Traditional religion is Dedza.** 6.13% there against 1.06% nationally: one district holds
**more than a quarter of every traditionalist NSO counted**, on 4.7% of the population.
Mzimba (3.25%) and Lilongwe (2.76%) follow. Read the national figure as a floor — see §7.

**The largest cell in the country is a residual.** `Other Christian Denominations` is 26.6%,
larger than the Catholics, and it runs from 48.3% in Nkhata Bay and 48.0% in Phalombe to 5.5%
on Likoma. A residual that varies ninefold across districts is carrying something specific
where it peaks (§9r's rule) and this source cannot say what. Most of it is Malawi's very
large independent and Zion church sector.

**And one thing that is the opposite of the usual shape:** `No Religion` is 2.15% nationally
and **7.57% in Lilongwe district**, a third of every unaffiliated person in the country —
while **Lilongwe City beside it is 1.73%**. That is a rural figure, not a capital-city one.

## 7. `Traditional` is a floor, and the reason is the question

sources.md §11b's continental rule applies here as sharply as anywhere: **every African
census that offers `Traditionalist` as a box exclusive of the Christian and Muslim boxes
undercounts it, by an unknown amount**, because traditional practice commonly accompanies one
of those rather than replacing it.

Malawi is a strong case for it. The Nyau societies of the Chewa — whose Gule Wamkulu is on
UNESCO's intangible heritage list and is practised across Dedza, Lilongwe, Mchinji, Ntchisi
and Kasungu — are not an alternative to church membership for most of their members; they
coexist with it, and the relationship with the Catholic and CCAP missions in exactly those
districts has a long documented history. A form with one box for `Traditional` and another
for `Catholic` cannot see anyone who is both.

1.06% is what the census measured. It is not what is there. `note_public` says so.

## 8. Ethics (§14)

Nothing here needs a §14 conversation. Malawi's state published this table itself, at this
geography, with no minority drawn finer than the office drew it; there is no persecuted group
in the category list and no cell whose mapping could identify one. The one category that
could carry a sensitivity — `Traditional` — is *under*-counted by the question's design and
is documented as a floor rather than a measurement.

The 2018 census is seven years old at the time of writing and Malawi's religious geography
has not been rearranged by violence against any drawn category in that time, so §11j's CAR
test is passed rather than dodged: the vintage note is a note, not a repair.

## 9. Not done

* **Traditional Authority level does not exist.** 433 TAs have boundaries in COD-AB and no
  religion table anywhere. The 2018 microdata would reach it and NSO does not publish
  microdata; there is no IPUMS Malawi 2018 sample either (IPUMS holds 1987, 1998 and 2008).
* **The 2018 census is the current one.** Malawi's next PHC is due 2028.
* **Buddhism and Hinduism cannot be split out of `other.mw`** — §3 above. If NSO ever
  publishes Table 3.4's categories by district, that is one join away.
* **`Other Christian Denominations` (26.6%) has no African-Instituted split.** This is
  gh2021.py's position before Kenya supplied `christianity.africaninstituted`, at twice the
  size. Any Malawian source that separates the Zion and Apostolic churches would open the
  largest cell on this map's Malawi.
