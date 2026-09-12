# Trinidad and Tobago — CSO, 2011 Population and Housing Census Demographic Report, Table 8

Wired 2026-09-07. 1,322,546 people, 15 municipalities, 16 drawn categories, **88.90% drawn**.

| | |
|---|---|
| source | Central Statistical Office, **Trinidad and Tobago 2011 Population and Housing Census Demographic Report**, **Table 8: Non-institutional population by sex, age group, religion and municipality**, pp. 130–183 of 442 |
| basis | `self_id`, **non-institutional** population, all ages |
| geography | **15 municipalities** — ~88,000 people each, between Guyana's 74,700 and Jamaica's 191,000 |
| categories | **17** plus the unit total; 16 drawn |
| drawn | **1,175,747 of 1,322,546 people, 88.90%** — the gap is `Not Stated` |
| licence | CSO publication, free to download and cite |

**Three categories are the reason to draw the country**, and two of them are new nodes:
`Orisha` (11,918) and `Baptist-Spiritual Shouter` (75,002) are the **first census counts of
either tradition anywhere on this map** — 86,920 people between them, 6.6% of the country,
more than its Muslims. `Rastafarian` (3,615) is the third census count of Rastafari here,
after Jamaica and Saint Vincent.

---

## 1. The source PDF the publisher links is truncated

**This is the finding worth carrying out of this country**, and it is in `sources.md` §11t
and `spec.md` §12 as well.

CSO serves the same report at two paths:

```
/wp-content/uploads/2019/03/TRINIDAD-AND-TOBAGO-2011-Demographic-Report.pdf     281,190 B  BROKEN
/wp-content/uploads/2020/01/2011-Demographic-Report.pdf                       7,013,046 B  INTACT, 442pp
```

The broken one is the copy the site links and the one CSO's own WordPress media search
returns first. It is not a redirect, an error page or a partial download:

* it **starts `%PDF-1.4`**;
* the server's **`Content-Length` matches the delivered bytes exactly** — 281,190 both sides,
  so the transfer was faithful;
* it **ends mid-stream** (`...teDecode>>\nstream\nx\x9c`) with **no `%%EOF`**;
* **PyMuPDF opens it, sets `is_repaired=True`, and reports `page_count = 0` without
  raising.**

So the file was already damaged when it was uploaded. **A complete download is not an intact
file — check the trailer, not the byte count.** This is §9k's *"a read that succeeds is not
a read that returned data"* one layer earlier: there the GDB opened and returned zero
features, here the PDF opens and returns zero pages.

`fetch()` in `sources/tt.py` asserts `%PDF-`, then `%%EOF` in the last 4 KB, then
`page_count == 442`.

### How the good copy was found, which is also the general lesson

Wayback CDX over `cso.gov.tt*` does hold two intact captures — a 16,969,403 B one from 2013
under the old Drupal path and the 5,809,837 B 2020 one. But **the publisher's own uploads
directory was enumerated first and had a live copy**, and a live copy is the one to cite.
The order to try is: the same publisher's other paths, then the archive.

## 2. Fifteen municipalities, and the tier is confirmed twice

Table 8 cuts by **municipality**: nine regional corporations, three boroughs, two cities and
Tobago. The tier is confirmed independently by CSO publishing **those same fifteen as
separate `... Individuals.xlsx` workbooks** on the same site.

Religion is not published finer. CSO *does* publish a **2011 Community Register** — 3,619
rows of community and enumeration district — but it carries population, households,
buildings and dwelling units only, with no religion. Checked, because §9p's lesson is that a
finer level can hide inside a coarser file. It does not here.

## 3. The parse, and the four things that would have broken it silently

The table is a fixed sequence — 17 unit blocks, each a unit label then nine age figures, then
17 category blocks of the same shape — so `read()` **walks the expected sequence token by
token and stops on the first deviation** rather than pattern-matching. That is the only way
to notice a re-typeset page. Four specific hazards:

* **`-` is the nil marker, not a missing value.** Small religions in small municipalities
  print a dash — Moravian in Point Fortin, Orisha in several boroughs. A parser that skipped
  non-numeric tokens would shift every later figure in the row left by one **and still
  produce nine plausible numbers**. Dashes are read as 0 explicitly.
* **Page 168 is typeset differently from every other page.** It packs three figures onto one
  line (`'113,771 123,008 105,161'`) and puts the national row's first figure on its *label*
  line (`'TRINIDAD AND TOBAGO 1,322,546'`). This is why the parse works on whitespace tokens
  rather than on lines.
* **One label changes case between panels.** The island is `Tobago` in BOTH SEXES and
  `TOBAGO` in MALE and FEMALE; nothing else differs. Matching is case-folded, which is safe
  only because the walk is expectation-driven — `TRINIDAD` and `TRINIDAD AND TOBAGO` are told
  apart by what is expected next, not by matching.
* **`Jehovah’s Witness` uses a curly apostrophe.** `taxonomy/tt2011.py` normalises it so a
  future vintage with a straight one still resolves.

Only the **even** pages of each panel are read: they carry column (1), `All Ages`. The odd
pages continue the five-year age bands, which this map does not use.

## 4. No identity is exact, and that is the source rather than the parse

Every reconciliation in this table is off by one or two people. **The figures are weighted
estimates**: CSO's own per-municipality `Individuals` workbooks publish *fractional* people
(`10.308105`, `195.796888`), so each integer printed in Table 8 has been rounded
independently and sums of rounded cells do not tie.

Measured across all 359 identity checks `sources/tt.py` runs:

```
   -2:  4     -1: 55     0: 249     +1: 47     +2:  4
```

**Symmetric, bounded at two people, 69% exact.** A parse error looks nothing like that — it
is one-sided and large. So the bound is asserted at 2 and **the distribution is printed on
every run**, because the distribution is what would actually reveal a re-typeset page.

The one figure that *is* exact is the national universe, 1,322,546, because it is printed
rather than summed.

### The identities themselves are unusually strong

* the 17 categories against each unit's own total, on all 17 units;
* **two nested universes**: the 14 Trinidad municipalities sum to the printed `TRINIDAD` row,
  and `TRINIDAD` + `Tobago` sums to `TRINIDAD AND TOBAGO`, on all 18 columns;
* **Male + Female == Both Sexes on all 306 cells.** The table repeats in full three times —
  BOTH SEXES pp. 168–185, MALE 186–203, FEMALE 204–221 — and only BOTH SEXES is drawn. The
  sex panels are the only check that would catch a token landing in the wrong column, since
  every other identity reconciles inside one panel whichever way its tokens were taken.

## 5. `None` is a category name, and pandas deletes it

**Fifth sighting of §12's Philippine trap**, after `ph`, `gy`, `zw` and `bz`. Milder here —
`None` is 28,842 people, 2.18% — but `_tt_counts` passes `keep_default_na=False` and asserts
the category survived, for the same reason as everywhere else.

## 6. The universe is non-institutional

Table 8 excludes the institutional population — prisons, hospitals, homes, barracks. The
2011 census counted **1,328,019** people in all; this table's universe is **1,322,546**, so
**5,473 people (0.41%)** are outside it. Nothing is scaled up to close the gap (§14.4), and
`gap=` states it in the legend, because a blank on a dot map cannot distinguish "nobody here"
from "nobody counted here" (§6.12).

## 7. Boundaries and placement

See `sources/tt_geo.md`. Short version: COD-AB ADM1 is the census's municipality tier
exactly, the join is **15/15 both ways** with ten name variants all handled by `fold()`, and
placement is Kontur's 400 m grid (4,610 hexes) because the units run from **13.1 km² to 931.0
km²** and the small ones are the dense ones.

## 8. What the map shows

**Trinidad and Tobago is the most plural country this map draws in the Americas.** No single
answer reaches 22%.

| | national |
|---|---|
| Roman Catholic | 21.60% |
| Hinduism | **18.15%** |
| Pentecostal / Evangelical / Full Gospel | 12.02% |
| *Not Stated* | *11.10%* |
| Other | 7.27% |
| Anglican | 5.67% |
| **Baptist-Spiritual Shouter** | **5.67%** |
| Islam | 4.97% |
| Seventh Day Adventist | 4.09% |
| Presbyterian / Congregational | 2.49% |
| None | 2.18% |
| Jehovah's Witness | 1.47% |
| Baptist-Other | 1.21% |
| **Orisha** | **0.90%** |
| Methodist | 0.65% |
| Moravian | 0.27% |
| Rastafarian | 0.27% |

### The Indo-Caribbean geography

Hinduism is **43.0% of Penal/Debe**, 31.3% of Couva/Tabaquite/Talparo, 30.0% of Chaguanas and
27.0% of Princes Town — the central and southern plain, which is where indenture put it —
against **0.7% of Tobago**. Islam follows the same belt (Chaguanas 8.6%, Princes Town 8.5%).
With Guyana (§9r) this is the Indo-Caribbean geography the map was missing; Suriname would
complete it (`sources.md` §11t).

### The Presbyterians are Indo-Trinidadian, and the map shows it unprompted

`Presbyterian/ Congregational` peaks in **San Fernando (5.5%) and Penal/Debe (5.3%)** — the
same units that are most Hindu — and is **0.2% in Tobago**. That is not the Scottish-settler
pattern the name suggests: it is the **Canadian Presbyterian Mission to the Indians**, which
from 1868 built schools among the indentured population and drew its converts from it.

### Tobago is a different country religiously

| | Tobago | Diego Martin |
|---|---|---|
| Roman Catholic | **6.6%** | 44.8% |
| Anglican | 12.8% | 8.5% |
| Seventh Day Adventist | **16.3%** | 3.6% |
| Pentecostal | 14.7% | 9.7% |
| Hinduism | 0.7% | 1.8% |

Trinidad is Catholic and Hindu; Tobago is Protestant, with the country's highest Anglican,
Adventist and Pentecostal shares. The 15-unit tier is just fine enough to show it.

### The two new nodes

**`afrodiasporic.spiritualbaptist`** — 75,002 people, more than Trinidad's Anglicans and
nearly five times its ordinary Baptists, which CSO counts as a *separate* answer. Baptist
Protestantism and West African practice fused rather than one absorbing the other. Banned
outright by the Shouters Prohibition Ordinance from 1917 to 1951; **30 March is a public
holiday, Spiritual Baptist / Shouter Liberation Day.** Geography 13.0% Point Fortin, 10.6%
Tobago, 9.6% San Juan/Laventille, against 2.6% Penal/Debe.

**`afrodiasporic.orisha`** — 11,918 people. Yoruba orisha worship, historically called
Shango, the direct sibling of Candomblé and Santería. Legally recognised since the Orisa
Marriage Act 1999. Geography 2.1% Mayaro/Rio Claro, 1.8% Point Fortin, 1.6% Port of Spain.

**Read the Orisha figure as a floor.** In Trinidad, Orisha and Spiritual Baptist practice
overlap heavily — many people take part in both, and much of the literature treats them as
one religious complex — while a census offers one box. So this counts people who chose Orisha
*over* the alternatives. Nothing here corrects it (§14.4); `note_public` says so.

The placement of both under `afrodiasporic` rather than under `christianity` follows Jamaica's
`Revivalist` (§9ab) and spec §3.3, and the objection to it is recorded rather than dismissed —
see `taxonomy/tt2011.py` and `taxonomy/branches.py`.

## 9. `Not Stated` at 11.10% changes how every share should be read

146,798 people, **the largest non-answer on this map outside the United States**. Every
percentage above is a share of the whole non-institutional population, so a religion's share
*among people who answered* is about a tenth higher than what is drawn. Nothing
redistributes it (§3.5).

**Its geography is uneven and unexplained** — 19.3% in Tunapuna/Piarco and 15.0% in Point
Fortin against 6.9% in Sangre Grande, a near-threefold spread — and CSO offers no comment.
It does not look like a refusal pattern (which would track the most religiously mixed places)
or an enumeration failure (which would track the remotest); Tunapuna/Piarco is the largest
and most suburban corporation in the country. Named as an open question rather than guessed
at.

## 10. Ethics (§14)

Nothing here is sensitive in §14.2's sense: the question is voluntary and ordinary, the state
persecutes no group in it, and 88,000 people per unit locates no community.

The one judgement that is not purely technical is **filing Spiritual Baptists outside
Christianity**. They describe themselves as Christians and as Baptists, and many would
reject the placement. The project made the identical call for Jamaica's Revivalists, and the
reasoning — that the tree is a genealogy of traditions rather than a register of
self-description — is written out in `taxonomy/branches.py` rather than left implicit, so it
can be argued with. **It is worth Anita's eye**, per §14's closing line: it is the kind of
call that is defensible and still someone else's to confirm.

## 11. Not done

* **The 2011 Community Register's finer geography**, because it carries no religion (§2).
* **The age and sex detail.** Table 8 has both and this map uses neither; the age columns
  would support a cohort read of secularisation like Chile's (§9k) if anyone wants it.
* **Splitting `Other`** (7.27%), because CSO publishes no composition. §14.4.
* **A 2022 vintage.** Trinidad's next census was repeatedly postponed and no religion
  tabulation from it was found; 2011 is current.
