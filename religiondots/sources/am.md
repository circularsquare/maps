# Armenia — Armstat, 2022 Population Census

Ingested 2026-09-08. `sources/am.py`, `sources/am_geo.py`, `taxonomy/am2022.py`.
Drawn at **marz**: 11 units, 15 nodes, 2,883,372 of 2,932,731 enumerated.

Summary: a country that §11o closed with one sentence and that publishes its census religion
tables in full, in Armenian, one volume per marz. The interesting sections are §1 (why the
sweep saw nothing, which is a page-shape problem and a language problem rather than a data
problem), §3 (why the marz tables have MORE categories than the national table they sum to,
and why a per-category equality check would have rejected a correct read) and §5 (what the
census's own religion-by-ethnicity table says about Armenia's Yazidis, which is the most
interesting thing in the file and is not drawn).

---

## 1. Why the sweep saw nothing, twice over

sources.md §11o recorded, in one row of a table: *"`armstat.am` census pages carry no
religion table."* That is wrong, and it is wrong for two independent reasons, either of which
would have been enough on its own.

**The results page is an image map whose links all go to the same place.**
`armstat.am/en/?nid=944`, *The Results of 2022 Population Census of RA*, renders a GIF of
Armenia with eleven `<area>` polygons over it. Every one of those eleven, plus the caption
underneath, points at `?nid=82&id=2623`, the national volume. Clicking Syunik and clicking
Shirak return the same file. Nothing on that page distinguishes the marzes, so the page reads
as "the national volume, presented decoratively" when in fact eleven separate marz volumes
exist. They are one `nid` each, `945`-`953`, `956` and `957`, reachable only through the left
navigation tree.

**And the English pages are the empty ones.** All eleven marz nids resolve under `/en/`, and
all eleven are a bare `<h1>` over the sentence *"Information is not available in English"*.
The same eleven nids under `/am/` each carry nine section archives. The English tree here is
a strict subset of the Armenian one, and the subset is missing exactly the geography.

> **The rule.** A statistics office's language versions are not translations of one site;
> they are separate trees that can differ in what exists at all. Check the national-language
> tree before recording a negative, and record which one you checked.

The national volume itself is not hidden and never was: `?nid=82&id=2623` is *The Main
Results of RA Census 2022*, nine chapters, each a `.7z` of `.xlsx`, in Armenian (`sector_N`),
English (`section_N`) and Russian (`Раздел_N`). Chapter 5 is *Ethnic structure, fluency in
languages and religious belief*. It has no geography in it, which is probably what the sweep
found and generalised from.

## 2. Which census, and why the newer one is not the finer one

The queue row priced Armenia at "11 categories" with the geography unknown, and the UNSD
oracle lists **2022, 11 categories, not a partition**. Both 2011 and 2022 asked religion and
both publish it the same way: national volume plus one volume per marz, religion in section 5
of each, and **section 5 stops at the marz in both**. Section 1 of the same volumes goes down
to the individual settlement, for population alone.

So Moldova's lesson does not repeat here in its own shape. The newer census is taken, and for
the ordinary reasons: it is eleven years fresher, its category list is longer (sixteen named
religions against the 2011 volume's shorter list), and it reconciles exactly. But the thing
worth carrying is that **checking cost nothing and the answer could have gone the other way**;
the 2011 volumes are still there at `?nid=533`-`543`, also Armenian-only, if anyone ever
wants the change over time.

**Nothing finer than the marz exists in the publication.** The microdata route is a formal
application (`?nid=864`, *Procedure for providing microdata*, Armenian PDF) and was not
pursued.

## 3. The marz tables are finer than the national table, and that breaks the obvious check

The national table 5.5 names thirteen religions. The eleven marz tables between them name
**fifteen**, adding `Բողոքական` (Protestant, in Ararat, Lori, Kotayk and Tavush) and
`Տրանսցենդետալ մեդիտացիա` (Transcendental Meditation, in Ararat alone) that national 5.5
folds into `Other religious groups`. Nationally those two are visible only in table 5.7,
whose universe is age 6 and over.

The reason is that **a marz table prints only the columns that marz has people in**. Syunik's
header runs to five religions and Yerevan's to fourteen. The consequence matters:

> **An answer with no column in a given marz is inside that marz's `Other religious groups`.**

Which makes a per-category equality check against the national table wrong. Islam is the
legible case, and it was worked through by hand rather than assumed:

| marz | Muslims printed |
|---|---:|
| Yerevan | 320 |
| Shirak | 120 |
| Armavir | 26 |
| Ararat | 17 |
| **sum** | **483** |
| national table | **515** |

The missing 32 are in the seven marzes with no Muslim column, inside their residuals. So a
small category's marz sum is a **floor**. Across all fifteen categories the shortfall is 169
people and it reappears exactly in `Other`.

**What does hold exactly, and it is the arithmetic that matters:**

- every marz's own columns sum to that marz's own published population, to the person;
- the eleven marz totals sum to **2,932,731**, the published national total, to the person;
- the twelve ethnicity rows of national table 5.5 also sum to 2,932,731, which is what proves
  the block boundary in §5 below.

**And the same comparison finds eight people that folding cannot explain.** Three categories
come out of the marz tables *larger* than the national table: `Refused to answer` by 6,
`Evangelical` by 1, `Jehovah's witness` by 1. Folding can only shrink a category, so these are
edits between two Armstat publications of one census. Total disagreement between the two
publications is 177 people, 0.0060% of the country, and `EDIT_CAP` in `sources/am.py` bounds
it so a real divergence would still fail.

### Four smaller traps in the same files

1. **The header is two rows and the top one is not decoration.** The religions sit under a
   spanning *Religious belief* title in the lower row, but `No religion` and `Refused to
   answer` are outside that span and printed one row higher, at the far right. Reading only
   the lower row loses 66,854 people per country and looks like a clean sub-total rather than
   a hole, because what is left still adds up to the printed `has a religious belief`.
2. **The labels carry line-break hyphens.** Armstat typeset these for a printed page, so the
   same answer is `Ավետարանական` in one marz and `Ավետարանա-կան` in the next, and likewise
   `Շարֆադինա-կան` and `Հեթանոսա-կան`. Fold hyphens and whitespace out before any lookup.
3. **Shirak's workbook has a second sheet of scratch working** (`Лист1`, 99 rows of
   intermediate arithmetic) beside the published `Sheet1`. Read sheet 0 only. Four of the
   archives also ship Excel's `~$` lock files.
4. **`table 5.6Е..xlsx` in the English national archive spells its `Е` in Cyrillic.** A glob
   on `5.6E` misses it.

## 4. The geography

geoBoundaries **ARM ADM1**, 11 polygons, joined to the eleven census marzes by name, both
ways, nothing spare on either side. The romanisations differ by the genitive ending Armenian
uses to form a province name (`Aragatsotni`/`Aragatsotn`, `Lorri`/`Lori`), which is an alias
table of a dozen entries rather than a fuzzy match.

**The join is proved on population, not on names.** Kontur's modelled 2023 population against
Armstat's 2022 census count, per marz: median 0.95x, and every one of the eleven inside
0.7-1.4x, which two swapped same-sized marzes would not be.

The two extremes are **Yerevan 0.79x** and **Kotayk 1.34x**, and that is §9q's city/ring pair
again, in another post-Soviet capital: Yerevan's ADM1 polygon is the city proper and Kotayk wraps its northern
edge, so Abovyan, Nor Hachn and Charentsavan are Kotayk's on paper and Yerevan's commuter belt
in practice. Both stay inside the band, so both are asserted rather than excused. It is a
placement fact and not a count fact; every marz's dot count still comes from the census.

Placement is the Kontur H3 r8 grid rather than §8.2's equal share, because 2,700 km² per unit
of Armenian plateau would otherwise put dots on rock.

## 5. What the census says about Armenia's Yazidis, which is not drawn

National table 5.5 crosses religion with the twelve census ethnicities. Those rows are carried
in `data/normalized/am.csv` at `geo_level=country_by_ethnicity`, are not a geography and are
not drawn; they are there because they are the only place the source says *who* is in a
category, and because they contain the most interesting fact in the file.

Of the **31,079** people who gave `Yezidi` as their ethnicity:

| answer | people |
|---|---:|
| Shar-fadinian | 13,256 |
| **Armenian apostolic** | **9,939** |
| Other religious groups | 3,246 |
| Pagan | 1,672 |
| Refused to answer | 2,053 |
| No religion | 424 |
| everything else | 489 |

Two things follow that the drawn map cannot show on its own.

**`Շարֆադինական` is the community's own name for its religion**, from *Şerfedîn*, and the
box is unusual: the three other censuses on this map that count Yazidis (Australia, Georgia,
the United Kingdom) all label it with the ethnonym. Armenia labels it with the faith.

**And `paganism` in Armenia is mostly a Yazidi answer.** The national Pagan cell is 2,132
people, of whom 1,672 are Yezidi and 215 Kurd against 237 ethnic Armenians. That is why the
cell peaks in Armavir (0.36%) and Ararat (0.19%) and not in Yerevan (0.03%), where an Armenian
neopagan revival would be. Moving those 1,672 to `yazidism` was considered and refused: the
census offered both boxes at the same question and these people chose this one, and
reassigning an answer on the ethnicity of the person who gave it is what spec §14.5 forbids.
It is recorded in `taxonomy/am2022.py`'s REVIEW instead, and in the public note, because a
reader looking at the Ararat plain should know the two colours there are largely one
community.

## 6. The one thing sent to Anita — ruled 2026-09-08, and Armenia moved

`christianity.oriental` had **2,793,041 Armenians on it, 95.2% of one country**, against the
109,041 Georgians that were previously its largest count, and Armenia shipped at that parent
because `au2021`, `ee2021`, `cy2021`, `pl2021` and `ro2021` all sat there too. Filed as
`ask/004-am`.

**Anita ruled: record whatever the source actually says.** Armstat's cell names the Armenian
church, so Armenia now files at `christianity.oriental.armenian`, a new node, and so do the
eight other sources that name it in their own languages. The premise the ask rested on was
wrong in two ways and the review caught both: the five European censuses do not file *Armenian*
bodies at the parent by choice, they name them, so they moved too; and the branch already had
ten children, two of them ASARB's American Armenian dioceses, so this added one legend row
rather than the first. The two ASARB rows are now the worldwide catholicosates
(`.armenian.etchmiadzin`, `.armenian.cilicia`) instead of nodes called *of North America* that
`origin_religion.py` was sending French and Spanish Armenians to. spec §2.7 carries the general
rule; four Oriental Orthodox cells stay at the parent on purpose because their sources cannot
distinguish.

## 7. Smaller calls, all recorded in `taxonomy/am2022.py`

- **Catholic → the parent, not `.eastern`**, although the geography says plainly which church
  it is: Lori 3.15% and Shirak 2.89% hold 77% of Armenia's Catholics, which is the Armenian
  Catholic country around Gyumri, Artik and Tashir. The census prints one undivided
  `Կաթոլիկ` and never names a rite, and Georgia's identical cell sits at the parent.
- **Molokan → `christianity.other`.** 1,982 people, 1,578 of them in Lori, the Spiritual
  Christian villages of Fioletovo and Lermontovo. A node of their own would be a legend row
  no other country uses, for two dots. They are emphatically **not** Orthodox.
- **Nestorian → `christianity.churchofeast`.** 479 people, 346 in Ararat: Verin Dvin and the
  Assyrian villages of the plain.
- **Evangelical → `christianity.evangelical`, not `.protestant`**, because the census prints
  a separate `Protestant` column in four marzes and therefore treats them as different
  answers. The body behind most of it is the Armenian Evangelical Church (Constantinople,
  1846), which is Congregational-Reformed, but the cell is a one-word answer.
- **TM → `other.am`.** Nine people, in one marz; the tree has no node for it and nine people
  are not a reason to make one.

## 8. Review, 2026-09-08, session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-am-rev`

`check_md.py`, `built_countries.py --check`, `check_rollup.py am` (2,883,372, all measured, no
derived, no orphans) and the import assertion all clean. Screenshot clean: dots follow the
Ararat plain, the Yerevan cluster, Gyumri, the Vanadzor corridor and Syunik down to Kapan,
Lake Sevan is empty, nothing crosses into Turkey, Georgia or Azerbaijan.

**Every figure in `note_public` recomputes from `data/normalized/am.csv`, and the min/max pair
is the real one.** Checked independently rather than against §3: Lori 92.079% is the lowest
Apostolic share of the eleven marzes and Syunik 98.825% the highest, with nothing outside them.
So are Lori as the Catholic maximum (3.151%), Tavush as the refusal minimum (0.261%) and
Shirak as the no-religion minimum (0.132%). `taxonomy/am2022.py`'s per-category claims check
out too, including the three single-marz ones: Judaism 96 and Krishna 200 are entirely in
Yerevan (Armavir prints a column and it is 0) and TM 9 is Ararat's alone. So does *"the three
other censuses on this map that count Yazidis"*: `au2021.py`, `ge2014.py` and `uk2021.py` are
the only other files mapping to `yazidism`, and all three label it with the ethnonym.

**One thing to change, recorded not applied. Three `note_public` figures are marz sums written
as national totals, and Armstat's national table 5.5 prints different numbers.**

| note_public says | table 5.5 | |
|---|---:|---|
| *"Armenia's **17,855** Catholics"* | 17,884 | 29 folded, §3 |
| *"Armenia's **1,982** Molokans"* | 2,000 | 18 folded, §3 |
| *"the country's **49,359** refusals"* | 49,353 | 6 the other way, the publication edit in §3 |

Both mechanisms are written up properly in §3, in `countries.py`'s internal `note` and in
`am2022.py`'s docstring, and none of them reaches the reader: `tiles.py:259` and `:365` export
`note_public` and nothing else, so `note` is internal. The reader meets only the figures, and
their precision to the person is what invites the check against a table that says something
else. Not applied by the review because the wording is the builder's or Anita's and
`countries.py` had a live builder in it; the smallest honest fix is to stop the sentence
asserting a national total, not to swap the number, since 17,855 is what is drawn. The 77% and
every share are unaffected.

**Applied 2026-09-08**, as three rewordings and no number changed. The marz sums, the country
table and the two sub-national figures were all re-derived from `data/normalized/am.csv` first
and reproduce exactly: marz Catholic 17,855 against country 17,884, marz Molokai 1,982 against
2,000, marz refusals 49,359 against 49,353; Lori 7,019 plus Shirak 6,813 is 77.47% of 17,855,
Lori holds 1,578 of the Molokans and Yerevan 38,931 of the refusals. `note_public` now says
*"77% of the 17,855 Catholics the marz tables print"*, *"1,578 of the 1,982 Molokans counted in
the marz tables"* and *"38,931 of the 49,359 refusals recorded across the eleven marzes"*. The
three phrasings differ on purpose so the hedge does not read as a tic, and none of them explains
the 29-, 18- and 6-person gaps, which is deliberate: the mechanism is §3's and is not something
a reader of the panel could check.

**`gap` is honest and complete.** Drawn 2,883,372, undrawn 49,359, and the undrawn are exactly
the refusals with nothing else missing; `gap_share=0.0168` is right on either publication's
figure (0.016831 / 0.016829). `basis="self-identification"` matches a direct census question.

**§3.5's lean was asked and does not clear, which is worth recording so nobody re-runs it.**
Correlating each marz's refusal share against each drawn category's share of that marz's
responders, n=11, 20k-permutation p: the largest real one is `No religion` at rho +0.518,
p=0.108, and it survives dropping Yerevan (+0.515). `Armenian apostolic` is rho -0.264,
p=0.436. `Molokai` reports rho 0.833 at p=0.0033 and is an artefact of ties, since it has a
column in five marzes and the other six are exact zeros; its Pearson is 0.058. So the sign is
Serbia's direction and the test has no power to say so: eleven units, one of which is 78.9% of
all the refusals, and the quantity a lean would move is 0.6% of the country
(`[[reference_check_needs_power]]`). Nothing to add to `note_public`, which already names the
direction of the hole geographically and says those people are not drawn.

**Mapping against precedent: nothing filed against convention.** The Oriental Orthodox question
is `ask/004` and is Anita's; the review appended a table to it rather than filing a second ask,
because the branch already has ten drawn children and the ask was written as though it had
none. `other.am` is the standard per-country residual, one of about ninety and no legend row of
its own.
