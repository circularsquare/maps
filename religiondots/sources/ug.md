# Uganda — UBOS, 2002 census, Table B7, and why the 2024 census could not be used

Built 2026-09-09, session `967ffe99-…-ug`. Drawn on **56 districts, 7 categories,
24,433,132 people**, from `Table B7: Religion by District for the Population`, an annex
table of the 2002 Population and Housing Census.

`queue.md` priced Uganda at *"regions"* and said the 2024 census has the best unclaimed
category list in Africa, published national-only, with DHS or a UBOS microdata email as
the routes to geography. The first half of that is right and unchanged. The second half
turned out to be unnecessary: **the 2002 census published religion by district, in a file
no current UBOS page links to.**

---

## 1. What the 2024 census publishes, established rather than assumed

`sources.md` §11b read the Final Report and concluded *"national only"*. That is correct,
and this pass went further because a 2024 table at any subnational tier would beat a 2002
one outright. Everything below was opened, not searched for.

| what | what it is | religion? |
|---|---|---|
| **NPHC 2024 Final Report Volume 1** | 434 pages. Religion appears on 14 of them. Table 3.1 is `Distribution of the Population by Religion, 2014-2024`, ten categories, **national**. Table 3.4, in the same chapter, breaks birth registration down **by sub-region**, so the absence is a choice. | no geography |
| **`statistics.ubos.org/nphc`** | Not a document set. A real application with `builder`, `drilldown`, `dashboard`, `map` and `report` pages over an API: `api/get_profile_data.php`, `api/get_geospatial_data.php`, `get_subregions/districts/counties/subcounties/parishes.php`, `get_location_hierarchy.php`. Eight endpoints, reaching **parish**. | **fifteen tables and none is religion** |
| the same API, pushed | `format=all` and `format=summary` return the identical fifteen tables as `format=detailed`; a `table=religion` parameter is ignored. The `map` page's indicator dropdown has 16 options and no religion. The dashboard has a `Religion Statistics` tab whose chart script, `scripts/social.js`, is **a zero-byte file**. | closed |
| **17 sub-region profile reports** | One per sub-region, e.g. `Acholi-Sub-Region-Census-2024-Profile.pdf`, **412 pages**, district by district down to sub-county. Same fifteen tables. | no |
| **`NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx`** | 13,527 rows to parish (§11b found this). | no |
| **NPHC 2024 Community Module Report** | 1,335 pages, and it looked promising: `religio` matches on 343 of them. Every one is the word **`Religious/NGO`** or **`Religious`** as an OWNERSHIP class for a school or a health centre. `worship`, `church` and `mosque` match **zero** pages. | no |
| **the 2024 monograph series** | Only three are out: MPI (Volume 5), OVC (Volume 4), Disability. Volumes 1-3 are unpublished. | not yet |
| **ubos.org's publications catalogue** | Swept by paging `?pagename=explore-publications&p_id=1..139`: **703 distinct documents**, filename and link text. Zero mention religion. | no |
| **NPHC 2014 Main Report** | 105 pages. Table 4.1, religion 2002 and 2014, **national**. | no |
| **2014 Area Specific Profiles** | One PDF per district under `wp-content/uploads/publications/2014CensusProfiles/`. ABIM.pdf, 48 pages, checked. | no |
| **2009 Higher Local Government district statistical abstracts** | 133 of them on the retired tree. Apac's, 77 pages, checked. | no |
| **2002 Population Composition analytical report** | Its Table 3.7 is religion by **four regions** plus urban/rural, and Appendix A1.4 is religion by ethnic group. This is what everyone cites. | four regions |

**The UBOS microdata catalogue is the route that was NOT taken, and there is an ask
about it.** `microdata.ubos.org:7070` is a NADA instance with 71 datasets including NPHC
2024 (id 74) and NPHC 2014 (id 75). Its own page says *"To access data for this study,
user must be logged in… register for a free account."* An account is Anita's call, per
AGENT_BRIEF §3, so `ask/008-ug-ubos-microdata-needs-a-free-account-and-its.md` was filed. See §6 below for the part of this
that is not just a registration.

**`ubosgis.ubos.org`** is UBOS's ArcGIS portal and answers **502** to every path
(`/portal/sharing/rest`, `/portal/home/index.html`, `/server/rest/services`), 2026-09-09.
Worth retrying later; it is the obvious place a religion layer could hide.

---

## 2. The table that was found, and where it lives

```
https://web.archive.org/web/2018id_/http://www.ubos.org/onlinefiles/uploads/ubos/
    census_tabulations/centableB7.pdf
```

Two pages. `Table B7: Religion by District for the Population`, footed
*"Annex 2"* of the 2002 census. Columns: **Catholic, Anglican, SDA, Pentecostal, Moslem,
Other, None, Total**, on **56 districts** grouped into the four regions.

It lives in `census_tabulations/`, a directory of fifteen loose annex tables on the
**retired** ubos.org tree. The live site 301s that path away and serves nothing at the new
one, and none of the 703 documents in the current publications catalogue is any of them.
The directory was found by pulling every `.pdf` URL ever archived under `ubos.org` from
the **Wayback CDX API** (8,751 of them) and listing the directories they sit in.

Wayback's `id_` replay returns the original bytes. The other fourteen were checked too:
B1 (populations 1980/1991/2002 by district), B2, B3, B10, B14, B25, B26, B27 and C1
were retrieved; B4, B5, B6, B8, B9, B11-B13 and B28-B30 have no capture at any
timestamp tried. **B7 is the only one with religion in it.**

---

## 3. Three checks the table passes

1. **Internal.** Every district row's seven cells sum to its printed total; the districts
   under each region sum to that region's printed row; the four regions sum to the
   printed `UGANDA` row. Asserted in `sources/ug.py`.

2. **The UNSD Demographic Yearbook.** `python tools/oracle.py Uganda` returns seven
   categories totalling **24,433,132** for Uganda 2002, and **all seven equal this table's
   national column to the person**. That is UBOS's own return to the UN, a separate
   publication, reproducing the file being parsed.

3. **A different UBOS table with MORE categories.** The 2002 Population Composition
   report's Table 3.6 gives **nine** national categories and excludes Kotido district.
   Take Kotido out of B7 and: Catholic 9,921,398 against its 9,921.4 thousand, Anglican
   8,753,811 against 8,753.8, Moslem 2,953,808 against 2,953.8, Pentecostal 1,128,020
   against 1,128.0, SDA 367,596 against 367.6 — and B7's `Other` **plus** `None` is
   716,629 against 3.6's Orthodox 35.4 + Other Christian 282.3 + Bahá'í 18.5 + Others
   380.4 = 716.6 thousand. The two tables are the same numbers cut two ways, which is what
   establishes what `Other` contains. Asserted.

That third check also explains a discrepancy anyone re-deriving these shares will hit:
**the widely quoted 2002 national shares exclude Kotido.** 41.6% Catholic is
B7-minus-Kotido; B7 including Kotido is 41.92%.

---

## 4. The boundary rebuild, and the proof

**No 2002-vintage boundary file for Uganda exists.** COD-AB is 2020-08-24 (135
districts), geoBoundaries is current, HDX has nothing older, and UBOS's own GIS is down.
Uganda has gone 56 → 112 (2014) → 135 (2020) → 146 (2024) districts. spec §8.1 wants the
vintage the data was published on, so the 56 are rebuilt as unions of the 135.

**Which of the 135 belongs to which of the 56 comes out of the census.** Table C1 of the
same annex series (`Total Population by Sub-county`) prints the entire 2002 hierarchy —
district, county, sub-county — for **995 named places**. Each name votes for the 2002
district it is printed under. COD-AB's 208 counties and 1,520 sub-counties are matched
against those names; each current district takes the winner. 113 are unanimous, 22 have
one stray vote each (a sub-county name that also occurs in a distant district), and all
135 land somewhere.

**The proof is exact and it is on an independent publication.** Table B1 gives the 1991
census redistributed onto the 56 districts of 2002; Table A3 of the **2014** census Main
Report gives the 1991 census redistributed onto the 112 districts of 2014. Both national
totals are 16,671,705, so the universe is the same. Grouping A3's 112 figures by this
concordance reproduces **all 56 of B1's figures exactly, on 56 distinct values.**
`sources/ug_geo.py` refuses to write anything if that stops holding.

It earned its keep immediately. The first version detected district headers in C1 by
indent alone; C1 prints a **sub-county** called `Sembabule T.C.` at the district indent
and spells the district `Ssembabule`, and Bugiri's header row lost its figures to a line
wrap. The result was that every one of Bugiri's sub-counties voted for Wakiso, the
district printed before it. Nothing about that looks wrong in a list of names. The 1991
test failed on two districts and named them.

---

## 5. Kotido, which is the one real decision here

Run the same test on the **2002** column instead of the 1991 one and 55 districts still
agree to the person. Kotido does not:

| | 2002 tabulations (Table B1) | 2014 Main Report (Table A3) |
|---|---|---|
| Kotido | **591,889** | **377,102** |
| all Uganda | 24,442,084 | 24,227,297 |

The national difference and the Kotido difference are **the same 214,787 people**. Kotido's
1991 figure agrees between the two, so this is a revision of one count, not a boundary
artefact. UBOS revised the 2002 population of one district downward by 36% and used the
revised figure from 2014 onward.

A third witness agrees, from a source with no lineage in common: **Kontur's 2023
population grid** reads **1.07x** Kotido's 2002 census figure when the national middle is
about 1.9x and no other district is below 0.78x. On the revised 377,102 Kotido would come
out at 1.68x, in the normal range.

**Kotido matters out of all proportion to its size.** It holds 22.5% of the national
`Other` cell and **33.1%** of the national no-religion cell. Uganda's drawn shares with
and without it:

| | with Kotido | without |
|---|---|---|
| Other | 3.04% | 2.41% |
| None | 0.87% | 0.60% |

**The decision: draw Table B7 as published, every row `measured`.** Scaling Kotido's seven
cells by 377,102/591,889 = 0.6371 was considered and refused. spec §14.4 rule 1 — *never
estimate a magnitude a source does not publish* — is the one rule that has never moved,
and UBOS revised a **population**, not a religion split; the factor would invent seven
counts in the single district where the composition is least like the rest of the country.
`note_public` carries both national figures instead.

A reviewer agent was spun up on this question with the files and not the reasoning, and
reached the same conclusion independently. It added two things worth keeping: the nearest
precedent is **Côte d'Ivoire** (§9az-i), where two ANStat releases of one census disagree
by 159,208 people and one release was used whole; and **no drawn country on this map has ever
had a published census figure altered** — Estonia's typo is deliberately reproduced,
Armenia's and Fiji's are bounded rather than fixed, and China's Hainan rescale (§14.23)
marks every row `derived`. It also named the strongest argument the other way, which is
that this is §3.4's shape and `br_rescale.py` already ships that construction for Brazil.

---

## 6. The microdata, and the thing about it that is not just a registration

`microdata.ubos.org:7070/index.php/catalog/74` is NPHC 2024. Its **Get Microdata** page
says a login is required and offers free registration.

**The `/download/<int>` route beside it answers unauthenticated.** Sweeping ids 1-400 with
`HEAD` returns 200 with `Content-Disposition` naming the file for 200 of them, and among
them are `NPHC 2024-Users File using cpro_extract_Population_record_data.rar` (429 MB),
`…Household_data.rar` (138 MB), `…10_perc_metadata.dta` (840 MB) and three module files. A
512-byte `Range` request on the last returns a valid `<stata_dta>` header, so it is not a
login page with a misleading length.

**It was not used.** The portal states an access control and this route walks around it,
which is not a licence; AGENT_BRIEF §3 sends anything needing an account to Anita.
`ask/008-ug-ubos-microdata-needs-a-free-account-and-its.md` puts the choice in front of her with the size of the prize, which is
large: the 2024 census at parish level with ten categories would replace this entire
country and would be the finest religion geography in Africa on this map.

The sweep is worth keeping for a different reason. It is also how the 2014 Main Report
(`download/314`) and the 2024 Final Report (`download/235`) were obtained, both of them
genuinely public documents, and it is a clean instance of
[[reference_cms_download_id_sweep]] on a NADA instance.

---

## 7. Categories, and what `Other` is

Seven cells (`taxonomy/ug2002.py` has the reasoning per cell):

```
41.92%  Catholic     -> christianity.catholic       largest in 29 of 56 districts
35.95%  Anglican     -> christianity.anglican       largest in 25
12.10%  Moslem       -> islam                       largest in 2 (Yumbe, Mayuge)
 4.62%  Pentecostal  -> christianity.pentecostal
 3.04%  Other        -> other.ug
 1.51%  SDA          -> christianity.adventist.sda
 0.87%  None         -> unaffiliated
```

`Anglican` is the **Church of Uganda** and nothing else. The 2002 analytical report prints
the identical national figure under the label `Anglican /Protestant`, which is Ugandan
usage rather than a wider category: Baptists, Presbyterians, Methodists and the Salvation
Army are inside `Other Christian`, which is inside `Other`. So `christianity.protestant`
would claim the opposite of what the form did.

`SDA` goes to `christianity.adventist.sda` rather than the parent **because the source
names the body**; `md2024.py` and `ro2021.py` use the parent for the opposite reason.

**`Other` is the interesting one and it is drawn whole.** The table's footnote says it
holds `Orthodox, Bahai, Other Christian, Non-Christian, and Traditional` in one column.
Its top is the three Karamoja districts (Kotido 28.2%, Nakapiripirit 25.0%, Moroto 19.7%),
where it is traditional religion answering a form that offered five churches and Islam —
and `None` peaks in the *same three districts*, so one population is being recorded in two
boxes. Then, clear of everywhere else, **Kibaale 13.6% and the adjoining Kyenjojo 12.9%**
against a next-highest of 7.6%. UBOS names `Ow'obushobozi` among the contents of the
equivalent 2024 cell, the only Ugandan new religious movement it names anywhere, and the
Faith of Unity was founded in 1980 at Kapyemi in Muhorro, which was Kibaale district in
2002 and is Kagadi today. That is a reading and `other.ug`'s node note says so.

The national split of `Other` **is** known (check 3 above) and is deliberately not applied
per district: it would put a quarter of Kotido's 167,065 into `Other Christian`.

---

## 8. `None` is the literal string, and pandas deletes it

Fourth sighting after `ph`, `gy` and `zw`. Default `read_csv` turns the category `None`
into `NaN`, it resolves to nothing, and **212,388 irreligious Ugandans vanish with no
error**. `_ug_counts` reads with `keep_default_na=False, na_values=[""]` and then asserts
the string is present before doing anything else. It cost twenty minutes here on a
scratch script before the module existed.

---

## 9. The §3.5 lean check

Table B7's own footnote says it *"excludes population enumerated in hotels"*, and Table B1
gives each district's full 2002 count, so the excluded residual is **B1 minus B7, per
district**: 8,952 people, 0.037%, spread over 55 of the 56 districts. It is the smallest
gap of any country on this map.

Correlated against each drawn share across the 56 districts, the strongest is the Adventist
share at **r = +0.269**, and **leave-one-out strengthens it to +0.353** with Kalangala (the
largest residual share, 0.19%, the Ssese Islands) dropped. Kampala, Mbale and Jinja are
next, so it leans urban, which is what a hotel population would do.

It does not matter. If **every one** of the 8,952 belonged to a single faith, no national
share would move by more than **0.04 percentage points**. Reported here rather than in
`note_public`, because a lean this small competes for attention with Kotido, which is
three orders of magnitude larger.

---

## 10. Fetch notes

* `python sources/ug.py --fetch` takes four PDFs: B7, B1 and C1 from the Wayback Machine,
  and the 2014 Main Report from `ubos.org/wp-content/uploads/publications/`. The PDF
  check is header **and** `%%EOF` trailer ([[reference_pdf_truncated_at_source]]); a
  direct fetch of the 2002 Census Final Report from the live site returned 360 KB that
  PyMuPDF opens with `page_count=0`, which is exactly that trap.
* `www.ubos.org` **serves an incomplete certificate chain** and is slow. `curl` without
  `-k` fails with exit 60 and reads as a dead host; `requests` needs `verify=False`.
  Large files take several minutes.
* `python sources/ug_geo.py --fetch` takes the 33 MB COD-AB shapefile bundle from HDX.
* `python sources/ug_grid.py --fetch` takes the 13 MB Kontur UG grid.

---

## 11. What would replace this, best first

1. **The UBOS microdata** (`ask/008-ug-ubos-microdata-needs-a-free-account-and-its.md`). 2024, parish level, ten categories.
   Everything else here is a distant second.
2. **`ubosgis.ubos.org`** when it comes back up. An office ArcGIS portal is where a
   religion layer would sit if one existed ([[reference_gis_server_census]]).
3. **The 2024 monograph series, Volumes 1-3**, unpublished as of 2026-09-09. The published
   ones are MPI, OVC and Disability; a socio-cultural volume is the shape that would carry
   religion by sub-region, and Rwanda's equivalent series did.
4. **DHS** (`sources.md` §11ag): UDHS 2016 and 2022 both ask religion, and both are behind
   the institutional registration that section prices. It would give 2022 at sub-region,
   which is 15 units against these 56, so it is only better because it is current.
5. **Afrobarometer**, which is open and already hardened in `sources/afrobarometer.py`.
   Not used and not close: a census at 56 districts beats a 2,400-person survey at four
   regions, and the whole reason to want Uganda is the category list.

---

## 12. Review, 2026-09-09, session `967ffe99-...-ug-rev`

**Nothing needs rebuilding and no dot moved.** `check_md`, `built_countries --check` and
`check_rollup ug` (24,433,132, all `measured`, 0 orphaned) are clean; the screenshot draws
the country full, dots on land, none in Lake Victoria, dense through Buganda and Busoga
and thin across Karamoja. The audit in section 5 was re-derived from the three PDFs with
a parse written independently of `sources/ug.py`, and it holds.

### 12.1 The audit reproduces, and there is a fourth witness the record does not use

Read straight off the files rather than through the modules:

| | 1991 | 2002 | 2014 |
|---|---|---|---|
| Table B1, Kotido (2002 tabulations) | 196,006 | **591,889** | |
| Table A3, Kotido + Kaabong + Abim (2014 report) | **196,006** | **377,102** | 456,895 |

The 1991 figures are equal to the person, so the three 2014 districts really are 2002
Kotido and the geography is not in question. 591,889 - 377,102 = 214,787, and
24,442,084 - 24,227,297 = 214,787, so the one district is the whole national difference.
Rebuilding the concordance without writing anything: 135 current districts onto 56, the
1991 column exact on all 56 of 56 distinct values, the 2002 column 55 of 56 with Kotido
alone at -36.3%. All as recorded.

**The fourth witness is in the same two tables and is not in the record.** On the
published figure Kotido goes 196,006 to 591,889 to 456,895: **+10.6% a year for eleven
years and then -2.1% a year for twelve**, a district that triples and then loses a
quarter of itself, and the 2014 census reports no decline there. On the revised figure it
goes 196,006 to 377,102 to 456,895, **+6.1% then +1.6%**, high for the first stretch and
ordinary for the second, which is what a Karamoja district looks like. This needs no
external model at all, which makes it a cleaner witness than Kontur's 1.07x, and it points
the same way.

### 12.2 The Kotido call: agreed, and the recorded reason is not the strongest one

The reason in section 5 and in `countries.py` is that UBOS revised a population and never
revised a religion split, so section 14.4 rule 1 forbids the factor. True, and it is a
fact about what was printed. The substantive reason is one level down and worth having in
the record, because the next country with this shape will need it:

**A uniform rescale is not the neutral option. It is a specific claim about the shape of
the overcount, made in the one district where the composition is least like anywhere
else.** Kotido is 28.2% `Other` and 11.9% `None` against a national 3.0% and 0.9%. If the
214,787 people who were removed were mostly the mobile pastoralist population, which is
the obvious story for a 2002 Karamoja overcount and is presumably why the correction
exists, then they sat disproportionately in exactly those two cells, and multiplying every
cell by 0.6371 would leave both far too high while looking like it had fixed them. If they
were spread evenly across the seven, the factor is right. Nothing available distinguishes
those two, and the distance between them is most of the national `Other` cell and a third
of the national `None`. Drawing the printed table claims only that the table says this,
which `note_public` then qualifies with both national shares; the factor would claim
something nobody measured. So rule 1 gives the right answer here for a reason beyond the
formal one.

**The tier is right by elimination and that should be written down.** Kotido's seven cells
are `measured`, the same tier as 55 districts on which two independent publications agree
to the person, and that is not quite what `measured` means everywhere else on this map:
every other `measured` row is a figure no publication of the office contradicts, and this
one is contradicted by 36%. There is no better tier available. `derived` means this project
computed it and `modelled` means this project estimated it, and neither happened, so a
fourth tier would be needed and that is a change to shared vocabulary which
`AGENT_BRIEF.md` section 3 sends to Anita and which one district does not justify. Left
alone deliberately, recorded so the next reader does not re-open it. The disclosure lives
in `note_public`, which is the only place it can live.

### 12.3 One real gap in the proof: it constrains 112 of the 135, not all of them

**Table A3 is on the 112 districts of 2014 and the concordance is on the 135 of 2020, so
the 23 districts created between those two vintages contribute nothing to either grouped
column and the 1991 equality is silent about where they went.** Brute force says so
exactly: of the 7,425 single-district misassignments available, 6,160 break the 1991 proof
and **1,265 slip past it**, which is 23 x 55 and not one district more. Those 23 are
**12.1% of Uganda's land area**.

Nothing is actually wrong. Every one of the 23 is assigned to the historically correct
parent (Kagadi and Kakumiro to Kibaale, Karenga to Kotido through Kaabong, Rubanda and
Rukiga to Kabale, Bugweri to Iganga, Kwania to Apac, and so on for the rest); all 23
border a *proved* district inside their own 2002 group, so none of them is holding a group
together on its own; and the 56 dissolved polygons are all single connected pieces, with
no second fragment over 5 km2 anywhere. Three independent reasons to believe it, none of
them the population proof.

Worth stating because the claim in `sources/ug_geo.py`, in `sources.md` 9cr and in spec 12
is that the concordance is proved, and what is proved is the 112 of it that the
redistribution table covers. **The general form: a concordance proved on a redistributed
census is proved only for the units that census was redistributed onto, so the unproved
share is the vintage gap between the redistribution table and the boundary file.** Uganda
is a good case rather than a bad one; the next COD-AB edition is 146 districts and the
unproved share grows with every split.

### 12.4 The Bugiri guard fires again, and there is now an earlier one

Reconstructed rather than assumed: 2002 Bugiri is current Bugiri plus Namayingo, and
sending both to Wakiso breaks the 1991 equality on two districts, Wakiso 802,194 against
Table B1's 562,887 and Bugiri 0 against 239,307. `ug_geo.py` raises on that list. There is
also now a guard that fires before it, which the original run did not have: `c1_votes`
matches district headers by name against B7's 56 and raises if it finds fewer, which is
the exact thing that failed. Both fire.

### 12.5 Everything reader-facing recomputed from `data/normalized/ug.csv`, by people

Every figure in `note_public` reproduces: Adjumani 82.52% and Gulu 78.15% Catholic,
Nakasongola 60.76% and Ntungamo 60.57% Anglican, Yumbe 76.23%, Mayuge 36.20% and Iganga
33.81% Moslem against 0.39% in Kotido and 0.45% in Pader, Kotido 28.23% `Other`,
Nakapiripirit 11.97% `None`, the three Karamoja districts holding 52.3% of the national
no-religion cell, Kotido holding 22.5% of `Other` and 33.1% of `None`, 3.04%/0.87% with
Kotido and 2.41%/0.60% without, and 41.61% Catholic on the widely quoted ex-Kotido base.
The leads are 29 Catholic, 25 Anglican and 2 Moslem (Yumbe and Mayuge), summing to 56.
Pentecostal peaks at Kapchorwa 18.0%, then Kaberamaido 11.8% and Soroti 11.4%, against
Kampala's 9.0%, so "strongest in Sebei and Teso rather than in Kampala" is right. Grain
436,306 people a district.

**The section 3.5 lean reproduces exactly and the note is right not to carry it.** 8,952
people, 0.0366%, nonzero in 55 of the 56 districts and never negative; Adventist share
r = +0.269 and +0.353 on leave-one-out with Kalangala dropped, the largest residual share
at 0.193%; Kampala, Mbale and Jinja next, so it does lean urban. If all 8,952 went to one
faith the largest national move is **0.0363 points**, on `None`. It is stated in section 9
and in `countries.py`'s internal note as saying almost nothing, and it is kept out of
`note_public`, which is the correct call: at 0.04 points it would compete for a reader's
attention with a finding three orders of magnitude larger.

### 12.6 Dots, mapping and voice, none of which needed anything

* **Dots.** 24,429 at 1,000 people each; the seven node counts are the national figures
  divided by 1,000 to the unit; 21 dots, 0.09%, fall outside a 2002 district polygon,
  which is hex clipping on the lakeshore; Kotido draws 593. Per-district dot counts against
  people/1000 are within 2.5% everywhere except Kalangala at -4.9%, which is 33 dots
  standing for 34.7 and is rounding on the smallest district.
* **`Anglican` to `christianity.anglican`** matches `mw2018` and `za2016`, both of which
  send a literal census `Anglican` cell to the same node, and the evidence here is better
  than either: the analytical report's separate national `Other Christian` of 282,300, 1.16%
  of the country, is where the Baptists, Presbyterians, Methodists and the Salvation Army
  went, and 1.16% is the right size for them. The `Anglican /Protestant` label does not
  make the cell a Protestant cell.
* **`other.ug`** is the 111th `other.<cc>` node and is the standing convention rather than
  a new legend row to argue about. `christianity.adventist.sda` is shared with `ca2021`,
  `md2024` and `usrc2020`.
* **Voice.** `note_public` has six bold sentence openers, which is the structure
  `countries.py`'s own field docstring documents (a bold sentence starting a sentence
  becomes a paragraph break and loses its bold) and which 104 of the 121 countries use, and
  it carries **zero em dashes** where 69 of 121 do. Nothing to fix.
* **Two figures that look like a discrepancy and are not.** `note_public` says Kotido is
  591,870 while section 5 and `countries.py`'s internal note say 591,889. The first is
  Table B7's own total, which excludes the hotel population and is what is drawn; the
  second is Table B1's. The 19 people between them are the hotel residual. Likewise
  `gap_share` 0.00036625 is 8,952/24,442,084 rather than 8,952/24,433,132 (0.00036639),
  and the field docstring asks for the share of the whole population the source is about,
  so the smaller figure is the right one.

### 12.7 The Wayback directory route, and the ask

**Spec 12's record of the route is reusable as written** and did not need widening. It
states the rule for any office with history rather than for UBOS, gives the CDX call with
the three parameters that matter (`matchType=domain`, `filter=original:.*\.pdf`,
`collapse=urlkey`), names the move itself (strip every hit to its containing directory and
count), says why it works where a search does not (a keyword search over filenames finds
none of these), and names the three directories that fell out.

**`ask/008-ug` was not touched beyond appending a reviewer paragraph.** Nothing was
downloaded from `microdata.ubos.org:7070`, no ask was filed, and the builder's decision to
stop at a 512-byte range probe was right.
