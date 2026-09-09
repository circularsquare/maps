# Micronesia — FSM Statistics, 2023 Population and Housing Census, Table B6

**DRAWN 2026-09-08.** Resumed from a checkpoint-B park; the parked session had `fm.csv` on disk
at four states and had costed the municipality option without taking it. §4 is where that
decision was reopened and settled the other way, with the numbers.

| | |
|---|---|
| source | 2023 Population and Housing Census, *Basic Tables*, `Table B6` |
| publisher | FSM Statistics (National Statistics Office), **`stats.gov.fm`** |
| route | WP File Download, `/download/<cat>/<slug>/<id>/…xlsx`, no key |
| tier | **33 units, two levels**: 20 Yap municipalities, 11 Pohnpei municipalities, Chuuk whole, Kosrae whole |
| people | **75,817** enumerated; **75,576 drawn (99.68%)** |
| categories | **11**, all drawn, nothing excluded |
| basis | self-identification, whole population, no age cutoff |
| outside witness | **none** — UNSD table 28 has no Micronesia row at any year |

## 1. The queue's host is gone and the office moved

`fsmstatistics.fm` resolves and returns HTTP 200. What it serves is a **LiteSpeed directory
index** with three entries: `cgi-bin`, `dasdas.png` and a stray `htaccess`. The WordPress site
that used to sit there is deleted, and its Wayback captures hold **no census PDFs or workbooks
at all** — the old site published its tables through the `wpDataTables` plugin, so the numbers
were AJAX and the archive kept only the shell.

The office is now **`stats.gov.fm`**, and it is WordPress with **WP File Download** — the sixth
Pacific office here on that plugin after Fiji, PNG, the Solomon Islands, Tonga and Kiribati
([[reference_wpfd_sweep]]). The whole library is:

```
https://stats.gov.fm/wp-admin/admin-ajax.php?juwpfisadmin=false&action=wpfd
    &task=files.getFiles&id=0&page=N
```

**71 files over eight pages** (seven of ten and one of one), which is the entire catalogue.
`task=categories.getFiles` returns the category only; `files.getFiles` is the one that returns
files, and `id=0` means the whole library rather than a category. Each row carries a
`linkdownload` field which is the public URL, so nothing has to be derived.

The sweep was run twice, on 2026-09-08, once by each session. Both got 71 files and the same
absence in §4.

## 2. What the census asks, and what it found

Religion is a census question and `Table B6` is a straight cross by state. **Its Total row
equals Table B1's population exactly**, so the universe is everybody enumerated and there is no
age cutoff to state in `grain`.

**The 2023 census counted 75,817 people.** `queue.md` priced Micronesia at 107,008, which is the
**2010** figure; the 2010 religion table's own universe was 102,843, and a decade of emigration
under the Compact of Free Association took the country down by a little over a quarter.

The eleven categories, national, as drawn (shares of the 75,576 drawn):

| category | people | share |
|---|---:|---:|
| Roman Catholic | 41,944 | 55.5% |
| Congregation/Protestant | 28,063 | 37.1% |
| Other religion | 1,167 | 1.5% |
| Mormon | 976 | 1.3% |
| Baptist | 880 | 1.2% |
| Assembly of God | 562 | 0.7% |
| No religion/Refused | 509 | 0.7% |
| Apostolic | 463 | 0.6% |
| SDA | 451 | 0.6% |
| Pentecostal | 319 | 0.4% |
| Jehovah's Witness | 242 | 0.3% |

**The geography is the point.** Kosrae is **88.7% Congregational and 1.7% Catholic**; Yap is
**79.6% Catholic and 4.6% Congregational**. Chuuk and Pohnpei sit between at 54.8/41.8 and
57.1/34.2. That is the Congregational mission in the east against the Catholic mission in the
west, and it is visible at four units — but see §4 for what four units cannot show.

## 3. `*` means suppressed or zero, and the loss is 241 people

The workbook's own note is *"Value is suppressed for confidentiality reasons or is zero"*.
`impute()` fills only a cell that is the **single unknown in its line**, from the table's own
subtotal column or from the unit's own Total, and never apportions.

At the four-state tier this recovers everything: five starred cells, all five recovered, nothing
left. At municipality it does not, because most lines have several unknowns at once — 58 of Yap
proper's 132 cells and 73 of Pohnpei's 132 are starred, and after imputation 55 and 73 unit cells
remain unknown and are set to zero.

**What that costs is 241 people, 0.32% of the country.** The starred cells are numerous but
small; the disclosure floor is somewhere around ten people.

### 3a. The lean, and the one that was not there — §3.5 run by the builder

**Per category the hole is real and it is not neutral.** Measured against the state column each
municipality block is anchored on, across Yap and Pohnpei together:

| category | in the two states | drawn at municipality | lost | |
|---|---:|---:|---:|---:|
| SDA | 334 | 275 | 59 | **17.7%** |
| Jehovah's Witness | 179 | 157 | 22 | **12.3%** |
| Assembly of God | 522 | 490 | 32 | 6.1% |
| No religion/Refused | 491 | 462 | 29 | 5.9% |
| Baptist | 776 | 744 | 32 | 4.1% |
| Apostolic | 293 | 282 | 11 | 3.8% |
| Other religion | 718 | 691 | 27 | 3.8% |
| Mormon | 653 | 639 | 14 | 2.1% |
| Pentecostal | 185 | 182 | 3 | 1.6% |
| Congregation/Protestant | 9,390 | 9,380 | 10 | 0.1% |
| **Roman Catholic** | 23,299 | 23,299 | **0** | **0.0%** |

So the finer tier **understates the small churches and not the large ones**, which is the exact
shape a disclosure threshold produces, and `gap` in `countries.py` says so in those terms.

**Per unit there is no lean, and the check that says so is the one that nearly went the other
way.** The residual share correlates **+0.469** with a unit's `No religion/Refused` share over
the 31 fine units, which reads as a finding: the units losing most to suppression are the ones
with the most irreligion. A permutation test puts it at p = 0.046, marginal. **Drop Kanifay,
the single extreme on both axes, and it is −0.011 over 30 units.** It is one municipality, not a
lean. [[reference_check_needs_power]]; without the permutation and the leave-one-out this would
have been written up as a real effect.

### The publisher perturbs its own margins by one or two, everywhere

The four printed state totals sum to **75,818** against a printed national **75,817**. Table B1's
own age rows come to Yap 10,738 / Chuuk 33,884 / Pohnpei 26,103 / Kosrae 5,092, which is 75,817
exactly, while B1's printed state totals read 10,739 / 33,885 / 26,102 / 5,092. Pohnpei's eleven
municipalities sum to **26,104** against its own printed **26,102**; Yap's twenty sum to 10,741
against a printed 10,739. Every one of these is within a couple of people of its own column.
`MARGIN_SLACK = 4` in `fm.py` is the tolerance and the discrepancy is printed on every run rather
than absorbed. **An equality assert here refuses a correct read.**

### There is no outside witness on any of these numbers

FSM has **never forwarded a religion tabulation to UNSD**, so `tools/oracle.py` has no row at any
year and the Kiribati-style check is unavailable. What replaces it is §3b.

### 3b. Two separately published workbooks agree exactly, which is the strongest check here

The national workbook and the per-state workbooks are different releases. Yap's and Pohnpei's own
`TOTAL` columns reproduce the national table's column for that state **on all eleven categories
with a difference of zero**, and their state totals match to the person (10,739 and 26,102).

That is the check a column permutation cannot survive: every within-table margin still closes if
a whole state's column is read into the wrong place, and this one does not.

## 4. THE TIER: municipalities where they exist, states where they do not

**This is the country's biggest call and it was reopened deliberately.** The parked session built
four states and wrote down why; this section is why it went the other way, and what would reverse
it.

FSM publishes a **per-state basic-tables workbook** beside the national one. The parked note said
Yap, Pohnpei and Kosrae each carry `Table B6. Religion by Municipality`. **That is wrong about
Kosrae.** Kosrae's 2023 workbook runs B1 age, B2 relationship, B3 birthplace, B4 citizenship,
B5 marital status and then the household tables. There is no B6 in it.

| state | people | 2023 per-state workbook | drawn as |
|---|---:|---|---|
| Yap | 10,739 | **file 2212**, religion by municipality | **20 municipalities** |
| Pohnpei | 26,102 | **file 2216**, religion by municipality | **11 municipalities** |
| Chuuk | 33,885 | **DOES NOT EXIST** | the state |
| Kosrae | 5,092 | exists, but **stops at Table B5** | the state |

The `id=0` sweep is the evidence for Chuuk's absence: 71 files, no Chuuk 2023 workbook and no
Chuuk 2023 factsheet either, while the other three states have both. Kosrae's absence is stronger
than that, because the file is on disk and `check_kosrae_has_no_b6()` in `fm.py` **asserts it on
every run** and fails the build if a B6 ever appears, naming the four municipalities to draw.

So the fine half is **36,841 people, 48.6% of the country**, and the coarse half is 51.4%.

### Why the mixed tier and not four states

**Because the project already has a written rule for exactly this, and it is not the one Albania
applied.** spec §12, from the Indonesia finding:

> Do not take the finer tier just because it exists. Measure it against the coarser one you
> already trust, per parent, and where it falls short prefer Ghana's answer: **draw the fine unit
> where it reconciles and the coarse one where it does not**, so the drawn tier is two
> `geo_level`s and every drawn row is still `measured`.

Measured, per parent: Yap's twenty reconcile to its state total within 2 and to the national
table exactly; Pohnpei's eleven within 2 and exactly. Chuuk's and Kosrae's do not exist. That is
Ghana's answer, so it is what was built.

And §9as-ii, from France: *"§14.3's never model finer than the source publishes is satisfied by
the source rather than by an argument… A country that declines a resolution its own source
publishes should record which of those two it is declining."* Nothing here is spread or modelled;
each state is drawn at the grain its own workbook prints.

**Albania (`sources/al.md` §3a) is the case that looks like this and is not.** There the finer
option cost **24.4% of the categories** and **twelve years of vintage**, and would have left half
the country living in a unit more than a quarter grey. Micronesia's finer half is the **same
census year, the same office, the same eleven categories and the same table number**. It costs
0.32% of the people and no category at all: the worst-hit is SDA at 17.7% of 334 people, and no
category is destroyed. That is a different trade and it goes the other way.

### What it buys, in figures

Four states cannot show any of this:

* **Inside Pohnpei island** the Catholic share runs from **86.3% in Nett** to **39.3% in
  Sokehs**, against a state figure of 57.1%.
* **Fais**, an outer island of 463 people in Yap, is **29.4% Assembly of God and 28.5%
  Baptist**, against national shares of 0.7% and 1.2% and a Yap state figure of 1.4% and 3.4%.
* **Sapwuahfik** in Pohnpei is **20.3% Seventh-day Adventist** against 0.6% nationally.
* **Sokehs** holds **282 of the country's 463 Apostolic**, 5.8% of that municipality.
* **Kanifay is 28.9% `No religion/Refused` and Rumung 33.3%**, both in Yap, against 0.1% of
  Chuuk. See `taxonomy/fm2023.py`'s REVIEW for why that is believed.
* Twelve of Yap's twenty municipalities record **no Congregational answer at all** and six of
  those are 100% Catholic; four of Pohnpei's outer atolls record **no Catholic answer** and are
  100% Congregational.

### The honest cost, stated

**44.8% of the people on this map are inside one Chuuk polygon.** Chuuk is where within-state
variation is likeliest to be largest — Chuuk Lagoon, the Mortlocks and the Northwest Islands are
three different places, and COD's own ADM2 file groups its forty municipalities into exactly
those — and it is drawn as 54.8% Catholic and 41.8% Congregational throughout. The map is much
sharper at the two ends of the country than in the middle, and `grain` and `note_public` both
say so rather than leaving the reader to infer it from the dots.

### How to reverse this cheaply, in either direction

* **To go back to four states**: `sources/fm.py` still parses the national table in full and
  `parse_national()` returns all four states' eleven categories; emit those four instead of
  calling `parse_yap()` and `parse_pohnpei()`, drop the `fm` row from `DEFAULT_LEVELS` in
  `tools/check_mapping.py`, and rebuild `fm_geo.py` dissolving all four states. About an hour.
* **To go finer**: **check the library for Chuuk's 2023 workbook first.** The other three landed
  together and Chuuk's may simply be late; that single file turns this into a ~73-unit country
  and removes the whole compromise. Kosrae's B6 is checked automatically on every run. The 2010
  per-state tabulations (files 596–599) cover all four states at municipality, but they are
  thirteen years stale against a country that lost a quarter of its people, and they lay their
  tables out side by side with region subtotals interleaved with municipalities, so a parser has
  to cut the hierarchy or double-count.

## 5. Boundaries and placement

**COD-AB `cod-ab-fsm` (2019), the ADM2 municipality shapefile**, 75 polygons: FM101–FM120 Yap,
FM201–FM240 Chuuk, FM301–FM311 Pohnpei, FM401–FM404 Kosrae. Each carries its ADM1 parent, a
`MAIN_OUTER` flag, a `GROUP`/`SUB_GROUP` pair and the office's own `STATS_MCOD`.

**The join is by name inside the state and the p-code tests it afterwards**, which is Benin's
rule (§9bb) and it pays here: **COD's FM101..FM120 and FM301..FM311 run in exactly the census's
own print order**, so a name that matched the wrong municipality would move a rank and be caught
even though every total still reconciles ([[reference_name_join_wrong_neighbour]]). 31 of 31 join;
two need an alias, both one letter — COD writes `Mwokilloa` and `Sapwuafik` for the census's
`Mwoakilloa` and `Sapwuahfik`.

**And COD's `MAIN_OUTER` reproduces the workbook's own table structure.** Yap's Table B6 is
printed in two blocks, `YAP PROPER` (ten) and `OUTER ISLANDS` (ten); COD independently flags the
same ten Main and the same ten Outer. Neither file has seen the other, so that is evidence about
the pairing and it is asserted.

Chuuk's 40 and Kosrae's 4 are dissolved whole, and each dissolve is checked against COD's own
separately published ADM1 polygon (1.0000 of it in both cases), so a dropped or duplicated
municipality cannot pass.

**Micronesia does not cross the antimeridian**, unlike Kiribati and Fiji: 137.49°E to 163.04°E,
1.03°N to 10.09°N. So the ordinary per-country bounding-box width assertion is the right one here
and is used ([[reference_antimeridian]]). No `view` is set; the dot bbox is the correct frame.

### Placement: Kontur, and it reads 1.50x the census on purpose

Kontur's `FM` extract (2023-11) has 731 hexes and **113,340 people against a census 75,817**, a
ratio of 1.495. **That is the country, not the grid**: the 2010 census counted 102,843 and FSM
lost a quarter of its people in between, so Kontur is at roughly the 2010 level.
`NATIONAL_TOLERANCE` is widened to 0.70 with that reason written next to it. The grid is only
ever a within-unit weight and is normalised by this ratio before any unit is judged.

**The per-unit band is what actually tests it** (spec §12's Eswatini finding): all 30
non-synthetic units inside a factor of 5, running 0.39 (Gilman) to 1.95 (Elato), and
**r = 0.9760 against a best of 0.5465 over 2,000 random pairings, none of which reach it**.

**30.2% of Kontur's people fall outside every unit and are snapped, not dropped** — 97.9% of the
strays are within 700 m. An atoll is a strip of land a few hundred metres wide, so nearly every
cell is a shoreline cell; this is Vanuatu's rule (§9bg) and Kiribati's precedent
([[reference_archipelago_grid_snap]]). 14 cells, 730 people of weight, remain unplaced and are
dropped; nobody leaves the map, because these are weights and not counts. Faraulep, Ifalik and
Ngulu are smaller than one 400 m hex and get their own polygon as a single cell (§8.2).

`water.py` clips 494 of the 720 placement polygons and reports 35 units losing over 95% of their
area to the sea, which are left unclipped. On this country that is normal rather than a warning.

## 6. The mapping

`taxonomy/fm2023.py`. Eleven categories, **nothing excluded**, one new node (`other.fm`). The two
calls worth reading are in its `REVIEW`:

* **`Congregation/Protestant` → `christianity.protestant`**, 37.1% of the country, and NOT
  `christianity.reformed.congregational`. `mh1999.py` and `pw2005.py` send the *same church* —
  the American Board's United Church of Christ — to `christianity.protestant` in the Marshall
  Islands (54.8%) and Palau (23.2%), and drawing Micronesia's in a different colour would put a
  religious boundary between three adjacent countries where none exists. The cell is also merged
  (`Congregation/` **`/Protestant`**) and Table B6 has **no `other Christian` cell at all**, so
  every Protestant the other ten categories do not name is inside it. And the Congregational
  reading is only safe for about half: Kosrae and Pohnpei are American Board ground, but Chuuk
  and Yap (14,654 between them) went to the German Liebenzell Mission around 1907, which is
  pietist and evangelical, and Chuuk is the state with no municipality table.
* **`No religion/Refused` → `unaffiliated`**, and **the 2010 census of the same country asked
  them apart, which is what settles it.** Table B09 of the 2010 workbook prints `No religion`
  **723** and `Refused` **72**, so 90.9% of a merged cell of this kind here is people with no
  religion; in Yap, which carries most of it, the 2010 split is 281 against 12, or 95.9%. This is
  Palau's method (`pw2005.py` used the 1995 split of 1,577 against 7) applied to the neighbour,
  and it was worth the download: the municipality tier makes this cell **28.9% of Kanifay**,
  which without the 2010 evidence would read as enumeration behaviour rather than belief. 2010
  found the same shape with refusals removed separately — Yap 2.5% no-religion against Chuuk
  0.05% — so the Yapese concentration is real.

`Apostolic` follows `ck2011.py` to `christianity.pentecostal`, the parent that asserts neither a
Welsh-Apostolic nor a Oneness reading. `Roman Catholic` follows `ki2015.py` to
`christianity.catholic.latin`. The rest are the usual nodes.

A second agent was asked to read the files and rule on the `Congregation/Protestant` call
without being told the intended answer; it reached `christianity.protestant` independently, added
the point that a `…congregational.<uccm>` leaf would sit beside the existing
`christianity.reformed.congregational.ucc` for the unrelated US denomination, and pushed back on
`No religion/Refused` — which is what sent this to the 2010 workbook.

---

## 7. Review, 2026-09-08 (`rd-review`, second perspective)

Everything in §§1-6 that this review could check independently held up. The 2023 workbooks were
re-parsed from `data/raw/fm/` without going through `sources/fm.py`, and every reader-facing
figure in `countries.py` reproduces exactly from `data/normalized/fm.csv`: 55.5 / 37.1 national,
Kosrae 88.7 / 1.7, Yap 79.6 / 4.6, Chuuk 54.8 / 41.8, 44.8% in one Chuuk polygon, Fais 29.4 and
28.5, Sapwuahfik 20.3, Kanifay 28.9, Rumung 33.3, Sokehs 282, twelve Yap municipalities with no
Congregational answer of which six are wholly Catholic, four Pohnpei atolls with no Catholic one.

**The three things this review was pointed at all survive.**

* **Kosrae's missing B6 is real and is not one reading.** Every cell of all three sheets of
  `kosrae_basic_tables_2023.xlsx` was swept for `religio|catholic|protestant|congregat|…`: zero
  hits. Its `List of Tables` stops at B5 and `Basic_Tables` holds exactly five tables. Yap's same
  sweep returns 16 hits. `check_kosrae_has_no_b6()` scans the one place a B6 would appear in this
  workbook family and would fire.
* **The mixed tier is spec §12's Ghana rule applied as written**, and §12 does say what §4 quotes.
* **The §3.5 lean check reproduces**: r = +0.469 on residual share against irreligion share over
  the 31 fine units, permutation p ≈ 0.045, and −0.011 with Kanifay dropped. Kanifay's residual
  share is 7.6%, more than double the next worst (Fanif, 3.4%), so it really is one point. The
  per-category figures are exact on the fine tier: SDA 59 of 334 = **17.7%**, Witnesses 22 of
  179 = **12.3%**, Roman Catholic 0 of 23,299 = **0.0%**. `gap` states the second, correctly, and
  correctly scopes it to Yap and Pohnpei; nationally those are 11.7% and 8.3%.

Also confirmed: `impute()` never apportions. It is an iterative single-unknown solve across the
national table's rows *and* columns, so Chuuk AoG = 11 comes from the AoG row, Kosrae JW = 14 and
No religion = 11 from their rows, and Kosrae Apostolic = 4 from the closed column. All 241 lost
people are in the fine tier; the coarse tier loses nobody.

### THE ONE REAL FINDING: the outside witness exists, and the record said it did not

`countries.py` said *"There is NO outside witness: UNSD table 28 has no Micronesia row at any
year"* and `note_public` told the reader Micronesia *"has never sent a religion tabulation to the
UN Statistics Division"*. **Both were wrong.** UNSD table 28 carries
`Micronesia (Federated States of)`, **2000**, 8 categories, **107,008**, an exact partition:

```
56,365 Roman Catholic   42,879 Congregational   4,066 Other Religions   1,123 Latter Day Saints
   972 Baptist             793 Seventh Day Adventist    753 No Religion     57 Refused to answer
```

**The cause is a tool trap worth knowing.** `tools/oracle.py` matches on **UNSD's own country
name**, never on a country code, so `python tools/oracle.py fm` prints *"ABSENT from the oracle
(proves no census tabulation was forwarded, nothing more)"* — and it prints that for **every** cc
ever passed to it. The working call is `python tools/oracle.py "Micronesia (Federated States of)"`.
This is a sibling of the failure `sources/bg.md` §3 already documents, where filtering the wrong
column *"looks exactly like 'this country is absent from the oracle', which is the one conclusion
§11r warns is expensive to get wrong."*

**The witness agrees, which is why this is worth having rather than merely worth correcting.**
Folded to 2000's eight categories: Catholic 52.67% → 55.50%, Congregational 40.07% → 37.13%,
Latter Day Saints 1.05% → 1.29%, Baptist 0.91% → 1.16%, Adventist 0.74% → 0.60%, no-religion plus
refused 0.76% → 0.67%, and Other Religions (which in 2000 must contain Assembly of God,
Pentecostal, Apostolic and Jehovah's Witness, since 2000 has no column for them) 3.80% → 3.64%.
Every category is within three points across twenty-three years, and the two large ones move
opposite by the same amount, which is drift rather than a parse error. The 3.64% agreement is a
free check on the 2023 read of the four categories 2000 does not name.

**And it corroborates the merged-cell call a second time.** The 2000 tabulation prints
`No Religion` 753 against `Refused to answer` 57, i.e. **93.0%** genuine no-religion, beside the
2010 census's 723 against 72, i.e. **90.9%**. `No religion/Refused` → `unaffiliated` now rests on
two independent rounds, not one.

`note_public` and the `note` were corrected in `countries.py` and `tiles.py --refresh-meta` run.
**`sources/fm.py` was NOT touched**: it still sets `ORACLE = None` and prints *"oracle check
SKIPPED — UNSD table 28 has no Micronesia row at any year"* at line 459. Wiring the real check in
is a code change on a country whose builder has released it, and belongs to whoever next opens
the file.

### Smaller notes, none of them blocking

* **`grain`'s only number describes the minority of the map.** It reads "municipalities in Yap
  and Pohnpei (1,200 people); whole states in Chuuk and Kosrae", and the coarse tier, which is
  51.6% of the people, gets no figure. Indonesia's mixed-tier `grain` has the same shape, so this
  is precedent-consistent, but Indonesia's coarse tier is not the majority. "(19,500)" after
  Kosrae would close it. Not changed here.
* **Shares in `note_public` are of the 75,576 drawn, not of the census's printed 75,817.** So
  Catholics are 55.5% here and 55.3% in the census; Adventists 0.6% here and 0.7% printed;
  "282 of the country's 463 Apostolic" is 473 in the census. This is internally consistent and
  matches what the viewer shows, which is the better choice, but it is worth knowing that the
  denominator is the drawn base. It bites hardest at **Kanifay**, quoted at 28.9%: the census
  prints 88 of 330, which is 26.7%, because 25 of Kanifay's 330 are in starred cells. Kanifay is
  the most-quoted unit in the note and the most suppressed unit in the country.
* **At the default 1,000 people per dot, seven of the ten drawn nodes reach zero dots** and show
  only as presence rings; Micronesia draws 70 dots in total. So the "small churches are
  island-sized" paragraph describes a geography the dots never render at any available dot value.
  This is the designed ring behaviour and is normal for a small country (Montserrat and Niue draw
  no dots at all, Tuvalu one), so it is not an fm defect, but the phrase "only the finer half
  shows it" is true of the figures rather than of the picture.
* **spec §12 says 48.6% fine and 44.7% Chuuk; the country record says 48.4% and 44.8%.** Both are
  right on their own base (printed 75,817 against drawn 75,576). Left alone.
* Screenshot taken: dots on land in all four states, no dots in the sea, colours read as Catholic
  and Protestant-unspecified as expected, panel and legend correct. Nothing looked off.
