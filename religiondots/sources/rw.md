# Rwanda — NISR, RPHC-5 2022, the thirty district profiles, Table 2.2

Wired 2026-09-08. 13,246,394 people, **30 districts**, 10 drawn categories, **99.87% drawn**.

| | |
|---|---|
| source | National Institute of Statistics of Rwanda, **RPHC-5 (2022)**, the **thirty district profiles** released May 2025, **Table 2.2** in each |
| basis | `self_id`, whole census population |
| geography | **30 districts** — 441,546 people each, against the **5 provinces** the queue priced |
| categories | **11** plus the universe total; 10 drawn, `Not stated` excluded |
| drawn | **13,228,609 people, 99.87%** |
| gap | `Not stated`, **17,785 people, 0.13%** — the smallest §3.5 residual of any census on this map that prints one |
| boundaries | NISR's own **`Population_2002_2022`** feature service on ArcGIS Online, 416 sector polygons, dissolved to district |
| placement | Kontur 400 m hexes **re-levelled sector by sector onto the census count** |
| licence | NISR publications, free to download and cite |

**The two things worth carrying off this country** are that the queue row was priced off the
wrong publication, and that the boundary layer proved the join instead of the code asserting
it. Both are §11p's and §9bv's lessons arriving together.

---

## 1. The queue said five provinces. It is thirty districts, and the reason is §11p's own rule

sources.md **§11p** closed the African sweep with a rule it had just derived: *the predictor
for an African source is whether the country runs a THEMATIC REPORT SERIES, and which theme
religion got.* It then applied that rule to Rwanda and got the right answer to the wrong
question. Religion **is** Chapter 4 of the RPHC-5 thematic report *Social-cultural
characteristics of the population*, and that report **does** work at province, so §11p wrote
Rwanda down as *"buildable and very coarse, 13.2M over 5 units"* and moved on.

What the rule cannot see is a series that is not the thematic series. In **May 2025** NISR
published a **district profile for every one of the thirty districts** — 80 to 140 pages
each, under `/district-statistics/<province>` rather than under the census pages — and
**every one carries the religion question in full as its Table 2.2**, the same eleven
categories, counts and percentages, split Total / Urban / Rural.

**That is Botswana's shape exactly (§9bu)**, three weeks after Botswana taught it: the
national report stops where the national report stops, and a per-district booklet series
published outside the report set the sweep tested carries the fine geography. The generalised
form is now in spec §12; the short version is **a national report that stops at the nation is
not evidence about the country.**

441,546 people per unit instead of 2,649,279. The difference is visible in one column more
than any other: **Adventists are 12.17% of Rwanda and run from 33.94% in Nyanza to 2.38% in
Gicumbi**, a belt across the middle of the country. At province level the same figures run
only **9.48% to 14.58%** and the belt does not exist.

## 2. What the eleven categories are, and the one that is a church

    39.91%  Catholic              5,286,003
    21.29%  ADEPR                 2,820,813
    14.56%  Protestant            1,928,741
    12.17%  Adventist             1,612,482
     4.18%  Other Christians        553,174
     3.04%  No Religion             402,517
     2.00%  Muslim                  265,317
     2.00%  Other religion          264,319
     0.70%  Jehovah witness          93,131
     0.13%  Not stated               17,785   <- the gap
     0.02%  Traditional/Animist       2,112

**`ADEPR` is a single named denomination counted as a census category at 21.29% of a
country, which nothing else on this map is.** The Association des Églises de Pentecôte du
Rwanda grew out of the Swedish Free Mission that reached Rwanda from the Belgian Congo in
1940 and took its present name in 1983. The **UNSD Demographic Yearbook prints the same
2,820,813 under the generic label `Pentecostal`**, which is how NISR described the cell to the
UN — so the booklets are strictly better than the Yearbook here, and the difference is the
church's name.

Its geography checks against its own history: ADEPR's first congregation was in **Rusizi**, on
the Congo border, and Rusizi is still its strongest district at **33.90%**, with Nyamasheke
next door at 30.70%; it thins eastward across the Catholic plateau to **11.12%** in Nyamagabe.

`Traditional/Animist` is **2,112 people, 0.02%**, eight times smaller than the next smallest
cell, and is read as a floor for §11b's standing reason — the box is exclusive of the
Christian ones, and *kubandwa* accompanies church membership rather than replacing it. See
`taxonomy/rw2022.py`'s REVIEW entry, which also notes that the cell is largest in Kigali,
which is one more reason not to read it as a measure of practice.

`Other religion` at **2.00%** is the same size as the Muslim cell, which is unusual for a
residual sitting beside boxes for Islam, traditional practice, the Adventists, the Witnesses
and ADEPR by name. NISR publishes no breakdown of it anywhere and none is invented; see
`taxonomy/branches.py`'s `other.rw`.

## 3. The join is PROVED, and that is the second thing to carry

§9bv found that Moldova's statistics office publishes its own boundary layer with the census
population on every polygon. Rwanda is the same finding on a different platform, and it is
worth writing down that **`gis.statistics.gov.rw` is a dead end while the content is real**:

* `gis.statistics.gov.rw` is a **DNS alias of the Drupal site**. `/Portal/sharing/rest`,
  `/server/rest/services` and `/arcgis/rest/services` all return a 61 KB Drupal **404 page**
  with HTTP 404, which reads exactly like an office that has no GIS.
* The content is an **ArcGIS Online organisation**, reachable only through
  `https://www.arcgis.com/sharing/rest/search`. `q=tags:nisr` returns **95 items**,
  `owner:NISR_Publisher` 58, `owner:GIS@NISR` 57, `owner:NisrProject` 12. All public.
* **There is no religion layer** and it was worth checking: `title:religion AND Rwanda`
  returns three items, none of them Rwandan.
* What there is: **`Population_2002_2022`** (`91890b6718b549c89efee8c4757a34c4`), 1,248
  features — **416 sector polygons × three census years** — carrying `tot_pop`, `pop_ur`,
  `pop_ru`, `hh_number`, `area_sqkm`, `district`, `district_id`, `province`.

Filtered to `census_time='2022'` it sums to **13,246,394, the census population exactly**.
Dissolved to district it gives thirty populations, and **every one equals its own booklet's
total to the person**. The thirty totals are all distinct (318,126 Nyaruguru to 879,505
Gasabo), so equality alone determines the pairing: there is no room for
`[[reference_name_join_wrong_neighbour]]` to hide, because a swapped pair does not look odd,
it fails. The layer's `province` field is checked against the province each booklet prints
first, so a pairing cannot cross a province before the counts are looked at.

**And the sector layer excludes Lake Kivu.** It covers 24,306 km² against Rwanda's 26,338 km²
of total area and about 24,668 km² of land, so §8.2c's dots-on-open-water problem does not
arise and `water.py` is not involved — the same way Lake Malawi and Lake Kariba resolved
themselves.

## 4. The placement grid is re-levelled onto the sectors, which is new here

Rwanda counted finer than it published religion. Religion stops at the 30 districts; the
population count exists for all **416 sectors**. So `sources/rw_grid.py` does not use Kontur
as the within-district weight in the usual way. It scales Kontur's hexes **per sector** so
that each sector's hexes sum to its census count, which makes everything coarser than about
15 km² and 32,000 people a measured quantity and leaves only the shape inside a sector
modelled.

**The check has to run on the raw ratios, before the scaling**, because afterwards every
ratio is 1.000 by construction and the file would prove nothing. Measured that way:

* Kontur 14,067,721 against the census 13,246,394 — **ratio 1.062**, which is the right size
  for a 2023 grid against a 2022 count.
* per sector, normalised by that ratio: **0.45x to 2.07x, median 0.99**, nothing outside 2.5x.
* the correlation, which is what carries the check here: **r = 0.9113 on 416 sectors against
  a best of 0.1507 over 2,000 random pairings, none of which reach it.**

That is the reverse of `sources/zw_grid.py`'s answer on the same pair of tests. Ten
provinces of similar size correlate by luck and the band discriminates; 416 sectors do not
correlate by luck and the correlation discriminates. **Measure both, use whichever the
country's own shape makes discriminating, and say which one it was.**

Three sectors have fewer than three Kontur cells (a Kontur cell is H3 resolution 8, about
0.74 km²) and they are the small dense ones in Kigali. Their people are placed inside that
cell, which is finer than the district grain could resolve anyway.

## 5. The parse, and two traps that would both have been silent

The text layer emits a row label and then its three counts and three percentages, one per
line, which invites a plain line read. **A plain line read is wrong on this source twice.**

**Table 2.1 shares the page, and one of ITS column headers is the word `Rwanda`.** Table 2.1
is nationality: `Total / Rwanda / Foreigners` twice over. So a reader anchoring on the first
line that says `Rwanda` lands in the nationality table and reads `Foreigners` as a figure.
Rutsiro fails loudly; a table laid out slightly differently would not have. The anchor used
instead is the first line that is exactly **`Catholic`**, which is inside Table 2.2 by
construction.

**Gakenke prints a BLANK cell instead of a zero.** Its `Traditional/Animist` row is 51 people,
all of them rural, and the urban cell is empty. In the text layer that row reads
`51 / 51 / 0.01 / 0.01`, so a reader taking the first three tokens as the three counts gives
the district **0.01 traditionalists** and shifts its percentages one column left — on a row of
51 people in one district, which no total would ever query. So the columns are cut on
**position**, using the `Catholic` row (full in all thirty booklets) as the template.
`[[reference_pdf_table_geometry]]`.

Two smaller ones: the table **breaks across a page** in several booklets, always repeating the
header and always leading with the printed folio, which is a bare integer where a figure is
expected. And the **title is not a constant** — most say *"by Religious Affiliation and
residence areas"*, Rwamagana says *"by Area of Residence and Religious Affiliation"* — so the
page is identified by having a `Catholic` row with six figures on it rather than by its title.

## 6. The reconciliation, and the outside witness

Internal, all clean:

* the 11 categories sum to the district total on all 30 districts;
* `Urban + Rural == Total` on all 360 cells;
* the printed percentages reproduce from the counts on all 985 cells to within a unit in the
  last place. **One cell rounds the wrong way in NISR's own arithmetic** — Gicumbi's
  `Other religion` rural cell is printed 1.3 where the counts give 1.2496 — which is why the
  bar is one unit in the last place and not half of one;
* the 30 districts sum to 13,246,394 / 3,701,245 / 9,545,149;
* 27 of the 30 booklets reprint the national row and all 27 agree; 27 reprint their province
  row and each province's districts sum to it. **Gakenke, Kirehe and Musanze start straight
  at the district row** and print neither, which is why those two rows are optional in the
  parser rather than asserted.

**The outside witness is the UNSD Demographic Yearbook table 28** (`tools/oracle.py Rwanda`),
which is NISR's own return to the UN and a wholly separate publication from these booklets.
**All eleven national totals reproduce to the person.** That is a check on the read rather
than on the arithmetic: every identity above holds inside one publication whichever way its
columns were taken, and this one does not.

## 7. What was NOT used, and why

* **The RPHC-5 microdata** at `microdata.statistics.gov.rw` (catalogue 109) is a **10% sample**
  — variable `V266`, *Religious affiliation (p13)*, 13 categories, 1,313,015 valid cases — and
  would in principle reach the 416 sectors. It is behind a **Login/Register** wall, which is
  `[[reference_ipums_account]]`'s class and not chased. The district profiles are a complete
  count and the sample would be a worse source at a finer grain, so this is not a loss.
* **The RPHC-5 Population Census Atlas** is an ArcGIS StoryMaps collection
  (`storymaps.arcgis.com/collections/37d4f88579d14f6cb910f988a10fe862`) over ten themes.
  Religion is not one of them.
* **COD-AB Rwanda** (`rwa_adm_2006_NISR_WGS1984_20181002`) would have served for boundaries
  and is NISR's own 2006 geography republished by OCHA. NISR's 2022 layer was taken instead
  because it carries the census count, which is what turns the join from a name match into a
  proof.
* **The 2012 (RPHC-4) and 2002 profiles** are on the same pages, under `/2025-06/` and
  `/2025-08/`. They are a different census and must not be read for this. `Population_2002_2022`
  carries 2002 and 2012 sector populations too, if anyone ever wants the change.

## 8. Fetch notes

* The thirty RPHC-5 profiles are at `/sites/default/files/2025-05/<District>.pdf` **except
  Nyabihu**, which is `/sites/default/files/2025-10/Nyabihu_RPHC2022.pdf`. That is the one
  file a pattern-guessing fetch misses. Three districts are shouted in the filename
  (`BURERA.pdf`, `BUGESERA.pdf`, `KIREHE.pdf`) and the rest are title case.
* `www.statistics.gov.rw` **fails TLS**: the certificate's altnames list `statistics.gov.rw`
  and not the `www` host, so a strict client errors while a browser follows the redirect. Use
  the bare host.
* The five listing pages are `/district-statistics/{kigali-city,southern-province,
  western-province,northern-province,eastern-province}` and answer a plain scripted GET.

---

## 9. Review, 2026-09-08 (session `967ffe99-…-rw-rev`)

Read the mapping, the normalized CSV, both grid modules and the PDFs, not the builder's
account of them. **Nothing here needs rebuilding.** What was checked, and what moved:

**Every reader-facing figure reproduces from `data/normalized/rw.csv`.** 46 district
percentages quoted across `countries.py`'s `note_public`, `taxonomy/rw2022.py`'s REVIEW
entries, `taxonomy/branches.py`'s `other.rw` note and this file were recomputed
independently; all 46 agree to the printed place. So do the unit counts and the
superlatives: Catholic largest in **27 of 30** districts with the other three being Karongi
and Nyanza (Adventist) and Nyamasheke (ADEPR); the Adventist range **14.27x** against
Catholic 3.08x, ADEPR 3.05x, Protestant 5.77x; the province-level collapse to 9.48%-14.58%;
Gasabo the largest district; `Traditional/Animist` the smallest cell by a factor of 8.42;
`No Religion` the flattest column at 2.83x. `gap_share` 0.001343 is 17,785/13,246,394.

**The re-levelling does what it claims, and the check is genuinely not circular.** The read
was reproduced without writing anything. `r = 0.9113` recomputes exactly on the raw
pre-scaling ratios, and **the same statistic run after the scaling comes out at 1.000000** —
which is what a circular check would have reported, so the distinction the module insists on
is a real one and not a formality. Kontur inside the sector layer is 14,067,721 against the
census 13,246,394 (ratio 1.0620), per-sector 0.452x-2.067x, median 0.991; three sectors have
fewer than three hexes and all three are inner Nyarugenge, as stated. After scaling, every
sector and every district sums to its census count to 0.000000, and the shipped
`rw_hexes.gpkg` reproduces hex for hex (24,330). The re-levelling is not cosmetic: within a
district it moves a **median 6.1%** of the dot weight off Kontur-alone (max 12.9%).

**One caveat on how §4 frames that correlation.** Two thousand permutations is a *sample*,
so its maximum is not a constant — the same code with a differently-consumed RNG stream
gives a best of 0.1878 rather than 0.1507. The conclusion is untouched (none of them come
close to 0.9113), but the figure should be read as one draw. Related, and worth knowing
rather than acted on: Kontur's population blend descends in part from grids that are
themselves census disaggregations, so the agreement is not quite two independent
measurements meeting. That does not weaken the test for the job it actually does — the code's
own failure messages have it right, the thing it discriminates is **whether the sector join
is carrying information** — but §4's framing claims a shade more than that.

**The join is proved, independently.** The 30 booklet totals were checked against the
dissolved layer's own `tot_pop` outside the builder's code: 30 equalities on 30 distinct
values, 318,126 to 879,505, so no other pairing of these names to these polygons reproduces
them. `[[reference_name_join_wrong_neighbour]]` has nowhere to sit.

**Both parse traps are real and both guards hold.** Confirmed by reading the PDFs directly:
Gakenke's `Traditional/Animist` row genuinely emits four tokens (`51 / 51 / 0.01 / 0.01`)
where every other row emits six, and the geometric cut places them correctly. The `Rwanda`
nationality header is on the Table 2.2 page in **four** booklets (Rutsiro, Rubavu, Rusizi,
Nyamasheke), carrying `13,246,394 / 13,129,019 / 117,375 / 100` — a four-column anchor set
that would have read `Foreigners` as a figure. Anchoring on the full `Catholic` row avoids
both. The full read reproduces `rw.csv` on all 360 rows, and NISR's one wrong-way rounding
is Gicumbi's `Other religion` rural cell, as §6 says.

**One screenshot, as a smoke test.** Dots fill the country and stop at the border, Lake Kivu
is empty as §3 predicts, Kigali reads as a dense cluster, and Akagera and the Nyungwe belt
are visibly thin. Legend totals match the table. Nothing that wants a human eye.

**Three countable claims were wrong and are corrected in place**, all of them prose rather
than data:

* `taxonomy/rw2022.py`, `Muslim`: *"Nyarugenge is 11.31%, five times any other district"*.
  It is **2.9x** the next district (Kicukiro 3.96%) and 2.0x by headcount (Rubavu 21,145).
  Reworded to "nearly three times the next district and the only one above 4%".
  `countries.py`'s public note already had this right and was not touched.
* *"Four district names are shouted in the filename (`BURERA`, `BUGESERA`, `KIREHE`)"* —
  three are named and three is what `PROFILES` holds. Fixed here, in `sources/rw.py` and in
  sources.md §9cb.
* sources.md §9cb called the `/2025-08/` files the **2008** profiles where §7 above calls
  them 2002. Rwanda's censuses are 1978, 1991, 2002, 2012, 2022 and the layer is
  `Population_2002_2022`, so 2002 is right; sources.md now says RPHC-3 (2002).

**`ask/005-rw` is untouched and nothing found here bears on it.** `other.rw` is the only new
node and it follows the 40-odd `other.<cc>` precedents, so it is not a legend row anyone has
to argue about.

**The `check_tiles.py` DIFFERENT the builder reported was not Rwanda, and it has healed.**
The Oriental Orthodox pass edited `taxonomy/au2021.py` at 17:58:38 today, re-pointing
`Coptic Orthodox Church`, `Syrian Orthodox Church`, `Armenian Apostolic` and
`Ethiopian Orthodox Church` off the bare parent onto their own nodes, which forces `au` to be
re-scattered — and `dots_au_10k.geojson` and `dots_au.geojson` were rewritten at 18:02:29 and
18:02:32. The builder's check ran against an archive that predated that; the rebuild at
18:04:24 picked it up. So it was this session's own maintenance agent and not a foreign
session, and there is nothing to chase.

Verified rather than reasoned. Whole-tile comparison cannot settle one country, because a
low-zoom tile is shared with a dozen others, but the multiset of `c='rw'` features in a tile
does not depend on what any other country puts there, so it can be compared on its own.
Against the shipped `religiondots.pmtiles`: **Rwanda IDENTICAL on all 51 tiles carrying it**,
z0-10, both editions and the rings. **Australia is now IDENTICAL too, on all 1,906**, which
is what confirms the diagnosis rather than merely fitting it. Also useful to know: a bare
`check_tiles.py <cc>` reports a catastrophe on any country, because the reference is built
from the named countries only while the archive tile holds every country in the frame, and
because COMMANDS.txt step 11 runs `--no-atomic` and the flag has to be passed back.
