# Zimbabwe — ZIMSTAT, 2022 Population and Housing Census Report, Table 2.14

Wired 2026-09-07. 15,178,957 people, 10 provinces, 11 drawn categories, **100% drawn**.

| | |
|---|---|
| source | Zimbabwe National Statistics Agency, **2022 PHC Report**, **Table 2.14(a)/(b)/(c)**, pages 144–145 of 259 |
| basis | `self_id`, whole census population |
| geography | **10 provinces** — ~1.5M people each, **the coarsest counting geography on this map** |
| categories | **11** plus the universe total; all 11 drawn |
| drawn | **15,178,957 people, 100%** — no `not stated`, no residual, no gap |
| licence | ZIMSTAT publication, free to download and cite |

**One category is the whole reason to draw the country.** `Apostolic Sect` is **6,112,503
people, 40.3% of Zimbabwe** — the Vapostori — and no other source anywhere on this map
counts them. It is larger than every other Christian cell put together and nearly twice the
size of everything `christianity.africaninstituted` held before Zimbabwe arrived.

---

## 1. The easiest country on the map, and worth saying why

Everything reconciled on the first run. There was no wall, no key, no hunt, no geometry
parse, no name-join subtlety and no vintage problem. It is worth recording what that
combination looks like, because §11p's own predictor said it in advance and was right:

* **the religion table is ONE PAGE** of a 259-page report (§11b's rule: the size of the table is the predictor, not the platform);
* **the text layer gives the figures one per line in column order**, so the parse is a line read rather than a geometry problem — the reverse of Benin's booklets and Malawi's rotated page;
* **ten province names, character-for-character identical** between ZIMSTAT and COD;
* **an exact partition** — the eleven categories sum to the province total on all ten rows;
* **a free read-check** in the two sex tables.

Total elapsed: well under an hour. Benin took a session. The difference was entirely in the
source, not in the country.

## 2. Ten provinces for 15.2 million people, and that is the ceiling

**~1.5M per unit, coarser than Kenya's 47 counties (1.0M), which was the previous extreme.**
Spec §3.9b is what makes it drawable: there is no minimum unit count, the rule is to take
the finest geography a country publishes, say what it therefore cannot show, and draw it.

It is ZIMSTAT's ceiling and not a choice made here, and three separate things say so:

* **Table 2.14 is the only religion table in the report.** Religion appears on six of 259 pages: a narrative and a figure at page 39, and 2.14(a)/(b)/(c) at 144–145. Nothing else in the report crosses religion with anything.
* **Religion got none of the five 2022 PHC thematic reports** — fertility, gender, disability, youth, projections. §11p's continental rule: *find the series index, read what religion was attached to, and that tells you the geography before you download anything.* Here it was attached to nothing.
* **The district and ward file carries population only.** `2022_Population_Distribution_by_District_Ward_SexandHouseholds` is exactly where a district religion table would live and it has none.

**What that costs the reader** is stated in the country note: a province is a sixth of a
country, so a cluster of dots says *this province, drawn where Zimbabweans live* and nothing
about which town. The Kontur placement makes the map look finer than the data is, which is
the standing risk with every coarse country here.

## 3. The parse, and the one check that earns its place

Three tables: **2.14(a) Male** and **2.14(b) Female** share page 144; **2.14(c) both sexes**
is on page 145. Only (c) is drawn.

The other two are parsed anyway, and the reason generalises: **every other identity in this
table reconciles inside one table whichever way its columns were read.** Categories summing
to the row total, provinces summing to the national row — both hold on a table whose columns
have been shuffled, as long as they were shuffled consistently. `Male + Female == Total` on
all 132 cells is the only check that crosses tables, so it is the only one that would catch
a column landing in the wrong place. Malawi's panel rule, on a source that lays its panels
out differently.

Nothing is trusted positionally: the table title, the eleven category labels **in order**,
and the ten province names **in order** are all asserted before any figure is taken. A
changed category list stops the run and says so, rather than relabelling the map.

## 4. `None` is a category name, and pandas deletes it

**Third sighting of §12's Philippine trap**, after `ph` and `gy`, and Zimbabwe is the worst
case of the three by share. ZIMSTAT's no-religion cell is the literal string `None`.
Default `read_csv` parsing turns it into `NaN`; it then fails to resolve in the taxonomy;
and every reader that drops unresolved rows — which is all of them, correctly — removes
**1,255,578 people, 8.3% of Zimbabwe**, with no error, no warning and no count anywhere.

`keep_default_na=False, na_values=[""]` is load-bearing on this country. `_zw_counts` also
**asserts the category is present** after reading, rather than trusting the flags to have
been kept through a future edit — which is the cheap version of the lesson Guyana taught,
where the checkers all read the file correctly and the one reader without the flags was the
one that mattered.

## 5. What the map shows

**Zimbabwe is two Christianities and the boundary is town against country.**

| | Apostolic Sect | Pentecost |
|---|---|---|
| Mashonaland Central | **53.3%** | 10.5% |
| Manicaland | 50.2% | 13.4% |
| **Harare** | 27.2% | **28.5%** |
| **Bulawayo** | 21.2% | **24.9%** |

The Vapostori are the Shona north and east — Manicaland is Johane Marange's own country —
and the Pentecostals (ZAOGA, the Apostolic Faith Mission, the newer prophetic ministries)
are the exact inverse. Between them they are 57% of the country.

**Matabeleland is the different half of Zimbabwe in nearly every column.** The two provinces
are 129,197 km², a third of the country, holding 1.59M people:

* **lowest Apostolic** — 34.3% North, 32.4% South, against 40.3% nationally;
* **highest `Other Christian`** — 16.1% and 13.2% against 7.8%, which is the Brethren in Christ and Seventh-day Adventist mission field;
* **highest `None`** — Matabeleland South at 13.5% against 4.5% in Manicaland.

**`None` is not an urban figure**, which is worth not mis-reading: Harare is 7.1% and
Bulawayo 7.8%, both below the national 8.3%, while the rural west is 12–13.5%. Benin's
`Aucune` warning (`sources/bj.md` §7) is the obvious thing to reach for — unaffiliated
traditional practice reported as no religion — and the evidence here is weaker than Benin's,
because `African Tradition` and `None` do not track each other cleanly: Mashonaland Central
is high on both, but Matabeleland South is highest on `None` and middling on tradition. It
is a possibility this table cannot settle, and it is left as one.

**African traditional religion is 5.0% and is a floor**, for §11b's continental reason — the
box is exclusive of the Christian ones, and Shona and Ndebele practice commonly accompanies
church membership. Zimbabwe is a sharper case than most because the Apostolic churches grew
out of exactly that overlap and are counted separately at eight times the size.

**The Protestant map is the mission-station map**, still: Bulawayo 20.2% (London Missionary
Society, from 1859) and Harare 17.9% against 7.4% in Mashonaland Central.

### The Jewish figure is almost certainly not what a reader will assume

6,845 people, 0.05%. Zimbabwe's historic Ashkenazi community — Harare and Bulawayo
synagogues, a few thousand at its 1960s peak — has largely emigrated and now numbers in the
hundreds. The census figure is an order of magnitude larger and **its geography is wrong for
that community**: it peaks in Midlands, Masvingo and Manicaland, rural provinces, not in the
two cities.

The likeliest reading is that it counts the **Lemba**, who claim Judaic descent and observe
dietary and circumcision laws, together with members of Judaising churches. The census does
not say, nothing here resolves it, and it is recorded in `taxonomy/zw2022.py` per §2.4 so a
source that does resolve it is a lookup rather than an investigation.

## 6. Boundaries and placement

`sources/zw_geo.md` has the file-level record. Two things worth having here:

**The join is the easiest on the map and the check is still free.** Ten names agree
character for character. The independent check is Malawi's and it holds exactly: ZIMSTAT's
print order (`ZW01`..`ZW10`, minted from the printed row) reproduces COD's `adm1_pcode`
order (`ZW10`..`ZW19`, an attribute) on all ten. Two numberings with different origins
agreeing is evidence; Benin is where the same check failed and why the minted id is
deliberately not shaped like the p-code.

**Zimbabwe is Benin's lesson with the answer reversed.** `bj_grid.py` found that a
Kontur/census ratio band could not distinguish a right join from a shuffled one across
Benin's 77 similar-sized communes, while the correlation could. Here it is the other way
round, and both halves were measured rather than assumed:

* **the band is tight and discriminating** — 0.83× to 1.11× across ten very uneven provinces, because the grid's vintage is one year off the census rather than ten; a shuffle puts a median 4 of 10 outside it and only 0.6% of shuffles pass cleanly;
* **the correlation is weak** — r = 0.9790 against a best of 0.9758 over 2,000 shuffles, because ten log-populations of similar size correlate by luck.

**The transferable form: measure both, use whichever the country's shape makes
discriminating, and say in the file which one it was.**

## 7. Ethics (§14)

Nothing here needs a §14 conversation. ZIMSTAT published this table itself, at this
geography, and no group is drawn finer than the office drew it — 1.5M-person provinces are
about as far from a targeting resolution as this map gets.

The one category that could carry a sensitivity is the Jewish cell (§5), and the treatment
is the conservative one: the number is drawn as published, the note says plainly that it
probably counts the Lemba rather than the Ashkenazi community, and **no attempt is made to
reassign it** — that would be inventing a magnitude (§14.4) about a small group's identity,
which is exactly the operation §14.5 restricts.

`African Tradition` is *under*-counted by the question's design and is documented as a
floor. The 2022 census is three years old at the time of writing, the newest African source
on this map.

## 8. Not done

* **There is no finer geography, from any source.** §2. If ZIMSTAT ever publishes a religion thematic report or a district cross-tabulation, that is one join away — `zw_geo.py` already reads a bundle containing ADM2 (districts) and ADM3 (wards).
* **The Vapostori cannot be split.** One cell for dozens of distinct churches — the African Apostolic Church, Johane Masowe eChishanu, the Gospel of God Church and many more. This is the largest single undifferentiated cell on the map after India's and Germany's, and the one most worth opening.
* **No branch for Protestant or Pentecostal**, and none inferred: the census gives none, and Zimbabwe's Protestants sit in five different places on the tree.
* **The 2012 census is not used.** It asked religion with a comparable list and would give a change-over-time reading, which is not something this map does (§13).
