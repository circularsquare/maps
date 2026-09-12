# Fiji — FBoS, 2007 Census of Population and Housing, Table P01-3

Wired 2026-09-08. 837,271 people, 15 provinces, 23 drawn categories, **99.89% of the country
drawn**.

| | |
|---|---|
| source | Fiji Bureau of Statistics, **2007 Census of Population and Housing**, Table P01-3 — *Relationship, Ethnicity, and Religion by Province of Enumeration* |
| basis | `self_id`, **whole population** — no age floor |
| geography | **15 provinces** — 56,000 people each (14 provinces plus the Rotuma dependency) |
| categories | **23 drawn** — eighteen named Christian bodies, plus Hindu, Sikh, Moslem, other religion and none |
| drawn | **836,376 of 837,271, 99.89%**; the 895 not drawn are a §3.5 residual the table does not print |
| licence | published census table, open download, no account |

**Fiji is the most religiously plural country in the Pacific and the only one on this map that
needs more than a Christian palette:** 64.9% Christian, **27.7% Hindu, 6.3% Muslim**, and 2,548
Sikhs with a printed line of their own.

---

## 1. Access — an open WordPress plugin API, and the id is the catalogue

`statsfiji.gov.fj` is WordPress. §11aa recorded it as one of the Pacific offices whose
`wp/v2/search` is *"disabled or 404"*; **that is wrong and it was one GET away** — the REST API
is open and `wp/v2/search?search=religion` returns the 2007 religion tables directly.

The files are not in `wp/v2/media` (which returns 0 for every census query). They live in a
**WP File Download** plugin under a `wpfd_file` post type, and the plugin's own AJAX route is
unauthenticated:

```
GET /wp-admin/admin-ajax.php?juwpfisadmin=false&action=wpfd&task=category.getFiles&id=<cat>
GET /wp-admin/admin-ajax.php?juwpfisadmin=false&action=wpfd&task=files.getFiles&id=<cat>
GET /wp-admin/admin-ajax.php?juwpfisadmin=false&action=wpfd&task=file.download
        &wpfd_category_id=<cat>&wpfd_file_id=<id>
```

`category.getFiles` returns the category's `name`, `parent` and `count`, so **sweeping the
integer id space maps the office's whole library** — the CMS download-id sweep again, in a
politer form because this one hands over JSON. `task=categories.getCategories` returns HTTP
500, which is the tell that the tree has to be walked by id rather than listed.

What the sweep finds, for the census:

```
  54  Census and Surveys
    114  PHC 2007
      115  Tables
        116  02_PLACE-OF-ENUMERATION    9 files   (urban/rural, NOT finer geography)
        117  01_PROVINCE-OF-ENUMERATION 7 files   <- file 686 is Table P01-3
        118  03_POPULATION-BY-FIVE-YEAR-AGE-GROUP  7 files
      119  Report                       4 files
      120  Provincial Profiles         14 files
    121  PHC 2017                       3 files   <- releases 1-3, and NO religion
```

## 2. The 2017 census does not publish religion, so 2007 is current

All three 2017 releases were downloaded and searched. **Release 3 is 484 pages and mentions
religion zero times**; releases 1 and 2 likewise. The oracle's row for Fiji is 2007 and that is
correct — 2007 is the most recent Fijian census with a published religion table, eighteen years
old at the time of writing.

## 3. THE TIER IS A TRADE-OFF, AND IT IS THE WHOLE DESIGN DECISION

Fiji publishes religion **twice**, from the same census, and the two are exclusive:

| | geography | units | people/unit | categories |
|---|---|---:|---:|---:|
| **drawn** — FBoS Table P01-3 | province | **15** | 56,000 | **23** |
| not used — SPC PopGIS | tikina | **86** | 9,700 | **6** |

PopGIS is **5.7x finer** and collapses all eighteen Christian bodies into a single `Christians`
column. Its six categories are `Christians / Hindu / Muslim / other religion / no religion /
not stated`.

**Taking the finer geography would delete Methodist**, which is 290,555 people — 34.7% of Fiji
and the largest single body in the country — and would leave the most plural country in the
Pacific looking like every other Pacific census on this map. The categories are the reason the
country is worth drawing at all, so the categories win.

This is §9k's *"has finer geography"* versus *"asks a better question"*, decided the opposite
way from Peru (§9bc) — and the difference is that **Peru had both and Fiji genuinely does
not**. `todo.txt` asked for Fiji as *"provinces, Methodist / Hindu / Muslim"*, which is the
same call.

**What it costs, stated:** the sugar belt is a tikina-scale pattern and Fiji is drawn at
province scale, so Ba province reads as 39.7% Hindu overall rather than showing which cane
districts inside it are 70% and which coastal villages are 5%. If FBoS ever publishes the
denominations at tikina, this file changes one identifier.

## 4. What is in the table

```
  Methodist                 290,555   34.70%      Hindu           232,103   27.72%
  Catholic                   76,603    9.15%      Moslem           52,594    6.28%
  Assembly of God            47,873    5.72%      Sikh              2,548    0.30%
  Seventh Day Adventist      32,370    3.87%      Other religion    1,294    0.15%
  Other Christian            17,019    2.03%      No religion       4,249    0.51%
  Penticostal                15,326    1.83%
  Christ Mission Fellowship  14,180    1.69%      ---------------------------------
  All Nation Christian       13,294    1.59%      Christian       543,588   64.92%
  Jehovah's Witnesses         8,450    1.01%        (nested universe, NOT drawn)
  Anglican                    6,328    0.76%      Not stated          895    0.11%
  Latter Day Saints           5,126    0.61%        (residual, NOT drawn)
  Apostolic                   5,089    0.61%      Total           837,271
  Presbyterian                2,907    0.35%
  Gospel                      2,835    0.34%
  Baptist                     1,772    0.21%
  United Pentecostal          1,361    0.16%
  Church of Christ            1,356    0.16%
  Salvation Army              1,144    0.14%
```

> **`Christian` IS A NESTED UNIVERSE AND DRAWING IT WOULD DOUBLE 65% OF THE COUNTRY.** The
> eighteen denominations sum to 543,588 exactly, on all fifteen provinces. `taxonomy/fj2007.py`
> has it in `EXCLUDED` beside `Total`, and `sources/fj.py` asserts the identity that makes that
> necessary.

**Two of the eighteen are Fijian foundations rather than imported missions.** *Christian
Mission Fellowship International* was started in Suva in 1990 by Suliasi Kurulo and now plants
churches in a hundred countries; *All Nations Christian Fellowship* is a Suva church that
describes itself as non-denominational. Both are in `taxonomy/fj2007.py`'s REVIEW, and the
second is the least certain call in the file — the census truncates every label to fourteen
characters, so `All Nation Chr` is a probable rather than proven identification.

**The census names Sikhs separately**, which very few censuses anywhere do. The Punjabi
minority inside the indenture-era migration was counted apart from the colonial period on.

## 5. The witness — SPC PopGIS, which shares no code path with the PDF

`fiji.popgis.spc.int` is a **GeoClip Observatory (PopGIS 3)** instance run by the Pacific
Community over FBoS microdata. Not used for the counts (§3) but ideal as a check: an SPC
database against a table typeset in 2008.

Its API, found by grepping `js/libs/gco5/gc_core.js` after `GC_init.php` answered:

```
GC_init.php?obs=main&lang=en             the whole configuration: 12 geographic levels,
                                         65 datasets, 3 indicator trees
GC_listIndics.php?tree=A00&theme=d9_religion&lang=en      41 religion indicators
GC_coldata.php?view=<map2|map3|map6>&dataset=d9_religion&indic=<i>&vars=<i>    the values
GC_refdata.php?nivgeo=<tid|pid>&view=<v>&extent=cid&lang=en                    the territories
```

**Three of the six figures come back exactly, and the other three disagree by 0.23% in a way
that is internally consistent:**

```
  total            837,271  ==  837,271   OK
  Hindu            232,103  ==  232,103   OK
  Moslem            52,594  ==   52,594   OK
  Christians       545,517  vs  543,588   +1,929
  no religion        2,295  vs    4,249   -1,954
  not stated            25  vs      (not printed)
  other religion     4,737  ==  Sikh 2,548 + Other 1,294 + the 895 residual   OK
```

**That last line is the useful one.** Table P01-3 states a total of 837,271 and prints six
top-level rows summing to 836,376, leaving **895 people unaccounted on the page**. PopGIS puts
exactly those 895 into `other religion` — which is where a residual goes and not where a
religion goes — and that is what identifies them as *not stated* rather than as a category
FBoS forgot to print. They are carried as `Not stated` and not drawn (§3.5).

The remaining 1,929 (0.23% of Fiji) sit on the Christian/no-religion boundary and are a coding
difference in a residual between two publications of one census, not a disagreement about the
census. Both are reported; the printed table is drawn.

## 6. The join is free, and for once that is not a trap

COD-AB Fiji is **sourced from FBoS's own POPGIS** — the HDX description says so — so it carries
`FBOS_PID`, the office's province id, beside the OCHA pcode. That is the same identifier the
census tabulates on and the same one SPC's PopGIS returns as `codgeo`. Three sources, one id
space, because two of them are the same office. 15 units, ids 1–15, names agreeing outright.

Unlike Nicaragua (§9ay, join on name because the codes were renumbered) and Peru (§9bc, join on
code because the names repeat across provinces), Fiji has no join problem. `sources/fj_geo.py`
still checks it both ways, because **a join that cannot fail is exactly the one nobody checks.**

**And it deliberately asserts no geographic witness.** Peru's lesson was that a witness naming
a region can fire on a correct join, and its replacement — spatial smoothness calibrated
against random re-pairings — needs enough units to calibrate on. Fifteen units scattered over
500 km of ocean cannot. The honest move is to say so and let the check with power do the work:
see §8.

## 7. THE ANTIMERIDIAN, WHICH BROKE TWO THINGS THAT LOOK NOTHING ALIKE

**Fiji is the first country on this map that straddles 180°**, and it is worth writing down
because everything about it fails silently.

**First: reprojecting the provinces to EPSG:4326 tears three of them.**

```
  Cakaudrove   lon -180.000 .. 180.000   span 360.000
  Lau          lon -179.886 .. 179.953   span 359.838
  Macuata      lon -180.000 .. 180.000   span 360.000
```

That is not a wide province, it is a polygon wrapped the wrong way round the globe. The file
opens, the feature count is right, the names are right, and a point-in-polygon join against it
is nonsense while every total still reconciles.

**Second, and worse because it is upstream: nine of Kontur's own hexes are stored torn**, with
an x-span of 40,075,017 m — the entire width of the EPSG:3857 plane. Their centroids therefore
compute to longitude ≈ 0, and this extract put six of them at 97°W, 26°E, 133°W, 76°W, 54°E and
94°W — in the Atlantic, the Sahara and the Indian Ocean, at Fiji's latitude. A centroid-in-
polygon join drops them without a word.

> **PROJECTING INTO A PACIFIC CRS DOES NOT FIX EITHER OF THESE — IT RELOCATES THE PROBLEM.**
> The first attempt reprojected everything into EPSG:3832 (PDC Mercator, central meridian
> 150°E) and asserted the result was Fiji-sized. **It fired, correctly.** pyproj does not wrap
> longitude: a point at 179.9°W is 329.5 degrees *west* of the 150°E origin as far as the
> transform is concerned, so it lands about 36,000 km off the map instead of 30 km east of
> Taveuni. The hex centroids came out spanning 28,670 km.

**What works is doing the arithmetic in degrees.** Repair the torn hexes in the tiling CRS by
shifting their negative-x vertices one plane width east; then take everything to EPSG:4326, add
360 to every negative longitude so Fiji is a continuous 176–181 band, and join there.
`sources/fj_grid.py` does that and asserts the country comes out about 5° wide — the assertion
that caught the first attempt.

Seven repaired cells would tear again when `scatter.py` reprojects the placement layer back to
4326, and are dropped: **385 people, 0.044% of the placement weights and no count at all.** The
strip either side of 180° in Cakaudrove is weighted very slightly light.

The boundaries are stored in **EPSG:3832**, not 4326, so nothing downstream can pick up a torn
polygon by accident. Natural Earth, checked separately, splits Fiji into 44 untorn island
polygons, so `country_shapes.py` and Auto are fine.

## 8. Placement — Kontur, and the check that actually has power

`sources/fj_grid.py`, 9,945 Kontur 400 m hexagons. Fiji needs a population grid because **its
units are archipelagos, not areas**: Lau is sixty-odd islands across 500 km of ocean holding
10,683 people; Cakaudrove is half of Vanua Levu plus Taveuni plus Rabi and Kioa. An equal share
per polygon would put dots in open sea and weight a copra island like a Suva suburb.

**Counts 2007, grid 2023 — a sixteen-year vintage gap**, second only to Nicaragua's eighteen.
It moves dots within a province, never between provinces.

```
  band          all 15 inside a factor of 3, and here the band IS evidence — unlike Peru's
                1,873 districts, no Fijian province is small enough for Kontur to be noise on
  correlation   r = 0.9896, against a best of 0.8884 over 2,000 random pairings (0 reach it)
```

**Six percent of Kontur's people fall outside every province, which looks alarming and is
not.** Measured against the province outline, **55,169 of those 56,337 — 98% — are within 500 m
of a boundary**: coastal cells whose centroid falls just seaward of a detailed island coastline
on a 400 m grid, which is what an archipelago costs. Only 457 people sit more than 5 km out, on
islets east of Taveuni that COD does not draw.

## 9. What the map shows

**The two halves of Fiji are not mixed, they are different islands.** Methodist is **83.7% of
Lau and 81.7% of Kadavu** — the outer eastern islands, where the Methodist mission landed in
the 1830s and where the chiefly system and the church grew together. Hindu is **44.3% of
Macuata and 39.7% of Ba** — northern Vanua Levu and western Viti Levu, which is where the cane
is and where the indentured labourers were sent from 1879. The plantation economy is legible
on the map.

**Methodism here is bigger than in any country that invented it**: 34.7%, the highest Methodist
share anywhere on this map.

**No religion is 0.51%**, one of the lowest figures on the map.

## 10. What else is on these servers, unused

- **PopGIS serves the 2017 census too** (`tree A07`, 508 indicators) — but with no religion
  theme; the religion dataset `d9_religion` exists only under `A00`, the 2007 tree.
- PopGIS's geography goes down to **wards and 2017 enumeration areas**, and its religion
  indicators cover only `tid`, `pid` and `dvid`. So the six-category tikina table in §3 is the
  floor there, not a choice.
- The 2007 table set also publishes religion **by urban/rural** and **by five-year age group**
  (categories 116 and 118), and Table P01-3's own page carries eight further panels crossing
  religion with ethnicity and with urban/rural — `P01-3F` (Fijian), `P01-3I` (Indian) and the
  rural/urban splits of each. **Fiji can therefore cross religion with ethnicity at province
  level**, which would say directly what §9's cane-belt reading infers. Not read here.
- The 14 **Provincial Profiles** (category 120) were not opened.
