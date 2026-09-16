# American Samoa — 2015 Household Income and Expenditure Survey, religion by county

**Drawn 2026-09-15** (session `d743fc47-as`). 10 units (the nine counties of Tutuila and Aunu'u,
and Manu'a), 16 categories on 15 nodes, the survey's shares laid on the 2020 census count of
49,710. Every row `modelled`. 43 dots and 8 rings at 1:1,000; 1 dot and 14 rings at 1:10,000.

- `sources/as.py` -> `data/normalized/as.csv` (the report in `data/raw/as/`, pinned)
- `sources/as_geo.py` -> `data/geo/as/as_counties.gpkg`, `as_hexes.gpkg`, `as_lookup.csv`
  (TIGER/Line 2020 county subdivisions, the 2020 census table, Kontur 400 m)
- `taxonomy/as2015.py` -> the mapping and `OWN_GEOGRAPHY`; `countries/as.py` -> the entry and the
  construction; `taxonomy/branches.py` gains `christianity.reformed.congregational.cccas` and
  `other.as`
- sources.md **§as-2026-09-15** is the summary; **§scout-2026-09-14-asia-oceania** was the scout's.

```
python sources/as.py     --fetch
python sources/as_geo.py --fetch
```

## 1. Why a survey, and what was looked at

The census cannot ask: the 2010 American Samoa summary file documentation says the Census Bureau
"cannot collect information on religion", and the 2020 Island Areas forms have no religion item
(the scout). UNSD table 28 has no American Samoa row (`tools/oracle.py`, 2026-09-15).

| release | religion | tier |
|---|---|---|
| **2015 HIES report (Department of Commerce, Statistics Division), section 11** | **Table 1.6 religion x county**; 2.3 x age, 3.3 x birthplace, 4.3 x citizenship and sex, 5.3 x education, 6.2 x work, 7.3 x occupation, 8.3 x industry, 9.3 x income; Table A county percents for three churches | 10 counties |
| 2015 HIES, text p.23 and Fig. 6 | the same three churches by county, as a chart | 10 counties |

**Not found or not opened, so not negatives:** the 2005 and 1995 HIES reports, which the 2015
report says used the same cross-tabulations (one web search, 2026-09-15, found neither); SPC's
American Samoa page (`sdd.spc.int/as`, 403 to WebFetch), which lists a Statistical Yearbook
2023-2024; SPC microdata catalogue 664 (403 to the scout); any HIES after 2015 (one search, none
found). The Census Bureau's data API now answers "Missing Key" and was not used; the county counts
come from the bureau's published CSV.

## 2. The file

| file | url | bytes | digest |
|---|---|---:|---|
| `as_hies2015_report.pdf` | `doi.gov/sites/default/files/uploads/american-samoa-2015-household-income-and-expenditure-report.pdf` | 6,856,994 | `K3PPMVA4Z24KN3ZPHPTQRFYCFTEAYLWS` |
| `tl_2020_60_cousub.zip` | `www2.census.gov/geo/tiger/TIGER2020/COUSUB/` | 31,892 | `ARX2J3O5XCT4UY34STD6DVTJPC4CM737` |
| `american-samoa-phc-table01.csv` | `www2.census.gov/programs-surveys/decennial/2020/data/island-areas/american-samoa/population-and-housing-unit-counts/` | 1,145 | `UGGNZBLDTO7MQQBEAJCHLPK3QGZ3GFU6` |
| `kontur_population_AS_20231101.gpkg.gz` | Kontur S3 | 18,701 | not pinned |

189 pages, a text layer on every page, one number per line in the text layer.

## 3. The checks (`sources/as.py::check`)

| check | result |
|---|---|
| Table 1.6 header rebuilt from its split names (`Maopu-` + `tasi`) | the ten counties in order |
| Table 1.6 parsed = transcription | 17 rows x 11 |
| every cell a whole multiple of 5.99668 | worst 0.478 of a person; 9,578 sampled persons, the counties sum to the same |
| rows sum to county totals; counties sum to row totals | within 2 and 1 (rounding of weighted cells) |
| Table 1.1 county totals | equal |
| Table A percents (CCCAS, Catholic, LDS) = Table 1.6 count over total | every column, to the decimal |
| Tables 2.3, 3.3, 4.3 national column | equal; 4.3 `NR` is 0 in all three blocks; males + females within 1 |
| the county test's verdict = `as2015.OWN_GEOGRAPHY` | equal |
| (`as_geo.py`) survey county totals against the 2010 census of the same name | 0.83x to 1.37x, inside 0.70-1.50: no column is shifted |

## 4. The sample, and what the table is

Printed pp.17-18: a systematic 20 percent sample of housing units on all islands on 2000 census
geography; 1,838 of 2,098 selected units completed a form. **One weight, 5.99668**, "the average of
the sample coverage of the three districts", used for housing and population alike. Religion was
a write-in for every household member (p.23), coded by the office into 16 rows. Households only;
group quarters were not sampled.

So each cell is a whole number of sampled persons times the weight, and the smallest rows are a
handful of people: Orthodox 4, Jewish 14, Nazarene 41, Bahá'í 49.

## 5. The population base is the 2020 census, not the survey's totals

The survey weighted to 57,436, above both censuses (2010: 55,519; 2020: 49,710). Its county totals
are completion by county times one number:

| unit | survey 2015 | census 2010 | ratio | census 2020 |
|---|---:|---:|---:|---:|
| Tualauta | 19,519 | 20,858 | 0.94 | 22,827 |
| Maoputasi | 11,052 | 10,299 | 1.07 | 8,568 |
| Lealataua | 6,968 | 5,103 | **1.37** | 4,293 |
| Ituau | 5,607 | 4,676 | 1.20 | 3,431 |
| Tualatai | 3,892 | 3,561 | 1.09 | 3,010 |
| Sua | 3,274 | 3,323 | 0.99 | 2,415 |
| Vaifanua | 2,489 | 2,545 | 0.98 | 1,487 |
| Saole | 1,811 | 2,187 | **0.83** | 1,158 |
| Leasina | 1,541 | 1,807 | 0.85 | 1,689 |
| Manu'a | 1,283 | 1,143 | 1.12 | 832 |

The census shows the Eastern District falling 26% from 2010 to 2020 and Tualauta rising 9%; the
survey shows neither. So the survey gives shares and the 2020 census gives people, which is how
every household-survey build here is laid (`pr`, `uy`, `do`, `bo` on the newest count). Mixed
vintage, 2015 shares on 2020 people, under ask 003. Rose Island and Swains Island have nobody in
2020 (Swains had 17 in 2010) and are not units, so there is no `gap`.

## 6. Which categories carry their own county geography

No microdata and one round, so the split-half (§9bi/§9bl) cannot run. **The test used instead**
(`sources/as.py::geography_test`): households per county are the sampled persons over the mean
household size (5.21); under the null each household holds the category at the territory rate
(binomial); the statistic is the 2 x 10 chi-square on households, against 20,000 draws, seed
20260915. A household counts once, since it mostly shares one church, and there is no
finite-population correction for a one-in-six sample; both make it conservative. The bar is the
house 95% per category (ask 007) with no multiplicity correction (open under ask 012).

| category | sampled persons | p | drawn |
|---|---:|---:|---|
| CCCAS | 3,193 | <0.0001 | own shares |
| Catholic | 1,736 | <0.0001 | own shares |
| LDS | 1,516 | <0.0001 | own shares |
| Other religion | 530 | <0.0001 | own shares |
| Jehovah's Witness | 119 | 0.015 | own shares |
| Nazarene | 41 | 0.026 | own shares |
| Methodist | 724 | 0.041 | own shares |
| SDA | 277 | 0.041 | own shares |
| No religion | 117 | 0.094 | territory rate |
| Assembly of God | 909 | 0.130 | territory rate |
| Full Gospel | 127 | 0.151 | territory rate |
| Baptist | 129 | 0.160 | territory rate |
| Orthodox | 4 | 0.54 | territory rate |
| Bahá'í | 49 | 0.65 | territory rate |
| Pentecostal | 93 | 0.89 | territory rate |
| Jewish | 14 | 0.93 | territory rate |

A failing category shares each unit's remainder (the census count less the passing categories'
shares) in its territory-wide proportion. A plain chi-square on the same household-equivalents with
a Bonferroni correction over 16 would pass only CCCAS, Catholic, LDS and Other religion; that was
the scratch first look and is not what is drawn. Methodist and SDA sit at 0.041. Nazarene passes on
perhaps eight households in two counties.

**An outside witness for two of the passing patterns:** Table 1.5 puts 1,397 of the survey's 1,607
Tongans in Tualauta, and Tualauta is where Latter-day Saints (25.0%) and Methodists (11.0%) are
highest; Tonga is the most Latter-day Saint country in UNSD's table and mostly Methodist.

## 7. Geography

**Units: TIGER/Line 2020 county subdivisions** (public domain), 16 features. Nine counties joined by
folded name (`Ma'oputasi` to the report's `Maoputasi`), a bijection; Manu'a is the five counties of
COUNTYFP 020 dissolved, their names asserted. The polygons include territorial water (TIGER's
AWATER), so coastal hex centroids fall inside without snapping. The census table closes: counties
to districts, districts plus Rose and Swains to the territory, both years.

**Placement: Kontur AS 2023-11**, 217 hexes, 43,916 people; 3 centroids outside every unit, all
dropped (10 people, presumably Swains). Kontur is 0.883x the 2020 census; per unit 0.58x (Ituau) to
1.93x (Saole), inside the 0.50-2.00 band. It moves dots only inside a unit. Median 21 hexes per
unit, above the grid floor. `kontur_cap.py as`: no stops.

## 8. What the table shows

CCCAS 33.3%, Catholic 18.1%, LDS 15.8%, Assembly of God 9.5%, Methodist 7.6%, Other religion 5.5%,
SDA 2.9%, and eight rows under 1.4%. CCCAS is 88.3% of Manu'a (no Catholic or LDS recorded), 55.7%
of Leasina, 47.7% of Saole; it is below Catholic only in Maoputasi (25.5% against 25.9%). Catholic
30.5% of Lealataua. LDS and Methodist peak in Tualauta. Other religion 11.6% of Maoputasi, none in
Saole, Vaifanua or Manu'a; 1,943 of its 3,178 were born in American Samoa and 138 in Asia (Table
3.3).

**The CCCAS is not Samoa's CCCS.** The World Council of Churches' member page for the Congregational
Christian Church in American Samoa: an independent assembly sought from 1964, constituted in 1980,
reconciliation with the Samoan church declared in 1982. Hence its own node beside `.cccs`, the
seventh of the Pacific Congregational set, rather than a merge.

## 9. §14 was considered and no ask was filed

Ten units averaging 5,000 people, the smallest Manu'a at 832. The small religions that could locate
a few households (Jewish 14 sampled persons, Bahá'í 49, Orthodox 4) are drawn at the territory-wide
rate, so the map does not place them by county; at 1:1,000 they draw rings, not dots. The table is
a US territorial government publication. No search for restrictions on religious minorities in
American Samoa was made; this rests on the grain and on the small groups being spread flat.

## 10. Gotchas

- **The survey's county totals are not populations** (§5). Now in `playbooks/geography.md`.
- **Table 1.6's header prints split names in two rows**, and the text layer gives the first row's
  fragments before the second row; `header_counties` rejoins them.
- **Table 2.3's age bands put bare numbers (`9`, `14`) in the header**; a row reader must skip
  everything before the `Total` row.
- **Labels vary between tables:** `Baha’i` (curly) in 1.6, `Bahai` elsewhere; `Jehovah's Witness`
  split over two lines in 1.6 and cut to `Jehovah's Witnes` elsewhere; `Assembly of` / `God` in
  3.3.
- **The report calls Saole `Sa'aole` in some charts** (its own note, p.20); the tables are right.
- **The Census Bureau's API needs a key since 2026**; the `www2.census.gov` CSVs do not.

## 11. Reopen when

- The 2005 (or 1995) HIES report turns up: a second sample on the same ten columns is the split-half
  this build could not run, and would settle Methodist, SDA and Nazarene.
- A post-2015 American Samoa HIES asks religion.
- SPC catalogue 664 opens: household-level data would replace the simulation with a real test.

## 12. Terms

The HIES report is a public US territorial government document hosted by the Department of the
Interior; no licence text was seen. TIGER/Line and the census table are US federal works, public
domain. Kontur Population is CC BY 4.0.

## 13. Review, 2026-09-15 (session `d743fc47-rev11`)

Full pass. `check_md`, `built_countries --check` and `check_rollup as` are clean (49,710 modelled,
0 orphaned). The figures in `note_public` were re-derived from `as.csv` and all match: CCCAS 88.3%
of Manu'a, Catholic 30.5% of Lealataua and 25.9% of Maoputasi, Tualauta 25.0% LDS and 11.0%
Methodist, Methodist 4.2 to 7.7% elsewhere.

**One wording error in `note_public`, not changed here.** "Eight smaller answers are drawn at the
territory-wide rate" is wrong twice. Assembly of God, at 9.5%, is larger than four of the categories
drawn on their own shares (Methodist, Other religion, SDA, Jehovah's Witness). And the eight are not
drawn at the territory-wide rate: they share each county's remainder in territory-wide proportions,
as the paragraph's last sentence says. Something like "Eight other answers share what is left in
each county" fixes both. The build tail has to run again after the edit.

**The two new nodes follow precedent.** `.cccas` is the seventh national church in the Pacific
Congregational set (`.cccs`, `.cicc`, `.ekt`, `.kpc`, `.ncc`, `.niue`), and the WCC's 1980 date
makes it a separate church. `other.as` sits beside some sixty `other.<cc>` nodes. No ask.

**Other religion is probably mostly small local churches.** 1,943 of its 3,178 were born in American
Samoa and 138 in Asia (Table 3.3). `other` means "an actual religion the source did not name"
(spec §6.3a-iv), so the node is right. But the note never mentions the row, although it is the sixth
largest and 11.6% of Maoputasi. One sentence saying the report does not list what it holds would
cover it.

**The remainder construction was checked against the 2x rule and for reversals** (a scratch
script, not kept). The largest category drawn where the survey found none is 1.61x its territory
share, the five small rows in Saole. No category of any size reverses.

**On the county test's close verdicts.** The null treats households as independent. In a
systematic sample of housing units, related households in one village often share a church, and
that makes the test too easy to pass for small clustered rows. That runs the other way from the two
safe-side choices §6 lists. At 16 tests and a 5% bar, about one false pass is expected. Methodist
has the Tongan witness; SDA (p=0.041) and Nazarene (about eight households) have none. At 1:1,000,
own shares against the remainder moves under a fifth of a dot per county for SDA and less for
Nazarene, so nothing changes. Reread this if a second HIES round turns up (§11).

**Map glance** (1400x900, Tutuila): about 40 dots, all on land along the coast and across the
Tafuna plain. None in the sea, nothing blank.
