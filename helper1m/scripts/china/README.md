# China — helper1m fetcher

Four admin levels, township at the bottom: province / prefecture / county /
township, 31 / 365 / 2,861 / 43,655 units, mainland only.

## Boundaries

Everything comes from one file, `data/asia1m/china/xiangzhen.shp` (43,655
township polygons, EPSG:4326, dated April 2018). It carries Chinese names for
all four levels and nothing else — no codes, no population. `prep_boundaries.py`
dissolves it upward, so the four levels nest exactly and populations roll up
without any reconciliation step.

Codes are synthetic positional strings (2 / 4 / 6 / 9 digits, each level a
prefix of the one below). The source has no GB/T 2260 codes and nothing joins to
it by code; the Chinese name tuple is the only real key and it is kept in the
output as `name_cn`.

Latin names come from `names.py`: pinyin for the proper-noun stem plus the
English division word, so 陈州回族街道 reads as "Chenzhou Huizu Subdistrict".
Ethnonyms are split off as their own words from a list of the 55 groups; the
first version ran them together ("Guangxizhuangzu"). After any change to
`names.py`, `patch_names.py` rewrites the names in the built gpkgs in a minute or
two, instead of re-running the six-minute dissolve. Characters with two readings
take pypinyin's guess, which is sometimes wrong for places — 长阳 comes out
Zhangyang.

Vintage is mid-2014 — 昌都地区 is still a 地区 while 日喀则地区 already is not.
Sichuan still has 4,754 townships, before the 2019–20 merger that cut it to about
3,100, and urban districts are smaller than their 2020 selves.

`prep_boundaries.py` applies the few changes since then that matter at adm2,
because a county under the wrong prefecture books its whole population against
the wrong unit and nothing at province or county level can see it. Six counties
move to the prefecture that now holds them — 简阳市 to 成都市, 公主岭市 to 长春市,
寿县 to 淮南市, 枞阳县 to 铜陵市, and both of 莱芜市's districts to 济南市, which
dissolves Laiwu as the 2019 change did. Six prefectures renamed from 地区 to 市
(Tibet's 昌都, 那曲, 山南, 林芝; Xinjiang's 吐鲁番, 哈密) get their current names;
their counties were already grouped correctly.

## Population, 2020

No open source publishes the township counts as a table. They are printed in
《中国人口普查分乡、镇、街道资料—2020》(China Statistics Press, 2022); the Excel
transcriptions that circulate are all behind Baidu Netdisk.

So the counts are recovered from a grid instead. ASPECT (Ju et al. 2025,
*Scientific Data*, CC BY 4.0) takes those printed township counts and spreads
each one over 100 m cells by dasymetric mapping. That spreading is
mass-preserving, so summing the grid back over a township returns the township's
census count. `zonal_pop.py` does the sum, in row blocks, in about 90 seconds.

This avoids joining 43,655 Chinese name tuples to a book we cannot download, and
it handles the boundary-vintage gap gracefully: where our 2018 townships differ
from the 2020 ones, people are counted where the grid says they live rather than
being dropped.

### The units trap

ASPECT documents its values as persons per hectare, but the grid is EPSG:4326
with square 0.00089832° cells whose true ground area shrinks with cos(latitude).
Weighting by true cell area gives a national total of 1,187 M against a census
1,411 M — short by exactly the population-weighted mean of 1/cos(latitude), which
reads like missing data rather than a units error. The values are persons per
*nominal* hectare, one per cell, so the recovery is a plain sum. `TRUE_CELL_AREA`
in `zonal_pop.py` keeps the other reading available for anyone re-checking this.

### How good the raw recovery is

Before any correction, 25 of the 31 provinces land within 0.25% of the published
2020 census and the national total is 0.18% high. Below the province the errors
are larger and come in matched pairs: an urban core district short and its
neighbouring rural county long, because those districts annexed populated fringe
between 2018 and 2020 and the grid puts people where they physically live.
Changsha County is 18% over while Tianxin District next door is 23% under.

Two provinces are off by more than boundary vintage explains — Anhui +4.5% and
Sichuan +3.4%. That excess is in the ASPECT grid, not in our polygons: summing
the same grid over the independent OCHA 2020 province outlines puts Anhui 4.8%
over as well.

`fetch.py` therefore puts every county it can match onto its own published census
figure (`ANCHOR_TO_CENSUS`), and lets the counties with no figure absorb what is
left of their province. The published province total minus the published figures
of the anchored counties is, by construction, what the rest of that province
holds — so province and national totals stay exact while the county level is
right wherever the census can say so.

That replaced a blanket province scaling, which multiplied every township in a
province by a single factor. It is the wrong instrument when a province's excess
sits in one or two broken counties: Anhui's is mostly Chuzhou's urban core and
Baohe, and the old scaling paid for them by shaving 4.3% off all 88 of Anhui's
other counties, which were already right. The county is the level these maps get
assembled at, so it is the level that has to be accurate.

Because anchoring uses the county census figures, scoring the built output
against them would only prove the anchoring ran. `validate_counties.py` therefore
scores the **raw** zonal sums instead — the recovery underneath, which is what
decides the counties anchoring cannot reach. It compares them as groups, because
a merged panel row covers several of our counties at once and only the group can
honestly be held against it. The median group is 0.5% out, 94% within 5% and 97%
within 10%, and it now also prints the counties it cannot check at all rather
than dropping them silently.

At 1.41 billion the country needs about 1,410 regions of a million. The median
township holds 18,500 people, so a region is around 54 of them and about 2.6
counties — the county level is the one to assemble from, with townships for
splitting a large county or following a line precisely.

## Population, 2010

The other half of the trend. `census_county_2010-2020_v1.csv` (Dong & Wang, github.com/leiii/census) has
county-level 2010 and 2020 counts on harmonised boundaries. Each township is
scaled by its county's ratio from that panel — the uniform-within-parent
assumption India's subdistricts also use.

**It is context, not arithmetic.** The viewer's current-year estimate is a line
through the last two years it holds, 2020 and 2024; 2010 only fills the history
table and raises the two "direction changed" flags. It is also the weakest column
in the build, being one county ratio applied uniformly to every township inside.
Worth reading, not worth more work — the figure that has to be right is the 2020
county total, which is measured rather than inferred.

Matching our counties to the panel by Chinese name gets 2,593 of 2,861. Misses
are mostly 2015–16 renames, 崇明县 becoming 崇明区 and 腾冲县 becoming 腾冲市, so
the name stem is tried after the full name; a match whose 2020 population
disagrees with our own sum by more than 25% is thrown out, which is what stops a
stem match picking a same-named neighbour.

Two further guards stop one panel row backing two of our counties. A key has to
be unique on our side as well as the panel's, and a row one of our counties has
already taken cannot back another. Jiangsu has a 鼓楼区 in both Nanjing and
Xuzhou: Nanjing's matches on the prefecture pass, and without the second guard
the Xuzhou one then claimed that same Nanjing row on the looser prov+name pass.
Neither looks wrong against that row on its own — only the pair does — so it
survived every per-county check until the validation started comparing groups.

### Development zones

139 panel rows are not one county but several joined with `+`, because the census
reports a development zone separately from the district it sits in:
蜀山区+高新区+经开区, coded 340104;340171;340172. Those zones are not
administrative divisions. The Ministry of Civil Affairs list for Hefei is four
districts, four counties and one county-level city, and 合肥高新区 and 合肥经开区
are management committees that 代管 subdistricts whose land stays legally
Shushan's. One of our polygons therefore covers the whole merged row, and the
row's ratio is the right one for it.

These rows cover 171 of our counties and 137 M people, and until they were
handled every one of them fell back to the province residual. Shushan came out
growing 27% across the decade where the census says its ground grew 65%.

The population check for a merged row is made against the group rather than the
single county: everything of ours landing on one row is summed and that sum is
tested against the row. Hangzhou's 上城区 is 9% of its four-district row and
would fail alone, while the group is within 1%. 116 of the 132 rows pass, and the
rest fall back to the province residual as before.

This is also why a district can read far above the figure printed against its
name. Hefei books 1.29 M people under four development zones, so the census row
for 蜀山区 is 1,047,150 while the district's own 653 km² holds 1,874,930.

**Do not average the matched ratios to fill the rest.** The counties that fail
to match are overwhelmingly the ones renamed when they became urban districts,
which are the fastest-growing ones, so their average understates growth badly —
the first attempt came out 46 M above the 2010 census. Unmatched counties instead
take the province residual: what the panel says is left in that province once the
matched counties are accounted for. Each province is then normalised onto its
published 2010/2020 growth, which fixes the level while keeping the
county-to-county variation. Xinjiang needs the largest correction, 0.89.

## Population, 2024

The viewer extrapolates from the last two years it has. With only 2010 and 2020
it carried a decade of growth forward and reached 1,456 M for 2026, while
China's population peaked around 2021 and was 1,405 M at the end of 2025 — so
every assembled region read about 3.6% high, worsening each year.

So each township is also scaled forward by its province's year-end population
from the China Statistical Yearbook 2025, table 2-5 (`yearbook_provinces.csv`).
Both the 2020 and 2024 columns are taken from that one table, because the
yearbook's year-end estimates are a different series from the November census
counts and a ratio only means anything inside one series. 11 provinces are up
since 2020 and 20 are down.

That table is published as an image, so the figures are transcribed by hand.
What makes the transcription checkable is the table's own footnote: the
national total counts servicemen and the provincial rows do not, and the 31 rows
sum to 140,628万 against a national 140,828万 — exactly the 200万 difference.

No township data exists after 2020, so every township in a province shares its
province's recent rate and the post-2020 trend carries no within-province
detail. The 2010-2020 step is the one that shows how a place was actually
moving, and it stays visible in the history table.

## Known gaps

261 townships come out zero. They are ASPECT's own missing-data units and are
almost all forestry stations, state farms, 经营所 and industrial parks —
special-purpose units holding a few hundred thousand people in total. 80 of them
are in Heilongjiang, whose provincial total is still within 0.12%.

## Order to run

```
python prep_boundaries.py     # ~6 min, writes boundaries/adm{1,2,3,4}.gpkg + units.csv
python zonal_pop.py           # ~90 s, needs the ASPECT tif extracted
python fetch.py               # instant, writes population.csv
python validate_provinces.py  # instant, both years against the census bulletins
python validate_counties.py   # instant, the test fetch.py cannot rig
python ../build_country.py china
```

The ASPECT raster is `data/asia1m/china/aspect_population_total_pop.zip` (488 MB,
figshare doi:10.6084/m9.figshare.27323106). Unzip it to
`aspect_population_total_pop.tif` (10.8 GB) beside the zip before running
`zonal_pop.py`; the tif can be deleted afterwards.

The township level is written as one geojson per province under
`countries/china/adm4/`, because 43,655 features in a single fetch is not
something the viewer can hold. It loads the provinces on screen, up to four.
