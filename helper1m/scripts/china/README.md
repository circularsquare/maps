# China — helper1m fetcher

Four admin levels, township at the bottom: province / prefecture / county /
township, 31 / 365 / 2,861 / 43,655 units.

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

Vintage is roughly 2016–2018. Sichuan still has 4,754 townships here, before the
2019–20 merger that cut it to about 3,100.

## Population

No open source publishes the township counts as a table. They are printed in
《中国人口普查分乡、镇、街道资料—2020》(China Statistics Press, 2022); the Excel
transcriptions that circulate are all behind Baidu Netdisk.

So the counts are recovered from a grid instead. ASPECT (Ju et al. 2025,
*Scientific Data*, CC BY 4.0) takes those printed township counts and spreads
each one over 100 m cells by dasymetric mapping. That spreading is
mass-preserving, so summing the grid back over a township returns the township's
census count. `zonal_pop.py` does the sum.

This avoids joining 43,655 Chinese name tuples to a book we cannot download, and
it handles the boundary-vintage gap gracefully: where our 2018 townships differ
from the 2020 ones, people are counted where the grid says they live rather than
being dropped.

### The units trap

ASPECT documents its values as persons per hectare, but the grid is EPSG:4326
with square 0.00089832° cells whose true ground area shrinks with cos(latitude).
Weighting by true cell area gives a national total of 1,187 M against a census
1,411 M — short by exactly the population-weighted mean of 1/cos(latitude). The
values are persons per *nominal* hectare, one per cell, so the recovery is a
plain sum. `TRUE_CELL_AREA` in `zonal_pop.py` keeps the other reading available
for anyone re-checking this.

### Second year

`census_county_2010-2020_v1.csv` (Dong & Wang, github.com/leiii/census) has
county-level 2010 and 2020 counts on harmonised boundaries, with official codes.
Township 2010 is planned as township 2020 scaled by its county's 2010/2020
ratio — the same trick India uses for subdistricts. Not built yet.

## Order to run

```
python prep_boundaries.py     # ~5 min, writes boundaries/adm{1,2,3,4}.gpkg + units.csv
python zonal_pop.py           # ~90 s, needs the 10.8 GB ASPECT tif extracted
python validate_provinces.py  # instant, compares against the 2020 census bulletin
```

The ASPECT raster is `data/asia1m/china/aspect_population_total_pop.zip` (488 MB,
figshare doi:10.6084/m9.figshare.27323106). Unzip it to
`aspect_population_total_pop.tif` (10.8 GB) beside the zip before running
`zonal_pop.py`; the tif can be deleted afterwards.
