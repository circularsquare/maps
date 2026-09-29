# China — helper1m fetcher

Four admin levels, township at the bottom: province / prefecture / county /
township, 31 / 364 / 2,861 / 43,655 units, mainland only.

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

`prep_boundaries.py` corrects the source before dissolving it, in three tables:

- `SOURCE_FIXES` — plain errors in the shapefile. Three of 郴州市's counties
  (桂阳县, 永兴县, 临武县) are labelled 四川省 成都市, which drew them as part of
  Chengdu and booked 1.56 M people to Sichuan, and the XPCC city 铁门关市 is
  labelled 西藏自治区. Found by checking which province's census polygons each
  township's people sit in; a sweep at prefecture level found nothing else.
- `COUNTY_MOVES` — counties that have since moved prefecture, because a county
  under the wrong prefecture books its whole population against the wrong unit
  and nothing at province or county level can see it. 简阳市 to 成都市, 公主岭市
  to 长春市, 寿县 to 淮南市, 枞阳县 to 铜陵市, and both of 莱芜市's districts to
  济南市, which dissolves Laiwu as the 2019 change did.
- `PREFECTURE_RENAMES` — six prefectures renamed from 地区 to 市 (Tibet's 昌都,
  那曲, 山南, 林芝; Xinjiang's 吐鲁番, 哈密), whose counties were already grouped
  correctly.

Any change here renumbers much of the country, since codes are positional. That
is harmless as long as `zonal_pop.py` is re-run afterwards; it reads the grid
straight out of the zip, so nothing has to be unpacked or re-keyed.

## Population, 2020

No open source publishes the township counts as a table. They are printed in
《中国人口普查分乡、镇、街道资料—2020》(China Statistics Press, 2022); the Excel
transcriptions that circulate are all behind Baidu Netdisk.

So the counts start from a grid. ASPECT (Ju et al. 2025, *Scientific Data*,
CC BY 4.0) takes those printed township counts and spreads each one over 100 m
cells by dasymetric mapping. That spreading is mass-preserving, so summing the
grid back over a township returns roughly that township's census count.
`zonal_pop.py` does the sum, reading the tif straight out of its zip, in about
two and a half minutes.

### The units trap

ASPECT documents its values as persons per hectare, but the grid is EPSG:4326
with square 0.00089832° cells whose true ground area shrinks with cos(latitude).
Weighting by true cell area gives a national total of 1,187 M against a census
1,411 M — short by exactly the population-weighted mean of 1/cos(latitude), which
reads like missing data rather than a units error. The values are persons per
*nominal* hectare, one per cell, so the recovery is a plain sum. `TRUE_CELL_AREA`
in `zonal_pop.py` keeps the other reading available for anyone re-checking this.

### Carrying the census onto our boundaries

The county census panel (`census_county_2010-2020_v1.csv`, Dong & Wang,
github.com/leiii/census) has a 2020 polygon for every county alongside its 2010
and 2020 figures. `zonal_pop.py` burns those polygons onto the grid in the same
pass as our townships and writes `township_panel_pop2020.csv`: how many grid
people sit in each (township, census county) piece. `fetch.py` then scales each
piece by a factor for its census county and sums the pieces back into
townships. Nothing is matched by name.

Matching by name was the previous method and it went wrong in three ways. A
2014 county and the 2020 county of the same name are often not the same ground
(Changsha's 天心区 was pushed 25% too high by being given 2020 Tianxin's figure),
268 counties found no name at all, and those 268 absorbed whatever their
province had left over — Hunan's two took all of Hunan's slack, which put 衡南县
at 2.14 M against a census 0.80 M.

How far each side is trusted:

- **Each census county's figure is taken as it is**, and the grid only says
  where inside that census county its people live. On the census's own polygons
  the grid is right to 0.2% at the median and within 5% for 92% of counties, and
  where it is badly out the error is usually the grid's: 衡南县 is +43% while
  every other county in Hengyang is within 2%, Chuzhou's urban core is +174%,
  Tongling's 义安区 +190%.
- **This holds even where the census's reporting units and the ground part
  company** — the decision is that the census is used wherever it has a figure.
  Zhengzhou's first row is 管城+金水+郑东新区+经开区+航空港区, and much of the
  Zhengdong and airport zones' land is legally 中牟县 and 新郑市, so the census
  books people the grid puts in Zhongmou (1.4 M) to Jinshui, and Zhongmou gets
  its census 703k. Anyang's 殷都区 (-71%) beside 安阳县 (+102%) looks like a
  polygon out of date; 新乡市's "原阳县+平原示范区" row actually sits on 新乡县,
  which has no row; XPCC cities' regiments live on other counties' ground. In
  all of these the census figure is used anyway. `CENSUS_EVERYWHERE = False` in
  `fetch.py` flips to letting the grid decide inside such prefectures, 20 of
  them, recognised by a development-zone row the grid is more than 10% off from
  or two census counties more than 25% off in opposite directions.
- **People the census books to no county are handed back.** In nine provinces
  the census counties sum to less than the provincial bulletin — Shaanxi 770k,
  Liaoning 630k, Fujian 550k, Anhui 460k. They are most likely development-zone
  populations reported only at province level (Shaanxi's would be Xixian New
  Area, whose host counties in Xianyang the grid holds 900k over). They are
  returned to the counties the first step cut, in proportion to the cut.

Provinces are then put onto their published census totals, which moves them by
well under a percent once the boundaries are right.

**The panel has no Xinjiang.** It was built from local census bulletins and
Xinjiang's counties published none, so all 106 Xinjiang rows are empty.
`xinjiang_counties.py` fills them from hongheiku.com, whose county pages carry
the census table in the layout of the national county book
(《中国人口普查分县资料—2020》, print only; Excel copies are sold by resellers).
Summed by prefecture, those figures
equal the official table in Xinjiang's own census bulletin No. 2
(`xinjiang_prefectures_2020.csv`) to the person in all 15 rows. As a check on the
site outside Xinjiang, its Hunan pages match the panel in 107 of 108 counties in
both years; the odd one, 衡东县, is a slip in the panel (it puts Hunan exactly
3,000 short of its bulletin), corrected in `fetch.py`'s `PANEL_FIXES`.

### How good it is

`validate_counties.py` prints three things. On the census's own polygons, the
raw grid is 0.2% out at the median. Where one of our counties is the same ground
as one census county (2,521 of 2,861), the published figure is within 5% of the
census for 99.2% and within 10% for 99.6%; the name-matched build before it
managed 97.9% and 98.3%, with 23 counties more than 25% off against 6 now. And it
lists every county sitting more than 15% from its census-implied figure, with the
reason — what remains is the handed-back people above. Only two counties cannot be checked at all, the island
groups 嵊泗县 and 长海县, where a tenth of the grid falls outside the census
polygons.

At 1.41 billion the country needs about 1,410 regions of a million. The median
township holds 18,500 people, so a region is around 54 of them and about 2.6
counties — the county level is the one to assemble from, with townships for
splitting a large county or following a line precisely.

## Population, 2010

The same pieces carry 2010: each piece is scaled by its census county's own
2010/2020 ratio from the panel (or, for Xinjiang, from hongheiku.com's 6th-census
column). A ratio outside 0.25–4 is a boundary moved between the two censuses
rather than growth — Harbin's 香坊区 0.21 beside 平房区 3.8 — and those pieces take
what their province has left over. Each province is then put onto its published
2010 total, which moves none by more than 1%.

Xinjiang used to need a 0.891 correction here. That was the empty panel: every
Xinjiang county got the national growth rate, and Xinjiang grew far faster.

**It is context, not arithmetic.** The viewer's current-year estimate is a line
through the last two years it holds, 2020 and 2024; 2010 only fills the history
table and raises the two "direction changed" flags.

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
detail.

## Known gaps

261 townships come out zero. They are ASPECT's own missing-data units and are
almost all forestry stations, state farms, 经营所 and industrial parks —
special-purpose units holding a few hundred thousand people in total. 80 of them
are in Heilongjiang.

The hand-back of people the census books to no county goes to every county the
first step cut in that province, in proportion. It cannot tell Xianyang's
Xixian counties from 商南县, whose grid excess is its own, so a little of
Shaanxi's 770k lands in the wrong place.

## Order to run

```
python prep_boundaries.py     # ~6 min, writes boundaries/adm{1,2,3,4}.gpkg + units.csv
python zonal_pop.py           # ~2.5 min, reads the ASPECT zip; township sums + census pieces
python xinjiang_counties.py   # instant once data/china/hongheiku_xinjiang.csv exists (else ~2 min scrape)
python fetch.py               # instant, writes population.csv + panel_rows.csv + counties.csv
python validate_provinces.py  # instant, all three years against the published totals
python validate_counties.py   # instant, against the county census
python ../build_country.py china
python ethnicity.py           # rebuilds composition.json on the new adm3; needs chinaethnicity's cells.npz
```

The ASPECT raster is `data/asia1m/china/aspect_population_total_pop.zip` (488 MB,
figshare doi:10.6084/m9.figshare.27323106). `zonal_pop.py` reads it through
GDAL's `/vsizip/`; an unpacked `aspect_population_total_pop.tif` beside the zip
is used instead if present.

The township level is written as one geojson per province under
`countries/china/adm4/`, because 43,655 features in a single fetch is not
something the viewer can hold. It loads the provinces on screen, up to four.
