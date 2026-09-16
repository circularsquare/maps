# Nationalities of China: notes

A dot map of China's 2020 census by nationality (minzu), one dot per 200 people, drawn from
county tables. `COMMANDS.txt` is the runbook, `provinces.md` is what is drawn and how; this
file is why things are the way they are.

Started 2026-09-14. Built in `religiondots`' image (merged-not-dropped tiles, Hilbert carry),
but it is a separate map with one country and one census, published by county wherever it is
open and estimated from the 2000 census where it is not.

## 1. What is drawn

Only a count the census published for a county, in the measured provinces. For each of them
that is table **1-4, population by region, sex and nationality**, from the province's own
2020 census yearbook: one row per county-level unit, 56 nationalities plus "nationality not
determined" (未定族称人口) and "foreign nationals naturalised" (入籍), each split by sex.

16 of 31 provinces publish it openly: 696,781,829 people, 49.4% of the 2020 census. The other
15 are estimated from the 2000 census's county table and hatched on the map (§7);
`provinces.md` lists them and how each could become measured.

### The files are hidden, and that is why this exists

Most 2020 provincial census yearbooks are online as an NBS-style frameset whose table list
(`left.htm`) links a JPG for every table. **An .xls with the same name sits beside each
JPG** and nothing links to it: `<yearbook>/zk/html/A0104.jpg` has `A0104.xls` next to it.
Spellings vary (Guangxi `A1-04.xls`, Fujian `html/a0104.xls` with no `zk/`, Shanghai
`ANJ-1-04.xls`, Beijing numbers the table 1-6). The national yearbook does the same, which
gives the province-level table the checks below rely on. `fetch.py` has every URL.

Three hosts refuse scripts from outside China and are fetched from the Wayback Machine
(Inner Mongolia, Zhejiang, Qinghai; Ningxia's whole yearbook PDF too). Python 3.9's
certificates reject several of the rest, so downloads go through curl.

## 2. Nothing is trusted until it adds up

`parse.py` rebuilds each table's prefecture/county hierarchy (from label indentation, or for
Ningxia's PDF from the numbers: a row is a prefecture when the rows under it sum to it) and
then requires, for every province:

- every prefecture row equals the sum of its counties, in all 59 columns;
- every row's total equals male plus female, in all 59 columns;
- the province total equals **the national yearbook's row for that province**, in all 59
  columns. This is a different publisher's copy, so a province yearbook that dropped or
  shifted a column cannot pass.

All 16 pass exactly. A province that fails is not written.

Ningxia is a text PDF (pages 30-49 of a 1,968-page yearbook). Its numbers are placed into
columns by x position under the column heads, because blank cells are simply not printed,
and a continuation page leaves out a whole row when none of that page's three groups live
there.

## 3. Counties: a name join, checked against a population grid

The tables have Chinese names and no codes. `join.py` matches each county row to DataV
GeoAtlas polygons (via religiondots' `sources/cn_geo.py`, which also clips them to de facto
administration) by exact name, then by stem, first inside the prefecture the table prints
the row under and only then province-wide; a province-wide match in a different prefecture
is refused.

Rows that are real units but do not match are listed in `OVERRIDES`, each with its reason:
2021 renames and mergers (Heng county to Hengzhou, Sha county to Shaxian district, Meilie
into Sanyuan, Jili into Mengjin, Gangzha into Chongchuan), an abbreviation (Dorbod), the
Daxing'anling forest districts that have no code, the Changbai Mountain management zones.

**Hangzhou** reorganised six districts in 2021 in ways the old rows cannot be divided
across, so those six rows share one placement area: the total there is exact, the
composition is averaged across the six old districts.

**Sansha** (2,333 people) is placed on the Paracels, where the city is seated. The Nansha
district polygon over the Spratlys is deliberately given nobody, because several states
administer features there. Kinmen (administered by Taiwan) has no row.

### Development zones go where ASPECT has people the table does not

About 130 rows in seven provinces are development zones, new areas, scenic areas and similar,
2.4 million people in Zhengzhou alone. They are census rows but not administrative units, so
no polygon exists and the table does not say which counties host them.

The ASPECT grid (§4) does say, because it is built from the same census's township counts and
puts the zones' people on their real ground. A county that hosts a zone holds more people in
ASPECT than in its own row; one that hosts none matches its row to about 1%. So within each
prefecture the zones are spread over the counties' surplus (ASPECT minus 1.05x the county's
own row), in proportion. `data/work/join_report.txt` prints each prefecture's zone total
against its surplus: mostly 0.8-0.9x, which is the check that the surplus is the zones and not
noise. Kaifeng is the clean example: a 418,307-person integration zone, and Longting district
holding 468,000 more in ASPECT than in its own row.

One zone had no surplus to find (Huanggang's Longganhu farm area, 27,855, whose host Huangmei
county sits just inside the 5% tolerance) and is an override instead.

## 4. Placement inside a county

The census says how many of each nationality live in a county, not where in it. So every
group in a county is spread the same way, over **ASPECT** (Ju et al. 2025, *Scientific Data*,
CC BY 4.0), which spreads the 2020 census township counts over 100 m cells by dasymetric
mapping. `aspect.py` sums it into ~500 m cells (5x5), reading the 10.8 GB tif straight out
of its zip, and burns the county polygons onto the same grid: 10.4 million populated cells,
0.04% of the grid's people outside every county polygon (coast and borders).

Its values are persons per nominal hectare, one per cell, so a plain sum is a count
(helper1m's `zonal_pop.py` explains why the true-area reading is 19% short).

Whole dots come from a carry along a Hilbert curve over (row, cell) pairs, never from ranking
(religiondots' rule): each group's national count is exact to under one dot. A dot lands at a
uniform random point in its 500 m cell.

### Tiles and dot size

`tiles.py` merges rather than drops, as religiondots does: dots of one group inside one merge
cell become one mark whose area is proportional to how many it holds. The merge cell is 1/64
of a tile (8 px), not religiondots' 1/32: at 1 dot per 200 people nearly every 16 px cell was
at the size cap, and the country read as a lattice of equal circles. The viewer's radius curve
is smaller than religiondots' at low zoom but keeps a lone dot about a pixel across from z8:
scaling the whole curve by sqrt(200/1000), which keeps ink per person constant, made single
dots sub-pixel and southern Ningxia's Hui villages all but disappeared.

## 5. Colours

`colors.csv` is the palette and is edited by hand. `palette.py` wrote the first version once
and refuses to overwrite it: it colours groups largest first, each taking the candidate
furthest (in OKLab) from the groups that share its counties. Han is pinned to a muted
grey-blue, because it is 91% of what is drawn and would otherwise set the colour of the whole
map; that is a legibility choice and is worth a second look.

## 6. Known limits

- The zone surplus method trusts ASPECT's township placement where it disagrees with DataV's
  county lines, e.g. Sanmenxia's integration zone mostly into Lingbao city and Xuzhou's
  development zone mostly into Jiawang district. Both are where ASPECT has the surplus.
- Inside a county, a minority concentrated in one town is drawn spread across the county. The
  2020 township tables that would fix this are not open.
- DataV's boundaries are 2025's and the census is November 2020's; §3 lists every place that
  mattered.

## 7. The estimated provinces

The 15 provinces with no open 2020 county table are drawn from the 2000 census's county
table, scaled to 2020 totals (`fallback.py`; the method is Anita's call, 2026-09-14). The
viewer hatches them, and hovering one names what it was scaled to.

**Pattern.** Table A0106 of the 2000 Population Census Data Assembly: county rows in the same
59 columns, as religiondots downloaded it from Harvard's `chinacensus` dataverse. Its county
names are romanised with no codes, and religiondots' resolver already matches them to DataV
adcodes (religiondots `sources/cn.md` §5), so fallback.py imports it rather than repeating
it. Every 2000 row in the 15 provinces resolves.

**Totals.** Where a province publishes its 2020 table 1-4 by prefecture (Hebei, Liaoning,
Hunan, Sichuan), each group is scaled per prefecture, so every prefecture's count of every
group is 2020's and only the split between its counties is 2000's. Elsewhere each group is
scaled to the national yearbook's row for the province. Every unit and every province adds
back to its 2020 row in all 58 columns. The 31 provinces together come to 1,409,778,724, the
census's provincial total (the national 1,411,778,724 includes 2 million serving military).

Hebei prints Shijiazhuang and Baoding twice, whole and marked ①. Shijiazhuang ① is the city
without Xinji, whose row sits under it, and the two add to the whole exactly. Baoding ① also
leaves out Xiong'an New Area, which has no row of its own: Baoding minus ① minus Dingzhou is
1,205,440 people with no column negative, and ASPECT holds 1,191,135 in Xiong, Rongcheng and
Anxin counties. So Xinji, Dingzhou and Xiong'an are units of their own (`REMAINDERS`).
Liaoning's Shenfu New Area row (171,820) is added to its host prefectures by ASPECT surplus,
as §3 does for zones. The signal is weak (0.20x) and gives all of it to Fushun.

**Carved districts.** 27 of today's polygons did not exist in 2000: Shenzhen's Longhua,
Guangming and Pingshan, Guangzhou's Nansha, nine Bingtuan cities and two border cities in
Xinjiang, and others. Left alone they would draw nobody while their parents held their people.
`CARVED` names each one's parent counties, and the script checks that every parent borders its
child. A parent's row is spread over its own polygon and its share of the child, the child's
population being divided among its parents in proportion to theirs. Where it was not clear
which towns moved, the parents are the counties the new unit borders most.

**What scaling cannot know.** A group with nobody in a unit in 2000 has no pattern, so it is
spread over the unit by 2020 population; the report lists each case, mostly a few dozen
people. A group that grew by migration is drawn where it lived in 2000: Guangdong's Yi grew
16 times, its Bouyei 10 times, Chengdu's Yi 22 times. The province or prefecture total is
right; where inside it those people live is 2000's.

**County totals are not fitted.** Each group is scaled on its own, so a county's drawn total
is its 2000 population grown at its unit's average rate. Counties that grew faster are drawn
thin: against ASPECT, Shenzhen's Bao'an is 2.2 million short, Longgang 1.7 million, Xi'an's
Weiyang 1.5 million. Counties that shrank are drawn thick, so Guangdong's median county is
drawn 26% above ASPECT. In the prefecture-scaled provinces the median county is within 7%.
religiondots rejected fitting to modern county totals (its cn.md §6): city growth in Xinjiang
and Tibet was mostly Han, so an IPF inflates minority counts in exactly those cities. A
narrower fix, letting Han alone absorb the gap, is open for Anita.

**Boundary drift.** Some 2000 units were redrawn rather than renamed, such as Urumqi county and
the districts around it. The resolver matches names, so 2000's people land on 2025's polygons:
Shihezi is drawn 371,000 above ASPECT and Urumqi's Xinshi district 514,000 below.

## 8. Handoff (end of the second session, 2026-09-14)

**State.** All 31 provinces are drawn: 16 measured, 15 estimated (§7). `fallback.py` and
`scatter.py` have run with every check passing; `tiles.py` had not been re-run at the end of
the session, so until it is, the archive holds only the 16 measured provinces while the
viewer already hatches the other 15. Local only: nothing published or committed. Anita's
first look at the 16 (first session): "a lot of good things, a lot of things we'll need to
tweak, but it looks great so far".

**Visual tweaks still open**, roughly in order of how much they show:

1. **Low and mid zoom (about z3-6) reads as tiled circles.** In dense provinces nearly every
   merge cell hits the radius cap, so eastern China is a mesh of equal grey discs with colour
   only at the edges. Ideas not yet tried: a smaller cap at low zoom only (per-stop caps in
   `radius()` in `index.html`), lower opacity for capped marks, or a finer merge cell at low
   zoom only (`tiles.py --cell-bits`, currently 6 everywhere).
2. **A faint lattice in rural areas around z6**, from one merged mark per cell at the cell's
   mean position.
3. **Palette.** `colors.csv` is `palette.py`'s first draft and has not been looked at by eye.
   Greens collide (Hui, Manchu, Hani, Maonan, Pumi) and so do pinks (Zhuang, Naxi, Dai,
   Evenki). Han's grey-blue is a legibility choice Anita has not confirmed. The estimated
   provinces bring Tujia, Bouyei, Yi, Tibetan and Uyghur in at full size for the first time.
4. **Default dot size** is a guess (`MERGED_STOPS`, `DOT_GAIN` in `index.html`); the slider
   covers it meanwhile.
5. The OpenFreeMap dark basemap's labels are multi-script and busy over China. religiondots
   uses the same basemap, so any change there is a shared decision.
6. **The hatch on estimated provinces** (`HATCH_ALPHA` in `index.html`) has not been judged by
   eye. It sits under the dots, so it shows only where dots are sparse.

**Paused by Anita at the end of this session (2026-09-14).** Everything below waits on her, so
the next session should start by asking these.

Questions
- Estimated provinces: should Han alone absorb the gap between drawn county totals and ASPECT
  (§7)? It would fix Shenzhen, Xi'an and the other fast-growing cities without changing any
  minority count. Suggested: build it behind a flag and compare the two by eye.
- Chongqing's "other nationalities" column (74,320 people): how to split it into the 52
  smaller groups once the short table is parsed (`provinces.md` queue 1). Suggested: each
  county's "other" spread by those groups' 2000 county pattern, fitted to the province totals.
- Can chinaethnicity get a "just run it" rule like religiondots and koreariders? `tiles.py`
  takes about 8 minutes at 7 million dots, over her 3-minute hand-off rule.

To test
- Run `python tiles.py` (about 8 minutes, with no `npx serve` holding the archive). Until then
  the map shows dots for the 16 measured provinces while hatching the other 15.
- Then judge the hatch on estimated provinces (`HATCH_ALPHA` 0.13 in `index.html`), especially
  over Sichuan and Guangdong, alongside visual tweaks 1-5 above.

To check
- `CARVED` in `fallback.py`: the parent counties come from memory plus a border check. Least
  certain: Gongqingcheng, Guanshanhu and the Bingtuan cities.
- Liaoning's Shenfu New Area (171,820) goes entirely to Fushun on a weak ASPECT surplus
  (0.20x); an even split by population might be just as defensible.

**Data work next**: the `provinces.md` queue, starting with Chongqing's short table.

**Decided by Anita, 2026-09-14**: provinces with no open 2020 county table are drawn from an
older county pattern scaled to 2020 totals (prefecture totals where published, province totals
otherwise), marked on the map as distinct from the measured provinces. Built in the second
session as §7.

**Decided by Anita, 2026-09-14**: Sansha's people stay on the Paracels only, with the Spratly
(Nansha district) polygon empty, as §3 describes.

**Things that cost time and are now handled**, so nobody rediscovers them: Python 3.9's
certificates reject several bureau hosts (downloads use curl); Windows `tar` cannot read the
Chinese member names in Yunnan's rar (7-Zip can); Ningxia's PDF omits blank rows on
continuation pages; the national table is also a hidden `.xls`; rasterio reads ASPECT
straight out of its zip, so the 10.8 GB tif is never unpacked. Chongqing's host drops the
connection partway through its 337 MB PDF, so `fetch.py` resumes a partial download. Hebei's
① rows are the prefecture without a county-level city, and Baoding's also without Xiong'an.
Wayback's Save Page Now cannot reach the Tianjin and Shaanxi hosts either (`provinces.md`
queue 4).
