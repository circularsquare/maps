# koreariders — a Korea-wide rail flow map

Prototype stage. The question: can Korea be mapped the way
[japanriders](../japanriders/) maps Japan — line thickness proportional to
passengers per segment — when Korea publishes no 輸送密度 equivalent?

Answer so far: **yes for the branch network, not yet for the spine.** Per-segment
figures are reconstructed rather than downloaded. 16 of 22 intercity lines
reconstruct to something usable; the four trunk lines do not, for a reason that
is understood but not yet solved.

## What Korea publishes

From the **철도통계연보** (Korail, annual, free, no login — `data/` holds the 2022
Excel bundle; 2023 is the newest on info.korail.com, later editions moved to
[railstat.korail.com](https://railstat.korail.com/statPortal/)):

| table | granularity | what it is |
|---|---|---|
| `4. 수송(여객)` sheet 8 | **station** | 역별 승하차, split 상행 / 하행, 253 stations |
| `4. 수송(여객)` sheets 9–13 | **station × train type** | the same, split KTX / SRT / 새마을 / ITX-새마을 / 무궁화 / 통근 |
| `4. 수송(여객)` sheet 5 | **line** | 선별 통과인원 — passengers who used each line |
| `4. 수송(여객)` sheet 4 | **line** | 선별 인거리 — see the warning below |
| `6. 운전` sheets 2(5)–2(7) | **segment** | 선구별 열차종별 운행횟수 — 117 sections, trains/day + 선로용량 |
| `8. 시설` sheet 2 | **station** | 노선 → 역명 roster (each station's *home* line, one only) |
| `8. 시설` sheet 4 | **line** | 기점, 종점, 영업거리 |

Passenger volume exists only per line; the only thing published per segment is
train frequency. The per-segment *passenger* numbers have to be built.

### 선별 인거리 is not a density — do not divide it by 영업거리

`probe_ingeori.py` divides each line's 인거리 by its 통과인원 and compares that
average distance with the line's own length:

```
line               통과인원          인거리     평균km   영업거리    비율
광주송북연          5473745      510334760      93.2       2.2    42.4
행신선             1792722      224823747     125.4       3.4    36.9
전라선             7714928      961928715     124.7     180.4     0.7
경부선           105147594     9389449004      89.3     441.7     0.2
중앙선            10547912      352964101      33.5     331.4     0.1
```

광주송정북연결선 is 2.2 km long and its average user apparently rides 93 km on
it. The ratio swings from 0.1 to 42 across the network. **통과인원 is a plain
count and is usable; 인거리 is not.**

## The reconstruction

Order a line's stations along the track and cumulate boardings minus alightings,
separately per direction, and you have the load on every segment.

Junctions are the difficulty: a junction station's counts mix every line through
it. The trick is to cumulate **inward from the far terminus**:

```
load(n-1, n) = 하차 at the terminus            (nothing boards there)
load(i-1, i) = load(i, i+1) - 승차_i + 하차_i    for i = n-1 .. 1
```

which never reads station 0, the junction end. The through flow crossing the
junction is never observed and never needs to be — it falls out of the
recursion.

Lines that junction at **both** ends have no clean terminus, so the anchor reads
a junction's whole traffic and lifts both profiles by a constant. The *shape*
survives, because adjacent-segment differences come from interior stations, so
exactly one number per line is missing and 통과인원 pins it (통과인원 moves by 2δ
when both profiles shift by δ). Same split japanriders uses: shape from one
source, magnitude from another.

### Results

`python build.py`:

```
line        verdict stops      km        통과인원    yearbook  mirror      수송밀도
전라선         good       18   180.4     7878650     7714928    2.2%     11262
경원선         solved     40    94.4     7502185     7502185    0.0%     20548
장항선         solved     28   152.8     4235851     4235851    8.4%      4320
경전선         solved     35   277.7     5583232     5583232    6.8%      3835
대구선         solved      6    26.1     1623828     1623828    3.6%      3665
충북선         solved     16   115.0     1816583     1816583    3.0%      2985
영동선         solved     24   188.9     1267405     1267405    3.1%      2647
중부내륙선       solved      5    56.9      130011      130011    5.2%       315
정선선         solved      7    45.9       16018       16018   11.4%        25
경부고속선       partial    27   398.2    52680935    95125521    1.4%     72505
호남고속선       partial     5   183.8    10741974    22425971    0.9%     22250
강릉선         partial     6   120.7     3801038     4864865    1.8%      7506
호남선         partial    19   252.5     2137234    16798094    3.5%      1929
태백선         partial    19   104.1      387560      711164    2.3%       808
경춘선         partial    20    80.7        2398        2399    0.7%         7
경의선         partial    25    56.0         820     1795352    0.0%         2
광주선         shaky       4    11.9      292146      496870    7.7%       742
경북선         shaky      11   115.0      191638      191638   23.6%       359
동해선         broken     31   188.9     3765490     4214757   12.8%      6125
중앙선         broken     35   332.2     2974985    10547912   35.1%      1631
수서고속선       broken      6    61.1      845244    19167014   13.8%     -3126
경부선         broken     98   441.7    22562701   105147594   10.9%    -21403
```

**mirror** is the worst disagreement between the 하행 and 상행 profiles, which
are built from disjoint columns of the source — it tests the method rather than
the data, and nothing but a broken cumulation can widen it.

The verdicts:

- **good** — clean terminus anchor, carries every train type, and its rebuilt
  통과인원 lands within 20 % of the published count. Only 전라선 qualifies fully
  (1.021, mirror 2.2 %).
- **solved** — no clean terminus, so the level came from 통과인원 and the ratio is
  an identity. Judge these on the mirror column alone. Eight lines, most under
  6 %.
- **partial** — clean anchor, but restricted to some train types, so 통과인원
  counts traffic the reconstruction deliberately excludes and the ratio is not a
  fair test. Mirrors are excellent (0.0–3.5 %); the profiles are probably fine
  and simply cannot be checked this way.
- **shaky / broken** — see below.

전라선, the one line that passes every test, in daily passengers both directions:

```
익산 ─17832─ 전주 ─12679─ 남원 ─10927─ 순천 ─6050─ 여천 ─4059─ 여수엑스포
```

## What is broken, and why

**Parallel lines sharing stations.** 경부선 and 경부고속선 both call at 서울,
대전, 동대구 and 부산, and sheet 8's combined counts cannot say which metals a
passenger rode. Cumulating the total along either line is meaningless — 경부선
came out at 22.6M against a published 105.1M, with negative loads.

Splitting by train type (sheets 9–13) fixes most of it: only the high-speed
services use the 고속선. That alone moved 경부고속선 from 15M to 53M and halved
several mirror gaps. **It does not finish the job**, because a train type is not
a line: 서울's KTX arrivals are 경부, 호남, 전라 and 강릉 KTX together, and
`lines.TYPES` currently hands all of them to every line listing KTX. 경부선
remains negative because 호남선 무궁화 trains join it at 대전조차장, so 대전's
conventional alightings include passengers who never rode 경부선 south of there.

The proper fix is the one already flagged for junctions, generalised: **solve
the network rather than each line independently.** Flow conservation at every
junction node, one equation per node, with each station's boardings allocated
across the lines that actually serve it — train frequency per section
(`6. 운전` sheets 2(5)–2(7)) is published and would weight that allocation.
The clean-terminus lines would then constrain the messy ones instead of each
line being solved alone.

## Geometry

Station order is the other thing the yearbook does not give — its rosters are
alphabetical. Two OSM approaches were tried:

- **Route relations** — rejected. `KTX 전라선: 용산 → 여수엑스포` is a clean
  391.9 km with zero gaps, but most lines have only a stub covering the first
  60 km.
- **Named track ways** — used. 64 % of Korean `railway=rail` ways carry a `name`,
  167 distinct, covering every line the yearbook reports on. `build.py` builds
  one national graph (117k nodes) and takes the shortest path between each
  line's end stations, which picks the through route and ignores sidings and
  triangles. Track belonging to another line costs 40× its length — cheap enough
  to bridge a station throat tagged with the crossing line's name, far too
  expensive to follow 경전선 out of 순천.

Stations snap to that path and sort by distance along it; chainage is rescaled to
the yearbook's 영업거리 (OSM runs 0.2–0.7 % short).

**Snap radius alone is not enough.** 천안아산 is a 경부고속선 KTX station about
100 m from 아산 on 장항선; snapping by distance pulled its 720k arrivals into the
장항선 cumulation and broke the line. Filtering against the roster fixes it —
the roster gives each station exactly one *home* line, so a station reporting
traffic whose home is elsewhere is somebody else's. The line's own two endpoints
are exempt: 익산's home is 호남선 but it is still where 장항선 ends.

That same roster fact is how junction ends are detected at all — a station
appears on only one roster, so "not on this line's roster" *is* the junction
test.

## Files

```
lines.py            yearbook parsing + the line-name table + anchor detection
build.py            the reconstruction over all lines; --line NAME for one
fetch_osm.py        Overpass: route relations, named rail ways, station nodes
kric_index.py       scrape the 레일포털 catalogue (475 datasets) to data/kric_index.csv
prototype.py        the original single-line version, with its workings printed
probe_ingeori.py    shows 선별 인거리 is not track-attributed
probe_geom.py       whether a route relation's geometry is contiguous
probe_ways.py       whether Korean rail ways carry line names
probe_routes.py     how a route relation is assembled
```

`data/` holds the 2022 yearbook zip, the OSM pulls (26 MB of named ways, 1,997
station nodes), the KRIC catalogue index and `segments.geojson`.

## What is not done

- **Network-wide solving**, per above. This is the main outstanding piece and
  what stands between 16 lines and all 22.
- **Geometry on the output.** `segments.geojson` currently carries the numbers
  and null geometry; the corridor polyline needs slicing per segment.
- **광역철도 and 도시철도.** Not in the intercity station table at all. seoulriders
  already has real per-segment 수도권 OD; Busan/Daegu/Gwangju/Daejeon have
  per-line totals in the yearbook's parts 2 and 3 plus station 승하차 from each
  city's portal, but no OD.
- **Station bubbles.** Available nationally and would sit well over the lines.
- **Newer data.** 2023 is the last year on info.korail.com; the download path on
  the railstat portal has not been found.
- The honest caveat for the finished map: unlike japanriders, these segment
  values are **derived, not published**. That belongs on the page.
