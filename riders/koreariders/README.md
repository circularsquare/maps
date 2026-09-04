# koreariders — a Korea-wide rail flow map

Prototype stage. The question: can Korea be mapped the way
[japanriders](../japanriders/) maps Japan — line thickness proportional to
passengers per segment — when Korea publishes no 輸送密度 equivalent?

Answer so far: **yes for 21 of 22 intercity lines, cumulating each line on its
own.** Per-segment figures are reconstructed rather than downloaded. Only 동해선
still fails outright.

The trunk lines used to fail too, and the cause turned out to be a bug rather
than the data. `resolve()` swaps a line's ends to put the clean anchor last, and
for 경부선, 중앙선 and 수서고속선 that leaves the chain running 종점 → 기점 while
the 승하차 columns stay labelled by the line's own 기점 → 종점. Cumulating with
the columns backwards is what put negative loads on 경부선. With the swap
recorded as `reversed` and the columns swapped to match, 경부선 goes from a
10.9 % mirror and a 수송밀도 of −21,403 to **2.6 % and 33,018**; 중앙선 from
35.1 % to **3.3 %**; 수서고속선 from a negative density to **44,783**. The
network solve in `solve.py` was built to rescue those four lines and is no
longer what rescues them — see [What the network solve is for](#what-the-network-solve-is-for).

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
중앙선         good       35   332.2     6380391    10547912    3.3%     12435
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
수서고속선       partial     6    61.1    18002340    19167014    1.1%     44783
경부선         partial    98   441.7    32909981   105147594    2.6%     33018
호남고속선       partial     5   183.8    10741974    22425971    0.9%     22250
강릉선         partial     6   120.7     3801038     4864865    1.8%      7506
호남선         partial    19   252.5     2137234    16798094    3.5%      1929
태백선         partial    19   104.1      387560      711164    2.3%       808
경춘선         partial    20    80.7        2398        2399    0.7%         7
경의선         partial    25    56.0         820     1795352    0.0%         2
광주선         shaky       4    11.9      292146      496870    7.7%       742
경북선         shaky      11   115.0      191638      191638   23.6%       359
동해선         broken     31   188.9     3765490     4214757   12.8%      6125
```

**mirror** is the worst disagreement between the 하행 and 상행 profiles, which
are built from disjoint columns of the source — it tests the method rather than
the data, and nothing but a broken cumulation can widen it.

The verdicts:

- **good** — clean terminus anchor, carries every train type, and its rebuilt
  통과인원 falls in the 0.6–1.5 band around the published count. 전라선 (1.021,
  mirror 2.2 %) and 중앙선 (0.605, mirror 3.3 %) qualify, though 중앙선 only
  scrapes the bottom of the band.
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

**Chain orientation, now fixed.** This was the whole of the trunk-line problem
and it was not a data problem at all. `resolve()` puts the clean anchor end last,
which for 경부선, 중앙선 and 수서고속선 reverses the chain relative to the
yearbook's 기점 → 종점 — and 상행/하행 are labelled by 기점 → 종점, so the columns
came out backwards. Anchoring 경부선 at 서울 then asserted that everything alights
at 서울 in 하행, when 하행 is where trains *depart* from. `spec["reversed"]` now
records the swap and `reconstruct()` takes the other column pair, which is what
turned three of the four broken lines into 2.6 %, 3.3 % and 1.1 % mirrors.

Everything below was written when those three lines were still failing, and it
is still true — it is just much smaller than it looked.

**Parallel lines sharing stations.** 경부선 and 경부고속선 both call at 서울,
대전, 동대구 and 부산, and sheet 8's combined counts cannot say which metals a
passenger rode. Splitting by train type (sheets 9–13) does most of the work, as
only the high-speed services use the 고속선. It does not finish the job, because
a train type is not a line: 서울's KTX arrivals are 경부, 호남, 전라 and 강릉 KTX
together, and `lines.TYPES` hands all of them to every line listing KTX.

**동해선**, the one line still broken, and 경북선, still shaky at a 23.6 % mirror.

## What the network solve is for

`solve.py` fits every line at once — entry flows, junction steps and per-station
allocations together — and was written to rescue the four trunk lines. The
orientation fix rescued three of them first, from the other direction, so the
solve's remaining job is much narrower than it was built for.

It has been through the same correctness pass: the 상행 cumulation had its sign
flipped (which the mirror residual could only satisfy by flattening every profile
to a constant), junction steps were being allocated at line ends where they could
not move a segment, the junction constraint ignored any line that *terminated* at
the junction — so 경부선's step at 대전조차장 had nothing to balance against and
was driven to zero — and it paired lines by each chain's own order rather than by
the direction passengers actually travel. Its 통과인원 residual is now a one-sided
**ceiling**: the figure counts everyone on the line's metals while the rebuild
sums only that line's train types, so a rebuild may land under it but never over,
and going over is exactly what leaking traffic in from a junction looks like.
Shares are allocated per train type, since 광주송정's KTX belong to 호남고속선 and
광주선 and have nothing to do with 호남선's 무궁화 at the same platforms.

That fixed the arithmetic but not the fit. The junction steps stayed free, and
free is what they could not be: 경부선 kept a 3.7M step at 용산 that punched its
load through zero, and no weight on the "steps are small" prior helped. Low left
경부선 at a 121.7 % mirror, high fixed 경부선 and broke 전라선 and 중앙선 instead,
and the response was not monotonic — a fit with many near-equivalent optima,
where the prior only chooses which line absorbs the error. No amount of weighting
could settle it, because the information was not in the residuals.

### What settled it: the published train counts

It was in the yearbook. `6. 운전` sheets 2(5)–2(7) give trains a day on each of
117 sections, by train type, and the count *changes* at a junction by exactly the
service stepping on or off:

```
경부선  서울-금천구청   새마을 26 + 무궁화 40 = 66
        금천구청-의왕   26 + 40 = 66
        의왕-천안      26 + 40 = 66
        천안-조치원     20 + 31 = 51      <- 15 trains leave at 천안
장항선  천안-신창       6 + 9   = 15      <- and here they are
```

`frequency.py` parses them. The useful half is the flat stretches: 66 trains
unchanged from 서울 to 천안 means nothing joins or leaves in between, so a step at
용산 is not unlikely but impossible, however well it suits the 승하차. Where the
count does change the sign is fixed too — trains leaving cannot put passengers
on. Steps at stations the count runs straight through are held to zero; the rest
keep the weak prior and let the 승하차 set the size.

That is the whole difference between guessing and knowing, and it shows:

| | before | after |
|---|---|---|
| 경부선 mirror | 121.7 % | **2.3 %** |
| 경전선 mirror | 111.5 % | **2.9 %** |
| 동해선 mirror | 90.5 % | **10.5 %** |
| lines with a negative segment | 2 | **0** |
| worst mirror, weighted by traffic | — | 5.3 % (중앙선), excluding two near-empty lines |

Every large step that survives is at a real interchange — 삼랑진, 익산, 서원주,
오송, 천안 — and none is at a station the counts run straight through.

**On the mirror column.** The worst single segment is a bad summary: it is
whichever segment carries almost nobody, where a handful of passengers reads as
100 %. 충북선's 조치원-오송 stub is 0 against 846 — two people a day — while its
other fifteen segments agree to 3 %. The report now leads with the disagreement
weighted by the traffic it applies to, and keeps the worst segment beside it.
Weighted, the median line is at 1.7 % and the trunk lines are at a few tenths of
a per cent.

Still imperfect: 대구선 (13.3 %), 경북선 (11.3 %) and 영동선 (9.0 %) are short
lines where one junction carries most of the traffic, and 경춘선 and 경의선 have
so little 일반열차 traffic left — densities of 9 and 1 — that nothing can be said
about them either way.

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
solve.py            the whole network fitted at once; --line NAME for one
frequency.py        선구별 운행횟수 — trains/day per section, which pins the junctions
membership.py       which lines physically serve each station, by track proximity
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

- **The short lines.** 대구선, 경북선 and 영동선 sit at 9–13 % weighted mirror,
  where one junction carries most of the line and the step size is still a guess.
  The section counts give a magnitude as well as a sign — 천안 is 66 trains down
  to 51, so roughly 15/66 of the load leaves — and using that ratio as the step's
  target, rather than only its sign, is the obvious next turn of the screw.
- **`build.py` and `solve.py` disagree on levels.** 경부선 is 33,018 from the
  single-line build and 20,051 from the network fit, both with clean mirrors.
  Nothing yet says which is right, and the map needs one number.
- **Non-Korail stations on the chains.** `solve.py` admits any station whose
  membership test passes, which puts 신용산, 삼각지 and 숙대입구 on 경부선 and a
  string of 부산 도시철도 stops on it through 부산진, plus a duplicate
  서울역/서울 pair. They carry no yearbook flow so they do not move the numbers
  much, but they split segments that are not really segments, and they are where
  경부선's junction steps go wrong.
- **Lines whose passengers board off-line.** 호남고속선 runs 오송–광주송정, but
  nearly everyone on it boards at 용산 or 서울, which are not on its chain. Its
  rebuild can only see 2.6M of a published 22.4M. Fixing that means modelling
  *services* rather than infrastructure lines.
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
