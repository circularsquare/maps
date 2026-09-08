# koreariders — a Korea-wide rail flow map

Prototype stage. The question: can Korea be mapped the way
[japanriders](../japanriders/) maps Japan — line thickness proportional to
passengers per segment — when Korea publishes no 輸送密度 equivalent?

Answer so far: **yes for 21 of 22 intercity lines, cumulating each line on its
own.** Per-segment figures are reconstructed rather than downloaded. Only 동해선
still fails outright.

One line is a different kind of no. 경춘선's numbers rebuild perfectly and are
worthless, because the source has no 승하차 for the ITX-청춘 and credits the whole
line with 2,399 passengers a year against 26 trains a day each way. It is drawn
grey and says so; see [The thin lines, and which of them are
wrong](#the-thin-lines-and-which-of-them-are-wrong-2026-09-07).

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

## Picking this up

Read this section, then `todo.txt` (Anita's own list — **hers to edit, not
ours**), then the rest of this file for the why behind any given decision.
Written 2026-09-07, at the end of a long session; the ordering below is a
recommendation, not a queue.

**Before you touch anything.** More than one Claude session works in this tree
at once. `lines.py` changed under this session's feet mid-run, and a
`segments.geojson` three hours old is a diff against *every* change since, not
against yours. To measure your own change, re-run with only that change
reverted, on the tree as it stands, and pass `solve.py --out <scratch>` so
`data/` is untouched. Check a file's mtime before blaming the code.

**Run order and cost.** `solve.py` is about eight minutes and writes
`data/segments.geojson`, which is what the page draws; run it yourself rather
than handing it over. `build.py` writes `segments_singleline.geojson`, which
nothing reads — it exists to be compared against. The city builds are seconds.
`check.py`, `check_cities.py` and `compare_builders.py` all read the output
files and need no solve.

**What is settled, so you do not reopen it.** The map draws `solve.py`, not
`build.py` — the published train counts overturned the 인거리 verdict that
favoured the single-line build. The city decays are fitted to published trip
lengths, not assumed. The train counts are per direction (편도, the sheet says
so in its own header). None of that needs revisiting.

### Worth doing next

1. ~~**경부고속선's 서울-광명 stub.**~~ Done 2026-09-07 — the line reaches 서울역
   and 인거리 cover went 86.5 % → 88.6 %. See [Reaching the end station, and
   drawing between
   stations](#reaching-the-end-station-and-drawing-between-stations-2026-09-07).

   What is left of it: **pull unnamed track into the graph.** The reason
   `reach_station` refuses 광주선's 1.77 km gap and 중부내륙선's 1.63 km is that
   `way[railway=rail][name]` excludes the junction throats that would connect
   them, so a 1.77 km gap routes 402 km. One more Overpass query and a graph
   rebuild would likely let both reach their end stations without naming a host
   line, and would leave `OVER` for the cases that really are one line running
   over another's metals. It touches every corridor, so it wants its own
   session and a full before/after.

2. ~~**The 호남선 handover split.**~~ Done 2026-09-07 — see [The 호남선 split,
   and one coordinate](#the-호남선-split-and-one-coordinate-2026-09-07).

3. **경원선's 24.6 % weighted mirror**, now the largest that means anything —
   경춘선's 113 % turned out to be the source not counting the line at all and
   정선선 at 26.8 % carries forty people a day. 경원선's *profile* is right; what
   disagrees is the level, by about 660k a year on the trunk, and it is all one
   step: the whole line's traffic moves onto 경춘선 at 청량리, the largest step in
   the network, and the two directions do not settle on the same size for it.
   `W_TAUSYM` already pushes them together and loses.

   **The cause is known and the obvious fix is measured and rejected** — see
   [경원선's level is
   contaminated](#경원선s-level-is-contaminated-and-taking-the-contamination-out-costs-more-than-it-buys).
   Its published 통과인원 counts 광역전철 riders the 승하차 cannot see, and
   deleting that constraint fixes 경원선 and breaks 충북선. Anything further has
   to correct the 7.50M rather than drop it.

4. **Busan lines 1-4, the one city model still uncalibrated.** 부산교통공사 files
   `공사에서 관리하지 않는 데이터임` in the yearbook sheet that measures every
   other operator's trip length, so its 20-minute decay is still an assumption
   where the other four are fitted. It is also the largest of the five. Needs a
   trip-length or OD source from somewhere other than the yearbook.

5. **The high-speed 통과인원 undercount is systemic**, not particular: 경부고속선
   reaches 0.54 of its published figure and 호남고속선 0.61. Both lines' riders
   board at stations off their own chains. `ENTRY_SHARE` addressed part of it
   and the capacity check cannot falsify what remains — 호남고속선 sits at 396
   passengers a train against a KTX-1's 935, and tightening that ceiling needs
   published rolling-stock allocation per line, which we do not have. Do not
   invent it from memory of which trains run where.

6. **Passenger sheet 3's attribution rule.** 선별 여객수송 is per line *and* per
   train type, which is exactly the shape needed to sharpen the 통과인원 ceiling
   for the nine `partial` lines. It is not usable until someone works out what
   rule assigns a journey to a line: 전라선 reads 3.27M against a published 7.71M
   line total and a 7.88M rebuild. Maybe an hour to characterise, unclear payoff.

### Smaller, and mostly legibility

- ~~**Non-Korail stations on the chains**~~ and ~~**stations a line's trains run
  past without stopping**~~ — both gone 2026-09-07, and neither needed naming:
  `merge_unserved` joins across any station with no 승하차 of the line's types,
  which is what both of them were. 426 features became 247.
- **중부내륙선 is still drawn 1.9 km short**, the last thing `check.py` flags.
  It is the same end-station fault as 경부고속선's 서울 and the bounded stub
  refuses it — 1.63 km straight routes 5.01 km, just over the 3× directness
  bound, because the junction with 경강선 is unnamed track and not in the graph.
  Pulling unnamed rail is the fix; see item 1 above.
- **2023 is the newest yearbook on info.korail.com**; we use 2022. The download
  path on the railstat portal has not been found.

### Things that will waste your time

- **Overpass needs a User-Agent.** overpass-api.de answers a default
  python-requests header with a bare **406**, kumi.systems with a **429** whose
  reason is only in the body. Neither is a rate limit. `fetch_osm.py` and
  `fetch_city_track.py` both send one; copy it. Ask `/api/status` before
  believing any refusal.
- **Sheet 15, 노선간 여객환승 실적**, looks like the junction steps `solve.py`
  fits and is a different quantity — 6,290 a day network-wide against steps of
  millions a year. It counts ticketed platform transfers; a tau is a through
  train whose passengers never get off.
- **Using section counts for a step's magnitude** cannot help 대구선 or 경북선,
  the two lines that need it: 대구선's only published boundary is 가천, not on
  the chain, and 경북선's is 점촌 at 5 trains either side.
- **The console dies on Korean text** under Windows cp1252. Write reports to a
  UTF-8 file and read that back.

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
중앙선         good       34   323.1     6380391    10547912    3.3%     12435
전라선         good       18   180.4     7878650     7714928    2.2%     11242
경원선         solved     38    94.4     7502185     7502185    0.0%     20548
장항선         solved     28   154.2     4235851     4235851    8.4%      4320
경전선         solved     33   277.7     5583232     5583232    6.8%      3835
대구선         solved      5    32.3     1623828     1623828    3.6%      3665
충북선         solved     16   115.0     1816583     1816583    3.0%      2985
영동선         solved     17   188.8     1267405     1267405    3.6%      2267
중부내륙선       solved      5    56.9      130011      130011    5.2%       314
정선선         solved      7    45.9       16018       16018   11.4%        25
경부고속선       partial    18   398.2    54453417    95125521    1.4%     74168
수서고속선       partial     5    53.2    18002340    19167014    1.1%     43751
경부선         partial    75   441.7    32909981   105147594    2.6%     33018
호남고속선       partial     5   183.8    10741974    22425971    0.9%     22250
호남선         partial    18   246.7     5227490    16798094    0.6%      9748
강릉선         partial     6   120.5     3801038     4864865    1.8%      7506
태백선         partial    18    95.4      387560      711164    2.3%       808
경춘선         partial    20    80.7        2398        2399    0.7%         7
경의선         partial    24    56.0         820     1795352    0.0%         2
광주선         shaky       3    12.2      292146      496870    7.7%       742
경북선         shaky      11   115.0      191638      191638   23.6%       359
동해선         broken     31   183.7     3765490     4214757   12.8%      6125
```

Stop counts are lower than they look against an older copy of this table: the
operator filter took 46 track-side stations off the chains and the corridor fix
took 태백선's stations off 영동선. `build.py` writes
`data/segments_singleline.geojson`, not the `segments.geojson` the page reads —
both used to write the latter, so whichever ran last decided what the map drew,
and only `solve.py` emits geometry.

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

**A line's type set is not uniform along it.** 호남선 was restricted to
conventional types, and south of 광주송정 that is simply false: 호남고속선 *ends*
at 광주송정 and the high-speed trains that carry on to 목포 run on 호남선's own
metals. `6. 운전` says so outright — 광주송정–목포 is KTX 19 + SRT 9 against
무궁화 7 + 새마을 4 — and none of those passengers reached the map. 목포's 1.55 M
KTX and 0.59 M SRT a year were dropped and the segment was drawn at **743 a
day**, for a stretch carrying 28 high-speed trains. `lines.PART_TYPES` now gives
호남선 those two types at the eight stations south of 광주송정 and nowhere else:
목포 goes to **6,593 a day**, 광주송정–나주 from 1,193 to 9,523, and the mirror
from 3.5 % to 0.6 %.

Two things about that fix are worth not rediscovering.

*It does not generalise from the section counts alone.* Five modelled lines run
types their table denies — 경부선 shows 123 KTX a day 서울–금천구청 — but those
are the same trains 경부고속선 already draws over 서울–광명, and granting them
twice double-counts. The test is whether another modelled line covers the same
trains, and south of 광주송정 nothing does.

*It has to be a span, not a line.* Granting the types line-wide did far more
damage than the bug it fixed, by two separate mechanisms. The types reach
서대전, 익산 and 정읍, whose high-speed traffic is 호남고속선's; and the line's
type set becomes the complete one, which flips its 통과인원 residual from a
one-sided ceiling to two-sided **equality** — so the fit was driven to reach a
published 16.8 M that counts through traffic the rebuild cannot see, and took it
from the only place available. 호남선 went to a 수송밀도 of 28,009, ahead of
경부선, while 호남고속선 collapsed from 5,774 to 2,412 with 정읍–광주송정 at zero
and a negative segment. `spec["full_types"]` now carries "is 통과인원
comparable" explicitly rather than inferring it from set equality, so a part
type cannot silently convert the ceiling into a target.

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

The counts also give the **level**, not just the steps, and the hardest case is
zero. 경원선 shows 전동차 only north of 청량리 and nothing at all north of
소요산 — that section has been shut for the 전철 works — so no 일반열차 runs
there and none of this map's passengers can be on it. Without that rule the fit
drew a flat 16,694 a day the length of the line, out to 백마고지 on the DMZ,
because with 36 of its 39 stops carrying no 승하차 row there was nothing in the
cumulation to make it taper. It now reads 13,200 on 용산–청량리, where the
ITX-청춘 actually runs, and 15–120 beyond. 경원선 is the only line the rule
touches, 31 of its 38 segments.

That is the whole difference between guessing and knowing, and it shows:

| | before | after |
|---|---|---|
| 경부선 mirror | 121.7 % | **2.3 %** |
| 경전선 mirror | 111.5 % | **2.9 %** |
| 동해선 mirror | 90.5 % | **10.5 %** |
| lines with a negative segment | 2 | **0** |
| median mirror, weighted by traffic | — | **1.9 %** |

Every large step that survives is at a real interchange — 삼랑진, 익산, 서원주,
오송, 천안 — and none is at a station the counts run straight through.

### Keeping a part type inside its span

One more residual joined later, for the same reason the train counts did: a
thing that was true, that the fit had no way to know. `lines.PART_TYPES` gives
호남선 the high-speed types south of 광주송정 only, and it follows that their
passengers ride only there — whatever boards inside the span leaves at the
boundary rather than carrying on up the line. Without it stated the fit carried
목포's KTX and SRT north up 호남선's conventional metals instead, lifting
서대전–계룡 by 8,330 a day, and neither suppressing the false anchor at 광주송정
nor any weighting of the existing priors moved it: both answers satisfy junction
conservation equally well.

`contain` states it one-sidedly, since the conventional step at the same station
is free to take more on top. Its magnitude is not fitted — it is the part-type
승하차 summed over the span, 4,175 a day southbound and 4,155 northbound, which
is the 8,330 the line had gained. It takes the median weighted mirror from
1.9 % to **1.4 %**, the best the network has been, and moves 광주선 from 3.3 %
to 1.0 % as a side effect of no longer having to absorb the discrepancy.

It stops about three-quarters satisfied because 호남고속선 cannot receive the
rest — see the entry in [What is not done](#what-is-not-done).

### Sizing an entry flow from the train counts

The same argument once more, at the other end of the same corridor. 호남고속선
begins at 오송 and almost nobody boards there: its riders start at 용산, 서울 or
광명, none of which is on the line, so the entry flow had nothing to size it and
the smallness prior put it on the floor. The line drew 9,416 against a
single-line rebuild of 22,250.

The counts size it. 경부고속선 runs 177 trains a day into 오송 and 127 out the
far side, and the 50 that vanish are exactly 호남고속선's own 오송–익산 count, so
the entry is that share — 28.2 % — of what the feeding line carries in.
`lines.ENTRY_SHARE` names the pairing and `solve.py`'s `entry` term applies it.

Read the share off the *feeding line's own* count either side of the junction,
not by dividing one line's count by the other's. Per segment they do not line
up: 천안아산–오송 spans SR분기 where the SRT join, so `runs_along` gives it the
117 from the near side while 177 arrive at 오송, which would have made the share
50/117 — half again too big.

It is a prior, not a measurement, since trains are not seats: a Honam KTX-산천
seats 363 against a Gyeongbu KTX-1's 935. Two published figures bracket it. The
line's 통과인원 is 23.6 % of 경부고속선's and its train share is 28.2 %, so they
disagree by a fifth rather than a factor. The fit had the line at 12.4 % and now
puts it at 27.5 %, between them.

The profile that comes out tapers the way the service does, with the 전라선 KTX
leaving at 익산:

```
오송 ─25,001─ 공주 ─24,706─ 익산 ─17,961─ 정읍 ─16,083─ 광주송정
```

Halving the weight was tried and is not worth it: 호남고속선 falls to 17,457 and
the network mirror does not improve at all.

### Handing over where the chains do not meet

Junction conservation pairs lines by station name, so it only works where the
handover point is a stop on **both** chains. Where it is not, the traffic does
not go anywhere — it stops existing, silently, because nothing in the report
counts passengers who left the network.

Scanning every line end for one that no other chain names found 수서고속선's.
It ends at 평택지제, where the SRT join 경부고속선, and 평택지제 is 2.93 km from
경부고속선's corridor against a 0.30 km snap radius, so it will never be a stop
on it. The line delivered **47,006 passengers a day** to that end and every one
of them vanished. The signature was plain once looked for — 경부고속선 read
105,490 a day on 광명–천안아산 and *less*, 101,907, on 천안아산–오송, when the
SRT should have joined in between.

`lines.HANDOVER` names the pairing and the conservation pool is keyed by where
the traffic lands rather than where it leaves. Two details matter:

- **The receiving stop is 천안아산, not 광명.** The SRT join south of 광명, so a
  step at 천안아산 puts them on 천안아산–오송 and not on 광명–천안아산. It leaves
  them off the stretch between the real junction and 천안아산, which nothing
  published can fix: 평택분기점 is not a station and has no 승하차 row.
- **The receiving stop must be exempt from the flat-step rule**, or the fix
  cannot work at all. 경부고속선's count changes at SR분기, which likewise has no
  platform, so read at 천안아산 it is 177 either side and looks flat — and a flat
  count forbids a step outright at weight 12. The count is not flat; the station
  it changes at is simply not one this build can draw.

| | before | after |
|---|---|---|
| 경부고속선 천안아산–오송 | 101,907 | **142,156** |
| 경부고속선 수송밀도 | 75,777 | **82,153** |
| 호남고속선 수송밀도 | 20,830 | **32,425** |
| 호남고속선 통과인원 cover | 0.39 | **0.61** |
| median weighted mirror | 2.2 % | **1.8 %** |
| lines with a negative segment | 1 | **0** |

호남고속선 rose because its entry is a share of the very segment the SRT were
missing from. Its rebuilt 통과인원 is now 25.8 % of 경부고속선's against a
published 23.6 %, where before the fault it was 8 %.

**The second handover changes the answer without improving the fit**, and that
is worth being plain about. 호남선's branch off 경부선 is now stated — the step
at 신탄진 is +7,495 a day, 경부선 reading 10,436 on 대전–신탄진 and 17,931 on
신탄진–부강 — and it moved 경부선 from 18,335 to 16,787 and 호남선 from 6,978 to
6,484. But the cost came out at 332.1 either way, to a decimal, and the median
weighted mirror stayed at 1.8 %. The constraint is satisfiable for free; what it
does is choose among optima the fit already considered equivalent. That is the
same shape the junction steps had before the train counts arrived, and it means
the gain here is in justification rather than in anything measurable: the answer
is now the one where the 호남선 trains come off 경부선, instead of whichever the
priors happened to like. Nothing external says the new levels are better.

It also put 정선선's 아우라지–구절리 back to −1 a day, having been +2. That
segment has oscillated between −3 and +2 all through this work; it is a closed
stub carrying 22 a day and the sign of it is noise.

**The mirror cannot see a level error, and must not be read as a confidence
bar.** It compares 하행 against 상행, built from disjoint columns of the same
source, so it tests whether the cumulation is arithmetically sound. It says
nothing about whether the line is carrying the right number of people. The
2026-09-06 work is the demonstration: 호남선's 임성리–목포 went from 743 a day to
6,593 and 호남고속선 from 5,774 to 32,413 — factors of 8.9 and 5.6 — and both
lines held a mirror under 0.5 % before, during and after. A tight mirror on a
line nobody has examined is not reassurance; 동해선, 대구선 and 경북선 are where
that should be remembered.

Nothing in this project has ever been checked against an observed segment count.
Everything is internal consistency plus published aggregates, which is exactly
the regime in which a fit can be tight and wrong. The 혼잡도 figures seoulriders
uses would be the first real external test.

**On the mirror column.** The worst single segment is a bad summary: it is
whichever segment carries almost nobody, where a handful of passengers reads as
100 %. 충북선's 조치원-오송 stub is 0 against 846 — two people a day — while its
other fifteen segments agree to 3 %. The report now leads with the disagreement
weighted by the traffic it applies to, and keeps the worst segment beside it.
Weighted, the median line is at 1.9 % and the trunk lines are at a few tenths of
a per cent — 경부고속선 0.2, 경부선 0.6, 전라선 0.2, 호남선 0.3, 동해선 0.5.

Still imperfect, weighted: 경원선 (24.6 %), 정선선 (15.8 %), 대구선 (13.0 %),
경북선 (8.0 %), 태백선 (5.2 %), 중앙선 (4.0 %). 경춘선 (113 %) and 경의선 (66 %)
have so little 일반열차 traffic left — densities of 10 and 2 — that nothing can be
said about them either way, and 정선선's 21 a day is barely different. For 경춘선
that turned out not to be a quiet railway but an uncounted one — 26 ITX-청춘 a day
against 2,399 riders a year in the sheet — and the map now says so rather than
drawing the figure; see [The thin lines, and which of them are
wrong](#the-thin-lines-and-which-of-them-are-wrong-2026-09-07).

경원선 is the odd one in that list, because its *profile* is right: 13,198 a day
on 용산–청량리 where the ITX-청춘 runs, 120 to 회기, and 14–20 out to the DMZ, which
is what the zero-train rule was built to produce. What disagrees is the level, by
660k a year on the trunk — 하행 2.74 M against 상행 2.08 M. The whole of 경원선's
traffic steps off at 청량리 onto 경춘선, which the section counts confirm outright
(26 trains to 0), and it is the largest step in the network; the two directions
simply do not settle on the same size for it.

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
the yearbook's 영업거리 (OSM runs 0.2–0.7 % short) — but only where the two
measure the same track. See below.

**영업거리 measures the legal line, not the one that gets drawn.** `lines.ENDS`
moves ten lines' endpoints to where trains actually run — 대구선 legally starts at
가천 junction, which has no station, so the chain starts at 동대구 instead — and
the 영업거리 stays behind on the legal extent. Rescaling the drawn chainage to it
then divides one extent by another: 대구선 drew 동대구–영천, 32.3 km, and rescaled
it onto a 가천–영천 26.1, so every segment came out 24 % short. 수서고속선 (평택지제
against 평택분기점), 태백선 (태백 against 백산), 중앙선, 호남선 and 동해선 all had
the same, smaller, version of it.

The rescale is what hid it: the stations stay in the right *proportion* of the
line whatever the scale, so the segment table looks perfectly ordered and only the
kilometres are wrong. Nothing published replaces the figure — the yearbook has no
station-to-station distances anywhere — so those lines now keep OSM's own
chainage, which the lines whose extents *do* match show to be good to a few tenths
of a per cent. `resolve()` sets `scaled` false for them.

Worth being clear about what this did and did not affect: 수송밀도 is
passenger-km over line length, and both scale together, so no density moved by a
digit. What was wrong was every segment's stated length — a 6.5 km stretch of
대구선 was labelled 5.2 — and with it the line lengths in the report and in the
page's info panel.

`check.py` is the check that found it, and it runs against `data/segments.geojson`
in a couple of seconds rather than behind an eight-minute solve: drawn length
against 영업거리, whether consecutive segments actually meet, and whether anything
carries a negative load. It is worth keeping — it caught `own_component`
regressing 호남고속선 from −0.5 % to −50 % and 태백선 to −37 % in the same run that
fixed 영동선, which nothing else would have noticed. Lines whose ends `ENDS` moved
are judged in km rather than per cent, since the gap is a stub of real length: the
largest honest one is under ten km, against the 38 and 92 km the misroutes cost.

**A line's metals do not always arrive in one piece.** OSM has 영동선 as a
2,687-node run from 봉화 to 강릉 plus a stranded 0.98 km stub at 영주 that comes
within 182 m of the rest and never touches it — the station throat between them
is mapped under another name. Anchoring the search on the node nearest the end
station put it on that stub, from which the line is unreachable along its own
track, so 영동선 was drawn 150 km round by 중앙선 and 태백선 through 영월 — over
태백선's own route, picking up its stations, and leaving eight of 영동선's real
ones (봉화, 춘양, 임기, 현동, 분천, 승부, 석포, 철암) with no track to sit on.
`own_component()` anchors on the run that comes nearest *both* ends instead, but
only where that costs it almost nothing: 호남고속선's track genuinely arrives in
two halves and wants the old behaviour, where the foreign-track penalty bridges
the gap. 영동선 now draws 188.8 km against a published 188.9 with no foreign
track at all.

**A name that does not match is a station that carries nobody.** The passenger
tables and OSM do not always spell a station alike, and where they differ the
station's traffic reaches no line at all — silently, since a missing key just
reads as zero. Sweeping every high-speed row against the chains found **10.4M
boardings a year, about 28,500 a day, belonging to no line**, in four kinds:

| | per year | |
|---|---|---|
| 용산 | 4.5M | on no high-speed chain; see ENTRY_SHARE above |
| 김천구미, 신경주 | 3.0M | spelling |
| 수원, 영등포, 구포, 밀양, 경산 | 1.6M | KTX on 경부선's metals |
| 행신 | 0.9M | 행신선 is not modelled |

`lines.STATION_ALIAS` fixes the plain one: the yearbook writes 김천구미 and OSM
writes 김천(구미), the station is already on 경부고속선's chain, and 1.2M a year
were falling down the gap. Merged rather than overwritten, since a target name
can already have a row of its own — 김천 is a different station 10 km away, on
경부선.

It is not free. That stop carried no flow at all before and now carries 1.4M, a
step its neighbours have to absorb, and the network's median weighted mirror
went from 1.4 % to 2.2 % — 경북선, which starts at the other 김천, from 8.1 % to
11.0 %. Worth it: the alternative is discarding a million real passengers to
keep the residuals tidy. Isolated by re-running with the entry weight halved,
which changed neither figure.

**신경주 cannot be aliased**, and this is the trap. OSM's 경주 node *is*
신경주 — the yearbook's own 경주 row is 164 passengers of residue from the old
station closing in December 2021 — but that node sits on 동해선's and 중앙선's
chains and not on 경부고속선's. Aliasing 신경주 to it would hand 1.8M of KTX and
SRT to two lines whose trains never carried them. It needs the station added to
the right chain, which is a different job.

**Snap radius alone is not enough.** 천안아산 is a 경부고속선 KTX station about
100 m from 아산 on 장항선; snapping by distance pulled its 720k arrivals into the
장항선 cumulation and broke the line. Filtering against the roster fixes it —
the roster gives each station exactly one *home* line, so a station reporting
traffic whose home is elsewhere is somebody else's. The line's own two endpoints
are exempt: 익산's home is 호남선 but it is still where 장항선 ends.

That same roster fact is how junction ends are detected at all — a station
appears on only one roster, so "not on this line's roster" *is* the junction
test.

**Somebody else's railway snaps on too.** The 부산 도시철도 follows the old 경부선
alignment through 개금 and 주례, Seoul's line 4 sits over it at 신용산 and 삼각지,
and 서울역 is a second node for 서울 — 46 stops in all, splitting corridors at
places no train stops. `membership.load_stations()` drops them on the OSM
`operator` tag, which is the only thing that separates them: a 광역전철 station is
still 한국철도공사's, so 노량진, 구로, 금천구청 and all of 경춘선 stay. They are
places people really travel between, and the map should draw the corridor there
even though the yearbook counts 일반열차 only and has no row for them. None of
the 46 carried a single passenger, so the numbers did not move at all — 경부선's
수송밀도 went from 20,051 to 20,064 and the fit's cost was identical.

Each segment is then sliced out of its line's own corridor by chainage, ends
interpolated, so consecutive segments join exactly — the worst gap network-wide
is 0 m.

## Station bubbles

`build_stations.py` writes `data/stations.geojson`: every station's 승하차,
boardings plus alightings, over 365. It is the one number on the map that is
**published rather than reconstructed**, and it is the very number the segment
lines were built out of — so the map can show the input and the inference at
once, which is the honest way to present a derived figure.

`build_metro_stations.py` adds `data/metro_stations.geojson`: **887 intracity
station complexes** — 625 around Seoul, 108 in Busan, 91 in Daegu, 22 in
Daejeon, 20 in Gwangju and 21 on the Busan–Gimhae LRT. The map draws all
station bubbles white. Each city outside Seoul uses the same published weekday
gates and reference month as its segment model. Seoul sums the completed
weekday OD matrix by origin and final
destination, so a trip contributes at its two ends and not again at transfers.
Its station totals inherit the Seoul model's uneven operator coverage.

Placement is the only real work. A name can carry several OSM nodes (동대구 and
수서 have one per operator), so each takes whichever node sits closest to the
network that actually got drawn — which both locates it and confirms it is on
the map at all. Which lines call there comes from the segment table rather than
from proximity, so the page can light up a line's stations exactly.

Three station names had to be merged by hand (`ALIAS`), the interesting one being
신경주: OSM's 경주 node *is* 신경주, and the yearbook's own 경주 row is 164
passengers of residue from the old station closing in December 2021.

## Files

```
lines.py            yearbook parsing + the line-name table + anchor detection
                    TYPES / PART_TYPES: which trains run on a line, and where
                    STATION_ALIAS: yearbook spellings -> OSM ones
                    ENTRY_SHARE: what sizes a line fed from another line
                    HANDOVER: where two chains meet at a place neither names
build.py            the reconstruction over all lines; --line NAME for one
solve.py            the whole network fitted at once; --line NAME for one
build_stations.py   역별 승하차 as map bubbles -> data/stations.geojson
build_daegu_busan.py  Daegu/Busan-Gimhae gates -> data/daegu_busan_segments.geojson
build_metro_stations.py  intracity entries + exits -> data/metro_stations.geojson
check.py            checks data/segments.geojson without re-solving; --line NAME
                    geometry, the mirror, the level against the published 인거리,
                    and passengers per train against the published 운행횟수
check_cities.py     the five city models against published trip length and 혼잡도
compare_builders.py  build.py against solve.py, per line; --line NAME per segment
fetch_city_track.py  Overpass: city metro track shapes -> data/osm_city_track.json
city_track.py       fits those metals to each city model's stations
check_metro_fresh.py  is the seoulriders import stale? exit 1 if so
yearbook_extra.py   the yearbook sheets nothing else reads: 인거리 by distance
                    band, city trip lengths, city peak-crowding segments
test_part_types.py  PART_TYPES / THROUGH_ENDS plumbing, yearbook only, ~1 min
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

## Seoul metro integration (2026-09-05, re-imported 2026-09-06)

Before rerunning any build, check that nobody else is already running it.

**Re-import after seoulriders changes, and check for it.** seoulriders is worked
on in the same tree and its model moves; the import here is a snapshot and goes
stale silently, since a stale GeoJSON draws perfectly well. It went unnoticed
for two days until 수인분당선 looked far busier in seoulriders than on this map.
The generated file records `source_build` and `source_sha256` for exactly this
— compare those against `../seoulriders/data/` rather than trusting the mtime,
and re-run `build_metro.py` **and** `build_metro_stations.py`, which read the
same source.

The 2026-09-06 re-import was not a small correction. seoulriders had gained
`split_boardings` and a `crush_cap` of 1.5, which redistributes load off the
lines that were over capacity:

| line | change |
|---|---|
| 서해선 | +347 % |
| 수인분당선 | **+244 %** |
| 경강선 | +206 % |
| 경의중앙선 | +131 % |
| 1호선 | +64 % |
| 9호선 | −9 % |

Modelled riders went from 6.85M to 8.38M a weekday over the same 759 segments.

`python build_metro.py` imports the full weekday `../seoulriders/data/stats.json`
and `link_shapes.json` into `data/metro_segments.geojson`. The page combines this
with the intercity GeoJSON at load time; neither intercity output nor the Seoul
source files are changed. The import takes less than a second locally and needs
only the Python standard library. `--source DIR` selects another source folder.

The import covers **all 22 lines in seoulriders: 759 station-pair segments**.
This includes Lines 1–9, Airport Railroad, Incheon, and the suburban lines.
Hourly routed
loads are summed and reverse links combined, retaining separate `daily_down`
and `daily_up` fields. Metro directions follow increasing station code except
the closing Line 2 ring edge (0243 → 0201). Names are namespaced with 수도권 to
avoid collisions with infrastructure lines. No annual passenger counts are
invented. The importer rejects sampled and non-weekday input.

Express loads (`hx`) are a subset of total loads (`h`), so subtract them first,
then distribute them over the shortest path through that line's local-stop
graph, weighted by geographic distance. Each express rider counts once on
**every** intermediate segment traversed. Local service is identified from
train counts (`n - nx`), including services with no local passengers. All 188
express links in the current source resolve within their own line; unresolved
links stop the import instead of silently losing riders. This assumes express
and local trains follow the same corridor. It does not combine metro service
with intercity service on shared infrastructure.

These are **modelled weekday** flows from the existing Seoul model, seeded by
2023 OD and reweighted to weekday station counts, whereas intercity is a 2022
annual average. They share a passengers/day scale, not a reference period.
Source build metadata and input SHA-256 fingerprints are preserved in the
generated file. The importer refuses to write if its source files changed during
the read/build and replaces the output atomically. Metro bubbles are built from
OD origins and final destinations, rather than routed train boardings, so
transfers are not double-counted.

Track shapes are reused without fetching OSM; 221 of the 759 segments currently
have only straight-line geometry. A reverse-direction shape is reused when it
is the only available one. Busan, Daegu, Daejeon, Gwangju and Busan–Gimhae have
separate experimental models below.

`python -m unittest test_metro` checks daily units, reverse pairing, ring closure,
express accounting across intermediate stops, isolation between lines, invalid
sources, and reverse geometry reuse.

## Busan pilot (2026-09-05)

Busan Lines 1–4 now load from `data/busan_segments.geojson`: **114 platforms,
108 station complexes, 110 physical segments**. The roster labels them `(est.)`
and every segment exposes the model's scenario range, including directional
ranges. These are **estimated OD flows**, with less evidence than Seoul's
measured OD seed. They are not calibrated against observed segment loads.

Sources, downloaded anonymously by `python fetch_busan.py` into `data/busan/`:

- [Busan station/day/hour gate counts](https://www.data.go.kr/data/3057229/fileData.do),
  January–July 2026 in the downloaded file; model uses the 23 Monday–Friday
  dates in July (no holiday exclusion).
- [Station sequence, distances and running times](https://www.data.go.kr/data/3033564/fileData.do),
  reference 2025-03-25.
- [Station coordinates and English names](https://www.data.go.kr/data/15043686/fileData.do),
  reference 2021-02-26. All 114 platform codes join to the distance table.

The source provides 112 reporting platforms. Shared-gate interchange platforms
need not each report separately. Aggregate **all gate rows at the same station
complex** before fitting; this also prevents counting a transfer as a new gate
entry. Every one of the 108 complexes has both entry and exit totals, and every
reporting platform has complete daily coverage. Routing can start/end at any
platform in the complex; it does not infer the entry gate's eventual train line.

`python build_busan.py` takes about a second. It builds the four-line network
from the published sequence and running times, links transfer platforms with a
five-minute penalty, and finds shortest-time paths. The OD seed is
`exp(-journey_minutes / decay_minutes)`, with same-complex trips forbidden.
IPF fits station entries and exits; exits are scaled by **1.001906** to balance
the observed 951,453 daily entries and 949,644 exits. Every trip is then counted
on each segment it passes. Metro directions run from lower to higher station
code, and back.

The central **20-minute** decay is still a provisional assumption rather
than a fitted parameter, and Busan is now the only city model of which that is
true — see [The city models had a published trip length all
along](#the-city-models-had-a-published-trip-length-all-along). 부산교통공사
writes `공사에서 관리하지 않는 데이터임` in the yearbook sheet that measures
every other operator's trip length, so there is nothing to fit against. The
10/20/30-minute scenarios produce mean journeys of **7.03/10.08/11.24 km**. The full scenario range spans a median **39.3%** of the
central segment load (maximum **59.9%**). These ranges test assumptions; they
are **not confidence intervals**. In the central run, station flow conservation
and fitted gate residuals are both below 0.000003 passengers/day. That is an
accounting check, not independent validation of the OD estimate.

`data/busan/model_report.json` records counts, balance factor, scenario results,
checks, and source fingerprints. `python -m unittest test_busan test_metro`
checks OD margins, forbidden self trips, transfers, direction, disconnected
networks, sensitivity and the Seoul importer.

Current limits: straight lines between published station coordinates (the OSM
route download returned HTTP 406; no workaround was attempted), one shortest
path per station pair, uncalibrated trip-length assumptions, and no Donghae
suburban rail. Busan–Gimhae is included through the separate model below, but
the Busan model cannot infer cross-system trips from the two operators' gate
totals. Average journey distance and segment crowding, named as the evidence
this needed, both turned out to be published — but not for 부산교통공사, which
files neither. Busan is the one city model still uncalibrated, and measured OD
or track geometry remains the next useful thing for it.

## Daejeon and Gwangju pilots (2026-09-05)

`python build_small_cities.py` writes `data/small_city_segments.geojson` for
Daejeon Line 1 and Gwangju Line 1: **42 stations and 40 segments**. Both systems
are single lines, so each OD pair has exactly one route and no transfer decision.
The unknown part is how far riders travel.

The model uses each operator's July 2026 station/day/hour gate counts, restricted
to all 23 Monday–Friday dates. It seeds OD with `exp(-distance / decay_km)`, fits
entries and proportionately balanced exits with IPF, then counts every trip on
each intervening segment.

**The decay is fitted, not assumed.** It was 5/10/15 km with the 10 km run
drawn, which put the mean trip at 5.52 km in Daejeon and 5.14 km in Gwangju
against published figures of 6.69 and 7.16 — both *outside* the old scenario
range, in the same direction. `calibrate_decay` now solves for the decay that
reproduces the published mean, and the tooltip range is ±15 % around it. The
median scenario span falls from **32.7 % to 19.5 %** in Daejeon and from
**31.7 % to 7.3 %** in Gwangju. It is still an assumption range and not a
confidence interval.

Gwangju's collapse to 7.3 % means the opposite of what it looks like, and
Daejeon is close behind it: their published trip lengths sit at or beyond what
these networks can produce under any distance decay, so the fits are pinned
against a boundary. Read the two entries in [the sheets
section](#the-city-models-had-a-published-trip-length-all-along) before quoting
either band.

Station order, coordinates and English names come from the national KRIC urban
rail station workbook. Four shortened Gwangju ridership names are joined through
explicit aliases. Straight distances between adjacent coordinates are scaled to
the operators' published 20.5km line lengths; the visual geometry remains straight.
Daily entries are **110,651** in Daejeon and **51,756** in Gwangju. Balanced gate
and routed station-flow residuals are below 0.000006 passengers/day in all six
scenarios. As with Busan, this proves accounting consistency, not the inferred OD.

Sources downloaded by `python fetch_busan.py`:

- [Daejeon station gate counts](https://www.data.go.kr/data/15060591/fileData.do)
- [Gwangju station gate counts](https://www.data.go.kr/data/15060048/fileData.do)
- [National urban-rail station coordinates](https://www.data.go.kr/data/15093755/fileData.do)
- [Daejeon Line 1 operating length](https://www.djtc.kr/kor/page.do?menuIdx=461)
- [Gwangju Line 1 operating length](https://www.grtc.co.kr/subway/contents/operationStatus)

`python -m unittest test_metro_stations test_small_cities test_busan test_metro`
checks bubble origin/destination accounting, linear route assignment,
station-flow conservation, IPF margins, transfer handling, scenario sensitivity
and the Seoul importer.

## Daegu and Busan–Gimhae pilots (2026-09-06)

`python build_daegu_busan.py` writes `data/daegu_busan_segments.geojson`:
**94 Daegu platforms grouped into 91 station complexes and 91 segments**, plus
the **21 stations and 20 segments** of the Busan–Gimhae LRT. Daegu Lines 1–3
are one connected model, with a 0.5km transfer cost at 명덕, 반월당 and
청라언덕. Busan–Gimhae is fitted separately because its reference period differs
from both Daegu and Busan Lines 1–4.

Daegu uses all 22 Monday–Friday dates in June 2026, averaging **442,379 entries
and 438,435 exits per day**. Busan–Gimhae uses all 23 Monday–Friday dates in
December 2025, averaging **51,347 entries and 50,939 exits per day**. Holidays
are not excluded. Both models use straight-line station distance, seed OD with
`exp(-distance / decay_km)`, balance it to the gates, and assign it to shortest
paths.

**The decay is fitted to the published mean trip length**, not assumed — see
[the sheets section](#the-city-models-had-a-published-trip-length-all-along).
The drawn run was the 10 km decay, giving 9.12 km in Daegu against a published
8.21 and 6.81 km on Busan–Gimhae against a published 8.06; Daegu was drawn too
heavy and Busan–Gimhae too light. Both now reproduce their published figure
exactly, at fitted decays of 8.21 km and 19.5 km, with the scenario range ±15 %
around it. The median span falls from **42.4 % to 29.8 %** in Daegu and from
**43.2 % to 33.6 %** on Busan–Gimhae. These are sensitivity checks, not
confidence intervals.

Busan–Gimhae's upper scenario needs a 135 km decay, which is very nearly a
uniform OD — the line is short enough that its published mean trip is already
most of the way to what the gates alone would give.

Sources downloaded anonymously by `python fetch_busan.py`:

- [Daegu station/day/hour gate counts](https://www.data.go.kr/data/15002503/fileData.do)
- [Busan–Gimhae station/day/hour gate counts](https://www.data.go.kr/data/15105181/fileData.do)
- [National urban-rail station coordinates](https://www.data.go.kr/data/15093755/fileData.do)

`python -m unittest test_daegu_busan test_metro_stations test_small_cities
test_busan test_metro` checks the source rosters and Daegu transfer grouping in
addition to the shared model invariants. Geometry is straight between stations.
Trip length is no longer a gap — it is fitted to the published figure and
checked by `check_cities.py` — so what remains is observed OD to constrain route
choice, and real track shapes.

## The sheets nobody had opened (2026-09-06)

Everything above was built from seven sheets. The passenger workbook has
**nineteen**, and the yearbook zip holds two further sections — `2.도시철도`
and `3.광역철도` — that nothing in this project had ever looked at. Four of
those sheets bear directly on questions recorded here as open, and the
calibration this model was described as lacking turned out to be in the file it
had been reading all along.

`yearbook_extra.py` reads them. `check_cities.py` is the new check they make
possible.

### The level has an external check now

Every check here was on the *shape* — the mirror, the geometry, positivity.
The level had none, which is why [What is not done](#what-is-not-done) recorded
경부선 at 33,018 from `build.py` and 20,051 from `solve.py` (17,343 since the
train-count fix below) with "nothing yet
says which is right".

Passenger sheet 14, `거리별 여객 수송실적`, publishes 2022 passengers **and
인거리** by distance band and train type. That 인거리 is not the per-line figure
`probe_ingeori.py` threw out for not being attributed to track — it is a plain
network total, 25.84 billion passenger-km over 141.0M journeys, mean trip 183 km.
Summing `load × length` over our own segments is the same quantity.

| | annual passenger-km | of published |
|---|---|---|
| published (sheet 14) | 25.84 bn | — |
| `build.py`, single-line | 23.99 bn | **92.8 %** |
| `solve.py`, network fit | 21.96 bn | **85.0 %** |

Both land under, which is the only direction they legitimately can: the rebuild
sums a subset of the traffic and knows it misses more — 10.4M boardings a year
on no chain, the high-speed lines seeing 0.54–0.61 of their 통과인원. Coming in
*over* would mean inventing passenger-km, and `check.py` now says so outright
if it ever does.

`build.py` is closer by about 1.9 billion passenger-km, and **that turns
out not to mean what it looks like** — the gap is made of segments `build.py`
draws at up to 2.8 times what a train holds. See [Passengers per
train](#passengers-per-train-and-why-the-인거리-verdict-was-backwards), which
was written after this and reverses its reading. Left standing as it was
first argued, because the correction is the useful part: a total can be closer
because two errors cancel, and nothing about the total itself says so.

What the check is good for regardless is a ceiling. A rebuild that sums a
subset of the traffic may not exceed the published figure, and `check.py` says
so outright if one ever does. It costs nothing, so it is a regression test as
much as a finding.

### The city models had a published trip length all along

`도시철도` sheet 10, `통행거리별 여객 승차실적`, gives boardings and 인거리 in
eight distance bands **per city line**. The mean trip falls straight out of it —
which is the number [Busan pilot](#busan-pilot-2026-09-05) called "a provisional
assumption, not a fitted parameter" and `todo.txt` listed as the thing to
calibrate.

`build_busan.calibrate_decay` bisects the decay until the fitted OD reproduces
it. The decay is no longer chosen; it is measured.

| | published mean trip | old central | new central | fitted decay | median scenario span |
|---|---|---|---|---|---|
| Daegu 1–3 | 8.21 km | 9.12 km | **8.21 km** | 8.21 km | 42.4 % → **29.8 %** |
| Busan–Gimhae | 8.06 km | 6.81 km | **8.06 km** | 19.5 km | 43.2 % → **33.6 %** |
| Daejeon 1 | 6.69 km | 5.52 km | **6.69 km** | 38.6 km | 32.7 % → **19.5 %** |
| Gwangju 1 | 7.16 km | 5.14 km | 6.57 km | at the ceiling | 31.7 % → **7.3 %** |

Three of the four old central runs sat outside their own published scenario
range, all in the same direction: real journeys are longer than the 10 km decay
assumed, so mid-line segments were drawn too thin. Only Daegu was too long.

Two things this turned up that are worth keeping.

**Bands longer than the line are not trips on it.** Gwangju reports 8.4 % of
its 인거리 beyond 20.5 km on a 20.5 km railway, including 290,907 passengers
averaging 33.9 km. That is integrated bus-and-rail or a filing error; either
way it is not on these metals. `yearbook_extra.urban_mean_trip` drops bands
whose *lower* bound is already past the network and keeps one that straddles
the end.

**Gwangju cannot be fitted at all, and the reason is interesting.** Even after
dropping the impossible bands, its published mean of 7.16 km is longer than the
network can produce under *any* distance decay — the ceiling, with every pair
weighted alike and only the gates shaping the OD, is 6.57 km. A decay can only
ever make trips shorter than uniform, so this is not a parameter that needs
tuning: **distance-decay gravity is the wrong model family for Gwangju.** Its
riders make disproportionately long trips, which a gravity model cannot
represent. The central run is pinned at the ceiling and still falls 0.59 km
short, so Gwangju's mid-line segments remain understated.

Its scenario span collapsing from 31.7 % to 7.3 % therefore means the opposite
of what it looks like. The band is narrow because the fit is against a
boundary, not because the answer is well determined. `trip_length_note` in the
model report says this in words, and it should reach the page rather than
sitting in a JSON file.

**Daejeon is nearly the same story without crossing the line.** Its published
6.69 km is 95 % of its own 7.05 km ceiling, so the +15 % scenario is
unreachable and `scenario_target_unreachable` records it. Distance decay is
doing almost no work on a line that short.

Busan lines 1–4, the largest city model, writes `공사에서 관리하지 않는
데이터임` in this sheet and still has no published trip length. It keeps its
20-minute assumption.

### The city models now have an output check

`도시철도` sheet 13, `연도별 최대 혼잡도`, names the busiest segment on every
city line for every year from 2011, with the half-hour it applies to. Five
models that had **no check on their output at all** — their gate residuals are
an accounting identity and their scenario spans a sweep over an assumption —
now have one that can falsify them.

`python check_cities.py`:

```
                  model's busiest      published busiest       혼잡도   apart
대구 1호선          반월당-중앙로           현충로→영대병원            77.9%   4 of 34
대구 2호선          청라언덕-반월당          반고개→내당              73.6%   2 of 28
대구 3호선          서문시장-청라언덕         북구청→달성공원           110.1%   2 of 29
부산김해경전철       공항-덕두              서부산→괘법              92.0%   2 of 20
광주 1호선          농성-화정              양동시장→돌고개            69.2%   2 of 19
대전 1호선          탄방-시청              탄방→용문               126.0%   1 of 21
부산 1호선          범내골-서면             부전→서면                95.0%   1 of 39
부산 2호선          전포-서면              부암→서면                78.0%   1 of 42
부산 3호선          물만골-연산             종합운동장→거제           108.0%   2 of 16
부산 4호선          동래-수안              충렬사→낙민               45.0%   2 of 13
```

**Median two segments apart on lines of 13 to 42.** Read the comparison
carefully before treating that as agreement or disagreement: these are not the
same quantity. 혼잡도 is a rate in one half-hour on the peak approach, and
these loads are a daily total, which peaks at the busiest interchange instead.
Every model sits *centre-ward* of the published peak, consistently, which is
what that difference predicts. What the check would catch is a model putting
the peak at the wrong end of a railway, and none does.

Measure it in segments along the line rather than by whether the two name the
same one. The first version of this check scored "one stop out" the same as
"opposite end" and reported 0 of 10 agreeing, which said nothing useful.

The sheet also abbreviates station names the way the KRIC portal does —
괘법르네시떼 and 서부산유통지구 appear as 괘법 and 서부산 — and an exact
comparison reports "not on chain" and silently stops checking. `resolve_pair`
matches a published name that is a *unique* prefix of a chain station and
refuses to guess when it is ambiguous.

### Two sheets that look like evidence and are not

Recorded so they are not rediscovered at cost.

**Passenger sheet 15, `노선간 여객환승 실적`** — 1,137 rows of line-to-line
transfers at named stations, 2022. It reads exactly like the junction steps
`solve.py` spends most of its residuals inferring, and it is a different
quantity. The network total is 2.3M a year, **6,290 a day**, against steps of
millions a year at a single junction; the largest single row is 동대구
경부선→경부선 at 387 a day. It counts ticketed platform transfers, while a tau
is a through train whose passengers never get off. Useful only as confirmation
that a junction's 승하차 is almost entirely local traffic, which the share
priors already assume.

**Passenger sheet 3, `선별 여객수송` by train type** — tempting, because the
whole `partial` verdict exists from a 통과인원 that counts train types the
rebuild excludes, and this is per line *and* per type. The attribution does not
line up: 전라선 reads 3.27M against a published 7.71M line total and a 7.88M
rebuild, and 경부고속선 reads about 37M against a 95M 통과인원 and a 54M
rebuild. It assigns a journey to one line by some rule that is not "touched
this line's metals", and until that rule is known it cannot replace the
ceiling. Left alone deliberately.

### Passengers per train, and why the 인거리 verdict was backwards

The 인거리 total puts the single-line build at 92.8 % of the published
passenger-km against the network fit's 85.0 %, and the obvious reading is that
`build.py` is nearer the truth. **That reading is wrong, and the way it is
wrong is the useful part.**

`compare_builders.py` differences the two per line. The gap is not spread
about: 경부선 alone is 2.5 of the 4.1 billion passenger-km the fit gives up,
with 경원선 0.65, 중앙선 0.41 and 호남선 0.37 behind it, while the fit gains
1.3 on 경부고속선 and 0.7 on 호남고속선. So the question is really about four
conventional lines, and 경부선 most of all — the fit puts it at 17,343
수송밀도 and the single-line build at 33,018, a factor of two on the busiest
railway in the country.

`6. 운전` decides it. It publishes trains a day on each section, so a segment's
load over its trains is the average number of people aboard, and that cannot
exceed what the train has room for. Nothing about the reconstruction enters the
comparison.

| | segments over capacity | worst | negative segments |
|---|---|---|---|
| `solve.py`, network fit | **0** | — | 0 |
| `build.py`, single-line | 20 | 1.4× | 3 |

And on the segment the whole disagreement turns on:

```
경부선 지천-신동, 30 trains a day each way (무궁화 19 + 새마을 11), ~432 seats

   single-line   36,462 명/일    608 per train   1.4x capacity
   network fit   15,195 명/일    253 per train   0.6x
```

Eight consecutive segments of the single-line build need half again what a
무궁화 holds, every train, all day. Standing tickets are 1.44M of 66.8M KTX
journeys — 2.2 % — so this is not a crowded train, it is a wrong number.
**The network fit is right about 경부선 and the single-line build is not.**

Which means the 인거리 total is closer for `build.py` **because two errors
cancel**. Both builders undercount — traffic that reaches no chain, high-speed
riders boarding off-line — and `build.py`'s overcount on 경부선, 중앙선 and
경원선 happens to fill the hole. An aggregate can be right for the wrong
reason, and only a per-segment check catches it.

경원선 is the same story in miniature and was visible in the per-line table all
along: the fit gives it 1,825 수송밀도 against 20,548, which reads as the fit
losing two thirds of a billion passenger-km. It is the fit applying the
zero-train rule to the 31 of 38 segments where no 일반열차 runs, exactly as
[the section counts](#what-settled-it-the-published-train-counts) intended.

**The counts are per direction**, and getting that backwards costs a factor of
two, so it is worth stating how it was settled. 수서고속선 settles it with no
model involved at all: SRT carried 19.56M journeys in 2022, every one of which
crosses that line since every SRT train starts or ends at 수서, and the sheet
gives it 60 SRT a day. Read as both directions that is 893 people on a
410-seat train, which is impossible; read as one it is 447, or 109 % of seats,
which is an ordinary busy service. Service levels agree — SR ran about 40
Gyeongbu and 20 Honam round trips a day in 2022, which is the 40 and 20 the
sheet gives those sections. So does the network as a whole: the published
인거리 over twice these counts is about 280 a train against a fleet averaging
some 500 seats, a load factor near half, where the both-directions reading
would put the entire network above 100 % of seats all day, every day.

The ordering between two outputs does not depend on the reading either way,
since both are divided by the same counts.

One known false positive remains: 경부고속선 천안아산-오송 reads 117 trains
where 177 run, because the count changes at SR분기, which has no platform —
the same fact that [the handover](#handing-over-where-the-chains-do-not-meet)
had to be exempted from the flat-step rule for.

### The train counts were being double-counted

Found while building the check above, and it turned out not to be cosmetic.
`6. 운전` has one 새마을 column covering both 새마을 and ITX-새마을 services,
and `frequency.py` exploded each column into the passenger types it serves, so
a line whose types are `CONVENTIONAL` added those trains twice. 경부선
서울-금천구청 came out at 92 trains a day against the 새마을 26 + 무궁화 40 = 66
that actually run — a 40 % inflation, on every conventional line.

The expectation was that nothing would move. The zero-train rule asks whether a
total is zero, and the sign rule which of two totals is larger; a consistent
double-count changes neither, and `ENTRY_SHARE`, the one place a magnitude is
used, pairs high-speed lines whose KTX and SRT have columns of their own.

**It moved 311 of 424 segments**, measured by re-solving today's tree with
`frequency.total` reverted and nothing else changed. That is worth doing
rather than diffing against an older output: this tree is worked on by more
than one session at a time, and an older `segments.geojson` can differ for
reasons that are not yours. The flat-step rule was the route: it holds a
junction step at zero where the count runs straight through, and whether two
totals are *equal* is not preserved by a double-count. A section pair like
새마을 10 + 무궁화 30 against 새마을 20 + 무궁화 10 reads 50 and 50 doubled —
flat, so a step is forbidden at weight 12 — and 40 against 30 undoubled, where
a step is not only allowed but has its sign fixed. Some junctions were being
forbidden a step they should have had, and others allowed one they should not.

Correcting it moved the network's passenger-km by −0.5 %, and it removed the
last negative segment: **0 of 22 lines now carry one**, where 정선선 had a
one-passenger negative before. Every segment of the fit now also fits inside
its trains.

`frequency.total()` counts each column once, matching it against the types
wanted rather than expanding it. Per line it made 경의선's mirror 28.0 % ->
17.9 % and 정선선's 18.5 % -> 26.8 %, the second being twenty-one passengers a
day on a line the report already discounts, and left the median at 1.8 %.

### The mirror without an eight-minute solve

`check.py` now reports it. The segments carry `down` and `up`, so the
project's headline quality number -- the traffic-weighted disagreement between
the two profiles -- can be read off the output file in seconds instead of
waiting for `solve.py` to finish and print it. The per-line figures agree with
solve.py's own report exactly; the median needs the same definition numpy uses,
the mean of the two middle values on an even count, or it lands a few tenths
out.

### More than one session works in this tree

`lines.py` changed under this work at 20:35 on 2026-09-06, from another session
in the same checkout. It cost a wrong conclusion before it was noticed: the
frequency fix was first attributed by diffing against a `segments.geojson`
written three hours earlier, which is a diff against *every* change since,
not against yours.

The habit that fixes it is cheap. Isolate a change by re-running with only that
change reverted, on the tree as it stands now, writing to a scratch path --
`solve.py --out` writes the geojson elsewhere, so nothing in
`data/` is touched and two sessions cannot clobber each other. Diffing two
runs you made yourself minutes apart is the only comparison that means what it
looks like. In the same vein: check a data file's mtime before blaming the
code.

The lesson is worth more than the fix. A published number that is only ever
compared with itself can be wrong indefinitely — this one had been read for
zero-ness, equality and sign, and all three survived a 40 % error. It is worth
asking, of any figure this project leans on, whether anything has ever read it
in units.

### Track this map's trains do not run on

경원선 was drawing a railway that was not there. North of 청량리 the intercity
layer read 14 to 120 passengers a day all the way to 백마고지 on the DMZ, and
that is a claim about ridership on track where the metro layer separately and
correctly draws 190,035 a day. The zero-train rule already knew — it holds 31
of 경원선's 38 segments near nil — but "held near nil in the fit" and "drawn as
carrying fourteen people" are different statements, and only the first was
true.

`6. 운전` separates the two reasons, and the map has to say which. Its 전동차
column is 광역전철, deliberately left out of every intercity total because that
service has no 승하차 row — but it is exactly what tells a section with no
intercity service apart from a section with no service at all:

```
경원선 청량리-광운대    전동차 184   intercity 0     commuter rail only
       광운대-동두천    전동차 111   intercity 0     commuter rail only
       동두천-소요산    전동차  38   intercity 0     commuter rail only
       소요산-신탄리    (nothing)    intercity 0     nothing ran in 2022
       신탄리-백마고지   (nothing)    intercity 0     nothing ran in 2022
```

`frequency.service_along` returns both counts per segment, `solve.py` writes
`service` (`commuter` or `none`) and `commuter_trains` onto the segment, and
the page draws those in a grey of their own at a constant width, with a legend
row and a tooltip that says what runs there instead of quoting a rider count.
Constant width matters: drawn at the width its number implies, a no-service
segment is a hairline, and a hairline still says "almost nobody rides here".

24 segments come out `commuter` and 7 `none`, all on 경원선, which is the only
line the rule touches.

**The grey is an authored colour, not the line's own dimmed**, and it has a
legend entry for that reason — a dimmed line still reads as that line, quietly.
It is also lifted well clear of the basemap's own greys, or it reads as a road.

Where the metro layer covers the same track the metro line is drawn on top and
wins the hover, so a reader at 동두천-소요산 gets Seoul line 1 and its real
4,446 a day rather than the grey. That is the right outcome and it means the
`commuter` tooltip is a fallback for track the metro layer does not reach.

### Two width changes

**The viewport cap is gone.** It scaled every line down whenever the thickest
visible one passed 40px, so the 경부 corridor could not overwhelm the map. But
the corridor overwhelming the map is the finding, and a cap that rescales on
pan turns a fixed quantity into one that depends on where the reader is
looking. The thickness slider already covers wanting it smaller. Its `autocap`
toggle and the `idle` handler that recomputed it are gone with it.

**The station hit floor is now a curve.** A flat 9px grab radius meant a
station drawn at a fraction of a pixel still claimed 9px, so hovering a line
anywhere near a station showed the station's tooltip instead of the segment's
— and at national zoom that is most of the map, with the bubble too small to
see and nothing to explain the wrong answer. The floor now runs 3px at zoom 3
to 9px at zoom 13. Lines keep their flat floor: a hairline is unclickable at
any zoom and nothing else competes for the cursor there. The same fix is in
[japanriders](../japanriders/), which had the identical code.

### A step drawn where it happens

경부고속선 read 142,189 a day south of 천안아산 and 102,429 north of it, and
those cannot both describe the same railway. The 39,760 difference is the SRT
joining — but they do not join at 천안아산. `6. 운전` says where:

```
경부고속선  금천구청-SR분기   KTX 117            no SRT yet
           SR분기-오송      KTX 117 + SRT 60   they have joined
```

SR분기 is 평택분기점, which has no platform and no 승하차 row, so the fit hangs
the step on the next station that exists. The map then drew one number over the
whole 74 km from 광명 when the last quarter of it carries the SRT as well.

The data cannot be fixed — there is no passenger row for a junction — but the
*drawing* is a separate question, and the junction's position is perfectly well
known. `split_at_junction` cuts the segment at the point on its corridor
nearest the handing line's own end, and the far side takes the through flow:

```
서울-광명        2.9 km    79,545
광명-천안아산    50.5 km   102,429   north of 평택분기점
광명-천안아산    23.6 km   145,772   south of it, SRT aboard
천안아산-오송    28.7 km   142,189   less 천안아산's own alightings
오송-대전        35.2 km   101,904   Honam services gone
```

The step to apply is the handover's own tau, not the difference between the two
segments' loads: the 39,760 gap mixes the SRT joining with 천안아산's platform
movement, and only the first belongs north of the station. Taking the tau gives
145,772, above 천안아산-오송 rather than equal to it, which is right — people get
off there.

It moves the network's 인거리 cover from **85.0 % to 86.4 %**, because 23.6 km
of real SRT travel was previously drawn at the wrong load. The mirror is
unchanged.

**The 호남선 handover did not split, and the guard was why.** Its receiving stop
is 신탄진 and the branch happens at 대전조차장, but the proxy for the junction was
호남선's own end at 서대전 — which sits 3 km *southwest* of 대전, so the nearest
point on the 대전→신탄진 corridor was 대전 itself, at index 0 of 195. A cut there
would be zero-length and in the wrong place, so `split_at_junction` declined,
as it does whenever either side would come out under a kilometre. **Fixed**;
see [The 호남선 split, and one
coordinate](#the-호남선-split-and-one-coordinate-2026-09-07).

Two smaller things this leaves. The two halves share a name, so the page's
tooltip names the junction on the heavier one — otherwise the reader hovers two
different-looking pieces of "광명 → 천안아산" and sees two different numbers with
no explanation. And `compare_builders.py --line` keys on the station pair, so it
now counts repeats and labels the second one.

**What that tooltip said was still wrong, and Anita caught it (2026-09-07).**
It read "traffic joining from 평택지제", because `junction` carries the giving
line's *last stop* — and 평택지제 is a station 3 km short of the merge, with a
dot on the map at neither the cut nor the step. So the note pointed at a place
that is not where the number changes. What joins at a flying junction is a
**service**, not a station: `join_line` now carries the giving line's name and
the note reads "Suseo HSR (SRT) trains join here, at a junction with no
station". The 경부선 cut reads "Honam Line trains join here" for the same
reason.

The step is worth being able to defend, because it is the largest unexplained
change on the map. `6. 운전` gives 경부고속선 **KTX 117 each way north of
SR분기 and KTX 117 + SRT 60 south of it**, and 수서고속선 **SRT 60** into
평택분기 — the same sixty trains. The drawn step is 43,343 a day, which over 120
SRT movements is **361 a train against an SRT's 410 seats**. The cut falls at
36.99969/127.04670, which is the junction itself rather than an approximation
of it.

**Unrelated, found while measuring this.** 경부고속선's 서울-광명 segment is drawn
2.9 km long where the two stations are 22 km apart: the line's own metals start
near 금천구청, so 서울 snaps to the corridor's north end. Nothing is missing from
the map, since 경부선 draws that approach and carries 123 KTX a day in its own
counts — but those 79,545 riders are credited with 2.9 km rather than 22, which
is roughly 0.5 bn passenger-km a year, about 2 % of the network total. Worth a
look if the 인거리 check is ever wanted tighter.

### City geometry from OSM, and the header that was blocking it

All five intracity models drew straight hops between published station
coordinates. Busan line 2 curves a long way round the bay and the map put a
chord across it. `fetch_city_track.py` pulls the metals and `city_track.py`
fits them; all **261 city segments** now follow real track.

**The block was a User-Agent, not a rate limit.** Both Overpass endpoints
filter on it and neither says so in a way the status code reveals:
overpass-api.de answers a default python-requests header with a bare Apache
**406 Not Acceptable** — the same 406 recorded here as an unexplained failure
of the earlier Busan attempt — while kumi.systems returns **429** with the
reason only in the body, "Please include a meaningful User-Agent string with
your requests to avoid rate-limiting". `fetch_osm.py` has always sent one,
which is why it works and the new fetcher did not. Hours of retrying looked
like a rate limit clearing slowly and was nothing of the kind. `/api/status` is
what told the truth; it is worth asking a refusing endpoint about itself before
believing its status code.

Two further things the pull needed:

- **Daegu line 3 is a monorail**, so `railway=subway|light_rail` silently
  missed it. It is in the query now, and Daegu 3 appeared with 8 ways and 2
  relations.
- **Ways and relations in one `out geom` reply time the server out** with a
  504, because a relation repeats every member way's full geometry. Two lighter
  queries come back where one heavy one does not.

**A metro is double track, and that is the whole difficulty in the fit.** OSM
maps each track as its own way, so chaining a line's name pool end to end
produces one out-and-back polyline of twice the real length — Busan line 1 came
out 79.8 km against a railway of 40.5. Route relations are the way out, since
the operator publishes one per direction. But grouping relations *by name*
looked right and was not: Busan line 2 and Daegu line 2 each have two relations
under a single name, and merging them rebuilt exactly the out-and-back the
relations were supposed to avoid. Each relation is now its own candidate.

Choosing between candidates needs the stations, so `pick_for` takes **the
shortest candidate that passes within 400 m of every station**. Shortest is the
discriminator — an out-and-back covers the stations just as well and is twice
as long — and full coverage is what stops a half-mapped relation winning on
shortness alone. It earns its keep on Daegu line 1, where the shortest
candidate is 30.9 km and misses the 안심~하양 extension, so the fit takes the
37.6 km one instead.

| line | segments | worst station offset | straight | on track |
|---|---|---|---|---|
| 부산 1호선 | 39 | 205 m | 37.9 km | 39.8 km |
| 부산 2호선 | 42 | 110 m | 43.3 km | 45.3 km |
| 부산 3호선 | 16 | 52 m | 17.1 km | 17.9 km |
| 부산 4호선 | 13 | 37 m | 11.2 km | 11.8 km |
| 대구 1호선 | 34 | 100 m | 36.7 km | 37.6 km |
| 대구 2호선 | 28 | 82 m | 30.1 km | 31.4 km |
| 대구 3호선 | 29 | 16 m | 22.4 km | 23.1 km |
| 부산김해경전철 | 20 | 20 m | 21.5 km | 22.4 km |
| 광주 1호선 | 19 | 76 m | 19.8 km | 20.4 km |
| 대전 1호선 | 21 | 77 m | 19.7 km | 20.7 km |

Track runs 2–5 % longer than the chords, which is what a railway following
streets should do. The stations are the check: at 400 m tolerance the worst is
205 m and most are under 110, so these are the right metals rather than a
neighbouring line's.

A line that cannot be fitted keeps its straight hops rather than taking a
corridor its stations do not sit on, and the stations must run monotonically
along the candidate or the slices would double back — which is what excludes
Gwangju's line 2 ring. Nothing currently falls back, but `geometry_source` says
`osm` or `straight` per segment either way, and `track_geometry` in each model
report records the fit.

**The flows did not change.** This is geometry only: the same loads, drawn
along the line they are actually carried on.

### The 호남선 split, and one coordinate (2026-09-07)

The guard that declined this cut was right to; what it was declining was a cut
aimed at the wrong place. `split_at_junction` had only one candidate for where a
handover happens — the giving line's own last stop — and for 호남선 that is
서대전, which is not on the metals being cut and not even on the right side of
대전. `lines.HANDOVER_POINT` now lets a handover name its junction outright, and
`write_geojson` prefers it where it exists:

```
                 before                     after
대전-신탄진      14.49 km   12,051      4.78 km   12,051    (대전 → 대전조차장)
                                        9.71 km   16,148    (대전조차장 → 신탄진)
신탄진-부강      12.10 km   17,113     12.10 km   17,113
```

4,097 a day is the 호남선 tau, not the 5,062 gap across 신탄진 — that gap mixes
the branch with 신탄진's own platform movement, the same distinction the SRT
split turned on. 인거리 cover goes 86.4 % → 86.5 %, which is the whole of what
this is worth; it was done because a 14.5 km segment drawn at one number when
its northern two-thirds carry a third more traffic is wrong on the map, not
because the total moved. Nothing else in the file changed: 425 features became
426, one `km` and one geometry differ, and every load is identical.

**Where the coordinate came from.** 대전조차장 is OSM node 7640162489,
`railway=yard`, `wikidata=Q188837`, at 36.3710255/127.4218344 — 4.4 km north of
대전 as the crow flies, 4.78 km along the track, which is the check that it is
the right node. It is absent from `data/osm_stations.json` because that pull
filtered to `railway=station|halt`; widening that filter would have put a
freight yard into the pool the membership test draws chains from, which is a
worse trade than recording one literal. `--out` was added to `solve.py` at the
same time, so the run that measured this wrote to a scratch path and `data/`
stayed as another session left it.

### The thin lines, and which of them are wrong (2026-09-07)

`todo.txt` asks for the suspiciously low counts to be verified, naming the far
end of 경원선 and 호남선. Dividing every segment's load by the trains the 운전
sheet runs on it answers all of them at once, and the three cases turn out to be
three different things. `check.py` now prints the emptiest lines under 승차율.

| line | passengers per train | what it is |
|---|---|---|
| 경춘선 | **0.2** | the source does not count this service at all |
| 정선선 | 12.3 | a real, tiny railway |
| 경의선 | 15.6 | one DMZ train a day |
| 광주선 | 20.8 | a 12 km shuttle |
| … | | |
| 호남선 | 78.1 | denominator artefact — see below |
| 경부선 | 201.9 | |
| 경부고속선 | 371.4 | |

**호남선 is not wrong.** It reads 46 a train over 장성-광주송정, the thinnest
stretch on the map's second conventional trunk, against 경부선's 202 and
전라선's 163. The gap is in the denominator: `6. 운전` books **20 SRT and 2 KTX
a day each way** to 호남선's 익산-광주송정 alongside 15 conventional trains, and
those high-speed passengers are credited to 호남고속선, not here. Against the
conventional service alone the same stretch carries **114 a train**, an ordinary
무궁화 load. The thinness is real and it is the 2015 opening of 호남고속선 — the
through traffic left the conventional line — not a reconstruction fault.

**경원선's far end was already right.** 14 a day to 백마고지 is the fit's
residual on track that ran nothing in 2022, and the map stopped quoting it when
the `service` grey went in. Nothing to do.

**경춘선 is wrong, and not by a little.** The 운전 sheet runs **26 ITX-청춘 a day
each way** plus 42 전동차; the 통과인원 sheet credits the whole line with **2,399
passengers for the year**, which is 춘천's own 승하차 and nothing else — no other
경춘선 station has an intercity row at all. So the fit faithfully reproduces the
sheet and draws nine people a day over a railway running fifty-two trains. That
is the 경원선 hairline again: a false claim about ridership, arrived at honestly.

The line now draws in the no-service grey at constant width, with a tooltip
saying that trains run and their riders are not recorded. The test is
`published 통과인원 / trains < 1 passenger per train`, both figures published and
neither modelled; 경춘선 comes to 0.13 and the next line up is 정선선 at 21.9, so
it is not a threshold slicing a continuum. The legend row is relabelled **"no
rider figure"** from "no intercity service", because that is now the one thing
true of all three greys. **No load changed** — this is what the map says about
the numbers, not the numbers.

**The colour was the smallest part of it, and nearly the wrong thing to fix
alone.** 경춘선's metals are covered end to end by the metro layer's own
수도권 경춘선, which draws on top and wins the hover, so the grey is invisible
and its tooltip unreachable. Every place the false figure actually surfaced was
somewhere else:

- the **roster** read `Gyeongchun Line 13`, its peak segment. Lines drawn grey
  end to end now show `—` and sort last. Partly-grey lines keep their number —
  경원선's 13k on 용산–청량리 is real and only the stretch past 청량리 is greyed.
- the **info panel** read `passengers / day  9 – 13`. It now carries the same
  refusal the tooltip does, in both languages.
- that panel's **range** was over every segment, so 경원선 read `14 – 13,193`
  where 14 is a greyed segment the line's own tooltips decline to quote. The
  range is now over the segments the map is willing to name, and 경원선 reads
  `13,193`.

The general lesson, which is not about 경춘선: the grey was introduced as a
drawing rule and every *other* surface kept quoting the number behind it. A
figure the map refuses to state in one place and states in three is not
refused.

It also disposes of 경춘선's 113.5 % mirror, which has been the network's worst
since the fit was built. There was never anything there to fix.

### 경원선's level is contaminated, and taking the contamination out costs more than it buys

Worth recording as a dead end, because the reasoning is sound and the result is
not. 통과인원 counts everyone on a line's metals, 광역전철 included, while the
승하차 sheets record intercity types only. Usually that just makes the published
figure a ceiling, which is how `solve.py` uses it. 경원선 is the one line where
it also has to set the **level** — 백마고지 has no 승하차 row, so `clean_end` is
false and `W_PASSING` pulls two-sided — and 24 of its 39 segments run 전동차 and
no intercity train at all. The fit reaches 2.87M of a published 7.50M and the
rest is riders with no 승하차 row anywhere. Asking it to close that gap is asking
it to invent passengers, and where it puts them is the 용산-청량리 entry flow,
one direction harder than the other. That is 경원선's 24.6 % mirror, README item
3, and the diagnosis holds up.

Dropping the two-sided term for lines with 광역전철-only sections — which fires
on 경원선 alone — does exactly what the diagnosis predicts and breaks other
things doing it:

```
                     before        after
경원선 mirror         24.6 %        4.2 %      the fix works
정선선 mirror         26.8 %        8.4 %
인거리 cover          86.5 %       87.2 %
network median        1.8 %        2.3 %      and this is the price
경북선                 7.4 %       15.0 %
태백선                 5.1 %        6.6 %
영동선                 2.2 %        3.5 %
중앙선                 4.0 %        4.8 %
충북선 조치원-오송   102k / 98k    0 / 2,155   하행 / 상행, a hard break
```

충북선's first segment is what settles it. Every train on the line crosses
조치원-오송, and a fit that gives it nothing 하행 and 2,155 상행 has traffic
appearing from nowhere at 오송 — the failure mode positivity and the mirror
exist to catch. 경부선's 수송밀도 also moves 17,343 → 20,735, a fifth of the
map's headline number, off a change to a line at the other end of the country;
that is toward `build.py` and toward the published 인거리, so it may even be an
improvement, but it is not one this change has earned.

What it means: the contaminated 통과인원 is carrying real information about the
rest of the network through junction conservation, and removing it is worse than
using it wrong. A fix for 경원선 has to *correct* the target rather than delete
it, which needs a defensible split of the 7.50M between intercity and 광역전철,
and nothing published gives one. The code keeps the old behaviour with a comment
pointing here.

### Reaching the end station, and drawing between stations (2026-09-07)

Two of `todo.txt`'s items and the first of "worth doing next" turned out to be
one bug and one design fault.

**경부고속선 now reaches 서울역, and 중부내륙선's shortfall was the same bug.**
The corridor search anchors each line at both ends on *its own* metals, so a
line whose end station sits on someone else's stopped short and the station was
drawn wherever the rails ended:

| line | end | drawn short by |
|---|---|---|
| 경부고속선 | 서울 | **14.29 km** |
| 중앙선 | 경주 | 3.20 km |
| 영동선 | 강릉 | 3.06 km |
| 광주선 | 광주송정 | 1.77 km |
| 중부내륙선 | 부발 | **1.63 km** |

That last row is the 1.9 km 중부내륙선 was drawn short by — the same fault, not a
separate one.

**Anchoring the search at the station instead is much worse, and this is worth
not rediscovering.** The nearest graph node to a station can sit on track that
does not connect locally, so the route goes the long way round: 광주선 came out
at **414.6 km for a 12.2 km railway**, 호남선 stopped connecting at all, and
중부내륙선 collected 78 stations it does not call at. The 40× foreign-track
penalty is not a leash when the search starts outside the fence.

**The cause is the pull, not the router.** `data/osm_railways.json` is
`way[railway=rail][name]`, so unnamed track is not in the graph at all — and
junction throats, crossovers and the connections between lines are exactly the
track nobody names. There is no local link from the high-speed metals to 경부선,
so a 14.29 km gap routes 111.96 km and 광주선's 1.77 km gap routes 402.42.

So there are two mechanisms, and the bounded one refuses more often than it
works:

- **`build.reach_station`**, a stub over any track, bounded at 30 km and at 3×
  the straight-line distance. It earns its keep on 영동선's 강릉 (3.69 km of
  track for 3.06 straight) and 중앙선's 경주 (7.80 for 3.20), and correctly
  refuses all three of the others. 중앙선 goes 323.1 → 330.9 km against a
  published 332.2, which is the independent check that it went the right way.
- **`lines.OVER`**, which names the line whose metals carry the trains and takes
  that line's own corridor — already built, already checked against its
  published length — sliced from the station to the junction. No search, nothing
  to wander. 경부고속선 gets 18.7 km of 경부선 and the two lines overlap on it,
  which is what they do in life; japanriders draws shared track the same way,
  one geometry per line, stacked.

```
                   before                     after
서울 drawn at      37.4399/126.8976           37.5538/126.9705
                   14.30 km from 서울역        0.11 km from 서울역
서울-광명           2.85 km                    20.86 km
인거리 cover        86.5 %                     88.6 %
```

**The rescale had to come off with it.** 경부고속선's legal ends really are
서울-부산, so it passed the `scaled` test and was being rescaled to a 영업거리 of
398.2 km that measures the high-speed metals alone — while a KTX from 서울 to
부산 runs 417. The 18.7 km added at 서울 was therefore taken back out of every
other segment: 서울-광명 came out right and 대전-동대구 came out short, and the
인거리 went *down*. `OVER` now suppresses the rescale for the same reason
`ENDS` does, and 경부고속선 draws 417.3 km. `check.py`'s allowance for a line
whose extent is not its 영업거리 goes 12 km → 25 km to cover it, which still
catches the 38 and 92 km misroutes it exists for.

**Segments now run between stations that something stops at.** 172 of 426 drawn
segments carried an identical load to their neighbour: 수원 to 안양 was eight
features all reading 29,471, because the stations between are Seoul line 1's and
no 무궁화 calls at them. The numbers were honest and the segmentation was not —
hovering any of the eight gave a piece of railway that was never measured as a
piece.

`merge_unserved` joins across a station with no 승하차 of the line's own train
types, since there is no measurement there to justify a break. **426 features
become 247**, over 179 stations. Four things still stop a merge, and each is a
real break: flow at the station; a change in `service`, so grey never absorbs
drawn track; a `junction` feature, which was split there on purpose; and the two
loads disagreeing by more than 1 %, which means the fit put a step there with no
passengers to show for it — a junction the model believes in, and hiding it
would be its own kind of lie. Across all 65 merged runs the **worst internal
spread is 0.03 %**, so nothing real was joined over.

This disposes of both of the "smaller, mostly legibility" items without naming
either: 경부고속선's 울산-부산 absorbs 범일 and 부산진, the pair it used to split
its approach into three identical segments at, and 수서고속선's 동탄-수서 absorbs
구성 and 성남, where the SRT runs in tunnel under stations of other lines.

**The two changes need each other.** Extending 경부고속선 to 서울역 drags nine
conventional stations within the 300 m snap — 용산, 노량진, 영등포 and the rest —
and none of them would have been a stop of anything a KTX runs. Because none has
KTX or SRT 승하차, the merge absorbs all nine and what is drawn is one honest
20.86 km segment. Run either change alone and the map is worse than before.

A merged segment carries `through`, the number of stations it runs past, and the
tooltip says so — the station bubbles are still drawn, so a length of track with
no break at them has to explain itself.

**Left alone.** Station names collide across operators and `flows` is keyed by
name, so the Korail 좌천 on 동해선 has traffic and that traffic blocks a merge at
the 부산 도시철도 좌천 on 경부선. It costs one spurious break near 부산진. Same
class of fault as the metro stations themselves, and cheap to fix if it ever
matters.

### The lines that stopped at the platform (2026-09-07)

Anita, looking at the map: *why does 수서고속선 look visibly disconnected from
경부고속선?* Because it was, by 7.5 km, and chasing that turned up a second fault
nobody had noticed.

**A corridor is routed between a line's two end *stations*.** For a line that
ends at a terminus that is right. For a handover line it is not: the SRT do not
stop at 평택지제 and turn round, they carry on to 평택분기점 and onto 경부고속선.
That last stretch was never drawn, so the line ended in mid-air pointing at
nothing — and it is the whole of a discrepancy `check.py` had been reporting as
a chainage curiosity for weeks:

```
                 drawn      영업거리    short by
수서고속선         53.2 km    61.1 km    7.9 km    the worst diff in the file, -12.9 %
호남선           246.7 km   252.5 km    5.8 km
```

Both gaps are exactly the distance from the last platform to the junction.
`RUN_ON` now extends those corridors along the line's own metals to the junction
`HANDOVER_POINT` already records, and the run-on piece is drawn as a segment of
its own carrying the handover's tau — everyone still aboard at 평택지제, who are
by definition the people who step onto 경부고속선. **수서고속선 goes to 60.7 km
against 61.1 (−0.7 %) and 호남선 to 252.8 against 252.5 (+0.1 %).**

Not double-counted: the same passengers ride 경부고속선's own segments, but over
different track on the far side of the junction. 인거리 cover 88.6 % → 88.8 %.

**The second fault: the step was drawn 5.7 km from where it happens.** With no
`HANDOVER_POINT` for 수서고속선, `split_at_junction` fell back to the giving
line's last stop, so it cut 경부고속선 at the point nearest **평택지제** —
36.99969/127.04670, which is a closest approach and not a junction at all. The
real 평택분기점 is findable exactly, because 수서평택고속선 and 경부고속선 **share
an OSM node**: 36.95148/127.07057, 7.48 km south of 평택지제. Recording it moves
the step to the merge and shortens the SRT-laden piece of 광명-천안아산 from
23.64 km to 17.75.

The check that it is right: the Suseo stub now ends at 36.95148/127.07057 and
경부고속선's cut begins at 36.95148/127.07063. **Six metres.** They meet.

**A stub is not a chain segment**, and `check.py` had to learn that — its `to`
is a line rather than a station, and leaving it in the stop list shifted every
section count by one. The page says `평택지제 → junction with Gyeongbu HSR` and
notes that these are the passengers still aboard past the last platform.

The lesson worth keeping: a published length that disagrees by 12.9 % had been
read as a rounding question about chainage, when it was 8 km of railway missing
from the picture. **A number that only ever gets compared with itself will not
tell you what it means.** It took someone looking at the map to ask why two
lines that meet were drawn apart.

## What is not done

- **The short lines.** 대구선 (13.0 % weighted mirror) and 경북선 (8.0 %) are
  where one junction carries most of the line and the step size is still a guess.
  영동선 has since come down to 2.2 % and is no longer in this company.

  The obvious next turn of the screw looked like using the section counts for a
  step's *magnitude* and not only its sign — 천안 is 66 trains down to 51, so
  roughly 15/66 of the load leaves. **It cannot help these two lines**, and this
  is worth not rediscovering: `python frequency.py` shows 대구선's only published
  boundary is 가천, which is not on the chain at all (the chain starts at 동대구),
  and 경북선's only one is 점촌 at 5 trains either side — a "no step" that the fit
  already enforces. The lines with informative boundaries are 중앙선, 호남선,
  경전선 and 동해선, and all four are already at 0.3–4.0 %. So the constraint is
  absent where it is needed and near-redundant where it applies.

  What is actually wrong with 대구선 is smaller than 13 % sounds: four segments,
  no interior boardings on the first three, and a flat 하행 660 k against 상행
  762 k — 280 passengers a day of level disagreement on a line carrying 3,900,
  with both its ends junctions and nothing but 통과인원 to set the level.
- **`build.py` and `solve.py` disagree on levels, and the fit wins.** 경부선
  is 33,018 from the single-line build and 17,343 from the network fit, both
  with clean mirrors. The published 인거리 favours `build.py` (92.8 % against
  85.0 %) and the published train counts overturn that: `build.py` needs 608
  people on each 무궁화 through 지천-신동, which seats about 432, over eight
  consecutive segments, while **every segment of the fit fits inside its
  trains**. See [Passengers per
  train](#passengers-per-train-and-why-the-인거리-verdict-was-backwards).

  **So the map should draw `solve.py`, which is what it already draws** —
  `build.py` writes `segments_singleline.geojson` and nothing reads it. What
  remains open is not which builder but the fit's own worst flag, 수서고속선 at
  1.9×, where the published train count and the published SRT ridership
  disagree with each other before any model is involved.
- **Non-Korail stations on the chains.** `solve.py` admits any station whose
  membership test passes, which puts 신용산, 삼각지 and 숙대입구 on 경부선 and a
  string of 부산 도시철도 stops on it through 부산진, plus a duplicate
  서울역/서울 pair. They carry no yearbook flow so they do not move the numbers
  much, but they split segments that are not really segments, and they are where
  경부선's junction steps go wrong.
- **Other line ends no chain names.** The scan that found 수서고속선's 평택지제
  found more. Most are real termini and want nothing — 목포, 여수엑스포, 춘천,
  영덕, 백마고지, 도라산, 광주, 구절리. Two are not:

  **서대전**, where 호남선 starts, has since been handed over to 경부선 at
  신탄진 — see above. It is a weaker fix than the SRT one: 대전조차장, where the
  branch really happens, falls between 대전 and 신탄진 and cannot be drawn, so
  the 대전–신탄진 segment is short of traffic that rides most of it. 대전 would
  place it better but is called by 경부고속선 too, and has a published count
  whose sign rule forbids the step outright.

  **부발**, where 중부내륙선 starts, is the junction with 경강선, which this build
  does not model at all.

- **Lines whose passengers board off-line — mostly addressed, see below.**
  호남고속선 runs 오송–광주송정 and nearly everyone on it boards at 용산, 서울 or
  광명, none of which is on its chain, so its rebuild saw a fifth of the line.
  `lines.ENTRY_SHARE` now sizes the entry flow from the published train counts
  and it draws 20,830 rather than 9,416. What remains is that 통과인원 still
  only reaches 0.39 — though 경부고속선, whose 서울 traffic *is* on its chain,
  only reaches 0.54, so the undercount is systemic to the high-speed lines
  rather than particular to this one.
- **Where the 목포 high-speed traffic transfers, and what 호남선's northern half
  really carries.** Giving 호남선 the high-speed types south of 광주송정 fixed
  that stretch and also lifted 서대전–계룡 from 16,574 a day to 24,904, which is
  a side effect rather than a finding. Both builders rise by the same 8,330, so
  it is the traffic leaking and not a change of method — only 909 a day step off
  at 광주송정 against the high-speed riders arriving from 목포, and the fit
  carries the rest north up the conventional line rather than onto the
  high-speed one they really came down. (Beware comparing across builders here:
  `build.py` has 서대전–계룡 at 4,338 before and 12,668 after, the same 8,330
  apart, because it anchors the level differently — see the two entries below.)

  What blocks the transfer is *not* what it looks like. 광주송정 is 호남고속선's
  on the facility roster, so it anchors the line, and an anchor asserts that
  everything alights at the last stop — false here, since 42 high-speed trains
  arrive 익산–광주송정 and 28 leave again towards 목포. `lines.THROUGH_ENDS`
  suppresses that anchor, and **on its own it changes nothing at all**: every
  figure in the report comes back identical, because nothing then pushes the
  through flow up. The smallness prior pushes it to zero and both answers
  satisfy junction conservation equally well. The same shape of problem as the
  original junction steps — many near-equivalent optima, the prior choosing
  which line absorbs the error, the information not in the residuals.

  Treating the end as a junction outright *does* move it, by handing the level
  to 통과인원: 호남고속선 goes from 5,774 to a plausible 40,697. But a hard
  22.4M target pulls on every neighbour through 오송 and 익산, and the median
  weighted mirror went 1.9 % → 4.1 % with cost 82 → 190, 광주선 at 43.6 % and
  장항선 at 30.1 %. Not worth it.

  What settles most of it is the span itself, not the section counts. PART_TYPES
  says those types run only south of 광주송정; it follows that their passengers
  ride only there, so whatever boards inside the span leaves at the boundary
  rather than carrying on. That is a consequence of the table rather than a new
  assumption, and nothing else in the fit was saying it. `solve.py`'s
  `contain` term states it one-sidedly — the first segment outside the span
  must be lighter than the last one inside by at least the contained traffic,
  and the conventional step at the same station stays free to take more on top.

  The contained traffic computes to 4,175 a day southbound and 4,155 northbound,
  summing to the 8,330 the northern half had risen by, which is the check that
  it is aimed at the right thing. With it the corridor comes out:

  | | before | leaking | contained |
  |---|---|---|---|
  | 임성리–목포 | 743 | 6,593 | **6,593** |
  | 장성–광주송정 | 2,103 | 10,432 | **3,399** |
  | 서대전–계룡 | 16,574 | 24,904 | **19,197** |
  | 호남고속선 정읍–광주송정 | 2,934 | 2,934 | **7,526** |
  | median weighted mirror | 1.9 % | 1.9 % | **1.4 %** |

  It stops about three-quarters satisfied, and the reason is worth having: the
  transfer is blocked at the far end. 호남고속선's last segment carries 1.37M
  against 3.27M high-speed passengers alighting at 광주송정 — a ratio of 0.42.
  A line ending at a junction has a through flow of the end segment's load less
  the platform movement, and that may not be negative, so the fit can only take
  more traffic off 호남선 by pushing 호남고속선's *share* of its own terminus
  down. 호남고속선 is simply too small to receive the transfer, seeing 4.2M of
  a published 22.4M, and that is the off-line boarding entry above rather than
  anything about 호남선. Fixing that one should let this finish on its own.

  The section counts' **magnitude** remains unused and would be the independent
  check: the KTX count across 광주송정 drops 19 to 2, so most of the high-speed
  service south of it demonstrably does not continue north. It cannot help
  대구선 or 경북선, as recorded above, but 호남선 is one of the four lines whose
  boundaries are informative.
- **Stations a line's trains run past without stopping.** 범일 and 부산진 are
  Korail's, so the operator filter keeps them, but no KTX calls at either and
  they have no 승하차 row — they split 경부고속선's approach to 부산 into three
  segments carrying an identical 43,601. Tightening `build_chains` to drop a
  station with no traffic in the line's own train types would fix it, and would
  also drop 노량진, 구로 and the rest of the 광역전철 stops from 경부선, which is
  a legibility question rather than a correctness one: no flow changes at any of
  them either way.
- **중부내륙선 is drawn 1.9 km short**, and it is the last line `check.py` still
  flags. OSM's 중부내륙선 metals stop 1.6 km from 부발 station — the junction
  throat off 경강선 is not mapped under the line's name — so the corridor starts
  short of the platform and the rescale stretches the rest by 3.4 % to cover it.
  `pick()` deliberately anchors on the line's own track, which is what stops
  영동선 setting off down 중앙선, so the fix is not simply to widen it.
- **Remaining intracity systems.** The Donghae commuter line and Daegyeong Line
  still need compatible current gate totals. The available 2021 Korail annual
  file catches the Donghae Ulsan extension only in its opening days, so dividing
  it by 365 would produce a misleading current-looking profile. Yongin EverLine
  is also absent from the 22-line seoulriders import.
- **Newer data.** 2023 is the last year on info.korail.com; the download path on
  the railstat portal has not been found.
- The honest caveat for the finished map: unlike japanriders, these segment
  values are **derived, not published**. That belongs on the page.
