# Seoul Subway Ridership Visualization

Animated map of Seoul subway trains sized by riders on board, in the style of
`../nycriders/`. **Status: end-to-end and working. Polish outstanding.**

Seoul came off `../metroslist.md` as a Tier 1.5 system — measured entry *and*
exit gates, so the numbers are real rather than inferred, but the published OD
carries no hour field and has to be disaggregated against hourly station counts.

**The map draws a typical weekday by default.** The OD is measured on one
Sunday and nothing else; the weekday is that pattern re-levelled onto measured
weekday station totals. Read "Getting off New Year's Eve" before quoting any
figure — it is the one modelled step in the pipeline, and `--day nye` still
builds the fully measured night.

---

## Picking this up

Read this section, then `todo.txt` (Anita's own list — **hers to edit, not
ours**), then the rest of this file for the why behind any given decision.

**Run order.** Each step reads the previous step's output from `data/`:

```
python fetch_schedules.py  # ~1 min  -> data/kric_*.xlsx, data/incheon2_*.csv
python fetch_ridership.py  # ~1 min  -> data/congestion_raw.csv, card_daily_*.csv,
                           #            congestion_line9.xlsx
python fetch_osm.py        # ~2 min  -> data/osm_routes.json, osm_stations.json
python kric.py             # ~40 s   -> data/timetable_extra.csv
python build_stations.py   # ~40 s   -> data/stations.json
python build_od.py         # ~4 min  -> data/od_hourly.npz
python build.py            # ~10 min -> data/trains.json, stats.json,
                           #            link_shapes.json  (--sample 60 for ~50 s)
python validate.py         # ~40 s   -> checks the build against published figures
python -m http.server 8000 # then open index.html
```

**`crowding.py` is optional and slow.** It re-runs `build.py` several times to
make crowded trains less attractive — see "Crowding" — and replaces
`trains.json` with the result. Skip it and the pipeline is exactly as it was;
`crowding.py --reset` undoes it. Budget ~15 minutes a round.

**Builds use half the cores** (`JOB_SHARE` in `build.py`), so the machine stays
usable while one runs. `--jobs` overrides.

**Which day is one decision, made once.** `build_od.py --day weekday|saturday|
sunday|nye` stamps its choice into `data/od_hourly.npz`; `build.py` and
`validate.py` read it back from there and take no day flag of their own. That
is deliberate. The timetable day type has to agree with the ridership day type,
and a mismatch is *silent* — weekday riders routed over a Sunday timetable do
not error, they just come out looking like a thinner weekday. `daytype.py`
holds the registry.

**Run `validate.py` after a build.** It compares each line's end-to-end
journey time, service span and station count against the operators' published
figures, walks every line's stop order looking for a station that landed far
from its neighbours, and prints the OD coverage table. It has already caught
two real bugs that looked like data — see "One bad stop time" below.

`validate.py --congestion` is the newest check and the only one that tests the
**output**. Everything else here checks an input. See "Checking the build
against 혼잡도".

The first three steps are what bring in the lines beyond 1–9. They only need
re-running when a source changes; `fetch_schedules.py` skips files it already
has.

**`lines.py` is the registry.** Every line's id, its name in each of the three
source files, its OSM relation pattern, colour and capacity live there and
nowhere else. Adding or renaming a line is a change to that file; `index.html`
reads names and colours out of `trains.json`, which `build.py` fills from the
registry, so the page needs no edit.

**`build.py` runs `build_shapes.py` itself now.** `build.py` lays down
straight-line hops between stations; `build_shapes.py` bends them onto the real
track, and it used to be a separate step that was easy to forget — miss it and
the trains visibly cut corners. It is now the last thing `build.py` does, and it
is idempotent (it drops any waypoints already present before it starts), so
running it by hand afterwards to re-tune the geometry is still fine and still
skips the routing. `--no-shapes` opts out.

**As of the end of the 2026-09-04 session:** the pipeline carries all 22 lines
and defaults to a weekday. `data/trains.json` is a full `--day weekday` build.

**How to tell a sample from a full build:** file size will not tell you. Check
`build.py`'s output — a full weekday build says `routed 6,3xx,xxx riders`, a
Sunday one `routed 3,3xx,xxx`, and a `--sample 60` run about `101,xxx`. The
page's own footer says so too when it is a sample.

**Three traps that cost real time.** All are fixed, all are documented below,
and all would be easy to reintroduce:

1. `00:00:00` in the timetable is a **null marker**, not midnight.
2. RAPTOR needs **one arrival label per round**; a single parent pointer
   silently loses riders.
3. Korean portal signup is **impossible from abroad** — never go down that road.
   See `[[reference_korea_open_data]]` in memory for the anonymous routes.
4. The map's midday peak is **Sunday, not New Year's Eve**. Do not spend an
   afternoon on the date; the two are indistinguishable in the hourly profile.
   See "Getting off New Year's Eve".

**Verify by checking output against source, not by reading code.** Every bug
found on day one looked completely plausible on the map. `build.py` prints two
invariants — riders boarding more than an hour off their spawn hour (should be
~0.3%) and unrouted (~0%) — and the scratch scripts that caught the rest are
described under "build.py output".

---

## Current state

Everything needed is downloaded. **No account, key or email was required** —
see "The account wall" below, which matters because the obvious route is
impassable from outside Korea.

| File | What | Size |
|---|---|---|
| `data/timetable_raw.csv` | Train timetable, lines 1–9 | 32 MB |
| `data/od_2023-12-31.csv` | Station-to-station OD, one day, 27 lines | 14 MB |
| `data/hourly_2023_raw.csv` | Daily × hourly counts, every day of 2023, lines 1–8 | 25 MB |
| `data/daily_hourly_raw.csv` | Same for 2024 | 25 MB |
| `data/card_daily_2023{11,12}.csv` | Daily counts, every station, **all 27 lines** | 1.2 MB each |
| `data/congestion_raw.csv` | 혼잡도 by station/direction/half hour, 평일·토·일 | 0.4 MB |
| `data/osm_stations.json` | 786 rail station nodes, Seoul-area bbox | 0.6 MB |
| `data/osm_routes.json` | 165 route relations, ordered stops + track geometry | 7 MB |

The Korean files are CP949-encoded **except `card_daily_*.csv`, which is UTF-8
with a BOM**; the OSM pulls are UTF-8 JSON. `data/` is gitignored repo-wide, so
none of it commits.

**The one real limitation is still the date.** The OD exists for exactly one
day, 2023-12-31, and that day is a Sunday. What the map can do about it is
"Getting off New Year's Eve", below.

## The data

### Timetable — lines 1–9, complete

[서울교통공사_서울 도시철도 열차운행시각표](https://www.data.go.kr/data/15098251/fileData.do).
Better than the MTA GTFS we used for NYC: both arrival *and* departure times at
every station, and `열차코드` works directly as a trip id.

Columns: `고유번호, 호선, 역사코드, 역사명, 주중주말, 방향, 급행여부, 열차코드,
열차도착시간, 열차출발시간, 출발역, 도착역`.

- **424,264 rows**, **5,146 train codes**, **458 station codes** / 405 names
- `주중주말` — DAY 160,136 / SAT 132,068 / END 132,060, so weekday, Saturday and
  Sunday-holiday timetables are all present. We need END for 2023-12-31.
- `방향` — DOWN/UP plus OUT/IN, the latter Line 2's 외선/내선 loop
- `급행여부` — 29,340 express rows, so Line 1 and 9 express service is flagged
- Times run past 24:00 for after-midnight service

Stations per line: 1 → 102, 2 → 51, 3 → 44, 4 → 51, 5 → 56, 6 → 39, 7 → 53,
8 → 24, 9 → 38.

**Through-running is included.** Line 1 at 102 stations is its full extent, and
terminals include 천안, 신창, 인천, 동인천, 서동탄, 양주, 오이도 — all Korail
track. No operating-boundary problem.

### OD — one day, whole network

[서울특별시_지하철 역별 OD](https://www.data.go.kr/data/15113638/fileData.do).
Labelled "샘플 데이터" on the portal; **that label is wrong.** The published file
is the real thing.

- **229,365 rows**, one date: **2023-12-31**
- **27 lines** — not just 1–9, but 경의중앙, 분당, 인천1/2, 공항철도, 신분당,
  우이신설, 경춘, 신림, 서해, 김포골드, 경강, 수인, 의정부, 진접, 에버라인
- **642 distinct stations**, 3,464,557 total passengers
- Columns: `기준일자, 승차_호선, 승차_역, 하차_호선, 하차_역, 총_승객수` plus a
  breakdown by fare category

It reads unmistakably as a holiday Sunday: busiest origins are 잠실, 홍대입구,
명동, 혜화, with 강남 only tenth; top pairs are 을지로입구→홍대입구 and 잠실→강남;
3.46M trips against a normal weekday's ~7M. Nightlife, not commute.

That last sentence is the reason for "Getting off New Year's Eve", and it is
also the limit on what re-levelling can fix: the volumes can be moved onto a
weekday, but 강남 ranking tenth is a property of the *pattern*, and the pattern
is all we have.

### Hourly counts — every day, lines 1–8

[서울교통공사_역별 일별 시간대별 승하차인원 정보](https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do)
(OA-12921). A per-year file archive going back to 2010; we took 2023 and 2024.

Columns: `연번, 수송일자, 호선, 역번호, 역명, 승하차구분` then hourly bins
`06시이전, 06-07시간대 … 23-24시간대, 24시이후`. Boarding and alighting are
separate rows via `승하차구분`, not separate columns.

- 2023 file: 199,270 rows, **365 days, 2023-01-01 to 2023-12-31**, 8 lines,
  282 station numbers / 247 names — **546 rows on 2023-12-31**, our date
- 2024 file: 199,424 rows, 366 days, same shape

Coverage is 서울교통공사 관할 only: Line 1 just 청량리–서울역, Line 4 to 남태령,
Line 8 to 암사역사공원. Narrower than the OD, which is the main scope constraint.

### Daily counts — every day, all 27 lines

[서울시 지하철호선별 역별 승하차 인원 정보](https://data.seoul.go.kr/dataList/OA-12914/S/1/datasetView.do)
(OA-12914), `CARD_SUBWAY_MONTH_<YYYYMM>.csv`, one file per month back to 2015.
`fetch_ridership.py` pulls them.

Columns: `사용일자, 노선명, 역명, 승차총승객수, 하차총승객수, 등록일자`. No hours —
that is the trade for the coverage. **UTF-8 with a BOM**, not CP949 like the
rest.

- ~19,000 rows a month: one per station per day
- **622 stations, 27 lines**, including 경부선/경인선/경원선/장항선 (line 1 track),
  과천선/안산선 (line 4), 일산선 (line 3), 경의선/중앙선, 공항철도, 분당, 수인,
  신림, 우이신설, 경춘, 경강, 서해
- Joins to `stations.json` at **99.6%** on name alone

The line names are Korail's routes rather than the through-service riders think
they are on, so `CARD_LINE_ALIAS` in `build_od.py` translates them. That
mapping is also what tells 5호선's 양평 from 경의중앙선's — the one name shared
by two complexes on this network.

**Not covered**, because they settle fares outside Seoul: 인천1/2호선,
신분당선, 김포골드라인, 서해선's 의정부 sibling, 의정부선, 진접선 and 7호선's
Incheon extension — 103 of our 626 complexes. In the re-levelling they simply
get no target and take whatever the Seoul end of their trips implies, which is
the right answer for lines whose OD is only their traffic *to and from* Seoul
anyway. See "What the OD actually contains".

### Congestion — 혼잡도, the output check

[서울교통공사_지하철혼잡도정보](https://www.data.go.kr/data/15071311/fileData.do)
(15071311), an anonymous `data.go.kr` file download, updated quarterly.

Columns: `구분, 호선, 역번호, 역명, 상하구분` then 39 half-hour columns
`5시30분 … 00시30분`. 1,671 rows = 282 stations × direction × three day types.
`구분` is 평일 / 토요일 / 일요일; `상하구분` is 상선/하선, or 내선/외선 on 2호선's
loop.

혼잡도 is riders on board over 정원, as a percentage — so **34% is a full seated
train**, not 100%. Lines 1–8 within 서울교통공사's boundary only, the same scope
as the hourly file.

**9호선 is not in this file** — it is not 서울교통공사's line. 서울시메트로9호선
publish their own at 서울 열린데이터광장 `OA-22197`, `congestion_line9.xlsx`:
eight sheets, 상선/하선 × 평일/휴일 × 일반/급행, keyed by station name with no
역번호 at all. Needs `openpyxl`. See "9호선 is checked separately".

See "Checking the build against 혼잡도" for what these are used for.

### The account wall

The obvious route to hourly counts is the Seoul Open Data API
(`CardSubwayTime`, 619 stations/month, monthly averages). It needs a
data.seoul.go.kr account, and signup requires Korean identity verification —
i-PIN with an 외국인등록번호, a Korean carrier phone, or a Korean bank
certificate. **Not passable from abroad.**

Route around it rather than fighting it: `data.go.kr` file downloads and the
Seoul portal's `nio_download.do` both work anonymously, and the OA-12921 archive
turned out to be strictly better anyway — real single days rather than monthly
averages. `fetch_hourly.py` is kept only as the sole route to hourly data for
Line 9 and the Korail lines, and is unusable without a key.

**`nio_download.do` has one trap.** It is a POST with `infId` and `seq`, and
also `infSeq`, which is **per dataset** — 3 for OA-12914. Get it wrong and the
server returns HTTP 200 with an HTML page containing
`alert('잘못된 접근입니다. 파일 목록에서 다시 선택하세요.')`, not an error status.
`fetch_ridership.py` scrapes the file list from `fileView.do` (which needs no
login) to map file name → `seq`, and refuses loudly if what comes back starts
with `<html`.

**T-Data is the same wall.** `t-data.seoul.go.kr` carries 교통카드 대중교통
이용정보 — raw T-money taps with transfer information, which would be better
than any of this — but it uses Seoul's integrated login, so it is the same
account and the same dead end.

### Geometry — from OSM

Two Overpass pulls, both anonymous.

`osm_stations.json` is a plain node pull over the Seoul-area bbox; matching our
458 timetable stations by Korean name hits **456**, missing only 연천 (which
opened in December 2023, after our date anyway) and 부천시청.

`osm_routes.json` is the better one: 165 `type=route` relations whose names
contain 호선, covering all of lines 1–9 plus Incheon 1/2. Each carries
`stop`-role nodes in running order *and* `way` members with full geometry — so it
gives ordered stations with unambiguous identity **and** curved track in a single
pull. Use it as the primary source, falling back to name matching only for gaps.

That matters because name matching alone is ambiguous: 83 of our station names
hit more than three OSM nodes, and Seoul genuinely reuses names across distant
places (양평 on line 5 versus 양평 on 경의중앙선). Route membership sidesteps it.

Line 1 appears as many relations, one per through-running pattern
(광운대 → 천안, 인천 → 동두천 급행, and so on); pick the most complete per line and
direction rather than merging them all.

nycriders drew straight lines between stops and lists that as a known
limitation. Seoul gets real track geometry —
`../londonriders/fetch_track_shapes.py` is the model.

## Getting off New Year's Eve

Read this before quoting any figure off a weekday build. Investigated
2026-09-04, after the map's midday peak looked wrong.

### It was never New Year's Eve. It was Sunday.

The complaint was that the map peaks around noon instead of at the morning
rush, and the obvious suspect was the date — 2023-12-31 is New Year's Eve, so
of course it is strange. It is not strange. Measured against the other 51
Sundays of 2023, from `hourly_2023_raw.csv`, which holds every day:

```
               <06 06-07 07-08 08-09 09-10 10-11 11-12 12-13 13-14 14-15 15-16 16-17 17-18 18-19 19-20 20-21 21-22 22-23 23-24   >24
NYE 12-31      1.1   1.4   1.8   3.0   4.4   5.0   5.3   6.4   6.8   7.0   7.4   8.0   7.9   7.1   6.0   6.0   5.6   4.7   2.0   3.1
Sundays        1.3   1.7   2.3   3.8   5.2   5.7   5.9   7.1   7.4   7.4   7.7   8.1   7.8   6.9   5.8   5.7   5.0   3.7   1.5   0.0
Wednesdays     1.4   2.9   8.2   9.9   5.7   3.8   3.7   4.1   4.2   4.4   5.1   6.1   9.0  12.0   5.8   4.5   4.3   3.3   1.4   0.3
```

2023-12-31 peaks at 16–17 like **48 of the other 51 Sundays**, and its total,
2.66M boardings on lines 1–8, sits 5% above the Sunday mean of 2.53M. The
*only* thing about it that is not an ordinary Sunday is the `>24` column, 3.1%
against zero — the extended service for the Bosingak bell, which is what
`EXTEND_LAST_HOUR` exists for.

So the broad afternoon hump is what a Seoul Sunday looks like. **Nothing that
reweights the hours of a Sunday can produce a weekday**, because a weekday's
8.2% in 07-08 and 12.0% in 18-19 are trips that are not in the file at all.
Getting a rush hour means changing which day's *volumes* are used, not which
hours they are spread over.

### There is no weekday OD, and this was checked properly

Searched 2026-09-04, all anonymous, all dead ends:

| Source | What it turned out to be |
|---|---|
| `data.go.kr/15113638` | still 2023-12-31, still the only date |
| `data.go.kr/15135167` 호선별 사용자 유형별 OD | line-to-line, not station-to-station, 100-row sample |
| `data.go.kr/15134768` 철도역 구간 | static section geometry, no ridership |
| `OA-20501` 대중교통 O/D | "파일이 없습니다" |
| T-Data `t-data.seoul.go.kr` | raw T-money taps with transfers — the real prize, behind Seoul's integrated login |
| `OA-12252` CardSubwayTime | hourly, all operators, **API only** — no file archive, needs a key |

The account wall is the same one as before; see "The account wall". Seoul's
integrated login covers data.seoul.go.kr and T-Data alike, and it is not
passable from abroad. `od_request.md` is the remaining route to a real weekday
OD and is worth sending — the provider's own description says the published
file was produced 제공요청에 의해.

### The two files that made a weekday possible anyway

Both are anonymous downloads, and `fetch_ridership.py` pulls them.

**`data.go.kr/15071311` 서울교통공사 지하철혼잡도정보** — riders on board as a
percentage of 정원, per station, per direction, per **30 minutes**, for a
typical **평일 / 토요일 / 일요일**. Lines 1–8, 282 stations, 1,671 rows.

This is *the quantity the map draws*, published. 서울교통공사 derive it the same
way we do — their description says 교통카드 데이터 분석 with an optimal-route
computation — so it is not an independent measurement of ridership so much as
an independent run of the same idea by the people who hold the raw taps. It is
the first thing this project has had to check its **output** against; every
other check in `validate.py` checks an input.

**`OA-12914` CARD_SUBWAY_MONTH, via `nio_download.do`** — daily boardings and
alightings per station, one file per month back to 2015. No hours, but **all 27
lines**: 622 stations including the Korail through-running sections, 공항철도,
분당, 신림, 우이신설, 경춘. It joins to `stations.json` at 99.6% on name alone,
and every miss is a case already documented under "Known data gaps" (자양,
불암산, 암사역사공원, 연천, 청산).

That split — level measured everywhere, shape measured on lines 1–8 — is
exactly the split `build_od.py` was already built to exploit.

### What a weekday build actually is

Two fits, not one.

1. **Re-level the pairs (Furness).** Take the measured pair totals as a seed
   and scale them, `od[o,d] × a[o] × b[d]`, until each station's daily origin
   and destination totals match what that station did on a typical weekday.
   The target is `kept_b[i] × (weekday gate total / 2023-12-31 gate total)` —
   a **ratio**, so whatever fraction of a station's gate traffic the OD covers
   on the measured date carries over untouched rather than having to be
   estimated.
2. **Split into hours (IPF), as before.** Weekday hourly gate counts are
   measured, so this half is unchanged in kind.

Reference days are the **Tue–Thu of 2023-11**, minus 2023-11-16 (수능, when
service is shifted and offices open late — it is the quietest Tue–Thu of the
month). Thirteen days. Monday runs ~4% below the Tue–Thu mean and Friday ~3%
above, which is the usual reason for the convention and holds here. 2023-11 is
the ordinary month nearest the OD date, so level and pair structure come from
the same season. All of this lives in `daytype.py`.

**The honest limit.** Volumes are measured; *who goes where* is carried over
from the Sunday. Furness fixes every station's row and column total, which is
most of the information in an OD matrix, but it cannot invent a commute that
the Sunday did not contain. A pair that was disproportionately busy on a
Sunday night — 을지로입구→홍대입구, say — stays disproportionately busy relative
to the rest of its row.

**This has to be said on the page, not just here.** It sits in the panel as
`#daynote`, in the same quiet grey as the static-view note, *outside* the
`about the data` fold — a caveat behind a disclosure triangle is not a caveat.
One sentence for the claim and one for its limit:

> Passenger numbers are measured. Which journeys make them up is carried over
> from a Sunday — the only day Seoul publishes station-to-station data for.

It is hidden on `--day nye`, which is the one build with nothing to qualify,
and shown in both view modes and both languages (`STR.en.dayNote` /
`STR.ko.dayNote`). The longer version stays in `about`. Anything published from
this map — a caption, a post, a screenshot — needs the same two sentences.

### Does it matter which ordinary month you pick?

Barely, and the exceptions are real rather than noise. Comparing the per-station
weekday/2023-12-31 boarding ratio computed from **2023-11** against the same
thing from **2024-11**, over the 499 stations with more than 500 boardings on
the OD date:

```
network-wide weekday ratio   2023-11 x1.853   2024-11 x1.893   (2.2% apart)
per-station, level removed   median 0.999   p10 0.959   p90 1.058
                             469 of 499 agree within 10%  (94%)
```

The stations that do move moved for reasons: **구리 ×0.64** because the 8호선
별내 extension opened in August 2024 and took its traffic; **구성 ×1.41**
because GTX-A opened there in March 2024. Those are the network changing, not
the method wobbling — and they are an argument for keeping the reference month
close to the OD date, which is why `daytype.REF_MONTH` is 2023-11 rather than
the more recent file. `build_od.py --month 202411` runs the comparison.

### The check that says the machinery is not doing something silly

`build_od.py --day sunday` re-levels by **×1.02** and reproduces the
New Year's Eve profile minus the midnight tail:

```
--day nye     16:00  269.1  ############################################################
--day sunday  16:00  279.9  ###########################################################
--day weekday 08:00  736.6  ############################################################
              12:00  251.3  ####################
              18:00  728.9  ###########################################################
```

The Furness step does nothing when the target day is the same kind of day as
the measured one, which is what it should do, and it does something large when
it is not. Total trips go 3.40M → 3.42M for Sunday and 3.40M → 6.34M for a
weekday, against a real weekday's ~7M gate journeys.

### Checking the build against 혼잡도

`python validate.py --congestion` — the only check here that looks at the
output. For every segment on lines 1–8 it computes riders-per-hour over
trains-per-hour over the line's 정원 and compares against the published figure
for that station, direction and hour.

- **Direction** is assigned from the station-number step, on the rule that
  Seoul numbers its stations in the 하행 direction (2호선's loop is 내선/외선
  instead, 외선순환 being the increasing one). That rule is an assumption, and
  a silent swap would leave every correlation positive — both directions are
  busy at both rushes — just worse. So the check runs it both ways and prints
  which won.
- **Read shape before level.** The hourly correlation per line is measured on
  both sides and independent of any capacity assumption. The level carries our
  160-per-car 정원 and theirs, so an offset shared by *every* line is a
  disagreement about what 정원 means, not a routing bug.
- **A sampled build cannot be read for level at all.** It keeps only the
  segments near the origins it kept, and those carry their riders in full, so
  the ratio is neither 1× nor 1/n. `--sample` says so out loud.

The per-segment train count this needs is new in `stats.json` as `"n"`,
counted over **every** trip and not only those carrying riders — an empty train
still dilutes the average. Segments also carry `"ca"`/`"cb"`, the platform
codes, which is what makes the direction test possible.

### 9호선 is checked separately, and split 일반 / 급행

9호선 is not 서울교통공사's line, so it is not in their file. 서울시메트로9호선
publish their own — 서울 열린데이터광장 `OA-22197`, an xlsx of eight sheets:
상선/하선 × 평일/휴일 × **일반/급행**. `fetch_ridership.py` pulls it,
`load_congestion9()` reads it. Only 40 populated rows a sheet; the 15 MB is
Excel bloat.

**The express split is the point, and it is why `stats.json` grew `hx`/`nx`.**
Those are the 급행 subset of `h`/`n` — the express riders and express trains on
that segment — so the local is what is left after subtracting them. Comparing
our blended average against either published sheet would have been meaningless:
9호선's 급행 runs at **65.4%** of 정원 against the 일반's **37.1%** in the same
hours, which is exactly why the operator publishes them apart. One number
describes neither.

The sheets are keyed by station *name* — no 역번호 anywhere in the file — which
is safe on 9호선 because it shares no station name with itself. Its day split is
also coarser than the 1–8 file's: 평일/휴일 only, so a Saturday borrows 휴일.
That is the file's limit, not ours; `LINE9_DAY` records it.

#### And it immediately found something: RAPTOR over-fills the 급행

The first full run of the split check, 2026-09-04:

```
line     ours  published  ratio   corr
9급행    90.6%      78.6%  1.15x   0.98
9일반    19.4%      41.5%  0.47x   0.95
```

Both shapes are excellent — 0.98 and 0.95, the best on the network — so the
*timing* is right on both services. The *split between them* is not. Counting
riders carried rather than percentages:

```
              ours      published-implied   ratio
express  1,945,523              1,898,133   1.02x
local    1,208,624              2,825,259   0.43x
express share of riders:  ours 61.7%   published 40.2%
```

**The express is right and the local is starved.** That is RAPTOR doing exactly
what it was asked: it minimises journey time, wait included, so every rider who
*can* take a 급행 does. Real passengers do not behave that way — the express is
already full, some would rather sit, some will not stand for twenty minutes to
save six. Nothing in the routing represents that, so 9호선's express carries
half again the share it should.

This is worth knowing before the express is drawn as its own service on the
map: doing that would make the most-wrong number in the build the most visible
one.

**Fixed, or at least attacked, by `crowding.py`** — see "Crowding" below. It is
a modelling addition rather than a data one, and it is optional: an ordinary
`build.py` run is still the uncrowded build unless a crowding pass has been
made. The figures in this section are the *uncrowded* ones, kept because they
are what the check found and what the fix has to beat.

It also generalises. The same over-assignment must apply to 1호선's 급행, where
28.8% of riders are on an express in our build; there is no published figure to
check it against, because 1호선's 서울교통공사 stretch has no express service and
the Korail sections that do are outside the file.

#### How much express traffic there is at all

Riders carried past a station over the whole day, from the same build:

| line | all riders | express | express share | segments with express |
|---|---|---|---|---|
| 1호선 | 8,707,927 | 2,503,839 | **28.8%** | 154 of 207 |
| 9호선 | 3,154,164 | 1,945,526 | **61.7%** | 30 of 102 |
| 공항철도 | 743,829 | 3,814 | 0.5% | 4 of 28 |
| **whole network** | 62,759,371 | 4,453,179 | **7.1%** | |

So express is 7% of the map but a *large* share of two lines and essentially
absent from the other twenty. That matters for how it should be drawn: anything
that distinguishes express costs nothing on most of the network, because there
is nothing there to distinguish.

### What it said, on the first full weekday build

2026-09-04, 545 segments matched against the published 평일 figures.

```
direction check: 상선/하선 as assigned r=0.801, flipped r=0.374  -> as assigned

 hour     ours  published   ratio        line     ours  published  ratio   corr
05:00    21.7%      32.7%   0.66x        1       34.7%      34.2%  1.01x   0.87
07:00    50.7%      58.8%   0.86x        2       43.5%      46.1%  0.94x   0.88
08:00    55.6%      65.1%   0.85x        3       39.3%      48.0%  0.82x   0.97
12:00    27.2%      33.3%   0.82x        4       40.3%      49.2%  0.82x   0.94
17:00    49.3%      59.6%   0.83x        5       43.4%      48.2%  0.90x   0.96
18:00    56.3%      69.6%   0.81x        6       31.4%      35.9%  0.88x   0.96
23:00    25.3%      23.7%   1.07x        7       42.9%      53.3%  0.80x   0.96
00:00    15.2%      10.3%   1.48x        8       43.3%      55.7%  0.78x   0.96
                                         9급행    90.6%      78.6%  1.15x   0.98
                                         9일반    19.4%      41.5%  0.47x   0.95
```

**The shape is right.** Every line correlates 0.87–0.98 against a measurement
the build has never seen, and the direction rule wins its own test by a mile.
The morning and evening peaks land in the right hours at the right relative
heights — which is the whole point of the exercise, since a Sunday build would
score near zero here.

**The level is 15–20% low, consistently, and that is the known coverage
limit rather than a bug.** The build routes **6,328,321 riders against
7,837,194 measured weekday gate boardings — 80.7%**. The missing fifth is
journeys the OD does not contain at all, for the reason in "What the OD
actually contains": a trip with neither end on 서울교통공사's network is not in
the source. Re-levelling deliberately preserves that coverage fraction
(`row_t = kept_b × ratio`, a ratio of gate totals rather than an absolute), so
the map under-draws by roughly the amount the OD under-measures. Inflating to
close the gap would be inventing riders.

**Re-measured after `spawn_gaps()` replaced the spawn grid**, and it moved
almost nothing: every ratio from 07:00 on is identical to three decimals and
the per-line correlations shift by at most 0.02. That is the right result
rather than a disappointing one. The grid rewrite was about *which train inside
an hour* a rider catches; 혼잡도 is an average over the trains in a half hour,
so it can see the hourly totals being right and is nearly blind to the
train-to-train lumpiness the rewrite fixed. The numbers to watch for that are
the load step and the platform-arrival histogram, under "Spawn times".

Three smaller things the table shows:

- **05:00 at 0.66×** is the other open-ended bin. `06시이전` is *everything*
  before 06:00, and some of those riders have no train to catch, so the hour
  under-reads. See "The first hourly bin is open-ended too".
- **Line 1 at 1.01×** is not a better model, it is better coverage. 혼잡도 for
  line 1 only covers the ten 서울교통공사 stations, 서울역–청량리, which is exactly
  the stretch where the OD is most complete.
- **00:00 at 1.47×** is the tail of `LATE_BIN_HOURS`. The `24시이후` gate column
  is open-ended and we spread it over one hour on an ordinary day; the
  published figure thins out faster. It is the last twenty minutes of service
  and carries almost nobody, so it has been left alone.

## Crowding — why the build iterates against itself

`crowding.py`. Added 2026-09-04, after the 9호선 check found that RAPTOR puts
half again as many riders on the 급행 as really ride it.

### The problem is not a bug, it is the objective

`build.py` gives every rider the fastest journey, wait included. On a line with
both 급행 and 일반 that means **everyone who can take the express does**,
because it always is faster. Real passengers do not behave that way: the
express is already full, and plenty of people would rather sit on a local for
twenty minutes than stand on an express for fourteen. Nothing in a shortest-path
search represents either fact.

### Why a capacity check does not work here

The obvious fix — refuse to board a train that is full — cannot be dropped into
this code. Whether you can board the 08:05 급행 at 노량진 depends on everyone
who got on upstream at 김포공항 and 여의도. That is a **shared, global**
constraint, and the routing is deliberately the opposite: 624 origins routed
independently across the cores, no worker knowing what any other worker loaded
onto a train. A live "is this train full" test would serialise the whole build
and break the parallelism the run time depends on.

### So iterate, and average

The standard transit-assignment answer, method of successive averages:

```
round 0   build with no penalty              -> loads
round i   loads -> a penalty per segment
          build again; riders now avoid the crush
          average the new loads into the old with weight 1/(i+1)
```

**The averaging is the part that matters.** Without it round 1 empties the
express, round 2 finds it empty and refills it, and the loop rings forever.
Averaging damps that into a fixed point.

Each round is still a fully parallel build, because the penalty is a read-only
table computed *between* rounds and left in `data/crowding.npz`. `build.py`
loads it if it exists and ignores it if it does not, so **an ordinary
`build.py` run is still the uncrowded build** unless a crowding pass has been
made. `crowding.py --reset` puts it back.

### How the penalty enters the search

Not as a cost array — that would mean a generalised-cost RAPTOR and a rewrite
of the scan. Instead the *arrival label* is inflated: riding from `board_si` to
`sj` on a crowded trip lands you at

```
arrr[ti][sj] + pcum[ti][sj] - pcum[ti][board_si]
```

so a crush-loaded ride simply looks longer than it is. Labels stay in seconds
and stay monotone, so the rest of RAPTOR is untouched, and `depc` — catching a
train — stays on real clock time, which is right: you board when it leaves,
however full it is.

**The cost of doing it that way**, written down because it is invisible
otherwise: the labels RAPTOR compares against real departure times are now
*perceived* times, so a rider who has just ridden a crowded train looks like
they arrived later than they did and can miss a connection they would really
make. It is a conservative distortion and a defensible one — someone at the
back of a crush-loaded train really is slower off it — but it is a distortion,
and it grows with the penalty. That is why `crowding.py` reports the median and
p90 penalty and not just the worst: a two-hour tail would wreck the transfers
while looking like it was only affecting express choice.

### The penalty function, and why 110% would have been wrong

Crowding `c` is riders over 정원. The instinct is to treat 100% as a wall. The
published figures say otherwise — every station-direction-halfhour on lines
1–8, weekday:

```
median    27.9%          cells at or above 100%   1.57%
p99      108.1%                            120%   0.46%
max      144.6%   (2호선 사당 외선 08:30)     150%   0.00%
```

**정원 is a design figure, not a physical limit.** Seoul routinely runs above
it and never above ~145%. So the penalty is zero up to `C_FREE = 1.00`, and
rises as a square to `ALPHA` times the segment's own run time at
`C_CRUSH = 1.45`:

```
factor(c) = 1 + ALPHA * clamp((c - C_FREE) / (C_CRUSH - C_FREE), 0, 1) ** 2
```

Square rather than linear so that 110% barely stings and 140% hurts, which is
what the distribution above implies: almost nothing is over 120%, so a linear
ramp would spend its whole range on cells that do not exist.

### What can and cannot be checked

**9호선 is the only line with a published express/local split**, so `ALPHA` is
fitted on one line and applied to all of them. 1호선's 급행 carries 28.8% of
that line's riders in the uncrowded build, with nothing to check it against —
서울교통공사's stretch of 1호선 has no express service at all, and the Korail
sections that do are outside their file. `crowding.py` prints both shares each
round for exactly this reason: one is a calibration target and the other is a
number to watch, and they must not be confused.

## Method

The disaggregation sketch in `metroslist.md` was "divide monthly OD by ~22
weekdays, then split across hours by the origin station's hourly entry share."
That only enforces the origin marginal, leaving the destination mix constant all
day — so the evening flow comes out as the morning one at different volume
rather than genuinely reversed. Not good enough; directionality is the story.

**Fit the hourly OD by iterative proportional fitting against both marginals.**

For each hour `h`, find `X(o,d,h)` such that

- `Σ_d X(o,d,h) = boardings(o,h)` — everyone who tapped in at `o` during hour `h`
- `Σ_o X(o,d,h') = alightings(d,h')` — where `h'` is the *arrival* hour,
  `h' = hour(h + traveltime(o,d))`

The arrival-hour shift is what makes this more than a textbook IPF: the
destination constraint lives in a different time index from the origin
constraint, so the fit couples adjacent hours and the travel-time matrix does
real work.

We now have a **measured daily OD for the exact date the hourly counts describe**,
so it seeds the fit as a prior rather than being modelled. The result matches
measured station-pair totals *and* measured hourly station counts for one real
day. That is as good as this gets without raw tap records.

Downstream is the nycriders shape: RAPTOR each OD-hour cell over the real
timetable, board riders onto specific `열차코드` runs, write `trains.json`,
animate. `../nycriders/build.py` should mostly port over.

## Scope

**v2 is 22 lines, 626 complexes, 761 platforms.** The universe is still set by
the *timetable* — we can only animate trains we have schedules for — but the
timetable is now three files rather than one. Every 수도권 line in the OD is in
except 에버라인.

Measured against the NYE OD, by the same name-and-호선 match `build_od.py` uses:

| network | routable trips | share |
|---|---|---|
| lines 1–9 only | 3,008,413 | 86.8% |
| **all 22 lines** | **3,397,486** | **98.1%** |

That is **+389,073 trips, 11.2 percentage points**.

**A correction worth keeping**, because the first estimate of this was wrong in
an instructive way. Counting by the OD's 호선 label put the lines-1–9 baseline at
83.5%, and implied that relabelling `7호선(인천)` and `진접선` would recover 1.5
points on its own. It does not: `build_od.py` matches on station *name*, so
부천시청 and 진접 were already in the network as line 7 and line 4 stops whatever
the OD chose to call them. The relabel is still right — it is what lets a
complex know which OD rows are its own, which is what makes the 양평 split
possible — but on its own it buys almost no coverage. Measure a change the way
the code will see it, not the way the source files are organised.

**388 of the 626 complexes lack measured hourly counts.** The hourly file is
서울교통공사's own, so it covers lines 1–8 inside their operating boundary and
nothing else: all of line 9, the Korail through-running sections, and every line
added since. That is a much larger unmeasured share than v1 had — 62% of
complexes rather than 39% — so the IPF is doing correspondingly more inferring.
It is still anchored, because almost every trip has at least one end at a
measured station, but it is the weakest part of the wider network and worth
saying out loud.

They do not have to be dropped. The OD file gives every station's exact daily
boardings and alightings for the day; what is missing is only the hourly
*shape*. So constrain the 280 measured stations hard, leave the other 178 with a
daily-total constraint only, and let IPF infer their hours through the fit — a
trip from an unmeasured suburb station into 강남 still has to arrive in an hour
consistent with 강남's measured alighting profile. Since 99.3% of trips touch at
least one line 1–9 station, almost nothing is fitted blind.

Extending past lines 1–9 needs schedules for the Korail 광역전철 lines, 신분당선,
공항철도 and the rest. **Those schedules have been found** — see the next
section. KTDB's nationwide GTFS was the assumed route and is no longer needed;
Korail's own hourly counts are monthly rather than daily, so they would be a
weaker constraint than what we have here.

## Extending past lines 1–9

Researched 2026-09-03. Two separate findings.

### Two lines cost nothing — the OD splits by operator, we split by line

`7호선(인천)` and `진접선` are not missing lines. They are stretches of lines 7
and 4 that a different operator runs, and the OD labels them by that operator.
Every one of their stations is already in our timetable:

- `7호선(인천)` — 11 stations (석남 … 부천종합운동장), all present under line 7
- `진접선` — 3 stations (진접, 오남, 별내별가람), all present under line 4

Both opened before our date (석남 2021-05, 진접 2022-03). Mapping those two OD
labels onto lines 7 and 4 takes routable trips from **83.6% to 85.1%** —
+52,725 trips — with no new data and no new geometry.

### Everything else is in one anonymous download

**[전체_도시철도운행정보](https://data.kric.go.kr/rips/M_01_01/detail.do?id=900)**
from 레일포털 (data.kric.go.kr, 국가철도공단). This is the upstream source of
data.go.kr's `전국도시철도운행정보표준데이터`; go to KRIC directly, the portal
listing only points back here. **No login, no key** — the download button is a
plain GET:

```
https://data.kric.go.kr/rips/dataset/download.file?type=filedata&id=900&operation=1
```

18 MB xlsx, 223,425 rows, 39 lines nationwide, saved as
`data/kric_urbanrail_timetable.xlsx`. Columns: `열차번호, 노선번호, 노선명,
운행구간기점명, 운행구간종점명, 운행유형, 요일구분, 운행구간정거장,
정거장도착시각, 정가장출발시각, 운행속도, 운영기관전화번호, 데이터기준일자`
(`정가장` is their typo, not ours). Arrival *and* departure times, a 급행 flag,
and weekday/Saturday/holiday variants — the same shape as the Seoul file, which
means `load_patterns` needs a second reader, not a second pipeline.

**It covers all 16 remaining lines**, and the day types we need:

| OD line | in KRIC as | trains | row shape |
|---|---|---|---|
| 분당선 + 수인선 | 수인분당선 | 779 | per stop |
| 경의중앙선 | 경의중앙선 | 438 | per stop |
| 경춘선 | 경춘선 | 216 | per stop |
| 서해선 | 서해선 | 320 | per stop |
| 경강선 | 경강선 | 221 | per stop |
| 공항철도1호선 | 인천국제공항선 | 794 | packed |
| 신분당선 + (연장2) | 신분당선 | 866 | packed |
| 우이신설선 | 수도권 경량도시철도 우이신설선 | 940 | packed |
| 신림선 | 수도권 경량도시철도 신림선 | 696 | packed |
| 인천1호선 | 인천지하철 1호선 | 574 | packed |
| 인천2호선 | 인천지하철 2호선 | 840 | packed |
| 김포골드라인 | 김포골드라인 | 866 | packed |
| 의정부선 | 의정부 | 793 | packed |
| 에버라인선 | 에버라인 | 526 | packed |

**The file carries two row shapes and you must handle both.** The Korail lines
write one row per station stop, exactly like our Seoul file. The metro
operators write **one row per train**, with the station list and the times
packed into single cells — and each operator picked its own conventions:

```
신분당선     D19-광교+D18-광교중앙+…      D19-10:17+D18-10:21+…      '+' , HH:MM
우이신설선    001-신설동+002-보문+…        001-:+002-5:31+…           '+' , H:MM, ':' = null
김포골드라인   001-장기역+002-운양역+…      001-5:26:10+002-5:28:39+…  '+' , H:MM:SS, 역 suffix
인천1호선    3125-예술회관,3126-인천터미널,… 3125-5:32:00,…            ',' , HH:MM:SS
공항철도     001-서울+002-인천공항1터미널,…  001+06:00+002+06:45        '+' both as sep and pair
```

Some lines fill only `정가장출발시각` and leave `정거장도착시각` empty.

**Station names need an alias step.** KRIC truncates to about three characters
and prefixes `신` where a name is used twice on the network:

- `강남구청`→`강남구`, `디지털미디어시티`→`디엠시`, `압구정로데오`→`로데오`,
  `평내호평`→`평내호`, `세종대왕릉`→`세종릉`, `남동인더스파크`→`남동인`
- `판교`→`신판교`, `이매`→`신이매`, `소사`→`신소사`, `수원`→`신수원`,
  `인천`→`신인천`, `초지`→`신초지`

8 of the 14 lines join to the OD at 100% as-is; the rest leave about 35 names
over. Do not hand-write that table — both KRIC and `osm_routes.json` list stops
in running order, so align the two sequences positionally and let the order
disambiguate the truncations, the way `build_stations.py` already uses route
membership rather than name matching.

**Geometry needs a wider Overpass pull.** `osm_routes.json` was fetched with a
filter on relation names containing `호선`, which is why it has Incheon 1 and 2
but none of 경의중앙, 수인분당, 신분당, 공항철도, 우이신설, 신림, 김포골드,
경춘, 서해, 경강, 의정부 or 에버라인. Widen the filter before re-running
`build_stations.py`.

**Vintage.** `데이터기준일자` is per operator and ranges from 2022-05
(우이신설) to 2026-06 (신분당). Our OD is 2023-12-31 and our own Seoul timetable
is already a 2026 one, so this adds no new kind of problem — but it makes the
existing "clip the network to stations present in the 2023 OD" rule load-bearing
for more of the map.

**What it buys**, cumulatively, on the NYE OD:

| after adding | routable trips |
|---|---|
| (today) lines 1–9 | 83.6% |
| + 7호선(인천), 진접선 relabel | 85.1% |
| + 분당선 | 88.7% |
| + 경의중앙선 | 91.4% |
| + 공항철도 | 93.6% |
| + 신분당선 | 95.0% |
| + the remaining nine | ~98% |

### How it was actually wired in — and the two bugs adding lines exposed

`kric.py` writes `data/timetable_extra.csv` in **exactly the columns of
`timetable_raw.csv`**, so `build_stations.py` and `build.py` gained fifteen
lines by reading a second file rather than by growing a second code path. The
awkwardness — five packed-cell conventions, truncated names, per-operator day
labels — is all absorbed there. Station codes are synthetic five-digit numbers
from 70001, clear of the four-digit 서울교통공사 codes.

Names are resolved back to the **OD's** spelling, because the OD is what the
ridership is keyed on: exact match, then unique prefix, then the `신` strip,
then a seven-entry alias table for the initialisms (`디엠시`→`디지털미디어시티`).
The run ends by asserting that every OD station on every new line was reached;
it currently reaches all 313. Five KRIC stops resolve to nothing and are
dropped, all correctly — the 인천1호선 검단 extension opened 2024-03, and
경강선 성남 and 경의중앙 운천 have no OD rows at all.

`항공대` was the one name settled by position rather than by rule: KRIC keeps
the old 경의선 name, the stop sits between 강매 and 수색, and that is where 화전
is. Check a doubtful alias against the stop sequence, not against a map search.

**Two bugs that only appeared once the network got bigger:**

1. **`양평` is two stations 27 km apart** — line 5 in 영등포구, 경의중앙선 in
   양평군 — and complexes were keyed on the ridership name alone, so they merged
   and the averaged coordinate landed in a field between them. The README had
   warned about exactly this pair. `build_stations.py` now splits any complex
   whose platforms are more than `SPLIT_M` apart, and records per complex which
   OD 호선 labels are its own; `build_od.py` looks trips up on **(name, 호선)**
   with a name-only fallback for the 625 names that are unambiguous. Real
   interchanges are nowhere near the threshold — 서울역, with four lines, spreads
   about 400 m.
2. **`osm_routes.json` was filtered on `호선` in the relation name**, so none of
   the commuter or light rail lines had track. `fetch_osm.py` replaces both
   ad-hoc Overpass pulls; the relation-name patterns live in `lines.py` and are
   anchored so `수도권 전철 1호선` cannot swallow `인천 도시철도 1호선`, and so
   공항철도 picks up neither the terminal shuttle nor the maglev.

**에버라인 is the one line left out.** The KRIC file gives it operating windows
and a headway instead of a timetable (`운영시간 - 17:00 ~ 20:00 운행간격 - 4분`),
국가철도공단 publishes its stations but not its schedule, and 용인시 publishes
only ridership. It is 0.05% of the day's trips. Generating stop times from the
headway would work — it is what GTFS `frequencies.txt` is for — but it would be
the second invented thing in a pipeline that has been careful to have exactly
one, so it is left out and said so on the page.

**Checked and rejected:** `한국철도공사_열차운임 및 시간표`
([15052169](https://www.data.go.kr/data/15052169/fileData.do)) is 51 rows of
fares, not a timetable. The KRIC Open API
(`openapi.kric.go.kr/openapi/trainUseInfo/subwayTimetable`) needs a key and the
file makes it unnecessary.

## The static view

`day` in the top-right swaps the animation for the whole day at once. The
animation answers *where is everyone right now*; this answers *how much moves
through here*, which is not a question you can get at by watching dots go past.

It is the **same routing, summed rather than sampled** — one more pass over the
numbers `build.py` already has, not a second model. `write_stats()` emits
`data/stats.json`: per station, boardings and alightings by line and hour; per
segment, riders carried past that point by hour. About 0.5 MB.

- **Line thickness** is riders carried past that point. **Circles** are
  boardings.
- The time slider becomes an hour selector, `0` being the whole day. The two
  are scaled separately — an all-day figure is the sum of twenty-one hours, so
  sharing one scale made every all-day line three times too thick — and each
  view is pinned to its own busiest value, so a quiet hour looks quiet instead
  of being renormalised back up to full width.
- Segments follow the **real track**, not chords between stations.
  `build_shapes.py` already works those polylines out for the animation, so it
  now also writes them to `data/link_shapes.json` rather than having the static
  view solve the same problem again.
- **Dashed segments are the partial-coverage lines**, with a legend key and a
  note in the panel. `build_stations.py` computes each line's within-line trip
  share from the OD and marks anything under `PARTIAL_BELOW` (20%); Seoul's
  nine land at 36–55% and every other operator at 0.3–12%, so nothing sits near
  the boundary. Their tooltips say "to/from Seoul only" too — the caveat
  travels with the number rather than living only in a footnote.

Why dashes and not colour or opacity: the lines already carry meaning in colour,
and dimming them would read as "less busy", which is the exact wrong idea. A
dash reads as "this line is drawn differently", which is what we mean. It needs
two MapLibre layers because `line-dasharray` cannot be data-driven.

## Two languages

`EN` / `한국어` in the panel header switches everything the page prints, the way
`../japanriders/` does. A device in `Asia/Seoul` opens in Korean; everything
else opens in English. Timezone rather than `navigator.language`, because that
is the OS locale and not where the reader is.

Three sources of text, and only the first is written by hand:

- **The furniture** — panel, buttons, legend, tooltip wording — is the `STR`
  table at the top of `index.html`. The counted strings are *functions*, not
  templates with holes in: Korean puts its counter after the number and orders
  the clauses differently, so `'159 waiting'` and `'159명 대기'` cannot come out
  of one format string.
- **Station names** come from OSM's `name:en`, which the node pull we already
  make carries on 811 of its 813 stations — so English costs no extra download
  and no romanisation of our own. `build_stations.py` cleans them (a *trailing*
  bracket is a 부역명 and goes; `Jongno 3(sam)-ga` keeps its bracket because it
  is mid-name) and votes across the several nodes a big interchange has, since
  they do not all spell 서울역 the same way. The five complexes with no OSM node
  are in `EN_FALLBACK`. A station with no English falls back to the Korean and
  `build_stations.py` prints a `NO ENGLISH NAME` list; it should stay empty.
- **Line names** are `display` / `display_en` in `lines.py`. There are only 22
  and their English is house style rather than data — 경의중앙선 is signed
  "Gyeongui-Jungang Line", which no romaniser produces — so they are written
  out.

The **basemap** switches too: openfreemap's dark style labels with `name:latin`
and `name:nonlatin` concatenated, so `applyBasemapLang()` rewrites `text-field`
on every symbol layer that draws a name. One trap there — do **not** gate that
on `map.isStyleLoaded()`. Switching language sets off a round of glyph loading,
so the style reads "not loaded" for a second or two afterwards and gating on it
silently drops every other switch. `getStyle()` throwing is the real "too
early" signal.

## What the OD actually contains — read this before quoting any figure

Found by `validate.py --coverage` on 2026-09-03, after the 15 lines went in.
**The OD holds only trips that touch 서울교통공사's own network.** A journey
confined to another operator's territory is not in the file at all.

The proof is a single pair of rows. `7호선` and `7호선(인천)` are the *same
line* — the OD splits them by which operator settles the fare:

| OD label | boardings | within-line | share |
|---|---|---|---|
| `7호선` | 293,469 | 129,854 | **44.2%** |
| `7호선(인천)` | 19,868 | 264 | **1.3%** |

Seoul's own lines all sit at 36–55% within-line trips, which is what a real
metro line looks like. Every other operator sits at 0.3–12%. Trips with both
ends on a non-Seoul operator are **0.03% of the entire file**.

So for the lines beyond 서울교통공사, the map shows **their traffic to and from
Seoul, not their ridership**. 인천1호선 draws 12,043 boardings here against a
real daily figure an order of magnitude higher; almost every Incheon-internal
journey is missing. The same caveat applies to the Korail-operated outer
sections of lines 1 and 4, which is why `1호선` sits at 36% while `2호선` sits at
55% — and that part predates the new lines. It was simply never measured before.

This does not make those lines wrong to draw. "Who travels between 인천 and
Seoul tonight" is a real and interesting thing, and it is what the file
measures. But it is not "how busy 인천1호선 is", the two must not be conflated in
any caption, and a static view that presents these as ridership totals would be
straightforwardly misleading. `validate.py --coverage` prints the table so the
number is never guessed at.

## Open questions

- ~~Does the timetable cover Korail through-running?~~ **Resolved — yes.**
- ~~Is the OD file real or a sample?~~ **Resolved — real, 229,365 rows.**
- ~~**Which day.** 2023-12-31 is all we have and it is New Year's Eve.~~
  **Half resolved, 2026-09-04.** It is not New Year's Eve that is the problem,
  it is Sunday — the two are indistinguishable in the hourly profile. The map
  now builds a weekday by re-levelling the measured pairs onto measured weekday
  station totals; see "Getting off New Year's Eve". A real weekday OD would
  still be better, and `od_request.md` is still the route to one. Nothing else
  found: the search is written down so it does not get repeated.
- ~~Station name joins will be the fiddly part.~~ **Resolved, and easier than
  feared.** OD names and hourly names match *exactly* — both use the
  parenthesised style, `잠실(송파구청)`. The timetable uses plain names but shares
  the station code with the hourly file (`0150` ↔ `150`, matching on 280 of 282).
  So: timetable ↔ hourly on code, OD ↔ hourly on (line, name), and strip the
  parenthetical only when reaching the timetable.
- **The timetable is 2026, the ridership is 2023.** Stations opened since — the
  8호선 별내 extension is 6 of them — exist in the timetable with no ridership,
  and would otherwise show as trains running empty down a branch that did not
  exist on the night we are drawing. Clip the network to stations present in the
  2023 OD file.
- **Express trains.** RAPTOR will pick 급행 runs where faster, which is correct,
  but Line 9 express crowding is notorious — worth eyeballing once it runs.

## Note for the rest of the list

The IPF-from-marginals approach applies equally to Paris and Bengaluru, which
`metroslist.md` files under Tier 3 precisely because they have hourly station
counts but no OD pairs. Both would promote.

The Furness step generalises further, and more usefully: **one measured OD of
any day, plus station totals for the day you want, gets you that day.** That
turns a single-snapshot OD — which is what most cities release, if they release
one at all — from a one-day map into a day-type map. The cost is stated in
"Getting off New Year's Eve": volumes measured, interaction structure borrowed.

## Pipeline

```
fetch_hourly.py    # CardSubwayTime -> data/hourly.csv
                   #   needs a Seoul API key, which cannot be obtained from
                   #   abroad. Superseded by the OA-12921 archive for
                   #   everything except lines 9 / Korail hourly counts.
                   #   Kept only because it is the sole route to those.
fetch_ridership.py # 혼잡도 (lines 1-8 and 9호선) + all-operator daily counts
daytype.py         # which day the map draws; the one place that decides
crowding.py        # optional: iterate build.py so crowded trains lose riders
build_stations.py  # complexes, platforms, coords, track geometry
build_od.py        # Furness onto the day, then IPF into hours -> od_hourly.npz
build.py           # RAPTOR routing + rider assignment -> trains.json
build_shapes.py    # bend train paths onto the track (build.py calls this)
index.html         # MapLibre animation, nycriders house style
```

`build.py` takes about **10 minutes** for all 624 origins on a weekday build,
shaping included — 7 minutes routing, 3 shaping. A Sunday is roughly half that:
the work scales with riders, and a weekday carries 6.34M against 3.40M. It
routes across the cores bar two, which it leaves alone so the machine stays
usable (`JOB_HEADROOM`); `--jobs` overrides that. Each worker wants ~300 MB, so
fourteen of them is a bit over 4 GB. `--sample 60` uses 11 origins and takes
~55 seconds, most of which is the shaping pass — add `--no-shapes` and it is
under 25. That is what to use while changing anything; the printed invariants
are just as meaningful on a sample as on the full run.

It used to take an hour and a half, and the whole difference is in how RAPTOR's
inner loop is fed. Two things did it:

- **Plain Python lists, not numpy, inside the scan.** `deps[ti, sj]` boxes a
  scalar and `np.searchsorted` on a column view spends ~2.7 µs on dispatch
  before it looks at anything, and the scan does that sixteen thousand times
  per search. The same numbers as lists of ints with `bisect_left` are ~7x
  faster; the arrays are still there for everything outside the loop.
  `prepare_scan()` builds both layouts — departures column-major because that
  is what gets bisected, arrivals row-major because that is read along a trip.
- **Not bisecting at all, most of the time.** Once you are on trip `ti`, an
  earlier one is only catchable if `dep[ti-1]` at this stop is still ahead of
  you. That is one array lookup, and it answers the question outright for the
  large majority of stops. Three of the 397 patterns have a train overtaking
  another partway along, so their columns do not rise and they keep the full
  search; `pattern["srt"]` is that flag.

Then the origin loop went across processes. Each `(origin, spawn time)` search
is independent, so workers rebuild the world from disk — the scan tables parse
faster than they unpickle — and hand back flat `(slot, riders)` arrays rather
than nested dicts. The negative side of every waiting bubble is exactly its
boarding, so it is recomputed at the merge instead of sent. Origins are cut
into a fixed 64 chunks and merged in chunk order however many workers there
are, so `--jobs` cannot change the output.

`build_shapes.py` rewrites `trains.json` in place and needs no re-routing, so
geometry can be re-tuned without paying for the routing again.

### Which build am I looking at

An afternoon went missing to this once: a sampled build quietly replaced the
full one, and the map went from a hundred thousand riders to fifteen hundred
with nothing on screen to say why. Three things keep that from recurring.

- **A sampled run writes its own filenames.** `--sample` sends its output to
  `trains.sample.json` / `stats.sample.json`, so it can never stand in for the
  real build. Load `index.html?sample` to look at one on purpose.
- **Every write is atomic.** `write_json()` serialises beside the target and
  renames over it, so a Ctrl-C in the middle of a 20 MB dump leaves the
  previous good file untouched rather than a truncated one. An interrupted run
  leaves a `.part` file behind; it is safe to delete.
- **Both outputs carry a `build` stamp** — when it was made, the sampling
  factor, how many origins and riders went into it. The page prints it under
  *about the data*, and puts an amber warning across the panel when it is
  drawing a sample, saying by how much the counts are low.
- **Both outputs also carry `day`**, and the page's subtitle reads off it: a
  weekday build says "a typical weekday" where an `--day nye` one names the
  date. `date` is empty on every day type but `nye`, because a re-levelled day
  is a class of days and naming a Wednesday would claim a precision the data
  has not got.

The quickest check by hand is the file size: a full weekday `trains.json` is
~35 MB. Anything much smaller is a sample, or a Sunday.

### One bad stop time, and how far it travelled

Worth reading before touching `kric.py`, because this took three attempts to
get right and each wrong answer looked correct.

The KRIC sources carry a handful of impossible stop times. 김포골드라인 stamps
`5:32:39` on station `005` of **every** weekday train, evidently the first
train's value pasted down the column. 인천1호선 has a few runs whose last three
stops count backwards. Neither is a large amount of data — about 1.2% of stop
times — but each one used to cost far more than itself:

1. **First attempt.** `monotonic` treated any backward step as a midnight
   crossing and added a day. One bad value therefore shifted *every stop after
   it*, and several bad values compounded: a 신림선 train arrived at **234:36**,
   nine days out.
2. **Second attempt.** Only count a backward step as midnight if it drops by
   more than 12 hours. Better, but 김포골드라인's `17:37 → 05:32` is a drop of
   12h04m — just the wrong side of any fixed threshold — so the weekday
   timetable still ran to 48:06.
3. **What actually works.** Ask whether the result would be a *believable hop*.
   Lifting `00:05` over midnight after `23:50` leaves 15 minutes, which is a
   train. Lifting `05:32` after `17:37` leaves 12 hours, which is not. The same
   test catches the forward direction, which matters because on a small-hours
   train the identical bad value reads as a five-hour jump *forwards* — and
   checking only one direction leaves half of them in. `MAX_HOP_S` is 90
   minutes; the longest real hop is 공항철도 직통 running 서울 to 인천공항1터미널
   non-stop in about 45 minutes.

A dropped stop costs one stop. `prev` stays where it was, so the rest of the
run survives, and every line still reaches 100% of its OD stations.

**Two things now stop this reaching the map again.** `validate.py` sweeps
**every service day**, not just the END one we draw — the 김포골드라인 bug lived
an extra day because every check ran on Sunday service only. And `build.py`
drops any run longer than `MAX_RUN_S` or ending past `MAX_END_S`, because the
page's time slider spans the last train it is given: one nine-day train made
the whole day occupy the first tenth of the slider, with the remaining 90%
showing a single dot creeping along. A parse failure upstream should cost one
train, not the entire time axis.

### Two clocks

`build.py` keeps the published timetable and the drawn timetable apart.
RAPTOR routes against `pattern["dep"]` exactly as the operator published it — a
rider is put on the train the schedule says they caught. Everything the page
animates comes from `pattern["vdep"]`, which is that same schedule after two
adjustments:

- **Jitter, ±10s per stop.** The source rounds to twelve distinct second
  values, 60% of them `:00` or `:30`. Drawn raw, trains scheduled to the same
  minute sit exactly on top of each other.
- **Speed smoothing.** Segments claiming more than their line's ceiling (110
  km/h on line 1, which shares Korail track; 90 on the rest) borrow seconds
  from slower neighbours until they are under it, which leaves the trip's
  start and end untouched. Seoul's timetable is already nearly clean against
  this — about 1% of segments before, 0.02% after — so the pass is mostly
  cleaning up after our own jitter. Ported from londonriders, where
  minute-rounded PDFs make it essential.

The page then adds a 15-second dwell at each platform and eases in and out of
it, so trains pull away and brake rather than teleporting between stops.

### Waiting bubbles

Each station carries a `wait` timeline: `[[t, count], ...]`, the crowd standing
on the platform. It is assembled in the routing loop — a rider is added when
they reach the platform (their spawn time, or the previous leg's arrival plus
`TRANSFER_SEC`) and removed when their train pulls out. Boarding times are the
train's exact `vdep`, so the bubble drops on the frame the train's own count
picks up; arrivals are bucketed to 20s so near-simultaneous spawns merge.

Entries are only written when the count moves the bubble by a visible amount.
The page draws `radius = sqrt(count) * 0.5`, so at 2,000 waiting it takes about
nine people to shift the edge by a twentieth of a pixel, and a busy station
changes by one person hundreds of times an hour. Writing every one of those
costs megabytes and draws nothing.

**A gap's riders trickle in, they do not land together.** `spawn_gaps()` gives
every rider in a gap the same spawn time, because one search has to stand for
the lot — but that is a routing convenience, not what happens on the platform.
Drawn literally, the bubble went from empty to full in one step and then
vanished when the train left, which read as a glitch rather than as a platform
filling. `spread_arrivals()` therefore splits a first leg's riders across the
gap they arrived in, up to `WAIT_MAX_SUB` steps. Measured on the rebuilt
sample, a bubble big enough to see now grows by about **9% of its level per
step** (p99 28%), and the page ramps between steps over `WAIT_TRANSITION`
on top of that.

Transfers are left as a single step on purpose: a trainload really does arrive
at once, and drawing that as a ramp would be the lie.

### Spawn times

The OD says how many people left a station in an hour, never when inside it.
**This section is history — it ends with the grid being thrown away.** Skip to
the bold line at the bottom for what `build.py` does now.

`build.py` *used to* spread riders over `DEP_BIN`-wide spawns and run RAPTOR
again at each, so a rider boarded the train that was next *then* rather than
the one that was next at the top of the hour.

Two things about that grid were wrong until 2026-09-04, and both showed on the
map as trains a few minutes apart carrying wildly different loads:

- **`DEP_BIN` was 600 s.** Ten minutes is longer than the headway on every
  trunk line, so most trains had no spawn tick in their window at all and
  boarded nobody.
- **The grid was shared.** Every origin in the network released at `:05`,
  `:15`, `:25` … together, so whichever train pulled out just after each tick
  scooped the lot. `dep_phase()` offset each origin by a stable fraction of a
  bin, so the network no longer breathed in step.

Measured on `--sample 60`, over eight busy stations between 16:00 and 22:00 —
the mean gap in load between one train and the next, over the mean load:

| | 600 s, shared grid | 300 s, phased |
|---|---|---|
| load step between successive trains | 1.24 | **1.13** |
| trains calling with nobody aboard | 23% | **16%** |
| platform arrivals landing in the `:05` minute | 18.5% | **11.1%** (flat) |

The `:05` spike is the signature of the bug and it is gone.

Narrowing the grid closes most of the rest of it. Simulating the quantisation
against the real departure times and the real hourly volumes — riders spawn on
the grid, each takes the first train after their tick — gives the whole curve,
against the continuous-arrival floor no grid can beat:

| `DEP_BIN` | load step | share of the achievable gap closed |
|---|---|---|
| 600 s | 1.96 | 0% |
| 300 s | 1.42 | 54% |
| 150 s | 0.99 | 97% |
| 60 s | 0.97 | 99% |
| continuous | 0.96 | — |

**So the grid was thrown away, and there is no `DEP_BIN` any more.** The floor
is reachable exactly, and for less work than a fine grid. A search from time
`t` is decided entirely by which departure is next at the origin, so *everyone
arriving between two departures catches the same train* — the one that ends
the gap. There is nothing finer to resolve. `spawn_gaps()` therefore puts one
spawn in each gap, carrying that gap's share of the hour, at the middle of the
gap because that is both the average arrival within it and the average platform
wait, which the waiting bubbles read off it.

This is what `../londonriders/` does — stratified random departure times inside
each quarter-hour — made exact: one spawn per gap rather than a sample of the
window, and no bin width or chunk size to tune. Anita asked why Seoul could not
just spread riders across time the way London does. It can; this is that.

Measured on the rebuilt `--sample 60`: load step **1.03** against a simulated
floor of **0.96** for the same data, and the platform-arrival histogram flat to
within half a point across the ten minutes. The residual is model error in the
floor estimate, not grid quantisation. What is left is the OD's own hour-level
lumpiness, which is data, not method.

The numbers to watch if this ever looks wrong again: the load step, the share
of calls with nobody aboard, and the flatness of the platform-arrival
histogram.

### The first hourly bin is open-ended too

`24시이후` is not the only open-ended column. The **first** one is `06시이전` —
*everything* before 06:00, not 05:00–06:00 — and it was being spread across the
whole 05:00 hour as though it were an ordinary one. The subway does not open
until about 05:30, so more than half of that bin was put on platforms before
any train ran, and because the first gap at each origin then stretched from
05:00 to its first departure, all of those riders spawned at its midpoint and
stood there.

The result looked like a finding: **at 05:19 the network held 33,818 people on
platforms against 42,147 at the 08:30 peak** — 80% of peak crowding on 10% of
the traffic, with four trains moving. It is entirely an artefact.

`early_window_start()` now begins that window one headway before each origin's
own first train. The same measurement afterwards: **68 on platforms at 05:19
against 1,036 at 08:30** (sample build), 6.6% — and the dawn curve ramps up
through the hour as service starts instead of spiking at 05:20. The morning
peak is unchanged, so the fix touched only the artefact.

`24시이후` gets the symmetric treatment from `LATE_BIN_HOURS`. **If a third
open-ended bin ever appears, it needs the same thought.**

### What the build actually spends its time on

**Not RAPTOR.** That was asserted here without measurement on 2026-09-04 and it
was wrong — it drove a whole round of optimising the wrong thing. Measured:
`--sample 8` routes **78 origins in 76 s** on fourteen workers, so all 626
scale to roughly **ten minutes** of routing. A `--sample 60` run is 36 s end to
end, nearly all of it setup — loading 600k timetable rows, the jitter and speed
smoothing pass over 200k segments, then building and writing the output.

So the spawn scheme is a small slice of the build, and the cost of making it
exact rather than approximate is not worth avoiding. **The full-build wall time
in "Picking this up" is still unverified** — time it and write the number down.

### Knobs worth knowing

| | in | what it does |
|---|---|---|
| `EXTEND_LAST_HOUR` | `build.py` | reconstructs New Year's Eve late service, and is set from `daytype.py` — on for `--day nye`, off for every other day. The one invented thing in the pipeline |
| `LATE_BIN_HOURS` | `build.py` | how many hours the open-ended `24시이후` bin is spread across. 2 |
| `MAX_ROUNDS` | `build.py` | trips per journey, so 4 means up to 3 transfers |
| `SAG_M` | `build_shapes.py` | how far the track must bow before a waypoint is kept |
| `RENAMES` / `EXCLUDE` | `build_stations.py` | 2026-timetable vs 2023-ridership differences, each with its reason |

### build_stations.py output

`data/stations.json`, 1.0 MB: **626 complexes, 761 platforms**, plus deduped
track geometry for all 22 lines (Line 1's 50-odd route relations all carry the
same track, so ways are kept once per line). Each complex also carries
`name_en`; see "Two languages".

Ridership attaches to the physical **complex**, following nycriders — which
platform a rider uses is a routing decision, not data. Of the 626:

- 203 fully measured hourly
- 43 partly measured — a complex whose line 1–9 platforms are split across the
  서울교통공사 boundary, 고속터미널 (3/7/9) being the type case
- 380 with hours to be inferred

Spot-checked against known positions: 서울역, 강남, 홍대입구, 잠실, 시청, 인천,
수원, 천안, 대화 and 오이도 all land within ~250 m.

### build_od.py output

`data/od_hourly.npz`: **178,227 pairs × 20 hours**. The rider total depends on
the day — 3.40M for `--day nye`, 3.42M for `--day sunday`, **6.34M for
`--day weekday`**. 2.0% of the OD's trips are dropped: one end off the network,
a same-complex round trip that carries no journey, or an unreachable pair.

The re-levelling reports two numbers worth reading. `day/OD-date ratio` is the
network-wide weekday-to-Sunday factor, **×1.85**, which is the size of what
Furness is being asked to do; and `origin totals hit to 0.143%`, which is how
close it got. Only 522 of 626 complexes have a card total on both days — the
other 104 are the operators outside Seoul's fare settlement and float, taking
their scale from the Seoul end of their trips.

The hourly fit converges in 40 rounds and reproduces the measured marginals
exactly. That is not too good to be true: the hourly profiles are rescaled so
each station's daily total equals what we actually carry, which makes the three
constraint families mutually consistent, and IPF on a consistent system has an
exact solution. Pair totals are preserved to 1e-6.

`measured complexes: 239 of 626` is the hourly file's reach, not a regression —
서울교통공사 lines 1–8 within their own boundary. Everything else takes its hours
from its measured partners, which is constraint (c) doing the work.

**The inference does real work.** By volume: 73.0% of trips have both ends
measured, 24.0% one end, and only 3.0% are seed-only. Unmeasured stations move
measurably away from the seed profile, and the ones that move furthest are the
outer suburban stations — 안산, 신창, 성환, 동두천, 오산 — exactly where trips are
long enough that the arrival-hour shift carries information.

Cross-checked against the raw hourly file on `--day nye`: per-station shapes
come through intact, and **잠실 shows 11,995 boardings in the midnight hour
against 1,378 at 23:00**, with arrivals peaking again at 23:00 before midnight.
The countdown crowd arriving and then going home is sitting right there in the
data. The system-wide dip at 23:00 on that build is real, not an artefact —
people are already at whatever they came for.

None of that survives into a weekday build, and it should not: it is the one
night's own signature.

### build.py output

`data/trains.json`, ~35 MB shaped. **445 patterns and 9,306 trips** make up
weekday service, against 268 patterns / 3,899 trips on a Sunday; 9,259 of those
trips end up carrying at least one rider. The file grew from 25 MB when
`spread_arrivals()` went in — the waiting timelines now have real ramps in them
rather than single steps. Riders spawn every 10 minutes inside their hour and RAPTOR runs afresh
per bin, so they board the train that is genuinely next — this is what stops an
hour of 잠실 piling onto one midnight train.

**`00:00:00` is a null marker, not midnight.** The timetable writes it for a
terminus that only arrives or only departs; genuine after-midnight times are
written `24:xx` and up. Read as a real time it puts a zero in the arrival
matrix, which then looks like the cheapest possible way to reach that stop, and
RAPTOR drags riders onto the first train of the day: **15% of all boardings were
landing in the 05:00 hour against 1% in the OD**, and one Line 4 "train" spanned
23.8 hours. Fixed in both `build.py` and `build_od.py`.

**RAPTOR needs one arrival label per round.** A single overwritten parent
pointer looks fine until a later round improves a stop that an earlier leg was
chained through; the backtrace then follows a state that no longer exists and
dies on its step guard. With three rounds that silently lost 8% of riders, and
raising it to four made it *worse*, 58%, which is what exposed it. Labels are
now `tau[k]` per round and the trace walks `k` downwards.

Final run: **2,961,900 riders routed of 3,007,753 (98.5%)** onto **3,984 trains**,
`trains.json` 8.3 MB after shaping.

Two checks the script prints, both worth watching:

- **riders boarding more than an hour off their spawn hour** — 0.25%, and the
  remainder are second legs after a transfer, which is legitimate. This is the
  number that was 15% before the null-time fix, so it is the one to look at
  first if anything ever looks wrong again.
- **unrouted** — 1.25%, of which 36,295 are in the midnight hour. Those are
  riders spawning late in the 00:00–01:00 bin who cannot reach their
  destination even with the reconstructed extra hour of service. Everything
  before 21:00 loses essentially nobody.

Boardings by hour now track the OD to within a few tenths of a percent, and the
system peaks at roughly 80,000 riders aboard through the middle of the
afternoon.

At 00:15 the map shows 16,612 riders on 49 trains, and by far the largest dot
on the network is a Line 2 train at 잠실 — the countdown crowd going home. That
is the shot the whole project is for.

## Known data gaps

Small, but write them down rather than rediscover them:

- **신림선's timetable carries 26 trips that are not trains.** Two rows each —
  `관악산 06:37 → 샛강 06:40`, the whole 7.8 km line in three minutes with none
  of the nine stations between — and 30 more like them across the other
  service patterns. They are summary rows. Left in, they drew a dot rocketing
  the length of the line at 400 km/h, and RAPTOR boarded riders onto them
  *because* they were the fastest thing going: 434 riders were teleported on
  the first full build. The speed smoothing cannot help, because it borrows
  seconds from neighbouring segments and a two-stop run has none. `load_patterns()`
  now drops any trip whose end-to-end straight-line speed is over 1.5x its
  line's ceiling; nothing genuine comes close, and the filter reports what it
  dropped. Found by auditing drawn speed, not by looking at the map — at
  Seoul-wide zoom one dot moving too fast is invisible.
- **자양 (line 7) has no rows in the OD at all** — not a naming mismatch, the
  station is simply absent from the source. Neighbouring 뚝섬유원지 has 6,594
  boardings, so a few thousand trips are missing. Dropped.
- **The 연천 extension** (연천, 청산) opened 2023-12-16, a fortnight before our
  date. 청산 has zero OD rows, 연천 has 545 trips (0.016% of the day), and
  neither is in OSM. Excluded.
- **The 8호선 별내 extension** opened August 2024 and is excluded. Its 구리 and
  별내 sit at complexes that did exist in 2023 on 경의중앙선 and 경춘선, so they
  pass a naive presence test and are named explicitly in `EXCLUDE`.
- **당고개 was renamed 불암산** in 2024; the timetable is 2026 and the ridership
  2023, so it needs an entry in `RENAMES`.
- **The OD files five complexes wholly under a line we do not carry** — 회기
  under 경의중앙선, 신내 under 경춘선, and 진접/오남/별내별가람 under 진접선,
  which is really Line 4's extension. Absorbed, but only where the complex has
  no line 1–9 rows at all, so that genuinely distinct same-name stations
  (양평 on line 5 versus 양평 on 경의중앙선) cannot swallow each other.
- **Five complexes are missing from OSM entirely** — 부천시청 (7), 화전 (경의중앙),
  박촌 and 임학 (인천 1호선), 서구청 (인천 2호선). They are placed *along their
  line's track* between the neighbours we did find, not on the chord between
  them: 인천 1호선 bends east through 계양구, so the chord put 박촌 807 m and
  임학 655 m out into open country, visibly off the drawn line. Following the
  track instead puts them 10 m and 46 m off it. See `fill_gaps()` in
  `build_stations.py`. They are still guesses about *where along* the line the
  station sits, just no longer guesses about whether it is on it.

## Known limitations of the build

- **The one invented thing: `EXTEND_LAST_HOUR` in `build.py`.** It applies to
  `--day nye` and nothing else — on an ordinary day the timetable and the gate
  counts agree about when service stops, so there is no gap to reconstruct and
  repeating an hour of departures would be inventing trains that did not run.
  `daytype.py` decides. On New Year's Eve: no trip in the
  regular Sunday timetable starts at or after 00:00 — only 105 trains are still
  finishing their runs, and the last ends at 00:42. Yet the gate counts record
  roughly 230,000 journeys in the post-midnight bin. That gap is the evidence
  that Seoul ran its customary New Year's Eve extended service, which the 2026
  timetable we have does not contain. So the 23:00 hour's departures are
  repeated an hour later at the same headways, 95 reconstructed trips, to stand
  in for it. Without this, the midnight exodus — the single most distinctive
  thing about this night — is mostly stranded. Set it `False` to see the night
  as the regular timetable would have it. Riders who still find no train are put
  on the last one that ran rather than deleted, so a few appear slightly early.
  Worth revisiting if an archived 2023 timetable turns up.
- ~~Trains move in straight lines between stations.~~ **Fixed** by
  `build_shapes.py`, which is the thing nycriders never got. Rather than trying
  to assemble the OSM ways into one ordered polyline per line — they arrive as
  an unordered soup with branches — it welds the way vertices into a graph and
  runs Dijkstra between the two stations. Branches then look after themselves,
  and express trains skipping stations still get a real path. Segments whose
  track is effectively straight are left alone: on the sample, 1,021 segments
  were shaped and 419 were already straight, none unmatched or rejected.
- ~~Trains shoot past a station and reverse into it.~~ **Fixed 2026-09-04.**
  Snapping each station to its own *nearest* vertex is wrong on a multi-track
  corridor: 경부선 carries four tracks a few metres apart and the welded graph
  joins them only at the crossovers, so 용산 and 노량진 landed on different ones
  and Dijkstra ran to the next crossover and back — 7,962 m for a 2,704 m
  chord, the train visibly overshooting to 신길 and reversing. `build_shapes.py`
  now takes every vertex within `CAND_SLACK_M` of each station as a candidate,
  runs one multi-source Dijkstra, and picks the pair minimising path length plus
  `SNAP_PENALTY` × the two offsets. Underneath that it rejects a path longer
  than `DETOUR_RATIO` × the chord, or one that doubles back past its own end,
  and falls back to the straight line. Worst detour on the network went from
  ×3.26 to ×1.95, and that one — 인천공항 T1↔T2 — is real track.
- **Nothing stops a train filling past capacity.** RAPTOR puts every rider on
  the first train that serves them, so on a night when everyone wants the same
  train, everyone gets it. Checked on the 2026-09-04 full build: the red rim
  fires on **138 of 207,379 drawn stops, 0.1%** — so it is calibrated about
  right and is not the noise `todo.txt` worried it might be. But 79 of those
  138 are in the **midnight hour**, and the tail there is not physical: line 2
  peaks at **4,819 riders against a 1,600 rating**, 3.0x, leaving 시청 at 00:20.
  Even crush load on a ten-car 2호선 train is nearer 2,500. So the midnight
  dots are the model saying "this many people wanted to leave 시청 at once",
  not "this many boarded". Two things feed it: no capacity constraint, and
  `EXTEND_LAST_HOUR` reconstructing the late service at 23:00 headways when
  Seoul very likely ran more trains than that. Fixing it properly means a
  capacity-constrained assignment, which is a different algorithm.
- **16 complexes are only partly measured** — their line 1–9 platforms straddle
  the 서울교통공사 boundary, 고속터미널 (3/7/9) being the type case. They are
  treated as measured, taking their hourly *shape* from the platforms that do
  report. Reasonable, but it is an assumption.
- **The post-midnight bin is open-ended in the source.** The hourly file lumps
  everything after 24:00 into one column, so the *shape* of the post-countdown
  exodus within that window is not measured, only its total. `LATE_BIN_HOURS`
  spreads it evenly across two hours. Compressing it into one hour instead —
  which is what the code did at first — crushed ~90,000 riders onto the handful
  of trains still running and made the dots balloon at exactly the moment the
  map is about. If the midnight dots ever look wrong again, look here.

## Where this could go next

`todo.txt` is Anita's list and takes priority. Two entries on it have answers
already worked out:

**"make trains follow line paths"** — done, by `build_shapes.py`. If it looks
undone, `build.py` has been re-run without it; see "Picking this up".

**"add more lines / korea-wide map in japanriders style"** — researched
2026-09-03. Lines 1–9 are **91.3% of all boardings** in the OD, so the other 18
lines buy 8.7% of boardings. But we drop 16.5% of *trips*, because a trip needs
both ends on the network — a 분당선 → 2호선 rider is lost even though only their
origin is off-map. Adding 분당선, 경의중앙, 공항철도 and 신분당 would recover most
of that; the blocker is schedules, not ridership.

Country-wide is more interesting as a *separate* project than as an extension,
because of an asymmetry:

| | data | OD pairs? |
|---|---|---|
| 수도권 | 642 stations, real OD | yes |
| 부산 | [3057229](https://www.data.go.kr/data/3057229/fileData.do) daily × hourly | no |
| 대구 | [15002503](https://www.data.go.kr/data/15002503/fileData.do) daily × hourly, 2018 on | no |
| 대전 | [15060591](https://www.data.go.kr/data/15060591/fileData.do) | no |
| 광주 | [15060048](https://www.data.go.kr/data/15060048/fileData.do) | no |

All four are anonymous downloads on `data.go.kr` in the same shape as the Seoul
hourly file. **None publish OD.** So a country-wide *rider-flow* map would be
one city measured and four modelled by gravity-seeded IPF — and Korea's metros
are disconnected islands anyway, with no through-riding between them.

A country-wide map in the `../japanriders/` style — station throughput by hour
rather than riders on trains — needs no OD at all and would be **fully measured
across all five systems**. That is the version that would be honest end to end.
Intercity rail could stitch them together visually, but KTX data is
station-level and monthly, so those flows would be coarse.
