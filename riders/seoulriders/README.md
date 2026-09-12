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
| `data/kric_station_monthly.csv` | KRIC station × month, 13 operators, 2023–24 | 0.8 MB |
| `data/incheon_hourly.csv` | 인천교통공사 station × hour × month, 2023–24 | 2 MB |
| `data/incheon_daily.csv` | 인천교통공사 station × day, current 12 months | 1 MB |
| `data/incheon_airport_hourly.csv` | 인천공항 passengers × hour × month, 도착/출발, transit separable | 20 KB |
| `data/osm_stations.json` | 786 rail station nodes, Seoul-area bbox | 0.6 MB |
| `data/osm_routes.json` | 165 route relations, ordered stops + track geometry | 7 MB |

The Korean files are CP949-encoded **except `card_daily_*.csv`, which is UTF-8
with a BOM**, and `incheon_daily.csv`, which is CP949 straight from data.go.kr
while the two files beside it are UTF-8 because `fetch_outside_seoul.py` writes
them itself; the OSM pulls are UTF-8 JSON. `data/` is gitignored repo-wide, so
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
Incheon extension — 103 of our 626 complexes. Those now come from their own
operators instead; see "The off-card operators" below. Do not list them by hand
anywhere — `outside.offcard_labels()` works the set out from the card file at
run time, because getting it wrong is a silent double count.

**Being in this file is not the same as being in the OD**, and conflating the
two cost a round of work. The card file settles the Korail 광역 lines perfectly
well; the *OD* still cannot see their internal journeys. What decides that is
whether a gate belongs to 서울교통공사, which this file also says — its 노선명 is
numbered for them and named for everyone else. `build_od.card_gates` splits on
it. See "What the build does about it now".

### The off-card operators — KRIC and 인천교통공사

`fetch_outside_seoul.py`, all three anonymous. Added 2026-09-04, after the map
drew Incheon at about a sixth of its real size.

**[KRIC 철도통계](https://www.kric.go.kr/jsp/industry/rss/citystapassList.jsp)**
— 철도운영현황 > 도시철도여객수송 > 역별 승강차실적(월). Per station, 승차 and
하차 separately, monthly, 2022 onwards, for every 도시철도 operator in the
country. **This is `www.kric.go.kr`, not the `data.kric.go.kr` 레일포털 that
`fetch_schedules.py` uses** — different site, different data, same lack of a
login.

Thirteen operators, of which eight lines are ones the card file misses:
인천1/2, 7호선(인천), 신분당 (three operators: 네오트랜스, 경기철도, 새서울철도),
김포골드, 의정부, 진접. 용인경량전철 is there too but this map carries no
에버라인선 complex, so it is skipped.

- The HTML view pages at 15 rows. Take the Excel export — the same form POST
  plus `mode=excel` — which returns the whole table.
- KRIC splits a transfer complex per line with a bracketed suffix (`고속터미널`,
  `고속터미널(7)`, `고속터미널(9)`). Sum them.
- **It agrees with the card file.** Joining KRIC's 서울교통공사 November 2023
  against `card_daily_202311.csv`: median ratio **1.0042** over 187 stations,
  147 of them inside 2%. Same measurement, so no scaling between the two.
- One known hole: KRIC's **2024-12 인천1호선** is short by 1.34M boardings
  against 인천교통공사's own figure for the same month. 2023 is clean. Cross-check
  before trusting a KRIC month.

**인천교통공사 「역 시간대별 통행량」** — on their
[사전정보 공표목록](https://www.ictr.or.kr/main/bbs/bbsMsgList.do?cate1=918&bcd=opendata)
board, *not* the open-data portal. 운수기획팀 has posted one .xlsx a month since
2015-10, unbroken: three sheets (인천1/2/7), 68 stations, **station × hour ×
승차/하차**. The same shape as OA-12921, which is what makes it useful — Incheon
stations move from "fitted through their Seoul partners" to measured on both
ends. November 2023 totals 12,663,474 boardings, KRIC's figure for that month
to the person.

`msg_seq` is not derivable, so the fetcher scans the board for it.

**[인천교통공사_역별일별 이용인원현황](https://www.data.go.kr/data/15004329/fileData.do)**
(15004329) — daily per station, 71 stations. Only ever the current 12 months;
data.go.kr rotates it annually with no archive, so it cannot reach 2023. It is
here for day-type factors, not levels: Incheon's own weekday 486,490 /
Saturday 368,932 / Sunday 271,472 boardings, Sunday/weekday **0.558**.

Its station names are the odd ones out — `서해구청` for 서구청 and
`7호선 부평구청` for the 부평구청 that is not on 인천1호선. `outside.ALIAS`.

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

**Do not run fewer than the default 4 rounds.** A `--rounds 2` weekday pass on
2026-09-04 went 65.6% → 33.0% → 59.2% on the Line 9 express share against a
published 40.2%: still mid-swing, and 19 points high in the round that wrote
`trains.json`. The per-round share is *meant* to oscillate — it is the raw
assignment, and only the penalty table is averaged, at `1/(it+1)`. But the
final round both writes the output and is a raw assignment, so it ships
whatever overshoot is left. Rounds are the only thing damping that.

How much overshoot is left is worth measuring rather than assuming, and it is
free: the averaged loads are in `data/crowding_loads_avg.npz` and the final raw
round's in `data/loads.npz`. On the 2026-09-05 build the *shares* were close —
52.3% averaged against 55.1% shipped — but the *peaks* were not: the average
had 9호선's 급행 topping out at 153% of 정원 and the shipped round at 265%.
An aggregate that has settled does not mean the individual trains have. That
gap is a symptom of the penalty having had no ceiling to push against, and is
the second thing the new shape is meant to fix.

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

### The penalty function

Crowding `c` is riders over 정원.

```
x = max(c - C_SEATS, 0) / (C_CRUSH - C_SEATS)
factor(c) = 1 + ALPHA * x * (1 + x) / 2
```

`C_SEATS = 0.34`, `C_CRUSH = 1.45`, and a minute at `C_CRUSH` is worth
`1 + ALPHA` perceived minutes. Two properties, both of which the first version
of this got wrong, and both of which the published 9호선 data forces:

**It starts at the seats, not at 정원.** 정원 counts every standing place at
crush — about 160 to a car against roughly 54 seats — so a train at a third of
it is already full of people standing, and that is where a rider starts to
mind. The original threshold was `C_FREE = 1.00`, which meant the penalty was
identically zero on 98% of the network and the express was free everywhere it
was not already at crush.

**It has no ceiling.** The original clipped `x` at 1, so a train at 265% of
정원 was penalised exactly as much as one at 145% — no marginal deterrent, and
no equilibrium for the loop to find. 1호선's 급행 duly ran away with 4,077
riders on a single train at 255% of 정원 while the 일반 behind it sat at a
fifth of 정원.

The shape is linear where people are merely standing and quadratic once it is
a crush. That convexity is not a guess either: the published express/local
load ratio *compresses* as the line fills — 1.96× off-peak, 1.44× at the
peaks — which is what a convex cost does and a linear one does not.

At `C_CRUSH` the new and old forms agree exactly, so `ALPHA` keeps its meaning
and its old fitted value is still the starting point. What changed is
everywhere else:

| c | old | new |
|---|---|---|
| 0.21 (일반, midday) | 1.00× | 1.00× |
| 0.39 (급행, midday) | 1.00× | 1.07× |
| 0.64 (일반, peak) | 1.00× | 1.51× |
| 0.96 (급행, peak) | 1.00× | 2.31× |
| 1.45 | 4.00× | 4.00× |
| 2.65 | 4.00× | 10.62× |

### What the published split actually says

The old shape was fitted on the assumption that the express/local choice is
made at crush. It is not. Checked hour by hour against 서울시메트로9호선's own
figures — our share against theirs, on the cells `validate.py` matches:

```
hour   ours    published        hour   ours    published
  8   41.0%      38.0%           12   62.1%      42.6%
 18   47.4%      36.6%           20   66.2%      48.1%
 all  55.1%      40.2%
```

**The over-assignment is worst where there is no crowding at all.** At the two
peaks we are 3 and 11 points high; in the quiet middle of the day we are 18–20
points high. And the real 급행 is 1.4×–2.25× fuller than the real 일반 at
*every hour* — at midday it sits at 39% of 정원 against the 일반's 21%.

So riders are trading crowding against time all day long, at load factors a
threshold-at-정원 model cannot see. That is why the penalty had to move down
to the seat line: not to punish the crush harder, but to exist at all in the
range where most of the day happens.

Rider-weighted over the whole day, the published 급행 runs at 88% of 정원 and
the 일반 at 58%, a ratio of **1.52×**. Ours ran at 85% and 39% — **2.21×**.
The express level is about right and the local is starved, which is the same
finding the uncrowded build gave, just smaller.

`crowding.py` now prints that ratio next to the share every round, because the
share alone cannot tell a model that splits riders correctly from one that
splits them for the wrong reason.

### What the new shape actually did

Full pass, 2026-09-05, `--rounds 4 --alpha 3`. The per-round 9호선 express
share, each round a raw assignment against the previous round's averaged
penalty:

```
round     0      1      2      3      4 (ships)
        66.3%   7.3%  58.9%  40.2%  43.2%
ratio   5.77x  0.17x  3.97x  1.46x  1.60x
```

It rings and then settles, which is the loop working. A `--sample 6` pass run
first as a rehearsal tracked it closely — 63.4 → 10.9 → 55.7 → 39.0 — so
sampling is a fair way to fit the knob.

Against the published figures, on the cells `validate.py` matches:

| | before | after | published |
|---|---|---|---|
| 9호선 express share | 55.1% | **43.2%** | 41.3% |
| 급행/일반 load ratio | 2.36× | **1.60×** | 1.52× |
| riders on trains ≥ 정원 | 4.79% | **1.52%** | — |

**Hour by hour is where it shows.** The old build was 18–20 points high all
through the middle of the day, which was the whole tell that the threshold was
in the wrong place. It is now within ±5 points at every hour from 05:00 to
22:00 — midday 11:00–15:00 comes in at +1.1 to +2.6. Only 23:00 (+9.2) and
24:00 (+8.0) are still notably high, on small volumes.

`validate.py --congestion` says the same thing from the other side:

```
                  uncrowded          now
9급행       90.6% / 1.15x / 0.98   0.80x / 0.94
9일반       19.4% / 0.47x / 0.95   0.80x / 0.89
```

**Both services now sit at the same 0.80× of published**, where before the
express was 1.15× and the local 0.47×. That is the shape of a *correct split
with a shared level offset* rather than a mis-split, and the level offset is
the known 80% OD coverage, not a routing error.

**Two things it cost, both worth knowing.**

The per-line shape correlations all fell a little — 1호선 0.87→0.86, 3호선
0.97→0.93, 4호선 0.94→**0.85**, 7호선 0.96→0.88 — and the 상선/하선 direction
check weakened from r=0.801 to r=0.749. The hourly profile shows why: we are
now slightly under at the peaks (08:00 0.89×, 18:00 0.86×) and over in the
hour *after* each of them (09:00 1.23×, 10:00 1.29×, 19:00 1.14×). Riders
avoiding a crush take slower paths, so they are still on board an hour later.
Real peak-spreading works by people leaving earlier or later, which this model
cannot do — the OD fixes how many depart in each hour. So some of that smear
is behaviour we have no way to represent, and ALPHA is buying the split partly
with it. **ALPHA 2 is the obvious thing to try next**: it would keep most of
the split correction with less smear, and 4호선 at 0.85 is sitting exactly on
the line the check calls a real problem.

#### ALPHA 3 against ALPHA 2, measured

Run 2026-09-06, both `--sample 6 --rounds 3`, so they are directly comparable
with each other (though *not* with a full build — see the caveat below). The
question was whether ALPHA drives the platform crowds at all, or whether those
are the structural peak-spreading limit and no ALPHA value would touch them.

| | ALPHA 3 | ALPHA 2 |
|---|---|---|
| 9호선 express share | 39.0% (−1.2 pts) | 43.4% (+3.2 pts) |
| 9호선 급행/일반 load ratio | 1.24× | **1.42×** (published 1.52×) |
| 사당 implied platform wait | 13.9 min | **10.5 min** |
| 사당 peak crowd | 4,571 | **3,295 (−28%)** |
| network median wait | 4.1 min | 3.6 min |
| p90 wait | 9.9 min | **7.8 min (−21%)** |
| worst station | 28.0 min | 21.4 min |

**ALPHA does drive the crowds.** That was the open question and it is now
answered: the bubbles are not purely the structural artefact. Sadang's crowd
falls by more than a quarter.

**The split cost is not the clean argument against it that it looks.** The
share gets worse (39.0 → 43.4) but the **load ratio gets better** (1.24 →
1.42 against a published 1.52), and the ratio is the measure that says whether
riders split for the right *reason* rather than merely in the right proportion.
The two calibration numbers disagree about which ALPHA is better.

**The caveat that matters most.** Neither run is converged — both oscillate
63.4 → 11/17 → 54/56 → 39/43, and sampled ALPHA 3 (39.0%) differs from the full
ALPHA 3 build (43.2%) by 4.2 points, which is the *same size* as the gap
between the two alphas. **So this experiment cannot resolve the express-share
difference at all.** It can resolve the 28% change in the crowds, which is far
larger than that noise. Read the table as "ALPHA 2 clearly shrinks the bubbles,
at a split cost too small for a sampled run to measure".

#### What the full ALPHA 2 build actually did

Run 2026-09-06 with `--rounds 4`, and carrying the airport fix as well, so the
two are not perfectly separable. Everything network-wide improved and both
9호선 split numbers got worse:

| | ALPHA 3 | ALPHA 2 |
|---|---|---|
| 혼잡도 correlation, all ten series | 0.84–0.94 | **0.86–0.97, every one up** |
| `validate.py --congestion` flags | 1 (4호선 at 0.84) | **0** |
| platform wait, median / p90 / worst | 3.0 / 6.3 / 16.7 min | **2.7 / 5.3 / 8.7 min** |
| 사당 implied wait | 9.3 min | **7.1 min** |
| worst single train | 230% | **179%**, none over 200% |
| 9호선 express share | 43.2% (+3.0) | 45.4% (+5.2) |
| 9호선 load ratio | 1.60× | 1.80× (published 1.52×) |

**The sampled A/B mispredicted the load ratio, and the reason is worth
keeping.** It said 1.24 → 1.42, moving *towards* 1.52; the full build went
1.60 → 1.80, moving away. Sampled ratios sit systematically *below* full ones
at the same ALPHA — 1.24 against 1.60 — so the sampled and full runs are on
opposite sides of the target, and "towards" in one is "away" in the other.
**A sampled fit can rank ALPHA on the express share and on the crowds. It
cannot be trusted on the load ratio at all.**

Whether the trade is worth it is a judgement, and it was made deliberately: ten
network-wide series improved and the shape check went clean for the first time,
against two numbers on the one line that has a published split.

The level ratios on every other line moved *toward* 1.00 as a side effect
(3호선 0.82→0.91, 7호선 0.80→0.94, 6호선 0.88→1.05), because riders on longer
paths are counted past more stations. Whether that is the build getting more
right or two errors cancelling is genuinely unclear, and it should not be read
as confirmation.

### The worst single train is a different bug

The fullest train in the shipped build carries **6,550 aboard, 409% of 정원**
— and it is a **1호선 all-stops** service, not an express, on a pattern with
only 13 trips in the whole day. Second and third are 1호선 급행 at 295% and
270%. The uncrowded build's worst was 6,923, so the crowding pass barely
touched it.

**There is no capacity check anywhere in the routing.** `capacity` appears in
`build.py` exactly once, at line 1595, copying it into `line_meta` for the
output file — the router never reads it. Boarding is unconditional. Crowding
only ever makes a train *look* slower; nothing has ever refused anyone a place.
That is deliberate and the reason is in "Why a capacity check does not work
here" above, but it is worth saying plainly, because "409% of 정원" invites the
question "why did they get on?" and the answer is "nothing stopped them".

The specific trip, dug out of `loads.npz`:

```
1호선 all-stops, departs 07:51, 53 stops, pattern runs 13 trips all day
   06:38 07:15 07:51 09:02 09:41 10:16 10:36 12:00 13:16 13:37 15:13 18:08 19:46
peak load at stop 31 of 53:            6,550 aboard   409% of 정원
perceived penalty it carried:          +71.5 min over the run
load the penalty was BUILT from:       2,553 aboard   160% of 정원
```

Two things there, and the second is the real one:

- **A sparse pattern concentrates demand.** Thirteen trips a day, gaps of half
  an hour to an hour and a half. Even +71 minutes of perceived penalty can lose
  to waiting an hour for the next one, so the model is not obviously wrong to
  cram people on — real passengers would too. What real passengers *cannot* do
  is all fit.
- **The deterrent is a round out of date and smoothed flat.** The penalty table
  is fixed for the whole round and built from the *previous* round's
  MSA-averaged loads. So the first rider aboard and the four-thousandth see
  exactly the same penalty, and this train was priced as if it would end up at
  160% when it ended up at 409%. Averaging is what makes the loop converge; it
  is also what hides every peak from the thing meant to deter it.

More ALPHA will not fix that — it would just move the whole network. What is
missing is **fail-to-board**: the standard transit-assignment treatment where a
full train leaves people behind and they take the next one. It could go in
without giving up the parallel search, as a separate time-ordered loading pass
after routing — riders already have their paths, so the pass is O(riders) with
no searching in it, and the overflow shifts to the next departure serving the
same stop pair. Sequential, but cheap.

The tail is small either way — per individual trip the median train is at 27%
of 정원 (published network median 27.9%), p90 59%, p99 105%, and 0.36% of
riders are aboard something over 200%. But it is the number a tooltip shows
when you hover the wrong train.

### The crush cap — what was actually done about it

Fail-to-board inside the search is still the right answer and is still not
built. What went in instead is a **cap applied after the routing, in the output
path only**: `cap_trains()` in `build.py`, run between `dump_loads()` and
`write_output()`. It walks each pattern forward in time and refuses boardings
that would put a train over `--cap` × 정원, default **1.5**, holding those
riders on the platform for a later trip of the same pattern.

**It is a fudge and it is the only one in the output path.** Read this before
quoting any single train's load off the map. What it may and may not touch:

- `data/loads.npz` is written **before** it runs, so `crowding.py` still sees
  the true, uncapped demand. Feeding the loop capped loads would hide the
  crowding from the one mechanism meant to deter it.
- `wait_deltas` is **not** patched. A rider deferred to the next trip of a
  sparse pattern would otherwise be drawn standing on the platform for an hour
  and a half, which is a worse artefact than the fat train was. The platform
  empties when the train they originally caught pulls out.
- Second legs are not re-planned. Someone deferred on leg 1 still boards their
  leg 2 on the old trip. At 0.24% of person-segments that is not visible, but
  it is a real inconsistency and not a rounding error.
- `--no-cap` turns it off and draws the raw model output.

**How much it has to move.** Measured on the shipped uncapped build, at 1.5×:
**413 cells of 241,952 (0.17%)**, on **86 trips of 9,306**, across 43 patterns
of 445, carrying **226,818 person-segments of excess out of 94M (0.24%)**. Line
1 is 58% of it, lines 7 and 4 most of the rest. Sparse patterns dominate:
patterns with under 20 trips run 0.66% excess, patterns over 100 trips 0.04%.

**Why 1.5.** Nothing 서울교통공사 publishes exceeds 144.6% (2호선 사당 외선,
08:30), and that is a half-hour average over trains, so a single train may sit
a little above it. 1.45 and 1.6 barely change the blast radius — 497 and 307
cells against 413.

**Where the riders go, and the ugly part.** The deferral is deliberately *not*
time-limited. The 409% train is on a 13-trip 1호선 all-stops pattern whose next
trip is 91 minutes later, so capping it means moving a rush-hour crowd onto a
mid-morning train. Limiting the deferral to a plausible wait would simply leave
that train at 409%, which is the thing being fixed. The saving grace is that
the queue drains: that pattern's next trip runs at 16% of 정원, so the overflow
lands somewhere nearly empty rather than cascading down the day.

**The pass has to run more than once, and the reason is worth not
rediscovering.** The first version deferred overflow forward and, when a
pattern was over the cap on *every* trip, the overflow found no room anywhere,
came back, and landed on top of the riders its own train had meanwhile taken
from the queue — turning a 312% train into a **462%** one. A capping pass that
makes the worst train worse is the one outcome that cannot be shipped.

So the loading pass runs up to `CAP_PASSES` (8) times per pattern. Riders it
could not place are pinned to the train the routing put them on, where they
board ahead of everyone and cannot be displaced, and the pass is re-run for the
rest. The early passes pin only the riders who were actually homeless; the
later ones pin the whole cell, which is blunter but cannot leave a leftover
behind, so the set of unpinned cells shrinks every pass and the loop has to
settle. A saturated pattern converges on doing nothing at all, which is the
honest answer for it.

If a pattern still will not settle, it is left **uncapped** rather than having
the strays added back — adding them back is precisely the unsafe thing above.
Uncapped but never worse.

Both of those were found by a deliberate stress run rather than by reasoning:

```
python build.py --sample 20 --no-shapes --cap 0.08
```

An absurd cap puts 53 patterns and 2,415 cells over it on real pattern
geometry, which is the only cheap way to exercise the paths a 1.5× cap touches
on two patterns in an hour-long build. It is worth re-running after any change
to `cap_trains()`. The first time it printed

```
   worst single train: 29% of 정원 -> 31%
   WARNING: the worst train got worse, 29% -> 31%.
```

which is the check earning its keep — `cap_trains()` prints that WARNING
whenever the worst train comes out fuller than it went in, and it should never
appear. After the fix the same run reads

```
   53 patterns had a train over the cap
   cells over the cap: 2415 -> 425 of 241,952
   worst single train: 29% of 정원 -> 27%
   riders held for a later train: 60,810 (1.749% of person-segments)
   how much later they board: median 29 min, p90 233 min, max 1016 min
   4 patterns would not settle in 8 passes and were left uncapped
```

**What the first capped full build actually did** (2026-09-06, 625 origins,
8.38M riders routed), read back off `trains.json`:

| | uncapped | capped |
|---|---|---|
| fullest single train | 6,550 aboard, **409%** of 정원 | 3,677 aboard, **230%** |
| trains over 150% | — | **7 of 9,298** (0.075%) |
| trains over 200% | — | 2 |
| p99.9 of drawn cells | 169% | **150.0%** — the cap binding |
| p50 / p90 of drawn cells | 27.0% / 58.6% | 26.5% / 58.3% |

The 409% train — the 13-trip 1호선 all-stops pattern — is gone from the list
entirely, capped to 150% with its overflow on the 16%-full trip behind it. What
survives above the cap is exactly the case the pass is honest about: **1호선
급행 patterns that are over the cap on every trip they run**, plus one that runs
a *single* trip in the whole day and so has nothing to defer to at all. Those
are left as the routing produced them, which is the right answer for them.

**The deferral leaves no visible artefact.** The worry was that moving a
rush-hour crowd 91 minutes later would show up as an implausibly full
mid-morning train. Person-segments by hour, uncapped against capped:

```
   08:00   9,931,401 -> 9,902,263   -0.3%
   09:00   7,598,313 -> 7,528,852   -0.9%
   10:00   4,778,101 -> 4,801,415   +0.5%
   11:00   3,561,087 -> 3,580,826   +0.6%
```

The peak sheds and the late morning gains, which is the smear doing exactly
what it is meant to, at under 1% — well under anything the eye can pick out of
the animation. No other hour moves by more than 0.3%.

**What the loading pass never needs to know is where anybody was going.**
Alightings are recomputed from a *rate* — what fraction of the people aboard
get off at each stop, taken from that trip's own nominal profile, with the last
stop forced to 1.0 so every train still empties at its terminus. That is what
buys the whole thing without a per-rider leg table. Riders deferred onto a
later trip alight in proportion to *that* trip's destination mix, which is an
approximation, but only ever of where the 0.24% get off.

### Re-fitting ALPHA, cheaply

A full pass is five builds and the better part of an hour, which is a miserable
inner loop for one number. Use `--sample`: it routes one origin in N and scales
the loads back up by N, so every load factor and every penalty is realistic
even though a sixth of the riders are being moved. The express share is a
*ratio* over the same network, so it survives sampling far better than any of
the levels do.

```
python crowding.py --sample 6 --rounds 3 --alpha 3
python crowding.py --sample 6 --rounds 3 --alpha 6
...
python crowding.py --rounds 4 --alpha <winner>     # confirm on the full build
python validate.py --congestion
```

Read the two 9호선 lines each round: the share walking towards 40.2% and the
급행/일반 load ratio towards 1.52×. A sampled pass leaves faked penalties in
`data/crowding.npz`, so `--reset` before any ordinary build.

**The seat-line threshold changes the whole network, not just the express.**
The old penalty touched 1.9% of pattern-stop cells; the new one touches
**33%**. That is the point — it is where the day happens — but it means the
perceived-time distortion now applies to ordinary crowded rides on lines with
no express at all, so it can move riders between *lines* as well as between
services. The per-ride cost is small (median +25s, p90 +2 min at ALPHA 3) but
the network-wide congestion correlations in `validate.py --congestion` are the
thing to read after a re-fit, not just the 9호선 pair. If lines 1–8 get worse
while 9호선 gets better, ALPHA is too high.

Ignore the "worst whole trip" figure the run prints when judging that: it is
the cumulative penalty over an entire end-to-end pattern, which only a rider
going all the way from 소요산 to 신창 would pay. The penalty enters as a
difference between two cumulative values, so what a real journey sees is the
median and the p90.

The other number to watch is `build.py`'s own `unrouted` and
`fell back to the last scheduled train` line. Inflated labels are compared
against *real* departure times, so a rider who has just ridden a crowded train
can miss a connection they would really make; with the threshold at the seat
line that now applies to a third of rides rather than a fiftieth. If either
figure climbs against the uncrowded baseline, ALPHA is buying its express
split by breaking transfers, and the trade is not worth it.

### Two things that bit, both worth not rediscovering

**`crowding.py` runs the rounds in-process, and that is load-bearing on
Windows.** It used to shell out to `python build.py` per round. Seventeen
minutes into round 0 of the first real run, the multiprocessing pool tried to
replace a worker and died with `PermissionError: [WinError 5]` out of
`DuplicateHandle` in `spawn_main`, taking the whole run with it. Nothing to do
with memory — the machine had 40 GB free. It is the process depth: shell →
crowding.py → build.py → workers is one level deeper than a plain
`python build.py`, which has never failed. `build.main()` therefore takes an
optional `argv` list so `crowding.py` can call it directly, and the pool sits
where it always has.

**The averaged loads are written every round**, not at the end. A round is a
quarter of an hour; losing four of them to a crash on the fifth is a bad trade
for one `np.savez_compressed`.

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

The diagonal-lines button in the top-right swaps the animation for the whole
day at once. The animation answers *where is everyone right now*; this answers
*how much moves through here*, which is not a question you can get at by
watching dots go past.

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
- **Every segment is drawn solid.** There used to be a second, dashed layer for
  the lines whose journey pattern is modelled, with its own legend key. It is
  gone: the modelled share is a *number*, it varies from 2.5% to 88% by line,
  and both tooltips already print it — "62% of journeys modelled". A dash could
  only say yes or no, so it drew a hard boundary through a continuum and then
  needed a legend key and a paragraph of panel text to explain that it was not
  saying "less busy". The tooltip says the thing the dash was standing in for,
  and says it precisely. The panel note now points at the tooltip instead.

### The drill panel

Ported from `../nycriders/`, which is the house style for this. Clicking a
segment or a station opens a panel along the bottom: one row per line, one bar
per hour, a running total per row and for the panel. The map says *how much*;
this says *when*, and *which service*.

- **A segment** gives every train that runs between those two stations, in both
  directions — the `>` / `<` arrow says which, with the western end on the left.
  Two metrics: **riders**, which is riders carried past that point, and
  **crowdedness**, which is riders per train against the line's 정원. In
  crowdedness the y-axis is pinned to at least 100% and a dashed line marks it,
  so a chart that looks half-full *is* half-full rather than being normalised
  back to the top of its own row. Row totals stay rider counts in both modes: a
  crowding percentage does not add up into a meaningful figure for a day.
- **The express is its own row**, tagged `급행` / `exp`. `stats.json` carries
  `hx`/`nx` as the express *subset* of `h`/`n`, so the all-stops row is the
  difference. This is the same reason 서울교통공사 publish 일반 and 급행
  congestion separately on 9호선: an express is far fuller than the local it
  overtakes and one blended average describes neither. See "Putting the
  express back on the track" below for why the front end has to rebuild that
  relationship rather than just read it.
- **A station** gives boardings per line per hour, with the same
  all / starting / changing split the day settings carry. It is one setting,
  not two — picking `changing` in the panel resizes the circles on the map, and
  vice versa. The bar scale is locked to *all* boardings so switching only ever
  shrinks a row, never grows it.
- The rows share one scale, so a line that carries a twentieth of its
  neighbour's traffic draws as a sliver. That is the point; per-row scales would
  make every row look equally busy.
- The clicked segments get a white halo and a wash of white on top, from a
  `static-hl` source drawn either side of the segment layers.

Which segments count as "the same place": platform codes are not shared between
operators, so the pair is matched on **station names and then on position**.
Seoul has two 양평 and two 신촌, nowhere near each other, and a names-only match
would have drawn them as one place.

### Putting the express back on the track

`write_stats` keys a segment by the **pattern's consecutive stops**, which is
the right key for counting and the wrong one for drawing. Where an express
happens to stop at both ends of a local link, the two share a key and `hx` is
the express subset of `h` — which is the case the drill panel was written
against. Where the express *skips* a station, its key is a leap the local
never makes, so it becomes a segment of its own: its own stripe on the map,
its own geometry, and no way to see the local running underneath it.

Counted on the 2026-09-05 build: **41 express-only segments** — 9호선 28,
1호선 11, 공항철도 2 — against 147 shared ones, of which 143 are 1호선. That
is the whole of the inconsistency:

- **1호선** mostly shared a key, so it mostly grouped — but not on the stretches
  the 급행 skips, which is why one direction of a pair could group and the other
  not.
- **9호선** shared only 2 keys out of 30. Its 급행 and 일반 are colinear, so the
  express drew as a thicker stripe directly over the local and a click always
  landed on the express. The local was there and invisible.
- **공항철도**'s 직통 runs 서울역 → 인천공항1터미널 in one hop, so it was a single
  50 km segment. `build_shapes.py` gave that leap 12 waypoints, which drew as a
  near-straight line nowhere near the track — the "duplicated" Airport Railroad.

`foldExpressIntoLocal()` fixes all three at load. It walks each express-only
link over the graph of everything that is *not* express-only, and adds that
link's riders and trains to every local link it passes — into `h`/`n` and into
`hx`/`nx` both, which is exactly the subset relationship the drill panel reads.
The skipping record is then dropped: its riders are not lost, they have been
spread along the way they actually travel. All 41 fold; the worst path detour
is 1.19× the straight line, so nothing has gone down a wrong branch. Afterwards
254 links carry an express instead of 147 (9호선 goes 2 → 72, 공항철도 2 → 26),
and every one of them opens as local + express in both directions.

Two consequences worth knowing:

- **The map gets busier, correctly.** A link an express passes now counts those
  riders as carried past that point, which it did not before. The segment width
  scale grows to match, so everything is slightly thinner relative to the new
  maximum.
- **This is a front-end fix, not a build fix.** It is a question about how the
  map reads rather than about what the numbers are, and it costs one pass over
  1,534 segments. Doing it in `write_stats` instead would be defensible, and
  would want the sub-links stamped with a parent key the way nycriders does it.

### The backdrop is drawn from the trains, not from OSM

Every line drew faintly under the animation, out of `DATA.lines[].ways` — the
route relations' own way members, deduplicated per line in `build_stations.py`.
OSM maps a four-track mainline **track by track**, so 1호선 arrives as **940
ways** against 3호선's 62, with 20 of them within 250 m of 영등포. The corridor
drew as a bundle of five parallel lines rather than as a line.

`backdropFeatures()` builds it from the train timelines instead. A timeline row
with four or more entries is a station stop and a three-entry row is a track
waypoint, so the rows between two stops *are* the link's polyline — the one
`build_shapes.py` already solved for the animation. Key each link by its two
station coordinates, unordered, one per line: **760 polylines, 18k
coordinates**, against several thousand ways before. They go through the same
spline as the day view's segments, with tangents carried across stations, so
the backdrop bends the way the segments do.

Two rules in there:

- **An all-stops run's path always wins**, and a link nothing but an express
  uses is dropped. Otherwise 공항철도's 직통 contributes its 50 km straight line
  to the backdrop too.
- **An unshaped build falls back to the OSM ways**, bundles and all: without
  waypoints the timelines are chords and the backdrop would be a chord diagram.

What this gives up: coverage is now every link a rider-carrying train
traverses, rather than every way OSM has for the line. `build.py` drops runs
that carry nobody at all and prints any line with no such run as
`NO TRAINS WITH RIDERS` — there are none, so in practice the two agree. It also
means `DATA.lines` is now only read on the unshaped path, and could come out of
`trains.json` on a future build.

### Smoothing the drawn track

`link_shapes.json` is built for the animation, where a train reads a position
off the polyline and `build_shapes.py`'s 25 m sag tolerance is invisible — its
median link is **three points**. Drawn ten pixels wide it is not invisible at
all: a bend between two stations arrives as a corner, and MapLibre's
`line-join: round` only fillets it by half the line's width.

So the day view runs the same points through a **centripetal Catmull-Rom
spline** (`splineThrough()`, shared with the backdrop above), which passes
through every original point and
rounds off everything between them. The end tangents come from the neighbouring
segment on the same line, so a line that bends *at* a station comes out smooth
across that join too; a terminus, with nothing to borrow from, mirrors its own
first span. Centripetal rather than uniform because uniform overshoots and can
cusp on a sharp reversal.

Checked before trusting it: mean maximum deviation from the source polyline is
**17 m**, worst relative case 4.9% of a segment's length. That is inside the
25 m the source geometry already carries, so the curve is not inventing a route
the simplification had not already smudged. The two 400 m outliers are both the
공항철도 직통 run, which is 50 km drawn with 12 points and was a sketch before
any of this. Whole network: ~35,000 coordinates, worked out once and cached.

### The control bar

Synced to `../nycriders/`, so the two maps read as one family: a cluster of
square icon buttons (settings, pause, 1×, 2×), then the slider, then the clock,
then the mode toggle on the right. The day view swaps the speed cluster for its
own — settings, pause, 1× — which steps an hour a second round the service day.
Each mode has its own settings drawer: speed / dot size / station and line
toggles for the animation, size and the boardings split for the day.

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

### What the build does about it now

**Fixed 2026-09-04, and `--coverage` still describes the source, not the
build.** That table reads the raw OD and should keep reading it; the numbers in
it are still true of the file. What changed is that the build no longer takes
the file's word for these lines.

Three things had to happen, and only the last is modelled:

1. **Volumes.** Every off-card complex now has a measured gate total from KRIC,
   and it enters the same per-station ratio as everything else. A complex can
   have both: 부평 counts 864,819 card boardings a month through 경인선's gates
   and a further 190,941 through 인천1호선's, which are two arrays of gates in
   one building, and those add.
2. **Hours.** 인천's 65 complexes get their own measured hourly profile instead
   of inheriting Seoul's through their partners.
3. **The pairs that are not in the file at all.** Scaling cannot invent them —
   Furness multiplies, and a cell that is zero stays zero. Without a seed,
   giving 인천1호선 its true total would pile all of it onto the handful of
   Seoul-bound pairs that do exist. `seed_outside` builds the block instead:
   each outside station's *missing* boardings spread over the other outside
   stations by a deterrence curve fitted to the measured OD, then balanced
   against the missing alightings. 147,840 pairs, pruned to the 126,803 that
   carry 99.5% of the volume; 1,140,939 trips at the OD date's level.

**Only step 3 is invented, and less of it survives than you would think.** The
seed sets trip length; Incheon's measured hourly constraints then reshape
direction. The 08:00 hour comes out running 부평구청 / 작전 / 계양 / 간석오거리 /
임학 → 갈산, 예술회관, 인천시청 and the 송도 cluster, which is Incheon's real
geography and is nowhere in the seed. Where the marginals and the hours are
both measured, the deterrence function is only breaking ties.

The test the section above sets — within-line share, 36–55% for a real metro
line — is met. 인천1호선 went 1.6% → **41.9%**, 인천2호선 0.3% → **32.5%**,
7호선(인천) 1.3% → **20.2%**. The last two sit below Seoul's band for a real
reason: both are feeders, one into 인천1호선 and one into the rest of line 7, so
most of their trips genuinely leave them. 신분당선(연장2) at 0.4% and 진접선 at
0.3% are three-station segments where a within-segment trip barely exists.

### Two things this got wrong first, both worth not repeating

**It is not "off-card", it is "not a 서울교통공사 gate".** The first version
seeded only the lines with no card row — 인천, 신분당, 김포골드, 의정부, 진접 —
on the reasoning that the card file was the boundary. It never was. The Korail
광역 lines *are* carded and had exactly the same hole: 수원 held 14.6% of its
gate count, 동인천 19.5%, 부평 25.8%. Incheon felt the difference too, because
the riders transferring in at 부평 off 경인선 were among the missing.

**And the split is per gate, not per complex.** Marking a whole complex
"inside" because it has one 서울교통공사 platform left 강남 at 77.5% of its gate
count, 모란 at 45.2%, 청량리 at 72.3% — 신분당선 강남 has its own fare gates, so
a 강남 → 양재 trip on 신분당 is as absent from the OD as 수원 → 안산 is.
`card_gates` splits each complex by who settles the gate, which is the
distinction the OD itself draws.

With both fixed, built boardings over measured gate count:

| | first pass | complex-level | per-gate |
|---|---|---|---|
| 서울교통공사 gates | 97.0% | 94.0% | **97.8%** |
| elsewhere | 37.7% | 96.5% | **97.3%** |

97–98% is the pipeline's own keep rate — the 2.0% it drops for an end off the
network, a same-complex round trip or an unroutable pair. Both groups now sit
on it, which is the most that can be asked.

**Do not compute that keep rate against gate counts.** `keep_frac` scales the
seeded block, and the obvious way to get it — trips kept over the gate count at
the stations we settle — gives 80%, because it folds in the source's own
coverage hole. Using it built 인천1호선 at 163k against a measured 203k. It is
`kept / (kept + dropped)`, measured against the file.

### What the totals should be

Total weekday trips: **8.42M**, against 8.62M of measured weekday gate
boardings across the whole network. **An earlier note here said a real weekday
was ~7M; that was the incomplete build describing itself.** Summing what the
operators publish — 서울교통공사 1,584M a year, Korail 광역 772M, 인천 147M,
공항철도 100M, 9호선 97M, 신분당 63M, and the rest — gives 2,832M a year, 7.8M
an average day and ~8.6M on a weekday.

**Boardings are not 수송인원, and headline figures are the latter.**
인천교통공사 publishes 수송인원 = 승차인원 + 유입인원, where 유입 is riders who
entered their network from another operator without passing an Incheon gate.
For 2023 that is 219,362,832 = 146,974,026 + 72,388,806, so **유입 is 33%** and
any remembered "인천1호선 does 300k a day" is 1.49× the boardings the station
bubbles draw. Line thickness does include those riders; the bubbles cannot.

### What the map says about it

The old caveat — "the origin–destination release only records their journeys to
and from the Seoul network, so what is drawn for them is that traffic, not
their ridership" — **stopped being true and had to be replaced, not deleted.**
Volumes are measured now. What is still modelled is *which journeys* those
riders make, and on some lines that is nearly all of them:

| line | modelled |
|---|---|
| 수인선 | 94.5% |
| 인천1호선 | 90.3% |
| 인천2호선 | 89.0% |
| 의정부선 | 88.1% |
| 분당선 | 68.5% |
| 신분당선 | 66.3% |
| **1호선** | **62.3%** |
| 경의중앙선 | 60.6% |
| 4호선 | 25.5% |
| 2/5/6/7/8/9호선 | 2.5–8.4% |

So both tooltips give the figure — "90% of journeys modelled" — rather than a
bare dash. Below 5% the note is dropped entirely. (The dashed line style and
its legend key were later dropped altogether; see "The static view".)

**1호선 is the line this newly catches.** It was never flagged before, because
the old test was the raw OD's within-line share and 1호선 sat at 36%, above the
threshold. But most of its length is Korail, so 62% of its riders' journeys are
seeded. Nothing about that was visible until the share was measured directly.

`build.py` computes it from the built OD: `build_od.py` stamps a per-pair
`modelled` fraction into `od_hourly.npz`, the router carries it along each leg,
and `write_stats` divides it out per line and per station. That is why it can
be attributed properly -- the router knows which line each leg used, where a
guess from the pair's endpoints would not. It overwrites `build_stations.py`'s
`partial`, which was read off the source and now describes the wrong thing.

### Still not fixed

- **No 혼잡도 for Incheon**, so these lines have no output check of the kind
  lines 1–8 get.
- **The 소사–원시 half of 서해선 — 12 stations — has no measured volume from any
  source found.** Seoul's card file carries only the three 대곡–소사 stations
  (김포공항, 부천종합운동장, 원종); KRIC covers 도시철도 operators and not Korail
  광역; and 한국철도공사_역별 승하차 현황 (15029727) explicitly excludes 광역전철.
  Those 12 float. 경기교통DB (`gits.gg.go.kr/gtdb`) publishes 전철 역별 승하차 by
  month and is the next place to look — 서해선 runs through 부천, 시흥 and 안산,
  all of them Gyeonggi.
- **Day-of-week factors for 신분당, 김포골드, 의정부 and 진접** are borrowed from
  the Seoul network, because only Incheon publishes a daily file. 신분당선
  commutes harder than the network mean, so that is where the borrow most
  likely costs something.

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
fetch_outside_seoul.py # the operators Seoul does not settle: KRIC station
                   #   monthlies + 인천교통공사 station x hour and station x day
outside.py         # those three, joined to the network and day-type corrected
fetch_airport.py   # 인천공항 passengers x hour, 도착/출발, transit separable
airport.py         # those, turned into the two shapes an airport trip has
daytype.py         # which day the map draws; the one place that decides
crowding.py        # optional: iterate build.py so crowded trains lose riders
build_stations.py  # complexes, platforms, coords, track geometry
build_od.py        # Furness onto the day, then IPF into hours -> od_hourly.npz
build.py           # RAPTOR routing + rider assignment -> trains.json
build_shapes.py    # bend train paths onto the track (build.py calls this)
index.html         # MapLibre animation, nycriders house style
```

### Boardings are counted per leg, and split two ways

A journey with a change of train is recorded at **both** trains: the rider adds
to the circle where they first board and again where they change. That is what
nycriders does, and it is also what an operator means by 수송인원 — Incheon's
three lines come out at 662,758 boardings against a published 수송인원 of
676,309, which is 98.0% of it and so the router's own keep rate. A station
circle is therefore **not** a gate-tap count; comparing one to 승차인원 makes it
look 1.5x too high.

`write_stats` keeps the two apart: `b` is every boarding, `b0` only the rider's
first. Transfers are `b - b0`, and the static view's **boardings: all /
starting / changing** buttons switch between them.

**A missing `b0` does not mean zero.** A station whose boardings are all
transfers has no `b0` key at all, so `stats.json` carries a
`build.split_boardings` flag to say the split exists; only its *absence* means
the file predates it. Reading a missing key as "no split here, count everything
as a first boarding" put 99.9% of the network into `starting`, which is how the
bug was caught — by running index.html's own `stationValue()` over a real
`stats.sample.json` in node rather than retyping the logic.

`build.py` takes about **7 minutes** for all 624 origins on a weekday build,
shaping included — 5 minutes routing, 3 shaping. A Sunday is roughly half that:
the work scales with riders, and a weekday carries 8.42M against 4.62M. It
routes on half the cores (`JOB_SHARE`), leaving the rest so the machine stays
usable; `--jobs` overrides that. Each worker wants ~300 MB, so eight of them is
about 2.5 GB. `--sample 60` uses 11 origins and takes
~55 seconds, most of which is the shaping pass — add `--no-shapes` and it is
under 25. That is what to use while changing anything; the printed invariants
are just as meaningful on a sample as on the full run.

It used to take an hour and a half, and most of the difference is in how
RAPTOR's inner loop is fed. Three things did it:

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
- **Nothing per-destination inside the spawn loop.** The destination's platform
  indices do not change with the spawn time, but were being rebuilt for each
  one — 239,038 spawns × ~245 destinations is 58M list comprehensions that
  always produced the same answer. `w.cx_dis` and `pattern["cx"]` are those
  answers, resolved once when the world is built.

Then the origin loop went across processes. Each `(origin, spawn time)` search
is independent, so workers rebuild the world from disk — the scan tables parse
faster than they unpickle — and hand back flat `(slot, riders)` arrays rather
than nested dicts. The negative side of every waiting bubble is exactly its
boarding, so it is recomputed at the merge instead of sent. Origins are cut
into a fixed 64 chunks and merged in chunk order however many workers there
are, so `--jobs` cannot change the output.

The one place speed was bought with fidelity is `MIN_SPAWN_GAP`. Spawning at
real departure gaps rather than on a 10-minute grid is exact, but it is 3x the
searches, and searches are the build. Merging gaps shorter than a minute gives
most of that back:

| `MIN_SPAWN_GAP` | searches | speed | boardings that move train |
| --- | --- | --- | --- |
| 0 (exact) | 239,038 | — | — |
| **60 (default)** | **-31%** | **1.43x** | **1.5%** |
| 120 | -45% | 1.93x | 4.1% |
| 180 | -54% | 2.45x | 6.2% |

Measured on a sample stratified by search count — the big interchanges own
nearly all the short gaps, so a flat every-Nth sample understates this badly
(it showed 1.05x where the stratified one showed 1.43x). A minute is well
inside every line's headway, so the two departures being merged are almost
always the same line's consecutive trains. Set it to 0 for the exact answer.

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
| `CRUSH_CAP` / `--cap` | `build.py` | how full a single train may be drawn, × 정원. 1.5, and `--no-cap` turns it off. The other invented thing in the pipeline — see "The crush cap" |
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

- **The airport train used to have a commuter rush hour. Fixed 2026-09-06;
  the two lag assumptions in `airport.py` are the cost.** What follows is the
  whole account, because the obvious fix is a dead end and the working one
  invents two numbers.

  Spotted on the map: 공항철도's 직통 showed a clean 08:00/18:00 double peak. The 직통 is
  modelled correctly — patterns 319/320, 26 trips each way, stopping only at
  서울역, 인천공항 T1 and T2, so RAPTOR cannot be abusing it as a commuter
  express; there is nowhere else to ride it to. The shape comes from the hourly
  fit. **Every one of 공항철도's 14 complexes is `measured=False`**, so in
  `build_od.py`'s `ipf()` they fall to the `else` branch of

  ```python
  seed[k] = B[o] / B[o].sum()   if measured[o] else sys_profile
  ```

  and `sys_profile` is the summed hourly profile of 서울교통공사's measured
  lines 1–8 — a commuter double peak by construction. The IPF then reshapes a
  pair only where an hourly constraint touches it, and for 서울역 → 인천공항
  neither end is measured, so **the seed survives into the output unchanged**.
  The airport line is wearing lines 1–8's rush hour.

  **The obvious fix was tried and does not work — do not spend the day on it
  again.** KRIC has an hourly endpoint nobody here had used,
  `citytimepassList.jsp`, the sibling of the `citystapassList.jsp` that
  `fetch_outside_seoul.py` already scrapes. One anonymous POST, no login,
  returns a whole year: 24 hourly bins × 15 operators × 승차/하차, and
  **공항철도 is one of the columns** (it reports boardings only; its alighting
  column is all zeros). So AREX's own measured hourly shape is free to get.

  It is the same shape we are already using. Against 서울교통공사's, Nov 2023:

  ```
  correlation between the two shapes   0.987
  share in the two busiest hours       AREX 19.1%   Seoul 1-8 20.0%
  total absolute difference            7.8 pts across 24 hours
  ```

  **Because at the operator level AREX *is* a commuter railway.** The 일반
  service through 김포공항, 계양, 검암 and 청라 carries roughly five times the
  직통's riders, so it dominates the line total and gives it a genuine double
  peak. The 일반 having a rush hour is correct. **The artefact is confined to
  the 직통, which is about 15% of the line and invisible in any operator-level
  aggregate.** No line-level source can fix it, and KRIC's per-station endpoint
  is monthly.

  **What does fix it, and what it costs.** 인천국제공항공사's 시간대별통계
  (`statisticCategoryOfTime.do` on airport.kr) gives hourly passenger counts
  split 도착/출발, no login and no key; `fetch_airport.py` pulls it into
  `data/incheon_airport_hourly.csv` and `airport.py` turns it into two seed
  profiles. 환승승객 are fetched separately and **netted out** — a transit
  passenger never leaves the airport, so they are not rail demand, and at 인천
  they are 10.6% of everyone.

  The two directions are not the same shape, and that is the point. People fly
  out in the late morning (인천's departures peak at 10:00), so they travel
  *to* the airport early and the airport-bound profile has **no evening peak at
  all**; arrivals peak late afternoon, so the outbound profile peaks at
  17:00-18:00 and stays high to the last train.

  **The invented part**, and it is the second invented thing in the pipeline
  after `EXTEND_LAST_HOUR`: the passenger counts are measured but converting
  them to *train* boardings needs two lags nobody publishes —
  `LANDING_TO_TRAIN` (deplane, immigration, baggage, walk) and
  `TRAIN_TO_FLIGHT` (how early you want to be at the terminal). Both are spread
  over a few hours rather than applied as a single offset, because a hard shift
  puts a spike in the profile that the real data does not have. They are named
  constants at the top of `airport.py` so they can be argued with.

  **What is *not* assumed is the mode share.** Only the shape is used — the fit
  rescales every pair to its own total — so whatever fraction of air passengers
  take the train divides out. Night is the exception and the timetable handles
  it: there are no trains, so nobody is put on one.

  Effect on the 2,315 pairs with an 인천공항 end, weekday build:

  | | before | after |
  |---|---|---|
  | airport-bound, 17:00-19:00 share | 27.2% | **15.4%** |
  | airport-bound, peak hour | 18:00 | 05:00 |
  | outbound, peak hour | 08:00 | 07:00, broad to 18:00 |

  **The large 05:00 bin is not a bug.** `HOURS` starts at 5 because the hourly
  file bins *everything before 06:00* into one open-ended column, and early
  flights genuinely load the first trains: 15.7% of 인천's net departures leave
  before 09:00 and every one of those passengers needs to be at the terminal
  before 06:00. The seed puts 8.1% there and the IPF's origin constraint pulls
  it to 12.5%.

  **The 직통 is no longer drawn as its own row in the drill panel, and that is
  the honest end of this.** Fixing the OD fixed the demand but not the
  assignment: the router has no fare, so it sees a faster train and nothing
  stopping anyone boarding it, and the 직통 becomes the escape valve whenever
  the 일반 is full. The drawn split came out *inverted* against the airport's
  own figures — 직통 towards the airport peaking at 18:00 when real departures
  peak at 10:00, and away from it at 08:00 when real arrivals peak at 17:00.
  It is a commuter pattern, because a commuter crush is what causes it.

  A fare penalty is the principled fix and was rejected on the numbers. Our
  직통 carries 9,726 a day against a real *record* day of 9,738 — but the whole
  line is 1.32× high (383,203 against a published 290,000 average), so as a
  **share** we are at 2.5% against a real 2-2.8%. The level barely needs
  moving; a penalty sized to fix the peaks would empty the service.

  So `NO_EXPRESS_SPLIT` in `index.html` folds 공항철도's express back into the
  total and the panel says why. The *total* stays drawn and is properly
  sourced — 공항철도's own measured hourly shape from KRIC correlates 0.987
  with what the build uses. Only the split between the two services is a claim
  nothing can support, and the map no longer makes it. The animation still runs
  직통 trains with riders aboard, which is a much weaker claim than an hourly
  bar chart.

  **The line total being 1.32× high is a separate, unchased thread.** Some of
  it is weekday-against-annual-average and boardings counted per leg, both
  pushing the same way, but it is above even the published maximum day.

  Read any *other* unmeasured station's hourly figure as "the network's shape
  at that station's volume" — the same weakness applies wherever a pair has an
  unmeasured station at both ends, and 인천공항 is only the case where it was
  visible enough to chase.

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
