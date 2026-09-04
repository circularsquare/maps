# Seoul Subway Ridership Visualization

Animated map of Seoul subway trains sized by riders on board, in the style of
`../nycriders/`. **Status: end-to-end and working. Polish outstanding.**

Seoul came off `../metroslist.md` as a Tier 1.5 system — measured entry *and*
exit gates, so the numbers are real rather than inferred, but the published OD
carries no hour field and has to be disaggregated against hourly station counts.

---

## Picking this up

Read this section, then `todo.txt` (Anita's own list — **hers to edit, not
ours**), then the rest of this file for the why behind any given decision.

**Run order.** Each step reads the previous step's output from `data/`:

```
python fetch_schedules.py  # ~1 min  -> data/kric_*.xlsx, data/incheon2_*.csv
python fetch_osm.py        # ~2 min  -> data/osm_routes.json, osm_stations.json
python kric.py             # ~40 s   -> data/timetable_extra.csv
python build_stations.py   # ~40 s   -> data/stations.json
python build_od.py         # ~4 min  -> data/od_hourly.npz
python build.py            # ~90 min -> data/trains.json, stats.json,
                           #            link_shapes.json  (--sample 60 for ~50 s)
python validate.py         # ~40 s   -> checks the build against published figures
python -m http.server 8000 # then open index.html
```

**Run `validate.py` after a build.** It compares each line's end-to-end
journey time, service span and station count against the operators' published
figures, walks every line's stop order looking for a station that landed far
from its neighbours, and prints the OD coverage table. It has already caught
two real bugs that looked like data — see "One bad stop time" below.

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

**As of the end of the 2026-09-03 session:** the pipeline carries all 22 lines,
but `data/trains.json` on disk is a small `--sample` run made while checking
them. Re-run `build.py` for the real thing.

**How to tell a sample from a full build:** file size will not tell you. Open
the page and look at the rider counter at 17:30 — a full build over all 22
lines reads roughly **90,000 riders on ~350 trains**, a `--sample 60` run about
**1,500 on ~200**. Or check that `build.py`'s output says
`routed 3,3xx,xxx riders` rather than `40,xxx`.

**Three traps that cost real time.** All are fixed, all are documented below,
and all would be easy to reintroduce:

1. `00:00:00` in the timetable is a **null marker**, not midnight.
2. RAPTOR needs **one arrival label per round**; a single parent pointer
   silently loses riders.
3. Korean portal signup is **impossible from abroad** — never go down that road.
   See `[[reference_korea_open_data]]` in memory for the anonymous routes.

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
| `data/osm_stations.json` | 786 rail station nodes, Seoul-area bbox | 0.6 MB |
| `data/osm_routes.json` | 165 route relations, ordered stops + track geometry | 7 MB |

The Korean files are CP949-encoded; the OSM pulls are UTF-8 JSON. `data/` is
gitignored repo-wide, so none of it commits.

**The one real limitation is the date.** The OD exists for exactly one day,
2023-12-31 — a Sunday, and New Year's Eve. Everything else is fine.

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
- **Which day.** 2023-12-31 is all we have and it is New Year's Eve. Either
  lean into that as the subject, or ask Seoul for an ordinary weekday — we hold
  hourly counts for every day of 2023 and 2024, so any date they give us pairs
  immediately. See `od_request.md`.
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

## Pipeline

```
fetch_hourly.py    # CardSubwayTime -> data/hourly.csv
                   #   needs a Seoul API key, which cannot be obtained from
                   #   abroad. Superseded by the OA-12921 archive for
                   #   everything except lines 9 / Korail hourly counts.
                   #   Kept only because it is the sole route to those.
build_stations.py  # complexes, platforms, coords, track geometry
build_od.py        # IPF against hourly marginals -> od_hourly.npz
build.py           # RAPTOR routing + rider assignment -> trains.json
build_shapes.py    # bend train paths onto the track (build.py calls this)
index.html         # MapLibre animation, nycriders house style
```

`build.py` takes about **3 minutes** for all 624 origins, shaping included. It
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

The quickest check by hand is the file size: a full `trains.json` is ~21 MB.
Anything much smaller is a sample.

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

### Spawn times

The OD says how many people left a station in an hour, never when inside it.
`build.py` spreads them over `DEP_BIN`-wide spawns and runs RAPTOR again at
each, so a rider boards the train that is next *then* rather than the one that
was next at the top of the hour.

Two things about that grid were wrong until 2026-09-04, and both showed on the
map as trains a few minutes apart carrying wildly different loads:

- **`DEP_BIN` was 600 s.** Ten minutes is longer than the headway on every
  trunk line, so most trains had no spawn tick in their window at all and
  boarded nobody.
- **The grid was shared.** Every origin in the network released at `:05`,
  `:15`, `:25` … together, so whichever train pulled out just after each tick
  scooped the lot. `dep_phase()` now offsets each origin by a stable fraction
  of a bin, so the network no longer breathes in step.

Measured on `--sample 60`, over eight busy stations between 16:00 and 22:00 —
the mean gap in load between one train and the next, over the mean load:

| | 600 s, shared grid | 300 s, phased |
|---|---|---|
| load step between successive trains | 1.24 | **1.13** |
| trains calling with nobody aboard | 23% | **16%** |
| platform arrivals landing in the `:05` minute | 18.5% | **11.1%** (flat) |

The `:05` spike is the signature of the bug and it is gone.

**Then `DEP_BIN` went to 150 s, and that is the last of it.** Simulating the
quantisation against the real departure times and the real hourly volumes —
riders spawn on the grid, each takes the first train after their tick — gives
the whole curve, against the continuous-arrival floor that no spawn splitting
can beat:

| `DEP_BIN` | load step | share of the achievable gap closed |
|---|---|---|
| 600 s | 1.93 | 0% |
| 300 s | 1.36 | 60% |
| **150 s** | **1.02** | **95%** |
| 60 s | 0.99 | 99% |
| continuous | 0.98 | — |

So 150 s is the knee: 60 s costs 2.5x as much for four more points. What is
left at 1.02 is the OD's own hour-level lumpiness, which is data, not method.

**And it is not slower.** RAPTOR from an origin at time `t` depends only on
which departure is next at each stop the search seeds — the origin's platforms
and whatever a transfer reaches. Between two of those departures every bin
gives a bit-identical answer, so `spawn_key()` keys the search on exactly that
and `route_origins()` reuses the previous result when the key has not moved.
On `--sample 8`, **52% of spawns never run a search**: 18,881 against the
19,524 the old 300 s grid needed, so the full build stays around 90 minutes
despite twice the spawns. Quiet outer stations are where it pays — four
departures an hour against twenty-four spawns.

The reuse was verified by building `--sample 60` with and without it and
diffing: every one of 5,456 trains and 626 stations identical, only the build
stamp differing. **Re-do that test if you touch `raptor()`'s seeding** — adding
anything to what round 0 depends on invalidates the key, and the failure mode
is silent mis-assignment rather than a crash.

The numbers to watch if this ever looks wrong again: the load step, the share
of calls with nobody aboard, and the flatness of the platform-arrival
histogram.

### Knobs worth knowing

| | in | what it does |
|---|---|---|
| `EXTEND_LAST_HOUR` | `build.py` | reconstructs New Year's Eve late service. The one invented thing in the pipeline; `False` shows the night as the regular timetable would have it |
| `LATE_BIN_HOURS` | `build.py` | how many hours the open-ended `24시이후` bin is spread across. 2 |
| `DEP_BIN` | `build.py` | how often riders spawn inside their hour. 150 s, which is the knee — see "Spawn times". Lowering it further costs real time and buys nothing |
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

`data/od_hourly.npz`: **112,774 pairs × 20 hours**, 3,007,753 riders. 13.2% of
the day's trips are dropped — one end off lines 1–9, or a same-complex round
trip that carries no journey.

The fit converges in 30 rounds and reproduces the measured marginals exactly.
That is not too good to be true: the hourly profiles are rescaled so each
station's daily total equals what we actually carry, which makes the three
constraint families mutually consistent, and IPF on a consistent system has an
exact solution. Pair totals are preserved to 1e-6.

**The inference does real work.** By volume: 73.0% of trips have both ends
measured, 24.0% one end, and only 3.0% are seed-only. Unmeasured stations move
measurably away from the seed profile, and the ones that move furthest are the
outer suburban stations — 안산, 신창, 성환, 동두천, 오산 — exactly where trips are
long enough that the arrival-hour shift carries information.

Cross-checked against the raw hourly file: per-station shapes come through
intact, and **잠실 shows 11,995 boardings in the midnight hour against 1,378 at
23:00**, with arrivals peaking again at 23:00 before midnight. The countdown
crowd arriving and then going home is sitting right there in the data.

The system-wide dip at 23:00 is real, not an artefact — people are already at
whatever they came for.

### build.py output

`data/trains.json`. 268 patterns and 3,899 trips make up the Sunday/holiday
service. Riders spawn every 10 minutes inside their hour and RAPTOR runs afresh
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

- **The one invented thing: `EXTEND_LAST_HOUR` in `build.py`.** No trip in the
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
