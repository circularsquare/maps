# Korea register sources (surveyed 2026-09-30)

What open Korean data can serve as noritetsu's line register, dataset by dataset,
then the recommended combination, the coverage gaps, and the published-length table
for `check_model.REGISTER`.

Everything here was downloaded without a login or key. Files are in this folder;
`probe_kric.py` fetches the KRIC ones (`--fetch`) and reproduces the two reports
beside this file (`--report` -> `probe_report.txt`, `--distance-table` ->
`probe_distance_table.txt`). The data.go.kr files were fetched by hand-rolled
requests: the dataset page `https://www.data.go.kr/data/<id>/fileData.do` carries
`fn_fileDataDown('<id>','uddi:...','','1',...)`; GET
`/tcs/dss/selectFileDataDownload.do?publicDataPk=<id>&publicDataDetailPk=<uddi>&fileDetailSn=1`
returns JSON with the `atchFileId`, and
`/cmm/cmm/fileDownload.do?atchFileId=<FILE_...>&fileDetailSn=1` is the file. Some
pages carry the `fileDownload.do` link directly.

`noritetsu/kr_sources.py` loads the three that matter (the 거리표 with SR's matrix, KRIC
1294, and the English names) keyed by OSM track name; `python kr_sources.py` writes
`kr_sources_summary.txt` here.

The downloads themselves are in `data/raw/kr/`, which is gitignored; this file is the
tracked record of them.

## The short answer

- **Korail's 각 선구별 거리표 (data.go.kr 15137040) is the register for every Korail
  and SR railway line.** It is the official distance table: one triangular matrix per
  line, stations *in order*, every section's km, junction points named
  (`시흥연결선북단`, `대전북연결선북단`, `익산분기기`, `광주선분기`...). It covers the
  intercity trunk and branch lines *and* the legal lines the 광역전철 runs on (경원,
  경인, 안산, 과천, 분당, 일산, 수인, 경의, 중앙, 경춘, 경강, 서해 Korail part, 교외).
  Dated 2024-09-01. Its line totals match the 2023 yearbook's 영업거리 exactly on 24
  of the 34 lines compared; the other ten are extent differences (서원주 vs 원주,
  판교 vs 성남, a shared section counted or not) or 0.2–1.4 km apart.
- **KRIC 1294 (전국 도시광역철도 역사정보) is the register for every metro and light
  rail operator**: stations in line order for the non-Korail operators, lon/lat, English
  names, and 상행거리/하행거리 (the km to the neighbouring station). Its Korail rows are
  scrambled and must not be used.
- **Lengths**: the 2023 철도통계연보 already on disk publishes 영업거리 for every Korail
  and SR legal line (part 1, `2. 역수 및 영업거리` sheet 2) and for every metro line
  (part 2, `2. 운영현황` sheet 1), as of 2023-12-31.
- **Coordinates for intercity stations: nothing open is trustworthy.** Use OSM. Every
  Korean coordinate file checked has gross errors (swapped lon/lat, another station's
  point, a Seoul point for a Busan station).
- **English names**: RAFIS 역 정보 (data.go.kr 15132601) has English and hanja for
  692 Korail stations including every intercity stop; KRIC 1294 has them for the
  urban operators.

## Dataset by dataset

### KRIC 18 — 표준데이터 노선정보(전체 기관)  `kric_18_line_all.xlsx`

47 rows, one per line (some lines split into two rows), 10 columns:
노선번호 (national line code, e.g. `I4101`), 노선명 (line name), 기점명 / 종점명 (start /
end), **정거장구성** (the station list, `code-name` pairs packed in one cell),
**노선연장** (length, metres), 개통일자 (opening date), 운영기관명 (operator),
운영기관전화번호, 데이터기준일자 (as-of date, Excel serial).

- Covers every urban operator plus Korail's 광역 lines (동해선 광역, 경강, 수인, 경춘,
  일산, 분당, 안산과천, 경의중앙, 경원, 경인, 경부 광역 서울–신창, 서해 김포공항–원시,
  대경선). No GTX-A, no 서해철도 (소사–원시 is folded into Korail's 서해선 row), no
  intercity lines.
- No coordinates, no English, no per-section km.
- 정거장구성 is **roughly** in order: metros are in order; Korail lists insert branch
  stations inline (경부선 lists 광명 between 독산 and 금천구청, 서동탄 after 병점),
  put 가좌 in the wrong place on 경의중앙, and omit 경의중앙's whole 용산–청량리 section.
- Traps: the list separator varies (`,`, `, `, `+`, embedded newlines, triple-quoted
  strings). 노선연장 is metres except 에버라인 (`18.143`, km) and blank or `-` for 광주 1,
  의정부, the maglev. 서울 9호선's 개통일자 and 운영기관명 cells hold multi-line prose.
  Seoul 2호선 is three rows (loop + 성수지선 + 신정지선). 노선번호 `I28K1` is used
  twice (수인선 and 분당선), `S1107` twice (Seoul 7 and 인천 7호선 extension).

### KRIC 32 — 표준데이터 역사정보(전체 기관)  `kric_32_station_all.xlsx`

1,099 stations, 15 columns: 역번호 (station number), 역사명 (name), 노선번호, 노선명,
영문역사명 (English), 한자역사명 (hanja), 환승역구분 (interchange flag), 환승노선번호/명,
역위도, 역경도 (lat, lon), 운영기관명, 역사도로명주소, 역사전화번호, 데이터기준일자.

- Same coverage as 18 (urban operators + Korail 광역, 경부선 광역 split out as
  경부선/장항선). Every row has coordinates and English. No order column; no km.
- Superseded by 1294 for the urban operators (1294 has the km). Useful as a second
  opinion on coordinates.

### KRIC 1294 — 전국 도시광역철도 역사정보 (29 fields)  `kric_1294_station_national.xlsx`

Sheet `1_역사정보`, 1,108 rows, 29 columns: 철도운영기관명 (operator), 운영노선 (line),
역 종류, 역 번호, 역명(한글), 역명(영어), 역명(로마자), 역명(일본어), 역명(중국어간체/번체),
역명(부역명) (secondary name), 환승역 여부, 환승노선명, 4 platform-facility columns,
승강장 유형, **역 위치(경도), 역 위치(위도)**, 2 addresses, 전화, 신설일자 (opened),
폐지일자, **상행거리, 하행거리** (km to the adjacent station, up / down), 데이터 기준일자,
참고사항. Sheet `Sheet1` is a copy of the first five columns; ignore it.

Operators and lines (row counts; every row has lon/lat and English):

| operator | lines |
|---|---|
| 공항철도 | 공항철도선 14 |
| 광주교통공사 | 1호선 20 |
| 김포골드라인 | 10 |
| 남서울경전철 | 신림선 11 |
| 남양주도시공사 | 진접선 3, 8호선 (별내선) 2 |
| 구리도시공사 | 8호선 (별내선) 3 |
| 대구교통공사 | 1호선 35 (+1 stray row), 2호선 29, 3호선 30 |
| 대전교통공사 | 1호선 22 |
| 부산교통공사 | 1호선 40, 2호선 43, 3호선 17, 4호선 14 |
| 부산김해경전철 | 21 |
| 서울교통공사 | 1–8호선 (10, 51, 34, 26, 56, 39, 42, 19), 9호선 언주–중앙보훈병원 13 |
| 서울시메트로9호선 | 9호선 개화–신논현 25 |
| 서해철도 | 서해선 소사–원시 12 |
| 신분당선 | 16 |
| 용인경전철 | 에버라인 15 |
| 우이신설도시철도 | 13 |
| 의정부경전철 | 15 |
| 인천교통공사 | 인천1호선 33 (incl. the 2025 검단 extension), 인천2호선 27, 7호선 extension 11 |
| 지티엑스에이운영 + ㈜SR | GTX-A 8 + 1 (동탄 filed under SR) |
| 인천공항 | 자기부상 6 |
| 코레일 | 1호선 92, 3호선 10, 4호선 22, 경의중앙 58, 경춘 25, 분당 35, 수인 22, 수인분당 6, 대경 8, 동해 23, 경강 12, 서해 9 |

- **Order**: for every non-Korail operator the rows are in line order, with branches
  appended after the trunk (Seoul 2호선's 성수·신정 지선; 5호선's 마천 branch; 9호선 in
  two operator blocks). **Korail's rows are sorted by station-code string**, so lines
  interleave row by row, codes repeat, and many stations are missing. Do not take
  Korail from this file.
- **Per-section km**: 상행거리/하행거리 exist for every urban operator, and summed they
  reproduce published lengths (Busan 1–4: 39.9 / 45.2 / 18.1 / 12.0, exactly the
  yearbook). But **the convention differs by operator**:
  - AREX, 신림선, 김포, 부산김해, 에버라인, 의정부, 우이신설, 부산: 상행 = km to the
    previous row, 하행 = km to the next row (down[i] = up[i+1]).
  - 광주, 대전, 인천 1·2, GTX-A: the reverse (상행 = km to the next row).
  - 서울교통공사 1–8호선: 상행 = 하행 = km to the *previous* row, with float noise
    (`1.1000000000000014`), and the first row is `0` or the loop-closing distance.
  - So take the km between consecutive rows as whichever of down[i] / up[i+1] the
    operator uses. Checked row by row: down[i] = up[i+1] holds throughout for AREX,
    Busan 1–4, 부산김해, 김포, 신림, 우이신설, 에버라인, 의정부 and 대구 3; it fails on
    nearly every row for 광주, 대전, 인천 1·2 and 서울교통공사 (the other conventions);
    신분당 (2 rows), 서해철도 (3), 대구 1 (10) and 대구 2 (2) have a few rows that
    disagree either way, i.e. typos.
- **Coordinate errors found by eye** (there will be more; check every point against
  OSM): GTX-A 운정중앙–서울역 lon/lat swapped; 인천공항 자기부상 all swapped; 부산 1호선
  구서, 두실, 남산, 범어사, 노포 placed at 128.96/35.05 (west Busan, 20 km off);
  대구 1호선 현충로 `128.3453/35.50272`, 율하 `128.41/35.52`, 각산 `1284326.88/35527.82`;
  대구 2호선 담티–신매 rounded to two decimals; 서울 2호선 용답 at 시청's point; 5호선
  마곡 and 발산 identical; 6호선 연신내 127.09 (should be 126.92); 신분당 판교 37.2337,
  동천 127.06/37.20; 의정부 발곡 37.275 (should be 37.72); 인천1 동수 off by 2 km.
- Other traps: names carry sub-names in brackets and sometimes a trailing `역`
  (`판암역`, `광주송정역`); some names contain newlines; km values carry a `km`
  suffix on the GTX-A rows; 대구 1호선 has a second one-row block (중앙로, code 3140)
  after 3호선; `-` and blank both mean "none".

### KRIC 31, 45, 1264, 216, 79, 602 — Korail's own

All six are **광역전철 only**; not one intercity station is in any of them.

- **45** 표준데이터 역사정보(코레일), 332 rows, same 15 columns as 32, by legal line
  (경부선, 경원선, 경인선, 경의중앙선, 장항선...). **The 역번호 column is misaligned
  with the names** (용산 and 용문 share `1003`; 구로 is `1026`); do not join on it.
  양원 carries the 봉화 (영동선) coordinates, not the Seoul station's.
- **1264** 코레일 역사정보, 301 rows, the 29-field layout of 1294 by service line (1호선,
  경의중앙...), with 상행/하행거리. Older (2022–23) data, same quality problems.
- **216** 코레일 역위치, 301 rows: 선명, 역명, 경도, 위도; some `-`.
- **79** 코레일 표준데이터의 역정보, 275 rows: code, name, English, romanised, Japanese,
  Chinese. 2018.
- **31** 표준데이터 노선정보(코레일), 13 rows: exactly the Korail rows of 18.
- **602** 코레일 호선정보, 9 rows of 2018 service-line extents. Useless.

### KRIC 624, 629, 633 — 호선구성역정보 (station order + chainage)

- **624 부산교통공사**: 114 rows: 선명, 역명, 역구성순서 (order), 구간키로 (section km),
  기점키로 (chainage). Complete for Busan 1–4, sums to 39.9 / 45.2 / 18.1(3호선
  구간키로 sum 18.3) / 12.0. Superseded by 1294, which has the same km and coordinates.
- **629 대구, 633 대전**: 2018, **no station names** (order numbers only), 구간키로 empty,
  기점키로 only. 대구 1호선 has 32 rows (before the 2024 하양 extension). Not needed.
- No other operator publishes a 구성역 table on KRIC (catalogue grep for 구성, 키로,
  거리, 순서). data.go.kr has per-line 역간거리 files from 국가철도공단 for Seoul 1–9,
  수인, 분당, 경강, 경의중앙, 경춘, 신분당, AREX, 에버라인, 우이신설, 의정부, 인천 1·2,
  코레일 (listed under "Other files" below); they repeat what 1294 has.

### data.go.kr 15137040 — 한국철도공사 철도운행거리_전체 (각 선구별 거리표)  `dgk_15137040_각 선구별 거리표.xlsx`

**The key file.** 21 sheets (+1 empty), dated 2024-09-01: 1 경부KTX운행거리, 2 호남KTX,
3 연결선, 4–5 경부, 6 경원·경춘, 7 경의·안산 (+교외선, 용산선, 서울–청량리), 8 장항·경인,
9 충북·경북 (+문경선), 10 호남, 11 전라, 12 경전 (+진해, 광양, 부산신항), 13 동해
(+가야, 우암, 괴동, 울산항 branches), 14 중앙, 15 태백 (+정선, 함백), 16 영동 (+삼척,
북평), 17 과천·분당·일산, 18 경강 (성남–여주 and 서원주–강릉), 19 수인, 20 서해,
21 중부내륙.

- **Layout**: each line is a triangular distance matrix. Station names sit on a
  down-right diagonal, (r,c), (r+1,c+1)...; pairwise distances are either to the right
  of each name on its row, or below it in its column. So a line is a diagonal chain of
  text cells, and the km between neighbours is the cell next to the diagonal.
  `probe_kric.py --distance-table` does exactly that and every chain's section sum
  matches its own end-to-end cell except 호남KTX (370.2 vs 370.6). Output:
  `probe_distance_table.txt`, about 80 chains.
- **What it gives**: stations in order, section km to 0.1 km (one 0.01: 이천–부발 4.49),
  junction points as named nodes, freight branches and depot spurs (filter by name:
  `기지`, `화물`, `신호소`, `분기`, `연결선`).
- **Traps**:
  - Continuation chains start with a **prefix jump**: 경부(2) begins `서울 -166.3- 대전`,
    중앙선 `청량리 -148.5- 도담`, 경전 `삼랑진 -209.3- 보성`, 연결선 `서울 -296.7- 노령`.
    The first "section" is the cumulative distance to the join, not a section. Drop a
    chain's first section when that station also opens another chain on the line.
  - Several matrices are the *service* routes rather than the legal line: sheet 1 is
    경부 KTX 서울–부산 over 경부선 to 금천구청, 시흥연결선, then the high-speed line;
    sheet 2 is 호남 KTX 용산–목포. The legal 경부고속선 is the `시흥연결선 종점 ~ 부산`
    chain on sheet 3 (398.2 km, 광명 천안아산 오송 대전 김천(구미) 동대구 경주 울산 부산).
    호남고속선 is the `오송 ~ 광주송정` chain on sheet 2 (182.4 km).
  - Names are written differently from the yearbook in places: `김천(구미)` vs
    `김천구미`, `대존조차장` (typo) / `대전조` / `대전조차장`, `DMC` vs
    `디지털미디어시티`, `망 우`, `제 천` with spaces, `거제\n해맞이` with a newline,
    `체천조` (typo for 제천조), `신경주분`, `대구남`, `부산북`, `익산분기기`.
  - 수인선 is 수원–인천 51.6 including the 한대앞–오이도 section shared with 안산선; the
    yearbook's 38.8 excludes it.
  - The 2-station connecting-curve chains (`호남선 -1.0- 경전선`, `중앙선 -0.7- 영동선`,
    `태백선 -1.3- 영동선`) are named by line, not station.
  - SR's 수서평택고속선 is **not** in it (it is SR's, not Korail's). See 15040194.
  - Lines opened after 2024-09-01 are not in it (see Gaps).

### data.go.kr 15040194 / 15071271 — ㈜SR 경부선 / 호남선 영업거리

11×11 distance matrices (2021): 수서 동탄 지제 천안아산 오송 대전 김천구미 동대구 신경주
울산 부산, and the 호남 equivalent. Gives 수서–동탄 32.4, 수서–지제 53.4, 지제–천안아산
25.1 (via 평택분기 onto 경부고속선). The only source for SR's own line.

### data.go.kr 15081858 — 국가철도공단_코레일 역간거리 (2026-05-06)

447 rows: 철도운영기관명, 선명, 역명, 역간거리(km). Korail 광역 **service** lines in
order: 1호선 as four full-route patterns (경인선, 광명선, 서동탄선, 경부선 — each
repeating 연천–회기), 3호선 (일산선), 4호선 (안산·과천), 수인분당, 경춘 (+상봉–광운대),
경의중앙 (도라산–지평, then 신촌–서울역 appended), 경강, 서해선 (일산–부천종합운동장),
동해 (부전–태화강), 대경선 (경산–구미). The km on a row is to the **next** row.
Traps: 1호선 jumps 회기 → 남영 across the 서울교통공사 section (회기's 1.4 is to
청량리, which is not listed); the last row of a pattern is blank. Newest file in this
survey; good for the service patterns OSM will also carry.

### data.go.kr 15132601 — 국가철도공단 RAFIS 역 정보 (2024-08-16)

692 rows, 8 columns: 역명, 역약어명 (abbreviation), **영문역명**, 한문역명, 고속선노선거리,
기준역간KP (chainage in 100 m units along the line), 역신설일자, 역폐지일자. CP949.
No line column and no coordinates; rows are grouped by line in file order, so the KP
sequences can be read as lines, but the 거리표 does that better. **Its value is English
and hanja for every Korail station, intercity included**, plus depots and signal boxes.

### data.go.kr 15153835 — 화물정보 역간최단거리 (valid from 2025-01-01)  44 MB

Every Korail station pair: 출발역명, 도착역명, 적용시작/종료일자, 구간거리내용 (the route
as `A→B(km,cum)→...`), 여객최단운행거리, 화물운행거리. Valid from 2025-01-01, so it
**includes 동해선 삼척–영덕** (근덕, 울진... appear). For any line newer than the
거리표, the passenger shortest distance from one terminus to each station is its
chainage. Heavy; delete once whatever is needed has been extracted.

### Station coordinates, Korail intercity

- **15127532 한국철도공사 역 위치 정보** (2024-04): 202 stations: 지역본부, 역명, 위도,
  경도, 출입구 개수. 행신 is at 37.36/126.50 (should be 37.61/126.83).
- **15067652 국가철도공단 철도역 정보** (2025-07): 215 major stations: address, lat,
  grade, lines served (`경부선(고속),호남선(고속)...`), trains per day, English,
  Chinese, history, description. 청량리 at 37.11/129.04 (should be 37.58/127.05).
- Neither covers all ~340 intercity stops, and both have gross errors. **Use OSM
  station nodes**, matched by name along the traced line; keep these as a cross-check.

### The 2023 철도통계연보 (riders/koreariders/data/korail_yearbook_2023_excel.zip)

What `riders/koreariders/lines.py` reads:

- `rosters()`: part 1 `8. 시설` sheet 2 (역사 내외부 시설 현황), ~405 stations each on
  **one** home line, sorted alphabetically within the line, Korail + SR. Misses small
  halts (양원) and all 광역-only stations. Not an order.
- `distances()`: part 1 `8. 시설` sheet 4 (영업선로별 철도거리): legal extent (기점, 종점)
  and track km split 단선/복선/복복선/3복선, 110 rows. This is *track* distance.
- **Not read by koreariders, and better for REGISTER**: part 1 `2. 역수 및 영업거리`
  sheet 2 (선별 역수 및 영업거리): per legal line, its 구간, station counts by class
  (보통역, 간이역, 조차장, 신호장, 신호소), **영업거리 여객 / 화물** and 철도거리. As of
  2023-12-31. The figures in the table below come from here.
- **No station-to-station distances anywhere in the yearbook**, as koreariders found.
  The 거리표 above is what fills that.
- Part 2 (도시철도) `2. 운영현황` sheet 1 (구간별 개통현황): per operator and line, the
  extent, station count, **영업거리**, 철도거리, 선로연장, by opening stage. 2023 xlsx.
- Part 3 (광역철도) `2. 운영현황` sheet 1: same for 광역 lines. **xlsb in the 2023
  bundle** (openpyxl cannot read it; pyxlsb is not installed); the 2022 bundle has it as
  xlsx, which is what was read here.

### Checked and not useful

- **KRIC 철도통계 `ABANEWList.jsp`** (선별역수 및 영업키로, www.kric.go.kr): same table
  as the yearbook's sheet, but only 2016–2021.
- **data.go.kr 15130547 국토교통부 철도역 구간 이용거리 (2026-08)**: pairwise
  distances, but only 부산교통공사 has station codes; the 747 Korail rows have none.
- **data.go.kr 3050970 역 운영현황 및 영업거리 (2022)**: 12 national totals.
- **data.go.kr 15131906 KR_STATIONS_DATA_TABLE**: 464 station names with a page number
  of some brochure. No data.
- **data.go.kr 15042115 역명(한글_영어_한자) (2022)**: 300 Korail stations, English and
  hanja and address. RAFIS supersedes it.
- The per-line 철도운행거리 files (강릉선, 경부고속선, 영동선, 전라선, 호남고속선;
  2024-01/08) are the same matrices as 15137040, older.

## Synthesis

### Recommended combination

| part of the network | lines, order, section km | coordinates | English |
|---|---|---|---|
| Korail legal lines, intercity and 광역 (경부, 호남, 전라, 경전, 장항, 중앙, 영동, 태백, 충북, 경북, 동해, 경춘, 경원, 경의, 경인, 안산, 과천, 분당, 일산, 수인, 경강 ×2, 서해 Korail part, 교외, 중부내륙, 경부고속, 호남고속, branches) | **15137040 거리표** | OSM (15127532 / 15067652 as cross-check) | RAFIS 15132601 |
| SR 수서평택고속선 | 15040194 (수서–동탄–지제) + OSM for the 평택분기 junction | OSM | RAFIS / 1294 |
| Every metro and light-rail operator, AREX, 신분당, 서해철도, GTX-A, maglev | **KRIC 1294** | 1294, each point checked against OSM | 1294 |
| Lengths for REGISTER | 2023 yearbook (part 1 sheet `2`, part 2 `2. 운영현황`), then KRIC 18 노선연장, then 1294 sums | | |
| Lines newer than 2024-09 (Korail) | 15153835 shortest distances, else OSM order | OSM | RAFIS |

The register unit should be the **legal line** (경부선, 경원선, 분당선...), as N02's 路線
is for Japan. The 광역 service lines (1호선, 수인분당선, 경의중앙선) are then operating
patterns, which OSM route relations already carry and 15081858 can confirm; they should
not be register lines, or 경부선 서울–천안 would be registered twice.

### Coverage, line by line

Covered with order and per-section km:

- **Korail intercity**: 경부선, 경부고속선, 호남선, 호남고속선, 전라선, 경전선, 장항선,
  중앙선, 영동선, 태백선, 정선선, 충북선, 경북선, 문경선 (freight-only now), 동해선
  부산진–영덕, 경강선 서원주–강릉, 중부내륙선 부발–충주, 진해선, 광주선, 대구선 (가천–영천,
  on sheet 5), 삼척선, 교외선 (reopened 2025-01-11), 가야선, 부전선, and the connecting
  curves (시흥, 대전남, 대구북, 건천, 강릉삼각, 망우, 용산, 구로, 북송정, 북영천...).
- **Korail 광역 legal lines**: 경원 (용산–백마고지), 경인, 경의 (서울–도라산), 용산선
  (용산–가좌–DMC), 중앙 (청량리–...), 경춘, 안산, 과천, 분당, 일산, 수인, 경강 판교–여주,
  서해선 대곡–원시.
- **Metros and light rail (1294)**: 서울교통공사 1–8, 9호선 (both operators), 인천 1·2,
  인천 7 extension, 부산 1–4, 대구 1–3, 대전 1, 광주 1, 부산김해, 의정부, 에버라인,
  우이신설, 신림선, 김포골드, 진접선, 별내선 (8호선 암사–별내, three operators), 신분당
  (신사–광교), AREX, 서해철도 소사–원시, GTX-A (운정중앙–서울역, 수서–동탄), 인천공항
  자기부상 (reopened 2025-10-17 as a tourist train under the 궤도운송법, 6.1 km).

**Covered by no dataset** (OSM order, then a named figure from somewhere else):

- **Korail lines opened after 2024-09-01**: 동해선 영덕–삼척 (2025-01-01; 15153835 has
  its distances), 중부내륙선 충주–문경 (2024-12), 서해선 홍성–서화성 intercity
  (2024-11), 중앙선 도담–영천 new double-track alignment (2024-12 / 2025-01). The
  거리표's 중앙선 shows the old stations.
- **GTX-A 서울역–수서** (with 삼성 passed without stopping) — planned for mid-2026 and
  reported delayed to about August; not in 1294. Check OSM for whether it runs.
- **대경선** is in 1294 and 15081858 as a service; it has no legal line of its own (it
  runs on 경부선), so it is an operating pattern, not a register line.
- **Tourist and heritage lines**: 해운대 해변열차 (미포–송정, 4.8 km on the old 동해남부선)
  and the 스카이캡슐 above it; 월미바다열차 (Incheon, about 6.1 km loop); 곡성 섬진강
  기차마을 steam line (about 10 km, 기차마을–가정). None is in any register; all are OSM
  only. (Rail bikes are not trains.)
- **Not open yet as of 2026-09-30**: 위례선 tram (planned 2026-12-26), 신안산선 (2028),
  동북선, 부전–마산 (not confirmed either way; check OSM).
- **Ownership changes**: SR was merged into KTX on 2026-09-01 (en.wikipedia "2026 in
  rail transport"), so 수서고속선's operator should read Korail now; the maglev left the
  수도권 전철 on 2025-07-07.

### Station order for intercity lines

It is published: the 거리표 gives every Korail line's stations in order. Nothing needs
ordering along OSM track except the post-2024-09 lines above, and SR's 수서–평택분기
(three stations).

### Pitfalls worth carrying into the reader

- Names differ between sources by brackets and sub-names (`판교 (판교테크노밸리)`,
  `김천(구미)`/`김천구미`, `신창(순천향대)`), a trailing `역`, and spaces or newlines inside
  names. Normalise before joining; koreariders' `STATION_ALIAS` has more.
- Two stations can share a name (양원 in Seoul and in 봉화; 좌천 on 동해선 and Busan
  metro 1; 판교 in 경기 and 충남 on 장항선; 송정 in Busan and Gwangju; 교대 in Seoul,
  Busan and Daegu). Match by line and position, never by name alone.
- The 거리표 names junctions and signal points as stations; flag them `junction`.

## Published lengths for check_model.REGISTER

Sources, in order of preference:

- **YB23-1**: 2023 철도통계연보, part 1 `2. 역수 및 영업거리` sheet 2, 영업거리 여객 (km),
  as of 2023-12-31.
- **YB23-2**: 2023 철도통계연보, part 2 `2. 운영현황` sheet 1, 영업거리 (km), 2023-12-31.
- **YB22-3**: 2022 철도통계연보, part 3 `2. 운영현황` sheet 1, 영업거리 (km).
- **DT24**: 15137040 거리표 (2024-09-01), the chain's end-to-end km.
- **K18**: KRIC 18 노선연장 (m, shown as km).
- **K1294**: sum of KRIC 1294's section km.
- **SR21**: 15040194.

Korail and SR (legal lines):

| line | extent | km | source | check |
|---|---|---|---|---|
| 경부고속선 | 서울~부산 (광명–부산 own metals) | 398.2 | YB23-1 | DT24 시흥연결선종점–부산 398.2 |
| 호남고속선 | 오송~광주송정 | 183.8 | YB23-1 | DT24 182.4 |
| 수서평택고속선 | 수서~평택(분기) | 61.1 | YB23-1 | SR21 수서–지제 53.4 |
| 경강선 | 원주~강릉 | 120.7 | YB23-1 | DT24 서원주–강릉 120.9 |
| 중부내륙선 | 부발~충주 | 56.9 | YB23-1 | DT24 56.3; since extended to 문경 |
| 경인선 | 구로~인천 | 27.0 | YB23-1 | DT24 27.0 |
| 경부선 | 서울~부산 | 441.7 | YB23-1 | DT24 441.7 |
| 경의선 | 서울~도라산 | 56.0 | YB23-1 | DT24 56.0 |
| 호남선 | 대전조차장~목포 | 252.5 | YB23-1 | DT24 252.5 |
| 경원선 | 용산~백마고지 | 94.3 | YB23-1 | DT24 94.3 |
| 충북선 | 조치원~봉양 | 115.0 | YB23-1 | DT24 115.0 |
| 경전선 | 삼랑진~광주송정 | 277.7 | YB23-1 | DT24 277.7 |
| 장항선 | 천안~익산 | 152.8 | YB23-1 | DT24 152.8 |
| 전라선 | 익산~여수엑스포 | 180.4 | YB23-1 | DT24 180.4 |
| 경춘선 | 망우~춘천 | 80.7 | YB23-1 | DT24 80.7 |
| 동해선 | 부산진~영덕 | 188.9 | YB23-1 | DT24 188.3; since extended to 삼척 (2025) |
| 중앙선 | 청량리~모량 | 332.2 | YB23-1 | DT24 332.2; new alignment 2024-12 |
| 영동선 | 영주~청량신호소 | 188.9 | YB23-1 | DT24 188.9 |
| 경북선 | 김천~영주 | 115.0 | YB23-1 | DT24 115.2 |
| 태백선 | 제천~백산 | 104.1 | YB23-1 | DT24 104.1 |
| 안산선 | 금정~오이도 | 26.0 | YB23-1 | DT24 26.0 |
| 과천선 | 금정~남태령 | 14.4 | YB23-1 | DT24 14.4 |
| 분당선 | 왕십리~수원 | 52.9 | YB23-1 | DT24 52.9 |
| 일산선 | 지축~대화 | 19.2 | YB23-1 | DT24 19.2 |
| 서해선 | 대곡~원시 | 38.5 | YB23-1 | DT24 40.3 (the 소사–원시 part is 서해철도's) |
| 경강선 | 성남~여주 | 57.0 | YB23-1 | DT24 판교–여주 54.8, K18 57.0 |
| 수인선 | 수원~인천 | 38.8 | YB23-1 | DT24 51.6 incl. the shared 한대앞–오이도 |
| 교외선 | 능곡~의정부 | 31.8 | YB23-1 | DT24 31.8 |
| 용산선 | 용산~가좌 | 7.0 | YB23-1 | DT24 용산–DMC 8.6 |
| 대구선 | 가천~영천 | 26.1 | YB23-1 | DT24 26.1 |
| 광주선 | 광주선분기~광주 | 11.9 | YB23-1 | DT24 11.9 |
| 진해선 | 창원~통해 | 21.2 | YB23-1 | DT24 21.2 |
| 정선선 | 민둥산~구절리 | 45.9 | YB23-1 | DT24 민둥산–아우라지 38.7 (passengers end at 아우라지) |
| 문경선 | 점촌~문경 | 22.3 | YB23-1 | no passenger service |
| 삼척선 | 동해~삼척 | 12.9 | YB23-1 | freight-only now |
| 가야선 | 사상~범일 | 8.3 | YB23-1 | |
| 시흥연결선 | 시흥~광명 | 1.5 | YB23-1 | |
| 병점기지선 | 병점~서동탄 | 2.2 | YB23-1 | DT24 2.2 |
| 오송선 | 서창~오송 | 4.6 | YB23-1 | DT24 3.6 |
| 대전선 | 대전~서대전 | 5.7 | YB23-1 | DT24 5.7 |
| 수색직결선 | 수색~검암 | 2.2 | YB23-1 | |

Metros, light rail and private lines:

| line | operator | km | source | check |
|---|---|---|---|---|
| 서울 1호선 | 서울교통공사 | 7.8 | YB23-2 | K18 7.8, K1294 7.8 |
| 서울 2호선 | 서울교통공사 | 60.2 | YB23-2 | K18 48.8+5.4+6.0 = 60.2 |
| 서울 3호선 | 서울교통공사 | 39.1 | YB23-2 | K18 38.2, K1294 38.2 |
| 서울 4호선 (당고개~남태령) | 서울교통공사 | 32.8 | YB23-2 | K18 31.1 (불암산~남태령) |
| 진접선 (당고개~진접) | 남양주도시공사 etc. | 14.892 | K18 | |
| 서울 5호선 | 서울교통공사 | 59.8 | YB23-2 | K18 59.8, K1294 59.8 |
| 서울 6호선 | 서울교통공사 | 36.4 | YB23-2 | K1294 36.4 |
| 서울 7호선 (장암~온수) | 서울교통공사 | 46.9 | YB23-2 | K1294 46.9 |
| 7호선 연장 (온수~석남) | 인천교통공사 | 13.98 | YB23-2 | K18 14.4 |
| 서울 8호선 (암사~모란) | 서울교통공사 | 17.7 | YB23-2 | before 별내선 |
| 8호선 별내선 (암사역사공원~별내) | 3 operators | — | K1294 sums per part | opened 2024-08-10 |
| 서울 9호선 | 9호선㈜ + 서울교통공사 | 40.6 | YB23-2 | K18 40.7 |
| 부산 1·2·3·4호선 | 부산교통공사 | 39.9 / 45.2 / 18.1 / 12.0 | YB23-2 | K1294 identical |
| 대구 1호선 (설화명곡~안심) | 대구교통공사 | 28.4 | YB23-2 | extended to 하양 2024-12; K1294 now 37–38 |
| 대구 2·3호선 | 대구교통공사 | 31.4 / 23.1 | YB23-2 | K1294 31.2–31.4 / 22.9 |
| 인천 1호선 (계양~송도달빛축제공원) | 인천교통공사 | 30.3 | YB23-2 | 검단 extension 2025-06; K1294 37.0 |
| 인천 2호선 | 인천교통공사 | 29.2 | YB23-2 | K1294 29.1 |
| 광주 1호선 | 광주교통공사 | 20.5 | YB23-2 | K1294 20.5 |
| 대전 1호선 | 대전교통공사 | 20.5 | YB23-2 | K18 20.47 |
| 부산김해경전철 | 부산-김해경전철 | 23.2 | YB23-2 | K18 22.361, K1294 22.4 |
| 의정부경전철 | 의정부경량전철 | 10.6 | YB23-2 | K1294 10.6 |
| 에버라인 | 용인경량전철 | 18.1 | YB23-2 | K18 18.143 |
| 우이신설선 | 우이신설경전철 | 11.0 | YB23-2 | K18 11.4 |
| 김포골드라인 | 김포골드라인운영 | 23.5 | YB23-2 | K18 23.67 |
| 신림선 | 남서울경전철 | 7.53 | YB23-2 | K18 7.76 |
| 공항철도 | 공항철도㈜ | 63.8 | YB22-3 | K18 63.8, K1294 63.8 |
| 신분당선 (신사~광교) | 신분당선/경기철도/새서울철도 | 17.3 + 13.8 + 2.4 = 33.5 | YB22-3 (+its footnote for 신사–강남) | K18 33.5 |
| 서해선 소사~원시 | 서해철도 | 23.4 | YB23-1 sheet 8.시설/4 (track) | K1294 about 22–23 |
| 동해선 광역 (부전~태화강) | 한국철도공사 | 63.8 | YB22-3 | K18 63.8 |
| 대경선 (구미~경산) | 한국철도공사 | 61.9 | K18 | runs on 경부선 |
| GTX-A 운정중앙~서울역 | 지티엑스에이운영 | 32.3 | K1294 | |
| GTX-A 수서~동탄 | SR / GTX-A | 32.4 | SR21 (수서–동탄 on 수서고속선) | K1294 32.8 via 성남·구성 |
| 인천공항 자기부상 | 인천국제공항공사 | 6.1 | K1294 sums 5.6 station to station | not a published figure |

Figures still needing a named source: every line in the "Covered by no dataset" list,
the 대구 1호선 and 인천 1호선 extended lengths (1294's sums are the only numbers on
disk), 별내선, and the tourist lines.

## Lines in pieces (2026-10-04)

Anita, 2026-10-04 ("yes, we can continue doing bridge over shared track"): the UK fix
(gb_sources.md "Lines in pieces") carried to Korea. A trip is entered station to station on a
line's strip diagram, so a register line whose sections do not all connect cannot be ridden
across its gap. kr_register now has the `split_pieces` hook build_model calls after
`drop_unridden_sections`, through the shared `pieces.py`: a gap is bridged over the track
between the pieces where trains run across (the `borrowed` sections credit the line whose
track it is), what cannot be bridged becomes one line per piece (the biggest keeps the id,
aliases.json `pieces` moves saved rides).

**Measured** on the build shipped 2026-10-03: 5 of 83 register lines in pieces, 168 km outside
each one's biggest piece. Each one, by what lies in the gap (the track found between the pieces
in data/proc/kr):

| line | pieces (km) | gap | cause | done |
|---|---|---|---|---|
| 수인선 | 19.8 + 18.9 | 오이도 - 한대앞 | shared track: 수인선 trains run over 안산선 there (the 거리표 lists the section on both lines) | bridged, 12.6 km over 안산선, 7 borrowed sections through 정왕, 신길온천, 안산, 초지, 고잔, 중앙 |
| 호남고속선 | 89.6 + 50.4 | 익산 - 정읍 | the line's own track: its named track enters 익산 over 0.8 km of 호남선 and unnamed station roads, so it never reached 익산 from the south | bridged over its own track, 43.0 km (40.7 its own name) |
| 인천 도시철도 1호선 | 30.2 + 2.9 | 계양 - 아라 | the line's own track: the 2025 검단 extension is mapped and named, but 0.6 km of unnamed track at 계양 cut it off | bridged over its own track, 계양 - 아라 4.0 km |
| 경강선 | 120.4 + 55.9 | 여주 - 서원주 | no track: 여주 - 원주 is being built. Two services today, the metro line 판교 - 여주 and the KTX line 서원주 - 강릉 | split |
| 서해선 | 63.9 + 40.1 | 원시 - 서화성 | no track: the link to 원시 is not built; the 2024 intercity line's trains reach Seoul by 안중 and 평택선 | split |

Two Korean settings beside the UK's (`kr_register.rules()`): the line's own named track costs
half and counts as under a route (`OWN_COST`), because OSM's KTX route relations lie on
호남선 beside 호남고속선 (the cheapest routed track from 익산 to 정읍 would otherwise have
been 호남선's); and `dense`, so a station on straight track with no OSM vertex within 150 m
still joins the track graph. Where a bridge or a split changed a line's sections,
`km_official` is worked out again from the published lists, or dropped if they do not give
every section.

**Split lines**: 경강선 keeps its id (`kfd60f34d9d`) on 서원주 - 강릉 (name_en "경강선
(Seowonju – Gangneung)"); 판교 - 여주 is `ked379c6f7e` ("경강선 (Pangyo – Yeoju)"). 서해선
keeps `kfa00e3def3` on 서화성 - 합덕 (the bigger piece by km); 대곡 - 원시, the metro line, is
`kb6b28601a6` ("서해선 (Wonsi – Daegok)"). Rides saved on the old ids between two stations of
the other piece move to it through aliases.json `pieces`.

**Crediting**: the only borrowed sections are 수인선's 12.6 km over 안산선, and riding them
credits 안산선 (12.3 km) and 안산연결선 (0.3); 0.4 km at 오이도 credits 수인선 itself where
its own track is. 호남고속선 and 인천 1호선 own their new sections (their own track). The
country's owned total (build_regions.owned_totals) goes 4,906.9 -> 4,950.2 km: the 43.3 km is
the two own-track bridges' track, newly owned; the borrowed km count once, for 안산선.

**Before -> after** (trial 2026-10-04): register lines 83 -> 85, 4,897.0 -> 4,956.6 km
(borrowed 12.6 km among them); lines in pieces 5 -> 0. check_model: 호남고속선 0.76 -> 1.00
(183.0 of 183.8), 수인선 now against the 거리표's 51.6 with the shared section (51.2), 인천
1호선 against KRIC 1294's 37.0 with the 검단 extension (37.1); 경강선 and 서해선 sum their
pieces as before (0.99, and 2.70 from the intercity piece the figure leaves out).
