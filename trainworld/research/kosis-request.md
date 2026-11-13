# What to ask the Korean volunteer for (2026-11-13)

Agent kosis, for the trainworld orchestrator. Covers trainworld demand, seoulriders and koreariders
(published as korearail). The draft message is `research/kosis-request-message.md`.

## For Anita

**The short version.** Most of KOSIS doesn't need a Korean ID: its tables download in your own
browser with no login, as the religiondots census table did. So the KOSIS tables trainworld wants are
on a "we get these ourselves" list below, and the volunteer is asked only for things that need a
Korean account. Those are requests to government bodies, mostly not KOSIS:

1. **A weekday subway journey file for Seoul (must, seoulriders).** Seoul publishes its
   station-to-station trips for one day only, a Sunday (31 Dec 2023), and sends other days by email
   when asked. Asking needs an account on the national open data portal (data.go.kr), which needs a
   Korean ID. A Korean request is already drafted (`riders/seoulriders/od_request.md`); the volunteer
   just submits it. One real weekday replaces the one guessed step in seoulriders and is a check on
   trainworld's Seoul. Free, credit-only licence, fine for a public map and the open-source game.
2. **Korail's station-to-station passengers (nice, koreariders and trainworld).** Not published at
   all. Anyone with a data.go.kr account can file a formal data request to Korail, which has to
   answer within about 10 working days. Korail may refuse; worth one try because it would turn
   koreariders' reconstructed loads into measured ones.
3. **The national transport database's passenger trips between regions (nice, trainworld).** The
   Korea Transport Database (KTDB) holds origin-destination trips by mode between about 250 districts.
   It needs a KTDB account and an approved application. **Its terms were not readable from here and
   very likely limit use to the applicant**, so the message asks the volunteer to check the terms
   before sending anything, and not to send it if passing it on is not allowed.
4. **Jobs for small areas (nice, trainworld).** The statistics agency's map service (SGIS) hands out
   jobs by neighbourhood (eup-myeon-dong) and finer, after login. Only needed if the free files we can
   fetch ourselves turn out not to have jobs; the message marks it "only if easy".

**Not asked: MDIS microdata.** MDIS is the census and survey microdata service. The fine-geography
files never leave its secure room (remote access only, and only approved result tables come out);
the public download versions are coarse (province level) and licensed to the person who downloads
them, so sending them on would very likely break the terms. Everything we would want from the 2020
census microdata is already published as KOSIS tables, so nothing is lost. If MDIS ever becomes
necessary, the only clean route is the volunteer computing a few named totals and sending those, and
that is a lot to ask a stranger.

**What can ship in the game.** The Seoul file and anything from data.go.kr or KOSIS is under Korea's
open government licence type 1 (credit the source, any use): fine for the open-source game, credited
on the data page. The KTDB data, if it comes at all, should be treated as "use for fitting, never put
in the repo or the packs": only fitted numbers (a curve, a few shares) would go into the game, and
only if its terms allow even that.

**How big, and how to send it.** The two likely items are small: about 15-20 MB for one weekday OD
(4-5 MB zipped), and probably under 20 MB for a Korail OD. Email works for those. KTDB is the only
possibly large one (tens to a few hundred MB, not checked). The message suggests: zip it; under
about 20 MB, email it; bigger, a shared Google Drive folder or a WeTransfer link. Seoul's
department replies by email to whatever address the request gives, so if you are happy to have your
own email address in a request to Seoul City, the volunteer can put it there and nothing has to be
forwarded. Your call; the message as drafted leaves it to the volunteer's address.

**We get these ourselves** (you in your browser, or an agent where a file link works without login):
the 2020 census commuting flows and the business census jobs by district for trainworld (task
T-504), plus three free data.go.kr files that look new to us (a KTX section-by-section statistics
file for koreariders, an hourly Korail commuter-line file for seoulriders, the statistics agency's
grid statistics for trainworld). Details in the list.

**Unverified.** KOSIS table pages redirect scripts to a login page, so no KOSIS table ID below could
be read off its page; the tables are given by their Korean names, which is how KOSIS search finds
them anyway. Statistics Korea was renamed in 2025 (it is now the "National Data Office",
gukgadeiteo-cheo) and its sites moved to `mods.go.kr` addresses; older links may redirect.

## The list

Priority is **must** or **nice**. "Needs ID" means the step can't be done without a Korean account or
identity check.

### Asked of the volunteer

| # | Project | What (Korean name) | Where | Years | Geography | Format | Size (est.) | Why | Needs ID | Priority |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | seoulriders (and trainworld's Seoul checks) | 서울특별시_지하철 역별 OD, one more date | data.go.kr dataset 15113638, its 오류신고 및 문의 button (or the 공공데이터 제공신청 form); provider 서울특별시 교통실 미래첨단교통과, 02-2133-4959 | one weekday: 2024-11-13 (Wed) asked, any non-holiday 2023-2024 weekday accepted (our hourly counts cover 2023-24, so any of those days pairs) | station to station, all 27 lines | CSV, same columns as the published file | 15-20 MB (the Sunday file is 14 MB; a weekday has twice the trips), 4-5 MB zipped | The only weekday OD; replaces the re-levelled Sunday pattern (`build_od.py --day` already reads a dated file). Korean request text ready in `riders/seoulriders/od_request.md` | Yes: data.go.kr signup needs Korean identity verification (seoulriders README, "The account wall") | **must** |
| V2 | koreariders; trainworld intercity checks | 한국철도공사 역간 여객 수송실적 (출발역-도착역별 승차인원, 열차종별) via 공공데이터 제공신청 | data.go.kr > 데이터 요청 > 공공데이터 제공신청 (`data.go.kr/tcs/dor/insertDataOfferReqstProcssView.do`), addressed to 한국철도공사 | 2023 (matches the yearbook koreariders draws) or latest full year | station pair, by train type (KTX, 새마을/ITX, 무궁화) | CSV or XLSX | 1-20 MB, depends on how they cut it | Korea publishes no station-pair or section loads; koreariders reconstructs them. A measured OD would check or replace the solve | Yes: the request form needs a data.go.kr login | nice (high value if granted; Korail may refuse) |
| V3 | trainworld (Korea's commute curve, intercity pull) | 국가교통DB 여객 O/D (전국 지역간 / 수도권 목적·수단 O/D), latest base year | ktdb.go.kr, 자료신청 after membership; KTDB is run by 한국교통연구원 | latest base year (2021 survey era, or newer if out) | about 250 si-gun-gu zones | as provided (often TXT or CSV matrices) | tens to a few hundred MB, unknown | Trips by mode and purpose between districts: the only measured intercity and regional OD for Korea. The register already marks the 2021 survey "G (account)" | Yes: membership and an approved application | nice, **only if its terms allow passing it on** |
| V4 | trainworld (jobs, Korea from archetype C toward A) | SGIS 자료제공: 소지역(읍면동·집계구) 사업체수·종사자수, 산업대분류별 | sgis.mods.go.kr > 자료제공 (login, then 자료신청) | latest (2023 or 2024 전국사업체조사) | eup-myeon-dong or 집계구 | CSV + boundary SHP | 10-100 MB nationwide (est.) | Jobs below si-gun-gu. Only needed if V4's free cousin (O5) has no jobs and KOSIS has no eup-myeon-dong table | Yes: SGIS login is the agency's integrated membership, Korean identity check | nice, only if easy |

Total if all four come: roughly 50-400 MB, almost all of it V3. Without V3: under 50 MB.

### Fetched ourselves (not asked)

| # | Project | What (Korean name) | Where | Years | Geography | Format | Size (est.) | Why | Priority |
|---|---|---|---|---|---|---|---|---|---|
| O1 | trainworld T-504 | 인구총조사 통근·통학 인구: 거주지(시군구) × 통근·통학지(시군구), 교통수단별 if the table has it | KOSIS > 국내통계 > 주제별 > 인구 > 인구총조사 > 인구부문 > 통근·통학 (table ID not verified) | 2020; use 2025 if its sample results are out (2020's came in late Nov 2021, so 2025's are due about now) | si-gun-gu to si-gun-gu | CSV | 5-30 MB | The commute flows to fit Korea's curve, replacing the single mean-commute figure | **must** |
| O2 | trainworld T-504 | 전국사업체조사: 시군구별, 산업별(대분류) 사업체수 및 종사자수; also the 읍면동 table if KOSIS carries one | KOSIS > 국내통계 > 주제별 > 경제 > 전국사업체조사 (table ID not verified) | latest (2023 or 2024) | si-gun-gu (eup-myeon-dong if offered) | CSV | under 5 MB (si-gun-gu); 20-50 MB (eup-myeon-dong) | Jobs by district and sector, replacing tier 0's proxy | **must** |
| O3 | trainworld | 인구총조사 통근·통학 소요시간별 / 교통수단별 인구 (시군구) | KOSIS, same branch as O1 | 2020 (or 2025) | si-gun-gu of residence | CSV | under 5 MB | A check on the fitted curve and on mode shares by district | nice |
| O4 | koreariders | 한국철도공사_KTX 구간별 통계 데이터 | data.go.kr 15119333 (XLSX, "이용허락범위 제한 없음", to Dec 2023) | to 2023 | per line, section and station (columns not seen) | XLSX | unknown, likely small | New to koreariders (no mention in its README): if it really has section volumes, it is the first published per-segment KTX figure | nice, check first |
| O5 | trainworld | 국가데이터처_SGIS 격자 통계 및 경계 | data.go.kr 15141768 (CSV + SHP, free, "이용허락범위 제한 없음") | 2024 | grid (100 m to 1 km; below 1 km totals only) | CSV, SHP | unknown | Sources listed include 전국사업체조사, so it may carry jobs per grid cell, which would make V4 unnecessary | nice, check first |
| O6 | seoulriders | 한국철도공사 광역철도 수송인원 및 승하차인원 (시간대별, 역별) | data.go.kr 15154382 (updated Nov 2025) | recent | station × hour, 경부·경인·경원선 | file | small | seoulriders' README says hourly Korail-line counts need an API key; this file may route round it | nice, check first |

O1-O3: KOSIS downloads in a browser with no login, up to 200,000 cells per download (KOSIS guide,
date unclear). A full si-gun-gu OD (about 250 × 250 × modes) is over that, so download it in parts,
for example one province of residence at a time. Scripts are walled; this is a browser job.

### Considered and left out

- **MDIS 2020 census microdata** (인구주택총조사 20% 표본): see For Anita. The public version is
  province level; the fine version stays in MDIS's remote-access room.
- **Seoul Open Data API key** (OA-12252 hourly counts): keys are personal; seoulriders already routed
  round it with the file archives.
- **T-Data** (`t-data.seoul.go.kr`, raw T-money taps): behind the same Seoul login. What it offers and
  on what terms was not checked. Worth asking about only if V1 fails.
- **Car ownership by small area**: demand reads no car ownership since T-854.
- **Universities and schools**: student counts are open on 대학알리미; not KOSIS-gated.

## Sources looked at

- data.go.kr 15113638 (Seoul station OD: "현재 파일은 샘플 데이터입니다", "데이터 용량 초과 문제로
  인해 자료 요청 시, 전자메일 이용하여 전달드립니다", "원하시는 기간을 명시하여 별도 요청바랍니다";
  licence 공공누리 제1유형).
- data.go.kr 15119333 (KTX 구간별 통계), 15141768 (SGIS grid), 15154382 (광역철도 hourly): dataset
  pages or search snippets only, files not opened.
- MDIS remote-access manual (`mdis.mods.go.kr/data/usrManual_permission.pdf`, contents page) and an
  MDIS introduction: approved-use data is used only through RAS or the data centre; outputs leave
  only after approval (반출). The public-download licence text itself was not read.
- KOSIS guide search snippet: browse up to 20,000 cells, download only from 20,000 to 200,000.
- `riders/seoulriders/README.md` ("There is no weekday OD", "The account wall"),
  `riders/seoulriders/od_request.md`, `riders/koreariders/README.md`, `docs/data-register.md`
  (South Korea rows), `docs/demand-worldwide.md` (South Korea), notes/T-504.md.
