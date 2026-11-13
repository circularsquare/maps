# Draft message to the volunteer

For Anita to send (edit freely). The research behind it is `research/kosis-request.md`. Two
Korean texts the volunteer can paste into the request forms are at the end; they go with the message.

Fill in before sending: `[your name]`, and whether the Seoul reply should come to you or to them
(see the note under item 1).

---

Hi, and thank you so much for offering to help with Korean data!

It turns out most KOSIS tables I can download myself in the browser, so I'd rather not use your time
on those. What I really can't do from abroad is anything that needs a Korean account. These are the
things I'd love help with, in order. Please only do what's easy for you; the first one matters most.

**1. Seoul subway trips for one weekday (most important)**

Dataset: 서울특별시_지하철 역별 OD, on data.go.kr (데이터 번호 15113638).
https://www.data.go.kr/data/15113638/fileData.do

The file there covers only one day, 2023-12-31, a Sunday. The page says other dates are sent by
email on request (데이터 용량 초과 문제로 인해 자료 요청 시, 전자메일 이용하여 전달드립니다). Could you
ask them for one ordinary weekday, ideally 2024년 11월 13일 (수), or any non-holiday weekday in 2023 or
2024? A ready request in Korean is below (요청 1). You can send it with the 오류신고 및 문의 button on
that page, or through 데이터 요청 > 공공데이터 제공신청. Both need a data.go.kr login.

They reply by email to the address in the request. [Either: "Please put my address, [email], as the
reply address, so you don't have to forward anything." Or: "Please use your own address and forward
the file to me."]

**2. Korail trips between stations (a long shot)**

Korail doesn't publish how many people travel between each pair of stations. Anyone can ask formally
through data.go.kr > 데이터 요청 > 공공데이터 제공신청, and Korail (한국철도공사) has to answer within
about 10 working days. They may say no, which is fine. A request in Korean is below (요청 2).

**3. Optional, only if you already have an account or it's easy**

- 국가교통DB (KTDB, ktdb.go.kr): 여객 O/D, the trips between regions by mode (지역간 / 수도권 목적·수단
  O/D), latest year. **Please check its terms first.** If they say the data is only for the person who
  applied, or can't be given to others, please don't send it. That's completely fine.
- SGIS (sgis.mods.go.kr) > 자료제공: 사업체수·종사자수 by 읍면동 (or 집계구), by 산업대분류, latest year
  (2023 or 2024 전국사업체조사).

I've left out MDIS microdata on purpose, since as far as I can tell its files can't be passed on to
someone else.

**Sending the files**

Item 1 should be about 15-20 MB (about 5 MB zipped), and item 2 probably smaller, so a zipped email
attachment is fine. If anything is bigger than about 20 MB (KTDB might be), a shared Google Drive
folder or a WeTransfer link works well. Whatever is easiest for you. CSV or Excel are both fine,
exactly as they come.

Everything goes into free, non-commercial maps and a free open-source train game, with the source
credited. Thank you again, this is really kind of you!

[your name]

---

## 요청 1: 서울특별시_지하철 역별 OD (paste into the data.go.kr form)

제목: 「서울특별시_지하철 역별 OD」 다른 일자 데이터 요청 (평일 1일)

안녕하십니까.

공공데이터포털에 공개된 「서울특별시_지하철 역별 OD」(데이터 번호 15113638) 데이터를 잘 활용하고
있습니다. 감사합니다.

현재 공개된 파일은 2023년 12월 31일 하루치 자료입니다. 해당 일자는 일요일이어서 출퇴근 시간대의
통행 패턴이 나타나지 않아, 일반적인 평일 하루치 자료를 추가로 요청드리고자 합니다.

데이터 설명에 해당 자료가 "제공요청에 의해" 제공되었다고 안내되어 있어 아래와 같이 문의드립니다.

- 요청 일자: 2024년 11월 13일(수) 1일치
  (해당 일자가 어려울 경우, 2023년 또는 2024년 중 공휴일이 아닌 평일 아무 날짜 하루치라도
  괜찮습니다)
- 요청 항목: 현재 공개된 파일과 동일한 항목
  (기준일자, 승차_호선, 승차_역, 하차_호선, 하차_역, 총_승객수 및 이용자 유형별 승객수)
- 요청 형식: CSV

활용 목적은 비영리 개인 프로젝트입니다. 서울 지하철의 시간대별 승객 흐름을 지도 위에 시각화하는
작업이며, 결과물은 무료로 공개하고 출처를 「서울특별시」로 명확히 표기하겠습니다.

회신 받을 이메일 주소: [회신받을 이메일 주소]

바쁘신 와중에 검토해 주셔서 감사합니다.

## 요청 2: 한국철도공사 역간 여객 수송실적 (공공데이터 제공신청)

제공기관: 한국철도공사

요청 데이터명: 역간(출발역-도착역) 여객 수송실적

요청 내용:
- 2023년(또는 가장 최근 1년) 연간 합계
- 출발역, 도착역, 열차종별(KTX, ITX-새마을·새마을, 무궁화 등), 승차인원
- 일반열차 및 고속열차 전체 (광역전철 제외 가능)
- 형식: CSV 또는 엑셀

활용 목적: 비영리 개인 프로젝트로, 전국 철도 노선별·구간별 승객 흐름을 지도로 시각화하려고
합니다. 현재는 철도통계연보의 역별 승하차 인원으로 구간 통행량을 추정하고 있어, 역간 실적이
있으면 추정 대신 실제 값을 사용할 수 있습니다. 결과물은 무료로 공개하고 출처를 「한국철도공사」로
표기하겠습니다. 개인을 식별할 수 있는 정보는 필요하지 않으며, 역 쌍별 합계만 있으면 됩니다.
