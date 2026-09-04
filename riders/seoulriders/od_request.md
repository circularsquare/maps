# Asking Seoul for an ordinary weekday

**This is now an upgrade, not a blocker.** We have a complete OD file and
matching hourly counts already — see `README.md`. The only problem with what we
have is *which day it is*.

## The situation

[서울특별시_지하철 역별 OD](https://www.data.go.kr/data/15113638/fileData.do) is
labelled "샘플 데이터", but that label is wrong. The published file is the real
thing: 229,365 rows, 27 lines, 642 stations, the whole 수도권 network.

It is a single date, though — **2023-12-31, a Sunday, and New Year's Eve.** The
numbers show it plainly: busiest origins are 잠실, 홍대입구, 명동 and 혜화, 강남
only ranks tenth, and the day totals 3.46M trips against a normal weekday's ~7M.
It is a night-out map, not a commute map.

Their own description says that date was produced 제공요청에 의해 — by request —
so producing another date is demonstrably something they do.

**We can pair any date they give us.** `data/hourly_2023_raw.csv` and
`data/daily_hourly_raw.csv` between them hold hourly boardings and alightings
for every single day of 2023 and 2024. So the request is purely "the same file,
a different date", with no follow-up needed.

**Suggested date: 2024년 11월 13일 (수).** A Wednesday, no Korean public holidays
that month, and clear of 수능 on 11/14 which distorts subway service. Worth
asking flexibly, since we do not know which dates they can produce.

## How to send it

Use the **오류신고 및 문의** button on the dataset page, which attaches to this
dataset and routes to the providing department. It needs a free 공공데이터포털
account.

**Note the account wall.** data.seoul.go.kr signup requires Korean identity
verification (i-PIN with an 외국인등록번호, a Korean carrier phone, or a Korean
bank certificate) and is not passable from abroad. data.go.kr may or may not be
the same — if it is, the remaining route is the phone number below, or simply
staying with New Year's Eve, which costs us nothing technically.

**Provider:** 서울특별시 교통실 미래첨단교통과 — tel 02-2133-4959. No public
email address; the page directs data questions to that number.

---

## Draft message

제목: 「서울특별시_지하철 역별 OD」 다른 일자 데이터 요청 (평일 1일)

안녕하십니까.

공공데이터포털에 공개된 「서울특별시_지하철 역별 OD」(데이터 번호 15113638)
데이터를 잘 활용하고 있습니다. 감사합니다.

현재 공개된 파일은 2023년 12월 31일 하루치 자료입니다. 해당 일자는 일요일이자
연말이어서 평상시의 통행 패턴과는 차이가 있어, 일반적인 평일 하루치 자료를
추가로 요청드리고자 합니다.

데이터 설명에 해당 자료가 "제공요청에 의해" 제공되었다고 안내되어 있어
아래와 같이 문의드립니다.

- 요청 일자: 2024년 11월 13일(수) 1일치
  (해당 일자가 어려울 경우, 2023년 또는 2024년 중 공휴일이 아닌
   평일 아무 날짜 하루치라도 괜찮습니다)
- 요청 항목: 현재 공개된 파일과 동일한 항목
  (기준일자, 승차_호선, 승차_역, 하차_호선, 하차_역, 총_승객수 및
   이용자 유형별 승객수)
- 요청 형식: CSV

활용 목적은 비영리 개인 프로젝트입니다. 서울 지하철의 시간대별 승객 흐름을
지도 위에 시각화하는 작업이며, 결과물은 무료로 공개하고 출처를
「서울특별시」로 명확히 표기하겠습니다.

회신 받을 이메일 주소: [회신받을 이메일 주소]

바쁘신 와중에 검토해 주셔서 감사합니다.

감사합니다.

---

## What the draft says, in English

Thanks them, notes the published file is 2023-12-31, and explains why that
particular day is awkward — a Sunday and year-end, so not representative.
Points out their own description says the file was produced on request, then
asks for one ordinary weekday in the same format, naming Wednesday 2024-11-13
but explicitly accepting any non-holiday weekday in 2023 or 2024. States the
purpose plainly: a non-commercial personal project, published free, crediting
서울특별시. Ends with a line for the reply address.

Deliberately asks for **one day, not a month** — a smaller ask than the file
they already publish, which makes it easy to say yes to.

Fill in the email address before sending.
