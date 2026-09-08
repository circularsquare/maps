# South Korea — KOSIS, Population Census 2015

`sources/kr.py` → `data/normalized/kr.csv`. 49,052,389 people, **229 drawn si/gun/gu**,
10 drawn categories.

| | |
|---|---|
| source | KOSIS table **`DT_1PM1502`**, 성별/연령별/종교별 인구-시군구 |
| org | 101, 국가데이터처 (Ministry of Data and Statistics, formerly KOSTAT) |
| url | `https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1PM1502` |
| geography | si/gun/gu — **229 drawn**, 252 published, 214,000 people each |
| categories | 12, of which 2 are universes |
| basis | `self_id`, on a **20% sample** of a register-based census |
| year | 2015 — **the last census that asked** |
| access | **bot-walled; downloaded by hand.** See §1. |
| encoding | CP949, never UTF-8 |

## 1. The metadata is open and the data is walled

`kosis.kr` answers normally and its 404 is a real router (1,058 bytes, distinct from every
other response), so this is not a site that is down or hostile in general.
**`/statHtml/statHtmlContent.do` returns 291 KB of the table's complete metadata with no
login** — the geography dimension's 327 items split 21/306, the 12 category ids and labels,
the period, the unit, `downloadable: true`. Every fact in the table above came from there.

Every DATA endpoint refuses:

| endpoint | what it is | response |
|---|---|---|
| `/statHtml/html.do` | the grid's own JSON | 200, HTML `alert()` |
| `/statHtml/downNormal.do` | download | 200, HTML `alert()` |
| `/statHtml/downGrid.do` | download | 200, HTML `alert()` |
| `/statHtml/downLarge.do` | download | 200, HTML `alert()` |

> `비정상적인 서비스 이용으로 접근이 차단되었습니다.` — *access blocked, abnormal service use*

That is bot protection, and `spec` §12 says a stop sign rather than a puzzle. It also
**spread**: the download endpoints refused first and `html.do` refused afterwards, so it is
reactive rather than a static rule, and probing harder makes it worse.

**Anita downloaded the file through a browser on 2026-09-05.** The click path, for next time:
open the table, set `행정구역별(시군구)` to its second level so all 306 units are selected,
leave all 12 items ticked, and download CSV. `sources.md` §11g has the same in more detail.
A KOSIS OpenAPI key would make this scriptable and **whether that registration is possible
from outside Korea is still unchecked** — it is the one thing worth trying.

**HTTP 200 with an HTML `alert()` is `sources.md` §5a again**, and worth naming as its own
disguise: not a truncated file, not an SPA shell, not a PNG of an error — a *success* status
carrying a JavaScript dialog. Anything that checks only `raise_for_status()` treats it as a
download.

## 2. Two nested tiers in one column, and neither is marked

### 45 rows are not places

`동부`, `읍부` and `면부` — urban, town, rural — appear once nationally and again inside most
provinces, in the same column as the districts and with no flag. They are a **cross-cutting
partition** of the same people, so counting them as units double-counts the country.

They are dropped, and asserted to sum to their province on all 12 categories. That assertion
is the point: it is what proves they are a universe rather than places being discarded by
mistake, and it is the same discipline Serbia's empty Kosovo row needed.

### 12 cities appear twice, as themselves and as their own gu

Suwon, Seongnam, Anyang, Bucheon, Ansan, Goyang, Yongin, Cheongju, Cheonan, Jeonju, Pohang
and Changwon are 특정시 subdivided into 구. KOSIS lists **both tiers in one column, same
indent, same column, no marker**. Summing the level as delivered overstates Gyeonggi alone by
5,997,676.

This is `spec` §12's **Serbia** case exactly — an extra level *inside* the drawn tier — and it
is found the same way: a parent's children are the consecutive rows whose totals sum to it
exactly. Verified afterwards **across all twelve categories, not just the total**, which is
what turns a plausible grouping into a proof.

### Which tier gets drawn is decided by the boundaries, not the source

| tier | units | what it is |
|---|---:|---|
| KOSIS level 2 | 252 | plain units + the 35 gu, city parents removed |
| **drawn** | **229** | plain units + the 12 cities, their gu removed |

geoBoundaries KOR ADM2 is 228 polygons and carries Suwon, Changwon and the rest **whole**;
no boundary set anywhere carries the 일반구. Drawing the 252 would mean 35 units with no
polygon. So the 229 is drawn and **the 35 gu are emitted at their own `gu` level, undrawn**,
so nothing published is thrown away (§2.4) and it is a lookup if a 일반구 file ever appears.

Seoul's 25 gu are unaffected — those are 자치구, full local governments, and ADM2 has them.

**Both tiers are asserted to partition every province independently.** Two complete partitions
of the same people, and checking only one would not notice the other going wrong.

## 3. No codes, and Korean district names collide worse than anywhere else

The download carries names only. Nationally:

```
동구 x6   중구 x6   서구 x5   남구 x5   북구 x4   강서구 x2   고성군 x2
```

A national name join would pair one metropolis's Jung-gu with another's and **every
provincial and national total would still reconcile** — `sources.md` §9n's Ghana `TMA`
failure, with more collisions than Ghana had. Names *are* unique inside a province, which
`kr.py` asserts, and row order gives the province, so every row carries its parent and
`sources/kr_geo.py` matches within it.

The code option was requested in the download and did not come through. It is not worth a
second manual export: the within-province join is checkable and checked.

## 4. The categories

| category | people | share | node |
|---|---:|---:|---|
| 종교없음-계 no religion | 27,498,715 | 56.06% | `unaffiliated` |
| 기독교(개신교) Protestant | 9,675,761 | 19.73% | `christianity.protestant` |
| 불교 Buddhist | 7,619,332 | 15.53% | `buddhism` |
| 기독교(천주교) Catholic | 3,890,311 | 7.93% | `christianity.catholic` |
| 기타 other | 98,185 | 0.20% | `other.kr` |
| 원불교 Won Buddhism | 84,141 | 0.17% | `buddhism.won` |
| 유교 Confucianism | 75,703 | 0.15% | `confucianism` |
| 천도교 Cheondogyo | 65,964 | 0.13% | `eastasiannew.korean.cheondogyo` |
| 대순진리회 Daesun Jinrihoe | 41,176 | 0.08% | `eastasiannew.korean.daesun` |
| 대종교 Daejonggyo | 3,101 | 0.01% | `eastasiannew.korean.daejong` |

Excluded: `계` (the total) and `종교있음-계` (has-a-religion, an intermediate universe of
21,553,674). Both identities — the nine religions summing to `종교있음-계`, and that plus
`종교없음-계` summing to `계` — hold on all 327 rows, which is also **what proves an empty
cell means zero** rather than suppressed.

**Five of these ten needed new nodes and four are religions no other source on this map
counts at all.** See `taxonomy/kr2015.py`; Korea is why `japanesenew` became `eastasiannew`
with Japanese and Korean children, and why `confucianism` exists.

## 5. Two caveats that are the source's, not the pipeline's

**The figures are a 20% sample grossed up.** 2015 was a register-based census (등록센서스) and
religion rode on the sample survey rather than the register. Every cell carries sampling
error, and it bites hardest exactly where this table is most interesting: Daejonggyo is 3,101
people nationally, so its district cells are a handful of sampled households each. Small
categories at small units are indicative, not measurements.

**The universe is 49,052,389 against a census population of 51,069,375** — a gap of 2,016,986,
3.95%, that neither the table nor its metadata explains. Candidate explanations were
considered and none of them fits cleanly: the resident-Korean population in 2015 was
49,705,663, which is closer but still 653,274 away, and the foreign-resident population is
the wrong size to account for the rest. **It is left unexplained and reported rather than
absorbed** (§3.5). Anyone who finds the definition should write it here.

## 6. Not done

- **A KOSIS OpenAPI key**, which would make this repeatable. Whether registration works from
  outside Korea is unchecked; `data.go.kr` and `data.seoul.go.kr` both gate signup behind
  Korean identity verification, and KOSIS may or may not.
- **The 35 general gu**, which are in `kr.csv` and undrawn for want of boundaries.
- **Denominations.** Korean Protestantism is majority Presbyterian, with the Hapdong and
  Tonghap assemblies each larger than most European national churches, and the census offers
  one Protestant box. The tree could hold them apart; nothing in KOSIS reaches them.
- **Anything after 2015.** The question was dropped. This is the final measurement.
