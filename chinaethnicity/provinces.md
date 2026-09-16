# Provinces: what is drawn, and how

Every province is drawn. 16 are **measured**: the province's statistics bureau publishes the
2020 census table **1-4, population by region, sex and nationality**, with a row for every
county, and the map draws those counts. The other 15 are **estimated** by `fallback.py`: the
2000 census county pattern scaled to 2020 totals (NOTES.md §7). The viewer hatches them.

The queue at the bottom is how an estimated province could become measured, or better
estimated, in the order worth trying.

## Measured (16)

| province | source | notes |
|---|---|---|
| Beijing | bureau, xls | numbered 1-6 in Beijing's yearbook |
| Inner Mongolia | bureau via Wayback | live host blocks foreign IPs; 5 development-zone rows |
| Jilin | bureau, xls | development zones; Changbai Mountain area rows |
| Heilongjiang | bureau, xls | |
| Shanghai | bureau, xls | 16 districts |
| Jiangsu | bureau, xls | development zones |
| Zhejiang | bureau via Wayback | Hangzhou's districts were reorganised in 2021 |
| Fujian | bureau, xls | |
| Shandong | bureau, xls (http only) | development zones |
| Henan | bureau, xls | many development-zone and "urban-rural integration zone" rows |
| Hubei | bureau, xlsx in the yearbook zip | development zones |
| Guangxi | bureau, xls | |
| Hainan | bureau, xls | |
| Yunnan | bureau, xlsx in the yearbook rar | |
| Qinghai | bureau via Wayback | live host sends a bot challenge |
| Ningxia | yearbook PDF via Wayback | parsed from the PDF text layer, pages 30-49 |

Source URLs are in `fetch.py`.

## Estimated (15)

| province | scaled to | notes |
|---|---|---|
| Hebei | 2020 prefecture rows | Xinji, Dingzhou and Xiong'an come out as their own units (NOTES.md §7) |
| Liaoning | 2020 prefecture rows | Shenfu New Area's row goes to Fushun, on a weak ASPECT signal |
| Hunan | 2020 prefecture rows | |
| Sichuan | 2020 prefecture rows | |
| Tianjin | province total | bureau file blocked; not in Wayback (queue 4) |
| Shanxi | province total | nothing open |
| Anhui | province total | bureau site blocked; not in Wayback (queue 4) |
| Jiangxi | province total | nothing open |
| Guangdong | province total | nothing open; Shenzhen is drawn far below its real size |
| Chongqing | province total | a short county table is downloaded, not yet used (queue 1) |
| Guizhou | province total | nothing open |
| Tibet | province total | prefecture communiqués could help (queue 2) |
| Shaanxi | province total | host unreachable; not in Wayback (queue 4) |
| Gansu | province total | nothing open |
| Xinjiang | province total | nothing found |

The 2020 prefecture tables are in `fetch.py` `PREFECTURE_SOURCES`. `data/work/fallback_report.txt`
has each province's large scaling factors and how far its drawn county totals sit from ASPECT.

## Queue

### 1. Chongqing: a short county table, downloaded

The 2020 yearbook (`fetch.py chongqing`, 1,636 pages of scans with an OCR text layer) has table
1-4 on PDF pages 44-46, with every district and county but only seven columns, each by sex:
total, Han, Tujia, Miao, Yi, Zhuang, Hui, and "other nationalities" (其他民族, 74,320 people).
Tables 1-4a/b/c (city, town, rural) follow on pages 47-55.

The OCR text cannot be read in order: rows shift where a cell is blank (page 53 puts Fuling's
numbers on Yuzhong's line) and characters are misread ("I" for 1, "九龙坡E"). A parse needs word
positions under the column heads, as Ningxia's does, and the arithmetic checks: total = male +
female, the districts and counties add to the city total, 綦江区 = 綦江区(不含万盛) + 万盛经开区,
and every named column against the national table.

Needs Anita's call first: how to split "other nationalities" across its 52 groups in each county.

### 2. Tibet and Gansu: prefecture communiqués for a few groups

- **Tibet**: each prefecture's 2020 census communiqué gives Tibetan, Han and other counts (seven
  prefectures). religiondots `sources/cn.md` §9e has three of them. They would replace the
  province total for those groups with prefecture totals.
- **Gansu**: Gannan prefecture's 2020 communiqué (`http://www.gnzrmzf.gov.cn/info/2452/70514.htm`)
  may give individual groups; it timed out from here.

### 3. Hebei: an open 2010 county table

`https://tjj.hebei.gov.cn/extra/col20/rkpc2010/html/A0106.htm` would be a newer pattern than
2000's inside each prefecture. Hebei's minorities are few, so this matters less than it sounds.

### 4. Tianjin, Shaanxi, Anhui: published, blocked from outside China

Anita's browser is blocked too (2026-09-14): Anhui answers with a Knownsec region-block page,
and Tianjin and Shaanxi do not respond. Everything tried on 2026-09-14 in the second session
failed:

- **Tianjin**: `https://stats.tj.gov.cn/ztzl_52045/wqhg/rkpchg/zk/html/A0104.xls` returns 403,
  and http redirects to it. Wayback holds the yearbook's `left.htm`, which lists table 1-4 as
  `zk/html/A0104.jpg`, and a few appendix PDFs, but neither A0104 file; its capture of
  `zk/html/` is itself a 403. Save Page Now returns a server error.
- **Shaanxi**: the host does not connect. Wayback holds the page that lists the 2020 volumes
  (`/tjsj/ndsj/pcsj/rkpc/202505/t20250522_3521544.html`, PDFs `P020250522415694808635.pdf`,
  `P020250522415699274008.pdf`, `P020250522415703212765.pdf`, `P020250522415717634203.pdf` in the
  same folder), but not the PDFs. Save Page Now returns a server error.
- **Anhui**: Wayback holds `http://tjj.ah.gov.cn/ssah/qwfbjd/rksj/147987521.html` and four later
  pages, with no links to the files in them.

A mirror inside China, or someone there, is the remaining route.

### 5. Prefecture yearbooks, and Xinjiang's statistical yearbook

- Prefecture-level census yearbooks would give real county rows. Resellers list Xiangxi (Hunan)
  and several Guizhou prefectures; none found open.
- **Guizhou** matters most of the province-total provinces (Miao, Bouyei, Dong, Tujia; about 14
  million people). Reseller: `https://www.tjcn.org/tjnj/24gz/41156.html`.
- **Guangdong**'s bureau says its CD went only to agencies and libraries.
- **Xinjiang**: no 2020 census yearbook was found anywhere. The annual Xinjiang Statistical
  Yearbook is believed to carry a county-by-nationality table, but of registered residents, a
  different number from the census that the map would have to name.
