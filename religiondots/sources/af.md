# Afghanistan (`af`)

**Drawn 2026-09-15** by session `cb8b206e-af`. 34 provinces, 34,935,197 settled people, every row
`modelled`, every row `islam`. Reopened from `queue.md`'s closed list (`sources.md` §11ao) on Anita's
priority of 2026-09-15 and her Mauritania ruling of 2026-09-16 (a near-uniformly Muslim country is
drawn on a compiler's figure). Code: `sources/af_geo.py`, `sources/af_grid.py`, `sources/af.py`,
`taxonomy/af2025.py`, `countries/af.py`. `sources.md` §af-2026-09-15. No ask filed.

## 1. Nothing asks religion

- **No census since 1979.** NSIA's own introduction to its 1404 estimates (pdf pp. 3-4, Dari and
  Pashto): the second complete census after 1358 has not been taken, for security and economic
  reasons.
- **Survey of the Afghan People** (Asia Foundation, all 34 provinces): no religion or sect item
  among the demographics; the 2019 report was read end to end (§11ao). It does ask ethnicity. The
  microdata is behind a download form.
- **UNSD Demographic Yearbook table 28:** no Afghanistan row (§11ao).
- **Not checked:** the 2015 Afghanistan DHS and the 2010-11 MICS questionnaires for a religion item.

## 2. Compiler figures, as read

**Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`, unrounded
counts; asserted in `sources/af.py`):

| year | population | Muslims | Christians | Buddhists | unaffiliated | Hindus | Jews | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 28,284,089 | 28,213,147 (99.749%) | 28,348 | 5,653 | 2,447 | 100 | 30 | 34,364 |
| 2020 | 39,068,979 | 39,015,051 (99.862%) | 7,571 | 7,814 | 3,304 | 50 | 10 | 35,179 |

**US State Department, 2023 Report on International Religious Freedom: Afghanistan**
(`state.gov/wp-content/uploads/2024/04/547499_AFGHANISTAN-2023-INTERNATIONAL-RELIGIOUS-FREEDOM-REPORT.pdf`,
22 pages, `%%EOF` present; the HTML page returns 403 to WebFetch, the PDF came with a browser
User-Agent). Section I:

- population 39.2 million (US government, mid-2023);
- World Religion Database 2022: Sunni about 89%, Shia about 11%; Columbia University's Gulf 2000
  project: Shia as high as 29%, Sunni about 70%;
- community leaders: about 90% of Shia are Hazaras, mostly Jaafari, including Ismailis; smaller
  Twelver and Ismaili numbers among Nuristani, Pamiri and other groups; "approximately 25 percent or
  more of Hazaras are Sunni" (some sources);
- "only six Sikh and Hindu individuals remain in the country", living in the Karte Parwan gurdwara;
- Ahmadis "number in the hundreds", largely in Kabul; "Reliable estimates of the Baha'i Faith and
  Christian communities are not available"; "no known Jews";
- Section II: Sikh and Hindu representatives said more than 1,000 were present at the Taliban
  takeover and more than 900 had left.
- Where people live is described by region only (Shia Hazaras in the central and western provinces,
  Kabul and parts of the north); no figure for any province.

**CIA World Factbook:** not read, so not quoted.

## 3. Sect: what exists, and why nothing is drawn or asked

The supervisor's brief: do not draw a Sunni/Shia split by province, and file one ask if a sourced
split exists. **Nothing opened gives a sect figure for any province**, so none is drawn and no ask
was filed.

- **Pew Research Center, *The World's Muslims: Unity and Diversity*** (2012, full report PDF, 164
  pages), p. 30, Afghanistan, % of Muslims: Sunni 90, Shia 7, something else 0, just a Muslim 3,
  nothing/DK 0 (Q31). Methodology page (read 2026-09-15 with a browser User-Agent): "Stratified area
  probability sample of all 34 Afghan provinces (excluding nomadic populations) proportional to
  population size and urban/rural population", face to face, 1,509 interviews, 27 November to 17
  December 2011. No quota is mentioned; the report says Afghan women were hard to reach despite
  gender matching. **The microdata** is Pew's (account) and ARDA's `WDMUS12`; the ARDA pages fetched
  did not show whether a province variable survives, and the file was not downloaded. About 106
  Shia respondents over 34 provinces is about 3 a province.
- **WikiShia** (Persian, *Afghanistan*): Shia 25-30% nationally, citing Bakhtyari, *Shiayan-e
  Afghanistan*, 1385 (2006), p. 7; places named, no province figures. Persian Wikipedia's range (15%
  to 35%) came from a search summary and was not opened.
- **Khwati, *Ismailiyya dar Afghanistan*, Tolou 7 (1382, 2003)**, on ensani.ir: Ismailis 3%
  nationally; lists provinces (Badakhshan the largest concentration), no numbers.
- **Ethnicity is not sect here.** The Survey of the Afghan People's ethnicity by province would give
  Hazaras, not Shia: the State Department's sources put a quarter or more of Hazaras as Sunni, and
  Ismailis are also Tajik and Pamiri. Spec §14.5's religio-ethnic test fails for Hazara as Shia.
- §11ao's line stands: a province map of Shia Afghans is a §14 conversation before it is a build.

Search record, 2026-09-15: English (Pew sect, Asia Foundation methodology, ARDA codebook) and
Persian (`جمعیت شیعیان افغانستان به تفکیک ولایات درصد`).

## 4. The construction, decided

- **Every settled person on `islam`** (Mauritania's construction, `sources/mr.md` §4). Pew 2020's
  53,928 non-Muslims are not drawn: 35,179 of them are `other religions`, and nothing says who or where
  any of them are. Rule 1 (§14): no magnitude a source does not publish.
- **The six Sikhs and Hindus** are not subtracted and not written as a row. Six people draw no dot,
  but a row would draw a presence ring on its unit, Kabul, for a community whose temple there was
  attacked in 2022; with Anita's "no rings" for the Maghreb as the nearest ruling, left out and
  named in `note_public` instead. Reversing it: one `sikhism` or `hinduism` row in `sources/af.py`.
- **Tier `modelled`**, bare `islam` (not `islam.sunni`), as `mr`.
- **Gap:** NSIA's 1,500,000 nomadic Kuchis, held at a fixed national figure since no census could
  count them (NSIA introduction), with no province: 1,500,000 / 36,435,197 = `gap_share` 0.04117.
  `tools/gap_share.py af` refuses (the mapping excludes nothing), so the figure is hand-written.
- **Not checked:** whether NSIA's settled estimate includes the Pakistani refugees in Khost and
  Paktika, or the 2023-25 returnees from Iran and Pakistan. Ask 033's rule would put non-nationals
  nobody counted in `gap`; UNHCR's Refugee Data Finder API (as `sources/mr.md` §8) is where to look.

## 5. Geography and placement

**Population base: NSIA, *Estimated Population of Afghanistan 2025-26*** (1404, September 2025),
`nsia.gov.af:8443/wp-content/uploads/2025/09/براورد-نفوس-کشور-سال-1404.pdf` (the URL is from
English Wikipedia's Parun District citation; the site is an Angular app and its certificate chain
is incomplete, so the fetch does not verify it). 168 pages with a text layer. Projected with
Pt = P0 e^rt from the 1381-1384 (2002-2006) household listing, base year 1383; 34 provinces, 394
original and 29 temporary districts and 34 provincial centres (457 units), 66 cities. Settled
34,935,197 (rural 25,465,725, urban 9,469,472), Kuchi 1,500,000, total 36,435,197. Read from Table 3
(pp. 29-30), checked against the 1402-1404 table (pp. 31-32; 1402 33,471,517, 1403 34,195,527).

- **Chosen over COD-PS** because it is the office's own. COD-PS 2026 (`cod-ps-afg`, "2021
  estimates based on 2017 study conducted by Flowminder/UNFPA") totals 48,595,633, 1.391x NSIA, and
  its per-province ratio over that runs 0.48 (Paktika) to 1.79 (Kunar). It ranks the provinces like
  NSIA (Spearman +0.892; best of 20,000 shuffles +0.639), which is the name join's witness.
- **Boundaries:** COD-AB `cod-ab-afg` version 03 (AGCHO and NSIA), 34 provinces and 401 districts;
  NSIA's English names pinned to pcodes (`Maydanwardag`, `Urozgan`, `Helmand` for Hilmand, `Herat` for
  Hirat).

**Kontur `AF`** (2023-11-01), 145,211 hexes, 42,302,486 people.

- 918 hexes with centroids outside every province; 906 snapped within 2 km, 12 dropped (731 people).
- **1.211x NSIA's settled figure** (band 0.80-1.50, written before the run). Rank witness +0.895
  (best shuffle +0.700). 8.0% of Kontur's people sit in a different province from NSIA.
- **Sar-e Pol reads 0.11** over the national ratio (88,475 against 678,598; median hex 1 person).
  Kept: against COD-PS's district figures only 5.9% of its Kontur people are in a different district
  (Spearman +0.86 over 7), better than the 10.9% NSIA-weighted mean, so calibration raises the level
  and keeps the shape. Next lowest Paktika 0.65 and Ghazni 0.70; highest Khost 1.37, Badghis 1.33.
- **Seats:** no hole. Sar-e Pul town reads 0.28 of GeoNames within 5 km; Bazarak 0.32 within 5 km
  and 0.89 within 10.
- **Kontur's cap, 18 blocks** (reviewed with GeoNames within 5 km and NSIA's urban column):
  - **left:** Kabul city, 187 hexes, 4,583,840 raw, 4,343,503 calibrated (0.81 of NSIA's urban
    Kabul, 5,361,333); Herat city, 41 hexes, 938,391 raw, 675,854 calibrated (0.89 of urban Herat,
    760,907). Capping to the 3 km ring would leave 158,469 and 29,044.
  - **capped (16):** one to five hexes each in rings of 2 to 2,106 people per km2: Gereshk,
    Torghundi and Farsi; Chaghcharan, Shahrak, Tulak and two more in Ghor (NSIA's urban Ghor is
    9,128); three in Daykundi (23.6% of the province's Kontur; no urban population); Panjwayi,
    Faryab, Samangan, Maidan Wardak, Badakhshan. 1,000,903 raw people moved.
  - **The arguable one is Gereshk** (4 hexes, 95,503 raw; GeoNames 43,588): left, it would hold 84%
    of NSIA's urban Helmand while Lashkar Gah is the larger town; capped, its surroundings still hold
    about 79,000 within 10 km.
- The calibrated layer's densest hex is 52,303/km2, above Kontur's limit, so `kontur_cap.apply`
  skips it at scatter time; the raw scan in `af_grid.py` is the only one (as `mr`).
- **Not done:** calibration to NSIA's district tables (Tables 6 onwards), which are in Dari and
  Pashto only and cover 457 units against COD's 401.

## 6. Built

- `check_mapping.py af`: 1 category, 1 node, 34,935,197 on `islam`.
- 1:1,000: 34,935 dots, 0 rings. 1:10,000: 3,493 dots, 0 rings.
- `check_md.py` clean; `coverage.py` ok (before the tail, so vacuous for `af`).

## 7. Reopen if

- A census or survey asks religion or sect with a province. The first thing to open is Pew's 2011
  microdata (ARDA `WDMUS12`) for a province variable; if it has one, that is an ask under §14 with
  what it shows, not a build.
- NSIA's district tables are ever joined (a Dari name join to 457 units).
- UNHCR's figures for non-nationals, for ask 033's `gap`.
