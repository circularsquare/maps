# United Arab Emirates (`ae`)

**Drawn 2026-10-03** by session `fafd1067-ae`, under a supervisor, on the scout's re-read of the
closed row (`sources.md` §scout-2026-10-03-negatives; earlier §11ao and the 2026-09-14 Gulf re-check).
7 emirates, 11,294,243 people (FCSC 2024), every row `modelled`, roll `NOWHERE`. Rulings:
`ask/RULINGS.md` 2026-09-15 (priority) and 2026-09-16 (Mauritania on a compiler's figure), asks
040/043 (Gulf citizens on one Islam). Code: `sources/ae.py`, `sources/ae_geo.py`,
`sources/ae_grid.py`, `taxonomy/ae2024.py`, `countries/ae.py`. No ask filed.
`sources.md` §ae-2026-10-03.

## 1. Nothing asks, and no counted table was found

Kuwait's PACI answer (religion by nationality group, `sources/kw.md`) was the thing to beat. Searched
2026-10-03:

- **2005 census form**: no religion item (UNSD `ARE2005en.pdf`, read 2026-09-14). UNSD's religion
  table has no UAE row (`tools/oracle.py`).
- **Dubai Statistics Center**: `dsc.gov.ae` rejects scripted requests (F5 "Request Rejected", also
  with browser headers) and WebFetch gets 403. The Wayback CDX of `dsc.gov.ae/Publication/*` lists
  71 population captures (bulletins 2012-2024, the 2005 census's social and demographic
  characteristics, the 2009 yearbook) and `dsc.gov.ae/Report/DSC_SYB_*` the yearbook tables; none
  has religion in its file name. **None could be opened**: from 2026-10-03 about 19:40 UTC the
  Wayback Machine answered every `/web/` request from this host with 429 ("flagged as suspected
  abusive bot traffic"), while CDX still answered; WebFetch cannot reach web.archive.org. The
  2005 census's *Social Characteristics* (`web/20210303084027id_/https://www.dsc.gov.ae/Publication/SocialCharacteristicOfPopulation_2005.pdf`)
  is the one to open first: the federal form had no item, so it is unlikely, not excluded.
  `geostat.dsc.gov.ae/Religious` (§11ao) redirects to a 404 and is probably places of worship.
- **Statistics Centre Abu Dhabi**: `scad.ae` 403; the 2024 release (scad.gov.ae) and the 2023
  census release give total, sex and region only. The census dashboard (§11ao) shows no religion.
- **Sharjah** 2015 and 2022 censuses, **Ajman** 2017, **Ras Al Khaimah** 2023, **Fujairah**: releases
  read through the press and u.ae; none mentions religion. Forms not read.
- **GLMM** (`gulfmigration.grc.net`) search for UAE tables: population by nationality, sex and
  emirate (1975-2005 censuses, estimates to 2016-2017), births by emirate 2022, a 2010-2022
  estimate of Emiratis; no religion table for any GCC state (as §11ao found).
- Web searches (English) for Dubai, Sharjah and Abu Dhabi religion tables: summaries only, every
  one saying no official religion count exists; not evidence either way.

## 2. The population, decided

Every figure below was opened, not taken from a search summary.

| emirate | newest total | 2024 as drawn | newest Emiratis | grown to 2024 |
|---|---|---|---|---|
| Abu Dhabi | 4,135,985 (SCAD, 2024) | 4,135,985 | 551,535 (2016) | 665,515 |
| Dubai | 3,863,600 (DSC, end 2024, via dubai.ae) | 3,863,600 | 233,430 (2016) | 281,670 |
| Sharjah | 1,800,000 (census 2022, as released) | 1,987,285 | 208,000 (2022) | 217,131 |
| Ajman | 504,846 (census 2017, via u.ae) | 557,374 | 39,231 (2005 census) | 66,065 |
| Ras Al Khaimah | 345,000 (2015, RAK media office via u.ae) | 380,896 | 127,000 (2015) | 157,631 |
| Fujairah | 314,829 (mid-2024, via u.ae) | 314,829 | 87,814 (2016) | 105,962 |
| Umm Al Quwain | 49,159 (census 2005) | 54,274 | 17,482 (2010, via u.ae) | 25,253 |

- **National**: FCSC 2024, 11,294,243 (7,235,074 men, 4,059,169 women; Gulf News, opened).
  The four emirates without a 2024 total are scaled together by 1.1040 so the seven close on it.
  One factor for four vintages (2005-2022) understates the recently grown ones; Umm Al Quwain is
  0.5% of the country.
- **Emiratis** grown at GLMM's FCSC-based series (births less deaths from the 2005 census; 2023-2024
  at 2022's 2.17%). They sum to 1,519,227, **1.093x** the series' 1,390,140. Not forced: the series
  is itself an estimate ("accuracy cannot be assessed", GLMM), and Abu Dhabi's own count grew 4.2% a
  year 2005-2016 against the series' 3%. Witness only, unopened: a search summary gives Fujairah
  105,554 Emiratis in 2023 (grown here: 105,962 for 2024).
- **Dubai's figures disagree**: 3,863,600 on dubai.ae (opened), 3,825,000 in Gulf News, and a
  search summary attributes 4,248,200 to DSC's 2024 bulletin (not opened, host blocked).
- Not used: Ras Al Khaimah 2012 (422,000 with 99,522 Emiratis, GLMM), inconsistent with the
  government's 2015 figure; Abu Dhabi's rounded 2024 sexes.

## 3. Religion, decided

- **Emiratis**: all on `islam`; no Sunni/Shia split (asks 040, 043).
- **Non-Emiratis** (9,775,016): one national mix, UN DESA *International Migrant Stock 2024*, UAE
  2024 column, 33 named origins (India 3,248,545, Bangladesh 1,024,949, Pakistan 932,356, Egypt
  841,883, Philippines 528,527), `Others` 248,004 (3.0%) at the named mix. DESA's UAE figures look
  modelled (shares nearly constant across years); they are the only origin split there is.
- **By sex, tried and dropped.** Saudi Arabia's construction was built first: Dubai's non-Emirati men
  as the national residual came out 0.635, below Abu Dhabi's 0.668, which is not believable, and
  DESA's men's and women's mixes give Christians 9.01% and 9.72%. So the split could move almost
  nothing and rested on a fitted residual.
- **Pew 2020 per origin** through `origin_religion.py`, Muslim branches folded. Corrections:
  India's Hindu share **28.63%** (India's row 79.37%) so the layer's Hindus equal Pew's UAE 11.754%
  of the 2024 total (1,327,472), removed Hindus on Islam (Saudi Arabia and Oman's rule); Gulf
  origins (BH, KW, QA, SA, 73,719 in DESA) on `islam`, which removes 17,334 non-Muslims, 15,433 of
  them on Kuwait's row (`taxonomy/ae2024.py` REVIEW). New node `other.ae` (32,684).
- **Witness**: Christians 903,356, 8.0% of the total against Pew's 14.3%, **0.56** (band 0.2-2.0;
  Oman 0.31). By origin: Philippines 597,448, India 88,917, Egypt 50,168, Indonesia 37,842.
  Buddhists 152,171 against Pew's 18,877 (8x; Sri Lanka 96,997), unaffiliated 34,989 against
  37,929, Jews 648 against 1,695. Not corrected. Indians' Christians at India's 2.3% are the likely
  shortfall (Kerala, 18.4% Christian in 2011, sends many of them); the Kerala Migration Survey by
  destination and religion is the fix route, as for Oman.
- **Result**: 77.4% Muslim (Pew 72.9%); non-Muslims 2,556,807: Hindu 1,327,472, Christian 903,356,
  Buddhist 152,171, Sikh 86,923, other.ae 32,684. Christians by emirate 4.9% (Umm Al Quwain) to 8.6%
  (Dubai), which is only each emirate's Emirati share speaking.

## 4. Geography and placement

- **COD-AB ARE ADM1** (HDX `cod-ab-are`, CC BY-IGO, valid 2023-05-15 to 2024-12-19): 7 emirates,
  pcodes AE01-07, exclaves as parts. Joined on pcode; names asserted. Witness: geoBoundaries ARE
  ADM1 (OSM 2017), IoU 0.991 Abu Dhabi to 0.764 Umm Al Quwain, each at least 5x its next-best
  overlap; the northern emirates' inland lines and coasts are traced differently.
- **Kontur AE** (2023-11-01), uncalibrated: 9,521,043 kept, 0.843 of FCSC. Per emirate over the
  national ratio: Abu Dhabi 0.74, Fujairah 0.80, Ajman 0.94, Dubai 1.10, Ras Al Khaimah 1.21,
  Sharjah 1.34, Umm Al Quwain 1.62 (geoBoundaries' polygons give the same within 0.06 except Umm Al
  Quwain 1.90). Rank witness +0.964 (7 of 5,040 orderings reach it). 107 hexes (16,004 people) inside
  Oman's wilayat dropped; 940 within 2 km of an emirate snapped; 17,052 dropped in all.
- **Kontur is flat in the city cores**: densest hex 12,838/km2 (Sharjah); 159,375 people in a box over
  Abu Dhabi island, 245,805 within 5 km of Deira. No count below the emirate was opened to test it
  (the Abu Dhabi census dashboard has districts and was not read), so not calibrated; `kontur_cap.py
  ae`: no stops.
- GeoNames seat check: every seat's GeoNames figure is its metro or emirate, so it is gated on the
  emirate's ratio (none under 0.5), as Oman.

## 5. Reopen if

- The Wayback Machine answers this host again: open DSC's 2005 *Social Characteristics* and the
  2024 bulletin (sex, Emiratis), and the Sharjah 2015/2022 and Ajman 2017 forms for a religion item.
- An emirate publishes Emiratis for 2020 or later (Abu Dhabi and Dubai stopped after 2016 in what
  was found), or a 2024 total for Sharjah, Ajman, Ras Al Khaimah or Umm Al Quwain.
- A count below the emirate (Abu Dhabi census districts, DSC communities) to calibrate Kontur.
- The Kerala Migration Survey's emigrants to the UAE by religion, for Indians' Christian share.
- Anita revisits the Gulf citizens' sect ruling.

## 6. The Gulf rule, 2026-10-03 (`fafd1067-gulf`)

Applied after Bahrain's builder found origin rows invert the Gulf's Christian/Hindu split
(`sources.md` §gulf-2026-10-03 has the rule and the evidence). After §3's India Hindu fit, Indians
are moved from Hindu to Christian until the layer's Christians / (Christians + Hindus) equals Pew
2020's UAE row, 0.549 (the layer gave 0.405). `origin_religion.gulf_christian_hindu`, called from
`ae.py`; it refuses to run if the move is under 1% of the country.

- 321,606 Indians moved; Indians end 20.6% Hindu, 10.2% Christian, 65.9% Muslim (India's row 79.4,
  2.2, 15.2). The non-Muslim total (2,556,807), the Muslim share (77.4%) and every other family are
  unchanged.
- Now drawn: Christians 1,224,962 (10.8%, **0.76** of Pew's share), Hindus 1,005,866 (0.76 of Pew's
  count; §3's "Hindus equal Pew" no longer holds after the move, by design). Per emirate Christians
  6.7% (Umm Al Quwain) to 11.6% (Dubai).
- The added Indian Christians take India's church split (Protestant 58%, Latin 37%, Orthodox 5%),
  which suits Kerala's Syro-Malabar, Latin and Malankara churches badly (Protestants now 389,940).
  Not changed: no opened source gives Gulf Indians' churches. Reopen with §5's Kerala Migration
  Survey item.
- `NOTE` now asserted in `ae.py`; both editions rescattered; `note_public` rewritten for it.

## 7. Review, 2026-10-03 (`fafd1067-rev11`)

Full pass: checks clean, note figures re-summed from `ae_foreign.csv` and `ae.csv` (non-Muslims
2,556,807 include Alevis 8,577 and Druze 1,756, both their own roots, so 77.4% Muslim is right),
map shot fine. One wording fix: "Kuwait, the one Gulf state that records religion by nationality"
contradicted Bahrain's panel on the same map (its census prints religion for Bahrainis and
non-Bahrainis); now "by region of origin", which is what PACI's groups are. Same fix in `om`.
Not reopened: Indians drawn 66% Muslim, which follows from fitting Hindus to Pew before the
Christian/Hindu move (§gulf-2026-10-03 weighed taking Pew's counts instead and declined).
Bahrain's Buddhists are the inconsistency, not these: `sources/bh.md` §5.

## 8. Top text before the 75-word cut, 2026-10-03 (`fafd1067-top75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/ae.py`; `note_public` was not changed.

- `how`: no source asks; Emiratis drawn as Muslim, foreign residents by country of origin
- `grain`: emirates, 1.6 million people on average
- `gap`: Emiratis who are not Muslim, whom no source counts; and the religion of the foreign residents in any one emirate, which nobody has measured
