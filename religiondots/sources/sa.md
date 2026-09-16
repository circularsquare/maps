# Saudi Arabia (`sa`)

**Drawn 2026-09-15** by session `cb8b206e-sa`, under a supervisor, reopening the row closed in
`sources.md` §11r, §11af and §11ao. 13 regions, 32,175,224 people, every row `modelled`. Rulings:
`ask/RULINGS.md` 2026-09-15 (priority; the Maghreb reopened) and 2026-09-16 (Mauritania on a
compiler's figure, foreigners by wilaya). Code: `sources/sa.py`, `sources/sa_geo.py`,
`sources/sa_grid.py`, `taxonomy/sa2022.py`, `countries/sa.py`. Ask 040 (Sunni/Shia split). `sources.md`
§sa-2026-09-15.

## 1. Nothing asks religion

- Census 2022: no religion item (§11r). The Arab Barometer's 1,404 wave II rows leave `Q1012` empty and
  wave V has no Saudi rows (§11af). The Ministry of Islamic Affairs is reported to count mosques (§11ao).
- Citizens are legally Muslim; no source counts a citizen who is not.

## 2. The census, and where it can be read

- `portal.saudicensus.sa` does not resolve (2026-09-15); `database.stats.gov.sa` is a bot wall (§11r).
- **GASTAT, *Population Summary Report*** (17 pp, June 2023), Wayback
  `web/20230531131948id_/https://portal.saudicensus.sa/static-assets/media/content/20230531_GASTAT_Population_Report.pdf`,
  393,645 bytes (pinned). 32,175,224; Saudis 18,792,262; non-Saudis 13,382,962 (10,244,464 men,
  3,138,498 women). Figure 3: 34 nationalities (sexes for the top nine). Figure 11: non-Saudi share per
  region, one decimal. Figure 12: males per 100 females by region, overall, Saudi and non-Saudi; the
  bars were read off the rendered page (the text layer holds the values out of order). Page 6 prose:
  non-Saudis from Asia 76.2%, Africa 23.2%, America 0.3%; women 67.5 / 31.4 / 0.5 Europe / 0.5 Americas.
- **GLMM** (`gulfmigration.grc.net`, mirrors the portal with citations): Saudis and non-Saudis by region
  (13 rows; its title says "governorate" and it is regions); Arab (18 countries), non-Arab Asian (by
  world region, all countries), European and Sub-Saharan African nationals by sex; the 34 largest.
  "By region" in two of its titles is the world region of origin, not the Saudi region.
- **RCRC open data** (`opendata.rcrc.gov.sa`, dataset
  `population-by-age-citizenship-gender-and-governorate-2022`): census by citizenship and sex for Ar
  Riyadh region in three parts; every other region is lumped. A witness only.
- **Nationality by Saudi region is not published anywhere found.** Searched: the report, GLMM's
  census module, RCRC's catalogue, `od.data.gov.sa` (times out from here; Wayback holds a
  `population-by-nationality-and-region` dataset page as a 302 only), MHRSD's private sector
  workbook 2021 (workers by nationality national only, establishments by region) and GLMM's MHRSD
  tables (national). No governorate table with Saudis and non-Saudis was found either.

**Checks in `sources/sa.py`:** GLMM's regions equal `sa_geo.REGION_2022` and the report's Figure 11 to
0.045 points in all 13; Figure 12's values are on page 15, and each region's overall bar follows from
its Saudi and non-Saudi bars within 1.5; the four tables' continent shares reproduce the page 6 prose
(76.16, 23.23, 0.31; women 67.51, 31.38, 0.53, 0.58); the 34 largest agree with the four tables. Pinned:
the Arab and Sub-Saharan tables round some rows (Kuwait 50,000 against 50,282; Ethiopia 159,300 against
159,221), and South Asia's `Other` row is one man with blank women and total.

## 3. Compiler figures

- **Pew 2020, Saudi Arabia** (everyone living there, 30,991,207): Muslims 92.694%, Christians 1,356,705
  (4.378%), Hindus 812,626 (2.622%), unaffiliated 34,742, Buddhists 22,774, other 36,581, Jews 851.
- **US State Department, *2023 Report on International Religious Freedom: Saudi Arabia*** (PDF, opened),
  Section I: citizens 85-90% Sunni; Shia 10-12% of citizens and 25-30% of the Eastern Province's
  population. It quotes the World Religion Database's 2.1 million Christians and 708,000 Hindus, not used
  (the project's ruling is Pew for totals).
- **Human Rights Watch**: *Denied Dignity* (2009), Shia 10-15%, Twelvers in the Eastern Province and
  Medina's Nakhawila; *The Ismailis of Najran* (2008), Ismailis "widely believed" a large majority of
  Najran.

## 4. The construction, decided

**Citizens**: all 18,792,262 on `islam`, as Mauritania (Pew's shares include foreigners, so its residual
on citizens would double-count). **No Sunni/Shia split**: ask 040 (spec §14, bombings of Shia and
Ismaili mosques in 2015; only the Eastern Province has a figure, on an unstated base).

**Non-Saudis, per region by sex.** Each region's non-Saudis are split into men and women with Figure
12's non-Saudi ratio (women summed to 3,138,966 against the census's 3,138,498, scaled by 0.99985; the
largest ratio moves 0.09). Riyadh region's men come out 3,139,875 against RCRC's 3,140,295. Each
region's men take the national mix of non-Saudi men and its women the mix of non-Saudi women. The sexes
differ: non-Saudi men 5.29% Christian, women 25.36%. Against one national mix for both sexes, Makkah
gains 31,000 Christians and the Eastern Province loses 20,000 (printed per region by `sa.py`).

**Nationality**: the four GLMM tables' rows by sex; `Other` rows at Pew's regional row; the 42,140 in no
table (the report's "America 0.3%", plus Oceania) at Pew's North America and Latin America rows summed.
Pew 2020 through `taxonomy/origin_religion.py`; Muslim branches folded to `islam`; `other.sa` new
(34,975).

**Two corrections, by name (origin_religion rule 2):**
- **Burma (163,717) on `islam`.** Refugee Law Initiative blog, Charlotte Lysa, 16 June 2023 (opened):
  the Rohingya are "often referred to as 'the Burmese' in Saudi Arabia", arrived on Bangladeshi,
  Pakistani and Indian documents, 250,000 special residency permits in 2017. MHRSD's 2022 private
  sector table (GLMM) lists `Myanmar/ holder of a Pakistani passport` and `...Bangladeshi passport`.
  Pew's Myanmar row would draw 158,240 non-Muslims.
- **India's Hindu share 19.69%, not 79.37%.** Pew, *Faith on the Move* (2012), pp.21-22 (opened): for
  migrants from India to Muslim-majority Middle Eastern destinations Pew used Egypt's census of its
  Indian migrants as a guide, and "most migrants from India to Egypt are Muslims". Pew's national Saudi
  row carries that; its India row does not. So India's share is set so the layer's Hindus equal Pew
  2020's Saudi 2.622% of the census count (843,672); every other nationality's Hindus (472,548) stay at
  their rows; the removed Hindus go to `islam`. On India's own row the layer would draw 1,968,185 Hindus,
  2.3x Pew. **This is fitting one origin to a published national figure**, a call someone might reverse;
  Indian Christians stay at India's row.

**Witness** (`CHRISTIAN_BAND` 0.5-2.0): 1,337,918 Christians, 4.158% of the census against Pew's 4.378%,
**0.95**. The band was written after a rough sum over the largest nationalities while scouting, so it is
not blind. Not corrected, printed only: Buddhists 120,949 against Pew's 23,644 (5.1x; Sri Lankans and
Nepalis at their home rows, no documentation found for either stream), unaffiliated 60,679 against 36,069,
Jews 457 against 884.

**Result**: 92.348% Muslim (Pew 92.694%); 2,461,948 non-Muslims: Latin Catholic 732,520, Hindu 843,672,
Protestant 377,914, Buddhist 120,949, Ethiopian Orthodox 64,742, Coptic 63,847, unaffiliated 60,679, Sikh
40,876, other.sa 34,975. Christians 5.21% of Makkah and 4.92% of Riyadh; 2.35% of Al Bahah.

**The ruling's test** (Algeria: foreigners on a national mix where the likely non-Muslims are small or
predictable): not small (7.7% of the country). Placement rests on the census's non-Saudi count and sex
ratio per region, both published; the unpublished part is the mix within each sex per region. Judged
predictable enough to draw at region grain and said in `note_public`; no ask.

## 5. Geography and placement

- **Boundaries:** COD-AB `cod-ab-sau` v01 (GADM lineage, edited 2015; 13 regions, unchanged since 1993;
  pcodes skip SA13). Polygon area equals COD's `area_sqkm`; every centre point inside its region. No
  GASTAT area table was found, so there is no area witness.
- **Kontur `SA`** (2023-11-01): 36.96M; 1,470 hexes outside every region (170,650 people), 1,302
  snapped within 5 km, 168 dropped (13,411, 0.036%). **1.148x the census**; per region over that,
  Hail 0.89 to Al Bahah 1.29; 1.5% of Kontur's people in a different region. Rank witness +0.989, best of
  20,000 shuffles +0.879. **Not calibrated** (no sub-region table). GeoNames seats of 50,000+: none lost;
  Riyadh reads 0.13 of GeoNames' 4.2M within 5 km and 0.47 within 10 km, which is the city's sprawl.
- **No Kontur block at the density cap** (`python kontur_cap.py sa`: no stops, no rows needed).

## 6. §14

- Shia and Ismaili citizens: ask 040.
- Non-Muslim foreigners may worship only in private (State Department 2023). Drawn at region grain
  (2.5 million people on average) on the census's own foreigner counts and a national mix; nothing
  finer is placed. Not escalated.

## 7. Reopen if

- GASTAT publishes non-Saudis by nationality and region (the census portal's thematic tables, if they
  come back; `od.data.gov.sa`'s `population-by-nationality-and-region` from a client in the region).
- A source gives Sri Lankan, Nepali or Indian migrants' religion in the Gulf (Buddhists 5.1x, §4).
- A governorate table with Saudis and non-Saudis (would calibrate Kontur).
- Anita rules on ask 040.

## 8. Review, 2026-09-15 (cb8b206e-rev9, full pass)

`check_md.py` clean; `built_countries.py --check` has `sa` in both editions; `check_rollup.py sa` all
modelled, nothing orphaned; `gap_share.py --check`: the mapping excludes nothing. Screenshot at the
country's extent: dots follow the cities and the Hejaz and Asir towns, none in the sea. Read
`sa2022.py`, both normalized CSVs and `countries/sa.py` before this file.

**Non-Muslims are drawn inside central Mecca, which they may not enter.** The State Department's 2023
religious freedom report: "The government prohibits non-Muslims from entering central Mecca and the
Prophet's Mosque in Medina" (read on ecoi.net, document 2111948; state.gov returns 403). Every region's
non-Saudis take one national mix and Kontur places them by population, so Makkah region's non-Muslims
fall in Mecca at the regional rate. In `dots_sa.geojson`, 12 of 93 dots within 3 km of the Kaaba are
non-Muslim, 32 of 327 within 6 km and 51 of 564 within 9 km, mostly Hindu, Latin Catholic and
Protestant. This is the one place where the non-Muslim share is known, and it is drawn at the average,
so the error is biased rather than noise; it is the weak point in §4's "predictable enough". The
Prophet's Mosque (2 of 26 dots within 1.5 km) is too small to matter. Suggested fix, not done: zero
place weight for Makkah's non-Muslim rows inside Mecca's haram boundary, or a sub-unit for it, so they
move to Jeddah, Taif and the rest of the region with the region's totals unchanged. Nothing in the
scatter code masks a node by area today. Sent to `queue.md`'s Saudi bullet for a fix batch; no ask.

**Alevis and Druze are drawn, although `sa.py` says no sect is.** `fold()` folds only node ids starting
`islam`, and `alevism` and `druze` are families of their own, so `sa_foreign.csv` has 5,019 Alevis
(Türkiye's row in `origin_religion.py`), 1,918 Druze and 9 Yazidis, about seven dots. Druze on its own
node is the project's convention. Alevis are the open one: Pew counts them among Türkiye's Muslims, and
the stated intent was one Islam for every Muslim nationality. Left for the fix batch.

**The proxy, against `[[feedback_proxy_residual_nameable]]` (spec §3.5b).** Nationality passes the way
England's country of birth did: each nationality's non-matching share is Pew's published row for its
country, and the build weights by it. The mix within a region does not pass on its own, since
nationality by region is unpublished; it stands on the Algeria ruling (2026-09-16), argued in §4, and
Mecca is where it is known to be wrong. India's correction has a documented direction (Pew, *Faith on
the Move*) and a fitted size, which `origin_religion.py` rule 2 ("documented rather than fitted") would
not allow on its own. Agreed as the builder's call, since the target is Pew's Saudi figure, built on the
same assumption. But Hindus are then no longer a check, and the only independent witness left is
Christians at 0.95.

Citizens "Muslim by law" (the note): the same report says applicants for citizenship must attest to
being Muslim and that children of Muslim fathers are deemed Muslim. Close enough.
