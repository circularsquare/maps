# Syria (`sy`)

Drawn 2026-10-03 by `fafd1067-sy`: Pew Research Center's 2020 national mix (the World Religion
Database's figure) at 14 governorates, on the Central Bureau of Statistics' end-2011 estimate,
placed on Kontur. Every row `modelled`. Built on Anita's rulings of 2026-09-15 (a §14 case files an
ask and carries on) and 2026-09-16 (a compiler's figure where nothing asks). Ask 050 holds the
placement of the Druze, Alawites and Christians. Code: `sources/sy_geo.py`, `sources/sy_grid.py`,
`sources/sy.py`, `taxonomy/sy2020.py`, `countries/sy.py`.

## 1. Nothing asks

- **Census.** The 2004 form has no religion item (`sources.md` §scout-2026-09-15-negatives). The
  1947 and 1960 censuses counted sects: Friedman, "The enigma of the number of ʿAlawis in Syria: 11%
  indeed?", *British Journal of Middle Eastern Studies* 53:1 (2024), pp.17-36, gives the 1947 census
  as 339,466 Alawis of 3,043,310 (11.15%) and the CBS's 1960s figures as 11.18-11.47% (search summary;
  the article answered 403 and was not read). Not used: Anita ruled Razmara's c.1950 Iranian
  gazetteer "too old to use" (2026-09-14), and a 1960 table is the same age and a §14 object.
  Not checked: whether either census printed sect by muhafaza.
- **Arab Barometer wave IX** (fieldwork 29 October-17 November 2025, 1,229 citizens, all 14
  governorates): the downloads page on 2026-10-03 still offers waves I-VIII only, with Syria's
  technical report. The factsheet on the Foreign Affairs piece prints no religious composition, only
  that Latakia and Tartus hold large Alawite shares and Suwayda the largest Druze share. REOPEN on
  the release; whether its sect item may place anyone is §14 and Anita's.

## 2. The level: Pew 2020, which is the World Religion Database

`data/raw/estimates/pew.zip`, unrounded counts, Syria 2020: 21,049,429 people; Muslims 19,821,403
(94.166%), Christians 808,455 (3.841%), unaffiliated 416,788 (1.980%), Hindus 2,105, other religions
559, Jews 118, Buddhists 0. The families sum to one less than the population. 2010: Muslims 89.46%,
Christians 8.59%, unaffiliated 1.93%.

Pew's Appendix A (`PR_2025.06.09_global-religious-change_appendix-a.pdf`, p.24, read 2026-10-03)
sources Syria's 2010 and 2020 composition to the **World Religion Database**, with UN WPP 2024 for
the population, and "Data unavailable" for switching. So the figure is a compiler's ascription, not
a survey. It is still what the national estimate layer draws (spec §15), and drawing the same figure
here keeps the two from disagreeing (Anita, 2026-09-09).

## 3. The Druze, and the sects

- Pew's `Other_religions` for Syria is 559 people (0.003%); its Lebanon row, from Pew's own surveys,
  carries 4.3% there, which is the Druze. The World Religion Database files Syria's Druze with the
  Muslims, so the map does too, and the note says so ("many Druze do not consider their faith part of
  Islam").
- A national Druze share was considered and rejected: it would draw Druze at the same rate in Deir
  ez-Zor as in Suwayda. The only honest Druze layer places them, which is ask 050.
- Alawites, Ismailis and Twelver Shia stay on bare `islam`: the Oman and Saudi ruling (2026-09-15),
  no sect split without placement.

## 4. Population base: CBS end-2011, chosen over everything current

| candidate | what it is | why not |
|---|---|---|
| OCHA Population Task Force baseline, Aug 2025, admin 1-4 (`syrian-arab-republic-baseline-population`) | current, IDPs where they are | HDX: "confidential ... cannot be shared for academic or research purposes" |
| HNO 2024 and 2025, HNRP 2026, JIAF 2025 (HDX) | people in need by sub-district | no total population column |
| US Census Bureau `syria_uscb_201811.xlsx` | 2004 census, 2011, 2014, 2016 estimates | wartime 2014-2016 reports; no later |
| Kontur SY 2023, WorldPop, GHSL | modelled surfaces | built on census-era admin totals: Kontur sits within 0.86-1.24 of CBS 2011 per governorate (Idlib 1.02), so it does not see displacement |
| **CBS, Statistical Abstract 2012, Table 3/2** | people actually living in Syria by governorate, 31/12/2011, thousands, excludes Syrians abroad; 21,377 | **used**: the state's last pre-war estimate, within 2% of Pew's 2020 total |

OCHA Syria republishes the table on HDX (`syrian-arab-republic-other-0`, `syr_pop_2011.xls`); the
same workbook has the civil-register count (24,504 thousand on 1/1/2011, which includes Syrians
abroad) and UNRWA's registered Palestinians by governorate (483,021 at end-2010), neither used.
Whether Table 3/2 includes Palestinians is not stated; they are inside Pew's national mix either way.

So the dots are pre-war positions. The note says so and gives UNHCR's regional figure (31 August
2026: 1.79 million registered by UNHCR in Egypt, Iraq, Jordan and Lebanon, 2.87 million registered by
Türkiye, more than 43,000 in North Africa; `data.unhcr.org/en/situations/syria`). No IDP figure was
opened, so the note gives none.

## 5. Geography and placement

- **COD-AB** `cod-ab-syr` v02, `syr_admin1.geojson`, 14 governorates SY01-SY14. CBS's table is in
  the same order; the join is by row, with English (four respellings pinned) and Arabic as witnesses.
- **The Golan** is drawn with Israel (`sources/il_geo.py`). COD's Quneitra includes it; Natural
  Earth's `ne_10m_admin_0_disputed_areas` "Golan Heights" (Admin. by Israel) is subtracted, 1,093 km2
  (1,718 -> 625). CBS's Quneitra (90,000) is the Syrian-held part. The UNDOF zone stays.
- **Kontur SY 20231101**: 75,619 hexes, 23.29 million. Hexes with a centroid in the Golan polygon
  are dropped before any snap (5 hexes, 43 people; Majdal Shams is about a kilometre from the line),
  1,297 border hexes snapped within 2 km, 26 dropped (882 people). Ratio to CBS 1.089; rank witness
  +0.982, no shuffle in 20,000 reaches it. Seat check: no hole (Ar Raqqah 0.45 and Al-Hasakah 0.54
  within 5 km are GeoNames figures above Kontur's, with their governorates at 0.93 and 0.96).
- **Cap blocks**: three, each with one hex at the cap, all registered `real` in `kontur_cap.csv`:
  Mezzeh and Kafr Sousa (10.3% of Damascus), Barzeh and Qaboun (7.5%), Yarmouk and Al-Hajar al-Aswad
  (4.5% of Rural Damascus; the pre-war camp and its neighbour were among Syria's densest districts,
  and the base is 2011).

## 6. As drawn

21,374,175 people: Muslim 20,129,864, Christian 821,038, no religion 423,273, the same 94.18 / 3.84
/ 1.98 in every governorate. 2,825 people (Pew's 0.0132% tail) are the gap. 21,373 dots at 1:1,000,
2,136 at 1:10,000, no rings.

## 7. What would make it better

- Arab Barometer IX's release (§1), under §14 first.
- A placement for the Druze (Suwayda), Christians and Alawites. Fabrice Balanche's 2010 estimates
  (Lyon 2; *Sectarianism in Syria's Civil War*, Washington Institute, 2018) give community shares for
  some governorates and cities, e.g. Suwayda governorate 90% Druze, 7% Christian, 3% Sunni of about
  375,000 (as quoted on Wikipedia, not read at source). A geographer's estimate published by a think
  tank, not a count. Ask 050.
- A current population base the UN allows for research.

## 8. Review, 2026-10-03 (`fafd1067-rev8`)

Full pass. Pew's Syria rows re-read from `pew.zip` (2020: 94.166 / 3.841 / 1.980; 2010 Christians
8.59%), and `sy.csv` is one mix in all 14 governorates summing to the note's figures. Mapping, gap and
`gap_share` hold; screenshot shows dots on the towns and the Euphrates, none on the Badia. Ask 050 left
as filed. One fix: the note said the dots "are drawn desaturated"; nothing desaturates any dot since
spec §7 removed it on 2026-09-04 (the shaders only hide modelled dots under `inferred dots: not shown`),
so the sentence now says they disappear when inferred dots are turned off. `tiles.py --refresh-meta` run.
