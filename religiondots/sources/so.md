# Somalia (`so`)

Session `cb8b206e-so`, 2026-09-15, under a supervisor. Reopened from `queue.md`'s closed row ("surveys
ask none", Africa swept a second time, 2026-09-14; `sources.md` §11aq) on Anita's priority of
2026-09-15 and her Maghreb and Mauritania rulings (`ask/RULINGS.md` 2026-09-15 and 2026-09-16). Code:
`sources/so_geo.py`, `sources/so_grid.py`, `sources/so.py`, `taxonomy/so2026.py`, `countries/so.py`.
`sources.md` §so-2026-09-15. No ask filed.

## 0. Outcome

Drawn at 18 regions, **19,442,160 people** (the 2026 humanitarian planning estimate, COD-PS), every
one on `islam`, every row `modelled`. Somaliland's five regions are inside. Foreign residents and
refugees are in `gap` (`gap_share` 0.00399). Nobody is placed on any other node.

## 1. Nothing asks

- `sources.md` §11aq (2026-09-14): PESS 2014 (report and 145 variables), SHDS 2020 (five files, 1,495
  variables), the Somali High Frequency Surveys 2016 and 2017 and the Somaliland Household Survey 2012
  have no religion item. Somalia is in no Afrobarometer round (§11aq's table). Not re-checked here.
- **Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), Somalia 2020,
  everyone living there: 16,651,191; Muslims 16,623,449 (99.833%); Christians 4,367; unaffiliated 2,684;
  Hindus 6,660; other religions 14,030. 2010: 99.807%.
- **US State Department, *2023 Report on International Religious Freedom: Somalia***
  (`state.gov/reports/2023-report-on-international-religious-freedom/somalia/`, opened 2026-09-15 with a
  browser user agent, HTTP 200): "According to the Federal Ministry of Endowments and Religious
  Affairs, more than 99 percent of the population are Sunni Muslim. According to the online reference
  World Atlas, ... a small Christian community of approximately 1,000". Also: al-Shabaab "threatened to
  execute anyone suspected of converting to Christianity"; the only non-Muslim place of worship is in
  the international airport compound; a Somaliland resident was imprisoned in 2022 for apostasy and
  "spreading Christianity"; Somaliland's constitution prohibits Muslims from converting.

## 2. Rulings and §14

- 2026-09-15 (priority): Somalia is one of the holes to fill first.
- 2026-09-15 and 2026-09-16 (Maghreb, Mauritania): a near-uniformly Muslim country is drawn on a
  compiler's figure; foreigners drawn where a count by province and a nationality mix allow, otherwise
  sized in `gap` (ask 033).
- **§14, not raised as an ask.** Converts have been threatened and killed (above). Nothing below the
  nation is placed for anyone but Muslims, and no subnational placement of Christians is on the table:
  no source says where they live. The note names the national figures only. The supervisor's brief
  said to keep them national or in the gap.

## 3. The construction

Mauritania's (`sources/mr.md` §4), taken whole: every person on `islam`, not Pew's residual. Pew's
non-Muslims (27,741) are mostly `other religions` and Hindus, cells nothing explains, and Pew's base
includes foreigners. Unlike Mauritania there is no foreigner layer to draw non-Muslims from, so the
map has no non-Muslim dot at all. Reversing it: a national residual in `sources/so.py`, rerun, scatter.

## 4. Units and population

**Boundaries.** COD-AB `cod-ab-som` v03 (OCHA; valid 2025-01-08, reviewed 2025-10-30, boundaries of
1984-06-23), `som_admin1.geojson` 18 regions SO11-SO28, `som_admin2.geojson` 91 districts (90 plus a
142.6 km2 `Unspecified` in Banadir).

**Population: the 2026 planning estimate.** COD-PS `cod-ps-som`, *Somalia District-level population
estimates for humanitarian response planning, 2026* (version 2026.V1, released 2026-05-25), 90
districts, 19,442,160 (the Total row equals the sum). Its read-me: the national figure "was shared by
the Government through UNFPA" and is "the official population figure"; district figures are
"planning estimates derived through remodelling of previous sub-national population data" and "not
official district population statistics". Joined on p-code; every district's own Region column agrees
with COD-AB's region for its p-code (asserted). The workbook swaps two pairs of Banadir district names
against COD-AB (SO2201/SO2202, SO2208/SO2209; pinned, region sums unaffected). The 2025 working file
(`CODs_working_file_Population_Data_v7.2`, 19,280,850, 91 rows with Gaalkacyo North and South on one
p-code) was read and not used.

**Why the 2026 figure over PESS 2014.** PESS 2014 (UNFPA, 12,327,528; region shares the survey
measured, districts interpolated from 2005 UNDP data per the COD-PS notes) is twelve years old. Kontur
2023 is no closer to either: half the summed absolute difference in regional shares is 0.161 against
PESS and 0.154 against 2026 (Spearman 0.77 and 0.82). Share ratios 2026/PESS run from 0.52 (Sanaag) and
0.62 (Awdal) to 1.35 (Mudug) and 1.56 (Lower Juba). **Why region and not district:** the district
figures are planning estimates, and Kontur disagrees with them far more (half-L1 0.249; Banadir's
districts 0.23x to 4.93x).

**Somaliland.** Awdal, Woqooyi Galbeed, Togdheer, Sool and Sanaag (4,077,761, 21.0%; the workbook's
`State` column says `Somaliland` for all five) are drawn inside `so`, named "Somalia". The reasons:
every population figure here (the 2026 estimate, PESS 2014, COD-AB) counts them as Somalia's; religion
is the same either side; spec §14.18's de facto rule settles which country's source covers land two
sources both count, and here only one does. Natural Earth draws Somaliland apart (`SOL`, "Self admin.;
Claimed by Somalia"), so `country_shapes.py` `ALSO` adds it to Somalia's outline; without that its dots
would sit outside any outline. Reversing it: a separate entry (an `x?` code, spec's Kosovo pattern)
built from the same five regions, and `SOL` out of `ALSO`. Not checked: where Somaliland's control ends
in Sool and Sanaag, which the regions do not follow anyway.

## 5. Placement

`so_grid.py`. Kontur `SO` 20231101, 97,713 hexes, 18,234,637 people. Kontur over the estimate 0.932;
rank witness over 18 regions +0.802, 2 of 20,000 shuffles reach it (bar 0.001). Per region over the
national ratio: Nugaal 0.37, Mudug 0.49, Sool 0.50 up to Woqooyi Galbeed 1.38, Lower Shabelle 1.62.
Placement is inside each region only, so these move no count. Seat check: 11 GeoNames seats of 50,000
or more, no hole (Hargeysa and Burao hold 3-4x GeoNames' old figures).

**Hexes outside every COD-AB region: 1,216, 654,684 people.** Measured by distance and by Natural
Earth country, then by whether Ethiopia's or Kenya's place layer (`et_hexes.gpkg`, `ke_hexes.gpkg`)
already holds the same hex (centroid within 10 m). Kontur's extracts overlap at borders: SO shares 457
h3 cells with ET and 273 with KE.

| rule | people |
|---|---:|
| snap, within 2 km, in no neighbour's layer | 370,852 |
| snap far, 2-10 km, inside NE Somalia/Somaliland or sea, in no neighbour's layer | 162,981 |
| drop, already in `et_hexes` | 84,082 |
| drop, already in `ke_hexes` | 28,345 |
| drop, neither (NE Ethiopia beyond 2 km) | 8,424 |

The far snap is Cabudwaaq: COD-AB's 1984 line leaves the Galmudug border town 2-7 km outside
Galgaduud, and no neighbour's layer holds it, so dropping it drew about 115,000 people nowhere. Witness:
124,326 Kontur people within 5 km of GeoNames' Cabudwaaq, in SO19. The Ethiopian drops include 54,490
people Natural Earth puts in Somaliland (Wajaale, where COD-AB and Ethiopia's layer disagree by under a
kilometre): they are drawn once, by Ethiopia. Kenya's are Beled Hawo and Mandera. Pinned in
`OUTSIDE_PINNED`.

**Kontur's cap: 28 blocks, 12 real and 16 capped, none unreviewed** (`kontur_cap.csv`; Sudan's ring-less
trap does not arise: every block has 3 or more populated hexes within 3 km). The rule was written
before the numbers: real where a GeoNames place within 5 km holds a third of the block, or the block is
within 15 km of Mogadishu, and the block holds at most 1.5x the 2026 estimate of the districts it
touches; capped otherwise. One slip corrected: the first pass compared Mogadishu's block (106 hexes,
3,072,354) with the one district its peak falls in (Heliwa, 110,961); the rule means the districts the
block covers (16, 2,668,138, 1.15x).

- **Real:** Mogadishu, Hargeysa, Borama, Kismayo, Jilib, Ceerigaabo, Jamaame, Marka, Qoryooley, and
  three blocks in the Afgooye corridor 7-11 km west of Mogadishu.
- **Capped:** Ceel Afweyn, Jawhar (154,128 against GeoNames' 47,086), Xuddur, Badhan, Wanlaweyn
  (233,253 against 22,022), Buurhakaba, Xarardheere, Waajid, Dujuuma, Balcad, Bulo Marer, Bur Salah,
  Miisra, Faarax Gololey, Baraawe (66,205, 0.98x its whole district), Shalaamboot.
- GeoNames gives many Somali towns no population (Ceel Afweyn, Badhan, Balcad, Baraawe), which the rule
  reads as capped. The Yemen trap (`playbooks/geography.md`, a town list too thin) applies; the
  district comparison is the second test.

## 6. Foreign residents and refugees: in `gap`

No source counts foreign residents by region, so none is drawn (ask 033). **UN DESA, *International
Migrant Stock 2024*** (`data/raw/mr/undesa_pd_2024_...xlsx`), Somalia as destination, mid-2024:
**77,972**, data type `I R` (imputed, with UNHCR's refugees added: there is no census to work from);
Ethiopia 28,964, Yemen 17,680, Eritrea 48, others 31,280. **UNHCR** Refugee Data Finder API
(`year=2024&coa=SOM&coo_all=true&cf_type=ISO`, read 2026-09-15): 19,503 refugees and 22,260 asylum
seekers, **41,763**; Ethiopians 27,041, Yemenis 12,332, Syrians 1,769. End 2025: 46,669.
`gap_share` = 77,972 / (19,442,160 + 77,972) = **0.00399**, an upper bound by whatever part of them the
planning estimate already holds (Sudan's construction). `tools/gap_share.py` refuses (nothing is
excluded from a table), so it is hand-written.

## 7. REOPEN

- A census, or a survey with a religion item (a post-2026 SHDS, an Afrobarometer round).
- Foreign residents or refugees by region with nationality: UNHCR Somalia publishes refugees by
  region; drawing Ethiopian refugees by region would place Christians below the nation, so §14 first.
- A separate Somaliland entry, if Anita wants de facto states apart (§4).
- Commune-level (district) calibration, if the planning estimates are ever replaced by counted
  district figures.

## 8. Calls someone might reverse

1. Everyone on `islam`, not Pew's residual (Mauritania's construction).
2. The 2026 planning estimate at region, not PESS 2014, and not the district figures.
3. Somaliland inside `so`, outline added through `ALSO`.
4. Border hexes: Cabudwaaq snapped from up to 10 km; hexes a neighbour's layer holds left to it.
5. Sixteen cap blocks lowered on a rule that reads a GeoNames town with no population as capped.
6. `gap_share` from UN DESA's imputed 77,972, not UNHCR's registered 41,763.
