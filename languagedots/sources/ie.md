# Ireland: Census of Population 2022, by Small Area

Drawn 2026-10-05 (session edd42a8c-ie). 5,149,139 people (the whole census population, all ages)
on 18,919 Small Areas, 65 nodes. Script `sources/ie_census.py`, mapping `taxonomy/ie2022.py`,
fragment `taxonomy/tree.d/ie.txt`, entry `countries/ie.py`.

## 1. The question, and why the map has to be built

The 2022 form has no first-language or main-language question. It has two language questions,
both asked of everyone:

- **Q14** "Can you speak Irish?" and, if yes, how often: daily within the education system;
  daily / weekly / less often / never outside it.
- **Q15** "Do you speak a language other than English or Irish at home?" and, if yes, which
  (one write-in). 751,507 people named a language.

English is never asked about. So the map is assembled from three parts (tiers in brackets):

1. **Languages other than English or Irish** (Q15). SAPS theme 2 table 5 names Polish, French
   and Spanish per Small Area (`measured`) and gives an `Other (incl. not stated)` remainder. The
   remainder is shared among the other 63 labels of PxStat **F5029** (language spoken at home x
   county of usual residence, 2022) in its county's proportions (`derived`).
2. **Irish**: the Irish speakers who speak it **daily outside the education system**,
   T3_2DIDOT (daily within and daily outside, 20,581) + T3_2DOEST (daily only outside, 51,387)
   = **71,968**, which is CSO's own headline figure for daily Irish speakers outside education
   (`derived`).
3. **English**: everyone else in the Small Area, T1_1AGETT less parts 1 and 2 (`derived`). No
   Small Area went below zero, so nothing was clipped.

`IRISH_DAILY = False` in the script draws nobody as Irish (they fall into English).

## 2. Sources (all CC BY 4.0, Central Statistics Office)

| file in `data/raw/ie/` | URL |
|---|---|
| `SAPS_2022_Small_Area_UR_171024.csv` | https://www.cso.ie/en/media/csoie/census/census2022/SAPS_2022_Small_Area_UR_171024.csv |
| `F5029.json` | https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F5029/JSON-stat/2.0/en |

Column meanings from CSO's SAPS glossary (`Glossary_Saps_2022_REVISED_21102024.xlsx`, the copy
in religiondots' raw folder). Boundaries are religiondots' Small Area layer, read-only
(`religiondots/data/geo/ie/smallareas2022/SMALL_AREA_2022.shp`, 2022 vintage; its
`sources/ie_geo.md` documents the 0/0 join on `SA_GUID__1`).

Other CSO tables seen and not used: F5015 (Q15 speakers by citizenship and English ability,
national); CPNI18/19 (joint Ireland/Northern Ireland publication, 15 languages, ages 3+:
735,514 in Ireland, so the 751,507 includes about 16,000 children under 3); the SAP2022 tables
on PxStat repeat SAPS.

## 3. Checks and their numbers

- The SAPS file's last row is the State (GUID `IE0`); every T column of the 18,919 Small Areas
  sums to it exactly.
- Per Small Area: Polish + French + Spanish + Other = T2_5T; the ten frequency rows sum to
  T3_2ALLT; T3_2ALLT = T3_1YES.
- SA to county: the boundary layer's 34-county field mapped onto F5029's 31 counties (Cork city
  and county merged, Limerick, Waterford and Tipperary merged, Galway city and county apart);
  18,919 Small Areas match both ways.
- F5029: the 66 labels sum to `All languages` in every county; the counties sum to the State.
- SAPS summed by county against F5029: State totals equal exactly (Polish 123,968, French 51,568,
  Spanish 48,113, all languages 751,507). Counties differ by a few: largest 19 for each named
  language, 99 for the total (Fingal), the absolute differences summing to 778 of 751,507 (0.1%).
  The pattern (small, scattered, every county, exact national totals) looks like CSO's
  disclosure control on SAPS moving households between areas rather than a join fault; the check
  allows 100 per county and 0.5% overall.
- Population 5,149,139 equals CSO's 2022 census population. Irish daily outside education
  71,968 equals CSO's published figure.

## 4. Calls

- **Irish on daily use outside school, not ability.** AGENT_BRIEF §2: a language most people name
  only as a learned second language is drawn on the main language. Irish ability (1,873,997 "yes",
  36%) is overwhelmingly school Irish, so it is drawn as English; but Irish is a real first
  language for a sizeable group (the Gaeltacht), so the subset that speaks it daily outside
  education is drawn as Irish. This is a census figure, but treating daily use as the home
  language is a choice: it will include some fluent second-language speakers (Dublin's
  Dún Laoghaire-Rathdown has 3,528) and miss children in Irish-speaking homes whose only answer
  was "daily only within the education system" (T3_2DIT, mostly ordinary pupils, which cannot
  be split). By county: Galway county 5.9%, Donegal 4.6%, Galway city 2.6%, Kerry
  2.4%; 78 Small Areas are over half daily Irish (9,377 people). **Flagged to the supervisor**
  per §2; reversing is the one switch `IRISH_DAILY`.
- **English is a remainder**, including people who answered nothing to Q15 and the usual
  residents absent on census night (SAPS counts them; F5029 is "usually resident and present").
  Neither is separable, so there is no `gap`.
- **Q15 answers as home languages.** "Speak at home" is not "main language": a bilingual home
  may name French or German beside English (French 51,568 is high for the migrant population,
  likely including Irish families raising children bilingually). The census gives no way to
  tell; drawn as answered.
- **County mix for the SAPS remainder**: within a county every Small Area's Other gets the same
  mix, so Romanian, Portuguese, Lithuanian, Malayalam etc. are placed by county. The F5029
  share includes its own `Other stated languages (incl. not stated)`, so that part stays on
  `other`.
- Mapping (all in `taxonomy/ie2022.py`): Filipino and Tagalog, and Bosnian, Croatian and
  Serbian, are printed apart and kept apart. "Chinese, nec" on `sinotibetan.sinitic` (no
  variety named). "Other Northern European" (1,311) on `indoeuropean` (what is left in the
  north after the named Nordic and Baltic languages is Norwegian, Icelandic, Faroese and
  Britain's Celtic languages and Scots). "Other Southern European", "Other Eastern European",
  "Other Asian" and "Other stated languages (incl. not stated)" on `other`; "Other African" on
  `africa_other`, as Germany does. Irish cant on `indoeuropean.celtic.shelta`; Irish sign
  language on `signlanguage.isl`; "Sign Language (Not Specified)" on `signlanguage`.
- **One new node**: `nigercongo.voltaniger.edoid.edo` Edo (Glottolog bini1246, "Bini"), 1,228
  people, a leaf under us.txt's Edoid group. Generated colour.
- Colours: English pale blue, Irish teal (#06bfa8), Polish green (#00a76c). Irish and Polish are
  the two nearest, but Irish sits in the western Gaeltacht and Polish in towns; left as is.

## 5. Scatter

1:1000, 5,117 dots across 4,895 Small Areas; 0.62% of people are in languages under one dot
nationally; 6 rings. Placement is the Small Area polygon itself (median about 260 people), no
weight, water-clipped.
