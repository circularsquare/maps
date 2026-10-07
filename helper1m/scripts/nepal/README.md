# Nepal

Three levels: 7 provinces, 77 districts, 753 palikas (local levels). Years
2021 (census), 2026 and 2031 (NSO projection). The viewer's current-year
estimate is the line through 2026 and 2031, so for 2026 it is NSO's own
projected figure for that palika.

Run order, from the repo root:

```
C:\Python39\python.exe helper1m\scripts\nepal\prep_boundaries.py
C:\Python39\python.exe helper1m\scripts\nepal\fetch.py
C:\Python39\python.exe helper1m\scripts\nepal\check.py
C:\Python39\python.exe helper1m\scripts\build_country.py nepal
```

`fetch.py` caches everything under `helper1m/data/nepal/raw/`. The first run
makes 838 small API calls one at a time and took about 25 minutes; after that
it is offline and takes seconds. `check.py` reads the Kontur hexes from
religiondots (read only) and needs `fetch.py` to have run.

## Sources

**Boundaries.** OCHA COD-AB Nepal v02 (valid 2024-03-14), already on disk at
`data/asia1m/nepal/npl_admin{1,2,3}.shp`. Provinces and districts are used as
they are. The palika file has 775 features: the 753 local levels plus 22
polygons for national parks, reserves and the Lumbini area, which in Nepal sit
outside every palika. The third digit of the unit code (character 6 of
`adm3_pcode`) is the unit type, and 5 means a park; filtering on it leaves
6/11/276/460 metropolitan, sub-metropolitan, municipal and rural units, which
is Nepal's published make-up. `prep_boundaries.py` drops the parks and writes
`helper1m/data/nepal/boundaries/adm3.gpkg`, so parks are holes at the palika
level (nobody is counted in them). The local-level map dates from 2017 and has
not changed since, so the 2024 file is the census geography.

**2021: census count.** National Statistics Office (NSO), National Population
and Housing Census 2021, `Religion_NPHC_2021.xlsx`, sheet
`Prov_District_local level`, the Total column:
https://censusresults.nsonepal.gov.np/files/caste/Religion_NPHC_2021.xlsx
It is the only census workbook found with all 753 local levels in one sheet.
It has no geographic codes; the hierarchy is indentation (province, district
and palika labels in columns B, C and D). Total 29,164,578.

Each district has an extra INSTITUTIONAL row (barracks, prisons, hospitals,
hostels, monasteries): 239,098 people, 0.82%, with no location finer than the
district. Those are spread over the district's palikas in proportion to their
population (largest remainder, so the district is exact). That means every
district and province equals its census row. It also means a palika with a big
barracks or prison is slightly understated and its neighbours slightly
overstated; at 0.8% overall this is small.

**2026, 2031: NSO projection.** NSO, *Population Projections for Nepal
2021-2051* (Thematic Report VIII of the 2021 census), medium scenario. The
report (`raw/Population Projections for Nepal.pdf`, from
https://censusresults.nsonepal.gov.np/files/result-folder/Population%20Projections%20for%20Nepal.pdf)
prints only national and province tables, but the projection was made with a
hierarchical cohort-component model down to the ward, and the site's
Population Projection page serves every province, district, palika and ward
from a public JSON API:

```
https://censusapi.cbs.gov.np/api/v1/population-projection/population-table
    ?province=<1-7>&district=<1-77>&municipality=<n within district>
```

Each call returns totals by sex for every year 2021-2051. The national series
matches the report's Annex 5 (medium scenario) exactly: 29,356,136 in 2021,
30,034,040 in 2026, 30,603,859 in 2031.

The API's numbering is not in the API. It lives in the page's JavaScript: one
of the Next.js chunks of `/population-projection` holds a literal list of 77
districts (`{label:"taplejung",value:"1",province:"1"}`) and 753 palikas
(`{district:"1",value:"1",label:"Phaktanlung Gaunpalika",no_of_wards:7}`).
`fetch.py` finds that chunk through the build manifest, so a redeploy with new
chunk hashes does not break it. The cached copy is `raw/nso_unit_lists.json`.

The projection's 2021 base is not the raw census. Per the report (section 2),
it is the census age-smoothed and corrected for under-five undercount, using
the post-enumeration survey; no adjustment was made above age four. That puts
2021 at 29,356,136, 0.66% above the census count. So the history step from the
2021 census to the 2026 projection carries that 0.66% on top of real growth.
The estimate is not affected, because it uses 2026 and 2031 only.

## How the pieces are joined

- **Census to NSO projection lists:** the palika list behind the projection
  page is in the census workbook's order with the census's exact labels (all
  753 identical, including NSO's spellings like `Metropolitian`). They are
  paired by position, district number and palika number within the district,
  and the labels are asserted equal. Districts are checked by name and
  province the same way.
- **Census to COD-AB:** by name inside each district, the same rules as
  religiondots (`religiondots/sources/np_geo.py`): strip the unit-type word
  (Gaunpalika, Nagarpalika, Municipality, Rural Municipality, Metropolitian
  City...), strip a leading district name (`Manang Ngisyang` vs `Ngisyang`),
  then a one-character fallback that must be unique among the district's
  unclaimed polygons. The fallback fires once: census `Melanchi` to COD
  `Melamchi` (Sindhupalchok). No manual aliases. Parks are dropped before the
  join, because four share a name with a palika in the same district
  (Shivapuri, Dhorpatan, Shuklaphanta, Lumbini Sanskritik). As a second check,
  each census district's palikas must all land in a single COD district.
- **The projection's institutional population.** NSO's projected palikas do
  not add up to its projected districts: in 2021 the districts hold 238,818
  more people than their palikas, which is the census's 239,098 institutional
  population again (within 5% in every district). So the projection keeps the
  institutional population at district level, as the census does. It grows to
  244,864 by 2026 and 253,167 by 2031. It is spread over palikas pro rata, the
  same as the census year. Mustang is the biggest share: 3,127 of its 14,344
  people in 2021.
- **Districts and provinces** are summed from palikas in every year. For 2021
  that equals the census district and province rows; for 2026 and 2031 it
  equals NSO's own district and province projection rows (also fetched, as a
  check), and those sum to NSO's province and national rows.

## Checks

From `fetch.py` and `check.py` (2026-10-02):

- **Census workbook nesting:** palikas plus the institutional row equal each
  district, districts equal each province, provinces equal Nepal (29,164,578).
- **National totals:** 2021 29,164,578 (census, exact); 2026 30,034,040 and
  2031 30,603,859 (both exactly NSO's Annex 5 medium scenario).
- **Provinces, 2021**, all 7 exact against the census rows: Koshi 4,961,412,
  Madhesh 6,114,600, Bagmati 6,116,866, Gandaki 2,466,427, Lumbini 5,122,078,
  Karnali 1,688,412, Sudurpashchim 2,694,783. **2026 and 2031:** all 7 exact
  against NSO's province projection rows.
- **Districts:** all 77 exact against the census (2021) and NSO's district
  projection rows (2026, 2031).
- **Coverage:** 7/7, 77/77 and 753/753 boundary units have all three years;
  no population row lacks a unit. `build_country.py` reports every feature
  with population at every level.
- **Palikas against an independent source.** The only one found is the Kontur
  Population grid (2023-11-01, 400 m hexes), already assigned to COD palika
  codes by religiondots (`religiondots/data/geo/np/np_hexes.gpkg`, read only).
  Compared with NSO's 2023 projection: Kontur's total is 4.9% higher. After
  taking that out, only 26.6% of palikas are within 10% and 61.0% within 25%
  (quartiles 0.84-1.24); for districts, 32.5% within 10% and 75.3% within 25%.
  This says more about Kontur than about NSO. The biggest gaps follow the
  pattern religiondots found: Kontur puts town people outside the municipal
  line (Butwal, Siddharthanagar, Bhimdatta, Birendranagar, Triyuga,
  Janakpurdham all read 0.25-0.33 while the rural palikas around them read
  high, e.g. Rohini 4.0 next to Siddharthanagar), and it over-models a block
  of the Parsa Terai (Kalikamai 8.0, Pakaha Mainpur 3.2). Kontur is not good
  enough at this grain to judge the 10% target. The palika figures are NSO's
  own, from a model that is consistent from ward to nation, which is the
  strongest evidence available.
- **Projected change by palika**, 2026 to 2031: median -1.2%, 5th-95th
  percentile -14.8% to +7.7%; extremes Biruwa (Syangja) -23.6% and Chame
  (Manang) +16.7%. From the 2021 census to 2026: median -0.5%, Biruwa -25.2%,
  Chame +23.3%. Most hill palikas are projected to shrink and the cities and
  Terai to grow.

## Known weaknesses

- The 2026 figure is a projection made from 2016-2021 migration rates. Palikas
  where those rates were extreme (fast-emptying hill units, small fast-growing
  ones) carry the most risk of being more than 10% off.
- The institutional population is spread pro rata, not placed where the
  barracks, prisons or monasteries are. It is 0.8% nationally but 22% of
  Mustang.
- The history step from the 2021 census to 2026 includes the projection's
  0.66% under-five undercount correction on top of real change.

## Gotchas and dead ends

- `censusnepal.cbs.gov.np` is a dead placeholder; the live site is
  `censusresults.nsonepal.gov.np`, a Next.js app with no `_next/data` routes.
  File lists and dropdown code lists are literals inside the page chunks.
- The site keeps files in different folders by section: the census workbook
  is under `/files/caste/`, the projection report under `/files/result-folder/`
  (`/files/thematic/` 404s for it, although it is listed on the thematic page).
- The API is at `censusapi.cbs.gov.np` (the old CBS hostname), not on the
  nsonepal domain.
- HDX's COD-PS for Nepal (`cod-ps-npl`) is a 2023 projection to districts only,
  so it was not needed.
- No 2011 figures by present-day palika were looked for: the projection gives
  a recent, same-method pair, which is what the estimate needs.
