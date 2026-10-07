# Sri Lanka

Three levels: 9 provinces, 25 districts, 340 Divisional Secretariat (DS) divisions, the
units of the 2024 census. Years 2012 and 2024, both census counts. The viewer's
current-year estimate is the line through them.

```
C:\Python39\python.exe helper1m\scripts\srilanka\download.py        # ~30 MB, cached in data/srilanka/raw/
C:\Python39\python.exe helper1m\scripts\srilanka\prep_boundaries.py
C:\Python39\python.exe helper1m\scripts\srilanka\fetch.py
C:\Python39\python.exe helper1m\scripts\srilanka\check.py
C:\Python39\python.exe helper1m\scripts\build_country.py srilanka
```

All of it runs in under a minute once downloaded.

## Sources

**2024: census count by GN division.** Department of Census and Statistics (DCS), Census
of Population and Housing 2024, "Grama Niladhari Division Level Population by Sex and Age
Group", from the CPH2024 page
(`https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024/GN_population_excel`,
served as an attachment with no extension). 14,008 GN divisions in 340 DS divisions, total
21,781,800. The sheet title still says "(Provisional)", but the total is the final
report's, and every district equals Table 3.2/3.3 of the final report. Summed to DS
divisions by the census's own district and DS codes.

**2012: census count by DS division.** CPH 2012 district reports, Table A1 "Population by
divisional secretariat division, sex and sector", one single-page PDF per district:
`https://www.statistics.gov.lk/PopHouSat/CPH2011/Pages/Activities/Reports/District/<District>/A1.pdf`
(Kandy's file is `Table%20A1.pdf`, Moneragala's folder is `Monaragala`). 331 DS
divisions, 20,359,439. `a1_2012.py` reads the text layer; every row passes both sexes =
male + female and every district's DS rows sum to its district row.

**Boundaries.** DCS's own cartography unit publishes the 2024 census geography on ArcGIS
Online (organisation `v2E6mIH6KqVu8t9L`, "cartographydcs"; the dashboards are linked from
the CPH2024 page). The layer `DSD_POP_DATA_NEW_Update` has 340 DS polygons keyed by
`ds_uid` (district code + DS code, e.g. 2103 = Kandy / Thumpane) with the 2024 count on
each, which equals ours for all 340. Districts and provinces are dissolved from it, so the
levels nest. The same organisation also serves `GND_POP_DATA_update` (14,008 GN polygons
with 2024 counts) and `GN_Division`; only the GN attributes are downloaded, as a reference.

This beats OCHA's COD-AB (v03, valid 2022-08-16, 339 DS divisions) for this job: it is the
census's own geography, with the 2024 codes and the Kalmunai split, so no name or code
crosswalk to the boundaries is needed at all. religiondots (`sources/lk_geo.md`) found that
COD's DS codes differ from the census's for 13 divisions, which this avoids.

**Kontur** population 2023-11-01 (400 m hexes, HDX) for `check.py` only.

## Carrying 2012 onto the 2024 DS divisions

The 2024 final report's Tables 2.3 and 2.4 give 331 DS divisions in 2012 and 340 in 2024
(Nuwara Eliya 5 to 10, Galle 19 to 22, Ratnapura 17 to 18); GN counts barely moved (14,021
to 14,008).

- **Same unit, same or respelled name (321):** paired by name within the district. `fold()`
  drops parenthetical glosses and evens out romanisation (th/t, dh/d, w/v, ee/i, oo/u,
  ck/k, doubled letters); `ALIAS` lists the 31 pairs it does not catch, and `check.py`
  prints every pairing whose names differ. Real renames: Hanwella is now Seethawaka
  (Colombo), Eragama is Irakkamam (Ampara, its Tamil name), Kalmunai Tamil Division is
  Kalmunai North Sub.
- **Splits (`SPLITS`):** Kothmale into Kothmale West and East; Mathurata from
  Hanguranketha; Nildandahinna from Walapane; Thalawakelle from Nuwara Eliya; Norwood from
  Ambagamuwa (now Ambagamuwa Koralaya); Rathgama and Madampagama from Hikkaduwa;
  Wanduramba from Baddegama; Kaltota from Balangoda. Each parent was identified by GN
  numbers (each child's GN numbers lie inside its parent's run, e.g. Wanduramba's 186-221
  interleave with Baddegama's 184-220) and confirmed by population: the children's 2024
  sum is 0.99-1.08 times the parent's 2012 count, in line with the district. The parent's
  2012 count is split between the children by their 2024 counts (largest remainder), so
  each child shows its parent's growth. No 2012 GN-level table exists to do better (see
  below).
- **Kurunegala:** 2012 printed Panduwasnuwara (63,742) and "Katupotha (Sub Office)"
  (32,386); 2024 has Panduwasnuwara West (71,186) and East (31,790), with interleaved GN
  numbers. Katupotha probably became East, but that would give East -1.8% against +11.7%
  for West in a district that grew 9.2%, and nothing confirms the pairing, so the 2012 pair
  is split over the 2024 pair by 2024 shares, like the other splits.

Every district's carried 2012 sum equals its A1 district row, and those equal Table 3.3.

## Checks (`check.py`, 2026-10-06)

- National 20,359,439 (2012) and 21,781,800 (2024), exactly the census.
- All 25 districts equal the final report's Table 3.3 in both years; all 9 provinces equal
  Table 3.2 for 2024.
- 340/340 DS 2024 figures equal the population DCS stores on its DS polygons.
- 9 / 25 / 340 polygons, every one with both years; no population row without a polygon.
- Growth 2012-2024 by DS: median +7.4%, 5th-95th percentile -3.9% to +23.1%. The fastest
  are the post-war resettlement areas of the north (Musali +123%, Puthukkudiyiruppu +66%,
  Pachchilaipalli +53%, Valikamam North +52%, where the high-security zone was released);
  the slowest are inner Colombo (Colombo -10%, Kotte -11%, Thimbirigasyaya -9%) and Delft
  (-17%).
- Kontur 2023 summed on hex centroids, against the census line at 2023: national factor
  0.996; DS divisions within 10% 72%, within 25% 91%; districts within 10% 88%. The
  outliers are all in the Vanni (Mullaitivu, Kilinochchi, Mannar), where Kontur reads
  0.1-0.5 of the census, the same under-modelling of resettled land seen elsewhere; Madhu
  reads 2x. Kontur is a good witness everywhere else, which says the DS polygons and the
  DS figures agree.
- Spot checks, 2024: Colombo DS 292,089; Thimbirigasyaya 217,118; Homagama 280,771; Kandy
  Four Gravets & Gangawata Korale 153,329; Galle Four Gravets 108,321; Jaffna 48,543;
  Negombo 137,952.

## Known weaknesses

- **The nine split divisions** carry their parent's 2012-2024 growth, not their own.
- **No newer year.** The census was taken in late 2024, so the line is short enough, but it
  averages 2012-2020 growth with the emigration after 2022; the 2026 estimate may run
  slightly high. DCS's mid-year district estimates were not added, because those before
  the census came from the 2012 base and are not on the same footing.
- **Fast post-war growth extrapolates.** Musali, Puthukkudiyiruppu and similar resettled
  divisions grew fast in 2012-2024; most of that happened early, and the straight line
  carries it on.
- **DCS's DS layer** overlaps itself by 19.8 km2, nearly all in Batticaloa between
  Koralai Pattu North and its neighbours: the report's footnote says Punani East GN
  (211B) belongs to Koralai Pattu North but part of it is run from Koralai Pattu Central.
  The polygons cover 252 km2 less land than COD's outline, mostly along the coast and
  lagoons.

## Tried and not used

- **GN level.** 2024 has GN counts and DCS GN polygons, but 2012 GN counts exist only as
  labels on the district GN maps (`/Resource/en/Population/CPH_2011/<District>.pdf`,
  "<GN number>, <population>"). The text layer breaks labels across lines, each map shows
  neighbouring districts' labels too, and 10-20% of numbers came back conflicting, so a
  2012 GN table could not be read cleanly. The CPH 2012 "Quick Stats" site stops at DS.
- **HDX COD-PS** for Sri Lanka is a 2023 projection to districts only.
- **COD-AB v03 DS polygons**: 2022, 339 units, and codes that disagree with the census for
  13 divisions (religiondots `sources/lk_geo.md`); DCS's layer has none of that.

## Files

- `download.py` — all downloads (DCS workbook, report, 25 A1 PDFs, ArcGIS layers, Kontur).
- `a1_2012.py` — Table A1 reader; run it alone to list every 2012 DS division.
- `prep_boundaries.py` — writes `data/srilanka/boundaries/adm{1,2,3}.gpkg`.
- `fetch.py` — writes `data/srilanka/population.csv`; `ALIAS` and `SPLITS` are every hand
  decision.
- `check.py` — the checks above.
