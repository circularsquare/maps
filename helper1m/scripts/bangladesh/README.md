# Bangladesh

Four levels. The first three are the OCHA COD-AB v03 boundaries (2023): 8
divisions, 64 zilas, and 507 level-3 units, which are 495 upazilas and 12 city
corporations. Level 4 has 4,936 units: unions, paurashavas, and city
corporation wards or thanas, drawn on the 2011 union polygons (see "Level 4").
Two census years, 2011 and 2022, both as enumerated (not the post-enumeration
adjusted figures).

```
C:\Python39\python.exe helper1m/scripts/bangladesh/fetch.py     # writes data/bangladesh/population.csv and level4.gpkg
C:\Python39\python.exe helper1m/scripts/bangladesh/check.py     # prints every check below
C:\Python39\python.exe helper1m/scripts/build_country.py bangladesh
```

`fetch.py` downloads what it is missing (the census report from the Wayback
Machine, the Union Statistics report from BBS, two HDX workbooks and one
district report for the checks) and decompresses Kontur from
`religiondots/data/geo/kontur/`. It reads, never writes, the 2011 USCB workbook
and geodatabase under `religiondots/data/raw/bd/`. Levels 1-3 take about a
minute; level 4 (`level4.py`, called from `fetch.py`) about ten more, mostly
fitting polygons. Level 4 is written one file per division (`adm4/BD10.geojson`
and so on, 12 MB in all).

## Sources

| what | file | where |
|---|---|---|
| boundaries | `maps/data/asia1m/bangladesh/bgd_admin{1,2,3}.shp` | OCHA COD-AB v03, https://data.humdata.org/dataset/cod-ab-bgd (already on disk for asia1m) |
| 2022 upazilas and city corporations | `data/bangladesh/raw/phc2022_national_report_vol1.pdf` | BBS, *Population and Housing Census 2022, National Report (Volume I)*, November 2023, 517 pages. Live URL was `bbs.portal.gov.bd/sites/default/files/files/bbs.portal.gov.bd/page/b343a8b4_956b_45ca_872f_4cf9b2f1a6e0/2023-11-20-05-20-e6676a7993679bfd72a663e39ef0cca7.pdf`; fetched from Wayback snapshot 20231125020331 |
| 2011 unions and wards | `religiondots/data/raw/bd/bangladesh_uscb_202107.xlsx` (Age-Sex sheet) and `Bangladesh.gdb` (layer `BD_GEOG_ADM4_2011_uscb_202107`) | USCB transcription of the BBS 2011 census on HDX, https://data.humdata.org/dataset/bangladesh-subnational-boundaries-and-tabular-data |
| check: 2022 by zila | `data/bangladesh/raw/unrco_phc2022_admin02.xlsx` | UN Resident Coordinator's Office upload of BBS's PHC 2022 dataset, https://data.humdata.org/dataset/populationa-and-housing-census-dataset |
| check: 2022 projection | `data/bangladesh/raw/bgd_admpop_2022.xlsx` | COD-PS, https://data.humdata.org/dataset/cod-ps-bgd |
| check: Kontur 2023 | `data/bangladesh/raw/kontur_population_BD_20231101.gpkg` | copy of `religiondots/data/geo/kontur/kontur_population_BD_20231101.gpkg.gz` |
| 2022 unions | `data/bangladesh/raw/phc2022_union_statistics.pdf` | BBS, *Population and Housing Census 2022: Union Statistics*, May 2025, 499 pages, Table U 01 (PDF pages 94-209). Linked from the census page of the new bbs.gov.bd (`/pages/static-pages/6922e073933eb65569e27220`); the file itself is on BBS's object storage, `objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/.../office-bbs/2024/12/f376ca7e7ee5405f8311de650415b9d8.pdf` |
| 2022 paurashavas, city corporation wards | the National Report above | Table P34 (paurashavas, report pages 388-394) and Table P32 (city corporation wards, pages 377-385) |
| level-4 polygons | `religiondots/data/raw/bd/Bangladesh.gdb`, layer `BD_GEOG_ADM4_2011_uscb_202107` | the same USCB 2011 union/ward polygons the 2011 re-cut uses |
| check: Thakurgaon unions | `data/bangladesh/raw/zila2022/thakurgaon.pdf` | BBS, PHC 2022 District Report: Thakurgaon, Table 01, Wayback snapshot 20250512071726 of `203.112.218.101/storage/files/1/Publications/PHCensus/Rangpur/District Report Thakurgaon Full.pdf` |

## The levels

**Level 3, upazila or city corporation (507).** The v03 level 3 is cut exactly
the way the 2022 census reports: each upazila without whatever part of it lies
inside a city corporation, and each city corporation as one unit. So 2022 joins
one to one.

- 2022: Table P35 (*Household, Population, Household Size and Literacy Rate by
  Upazila*, report pages 395-404, PDF pages 442-451) gives the 495 upazilas,
  each labelled "(Except City Corporation)" where it applies. Table P33 (*... of
  City Corporation by Thana*, report pages 386-387) gives the 12 city
  corporations and their 105 thanas. `report.py` parses both; every district
  row equals the sum of its upazilas, every city corporation row the sum of its
  thanas, and the table total the sum of everything, in households, total, male
  and female. The join to v03 is by (zila, name) and every one of the 507 names
  matched without a single alias, because v03 was drawn from the same BBS
  naming.
- The report's "Total" column is male plus female. The 8,124 people counted as
  third gender (hijra) appear only at zila level, so they are not in these
  figures. That is 0.005% of the country, at most 677 in any zila.
- 2011: the 2011 units are not the 2023 ones. Since 2011, Gazipur (2013),
  Rangpur (2012) and Mymensingh (2018) became city corporations, Dhaka's two
  city corporations absorbed outlying unions (2016), Narayanganj and Cumilla were
  formed from paurashavas, the 61 metropolitan thanas of 2011 disappeared into
  their city corporations, and 13 upazilas were created. So the 2011 count is
  rebuilt from the 5,161 unions and wards of 2011, each moved whole onto one
  v03 unit:
  - Its home is the v03 upazila with the same BBS geocode: 2011 code `100409`
    (Amtali) is v03 `BD10040009`. This holds for 482 of the 544 units of 2011,
    with spelling differences only; Zianagar is now Indurkani and Matlab is now
    Matlab Dakkhin, same codes. Dakshin Sunamganj was renamed Shantiganj with a
    new code, and is aliased by hand (`HOME_ALIAS`).
  - It leaves home only for a unit that did not exist in 2011, when at least
    half its Kontur 2023 population lies inside that unit.
  - A union with no home (the 2011 metropolitan thanas) goes to the v03 unit
    holding most of its Kontur population.

  Result: 4,752 unions stayed home, 153 moved to a new unit, 256 thana unions
  went by largest share. 68 city wards smaller than a Kontur hex were placed by
  a point inside them. `data/bangladesh/unions_2011_to_v03.csv` records every
  union's destination and why.

  Geometry alone was tried first and was too noisy: the USCB 2011 polygons and
  the v03 lines are a few hundred metres apart, enough to move 10-20% of the hex
  centres of 525 border unions across the line. Whole-union moves are also
  structurally right, since new upazilas and city corporation extensions are
  made of whole unions and wards.

  Where the 2011 units went, by destination:

  | v03 unit (new since 2011) | from 2011 | 2011 people |
  |---|---|---|
  | Dhaka North and South City Corporations | 41 metropolitan thanas (minus Kotwali's one ward placed in Keraniganj) | 8,892,508 |
  | Chattogram City Corporation | 11 metropolitan thanas | 2,592,439 |
  | Gazipur City Corporation | 9 unions/wards of Gazipur Sadar | 1,626,077 |
  | Narayanganj City Corporation | Narayanganj, Siddhirganj, Kadam Rasul paurashavas | 709,381 |
  | Khulna City Corporation | 5 thanas (part of Daulatpur and Khan Jahan Ali went to Dighalia and Phultala) | 647,562 |
  | Rangpur City Corporation | 9 unions/wards of Rangpur Sadar | 568,185 |
  | Sylhet City Corporation | 26 wards of Sylhet Sadar | 451,643 |
  | Rajshahi City Corporation | 4 thanas (one Rajpara ward to Paba) | 438,842 |
  | Mymensingh City Corporation | 5 units of Mymensingh Sadar | 430,214 |
  | Cumilla City Corporation | Comilla paurashava, plus Comilla Dakshin paurashava and one union of Sadar Dakshin | 348,891 |
  | Barishal City Corporation | 30 wards of Barishal Sadar | 328,278 |
  | Tarakanda | 10 unions of Phulpur | 298,220 |
  | Osmaninagar | 8 unions of Balaganj | 201,354 |
  | Lalmai | 6 unions of Sadar Dakshin, 1 of Laksam | 210,182 |
  | Karnaphuli | 5 unions of Patiya | 162,110 |
  | Naldanga | 6 unions of Natore Sadar | 129,304 |
  | Eidgaon | 5 unions of Cox's Bazar Sadar | 120,322 |
  | Rangabali | 5 unions of Galachipa | 103,003 |
  | Madhyanagar | 4 unions of Dharampasha | 92,745 |
  | Taltali | 7 unions of Amtali | 88,004 |
  | Dasar | 5 unions of Kalkini | 71,494 |
  | Shayestaganj | 3 unions of Habiganj Sadar | 65,398 |
  | Guimara | one union each of Matiranga, Ramgarh, Mahalchhari | 44,202 |

  18 unions (231,925 people) were kept home although most of their Kontur
  population lies in another pre-existing upazila. Mostly this is the old
  polygons disagreeing with the new lines in the Hill Tracts and along moving
  rivers; for example Tarachha union is 90% inside Bandarban Sadar by geometry,
  but Rowangchhari's 2011 and 2022 totals (27,264 and 27,719) only make sense
  with it kept. One may be a real transfer: Khaleya union (26,446) of
  Gangachara sits 94% inside Rangpur Sadar, and Gangachara is the one upazila
  in Rangpur division whose population falls (2022/2011 = 0.96). If Khaleya did
  move, Gangachara's 2026 estimate is about 4% low.

**Level 2, zila (64), and level 1, division (8).** Sums of level 3 in both
years.

## Level 4 (`level4.py`, `level4_tables.py`)

4,936 units: 4,389 that are one or more unions, 298 paurashavas, 27 a
paurashava with unions around it, 157 city corporation wards, 53 city
corporation thanas or groups of them (40 of them in Dhaka), and 12 holding a
cantonment or other non-union area. Median 26,600 people
in 2022; 4,738 units are exactly one 2022 union or paurashava over exactly one
2011 polygon. Every level-4 unit sits inside one level-3 unit and the units add
up to it exactly, both years. `data/bangladesh/level4_lineage.csv` lists every
2022 row and 2011 polygon behind every unit, and the rule that put it there.

**Counts.**

- 2022 unions: *Union Statistics* Table U 01 (4,584 unions, under their
  division, zila and upazila). `level4_tables.py` reads the text layer. Two
  print quirks are handled there: a long number broken over two lines (rejoined
  only where the row's own arithmetic, households = general + institutional +
  others and every age group = male + female, accepts exactly one way), and a
  row label cut by a page break ("Adamdighi" at the foot of a page, "Upazila"
  at the head of the next). Every upazila's unions add up to its printed row and
  the whole table to its printed "Union Total", 126,055,919.
- 2022 paurashavas: National Report Table P34 (327, by zila only). Each is
  placed in its upazila by name against the 2011 paurashavas (294), the
  upazila names (26), or, for the 5 left, by room: an upazila's P35 figure
  minus its unions is what its paurashavas hold. Bogura paurashava straddles
  Bogura Sadar and Shajahanpur as in 2011, and is split between them by that
  room (422,900 and 63,044). After placing, what each upazila holds beyond its
  unions and paurashavas, 75,158 people in 12 upazilas (cantonments, Mongla
  port and the like), becomes an "Other areas" member, joined to a 2011
  cantonment where the upazila had one.
- 2022 city corporations: Table P32 by ward for Barishal, Chattogram, Khulna,
  Rajshahi and Sylhet; Table P33 by thana for the other seven.
- 2011: the 5,161 unions and wards already placed on level-3 units for level 3.
- All 2022 figures are male + female, like level 3; hijra are left out.

**Polygons.** No open polygons of the 2022 unions exist. COD-AB v03 stops at
upazila (the HDX page says so; v01's union layer, from 2015, is the same 2011
union set), geoBoundaries' ADM4 is that 2015 OCHA layer (5,160 units, and it
nests in v03 no better than USCB's: 43% of units 99% inside one upazila for
both), and OpenStreetMap has almost no union boundaries in Bangladesh (Kontur's
OSM boundary extract, June 2023: 405 upazila-level areas, nothing at union
level). So level 4 is drawn from the USCB 2011 union/ward polygons, and a
level-4 unit is the smallest set of 2011 polygons and 2022 rows that match:

1. inside each level-3 unit, 2022 unions to 2011 unions by name: direction
   words normalised (Uttar/North, Dakshin/Dakkhin/South, Purba, Paschim), word
   order ignored, a bracketed alias tried both ways ("Bitghar (Tiara)"), and a
   consonant-skeleton comparison for spellings (Atharagashia / Atharogachhia);
   the best pairs first, one to one, at a score of 0.8 or more. Paurashavas to
   paurashavas the same way; wards by ward number;
2. a 2022 union left over whose name minus its direction word is a 2011
   union's joins it (Bharella -> Bharella Uttar + Bharella Dakshin), and the
   other way round;
3. whatever is still unmatched on both sides of a level-3 unit is merged into
   one unit (411 rows and polygons in all);
4. a 2011 union left alone joins the upazila's paurashava (absorbed), else its
   longest-border neighbour; a 2022 union left alone joins the unit whose
   2022/2011 ratio it brings closest to the upazila's (65, the weakest rule);
5. `rebalance()`: neighbouring units whose ratios are off in opposite
   directions (one above 1.5 times the upazila's ratio, one below 0.7) are
   merged until neither is. A union carved out of its neighbours after 2011
   that the names could not tie to them otherwise shows as a thirteen-fold rise
   beside a halving (Ghatail: Lakkhindar, Sagardighi and Sangrampur next to
   Rasulpur, Sandhanpur, Dhalapara), and the line through 2011 and 2022 would
   run any estimate off. Real growth with no shrinking neighbour (Savar,
   Ashulia, Sreepur) is left alone. Outside city corporations, a pair that does
   not touch may merge when nothing touching is left (Bishwanath union and the
   new Bishwanath paurashava). Units under 500 people in either year are
   folded into a neighbour.

Each unit's polygon is then the union of its 2011 polygons clipped to its v03
level-3 polygon, with whatever part of the level-3 polygon no 2011 polygon
covers (the two layers are a few hundred metres apart; rivers) given to the
nearest unit by Voronoi cells of points along the unit borders. So level 4
tiles level 3 exactly, and each union is where 2011 drew it.

**City corporations.**

- Barishal, Chattogram, Khulna, Rajshahi, Sylhet: by ward. Ward numbers
  survived from 2011 (ratios 0.73-2.4, median 1.04-1.24 per city). Khulna's
  2011 ward 4 went to Dighalia/Phultala in the level-3 re-cut, so its 2022 row
  joined another ward; the same for Sylhet's ward 8 (now "Ward 08 + Ward 15").
- Dhaka North and South: by thana, 20 units each. The 2011 wards are numbered
  across the undivided city (1-92), and the 2011 split renumbered each half in
  a way no open table records: numbering DNCC's old wards in order gives
  plausible ratios for 1-23 and nonsense after (0.62 to 2.33), DSCC's nonsense
  throughout. Thanas match by name; the thanas created since 2011 are tied to
  their parents by hand (`HAND_LINKS`): Banani to Gulshan, Bhatara to Badda,
  Hatirjheel to Tejgaon and Rampura (they lost what it holds), Rupnagar to
  Mirpur and Pallabi, Bhasantek to Kafrul, Wari to Sutrapur, Mugda to
  Sabujbagh, Shahjahanpur to Motijheel, Kamrangichar to Lalbagh and the 2011
  Kamrangir Char. Pallabi + Mirpur + Rupnagar is the largest unit in the
  country, 1,321,760.
- Gazipur, Narayanganj, Cumilla, Mymensingh, Rangpur were paurashavas and
  unions in 2011. Their 2022 thanas are matched to those by name (Joydebpur to
  Gazipur paurashava, Bandar to Kadam Rasul paurashava, by hand): Gazipur
  splits into 7 (Tongi Purba + Pashchim 758,550 the largest), Cumilla into 2,
  Rangpur into 2; Narayanganj (Bandar fell to half its 2011 paurashava,
  Narayanganj Sadar grew 2.6x, so `rebalance` merged them) and Mymensingh stay
  whole.

## Checks (`check.py`)

1. National. 2022: 165,150,492 against the published enumerated 165,158,616,
   short by the 8,124 hijra. 2011: 144,043,697 against BBS's 144,043,696; the
   USCB Age-Sex sheet's own national row is one person above BBS (its Religion
   sheet has the exact figure), and every union and upazila in it adds up to
   that row.
2. 2022 zilas against the UNRCO/BBS district table, an independent
   transcription: all 64 equal its male plus female exactly; against its total
   with hijra, every zila is short by its hijra count, at most 677. Divisions
   likewise (Dhaka 2,481 short, the largest).
3. 2011 zilas and divisions after the union move, against the USCB rows: all
   64 zilas and all 8 divisions exact. No union crossed a zila line.
4. Level 3, 2022 against Kontur 2023 (hex centres per v03 polygon, scaled to the
   census total): 58.4% of units within 10%, 79.5% within 20%. Kontur is weak
   here and is not evidence against the census. Its worst cases are Laksam
   (census 333,706 on 141 km², 2,360/km², in line with its neighbours; Kontur
   80,600, 570/km²),
   Taltali, Ramgarh, Subarnachar (all below), and Alikadam, Barishal Sadar,
   Keraniganj, Dohar (above). Kontur spreads Dhaka and Barishal into the
   surrounding upazilas and pulls people out of char and hill land.
5. Level 3, 2022 census against the USCB 2022 projection (COD-PS), on the 455
   units whose 2011 lineage is exactly one untouched 2011 upazila: 76.0% within
   10%, 92.1% within 20%. The misses are growth the projection did not foresee
   (Savar -32%, Kaliakair -22%: the census is higher than 2011 but lower than
   the projection) and places the projection had declining (Harirampur +48%,
   Char Bhadrasan +45%). No pattern of mirrored misses between neighbours, which
   is what a swapped join would show.
6. Level 3, 2022/2011 ratio: national 1.147; units from 0.86 (Chouhali, Jamuna
   erosion) to 1.78 (Gazipur Sadar outside the city corporation); quartiles
   1.08 and 1.15. The top end is the Dhaka-Gazipur-Narayanganj industrial belt
   (Sreepur 1.74, Savar 1.67, Gazipur CC 1.65, Kaliakair 1.44).
7. Coverage: every unit at every level has both years; every row in
   `population.csv` lands on a unit.
8. Level 4 (section 6 of the output). Sums to level 3 exactly in all 507
   units, both years. 2022 unions add to the Union Statistics' printed total
   (126,055,919) and paurashavas to P34's (17,897,874). Thakurgaon's District
   Report (BBS 2024, an independent typesetting) prints all 54 of its unions:
   36 equal, 18 higher by 1-15 (its Total counts hijra), none otherwise.
   Kontur 2023 is no use at this scale: 17% of units within 10%, median miss
   33%, and it does just as badly on the raw 2011 polygons against the 2011
   counts (median 34%), so that is Kontur, not the fitted polygons. 2022/2011
   ratios: 1% of units below 0.76, 1% above 1.64; 14 below 0.6 (Char Janajat
   0.12, Saheber Hat 0.22: Padma and Meghna char erosion, each its own 2011
   union with no growing neighbour) and 16 above 2 (Savar's unions, Kashimpur
   in Gazipur, Chattogram ward 39, Rajshahi ward 14).

## Weaknesses

- At level 3 each city corporation is one unit (Dhaka North 5.99 million);
  level 4 splits them as above. No open polygons of the 2022 thanas or wards
  exist (COD-AB v03 mentions a thana and a ward layer but publishes neither;
  OSM has none in Dhaka), so the city splits rest on the 2011 polygons and are
  coarser where the units changed: Narayanganj (967,887) and Mymensingh
  (576,927) stay whole.
- Level 4 polygons are 2011's. A union split or created since 2011 is drawn
  as the 2011 union(s) it came from, with all the 2022 rows inside, so those
  units are coarser than the census. Paurashavas that grew into the unions
  around them since 2011 keep their 2011 outline; where that left a sharp
  ratio pair the two were merged, elsewhere a paurashava's 2011-2022 rise is
  partly area gained.
- Rule 4 above (a 2022 union with no name match and no unmatched 2011 polygon
  goes where the ratio fits) is a guess for 65 rows; the lineage file marks
  them "left over 2022".
- The 18 unions kept home at level 3 although their polygon lies mostly in
  another upazila are clipped to their home upazila, so they are drawn small
  and their land is given to the neighbouring unions.
- 2011 for the 25 new units (13 upazilas, 12 city corporations) and for the
  units they came from is built from whole unions placed by geometry, so it can
  be off where a union was split between two 2023 units. The checks above put
  that at a few per cent at most for any unit.
- Both years are enumerated counts. BBS's post-enumeration check puts the 2022
  undercount at 2.75% (adjusted total 169,828,911); no adjusted figure is
  published by upazila, so it is not applied. 2011 is enumerated too, so the
  two years are like for like.

## What was tried and failed

- `bbs.gov.bd`: TLS certificate failure from this machine (curl exit 60). The
  server sends no intermediate certificate; adding Sectigo's "Public Server
  Authentication CA DV R36" (from the AIA URL in the certificate) to certifi's
  bundle makes it verify, which is how the census page that links the Union
  Statistics was read. The files themselves sit on BBS's Oracle object
  storage, which needs no workaround. The 64 District Reports (union tables,
  and an Excel "community series") are linked from the same page; only
  Thakurgaon's is used, as a check. `file-dhaka.portal.gov.bd`, where a search found a single
  page of the report, times out; Wayback has that page alone.
  `bbs.portal.gov.bd` is retired, but the Wayback CDX listing of its
  publications folder had the full report on the first guess (the two 11 MB
  PDFs dated 19 and 20 November 2023).
- HDX COD-PS (`cod-ps-bgd`) looked like the 2022 upazila source but is a USCB
  projection from 2011 on the 544 units of 2011 (Metadata sheet: "Baseline
  population 2011, Reference year of projections 2022"). Used only as a check.
- The HDX COD-AB v03 geodatabase (`bgd_admin_boundaries.gdb.zip`) opens with 0
  features in this GDAL; the asia1m shapefiles are the same v03 release.
- Union-level 2022 figures: the old `203.112.218.101` host of the zila
  reports times out, and Wayback holds only about 45 of the 64. The national
  Union Statistics volume (May 2025) has every union in one table and replaced
  them.
- Dhaka's 2011 wards by number: see "City corporations" above.
