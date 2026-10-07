# Aruba (aw): record

Drawn 2026-10-05 (session edd42a8c-aw). Census 2010, language most spoken in the household, 49
populated census zones, 99,495 people drawn (101,484 counted); placed on Kontur hexes cut along
CBS Aruba's zone polygons.

Files: `sources/aw_census.py` (counts), `sources/aw_geo.py` (placement layer),
`taxonomy/aw2010.py`, `countries/aw.py`, `data/normalized/aw.csv`, `data/geo/aw/aw_hexes.gpkg`,
`data/raw/aw/` (the census report PDF, the zone GeoJSON). No tree fragment: every node exists.

## 1. What exists, and why 2010

- **Census 2010, Table P-D.2** (report pp.111-114): population by language most spoken in the
  household, by region (8) and zone (55, seven "<region> other" zones, six of them empty), by
  sex. Nine columns: Papiamento, Spanish, Dutch, English, Chinese, Does not speak (yet), Others,
  Not reported, All. One answer per household, given to each member. P-D.1 has the same by age;
  P-D.3 by region and age. The report is the PDF religiondots already drew Aruba's religion from
  (Wayback copy of cbs.aw; cbs.aw is now behind a Sucuri JavaScript challenge, and the Wayback
  Machine returned 429 at fetch time, so `--fetch` copies religiondots' bytes when present).
- **Census 2020** (held Sept-Dec 2020) asked for the two most spoken household languages but, as
  far as is published, only as five pair groups by region (Mapping Census 2020 StoryMap 3; ArcGIS
  service `Social_Atlas_Languages_by_Region`, fields PD1-PD5): Papiamento only 43.2%, Papiamento
  and Spanish 18.9%, and English 9.4%, and Dutch 6.8%, Papiamento not among the two 18.6%. The
  last group names no language and the StoryMap says language is "only available by region". The
  2020 population tables on cbs.aw (P-B, housing) have no language table; the REDATAM server
  (`prod.redatam.org/binabw`, BASE=AUA2010) holds 2010 only. 2010 it is; Spanish has very likely
  grown since (Venezuelan arrivals after 2015), which 2010 cannot show.
- **The coverage sweep's lead** (REDATAM 2010, region/zone) was right about the question; the
  printed zone table made the REDATAM route unnecessary.

## 2. Checks (sources/aw_census.py)

Cells are rounded one by one (P-D.1 itself has 3,326 + 3,181 = 6,508), so sums are checked within
a tolerance and the largest deviation printed: male + female against total, at most 1; the eight
columns against All, at most 3; zones against their region's TOTAL, at most 2; zones against the
island row, at most 5. The island total is 101,484, as P-D.1 and the religion table. P-D.2's
column totals against P-D.1's: Papiamento 69,358 / 69,354, Spanish 13,711 / 13,710, Does not speak
1,563 / 1,568, the other five equal. The column order is asserted from the header words' x
positions (Dutch left of English on p.111; Chinese, Does not speak, Others, Not rep., All on
p.113): the age profile agrees (1,479 of the 1,568 non-speakers are 0-4) and so does the
geography (English 44% in Village and Kustbatterij, San Nicolas).

## 3. Mapping

Papiamento on `creole.portuguese_based.papiamento` (pl2021's and cw2023's node). Spanish, Dutch,
English on the shared nodes. Chinese on `sinotibetan.sinitic`, as cw, cy, kg, kh. Others (1,725)
on `other`: no breakdown printed, no indigenous language on the island. Does not speak (yet)
(1,563) and Not reported (432) are not drawn and make up `gap` (1,995, 2.0%). Colours as Curaçao:
Papiamento pale green, Spanish ochre, English pale blue, Dutch blue; nothing hand-picked.

## 4. Geography and placement

- **Zones.** CBS Aruba's ArcGIS Online service `Population_Tables_Census_2010_2020` (account
  R.vdBiezen, the office's GIS account): 55 polygons, `Zone` = GAC2 code (region digit + zone
  digit), `TotPop_10` summing to 101,484. The table's zones are keyed by print order within each
  region (asserted against the printed names for regions 1-6); the join check is that each code's
  table total equals TotPop_10 within 2, and the populated sets match both ways (49 zones; San
  Nicolas North other has 43 people, the other six "other" zones none).
- **Placement.** religiondots' Kontur hexes for AW (279, read only), intersected with the zones
  and each hex's people shared among its pieces by area (519 pieces in the 49 zones): the zones
  are too small (median about 2 km²) for whole-hex centroid assignment, which would leave small
  zones empty. 12 Kontur people fall in hexes touching no zone.
- **Witness.** Kontur per zone against the 2010 census: national ratio 1.026, normalised p10 0.70,
  median 1.02, p90 1.23; log r = 0.889 against a best of 0.520 over 500 shuffles. Highest Eagle/
  Paardenbaai 4.48 and San Nicolas North other 3.97 (hotel strip and industrial land; Kontur
  counts what census residents do not), lowest Village 0.26. Only shares inside a zone matter, so
  nothing is calibrated.
- **No nationality weighting.** Unlike Curaçao the counts are already per zone; CBS's other
  sub-zone layers were checked and are partial (hexagon grid `HegaxonGRID_60000SqM_PopData`
  holds 5,544 people; `Population_street_neighbourhood` 32,915), so plain Kontur population.

## 5. Calls someone might reverse

- 2010 single answers over 2020's pair groups (§1).
- "Does not speak (yet)" left undrawn rather than given its household's language (the census
  gave none).
- Hex pieces weighted by area within a hex, not by any finer population.

## 6. Scatter

97 dots at 1:1000 over 85 polygons; 2,495 people fall under one dot per language and draw none.
scatter.py's water clip reported one unit losing over 95% to the sea and left it unclipped; not
traced to a zone.
