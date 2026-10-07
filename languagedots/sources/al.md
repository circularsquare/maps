# Albania — INSTAT, Census of Population and Housing 2011, mother tongue

Drawn 2026-10-05 (session edd42a8c-al). 2,796,295 people in 9 languages on the 12 qarqe, with
Albanian, Greek and Macedonian placed inside each qark by the census's own counts for the 373
administrative units of 2011. 2,792 dots.

Files: `sources/al_census.py` (fetch + normalise), `sources/al_geo.py` (placement layer),
`taxonomy/al2011.py`, `countries/al.py`. No tree fragment: every node already existed.

## 1. Why 2011 and not 2023

The 2023 census asked **language usually spoken at home** (gjuha që flitet zakonisht në shtëpi),
table 1.14 (`/media/14412/tab_1_14_..._qarqe.xls`, kept in `data/raw/al/`) and national PxWeb
`START__Census2023__Cens/CE16`. Its categories at every level are Albanian / Other language /
Mixed languages / Prefer not to answer / Not available. No minority language is named, even
nationally, and the two missing categories are about 21% of the country (Berat 26%: 37,225 "not
available" of 140,956). It would draw a country of Albanian and "other".

The 2011 census asked **mother tongue** and printed eight languages by name per prefecture, with
only 3,843 (0.14%) invalid or undetermined. Brief §2: old vintage is fine, say the year. Taken.
A 2023-counts / 2011-shares blend would change counts and is a proxy, so not done.

## 2. The tables

* **Counts drawn**: PxWeb `START__Census2011__Census_Prefecture/CENSUS_P_1114` ("1.1.14 Popullsia
  banuese sipas gjuhës amtare 2011"), 12 qarqe × Gjithsej + 10 categories.
  Database: `https://databaza.instat.gov.al:8083/pxweb/sq/DST/` (religiondots found it; the
  Albanian interface, no REST API). The script posts the ASP.NET selection form back with every
  value selected (counts, not the percentage variable) and parses the HTML table. Numbers print
  with `.` as thousands separator.
* **National check**: `START__Census2011/Census1115` (1.1.15), same categories.
* **Placement**: INSTAT's ArcGIS Online organisation
  (`services7.arcgis.com/E9FE1JuiACmTPbPv`): `administrativeunit_p_mtong{alb,gre,mac}_2011_view`
  (full-precision % of resident population per unit) and `administrativeunit_p_distrib_2011_view`
  (2011 population per unit, with polygons). Mother tongue exists at unit level for these three
  languages only; municipality (61) and prefecture versions exist too. There is a
  `prefecture_p_mtongalb_2023_view`, whose values are table 1.14's 2023 *home language* Albanian
  share despite the "mother tongue" layer name.

## 3. Checks, with numbers

* Each qark's ten categories sum to its printed total.
* The twelve qarqe sum to national table 1.1.15 in every category: total 2,800,138; Albanian
  2,765,610 (98.77%), Greek 15,196, Macedonian 4,443, Romani 4,025, Aromanian 3,848, Turkish 714,
  Italian 523, Serbo-Croatian 66, Other 1,870, invalid/undetermined 3,843.
* ArcGIS: share × population lands within 1.5e-11 of an integer on all 373 × 3 values, so these are
  the census counts. Unit populations sum to every qark's printed total, and unit Albanian, Greek
  and Macedonian sum to the qark table's figures exactly, all 36 pairs. The join (CODE_PREFECTURE
  = religiondots' qark `unit`) is therefore proved on population and on three languages.
* Placement layer: religiondots' 22,289 Kontur hexes (read only), each tagged with the 2011 unit
  its centroid falls in, restricted to units of the hex's own qark; 341 centroids outside every
  unit of their qark took the nearest one. All 373 units hold a hex. Kontur / census 2011 per
  unit: median 0.87 (10th pct 0.61, 90th 1.32), the expected post-2011 drift to Tirana and Durrës.

## 4. Calls

* **Rumanisht = Aromanian.** The prefecture table says `Rumanisht`, the national `Arumanisht`; the
  prefecture figures sum to the national Arumanisht exactly and there is no Romanian answer.
  Node `romance.aromanian`.
* **Macedonian kept as printed.** 3,183 of 4,443 are in Liqenas (Pustec, Prespa, 97% of the
  unit). 105 are in Shishtavec (Kukës), which is Gora; their speech is Gorani, but the census
  label is Macedonian and the table cannot split it.
* **Greek**: Mesopotam 2,474 (89%), Dropull i Poshtëm 1,982 (94%), Dhivër 1,062 (76%), Aliko,
  Sarandë, Dropull i Sipërm, Himarë, then Tirana 798 (0.2%).
* **Other (1,870)** on `other`: the census does not separate indigenous from foreign remainders.
* **Albanian** as one leaf; Gheg and Tosk are not asked apart.
* **Placement proxy (brief §4.4)**: Albanian, Greek and Macedonian inside each qark go to each
  2011 unit in proportion to that unit's census count, then by Kontur population inside the
  unit. Counts stay the qark table's. Romani, Aromanian and the small rest go by population
  within the qark (Aromanian is really concentrated around Korçë, Fier/Myzeqe and Vlorë, Romani
  in town edges; nothing published places them).
* **Colours** unchanged: Albanian #b17000 against Greek 0.29, Macedonian 0.26, Romani 0.18,
  Aromanian 0.34 (OKLab). Albanian and other 0.13 is the closest, and other is 1,870 people.

## 5. Reliability, written into note_public

The 2011 census was contested: Omonia and the Greek minority party PBDNJ called for a boycott;
Article 20 of the census law set a fine for declaring an ethnicity different from the birth
certificate; the Council of Europe's minority committee judged the minority figures unreliable
(Balkan Insight, 2011-10-05 and 2011-07-06; Wikipedia "Minorities of Albania"). The minority
languages are most likely undercounted. Not corrected; no other source counts them by place.

## 6. Not used

* 2023 qark/national home-language table: no languages named (§1).
* 2011 public microdata (`/media/1548/...mikrodata.rar`, religiondots §2): prefecture geography only.
