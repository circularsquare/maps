# North Macedonia — SSO, Popis 2021, mother tongue

Drawn 2026-10-05 (session edd42a8c-mk). 1,704,051 people in 9 categories on the 80
municipalities, placed inside each municipality by the census's ethnicity of each of its 1,781
settlements. 1,701 dots at 1:1000.

Files: `sources/mk_census.py` (fetch + normalise), `sources/mk_geo.py` (placement layer and
settlement points), `taxonomy/mk2021.py`, `taxonomy/tree.d/mk.txt` (a colour only),
`countries/mk.py`. Data: `data/raw/mk/` (three PxWeb cubes for the counts, two for settlements,
`osm_places.csv`), `data/normalized/mk.csv`, `data/geo/mk/mk_hexes.gpkg`,
`data/geo/mk/mk_settlements.csv`. Read only from religiondots: `data/geo/mk/mk_opstini.gpkg`
(GISCO LAU 2021) and `mk_lookup.csv` (SSO code -> LAU code), `ne_10m_admin_0_countries.geojson`.

## 1. The tables

MakStat's PxWeb API (`https://makstat.stat.gov.mk/pxweb/api/v1/<en|mk>/MakStat/Popisi/Popis2021/`),
walked from the root; religiondots found the religion table in the same folder.

* **Counts drawn**: `NaselenieVkupno/NaseleniePopis2021/EtnoKulturniKarakteristiki/T1015P21.px`,
  mother tongue x sex x 82 geographies (country, City of Skopje, 80 municipalities). Categories:
  Macedonian, Albanian, Turkish, Romani, Vlachs, Serbian, Bosniak, Other languages not
  mentioned, Sign language, Unknown, and code 88 "persons for whom data are taken from
  administrative sources". Fetched in English (labels) and Macedonian (names). City of Skopje
  dropped (the sum of its ten municipalities).
* **National check and the breakdown of `other`**: `NaselenieSet/T1013P21.px`, national only, 39
  categories.
* **Placement**: `NaselenieVkupno/PodatociNaselenie/T1503P21.px`, ethnicity by settlement
  (1,781 settlements, labels `<settlement> (<municipality>)`, 6-digit codes). Mother tongue is
  not published below the municipality anywhere in the tree.
* Not used: `NaselenieSet/T1016P21.px`, language usually spoken in the household, national only
  (the same 35 languages). There is no municipal version.

Question asked: mother tongue (majchin jazik), one answer. The 2021 census also counted
non-residents from registers; they are not in these resident tables.

## 2. Checks, with numbers

`python sources/mk_census.py`:
* national total 1,836,713, as published; 80 municipalities;
* all 12 categories (with the total) sum from the 80 municipalities to the national row exactly;
  the 11 categories partition every municipality;
* every municipality's total equals religiondots' total for the same SSO code from the religion
  table T1012P21 of the same census (80 of 80). This is the join check: the codes are shared, and
  religiondots' lookup to LAU codes was itself checked against GISCO populations;
* T1013P21 equals T1015P21's national row in all 10 shared categories, and its 28 extra
  languages plus its own unnamed "other" (106) sum to the municipal `Other languages not
  mentioned` exactly (5,167).

`python sources/mk_geo.py`:
* the 1,781 settlements sum to every municipality's total in the mother-tongue table exactly
  (0 off; municipality named in the label, joined by folded Cyrillic name to mk.csv);
* the 11 ethnic categories partition every settlement;
* Kontur MK 12,576 hexes, 2,089,879 people. 222 hexes (11,175 people) fall outside the GISCO
  polygons: 58 (5,312 people) inside Natural Earth's North Macedonia and clear of a border, snapped
  to the nearest unit within 1 km (mostly lake shores, where GISCO's shore is generalised); 164
  (5,863) near a border or in a neighbour, dropped;
* Kontur / census nationally 1.135; per unit normalised p10 0.81, median 1.02, p90 1.49; log
  correlation 0.939 against a best of 0.334 over 500 shuffles. The outliers are Skopje: Shuto
  Orizari 0.18 and Chair 0.33 (Kontur thins the densest quarters and gives the people to Butel,
  1.89), and Sopishte 6.02 (Kontur has Skopje's southern edge inside it). These change only where
  inside a unit its own census count goes, so Sopishte's 5,600 people are drawn towards the Skopje
  edge of the municipality. Not corrected.

## 3. Calls

* **Bosniak -> `bosnian`.** "Boshnjachki jazik" is Macedonian usage's and SSO's name for the
  language of the Bosniaks; there is no separate Bosnian answer, so it is one answer here, not a
  rival label. Bosnia's census prints both and keeps them apart (ba2013); that does not apply.
* **Vlachs -> `romance.aromanian`.** North Macedonia's Vlachs are Aromanians (most Vlachs by ethnicity
  are in Shtip, Bitola, Krushevo and Skopje); a few hundred Megleno-Romanian speakers around Gevgelija (Huma) are
  inside the same label and cannot be told apart. Not Serbia's Vlach (rs2022), which is Romanian.
* **Other -> `other`.** Nationally: Bulgarian 1,519, Croatian 958, Russian 437, Serbo-Croatian 416,
  English 378, Greek 182, Slovenian 146, German 140, Polish 133, Romanian 123, Montenegrin 87,
  and 17 more under 65 each, 106 unnamed. Not split by municipality; drawing the national shares
  into each municipality's `other` would change counts, so not done. No indigenous remainder to
  keep apart.
* **Not drawn (gap)**: 132,260 residents counted from administrative registers, who were never
  asked (7.2%; the same category religiondots excludes), and 402 unknown. The register share is
  uneven by municipality (it is how each place was enumerated), so municipal shares are of those
  who answered.
* **Placement by settlement ethnicity** (brief §4.4: moves people only inside the unit the census
  counted them in, so no ask). Each language follows the ethnicity of the same name: Macedonian ->
  Macedonians, Albanian -> Albanians, Turkish -> Turks, Romani -> Roma, Vlachs -> Vlachs, Serbian
  -> Serbs, Bosniak -> Bosniaks; sign language and other follow Kontur population. A hex's weight
  is its Kontur population times its settlement's share of that ethnicity. Nationally the pairs
  are close (Macedonian 1,127,394 speakers against 1,073,299 Macedonians; Albanian 447,001 against
  446,245; Turkish 62,723 against 70,961; Romani 31,721 against 46,433 Roma; Aromanian 3,151 against 8,714 Vlachs), but where they part
  (Torbeshi who declared Turkish or Albanian ethnicity but speak Macedonian; Roma whose mother
  tongue is Albanian or Macedonian) speakers can land in the wrong villages of the right
  municipality. `note_public` says so. Ethnicity is not used to change any count. Scatter: 180
  (municipality, language) rows placed by ethnic mix, 5 on population.
  Spot check on the dots: the median Albanian dot sits in a settlement 92% Albanian, the median
  Macedonian dot in one 85% Macedonian; 2.2% and 0.9% in settlements under 10%. Romani's median is
  7%, because most Roma live in quarters of mixed towns, which are one settlement each.
* **Settlement points** are OSM place nodes from Geofabrik's macedonia-latest.osm.pbf (read
  2026-10-05, then deleted; `data/raw/mk/osm_places.csv` keeps the 2,805 nodes). Overpass answered
  504 on every endpoint that day. Matched by folded Cyrillic name inside the settlement's
  municipality buffered 1.5 km: 1,681 matched to a node; 9 unmatched settlements holding over half
  their municipality (Skopje's urban settlements, `Skopje - Aerodrom` and the like) sit at their
  municipality's population centroid; 91 settlements (12,927 people, 0.70%) not placed, the
  largest Skopje - Saraj 6,265 (Saraj's urban part, under half the municipality) and Negotino -
  Poloshko 3,068 (Vrapchishte). Hexes go to the nearest placed settlement of their own
  municipality, so an unplaced settlement's area is weighted by its neighbours' mix.
* **Colour**: Macedonian was generated at 0.089 OKLab from Bosnian's hand-picked lime, and the
  two share the Bosniak municipalities (Chair, Dolneni, Veles, Petrovec, Studenichani). Pinned at `0.92 0.16 135` (light yellow-green), 0.165
  from Bosnian; diffing languages.json before and after shows no other node moved. Nearest
  remaining: Croatian 0.094, Slovenian 0.121, neither present here. Albanian brown, Turkish pink,
  Romani purple, Aromanian pale blue and Serbian teal were already far apart (all over 0.12).

## 4. Not done

* The 2002 census (religiondots found its municipal tables under `PopisNaNaselenie`) was not
  looked at for mother tongue; the spec rules out a time slider.
* Megleno-Romanian apart from Aromanian: no table splits them.
