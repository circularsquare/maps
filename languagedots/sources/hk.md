# Hong Kong: 2021 census, usual spoken language by Large Subunit Group

Session d9e44929-hk, 2026-10-04. Drawn: 7,178,042 people aged 5+ on 1,746 Large Subunit Groups
(LSUGs), 7,170 dots, 14 languages.

## Source

Census and Statistics Department (C&SD), 2021 Population Census. The question is **usual spoken
language**: "the language/dialect a person used in daily communication at home", asked of
everyone aged 5 and over; mute persons are outside it. `how` = "census, 2021, usual spoken
language at home".

Two tables of the same census, because neither alone has both the grain and the labels:

- **`LSUG_21C.zip`** (https://www.census2021.gov.hk/doc/LSUG_21C.zip, data.gov.hk dataset
  `hk-censtatd-census_geo-2021-population-census-by-lsg`): 1,746 LSUGs, about 4,200 people each,
  FIVE groups: Cantonese, Putonghua, Other Chinese dialects, English, Other languages. Complete:
  "-" is nil, and the release's "**" (not released) never appears in a language column. The
  coverage sweep's lead was the 159 Large TPU Groups; the LSUG release is the same five columns
  eleven times finer, and C&SD publishes it for 1,746 groups of at least 1,000 people.
- **The IDDS district table**: the census's Interactive Data Dissemination Service has an open
  GET API (`https://idds.census2021.gov.hk/api/query?cv.LANG1=...&cv.AREA=...&sv.RP=RP_NPER&period=2021&lang=en`,
  no key, no captcha on the API; found in the app's `idds-func-api.js`). Its 15-group language
  classification (`LANG1_5_15G`) opens the two remainders into Hakka, Chiu Chau, Fukien, Sze Yap,
  Shanghainese, Other Chinese dialects / Filipino (Tagalog), Indonesian (Bahasa Indonesia),
  Japanese, Thai, Others. **It is released only by District Council district**: for
  constituency areas, TPU groups and subunit groups the same query returns Cantonese, Putonghua,
  English and "under 5 or mute" and nothing else (tested for DCCA, LTPUG, STPUG and LSBG). Saved
  as `data/raw/hk/idds_dc_lang15.json` with its URL.

Terms: DATA.GOV.HK terms, free reuse with attribution; the IDDS is C&SD's public service.

National totals (aged 5+): Cantonese 6,327,891 (88.2%), English 330,773, Putonghua 165,433,
Others 83,546, Fukien 60,864, Hakka 41,514, Chiu Chau 37,621, Filipino 29,413, Sze Yap 26,851,
Other Chinese dialects 26,710, Indonesian 24,244, Shanghainese 11,011, Japanese 8,704, Thai 3,467.

## Checks (`python sources/hk_census.py`), none a tolerance

- The 1,746 LSUGs' five groups and populations each sum to the release's own land total
  (7,411,945 people; under 5 or mute 233,903).
- **Second release, per unit**: the LSUGs rebuilt into the 159 Large TPU Groups by the TPU at the
  head of their names equal `LTPUG_21C.CSV` in all five groups exactly. Groups joined by an LSUG
  that straddles them, or a TPU the LTPU release splits by subunit, are compared as one block:
  153 blocks, 148 of them single groups.
- The IDDS 15 groups collapse onto `DC_21C.CSV`'s five groups exactly for all 18 districts, and
  the districts' totals equal the LSUG land total.

## How the remainders are shared out (`countries/hk.py`)

Cantonese, Putonghua and English are the LSUG table as published (`measured`). Each LSUG's
"Other Chinese dialects" and "Other languages" are shared by its district's mix inside that
group (`derived`), the US entry's arrangement (tract groups by PUMA mix).

**LSUGs do not nest in districts.** Each LSUG's districts are weighted by where Kontur puts its
people (area where Kontur has nobody), 0.5% floor: 83 LSUGs get two or more districts
(`data/geo/hk/hk_lookup.csv`). Witness: the LSUGs' five groups, shared by these weights, miss
DC_21C's district table by 40,380 people of 7.18M (0.6%); each LSUG wholly in its main district
misses by 45,289. The same miss comes out on HAD's district file and on the census's own 452
constituency areas dissolved to districts, so it is the LSUG and district polygons disagreeing
about a few big straddlers (one LSUG's ~20,600 Cantonese speakers are counted in Kwun Tong while
its polygon is mostly in Sai Kung), not a join error. The district mix is used only for the two
remainders, so this moves a few hundred derived people between district mixes.

Drawn national totals against the IDDS table: Others 83,466 (83,546), Fukien 61,038 (60,864),
Hakka 41,589 (41,514), Chiu Chau 37,613 (37,621), Filipino 29,403 (29,413); all within 0.3%.

## Labels (`taxonomy/hk2021.py`, `taxonomy/tree.d/hk.txt`)

- Putonghua on `mandarin`; Fukien (Hokkien) on `min_nan`; Shanghainese on `wu`; Filipino
  (Tagalog) on `tagalog` (the UK's choice for "Tagalog or Filipino").
- New nodes: `sinitic.teochew` "Teochew (Chiu Chau)" (Glottolog chao1238, a dialect of Min Nan)
  and `sinitic.siyi` "Sze Yap (Taishanese)" (Glottolog siyi1236, a dialect of Yue). Both sit
  directly under `sinitic`, beside `min_nan` and `cantonese`: a child under `min_nan` would turn
  it into a group node and wash out the Hokkien already drawn elsewhere.
- "Other Chinese dialects" on `sinitic` (Chinese, language not named). "Others" on the root
  `other`: IDDS's code list behind it (41, 43, 46-49, 51-54, 59-67, 69, 92) is everything not
  listed, Urdu, Nepali, Hindi, Punjabi, Korean, Vietnamese and European languages among them, and
  nothing finer is published at any level.

## Not drawn (`gap`)

Children under 5 and mute persons, 233,903 (3.2%); the marine population, 1,125, which the census
places in no district or LSUG (land total 7,411,945 against 7,413,070 enumerated).

## Geography (`python sources/hk_geo.py` -> `data/geo/hk/`)

- **Units**: C&SD's own LSUG boundaries from the CSDI portal (`censtatd_rcd_1635933282224_58228`,
  layer `LSUG_21C`, GeoJSON, EPSG:4326), keyed by the table's `lsbg` code; 1,746 polygons,
  matched both ways. Median 0.059 km2, p10 0.010, p90 0.75; 1,111 km2 in all.
- **Placement**: Kontur 2023 (religiondots' `data/raw/hk/kontur_population_HK_20231101.gpkg.gz`,
  read-only, unpacked into `data/geo/kontur/`) cut by the LSUGs and districts; each hex's people
  shared over its pieces by area, **divided by the whole hex**. Malta's rule (divide by the land
  pieces) was tried first and put 78% of Central's LSUG 12102L on a 7,357 m2 strip of harbourfront
  at 5.3 million/km2: Kontur gives harbour hexes that are mostly water the full density of the
  waterfront. 450 slivers under 50 m2 dropped (62 Kontur people); 6 hexes (7,760 people) touch no
  LSUG. 5,531 pieces, median 2 per LSUG, 610 LSUGs in one piece; every LSUG has a populated piece.
- **Kontur against the census per LSUG** (normalised): p10 0.22, median 0.52, p90 3.04; log
  correlation 0.481 against a best of 0.067 over 200 shuffles. The band is wide because a 0.06 km2
  unit inside a 0.74 km2 hex can only get its area share; Kontur never decides how many dots an
  LSUG gets, only where inside it they go.
- **Grid floor**: median 2 pieces per unit, which spec §8.2e's warning calls too few to weight.
  Accepted, as South Africa's: the pieces are the units' own shape and the units are already
  finer than a hex, and uniform placement would put the 82 LSUGs over 2 km2 (293,000 people over
  831 km2, three quarters of Hong Kong's land) across country park.
- **Kontur's cap**: religiondots reviewed Hong Kong's 62 cap blocks as `real` (2026-09-14).
  `hk_geo.py` runs `kontur_cap.apply` on the uncut hexes with the merged registry and stops if
  any block is unregistered or any weight changes. The cut layer is named `hk_cut.gpkg`, not
  `*_hexes.gpkg`, so the scatter does not re-run the check on pieces: there the block-finder,
  spacing by the median distance between neighbouring pieces, splits the 62 blocks into 249
  fragments that would each need a registry row saying the same thing. This also silences the
  scatter's grid-floor warning (accepted above).
- Water clip: 703 pieces clipped (0.36% of their area was sea); 3 LSUGs left unclipped by the
  over-95% rule.

## Colours

Teochew's generated colour (#7954de) sat on top of Mandarin's (#8258c9), and the two share
districts: hand-picked `0.66 0.11 222` in the fragment (teal-blue, #2da1c2). Someone else retuned
`sinitic`, `mandarin` and `min_nan` in this fragment during the session; left as they are.

## Calls someone might reverse

- LSUG grain with district mixes for the remainders, over the district table alone (18 units,
  every label measured).
- The Kontur-weighted district shares for the 83 straddling LSUGs.
- Whole-hex Kontur sharing (South Africa) over land-only sharing (Malta), for the harbour.
- `hk_cut.gpkg` named to keep the scatter's cap check off the pieces; the check runs on the
  uncut hexes in `hk_geo.py` instead.
- Teochew and Sze Yap as their own nodes under `sinitic`, not under `min_nan` / a Yue group.
