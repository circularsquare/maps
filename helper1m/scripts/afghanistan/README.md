# Afghanistan

Two levels on COD-AB v03 boundaries: 34 provinces and 401 districts. Figures
are NSIA's settled-population estimates for the solar years 1403, 1404 and
1405 (starting March 2024, 2025, 2026; stored as 2024, 2025, 2026), so the
viewer's 2026 estimate is NSIA's own 1405 figure and its line runs through
1404 and 1405, two editions of the same projection on the same units.

## Run order

```
python download.py          # GSIA files + COD-PS 2021 into data/afghanistan/raw/
python prep_boundaries.py   # COD-AB v03 -> data/afghanistan/boundaries/adm{1,2}.gpkg
python kontur.py            # Kontur hexes summed per district -> data/afghanistan/kontur_adm2.csv
python make_mapping.py      # only to regenerate nsia_to_cod.csv (it is checked in)
python fetch.py             # -> data/afghanistan/population.csv  (--source codps for COD-PS)
python ../build_country.py afghanistan
python check.py
python language.py          # optional: composition pies -> countries/afghanistan/composition.json
```

All with `C:\Python39\python.exe`. Everything runs in under a minute.

## Language pies (`language.py`): a proxy, province level only

No census since 1979 and no open survey gives language below the national level, so this is
`maps/languagedots`' Afghanistan figure, read as it stands (`data/normalized/af.csv`, mapped by
its `taxonomy/af2007.py`; its record is `languagedots/sources/af.md`). It is the Ministry of
Rural Rehabilitation and Development's provincial profiles (c. 2006-07): the share of each
province's people living in villages where most people speak each language. Everyone in a
village counts under its majority language, so minorities in mixed villages vanish, and
Hazaragi is inside Dari because the profiles never name it. languagedots applies the shares to
NSIA's 1404 settled population, which is helper1m's 2025 column, so every province's pie total
equals its 2025 figure (ratio 1.000 in all 34). Kabul city and Herat have no usable figure of
their own and are split 93/7 Dari/Pashto by the Asia Foundation's 2006 national shares; the
Pamiri languages, Kyrgyz, Parachi, Gawar-Bati and Brahui are published speaker estimates.

- **Province only.** The profiles are per province; nothing splits them by district, so the
  district level draws no pies.
- **Codes.** languagedots' AF01-AF34 are religiondots' join of NSIA to COD-AB's pcodes, the same
  codes as helper1m's v03 provinces; `language.py` checks each against its name (six spell
  differently: Herat/Hirat, Helmand/Hilmand, Paktia/Paktya, Panjshir/Panjsher, Sar-e Pol/Sar-e-Pul,
  Jowzjan/Jawzjan).
- **"Not described"** (3.18 million, pale grey) is kept as a group: the part of each province
  the profile leaves out, and all of Takhar and Kunduz, whose profiles give no usable figure.
  languagedots leaves it undrawn; here it keeps each pie's percentages a share of the whole
  province. Click it in the legend to hide it and see the described part alone.
- **Groups:** Dari 45.3%, Pashto 37.0%, Not described 9.1%, Uzbek 4.4%, Pashai, Turkmen, Brahui,
  Nuristani languages, Balochi, "Balochi or Dari (one figure)" (Kandahar and Helmand),
  "Turkmen or Uzbek (one figure)" (Herat), and Other languages (the profiles' own "other" plus
  Shughni, Wakhi, Gawar-Bati, Munji, Parachi, Sanglechi, Kyrgyz, Ishkashimi, each under 0.1% of
  the country and under 30% of its province). Grouping and colours: `scripts/language_common.py`
  and `scripts/language_colors.csv`, shared with Pakistan.

## Sources

- **NSIA / GSIA, Estimated Population of Afghanistan.** NSIA is now called
  GSIA (General Statistics and Information Authority) and its site moved to
  `https://gsia.gov.af:8443/`. Files used (all under `wp-content/uploads/`):
  - 1405 (2026-27): `2026/08/براورد-نفوس-1405-1.pdf`, 169 pages, uploaded
    2026-08-25. Three earlier uploads of the same month carry the same
    numbers; they differ only in spellings and in which of two Baghlan rows is
    labelled the provincial capital (see Gotchas).
  - 1404 (2025-26): `2023/01/براورد-نفوس-کشور-1404.xlsx`, sheet `نفوس ولایات`
    (Tables 6-73, one per province). The PDF of the same year
    (`2025/09/براورد-نفوس-کشور-سال-1404.pdf`) is also downloaded.
  - 1403 (2024-25): `2023/01/براورد-نفوس-کشور-بابت-سال-1403.xlsx`.
  - 1390-1402: `2024/01/برآورد-نفوس-کشور-بابت-سال-<year>.xlsx`, used only to
    trace where new districts came from (below), not shipped.
  The district tables print English and Dari names. Units are 394 districts,
  29 temporary districts and 34 provincial centres (457). Settled population
  only; NSIA adds a fixed 1.5 million Kuchi nomads at national level with no
  province, and those are left out. Method (preface): Pt = P0 e^rt from the
  2003-05 household listing, base year 1383; no census since 1979.
- **COD-AB v03** (`cod-ab-afg`, AGCHO and NSIA via OCHA, valid 2025-06-01), read
  from `religiondots/data/raw/af/afg_admin_boundaries.geojson.zip`. v03 rather
  than asia1m's v02 because v03's codes are COD-PS's (v02 still files Khulm
  under Balkh as AF2110; v03 has it as AF2008 in Samangan, as NSIA does).
- **COD-PS** (`cod-ps-afg`, UNFPA/Flowminder lineage, "2021 estimates based on
  2017 study"): `afg_admpop_adm2_2026.csv` (religiondots' copy, 401 districts)
  and `afg_admpop_2021_v2.xlsx` (HDX; provinces only, no district table).
- **Kontur Population** AF 2023-11-01 (religiondots' copy), for checks and for
  one split.
- **UN WPP 2024**, medium variant, mid-year: 43,844,111 (2025), 45,047,069
  (2026), via one web search (worldometers / populationpyramid quoting WPP).

## NSIA's 457 units onto COD's 401 districts

`nsia_to_cod.csv` gives a target for every unit in the 1404 tables;
`make_mapping.py` wrote it. 351 units matched automatically: the COD Dari name
found inside NSIA's Dari name (NSIA often prints Pashto and Dari spellings run
together), taken only when exactly one COD district in the province matches,
else the English names letter for letter; provincial centres go to the COD
"Provincial Centre". The other 106 are hand-set in `make_mapping.py`, with a
reason in the CSV's `note` column (19 of them in pools). Kinds:

- **Spelling** (Momand Darah = Muhmand Dara, Sawkai = Chawkay, Chahar Chino =
  Shahid-e-Hassas, Jolga = Khwaja Hejran, ...). Two automatic hits were wrong
  and are overridden: Batikot (also hit Kot) and Farahrod (hit Farah city).
- **Temporary districts with a known parent**: the five parts of Shindand
  (Herat), Sheltan (COD's district is "Shigal wa Sheltan"), Yakawlang No. 2,
  Marja (Nad-e-Ali), Spinghar (Achin), Badpash (Mehtarlam), Kalbad (Imam
  Sahib), Gul Tepa (Kunduz), Aqtash (Khan Abad), Mirzaka (Ahmadaba), Gerda
  Serai (Zadran) from Wikipedia's province pages; Rohani Baba (Zurmat) from an
  AAN election report; Nawamish (Baghran) from Etilaat-e Roz; Laja Mangal,
  Dand, Takhtapul, Delaram from GeoNames points falling in a COD polygon.
- **Districts NSIA created in 1402-1404.** The yearly workbooks show it
  directly: in the year a unit appears, its parent's figure drops by the same
  amount (projections are smooth, so the drop stands out). Single parents:
  Murghab (Feroz Koh), Kantowa (Parun), Want (Waygal), Chahi (Dawlat Abad),
  Kohi-e-Alborz (Chimtal), Seorai (Shinkai), Kshata Shahwalikot and Kshata
  Maiwand, Bughni (Baghran), Sang-e-Atash (Ab Kamari), Bandar (Kohistan,
  Faryab), Dand-e-Ghori (Pul-e-Khumri), Frashgan (Dawlat Shah), Pamir
  (Wakhan), Ferdows (Pashtun Kot). Where a batch of new units came out of
  several old districts at once, the batch is a **pool**: its people are
  split among the old districts in proportion to what each lost that year.
  Pools: five Paktika units from Gomal, Jani Khel, Zarghun Shahr (losses
  54.3k against the five's 54.4k); four Sar-e-Pul units from Kohistanat and
  Sar-e-Pul centre (105.7k against 106.0k); Babaji, Bahramcha (Helmand);
  Chehel Gazi, Khawaja Musa, Khaibar (Faryab); Darah-e-Bom, Tagab Alam
  (Badghis); Alfarooq, Allah Yar (Ghor); Farahrod (Farah).
- **Abshar** (Panjshir, temporary, 13.6k): no source found for its parent.
  Put in Dara, which alone reads 0.39 of COD-PS against 0.74 for the province
  (0.70 with Abshar), and is in the south-east where Wikipedia puts Abshar.
- **Kaldar / Sharak-e-Hayratan**: COD has Hayratan as its own district, NSIA
  counts it in Kaldar. Kaldar is split between the two by their Kontur
  populations; this is the one place Kontur sets a figure.

Other years are paired to the 1404 units by name within the province (for a
provincial centre, the name in brackets), then by the closest name. 1403 lacks
the three units new in 1404, whose parents are in the same COD districts, so
the district figures are on the same footing in all three years.

Every COD district receives at least one unit; every unit lands on a district.

## Checks (`check.py`)

- National settled population against NSIA's own three-year province table
  (1405 PDF, Table 4): 2024 -7, 2025 0, 2026 +1 people (rounding of the
  workbook's unrounded values). All 34 provinces x 3 years within 4 people.
- 1404 -> 1405 growth by district runs 1.017-1.037 except four districts
  where NSIA itself moved people: Burka 1.003, Andar 1.010, Ghazni city 1.025,
  Aliabad (Kunduz) 1.033. Kabul city grows 3.7% a year, Herat and Mazar 3.2%.

**Which source to ship.** National level against WPP 2024 for 2026: NSIA
37.19 million including Kuchi, **0.83** of WPP's 45.05 million; COD-PS 48.60
million, **1.08**; Kontur's 2023 grid holds 42.30 million, 0.96 of WPP 2025. Distribution against
Kontur (scaled to each source's total; for districts, within the province):

| | provinces within 10% | districts within 10% | within 25% | districts over 100k within 10% |
|---|---:|---:|---:|---:|
| NSIA 2026 | 13/34 | 245/401 | 342/401 | 67/88 |
| COD-PS 2026 | 11/34 | 137/401 | 267/401 | 55/153 |

NSIA and COD-PS agree on only 141/401 district shapes (COD-PS scaled to NSIA's
province totals). NSIA is the default: it is the office's own figure, its two
latest years are consistent editions of one projection, and its districts
follow Kontur's grid much more closely. Its weakness is the level: about 17%
under the UN. COD-PS is closer on the level but spreads people within
provinces very differently (Bagrami 657k, Nahr-e-Shahi 401k against Kontur's
~100k and ~66k) and has only one district year.

Worst NSIA districts against Kontur, with the likely reason:
- Shakar Dara 7.1x, Deh Sabz 0.48, Bagrami 0.64 (Kabul): the Kabul city edge.
  Kontur puts few people in Shakar Dara (14.5k) and many in Deh Sabz.
- Injil 3.9x (Herat), Daman 0.32 (Kandahar): Herat and Kandahar city edges.
- Dara-e-Suf-e-Payin 4.8x (Samangan), Marmul 6.2x (Balkh): COD-PS is off
  from Kontur the same way (5.0x, 5.4x), so probably Kontur.
- Hazrat-e-Sultan 0.40 (Samangan), Patoo 3.8x (Daykundi), Turwo 3.3x
  (Paktika), Nili 0.60 (Daykundi): not explained.
- Sar-e-Pul province reads 9.25 against Kontur, and COD-PS 8.54: Kontur has
  almost nobody there (religiondots' `sources/af.md` found the same).

## Known weaknesses

- The whole country reads about 17% under the UN, more where returnees from
  Iran and Pakistan have settled since 2023 (NSIA's projection cannot see them).
- Everything rests on a fixed-rate projection from a 2003-05 listing: a
  district's recent growth is NSIA's assumption, not a measurement.
- Pools and Abshar are an allocation, not a count; Kaldar/Hayratan is split by
  Kontur.

## Gotchas

- **TLS:** gsia.gov.af:8443 omits its intermediate certificate. `download.py`
  fetches it (Certum DV TLS G2 R39) from the URL in the server's certificate
  and verifies against certifi plus that; nothing is fetched unverified.
  Wayback has none of these files.
- **Finding files:** the site is an Angular front end on WordPress; the media
  API (`/wp-json/wp/v2/media?search=نفوس&per_page=100`) lists every upload
  including the Excel workbooks no page links to.
- 1401's workbook is in thousands of people. The 1390-1400 workbooks print a
  Nangarhar and a Badakhshan "Total" that do not equal their units; not
  investigated, those years are not shipped.
- The workbooks hold unrounded figures; rural + urban can differ from total by
  one, and units from their province total by a few. Tolerances in `nsia.py`.
- 1405 PDF: the first uploads labelled the row of 234,991 as "Provincial
  Capital (Baghlan-e-Markazi)"; the 2026-08-25 upload labels it Pul-e-Khumri.
  The 1404 -> 1405 growth confirms the later label (230,112 -> 234,991 for
  Pul-e-Khumri), and pairing by name uses it.
- The PDF's running head ("17 / Population of Afghanistan") followed by the
  column heads parses as a row; `nsia.read_pdf` uses it as the province total
  check. Long tables (Nangarhar, Badakhshan) run onto a page with no caption.

## Tried and not used

- OSM district relations (Overpass, kumi endpoint): 450 relations, none for
  the new or temporary districts. overpass-api.de returned 406.
- GeoNames (religiondots' AF dump): placed 8 of about 40 new or temporary
  districts; the rest have no entry.
- COD-PS for a second district year: its 2021 file has provinces only.
