# Vietnam (vn)

Drawn 2026-10-05 by session edd42a8c-vn. Ethnicity read as language under AGENT_BRIEF section 2
(Anita, 2026-10-05), with a measured retention correction. Every row `derived`.

## Sources

| | |
|---|---|
| counts | 2019 Population and Housing Census, "Ket qua toan bo" volume, Bieu 2 (Table 2): population by ethnic group x urban/rural x sex, country, 6 regions, 63 provinces. PDF pages 44-210. `https://www.nso.gov.vn/wp-content/uploads/2019/12/Ket-qua-toan-bo-Tong-dieu-tra-dan-so-va-nha-o-2019.pdf` (9.5 MB; gso.gov.vn no longer resolves, the office is now NSO at nso.gov.vn) |
| retention | 2024 survey of the socio-economic situation of the 53 ethnic minorities (Dieu tra 53 DTTS 2024; NSO and the Committee for Ethnic Minority Affairs, published June 2026), Bieu 3.9, PDF pages 178-179: share of each minority's households by the language mainly used in family communication (own ethnic language / Vietnamese / another ethnic language). `https://www.nso.gov.vn/wp-content/uploads/2026/06/Ket-qua-Dieu-tra-thu-thap-thong-tin-ve-thuc-trang-kinh-te-xa-hoi-cua-53-dan-toc-thieu-so-nam-2024.pdf.pdf` |
| geography | religiondots' 400m Kontur hex layer for the 63 provinces (`RD_GEO/vn/vn_grid_400m.gpkg`, `vn_lookup.csv`), read-only, re-keyed here into urban and rural halves |

Scripts: `sources/vn_census.py` (-> `data/normalized/vn.csv`, `vn_retention_2024.csv`),
`sources/vn_geo.py` (-> `data/geo/vn/vn_hexes.gpkg`), `taxonomy/vn2019.py`,
`taxonomy/tree.d/vn.txt`, `countries/vn.py`.

## Which retention source, and what was searched

- **2019 census**: no language question (asks ethnicity, and literacy in Vietnamese). The 2019
  and 2009 volumes have no language table.
- **2019 survey of the 53 minorities** (GSO, July 2020; parts 01 and 02 on nso.gov.vn): the
  person-level question "can speak an ethnic language" exists, but the publications give only
  the overall 88.7% (age 5+), by age, and Ngai's 30.5%. The UN Women / CEMA figures booklet
  (2021, `data/raw/vn/unwomen_53em_figures_2021.pdf`, p.88) says the 88.7% is "any minority
  language", not the group's own, and prints no per-group table. Not usable.
- **2015 survey** (UBDT/UNDP report 2017): no language section. Not kept.
- **2024 survey, Bieu 3.9**: per group, household main home language. Used. This is closer to a
  home-language question than a retention share, which is what this map draws.

## Method

1. Table 2 gives each group's urban and rural count per province.
2. Each minority's count is split on its national Bieu 3.9 shares into its own language,
   Vietnamese, and "another minority language" (`seasia_other`, unnamed: the survey does not say
   which). Integer rounding keeps every unit's total to the person. Kinh are drawn as Vietnamese
   whole; foreigners (3,553) on `other`; not stated (349) not drawn.
3. Placement: within each province the densest Kontur hexes holding the census's urban share of
   the population are the urban half, the rest rural; urban and rural counts are placed on their
   own halves by Kontur population. A density proxy for the urban boundary (official phuong and
   thi tran boundaries are not used). Moves people only within their province (brief 4.4).

## Checks (all asserted in the scripts)

- Table 2: 70 blocks x 56 rows; every block's rows sum to its total row in all 9 columns;
  total = urban + rural = male + female on every row; 63 provinces sum to the country for every
  group; national 96,208,984.
- Bieu 3.9: 53 groups; own + other = 100 and Kinh + other ethnic = other within 0.15. Its
  spellings `Đê`, `Khơ mú` matched to Table 2's `Ê Đê`, `Khơ Mú`.
- Province names join religiondots' lookup 63/63 both ways, distinct codes.
- After the split, every province half equals the census; vn.csv sums to 96,208,984.
- check_country: ok, 96,208,635 drawn (less the 349 not stated), 55 languages, 126 units.
- Scatter: 96,184 dots; 3 Kontur cap blocks already registered as real cores.

## Result

Vietnamese 88.6% (85.2M: 82.1M Kinh plus 3.15M minority people in Vietnamese-speaking homes,
22.3% of minorities person-weighted). Then Tai (Thai) 1.62M, Hmong 1.36M, Muong 1.08M, Tay
1.05M, Khmer 1.00M, Dao 0.72M, Nung 0.61M, Jarai 0.50M, Rade 0.38M, Chinese 0.32M. Another
minority language: 0.28M.

## Calls someone might reverse

- **Household shares applied to persons, one national share per group.** The survey's urban
  51.5% vs rural 79.6% own-language gap is not applied per group (the table has no group x
  urban cell); applying the total gap would be a model, not a source.
- **"Another minority language" on `seasia_other`**, not guessed (Xinh Mun 41%, La Chi 41%,
  La Ha 40%, Co Lao 53%, O Du 96% are big; likely Tai or Tay, but nothing says).
- **Hoa on `sinitic`** (Cantonese, Teochew, Hakka, Hokkien unsplit), as every country files it.
- **Ngai on Hakka**; **San Diu** its own leaf under Chinese (a Yue-like variety; no Glottolog
  leaf). **San Chay** on Cao Lan, though the San Chi half speak a Chinese variety.
- **Dao** its own leaf, `hmongmien.dao` "Dao (Iu Mien and Kim Mun)": no source splits them.
  Not on `hmongmien.iu_mien`, which is one language.
- **Thai** is `kradai.tai_vietnam` "Tai Dam and Tai Don", not Thailand's `kradai.thai`.
- **Lo Lo** (Mantsi) under Loloish though Glottolog puts Mondzish beside Ngwi; readers know the
  Lo Lo as a Yi people.
- Vietic Muong, Tho, Chut and Palaungic Khang, Mangic Mang are leaves directly under
  Austroasiatic, beside Vietnamese, which already sits there.
- Colours hand-picked: Tay, Tai, Nung, Muong, Dao, Bahnar, Sedang, Koho, Rade, Khmu, Roglai (was
  Cham's teal in Ninh Thuan), San Diu (was Muong's pink).

## Room for improvement

- A census or survey asking each person's first language, by province or district, would replace
  the proxy. The 2019 survey asked person-level ability to speak an ethnic language; a per-group
  (and per-province) tabulation of that, or the 2024 Bieu 3.9 by province, would sharpen it.
- District grain: Table 2 is province only. IPUMS holds the 2019 census sample with district
  geography (gated, and the IPUMS account is blocked).
- Urban boundaries: official phuong/thi tran polygons instead of the density proxy.

## Moved from countries/vn.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- Those shares, one per group, are applied to every province: 42% of Tay households and 57% of Hoa households speak Vietnamese at home, against 2% of Hmong and Jarai households.
