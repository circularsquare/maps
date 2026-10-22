# Myanmar (mm): record

Drawn 2026-10-05, session edd42a8c-mm. Ethnic nationality per township from GAD administrative
records, read as language under AGENT_BRIEF §2's ethnicity rule (Anita, 2026-10-05); every row
`derived`. 47,798,404 people, 325 townships, 29 nodes; plus 577,182 Rohingya in Rakhine read from
GAD's religion table (added the same day, session edd42a8c-fix, §3): 48,375,586 people, 30 nodes,
48,359 dots.

## 1. Source and vintage

- **Table:** General Administration Department (GAD), *2018 Township Profiles* (published 2019),
  Table 14 "Ethnic Nationalities Living" (Burmese: နေထိုင်သည့်တိုင်းရင်းသားလူမျိုးစုများ), figures as of
  1 April 2017. Administrative records kept by township offices: not a census, not self-report.
- **Transcription used:** U.S. Census Bureau, "Burma Subnational Population and Housing Data
  Tables with Administrative Boundaries" (HDX, CC BY), `burma_uscb_202003.xlsx` sheet `Ethnicity`
  (USCB layer `MM_ETHNICITY_2018record_uscb_202003`), plus the same dataset's `burma.gdb.zip` for
  boundaries. Both in `data/raw/mm/`; `sources/mm_gad.py --fetch` re-downloads.
- 31 named groups plus "Foreign" and "Other". USCB's data dictionary keeps GAD's Burmese field
  name for every column, which is how the odd labels below were identified.
- MIMU publishes the same GAD township profiles as PDFs, one per township; USCB's sheet is the
  machine-readable form of exactly that table, so MIMU was not re-transcribed.

## 2. What else was looked for (none usable)

- **2014 census**: asked ethnicity (135 codes); the ethnicity results were never released. No
  language question.
- **2024 census**: reported (Wikipedia, "Census in Myanmar") to have asked main language at home.
  The 2024 Union Report (DOP, `2024mphc_unionreport_en_26nov.pdf`, fetched 2026-10-05, 146 pp.)
  has no language table at all; the only mention of language is interpreters for enumerators.
  The census was only partly carried out (much of the country was not enumerated during the war).
  **If DOP ever publishes the language table by township, it replaces all of this.**
- **2019 Inter-censal Survey**: topics are demography, education, labour, fertility, migration,
  disability, housing, WASH; no ethnicity or language tables found in its published listings.
- **DHS 2015-16** (FR324, 485 pp.): no table of native language or interview language. Microdata
  needs DHS registration (off limits, memory note).
- No retention study with numbers found. Literature (Jenny 2015, *The Mon language in Thailand
  and Myanmar*; Minority Rights Group) says most ethnic Mon speak and read only Burmese, but
  gives no share.

## 3. Checks (`python sources/mm_gad.py`)

- 330 townships under 15 states/regions; the table is blank for exactly five: Mongmao, Pangwaun,
  Pangsang, Narphan (Wa Self-Administered Division) and Mongla (Special Region 4). GAD has no
  office there. 2014 census population of the five: 433,000. Undrawn, in `gap`.
- Townships sum to the national row exactly in every category; national 47,809,979.
- 28 townships' categories do not add to their printed total (largest Yamethin, -8,710; net
  -11,575 nationally). The categories are drawn, not the printed totals.
- **Unrecorded people.** The same sheet has GAD's religion table for the same townships and date.
  Religion total minus ethnicity total, where positive: 1,204,383 nationally (written per township
  to `data/normalized/mm_unrecorded.csv`). By state: Rakhine 584,142 (GAD's Islam there: 588,353),
  Yangon 271,848, Bago 119,329, Mandalay 60,440, Kachin 50,226, Ayeyarwady 40,698, Shan 37,321,
  others under 11,000. In Rakhine this is the Rohingya: Buthidaung records 46,067 by ethnicity
  against 308,259 by religion (Islam 260,635). (Maungdaw's 2017 religion total, 107,271, is far
  below its 2014 census figure; GAD's own count there is incomplete as well.) Elsewhere it is probably mixed (Muslims, people of Indian and
  Chinese descent), but nothing says.
- **Rakhine's shortfall is drawn as Rohingya** (Anita's ruling, 2026-10-05; `countries/mm.py`
  `_rohingya`). Per Rakhine township, min(religion total - ethnicity total, Muslims in the
  religion table): the religion table's Muslims missing from the ethnicity table. 12 townships,
  577,182 people (Buthidaung 260,635, Sittwe 90,914, Maungdaw 72,266, Kyauktaw 42,778, Pauktaw
  32,568, Myauk U 26,475, Minbya 26,153, the rest under 10,000); the 6,960 of Rakhine's
  shortfall above its Muslim count stay undrawn. Rakhine State totals: 2,670,819 by religion,
  2,086,677 by ethnic group. Rows `derived`, on node
  `indoeuropean.indoaryan.eastern.rohingya` (the id ca.txt and sa.txt already use; Glottolog
  rohi1238, Bengali-Assamese, which this tree keeps flat under Eastern), colour pinned in mm.txt
  to an olive (0.66 0.14 100), clear of Rakhine's pink. Rakhine's Kaman Muslims are a recognised
  group and so in the ethnicity table, not in the shortfall. **The non-Rakhine shortfall
  (620,241; Yangon 271,848) stays undrawn**: nothing says it is Rohingya.
- Placement join: GEO_MATCH codes shared by table and polygons, asserted 325 into 330 with the
  five blank townships left out by name. Kontur against GAD per township: log r = 0.942, best of
  500 shuffles 0.173; Kontur/GAD nationally 1.121 (Kontur is 2023 and includes the unrecorded);
  normalised p10 0.78, median 0.96, p90 1.33; 8 of 325 outside a factor of 3.

## 4. Mapping calls (`taxonomy/mm2018.py`, `taxonomy/tree.d/mm.txt`)

- **Group nodes** (drawn washed out as "language not named"), as the supervisor directed for
  broad national races: Karen → `sinotibetan.karen`; Chin → `sinotibetan.kukichin`; Naga →
  `sinotibetan.naga`; Chinese → `sinotibetan.sinitic`; Kachin → new `sinotibetan.kachin`
  ("Kachin languages", children Zaiwa, Lhaovo, Rawang, Lachik; Jingpho stays at
  `sinotibetan.jingpho` where other fragments put it); Kayah → new `sinotibetan.karen.kayah`
  (Glottolog's Kayah family, children Eastern and Western Kayah Li). The children exist only so
  the viewer treats the two as groups; nobody is drawn on them. Kachin is not a genealogical
  group (Jingpho-Luish, Burmish and Nungish members); it is the conventional cover term.
- **Shan → `kradai.shan`**, a leaf, not a group: GAD lists Pa'o, Palaung, Danu, Intha, Lahu, Wa,
  Akha, Kokang and others separately, so what is left under "Shan" is mostly Shan speakers.
  Eastern Shan's Tai Khun and Tai Lue are inside it and cannot be split.
- Identified from GAD's Burmese field names: "Liz" (လီရှော, Lishaw) and "Kho Lone Li Shaw" are
  Lisu, merged with Lisu as spelling variants; "Myaing" (မြောင်ဇီး, Myaungzi) is the Hmong of
  northern Shan → `hmongmien.hmong`; "Htanot" (ထနော့, Kalaw only) is Danau (dnu, Palaungic,
  Glottolog point at Kalaw) → new `austroasiatic.danau`; "Yinn (Kya and/or Net)" in Nansang and
  Monghsu is Yinchia (yin, Palaungic) and Yinnet → new `austroasiatic.yinchia`; "Kanan" is Ganan
  (Luish), in Banmauk beside Kadu.
- **Kokang and Mong Wong → `sinotibetan.sinitic.mandarin`**: both speak Yunnanese
  (Southwestern Mandarin). Wikipedia's Mong Wong article: Yunnanese-speaking, recognised on ID
  cards as "Mong Wong (Bamar)".
- New leaves: Pa'o and Kayan under Karen; Danu, Intha, Taungyo under Burmish (Glottolog files
  Danu and Intha as dialects of Intha-Danu, a Burmese variety); Akha under Loloish; Kadu and
  Ganan directly under Sino-Tibetan (no Luish group added for two small leaves); Moken under
  Austronesian; `other.indian` ("Indian, no language named", 1,921).
- "Foreign" (28,025: 18,317 in Mu Se on the Chinese border, 9,562 in Thandwe) and "Other"
  (594,341) → `other`. "Other" is 79% of Laukkai and 99.8% of Konkyan, the Kokang
  Self-Administered Zone's townships, where the people are presumably Kokang; not moved, since
  GAD did not say so.
- No retention correction anywhere: no source gives a share for any group (§2).
- Colours: Kachin set to a yellow (0.82 0.14 90) clear of Shan's green; Pa'o a deep blue
  (0.56 0.12 232) clear of Shan and of Intha round Inle Lake. The rest are generated.

## 5. Placement (`sources/mm_geo.py`)

Kontur 2023 400 m hexes keyed to USCB's ADM3 polygons (MIMU 2019 townships) by centroid. Three
downtown Yangon townships smaller than a hex (Pabedan, Pazundaung, Seikkan) got no hex and are
placed on their own polygons, weighted by GAD's count. religiondots draws Myanmar on 15 states,
so its layer is not reused.

**Kontur cap blocks, not settled.** Eight blocks at Kontur's cap in the Ayeyarwady delta and
rural Yangon are `unreviewed` in religiondots' registry, where each was under 5% of a state. At
township grain they hold 21-60% of a township: Mawlamyinegyun 60% (6 hexes, of 321,977),
Ma-ubin 50%, Kayan 49%, Bogale 48%, Taikkyi 31%, Twantay 28%, Labutta 24%, Danubyu 21%. Several
sit with no town within 20 km; they look false and should be `capped`. They cannot be fixed
from languagedots: a languagedots row with a different status for the same block makes
`kontur_cap.apply` stop on "rows that disagree", and religiondots' file is read-only here. They
are drawn as Kontur has them, with a warning. Needs either religiondots' rows promoted or rdlink
letting a languagedots row win.

## 6. Room for improvement

A language table by township would replace all of this: the 2024 census's home-language question
if DOP publishes it, or the 2014 ethnicity table if ever released. Short of that, any survey
crossing ethnicity with home language would give retention shares (Mon above all; also Karen in
the delta, Shan outside Shan State, Chinese and Indian descent). The 620,000 unrecorded outside
Rakhine and the five Wa and Mongla townships are the gaps; the Rohingya figure is a residual of
two office tables, not a count of anyone.

## 7. Coverage gap: the Wa and Mongla townships (2026-10-19, session eca0185b-gap)

The five townships GAD has no office in (section 3) are now drawn, `modelled`: **433,000 people**,
so Myanmar is 48,808,586 drawn (was 48,375,586). Files: `sources/mm_wa.py` (counts, mixes, and
`--build` for the hexes), `data/geo/mm/mm_hexes.gpkg` (2,527 hexes added; 330 units), `countries/mm.py`.

**People.** The 2014 census table of the same townships, in USCB's transcription already in
`data/raw/mm/burma_uscb_202003.xlsx` (sheet Age-Sex, `BTOTL`): Mongmao 70,683, Pangwaun 97,097,
Pangsang 105,972 (incl. Mankan subtownship), Narphan 116,180, Mongla 43,068 = 433,000. Not grown to
2017 (the rest of the map is April 2017 records) and not checked against the 2024 census, whose
township tables for these areas are partial. Kontur against the census: Mongmao 1.22, Pangwaun
1.07, Pangsang 1.44, Narphan 0.93, Mongla 1.13, so the grid is usable as a within-township weight.

**Languages.** No count of ethnicity or language exists for them; the 2014 ethnicity tables were
never released. Ethnic shares read as language, as everywhere in this country; rows `modelled`
(AGENT_BRIEF section 2, no measured source).
- *Wa Self-Administered Division* (Mongmao, Pangwaun, Pangsang, Narphan): World Food Programme,
  "Wa Self-Administered Division" (docs.wfp.org/api/documents/WFP-0000107118/download): 558,000
  people, 16 minority groups, "ethnic Wa account for 70 percent of the total population, with
  Lahu and Shan as the second and third largest". Mix used, one for all four: Wa 70, Lahu 12, Shan 8,
  Akha 4, Mandarin 3 (Han and Kokang Chinese), Lisu 1, Palaung 1, other 1. Only the 70% and the
  Lahu-then-Shan order are cited; the rest is judgement. The two tabulated townships of the same
  division, Hopan (Kokang 38, Wa 33, Shan 17) and Makman (Wa 47, Lahu 42, Shan 11), are on the
  division's edge and under government offices; not representative of the army-held core, not used.
- *Mongla* (Special Region 4, NDAA): Burma Link's page on Special Region 4, "approximately 35% of the
  population are Wa and the rest are of Shan, Lahu, Akha, Lisu and Loi ethnicities"; RFA, 17 Sep 2019,
  "In Myanmar's Remote Mongla Region, Mandarin Supplants the Burmese Language": 13 minority
  languages besides Burmese (Shan, Akha, Lahu, Lweh named), children mainly speak Chinese. Mix: Wa 35,
  Shan 25, Akha 15, Lahu 12, Lisu 5, Mandarin 5, other 3 (Loi, with no node here). The Wa 35% is cited;
  the rest is judgement. Burma Link is an advocacy site and the figure is undated.
- Wa people in these townships who speak Chinese at home are not separated: no source gives it
  (Mandarin is the Wa State's working language, and children are taught in it).

Result added: Wa 288,026, Lahu 51,960, Shan 41,962, Akha 22,057, Mandarin 13,851, Lisu 6,053, other
5,191, Palaung 3,899. Wa goes from 117,483 (GAD's records elsewhere in Shan State) to 405,509 drawn nationally.
The heartland had no records, which is why the recorded Wa count was so low.

**Placement.** Their own Kontur hexes (the layer had left them out by name, `NO_TABLE` in
`sources/mm_geo.py`); `python sources/mm_wa.py --build` appends them to `mm_hexes.gpkg` and is
idempotent. `sources/mm_geo.py` was not edited: re-running it alone would drop the five units again,
so run `mm_wa.py --build` after it.

**Calls someone might reverse:** 2014 figures not grown; one mix for all four Wa townships; the
Wa and Mongla non-Wa shares. **Room for improvement:** the 2024 census township language table if
DOP publishes it, MIMU's Wa SAD baseline, a language survey of the Wa State.
