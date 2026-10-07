# East Asia, Southeast Asia and Oceania: language coverage

Swept 2026-10-03/04. 36 countries, 14 web searches. Several answers came from census files already cached under `religiondots/data/raw/`.

**The overall picture.** Only a few places ask a real first or home language at a fine level: Australia (SA1, about 500 languages, CC BY), Hong Kong (159 large TPU groups), French Polynesia and Guam (commune or village), and Singapore (planning area, from the same data.gov.sg series as religion). New Zealand also covers SA2, but its question lets people list every language they speak, so nobody has a single first language. The large countries mostly ask ethnicity instead: China, Vietnam, Malaysia, Mongolia, the Philippines, Myanmar and Laos. In Southeast Asia ethnicity mostly lines up with language, but in China it does not.

**Best sources found**
- Philippines 2020 ethnicity: about 290 ethnolinguistic groups by province or highly urbanised city (116 units). It comes as a layer in the USCB geodatabase, which is already cached. This is the richest category list in the region.
- Laos 2015 and 2005: 10 ethno-linguistic categories for each of about 8,500 villages, on the open Decide/K4D ArcGIS server (`gis.cde.unibe.ch`), free to use with citation.
- Myanmar: the USCB file on HDX carries GAD 2018 township-profile ethnicity, 33 groups across 330 townships. These are administrative records rather than the census. Rohingya don't appear as a category and the Chinese count is clearly too low.
- Solomon Islands 2019: about 70 first languages are named, but only as national totals. Each language belongs to one area, though, so language-area polygons could place most of them.

**Biggest gaps**
- China: no census has ever asked about language. Han dialect groups exist only as county-level zone labels from the Language Atlas of China (Harvard Dataverse, non-commercial use only) or an unlicensed GitHub county classification.
- Indonesia: the 2020 regency table (USCB) only splits Indonesian / regional / foreign, with no named languages. The detailed 2010 languages are published by province only, or in IPUMS microdata by regency, which needs an account.
- Thailand 2010: the language question is asked per household, and Isan, Northern and Southern Thai are all counted as "Thai".
- Taiwan 2020: language results are open by county only (22 units).
- Cambodia 2019: mother tongue is published nationally only.
- PNG: only literacy by language (four categories). Vanuatu lumps all of its roughly 110 vernaculars into one category.
- Japan and both Koreas have no language question.

**Surprises.** Myanmar's township ethnicity is on HDX. Laos has village-level ethno-linguistic data. Kiribati 2020 asks only "speaks English at home?".
