# Qatar (qa): record

Drawn 2026-10-05 (session edd42a8c-gulf). Method shared with the other Gulf states:
`sources/gulf.md`. 2020 census, 8 municipalities, 2,846,118 people, 99 nodes, every row
`derived`. 2,797 dots at 1:1,000; 21 rings.

```
python sources/qa_geo.py
python sources/qa_build.py
python taxonomy/build.py
python tools/check_country.py qa
python scatter.py --country qa
```

Files: `sources/qa_geo.py`, `sources/qa_build.py`, `sources/gulf_mix.py`, `taxonomy/qa2020.py`,
`taxonomy/tree.d/qa.txt`, `countries/qa.py`, `data/raw/qa/Census_Final_Results.xlsx`
(npc.qa, `.../census2020/results/Documents/Census_Final_Results.xlsx`, fetched with `curl -k`),
`data/geo/qa/qa_hexes.gpkg`, `data/normalized/qa.csv`.

## Tables

- **Population**: Table 1 (8 municipalities x sex), Table 2 (87 zones x sex). Asserted: zones
  summed by their COD-AB municipality equal Table 1 exactly (that is also how COD-AB's
  municipality codes were decoded: 001 Al Daayen, 002 Al Khor and Al Thakhira, 003 Al Rayyan,
  004 Al Shamal, 005 Al Wakra, 006 Doha, 007 Umm Slal, 008 Al Sheehaniya).
- **Qataris**: Qatar publishes no citizen total. Table 32 counts Qataris aged 10+ by municipality
  and sex (246,256). Under-10s: each municipality's under-10s by sex (Tables 4, 5) x the national
  Qatari share of 10-14 year olds (Table 20's 37,770 of Table 3's 128,748, **29.3%**). Result
  **340,298 Qataris (12.0%)**, in line with the outside estimates usually quoted (about 300,000
  to 380,000). The share is national, so a municipality of migrant families gets too many Qatari
  children and a Qatari suburb too few.
- **Nationalities**: none published by the census. UN DESA International Migrant Stock 2020,
  Qatar, men and women separately (2,182,000; `Others` 32,601 on `other`). DESA's Qatar figures
  are an estimate with a thin base. Each municipality's non-Qatari men and women take their sex's
  mix (Al Sheehaniya: 148,384 men, 12,856 women).
- Indians: Keralites 196,039 (KMS 2023, 9.1%) = 28.5% of DESA 2020's Indians.

## Geography

religiondots built Qatar on the ten municipalities of 2004 (`../religiondots/sources/qa.md` §5).
`qa_geo.py` re-keys its Kontur 2023 pieces (each carries its COD-AB zone) to the 2020
municipalities, and scales each zone's pieces to that zone's 2020 population (Table 2), keeping
Kontur's weights inside the zone. Five zones religiondots gave no piece because they were empty
in 2004 (46 Al Thumama 22,284; 49 Hamad airport 2,297; 50 3,025; 58 221; 98 4) are placed on the
whole zone polygon, evenly. Every 2020 zone with people is a COD-AB zone (asserted).

## Results

Gulf Arabic 12.6%, Bengali 11.7%, Malayalam 7.7%, Egyptian Arabic 7.3%, Hindi 7.1%, Nepali 5.0%,
Punjabi 4.9%, Sinhala 4.6%, Tamil 3.9%, Pashto 2.9%.

## Room for improvement

- A citizen count by municipality and age (the census has it; only 10+ is printed).
- Nationality by anything: the Planning and Statistics Authority publishes none; Qatar's labour
  force survey has nationality groups, not opened.

## The grey wedge (2026-10-06, session 5d7dac7e-oth)

Anita: the grey "other languages" segment is big in Qatar. Measured on the dots: 43% of the
country's pie is the viewer's fold of everything past its 7 largest languages (`index.html`,
PIE_K = 8): 72 languages, nearly all named (Sinhala 4.7%, Tamil 3.9%, Pashto, Levantine Arabic,
Tagalog, Bhojpuri, Telugu, Cebuano...). Qatar's languages are many and none is large, so the
fold is bigger here than anywhere else looked at. The data's own unnamed rows are 1.9%: `other`
1.4% (UN DESA's `Others`, 32,601, which DESA does not break down, plus the 1% tails home mixes
leave on `other`) and Indo-Aryan 0.5% (Indian states' unnamed remainders). Nothing finer exists:
Qatar publishes no nationality table at all. **Not changed.** The fix is in the viewer (family
wedges or a larger PIE_K; `followups.md`).

## Placement inside units (2026-10-06)

Citizens and foreign residents are now placed apart inside each unit (session 5d7dac7e-gulf): foreign dots lean to dense hexes and fill OSM industrial land and labour camps, citizens take the rest; one rule for the six Gulf states, fitted on Kuwait's areas and Oman's wilayat. Method, fit, data searched: `sources/gulf_place.md`. Doha's Industrial Area 90.3 -> 95.8% non-Qatari; municipalities otherwise barely move (Qatar's zones carry no citizenship).
