# Oman (om): record

Drawn 2026-10-05 (session edd42a8c-gulf). Method shared with the other Gulf states:
`sources/gulf.md`. NCSI register, end 2024, 63 wilayat on religiondots' 61 units, 5,268,072
people, 97 nodes, every row `derived`. 5,221 dots at 1:1,000; 33 rings.

```
python sources/om_build.py --fetch     # --fetch re-runs om_extract.py
python taxonomy/build.py
python tools/check_country.py om
python scatter.py --country om
```

Files: `sources/om_extract.py`, `sources/om_build.py`, `sources/gulf_mix.py`,
`taxonomy/om2024.py`, `taxonomy/tree.d/om.txt`, `countries/om.py`, `data/raw/om/`,
`data/normalized/om.csv`.

## Tables

All through religiondots' `sources/om.py` (record `../religiondots/sources/om.md` §2, §4), loaded
by path in its own process by `om_extract.py`, its parsers and checks run, read-only:

- NCSI *Statistical Year Book 2025*: Table 7-2, Omanis 2,984,793 and expatriates 2,283,279 by
  wilaya; Table 8-4, expatriate workers by governorate and sex; Tables 17-4 and 18-4, workers by
  nationality and sex (Bangladesh, India, Pakistan, the Philippines, Egypt, Myanmar, Sri Lanka,
  Tanzania, Sudan, Jordan, "Other Arabs", "Other nationalities").
- GLMM's copy of NCSI's mid-2018 population by nationality and sex, and 2018 workers.
- The construction is religiondots' to the letter: per governorate, male workers, female workers
  and dependants (expatriates less workers), each at its national nationality mix; "other
  nationalities" women at 2018's Uganda, Indonesia, Ethiopia, Nepal, men at the named men's mix;
  dependants at mid-2018 population less 2018 workers. Each wilaya takes its governorate's mix.
- "Other Arabs" (3,164) on `afroasiatic.arabic`: Arab, nationality unnamed.
- Myanmar (33,083, 99% women domestic workers) at Myanmar's drawn mix, not Rohingya
  (`sources/gulf.md` §2). Tanzanians (30,610) at Tanzania's mix (Swahili mostly).
- Indians: Keralites 137,874 (KMS 2023, 6.4%) = 18.1% of the layer's 760,295 Indians.

## Results

Omani Arabic 56.7%, Bengali 15.3%, Hindi 4.2%, Punjabi 3.9%, Pashto 2.6%, Malayalam 2.6%,
Egyptian Arabic 1.8%, Tamil 1.5%, Telugu 1.1%, Urdu 1.0%. Female workers: Bengali 11%, Burmese
9%, Swahili 6%.

## Calls and gaps

- **Omanis all on Omani Arabic.** Large minorities of citizens speak something else: Baluchi
  (Omani Baluch, the Batinah coast and Muscat), Jibbali, Mehri, Harsusi, Bathari and Hobyot in
  Dhofar and Al Wusta, Kumzari in Musandam, Swahili among families from East Africa, Lawati
  (Khoja Sindhi) in Muttrah. No source counts any of them; sa's no-guessing rule.
- **Pakistanis** at the BEOE province mix; Oman's Pakistanis are probably far more Baluch (Makran
  coast) than that mix (Balochistan under 1%) shows. A destination-by-province BEOE table would
  fix it.

## Placement inside units (2026-10-06)

Citizens and foreign residents are now placed apart inside each unit (session 5d7dac7e-gulf): foreign dots lean to dense hexes and fill OSM industrial land and labour camps, citizens take the rest; one rule for the six Gulf states, fitted on Kuwait's areas and Oman's wilayat. Method, fit, data searched: `sources/gulf_place.md`. People in hexes 90%+ expatriate: 0 -> 6%.
