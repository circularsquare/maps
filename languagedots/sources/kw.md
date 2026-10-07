# Kuwait (kw): record

Drawn 2026-10-05 (session edd42a8c-gulf). Method shared with the other Gulf states:
`sources/gulf.md`. 2021 census, 157 areas on religiondots' 143 units, 4,381,139 people (the
4,578 with no area stated left out), 100 nodes, every row `derived`. 4,331 dots at 1:1,000; 22
rings.

```
python sources/kw_build.py --fetch     # --fetch re-runs kw_extract.py
python taxonomy/build.py
python tools/check_country.py kw
python scatter.py --country kw
```

Files: `sources/kw_extract.py`, `sources/kw_build.py`, `sources/gulf_mix.py`,
`taxonomy/kw2021.py`, `taxonomy/tree.d/kw.txt`, `countries/kw.py`, `data/raw/kw/`,
`data/normalized/kw.csv`.

## Tables

- **Areas, Kuwaitis and non-Kuwaitis by sex and nationality group**: religiondots'
  `sources/kw.py` (record `../religiondots/sources/kw.md` §1-2), loaded by path in its own process
  by `kw_extract.py`, read-only. Census 2021 Table 52 (157 areas x Kuwaiti/non-Kuwaiti x sex),
  Table 6 (groups by governorate and sex); each area's December 2014 group mix (PACI via GLMM)
  raked per governorate and sex to Table 6, the same arithmetic as religiondots' `main()`.
  Asserted: 1,488,435 Kuwaitis, 2,892,704 non-Kuwaitis.
- **Nationalities**: PACI mid-2018 non-Kuwaitis by nationality and sex (GLMM): Egypt 670,524,
  Syria 160,120, Saudi Arabia 127,604; India 1,012,104, Bangladesh 281,131, the Philippines
  213,989, Pakistan 109,427, Sri Lanka 93,749, Nepal 70,378. Each group's unnamed rest (Arab
  302,814; Asian 87,430) at UN DESA 2024's other origins in that group and sex, scaled to fill it;
  Africa, Europe and North America at DESA alone; South Americans on `indoeuropean.romance`
  (narrowest node for Spanish and Portuguese together), Australians on English. DESA's own Kuwait
  figures miss what PACI counts (Saudi nationals 2,406 in DESA against PACI's 127,604), so PACI
  wins where it names. Vintages: groups 2021, nationalities 2018 and 2024.
- Indians: Keralites 124,948 (KMS 2023, 5.8% of Kerala's emigrants) = 12.3% of PACI's Indians.

## Results

Gulf Arabic 34.8% (Kuwaitis 1,488,435 plus other GCC nationals), Egyptian Arabic 14.6%,
Bengali 6.6%, Hindi 6.1%, Levantine Arabic 5.8%, Saudi Arabic 2.8%, Tamil 2.4%, Malayalam 2.4%,
Punjabi 2.0%, Yemeni Arabic 1.9%.

## Calls and gaps

- **Bidoon not named.** Stateless residents (about 100,000 by Kuwait's own Central Agency, not
  opened) are non-Kuwaitis in the register, but no table found gives them a row or an area. They
  are presumably inside PACI's unnamed Arabs, drawn at DESA's other Arab origins (Jordanian,
  Yemeni, Sudanese, Palestinian...), which is wrong for them: they speak Gulf Arabic. A PACI
  "without nationality" count by area would fix it.
- Kuwaitis of Persian origin ('Ajam) who keep Persian (or Larestani/Achomi) at home: no count.
- Keralites at 12% of Indians may be low (KMS emigrant counts vs PACI's register).

## Placement inside units (2026-10-06)

Citizens and foreign residents are now placed apart inside each unit (session 5d7dac7e-gulf): foreign dots lean to dense hexes and fill OSM industrial land and labour camps, citizens take the rest; one rule for the six Gulf states, fitted on Kuwait's areas and Oman's wilayat. Method, fit, data searched: `sources/gulf_place.md`. Next to no change here: the 143 areas already carry the census split, and religiondots' layer is flat inside each area, so only industrial land moves dots.
