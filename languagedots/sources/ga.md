# Gabon (ga): record

Drawn 2026-10-05 (session edd42a8c-mono4). 3,518,621 people (RGPL 2026), 9 provinces, 206 nodes
(most from the foreign residents' origin mixes). Gabonese rows `modelled`, foreign rows
`derived`. 3,439 dots at 1:1000, 111 rings.

```
python sources/ga_afro.py
python sources/origin_mix.py --fragment ga
python taxonomy/build.py
python tools/check_country.py ga
python scatter.py --country ga
```

Files: `sources/ga_afro.py`, `taxonomy/ga2026.py`, `taxonomy/tree.d/ga.txt`, `countries/ga.py`,
`data/normalized/ga.csv`. Read-only from religiondots: Afrobarometer merged files (through
`sources/mono_afro.py`), `data/geo/ga/` (lookup and hexes), the UN DESA workbook.

## 1. What exists

- **Census**: RGPL 2013 *Résultats globaux* (religiondots' copy) has nationality, no language,
  and no nationality breakdown of its 287,379 foreigners (Tableaux 23-26). RGPL 2026: totals,
  provinces and the national Gabonese/foreign split only.
- **Afrobarometer** R6 2015, R7 2017, R8 2020, R9 2022: 4,797 adult citizens, region in every
  round; Estuaire 2,461 to Nyanga 136. Not in R4-R5. The source.
- **UN DESA migrant stock 2020** for Gabon: 416,651, of which 405,666 by named origin
  (Equatorial Guinea 87,411, Mali 55,434, Benin 52,460, Cameroon 50,906, Senegal 30,652,
  Nigeria 24,029, Togo 23,129, Congo 16,396, France 16,146).

## 2. How the counts are made

- **Gabonese per province** (religiondots' IPF of RGPL 2026 totals on 2013 foreign shares,
  2,318,365) x Afrobarometer shares.
- **French (ask 018; the call most worth reversing).** French by round: R6 29.2%, R7 74.9%, R8
  74.7%, R9 73.0%. R6 asked "home language", R7-R9 "language spoken in home": the same wording
  break as Cameroon, Kenya, Tanzania. Cameroon's rule: French's share per province from R6 alone
  (`LF_ROUNDS = [6]`), other languages from all rounds among non-French answers, scaled to what
  French leaves. Drawn: French 29.8% of Gabonese (Ogooué-Maritime 54%, Ngounié 43%, Estuaire
  30%, Ogooué-Ivindo 2%); R6 has 30-70 respondents per province outside Estuaire, so the provincial
  French shares are rough. `LF_ROUNDS = [6, 7, 8, 9]` draws French as answered (~74%).
  **Replaced 2026-10-05, below.**
- **French at R7's mother tongue (2026-10-05, ask 018).** Anita's ruling: lingua francas at
  Afrobarometer R7's separate **mother tongue** question (Q2A). `LF_ROUNDS = "R7Q2A"`: each
  province's Q2A share shrunk to the national 1.42% by 50 respondents (`wafr_afro.r7_mother`,
  `shrink`); 16 of R7's 1,199 named French, all in French interviews. French among Gabonese
  690k (29.8%) -> 36k (1.6%): Estuaire 18 -> 1.2%, Ogooué-Maritime 37 -> 0.8%, Ngounié 37 ->
  0.5%, Woleu-Ntem 19 -> 0.3%. The other languages scale up (Fang 25.5 -> 35.4%, Punu 13.8 ->
  20.5%, Nzebi 9.2 -> 12.8%). French from foreign residents' mixes (France, Belgium...) is
  unchanged.
- Gabonese drawn: Fang 35.4%, Punu 20.5%, Nzebi 12.8%, Mbede 8.7%, Myene 6.0%, Kota 5.5%, Tsogho
  4.4%, French 1.6%.
- **Foreign residents** (1,200,256; per province from religiondots' IPF): DESA's named origins,
  `Others` assumed alike, each through `origin_mix.mix(iso, "ga")` (home mix where the origin is
  drawn on this map, else its main language). One national mix in every province: nothing gives
  nationality by province. Equatorial Guineans take gq's drawn mix (Fang 77%).
- Every province sums to its RGPL 2026 total (asserted).

## 3. Calls someone might reverse

- French at R7's mother tongue (above; Anita's ruling; the shrink is mine).
- "Mbédè" drawn as Mbere (mber1257; Haut-Ogooué, where 137 of its answers are); "Nzébi" as Njebi.
- "Bateke" on cd.txt's `teke` leaf: the card does not say which Teke.
- Baloumbou, Masangu, Kélé get Gabon-specific ids (`lumbu_gabon`, `sangu_gabon`, `kele_gabon`):
  Zambia's Lumbu and Tanzania's Sangu are other languages.
- "Kélé"/"Kélè" merged (spelling variants across rounds).
- Foreigners: DESA's 2020 origin mix on 2026's foreign count, which grew from 287,379 (2013) to
  1.2M; the mix may have moved since.
- Hand colours for Punu, Nzebi, Mbere, Kota, Tsogo, Myene, Sira (fragment).

## 4. Room for improvement

A census language or ethnicity table; a nationality-by-province table from RGPL 2026.
