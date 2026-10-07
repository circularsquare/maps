# North Korea (kp): record

Drawn 2026-10-05 (session edd42a8c-mono2). No language source: all 23,349,859 people the 2008
census counted by province drawn on Korean across the 11 provinces, rows `derived`. Placed on
religiondots' Kontur hexes (read-only). Haiti's model (`sources/ht.md`).

Files: `sources/kp_pop.py`, `taxonomy/kp2008.py`, `taxonomy/tree.d/kp.txt` (bare repeats),
`countries/kp.py`, `data/normalized/kp.csv`.

## Why everyone is drawn as Korean

- The 1993 and 2008 censuses ask no language question; no survey inside the country asks one.
- Korean is the first language of practically everyone. The one minority of any size, the ethnic
  Chinese (hwagyo, commonly put at a few thousand), has no published count anywhere and is drawn
  as Korean; said in `note_public`.

## Population and checks

2008 census national report Table 2, as religiondots re-cut it onto COD-AB's 11 provinces
(`kp_lookup.csv`, read-only): 23,349,859, asserted 11 rows, geo_id == unit. HDX `cod-ps-prk`
carries the same 2008 figures, so there is no newer count by province. The census's 702,372 people
in no province (institutional population, mostly military) are in `gap`, as religiondots has it.

## Calls someone might reverse

- 2008 population rather than a projection scaled to today (~26M, UN WPP): no projection by
  province exists, and a national scale-up would add nothing a reader could check.
- Chinese not drawn (no count).

## Scatter

23,349 dots over 8,453 hexes. `check_country.py kp`: ok.
