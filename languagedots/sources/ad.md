# Andorra (ad): record

Drawn 2026-10-05 (session edd42a8c-mono3). No census; World Values Survey wave 7 (2018) Q272
"language at home", national, on the 2018 population (76,177): Spanish 36.8%, Catalan 35.4%,
`other` 11.4%, Portuguese 11.3%, French 5.3%. Every row `modelled`. One national unit on
religiondots' Kontur hexes (read-only). 73 dots.

Files: `sources/ad_wvs.py`, `taxonomy/ad2018.py`, `taxonomy/tree.d/ad.txt` (bare repeats),
`countries/ad.py`, `data/normalized/ad.csv`, `data/raw/ad/ihsn_11550_Q272.html`.

## Source

WVS-7 Andorra 2018 (Institut d'Estudis Andorrans), 1,004 face-to-face interviews, residents 18+,
unweighted. Q272 frequencies from the IHSN catalogue's variable page (catalog 11550, V312), the
same file religiondots read Q289 from. Codes: Catalan 355, Spanish 369, French 53, Portuguese
113, Other European 80, Other 34 (sum 1,004 = N, asserted). Population: religiondots' pinned
2018 figure from the Observatori Social series.

## Calls

- WVS over the register-plus-proxy route (§2's rich-country rule): a survey that asks home
  language beats nationality as a proxy, and the brief's survey ruling covers it. The Govern's
  own "Coneixements i usos lingüístics" survey (periodic, national) was not fetched; it asks
  first language and would be the better source if someone gets its tables.
- "Other European" (80) and "Other" (34) on `other`: unnamed, no European-remainder node.
- National only. WVS carries parish codes (N_REGION_ISO, 7 parishes, 31-295 interviews each);
  per-parish shares from 31-63 interviews would be noise, so not split.
- Adults' shares applied to everyone, children included.
