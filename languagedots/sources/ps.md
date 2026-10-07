# Palestine (ps): record

Drawn 2026-10-05 (session edd42a8c-arab). PCBS 2017 persons counted per governorate
(4,705,601, Table 2), all on Levantine Arabic, since every survey answer was Arabic: 16
governorates, 1 node, rows `modelled`. Placed on religiondots' Kontur 400 m hexes (Kontur PS + IL,
settlement population taken off there). 4,705 dots.

Files: `sources/ps_surveys.py`, `sources/ab_firstlang.py`, `taxonomy/ps2017.py`,
`taxonomy/tree.d/ps.txt`, `countries/ps.py`, `data/normalized/ps.csv`.

## Sources

| source | item | n | result |
|---|---|---:|---|
| PCBS census 2017 | no language item | 4,705,601 counted | population (religiondots' `ps_lookup.csv`, `counted_t2`) |
| Arab Barometer II, III | first language | 1,200 each | item empty for Palestine |
| **Arab Barometer IV 2016** | first language | 1,200 | Arabic 1,200 |
| **Arab Barometer VII 2021-22** | ethnic group | 1,800 | Arab 1,795, Other 2 (Jerusalem), missing 3 |
| WVS 6 (2013) | language at home | ~1,000 | not fetched |

Table 2 rather than religiondots' Table 3 (Palestinians only): Table 2 includes the 40,175
non-Palestinians counted, whose language nothing gives; they are drawn on the survey shares like
everyone else. AB VII's two "Other" go on the country's language (Algeria rule). Survey region
names (Jabalia = North Gaza, Gaza City = Gaza) asserted against the 16 units.

## Calls someone might reverse

- Levantine Arabic for everyone, including the 40,175 non-Palestinians counted.
- Domari not drawn (Dom in Jerusalem and Gaza; no count found, and most no longer speak it).
- Settlers are outside the census and on neither Palestine's nor Israel's entry: they are the
  `xs` entry (`sources/xs.md`), on the same units religiondots took off Palestine's placement.
- 2017 placement in Gaza kept despite the displacement since October 2023 (said in note_public).

## Terms

PCBS tables via religiondots. Arab Barometer: free, citation requested. Kontur CC BY 4.0.
