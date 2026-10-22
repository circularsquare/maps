# Guernsey: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 64,091 people (Guernsey, Herm and Jethou
at 31 March 2023), one unit, 201 nodes (mostly tiny origin-mix tails), 56 dots and 195 rings at
1:1000. Every row `derived`. Scripts: `sources/gg_census.py`, `sources/gg_geo.py`,
`sources/terr_common.py`, `taxonomy/gg2023.py`, `taxonomy/tree.d/gg.txt` (origin_mix block
generated), `countries/gg.py`. No asks. Built as Bermuda (`sources/bm.md`).

## 1. Source

Guernsey has had no traditional census since 2001 and no language question; it counts people from
records (the Rolling Electronic Census). Table used: States of Guernsey Data & Analysis, "Guernsey
Quarterly Population, Employment and Earnings Bulletin, Supplementary Data (population up to
December 2023)", sheet "Country of Birth", column 31.03.23 (`data/raw/gg/supplementary_183493.xlsx`,
gov.gg/CHttpHandler.ashx?id=183493; the server's certificate chain is incomplete, fetched without
verification). It has 68 country rows and 6,712 "country of birth unknown", total 64,091. The
Annual Electronic Census Report 2023, Table 5.2.1 (p. 16, `data/raw/gg/electronic_census_2023.pdf`)
prints the same figures in groups: Guernsey 33,955, other Crown Dependencies 420, UK and ROI
15,107, Portugal 1,326, Latvia 887, Other Europe 1,699, Rest of world 3,985, unknown 6,712.
The spreadsheet's rows reproduce all of these except that Other Europe and Rest of world differ
by 4 people the other way (their sum agrees). The report notes birthplace is known for 90%.

## 2. Calls

- Guernsey, other Crown Dependencies, UK parts, Ireland, British Overseas Territories, the US,
  Canada, Australia and New Zealand: English (the Bermuda rule).
- Every other named country: `origin_mix.mix(iso, "gg")`.
- Group rows without a country (Other Europe, Africa, Asia, North and South America; 509 people):
  `other`.
- The 6,712 with no birthplace (10.5%): shared pro rata over the rest, so every row is scaled by
  1.117. Table 5.2.2 shows the unknowns skew towards 5 to 15 years on island, so more of them may be
  recent arrivals than among the known; no figure says how many.
- Guernsey-born children of immigrants are drawn as English: Portuguese (1,326 born in Portugal,
  scaled 1,516) and Latvian are probably understated; Jersey's identity table suggests about
  a fifth more there (9,739 by identity against 8,280 by birth) (`sources/je.md`), but Guernsey has no equivalent.

## 3. Geography (`sources/gg_geo.py`)

Not in religiondots. Kontur GG (Bailiwick, 155 hexes, 62,849 people) cut at longitude -2.40 to
leave Alderney (about 2,000) and Sark (about 560) out: 120 hexes, 60,362 people against 64,091
(0.94). One unit `GG`: the electronic census publishes parish populations (10 parishes plus Herm
and Jethou, Table 6.1.1) but no parish table of birthplace, and parish populations alone only
restate where Kontur already puts people.

## 4. Room for improvement

The States' postcode file (`data/raw/gg/postcode_180740.xlsx`) is flags only, no birthplace. A
birthplace-by-parish table, or a second-generation estimate (Jersey's identity table), would move
Portuguese and Latvian up.
