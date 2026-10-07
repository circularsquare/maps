# Turkmenistan: 2022 census mother tongue, Ashgabat and five velayats

Drawn 2026-10-05 by session `d9e44929-tm`. Code: `sources/tm_census.py`, `taxonomy/tm2022.py`,
`taxonomy/tree.d/tm.txt`, `countries/tm.py`. Raw: `data/raw/tm/tm_census2022_results_en_4.pdf`
(copied from religiondots' raw folder, where its session cached it; the script downloads it from
stat.gov.tm if neither copy exists). No asks filed.

## Source

| | |
|---|---|
| office | State Committee of Turkmenistan on Statistics, `stat.gov.tm` |
| table | *Results of the Complete Population and Housing Census of Turkmenistan 2022*, section 4, "National composition of the population and language proficiency" (English, 61 pages, text layer). Tables 4.9-4.29: population by nationality (62 rows) and "considered as their mother tongue" (10 columns), for the country, Ashgabat city and each velayat, both sexes, male, female. Drawn: 4.12, 4.15, 4.18, 4.21, 4.24, 4.27 (both sexes per unit). |
| URL | `https://www.stat.gov.tm/population-census-pdfs/results/en/4.pdf`; index `stat.gov.tm/en/population-census` |
| census day | 17 December 2022; 7,057,841 people |
| question | mother tongue, one answer. The columns: Turkmen, Russian, Ukrainian, Uzbek, Kazakh, Tatar, Armenian, Azerbaijani, Baloch, other languages |
| grain | Ashgabat and the five velayats (Ahal, Balkan, Dashoguz, Lebap, Mary). No etrap-level language table in any of the eleven sections. Arkadag (velayat since 2023) is inside Ahal, 567 people. |
| licence | the PDF states none (no copyright or reuse line); the site's terms not checked. An official statistical publication, cited, as religiondots uses the same volume |
| also in the volume | tables 4.30-4.50, nationality by knowledge of other languages (Russian, Turkmen, Uzbek, Azerbaijani, Kazakh, other). Not used: second languages are not drawn. |

## Checks (all in `sources/tm_census.py`, all pass)

1. Every row's ten language columns sum to its total, in all 21 tables.
2. In every table the 62 nationality rows sum to "All nationalities", column by column.
3. Male plus female equals both sexes in every cell, for the country and all six units.
4. The six units sum to the national table 4.9 in every cell.
5. Each unit's total equals section 1's table 1.3, and 90 nationality totals (15 nationalities x
   6 units) equal religiondots' transcription of tables 4.3-4.8 (both read-only from
   `religiondots/data/geo/tm/`).
6. Every figure in `note_public` is recomputed and asserted.

**Three print defects, each handled and tested:**
- Page 10 (table 4.12, Ashgabat) heads the sixth column "Kazakh" a second time; every other page
  says "Tatar". It is Tatar (the Tatars' row has 664 of 2,585 there; check 4 pins the column).
- Table 4.10 (male, national) leaves the last cell ("other languages") blank, no figure and no
  dash, for 15 rows from Slovaks to Estonians. Each is filled as the row's residual (0 to 206) and
  check 3 confirms every one against both sexes minus female. Table 4.10 is only used for that
  check anyway.
- Table 4.24 (Lebap) sets the Turkmens row across two baselines, with 1 288 630 split "1 288" /
  "630". The parser joins words within 11 pt vertically (rows are at least 13.5 pt apart);
  checks 1-4 pass on the result.

Column placement for lines with fewer than eleven figures uses the header's vertical rules; on
some pages those rules sit 3 pt off the figures, so complete lines are read in order instead.

## Result

National: Turkmen 6,297,965 (89.2%), Uzbek 478,212 (6.8%), Russian 135,565 (1.9%), Balochi
82,767 (1.2%), Azerbaijani 21,274, other 17,596, Kazakh 10,578, Armenian 9,154, Tatar 3,900,
Ukrainian 830.

```
              people    Turkmen  Uzbek  Russian  Baloch  Azerb.  Kazakh  other
Ashgabat   1,030,063     90.07    0.31    8.05    0.01    0.68    0.05   0.20
Ahal         886,845     98.92    0.12    0.28    0.09    0.11    0.02   0.43
Balkan       529,895     93.85    0.42    3.17    0.05    1.28    0.49   0.25
Dashoguz   1,550,354     71.62   27.53    0.23    0.01    0.02    0.34   0.17
Lebap      1,447,298     96.09    2.77    0.87    0.03    0.04    0.10   0.04
Mary       1,613,386     92.63    0.29    1.07    5.02    0.34    0.04   0.44
```

10 nodes, 7,052 dots at 1:1,000, 1 ring. 5,841 people (0.08%) are under one dot per language
and draw none.

**Of interest, all from the census's nationality rows:**
- 169,990 of 642,476 Uzbeks (26%) named Turkmen. In Lebap it is 97,522 of 136,499 (71%), against
  67,052 of 489,453 (14%) in Dashoguz. Lebap is 9.4% Uzbek by nationality and 2.8% by mother
  tongue. Whether this is language shift, how the question was put, or how answers were recorded,
  the census cannot say. The map draws what it printed.
- 32,777 of the 135,565 who named Russian are not Russians (Armenians 4,737, Ukrainians 1,637,
  Tatars 3,216, Azerbaijanis 3,434, Turkmens 14,301). 11,140 Russians named Turkmen.
- Balochi: 80,365 of 85,384 Baloch in Mary named Balochi; 4,926 Baloch nationally named Turkmen.
- "Other languages", 17,596 across 57 nationalities: Persians 7,190 (Ahal 3,397, Mary 3,506),
  Afghans 1,796, Karakalpaks 1,613 (1,580 in Dashoguz), Lezgins 1,241, Turkmens 888, Kurds 856,
  Balochi 576, Koreans 556.

## Geography

religiondots' placement layer, read-only: `religiondots/data/geo/tm/tm_hexes.gpkg`, Kontur 400 m
hexes keyed to the six units of Kontur's OSM boundaries of 2022-04-07 (religiondots `sources/tm.md`
§2 and §7: the join has three witnesses, no hex at the Kontur cap). The census's unit ids are
religiondots' own (`TM-S`, `TM-A` ...), joined through its `tm_lookup.csv`; `countries/tm.py`
asserts all six match. Weight: the layer's Kontur `pop`. Kontur reads Ashgabat at 1.75x and
Dashoguz at 0.35x the census (religiondots §7); that only moves dots within a unit.

## Calls someone might reverse

- **"Other languages" on `other`**, not split by nationality (Persians' share onto Persian,
  Karakalpaks' onto Karakalpak). The census does not name the language, and §3 says an unnamed
  remainder is never guessed into a member. Reversing: a nationality-keyed mapping in `tm2022.py`
  and `nationality` in `countries/tm.py`'s groupby; the normalized CSV keeps the nationality.
- **Baloch on tree.txt's single `balochi` leaf**, not a Western (Rakhshani) Balochi node.
- **The census taken as printed**, Uzbek-to-Turkmen answers included, with the closed-state
  caveat in `note_public`. The total (7,057,841) is the same one religiondots drew, which Anita
  approved there on 2026-09-14.
- **Colours unchanged.** Turkmen (crimson, uz's pick), Uzbek (pink), Balochi (generated gold-brown),
  Azerbaijani (generated plum) and Russian (green) are told apart; at velayat grain they mix
  inside units rather than meet at borders.

## What reopens it

- An etrap-level language table (none in the eleven sections of the 2022 results).
- Microdata, which the committee does not publish.
