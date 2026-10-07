# Singapore: 2020 census, language most frequently spoken at home, by planning area

Session d9e44929-sg, 2026-10-04. Drawn: 3,596,282 residents aged 5+ on 31 units (30 planning
areas and an `Others` row), 3,593 dots, 9 nodes. Files: `sources/sg_census.py`,
`taxonomy/sg2020.py`, `taxonomy/tree.d/sg.txt`, `countries/sg.py`; raw CSVs in `data/raw/sg/`.

## Source

Department of Statistics Singapore (SingStat), Census of Population 2020. The question is
**language most frequently spoken at home**: "the language or dialect that a person uses most
frequently at home when speaking to other household member(s)" (Statistical Release 1,
glossary). `how` = "census, 2020, language most frequently spoken at home". The census also
asks the second most frequent language; this map uses only the first.

Three TableBuilder tables, republished on data.gov.sg under the Singapore Open Data Licence (free
reuse with attribution, no key, no login). Download is a two-step: `GET
https://api-open.data.gov.sg/v1/public/api/datasets/<id>/poll-download` returns a signed,
expiring S3 URL. **Anonymous calls are rate-limited** (code 24, "try again in 10 seconds"); the
fetch retries.

| table | data.gov.sg id | what it gives |
|---|---|---|
| CT/17596 | `d_21f546492a87dec38391fc72eb4c7890` | by planning area of residence: seven groups |
| CT/17439 | `d_2a1a771fe8515745d21ee64e43d6b46d` | national, by ethnic group and sex: the same seven (check only) |
| Table 41 | `d_ad4a8ccbdab03d16c486a9ee6988289d` | national, by age: Chinese Dialects opened into Hokkien, Teochew, Cantonese, Other |
| 2010 CT/5980 | `d_cde207636c5c3a4e76b796fa9ba17971` | 2010 national, with the same dialect split (drift check only) |

The coverage sweep's lead (`d_2a1a77...`) was the national table; the planning-area one was found
by listing all 4,643 data.gov.sg datasets through
`https://api-production.data.gov.sg/v2/public/api/datasets?page=N` (the site's own search is JS
and the API's `query=` parameter is ignored) and grepping the names.

**Nothing finer than national splits the dialects.** The 2010 census planning-area table
(`d_c3ed3269...`, CT/8625) and GHS 2015's (`d_58c5248e...`) carry the same seven groups; the
dialect-GROUP tables (Hokkien, Teochew... as ethnicity) are national too and are ancestry, not
language.

Printed counterpart: Statistical Release 1, Table 41 (printed page 151, PDF pages 167-168),
`https://www.singstat.gov.sg/files/467e88fc-0c55-453c-885f-268b89731904.pdf`; read from
religiondots' cached copy (`religiondots/data/raw/sg/cop2020sr1.pdf`), not re-downloaded.

National, most frequent language: English 1,735,242 (48.3%), Mandarin 1,075,172 (29.9%), Malay
332,256, Chinese Dialects 313,258 (Hokkien 157,259, Cantonese 79,216, Teochew 59,424, Other
17,359), Tamil 89,946, Other Languages 26,592, Other Indian Languages 23,818. Total 3,596,284.

## Universe and gap

Residents (citizens and PRs) aged 5+, **excluding people unable to speak and those in one-person
households or households of unrelated persons** (every table's footnote). 3,596,284 of the June
2020 population of 5,685,800 (63.3%). Outside it: 1,641,590 non-residents (not asked), and
447,926 residents (under 5, living alone or with unrelated people, unable to speak). Not scaled
up, as religiondots did not scale its Singapore universe: nothing published says what language
the people living alone speak, and non-residents are not tabulated for language by anybody.

The counted share of each planning area's all-ages resident population (subzone `pop` in the
placement layer) has the right shape: lowest in the old centre (Downtown Core 0.65, Others 0.74,
Outram 0.76, where people living alone are common), highest in the family towns (Pasir Ris and
Choa Chu Kang 0.94, Bukit Panjang 0.93), national 0.89. A scrambled join would not sort like this.

## Checks (`python sources/sg_census.py`)

- `-` is "nil or negligible" and reads 0; anything else non-numeric raises.
- Every unit's seven groups sum to its Total (worst gap 1, SingStat's random rounding); English's
  seven "English only / English & X" sub-columns sum to English. The other groups' sub-columns
  are short by design (footnote 1/: some second languages "not shown") and are not checked.
- The 31 units sum to the table's Total row per group within 3 (random rounding).
- The Total row equals CT/17439's national figures exactly, and Table 41's file equals the
  printed Table 41 exactly for all twelve figures; the four dialects sum to Chinese Dialects.
- Dialect mix, share of Chinese Dialects, 2010 vs 2020: Hokkien 49.0% / 50.2%, Teochew 19.4% /
  19.0%, Cantonese 24.9% / 25.3%, Other 6.7% / 5.5%. The build stops if any moves over 3 points.
- `countries/sg.py` asserts the table's 31 unit names equal the placement layer's 31 `unit`
  values both ways.
- `check_country.py sg`: ok, 3,596,282 people (2 short of the national total by rounding).

## Geography

religiondots' layer, read-only: `religiondots/data/geo/sg/sg_subzones.gpkg`, the 332 URA Master
Plan 2019 subzones labelled with the 31 units of its religion table, weighted by census 2020
resident population per subzone (`pop`). The language table has **the same 31 rows**: the same
30 named planning areas and `Others` for the remaining 25 (religiondots' `sources/sg.md` §7 has
the identification). So `place_weight=pop_weight` and nothing new was built. Not Kontur, so no
cap blocks; religiondots' reasoning (Kontur counts the 1.64M non-residents in dormitories) holds
equally here.

## Calls

- **Chinese Dialects shared out by the national mix, tier `derived`.** The Bulgaria and Hong
  Kong precedent: the planning-area total is measured, the split is the census's own national
  figure, the shape was stable 2010-2020, and the derived rows roll back to one measured cell.
  The alternative, `sinitic` washed out as "language not named", would hide 313,258 people the
  census did name. Cost: dialect geography inside Singapore is flat on this map; older central
  estates are likely more Cantonese than the national mix says.
- **Other Indian Languages on `other`**, not split. It crosses Indo-Aryan and Dravidian and no
  table opens it (the Indian dialect-group tables are ethnicity, and a proxy is Anita's to
  allow). The UK and US entries do the same.
- **Other Chinese Dialects on `sinitic`**, the census's own unnamed remainder.
- **Malay hand-coloured** in `tree.d/sg.txt` (0.82 0.15 150, `#75df8f`): generated near Malayic it
  came out on top of Tamil's teal (`#00b8a9`), and the two share every housing estate. Malay is
  also drawn small in au, ca and us, which this recolours.
- **`note_public` gives the national dialect shares** so a reader knows the split is uniform.
- Universe not scaled up (above).

## Not done

Teochew (`#2da1c2`) and Tamil (`#00b8a9`) are both cyan-teal; Teochew is 1.7% and derived, so
left as Hong Kong set it.

## Moved from countries/sg.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- The 1.64 million non-residents, the work permit and employment pass holders, dependants and students, are not asked at all: the dormitories at Tuas and the domestic workers in flats across the island are missing from these numbers.
- Thirty planning areas are named and the other twenty-five, home to 25,353 of the people counted, share one row: Rochor, with Little India and Kampong Glam, is among them.
