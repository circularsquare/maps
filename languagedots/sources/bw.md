# Botswana: 2011 census, language spoken at home, 28 census districts

Drawn 2026-10-04 (session d9e44929-bw). `sources/bw_census.py`, `taxonomy/bw2011.py`,
`taxonomy/tree.d/bw.txt`, `countries/bw.py`.

**1,919,350 people aged two and over, 14 nodes, 28 census districts (69,000 people on average),
1,912 dots.**

## 1. The table

*Population & Housing Census 2011: National Statistical Tables Report* (Statistics Botswana,
2015), Category B, **Table 2**, "Distribution of Numbers of Persons Aged Two Years and Over By
District and Language Spoken; Botswana 2011", PDF page 52 (printed 40).
`https://www.statsbots.org.bw/sites/default/files/publications/national_statisticsreport.pdf`,
saved as `data/raw/bw/phc2011_national_statistical_tables.pdf`. Table 4 on PDF page 54 is the same
table printed again (with a grand total cell, 1,920,238).

One answer per person, the language spoken at home, persons aged 2+. Columns: Setswana, English,
Sekalanga, Shekgalagadi, Sesubiya, Sesarwa, Seyeyi, Sembukushu, Afrikaans, Ndebele, Zezuru/Shona,
Seherero, Other African, Other European, Other Asian, Other (NEC), Not Stated. Rows: the 28 census
districts and the six districts that group them. The PDF's text layer is clean (one cell per line),
so it is parsed from PyMuPDF text, with the column order pinned and checked by sums.

## 2. Why 2011 and not 2022

The coverage sweep's lead was the 2022 census, *Analytical Report Volume 1*, "appendices 3-4
tabulate both [home and early-childhood language] by census district". **They do not.**
`data/raw/bw/phc2022_analytical_vol1.pdf`, pages 77-82:

- Appendix 1: how 2022 grouped its answers (Sesarwa = Gana, Naro, Nama, Ju|'hoan, Sekhwedam, Shua,
  Gwi, Kx'au||'ein, Tsowa, !Xoo, |Hua, Kua; Setswana and Shekgalagadi with their dialects).
- Appendix 2: national, 2001/2011/2022 home language and 2022 early-childhood language.
- Appendix 3: by census district, but **only the few districts where each language is largest**
  (Setswana in seven, Kalanga in six, Sesarwa in five...). Not a full table; it cannot be drawn.
- Appendix 4: national only, with the dialects and the dozen San languages split out. (Its title
  says "by Census Districts"; the table has no district column.)

Appendices 2 and 4 disagree on 2022's Sesarwa and Thimbukushu (33,405 and 18,912 swapped between
them), so the 2022 report's tables are not careful either. The 2022 data page
(`/census-2022-data`) is population by locality only. So the full district table is 2011's.
The 2022 early-childhood question would have been the better fit for a first-language map (spec
§1 prefers mother tongue over home language); nationally it is close to the home answer (Setswana
75.3% against 77.5%), so the choice of question costs little here.

## 3. Checks (all asserted in `bw_census.py`)

1. Every row's 17 cells sum to its printed Total (34 rows).
2. The census districts of Southern, Kweneng, Central, North West, Ghanzi and Kgalagadi sum to
   those rows, every column.
3. The 28 census districts sum to the Total row, every column.
4. Table 2 = Table 4 cell by cell (a reprint: checks the parse, not the census).
5. **A second publication agrees per unit**: the 2022 report's Appendix 3 reprints the 2011 figure
   for 49 (language, census district) cells; all 49 equal Table 2. (Transcribed by hand from the
   rendered page, so this also checks the transcription.)
6. The 2022 report's Appendix 2 gives 2011 as 1,919,350 stated + 105,554 not applicable =
   2,024,904, the 2011 census total. Table 2: 1,919,350 stated, 888 not stated, so the under-twos
   are 104,666.

## 4. Geography

Religiondots' `data/geo/bw/bw_hexes.gpkg` (Kontur 400 m hexes keyed to COD-AB 2011 ADM3
localities), read-only. An ADM3 pcode `BWddssnn` sits in census district `BWddss`, which is the
COD-AB ADM2 pcode; there are exactly 28 ADM2 units and they are the census's 28 census districts
(Ngwaketse is COD-AB's "SOUTHERN" BW0801, Okavango Delta its "NGAMILAND DELTA" BW1403, CKGR
BW1602). `countries/bw.py` truncates the hex unit to six characters and asserts every unit is an
8-character BW pcode; `check_country` confirms the 28 census districts and 28 placement units
match both ways. The name → code table is explicit in `bw_census.py` (`ROWS`), checked against the
printed row labels in order; "Ghanzi" is printed twice (the district, then its census district)
and is told apart by position.

Unlike religiondots, which needed Central Boteti and Central Bobonong left out (no booklet), this
table covers all 28, including both.

Kontur is odd inside two towns: Gaborone's ADM2 holds 134,876 Kontur people against 222,957 census
2+, Jwaneng's 976 against 17,275. That only moves dots within those units (placement is by Kontur
share inside the census district); the district totals are the census's.

## 5. Mapping calls (`taxonomy/bw2011.py` has the full reasoning)

- New leaves: **Shekgalagadi** under Sotho-Tswana (Glottolog kgal1244 in soth1248); **Kalanga**
  under Bantu beside Shona (Glottolog puts it inside Shona S.10, but `nigercongo.bantu.shona` is a
  leaf other countries draw on); **Yeyi** under Bantu (Glottolog: directly under Eastern Narrow
  Bantu).
- Reused: Setswana (za), English, Afrikaans, Shona (us; Zezuru is a Shona dialect), Ndebele
  (Zimbabwe) (uk), Subiya (zm, Botatwe), Mbukushu (zm, Luyana), Otjiherero (na; pl.txt also has a
  bare `nigercongo.bantu.herero` for the same language, a duplicate someone may want to fold).
- **Sesarwa → the `khoisan` root**, washed out as "language not named": it is the census's name for
  the San languages together, which span Khoe-Kwadi, Kx'a and Tuu (na2011.py did the same with
  Namibia's "San languages"). 31,778 people, 1.7%; Ghanzi 13,372, Central Boteti 4,650, Ngamiland
  West 3,924, Central Tutume 2,941, Kgalagadi North 1,703.
- Other African → `africa_other`; Other European, Other Asian, Other (NEC) → `other` (na's call).
- Not Stated (888) not drawn, in `gap`.

## 6. Colours

Hand-picked in `tree.d/bw.txt`: Shekgalagadi light sand (against Setswana's olive), Kalanga
blue-violet (against Shona pink and Ndebele orange in the North East), Yeyi dark brick red, and
**Mbukushu recoloured** to a light periwinkle: it is zm.txt's node, bare there, and generated as a
pale mint indistinguishable from the washed Khoisan green it shares Ngamiland West with. Zambia
draws only a few thousand Mbukushu speakers, so the change costs it nothing visible.

## 7. Leads not used

- **Village level, 2011, partial.** Several of the 2011 district *Selected Indicators* booklets
  (religiondots' `data/raw/bw/mono/`) print language by village, by their table titles: Chobe and
  Delta (Table 10), Ghanzi (Table 10), Kweneng West (Table 13), Ngamiland East (Table 12),
  Ngamiland West (Table 8), South East (Table 14A). North East, Kweneng East and Ngwaketse have a
  language table or figure whose grain was not checked. The other booklets have none, and two
  districts have no booklet. A mixed grain (villages in the west and north-east,
  districts elsewhere) is possible and would sharpen exactly the places where it matters
  (Ngamiland, Ghanzi, the North East); not done here.
- **Microdata.** `microdata.statsbots.org.bw` (NADA) lists the 2011 census with variable
  `P13_LANGUAG`; access terms not checked. IPUMS carries 2001 and 2011 (account blocked).
- **Afrobarometer** home language by district (the sweep's fallback): not compared.
