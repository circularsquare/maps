# Armenia (`am`): 2011 census mother tongue, 11 marzes

Drawn 2026-10-05 by session `d9e44929-am`. No asks filed.

Files: `sources/am_census.py` (normaliser), `taxonomy/am2011.py`, `taxonomy/tree.d/am.txt`,
`countries/am.py`. Outputs `data/raw/am/` (11 marz PDFs `AM-xx.pdf`, `national_hy.pdf`,
`national_en.pdf`), `data/normalized/am.csv` (87 rows), `dots_am.geojson` (3,014 dots),
`rings_am.geojson` (5). Placement is religiondots' `data/geo/am/am_grid_400m.gpkg`, read only.

## 1. The table, and why 2011 and not 2022

The queue's lead was the 2022 census at marz. Both censuses asked mother tongue (`մայրենի լեզու`)
and both publish it for the eleven marzes (ten provinces and Yerevan) and no lower; section 1 of
each volume goes to the settlement for population alone.

- **2022**, table 5.2 in section 5 of each marz volume (`armstat.am/am/?nid=945`-`957`, 7z of xlsx,
  Armenian only; national `file/article/section_5.7z`, English). Its columns are **"the language of
  his nationality", "other tongue", "refused"**, crossed with eleven ethnicities. No language is
  named: the 46,846 people (1.6%) whose mother tongue is not their nationality's are one unnamed
  remainder (33,172 of them ethnic Armenians), and everyone else is named only through the
  ethnicity row ("Indian" and "other" ethnicities have no single language at all).
- **2011**, table 5.2-1 of each marz volume (`armstat.am/am/?nid=533` Yerevan to `?nid=543` Tavush,
  one PDF per table; doc ids are in `MARZES` in the script) and of the national volume (`?nid=532`,
  doc 99478358 Armenian, 99486258 English). Columns name each mother tongue: Armenian, Yezidi,
  Russian, Assyrian, Kurdish, Ukrainian, English, Georgian, Persian, Greek, Other, Refused.

So 2011: same grain, the same question, and every answer named. 2022 is a fine witness (section 4).
The `/en/` marz pages are empty for both censuses (religiondots found the same for religion); the
Armenian tree has everything.

**Each marz prints only the languages it has speakers of**: Yerevan 9 languages plus Other and no
Kurdish; Gegharkunik, Syunik and Vayots Dzor only Armenian and Russian plus Other; Tavush the same
with no refusal column. A
language a marz gives no column is in that marz's Other: 1,270 people nationally (Ukrainian 342,
Kurdish 289, Georgian 271, English 192, Greek 98, Assyrian 55, Yezidi 19, Persian 4).

Reading: PyMuPDF word boxes, not text order (plain text puts Yerevan's total row at the end of the
page). Heads are set vertically in ten files and horizontally in Tavush; the refusal head is
typeset differently in every file (Kotayk: `Հրահրաժարվել պատաս- խանել են`); the refusal column sits
2-3 pt off its row in three files, so rows are anchored on their Total cell; Tavush writes
thousands with a space (`128 609`). All handled in `read_total_row` and noted where they are.

## 2. Checks (all asserted in `am_census.py`)

1. Every marz's columns sum to its own Total exactly; the eleven Totals sum to 3,018,854, the
   national total, exactly.
2. The Armenian national PDF reads to exactly the English national table's figures (typed in
   `NATIONAL_EN`), 13 of 13.
3. Marz sum vs national per language is a floor (shortfalls above), and the marz Others exceed the
   national Other (913) by exactly the sum of the shortfalls (1,270).
4. **Second cut, same census:** each table's ethnicity rows add up to its total row in every
   column, in all ten marzes and the country. Tavush is the one exception, as printed: its Other
   column's rows give 25 against the total row's 118 (93 short; every row's own Total and the total
   row are consistent). Pinned in `KNOWN_ROW_GAPS`; am.csv takes the total row.
5. **Column order** (which the sums cannot catch): for every nationality of 300+ in a unit whose own
   language has a column, that column is the row's largest apart from Armenian. 33 rows across
   the marzes and 9 nationally, all pass (Yerevan's 603 Ukrainians tie Ukrainian and Russian at 251).

## 3. Mapping (`taxonomy/am2011.py`)

Eleven answers, ten existing nodes and one new one.
- Armenian -> `indoeuropean.armenian.armenian`; Russian, Ukrainian, Greek, Georgian, Persian,
  English -> the existing leaves; Assyrian -> `afroasiatic.assyrian` (Assyrian Neo-Aramaic, from
  us.txt; Armenia's Assyrians came from Urmia in the 1820s).
- **Yezidi -> new `indoeuropean.iranian.yezidi`, a sibling of Kurdish.** Glottolog has no Yezidi
  language: Armenia's Yezidis and Kurds both speak "Erevan Kurmanji" (erev1241), a dialect of
  Northern Kurdish (nort2641). The census prints the two answers apart and the names follow the
  communities (national cross-tab: of 35,308 Yezidis, 30,628 named Yezidi, 323 Kurdish, 4,271
  Armenian; of 2,162 Kurds, 1,684 Kurdish, 39 Yezidi). Every printed label gets a node, and a child
  of Kurdish would wash Kurdish out as a group. No glottocode.
- Other -> bare `other` (2,183 in the marz tables): unnamed languages and named ones a marz gives no
  column, together.
- Refused (29) -> not drawn; `gap`.

The Yezidi/Kurdish naming is a live identity question in Armenia. It was not raised as an ask: the
map draws exactly the two answers the census prints, at marz grain, with nothing inferred, and
`note_public` says linguists count both as Kurmanji.

## 4. 2022 as a witness

2022 table 5.2, national: Yezidi ethnicity 31,079, of whom 25,430 name their own language (2011:
35,308 and 30,628); Russian ethnicity 14,076 / 13,146 (2011: 11,911 / 10,466). Ethnic Armenians with
another mother tongue rose from 13,028 in 2011 to 33,172 in 2022, plausibly including the
Russian-speaking arrivals of 2022; the 2011 map does not show them. Not compared by marz.

## 5. Geography

Religiondots' layer, read only: geoBoundaries ADM1 (11 marzes, keyed by ISO 3166-2, `AM-ER` etc.)
cut into 12,340 Kontur 400 m hexes with `pop`. am.csv's `geo_id` is the same code; the join is 11
of 11 (check_country). Marz boundaries are unchanged since 1995, so the 2011 table sits on them as
the 2022 one does. Kontur against the 2011 totals: Yerevan 0.81, Kotayk 1.42, the city/ring pair
religiondots recorded; placement only moves people inside their marz.

## 6. Calls someone might reverse

- 2011 with every language named over 2022 with the non-nationality languages unnamed.
- Yezidi as its own node beside Kurdish rather than inside it.
- Marz Other kept whole on `other`, though 1,270 of it is named languages the national table
  places.

## 7. Scatter and colours

3,018,825 people drawn, 3,014 dots at 1:1000, 5 rings; 4,825 (0.16%) under one dot per language
nationally. Yezidi hand-picked a saturated orange (0.72 0.16 55) to stand off Armenian's light teal
in Armavir and Aragatsotn and off Kurdish's generated green. Kurdish (green, #77ae7e) sits near
Russian's green and Assyrian's pale green near Armenian's teal, but each of those is two dots and
the nodes are shared with other countries, so they were left.
