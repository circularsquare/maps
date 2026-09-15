# Brunei — 2021 Population and Housing Census, religion by district

**Drawn 2026-09-15** (session `d743fc47-bn`). 4 districts, 4 categories, 440,715 people (the whole
enumerated population, temporary residents included), every row `measured`. 439 dots at 1:1,000,
42 at 1:10,000.

- `sources/bn.py` -> `data/normalized/bn.csv` (the workbook and Annex A in `data/raw/bn/`, pinned;
  the questionnaire is kept beside them as `bpp2021_questionnaire.pdf`)
- `sources/bn_geo.py` -> `data/geo/bn/bn_districts.gpkg`, `bn_hexes.gpkg`, `bn_lookup.csv`,
  `bn_mukim_lookup.csv` (geoBoundaries ADM1 districts, ADM2 mukims as a witness, Kontur 400 m)
- `taxonomy/bn2021.py` -> the mapping; `countries/bn.py` -> the entry; `taxonomy/branches.py`
  `other.bn` is the one new node
- sources.md **§bn-2026-09-15** is the summary; **§scout-2026-09-14-asia-oceania** was the scout's.

```
python sources/bn.py     --fetch
python sources/bn_geo.py --fetch
```

## 1. What DEPS publishes

| release | religion | tier |
|---|---|---|
| **BPP 2021 report, *Demographic, Household and Housing Characteristics*, Annex A** | **A4, religion x district x sex, persons**; A9 age x religion (nation); A10 (a)-(d) age x religion per district; A11 religion x residential status (nation); A12 (a)-(d) the same per district | **4 districts** |
| the same tables as `EXCEL TABLE A-C.xls` | sheets A1-A12, B1-16, C1-C10 | 4 districts |
| Annex C / sheets C1-C10 | population by mukim and kampung, residential status, age, sex; **no religion** | 39 mukims |
| live report `deps.gov.bn/wp-content/uploads/2025/11/RPT-2.pdf` | national chart only (the scout) | national |
| UNSD Demographic Yearbook table 28 | 2021 and 2011, four categories, Total / Urban / Rural | national |

No table in Annex A to C crosses religion with race (A3 is race x district). **Not opened this
session, so not a negative:** the full 2021 report `RPT.pdf` (37.9 MB on Wayback), for a
definitions chapter or a religion table outside the annexes; the 2011 census tables; the 2001
census.

## 2. The files

Both survive only on Wayback: the new DEPS site links the old `deps.mofe.gov.bn` document library,
which no longer answers.

| file | capture | digest | bytes |
|---|---|---|---|
| `EXCEL TABLE A-C.xls` | 20231023184321 (one digest in every capture to 2025-09-08) | `TBSZB7XQDR7RSAHO3CA4B7IKOZNPXJCS` | 856,576 |
| `ANNEX A.pdf` | 20240624173053 (31 pages, `%%EOF`) | `TJ5JRHQ7FQH5DTOPRD4RWXIZWIWGZNY7` | 1,329,106 |
| `ANNEX A.pdf`, **not used** | 20230930115000 | `2WW3Z24POL5YRMBY3S5EGEX7XVAAOH4I` | **1,048,576, no `%%EOF`** |
| `Q_BPP2021.pdf` (questionnaire) | 20220626223640 | `MWCR4LRZBV46MVX4MBS32GK7UTNDSDWU` | 6,764,472 |

The 2023 Annex A capture is exactly the first 1 MiB of the file, the playbook's truncated-capture
trap; the scout's URL pointed at it.

## 3. The checks (`sources/bn.py::check`)

| check | result |
|---|---|
| both files pinned | digests above |
| sheet A4 = transcription = Table A4 parsed off printed p.83 | identical, 5 rows x 15 counts |
| persons = males + females; districts sum to total; religions sum to total | every cell |
| A4's district totals = A1's | persons, males, females |
| A11 (religion x status) total column = A4's national column; statuses sum | yes |
| A12 (a)-(d) total columns = A4's districts; statuses sum; districts sum to A11; status totals = A1 | yes |
| A10 (a)-(d) age rows sum to totals, totals = A4 per religion | yes, 2 empty cells read as zero (Temburong) |
| C1's 39 mukims (18 / 8 / 8 / 5) sum to A1's districts | yes |
| UNSD table 28, Brunei Darussalam 2021 | Muslim 362,035, Christian 29,462, Buddhist 27,745, Other Religions 21,473; equal to the person |

## 4. The questionnaire, and what `Others` holds

Household form, p.10 as printed, **item E10 Ugama / Religion**: 1 Islam, 2 Kristian / Christianity,
3 Buddha / Buddhism, 4 Hindu / Hinduism, 5 Lain-lain / Others (Sila nyatakan / Please specify).
Rendered and read. **There is no box for no religion and none for not stated.** Every table folds
code 4 into `Others`. The scout's line that the tables fold "Hindu, no religion and not stated" into
Others is right about Hindu; no religion and not stated are not codes, so they can only be write-ins
under code 5, or blanks the office handled somehow. A4's total is A1's whole population, so no one
was left out of the table and no non-response row exists. No `gap`.

What the annex does say about `Others` (21,473, 4.87%):

| district | Others | share | citizens / permanent / temporary (A12) | under 15 (A10), Others / everyone | male (A4) |
|---|---:|---:|---|---|---:|
| Brunei Muara | 10,074 | 3.16% | 1,732 / 727 / **7,615** | 10.3% / 20.6% | 66.3% |
| Belait | 5,196 | 7.93% | 1,866 / 1,277 / 2,053 | 15.1% / 19.5% | 57.6% |
| Tutong | 5,192 | **11.00%** | **4,718** / 212 / 262 | 15.3% / 21.5% | 51.8% |
| Temburong | 1,011 | 10.71% | 666 / 262 / 83 | 17.3% / 20.8% | 53.4% |

Two populations in one cell: in Brunei Muara, mostly temporary residents and mostly men (5,442 of
the 7,615 temporary are men); in Tutong, citizens with an ordinary age and sex structure. Which
religions they are is not published. Drawn on one node, `other.bn`, because the source gives one
number (§6.3a-iv: `other` is a religion the source did not name). Not `unknown`: the box is not a
no-religion box, and `unknown` would hide the Hindus, whom the form asked about by name.
`tools/check_no_religion.py` does not see the box (its label is not a no-religion answer); the call
is in `taxonomy/bn2021.py`'s REVIEW.

**Christianity (29,462) is 63.3% temporary residents** (18,653, A11): 14,893 of Brunei Muara's
20,076. Temburong is the exception, 955 citizens of 1,207. **Buddhism** splits evenly across the
three statuses (9,357 / 9,126 / 9,262); 4,418 of Belait's 7,235 are permanent residents.

## 5. Geography

**Boundaries: geoBoundaries gbOpen BRN ADM1** (commit 9469f09; 4 districts, `boundaryYear` 2011,
traced from a Wikimedia Commons map by user Tachymetre, public domain). No HDX COD-AB exists.
Joined by folded name (`Brunei-Muara` against the census's `Brunei Muara`), a bijection. Areas:
Belait 2,797 km2, Tutong 1,227, Temburong 1,291, Brunei Muara 557; 5,871 in all.

**Witness: the mukims.** geoBoundaries ADM2 is 38 mukims (2006, a separate tracing by user
Rarelibra). Census Table C1 lists 39: Gadong A and Gadong B (one `Gadong` in 2006), and `Bokok`,
which geoBoundaries spells `Bunkok`. With those folds the 38 join one to one, and every mukim lies
mostly in the district C1 lists it under, measured on the part of its area inside any district
(lowest: Liang 97.7%, Pangkalan Batu 98.0%).

**An area test against the mukims was tried and dropped, because area was the wrong measure.** The
first build required the union of each district's mukims to overlap the district at IoU 0.85.
Brunei Muara failed at 0.831; Belait passed at 0.942. Measured in Kontur people instead
(scratch `which_outline.py`, 2026-09-15):

| district | census | Kontur on the districts | Kontur on the mukims dissolved by C1 |
|---|---:|---:|---:|
| Brunei Muara | 318,530 | 323,172 (1.01x) | 317,588 (1.00x) |
| Belait | 65,531 | 62,819 (0.96x) | **46,272 (0.71x)** |
| Tutong | 47,210 | 46,647 (0.99x) | 45,506 (0.96x) |
| Temburong | 9,444 | 15,678 (1.66x) | 14,478 (1.53x) |
| in no district | | 658 | **25,130** |

The mukim tracing leaves out a single 42.5 km2 strip of the Kuala Belait and Seria coast (centred
4.622N 114.329E) holding **29,972** Kontur people, so every Belait coastal mukim reads about 0.7x on
it, and it draws the Brunei River inside Sungai Kebun (58% of its traced area in no district),
Serasa, Burong Pingai Ayer and Kota Batu. Belait passed the area test while its populated coast was
the part in dispute; Brunei Muara failed on river. The district outlines put people where the
census does, so they are the placement layer, and the mukims are a membership witness only.

**Placement.** Kontur BN 2023-11, 2,064 hexes, 448,974 people. 168 centroids fell outside every
district; 126 snapped within 500 m, 42 dropped (658 people, 0.147%). Kontur 2023 is 1.017x the 2021
census. Per district 0.96x to 1.01x, except **Temburong at 1.66x** (15,678 against 9,444, inside the
0.60-1.70 band); it moves only placement inside Temburong, since the census fixes each district's
total. `kontur_cap.py bn` found no block at the cap. Median unit about 1,260 km2, 296 to 682 hexes
each, so no grid-floor concern. The water clip left 4 hexes that are over 95% sea unclipped.

Per mukim, Kontur against C1 is printed and not used: Mentiri 0.43x, Serasa 0.60x and Telisai 0.58x
low; Lumapas 2.39x, Kianggeh 1.82x and Bangar 2.17x high; the river mukims Sungai Kebun, Peramu and
Tamoi hold almost no hex. With the district layer and Kontur agreeing within 4% for the three large
districts, no calibration to mukims was made.

## 6. What the table shows

Islam 82.15%, Christianity 6.69%, Buddhism 6.30%, Others 4.87%. Every district has a Muslim
majority: Brunei Muara 84.47%, Tutong 84.23%, Temburong 75.46%, Belait 70.31%. Belait is 11.04%
Buddhist and 10.72% Christian; Temburong 12.78% Christian and 1.06% Buddhist; Tutong 11.00% Others
with 2.44% Christian and 2.34% Buddhist. Brunei Muara holds 72.3% of the people and 68.1% of the
Christians.

## 7. §14 was considered and no ask was filed

Not escalated. The grain is four districts averaging 110,000 people, coarser than the units asks
001, 017 and 018 cleared; the table is the state's own publication, forwarded to UNSD; and nothing
here locates a group more finely than DEPS already has. No search for restrictions on or attacks
against religious minorities in Brunei was made for this build, so this rests on the grain and the
publisher, not on a safety check; a session that reads such a record should reread this.

## 8. Gotchas

- **The 2023-09-30 Annex A capture is a 1 MiB fragment.** The whole file is in every capture from
  2024-06-24.
- **The Malay and English row labels for Islam are the same word**, on separate rows; the Malay row
  carries no numbers. `read_grid` takes only label rows that hold a number.
- **Temburong's A10 has two empty cells** where a zero belongs.
- **geoBoundaries spells Bokok `Bunkok`**, and its 2006 mukims have one `Gadong`.
- **C1's mukim rows are indented with spaces; district rows are not.** That is the only thing
  telling them apart.
- **A two-layer area test hid the misplacement that mattered** (§5). Now in `playbooks/geography.md`.

## 9. Reopen when

- A table crossing religion with race or mukim turns up (the full `RPT.pdf`, the 2011 tables).
- A published split of Others: Hindus, no religion, indigenous religion.
- An official district boundary file (DEPS, Survey Department) appears; the tracing is the weakest
  link, though it agrees with Kontur.

## 10. Terms

DEPS's reports and workbook are public files on its former site; no licence text was seen and none
was looked for. geoBoundaries gbOpen BRN ADM1 and ADM2 are public domain (Wikimedia Commons
tracings). Kontur Population is CC BY 4.0.
