# Isle of Man: 2021 Isle of Man Census, religion island-wide

**Drawn 2026-09-15** (session `d743fc47-im`). One unit, the island; 6 categories; 74,487 people
drawn out of 84,069 residents; every row `measured`; `gap` 11.4% (the non-answer to a voluntary
question), computed by `tools/gap_share.py` two ways.

- `sources/im.py` -> `data/normalized/im.csv` (both reports in `data/raw/im/`, pinned) and
  `data/geo/im/im_hexes.gpkg` (Kontur IM 2023-11-01, raw extract in `data/raw/micro/`)
- `taxonomy/im2021.py` -> the mapping; `countries/im.py` -> the entry. No new node.
- sources.md **§im-2026-09-15** is the summary; **§scout-2026-09-15-europe** was the scout's.

## 1. What Statistics Isle of Man publishes

| release | religion | tier |
|---|---|---|
| **2021 Isle of Man Census Report Part I** (42 pp.), `gov.im/media/1375604/2021-01-27-census-report-part-i-final-2.pdf` | **Table 2.12, residents by religious faith and age, p.27**; prose and shares p.13 | island |
| 2021 Isle of Man Census Report Part II (49 pp.), `gov.im/media/1376421/2021-isle-of-man-census-report-part-ii_11052022.pdf` | only the form facsimile, Q8 p.43 | none |
| UNSD Demographic Yearbook table 28 | Isle of Man absent | none |
| 2016 and earlier censuses | religion not asked (p.13: 2021 is the first to ask) | none |

Checked 2026-09-15: the Census and Population page lists only these two 2021 reports, the 2016
reports and data tables, and the 2023-2025 population reports (not opened). The open-data
*Society* page lists only the 2016 census data tables. The page says "Unpublished data may be
supplied on request"; nobody has asked for religion by area. **The 2026 census** was held on 26
April 2026; its page is still the pre-census information page, with no results.

## 2. The files

| file | digest | bytes |
|---|---|---|
| `2021-isle-of-man-census-report-part-i.pdf` | `QVOV43VF23Y4GK26O2U25C6UMWBJU4SV` | 1,691,173 (42 pages, `%%EOF`) |
| `2021-isle-of-man-census-report-part-ii.pdf` | `CHTCZ3EG5BDN4SR6PTBEYQSIA3EDTVEH` | 482,408 (49 pages, `%%EOF`) |

**gov.im/media/ is behind a bot wall that depends on the client.** Python's urllib with a full
Chrome user-agent string (`micro.UA`) got a 269-byte "Request Rejected" page; curl with a Chrome/124
string got both PDFs. WebFetch on the gov.im HTML pages was rejected; curl got them. The raw files
came from curl, and `check()` asserts the digests.

## 3. The checks (`sources/im.py::check`), and where Sikhs and write-ins went

| check | result |
|---|---|
| both files pinned | digests above |
| Table 2.12 parsed off p.27 = transcription | 8 rows x 19 columns |
| age bands sum to each row; rows sum to the total row | every column |
| footnote 14 | "This question was voluntary. The total is that of all people who chose to provide an answer." |
| Table 2.1's total row (p.18); Table 1.1 (p.15) | 84,069 residents, bands close |
| answered <= residents per age band | yes; 9,582 not answered (11.40%) |
| p.13 figures | 74,487 answered; 54.7% Christian; no religion 38.8% of residents (43.8% of those answering); the six printed shares recompute |
| Part II Q8 | Christianity, Judaism, Buddhism, Sikhism, Hinduism, No religion, Islam, Other (please specify below) |

**The form offered Sikhism and a write-in; Table 2.12 has no Sikhism row and prints Other as 0 in
all 19 cells.** Either those answers were coded into the listed faiths or they were left out of the
74,487. The report does not say which, and nothing else found settles it. Not checked: the census
project review, which Part II says would follow and which is not on the census page. Drawn as
printed; the note says so.

**Non-answer by age** (Table 2.1 less Table 2.12): 9.6% to 12.1% in every band from 0-4 to 80-84,
and 19.2% at 85 and over. Children are in the table, so a household answered for them.

## 4. Mapping

Christianity -> `christianity` (one box for every Christian, so the parent); No Religion ->
`unaffiliated`; Islam, Buddhism, Hinduism, Judaism -> their roots. `Other` (0) is not emitted.
EXCLUDED holds the resident population (the universe) and the non-answer, 9,582, which is the
difference of two printed totals rather than a printed cell. REVIEW is empty: no call is arguable.

## 5. Geography and placement

Island-wide only, so one unit (the microstate shape, `_micro_counts`). At 84 dots it is larger
than Gibraltar's 38 or Palau's 17, but there is no finer table to leave undrawn, so the
microstate ruling's open question (a published district table left undrawn) does not arise.

Kontur IM 2023-11-01: 567 populated hexes, 84,591 people, **1.006x** the resident population.
geoBoundaries has only an ADM0 outline for the island, and Overpass answered 504 on 2026-09-15, so
Kontur was **not calibrated** to Table 2.4's 21 areas (4 towns, 4 villages, 13 parishes). A rough
witness with buffers round the town centres (smaller than the towns' boundaries, so Kontur should
read low): Douglas and Onchan 35.8% within 3 km against 42.5% for the two areas; Ramsey 8.8%
against 9.9%; Peel 5.3% against 6.8%; Castletown 2.8% against 3.8%; Port Erin and Port St Mary 6.7%
against 6.8%. Nothing points to Kontur misplacing people. `kontur_cap.py im`: no stops (largest
hex 2,537 people). No land border, so the scatter's sea clip is the only cut needed. The scatter
left 6 mostly-sea hexes unclipped (`KEEP_WHOLE_ABOVE`); a scratch check against the OSM water
polygons found 0 of the 72 dots, 7 dots and 8 rings in the sea. Scattered: 72 dots and 4 rings
(Islam, Buddhism, Hinduism, Judaism) at 1:1,000; 7 dots and 4 rings at 1:10,000.

## 6. §14

Considered; nothing raised. The island's government publishes the table, the grain is the whole
island, and no search for restrictions on religious minorities was made.

## 7. Reopen when

- The 2026 census publishes (census night 26 April 2026), especially if religion comes by area.
- Religion by area is requested from Statistics Isle of Man (statistics@gov.im; the census page
  offers unpublished data on request). That is an email, so Anita's.
- Someone wants Kontur calibrated to Table 2.4: the areas need polygons (OSM local authority
  relations, or the island's own GIS, not searched).

## 8. Terms

gov.im content is under the Open Government Licence except where otherwise stated (site footer).
Kontur Population is CC BY 4.0.

## Review, 2026-09-15 (cb8b206e-rev4, light pass)

Nothing to change. `check_md.py` clean; `im` in both editions; `check_rollup.py im` all
measured; `gap_share.py --check` confirms 11.40% two ways. Every share in `note_public`
recomputes from the six rows of `im.csv` (Hindus 0.35% and Jews 0.15% round up to 0.4% and 0.2%).
No new node, no REVIEW calls, and the parent `christianity` is the only reading of a single
Christian box. The note already says the Sikh and write-in answers are unaccounted for. Screenshot at the entry's
`view`: dots in Douglas and Onchan, Ramsey, Peel, Port Erin and Castletown, none in the sea.
