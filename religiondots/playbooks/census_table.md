# Census table playbook

A statistics office's own religion table (census or register count), read from xlsx, PDF, a tabulation
server or UNSD's copy into `data/normalized/<cc>.csv` and a mapping. It is the route wherever a census
asked religion; boundaries, name joins, population bases and placement are in `playbooks/geography.md`.

## Used by
- `gw` Guinea-Bissau: PDF annex in counts, equal to UNSD to the person; misprints, a prose swap, a pinned digest (sources.md §9dx).
- `mz` Mozambique: per-province xlsx on INE's retired Plone site, found by a Wayback CDX prefix query (§9df, §9dy); since 2026-09-15 also the 2007 district volumes' one-decimal shares with N, fitted by IPF to 2017's province answers and district populations (sources.md §mz-2026-09-15).
- `cg` Republic of the Congo: Wayback `id_` copy of a squatted domain's PDF; the form's codes fix the column order (§9dv).
- `gn` Guinea: one-decimal shares by région times printed populations, rescaled to UNSD (§9dh).
- `ir` Iran: SCI's 1395 yearbook table in counts, Persian digits in the text layer, the bold national row a picture (§ir-2026-09-14). A Persian journal PDF (§ir-2026-09-15) decomposed the `لا` ligature (`گیالن` for `گیلان`) and mixed Persian and Latin digits within one column: fold both before matching a name or a number, and join on the printed population as well as the name (`ir_split.py::_fold`).
- `zm` Zambia: PDF tables with two wrong column headers, settled by the office's analytical report (§9db).
- `pk` Pakistan: PBS Table 9 PDFs under `wp-content/uploads/`, right-aligned columns read by drawn rules (§9du).
- `td` Chad: shares by région from a Wayback copy, raked to two printed margins; animist offered beside no religion; COD's provinces rebuilt to 2009 on the office's areas (sources.md §td-2026-09-14).
- `bf` Burkina Faso: annex counts by province on a retired tree, equal to UNSD, summed by région to a second annex table; geoBoundaries because COD-AB moved to a 2025 reform (sources.md §bf-2026-09-14).
- `sl` Sierra Leone: one-decimal shares by district on printed district totals, scaled to the household population; no national count in persons exists, so no rescale; a misprinted national row pinned (sources.md §sl-2026-09-15).
- `sn` Senegal: one-decimal shares by région with the Sufi brotherhoods as codes, on printed populations; a cell printed in the wrong column, settled by a regional report in counts, which also draws Diourbel at département (sources.md §sn-2026-09-15).
- `bn` Brunei: counts by district in a workbook and an annex PDF, both only on Wayback (one capture a 1 MiB fragment), equal to UNSD; the form's Hindu folded into Others (sources.md §bn-2026-09-15).
- `gi` Gibraltar: report PDF, counts by residential area, drawn as one unit on 20 Kontur hexes; the 78 enumeration areas rebuild every area and place one institutional EA the appendix does not (sources.md §gi-2026-09-15).
- `mt` Malta: counts by the 68 localities in a born-digital PDF, crossed with the single-year age table to show the religion totals are the whole population aged 15 and over; GISCO LAU joined by name, witnessed by code district and printed area (sources.md §mt-2026-09-15).
- `ps` Palestine: counts by governorate in a bilingual PDF read by English row label; three nested universes (Palestinians counted, everyone counted, plus the estimate) settle East Jerusalem (sources.md §ps-2026-09-15).
- `ne` Niger: annex counts by région equal to UNSD; the body's share table is of those who stated a religion, with three cells forced to close, and the density table doubles as the office's area per unit (sources.md §ne-2026-09-15).
- `gm` The Gambia: annex counts by LGA with sex and urban/rural tables; one both-sexes table reprints the urban one and closes on itself, caught by the sex and urban-rural identities (sources.md §gm-2026-09-15).
- `fo` Faroe Islands: PxWeb API counts by district from a select-all-that-apply question, ticks scaled to people per district; suppressed small bodies folded into Other; districts rebuilt from the register's villages (sources.md §fo-2026-09-15).
- `im` Isle of Man: island-wide counts in a report PDF, one unit on Kontur; the form offers Sikhism and a write-in that the table never prints (Other 0 in every cell), so read the form against the rows; gov.im refuses urllib and serves curl (sources.md §im-2026-09-15).
- `qa` Qatar: HTML table pages only on Wayback, pinned by CDX digest; a count for Qataris and a sample calibrated to the counted totals for everyone else, said only on the census's introduction page (sources.md §qa-2026-09-15).
- `us` United States (Sikhs, Yazidis): 2020 Census Detailed DHC-A race write-in by county and tract, noise-infused and thresholded, subtracted from a survey line (sources.md §us-2026-09-15).
- `tk` Tokelau: counts by atoll in a workbook for the usual residents present on census night, a fifth short of the official de jure count; UNSD's 2016 row is 4 people off, its 2011 row equal (sources.md §tk-2026-09-15).
- `mn` Mongolia: 22 per-aimag PDF volumes of chained one-decimal shares, each table found by its identity; Ulaanbaatar's düüregs measured off a vector chart (§9bt, sources.md §mn-2026-09-15).
- `pw` `ck` `tv` and eight more: UNSD table 28 is the whole source (`sources/micro.py`); `vg` `aw` in `sources/terr.py`.
- `pe` Peru, `ni` Nicaragua: REDATAM. `et` Ethiopia: USCB geodatabase (also `bd`, `jm`, `vc`, `cf`, `pk2017`).
- `lc` Saint Lucia: questionnaire catches a mislabelled column. `cv` Cabo Verde: UNSD `Unknown` is the under-15s.

## Loading it
- `python tools/oracle.py --fetch` once (cache `data/raw/unsd/dyb_table28_values.zip`), then
  `python tools/oracle.py "<UNSD name>"` for the categories and counts to expect. Use UNSD's name (`--list`).
- Raw files in `data/raw/<cc>/`, written through `.part` and `os.replace`. `sources/<cc>.py` has `fetch()`, a
  reader, `check()`, `emit()`, and writes `micro.COLUMNS` with the source's label verbatim, NFC. Copy the shape of
  `sources/gw.py` or `cg.py` (PDF), `mz.py` (xlsx), `pe.py` (REDATAM): transcribe the table as constants, parse it
  off the page, assert both agree, run checks through `say(ok, msg)`, `SystemExit` if any failed.
- Mapping `taxonomy/<cc><YYYY>.py`: `EXCLUDED`, `REVIEW`, `MAP`, `COLUMNS`, `_key`, `resolve` (model:
  `taxonomy/gw2009.py`). A source script with a year takes an underscore (`sources/pk_2023.py`); a second mapping
  vintage needs `taxonomy/registry.py::OVERRIDE` first, or `discover` raises for every session.
- `python sources/<cc>.py --fetch`; `python taxonomy/build_tree.py` after a new node;
  `python tools/check_mapping.py <cc>`; `python tools/gap_share.py <cc> -v`. The rest is `COMMANDS.txt`.

## Traps
- **A negative from one publication is not a negative for the country.** Zambia, Mozambique, Guinea and
  Guinea-Bissau were closed on one release or a volume title, then drawn from the same office; Equatorial Guinea was closed on its 2015 results volume while question 9 of its form asks religion (sources.md §scout-2026-09-15-negatives). Open each volume's
  table list (the structure volume first in francophone Africa), newer and older censuses, per-province series,
  the national-language tree, and the national statistical yearbook (Iran's 1395 census tree has no religion
  topic; the 1395 yearbook's population chapter prints it by province); record what was asked, of what, when. Caught by: Not checked yet
  (`WORKFLOW_PLAN.md` item 8). Detail: spec §12 "No country is closed for good", "A VOLUME'S TITLE IS NOT ITS TABLE LIST".
- **UNSD table 28 is a floor and a transcription.** Absence proves only nothing was forwarded. A row can be the
  under-15s (Cabo Verde) or collective households (Guinea); a value can be a typo (Burundi 494,533 for 484,533)
  or a later edit (Mozambique). `NOT a partition` means open the office's table; the office wins. Caught by:
  `tools/oracle.py::main`, `sources/micro.py::normalise`, `sources/terr.py::ORACLE`, `sources/gn.py::check`,
  `sources/cv.py::check`; a known disagreement pinned, `sources/mz.py::check` via
  `sources/fetch_checks.py::pinned_differences`. Detail: spec §12 "AN ORACLE CATEGORY COUNT COUNTS ROWS", "UNSD'S NUMBERS CAN BE WRONG TOO".
- **Ask for a REDATAM server before reading PDFs.** Nicaragua prints 17 departments and serves 2,579 comarcas.
  `RpWebStats.exe/Frequency?BASE=<base>&ITEM=FREQPOB` lists variables (unclosed `<option>` tags); a bad program
  answers `Tabla vacía` with 200; `prod.redatam.org` does not namespace `BASE=` by country. Caught by:
  `sources/pe.py::_save`, `sources/ni.py::fetch`; the shared host Not checked yet (assert the base's area names).
  Detail: spec §12 "Finding the data".
- **USCB geodatabases carry census tables.** `data.humdata.org/api/3/action/package_search?q=organization:us-census-bureau&rows=100`;
  religion for Ethiopia, Pakistan 2017, Bangladesh, Jamaica, Saint Vincent, CAR. `-999` is a null that sums;
  the `Metadata` sheet says what tables omit (Jamaica's parishes drop four religions). Caught by:
  `sources/et.py::check` (counts `-999`); Metadata Not checked yet. Detail: `sources/et.py` docstring.
- **A retired site is usually whole in the Wayback Machine.** Query the old path prefix, not the new portal
  (`web.archive.org/cdx/search/cdx?url=<host>/<old path>/&matchType=prefix&fl=original,timestamp,digest`), or
  `matchType=domain` on the whole government domain (Eswatini). https only; grep locally, as a bad `filter=`
  returns 500 and nothing; retry 503s; raw bytes via `web/<ts>id_/<url>`. Caught by:
  `sources/fetch_checks.py::parse_cdx` (a non-CDX answer raises), `::cdx_url`, `::one_per_digest`, `::wayback_raw`;
  no loader uses them yet. Detail: spec §12 "A PORTAL MIGRATION HIDES FILES".
- **A host that answers can be the wrong host.** A lapsed domain serves a parking page (`isteebu.bi`) or the
  real PDF with spam links (`cnsee.org`). A 200 under a kilobyte is a bot wall; "unable to get local issuer
  certificate" is a missing intermediate; try a second client. Never iterate on headers: give Anita the URL.
  Caught by: `sources/fetch_checks.py::check_body` (a 200 under 1 KB, `forbid=`), from `sources/cg.py::fetch`;
  the second client Not checked yet. Detail:
  spec §12 "A DEAD OFFICE CAN ANSWER 200", "A BOT WALL AND A TLS FAILURE", "A SQUATTED OFFICE DOMAIN".
- **An API index lists what registers with it, not what the site serves.** WordPress: `wp/v2/search` on the
  publication name, then `wp/v2/media` (or `index.php?rest_route=`) paged in full; media misses `wp-content/uploads`
  (Pakistan). WP File Download: grep the page for `action=wpfd`, then `files.getFiles&id=0` paged by `page`. A
  shut listing often sits beside an open `/download/<int>` naming files in `Content-Disposition` (`sources/mw.md`
  §1). Count requests first. Caught by: `sources/fetch_checks.py::wp_json` (HTML, error object, `X-WP-TotalPages`);
  coverage of `uploads` and WPFD Not checked yet. Detail: spec §12 "A WP FILE DOWNLOAD INSTALL IS INVISIBLE".
- **A JavaScript portal or dashboard sits on a real API.** Compare the 404s, grep the bundle (or its Wayback
  copy) for `/api`, search `<host> api`; a Qlik dashboard held Kazakhstan's microdata. PxWeb: try
  `/pxweb/api/v1/en/` and `/api/v1/en/`; a 403 can be a cell limit. Caught by: Not checked yet. Detail: spec §12
  "Finding the data".
- **A 200 is not the file.** Assert magic bytes: `%PDF-`, `PK\x03\x04` (xlsx), `\xd0\xcf\x11\xe0` (xls); `<?xm` is
  SpreadsheetML, read by `xml.etree` (honour `ss:Index`). A slow openpyxl open is `xl/styles.xml`: drop it, open
  `read_only=True`. Find header rows by label. Caught by: `sources/fetch_checks.py::check_body` (magic bytes),
  `sources/mz.py::fetch`, `::read_one`, `sources/al.py::fetch`; the stylesheet strip Not checked yet. Detail: spec §12 "Estimating the work".
- **A truncated PDF opens.** A 1 MiB Wayback fragment with no `%%EOF` opened as all 92 pages, yet a Word PDF can
  keep 93 KB after `%%EOF` and be whole (Congo). Pin the SHA-1 in CDX base32 `digest` form or the exact size,
  fetch one capture per digest, check pages and text on every page. Caught by: `sources/fetch_checks.py::check_body`
  (a pin is the verdict; unpinned without `%%EOF` stops and says where the markers are), from `gw.py`, `gn.py`,
  `cg.py`, `td.py`; `::check_pdf_doc`; `sources/gw.py::check`, `sources/cg.py::check`. Detail: spec §12 "A TRUNCATED WAYBACK CAPTURE OPENS AS THE WHOLE DOCUMENT".
- **Cells that look like numbers.** Classify every cell through one function that raises on the unknown; never
  `errors="coerce"` or filter by type (Germany lost 2.2M counts stored as text). List count columns; a `%` twin
  sits beside each. A clipped PDF cell parses short (Guyana `38,96`): assert it prefixes the sum. Caught by:
  `sources/gy.py::check`; a shared classifier Not checked yet. Detail: spec §12 "Parsing the table".
- **PDF tables are read by geometry.** Assign right-aligned numbers by right edge, to edges from the header's
  drawn rules (`page.get_drawings()`) or the n-1 largest gaps; anchor on the header row; bound by caption and next
  caption; cluster rows on vertical centre; name columns by label (Austria prints `1 2 3 5 4`). Render before
  blaming the parser. Caught by: `sources/pk_2023.py::_column_edges`, `::check_block`. Detail: spec §12
  "Reading numbers out of a PDF", "A RIGHT-ALIGNED PDF TABLE IS READ BY ITS DRAWN RULES".
- **Scanned tables.** Captions can be text while numbers are images: compare `len(page.get_images())` with rows
  read. Render, transcribe, let arithmetic check it; a chart can be measured against its printed totals
  (`sources/jp_checks.py`). Caught by: `sources/fetch_checks.py::image_pages`; no reader calls it yet (belongs in
  each PDF reader). Detail: spec §12 "A PDF CAN
  HAVE A TEXT LAYER FOR ITS PROSE AND PICTURES FOR ITS TABLES".
- **A chart in a PDF is usually vector, so measure it, and its prose can name the wrong segment.**
  Ulaanbaatar's 2020 volume gives religion by düüreg only as two charts: every bar is a filled
  rectangle in `page.get_drawings()`, and the labels and names are glyph outlines with no text.
  Take widths at one scale fitted on the printed labels, never per row (Baganuur's printed values
  sum to 100.1, so its bar is wider); check transcribed names by glyph path signature (one outline,
  one letter, across every name); check each label sits inside the segment it is credited to. The
  prose called Nalaikh's 13.6% Christian; the fill, the label and a reweighting of the nine units
  against the city's printed table all say Islam. Caught by: `sources/mn_ub.py::read_fig36`,
  `sources/mn.py::read_ub`. Detail: `sources/mn.md` §11.
- **A printed 0.00% religion non-response means the blanks went somewhere.** A form with no
  non-response code and a report printing none leave blank answers inside a real code. Cross another
  table's non-response row: Burkina Faso's age-not-recorded row (A5.7) is 18.7% `Autre` against 0.57%
  overall. Say so in REVIEW and the `other.<cc>` description; do not move dots on a guess. Caught by:
  `sources/bf.py::check` (A5.7's ND row); a shared check Not checked yet. Detail: `sources/bf.md` §4.
- **A share table can have non-response prorated in, and say so only in another volume.** Mali's
  RGPH5 prints no religion non-response anywhere in its religion chapters; the cultural volume's
  annex A01 counts 48,746 `Non Déclaré`, and Tableau 2.01's counts are A01's with that row spread
  in proportion to within a person. Record it, never undo it. And a one-decimal table can be
  forced to close: all 21 rows of Tableau 6.13 sum to exactly 100.0, with the slack in the last
  cells (Koulikoro `Autre religion` 0.5 against 0.36 at two decimals). Where two volumes print
  one table, take each cell from the finer one. Caught by: `sources/ml.py::check` (the proration,
  the closure, cell-by-cell against 2.03); a shared check Not checked yet. Detail: `sources/ml.md` §2.
- **A sum that does not close can be three universes nesting.** PCBS's 2017 book prints Palestinians
  counted (Table 3, the religion table), everyone counted (Table 2) and counted plus the
  post-enumeration estimate (Table 25). The scout read Table 25's J1 + J2 = 435,483 against Table 3's
  392,835 as a contradiction; it was the largest universe against the smallest. Name each table's
  universe from its caption and footnote, then assert the order per unit before calling a table
  inconsistent. Caught by: `sources/ps.py::check` (steps 5 and 6). Detail: `sources/ps.md` §2.
- **Space as the thousands separator makes a row ambiguous as text** (`2 622 730 55`). Solve each
  row's cell boundaries against its printed total, then compare digit strings with the transcription.
  Caught by: `sources/bf.py::segments` with `check`; the solver lived in a scratch script, Not shared yet.
- **A religion table can be multi-response, and suppressed small cells can hide inside `Other`.** The Faroe
  Islands' 2011 congregation table counts a member of two bodies in both (a `More than one` row of 4,596 people),
  so its categories sum 5,122 above its responses; its district cells blank the small bodies (`...`) and fold them
  into each district's `Other congregations`, which sums to 585 over the districts against 106 nationally. Sum
  each level's categories against its responses, and each category over the units against its national cell.
  Before choosing a rule to turn ticks into people, test the rule's premise on the table's own cells: the
  Faroese overlaps could not all be National Church plus missionary movement (six districts too few), so
  the rescale is proportional. Caught by: `sources/fo.py::check` (checks 4 to 6). Detail: `sources/fo.md` §3.
- **An appendix's list of what makes up each unit can be wrong for one piece, while the table closes.**
  Gibraltar's Appendix 9 builds residential areas from enumeration areas and leaves institutional EAs
  80-87 to the Institutions row; Table 42 counts EA 85 (70 people) in South District. Only rebuilding
  every area from Table 43's EA rows shows it: Institutions short and South District over by EA 85's
  row in all 19 cells. Rebuild each unit from the finer table, fix shared pieces from the units that
  pin them, and let the shortfall name the piece. Caught by: `sources/gi.py::solve_shared`, `::check`.
  Detail: `sources/gi.md` §3.
- **One volume can name a unit two ways, and only a name-set comparison catches it.** Malta's Table
  1.5 heads Gozo's Żebbuġ page `Iż-Żebbuġ, Għawdex` where Tables 1.2, 1.10 and 5.3 print `Iż-Żebbuġ`.
  A count of 68 pages passed; comparing the sets failed, and the under-15s came out 423 short, exactly
  that page's. Compare name sets, not counts, between every pair of tables a check crosses, and alias
  a variant by name in code. Caught by: `sources/mt.py::read_age_pages` (`T15_ALIAS`), `::check`.
  Detail: `sources/mt.md` §2.
- **pandas deletes a category named `None`**; all three cases were the no-religion row (Zimbabwe 1.26M). Read
  normalised files with `keep_default_na=False, na_values=[""]`; NFC labels both ways. Caught by:
  `tools/check_na_readers.py::main` (every `read_csv` of a file holding an NA string; `countries.py::_micro_counts`
  still drops Bermuda's, Niue's and Tuvalu's `None`, a registered WARN), `tools/check_mapping.py::main` for NFC.
- **Reconcile every column at every level, then cross a second table.** A Total-only test passed units with no
  religion columns (Indonesia) and a doubled `Grad` level (Serbia); a national line given to no unit (China's
  military) is asserted by name. A consistent column swap survives in-table identities: cross sexes, rural and
  urban, form codes or UNSD. Caught by: each `check()`, `sources/pk_2023.py::check_block`, `sources/mz.py::read_one`.
  Detail: spec §12 "Reconciliation discipline".
- **Read the questionnaire.** It alone catches a mislabelled column (Saint Lucia's `Mennonite` is `Evangelical`);
  says if figures were weighted up or non-response prorated (Guyana: record, never undo), if a non-response code
  exists (Congo: none, no `gap`), if write-ins were back-coded (Nauru). Caught by:
  `sources/lc.py::confirm_questionnaire`, `sources/cg.py::check`; proration Not checked yet. Detail: spec §12 "A
  QUESTIONNAIRE CAN MAKE A TABLE DEEPER THAN THE QUESTION".
- **The office's table can be wrong while it closes.** A typo is one row and one column short by one round number
  (Guinea-Bissau 21,000): fix the slipped digit, say other splits close too. Zambia's `Judaism` is traditional
  religion, by the analytical report read every way: assert both ways. Guinea-Bissau's prose swaps two regiões.
  Caught by: `sources/gw.py::ethnic_witness`, `::check`, `sources/zm.py::check_relabel`. Detail: spec §12 "A
  TABLE'S OWN TOTALS FIND THE OFFICE'S TYPOS", "A PRINTED COLUMN HEADER CAN BE WRONG", "PROSE CAN SWAP".
- **A cell can sit in the wrong column with every row still closing, and the national row can be
  another tabulation.** Senegal 1988's Tableau 1.15 prints Diourbel's Khadriya under Layène (the
  orders still sum to Musulmans), and its Ensemble row sums to 99.7 and misses the population-weighted
  régions by up to 0.8 points. Weight the rows by the printed populations and compare each column: one
  misplaced cell moves one column away (Layène 0.93 against 0.6, 0.60 swapped back), while a row
  that disagrees in most columns is a different tabulation, so do not rake to it. A regional report
  in counts settles both. Caught by: `sources/sn.py::check`. Detail: `sources/sn.md` §4.
- **A religion can hide in another question's write-ins, and a noise-infused table does not add up by
  design.** The US census asks no religion, but its race write-in codes "Sikh" and "Yazidi" as detailed
  groups (DHC-A T01001, ITERID 3845 and 1207). The Bureau adds noise to every cell separately and prints
  a county or tract only at 22 or more, so tracts need not sum to their county (checked at 11 per tract)
  and counties sum to 96.9% of the nation. A withheld cell is `-888888888` with `ANN` `X`; DC's county
  row repeats its state row and sits under the threshold (19). Look up each ITERID's label in the
  iterations list rather than trusting a code. The count is of people who wrote the word, a floor (70,697
  against a community estimate of 500,000), so draw it as counted and subtract it from any survey line
  that already holds those people. Caught by: `sources/us_dhca.py::classify`, `::check`. Detail:
  `sources/us_dhca.md` §2.
- **An unmapped category vanishes.** `EXCLUDED` holds universe rows and non-answers, each with a sentence and
  which way it leans (§3.5); `REVIEW` every arguable call, naming the node wanted. A parent beside children that
  sum to it on every row is a duplicate; else emit the remainder. Never map on the string alone. Caught by:
  `tools/check_mapping.py::main`; the lean Not checked yet (`tools/gap_share.py`). Detail: spec §12 "Taxonomy".
- **`gap` and `gap_share`.** A printed non-response column is computed (`tools/gap_share.py <cc> -v`, `--write`).
  People in no table (collective households, restricted areas, unasked ages) are hand-written and join `gap`
  (Guinea's precedent); never add the two silently. Caught by: `tools/gap_share.py --check`; the foot of
  `countries.py` asserts `gap` states `gap_share`. Detail: spec §10.4a; `countries.py` docstring.
- **A refugee block given its origin country's national mix is wrong where refugees come from one end of it.**
  Mauritania's 46,800 Mbera refugees on Pew's Mali row carried 2,745 non-Muslims, 21.6% of every non-Muslim
  drawn in the country; they are from northern Mali, 99.7% Muslim in Mali's own census, which gives 379. Look
  for UNHCR's areas-of-origin map for the camp and weight the origin census's regions by it. Caught by:
  `sources/mr.py::refugee_composition`, which prints both mixes; no shared helper. Example: `mr`. Detail:
  `sources/mr.md` §7.
- **Refugees the census missed look like refugees who said they were nationals.** Set the census's count of
  the camp itself against UNHCR's camp figure: a refugee who told the census they were a national is still in
  the camp's census population, so a short camp count is people never counted, and they go in `gap` (Mbera:
  about 41,200 counted, almost 100,000 by UNHCR). UNHCR's API (`api.unhcr.org/population/v1/population/`)
  returns no rows unless `cf_type=ISO` is passed with ISO3 codes. Caught by: Not checked yet; it belongs beside
  `tools/gap_share.py`. Example: `mr`. Detail: `sources/mr.md` §8.
- **A foreigner layer on a national nationality mix can use the sexes, and two origin rows are wrong in the
  Gulf.** Saudi Arabia's census gives non-Saudis per region and their nationality only nationally, but both by
  sex (a regional sex-ratio chart, national nationality tables by sex). The sexes carry different streams
  (non-Saudi men 5.3% Christian, women 25.4%), so each region's men take the men's mix and its women the
  women's; against one mix Makkah gains 31,000 Christians. Before trusting Pew's origin rows, look for the
  stream: GAStat's `Burma` are the Rohingya (Pew's Myanmar row would draw 158,240 non-Muslims), and India's row
  draws 2.3x Pew's own Saudi Hindu figure, because Indians in the Gulf are mostly Muslim (Pew, *Faith on the
  Move*, pp.21-22). A chart's bar values can be in the text layer out of order: read them off the rendered
  page, then assert the values are on the page and that each overall bar follows from its two parts. Caught by:
  `sources/sa.py::report_checks`, `::main` (the Rohingya and India lines, the by-sex against one-mix table).
  Example: `sa`. Detail: `sources/sa.md` §4.
- **`other.<cc>`, `unknown` and tiers.** A source's `Other` gets a new `other.<cc>` in `taxonomy/branches.py`,
  a real religion (§6.3a-iv); `unknown` is counted people no geography can place (§6.3a-ii). Own-geography counts
  and shares times a total in the same publication (Benin) are `measured`; spread from coarser is `derived` with
  `fill=` (Zambia). Caught by: `tools/check_mapping.py::main`; tier Not checked yet. Detail: spec §7, §2.7a.
- **A no-religion box (draft, `WORKFLOW_PLAN.md`).** Read the form. Traditional offered separately:
  `unaffiliated` (Guinea-Bissau, Chad). Lumped: `unknown` until a national source asking both (MICS/LSIS, DHS,
  Afrobarometer, Pew) puts one reading at 80%+; draw it, say whether it is self-description or practice, record
  source and share in `sources/<cc>.md` and `REVIEW`. Caught by: `tools/check_no_religion.py::main` (a box naming
  traditional religion needs a traditional sibling or a classification; a lumped box drawn as one reading needs source,
  80%+ and what it measures); a form nobody read Not checked yet. Detail: sources.md §9dn, §9dy.
- **A census table can be a count for some people and a calibrated sample for the rest, and print
  plain counts that close every way.** Qatar's 2004 census enumerated Qataris in full and sampled
  non-Qataris (households and labour gatherings), then weighted the sample to the counted
  population by municipality and sex. Table 6 closes on Tables 1-5 and equals UNSD to the person,
  and nothing on it or on the form says part of it is an estimate; only the census's introduction
  page does. Read the methodology or introduction beside the form, name who was sampled, and say
  in the tier reasoning whether the printed unit is the sample's stratum (Qatar's is, so
  `measured`). Caught by: Not checked yet (a reading step; record it in `sources/<cc>.md`).
  Detail: `sources/qa.md` §3.
- **A religion table can cover only the residents present while the official count is de jure.**
  Tokelau's 2016 Table 5.8 counts the 1,197 usual residents present on census night; the official
  count is 1,499, adding 254 absentees their household head described and 48 public servants in Apia
  on a short form, none asked religion. The table closes on itself and on UNSD, so nothing flags the
  missing fifth. Read each table's universe line against the report's population definitions,
  rebuild its unit totals from the de jure and absentee tables, and put people no table asks in `gap`
  in the official count's universe. Caught by: `sources/tk.py::check` (step 3, de jure minus
  absentees per atoll). Detail: `sources/tk.md` §4.
- **A census microdata sample supports a tier by its noise against its parent, not by significance.**
  Uganda's 10% file has 2,207 subcounties and 10,852 parishes; a split-half rank test against zero passes
  nearly every answer at parish, because district geography alone ranks parishes. Deal HOUSEHOLDS into
  waves (a quarter of Ugandan households hold two religions), test each tier's departure from its parent
  with a within-parent shuffle null, and take a tier only where the median half-sample Pearson of the
  departures is 1/3 or more (the unit's own share then has lower expected squared error than the parent's).
  Take people from the full-count unit table, not the sample, and check the sampling fraction per unit.
  Caught by: `sources/ug_2024.py::stability`, `::join` (fraction band). Detail: `sources/ug.md` §0.4.
- **A share table's N can sit a line off its row, and a Total row can be the right numbers in the
  wrong cells.** Mozambique's 2007 district volumes print one-decimal shares with each district's
  `N`; in five of nine the text layer puts some `N` on the line above the shares, or between the
  name and the shares, so a line parser pairs the wrong population. Read the table as one token
  stream and pair the kth count with the kth row, then prove it: the `N` column sums to the Total
  row, and the N-weighted shares match the Total row within the rounding. That second check is
  also what found Gaza's printed Total row to be its districts' eight numbers set in the wrong
  cells; pin the misprint and keep the districts. Caught by: `sources/mz_2007.py::read_volume`,
  `::check_volume` (`GAZA_PRINTED`, `GAZA_MEANT`). Detail: `sources/mz.md` §7.

## Shared code
Import these, do not copy them.
- `tools/oracle.py`: `fetch`, `oracle(name, year)`, `latest`, `table`, `partition`, `TOTAL`.
- `sources/micro.py`: `COLUMNS`; a national-only country is a row in `COUNTRIES`, not a module.
- `tools/check_mapping.py` (`DEFAULT_LEVELS`), `tools/gap_share.py` (`REPLACED`, `SKIP`), `taxonomy/registry.py`.
- `sources/fetch_checks.py`: `check_body` (size, magic, pins, trailer), `digest`, `check_pdf_doc`, `image_pages`,
  `cdx_url`, `parse_cdx`, `one_per_digest`, `wayback_raw`, `wp_json`, `pinned_differences`.
- Lints over every mapping or reader: `tools/check_na_readers.py`, `tools/check_no_religion.py`.
- Not shared yet, copied per country: the GET itself, `despace`, `say()` (`sources/gw.py`); the REDATAM POST
  (`sources/pe.py::_collect`). Name the source in the docstring.

## Rulings
- **Nothing is truly dead** (2026-09-07): stopping early is fine; record the search. Does not pick what to reopen.
- **§3.9b, §3.9c**: no minimum unit count; uniform is no reason to skip. Take the finest tier published.
- **Microstates** (2026-09-08): a national table is complete for a country of tens of dots; not for larger ones.
- **004-am, 005-rw**: record what the source says; ADEPR stays on `christianity.pentecostal`. Not new nodes generally.
- **§2.7a** (2026-09-14): a large unspecified share does not stop a named split being drawn.
- **No-religion boxes** (2026-09-14): Mozambique `unaffiliated` (drawn); Laos to traditional religion (queued;
  `la2015.py` still maps `unknown`); China stays `unknown`. A merged node is Anita's, undecided.
- **002-za, 008-ug**: accounts are Anita's to make; UBOS: extract what is needed, delete the rest.
- **003-cn**: mixed vintages in one country are fine when the method is sound; Tibet was declined on size only.
- **017-td, 018** (2026-09-14): draw Chad at 22 régions, Burkina Faso at 45 provinces, Mali at 20 régions, as
  published. Other §14 safety cases are still raised, not decided.
- **023-ir** (2026-09-14): draw Iran's 31 provinces; a small religion with no census answer sitting in other does
  not hold a build.
- **Pakistan** stays at district by Anita's choice (`sources/pk.md` §9); not a rule for other countries.
