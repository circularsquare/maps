# Census table playbook

A statistics office's own religion table (census or register count), read from xlsx, PDF, a tabulation
server or UNSD's copy into `data/normalized/<cc>.csv` and a mapping. It is the route wherever a census
asked religion; boundaries, name joins, population bases and placement are in `playbooks/geography.md`.

## Used by
- `gw` Guinea-Bissau: PDF annex in counts, equal to UNSD to the person; misprints, a prose swap, a pinned digest (sources.md §9dx).
- `mz` Mozambique: per-province xlsx on INE's retired Plone site, found by a Wayback CDX prefix query (§9df, §9dy).
- `cg` Republic of the Congo: Wayback `id_` copy of a squatted domain's PDF; the form's codes fix the column order (§9dv).
- `gn` Guinea: one-decimal shares by région times printed populations, rescaled to UNSD (§9dh).
- `ir` Iran: SCI's 1395 yearbook table in counts, Persian digits in the text layer, the bold national row a picture (§ir-2026-09-14).
- `zm` Zambia: PDF tables with two wrong column headers, settled by the office's analytical report (§9db).
- `pk` Pakistan: PBS Table 9 PDFs under `wp-content/uploads/`, right-aligned columns read by drawn rules (§9du).
- `td` Chad: shares by région from a Wayback copy, raked to two printed margins; animist offered beside no religion; COD's provinces rebuilt to 2009 on the office's areas (sources.md §td-2026-09-14).
- `bf` Burkina Faso: annex counts by province on a retired tree, equal to UNSD, summed by région to a second annex table; geoBoundaries because COD-AB moved to a 2025 reform (sources.md §bf-2026-09-14).
- `sl` Sierra Leone: one-decimal shares by district on printed district totals, scaled to the household population; no national count in persons exists, so no rescale; a misprinted national row pinned (sources.md §sl-2026-09-15).
- `sn` Senegal: one-decimal shares by région with the Sufi brotherhoods as codes, on printed populations; a cell printed in the wrong column, settled by a regional report in counts, which also draws Diourbel at département (sources.md §sn-2026-09-15).
- `bn` Brunei: counts by district in a workbook and an annex PDF, both only on Wayback (one capture a 1 MiB fragment), equal to UNSD; the form's Hindu folded into Others (sources.md §bn-2026-09-15).
- `gi` Gibraltar: report PDF, counts by residential area, drawn as one unit on 20 Kontur hexes; the 78 enumeration areas rebuild every area and place one institutional EA the appendix does not (sources.md §gi-2026-09-15).
- `ps` Palestine: counts by governorate in a bilingual PDF read by English row label; three nested universes (Palestinians counted, everyone counted, plus the estimate) settle East Jerusalem (sources.md §ps-2026-09-15).
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
  Guinea-Bissau were closed on one release or a volume title, then drawn from the same office. Open each volume's
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
  Caught by: Not checked yet (belongs in `sources/fo.py::check`). Detail: sources.md §scout-2026-09-15-europe.
- **An appendix's list of what makes up each unit can be wrong for one piece, while the table closes.**
  Gibraltar's Appendix 9 builds residential areas from enumeration areas and leaves institutional EAs
  80-87 to the Institutions row; Table 42 counts EA 85 (70 people) in South District. Only rebuilding
  every area from Table 43's EA rows shows it: Institutions short and South District over by EA 85's
  row in all 19 cells. Rebuild each unit from the finer table, fix shared pieces from the units that
  pin them, and let the shortfall name the piece. Caught by: `sources/gi.py::solve_shared`, `::check`.
  Detail: `sources/gi.md` §3.
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
- **An unmapped category vanishes.** `EXCLUDED` holds universe rows and non-answers, each with a sentence and
  which way it leans (§3.5); `REVIEW` every arguable call, naming the node wanted. A parent beside children that
  sum to it on every row is a duplicate; else emit the remainder. Never map on the string alone. Caught by:
  `tools/check_mapping.py::main`; the lean Not checked yet (`tools/gap_share.py`). Detail: spec §12 "Taxonomy".
- **`gap` and `gap_share`.** A printed non-response column is computed (`tools/gap_share.py <cc> -v`, `--write`).
  People in no table (collective households, restricted areas, unasked ages) are hand-written and join `gap`
  (Guinea's precedent); never add the two silently. Caught by: `tools/gap_share.py --check`; the foot of
  `countries.py` asserts `gap` states `gap_share`. Detail: spec §10.4a; `countries.py` docstring.
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
