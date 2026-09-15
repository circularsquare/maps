# Census table playbook

A statistics office's own religion table (census or register count), read from xlsx, PDF, a tabulation
server or UNSD's copy into `data/normalized/<cc>.csv` and a mapping. It is the route wherever a census
asked religion; boundaries, name joins, population bases and placement are in `playbooks/geography.md`.

## Used by
- `gw` Guinea-Bissau: PDF annex in counts, equal to UNSD to the person; misprints, a prose swap, a pinned digest (sources.md §9dx).
- `mz` Mozambique: per-province xlsx on INE's retired Plone site, found by a Wayback CDX prefix query (§9df, §9dy).
- `cg` Republic of the Congo: Wayback `id_` copy of a squatted domain's PDF; the form's codes fix the column order (§9dv).
- `gn` Guinea: one-decimal shares by région times printed populations, rescaled to UNSD (§9dh).
- `zm` Zambia: PDF tables with two wrong column headers, settled by the office's analytical report (§9db).
- `pk` Pakistan: PBS Table 9 PDFs under `wp-content/uploads/`, right-aligned columns read by drawn rules (§9du).
- `td` Chad: shares by région from a Wayback copy; animist offered beside no religion (built, held on ask 017).
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
  the national-language tree; record what was asked, of what, when. Caught by: Not checked yet
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
- **017-td, 018** (open): Chad, Burkina Faso, Mali held on §14 safety. Raise such cases; do not decide them.
- **Pakistan** stays at district by Anita's choice (`sources/pk.md` §9); not a rule for other countries.
