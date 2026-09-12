# Philippines — PSA 2020 Census of Population and Housing, religious affiliation

`source_id: ph_cph_2020` · ingested 2026-08-30 · `sources/ph.py` → `data/normalized/ph.csv`

**What it is.** One wide matrix — 129 religious-affiliation columns × 135 geographic rows — from
the Philippine Statistics Authority's 2020 CPH. It is the *finest nationally complete* religion
tabulation the Philippines publishes, and it is very good: 129 named bodies is second only to the
US Religion Census among anything in `sources.md`, and unlike ASARB it is `self_id`.

---

## 1. Files, URLs, and how to re-fetch

Everything is in `data/raw/ph/` (gitignored).

| file | what | live URL |
|---|---|---|
| `3_Statistical_Table_for_Religious_Affiliation_RML_12082022_PMMJ_CRD_1.xlsx` | **the data.** Sheet `A` = Table A, household population by religious affiliation × region/province/HUC. Sheet `Country of Citizenship_w PH` = an unrelated Table D, ignored. | `https://psa.gov.ph/system/files/phcd/3_Statistical%20Table%20for%20Religious%20Affiliation%20(for%20Posting)_RML_12082022_PMMJ_CRD_1.xlsx` |
| `1_Press_Release_on_Religious_Affiliation_RML_01272023.pdf` | the 21 Feb 2023 press release; the national/regional/HUC/province headline figures used as the reconciliation target | `https://psa.gov.ph/system/files/phcd/1_Press%20Release%20on%20Religious%20Affiliation_RML_01272023_FJRA_PMMJ_CRD-signed_0.pdf` |
| `2_Technical_Notes_for_Religious_Affiliation_RML_12082022.pdf` | 3 pages; the question wording, the de jure basis, and **the PSGC edition** | `https://psa.gov.ph/system/files/phcd/2_Technical%20Notes%20for%20Religious%20Affiliation_RML_12082022_PMMJ_CRD_0.pdf` |
| `PSGC-1Q-2022-Publication-Datafile.xlsx` | Philippine Standard Geographic Code as of **31 March 2022** — the codes for `geo_id` | `https://psa.gov.ph/system/files/scd/PSGC%201Q-2022-Publication-Datafile.xlsx` |

All four landing pages are at `https://psa.gov.ph/content/religious-affiliation-philippines-2020-census-population-and-housing`
and `https://psa.gov.ph/classification/psgc/`.

### psa.gov.ph is behind Cloudflare for scripted clients

Every `psa.gov.ph` and `rsso*.psa.gov.ph` URL returns **HTTP 403 with a Cloudflare "Just a
moment…" interstitial** to curl and to WebFetch, with or without a browser user-agent. A real
browser gets the files fine. Rather than fight it, all four files were pulled from the **Wayback
Machine's `id_` (raw, unrewritten) endpoint**, which mirrors them byte-for-byte:

```
https://web.archive.org/web/20230812114505id_/https://psa.gov.ph/system/files/phcd/3_Statistical%20Table%20for%20Religious%20Affiliation%20(for%20Posting)_RML_12082022_PMMJ_CRD_1.xlsx
https://web.archive.org/web/20230812114943id_/https://psa.gov.ph/system/files/phcd/1_Press%20Release%20on%20Religious%20Affiliation_RML_01272023_FJRA_PMMJ_CRD-signed_0.pdf
https://web.archive.org/web/20230812114416id_/https://psa.gov.ph/system/files/phcd/2_Technical%20Notes%20for%20Religious%20Affiliation_RML_12082022_PMMJ_CRD_0.pdf
https://web.archive.org/web/20250521122525id_/https://psa.gov.ph/system/files/scd/PSGC%25201Q-2022-Publication-Datafile.xlsx
```

To find a PSA file's snapshots without loading the whole domain (which times out the CDX API):
`http://web.archive.org/cdx/search/cdx?url=psa.gov.ph/system/files/&matchType=prefix&collapse=urlkey&filter=urlkey:.*psgc.*`.
Note the PSGC URL is **double-escaped** in the CDX index (`%2520` for a literal `%20`); pass the
CDX `original` string through unchanged.

---

## 2. Geography and vintage

**Level.** The rows are a single tier of **province + highly urbanized city**, plus one
municipality and one interim province, with region and national aggregates above them:

| `geo_level` | rows | note |
|---|---|---|
| `country` | 1 | `geo_id` is `PH` — the PSGC has no code for the nation |
| `region` | 17 | aggregate |
| `province` | 82 | the 81 PSGC provinces, **each excluding any HUC inside it**, plus the BARMM interim province |
| `city` | 34 | the 33 HUCs, plus the City of Isabela |
| `municipality` | 1 | Pateros, the only municipality in NCR |

`province` + `city` + `municipality` = **117 units that partition the country exactly**
(108,667,043 people, checked). Region and country rows are aggregates and carry a `note` saying
so; do not add them to the fine tier.

**Vintage: PSGC as of 31 March 2022** — stated outright in the technical notes ("The Philippine
Standard Geographic Code (PSGC), as of March 2022 was used to disaggregate geographic levels of
the 2020 CPH"). That is the **1Q 2022 publication datafile**, and it is the 10-digit PSGC
(`RRPPPMMBBB`); the file also carries the old 9-digit code as `Correspondence Code`. Every one of
the 134 coded rows resolved against it, no fuzzy matching.

### spec.md §8.1 in a country that reorganises constantly

Taking a newer boundary set would silently delete or misplace, at minimum:

- **Maguindanao** was split into Maguindanao del Norte and Maguindanao del Sur in September 2022,
  six months after this vintage. Here it is one undivided province — **and the census row is
  labelled "Maguindanao (including the City of Cotabato)"**, so it also swallows an independent
  component city that most boundary files hold separately.
- **The BARMM Special Geographic Area** — 63 barangays detached from six Cotabato municipalities
  by the 2019 plebiscite. The census gives them one row, "Interim Province"; PSGC 1Q 2022 gives
  them the interim province code `1999900000` and then splits them into **eight "SGU" cluster
  codes** `1999901000`–`1999908000`. There is no single polygon anywhere for this unit, and its
  215,348 people are *not* in the Cotabato row.
- **Negros Occidental, Negros Oriental, Siquijor and Bacolod** were moved into the new Negros
  Island Region in 2024. Under the 2020 vintage they are in Regions VI and VII.
- **33 HUCs are carved out of their provinces.** Cebu here means Cebu minus Cebu City,
  Lapu-Lapu and Mandaue. geoBoundaries PHL ADM2 does not make that cut, so an ADM2 join will
  double count unless the HUCs are cut out first.
- **City of Isabela** is geographically inside Basilan but administratively Region IX; PSGC gives
  it a special province-slot code (`0990100000`, "City of Isabela (Not a Province)"), the city
  itself being `0990101000`, which is what `geo_id` uses. Basilan's row says "excluding the City
  of Isabela".
- **Davao de Oro** is labelled "(Compostela Valley)" — renamed 2019.
- NCR's four legislative districts have 9-digit PSGC codes and **are dropped in the 10-digit
  PSGC**, so anything keyed on the old codes has four orphan units.

All of these are carried in the `note` column of the affected rows.

### Two name collisions that a name join gets wrong

Isabela province contains a **municipality called Quirino**, and Laguna contains a **municipality
called Rizal** — and Quirino and Rizal are also province names in the same regions. The first
draft of `ph.py` matched both census province rows to the municipality and produced 80 provinces
instead of 82. `load_psgc()` now ranks candidates by geographic level so the province wins its own
name. Worth remembering for any other country matched by name.

---

## 3. Basis

**`self_id`, every row.** The 2020 CPH is a de jure census; the question was *"What is ____'s
religious affiliation?"*, asked of a household respondent about every member. So it is
self-identification in spec.md §3.1's sense — a population question, not an institutional roll —
but it is **proxy-reported**: one household member answered for the others. The technical notes
say so and add their own caveat that the statistics "should be used with caution".

There are no `roll` or `estimate` rows in this source. The one place a per-row basis distinction
would have been tempting — *Oblates of Mary Immaculate, Incorporated*, a religious **order**
appearing as a self-identified affiliation with 528 people — is still `self_id`, because 528
people said it about themselves. (It is also spec.md §3.2's "an order is a slice of its parent"
case arriving in a self-id census rather than a roll, which is unexpected and will need a
decision when the taxonomy mapping happens.)

---

## 4. The count is the HOUSEHOLD population, not the total population

| | |
|---|---|
| 2020 CPH **total** population | **109,035,343** |
| 2020 CPH **household** population, the base of every figure here | **108,667,043** |
| difference | **368,300 people = 0.338%** |

PSA defines a household as people who sleep in the same housing unit and share food preparation,
and the household population as the members of households. The 368,300 who are not in it are
therefore everyone living in an institutional or other non-household arrangement — prisons,
hospitals, military camps, orphanages, dormitories, seminaries, convents and monasteries — plus
the homeless, plus the 2,098 Filipinos in Philippine embassies, consulates and missions abroad
that the PSGC national summary counts separately. (The 2,098 is the only component PSA breaks
out; the rest is a deduction from the definitions, not a published split.)

For this project that is not a rounding error in the interesting direction: **the excluded
institutional population is exactly where monastic and seminary communities live.** A census
religion table structurally cannot see a charterhouse. spec.md §4.3's rings will have to come from
elsewhere for the Philippines, not from this file.

The total population figure above is from the PSGC workbook's own `Nat'l Sum` sheet, so the
comparison uses two PSA sources and no outside number.

---

## 5. Reconciliation against the published figures

`sources/ph.py` checks all of this on every run and exits non-zero if anything moves.

| | from sheet A | PSA press release | diff |
|---|---|---|---|
| household population | 108,667,043 | 108,667,043 | 0 |
| Roman Catholic, excluding Catholic Charismatics | 85,645,362 (78.814%) | 85,645,362 (78.8%) | **0** |
| Catholic Charismatic | 74,096 | 74,096 | 0 |
| Islam | 6,981,710 | 6,981,710 | 0 |
| Iglesia ni Cristo | 2,806,524 | 2,806,524 | 0 |
| Seventh Day Adventist | 862,725 | 862,725 | 0 |
| Aglipay | 818,916 | 818,916 | 0 |
| Iglesia Filipina Independiente | 640,076 | 640,076 | 0 |
| Bible Baptist Church | 540,364 | 540,364 | 0 |
| United Church of Christ in the Philippines | 470,792 | 470,792 | 0 |
| Jehovah's Witness | 457,245 | 457,245 | 0 |
| Church of Christ | 429,921 | 429,921 | 0 |
| None | 43,931 | 43,931 | 0 |
| Not reported | 15,186 | 15,186 | 0 |

Plus two structural checks, both exact:

- **the 129 categories sum to the household population in every one of the 135 rows**, national
  row included — 0 discrepancy anywhere. The categories are a true partition; there is no
  residual to compute (spec.md §3.2 has nothing to do here);
- **the 117 fine units sum to 108,667,043, and so do the 17 regions, and so does the national
  row.** Geographic nesting is exact too.

**The 78.8% headline excludes Catholic Charismatics.** The press release's asterisk says so.
Catholics *including* charismatics are **85,719,458 = 78.881%**. Both slices are in the CSV as
separate `source_category` values, because that is how the source has them.

**Trap: "Other religious affiliations" means two different things.** The press release's
`Other religious affiliations = 8,954,291 (8.2%)` is the residual of *its own top-ten table*
(108,667,043 − the ten named − None − Not reported = 8,954,291, verified). Sheet A has a column
**literally called "Other religious affiliations"** and it is **1,893,134 (1.74%)** — a real
catch-all category alongside the other 128, not the press release's residual. Do not cite one for
the other.

---

## 6. The 129 categories

Alphabetical, flat, mutually exclusive and exhaustive. **They do not nest** — there is no
"Protestant" or "Christian" parent anywhere in the file, so the whole taxonomy tree above the leaf
has to come from `taxonomy/`. Five of the 129 are explicit catch-alls, and they are the file's own
residuals: **Other Baptists** 361,332 · **Other Evangelical Churches** 254,489 · **Other
Methodists** 49,179 · **Other Protestants** 332,173 · **Other religious affiliations** 1,893,134.
Two more are not affiliations at all: **None** 43,931 and **Not reported** 15,186.

National counts, largest first (full list is the `country` rows of `ph.csv`):

| # | category | count | % |
|---|---|---|---|
| 1 | Roman Catholic, excluding Catholic Charismatics | 85,645,362 | 78.81 |
| 2 | Islam | 6,981,710 | 6.43 |
| 3 | Iglesia ni Cristo | 2,806,524 | 2.58 |
| 4 | *Other religious affiliations* | 1,893,134 | 1.74 |
| 5 | Seventh Day Adventist | 862,725 | 0.79 |
| 6 | Aglipay | 818,916 | 0.75 |
| 7 | Iglesia Filipina Independiente | 640,076 | 0.59 |
| 8 | Bible Baptist Church | 540,364 | 0.50 |
| 9 | United Church of Christ in the Philippines | 470,792 | 0.43 |
| 10 | Jehovah's Witness | 457,245 | 0.42 |
| 11 | Church of Christ | 429,921 | 0.40 |
| 12 | Assemblies of God | 413,703 | 0.38 |
| 13 | *Other Baptists* | 361,332 | 0.33 |
| 14 | Jesus is Lord Church | 333,506 | 0.31 |
| 15 | *Other Protestants* | 332,173 | 0.31 |
| 16 | Christian and Missionary Alliance Church of the Philippines | 327,537 | 0.30 |
| 17 | Pentecostal Church of God Asia Mission | 301,165 | 0.28 |
| 18 | United Methodists Church | 300,095 | 0.28 |
| 19 | Baptist Conference of the Philippines | 274,309 | 0.25 |
| 20 | *Other Evangelical Churches* | 254,489 | 0.23 |
| 21 | Tribal religion | 251,548 | 0.23 |
| 22 | Alliance of Bible Christian Communities of the Philippines | 236,408 | 0.22 |
| 23 | United Pentecostal Church (Philippines), Inc. | 212,425 | 0.20 |
| 24 | Southern Baptist Church | 190,336 | 0.18 |
| 25 | Christian Missions in the Philippines | 183,584 | 0.17 |

and a long tail: **98 of the 129 are under 100,000 people, 39 are under 10,000**, and four are
under 1,000 — *Jireh-Evangel Church Planting Philippines, Inc.* 973, *Don Stewart Ministries
Miracle Revivals, Inc.* 812, *Oblates of Mary Immaculate, Incorporated* 528, and the smallest,
**Faith Tabernacle Church (Living Rock Ministries), 358 people spread over 64 of the 117 units**.

**Per-unit richness** (117 fine units): distinct categories present — min **43** (Tawi-Tawi),
median **122**, max **129** (City of Manila, which reports every single body). 14 categories are
present in all 117 units. Compare the US Religion Census's median of 20 bodies per county: the
Philippine file is far denser per unit, because these are 129 *national* response options rather
than an enumeration of local congregations.

---

## 7. Things that surprised me

1. **The entire non-Abrahamic world gets three rows.** Islam, Buddhist (39,158) and Tribal
   religion (251,548) are the only non-Christian named categories in 129. There is **no Hindu, no
   Jewish, no Sikh, no Chinese folk religion category at all** — those people are inside *Other
   religious affiliations*. So the source is four or five levels deep on Philippine
   evangelicalism and zero levels deep on everything else. This is spec.md §2.2's "the tree
   records what is countable" seen from a new angle: the depth follows the *country's own*
   religious politics, not world religion.

2. **Aglipay and Iglesia Filipina Independiente are the same church, listed twice.** The IFI *is*
   the Aglipayan Church — one body, founded by Gregorio Aglipay in 1902. The census offered both
   as response options and respondents split 818,916 / 640,076 between them, **1,458,992 = 1.34%
   of the country**, which would make it the fourth-largest religious body in the Philippines if
   merged. This is a decision for the taxonomy mapping, not for ingest, and it is left raw per
   spec.md §2.4. There is a comparable cluster on the Catholic side: *Philippine Independent
   Catholic Church* (52,637) and *Apostolic Catholic Church, Inc.* (54,543) are separate bodies
   again, and neither is the IFI.

3. **43,931 people, 0.040%, reported "None".** Four hundredths of one percent. Whatever else this
   census measures, it is not measuring irreligion the way a European census does, and any
   unaffiliated residual drawn from it will be a claim about the question rather than about the
   country.

4. **A religious order is a response option.** *Oblates of Mary Immaculate, Incorporated*, 528
   people. spec.md §3.2 argues at length that an order is a slice of its parent and not an extra
   layer — that argument was written about `roll` sources, and here it arrives inside a `self_id`
   census, where the 528 are genuinely *not* also counted as Roman Catholic. The partition holds;
   the tree placement is the open question.

5. **The columns are alphabetical, except Islam.** Islam sits at column 47, between *Faith
   Tabernacle Church (Living Rock Ministries)* and *FIFCOP Mission, Inc.* — where a label
   beginning "Fi…" would have sorted. Some earlier working label ("Filipino Muslim"?) was
   evidently renamed without re-sorting. Harmless, but it means column position carries no
   meaning and the header text is the only key.

6. **The finest published level is province, but not for want of finer data.** PSA regional
   offices publish per-*municipality* religion special releases — e.g.
   `rssomimaropa.psa.gov.ph/.../religious-affiliation-baco-oriental-mindoro-2020-census-population-and-housing`,
   with barangay-level tables inside. A Wayback index scan found roughly a dozen of them, **all in
   Oriental Mindoro**, as infographic PDFs. There is no national municipality-level release and no
   sign of one; assembling one would mean scraping 1,600+ Cloudflare-protected pages that mostly
   do not exist. Province + HUC is the ceiling for this country.

---

## 8. Licence

- **PSGC workbook, `Metadata` sheet, verbatim** — *Access constraints: None.* *Use constraints:
  Acknowledgement of the Philippine Statistics Authority (PSA) as the source.* *Disclaimer: The
  PSGC is being distributed without warranty of any kind…*
- **The census tables and press release** carry `Source: Philippine Statistics Authority, 2020
  Census of Population and Housing` on every table and a citation section in the technical notes,
  which is the same requirement stated less formally. The files are free, unauthenticated
  downloads with no click-through terms.
- Background: works of the Government of the Philippines are not subject to copyright under
  §176 of the Intellectual Property Code (RA 8293), though that section reserves the right to
  require prior approval for exploitation for profit. **Attribution to the PSA is the operative
  obligation and is cheap**; anything commercial should read §176 properly first.

Suggested citation: *Philippine Statistics Authority, 2020 Census of Population and Housing,
Religious Affiliation (released 21 February 2023).*

---

## 9. What `sources/ph.py` does

1. reads sheet `A`, taking the category names from row index 3 and the units from row 5 down,
   skipping the blank spacer rows, the footnote and the source line;
2. strips the trailing footnote marker from `Interim Province 1`;
3. loads every non-barangay PSGC entry, keyed by `(2-digit region, folded name)` with a
   level ranking so a province beats a same-named municipality;
4. walks the census rows in order — a region header sets the region context, every row after it
   resolves inside that region — with a 9-entry `LABEL_OVERRIDES` table for the labels that
   carry parentheticals the PSGC does not (`Samar (Western Samar)`, `Cotabato (North Cotabato)`,
   `Davao de Oro (Compostela Valley)`, `City of General Santos (Dadiangas)`, the two BARMM
   specials, Pateros, and NCR). An unresolved label is a hard error, never a skip;
5. runs the §5 checks and exits non-zero if any fails;
6. writes 17,550 rows: 135 units × (129 categories + 1 `Household Population` row).

`Household Population` is emitted as a `source_category` because spec.md §8.2 wants the census's
own denominator alongside the counts; its `note` says it is the denominator and not a religion, so
it can be filtered in one predicate.
