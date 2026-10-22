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

## Coverage gap: the 1,641,590 non-residents (2026-10-19, session eca0185b-gap)

Added as `modelled` rows (AGENT_BRIEF §2, "no measured source at all"), so Singapore now draws
5,237,872 people: the 3,596,282 residents the census asked plus 1,641,590 it did not. The census
rows are unchanged. Files: `sources/sg_nonres.py` (stock, mixes, placement; `--build` writes the
layer), `data/geo/sg/sg_places.gpkg` (religiondots' 332 subzones + 29 dormitory sites + 66
industrial units), `data/normalized/sg_nonres_sites.csv`, `data/raw/sg/osm_worker_dorms.json` and
`mp2019_business2.geojson` (Master Plan 2019 Business 2 land, cut from data.gov.sg
`d_90d86daa5bfaa371668b84fa5f01424f`; the 174 MB source was deleted).

**Stock, June 2020** (the date of the 1,641,590). MOM Foreign workforce numbers, Jun 2020
column, read from a Wayback snapshot of mom.gov.sg/documents-and-publications/foreign-workforce-numbers
(2021-03-01; the live page starts at Dec 2021): Employment Pass 189,700; S Pass 188,800; Work Permit
940,200 = foreign domestic workers 252,600 + construction/marine/process 351,800 + other 335,800;
other work passes 33,100; total 1,351,800. The 289,790 left of 1,641,590 are dependants,
long-term visit pass holders and students (MHA's 2018-23 averages are ~150,000 Dependant's Pass
and ~100,000 non-family LTVP, so the residue is plausible; no student pass count was found).
Malaysian work permit holders who commute daily are probably in MOM's table and not in the
non-resident population; the residue absorbs that, so the 289,790 is a floor on dependants.

**Dormitories** (SUTD "Rethinking dorms", 26 May 2020, quoting MOM): about 323,000 workers in
dormitories: 43 purpose-built (198,320 residents, Tan et al., JAMA 2021), about 1,200
factory-converted (95,000), construction temporary quarters (20,000). TODAY, May 2020: 85,000
construction work permit and S Pass holders lived outside dormitories. So: 266,800 construction/
marine/process workers in dormitories (351,800 - 85,000) and 56,200 other work permit holders.

**Nationality.** MOM does not publish nationality for any pass type (it says so in reply to
questions; searched parliamentary replies, MOM, Population in Brief 2020 and DESA without finding one).
Cited fragments: dormitory workers 44% Bangladeshi, 40% Indian, 16% other (a 1,089-worker survey at
five dormitories, Aug-Nov 2022, PMC11471231; its sectors were 60% construction, 25% marine); Myanmar
domestic workers ~30,000 of ~250,000 (Inquirer 2015). Everything else is **uncited judgement**:
the 16% split (Chinese 7, Burmese 4, Thai 2, Malaysian 2, Filipino 1); Indonesian 48 / Filipino 40 /
Burmese 12 for domestic workers; other work permits Malaysian 25, Chinese 20, Indian 15,
Bangladeshi 10...; S Pass Malaysian 28, Indian 22, Chinese 20...; Employment Pass Indian 27, Chinese
15, Malaysian 10, Japanese/American/British 5 each...; dependants and students Indian 25, Chinese
22, Malaysian 14... (all in `sg_nonres.py`). Nationality -> language through
`origin_mix.mix(iso, "sg")` (home mix of the origin's drawn map), with two local overrides: Indian
dormitory workers 60% Tamil (Tamil Nadu is the main source of Indian construction labour; uncited),
and Malaysians 45% Malay / 38% Chinese languages / 8% Tamil / 6% English (home mix would draw them
60% Malay; uncited).

**Placement.** Dormitory residents (323,000) on dormitory sites: 29 OSM-named worker dormitories
(Sungei Tengah Lodge, S11, Tuas View, Westlite sites, Penjuru...) with MOM/press bed counts where
found (Papan 7,900, Jalan Tukang 3,420, Kranji Way 1,300) and otherwise the mean of the 43
(198,320/43 = 4,612); the purpose-built residents those 29 do not hold, and the factory-converted
95,000, go on Master Plan 2019 Business 2 land by subzone (66 subzones with 5+ ha). OSM footprints
do not track beds, so area was not used. The remaining 29,680 dormitory residents (construction
quarters and the rounding in "about") and every other group go to planning areas by resident
population (the same subzone weight as residents). No open source places domestic workers, Employment
Pass holders or students more finely; HDB/private dwelling type or campus locations would improve it.

**Result.** Non-residents add Bengali 208,000, Mandarin 205,000, Tamil 156,000, Malay 96,000, Hindi
82,000, Tagalog 63,000, Javanese 60,000, English 59,000, Burmese 57,000, Cebuano 49,000, Indonesian,
Sundanese, Vietnamese... 102 languages. Singapore's Bengali goes from 0 to 208,000. English (residents'
48%) falls to 34% of the drawn people.

**Calls someone might reverse:** modelled nationality mixes (above); the dormitory/home split; Malaysian
commuters treated as absorbed in the residue; using residents' population as the home weight.
**Room for improvement:** MOM's dormitory list with addresses and capacities (MOM's site did not give one
openly); MOM nationality by pass type if ever published; HDB sublet and dwelling type by planning area.
