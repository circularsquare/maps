# The Bahamas (bs): record

Drawn 2026-10-05 (session edd42a8c-mono3). No census language question. Built under the
2026-10-05 ruling for countries with no language question (AGENT_BRIEF §2): national language for
citizens, immigrant languages proxied by citizenship. 2010 census shares per island (citizenship
and race), applied to the 2022 census island populations: 398,165 people, 18 islands, 38 nodes,
every row `derived`. Placed on religiondots' Kontur hexes for the 18 census islands (read-only).
392 dots at 1:1,000, 38 rings.

Files: `sources/bs_census.py`, `taxonomy/bs2010.py`, `taxonomy/tree.d/bs.txt`, `countries/bs.py`,
`data/normalized/bs.csv`, `data/raw/bs/` (eighteen 2010 island reports, `bs_2010_<island>.pdf`,
and the 2010 International Migration report).

```
python sources/bs_census.py --fetch
python taxonomy/build.py
python tools/check_country.py bs
python scatter.py --country bs
```

## 1. Sources

- 2022 census: no language question (questionnaire in BNSI's first release, appended); it asks
  citizenship and birthplace but neither the first release nor the All-Island Report tabulates
  them by island (religiondots' copies searched). Island totals from religiondots' parse of the
  All-Island Report (`../religiondots/data/normalized/bs.csv`, TOTAL rows), read only.
- 2010 census island reports (stats.gov.bs/wp-content/uploads/2020/08/<ISLAND>-2010-CENSUS-REPORT.pdf,
  listed on stats.gov.bs/subjects/population-and-demography/), one per census island, the same 18
  islands as 2022. **Table 8.0** total population by racial group; **Table 9.0** total population
  by country of citizenship. Only these carry citizenship per island.
- 2010 International Migration report (`bs_2010_international_migration.pdf`): foreign-born by
  island and country of birth; read for orientation, not used (birthplace misses Haitian
  children born in the Bahamas, citizenship catches them).

## 2. The model

Per island, from the 2010 tables:

- **Foreign citizenship → that country's language** (`taxonomy/bs2010.py`), the
  `fr_build.COUNTRY_LANG` convention, with Canada → English (COUNTRY_LANG's French is a
  France-specific choice) and Belgium → Dutch. Haiti → Haitian Creole, Jamaica → Jamaican Creole,
  Dominica and St Lucia → Antillean Creole. The English-speaking Caribbean's nationals go on
  their own creole, as Bahamians do on theirs (Barbados → Bajan, Guyana → Guyanese, Trinidad,
  Turks and Caicos, Antigua and St Kitts → Antiguan, St Vincent, Grenada; nodes from
  `tree.d/bb.txt`, repeated here, plus Turks and Caicos Creole, Glottolog turk1310, new here).
  Cayman, BVI and Belize nationals (a few dozen) stay on English.
  "Other Commonwealth" and "Non-Commonwealth countries" are pooled unnamed rows → `other`.
- **Bahamian citizens → Bahamian Creole** (new node `creole.english_based.bahamian`, Glottolog
  baha1260 "Bahamas Creole English"), **except white Bahamians → English.** White Bahamian
  English (Spanish Wells, Abaco, Long Island) is described in the dialect literature as an English
  variety, not a creole. White citizens are estimated per island as WHITE (Table 8.0) minus the
  island's nationals of North America, Europe and Oceania, floored at 0: 8,411 nationally, 2.9%
  of citizens (Spanish Wells 1,223 of 1,271 white; Abaco 1,920; New Providence 4,464).
- **Not stated citizenship** is left out of the shares (`gap`).
- The 2010 shares are applied to each island's 2022 population, largest-remainder rounded so each
  island sums exactly.

## 3. Checks (all asserted on every run)

- Each island's citizenship rows sum to its Table 9.0 TOTAL, race rows to its Table 8.0 TOTAL,
  and the two TOTALs agree (18/18).
- The 18 islands sum to the 2010 census total, 351,461; the 2022 islands to 398,165.
- **Haitian nationals sum to 39,144**, the figure the 2010 census is widely quoted for ("39,000
  Haitians or persons of Haitian descent"). Bahamian citizens over the 18 island tables sum to
  290,725, exactly the migration report's Table (v) national count (printed, not asserted).
- Island names joined to religiondots' ids by an explicit table, asserted against its geo_name.

Parse traps: a page number heads every page and the last header line of a page is a label, so
the parser drops line one of each page; a few rows come out of the PDF as `label, MALE, <total
row>, FEMALE, ...` (Inagua's Canada and U.S.A), handled; spelling variants (`JAMACIA`,
`TRINADAD`, `SW ITZERLAND`, `OTHER COMMOMWEALTH`) merged.

## 4. Calls someone might reverse

- **Bahamian Creole for all non-white Bahamian citizens** (80.8% of the drawn country). The
  Bahamas is a creole continuum; an English-dominant Afro-Bahamian middle class exists, and
  unlike Jamaica (`sources/jm.md`) no survey measures it, so none is drawn as English.
- **White Bahamians as English**, estimated by subtraction; the race x citizenship cross is not
  published by island.
- **2010 shares on 2022 populations.** Hurricane Dorian (2019) destroyed the Mudd and Pigeon Peas,
  Abaco's large Haitian settlements, so Abaco's 26% Haitian Creole is likely high for 2022; said
  in `note_public`.
- **Bahamian citizens of Haitian descent** (naturalised, or registered at 18) are drawn as
  Bahamian Creole; Haitian Creole is an undercount of home use to that extent.

## 5. Room for improvement

A 2022 citizenship-by-island table (the question was asked; BNSI has not published it) would
replace the 2010 shares. A language-use survey like Jamaica's would settle the creole/English
split among Afro-Bahamians.
