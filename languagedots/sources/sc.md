# Seychelles (`sc`)

Drawn 2026-10-05 by `edd42a8c-sc`. **Census 2022, first main language spoken at home, aged 3+,
25 districts and 18 islands, 87,407 people drawn on 7 nodes; 11,545 (11.7%) with no language
not drawn.** Files: `sources/sc_census.py`, `sources/sc_geo.py`, `taxonomy/sc2022.py`,
`taxonomy/tree.d/sc.txt`, `countries/sc.py`; data in `data/raw/sc/`, `data/normalized/sc.csv`,
`data/geo/sc/`.

## 1. Source

NBS, *Seychelles Population and Housing Census 2022* (full report, 301 pages),
`nbs.gov.sc/downloads/1555-seychelles-population-and-housing-census-2022/download`, fetched
2026-10-05, 10,525,457 bytes (byte-identical in size to religiondots' copy of 2026-09-14).

- **Table B3.1a** (PDF page 116, print 92): population aged 3+ by main first language spoken at
  home, by region, district and island. Columns Creole, English, French, Gujarati, Hindi, Tamil,
  Other (Specify), Do Not Know, Refusal, Missing. This is drawn.
- Question Q.01.13 (report page 265), asked of each person aged 3+: "What main language does
  [Name] speak most often at home?" A home language, one answer. Q.01.14 asks a second one
  (Table B3.1b); not drawn.
- 2010 and 2002 asked the language at household level and published it nationally only (2010
  report Table 8.1; the 2010 district supplement religiondots used has no language table). 2022
  is the only choice for a district map.
- `queue.csv` pointed at the provisional results report (1308); the full report (1555) is the
  one with district tables.

## 2. The parse and its checks (`sources/sc_census.py`)

PDF text layer, a label then its cells, one per line. Asserted:

1. every one of 51 rows: ten cells sum to the printed total;
2. districts and islands sum to their region rows in every column, regions to the national row
   (98,952);
3. Table B3.1b (second language) prints the same total in all 51 rows;
4. Table B4.1 (religion, all ages, all households) is at least B3.1a's 3+ total in every row
   (102,612 against 98,952 nationally);
5. the Outer Islands witness (§3).

National, of 98,952: Creole 74,542, English 6,986, French 401, Gujarati 1,635, Hindi 1,618,
Tamil 771, Other 1,454, Do Not Know 115, Refusal 13, Missing 11,417. Of the 87,407 who named a
language: Creole 85.3%, English 8.0%, Gujarati 1.9%, Hindi 1.9%, Other 1.7%, Tamil 0.9%,
French 0.5%. The report's own 85.1% / 8.0% / 0.5% (its Table 3.3) are on a slightly different
denominator; not chased.

## 3. Calls

- **The Outer Islands are drawn as printed.** B3.1a records no Creole speaker on any outer
  island; 707 of their 985 people are Gujarati speakers (Platte 353, Desroches 162, D'Arros 110,
  and a few on eight more), 53 English (Desroches), 225 missing. Table B4.1 counts exactly 707
  Hindus on the Outer Islands, island by island, and B3.1b has the whole region as missing. That
  looks like a block of contract workers recorded together, possibly with defaults, and the
  outer islands' Seychellois staff (who must exist on Desroches, Alphonse, Coetivy, Farquhar)
  are not visible in it. Two tables agree, there is no published figure to correct it with, and
  it is 0.8% of the drawn total, so it is drawn and the public note says so. At 1:1,000 it draws
  almost nothing, and the islands are outside the entry's view.
- **Missing is not drawn** and is in `gap`. The report (page 56) says 11,225 of those with no
  first language were non-Seychellois, so foreign workers' languages are undercounted, most in
  Cascade (1,543 of 6,481 missing, 24%), Roche Caiman 20%, Mont Fleuri 17%, Anse Boileau 16%,
  Ile Perseverance 14%; and Silhouette, 260 of 298 (a resort island).
- `Other (Specify)` 1,454 -> `other`: no breakdown anywhere, no indigenous languages in
  Seychelles (no people before 1770), so nothing for an indigenous remainder.
- Creole -> `creole.french_based.seselwa` (au.txt's node); the rest to the leaves au2021 and
  mu2022 use. Gujarati on the leaf `gujarati.gujarati`, as au2021 and ca2021.
- Colour: Seychellois Creole hand-picked `0.78 0.12 218` in `tree.d/sc.txt`, a clear cyan in the
  creoles' part of the wheel, bluer than Morisyen (205), apart from English (near white),
  French (violet), Hindi (orange), Tamil (teal 185). au.txt added the node bare, so this is
  the only colour it has.

## 4. Geography (`sources/sc_geo.py`)

43 units, ids as religiondots' `sc_lookup.csv` (ISO 3166-2:SC numbers) for the 2010 districts,
`SC-PI` for Ile Perseverance (own id; ISO's code for it not checked), `SC-I-<NAME>` for islands.

- COD-AB `syc_admbnda_adm3_nbs2010` (religiondots' copy, read-only). 24 districts as COD draws
  them, name and pcode ISO number asserted to agree; **English River without Perseverance Island**
  and COD's `Perseverance Island` feature as the new district (religiondots merged them for 2010).
- **La Digue** = COD's La Digue + 11 Other Islands parts in a box round Félicité, Marianne, the
  Soeurs and Cocos (4.71 km2), as religiondots showed the 2010 census counts them; the 2022 table
  names no row for them.
- **17 islands cut out of COD's Other Islands** by position: every part whose centroid is within
  a radius of the island; each must catch at least one part, inside an area band, none twice
  (Aldabra 944 parts 152 km2, Silhouette 19.9, Assumption 11.0, Coetivy 8.9, Farquhar 7.1, ...).
- 74 leftover parts (15.7 km2: Astove, Cosmoledo, St Pierre, Desnoeufs, the Mahé islets,
  Cousin and Cousine, D'Arros atoll's islets) are in no 2022 row and are dropped.
- Join both ways: sc.csv's 43 unit ids = the 43 polygons.

**Placement**: Kontur 2023 (religiondots' copy) cut by overlap as religiondots' `sc_grid.py` does,
but **Perseverance's hexes kept** (5,143 people aged 3+ there in 2022). 14 hexes touch no unit:
4 snapped within 700 m (81 people), 10 dropped (56). Every unit has Kontur population, so the
pop-1 polygon fallback did not fire. Kontur / census 1.074; every district and La Digue inside a
factor of 2; log r = 0.564 over 26 districts, 4 of 2,000 shuffles reach it (lower than
religiondots' 0.89 because that figure includes the Other Islands outlier; 2022's districts are
all 2,100-6,500 people). Islands range widely (Platte 0.09, Bird 37); that only moves dots
within an island.

## 5. Build

`check_country.py sc` ok: 87,407 people, 39 units with language rows (Bird, Denis, Frégate and
Aldabra have only missing answers), 7 languages. `scatter.py --country sc`: 83 dots, 2 rings;
water.py left 2 units unclipped (over 95% sea, atolls). Kontur cap not checked (overlap slivers;
no district near the cap).

## 6. Open

- The Outer Islands block (§3): NBS microdata or a query to NBS would say whether those 707 are
  Gujarati speakers or a coding default.
- Second-language table B3.1b (45,742 English as second language) is not used.

## Moved from countries/sc.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- Gujarati, Hindi and Tamil, 4.6% together, have grown with workers coming from India, the report says; in Cascade 23% of those who answered named Gujarati or Hindi.
