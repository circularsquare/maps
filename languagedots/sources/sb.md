# Solomon Islands (`sb`): 2019 census first language, national counts placed on wards by a model

Drawn 2026-10-05 by session `d9e44929-sb`. Files: `sources/sb_census.py` (normaliser),
`sources/sb_model.py` (placement model, drawn), `taxonomy/sb2019.py`, `taxonomy/tree.d/sb.txt`,
`countries/sb.py` (`MODEL = True`). Outputs `data/normalized/sb.csv` (national, 60 rows),
`data/normalized/sb_inputs.csv` (the province tables), `data/normalized/sb_model.csv` (9,869 ward
rows, all `modelled`), `data/processed/dots_sb.geojson` (607 dots), `rings_sb.geojson` (20).

**631,061 people aged 5 and over, 60 nodes, 183 wards (3,400 aged 5+ on average).** Pidgin 16.1%,
the remainder the report does not name 14.4%, Kwara'ae 8.0%, Are'are 4.1%, Lau 3.6%, Kwaio 3.6%,
Lengo 3.4%, To'abaita 3.1%, Tolo/Talise 3.1%, Gela 2.8%.

## 1. The table

*2019 National Population and Housing Census, National Report Volume 1* (SINSO, 2023), section
9.6, PDF pp. 149-152 (printed 112-115). Read in place from religiondots' cached copy
(`../religiondots/data/raw/sb/sb_2019_national_report_vol1.pdf`, read-only; `--fetch` downloads
into `data/raw/sb/` only if that copy is gone). URL:
`https://solomons.gov.sb/wp-content/uploads/2023/09/Solomon-Islands-2019-Population-and-Housing-Census_National-Report-Vol-1.pdf`.
The "Vol-1-2" file on the same site and the statistics.gov.sb download are byte-identical
(8,046,759 bytes).

Question (questionnaire P18a, Volume 2 appendix): "What is the first language this person (name)
learnt as a child?", aged 5+, answers 1 Pidjin, 2 English, 3 Local language (specify), 4 Other
(specify). One answer.

Three tables, all national counts:
- **9.6.1** "Larger local languages by province": Pidgin and 24 languages under the province each
  belongs to (the heading is the language's home; every figure is national).
- **9.6.2**: 21 rows, three repeating 9.6.1 (Lau, Marovo, RenBell = 9.6.1's "Rennell-Bellona").
- **9.6.3** "Endangered and new listed languages": 17 rows by province, Anuta repeating 9.6.2.

59 distinct answers, 540,202 people. English and Other are never printed; nor are the local
languages too small or unremarkable for these tables. **Remainder 90,859 (14.4%) = 631,061 less
everything named.** Volume 2 (Basic Tables) has no first-language table (its P9.7-P9.9 are
literacy by language). The coverage sweep's lead (national only, Pidgin by province as a figure)
was right.

**Grain: national.** Searched for anything finer: Volume 2's table list (no language table but
literacy); statistics.gov.sb's census page (only the two volumes and the projections); the
SINSO/SITAG news item (Nov 2022, no tables); pacificdata.org and sdd.spc.int (both 403 to a
browser User-Agent: the SPC portals, where the 1999 census report with its first-language tables
would be). The 2019 microdata is on the SPC microdata library by application only (gated, not
tried). The 2009 census did not ask the question.

## 2. Checks (`python sources/sb_census.py`, all asserted)

- The three tables give the same 2019 figure wherever they repeat a language (Lau 22,806, Marovo
  11,266, RenBell 4,438, Anuta 272). **Their 1999 columns do not agree** (Lau 15,747 vs 17,079,
  Marovo 7,566 vs 8,094, RenBell 2,998 vs 4,394, Anuta 267 vs 249), and 9.6.1's Aiwoo row repeats
  Babatana's 1976 and 1999 figures (2,355 and 5,255) exactly, a copying slip. Only 2019 is used.
- Pidgin 101,588 is 16.1% of 631,061, as the text says.
- Table 9.2.2 (5+ by province) sums to 631,061.
- P7.2 (vol 2, birthplace x province of enumeration): every row's birth columns sum to its Total,
  the ten province blocks sum to the national row in every column, and each province's Total
  equals Table 8.4.3's (ethnicity by province).
- Table 8.4.3's Micronesian row sums to its printed 8,647.
- Named answers sum to less than 631,061.

**Figure 9.6.1 (Pidgin by province) is an image**, so its ten labels are transcribed in
`sb_census.py` from the rendered page. They sum to 100.1, so they are each province's share of
the 101,588 Pidgin speakers. The other reading (Pidgin's share of each province) would give
106,276, 4.6% over the table; and the text's "Honiara, where the majority (47.2%) of people spoke
Pidgin" is the report misreading its own chart. Honiara: 47,902 Pidgin speakers, 41.1% of its 5+.

## 3. The placement model (`sources/sb_model.py`), drawn

Without it, Kwara'ae would be spread over all 900 islands by population. Every national count is
kept exactly; the model only decides where inside the country each language's speakers go. My
reasons for drawing it without an ask: AGENT_BRIEF 4.4 (a placement within the census's own unit,
which here is the nation, counts unchanged), and Anita allowed the same kind of model for
Azerbaijan (ask 006) and Indonesia (ask 009). `MODEL = False` in `countries/sb.py` draws the
national grain instead.

**Step 1, provinces**: IPF on languages x provinces. Rows = the census's national counts.
Columns = Table 9.2.2's 5+ population less Figure 9.6.1's Pidgin (Pidgin is fixed, not fitted).
Seeds: a local language starts out like the people born in its home province (P7.2: of the
Malaita-born, 79.9% were counted in Malaita, 12.0% in Honiara, 5.9% on Guadalcanal); Kiribati
like the Micronesian population (8.4.3); the remainder like the 5+ population. The fit then
raises every language's Honiara share alike, because Honiara-born children of migrants are
counted as born in Honiara. Result: Malaita's languages 68% Malaita, 18% Honiara, 10%
Guadalcanal; Lengo 96% Guadalcanal; Gela 85% Central; RenBell 62% Rennell-Bellona, 27% Honiara.
Province mixes: Honiara Pidgin 41%, unnamed 15%, Kwara'ae 8%; Malaita Kwara'ae 23%, Are'are 12%,
Lau 10%, Kwaio 10%; Isabel Cheke Holo 48%, unnamed 30%; Temotu Aiwoo 40%, unnamed 27%.

**Step 2, wards**: IPF on languages x wards inside each province. Columns = each ward's P8.3
total (religiondots' `sb.csv`, the `Total` rows) scaled by its province's 5+ share. Seeds:
`exp(-d / 10 km)` from the language's Glottolog location (`data/raw/glottolog/languages.csv`,
every code looked up there, listed in `sb_model.py`) for languages in their home province, or
from the census's own ward names where they say where it is spoken (Ulawa 801-803, Arosi
805-808, Bauro 809-811, Owa 815-816, Kiribati on Wagina 101); flat for Pidgin, the remainder and
everyone away from home. Ward population-weighted centres come from religiondots' Kontur hexes.

Sanity, Malaita's largest language by ward: Auki Kwara'ae 62%, Malu'u To'abaita 62%,
Sulufou/Kwarande (the Lau Lagoon) Lau 55%, East Baegu Lau 66%, Are'are Are'are 85%, Asimae Sa'a
75%, Sikaiana Sikaiana 92%, and Luaniua and Pelau (Ontong Java, whose Luangiua the report does
not name) unnamed 63%. Ghari's Glottolog point is 5 km from a Honiara ward; it is placed within
Guadalcanal only, so this does not matter.

Weaknesses, said in `note_public` in short: who in Honiara speaks what is only the migrants'
proportions; a territory is a disc around one point; Pidgin is flat inside each province, so the
towns (Auki, Gizo, Noro, Lata) are not favoured.

## 4. Mapping and tree (`taxonomy/sb2019.py`, `tree.d/sb.txt`)

Every printed label has a node, including the ones the report's commentary calls dialects or
lineage names (Mae, Ghoighoi, Laghu, Dororo, Guliguli, Kazukuru, Ririo). Groups, checked against
Glottolog's classification paths: `austronesian.oceanic.nw_solomonic`, `.se_solomonic`,
`.temotu` (new, with colours); the Polynesian outliers directly under Oceanic as Samoan and Tongan
already are; Bilua, Touo and Lavukaleve under `papuan.central_solomons` (new; Glottolog has each
as an unclassified isolate, the usual reading is a Central Solomons family, and readers know them
as Papuan). Pidgin on au.txt's `creole.english_based.pijin`, Kiribati on `gilbertese`.

Name calls: Tairaha is Bauro (Wikipedia's Bauro article gives Tairaha as its other name; it is
also the only way Bauro, a language of ~10,000, could be missing from the list); Laube is
Lavukaleve (the report says so; only the 36 who said "Laube"); Asumnoa = Asumboa; Engdewu and
Noipa both footnoted by the report to Ethnologue `ngr`, Glottolog's Nanggu, so both are anchored
there but keep their own nodes; Tauma taken as Taumako (Duff Islands, Polynesian), which the
report cannot verify; Wala is Glottolog's name for the Langalanga language (ISO lgl), and is
anchored in the Langalanga lagoon.

**The remainder (90,859) is on `other`**, drawn grey as language not named. It holds English and
foreign languages together with every unlisted local language, and nothing published separates
them, so spec 3.2's narrowest node is the root. The rule against putting indigenous remainders on
`other` applies where a census lets the two be told apart; this one does not. It is probably
mostly local languages (Tikopia, Natügu, Bughotu, Kokota, Zabana, Blablanga, Savosavo,
Lavukaleve, Luangiua, Fataleka, Langalanga's neighbours, Longgu, Malango, Marau, Fagani...).

Colours: 21 Southeast Solomonic languages hand-picked in the fragment so Malaita's, Guadalcanal's
and Makira's neighbours differ (the generated ones put Baegu next to Lau at 0.043 OKLab and Kwaio
next to Are'are at 0.047). After that every pair of languages that both hold 8%+ of some ward is
at least 0.073 apart, except Amba (501 people) beside Pijin in one Utupua ward.

## 5. Geography and scatter

religiondots' placement layer as is: `../religiondots/data/geo/sb/sb_hexes.gpkg` (6,385 Kontur
hexes over the 183 wards, joined there on SINSO's own ward id and checked against OCHA's ADM1),
via its `sb_lookup.csv`; `pop_weight`. Scatter: 607 dots at 1:1000, 20 rings (all modelled rows);
24,060 people (3.81%) fall under one dot per language nationally, the carry of 60 languages
spread thin over 183 wards. water.py leaves 111 units unclipped (over 95% sea), the same as
religiondots since it shares the cache.

## 6. Leads not followed

- The 1999 census *Report on the Census, Basic Tables* had first language by province (the
  coverage sweep's note; 9.6.1 quotes its 1999 column). It would be a real province-level check
  on step 1, 20 years older. The SPC digital library and Pacific Data Hub, where it would be,
  returned 403 from here (2026-10-05); not tried via Wayback.
- SPC microdata library: 2019 census microdata, by application. Gated.
