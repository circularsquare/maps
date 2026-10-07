# Caribbean Netherlands (bq): record

Drawn 2026-10-05 (session edd42a8c-bq). CBS Omnibus survey 2021, the language each person aged
15+ speaks most, shares per island, applied to each island's population on 1 January 2022:
27,749 people in 5 answers on 3 islands, all `modelled`. Placed inside each island on WorldPop's
constrained 2020 grid.

Files: `sources/bq_survey.py` (counts), `sources/bq_geo.py` (placement layer),
`taxonomy/bq2021.py`, `countries/bq.py`, `data/normalized/bq.csv`, `data/geo/bq/bq_cells.gpkg`,
`data/raw/bq/` (82867NED and 83774NED as JSON, two WorldPop GeoTIFFs). No tree fragment: every
node exists.

## 1. What exists, and why this table

- **No census asks language.** The Caribbean Netherlands has had no census since the Netherlands
  Antilles' 2001 round (the coverage sweep's lead, "home language, 2001"). 2001 is a different
  population: Bonaire has since roughly doubled, mostly through immigration from Curaçao, the
  Netherlands and Latin America. Not searched for or fetched.
- **CBS StatLine 82867NED**, *Caribisch Nederland; gesproken talen en voertaal,
  persoonskenmerken*, from the Omnibus survey (rounds 2013, 2017/2018, 2021; 2021 provisional,
  table modified 2022-09-27; a new round every four years, none published since). Question:
  "welke taal of welke talen spreekt u?" (Papiaments, Engels, Nederlands, Spaans, Anders; several
  allowed), then of anyone naming more than one, "welke taal spreekt u het meest?". The
  **Voertaal** columns are that most-spoken language, one per person: a single-answer main
  language, so no sharing across answers is needed (AGENT_BRIEF §2). Fetched through the OData
  API (`opendata.cbs.nl/ODataApi/odata/82867NED/TypedDataSet`, all persons `T009002`).
- The 2025 CBS longread *De Nederlandse Caraïben vijftien jaar na de staatskundige hervorming*
  (Table 3.3.1) quotes the same 2021 figures; nothing newer exists.
- religiondots draws bq from the sister table 82868NED (religion) of the same survey, and this
  follows its method: shares times the 1 January 2022 populations (22,573 Bonaire, 3,242 Sint
  Eustatius, 1,911 Saba), the date nearest the October-December 2021 fieldwork.

## 2. Figures and checks (sources/bq_survey.py)

| island | Papiamentu | English | Dutch | Spanish | other | sum |
|---|---|---|---|---|---|---|
| Bonaire | 62.4 | 5.6 | 15.0 | 15.4 | 1.7 | 100.1 |
| Sint Eustatius | 1.3 | 81.2 | 3.6 | 12.8 | 1.2 | 100.1 |
| Saba | withheld | 83.3 | 4.1 | 9.9 | 2.5 | 99.8 |

- Populations pinned; four 2021 shares pinned (a revision of the provisional figures fails).
- Each island's shares sum within 99-101; rounding puts 23 more people on Bonaire and 4 on Sint
  Eustatius than live there (27,749 drawn against 27,726). Left as is: 0.1%.
- Every main-language share is at most the share speaking that language at all (Bonaire:
  Papiamentu 62.4 main, 88.4 speak it; Dutch 15.0 against 76.6).
- The earlier rounds hold the same shape (Bonaire Papiamentu 63.8 / 60.2 / 62.4; Sint Eustatius
  English 84.7 / 80.3 / 81.2), with Spanish rising on the two Windward islands.
- **CBS's own all-islands row (CN01) is not population-weighted**: it gives English 43.1% and
  Papiamentu 32.2%, against 19.8% and 51.0% from the island rows weighted by population. It looks
  like the respondents pooled unweighted across islands (the small islands are oversampled). Not
  used; anyone quoting a Caribbean Netherlands total should not take it from that row.

## 3. Mapping (taxonomy/bq2021.py)

Papiaments on Papiamento, Engels on English, Nederlands on Dutch, Spaans on Spanish, Anders
(other; 1.2-2.5%) on `other`, since the survey offered only five answers and "Anders" holds every
other language unnamed. English on Sint Eustatius and Saba is largely the islands' own
English-lexifier vernacular; the survey has no answer for it, so it is drawn as English. Saba's
withheld Papiamentu (4 people of remainder) is not drawn and is the gap. Colours unchanged; the
four named languages are the same pale green, pale blue, blue and ochre as on Curaçao, Aruba and
Sint Maarten, and tell apart.

## 4. Placement (sources/bq_geo.py)

- Kontur has no cells for BQ (religiondots found its extract empty), and religiondots spreads
  dots evenly over Natural Earth's island polygons. Here **WorldPop's constrained 2020 raster for
  BES** (`Global_2000_2020_Constrained/2020/BSGM/BES/bes_ppp_2020_UNadj_constrained.tif`), summed
  in 4x4 blocks to ~370 m cells (482 populated), is the within-island weight. It moves people only
  inside their island (AGENT_BRIEF §4.4). Every language is placed by population: nothing gives
  where Spanish or Dutch speakers live within an island.
- Cells keyed to islands by religiondots' own longitude/latitude split of the NLY unit; asserted
  that every cell is within 0.05 degrees of religiondots' polygon for its island.
- WorldPop per island against CBS 1 January 2020, each normalised: Bonaire 0.92, Sint Eustatius
  1.54, Saba 1.03. Sint Eustatius's excess is a level difference (WorldPop's UN-adjusted island
  total runs ahead of CBS) and does not matter to a within-island weight; the check fails only
  beyond a factor of 2. The unconstrained 2020 raster gives the same island totals (same
  adjustment), so it is a weak control.
- Shape sanity check on Bonaire: about 14,900 of WorldPop's 19,300 in a box around Kralendijk,
  1,900 around Rincon, 208 in Washington Slagbaai park, 46 south of 12.08 N (the salt pans).

## 5. Calls someone might reverse

- A 15+ survey drawn on whole island populations, children included (as religiondots does).
- 2021 survey over the 2001 census (§1).
- WorldPop placement within islands, where religiondots spreads evenly.

## 6. Scatter

26 dots at 1:1000 over 26 cells, 1 ring; 1,749 people (6.3%) fall under one dot per language and
draw none. scatter.py's water check left 2 cells that were over 95% sea unclipped.
