# Germany (de): the record

Drawn 2026-10-05 by session d9e44929-de. 83,875,101 people, 16 Laender, 31 nodes, 83,860 dots;
since 2026-10-06 (session 5d7dac7e-oth) 127 nodes, 83,815 dots, the remainders split by citizenship;
since 2026-10-06 (session 5d7dac7e-dereg) 133 nodes, 83,814 dots, with Low German, Sorbian,
Frisian and the Danish minority (`sources/de_regional.py`).
Build: `python sources/de_mz.py --fetch`, `python sources/de_place.py --fetch` (placement,
added 2026-10-05 by d9e44929-de2), `countries/de.py`, `taxonomy/de2023.py`,
`taxonomy/tree.d/de.txt` (no new nodes; bare repeats only).

## What was asked, and by whom

Zensus 2022 (and 2011) asks nothing about language. The only official source is the
**Mikrozensus**, the annual 1% household sample survey, which since 2017 asks "Welche Sprache
bzw. welche Sprachen sprechen Sie zu Hause?" and then "Welche Sprache sprechen Sie vorwiegend zu
Hause?" (Berlin-Brandenburg's metadata sheet MD_12211_2024 prints the questions). The drawn item is
the second one, the language spoken **mainly at home**, one answer per person. It covers people in
private households only; the figures are survey extrapolations rounded to the thousand.

## Tables

1. **IntMK Integrationsmonitoring der Laender 2025, indicator J2** ("Bevoelkerung mit
   Migrationsgeschichte nach vorwiegend gesprochener Sprache"), 2017-2023, by Land,
   https://www.integrationsmonitoring-laender.de/documents/j2-2017-2023-1743420780_1743531218.xlsx
   -> `data/raw/de/intmk_j2_2017_2023.xlsx`. 2023 is Mikrozensus first results (Erstergebnisse),
   extrapolated on the Zensus 2011 basis. Eight groups: deutsch, west-European (English, French,
   Italian, Spanish, Dutch, footnote 1), Polish, Russian, Turkish, other European, Arabic, other.
   Only people WITH Migrationsgeschichte (they or at least one parent not German-born-German,
   counted "im weiteren Sinn", parents outside the household included).
2. **IntMK indicator A1a**, the same release: each Land's population in private households, with
   and without Migrationsgeschichte,
   https://www.integrationsmonitoring-laender.de/documents/a1a-2011-2023-1743417718_1743504177.xlsx
3. **Destatis, Statistischer Bericht "Mikrozensus - Bevoelkerung nach Einwanderungsgeschichte",
   Endergebnisse 2023** (19.05.2025), table 12211-40 / csv-12211-40: 32 named languages plus
   German, NATIONAL only, by Bevoelkerungsgruppe (Einwanderungsgeschichte, citizenship, age,
   country of birth). Statistische Bibliothek zip DEHeft_derivate_00093474.
4. The same report for **2024** (04.02.2026, DEHeft_derivate_00099265), used only as a drift
   check on the borrowed shares.

Where else I looked, and why not:
- **GENESIS-Online** has no language table. Its open API (base
  `https://genesis.destatis.de/genesis/api/rest/`, found in the SPA's `environment.json`; the old
  `genesisWS` endpoints want an account) lists 132 tables under 12211 and none carries a
  language variable; its search for "Sprache" hits only occupation codes. The Statistischer
  Bericht's own GENESIS sheet says 12211 tables are "in Vorbereitung".
- **Land offices' own releases** (Hessen 2024, Rheinland-Pfalz) print a few shares, on
  differing definitions (Hessen's is "another language besides German", several allowed). Not a
  consistent 16-Land table; J2 is the Laender's own harmonised one.
- **Mikrozensus Campus-File / Scientific Use File**: registration-gated (FDZ). Skipped.
- Below Land: nothing. The Mikrozensus is not published by Kreis for this item.

## How the drawn rows are built

Per Land (all figures in persons, from thousands):

| part | rows | tier |
|---|---|---|
| people with Migrationsgeschichte speaking German, Polish, Russian, Turkish, Arabic | J2's own columns | measured, 14.73M |
| J2's west-European, other European and other groups | split into Destatis' named languages by the national 2023 shares among everyone not "ohne Einwanderungsgeschichte" (12211-40 Insgesamt minus Ohne) | derived |
| four suppressed J2 cells ("/", under 71 sample cases): Turkish in Brandenburg, Mecklenburg-Vorpommern, Sachsen; Polish in Saarland | the row total minus the shown cells: 2,500, 2,000, 7,000, 5,400 | derived |
| people without Migrationsgeschichte (A1a), 60.70M | the national 2023 mix of "ohne Einwanderungsgeschichte": 98.83% German; its shown non-German cells (suppressed ones count 0) scaled by 1.069 to the row's "vorwiegend nicht-deutsch" | derived |

So 17.6% of the people drawn are measured and the rest derived; turning off inferred dots leaves
the measured migrant-language picture, the same shape as the Americas' indigenous-only countries.
This follows the Bulgaria test in Claude's memory (published totals, a borrowed shape that can be
checked, reversible in the viewer). J2's Land totals are measured; only their composition inside
the three mixed groups, and the language of the "ohne" group, are borrowed from the nation.

## Checks (sources/de_mz.py prints all of these and fails on them)

- J2's Land totals equal A1a's "mit Migrationsgeschichte" in all 16 Laender (to 0.15k).
- The 16 Laender sum to the Deutschland row in every complete J2 column and in A1a
  (83,875.3k in private households, 23,172.4k with Migrationsgeschichte).
- Destatis 2023: the 32 languages sum to 14,838k against "vorwiegend nicht-deutsch" 14,837k;
  German + non-German = total (82,472k; the Endergebnisse are re-extrapolated on Zensus 2022,
  hence lower than J2's Erstergebnisse on the Zensus 2011 basis).
- **The grouping.** Every Destatis language sits in exactly one J2 group, and J2's national
  non-German mix matches Destatis' (not-ohne) mix under that grouping: west-European 12.2 vs
  12.1%, Polish 6.9 vs 6.9, Russian 12.9 vs 12.8, Turkish 12.6 vs 13.8, other European 28.1 vs
  27.7, Arabic 9.8 vs 9.4, other 17.5 vs 17.3. This is what settles Kurdish (2.8% of non-German)
  as "sonstige Sprache", not European. Turkish is the loosest: Destatis' not-ohne base includes
  people with one immigrant parent that J2 counts as without.
- **Drift.** The borrowed within-group shares move at most 3.2 points from 2023 to 2024 (the
  build fails above 6); the "ohne" German share goes 98.83% -> 98.76%.
- Drawn rows sum to A1a per Land within 300 people (rounding); 83,875,101 against 83,875,300.
- check_country: ok. scatter: 83,860 dots on 44,966 grid cells; 15,101 people (0.02%) under one
  dot per language draw none.

## Placement inside a Land: citizenship (2026-10-05)

Anita, 2026-10-05: "Germany immigrant languages by foreign citizen is good. Ideally if we have
the specific foreign origin, that'd be best" (AGENT_BRIEF §4 item 4). The counts above are
untouched; only where a language's dots go inside its Land changed. `sources/de_place.py`
writes `data/geo/de/de_grid_1km.gpkg`: religiondots' 1km layer (geometry, `ars`, `pop`, read
only), re-keyed to its INSPIRE cell id, plus one column `w_<slug>` per Mikrozensus label.

**Sources, all Zensus 2022 (15 May 2022), Datenlizenz Deutschland Namensnennung 2.0:**
- Zensus database table **1000A-1023 "Personen: Staatsangehoerigkeit (Laender)"**, 203
  citizenships x all 10,786 Gemeinden. Open, no login: the Zensusdatenbank SPA's REST API at
  `https://ergebnisse.zensus2022.de/proxy/api/rest/` (GET `tables/1000A-1023/structure`, swap
  the Laender block in the column layout for the Gemeinden block, POST it to
  `tables/1000A-1023/download/ffcsv/de`; 48 MB zip). -> `data/raw/de/zensus2022_1000A-1023_gemeinden.zip`.
  Not GENESIS-Online: its API base in `sources/de.md` above returns 404 on catalogue routes,
  and the Zensus results live in the separate Zensusdatenbank anyway.
- 1km grids from destatis' Gitterzellen page: **Staatsangehoerigkeit nach ausgewaehlten
  Laendern** (12 countries: Turkey, Poland, Russia, Kazakhstan, Ukraine, Romania, Italy, Greece,
  Croatia, Bosnia, Netherlands, Austria) and **Staatsangehoerigkeit Gruppen** (German, EU27,
  other Europe, rest of the world, stateless/unknown). Also on the page and unused: Geburtsland
  (Gruppen), country of birth in the same five groups.

**The weight**, for language L in cell c of Gemeinde g: for each citizenship k listed for L,
k's people in g (table) spread over g's cells by k's own grid column, or, where the 12-country
grid lacks k, by its group's column less the countries that grid carries (`EU27_rest`,
`Europa_rest`), or the rest-of-the-world column. Summed over k. Where a column is zero across a
Gemeinde (the Cell-Key method zeroes small grid counts) all foreign citizens stand in, then
population. German is placed on German citizens rather than population: in a cell where foreign
languages now draw thicker, German draws thinner, so the total dot density still follows people.
A node fed by several labels (`other`) blends the labels' weights by their share of it in that
Land (`countries/de.py`).

**Language -> citizenships** (`LANGS` in de_place.py; kept simple, a citizenship may serve
several languages):

| language | citizenships |
|---|---|
| German | Germany |
| Turkish | Turkey |
| Polish | Poland |
| Russian | Russia, Kazakhstan, Ukraine at half (646k Ukrainians on Zensus day, mostly that spring's refugees, against 241k Russians) |
| Arabic | Syria, Iraq, Lebanon, Morocco, Algeria, Tunisia, Egypt, Jordan, Palestine, Libya, Yemen, Sudan, Saudi Arabia, UAE, Kuwait, Qatar, Bahrain, Oman, Mauritania |
| Kurdish | Turkey, Syria, Iraq, Iran |
| Persian | Iran, Afghanistan; Pashto: Afghanistan, Pakistan; Urdu: Pakistan; Hindi: India |
| Chinese | China, Hong Kong, Macau, Taiwan; Vietnamese: Vietnam |
| English | UK (and overseas territories), Ireland, USA, Canada, Australia, New Zealand |
| French | France, Belgium, Luxembourg, Cameroon |
| Spanish | Spain and 18 Spanish-speaking American countries |
| Portuguese | Portugal, Brazil, Angola, Mozambique, Cabo Verde, Guinea-Bissau |
| Dutch, Italian, Croatian, Bosnian, Bulgarian, Hungarian, Ukrainian, Danish | their own country |
| Romanian | Romania, Moldova; Greek: Greece, Cyprus; Serbian: Serbia, Montenegro |
| Albanian | Albania, Kosovo, North Macedonia; Macedonian: North Macedonia |
| another language of Europe / Asia / Africa | that continent's citizenships less every one listed above (and less Austria, Switzerland, Liechtenstein) |
| any other language | every foreign citizenship |

**Checks** (de_place.py prints them; the first four stop the build):
- Every Mikrozensus label has a mapping; every named citizenship is a label in the table.
- Cell ids recovered from the geometry: 209,154 rows, 33 whole-Gemeinde polygons (no cell
  centre fell in them; they place by population), one cell split across two rows (kept on the
  larger piece). Layer -> groups grid: all found. Layer -> 12-country grid: 327 tiny cells
  (1,625 people) absent there, taking zero. Grid -> layer: 1,436 cells (159,392 people) not in
  the layer, the cells religiondots dropped for a centre in no Gemeinde.
- Table <-> layer `ars`: 10,786 of 10,786 both ways.
- National totals, table against the 12-country grid: all 12 within 0.6% (Turkey 1,298,737 vs
  1,298,674). Groups: EU27 4.31M vs 4.29M, other Europe 3.28M vs 3.28M, rest of the world 3.32M
  vs 3.20M (the grid files 125k stateless/unknown apart).
- Dots per (Land, language), before vs after: all 482 pairs identical.
- **Where Turkish lands**, dots before (population) -> after (citizenship): in the 78
  Gemeinden of 100,000+ (31.7% of people) 37.6% -> 49.2%; in Gemeinden under 20,000 (40.6% of
  people) 33.9% -> 20.8%; of NRW's Turkish dots, Duisburg + Gelsenkirchen 3.4% -> 10.8%; of
  Berlin's, in its most-Turkish fifth (the cells with the highest Turkish-citizen share,
  holding 20% of Berliners) 19.5% -> 60.2%. Arabic in 100k+ cities 38.8% -> 50.9%, Polish 36.5%
  -> 37.2%, Russian 33.1% -> 39.8%, German 31.0% -> 29.3%.

**Calls someone might reverse:**
- Citizenship as the locator, not country of birth: the ruling names citizenship, and only
  citizenship has per-country counts at Gemeinde level and on the grid. Naturalised speakers
  (most Russian speakers are German-citizen Aussiedler) are assumed to live where foreign
  citizens of the same origin do. Said in note_public.
- Ukraine at half weight for Russian.
- German on German citizens, not plain population (reads better: the dot total per cell stays
  close to population).
- The layer copies religiondots' geometry; if religiondots rebuilds `de_grid_1km.gpkg`, re-run
  `sources/de_place.py`.

## The remainders split by citizenship (2026-10-06)

Session 5d7dac7e-oth, after Anita's note that the grey "other languages" wedge is big in
Munich. `python sources/de_rest.py --fetch` -> `data/normalized/de_rest.csv`, applied in
`countries/de.py` `_rows()`.

**What was grey.** In Munich's dots before this: `other` 2.1% (the Mikrozensus' "another
language of Europe" and "of Asia" and "any other language"), Arabic 1.1% and Chinese 0.5%
(named, on group nodes, drawn washed in their family colour), "other African" 0.4%. Nationally
`other` 1.07M (1.27%), `africa_other` 0.28M. **But the grey wedge in the pies is mostly not
these.** The viewer's pies keep 7 languages and fold the rest into one grey "N other languages"
wedge (`index.html`, PIE_K = 8): in Munich that fold is 17% of the pie, 46 languages, and
almost all of it is named languages (Italian, Croatian, Greek, Albanian, Spanish...). See the
report and `followups.md`.

**The split.** Destatis 12211-40's three continental remainders (Europe 260k, Asia 448k,
Africa 277k) are each split per Land in proportion to Σ_k citizens_k(Land) × mix_k(L): k every
foreign citizenship of that continent (Zensus 2022 table 1000A-1023, by Land, the table's code
ranges; Turkey with Asia for its Zazaki), mix_k = `origin_mix.mix(k, "de")`, and L only the
languages the Mikrozensus does not name (named languages, their varieties, Arabic and Chinese
varieties, Dari, groups holding a named language and `other` dropped from each mix). A
language under 0.5% of a remainder's pool nationally is not drawn on its own; its share stays
on `other` / `africa_other`. "Eine sonstige Sprache" (359k) stays on `other`: nothing says what
it holds.

Uncited exclusions (`EXCLUDE`): Italy's regional languages (an Italian answers "Italienisch";
they would have been 30% of Europe's remainder), the Netherlands' Westphalian, Limburgish and
Frisian, Moldova's "Moldovan" (Romanian, named), and Kazakhstan entirely (its citizens in
Germany are mostly Russian-speaking Aussiedler families, not at Kazakhstan's 75% Kazakh).

**Checks** (de_rest.py prints them; `countries/de.py` asserts the split keeps each remainder's
total): Land table foreign citizens 10,913,291 against 10,913,338 nationally. Candidate pools
against the remainders they split: Europe 338k for 260k (130%), Asia 580k for 448k (130%),
Africa 398k for 277k (144%); a pool above its remainder is expected, since many foreign
citizens speak German or a named language at home. Kept: Europe 21 languages (98.7% of its
pool), Asia 46 (93.1%), Africa 30 (70.4%; Nigeria's, Ghana's and Cameroon's long tails stay on
`africa_other`). check_country ok, 83,875,101 people unchanged, 127 nodes (31 before);
scatter 83,815 dots, 0 rings.

Largest results nationally: Europe: Czech 16%, Slovak 14%, Lithuanian 13%, Romani 10%,
Slovenian, Latvian, Swedish, Luxembourgish. Asia: Azerbaijani 8%, Korean, Bengali, Punjabi,
Japanese, Thai, Armenian, Uzbek, Georgian (4-6% each), Sinhala, Tamil, Zazaki, Isan. Africa:
Tigrinya 12%, Somali 7%, Hausa 7%, Akan, Fulah, Yoruba, Tigre, Igbo.

**What it assumes, and where it is weak.** Citizenship misses naturalised speakers: Aramaic and
Assyrian (mostly German citizens), Sinti Romani and Sorbian are absent, and their people go to
the languages that are in the pool. Home mixes are the origin's whole country (Nigeria's Hausa
35%, though Nigerians in Germany are likelier southern; India's at its national mix, though
Germany's Indians lean south). Placement: a split language is placed by its remainder's
column (all of that continent's unnamed citizenships), not by its own origin's citizens; a
per-language weight would need new columns in `de_place.py`. All rows `derived`, as the
remainders already were. Reverse: drop the merge in `countries/de.py` `_rows()`.

Munich after: `other` 0.7% of dots (was 2.1%); the pie's grey fold 17% either way.

## Regional and minority languages (2026-10-06)

Session 5d7dac7e-dereg, Anita's task of 2026-10-06 under the rich-country ruling (AGENT_BRIEF
§2: regional languages from regional language surveys, as Wales and France) and ask 019
(published estimates placed by homeland). `python sources/de_regional.py` ->
`data/normalized/de_regional.csv` (Land, node, count, area) and `de_regional_areas.csv` (ars,
area, factor), applied in `countries/de.py` `_regional()`. Every row `modelled`; each Land's
total is taken out of that Land's derived German row (people without a migration history), so
Land totals are unchanged (83,875,101). 1,127,923 people, 1.3% of Germany.

**Low German.** Adler, Ehlers, Goltz, Kleene, Plewnia, *Status und Gebrauch des
Niederdeutschen 2016. Erste Ergebnisse einer repräsentativen Erhebung* (IDS/INS, Mannheim
2016), https://ids-pub.bsz-bw.de/frontdoor/index/index/docId/9037 (Anubis bot wall; fetched from
the Wayback copy of 2025-04-26 -> `data/raw/de/ids_niederdeutsch_2016.pdf`). Telephone survey,
June 2016, 1,632 German-speaking people aged 16+ in SH, HH, HB, NI, MV and the north of NW, ST,
BB (Hessen's small Low German strip not surveyed, so not drawn). It has **no first-language or
home-language question** (Abb. 16 "family and friends" 26.6% counts anyone who speaks it there
even rarely). Drawn: the share who speak it **"sehr gut"**, Abb. 10, read off the chart, which
reproduces the text's (sehr) gut sums for SH, MV, NW, ST, BB exactly:

| Land | sehr gut | gut | area | drawn |
|---|---|---|---|---|
| SH | 16.5 | 8.0 | whole Land | 354,705 |
| HH | 3.2 | 6.3 | whole | 37,592 |
| NI | 4.7 | 12.7 | whole | 272,628 |
| HB | 9.9 | 7.7 | whole (n=49) | 40,730 |
| MV | 5.9 | 14.8 | whole | 74,598 |
| NW | 5.2 | 6.6 | RB Münster, Detmold, Arnsberg less Siegen-Wittgenstein (45% of NW) | 282,667, as Westphalian |
| ST | 2.2 | 9.6 | Altmark (2 Kreise), Börde, Magdeburg, Jerichower Land (32%) | 11,673 |
| BB | 2.6 | 0.2 | Prignitz, Ostprignitz-Ruppin, Uckermark, Oberhavel (20%) | 10,330 |

count = share x 0.85 (aged 16+; under-20s 0.8% speak it well, Abb. 11) x the Land's German
speakers x the area's share of the Land's population (1km grid). "(Sehr) gut" would give 2.70M
instead of 1.08M; drawing that many as people whose main language is Platt would overstate it,
since nearly all speak High German day to day. Placement inside the area: cells by population,
Gemeinden under 20,000 people at double weight (the survey finds competence higher in places up
to 20,000, and Platt heard in the neighbourhood by 36.6% in places under 2,000 against under 17%
over 50,000, p. 19; the factor 2 is mine). Node: NW's Platt on nl's `lowgerman.westphalian`
(Glottolog Westphalic west2356; the area drawn is Westfalen-Lippe). Elsewhere the Land figure
mixes Northern Low Saxon, Eastphalian, Westphalian (Osnabrück, Emsland) and East Low German, so
it goes on a new leaf `lowgerman.plattdeutsch` "Low German (Plattdeutsch)", hand-coloured light
teal (0.80 0.12 180) against German's blue. Not split further; a dialect split would follow
whatever Anita rules for High German.

**Published estimates, by homeland** (Sorbian settlement area per de.wikipedia "Sorbisches
Siedlungsgebiet"; Gemeinden in it only with a few named villages left out, those in it "except"
a few villages taken whole; names matched to Zensus 2022 ARS, each exactly once):

| language | count | source | placed |
|---|---|---|---|
| Upper Sorbian | 20,000 | 20,000-25,000 speakers, Schön, Scholze (eds.), *Sorbisches Kulturlexikon*, Bautzen 2014, p. 291; low end | 60% of residents (low end of the 60-90% de.wikipedia gives) in the five Catholic core Gemeinden Crostwitz, Panschwitz-Kuckau, Ralbitz-Rosenthal, Nebelschütz, Räckelwitz: 4,340 of 7,234; 15,660 over the other 32 Saxon settlement Gemeinden, rural-weighted |
| Lower Sorbian | 5,000 | about 5,000 speakers, Starosta, Bartels, "Niedersorbisch", Sorabicon (via de.wikipedia) | 34 Brandenburg settlement Gemeinden incl. Cottbus, rural-weighted |
| North Frisian | 8,000 | 8,000-10,000, Land Schleswig-Holstein's official figure (via de.wikipedia "Nordfriesische Sprache"); low end | 3,500 on Föhr and Amrum (Amt Föhr-Amrum, 15 Gemeinden; the article's estimate of speakers there), 4,500 over the rest of Kreis Nordfriesland and Helgoland, rural-weighted |
| Saterland Frisian | 2,000 | 1,500-2,500 (Fort 2001, *Handbuch des Friesischen* p. 410; Stellmacher's 1995 count 2,225); midpoint | Gemeinde Saterland |
| Danish (minority) | 8,000 | 8,000-10,000 use Danish daily (Institut for Grænseregionsforskning / Gesellschaft für bedrohte Völker, via de.wikipedia "Dänische Minderheit in Deutschland"); low end | Flensburg, Kreise Schleswig-Flensburg, Nordfriesland, Rendsburg-Eckernförde, by population |

Danish: the Mikrozensus' Danish (395 people in SH) is only people with a migration history; its
"ohne" Danish cell is suppressed, so the minority was on German. The 8,000 join the same node
and are placed on the minority's area; the Mikrozensus' Danish stays on Danish citizens.

**Not drawn: Romani / Sinti.** Estimates exist (Zentralrat: about 70,000 German Sinti and Roma;
Council of Europe 70,000-150,000), but they count people, not speakers, and German Sinti have no
home region to place them by. Left as before.

**Checks** (de_regional.py prints them): every Land keeps most of its German (largest take SH
14.7%, HB 8.4%); all names matched once; area rows = row areas. check_country ok, 133 nodes;
scatter 83,814 dots, 0 rings; per-node bounding boxes land where they should (Upper Sorbian
51.15-51.53 N, 14.17-14.77 E; North Frisian 54.39-54.84 N; Saterland one point 53.1 N 7.69 E;
Westphalian 50.97-52.45 N, 6.46-9.38 E).

**Calls someone might reverse:** "sehr gut" as the stand-in for first language (no such
question); the 0.85 adult factor; the NW/ST/BB area boundaries; the double weight under 20,000;
low ends of each published range; Westphalian for NW only. Colour: Westphalian is nl's
(0.52 0.12 245), a darker shade of German's hue; in NRW it may not stand out from German. Not
recoloured, since nl owns it.

## Calls

- **Source and vintage.** 2023, because J2's latest Land figures are 2023 Erstergebnisse;
  the national shares come from the same year's Endergebnisse.
- **Splitting J2's mixed groups by national shares** rather than leaving them on group nodes.
  "West-European" is five named languages, so it may not sit on a group node (a named label on
  a group draws as "language not named"); the two "other" groups (4.09M and 2.55M nationally in J2)
  would have drawn 6.6M people as unnamed. Reversible: the split rows are all `derived`.
- **People without Migrationsgeschichte at the national mix**, not all German. 0.7M of them
  speak mainly another language nationally (English 260k of them), and the national figure
  is published; drawing them all German would assert something the source contradicts.
- **Placement**: unit = the first two digits of `ars` on the 1km Zensus 2022 grid. Until
  2026-10-05 weighted by cell population, so Turkish in NRW drew across the Sauerland as
  densely as in Duisburg; now by citizenship, see "Placement inside a Land" below.
- Labels: "Chinesisch" on Sinitic (precedent cz/pl/us/uk); "Kurdisch" on the Kurdish leaf; (the three continental remainders below are split since 2026-10-06, see above)
  "andere in Europa/Asien" and "sonstige" on `other`; "andere in Afrika" on `africa_other`.
- **Minority languages.** Sorbian, North and Saterland Frisian, Romani and Low German are not
  named by the Mikrozensus; speakers who answered one are inside "other European" (`other`),
  and most Low German speakers will have said German. Danish is named (21k nationally in 2024).
  Since 2026-10-06 the regional ones are drawn as modelled rows; see "Regional and minority
  languages" below. Romani is still drawn only for foreign Roma.
- gap: about 0.8M people in communal housing (end-2023 population 84.67M on the Zensus 2011
  basis less A1a's 83.88M in private households).

## Colours

Nothing re-coloured. German is #359bd9; Greek (#5aa3ec, hand-set in other fragments) is
nearly the same blue and its ~300 dots will vanish among German. Polish (#00a76c) and Russian
(#54b85b) are both green but tell apart. Turkish pink, Arabic pale green, Romanian pink-lilac,
Albanian ochre, Kurdish sand. Greek is the one worth a look when colours are tuned.
