# Netherlands (nl): the record

Drawn 2026-10-05 by session edd42a8c-nl. European Netherlands only (the Caribbean islands are
the bq, aw, cw, sx rows). 18,127,984 people (CBS, 1 Jan 2026; 2,224 dropped as rows under 0.5),
342 gemeenten (2026 classification), 129 languages, 18,081 dots. Every row is `derived`.

Build: `python sources/nl_cbs.py --fetch` (CBS StatLine OData into `data/raw/nl/`, then
`sources/nl_build.py` writes `data/normalized/nl.csv` and `data/geo/nl/nl_weights.csv`);
`python sources/nl_geo.py --fetch` (PDOK polygons, CBS PC4 table, `data/geo/nl/nl_hexes.gpkg`).
Mapping `taxonomy/nl2026.py` (France's labels plus the Netherlands'), nodes
`taxonomy/tree.d/nl.txt`, entry `countries/nl.py`. Survey PDFs found by the research are in the
session scratchpad (`edd42a8c-nl/regional/`, `edd42a8c-nl/retention/`), not copied to the repo.

## 1. The rule

No Dutch census has asked language. Anita's 2026-10-05 ruling for rich countries with no
language question (AGENT_BRIEF §2): Dutch, plus (a) regional languages from surveys, (b)
immigrant languages by origin, all proxy rows derived.

## 2. Immigrants: CBS herkomstland, 1 Jan 2026

- **85458NED** (Bevolking; herkomstland, geboorteland, leeftijd, regio): every gemeente, 48 named
  countries plus continents and CBS groups, split born abroad / born in NL. CBS's herkomstland
  (2022 definition) is one's own birth country if born abroad, else the mother's, else the
  father's. 5,271,067 people of foreign origin: 3,124,728 born abroad, 2,146,339 born in NL.
  "Onbekend herkomstland" is 0.
- **85384NED**: the same nationally with all 262 origins, which splits the gemeente table's
  remainders into countries by the national mix of the unnamed countries. Eight remainders
  (other EU, other central/eastern EU, other non-EU Europe, other Dutch Caribbean, other Africa,
  other Americas/Oceania, other Asia; GIPS is fully named): **each reconciles to the national
  members to the person**, both generations. Every gemeente's foreign-origin total is rebuilt
  exactly (worst gemeente off by 0.00).

Country -> language: France's `COUNTRY_LANG` (`sources/fr.md` §2) with Dutch overrides
(`NL_OVERRIDES`): Belgium -> Dutch (Flemish), South Africa -> Afrikaans, Indonesia -> Dutch
(the Indonesia-born are mostly repatriated Indo-Dutch; CBS's own commentary), Sint Maarten ->
English, Caribbean NL and the old Antilles -> Papiamento, Hong Kong and Macau -> Cantonese,
dissolved states on their main successor's language. Splits (`SPLITS`) from **NIDI, Hogendoorn,
"Taal en taligheid van mensen met een migratieachtergrond", Demos 39(4), 2023, p. 6** (SIM
2006/2011/2015 + SING 2009 pooled; among those who understand an origin language, the one they
understand best, first generation), normalised over the languages named:
- Suriname: Sranan 64, "Hindi" 29 (Sarnami), Javanese 2 -> 67/31/2%.
- Morocco: Arabic 51, Berber 43 -> Moroccan Arabic 54%, Tarifit 46%. **All Berber on Tarifit**:
  the Moroccan Dutch came overwhelmingly from the Rif. The often-quoted "~70% Tarifit"
  (Laghzaoui 2007) was found only second-hand; NIDI's 43-45% is the citable figure and may
  undercount (interviews could be in Arabic). France and Belgium used Morocco's 2024 census mix
  (Tarifit 3%); the Netherlands differs on purpose.
- Iraq: Arabic 55, Kurdish 35. Afghanistan: Dari 79, Pashto 16. Turkey: Turkish 94, Kurdish 4.

### 2b. Retention

The measure: share NOT speaking Dutch often or always with their children, first generation
(SCP). No Dutch source asks the single "language usually spoken at home" by origin; "with
children often/always" is the closest and is looser than France's "only French".

| origin | foreign share | source |
|---|---|---|
| Turkey | 63% | SIM 2015, *Bijlagen bij Integratie in zicht?* (SCP 2016) table B3.2 |
| Morocco | 46% | same |
| Suriname | 5% | same |
| Dutch Caribbean | 20% | same |
| Somalia (Eritrea, Ethiopia borrow it) | 64% | SIM 2015, *Gevlucht met weinig bagage* (SCP 2017) fig. 3.9 |
| Poland (Ukraine, Russia, Romania, Hungary, other central/eastern EU borrow it) | 86% | SIM 2015, *Bouwend aan een toekomst* (SCP 2018) fig. 4.3 |
| Afghanistan / Iraq / Iran | 76 / 71 / 65% | SING 2009, *Vluchtelingengroepen in Nederland* (SCP 2011) table 3.4 |
| Syria | 90% | NSN 2019, *Syrische statushouders op weg* (SCP 2020) table 1 |
| Bulgaria | 90% | *Langer in Nederland* (SCP 2015) table 3.2, with partner |
| Indonesia, Belgium | 0% | drawn Dutch |
| everything else | 23.4% | common rate, below |

**Calibration to CBS**: Schmeets & Cornips, "Talen en dialecten in Nederland", CBS Statistische
Trends 2021, table 2.2 (SSW 2019, 15+, language most spoken at home): first generation 44% mostly
another language, second generation 16%. The common rate (0.234) makes the first generation as
a whole 44%; the second generation takes k x its origin's first-generation rate with k = 0.502,
so it is 16% as a whole. This differs from France and Belgium, which drew the second generation
wholly on the national language: here a Dutch source measures it.

## 3. Regional languages

**CBS SSW 2019 (Statistische Trends 2021, table 2.1), language most spoken at home, 15+, by
province, one answer** (n = 7,652). CBS codes Stadsfries, Bildts and Veluws as "dialect" and
Stellingwerfs, Gronings, Drents, Twents, Achterhoeks as Nedersaksisch. Applied to each gemeente's
15+ (CBS 2026), children under 15 at Driessen 2012's child-to-mother ratio (ITS Nijmegen,
"Ontwikkelingen in het gebruik van Fries, streektalen en dialecten 1995-2011", 2011 wave:
Frisian 37 vs 39.6 adult, Limburgish 39 vs 47.9, Low Saxon 1 vs 26.8, Zeeuws 10 vs 29.6).

Labels, by Glottolog (as Italy drew its dialects):
- **Frisian** (Western Frisian): Fryslân's 39.6%, spread over gemeenten by **De Fryske Taalatlas
  2020** (Provinsje Fryslân; map 1.6, Frisian mother tongue, band midpoints: Dantumadiel 85 ...
  Harlingen, Weststellingwerf 25; relative weights, CBS's total). The four Wadden islands were not
  surveyed and are drawn Dutch. Groningen's 1.9% on Westerkwartier; Noord-Holland's 2.2% "Fries"
  read as West-Fries, a Hollandic dialect -> Dutch; small shares elsewhere by population.
- **Gronings** (a Glottolog language): Groningen's 25.5% "Nedersaksisch".
- **Westphalian** (Glottolog's Westphalic holds Drents, Twents, Sallands, Achterhoeks,
  Stellingwerfs, Veluws as dialects): the other provinces' Nedersaksisch; Gelderland's 10.2% on
  the Veluwe and Achterhoek COROPs only, plus its 0.9% "dialect" on the Veluwe (Veluws);
  Fryslân's 2.8% on the Stellingwerven.
- **Limburgish**: Limburg's 47.9%, by Veldeke / R&M Matrix 2021's fluent-speaker share by region
  as relative weight (North 60, Midden 74, West South 76, Parkstad 54); small shares elsewhere.
- **Zeeuws** (a Glottolog language): Zeeland's 29.6% "dialect", Zeeuws-Vlaanderen included.
- **Dutch**: Brabant's 25% dialect (Brabants is Dutch in Glottolog) and every other "dialect".

Result: Westphalian 627,907, Limburgish 571,837, Frisian 292,311 (39.3% of Fryslân), Gronings
134,105, Zeeuws 105,564.

## 4. Geography

`data/geo/nl/nl_hexes.gpkg`: Kontur 2023 hexes (religiondots' NL .gz, copied, read-only) keyed
to the 2026 gemeenten from PDOK (`gebiedsindelingen/2026`, 342, both directions match CBS) and
to postcode-4 areas (PDOK `postcode4/2024`; 2025/2026 not served). hex_layer: 2,495 hexes
(488,466 people) outside every gemeente dropped (Kontur's NL extract carries a slab of Belgium,
religiondots `countries/nl.py`); Kontur/CBS median 1.01, 0 of 342 outside a factor 3, log r 0.994
vs 0.167 shuffled. Each hex carries its PC4's CBS 85640NED counts (born in NL of Dutch origin,
born in NL, born abroad by 12 origin groups that partition the foreign-born), spread by Kontur
population; 99.9% of Kontur people sit in a PC4 the 2026 table has. Placement (§4.4): Dutch on
the NL-born, regional languages on NL-born of Dutch origin, each immigrant language on its
origin groups' foreign-born in the gemeente's mix (`nl_weights.csv`).

## 5. Checks and result

Gemeenten sum to CBS 2026 within 11.7 people (rows under 0.5 dropped). check_country ok; scatter
18,081 dots on 10,287 polygons, 55 rings (all derived), 46,984 people (0.26%) under one dot.
National: Dutch 81.0%, Westphalian 3.5%, Limburgish 3.2%, Frisian 1.6%, Arabic 1.3%, Turkish
1.2%, Polish 1.0%, Gronings 0.7%, Ukrainian 0.7%, Zeeuws 0.6%.

Corroboration, immigrant languages (all ages, 2026) vs SSW 2019 "other language" (15+): Zuid-
Holland 12.3 / 12.3, Noord-Holland 12.0 / 11.1, Flevoland 11.7 / 10.7, Gelderland 6.8 / 7.0,
Overijssel 7.2 / 7.8, Fryslân 4.8 / 4.6; high in Noord-Brabant 9.0 / 5.6, Limburg 8.8 / 5.1,
Utrecht 8.7 / 5.5, Drenthe 5.0 / 2.8 (Ukrainian and eastern European arrivals since 2019 explain
part; the borrowed Polish rate may be high for them).

## 6. Colours

New: Gronings (violet), Westphalian (dark blue), Zeeuws (green), Limburgish hand-set teal (it
was bare in fi/pl). **Frisian's colour is ca.txt's (0.80 0.10 200) and sits close to Dutch's
(0.78 0.12 225, cz/us)** on the one border that matters most here; not changed, because the
build refuses a second colour for a node and ca.txt is not this session's.

## 7. Room for improvement

- Schmeets & Cornips, Taal en Tongval 78 (2026) 26-44 (doi 10.5117/TET2026.1.002.SCHM) pools SSW
  2019-2024 (n > 22,000) with language maps, likely finer than province. Open access but
  aup-online.com returns 403 to scripts: fetch in a browser.
- Low Saxon and Limburgish are uniform within province (or region); cities speak less of both.
  Bloemhoff's Taaltelling Nedersaksisch (2005) has regional counts but is not online.
- Retention by origin is "with children often/always Dutch", looser than "mostly Dutch"; most
  origins take a common rate; Ukrainians borrow the Polish figure.
- Frisian gemeente weights are atlas bands (ranges), not exact values.
