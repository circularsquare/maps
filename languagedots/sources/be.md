# Belgium (be): the record

Drawn 2026-10-05 by session edd42a8c-be. 11,554,767 people (census 2021), 44 arrondissements
(NUTS 3; Verviers is two, its nine German-speaking communes being BE336), placed on 581
communes; 122 languages, 11,508 dots. Every row is `derived`.

Build: `python sources/be_census.py --fetch` (Eurostat JSON into `data/raw/be/`, then builds
`data/normalized/be.csv` and `be_communes.csv`), `python sources/be_geo.py` (placement layer).
Mapping `taxonomy/be2021.py` (France's labels, imported from `fr2023`), no new tree nodes, entry
`countries/be.py`. Survey tables as images in `data/raw/be/brio_*`.

## 1. The rule

Belgium asked language in its censuses up to 1947 (published 1954); the 1961 law abolished the
question and the 1962-63 laws froze the language areas. Nothing official has been measured
since. Anita's 2026-10-05 ruling for rich countries with no language question (AGENT_BRIEF §2):
national language by area, plus surveys, plus immigrant languages by origin, all `derived`.

## 2. Immigrants: Eurostat census 2021, country of birth by NUTS 3

`cens_21cob_r3` (Eurostat API, no key), sex T, age TOTAL, the 44 Belgian NUTS 3. Census 2021 is
Statbel's register-based census. Checks (printed; the first two stop the build): TOTAL = NAT +
FOR + UNK in every arrondissement; FOR minus the 225 named countries equals EUR_OTH exactly
(113,997, "other European countries", presumably people born in the USSR, Yugoslavia or
Czechoslovakia before the split), spread over the arrondissement's births in the successor
states; LAU 2021 commune populations agree with the census total within 0.27% per
arrondissement. Born abroad: 2,065,727; unknown 4,958 (drawn on the area language).

Country of birth, not nationality: it catches naturalised immigrants (most Moroccan and Turkish
Belgians), and puts the Belgian-born second generation on the area language, as France did.
Its weakness: Belgians born abroad (colonial-era Congo, France, the Netherlands) count as
immigrants of that country. Statbel publishes commune figures only as Belgian/foreign
nationality; its site sits behind an F5 bot wall (religiondots `sources/be.md` §1a) and was not
used. So immigrant languages are spread over an arrondissement's communes by population, the
brief's last fallback (§4.4). In Brussels that puts Moroccan Arabic as much in Uccle as in
Molenbeek proportionally; see §7.

Country -> language: France's `COUNTRY_LANG` (`sources/fr_build.py`; `sources/fr.md` §2 gives
the calls), plus France -> French and a few territories (English, French; Faroe and Greenland
on Danish, a handful of people). Calls worth reversing for Belgium specifically:
- **Morocco** uses Morocco's 2024 census mix (Darija 78%, Tachelhit 12%, Tamazight 6%, Tarifit
  3%). Belgian Moroccans come largely from the Rif (60-80% is the figure cited for the
  Netherlands; nothing Belgian with a source was found), so Tarifit is far too low and Darija
  too high. Kept for consistency with France, said in note_public.
- **DR Congo -> Lingala** (Congolese Brussels is Lingala and French; the TeO2 Central Africa
  retention moves 72% of them to the area language anyway). Rwanda, Burundi as France.
- **Turkey -> Turkish**; Belgian Turks are largely from Emirdağ and Afyon, few Kurds counted.

Superseded since (as the entry's note_public said before the 2026-10-06 text sweep moved it
here): people born in Morocco are drawn 40% Tarifit, since over 40% of Belgium's Moroccans grew
up in the Rif (Reniers, 1999); people born in Turkey 6% Kurdish; people born in DR Congo in the
Congo's own shares, so Lingala, the language of Kinshasa, is probably undercounted. Brussels'
BRIO 2024 answers: 41% French only, 8% Dutch only, 4% both, 18% French with another language,
29% neither (two home languages counted half to each). Regional languages (West Flemish,
Limburgish, Walloon, Picard, Luxembourgish round Arlon) are drawn as Dutch, French or German.

### 2b. Retention: no Belgian survey, France's TeO2

No Belgian survey gives home-language retention by origin (BRIO measures Brussels as a whole;
the Flemish integration monitors give pupils' home language as Dutch / not Dutch). France's
TeO2 shares (`sources/fr.md` §2b, the share who speak only French with their children by
region of origin) are applied as "speak only the national language", and the moved share goes
to the commune's area language. In Flanders that assumes immigrants shift to Dutch as they shift
to French in France; many shift to French instead. Not applied in Brussels and the Rand (the
surveys measure them directly). Moved: 400,848, largest Lingala 42,972, Moroccan Arabic 40,672,
Italian 33,535, Arabic 26,618, Romanian 21,521.

## 3. Language areas and the border communes

Dutch for BE2 (Flanders), French for BE3, German for BE336 (exactly the nine communes:
Eupen, Kelmis, Lontzen, Raeren, Amel, Büllingen, Burg-Reuland, Bütgenbach, Sankt Vith).
Malmedy and Waimes (French area with German facilities) are French. Regional languages (West
Flemish, Limburgish, Brabantian, Walloon, Picard, Gaumais, Luxembourgish round Arlon, Platdiets
in Welkenraedt-Plombières) are not drawn: no survey counts them as first languages.

Border communes with a printed figure (applied to the Belgian-born and to the folded immigrants):

| commune | minority | share | source |
|---|---|---|---|
| Voeren (73109) | French | 40% | "about 40% francophone, 60% Dutch-speaking according to electoral lists" (CEFAN, Université Laval, *La commune des Fourons*). 1947 gave French majorities in five of six villages (Encyclopedie Vlaamse Beweging); one search summary claimed 20% from a 2011 integration monitor, not traced, not used |
| Comines-Warneton (57097) | Dutch | 7.5% | 7-8% of identity cards issued in Dutch, October 2012 (nl.wikipedia Komen-Waasten); 1947 language mostly spoken 14.3% Dutch |
| Mouscron (57096) | Dutch | 12.06% | 1947 language mostly spoken 23.0% Dutch (nl.wikipedia Moeskroen) x Comines-Warneton's 2012/1947 ratio (7.5/14.3); no later figure |

No figure found for Ronse, Spiere-Helkijn, Mesen, Bever, Herstappe, Enghien, Flobecq; they are
drawn on their area language. The 1947 per-commune table was not found online (nl.wikipedia
Talentelling has 1930/1933 per commune, 1947 only in aggregate).

## 4. Brussels and the Rand: BRIO

**Brussels-Capital (BE100, 19 communes)**: BRIO Taalbarometer 5 (2024, 1,627 adults), table 3,
"oorspronkelijke thuistaal" (language(s) spoken at home growing up): French 41.3, Dutch 7.5,
Dutch/French 4.3, French/other 18.0, other 28.8 (`data/raw/be/brio_tb5_tabel3.png`, from
briobrussel.be/node/19094). Pairs half each: French 52.5%, Dutch 9.66%, other 37.84%. Applied
to everyone, children included. TB4 (2018) gave 52.2 / 5.6 / 10.7 / 10.1 / 21.4, so French alone
fell 11 points in six years with a smaller sample; TB5 is used as the newest. "Other" is split
over the languages of Brussels' foreign-born (France- and Netherlands-born and francophone
origins left out), in proportion.

**Vlaamse Rand (19 communes, BE241 and Tervuren in BE242)**: BRIO Taalbarometer Rand 2
(Janssens, 2019; briobrussel.be/node/14829). Table 1, original home language, all 19: Dutch
45.0, Dutch/French 10.2, Dutch/other 0.7, French 20.4, French/other 6.8, other 17.0. Figure 1
gives *current* home language by commune cluster, read off the bars (no numbers printed, about
half a point): facility cluster Dutch 18.5, Dutch/French 14.6, Dutch/other 1.2, French 47.2,
French/other 10.5, other 7.8 -> French 59.9%, Dutch 26.5%, other 13.7%.
- The six facility communes take the cluster figure, varied between them by Le Soir's 2005
  French shares (Drogenbos 55, Kraainem 78, Linkebeek 79, Sint-Genesius-Rode 58, Wemmel 54,
  Wezembeek-Oppem 72; as cited by en.wikipedia "Municipalities of Belgium with language
  facilities"), scaled x0.926 so the population-weighted mean is BRIO's. Result French 50-73%.
- The other 13 take the residual that makes all 19 match table 1: French 22.7%, Dutch 55.2%,
  other 22.1%. (Mixes current and original home language; the cluster figure is current.)
- Other split by the arrondissement's foreign-born, as Brussels.

## 5. Geography

`data/geo/be/be_place.gpkg` (`sources/be_geo.py`): religiondots' `be_lau.gpkg` (GISCO LAU 2021,
581 communes, 2021 population), read-only, re-keyed from NUTS 2 to NUTS 3 by GISCO's
correspondence workbook; 581 joined both ways, 44 units. Placement: each language on the
communes' own figures from `be_communes.csv` (so Dutch never lands in a Walloon commune except
as immigrants, German stays in BE336, the facility shares land in their communes).

## 6. Checks and result

check_country ok. Communes sum to their census population (to 0.5 person); be.csv sums to the
census total. Scatter: 11,508 dots on 580 polygons, 44 rings, 46,767 people (0.40%) under one
dot per language. National: Dutch 52.7%, French 35.6%, German 1.2%, Moroccan Arabic 1.2%,
Romanian 0.8%, Arabic 0.8%. Flanders: Dutch 89.3%, French 2.4%. Wallonia: French 90.3%, German
2.5%. Brussels: French 52.5%, Dutch 9.7%, Moroccan Arabic 5.9%. BE336: German 91.8%.

Colours: all nodes exist (France and Germany); no hand-set colours.

## 7. Room for improvement

- Commune-level origin (Statbel publishes nationality by commune and statistical sector; walled)
  would place Brussels' immigrant languages where they live (Molenbeek, Schaerbeek,
  Anderlecht, Saint-Josse) instead of by population across all 19 communes.
- A Rif share for Belgian Moroccans with a source; Kurdish among Turkey-born.
- A Belgian retention survey, or Flanders' split of non-Dutch home language into French and
  other (Kind en Gezin's language spoken with babies, by commune, would measure Flanders directly).
- Modern figures for Ronse, Enghien and the other border facility communes; the 1947 per-commune
  table.
