# Belarus (by): 2019 census, language usually spoken at home, by raion and city

Drawn 2026-10-05 (session d9e44929-by) on native language; **switched to home language the same
evening** on Anita's ruling (session edd42a8c-rulings, see "Question" below). Now 9,196,927
people on 117 languages, 129 units; 216,519 (2.30%) whose home language was not stated, not
drawn.

## 1. The table

**Belstat's census database**, https://census.belstat.gov.by/ (an anonymous React app over a
Saiku OLAP server). The published bulletins on belstat.gov.by stop at the oblast; the database
is the only place language is published below it. Of its cubes only **F503 / `tb503`**,
"population by nationality, native language and language usually spoken at home", has a
language dimension at level 3 (`tb034`: raions, cities of oblast subordination, Minsk's nine
districts). Indicators `rodn_kol_19` (native language, 2019) and `razg_kol_19` (language usually
spoken at home, 2019); the 2009 pair is there too and not used. Other cubes with language
(F506, F509) are oblast-only and have other universes (F509 is the population by education
level, aged 10 and over by the look of its totals).

**The route.** The app's choropleth endpoint, found in the JS bundle (`static/js/0.*.chunk.js`
embeds every cube's table, indicator and dimension keys):

    POST https://census.belstat.gov.by/sdpn/rest/api/map/l3
    {"cubeKeys": {"table": "tb503", "indicatorKeys": ["rodn_kol_19", "razg_kol_19"],
                  "dimensions": [{"table": "tb016", "tablePK": "iazyk_id", "value": ["29"]}]}}

returns one value per level-3 unit, summed over every dimension not named. One POST per value of
`tb016` (171 languages), one with no filter for unit totals, one at `/map/l2` for oblasts. Code
lists: `POST /sdpn/rest/nsi/simple?code=tb016` and `?code=tb034` (with SOATO codes). No login,
no key. The host's TLS certificate chains to Belarus's national root CA, which Python does not
trust; verification is off for this host only (public data, nothing sent).
`sources/by_census.py --fetch` does all of it in about two minutes.

**Question.** The 2019 form asks native language (родной язык) and the language usually spoken
at home. Both are kept in `data/normalized/by.csv` (`question` = native / home); **home language is
drawn**. Native was drawn first (spec §1: mother tongue over home language; AGENT_BRIEF §2 names
Belarus as the case for native language with an identity caveat). Anita ruled on 2026-10-05 to
switch: "it does seem like Russian majority is reality". Native language ("родной язык") in
Belarus and the other ex-Soviet censuses leans towards ethnic identity: 54.1% Belarusian and
42.3% Russian native, against 26.0% Belarusian and 71.4% Russian at home. The home figures are
the language people actually use, which is what the map shows elsewhere. `note_public` gives both
pairs and the city gap (Brest city 71.4% Belarusian native, 5.9% at home; Vitebsk city 37.7% and
2.0%, re-read from `by.csv`). The native-language figures and checks below are kept as the record
of the other question; every check covers both.

**Second source for the check.** Belstat's statistical bulletin on the 2019 census, "Общая
численность населения, численность населения по возрасту и полу, ... национальностям, языку ...
по Республике Беларусь" (https://www.belstat.gov.by/upload/iblock/471/471b4693ab545e3c40d206338ff4ec9e.pdf),
p.44: population, native Belarusian, native Russian, Belarusian at home, Russian at home, for
the country and each oblast and Minsk. Transcribed in `by_census.py` and re-read from the PDF's
own text on every run.

## 2. Checks (all pass, `python sources/by_census.py`)

- National total 9,413,446 for both questions; every one of the 137 units' languages sums to its
  own total, for both questions.
- `tb034` lists 142 units; exactly five are empty in 2019, asserted by name: Polotsk city and
  the Vitebsk, Novopolotsk, Orsha and Gomel city councils (2009-era units). 137 left: 118
  raions, 10 cities, 9 Minsk districts.
- The units summed per oblast equal the bulletin's p.44 in all 35 figures (seven units x
  population, native Belarusian, native Russian, home Belarusian, home Russian), to the person;
  the national five too (9,413,446; 5,094,928; 3,983,765; 2,447,764; 6,718,557).
- The cube's own oblast endpoint gives the bulletin's seven populations.

National, native | at home: Belarusian 5,094,928 (54.12%) | 2,447,764 (26.00%); Russian
3,983,765 (42.32%) | 6,718,557 (71.37%); not stated 221,588 (2.35%) | 216,519; Ukrainian 55,020
| 8,055; Polish 20,740 | 4,508; Armenian 5,348; Romani 5,067; Turkmen 4,986; Azerbaijani 4,532.
138 labels carry someone as native language, 118 at home. "Two or more languages" and "language of
one's own nationality" are on the code list and hold nobody.

Shape of the data, for the record: Belarusian native ranges 23% (Dobrush raion, Gomel) to 91%
across raions, median 62%. City and its surrounding raion are close (Brest 71% / 70%, Vitebsk 38%
/ 40%, Gomel 41% / 42%, Grodno 43% / 46%); Minsk's nine districts 46.7-49.8%. Not stated is
highest around Mogilev and Bobruisk (6-9%). Polish peaks at 4.6% native (Grodno raion), though
ethnic Poles are a fifth of Grodno oblast: of the country's 287,693 Poles (bulletin p.37) 19,141
named Polish, 156,650 Belarusian and 110,727 Russian.

## 3. Geography (`sources/by_geo.py`)

129 polygons, all OpenStreetMap relations via Nominatim:

- **118 raions**: one Nominatim search per census raion name ("Барановичский район"), the one
  admin_level-6 boundary (place_rank 12) whose `address.state` is the census oblast. Every search
  had exactly one. OSM's raion polygons already exclude the cities of oblast subordination and
  Minsk, as the census does.
- **10 cities**: OSM city relations (Brest 72615, Baranovichi 3629362, Pinsk 1749248, Vitebsk
  6825777, Novopolotsk 6825778, Gomel 163244, Grodno 130921, Zhodino 79911, Mogilev 62145,
  Bobruisk 167857). Polotsk city has no 2019 row; OSM's Polotsk raion holds it.
- **Minsk**: OSM relation 59195 (religiondots' copy), 353 km2, unit id `BY005` so religiondots'
  kontur_cap.csv row (Frunzensky, `real`) applies. The nine districts are one unit; their mixes
  are within 3 points of each other.

**COD-AB was tried first and dropped.** COD admin 2 (religiondots' raw copy) is a coarse
drawing: same-named raion IoU against OSM median 0.70, lowest 0.31 (Beshenkovichi); its
Orsha/Dubrovno line runs through Orsha city, so Dubrovno raion read 2.83x its census people in
Kontur and two thirds of its dots would have sat in Orsha's suburbs; its Kirovsk line takes half
of Bobruisk city; it draws each city inside its raion. Kontur/census fit per unit: COD p10 0.71,
p90 1.48, log r 0.951; OSM p10 0.83, p90 1.30, log r 0.978 (shuffles best 0.274), no unit outside
a factor of 3.

**Witness for the OSM pick, which neither name decides**: COD is name-joined to the census
independently (translit prefix match within the oblast, unique, 1:1, three pinned aliases) and
for all 118 raions the COD raion holding most of the OSM polygon is that same raion (asserted).

Other numbers: units overlap 0.0 km2; Belarus outside every unit about 980 km2 of border
slivers (OSM's and COD's national lines differ), 575 hexes, 9,011 Kontur people, not placed on.
Kontur/census highest: Mogilev raion 2.05, Bobruisk raion 2.04, Smolevichi raion 2.03, Zhodino
1.85, Vitebsk raion 1.77 (suburban raions round the cities; Kontur dense there); lowest Pinsk
city 0.67. Zhodino is allowed to 2.0 by name: its top hexes are on the town's own blocks.

Placement is plain Kontur population inside each unit (AGENT_BRIEF §4.4, the last fallback);
nothing finer by origin was looked for, the minority languages being under 1%.

## 4. Mapping (`taxonomy/by2019.py`, `taxonomy/tree.d/by.txt`)

Belstat's code list is the Soviet-lineage list Russia's census uses, so any label Russia's 2021
table prints maps through `taxonomy/ru2021.py`'s NAMES (capitalised). Labels it lacks are in
`by2019.NAMES` with the calls in the module docstring: Afghan to `other.afghan`; "Australian" to
a new leaf `other.australian` (a family name, no language named; the `australian` root would
draw it as an Australian Indigenous remainder over Belarus); Gaelic to Scottish Gaelic (Irish is
listed apart); Orok to Uilta; the Pamir languages flat under Iranian; Chuvan under Yukaghir;
Baraba Tatar and Nagaybak under Turkic; Svan, Bats, Homshetsi, Livonian, Taz. "Другой язык"
(542) on `other`; not stated is the gap. Tiny odd labels (Aleut 45, Koryak 11) are drawn as
coded.

## 5. Colour

Belarusian and Russian are every raion's two languages, and generated they were two greens of
the same lightness (#00bc99, #54b85b). Belarusian is hand-picked in `tree.d/by.txt` to OKLCH
0.56 0.11 185 (#00897d), a darker teal. That recolours Belarusian everywhere; checked: in
Lithuania (sources/lt.md flagged Russian, Polish and Belarusian as three greens) it now differs
from both by lightness; in Poland's Hajnowka the hand-picked Podlachian (0.72 0.11 215) stays
apart by lightness. Polish (#00a76c) and the new Belarusian are both teal-green, 0.07 apart in
lightness; Polish is under 5% in every raion here.

## 6. Calls someone might reverse

- **Home language drawn, not native language** (Anita's ruling, 2026-10-05). Native, Belarus is
  54% Belarusian and the map would look entirely different. The normalized CSV carries both;
  flipping back is one filter in `countries/by.py::_counts`.
- Minsk as one unit rather than nine districts.
- Belarusian's colour changed for every country.
