# Kosovo — ASK, Census 2024, mother tongue

Drawn 2026-10-05 (session edd42a8c-xk). 1,602,515 people in 6 categories on the 38
municipalities, placed inside each municipality by the 2011 census's ethnicity of each of its
settlements (the four northern municipalities on population). 1,599 dots at 1:1000; 16,806 people
in the north are `tier="derived"` (ASK's own estimate, below).

Files: `sources/xk_census.py` (fetch + normalise), `sources/xk_geo.py` (settlements and the
placement layer), `taxonomy/xk2024.py`, `countries/xk.py`. No tree fragment: every language was
already a node, and the colours already differ (checked for North Macedonia: Albanian brown,
Serbian teal, Bosnian lime, Turkish pink, Romani purple, all over 0.12 OKLab apart).
Data: `data/raw/xk/` (three PxWeb cubes, `settlements/censusetn<code>.json` for 35
municipalities, `osm_places.csv`), `data/normalized/xk.csv`, `data/normalized/xk_north.csv`,
`data/geo/xk/xk_hexes.gpkg`, `data/geo/xk/xk_settlements.csv`. Read only from religiondots:
`data/geo/xk/xk_municipalities.gpkg` (geoBoundaries ADM2, `unit` = ASK code) and
`xk_grid_400m.gpkg` (Kontur 400 m hexes with `unit`, `pop`).

## 1. The tables

ASK's PxWeb, `https://askdata.rks-gov.net/api/v1/en/ASKdata/Census population/`, open, no key.
Requests are slow (about 45 s each), and the root answers `dbid` rather than `id`.

* **Counts drawn**: `1_Demographic_Characteristics/census2024_22.px`, mother tongue x sex x
  country + 38 municipalities, 2011 and 2024. Categories: Albanian, Serb (Serbian), Bosnian,
  Turkish, Romani, Other (specify), Not available (0 everywhere in 2024). The national table by
  age (`census2024_56`) and the final-results PDF (Tab. 3.7, 4.6;
  `askapi.rks-gov.net/Custom/bffbac3c-...pdf`, the queue's lead) have the same six categories;
  nothing finer is published. ASK's definition: "the first language a person has learned from
  birth and is still able to speak". One answer.
* **Ethnicity**: `census2024_05.px` (enumerated) and `census2024_63.px` (with estimation), the
  pair religiondots reads.
* **Placement**: `6_Sipas vendbanimeve/<code> Popullsia e komunës .../censusetn<code>.px`,
  ethnicity by settlement, **2011 only** (one table per municipality; Deçan's is
  `PopEtnicititeti03.px`; North Mitrovica, created 2013, has none). The 2024 settlement tables
  give population by sex and age only. So, unlike North Macedonia, the settlement mix is 13 years
  older than the counts.

## 2. Checks, with numbers

`python sources/xk_census.py`:
* national total 1,602,515; the 7 categories partition every municipality; the 38 municipalities
  sum to the national row in every category;
* **the mother-tongue table already includes ASK's estimate for the north.** Its municipal totals
  equal the *estimated* ethnicity table's in all 38, and differ from the enumerated one only in
  Leposaviq, Zubin Potok, Zveçan and North Mitrovica (national enumerated 1,585,566). The
  estimate adds 16,949 people there, 16,369 Serbs (96.6%). In Leposaviq, Zubin Potok and Zveçan
  the Serbian mother-tongue count equals the estimated Serb count exactly;
* nationally, mother tongue against ethnicity: Albanian 1,485,170 against 1,454,999 Albanians +
  16,207 Ashkali + 10,581 Egyptians (1,481,787); Serbian 55,168 / 53,021 Serbs; Bosnian 28,854 /
  27,152 Bosniaks; Turkish 18,748 / 19,419 Turks; Romani 5,597 / 8,730 Roma; Gorani 9,140.

`python sources/xk_geo.py`:
* the 38 polygons' ASK codes are the mother-tongue table's (names agree after stemming, except
  Prishtinë / Pristina);
* each settlement table ends with a municipality-total row in capitals (DEÇAN, PRIZREN), equal to
  the sum of the rest in every group; dropped (34 rows; Mamushë's single settlement handled);
* the 1,468 settlements sum to the 2011 municipal ethnicity table (`census2024_05`, year 2011) in
  every group in every municipality, once the settlement tables' "Not available" is folded into
  "Prefers not to answer" as the municipal table does. Gjilan's table prints "Albanian," with a
  comma; stripped;
* Leposaviq, Zubin Potok and Zveçan tables are all blank for 2011 (not enumerated), and North
  Mitrovica has none. Those four are placed on Kontur population alone;
* settlement points: OSM place nodes, place areas and admin-level 8-10 boundaries from Geofabrik's
  kosovo-latest.osm.pbf (read 2026-10-05 with pyosmium, then deleted; 2,526 rows kept in
  `osm_places.csv`). Overpass was returning 504 that day. 1,198 of 1,297 settlements matched by
  folded name (`name:sq`, `name`, `name:sr-Latn`, `name:sr`) inside the municipality buffered
  1.5 km, with "upper/lower" spellings unified, a final vowel dropped, and j/y read as i; two
  aliases (Gllogoc town = Drenas; Nëntë Jugoviq = Bardhosh). 98 settlements (79,133 people in
  2011, 4.55%) unplaced, mostly villages OSM does not carry at all (Pozharan 4,247, Stanoc i
  Epërm, Gjylekar, Hodonoc). Their area goes to the nearest placed settlement, so they blur the
  mix rather than lose people. Lowest share placed: Novobërdë 45% (Serb villages missing from
  OSM), Viti 77%, Obiliq 83%;
* the Kontur join itself is religiondots' (Kontur against census: median 1.06 on the 34
  enumerated municipalities; the north far above, the boycott, and North Mitrovica's north-bank
  hexes partly in the southern municipality because geoBoundaries splits the city on the Ibar).

Spot check on the dots (settlement of the hex each dot falls in): median Albanian dot in a
settlement 99.8% Albanian/Ashkali/Egyptian; Serbian 70.5% Serb/Gorani; Bosnian 94.9%. Turkish
10% and Romani 7%, because Turks and Roma live in quarters of Prizren, Gjilan and the other
towns, each one settlement.

## 3. Calls

* **The north is drawn as ASK published it, with the estimated part marked derived.** No
  enumerated-only mother-tongue table exists, so the estimated share is taken from the ethnicity
  pair: in each northern municipality, the people ASK added in an ethnicity move to `derived` in
  the matching language (Serb -> Serbian, Bosniak -> Bosnian, Albanian, Turk -> Turkish, Others
  -> other), capped at that language's count: 16,806 people, 16,369 Serbian. The 49 added who
  preferred not to give ethnicity stay measured. This follows Anita's ruling for the same census
  in religiondots (2026-09-06: draw the north from ASK's estimate, tier derived), and here no
  inference of mine is needed, since ASK put the estimate into the language table itself. No ask.
* **Serb -> `serbian`.** The PxWeb English label is "Serb"; the national table and the PDF say
  "Serbian".
* **Other (specify) -> `other`.** No breakdown is published. 5,980 of the 8,978 are in Dragash,
  where 7,828 declared Gorani: mostly the Gorani's own speech (nashinski), but nothing says how
  much, so it is not guessed onto a node. No indigenous remainder to keep apart.
* **Placement by 2011 settlement ethnicity** (brief §4.4: moves people only inside the
  municipality the census counted them in). Albanian follows Albanians + Ashkali + Egyptians (the
  national pair agrees within 0.2%); Serbian follows Serbs + Gorani, Bosnian Bosniaks + Gorani,
  other Others + Gorani (Dragash's Gorani answered all three: 1,676 Serbian, 3,674 Bosnian, 5,980
  other against 17 Serbs and 2,900 Bosniaks); Turkish follows Turks, Romani Roma. A hex's weight
  is its Kontur population times its settlement's share of those groups. 65 (municipality,
  language) rows placed by settlement mix, 9 on population. The 2011 mix is older than the counts;
  the big movements since are within towns (one settlement each), so village-level mixes changed
  little, but `how` and `note_public` say 2011.
* **Gap**: none. "Not available" is 0, and the table has no refusals.

## 4. Not done

* 2011 mother tongue (in the same cube) is not drawn; the spec has no time slider.
* No finer split of `other`, and Gorani is not a node: no table names it as a language.

## Moved from countries/xk.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- The statistics agency's language table includes its own estimate for them, about 17,000 people, nearly all Serbian speakers; those dots are an estimate, not answers, and the confidence control removes them.
