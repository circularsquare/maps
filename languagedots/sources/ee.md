# Estonia: Rahvaloendus 2021 (REL2021), mother tongue

Built 2026-10-05 (session edd42a8c-ee). Rebuild:

```
python sources/ee_census.py [--fetch]   -> data/normalized/ee.csv
python taxonomy/build.py
python sources/ee_geo.py                -> data/geo/ee/ee_units.gpkg, ee_hexes.gpkg
python tools/check_country.py ee
python scatter.py --country ee
```

Drawn: 1,323,649 people on 127 parts (45 whole municipalities, 37 towns, 32 municipality
remainders, 13 city districts), 16 labels on 15 nodes, 1,316 dots, 2 rings. Not drawn: mother
tongue unknown, 8,176 (0.61%); 24 people in Kohtla-Järve placed in no district (section 3).
Everything is `measured`. No new tree nodes, no asks.

## 1. The table

Statistics Estonia's PxWeb API, no key, no login (`verify=False`, as religiondots' `ee.py`):
`https://andmed.stat.ee/api/v1/en/stat/rahvaloendus/rel2021/rahvastiku-demograafilised-ja-etno-kultuurilised-naitajad/rahvus-emakeel/`.

| table | content | use |
|---|---|---|
| `RL21434` | mother tongue x sex x age group x place (administrative unit), 17 categories | drawn (both sexes, all ages) |
| `RL21431` | mother tongue x sex x settlement region, 245 languages, 5 places | national check; what "Other" holds |
| `RL214312` | first x second mother tongue, national | how many gave two |

**Question**: emakeel, mother tongue, for the whole population (the religion question in
religiondots was 15+; this one is not). REL2021 was register-based. A person could give two
mother tongues: 30,710 did (RL214312, figures rounded to 10), the largest pairs Estonian + Russian
18,160, Russian + Ukrainian 3,810, Estonian + English 1,220, Russian + Belarusian 1,110. RL21434
counts each person once; the national row equals RL21431's exactly, so it is the first mother
tongue. Drawn as the first; `note_public` says so. Vintage: 31 December 2021.

**Places.** RL21434 goes below the municipality. Each municipality with a town is published whole
and also as "X city as a settlement unit" plus "municipality, excl. X" (a bare serial code such
as `4`); Tallinn has its 8 linnaosad and Kohtla-Järve its 5. The drawn cover takes a
municipality's parts where it has any, else the municipality: 127 parts. This matters most in
the mixed areas: Narva-Jõesuu town is 89% Russian, and Paldiski, Kehra and Tapa towns hold
2.3-2.7% Ukrainian, among the highest shares in the country.

## 2. Checks (`sources/ee_census.py`; all pass)

- Unit counts 1, 15, 79; national total 1,331,824.
- **The cube carries small disclosure noise.** Below the national row, the 16 categories miss a
  place's total by -7 to +9 (52 of 177 places exact), and a parent misses the sum of its
  children by -5 to +6 in 219 of 850 (parent, category) cells. Nothing documents the method in
  the table metadata; it looks like the random perturbation offices apply to register-census
  tables, the same in kind as religiondots' base-10 rounding of Estonia's religion table. Drawn
  as published, checked against a bar of 10. `note_public` mentions it in plain words.
- **Kohtla-Järve is 24 more than its five districts** (33,499 against 33,475: Russian 19,
  Estonian 5), the one gap above the noise: people the register puts in the city and in no
  district. Drawing the districts leaves them out (0.002%); asserted exactly.
- The 127 parts sum to the country less those 24.
- The national row against RL21431's "Whole country", its 245 languages folded into RL21434's
  list: all 17 categories exact.

## 3. Mapping (`taxonomy/ee2021.py`; no tree fragment)

Estonian, Russian, Ukrainian, Finnish, English, Latvian, German, Belarusian, Spanish,
Lithuanian, French, Azerbaijani, Armenian, Tatar: existing leaves. `Other mother tongue` (15,848,
1.19%) on `other`: RL21431 shows it is 229 languages, largest Italian 1,048, Swedish 818,
Turkish 742, Portuguese 732, Polish 693, Georgian 692, Hindi 630, Arabic 628, then Romani 457,
Estonian Sign Language 444, Russian Sign Language 281, Karelian 44, Ingrian 5, Votic 3,
Livonian 1. Indigenous and foreign answers are mixed in the drawn table, so `other` rather than
a family node. Võro and Seto have no category: the census counts them under Estonian and asks
about them only in the dialect-knowledge tables (`voorkeeleoskus-murded`), which measure ability,
not mother tongue. Not used.

## 4. Geography (`sources/ee_geo.py`)

Polygons from Maa-amet's municipality and settlement-unit shapefiles (EHAK, 2024-12-01), read
in place from religiondots' `data/geo/ee/` (read only; its `ee_geo.md` describes the files and
the dud `linnaosa_shp.zip`). religiondots' own `ee_finest.gpkg` (86 polygons, no population)
was not reused because the language table is finer.

- Municipalities: `code[4:8]` = OKOOD. Four codes changed between 2021 and the file (0142->0145
  Antsla, 0514->0515 Narva-Jõesuu, 0735->0736 Sillamäe, 0855->0857 Valga); re-joined by name with
  one candidate per side, asserted as that exact set, and three of the four witnessed by their
  town's settlement unit lying in the new code.
- Towns and districts: `code[8:12]` = AKOOD, asserted inside its municipality. One settlement
  code changed: Paide 5860 -> 5861, re-joined by name within the municipality.
- Remainders: the municipality less its towns. Each split municipality's parts tile it (area
  miss 0.000%).
- Placement: Kontur EE (2023-11-01, downloaded into languagedots' `data/geo/kontur/`), 400 m
  hexes by centroid via `_grid.hex_layer`, weighted by Kontur population. Smallest parts: Oru
  district 1.2 km² (2 hexes), Kukruse 1.5 km² (1 hex); median part 115 km². Every part has
  populated hexes.
- **Kontur is badly wrong between towns here, and it does not move any dot between parts.**
  Kontur puts 900,302 people in Tallinn (census 437,817; 2.09x after the national ratio) and
  9,089 in Tartu town (95,190; 0.10x); Narva, Sillamäe, Rakvere, Viljandi, Pärnu and Keila read
  0.11-0.17; 37 of 127 parts are outside a factor of 3. The shuffle control passes (log r 0.810
  against a best of 0.367). The census fixes each part's count; Kontur only spreads a part's dots
  over its own hexes, where its footprint is what matters. Accepted, not calibrated. 962 hexes
  (15,127 people) have centroids outside every unit (sea or over the border) and are dropped.
- No Kontur cap block stopped the scatter.

## 5. What the map shows

Russian is Narva 95.6%, Sillamäe 95.3%, Kohtla-Järve's Ahtme 90.3%, Narva-Jõesuu town 88.8%;
in Tallinn, Lasnamäe 80,980 of 115,038. Ukrainian peaks at 2-4% in Loksa, Kehra, Paldiski, Tapa
and Maardu. Finnish is 1.7% of Tallinn's Kesklinn. "Other" is highest in Kesklinn (5.3%) and
Tartu town (2.1%).

**Colour.** Estonian and Russian are in different families and read apart; Finnish is the other Uralic
leaf and so near Estonian in hue, but it is under 2% everywhere. Not changed, not looked at in
the viewer.

## 6. Second sources

None needed: the question covers everyone. The 2011 census (on the same API) would corroborate at
municipality level; not looked for.
