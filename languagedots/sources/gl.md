# Greenland (gl): record

Drawn 2026-10-05 (session edd42a8c-mono5). No language question anywhere (register, no census).
Built under the 2026-10-05 ruling for countries with no language question (AGENT_BRIEF §2):
birthplace by locality from the population register, 1 January 2026, 56,740 people, 78 towns and
settlements, 125 nodes (most tiny), every row `derived`. Placed on religiondots' locality discs
(read-only), one unit per locality. 52 dots, 69 rings.

Files: `sources/gl_census.py`, `taxonomy/gl2026.py`, `taxonomy/tree.d/gl.txt` (origin_mix block
generated), `countries/gl.py`, `data/normalized/gl.csv`, `data/geo/gl/gl_places.gpkg`,
`data/raw/gl/bexst6_2026.csv`, `bexst6nuk_2026.csv`.

## Sources (Statistics Greenland Statbank, open PxWeb API, no key)

- BEXSTD: place of birth (Greenland / born outside) by locality. religiondots' download, read
  only, with its locality codes (`gl_localities_born_2026.csv`).
- BEXST6 (Greenland) and BEXST6NUK (Nuuk): citizenship, 2026. Danish citizenship covers
  Greenlanders, so the foreign citizens (2,829) are what it adds.

## The model

- **Greenland-born (49,721) -> Greenlandic by district**: Tunumiisut in Tasiilaq and
  Ittoqqortoormiit districts (2,733; Glottolog tunu1234, a separate language), Inuktun in
  Qaanaaq district (666; Glottolog's Polar Eskimo), Kalaallisut elsewhere (46,322). Locality
  code digits 4-5 give the district.
- **Born outside (7,019) -> foreign citizens on `origin_mix.mix(iso, "gl")`**, the rest (4,196:
  Danish citizens born in Denmark or elsewhere, plus 6 stateless/unknown) on Danish. Nuuk town
  uses Nuuk's own table; every other locality the national table minus Nuuk, spread by its
  born-outside count. Pooled "Other Europe/America/Asia", Oceania -> `other`; Other Africa ->
  `africa_other`.

## Checks (asserted)

Greenland + outside = Total for every locality; localities sum to 56,740; each citizenship table
sums to its total; Nuuk <= national per citizenship; foreign citizens <= born-outside in both
parts; every locality with people has a disc (none folded).

## Calls someone might reverse

- **Danish-dominant Greenland-born drawn as Greenlandic.** Common in Nuuk; no count exists.
- Greenland-born children of Danish parents likewise on Greenlandic.
- Tunumiisut and Inuktun as languages beside Kalaallisut (siblings, not children, so Kalaallisut
  keeps fi's leaf node). Everyone Greenland-born in those districts is put on the local variety,
  including West Greenlanders who moved there.
- Foreign citizens born in Greenland are counted inside the born-outside (no cross-table); small.

## Room for improvement

A language-use survey (none found) would size Danish among the Greenland-born.
