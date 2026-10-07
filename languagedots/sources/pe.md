# Peru: Censo 2017, mother tongue, district

**Drawn.** 27,715,996 people on 1,873 district polygons (1,874 census districts), 44 nodes,
27,698 dots at 1:1000, 20 rings.

## Source

- INEI, Censos Nacionales 2017: XII de Población, VII de Vivienda y III de Comunidades Indígenas.
  Variable `Poblacio.C5P11`, "P3a+: Idioma o lengua con el que aprendió hablar": the language
  learned in childhood, asked of everyone aged 3 and over. That is mother tongue in the brief's
  sense; `how` says "mother tongue".
- Route: INEI's own REDATAM webserver, `https://censos2017.inei.gob.pe/bininei/`, base
  `CPV2017DI`, the same open, unauthenticated route religiondots uses for Peru's religion
  question (`../religiondots/sources/pe.py`). No login, no terms gate. Generic browser UA.
- `sources/pe_censo.py --fetch` runs five queries (AREALIST at department, province and district;
  FREQUENCY with AREABREAK at district for names and a second rendering of the counts; a national
  FREQUENCY restricted to age 5+ for the published check). Raw output in `data/raw/pe/`.
- The coverage sweep's lead was right on table and level; it guessed "~15" categories. The
  variable holds 46: the form's precoded boxes (8 native languages, Castellano, Portugués, other
  foreign, Peruvian Sign Language, "no escucha, ni habla", "otra lengua nativa"), 31 more native
  languages that INEI coded from the write-ins, and "no sabe / no responde".
- Peru's 2025 census was not checked for a language release; 2017 is the vintage.

## Checks (all pass, printed by the script)

| check | result |
|---|---|
| units | 25 departments, 196 provinces, 1,874 districts |
| national universe, aged 3+ | 27,946,060 (census population 29,381,884; 1,435,824 under 3 not asked) |
| 46 categories sum to the row's Total | all 1,874 districts |
| districts rebuild provinces / provinces rebuild departments | 9,212 and 1,175 cells, 0 failures |
| AREABREAK (rows paired by label) vs AREALIST (columns paired by position) | 86,204 cells, 0 failures |
| INEI's published national figures, aged 5+ (Perú: Perfil sociodemográfico, Censo 2017) | 5 of 5 exact |

The published 5+ figures are Castellano 22,209,686, Quechua 3,735,682, Aimara 444,389, otra
lengua nativa 210,017, otro tipo de lengua 83,981. A 5+ query reproduces all five, and shows
INEI's grouping: "otra lengua nativa" is every native label except Quechua and Aimara (Jaqaru
and Cauqui included), "otro tipo de lengua" is Portuguese + other foreign + sign language + "no
escucha, ni habla". This is the check independent of the server's own arithmetic.

Two parser traps, both handled in the script: the AREABREAK output for the last district runs on
into a national RESUMEN table that repeats every label (and the per-district tables omit zero
rows, so the summary filled them in); and that endpoint writes the ñ of "señas" as U+FFFD.

## Geography

Religiondots' Peru layer, read-only: `RD_GEO/pe/pe_hexes.gpkg` (258,279 Kontur 400 m hexes,
`pop`) and `pe_lookup.csv` (six-digit ubigeo -> COD-AB adm3 p-code). The language table is on
exactly the census districts religiondots joined, so every one of the 1,874 codes maps; no new
geography. Mazamari and Pangoa (Satipo, Junín) share one COD polygon, PE120699, and are summed
there (religiondots' finding; said in note_public). Placement is `pop_weight`: every language in a
district is spread by Kontur population alone.

## Mapping calls (taxonomy/pe2017.py, taxonomy/tree.d/pe.txt)

- Every printed language label has its own node. Families from Glottolog. New roots: Chicham
  (Awajún, Wampis, Achuar), Cahuapanan (Shawi, Shiwilu), Harakmbut, Zaparoan (Arabela), Tacanan
  (Ese Eja). Glottolog puts Tacanan inside Pano-Tacanan; the tree already has Panoan as a root
  (br.txt), so Tacanan gets its own, the split most readers know.
- Kandozi-Chapra and Urarina go on `isolate` (Glottolog isolates); Tikuna joins br.txt's
  `isolate.tikuna`.
- Same language as an existing node, reused: Cashinahua = Kaxinawá, Yaminahua = Yamináwa,
  Matses = Matsés, Kukama kukamiria = Kokama (Cocama-Cocamilla), Murui-Muinani = Witóto (the
  Murui and Mɨnɨka Witoto of the Putumayo, the same people's language as Brazil's), Ocaina,
  Yagua.
- Kichwa (the Quechua of Loreto and San Martín) is printed apart from Quechua, so it is its own
  node, `quechuan.kichwa`.
- Quechua is one census label covering many varieties; drawn as one, said in note_public.
- "Otra lengua nativa u originaria" (1,060) -> `americas_other`, never guessed. A quarter of it
  (277) is in Purús, Ucayali, where it is 10% of the district: very likely Culina (Madija, Arawan)
  and Mastanahua, which INEI did not code. Not split, nothing publishes it.
- "Otra lengua extranjera" -> `other`; "Lengua de señas peruanas" -> `signlanguage.lsp`, a child
  node since the label names one sign language.
- Not drawn, in `gap`: under-3s (1,435,824), "no sabe / no responde" (204,301), "no escucha, ni
  habla" (25,763).

## Colour

Spanish (red-orange, us.txt) against Quechua is the map's main edge, across the whole Andes, and
Quechua was 0.74 0.16 30, a salmon only a little lighter than Spanish. **Changed in br.txt** (which
defines the node) to 0.86 0.11 45, a light peach in the same Quechuan hue region. Checked where
else Quechua is drawn: Brazil (tiny, scattered); Argentina, where it sits beside Kolla (0.78 0.12
5) and Kolla Quechua (0.66 0.13 15), both still clearly apart, and Diaguita Quechua (0.86 0.09 25),
now close, but that is a Quechua variety of the same region. Hand-picked in pe.txt: the Chicham
violets away from the isolates' pinks in Loreto, Shawi lime, Kichwa amber, Shipibo teal against
Ashaninka's olive, the Campa languages through the Arawakan greens.

## Corroboration of the remainder

Not needed: the question went to everyone aged 3+, nobody is drawn as a derived Spanish speaker.
