# Uruguay (uy): record

Drawn 2026-10-05 (session edd42a8c-amer). Censos 2011 country of birth, per department and
Montevideo barrio: Uruguayan-born on Spanish, foreign-born by `origin_mix.mix(iso, "uy")`.
3,285,877 people on 80 units (religiondots' 18 departments + 62 barrios), every row `derived`.
Spanish 99.0%, Portuguese 13,640, Italian 4,642, English 3,342. 3,276 dots, 208 rings.

Files: `sources/uy_census.py`, `taxonomy/uy2011.py`, `taxonomy/tree.d/uy.txt` (borrowed nodes
only), `countries/uy.py`, `data/raw/uy/` (3 programs + outputs), `data/normalized/uy.csv`.

## 1. Sources

- **INE Censos 2011**, REDATAM base CPV2011 at prod.redatam.org/binury (open; outputs come back
  in an iframe under /redury/tempo/). No language question. The ethnic-ancestry question (afro,
  indigenous, Asian, white) names no group with a language in use, so it is not drawn.
- PAISNAC by unit U (DEPTO.DEPTO outside Montevideo, 100 + VIVIENDA.CBAR inside). A variable in
  a SWITCH DEFAULT evaluates to 0 on this engine (Honduras's server too), so every department has
  its own INCASE. Checks pass: 80 units in the crosstab and in a separate FREQUENCY of U; units sum
  to 3,285,877 (the public base; INE publishes 3,286,314); country totals equal the national
  PAISNAC frequency; every label mapped.
- **"No relevado" 115,797** (birthplace not collected) and "No declarado o ignorado" are not
  foreign-born: spread over each unit's known birthplaces, Uruguay included. Known foreign-born
  about 77,000 (Argentina 26,782, Brazil 12,882, Spain 12,667, Italy 5,541).
- **Portuguese on the border: no first-language source.** INE's Encuesta Telefónica de Idiomas
  2019 (www5.ine.gub.uy, ETI, 4,029 people 15-60 in towns of 5,000+) gives knowledge of
  Portuguese (29.7%, highest in Artigas, Rivera, Cerro Largo; 18% of knowers learned it at the
  border). Knowledge is a learned-language measure (§2 rule), so not drawn. Wikipedia's
  "100,000 native speakers" of Portuñol riverense cites no survey.

## 2. Calls

- Uruguayan-born all on Spanish, including the northern border where Uruguayan Portuguese is a
  home language for many; said in note_public.
- Brazil-born on Portuguese via origin_mix (Brazil's home mix); Argentines on Argentina's mix.
- Placement: religiondots' hexes, by plain population (`pop_weight`).

## 3. Room for improvement

Any survey asking home or first language in the north (a sociolinguistic study with a sample
frame, or an ECH module) would let Uruguayan Portuguese be drawn in Artigas, Rivera, Cerro Largo
and Rocha. The 2023 census asked nothing on language either.
