# El Salvador: Censo 2024, languages spoken besides Spanish, distrito

**Drawn** 2026-10-05 (agent d9e44929-sv). 5,922,921 people on 262 distritos, 5 nodes, 5,921 dots
at 1:1000, 4 rings. Every row `derived` (a multi-answer question, spec §3.6). Redrawn the same day
(d9e44929-svec) with the foreign languages drawn as Spanish, Anita's ruling below.

## Source

- ONEC / Banco Central de Reserva, VII Censo de Población y VI de Vivienda 2024 (fieldwork
  2 May to 26 June 2024). The question: "¿Habla otro idioma aparte del español? ¿Cuáles?",
  people aged 3 and over, several answers. The BCR's language dashboard ("Resultados del VII
  Censo ... Idiomas Tercera Entrega", ArcGIS item 8c4eed18c1ba4a3bb4d2480cb7c7e06a) states the
  universe ("Total de personas de 3 años o más que habla otro idioma") and that shares can sum
  past 100% because one person can speak several languages.
- **Where the figures are.** The GeoPortal `poblacion.bcr.gob.sv` is an ArcGIS Hub (org
  `xSn1Pefr3M9zkw7Z`); its page is an empty shell, and the figures are open, keyless feature
  services under `services8.arcgis.com/xSn1Pefr3M9zkw7Z/arcgis/rest/services/`. Found with
  `hub.arcgis.com/api/v3/domains/poblacion.bcr.gob.sv` then a `sharing/rest/search` on the org
  (memory note "Census data on GIS servers"). Services used, all fetched by
  `sources/sv_censo.py --fetch` into `data/raw/sv/`:

| service / layer | what |
|---|---|
| `tabla_idioma/0` | mentions per language per distrito, plus municipio, department and national aggregate rows (id `000000`) |
| `Limites_idiomas/2` | the same mentions as a map layer, plus `Todos`; its `Todos` polygons are the distrito boundaries |
| `Si_habla_otro_idioma___Copy/0` | `si`: persons who speak another language, per distrito |
| `capas_población_1/6`, `/3` | population per distrito and department, all ages, 5-year bands by sex |
| `Otro_30_09/0` | the 30 September department release, the only one with `No` (used for a printed check) |

- Categories: Inglés, Francés, Italiano, Náhuat, Pisbi, Potón, LESSA, Otro, Español. The
  department layer spells out "Pisbi (Cacaopera)", "Potón (Lenca)", "Lengua de señas (LESSA)".
- Grain: 262 distritos, the pre-2024 municipalities, which the 2024 reform kept as districts of
  44 new municipios. The coverage sweep had "national (as reported so far)".
- Licence: none stated on the services; an official statistics office's published results,
  open and keyless, as for other countries here.

## Checks (`sources/sv_censo.py`, all asserted, all pass)

1. 262 distritos in every table, the same six-digit ids.
2. `tabla_idioma` and `Limites_idiomas` agree on every distrito and language (|diff| 0), and
   `Todos` equals `si` everywhere.
3. Distritos sum exactly to the table's own 44 municipio rows, 14 department rows and national
   row, on all nine languages.
4. National totals equal the BCR's released figures (press, March 2026): 427,368 people speak
   another language; Inglés 414,887, Francés 16,741, Italiano 6,911, Náhuat 1,135, Pisbi 24,
   Potón 32, LESSA 2,025, Otro 13,326.
5. Per distrito no language exceeds `si`, and `si` is at most the sum of mentions (Español
   included: 32 yeses are not covered by any non-Spanish mention, people who named only Spanish).
   1.065 non-Spanish mentions per yes nationally.
6. Population: 262 distritos sum to the 14 department totals and to 5,922,921; age bands sum to
   each total; names agree with the language table but for two spellings (Dolores / Villa
   Dolores, San Buenaventura / San Buena Ventura), same ids.
7. Printed: the universe. The 30 September release's Sí + No per department over population
   less three fifths of the 0-4 band (ages 0-2) reads 0.990 to 1.006, national 5,695,340 against
   5,698,981. So the question's universe is the 3+ population of this same table. (That release
   had Sí 436,219; the revised figure is 427,368. Only the revised one is used.)

**The headline 6,029,976 is not this table.** The census's announced total is 6,029,976; the
per-distrito population layer sums to 5,922,921, 107,055 (1.8%) fewer, and the language universe
reconciles with the smaller one (check 7). Nothing found says what the difference is (people
outside private dwellings, or an omission adjustment, are guesses). It is in `gap`.

## The split (`countries/sv.py`)

The question presupposes Spanish: nobody names it as such, everyone is taken to speak it, and a
yes means Spanish plus at least one other language. Spec §3.6 says scale each unit's mentions to
its population. Applied to the whole distrito that is wrong here: with everyone's Spanish as a
mention it shrinks the 93% who said no as well, and would draw English at about 0.93 of its
mentions instead of about half. So the scaling is applied to the group the census separates,
the yes-sayers: within a distrito, the group holds S people and S + M mentions (Spanish once each,
plus M other mentions), each language gets `mentions * S / (S + M)` and Spanish gets
`S * S / (S + M)`; everyone else (`P - S`, under-3s included) is Spanish. A yes naming one other
language counts exactly half to each; only the 6.5% extra mentions are approximated.
`Español` answers are not added: the person is already counted as a Spanish speaker.

The split gives: Spanish 5,702,563, English 200,966, French 8,070, other 6,418, Italian 3,338,
LESSA 984, Náhuat 554, Potón 16, Cacaopera 12. English is 68% in San Salvador (91,504) and La
Libertad (45,904).

**Foreign languages are drawn as Spanish** (Anita, 2026-10-05: "El Salvador: ideally we don't
use English like this, to be consistent with neighbours"). Guatemala, Nicaragua, Costa Rica and
Mexico draw only the indigenous languages as measured and everyone else as Spanish (spec §3.5);
here English, French, Italian and Otro were drawn from a second-language question, English at
3.4% of the country against 554 Náhuat. So in `countries/sv.py` the shares the split gives to
Inglés, Francés, Italiano and Otro go to Spanish; the indigenous and LESSA shares stay exactly as
the split gives them (a Náhuat + English yes-sayer is still 1/3 Náhuat, now 2/3 Spanish). Otro
goes with them: the census names all three of the country's indigenous languages, so Otro is
mostly foreign. Kept out of the map, kept here:

| answer | mentions | share the split gave | now drawn as |
|---|---|---|---|
| Inglés | 414,887 | 200,966 | Spanish |
| Francés | 16,741 | 8,070 | Spanish |
| Italiano | 6,911 | 3,338 | Spanish |
| Otro | 13,326 | 6,418 | Spanish |

Nothing published says how many named only a foreign language besides Spanish (no combination
table); 427,368 yes-sayers made 455,081 non-Spanish mentions, only 3,216 of them indigenous or LESSA, so
at least 424,152 yes-sayers named no indigenous or sign language at all.

Drawn now: Spanish 5,921,356 (99.97%), LESSA 984, Náhuat 554, Potón 16, Cacaopera 12.
`countries/sv.py` asserts the shares add back to each distrito's population, all 262.

**Under-3s are drawn as Spanish**, not put in `gap` as Ecuador's under-1s are. Here no Spanish
row is measured anyway (it is the complement of the yes-sayers for everyone), so the under-3s
are the same kind of derived remainder; leaving them out would need an estimated age cut from
the 0-4 band (playbook: cutting a band is easy to get wrong). Reversing it: subtract
`0.6 * (H_0_4 + M_0_4)` from P in `_counts()`.

## Geography (`sources/sv_geo.py`)

- Boundaries: the BCR's own distrito polygons from `Limites_idiomas/2`, the same service and ids
  as the figures, so there is no name join. COD-AB admin2 is the same 262 units but on
  alphabetical pcodes, which religiondots found mispair the official order; not used.
- Placement: religiondots' `sv_hexes.gpkg` (raw Kontur SV 2023-11, keyed to COD department;
  the Kontur extract itself is no longer on disk) re-keyed by hex centroid to distrito. 46 hexes
  (4,242 people) centred in no distrito go to the nearest distrito of their own department, at
  most 530 m.
- Witnesses: department names agree with religiondots' lookup (official number = LAPOP prov −
  300) on all 14; **COD's and the BCR's department lines are different tracings**, IoU 0.86
  (Cuscatlán) to 0.95, areas e.g. Chalatenango 2,021 km2 (BCR) against 1,950 (COD), Cuscatlán
  748 against 678, and 579 hexes (105,154 Kontur people, 1.8%) fall in a different department on
  the two. The bars (IoU > 0.85, < 3%) only ask that they are the same departments; the census
  counted on the BCR's lines. Kontur per distrito against the census: national ratio 1.006, p10
  0.77, median 0.96, p90 1.24; log r = 0.957 against a best of 0.216 over 500 shuffles.
- **San Isidro Labrador (040326) reads 52.8x**: Kontur puts 31,108 people in 15 hexes of a
  25.7 km2 rural distrito of Chalatenango that the census counts at about 590. A Kontur artefact
  (not at the cap, so the scatter's cap check does not see it). It moves nothing between
  distritos; it only decides where that distrito's few dots fall inside it. Left.
- Median distrito 51.9 km2, about 70 hexes: well above the grid floor.
- Scatter: 232 hexes clipped for sea (0.19% of area); one unit left unclipped by water.py's 95%
  rule; no Kontur cap block stopped it.

## Mapping calls (`taxonomy/sv2024.py`, `taxonomy/tree.d/sv.txt`)

- Náhuat → new `utoaztecan.pipil`, "Náhuat (Pipil)", Glottolog pipi1250, a sister of mx.txt's
  `utoaztecan.nahuatl` (that node is a language, so Pipil cannot sit under it).
- Pisbi → new `misumalpan.cacaopera`, Glottolog caca1247 under misu1242 (ni.txt's root).
- Potón → new `isolate.lenca_salvador`, "Potón (Salvadoran Lenca)", Glottolog lenc1243 in the
  two-language Lencan family lenc1239. Under `isolate` as gt.txt does with Xinka (a tiny family
  most readers know as one language); kept to the Salvadoran variety so Honduras's Lenca can
  have its own node.
- LESSA → new `signlanguage.lessa`, Glottolog salv1237.
- Otro → `other` in the mapping. The census names the country's three indigenous languages, so
  Otro is mostly foreign, but nothing separates an indigenous answer from a foreign one in it.
  Since the 2026-10-05 ruling `countries/sv.py` draws its share as Spanish, with English, French
  and Italian (the mapping still says what the label means).
- Colours all generated: Spanish (sand) is now the only node with more than a handful of dots.

## Calls someone might reverse

- Foreign languages drawn as Spanish (Anita's ruling, 2026-10-05, above), replacing the first
  build's §3.6 sharing, which drew English at 3.4% of the country. Ecuador was changed the same
  way. Otro going with the foreign languages is this session's call; it could hold a few
  indigenous answers (Guatemalan or Honduran migrants), 6,418 people's share at most.
- Under-3s drawn as Spanish (above).
- The scaling inside the yes group instead of over the whole distrito (above).

## Not done

- The 2007 census (REDATAM at onec.bcr.gob.sv per the coverage sweep) was not probed; it is
  older, and 2024 is open at a finer grain than the sweep expected.
- The indigenous self-identification tables (68,148 people, `map_indigenes_segunda_entrega_WFL1`)
  are on the same portal; they are ethnicity, not language, and are not drawn.

## Immigrant languages: not drawn (2026-10-05, session edd42a8c-latn)

Looked for country of birth to draw immigrant languages as Mexico, Costa Rica, Panama, Honduras
and Nicaragua now do (`sources/mx.md`, "Immigrant and settler languages"). The 2024 census on
the BCR's ArcGIS org publishes birthplace only as "aquí / otro municipio / otro país" per
department (`Residencia_Oct/FeatureServer/0`, 14 rows; search of the org for nacimiento,
migración, país found nothing finer; the other migration services are internal moves and work
commuting). The 2007 census's REDATAM base (CPVSLV2007) is not reachable: redatam.org/binslv
answers 404, prod.redatam.org/binslv 404, redatam.digestyc.gob.sv does not resolve. So nothing is
drawn; the non-Spanish-speaking foreign-born are a small share here (Honduras, Nicaragua and
Guatemala dominate, and US-born children of returnees would be drawn as Spanish anyway).

## Moved from countries/sv.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- 1,135 people named Náhuat, the Nahua language of western El Salvador, which drawn shared comes to fewer than one dot; the most are in Sonsonate (428), above all Santo Domingo de Guzmán and Izalco, and in San Salvador (294).
- The census does not publish how many languages each person named, so inside each district the people who named another language are shared in proportion to how often each language was named.
