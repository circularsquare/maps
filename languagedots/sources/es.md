# Spain (es): the record

Drawn 2026-10-05 by session edd42a8c-es. 46,180,037 people, 52 provinces, 73 nodes, 46,148 dots.
Build: `python sources/es_ecepov.py --fetch`, `python sources/es_place.py --fetch`,
`countries/es.py`, `taxonomy/es2021.py`, `taxonomy/tree.d/es.txt` (one new node).

## What was asked, and by whom

Spain's 2021 census is register-based and asks nothing about language. Its companion sample
survey asks it of everyone: INE's **Encuesta de Características Esenciales de la Población y las
Viviendas (ECEPOV) 2021**, reference date 1 July 2021, about 309,000 dwellings (methodology,
https://www.ine.es/metodologia/metodologia_ECEPOV_2021.pdf). Question 4.21 lists every language
a person knows (Spanish printed, five write-in lines, each with knowledge and use), then asks
which of them the person **spoke first** ("lengua inicial"). Asked for everyone aged 2 and over in
family dwellings; INE imputes, and the tables have no "not stated".

**This replaces the plan the supervisor sketched** (regional sociolinguistic surveys for the
co-official languages plus a nationality proxy for immigrants). ECEPOV is one question, one
year, all of Spain, a sample about fifty times any regional survey's, and it names the
immigrant languages too. The regional surveys are kept for what ECEPOV cannot say (where in a
province the regional language is) and as checks, which it passes (below).

## Tables

1. **ECEPOV 2021, "Personas según la lengua inicial más frecuente por sexo, grupo de edad y
   nacionalidad (española/extranjera)"**, one table per province, tpx 55725-55776 in INE
   province-code order 01-52, `https://www.ine.es/jaxi/files/tpx/es/csv_bdsc/<tpx>.csv` ->
   `data/raw/es/ecepov/`. Each names the languages frequent in that province, combinations
   ("Castellano y catalán"), and "Otra". The 19 community tables (tpx 55547-55565) are the check.
2. **Padrón 1 Jan 2022, foreign residents by nationality and province** (INE 03005, 137
   nationalities) -> `data/raw/es/ine_03005.px`. For the split of "Otra".
3. **Padrón 1 Jan 2022, residents by municipio and main nationality** (INE 33572, national, every
   municipio, about 30 named nationalities plus continent remainders) -> `data/raw/es/padron2022_muni_nationality.csv`
   (the per-nationality cache the fetch writes was deleted once combined). Placement only. The API refuses the table whole
   ("restricciones de volumen"); it is fetched one nationality at a time.
4. **Eustat, "Población de la C.A. de Euskadi por ámbitos territoriales, primera lengua y sexo"**
   (PX_010123_cepv3_lm01, 2021, every municipio) -> `eustat_lm01_2021.json`. Placement and check.
5. **Idescat EULP 2023** (Enquesta d'usos lingüístics de la població), first language of people
   15+, by area of the territorial plan, Barcelona city, Aran and Catalonia, plus the "most
   frequent languages" table -> `eulp2023_*.json`; Idescat's EMEX comarca-municipio list ->
   `idescat_emex_com_mun.json`. Placement, the Aranese count, the Moroccan Berber share, check.
6. Zone lists printed in `sources/es_place.py`: Navarre's Ley Foral del Vascuence zones (2017
   version), the Valencian Llei 4/1983 title V Castilian-speaking municipios, the Franja's
   Catalan-speaking municipios (2001 draft Aragonese language law, the only official list), the
   Galician-speaking municipios of El Bierzo and As Portelas. Transcribed from Wikipedia's
   reproduction of the laws; joined to INE's names and asserted (every name matches once).

Looked at and not used: the Valencian 2021 survey (synthesis gives the childhood home language
for the whole Community only); Navarre's Nastat 2021 Basque knowledge by municipio (published as
a map only; its zone shares are used); the INE 2021 census country-of-birth tables (20 countries,
coarser than the Padrón's nationality ones); ECEPOV microdata (would give every language, but
the published tables suffice and province is the finest geography either way).

## How the drawn rows are built (`sources/es_ecepov.py`)

Per province, from the Total-nationality column:

| part | rows | tier | people |
|---|---|---|---|
| single languages | as printed | modelled | |
| combinations | each person shared equally across the 2-3 languages named (spec 3.6) | derived | |
| foreign nationals' "Otra" | shared by the province's foreign residents by nationality (03005), each nationality on its main first language (es2021.ORIGIN), only languages the province's table does not name, capped at the cell: 1,423,776 of 1,471,298 (96.8%) | derived | |
| Spanish nationals' "Otra" above the Spain-wide level, Balearics, Asturias, Melilla | Catalan, Asturian, Tarifit | derived | 58,441 / 35,891 / 5,987 |
| Aranese (`occitan`, fr.txt's node) | EULP 2023 Aran share 21.2% (15+) x Aran's Padrón 10,268, out of Lleida's Spanish "Otra" (8,200) | derived | 2,176 |
| the rest of "Otra" | `other` | modelled | |

Eight provinces suppress one small Total cell (".", Ávila English, Badajoz Spanish-Romanian,
Burgos Italian, Córdoba French, León German, Lugo English, Palencia Spanish-Italian, Salamanca
Spanish-Romanian); the column total gives each back exactly.

**Spanish nationals' "Otra".** In 49 provinces it follows the foreign share closely: share =
1.13% + 0.122 x foreign share, residual sd 0.75 points (it is mostly naturalised immigrants).
Three provinces stand far off the line: Illes Balears 10.2% (expected 3.8%, +8.5 sd), Asturias
5.5% (1.7%, +5.1 sd), Melilla 10.6% (2.3%, +11 sd). Each has a local language INE's provincial
table does not list: Mallorquín/Menorquín/Eivissenc (the release's own footnote files them
under Catalan, but this table apparently did not recode the write-ins: the Balearic "Otra" has
Catalan's age profile, highest at 60+), Asturian (Otra 6.3% at 60+ against Cantabria's 1.5%),
and Tarifit (Melilla, peak at 40-59). Only the excess over the line moves; the expected part
stays on `other`.

**Moroccans.** Morocco's nationals go 79.8% Arabic, 20.2% Berber (group node, variety unknown):
EULP 2023's Catalonia counts of Tamazight (45,600) and Arabic (179,600) first language. Arabic is
named in every province, so only the Berber part enters the "Otra" split.

**Latin Americans** are on Spanish and never enter the split. Brazil is Portuguese.

## Checks (all in the script's output)

- Language cells sum to each province's total: worst 0.34% (Ourense, an unlisted remainder INE
  does not print); Spanish + foreign = total within 0.001%.
- The 52 provinces sum to the 19 community tables, total and Spanish: 0.00%.
- Drawn rows sum to each province's ECEPOV total within 0.5%.
- **Catalan first language in Catalonia**: ECEPOV 33.9% (all ages 2+, combinations shared) vs
  Idescat EULP 2023 31.8% (15+). **Basque**: Álava 5.6% vs Eustat 6.9%, Bizkaia 15.5% vs 15.2%,
  Gipuzkoa 37.4% vs 37.7%. Independent sources, different universes; they agree.
- ECEPOV's universe is 46.18M against the Padrón's 47.43M (1 Jan 2022): children under 2 are
  not asked and communal housing is out of scope.

## Placement (`sources/es_place.py`, placement only)

Every (province, language) row is placed on its own municipal weight (scatter: 953 rows, none
fell back). Locals = Spanish nationals + nationals of Spanish-speaking countries.

- **Immigrant languages** (and English, French, German, Romanian, Arabic): the municipio's
  Padrón 2022 residents of the nationalities that speak them (es2021.ORIGIN shares). INE 33572
  names about 30 nationalities per municipio; each continent remainder is spread by the
  province's own mix of the unnamed nationalities (03005). 5,542,908 of 5,542,932 foreign
  residents carry a nationality this way. `other` goes on all foreign residents.
- **Regional languages**: locals x g_m, g_m = f_m scaled per province so the weights sum to
  ECEPOV's count (no municipio needed capping at 95%); Spanish = locals x (1 - g_m).
  f_m is Eustat's 2021 municipal first-language share (Basque Country, all 251 municipios);
  Nastat's Basque-speaker share by Navarre's zone (62.1 / 13.5 / 2.7%; 64 + 98 municipios
  joined); EULP 2023 Catalan share by area (Metropolità 25.2% ... Terres de l'Ebre 59.6%;
  Barcelona city and Aran on their own figures; comarca -> area mapping checked by each area's
  15+ / Padrón ratio, 0.857-0.892); Valencian 1 in the Valencian-speaking zone, 0.05 in the
  145 Castilian-speaking municipios; Franja 1 / 0.03 outside; Galician in El Bierzo and As
  Portelas 1 (Ponferrada 0.3) / 0.02 outside; Aranese only in Aran's nine municipios. Flat
  elsewhere (Galicia, the Balearics, Asturias, Melilla).
- Names: INE prints co-official names ("Altsasu/Alsasua"); the lists use Spanish forms. 47
  aliases (Navarre's Basque forms, Veracruz -> Beranuy, Valdetormo -> Valdeltormo, Candín ->
  Valle de Ancares), every list name asserted to match exactly one municipio.
- Plotted (scratch): Valencian follows the coast and stops at the Vega Baja and the inland
  Castilian zone; Basque dense in Gipuzkoa and northern Navarre.

Colour: the `afroasiatic.berber` group (Moroccans' Berber, variety unknown) draws washed out
close to Spanish's yellow, and the two share towns. Left for the colour pass.

## Calls someone might reverse

- ECEPOV instead of the regional surveys for the counts (above).
- Valenciano on its own node, as INE prints it, not merged into Catalan.
- Arabic in Ceuta and Melilla stays `arabic` (the label), though the speech is Darija.
- The three regional "Otra" excesses (Balearic Catalan 58,441, Asturian 35,891, Tarifit
  5,987), and Aranese (2,176).
- Placement factors chosen by hand: Valencian 0.05 outside its zone, Franja 0.03, Galician
  0.02 outside El Bierzo/As Portelas, Ponferrada 0.3.
- Origin languages: India Hindi, Pakistan Punjabi, China Sinitic (group), Nigeria English,
  Ukraine 70/30 Ukrainian/Russian (2001 census native language), Belarus Russian, Switzerland
  65/25/10, Belgium 60/40 Dutch/French, Canada 75/25.

## Room for improvement

- ECEPOV microdata (open, INE) would name every write-in language instead of "Otra" and settle
  the Balearic, Asturian and Melilla calls directly.
- Aragonese, Leonese, Fala and Galician-Asturian are not drawn: none stands out of "Otra" in
  its province (Huesca 2.3%, León 1.8%, Cáceres 1.5% against 1.1-2% expected).
- Galicia's Galician and the Balearics' Catalan are placed flat by local population inside each
  province; IGE (Galicia) and the Balearic surveys have finer figures.
