# Serbia: Popis 2022, mother tongue

Built 2026-10-04 (session d9e44929-rs). Rebuild:

```
python sources/rs_census.py [--fetch]   -> data/normalized/rs.csv
python taxonomy/build.py
python tools/check_country.py rs
python scatter.py --country rs
```

Drawn: 6,255,702 people on 168 municipalities and city municipalities, 19 labels on 19 nodes,
6,246 dots, no rings. Not drawn: 391,301 (5.9%), the gap: 303,179 "Unknown" and 88,122 "Did not
declare". Kosovo was not enumerated and is not drawn from this source.

## 1. The table

Republički zavod za statistiku (RZS), Popis stanovništva, domaćinstava i stanova 2022. The census results portal lists every published table as a direct Excel
file: https://popis2022.stat.gov.rs/sr-latn/popisni-podaci-eksel-tabele/. Two are used, no login,
browser User-Agent, TLS verifies:

| file | what | use |
|---|---|---|
| `media/31330/7_stanovnistvo-prema-maternjem-jeziku.xls` (570 KB, sheet `opstine`) | mother tongue by municipality and city, by sex and settlement type | drawn |
| `media/31348/1_stanovnistvo-prema-nacionalnoj-pripadnosti-i-maternjem-jeziku.xlsx` (34 KB) | ethnicity x mother tongue by region | check |

**The question**: maternji jezik, mother tongue, one answer, which people may decline ("Did not
declare" is those who did). 18 named
languages, "Other languages", "Did not declare", "Unknown". No breakdown of "Other languages" at
any level.

**Finer than municipality**: not found. The portal's only settlement table is total population
(`0_ukupan-broj-stanovnika-naselja.xlsx`); religiondots found the same ceiling for religion. The
office's main site (www.stat.gov.rs, data.stat.gov.rs) fails TLS verification from this machine,
so its publication list was not read. Not pursued further.

## 2. Reading the sheet

The layout is religiondots' religion workbook (`religiondots/sources/rs.py` and `rs.md` §2-4) with
one more row type, and the parser follows it so that its ids are the same:

- six levels share column 0 with no codes: republic, Srbija-sever/jug, 4 regions, 25 oblasti,
  municipalities. Each unit's block is its total row (sex `с`/`t`), male, female, then `Градска`
  (urban) and `Остала` (other settlements) with their own sexes. Only the total row is kept.
- `Grad Niš`, `Grad Požarevac`, `Grad Užice`, `Grad Vranje` sit among municipalities as parents of
  their own city municipalities; their children are the following rows that sum to them exactly.
  Re-levelled `city`, not drawn.
- Palilula is in Belgrade and in Niš; city municipalities are keyed `<city> - <name>`, all others
  by bare name (religiondots' keys).
- Kosovo's region row is all `...`; dropped and reported.

## 3. Checks (all pass)

- national total 6,647,003, as published.
- every level (2 halves, 4 regions, 25 oblasti, 168 municipalities) sums to the national row in
  all 22 columns, exactly. No rounding or suppression anywhere.
- the 21 categories partition every municipality.
- **join**: all 168 municipality ids match religiondots' `data/normalized/rs.csv` (the religion
  table of the same census) both ways, with the identical total population in every unit. The
  only difference is whitespace: religiondots keeps RZS's double space in `Beograd - Stari  grad`;
  rs.csv collapses it, and `countries/rs.py` collapses the hex layer's keys the same way.
- **second table**: the municipalities summed by region equal the ethnicity x mother tongue
  table's four region rows in all 22 columns, exactly.

## 4. Mapping (taxonomy/rs2022.py)

All 18 named languages get their own leaf. Two new nodes, in `taxonomy/tree.d/rs.txt`:

- **Bunjevac** (`slavic.south.bunjevac`, 3,319, 3,005 in Subotica). Glottolog files it as a
  dialect of Serbian-Croatian-Bosnian (bunj1247), as it does the Serbian, Bosnian and Montenegrin
  standards; the census prints it apart from Serbian and Croatian.
- **Vlach** (`romance.vlach`, 23,216; Kučevo 19.6%, Žagubica 16.4%, Negotin 12.8%, Petrovac na
  Mlavi, Boljevac). The Romanian speech of eastern Serbia's Vlachs; Glottolog lists its dialect
  groups (Ungureni, Tarani) under Romanian. Printed apart from Romanian, which is the Banat
  (Alibunar 22.3%, Vršac, Žitište), so a sibling of Romanian, not a child. The distribution rules
  out Aromanian.

Existing nodes: Serbian, Bosnian, Croatian, Montenegrin (as printed; they follow ethnicity, which
note_public says), Ruthenian -> Rusyn (Pannonian Rusyn, a Rusyn dialect in Glottolog, pann1240;
as ro2021 and hu2022), Roma language -> `romani.romani` (variety not stated), Other languages ->
`other` (no split; RZS does not separate regional from migrant languages, so there is no
indigenous remainder to keep apart).

"Other languages" is largest in Dimitrovgrad (9.7%) and Subotica (4.4%). Nothing published says
what was written there; plausible guesses (a local name for the Bulgarian or Bunjevac speech,
"Serbo-Croatian") are not drawn as such.

## 5. Geography

religiondots' `data/geo/rs/rs_grid_400m.gpkg` (Kontur H3 r8, 59,823 hexes, `unit` and `pop`),
read-only, keyed exactly as rs.csv apart from the whitespace above. GISCO LAU 2021 boundaries with
Petrovaradin dissolved into Novi Sad; religiondots' `sources/rs_geo.py` and `rs_geo.md` hold the
join and its checks. Its caveat carries over: Kontur under-models the Preševo valley, so Preševo's
and Bujanovac's Albanian dots sit on the weakest surface in the country. No Kontur cap blocks hit.

## 6. Colour

Serbian was bare and generated as an olive 0.07 (OKLab) from Slovak's lime, so the Slovak towns of
Vojvodina disappeared. Hand-picked Serbian as a light teal-green (0.77 0.14 168), at least 0.11
from everything sharing a municipality with it; Bunjevac (0.64 0.13 115) and Vlach (0.72 0.15 20)
hand-picked too. Serbian elsewhere (hu, ro, cz, us) is a small migrant language and is clearer
against Hungarian and Romanian than before. Left alone: Bosnian and Montenegrin are only 0.06 apart
(pre-existing colours); they barely meet in Serbia but will in Montenegro.

## 7. Wording

- how: "census, 2022, mother tongue"
- grain: "168 municipalities and city municipalities, 40,000 people on average" (6,647,003 / 168)
- gap: the 391,301 unknown or undeclared, and Kosovo not enumerated.
- note_public: identity-following labels (Serbian/Bosnian/Croatian/Montenegrin/Bunjevac), Vlach
  vs Romanian, the 5.9% not drawn, the single "other".

"Unknown" (4.6%) is highest in central Belgrade (Savski venac 15.2%, Stari grad 11.3%) and in
Majdanpek and Kučevo (11.9%, 9.1%), the second pair Vlach country; leaving it out may slightly
understate Vlach in the east. Not corrected: nothing published says what the unknowns speak.
