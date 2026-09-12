# Serbia — boundaries and placement

Built 2026-09-05 by `sources/rs_geo.py`. One 4 MB download; the boundaries cost nothing.

| | |
|---|---|
| units | 168 municipalities, Eurostat **GISCO LAU 2021** (`LAU_RG_01M_2021_4326.shp`) |
| placement | 59,823 **Kontur H3 r8** hexes (~0.74 km²), `kontur_population_RS_20231101` |
| outputs | `data/geo/rs/rs_municipalities.gpkg`, `data/geo/rs/rs_grid_400m.gpkg` |

---

## 1. GISCO's LAU set is not the EU27, and this keeps paying

The file fetched for North Macedonia in §9i carries **169 Serbian polygons** with no
further download. It is worth writing down what else is in it, because it decides which
country is cheap next:

| | | | | | | |
|---|---|---|---|---|---|---|
| AL 61 | AT 2,095 | BE 581 | BG 265 | CH 2,242 | CY 615 | CZ 6,258 |
| DE 11,001 | DK 99 | EE 79 | EL 6,137 | ES 8,131 | FI 310 | FR 34,966 |
| HR 556 | HU 3,155 | IE 166 | IS 69 | IT 7,903 | LI 11 | LT 60 |
| LU 102 | LV 119 | MK 80 | MT 68 | NL 355 | NO 356 | PL 2,477 |
| PT 3,092 | RO 3,181 | RS 169 | SE 290 | SI 212 | SK 2,927 | |

98,188 polygons over 34 countries. Serbia, Albania, Switzerland and Bulgaria are all in
there, so for any of those the boundary half of §12's step 2 is already done and only the
counts have to be found. **Slovakia's 2,927 obce have been in this file since §9e** — its
blocker was never the geography.

## 2. 169 polygons against 168 census units, and the odd one is Petrovaradin

GISCO splits Novi Sad into `Novi Sad` and `Petrovaradin`, the city municipality across the
Danube. The census publishes Novi Sad whole.

So Petrovaradin's polygon is **dissolved into Novi Sad's**, and the direction matters:
merging two polygons loses nothing, whereas splitting the census figure between them would
be inventing a magnitude at a resolution the source does not publish (§14.4). After the
dissolve the join is exactly 1:1 on all 168.

GISCO does split what the census splits: Belgrade's 17 city municipalities are there as
`Belgrade - Barajevo`, `Belgrade - Voždovac`… and Niš's 5 as `Niš - Medijana` and so on.

## 3. The join is by name, and three things make it safe

There are no codes on the census side at all (`rs.md` §3), so this is Romania's and
Ghana's situation. GISCO's `LAU_ID` is the Serbian municipality code (`70017`) and the
workbook has nothing to match it against.

1. **The two sides disagree only about diacritics**, and normalising them away is lossless
   here — no two Serbian municipalities differ only by an accent. `đ` has to be replaced
   before NFKD, which does not decompose it: `Aranđelovac` → `Arandjelovac`,
   `Žitorađa` → `Zitoradja`. GISCO's own spelling is already half-transliterated
   (`Arandjelovac`, `Niš - Crveni Krst` against the census's `Crveni krst`), so both sides
   need the same normaliser rather than one side needing a fix.
2. **`Palilula` is a municipality of Belgrade and of Niš.** `_keys()` finds which bare
   names repeat on *either* side and qualifies only those with their city, rather than
   carrying a hand-written list of exceptions. One name qualifies today; the code does not
   need editing when a second one does.
3. **GISCO writes `Belgrade` where the census writes `Beograd`.** One alias, and it is the
   only translated place name in the Serbian half of the file. It is in `CITY_ALIAS` and
   nowhere else.

## 4. The independent check, and why it is a band and not an equality

GISCO carries its own `POP_2021` per polygon. Checking it against the census total is the
§9i check: **the two are different quantities** — a 2021 estimate against a 2022
enumeration — so demanding equality would either fail on every honest difference or be
loosened until it detected nothing. What a correct join looks like is a tight band; what a
scrambled one looks like is orders of magnitude.

    national 7,020,858 / 6,647,003 = 1.056x
    per unit: median 1.094, min 0.893 (Preševo), max 1.231 (Senta)
    every unit inside 0.8–1.3x

A 0.34-wide band across 168 units is not something a shuffled join produces.

## 5. Placement: the country extract, not the global grid

Serbian opštine average **461 km²** and are not built to a population target — they are
historical districts, typically one town and a scatter of villages in a valley with forest
and mountain around them. §8.2's equal share has nothing here to be equal over, so the
weight is a measured population surface, as for Russia and Kenya.

**The global r6 grid is the wrong file for a country this size, and it says so out loud.**
At r6 (~36 km² hexes) Serbia is 1,991 hexes, and **four municipalities hold no hex centre
at all** — Vračar, Stari grad, Medijana and Sremski Karlovci, which are among the densest
places in the country. A layer that fails hardest where the people are is the wrong layer.

Kontur publishes per-country extracts at r8 and Serbia's is **4.2 MB**:

    https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/
        kontur_datasets/kontur_population_RS_20231101.gpkg.gz

60,540 hexes, 7,179,994 people. 717 fall outside the drawn municipalities — Kosovo, and
the border strip Kontur rounds outwards — leaving **59,823**, a median of **312 hexes per
municipality** against the 1 polygon §8.2 would have used. The URL pattern takes any ISO
code, so this is the route for any future country needing a population surface;
`sources/ke_grid.py` takes Kenya's the same way.

Two approximations, both Russia's, both bounded by the hex size: a hex belongs to the
municipality containing its **centre**, and is then **clipped** to it so no dot lands
across a border (6,264 of 59,823 clipped). A unit too small to hold any centre would fall
out of the layer entirely and its dots would have nowhere to go, so it carries its own
polygon as a single cell instead — `de_grid.py`'s answer for 34 German Gemeinden. At r8 no
Serbian municipality needs it.

## 6. Where Kontur is least trustworthy, named

The hex populations are **relative weights inside a unit only**; every municipality's dot
count comes from the census. So a unit Kontur models badly is not miscounted — its dots are
placed on a worse surface. Three are worth naming:

| municipality | census | Kontur | ratio | hexes |
|---|---|---|---|---|
| Bujanovac | 41,068 | 7,273 | 0.18× | 290 |
| Preševo | 33,449 | 6,241 | 0.19× | 216 |
| Crna Trava | 1,063 | 1,745 | 1.64× | 197 |

**The two bad ones are the Albanian-majority Preševo valley**, and Kontur — built from
GHSL, HRSL and building footprints — misses about four fifths of the people there. Each
still has a couple of hundred hexes, so the *shape* of the surface survives and the dots go
to the modelled settlements; what is lost is any confidence that the relative weights
between those settlements are right. Serbia's two most Muslim southern municipalities are
therefore its worst-placed. Everything else sits between 0.6× and 1.5×, median 1.095.
