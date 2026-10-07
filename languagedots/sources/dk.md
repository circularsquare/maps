# Denmark (dk): record

Drawn 2026-10-05 (session edd42a8c-last). 6,025,603 people (Statistics Denmark, 1 January 2026),
99 units (98 kommuner and Christiansø), 531 nodes (nearly all from origin mixes), every row
`derived`. 5,917 dots, 436 rings.

```
python sources/dk_dst.py --fetch      # FOLK1C 2026K1, BEF5G, BEF5F into data/raw/dk/
python sources/dk_dst.py              # data/normalized/dk.csv; data/geo/dk/dk_hexes.gpkg if missing
python sources/origin_mix.py --fragment dk
python taxonomy/build.py
python tools/check_country.py dk
python scatter.py --country dk
```

Files: `sources/dk_dst.py`, `taxonomy/dk2026.py` (identity), `taxonomy/tree.d/dk.txt`,
`countries/dk.py`, `data/normalized/dk.csv`, `data/raw/dk/`, `data/geo/dk/dk_hexes.gpkg`; two rows
in `kontur_cap.csv`. Kontur DK downloaded into `data/geo/kontur/`.

## 1. Sources

No census or register asks language. Anita's 2026-10-05 rule for rich countries; Norway's method
(`sources/no.md`).

- **FOLK1C** (StatBank API, open): population per kommune by ancestry (Danish origin 5,014,567,
  immigrants 780,954, descendants 230,082) and country of origin (209 named). Checks: ancestry
  parts sum to the total in every kommune (gap 0); countries sum to each kommune's total (gap 0).
  Area codes = religiondots' `dk_lau.gpkg` LAU codes, 99 both ways.
- DST English country names -> alpha-2 through queue.csv names plus `NAME_FIX` (UK, USA,
  Czechoslovakia, Yugoslavia incl. Serbia and Montenegro, Palestine's three parts...). 1,538
  people of stateless or "not stated" origin on `other` (in `gap`).
- **BEF5G / BEF5F**: people born in Greenland (17,483) and the Faroe Islands (10,804) living in
  Denmark, by parents' birthplace, national only. FOLK1C counts them as Danish origin, so without
  these they would all be Danish.
- **German minority**: "ca. 15.000 personer ... ca. 6 % af befolkningen i de fire sønderjyske
  kommuner (2022)" (Grænseforeningen, leksikon "Tyske mindretal, Det"); "omkring to tredjedele
  [taler] dansk (ofte sønderjysk) som hjemmesprog" (Den Store Danske vol. 4, 1996, via
  da.wikipedia "Det tyske mindretal i Nordslesvig"). nordics.info agrees most members do not use
  a minority language foremost.

## 2. How the counts are made

- Each origin through `origin_mix.mix(iso, "dk")`. Immigrants keep it whole; descendants at
  78% (Parkvall 2009, Sweden, borrowed as Norway did), the rest Danish.
- Greenland- and Faroe-born with no parent born in Denmark (9,414 and 9,313) on Greenland's and
  the Faroes' home mixes; spread over kommuner by Danish-origin population (no kommune figure).
- German 5,000 (a third of 15,000) over Aabenraa, Haderslev, Sønderborg, Tønder by population
  (2.2% each), carved from Danish. The 51,925 German total is mostly German immigrants.
- Drawn: Danish 83.7%, Levantine Arabic 1.0%, Polish 0.9%, German 0.9%, Turkish 0.8%.

**Placement**: Kontur hexes keyed to the 99 LAUs by centroid: log correlation 0.981 (shuffled
best 0.426); one unit outside a factor of 3 (185 Tårnby, airport, 4.3x). Two cap blocks
registered `capped`: Kastrup/airport (Tårnby) and Østerbro/Nordhavn (Copenhagen, 39% of the
kommune in 12 hexes).

## 3. Calls someone might reverse

- Greenlanders and Faroese with a parent born in Denmark drawn as Danish; those with both
  parents "unknown" (older people, 5,110 and 6,710) drawn on the home mix.
- Greenland/Faroe-born spread by population, not by where they live (Greenlanders cluster in
  Copenhagen, Aalborg, Aarhus, Odense).
- German minority's home German is one third of a rounded estimate from a 1996 encyclopedia.
- Sønderjysk is drawn as Danish (a dialect; DST and nobody else counts it).
- 78% descendant retention borrowed from Sweden. Vintage 2026Q1.

## 4. Room for improvement

A kommune breakdown of Greenland-born residents (DST has none open); a survey of the German
minority's home language.
