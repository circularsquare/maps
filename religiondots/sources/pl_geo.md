# Poland — gmina boundaries

Built 2026-09-03 by `python sources/pl_geo.py --fetch`. Writes
`data/geo/pl/pl_gminy.gpkg`, 2,477 polygons, the single layer `countries.py` reads.

## 1. Source, and why not the other two

**Eurostat GISCO, LAU 2021, 1:1,000,000, EPSG:4326.**

```
https://gisco-services.ec.europa.eu/distribution/v2/lau/download/ref-lau-2021-01m.shp.zip
```

98MB, a zip of zips — one shapefile per projection, take `LAU_RG_01M_2021_4326.shp.zip`.
Saved under `data/geo/lau2021/`.

Rejected alternatives:

- **geoBoundaries POL ADM3** — the API returns `boundaryYearRepresented: 2017`. Polish
  gminy have been created and merged since, so this is precisely the spec §8.1 hazard:
  it would join, mostly, and be quietly wrong. Not used.
- **GUGiK PRG** (Państwowy Rejestr Granic) — authoritative and carries TERYT directly, so
  it needs none of §2's decoding. Rejected only because it is a large national-only
  download and GISCO covers 34 countries in one file, which matters because **Slovakia,
  Hungary and Romania are the next candidates and are in the same file** (sources.md §11).
  If the join rule below ever breaks, PRG is the fallback and it is a better file.

**The vintage is 2021, the same year as the census.** That is the point: 2,477 GISCO
features against 2,477 census gminy, nothing unmatched either way.

## 2. THE JOIN IS NOT THE OBVIOUS ONE

GISCO gives Poland a **13-digit** `LAU_ID`; GUS gives a **7-digit** TERYT code. Neither
contains the other as a substring:

```
LAU_ID   1006061110802
TERYT       0608022
```

Joining on the ids as delivered matches **0 of 2,477**. The dangerous part is that both
sides have exactly 2,477 units, so a unit-count check passes while the join fails
completely — the §5c shape of mistake, where the reassuring number is not the one that
would catch the error.

The `LAU_ID` embeds TERYT at fixed offsets and **drops the type digit**, the trailing
1/2/3 that distinguishes an urban gmina from a rural one from a mixed one:

```
voivodeship = LAU_ID[4:6]     powiat = LAU_ID[9:11]     gmina = LAU_ID[11:13]
```

so `LAU_ID[4:6] + LAU_ID[9:11] + LAU_ID[11:13]` reproduces TERYT's first six digits, and
those six are already unique per gmina. `pl_gminy.gpkg` is keyed on that six-digit `kod`,
and `_pl_counts()` in `countries.py` slices `geo_id[:6]` to match. **The seventh digit is
not lost information that matters** — it is a property of the gmina, not part of its
identity.

`pl_geo.py` asserts the id is 13 digits before slicing, so a reissue that changes the
format fails loudly instead of producing a wrong six-digit key.

### The evidence the offsets are right rather than lucky

Names. 2,476 of the 2,477 joined pairs agree on the gmina name after stripping GUS's
`gm. m.` / `gm. w.` / `gm. m-w.` prefixes.

The single disagreement is real and is **not** an error:

```
260417   GUS='gm. w. Nowiny'   GISCO='Sitkówka-Nowiny'
```

which is the name the gmina had before it was renamed in 2021. A join that were wrong
would not produce 2,476 name agreements; a join that were right would produce exactly one
disagreement of this kind.

## 3. Placement layer

Gminy are the placement layer as well as the count layer, because **GUS publishes religion
at no finer unit**. Same situation as Czechia and Ireland: no separate placement geography
and no allocation inside a unit.

Median gmina population is about 7,500 — twice a US census tract, finer than a Mexican
municipio — so an equal share per polygon is a reasonable population weighting nearly
everywhere (spec §8.2).

## 4. Where it is not reasonable: Warsaw

This is Czechia's Prague problem, one size smaller, and unlike Czechia there is no fix.

| gmina | population | area | share of Poland |
|---|---|---|---|
| Warszawa | 1,794,166 | 517.2 km² | **4.69%** |
| Kraków | 779,966 | 326.8 km² | 2.04% |
| Łódź | 672,185 | 293.3 km² | 1.76% |
| Wrocław | 641,928 | 292.8 km² | 1.68% |
| Poznań | 532,048 | 261.9 km² | 1.39% |

`cz_geo.py` could subdivide Prague because ČSÚ publishes the 142 city districts *with
religion counts on them*. **GUS does not**: TABL.7 is gminy and stops. Warsaw's 18
dzielnice exist as boundaries, but there are no counts for them, so subdividing would be
an allocation inventing structure the source does not have — spec §3.10's line. Left as
one polygon, and 4.7% of Poland's dots spread evenly over 517 km² including the Vistula
and the forests.

Prague was 12.4% of Czechia in one polygon, so this is about a third as bad.

## 5. Reuse

The same download is the boundary source for every LAU country in Europe:

```
AL AT BE BG CH CY CZ DE DK EE EL ES FI FR HR HU IE IS IT LI LT LU LV MK MT
NL NO PL PT RO RS SE SI SK
```

98,188 features. Slovakia, Hungary and Romania — the rest of sources.md §11's Central
European cluster — are in it, and each will need its own `LAU_ID` decoding checked the way
§2 checks Poland's, because **the id format is per-country** and Poland's is not a
general rule.
