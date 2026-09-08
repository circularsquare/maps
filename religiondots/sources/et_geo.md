# Ethiopia — boundaries and placement

`sources/et_geo.py` → `data/geo/et/`. 738 woreda polygons, 422,726 Kontur hexes.

| | |
|---|---|
| boundaries | `ET_GEOG_ADM3_2021_uscb_202308`, a layer of the **same geodatabase as the counts** |
| join | **identity** — both tables are keyed on `GEO_MATCH` |
| placement | Kontur `kontur_population_ET_20231101`, H3 r8 (~400 m), 423,833 hexes |
| outputs | `et_woredas.gpkg`, `et_hexes.gpkg`, `et_lookup.csv` |

---

## 1. The join is an identity, and that is the whole reason this country was cheap

Every other country on this map needed its boundaries found separately from its counts and
then married to them, and the marriage is where the time goes:

- **Chile** — names from the statistical source, never the boundary file (`spec` §12)
- **Guyana** — no names at all in the census; joined through ISO 3166-2 (§9r)
- **Ghana** — an acronym collision between two districts (§9n)
- **Kenya** — two files labelled ADM2 turned out to be different tiers (§9o)
- **China** — the join *was* the ingest: 2,691 of 2,859 counties, by name, by code order and
  by a hand table, and geoBoundaries had to be rejected outright (§9r)

Ethiopia has none of that. `ET_GEOG_ADM3_2021` and `ET_RELIGION_2007census` are two layers of
one file, both keyed on `GEO_MATCH`, produced together by one publisher. There is nothing to
match, nothing to transliterate, and nothing to verify — a `merge` on the key is exact by
construction and `et_geo.py` asserts the feature counts rather than a match rate, because a
match rate would be meaningless here.

**This is the argument for the rest of the USCB series** (`sources.md` §11h): the reason to
prefer these files over a national office's own is not that the tabulations are better — they
are the same tabulations — but that the geography arrives attached.

## 2. Inland water and the parks are already cut out

18 of the 756 ADM3 polygons carry no counts. Four are the woredas with no census data
(`sources/et.md` §3); the other 14 are water and protected land, each its own polygon:

```
LAKE T'ANA (3,214 km²)   LAKE ABAYA   LAKE CHAMO    LAKE LANGANO   LAKE SHALA
LAKE ZIWAY               LAKE AWASSA  LAKE CHOMEN   LAKE AFERA     LAKE K'OK'A HAYK
TEZEKÉ DAM               MAGO NATIONAL PARK         NECH SAR NATIONAL PARK
GAMBELLA WILD LIFE RESERVE
```

They are **holes in the woreda cover, not areas inside it**, so nothing has to be clipped
afterwards. §9n records inland water finally costing something in Ghana; here it costs
nothing, and the difference is that the publisher cut the lakes out before shipping. Lake
Tana reads on the map as a clean void in the middle of the Orthodox highland, which is a
useful check that the polygons are the right ones.

`et_geo.py` prints all 18 by name rather than silently dropping the ones with no counts —
a polygon with no data and a polygon that is a lake need to be distinguishable in the log,
because only one of them is a problem.

## 3. Kontur is needed here for Kenya's reason, more sharply

`ke_grid.py` exists because Kenya's counts are 47 counties and its counties are wildly uneven
in habitability. Ethiopia's counts are 738 woredas — sixteen times finer — and it needs the
same treatment anyway:

| | |
|---|---|
| the 50 largest woredas by area | **38.4% of Ethiopia's land** |
| the people in them | **5.3%** |
| their mean Muslim share | **75%** |
| largest single woreda | Warder, 22,626 km², 58,035 people |

Spread those uniformly and two fifths of the map fills with an evenly spaced wash over the
Ogaden, where almost nobody lives — and because the Somali region is 98.4% Muslim, that wash
is **one colour** and it is the most visually dominant thing on the country. Exactly §8.2's
failure, on more ground than Kenya offered.

So the weight is the hexagon's own modelled population. It is a **population** weight and not
a religion one: nothing measures where a woreda's Orthodox sit inside it, so an Orthodox dot
and a Muslim dot are spread identically. Read the map as *"religion by woreda, drawn where
Ethiopians live"*, never as woreda-internal detail.

The join is on hex **centroids**, so a hex on a woreda line belongs wholly to one side and no
population is double-counted. 1,107 hexes (316,142 people, 0.249%) have a centroid in no
woreda — the border overrun and the lake holes — and are dropped.

## 4. The ratio band is wide on purpose, and that is the difference from Kenya

`ke_grid.py` asserts Kontur within **25%** of the census, because Kenya's census is 2019 and
Kontur is 2023. Copying that constant here would fail the build, and failing would be the
wrong answer:

```
Kontur 2023   126,399,684
census 2007    73,750,932
ratio               1.714
```

**Ethiopia's census is sixteen years older than the grid**, over which the country grew from
73.8 million to something near 125 million. A ratio near 1.0 would be evidence of a *bad*
download, not a good one. So the band is `[1.15, 2.10]`, derived from that growth and
bracketed loosely — it is there to catch a truncated file or a scrambled join, not to assert
a population.

**The per-woreda distribution is the check that actually has teeth**, and it is reported
rather than asserted:

```
median 1.72     quartiles 1.59–1.85     1–99% 0.69–2.91
```

A tight median matching the national ratio is what a correct assignment looks like; a
scrambled one would pair Addis Ababa's 2.7 million with a Somali woreda's 58,000 and scatter
the ratios over orders of magnitude. That is §9i's North Macedonia check. The outliers are all
explicable and none is a join failure — Durame town 5.9×, Dawe Qachen 4.6×, Gambella Zuriya
3.5× are fast-growing towns and frontier woredas, and Ethiopia's urban growth over those
sixteen years was extremely uneven.

Every one of the 738 woredas gets hexes (4 to 3,429 each), which is the assertion that matters
most: a woreda with none would fall back to an equal share over nothing and silently empty.

## 5. Not done

- **`et_woredas.gpkg` is written but not read by `countries.py`.** Placement is on the hex
  layer, so the polygons only feed `et_geo.py` itself — the same wiring as Kenya, Serbia,
  Lithuania and China. It is kept because it is the thing to draw outlines from if the
  viewer ever wants them.
- **The ADM1 and ADM2 layers are not extracted.** The woreda tier reconciles exactly, so
  there is no reason to draw a coarser one.
- **Kontur's global r6 file is already on disk** (`data/geo/kontur/`) and is not used:
  §9p's rule is that a per-country extract is the right one, and at 400 m it is also four
  levels finer than r6.
