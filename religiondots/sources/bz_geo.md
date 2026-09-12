# Belize — boundaries and placement

Wired 2026-09-07. 6 district polygons, 5,095 Kontur hexes.

## What was downloaded

```
COD-AB   https://data.humdata.org/dataset/8b2c9a50-82d5-4c31-9b4d-9f7bec2934c6/resource/
         113e7382-21bb-4cdb-a5c6-bde8217a0131/download/blz_admin_boundaries.shp.zip   500,158 B
Kontur   https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/
         kontur_population_BZ_20231101.gpkg.gz                                        426,091 B
```

**The shapefile bundle and not the geodatabase**, on §12's Chile rule: GDAL's OpenFileGDB
driver has been seen to open a `.gdb`, list its layers, report the right CRS and return ZERO
features while raising nothing. The bundle holds `blz_admin0/1/lines/points`; only
`blz_admin1` is read, and the feature count is asserted after the read either way.

## The join

**COD's ADM1 is the census's district tier exactly.** Six polygons, six census rows. Belize
has had the same six districts since 1882, so unlike Malawi (where a 28-district file would
have silently absorbed four cities) there is no tier ambiguity to check for and nothing to
guard beyond the count.

```
  census districts              6
  COD polygons                  6
  matched                       6
  census with no polygon        0
  polygons with no census       0
```

**No name differs.** All six match on `fold()` with no variants at all — the first country
here of which that is true. No alias table is written and none should be added later (§12: a
frozen list of renames goes stale in silence at the next release).

### The p-code check is not redundant here, and that is unusual

In most countries the name join and the p-code agree trivially. In Belize they encode
**different orders**: SIB prints north to south (Corozal first), COD codes alphabetically
(`BZ01` = Belize District). Five of six units would be mismatched by row position.

So `sources/bz.py` carries the pcode against each district *name* rather than numbering by
row, and this file asserts the same pairing from the boundary side — 6/6. If it ever fails,
every total in `bz.py` would still reconcile and only the dots would be in the wrong
districts (§9n's `TMA` lesson).

## Placement: Kontur, and the reason is area rather than lake

Six districts over **22,966 km²** is **3,828 km² per unit** — five times Jamaica's 785 and
the worst area-per-unit ratio on this map after Zimbabwe's provinces. And Belize is nothing
like uniformly habitable:

* **Cayo** contains the Chiquibul Forest Reserve and most of the Maya Mountains;
* **Toledo** is largely rainforest, with its people strung along the Southern Highway and the
  coast;
* **Orange Walk** runs north into the Rio Bravo conservation area;
* **Belize District** is 64,000 people in Belize City plus mangrove, lagoon and the cayes.

Spread dots evenly over those polygons and a large share of the country lands in forest
nobody lives in. Kontur removes the problem rather than patching it — an empty hex has no
population and takes no dots — so `water.py` is not doing this work.

```
  Kontur hexes                       5,463
  centroid outside every district      368   (28,202 people, 6.833%)  dropped
  kept                               5,095
  Kontur 384,535 vs census 397,483 — ratio 0.967
```

The 6.8% dropped is the **Mexican and Guatemalan border overrun** in Kontur's per-country
extract — Chetumal and Melchor de Mencos both sit on the line. **Ambergris Caye is inside**,
which matters because San Pedro Town is 15,456 people; the 1.02x ratio for Belize District is
the evidence, since losing the caye would have pulled it to ~0.88x.

### The per-district ratio, which is the check that matters

| district | hexes | ratio |
|---|---|---|
| Belize | 664 | 1.02x |
| Cayo | 1,454 | 0.99x |
| Corozal | 604 | 1.10x |
| Orange Walk | 901 | 0.80x |
| Stann Creek | 594 | 0.89x |
| Toledo | 878 | 0.94x |

**0.80x–1.10x is tight** for an 18-month vintage gap, and the level (0.967 nationally) does
not matter at all — Kontur is used only as a *within*-district weight, so only the shape is
load-bearing. Printed rather than asserted per unit: a district Kontur models badly gets its
dots on a worse surface, not the wrong number of them (§9t).

Orange Walk at 0.80x is the loosest and is worth one line: it is the district with the
largest Mennonite farming population, and a model built from building footprints and
night-time signal will tend to under-read dispersed low-density farmsteads. That biases
*where inside Orange Walk* its dots go, slightly toward the town — it does not change how
many Mennonites Orange Walk has.

## Not done

* **A finer placement tier.** SIB publishes population by city/town/village and it would make
  a plausible alternative weight, but it is a point list rather than polygons and Kontur
  already covers the same settlements with area. Not worth the work.
* **Water clipping beyond `water.py`'s default.** `scatter.py` clipped 181 of 5,095 hexes
  (1.15% of their area was sea) and left 2 units untouched under the `KEEP_WHOLE_ABOVE`
  rule. That is the normal coastal case and needed no special handling.
