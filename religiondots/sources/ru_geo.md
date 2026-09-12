# Russia — boundaries and placement

Built 2026-09-05 by `sources/ru_geo.py`. Writes `data/geo/ru/ru_subjects.gpkg` (**83** ADM1
polygons) and `data/geo/ru/ru_grid_3km.gpkg` (108,632 population hexes). `data/` is
gitignored, so this file is the record.

**It was 79 subjects until `ru_fill.py` landed the same day**, and the change is worth
stating as a rule rather than a fact about Russia: the geography layer now carries every
subject and the COUNTS decide what is drawn. A boundary file that silently omits a region
makes a missing count look like missing land, which is the one thing a dot map must not do.

## 1. The units: geoBoundaries ADM1, and it needed no editing at all

`https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RUS/ADM1/geoBoundaries-RUS-ADM1.geojson`
— 59,162,098 bytes, ODbL, OSM-derived, 2017 vintage.

**83 features, and no Crimea or Sevastopol.** That is the internationally recognised
composition and it is also exactly what Arena surveyed, so the two sources agree about what
Russia is without a single line of editing — which is not something to take for granted with
this country, and is the reason this boundary set was preferred over anything Rosstat-derived.

**It carries `shapeISO` for all 83, fully populated, and that is the whole join.** Arena
gives English region names, Wikipedia gives different English region names, geoBoundaries
gives a third set — "Tyva Republic", "Tuva", "Tuva"; "St.Petersburg", "Saint Petersburg",
"Saint Petersburg"; "Sakha (Yakutia) Republic", "Sakha", "Sakha Republic". None is derivable
from another, so `sources/ru.py` holds **two explicit dictionaries**, `ARENA_TO_ISO` (79
entries) and `WP_TO_ISO` (83), written out in full and asserted exhaustive. No similarity
matching anywhere.

This is spec §9j's lesson applied before it could bite: Sri Lanka's code join matched 96% of
the country and put 762,824 people in the wrong district in silence. With 79 units a hand
mapping costs twenty minutes and cannot fail that way — an unknown name stops the run.

**The four dropped units fall out of the ISO codes independently**: `RU-CE`, `RU-CHU`,
`RU-IN`, `RU-NEN` — Chechnya, Chukotka, Ingushetia, Nenets. That is the same four the Arena
workbook is missing, derived from a completely different direction, which is a free check
that the mapping is not quietly wrong.

## 2. The placement layer, and why Russia needs one more than anywhere else

**Kontur Population, H3 r6 (~3km), 2023-11-01**, from HDX under CC BY:
`kontur_population_20231101_r6.gpkg.gz`, 185,340,844 bytes compressed, 509 MB open,
2,016,971 hexes globally in **EPSG:3857**.

spec §8.2's usual trick — spread a unit's dots equally over a finer layer an agency built to
a population target — has nothing to work with here. The counts are at federal subject: a
mean of 1.8 million people over a mean of 200,000 km². **Sakha alone is 3.08 million km²
with about a million people living along four rivers.** Spread evenly its dots would cover
an area the size of India, which is the Brazil-município problem (§8.2a) at maximum severity.

So the weight is a measured population surface, per §8.2d: where a fine layer carries a
population, use it and stop reasoning about proxies.

**Why r6 and not r8 (400m)** — the same argument `de_grid.py` makes for Germany's 1km over
its 100m grid. Russia draws 134,821 dots at 1:1,000 and r6 gives 107,240 populated hexes, so
the grid is already the same order as the dots and is not what limits the picture. r8 is a
2.4 GB download for ~13× more cells than can be shown. And the part that settles it: **every
dot inside a subject is drawn from one distribution**, because a subject is all Arena
measures. A finer grid would place the same undifferentiated mixture more precisely and tell
the reader nothing.

**Kontur ships only populated hexes**, so the 107,240 cover about 3.9 million km² of a
16.4 million km² country. That is the single most useful thing this layer does here: the
other three quarters of Russia is empty, and now the dots know it.

## 3. The join, and the check that it is right

| step | result |
|---|---|
| hexes in the two bboxes | 326,663 (645.1M people — they reach deep into Europe and Asia) |
| centres outside Russia's 83 subjects | 218,031, dropped |
| hexes kept | **108,632** |
| boundary hexes clipped to their subject | 9,094 |
| left whole | 99,538 |
| clips that came out empty | 0 |

**Two bboxes, because Chukotka crosses 180°.** The first build used one, and said in its own
docstring that the antimeridian needed no handling *because Arena does not cover Chukotka* —
which stopped being true the moment the four missing subjects were filled. The second box
(180°W–165°W) finds 269 hexes. A bbox justified by what the data happens to contain is a
bbox that goes stale when the data changes, and this one did, within a day.

Two approximations, both the ones `de_grid.py` names: a hex is assigned to the subject
containing its **centre**, so a boundary hex's people may belong to either side; and each hex
is then **clipped** to its subject so a dot cannot cross a border or the coast.

**The clip is cheap because interior hexes are tested first.** `shapely.prepare()` on the
subject, then `contains_properly` over its hexes — 98,405 of 107,240 need no intersection at
all, which matters when Sakha's polygon has 227,194 vertices. Doing the intersection
unconditionally is the version of this that runs for an hour.

### The population check — a relationship, not an equality (spec §9i)

Kontur's surface is a 2023 model built from GHSL and building footprints; the counts come
from a 2021 enumeration. They measure different things, so demanding equality would either
fail on every honest difference or be loosened until it detected nothing. What a correct join
looks like is every subject's ratio in a tight band; a scrambled one pairs cities with
tundra and spreads it over orders of magnitude.

```
  totals:  census 142,590,384   kontur 140,297,347   (0.984x)
  per-subject ratio: median 1.015, min 0.695 (RU-KAM), max 1.272 (RU-LEN)
  every subject inside a factor of two
```

0.984× nationally on two independent constructions is better agreement than this check
needed. The two outliers are both explicable and neither is a join error: **Leningrad Oblast
at 1.272** is the St Petersburg commuter belt, where a built-up surface reads population that
the census registers inside the city; **Kamchatka at 0.695** is a thinly settled region where
GHSL has little built-up area to disaggregate onto.

## 4. Water

`water.py` clipped 1,281 of 107,240 placement polygons, removing 0.02% of their area, with
122 geometries repaired. One unit lost over 95% to the sea and was left unclipped, per
water.py's own rule.

The figure is small because Kontur hexes are already population-derived and there are few of
them over open water. **Inland water is not subtracted** — Ghana needed HydroLAKES cut out
because GSS runs districts across Lake Volta; Russia has Baikal, Ladoga and the Rybinsk
Reservoir inside its subjects, but a hex only exists where Kontur found people, so almost
nothing is over them in the first place. Worth re-checking if the placement layer is ever
changed to something that covers area rather than population.
