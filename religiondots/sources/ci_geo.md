# Côte d'Ivoire — région boundaries and the placement grid

**Built 2026-09-07.** `sources/ci_geo.py`. Counts: `sources/ci.md`.

Writes `data/geo/ci/ci_regions.gpkg` (33 units), `ci_hexes.gpkg` (142,653 Kontur hexes) and
`ci_lookup.csv`.

## 1. geoBoundaries fits, which is not the usual outcome

`gbOpen/CIV/ADM2` is **33 polygons** against the census's **33 units** — 31 régions plus the
autonomous districts of Abidjan and Yamoussoukro. ADM1 is the 14 districts, which is the tier
this map deliberately does not draw (`ci.md` §5).

That is a better start than most countries here have had: §9m rebuilt the Philippines' BARMM
from tagged barangays, §9r disqualified geoBoundaries CHN outright, §9ah found it missing an
entire Korean county, §9at found *both* available boundary sets wrong for the Cayman Islands.
Here the count matches on the first try.

## 2. The name join, and why an alias table is safe this time

Accent- and punctuation-folding matches **30 of 33**. The other three:

| census | geoBoundaries |
|---|---|
| `District D'Abidjan` | `District Autonome D'Abidjan` |
| `District De Yamoussoukro` | `District Autonome De Yamoussoukro` |
| `La Mé` | `Me` |

**None is ambiguous.** There is exactly one Abidjan, one Yamoussoukro and one Mé, and no other
candidate on either side is close to any of them. So an explicit alias table is enough here,
where §9af needed a *spatial* check against 183 Mauritian place names because two of them
genuinely split the same ground differently.

The join is still asserted rather than trusted: **any census unit that matches zero or more
than one polygon raises**, and so does any polygon being claimed twice. The alias table
cannot silently rot into a wrong pairing.

## 3. Kontur, and the région it exists for

Côte d'Ivoire is 322,463 km² over 33 units and the split is very uneven:

| | area | people | density |
|---|---:|---:|---:|
| District autonome d'Abidjan | 2,153 km² | 6,321,017 | 2,936/km² |
| Bounkani | 21,800 km² | 427,037 | 19.6/km² |

**The 10 largest régions are 47.6% of the land and 28.9% of the people.** An equal share per
polygon would spread the north's dots evenly across country that is largely empty Comoé
National Park — **and Bounkani is exactly where `Animiste` is 24.7%**, ten times the national
figure. So a uniform fill would smear the single sharpest thing this country has to show
across the largest and emptiest polygon in it. That is a stronger argument for a grid than
"the map looks wrong".

```
kontur_population_CI_20231101.gpkg.gz   11,706,001 bytes -> 28,127,232
143,307 hexes, 28,920,846 people
654 hexes (79,427 people, 0.275%) have their centroid in no région — border
overrun between Kontur's extract and geoBoundaries; dropped.
142,653 kept; every région gets 902–13,388 hexes.
```

Joined on hex **centroids**, so no hex is split across two régions. The §8.2e floor is met
with room to spare: the smallest région is 2,113 km² against a 0.67 km² hex, and the measured
minimum is 902 hexes per unit.

## 4. THE §9av CHECK, RUN ON A SECOND COUNTRY — and it comes out the other way

§9av found that the Central African Republic's Kontur extract is close to a flat rescale of
the census being drawn, because **CAR has had no census since 2003** and Kontur had nothing
newer to build from. The consequence recorded there was general: *a population grid's
agreement with a census is evidence only when the grid had a different census to be built
from.* Côte d'Ivoire is the first test of that.

| within ±5% of the median ratio | |
|---|---:|
| **Central African Republic** (177 communes, 2003 census) | **78.5%** |
| **Ethiopia** (738 woredas, 2007 census) | 34.0% |
| **Côte d'Ivoire** (33 régions, 2021 census) | **27.3%** |

**Côte d'Ivoire is the widest of the three**, which is what a 2021 census against a 2023 grid
should look like — two years apart, and Kontur demonstrably not built on this table. The
national ratio is **0.985**, and the per-région quartiles are 0.91–1.12 against CAR's
1.47–1.52.

So the rule survives its first test in the direction that matters: **the tightness is
diagnostic of dependence, not of quality.** `ci_geo.py` measures it and prints the CAR and
Ethiopia figures beside it on every run, and warns if it ever exceeds 60%.

The largest disagreements are ordinary and are reported rather than asserted:

| ratio | région |
|---:|---|
| 1.45x | Guemon |
| 1.31x | Marahoué |
| 1.31x | N'Zi |
| 1.22x | Goh |

Guémon and the west are where the post-2011 returnee movement was largest, so a 2023 grid
sitting above a 2021 census there is plausible rather than suspicious. Since only the
within-région shape is read, none of it affects a count.

## 5. Water

`scatter.py` clips 299 of 142,653 placement polygons, 0.08% of their area being sea, and
leaves **7 units that lose over 95% to the sea** unclipped — the lagoon settlements of the
Abidjan and Grand-Lahou coast, where the census's people genuinely live on water. That is
`water.py`'s standing behaviour and the right one here: Côte d'Ivoire's southern lagoons are
inhabited, and clipping them would move real people inland.
