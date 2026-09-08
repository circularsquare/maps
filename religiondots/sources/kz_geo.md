# Kazakhstan — boundaries and placement for the 17 regions of the 2021 census

Built 2026-09-07 alongside `sources/kz.md`. Two modules: `kz_geo.py` (the units) and
`kz_grid.py` (the placement layer).

| | |
|---|---|
| units | OCHA **COD-AB Kazakhstan** (`cod-ab-kaz`, UNHCR from OpenStreetMap, 2023), shapefile bundle, 5.3 MB, one GET from HDX |
| tier | **ADM1, dissolved from 20 back to 17** — see §1 |
| placement | **Kontur 400 m population hexagons**, `kontur_population_KZ_20231101`, 14.9 MB gzipped → 175,118 hexes kept |
| join | **17 names**, English COD ↔ Russian census, written out and asserted 1:1 both ways |
| result | **17 / 17 matched, 0 unmatched either way** |

---

## 1. The boundary file is the wrong vintage, and is put back

**In 2022 Kazakhstan created three new oblasts**, so COD's ADM1 has **20** regions and the
2021 census has **17**. Spec §8.1: boundaries must be the vintage the data was published on.

The reform is the cleanest kind there is — three new oblasts carved out of three existing
ones, nothing else touched:

```
    Abay Region    <- East Kazakhstan Region      Ulytau Region  <- Karaganda Region
    Jetisu Region  <- Almaty Region
```

So the 2021 map is recovered by **dissolving three pairs of whole ADM1 polygons**. No polygon
is cut, nothing is apportioned by area or population, and the result is exactly the union of
features the file already contains. That is a much weaker operation than a general boundary
reconstruction and is the reason this is safe.

### Two independent checks that the merge is the right one

**ADM2 counts.** COD ships **218** ADM2 polygons and the census publishes **218** level-2
units, because the reform regrouped rayons without splitting any. Every ADM2 is assigned to
its post-merge parent and the counts must add up:

```
    Abay 10 + East Kazakhstan 11 = 21        Jetisu 10 + Almaty 10 = 20
    Ulytau 5 + Karaganda 13 = 18             all 218 rayons land in exactly one of the 17
```

If the reform had moved a single rayon anywhere unexpected, this fails.

**Kontur, in `kz_grid.py`, and it is the one that would really catch a mistake.** A region
dissolved onto the wrong parent would have its modelled population compared against the wrong
polygon's population grid, and East Kazakhstan and Karaganda are nothing alike in density. The
per-region band is therefore doing double duty. It comes back **0 of 17 outside a factor of
1.8**, against a shuffled median of 5.

## 2. The name join

Seventeen pairs, written out rather than derived. A transliteration rule cannot get
`Северо-Казахстанская область` from `North Kazakhstan Region`, and pretending otherwise on a
list this short would be worse than the list. Every COD name must find a census row and every
census row must be claimed exactly once; the module raises on either failure.

`geo_id` **is** the KATO code — Kazakhstan's official administrative-territorial classifier,
straight from the census workbook (`110000000` = Akmola). So `kz_lookup.csv` is an identity
mapping, which is unusual here and is a good sign: the statistical source brought its own
stable key.

## 3. Why the grid matters more here than anywhere

**17 regions over 2.7 million km², at 7 people per km².** Karaganda region alone is 428,000
km² — bigger than Germany and Poland together, and larger than every country on this map
except Russia, China, India and Brazil. The population sits in a ring around the edge of the
country plus Astana and Almaty; the middle is the Betpak-Dala desert and is genuinely empty.

An equal-share wash would scatter Karaganda's dots evenly across that desert. §8.2's trick
(fine units make a population layer unnecessary) is as far from applying as it gets.

### The numbers

| | |
|---|---|
| national ratio | Kontur 19,603,341 / census 19,186,015 = **1.022** (2023 grid, 2021 census) |
| BAND | **0** of 17 regions outside a factor of 1.8, against a shuffled median of **5** |
| CORRELATION | r = **0.8699** on log populations, **0** of 2,000 shuffles reach it |
| dropped | 840 hexes, 57,259 people (**0.291%**) whose centroid falls outside every region |

**The three regions furthest from 1.0 are all the same effect and it is the familiar one.**
Astana reads 0.64 and Almaty 0.71 — Kontur under-models the two big cities — while Akmola
region, which *surrounds* Astana, reads **1.54**. That is a city smeared into its hinterland,
the same pattern Nepal's Rohini/Siddharthanagar pair shows (`sources/np_geo.md` §4), and it
pools away: Astana plus Akmola together come to 1.01.

**It changes no count** (§8.2). The grid is a within-region weight, so Astana receives exactly
its modelled population and only the shape inside the boundary comes from Kontur.

## 4. What this country's placement really means

Worth stating plainly, because Kazakhstan stacks two different kinds of estimate:

1. **Which religion** a dot is comes from the model (`sources/kz.md`) — the region's ethnic
   composition applied to the national religion-by-ethnicity table.
2. **Where in the region** the dot sits comes from Kontur — the population grid.

Neither says anything about where adherents of a given religion live *within* a region. A
cluster of Orthodox dots over Petropavl means "North Kazakhstan region is 49% Christian, and
this is where North Kazakhstan's people are" — nothing more.
