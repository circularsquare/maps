# Suriname — boundaries and placement

Wired 2026-09-07. 62 ressort polygons, 5,689 Kontur hexes.

## What was downloaded

```
COD-AB   https://data.humdata.org/dataset/ab35e673-76f2-43e5-a6a4-a5b81f9e093c/resource/
         fd31ebfd-bf77-4b7d-902f-3ea3a9c2d7a2/download/sur_adm_2017_shp.zip           682,279 B
Kontur   https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/
         kontur_population_SR_20231101.gpkg.gz                                        466,810 B
```

## The join

**COD's ADM2 is the census's ressort tier exactly** — 62 polygons against 62 census columns
— and its ADM1 is the ten districts, so the two-level structure the census file implies is
present in the boundary file too. That matters, because the join has to use both levels.

### It must be a (district, ressort) join

Ressort names are **not unique**. There is a `Welgelegen` in both Paramaribo and Coronie and
a `Centrum` in both Paramaribo and Brokopondo. ABS disambiguates in its own header —
`Welgelegen (Par'bo)`, `Welgelegen (Coronie)`, `Brokopondo Centrum` — and COD carries neither
the parenthetical nor the prefix, so `fold()` strips both before comparing and the district
supplies the disambiguation instead.

The census file has no district column at all; `sources/sr.py` reconstructs it from
`district-profiel-census.xls`. See `sources/sr.md` §3.

### Four names do not fold, and the resolution is FORCED rather than chosen

```
  Wanica       census `Koewarasan`      COD `Kwarasan`
  Marowijne    census `Moengo Tapoe`    COD `Moengo Tapu`
  Brokopondo   census `Marchallkreeek`  COD `Marechallkreek`    (three e's — ABS's typo)
  Sipaliwini   census `Coeroeni`        COD `Coeroenie`
```

**No alias table is written for these**, and that is deliberate. §12's rule is that a frozen
list of renames goes stale in silence at the next release. What runs instead:

1. the exact fold, on `(district, name)`, matching **58 of 62**;
2. then, per district, **if exactly one census ressort and exactly one polygon are left
   over, they are paired by elimination** — and the run raises if a district ever has two of
   either.

That is a derivation rather than a lookup: it re-computes itself every run, it cannot rot,
and it fails loudly if a future vintage genuinely renames something. All four resolve, one
per district, and each is printed.

```
  census ressorten             62
  COD polygons                 62
  matched                      62
  polygons with no census       0
```

**The independent check** is that `sources/sr.py`'s hardcoded `RESSORTEN` pcodes reproduce
this derivation on all 62. Nothing else would catch a transposition — every total in `sr.py`
reconciles whichever polygon a ressort is paired with (§9n's `TMA` lesson).

ABS's spellings, typo included, are transcribed as printed and are what the map labels use
(§12, Chile: take names from the statistical source, not the boundary file).

## Placement: Kontur, and Suriname needs it more than anywhere

**163,820 km², and roughly 90% of the people live on the coastal strip.** The three interior
Sipaliwini ressorten — Kabalebo, Coeroeni, Boven-Coppename — are together larger than the
Netherlands and hold a few thousand people between them. Uniform scatter would put most of
Suriname's dots in rainforest.

It is also a small-unit problem at the other end: the twelve Paramaribo ressorten are a few
km² each and hold half the country. Kontur handles both ends without a special case.

```
  Kontur hexes                       5,773
  centroid outside every ressort        84   (19,358 people, 2.990%)  dropped
  kept                               5,689
  Kontur 628,041 vs census 492,829 — ratio 1.274
```

### The band is wide on purpose

This is **the largest vintage gap on the map**: a 2023 modelled grid against a **2004**
census, nineteen years, over which Suriname went from ~493k to ~620k people. The grid
*should* read about 25% high, so the assertion band is **[1.00, 1.60]** rather than something
centred on 1.0 — a band around parity would be the wrong shape of check even if it happened
to pass. Measured 1.274.

Kontur is used only as a **within-ressort** weight, so the level is irrelevant and only the
shape matters.

### §9ac's rule, checked before use

Saint Vincent's lesson is that **a population grid must be finer than the counting tier to
be worth using** — there, 43 enumeration districts were smaller than a single hex and Kontur
was measured, rejected and removed. Checked here and it passes: every one of the 62 ressorten
has hexes, from 4 to 700 each, and the median ressort is orders of magnitude larger than a
0.67 km² hex.

### Name where it is worst

The per-ressort Kontur/census ratio runs **0.06x to 2.86x**, which is wide, and the reason is
real rather than modelling error: nineteen years of Suriname's growth went to Wanica and the
Paramaribo fringe (Koewarasan 2.86x, Bigi Poika 2.67x) while the interior and old Paramaribo
did not move (Centrum 0.50x, Sarakreek 0.54x).

**Galibi is the worst-placed unit on this country and should be named.** 4 hexes and **43
modelled people** — a ratio of 0.06x. Galibi is the Kalina (Carib) village area at the mouth
of the Marowijne, and a model built from building footprints and night-time signal reads a
village of traditional houses under forest canopy as very nearly empty.

What that costs is precise and limited: Galibi's dots are still **inside Galibi** and still
the **right number**, because Kontur is only a within-unit weight — they simply sit on a
surface that carries no information (§9t: *a badly modelled unit gets its dots on a worse
surface, it does not get the wrong number of them*). The five thinnest units are otherwise
all urban Paramaribo, where near-uniform placement is close to right anyway.

## Not done

* **A better placement layer for the interior.** Suriname's Maroon and indigenous villages
  are strung along rivers and are exactly what a global population model handles worst. A
  village point layer would improve Sipaliwini, Brokopondo and Galibi materially; none was
  found in a form that joins to these units.
* **Water clipping beyond `water.py`'s default.** 42 of 5,689 hexes were clipped (0.19% of
  their area was sea), which is the ordinary coastal case.
