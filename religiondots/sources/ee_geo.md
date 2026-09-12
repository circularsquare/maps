# Estonia — the finest complete cover, with Tallinn broken up

Built 2026-09-03 by `python sources/ee_geo.py --fetch`. Writes
`data/geo/ee/ee_finest.gpkg` (86 polygons) and `data/geo/ee/ee_replaced.csv`.

## 1. Why not GISCO

Estonia is in the same Eurostat LAU 2021 file Poland and Romania use, with **79 features
that join on the EHAK code with no decoding at all** — by far the easiest join available.
It is not used.

**Tallinn is 33.05% of Estonia in a single 159 km² polygon.** That is three times worse
than Bucharest (9.8%), seven times worse than Warsaw (4.7%), and worse even than Prague
(12.4%). A third of the Estonian map would have been one uniform smear across the capital,
which is precisely the failure spec §8.2 exists to avoid.

Statistics Estonia publishes RL21452 for **Tallinn's 8 linnaosad** as well as for the 79
municipalities, so splitting it is **measured data, not an allocation** — the same
situation as Czechia's 142 city districts, and the opposite of Poland and Romania, where
GUS and INS publish nothing below the capital and it had to stand as one polygon.

Result: 78 municipalities + 8 Tallinn city districts = **86 units**.

## 2. Sources — Maa-amet (Estonian Land Board), EPSG:3301

```
https://geoportaal.maaamet.ee/docs/haldus_asustus/omavalitsus_shp.zip     79 municipalities
https://geoportaal.maaamet.ee/docs/haldus_asustus/asustusyksus_shp.zip    4,714 settlement units
```

### `linnaosa_shp.zip` is §5a's rule in its most literal form

There is an obvious third URL, `.../haldus_asustus/linnaosa_shp.zip`, and it looks right.
A `HEAD` returns **HTTP 200**. What it actually serves is a **282-byte PNG** — a picture of
an error message. Not a 404, not an error status, not even text: an image.

`ee_geo.py` therefore checks `zipfile.is_zipfile()` and a minimum size after every
download, rather than trusting the status code. The directory listing is 403, so there is
no way to enumerate the real filenames either.

The city districts are inside `asustusyksus_shp.zip`, as the rows with `TYYP == 6` and
`OKOOD == 0784`.

## 3. The keys

The PxWeb place code is a 14-character concatenation of EHAK codes:

```
00370141000001   county 0037 + municipality 0141 + filler    -> OKOOD = code[4:8]
003707840000L4   county 0037 + municipality 0784 (Tallinn)   -> OKOOD = code[4:8]
003707840176L6   county 0037 + Tallinn 0784 + district 0176  -> AKOOD = code[8:12]
```

so a municipality is keyed on `code[4:8]` and a city district on `code[8:12]`. EHAK codes
are unique across both kinds; the script checks that rather than assuming it, and fails on
a collision.

## 4. Four EHAK codes were retired between the census and the boundary file

Maa-amet's files are stamped **2024-12-01** and the census is **2021**. Normally that is the
spec §8.1 hazard, and here it half-bites:

The 79 municipalities are the same 79 — Estonia's 2017 haldusreform cut 213 to 79 and
nothing has merged since — but **four of them changed code**:

```
census 0142 -> polygon 0145   Antsla vald
census 0514 -> polygon 0515   Narva-Jõesuu linn
census 0735 -> polygon 0736   Sillamäe linn
census 0855 -> polygon 0857   Valga vald
```

The old codes appear **nowhere** in the current Maa-amet file — not as a municipality, not
as a settlement unit — which is how we know this is a real code change and not a level
mix-up.

`ee_geo.py` re-joins those four **by name**, and only where exactly one candidate is left
unmatched on each side, so nothing is guessed. The alias map is **derived, not hard-coded**,
because the next Maa-amet release may retire different codes and a frozen list would go
stale in silence — the same reasoning as `ro_geo.py`'s derived NUTS3↔judeţ pairing.

Name matching needs a small language bridge: Statistics Estonia writes the English unit
type (`Antsla rural municipality`, `Narva-Jõesuu city`) and Maa-amet writes the Estonian
one (`Antsla vald`, `Narva-Jõesuu linn`), so neither the whole string nor a shared suffix
matches. `bare()` strips both sets of type words.

## 5. Placement

The 86 units are both the count layer and the placement layer. Median unit is about 5,000
people aged 15+.

The largest remaining polygons are rural and empty rather than urban and full, which is the
right way round:

```
Saaremaa vald       2,719 km2      Lääne-Nigula vald   1,448 km2
Alutaguse vald      1,459 km2      Viljandi vald       1,372 km2
```

Saaremaa is an island municipality of 31,000 people over 2,719 km², so its dots spread
thinly over the whole island — acceptable, and the opposite of the Tallinn problem.

## 6. Capitals, across the four Central and Eastern European countries now on the map

| capital | share of country in one unit | split available? |
|---|---|---|
| Tallinn | 33.1% | **yes** — 8 linnaosad, published |
| Prague | 12.4% | **yes** — 57 city districts, published |
| Bucharest | 9.8% | no — 6 sectors exist, no religion published |
| Warsaw | 4.7% | no — 18 dzielnice exist, no religion published |

The pattern is that Central and Eastern European capitals are single municipalities, and
whether the map can do anything about it depends entirely on whether the statistical office
publishes below that level. Two of four do.
