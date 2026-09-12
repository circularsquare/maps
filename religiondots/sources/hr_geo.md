# Croatia — town/municipality boundaries

Built 2026-09-03 by `python sources/hr_geo.py`. Writes `data/geo/hr/hr_opcine.gpkg`
(556 polygons) and `data/geo/hr/hr_lookup.csv`.

Reuses both files Poland and Romania already downloaded — **Eurostat GISCO LAU 2021** and
the **LAU 2021 – NUTS 2021 correspondence workbook** — so `--fetch` is only needed on a
clean checkout. Croatia has 556 LAU features and `LAU_ID` is the national municipality
code, needing no decoding.

## 1. The method is Romania's

The DZS workbook has **no geographic codes**, only names, so `geo_id` is `"ŽUPANIJA|NAME"`
and the correspondence table bridges it: `(županija, name)` → `(NUTS3, name)` →
`LAU CODE` → polygon. The NUTS3↔županija pairing is **derived** by name-set overlap and
required to be a perfect bijection, not typed out — see `ro_geo.md` §2 for why.

It came out cleaner than Romania: **all 20 counties matched on every single name**, and
555 of 555 municipalities resolved with no elimination pass needed.

## 2. The one real trap: an en dash

Istria is officially bilingual and its municipalities carry Croatian–Italian double names.
DZS writes them with an **EN DASH** (U+2013) and Eurostat with a **HYPHEN-MINUS** (U+002D):

```
DZS       Bale – Valle        Poreč – Parenzo      Rovinj – Rovigno
Eurostat  Bale - Valle        Poreč - Parenzo      Rovinj - Rovigno
```

That is 20 of 555 units, all in one county, failing for a reason with nothing to do with
geography — the same category of error as Romania's `ş`/`ș`, and it would read the same
way if you did not know: like a boundary set from the wrong year.

`squash()` folds every dash to a hyphen and strips the spaces around it, which also fixes
`Murter-Kornati` vs `Murter - Kornati`.

## 3. Three names repeat across counties

`Otok`, `Privlaka` and `Sveta Nedelja` are each **two different municipalities** in
different counties. The county half of the key resolves all three — which is the reason the
NUTS3↔županija pairing is established *before* the join rather than matching on bare names
and hoping.

## 4. Zagreb is one polygon, and it is the cheapest fix outstanding

GISCO has all 556 municipalities including `01333 Grad Zagreb`. The census has 555 —
Zagreb appears only as its 17 gradske četvrti (see `hr.md` §2). So Grad Zagreb is the one
polygon with no census municipality, which is **expected rather than an error**, and
`hr_geo.py` asserts that the leftover set is exactly `{01333}`.

`hr_lookup.csv` therefore routes all 17 districts to `01333`, and `countries.py` sums them.
The result: **18.44% of Croatia in one 641 km² polygon.**

| capital | share in one unit | why |
|---|---|---|
| Tallinn | 33.1% | fixed — 8 districts published, boundaries found |
| Prague | 12.4% | fixed — 57 districts published, boundaries found |
| **Zagreb** | **18.4%** | **NOT fixed — 17 districts published, boundaries not found** |
| Bucharest | 9.8% | not fixable — no district data published |
| Warsaw | 4.7% | not fixable — no district data published |

**Zagreb is the only capital in the project where the split exists in the data and is
blocked by the geometry.** Everywhere else the constraint is the statistical office. What is
needed is one boundary file with 17 polygons. What was tried and failed:

- GISCO LAU stops at the municipality.
- Overpass, `admin_level` 9 and 10 inside `Grad Zagreb`: **zero relations**. OSM does not
  carry them at the obvious tagging (a third query was rate-limited at 429 and not retried).

Untried and plausible: DGU's **Registar prostornih jedinica**, `data.gov.hr`, and Zagreb's
own geoportal. Whoever picks this up gets a visible improvement to 18% of a country's map
for one download.

## 5. `hr_lookup.csv`

`geo_id → kod` for all 572 census units (555 municipalities plus 17 districts pointing at
Zagreb). Because the census carries no codes, **re-run `hr_geo.py` after any re-run of
`hr.py`**; `_hr_counts()` raises rather than silently dropping rows if the lookup goes
stale.
