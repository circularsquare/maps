# Ghana — boundaries

Built 2026-09-05 by `sources/gh_geo.py`. 272 polygons, from Ghana Statistical Service.

| | |
|---|---|
| source | GSS StatsBank Census Atlas, `statsbank.statsghana.gov.gh/assets/geofiles.zip` (26 MB) |
| vintage | 2021, the census's own |
| tiers | `Districts_261.zip` → 261 features; `Districts_271.zip` → **272** features |
| drawn | the 272 tier, as `data/geo/gh/gh_districts.gpkg` |
| join | by name, **272 of 272 both ways**, no spares, no ambiguity |
| licence | GSS open data, attribution expected |

---

## 1. The statistical office ships the boundaries for its own tables — and both tiers

One link on the Census Atlas page holds everything. Inside are two shapefiles, and the
second is **misnamed**: `Districts_271.zip` contains `District_272.shp` with **272**
features. That is exactly the 261 MMDAs with the six metropolitan districts replaced by
their 17 sub-metros — precisely the tier `sources/gh.py` selects out of the census cube. The
match is not luck; both files are cut for the same publication.

The general boundary sources are worse here on every axis, which is worth recording because
reaching for them first is the reflex:

- **geoBoundaries GHA ADM2** — 260 units, `boundaryYearRepresented` 2019, sourced from
  *USAID Ghana HPNO, Ghana Statistical Service*. One unit short of the census, two years
  stale, and no sub-metros at all.
- **COD / HDX** — not checked past this, because there was no reason to.

**Generalises:** before reaching for geoBoundaries or COD, look for a *census atlas*,
*geo-files* or *GIS* link on the statistical office's own dissemination platform. When it
exists it is the census-year vintage by construction, and it is cut to the tabulation
geography rather than to the administrative one — which is the distinction that cost the
Philippines a whole session (§9m). This one was a single link on a page already open.

## 2. The join is by name because there are no codes anywhere

Neither side has a code. The census cube publishes names only; the shapefile carries
`Label`, `District`, `Region` and nothing else. That is Romania's situation, and it works
out better here: both sides are the same office's own spelling of the same list, so a
conservative fold (case, accents, punctuation, and `&`→`and`) matches **272 of 272 in both
directions with no spares**, and the same fold matches 261 of 261 on the other tier.

Folded-name collisions are asserted against on both sides before matching, so a fold that
became too aggressive would fail rather than silently pair two districts.

## 3. The independent check is the REGION, and it is a real one

§12: a 100% name join is also what a subtly wrong name join looks like, so it has to be
verified against a quantity the join does not determine. There is no population column in
this shapefile, so the usual check is unavailable. The **region** does the job instead, and
does it unusually well:

- on the **census** side, a unit's region comes from **row order** — the cube lists a region
  header followed by its own districts, and *nothing else in the source says which region a
  district is in*;
- on the **boundary** side, it comes from an **attribute column**.

Those two are genuinely independent, and they agree on all 272. So the check confirms the
join and the positional parse in `gh.py` **at the same time** — it is exactly the check that
would have caught Romania's county-header bug, and it is available for free wherever a
boundary file carries a parent column.

## 4. One disagreement, and it is a typo in the boundary file

`AMA-Ablekuma South` is `Greater Accra` in the census and **`Greate Accra`** in
`District_272.shp`. One missing `r`, on one row, nowhere else in either file.

Resolved rather than folded away, on Chile's rule (§12): the unit's own name matched
exactly, and it is a sub-metro of Accra Metropolitan Area, which cannot be in another
region. It is listed by name in `RESOLVED_REGION` in `sources/gh_geo.py` and the list is
exhaustive, so a **second** disagreement fails the build instead of joining a pile of
known-harmless ones.

It is **not** repaired in `gh.py`: the normalised CSV reproduces the source (§2.4), and the
typo belongs to the boundary file, not to the table.

## 5. The two tiers are checked against each other

The 272 tier must be a *subdivision* of the 261 tier, not a different file that happens to
have more rows — the Budapest trap (§12), where a borrowed sub-layer agreed on total area
and still overhung the parent by tens of metres. Here both come from GSS and the total areas
agree to **0.0000%**, so the sub-metros tile their parents exactly and no metro dot can land
outside its metro.

## 6. Lake Volta had to be cut out, and Ghana is the first country where that mattered

`water.py` subtracts the **sea** and says plainly that inland water is a known gap: "where an
agency has NOT done it the lake will still take dots". Ghana is where that gap gets large.
**Lake Volta is 6,045 km², the largest reservoir on earth by surface area, it sits in the
middle of the country, and GSS runs its district boundaries straight across it.**

Measured on the first build: **397 of 30,750 dots, 1.29% of Ghana, in open water** — and
they are the most visible dots on the map, because nothing else is drawn there. For scale,
the figure that made `water.py` exist at all was 3.0% of the dots in the New York bbox.

`_drop_lakes()` subtracts **HydroLAKES v1.0** (Messager et al. 2016, CC BY 4.0, already in
`../data/` for the river maps) from the placement polygons. 99 of 272 districts are clipped,
2.60% of the country's district area comes out, and the dots in water go to **zero**. The
worst-hit units are real: Kwahu Afram Plains North loses 49.0%, Krachi West 42.2%, North
Dayi 40.8%, Asuogyaman 40.7% — these are the lakeside districts and half of each really is
reservoir.

Two things about how it is done:

- **It is placement, not counting.** Every dot stays in the district it was counted in; the
  clip changes only where inside that district it may land (spec §8.2).
- **It lives in `gh_geo.py`, not in `water.py`.** The gap `water.py` names is global; the
  knowledge that Ghana specifically needs it filled is local, and no other country on the
  map has been shown to need it yet. A `KEEP_WHOLE_ABOVE` guard mirrors `water.py`'s, at
  0.90 — nothing in Ghana comes close, so it is a tripwire rather than a working threshold.
  **If a second country needs inland water, this is the code that should move into
  `water.py`.**

## 7. Placement

Uniform inside each district. There is no population weighting on the Ghanaian side of this
map, so the empty half of a large northern district gets as many dots as the town in it.

That is a live improvement rather than a settled choice — Ghana's districts average 113,000
people and some northern ones are very large and very empty. Kontur Population (400m H3, on
HDX) or GHS-POP would do for Ghana what `ph_geo.py`'s barangay weighting does for the
Philippines. Not done, and the reason is only that the district polygons were enough to land
the country.
