# citydirections

For each city, which direction are the rich neighbourhoods in?

US prototype. The point of this stage was to find out whether the question has a
stable answer at all before spending effort on global coverage.

## Run

```
python fetch.py     # ~30 MB, keyless
python build.py     # ~1 min
python plot.py      # ~1 min
```

## Data

All three inputs are public and need no API key. Note that the Census *API*
(`api.census.gov`) now requires a key — the bulk files under `www2.census.gov`
used here do not.

| file | what |
|---|---|
| `CenPop2020_Mean_BG.txt` | population-weighted centroid + population of every 2020 census block group |
| `acsdt5y2023-b19013.dat` | ACS 2019–23 median household income (table B19013), from the table-based summary files |
| `list1_2023.xlsx` | county → CBSA crosswalk |

Using block group *centroids* rather than polygons is what keeps this cheap: no
TIGER shapefiles, no geometry ops, one 10 MB text file for the whole country.

## Method

**Wealth field.** Within each metro, income is converted to a population-weighted
percentile rank, then z-scored. Ranks rather than dollars, because ACS top-codes
median household income at $250,001 and the distribution is right-skewed — in raw
dollars a handful of block groups would dominate every fit. Ranks also make the
numbers portable to other countries whose income data is shaped differently.

**Direction.** A population-weighted least-squares plane through the wealth field:

```
z ~ a + bx·X + by·Y            X, Y in km
```

The gradient `(bx, by)` points toward money. This needs **no city center**, which
is deliberate: "downtown" is the single hardest thing to define consistently
across cities, and in many Sunbelt metros the population-density peak is a dense
poor neighbourhood rather than the CBD. Anchoring on a mis-placed center would
inject a spurious direction into every arrow at once — the kind of error that
produces a beautiful map that is wrong.

`strength` is sd of income rank gained per 10 km travelled in the rich direction.
`r2` is the honest check on whether the metro is directional at all.

**Radial term.** A second, center-based model exists only to characterise rich-core
vs rich-suburbs, which genuinely does need a center (taken as the smoothed
population-density peak). Treat it as lower confidence than the direction.

**Trimming.** CBSAs are county-based and some sprawl for 200 km (Riverside–San
Bernardino reaches the Arizona border). Block groups are trimmed to the smaller of
50 km or the 97.5th percentile of population-weighted distance from the metro's
population-weighted centroid.

## Results

103 metros over 500k population.

- Directions match ground truth where it is well known: DC → **WNW**, Atlanta →
  **N**, Seattle → **ENE**, Detroit → **NW**, St. Louis → **W**, Dallas → **N**,
  San Antonio → **N**, Austin → **WNW**, Boston → **WSW**, Cincinnati → **E**,
  San Diego → **NNW**, Minneapolis → **WSW**.
- **93 of 101** CONUS metros have rich suburbs and poor cores. The eight
  exceptions include Seattle, the strongest.
- New York comes out at R² = 0.003 and Los Angeles at 0.024 — the model correctly
  declines to claim a direction where wealth is arranged in wedges and rings
  rather than a gradient. This is a feature; a method without an R² would have
  confidently reported "NNE" for New York.
- Aggregate tilt: the r²-weighted circular mean bearing is **≈330° (NNW)**, and it
  strengthens as you restrict to more directional metros (335° at R² ≥ 0.08).
  But circular concentration is only R̄ ≈ 0.19–0.27, so this is a mild tendency,
  not a rule. Among directional metros the quadrant split is W 15 / E 14 / N 13 /
  S 7 — the real pattern is that **south is rare**, not that west dominates.
- Only about a third of metros (35 of 101 at R² ≥ 0.08) have a meaningfully
  directional wealth field.

## Output

- `out/metros.csv` — one row per metro: bearing, compass, strength, R², radial term
- `out/field.csv` — the per-block-group wealth field, for other renderings
- `out/arrows_us.png` — one arrow per metro; hue = bearing, length = strength,
  opacity = R²
- `out/medallions.png` — the intra-city wealth field for the 16 largest metros,
  drawn as circular medallions. This is the preview of the bubbles-on-a-world-map
  version: the field is the artifact, arrows and bubbles are both just renderings
  of it.

In the medallions, **hue is wealth and opacity is population density** — smoothstep
over log density from 30 to 600 people/km², so built-up cores stay solid and each
city dissolves through its own exurbs instead of ending at a threshold. The rim is
feathered over the outer 7% of the radius so that a metro filling its circle
doesn't get a hard edge where the 50 km trim happened to land.

Note the medallions re-rank *after* smoothing, so every city spans the full colour
ramp. That maximises legibility of each city's internal pattern but means
medallions cannot be compared to each other for degree of segregation.

## Meta Relative Wealth Index — tried, does not work

`build_rwi.py` runs the same estimator over Meta's RWI (2.4 km cells, 93 LMICs,
one 85 MB download from HDX) and produces 624 cities. **The output should not be
used.** RWI does not resolve within-city wealth, and where cities have low-density
affluent districts it is *inverted*. Mean RWI within 3 km of known neighbourhoods:

| city | wealthy district | RWI | poor district | RWI |
|---|---|---|---|---|
| Dar es Salaam | Masaki / Oyster Bay | **0.72** | Tandale | **1.59** |
| Kinshasa | Gombe | **0.64** | Masina | **1.40** |
| Lagos | Ikoyi / Victoria Island | 1.29 | Mushin | **1.36** |
| Nairobi | Karen / Runda | 1.07 | Kibera | **1.10** |
| Bogotá | Chapinero | 1.09 | Kennedy | **1.21** |
| Mexico City | Polanco | 1.38 | Iztapalapa | 1.24 |

Dar es Salaam is the worst case precisely because it looked the best: it has the
highest directional R² (0.32) of any large city, and its arrow is backwards.

Aggregate evidence across all 624 cities, against the 103 US metros as a control:

| | RWI cities | US metros |
|---|---|---|
| "rich core" (radial < 0) | **98.1%** | 8.7% |
| median radial coefficient | **−0.996** | +0.242 |
| median R², direction only | 0.068 | 0.048 |
| median R², adding log(distance from centre) | **0.355** | 0.144 |

So RWI's within-city variance is almost entirely a radial urbanisation gradient,
with a suspiciously uniform coefficient. What residual direction remains is *not*
a centre-offset artifact — the fitted bearing sits a median 86° from the bearing
to the settled-mass peak, which is essentially random — it simply does not track
wealth.

This is not a defect in RWI. It is built and validated for relative poverty at
aggregate and rural scale, where denser and better-connected does mean richer.
Inside a metro that relationship inverts, because urban affluence looks like
low-density leafy plots that read as *less* developed to satellite and
connectivity features, while asset ownership saturates across the built-up core.
Intra-urban gradients are outside its design envelope.

One real bug was found and fixed along the way, worth remembering: the cities
list carries satellites as separate entries (Soacha, Ecatepec, Chimalhuacán), and
these are overwhelmingly the *poor* peripheries. Letting them claim their own
cells stripped the low end out of the parent metro and silently reversed its
gradient — Mexico City came out a confident ENE. Absorption now runs to the full
parent radius. This is the same failure as Miami's CBSA, inverted: over-splitting
rather than over-merging.

## Going global — what is actually left

There is no one-download global layer for this. It is a country-by-country build.

- **Do not** use global gridded GDP (Kummu, Chen, LitPop): downscaled from
  national totals by population × nighttime lights, so within a city it encodes
  where the CBD and the industry are.
- **Do not** use RWI, per above.
- **Middle income, excellent data:** Colombia DANE *estrato socioeconómico* — a
  legally defined 1–6 wealth stratum per city block, which would nail Bogotá.
  Brazil IBGE setores censitários, Mexico INEGI AGEB, South Africa StatsSA small
  area, Argentina radio censal, Chile, Peru.
- **High income:** France Filosofi (200 m grid) and the Netherlands (100 m) beat
  anything the US has; then Spain INE income atlas, UK LSOA, Sweden DeSO,
  Canada DA, Australia SA1.

City extents should come from the GHSL Urban Centre Database rather than a
point-plus-radius list.

Because direction is scale-free and centre-free, mixing sources across countries
is fine as long as each city is internally consistent — but every source needs
the same neighbourhood spot-check RWI just failed, before it goes on a map.
