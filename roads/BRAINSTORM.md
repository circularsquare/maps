# roads — US road traffic map (brainstorming)

A road-traffic analog of [`riders/japanriders`](../riders/japanriders/) (line
thickness ∝ rail throughput / 輸送密度). For roads the equivalent per-segment
metric is **AADT — Annual Average Daily Traffic** (vehicles/day on a segment).
The fit is actually *cleaner* than rail: AADT is a single measured/estimated
figure per physical segment, no origin–destination or through-running ambiguity.
Same "static-thickness, one number per segment" shape japanriders already uses.

Status: **scoping only.** No build yet. Goal of this doc is to record the data
landscape, what already exists online, and the candidate directions.

---

## Data sources (US AADT)

Three tiers, trading coverage vs. granularity vs. ingestion effort.

1. **FHWA HPMS** — *most granular, national.* The
   [Highway Performance Monitoring System](https://www.fhwa.dot.gov/policyinformation/hpms.cfm).
   AADT per segment for the **entire Federal-Aid System** (interstates +
   arterials + collectors + NHS), ~1M miles, hundreds of thousands of segments.
   Carries **truck AADT** separately. Released **by state** as ArcGIS feature
   services / shapefiles ([FHWA shapefile page](https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm),
   services there top out at 2018); newer **2024 full national** release as
   GeoJSON via [data.transportation.gov](https://data.transportation.gov/Roadways-and-Bridges/Highway-Performance-Monitoring-System-HPMS-/jc5k-rzm8).
   Tradeoff: big by-state stitch; full coverage on higher-order roads, sampled
   on collectors.

2. **BTS NTAD "AADT"** — *cleanest start.* A single national AADT layer for the
   **National Highway System** (interstates + major US/state routes, ~230k mi),
   one download, one schema, GeoJSON / shapefile / file-gdb, from the
   [BTS geospatial hub](https://data-usdot.opendata.arcgis.com/). Coarser than
   HPMS (NHS only, no minor arterials/collectors) but trivial to ingest. This is
   the analog of japanriders' `build.py` national base — ideal to prototype on.

3. **State DOT layers** — *finest, most work.* Every state DOT publishes its own
   AADT GIS (segment layers + count-station points), often finer/more current
   than HPMS. ~50 schemas to harmonize. Analog of japanriders' census stitch.
   [Caltrans](https://gisdata-caltrans.opendata.arcgis.com/datasets/d8833219913c44358f2a9a71bda57f76_0/about),
   [NCDOT](https://connect.ncdot.gov/resources/State-Mapping/Pages/Traffic-Survey-GIS-Data.aspx),
   [NYSDOT TDV](https://www.dot.ny.gov/tdv),
   [Maryland iMap](https://data.imap.maryland.gov/maps/77010abe7558425997b4fcdab02e2b64).
   [Cubit's list of 52 sources](https://blog.cubitplanning.com/2019/07/52-sources-traffic-counts-aadt-data/).

### Time granularity

- **Multi-year (trends):** YES, network-wide. HPMS published annually back
  through the 2010s → per-segment growth / COVID crater + recovery. Caveat:
  segmentation & Route IDs shift year to year, so joining a segment to its past
  self is fiddly.
- **Hourly / time-of-day:** only at **points**, not the network. The
  [FHWA TMAS / Continuous Count Stations](https://data.transportation.gov/stories/s/TMAS-Data-Program/katt-tac5/)
  network reports hourly + monthly + day-of-week at a few thousand permanent
  sensors. Downloadable as GIS
  ([stations](https://data-usdot.opendata.arcgis.com/datasets/usdot::travel-monitoring-analysis-system-stations/about),
  [volume](https://geodata.bts.gov/datasets/5a9462b519854ec6a2334b3c0bdfc3c1)).
- **Peak-hour structure (no time series needed):** HPMS *sample-panel* segments
  carry **K-factor** (peakiness — peak-hour share of AADT) and **D-factor /
  Dir_Factor** (directional lopsidedness of the peak; 50% balanced → ~75% strong
  one-way tide). Sparser than AADT (sample only) but national + segment-level.
  [HPMS traffic requirements](https://www.fhwa.dot.gov/policyinformation/tmguide/tmg_2013/hpms-requirements.cfm).

---

## What already exists online (don't rebuild this)

- **OpenHPMS** ([openhpms.com](https://openhpms.com/)) — national interactive
  explorer of 2024 HPMS: filter AADT / freight / speed / lanes by
  state/county/route. Most complete national viewer, but an *analytical/filtering
  tool*, not an aesthetic picture.
- **Esri ArcGIS Living Atlas HPMS layer** /
  ["AADT along US highways"](https://www.arcgis.com/home/item.html?id=7b3a42009dbc432493ecd19f34695d3f)
  — national HPMS network as a ready-made layer; **renders quite well** (segments
  colored + sized by AADT, click for AADT/AVMT popup). The strongest existing
  thing — close to a finished picture, not just a tool. Honest gut-check: the
  "just make it prettier" angle is weak against this.
- **Maptitude/Caliper AADT infographic** — single static national image styled
  by AADT. Closest in *spirit*, but static + commercial showcase.
- **State DOT viewers** — one per state, click-a-segment, utilitarian.

**Verdict:** raw national AADT-on-a-map is well covered (esp. Esri). The open
niche is a map with a **point of view** — a derived/normalized lens that reveals
something raw volume doesn't, framed as a shareable object rather than a query
tool. ("Putting AADT on a map" is not novel; the *thesis* is.)

---

## Candidate directions

The Esri map shows raw volume well, so the differentiator has to be an angle it
doesn't take. Strongest-to-weakest by my read:

1. **Flow-orientation field** ⭐ — *needs zero extra data.* Every segment's
   geometry gives a **bearing**; AADT gives **strength**. Color by bearing
   (cyclic hue ramp), intensity/width by AADT → the *grain* of American movement
   as a field: N–S Atlantic spine, E–W transcontinentals, radial metro fans,
   Plains grid-grain. Wind-map / flow-field aesthetic. Buildable from data in
   hand; nobody renders it. Most japanriders-shaped: one elegant static national
   object showing something real that no existing map shows.
   - Catch: this is *axis* (orientation), not *signed* direction. D-factor gives
     imbalance magnitude but not which compass direction wins, so true signed
     flow isn't network-wide available.

2. **Daily tide / "breathing" map** — the literal which-way-is-it-flowing-now
   version. Needs signed direction → only clean at TMAS continuous-count points.
   Animated national map, morning rush sweeps inbound, reverses at night. More
   spectacular, more fragile (point-based, not network-wide).

3. **Freight / truck flow** — interstate truck-AADT only. Clean (freight ≈ ~20
   corridors, no hairball), under-seen, strong thesis ("how America moves its
   goods"). Truck AADT already in HPMS.

4. **Congestion / per-lane** — normalize AADT by lane count → which roads are at
   their limit vs. just big. Reveals stress not size; denser/harder to render
   cleanly.

5. **Change over time** — multi-year AADT: growth corridors, COVID crater +
   recovery. Adds the time dimension Esri lacks; depends on clean multi-year
   joins.

6. **Art-object render** (raw AADT, just framed as a shareable piece) — weakest
   standalone; Esri is already attractive. Better as the *treatment* applied to
   one of the above than as the thesis itself.

### Leading idea (2026-06-02)

Anita is drawn to **#1 (flow orientation / direction + strength)**, time
granularity aside. Next step floated: pull the NTAD AADT layer (tier 2) and
throw together a rough orientation render to see whether the "grain of the
country" reads at national zoom before committing.
