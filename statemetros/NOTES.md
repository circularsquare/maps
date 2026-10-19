# statemetros

Animated map: the biggest metro area in each US state, 1900 to 2020. Served at
http://localhost:8800/statemetros/ (registered in served.json).

## Files

- `build_data.py` reads `raw/` and writes `data/statemetros.json` (64 areas, about 15 KB).
- `index.html` is the whole viewer (d3 for the projection, plain SVG). `?t=1950` opens at a year, paused.
- `data/states-10m.json` is us-atlas 3.0.1 state outlines, projected in the page with `geoAlbersUsa`
  so the dots and the outlines always agree (the pre-projected us-atlas file put Hawaii elsewhere).

## Method

- **Today's metro definitions, applied backwards.** July 2023 CBSA delineation (metropolitan and
  micropolitan areas), each one a list of counties. Its population in a census year is the sum of
  those counties' populations. Nothing about the 1910-onwards official definitions is used.
- **Metros across state lines count once per state**, using only the counties in that state.
  New Jersey's leader is the New Jersey side of the New York metro, Delaware's the Delaware side
  of Philadelphia. A state's part only counts if the metro has a principal city in that state
  (otherwise Boston's New Hampshire counties would beat Manchester).
- **Label** is the principal city that comes first in the CBSA title, within that state.
- **Between censuses** populations are interpolated log-linearly, and the leader is taken year by
  year, so a lead change happens when the curves cross.
- Bubble position is the 2020 population-weighted centre of the state's counties in the metro.
- Bubbles under 5,000 people are not drawn (Alaska before the 1960s has no data).

## Sources (all in `raw/`)

- `county_pop.csv`: county populations 1900-2020, Andrew Van Leuven's update of Forstall's
  1900-1990 table (https://andrewvanleuven.com/post/countypopulations, Harvard Dataverse
  doi:10.7910/DVN/WLS5GF). The NBER and census.gov copies of the 1900-1990 file are dead.
- `list1_2023.xlsx`, `list2_2023.xlsx`: Census Bureau CBSA delineation and principal cities.
- `2023_Gaz_counties_national.txt`: county centres. Its Connecticut rows are planning regions, so
  the eight old counties have hand-typed approximate centres in `build_data.py`.

## Known soft spots

- **Connecticut** planning regions are mapped to old counties by hand. Waterbury is dropped.
- **Broomfield County CO** (created 2001) has no row in the population file; about 1% of Denver.
- **Counties that did not exist yet** are not carved out of their parents, so early populations of
  metros built from newly created counties can be off.
- **Back-casting makes old metros big.** A 2023 metro in 1900 includes a lot of farmland, so
  rankings between a rural-heavy metro and a compact one can differ from what the cities looked like.
  Greenville SC leading South Carolina all century is probably this.
- **Virginia** is led by the Northern Virginia part of the Washington metro (labelled Arlington) from 1971.
- **Before 1900** is not built. Plan: one county, or city proper where a county would be silly.
