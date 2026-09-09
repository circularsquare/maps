# elevationpop

Population density draped over real 3D terrain, interactive. Türkiye is the
first subject, after a static version of the same idea by Milos Makes Maps.

MapLibre GL JS does the 3D part out of the box: a `raster-dem` source plus
`terrain` in the style turns the basemap into a displaced mesh, and raster,
fill, line and hillshade layers get draped onto it. Nothing here is a custom
renderer.

## The layer stack

Bottom to top, and this order is the whole design:

1. `background` — near-black, what empty country shows.
2. `pop` — the WorldPop raster, coloured here, tiled locally.
3. `hillshade` — shading computed from the DEM, translucent, **on top**.

A draped raster gets no lighting from the terrain mesh in MapLibre. Pitching
the camera gives you parallax and silhouette but the surface stays flat-lit,
which is why the hillshade sits above the population rather than under it: it
shades the colours instead of being hidden by them. The alternative is to
multiply a hillshade into the population tiles at generation time, which is
what a static render does, but then the shading is frozen and the light
direction cannot move.

## Camera

The right-drag rotate felt reversed and, in Anita's word, nauseating. It was
not the direction. **It was a latched drag**, and that is the thing to check
first next time: the handler listened for `mouseup` on `window`, and releasing
the button outside the browser window never delivers one, so the drag stayed
live and the map kept turning under a button that was no longer held.

`setPointerCapture` plus an `e.buttons === 0` guard fixes it. Both were tried,
and the directions that won are MapLibre's own on both axes, so `TURN_BASE` and
`TILT_BASE` encode the stock behaviour. What is not stock is the speed, half of
MapLibre's 0.8 degrees per pixel, and the absence of inertia. Both are calmer.

The built-in handler is still disabled rather than reused, because that speed
and that inertia are not configurable.

## Shading

The ground colour is deliberately not black. A hillshade can only darken from
the colour underneath it, so on a near-black base the shadows have nowhere to
go and the relief reads as noise. Lifting the base and widening the shadow and
highlight range is what makes the topography legible. Three values move
together and want changing together: `background-color` in `index.html`,
`#ramp` behind the legend gradient, and `BASE` in `preview.py`.

## Data

- **Population** — WorldPop 2020, UN-adjusted constrained, 100 m,
  `Global_2000_2020_Constrained/2020/BSGM/TUR/`. Sums to 84.3 M, which matches
  Türkiye's 2020 UN figure, so the download is intact.
- **Elevation** — Terrain Tiles on AWS Open Data, terrarium encoding, global,
  no key. MapLibre reads terrarium directly. This host has been unreliable in
  the past; for anything published, cut Copernicus GLO-30 to terrain-rgb and
  host it instead.

Neither is redistributed here. `data/` is gitignored.

## Pipeline

    python prep.py       # warp to Web Mercator, aligned to the tile grid, ~25 s
    python preview.py    # one PNG of the whole country, to judge the ramp
    python tiles.py      # 512 px XYZ tiles, ~40 s
    python ramp.py       # print the legend gradient CSS for index.html

Then serve the directory and open it. Plain PNG tiles, so anything static
works and there is no byte-range requirement:

    python serve.py 8971

Use `serve.py` rather than `python -m http.server`. It sends `no-store`, and
the browser cache on this page is bad enough that an edit can look like it did
nothing, twice in a row, including across a fresh port.

## What a pixel means

`prep.py` warps nearest (the destination is finer than the source, so it only
replicates) and builds the overview pyramid with average. A pixel at any zoom
therefore means *mean people per 100 m cell over the area it covers*, which is
a density, so one legend stays honest at every zoom. The cost is that a lone
one-cell hamlet fades out when zoomed far out. A town does not, because a town
is hundreds of contiguous cells and averaging within it returns roughly its own
density.

Only about 2% of cells in Türkiye hold anyone, so most of the map is empty by
design and the dark base and hillshade show through.

## Colour

`ramp.py` holds the ramp and nothing else does. Edit `STOPS`, `LO`, `HI` or
`FADE`, re-run `tiles.py`, then re-run `ramp.py` and paste the gradient into
the legend block in `index.html` so the two cannot drift.

## Known rough edges

- Terrain and hillshade run off two `raster-dem` sources with identical URLs.
  MapLibre warns if one source drives both. The browser cache means it is not
  a second download.
- The hillshade covers the whole world while the population stops at the
  border, so Türkiye sits in a shaded but uncoloured neighbourhood. At a
  distance that reads as context; up close it reads as a missing country.
- Terrarium carries bathymetry, so the sea floor dips rather than sitting flat
  at zero, the default 6.5x multiplies that, and the lifted base now makes it
  visible. Offshore canyons and the shelf edge are legitimately there, but they
  compete with the coastline. A published version should clamp negative
  elevations to zero in its own DEM tiles.
- City labels are HTML markers, so they do not collide with each other.
  `cull()` hides them past a zoom-scaled distance, which is a blunt stand-in
  for real label placement. MapLibre does occlude markers behind terrain, but
  at 6.5x it decides almost every city is behind a ridge and greys the lot, so
  `opacityWhenCovered` is pinned to 1.
