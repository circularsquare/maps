# Posters

Working notes for turning the interactive maps into printable posters.
(Shared plan lives here; `todo.txt` files stay Anita's own notes.)

## Print vendor

Lumaprints, semi-glossy fine art paper. Used to require one of their fixed
sizes; now accepts arbitrary inch dimensions, though fixed sizes may still
price better — worth comparing before ordering.

Reference prices (2025-09-24 order): 30x30 = $21.84, 36x36 = $30.50,
plus ~$13 shipping.

Specs:
- 300 DPI recommended for fine art paper. (72 DPI is their hard floor, not a target.)
- Accepted formats: JPG / JPEG / PNG only
- **Max upload 100 MB** <- the binding constraint, see below
- Recommended colour profile: Adobe RGB (they handle the CMYK conversion)

Source: lumaprints.com/faq and their file-prep blog posts.

## Sizing

~30 in is the practical minimum for legibility, established by the
equal-population maps. Those are 12000 px at 30 in = 400 ppi, which is
over-specced against Lumaprints' own 300 DPI recommendation. 300 is fine
and makes the file-size problem much easier.

At 300 ppi:

| size  | pixels      |
|-------|-------------|
| 30x30 | 9000x9000   |
| 36x24 | 10800x7200  |
| 36x36 | 10800x10800 |
| 40x30 | 12000x9000  |

## The 100 MB problem

The equal-population maps are flat colour regions, which PNG compresses
extremely well. Dot maps are high-entropy noise and PNG-24 will do badly —
a 9000x9000 dot map could plausibly land at 150-300 MB, over the cap.

Options, best first:

1. **Indexed PNG (PNG-8).** Dot maps have a genuinely small palette. Lossless,
   compresses far better than PNG-24, and it's Aseprite's native mode. Needs
   dots drawn with limited antialiasing, or quantised at the end.
2. **JPEG at quality 95+ with 4:4:4 chroma** (no subsampling). The default
   4:2:0 will smear tiny coloured dots badly — must be set explicitly.
3. Drop to 200-250 ppi.

Test this early — render one full-size frame before tuning anything else.

## Per-map status

- **equal population** — DONE, printed
- **ancestrydots** — IN PROGRESS
- **flights** — queued. Needs a renderer (currently web-only), reprojection off
  flat (Winkel Tripel / Robinson), top-routes table in the margin. Consider a
  Pacific-centred companion.
- **japanrail** — queued. Tokyo/Osaka insets, numbered top segments decoded by a
  margin table, line-thickness legend.
- **nycriders / londonriders** — experimental. Small-multiples grid of the day
  (8-12 frames) rather than one static view. Test whether the parallel-line
  rendering cleans up at print resolution — it looks good in detail crops and
  bad at whole-city zoom, which is a pixel-budget problem print may just fix.
- **nycatchment** — queued. Too dark to print as-is; crop tight to the built-up
  area, margin space to legend.
- **OUT** — world metro map, nybus, nystreets (unpolished); graffiti (not 2D);
  cityhistory (needs animation).

## Shared open questions

- **Dark vs light.** Dark mode is part of these maps' identity, but large flat
  near-black on paper bands, scuffs, shows handling marks and eats ink, and
  saturated neon shifts under print. Test a light variant of ancestrydots and
  decide from a physical proof, not from the screen.
- **Basemap.** Stop screenshotting MapLibre — it can't reach print resolution
  and the style is tuned for a backlit screen. Draw coastline/water directly
  (Natural Earth 10m, or the JRC global surface water pipeline from asia1m).
  That removes the maritime borders by never drawing them. Hand-place labels
  in Aseprite; a poster needs ~40-80 and they only get placed once.
- **Proof first.** Get one small cheap print before committing to a 30 in run.

## ancestrydots

Data already exists and does not need rebuilding:

- `data/processed/dots_all_1per100.geojson` — 662 MB, US, 1 dot = 100 people
- `canada/` has the Canadian equivalent; `combined/` merges the legends
- `ancestry_colors.csv` is the palette and is **hand-edited** — never run
  `--write-colors`

Steps:

1. Bake dots to a compact binary (lon, lat, colour index) so re-renders are
   fast. Parsing 662 MB of GeoJSON on every iteration is not workable.
2. Render layers to PNG at print res with matplotlib, in Albers equal-area
   conic for North America. One layer each: land/coast, water, dots, labels.
3. Composite and hand-label in Aseprite.
4. City insets — magnified crops at the *same* dots-per-person as the main map,
   so density reads consistently across the sheet. Candidates: NYC, LA,
   Chicago, Bay Area, Houston, Miami, Toronto, Vancouver, Montreal.
5. Legend — small-multiples strip: one thumbnail map per top-level category
   showing only that category's dots. Doubles as content, not just a key.

### Extent (settled)

Anita's bounds: south to Key West / Brownsville, north to Edmonton, west to
Campbell River, east to Cape Breton.

    lat 24.55 - 53.55 N,  lon -125.24 - -59.75 W
    projection: Albers equal-area conic, +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96

Projected: **6589 x 3684 km, aspect 1.789.** Tuning the standard parallels for
this latitude band (29.5/48.5) changes the aspect by <1% — not worth it, stay
with the EPSG:5070-style parameters.

The projected *rectangle* around that curved lat/lon box is generous at the
corners, so all of Newfoundland and all of Vancouver Island (incl. Cape Scott
and Tofino) come in free, even though the named bounds don't reach them.

Conic curvature adds 471 km of height (14.6%) over the central-meridian span.
That extra height is all empty corner — which is where the insets go.

### Sheet size

At 36 in wide: **183 km/inch**, map is 36 x 20.13 in, 1 px @300ppi = 610 m.

Recommend **36 x 24 in** — standard frame size, and 36/1.789 = 20.13 leaves a
**3.87 in band** along the bottom for the legend. 10800 x 7200 px at 300 ppi.

### Inset budget (measured, not guessed)

City insets, span -> size on the sheet:

| metro       | at 1x  | at 8x  | at 12x |
|-------------|--------|--------|--------|
| NYC         | 0.55in | 4.4in  | 6.6in  |
| LA          | 0.60in | 4.8in  | 7.2in  |
| Chicago     | 0.49in | 3.9in  | 5.9in  |
| Bay Area    | 0.49in | 3.9in  | 5.9in  |
| Toronto GTA | 0.44in | 3.5in  | 5.2in  |

So each city inset costs 4-7 in. **Four or five fit in the ocean dead space,
not nine.** Trim the list.

Off-map states at TRUE scale (same km/inch as the main map):

- Alaska  2423 x 2034 km -> **13.2 x 11.1 in** — not viable, see below
- Hawaii   579 x  380 km -> 3.2 x 2.1 in — fits fine in the Pacific
- Puerto Rico 222 x 72 km -> 1.2 x 0.4 in — fits anywhere
  (`dots_72_1per100.geojson` already exists, so PR is free)

**The Alaska problem.** True scale is a third of the poster width, and Alaska
is only 733k people = ~7,300 dots at 1:100, so a full-state true-scale inset
spends ~145 sq in to show almost nothing. Options: shrink it with the scale
change labelled, or crop to the populated strips (Anchorage/Mat-Su, Fairbanks,
Southeast) at true scale. Leaning toward putting AK/HI/PR in the legend band
with their own scale bar rather than floating them in the ocean.

### Known issues with this framing

- **Mexico has no data.** Baja and northern Mexico are a large visible chunk of
  the bottom of the frame and will render as empty land — on a dark map, solid
  black, which reads as "nobody lives there". Needs a deliberate treatment:
  grey/hatch as out-of-scope, or crop tighter.
- **The north is very empty.** Holding the Edmonton line costs a lot of near-
  blank paper across the prairies and Canadian Shield. Defensible (Edmonton is
  1.5M) but worth a look at the proof before committing.
