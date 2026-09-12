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
- **japanrail** — IN PROGRESS. Tilted sheet, Tokyo/Osaka insets, numbered top
  segments decoded by a margin table, line-thickness legend.
- **nycriders / londonriders** — experimental. Small-multiples grid of the day
  (8-12 frames) rather than one static view. Test whether the parallel-line
  rendering cleans up at print resolution — it looks good in detail crops and
  bad at whole-city zoom, which is a pixel-budget problem print may just fix.

  **nycriders needs a credit line.** MTA's data itself is free to use and
  redistribute, but its developer page asks you to license "logos, maps,
  symbols or other intellectual property", and `ROUTE_COLORS` is the official
  palette — already jittered off it (median dE76 10.8), but jittered *for*
  within-group separation, which is down at 3.1 between N and W. There is no
  slack to rejitter further, and moving whole groups would break the one thing
  a reader navigates by: red means the 1/2/3. **Anita's call: nobody would
  think MTA made it, so keep the colours and add a credit line** naming MTA
  open data as the source and saying the map is unofficial. Same for
  londonriders, whose TfL terms explicitly require `Powered by TfL Open Data`
  and forbid passing the result off as a TfL product.
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

### Pipeline (built)

    posters/ancestrydots/bake.py     GeoJSON -> build/dots_na.npz (46 MB, 4,403,741 dots)
    posters/ancestrydots/render.py   npz -> layered PNGs

`bake.py` regex-scans the single-line GeoJSON at ~60 MB/s; the whole bake is
about 15s. Two gotchas it now handles, both of which were live bugs:

- Accented labels (Québécois, Métis — 14k dots) are `\uXXXX`-escaped in the
  GeoJSON but decoded in legend.json, so a naive byte match drops them. The
  index is keyed on both forms.
- **The US `--all` GeoJSON already contains Puerto Rico.** Adding
  `dots_72_1per100.geojson` on top double-counts it (51,980 vs 51,981 dots in
  the same bbox). PR is deliberately not in SOURCES; there is a guard that
  re-checks this on every bake.

Totals: 4.40M dots = 440M *ancestry responses*, not people. ACS ancestry is
multiple-response, so the count exceeds the ~371M population — consistent with
the interactive map's own legend, which sums to ~449M.

`render.py` writes an **index raster** (uint16 ancestry index per pixel), not
RGB, so the hand-tuned palette can be re-applied without re-rendering and the
output stays palette-limited for indexed PNG. Dots are hard-edged, last-write-
wins in shuffled order — unbiased, and it keeps the colour count low.

    python render.py --scale 0.2                        # preview, ~4s
    python render.py --scale 0.2 --theme light
    python render.py --crop " -74.0,40.75,3,2.4"        # full-dpi texture check
    python render.py                                    # full 10800 x 6036

(Note the leading space inside the quotes on `--crop` — argparse otherwise
reads the negative longitude as a flag.)

### The 100 MB cap is a non-issue (resolved)

Full 36in render at 10800 x 6038 comes out at **12 MB**. The hard-edged,
palette-limited dots compress far better than feared, so PNG-24 is fine — no
need for indexed PNG or JPEG. Plenty of headroom to go to 400 ppi or larger
dots if wanted. Whole render takes **24 s** for both themes (the dot pass is
theme-independent and gets reused).

### Previews lie — use 1:1 crops

Two opposite artifacts, both misleading, both encountered:

- A **reduced-scale render** (`--scale 0.2`) forces dots down to 1 px and piles
  ~11 of them per pixel, so it looks far muddier and more saturated than print.
- A **downscaled view** of the full render averages each 1-2 px dot with ~20
  background pixels, so it looks far paler and washed out than print.

Only `--crop` at full `--dpi` shows what the paper will actually look like.
Judge colour and texture from those, and use the whole-sheet view for
composition only.

### What the first renders showed

- **Full-res texture is good.** The 1/5-scale previews look muddy, but that's a
  preview artifact — at 1 px = 610 m the dots resolve properly across almost
  the whole continent.
- **Metro cores still saturate**, exactly as the density maths predicted. The
  NYC core is a solid magenta mass at full res. Confirms city insets are
  load-bearing rather than decorative.
- **Light mode is markedly more legible** than dark, and it also softens the
  Mexico problem — empty land reads as land rather than as void. The palette is
  tuned for a dark ground though, so the pale categories (yellows, light
  greens) are weak on cream and would want darkening.

### Theme: DARK (decided)

Dark won on legibility at 1:1 — the palette's colours were tuned to glow
against black and they collapse toward a single mid-blue on cream. The only
open dark-mode risk is print physics (flat near-black on semi-gloss), which a
small proof settles.

### City insets (built)

`posters/ancestrydots/insets.py` — 16 cities at 5x the main scale
(36.6 km/inch, 1 px = 122 m), **93.1 of the sheet's 864 sq in**, whole set
renders in ~27 s.

Each inset gets its **own local Albers** centred on the city rather than the
main CONUS projection. Necessary for Honolulu (60 deg off the main central
meridian, where CONUS Albers is unusable) and more accurate for all of them.
Equal-area throughout, so dots-per-km stays comparable.

Water is the **OSM water polygons** (`data/water-polygons-split-4326`), not
Natural Earth — at 1 px = 122 m, NE 10m coastlines read as visibly polygonal.
The bbox-filtered read is 1-3 s per city even off the 1.26 GB shapefile.

**5x works better than the density maths predicted.** I expected metro cores to
collapse into solid colour; in practice NYC at 5x shows clear neighbourhood
structure. The earlier "you'd need 20x" estimate assumed dots must be
individually resolvable, but the colour *mixture* reads fine before that point.

### Dot rendering: antialiasing

`dotraster.py` is shared by the main map and the insets.

Dots were originally hard-edged, which made them read as little **plus shapes**
rather than dots — `disc()` quantises hard, so r=1.4 gives a 5px plus and r=0.9
gives a single pixel. That was a deliberate choice to keep the palette small
for indexed PNG, but the 12 MB output killed that rationale, so it is now just
a defect.

Antialiasing is by **supersampling** (`--ss`, default 3), not per-dot alpha:

- Per-dot alpha compositing needs dots drawn sequentially to get occlusion
  right, which is not vectorisable.
- Averaging overlapping dots instead of occluding them desaturates dense areas
  toward a muddy mean.
- Rendering hard at ss x and box-downsampling keeps correct occlusion *and*
  gives round soft dots. Costs ss^2 memory, so it runs in horizontal strips.

Fractional radii only mean anything once ss > 1. Inset default is now
**r=0.91, ss=3** (~35% smaller than the old 1.4).

Trade-off to keep in mind: antialiasing is smoother but slightly less "dotty" —
in dense areas it reads as a continuous colour field rather than distinct dots.
r=1.2 ss=3 sits between the two if the AA version feels too soft.

### Per-inset mini-legends

Each inset now emits its top ancestries by dot count inside its own window:

    build/insets/<name>_top.json     ranked list with colours, dots, people, pct
    build/insets/<name>_legend.png   rendered starting-point block
    build/insets_top_ancestries.md   all cities in one readable table

Percentages are of ancestry **responses** in the window, not of people (ACS
ancestry is multiple-response). Worth wording carefully on the poster.

`--exclude-groups no_ancestry` drops the residual "Black/White, no ancestry
reported" categories from the listing — they otherwise take the top slots in
most cities. The denominator always stays all responses, so shares remain
honest either way.

Known gaps in the insets:

- The OSM water file is **ocean only**; inland lakes come from coarse NE 10m,
  and rivers are not drawn at all. Rivers still read, because the dots have
  areawater subtracted and so leave a gap — but the gap is land-coloured rather
  than water-coloured. Fixable with NHD/OSM waterways if it bothers.
- Window framing is a first pass. Boston carries too much open ocean and
  Seattle a lot of empty land; those want re-centring.
- No city labels yet — Aseprite job, or add to the script.

### Settled

- Theme **dark**.
- Dots **antialiased** (ss=3). Main map r=1.0, insets r=0.91.
- Mini-legends **keep** "Black/White, no ancestry reported" rather than
  excluding them.

### Layout: edge rails, two-column right (current)

**Sheet is now 34 x 24 in** (map 34 x 20.13 + a 3.87 in band), at
`--extend-left 1.0 --trim-right 3.0`.

    left    alaska seattle sf la honolulu hawaii     one column, 15.1 in
    right   newfoundland / boston nyc / philly dc /  two columns, 11.5 in
            atlanta miami / pr
    top     chicago detroit toronto montreal         one row, 10.8 in
    bottom  dallas houston                           one row, under Texas

**294 of 4,314,046 dots covered — 0.01%**, all of it the top rail. Left, right
and bottom rails cover literally zero.

Why the numbers moved:

- Two columns shortened the right rail from 16.1 in to 11.5, which is what let
  3 in come off the Atlantic. **4 in is too far** — it pushes the inner column
  onto the New England coast and costs 10,496 dots.
- Losing St John's to that trim is deliberate: **Newfoundland now has its own
  1x inset**, sitting above Boston/NYC. At 1x it reads as a relocated piece of
  the main map rather than a magnified window.
- `--extend-left 1.0` drops Haida Gwaii and is all the left rail needs.
- Rails now pull toward the mean position of the cities they show (`--pull`),
  which is what moves Dallas/Houston out of the corner to under Texas.
- Rails are **normalised**: on a side rail every box takes the rail's widest
  width and its row's tallest height. A wider window just shows more
  surrounding country at the same magnification, so nothing is lost.

Alaska, Hawaii, Puerto Rico and Newfoundland now live in `insets.CITIES` with
their own magnification in `insets.MAG` (0.24, 1.0, 2.0, 1.0), so sizes are
defined in one place. Run order matters:

    python layout.py                    # normalises sizes -> layout.json
    python insets.py --from-layout      # renders at those sizes

Two bugs found and fixed doing this:

- matplotlib rounds a figure a pixel short (9.45 in x 100 dpi -> 944, not 945),
  so the base and the dot raster failed to broadcast. Both scripts now force
  the base to the raster's exact size.
- The OSM water file is **split into tiles**, so stroking its polygons drew the
  tile seams as a grid across open water — glaring on Alaska. Water is now
  fill-only; NE lakes are single polygons and keep their edge.

### Superseded: single-column rails

Free placement was **tried and rejected**. Scoring each inset independently
scattered them across sparse country, and sparse areas are exactly where a dot
map is worth reading — hiding 700 dots in rural Texas costs more than the
number suggests.

Current approach is a 19th-c atlas plate: a rigid column down each side, a row
along the top, a short row at the bottom left. Order within each rail is fixed
(north to south, west to east); the only free parameter is the rail's offset
along its own axis, chosen against the density table.

    left    alaska seattle sf la honolulu hawaii      15.7 in run
    right   boston nyc philly dc atlanta miami pr     16.1 in run
    top     chicago detroit toronto montreal          10.8 in run
    bottom  dallas houston                             5.5 in run

**1,866 of 4,317,493 dots covered — 0.04%**, against 0.12% for free placement.
Tidier *and* less destructive.

The left and bottom rails cover literally zero. The top rail covers only 184,
because it slides to a gap in the Canadian Shield. The single real offender is
**Boston at 1,682 dots**: it is northernmost in the right rail, which puts it
over Nova Scotia, and the rail cannot slide down because Puerto Rico is already
near the bottom margin.

Open aesthetic issue: **the rails are ragged.** Box widths in the left rail run
2.0-3.97 in and the right rail 1.8-3.0, so the inner edges are uneven, which
the atlas-plate style really wants squared off. Fix is to normalise each rail
to a single cross-axis size — a wider window just shows more surrounding
country at the same 5x, which is no loss. Needs the inset windows re-rendered
at the new sizes.

Also unused: a large empty block bottom-centre (Gulf of Mexico and northern
Mexico). Natural home for the title / credit / data-vintage text.

### Superseded: free placement

`posters/ancestrydots/layout.py` — places the 16 insets plus Alaska, Hawaii and
Puerto Rico by scoring every candidate position against a summed-area table of
dot density. "On the water / on empty land" therefore falls out of the data
rather than being hand-guessed. Cost is `dots covered + dist_weight x distance
from the city`, with overlap and the legend band as hard constraints.

`--dist-weight` is the knob. It was initially 900, which valued 1 inch of
travel at 18,000 dots and parked Boston on 21,299 of them. At **120** (1 inch ≈
2,400 dots) everything moves out to water.

**Frame: `--extend-left 2.5 --trim-right 2.5`, now the default.** Trades 2.5 in
of empty Atlantic for 2.5 in of Pacific at a fixed 183 km/inch, so the sheet
stays 36 x 24. This is what gets Alaska, Hawaii, Honolulu and Puerto Rico onto
open water — before it, Alaska sat on 11,924 dots. 3.0 in is too far: it loses
St John's. There is a landmark check in the script that reports this.

Result: **5,053 of 4,317,490 dots covered, 0.12%.** Worst offenders are
Toronto (2,612), Chicago (723) and Dallas (706) — the Great Lakes and Texas
interior simply have less empty space nearby.

Outputs: `build/layout_plan.png` (the picture) and `build/layout.json`
(every box's position and size in inches, for Aseprite).

Still rough in the plan:

- Leader lines cross around New York / Philadelphia / Boston.
- Detroit ends up 6.1 in from its source, further than ideal.
- Toronto and Chicago still sit on dots.

### Legend band (built)

`posters/ancestrydots/legend.py` -> `build/legend_band.png`, 34 x 3.87 in at
300 ppi (10200 x 1161).

Names every ancestry reported by 250,000+ people: **101 of 343**, grouped under
the 14 top-level categories, flowed into 5 columns x 23 rows. A group header is
never stranded at the foot of a column. Entry type is 7 pt, headers 8 pt —
7 pt is close to the print floor but comfortable at poster reading distance.

The band states plainly that the list is partial: "101 ancestries are named
here — those reported by 250,000 or more people. All 343 reported ancestries
are mapped."

Title block (left 8.5 in) carries the title, the dot key (1 dot = 100 people),
a 500 km scale bar with the projection named, and the sources:

    US Census Bureau, American Community Survey 2020-2024, table B04006
    Statistics Canada, 2021 Census of Population

`--threshold` moves the floor (100k -> 147 named, 500k -> 73).

Note the title currently lives in the band, not in the empty Gulf block. Easy
to move if the bottom-centre placement reads better.

### Rail clearance

A box that only pays for dots *strictly inside* it can hug a coastline for
free, which is how Boston ended up touching Long Island. `--clearance`
(default 0.35 in) scores an expanded box, so near-misses cost something. That
alone dropped the right rail 0.6 in, no hand-nudging.

### Composite (built)

`posters/ancestrydots/compose.py` places everything from `layout.json`, so the
layers register with each other by construction. Full sheet 10200 x 7200:

    poster_1_base.png     land, water, band ground          opaque   2.2 MB
    poster_2_dots.png     the dots                          alpha   17.3 MB
    poster_3_insets.png   20 insets + captions              alpha   10.9 MB
    poster_4_frames.png   inset borders, on-map locators    alpha    0.3 MB
    poster_5_text.png     title block + ancestry legend     alpha    1.1 MB
    poster_flat.png       all five flattened               27.7 MB — under the cap

Frames are their own layer so they can be restyled or dropped without
re-rendering. `legend.py` now writes transparent by default (`--opaque` to
paint the ground back in) so it can be layer 5.

Full pipeline, in order — the steps feed each other:

    python bake.py                    # only if source data changed
    python layout.py                  # frame + sizes -> layout.json
    python insets.py --from-layout
    python render.py --theme dark --layers
    python legend.py
    python compose.py

**render.py and layout.py disagreed about the frame** for a while — render.py
had no idea about `--extend-left/--trim-right`, so the map was still 36 in on
the original extent while the rails were planned for 34 in. Every inset would
have been misplaced. render.py now reads the frame from `layout.json`, and
compose.py refuses to run if the two are out of step. It also picks the render
*matching* the layout width rather than the widest on disk, because after a trim
the widest file is the stale one.

**Captions come out of the inset's window, not on top of the box.** Growing
every box by the 0.62 in caption pushed the right rail onto Newfoundland and
Nova Scotia (10,449 dots) and shoved Dallas/Houston away from Texas. Taking the
strip from the window costs a little inset area instead. Puerto Rico is the one
exception — its box is 0.80 in tall, so subtracting 0.62 would leave a sliver;
below `MIN_WIN_IN` (0.80) the box grows instead, costing the right rail 0.62 in.

### Inset and legend revisions

- Captions are **two columns of six** ancestries in the same 0.62 in strip.
  Caption type is 5.4 pt, deliberately the smallest in the project.
  `--top` must be >= `--caption-top` or the caption silently comes up short.
- Long labels overran their percentage in a half-width column, so `SHORT`
  rewrites the worst ("Black, no ancestry reported" -> "Black (no ancestry)")
  with a 24-character fallback truncation. The band legend keeps full wording.
- City name row carries the population in bounds, right-aligned, and a divider
  rule sits above it.
- **No all-caps anywhere** — city names, poster title, and legend group headers
  are all sentence case now.
- Newfoundland inset dropped.
- Each inset gets its own scale bar, bottom-right of the map region, drawn on
  the frames layer. The distance is chosen per inset (round number nearest a
  fifth of the window) because insets run at different magnifications.
- Band scale bar is now a single plain bar — the two-tone one read as two bars,
  leaving it ambiguous whether 500 km covered one or both. Dot key is one dot,
  drawn at its true printed size.
- Added `anita.garden/ancestrydotsna`.

**Printed dot size: 0.197 mm.** A main-map dot is `disc(radius*ss)` =
`disc(3.0)`, spanning 7 supersampled px, downsampling to 2.33 px; at 300 ppi
that is 0.197 mm on the 35 in sheet, or 0.169 mm if the same file is printed
at 30 in. Inset dots (r=0.91) are 0.141 mm.

### Sheet grew to 35 x 24

Windows were enlarged where the crop was cutting things off: Honolulu now
holds all of Oahu, LA reaches Long Beach and Orange County, SF runs past San
Jose, Hawaii clears top and bottom, PR is less tight, and Alaska went to
4.2 x 3.9 in for the panhandle and Aleutians.

Alaska sets the left rail's column width, so the whole rail went 3.18 -> 4.2 in
and at `--extend-left 1.0` reached the California coast under San Francisco
(680 dots). Trimming the right to compensate is not available — at 4.0 in
Boston lands on 56,327 dots. So `--extend-left 2.0`, sheet 35 x 24, back to
294 dots covered (0.01%).

Side effect: Honolulu and Hawaii now sit in wide panels with a lot of empty
ocean, because they inherit Alaska's column width.

### Typeface

**Nunito**, matching the interactive map. `typeface.py` registers it;
`make_fonts.py` cuts the static weights.

Gotcha: `Nunito.ttf` from Google Fonts is a **variable** font whose default
instance is **ExtraLight**, and matplotlib does not set variable axes — point
it at the variable file and everything silently renders hairline-thin. The
`static/` path in the google/fonts repo 404s for Nunito, so the weights are
instanced locally with fontTools:

    curl -sSL -o Nunito.ttf 'https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf'
    python make_fonts.py        # -> Nunito-Regular.ttf, Nunito-Bold.ttf

### Measured sizes (Nunito, at 300 ppi)

    5.4 pt  inset caption + inset scale bar   'l' 1.37 mm   x-height 0.95 mm
    6.6 pt  inset population figure           'l' 1.68 mm
    7.0 pt  legend band entries               'l' 1.78 mm
    7.6 pt  inset city name                   'l' 1.93 mm
    8.0 pt  legend band group headers         'l' 2.03 mm
     26 pt  poster title                      'l' 6.61 mm

**One map dot is 0.197 mm across** (0.141 mm in the insets). At a 40 cm reading
distance one arcminute of visual acuity is about 0.116 mm, so adjacent dots are
just separable close up; at a metre it is 0.29 mm, so they merge into tone.
Detecting an isolated dot is far easier than resolving two, so a lone dot in
empty Nevada still reads as a speck from across the room. Roughly one pixel of
a 27-inch 1440p monitor.

### Water tile seams, again

Fill-only was not enough. Adjacent OSM water tiles leave an antialiased
hairline along every shared edge, which reads as the same grid the coloured
stroke produced. Stroking each tile in **its own fill colour** closes the seams
without drawing anything.

### Caption spacing rules

Padding is uniform: the gap above the city name, left of everything, and below
the last ancestry row are all `PAD_IN` (0.10 in). Rows are distributed between
a computed first and last centre rather than by a fixed step, so the last row
lands exactly on the bottom padding. `NAME_GAP` (0.055 in) separates the name
from the first row.

Both columns' percentages right-align, and the second column's edge is flush
with the ancestry count above it — the column grid is
`col_w = (width - 2*PAD - gutter) / 2` so that falls out rather than needing a
fudge. Caption type is 6 pt.

Heights use 0.72 em, which is where Nunito puts both caps and ascenders, so
that is what reads as the visual height of a line.

The population figure is labelled **"19.2m ancestries"** rather than left bare,
because it counts ancestry responses and not people — ACS ancestry is
multiple-response, so NYC's window shows 19.2m against a real population of
about 14m.

### Scale bars

- Inset bars: a **twelfth** of the panel width, 3 px thick, labelled at caption
  size. LA is in `LEFT_SCALE` because its bottom-right corner is dense; its bar
  sits bottom-left instead.
- The main 500 km bar is **on the map**, bottom-left of the map area, directly
  above the title. `legend.py` no longer draws it; `compose.py` puts it on the
  text layer.

### Legend band: auto-fitted

The band no longer uses a fixed 5 columns stretched to fill the width — that
left a huge gap between each ancestry and its count and between columns.

`fit()` searches rows-per-column, choosing the largest type size whose grid
still fits the width. Column count comes from actually flowing the lines, not
from dividing, because the no-stranded-header rule can add a column. Entry
width is **measured from the font's advance widths** (`Metrics`), not guessed
from a character average.

Now: **100 largest ancestries, 8 columns x 15 rows at 9.8 pt**, grid 20.6 in,
title block 12.2 in. Whatever the grid does not need goes to the title block
rather than becoming gutter.

Two things the fitter needs guarding:

- `--max-fs` too low forces many narrow columns and a short band. At 8.5 it
  picked 7 columns and left the band looking empty.
- Without `--title-min` (10.5 in) it maximises legend type until the title
  block is squeezed to 6.6 in, which is narrower than the title itself.
- The old column break tested `y < bottom`, which let a 14th line land exactly
  on the margin, spilling an extra column and clipping the last row. Columns
  are now flowed explicitly.

Named ancestries switched from a population threshold to the **N largest**
(default 100), so the copy can say "the largest 100" without the awkward 101
that a 250k floor produced.

### Title block copy

    Ancestry Dots in North America                       26 pt
    1 dot = 100 reported ancestries                      16 pt
    Data is self-reported ancestries from the 2020-2024 American Community
      Survey and the 2021 Census of Canada. Surveyed people can report
      multiple or zero ancestries, so dot count does not directly correspond
      to number of people.                                9 pt, wrapped to 8 in
    The legend shows the largest 100 ancestries in the dataset, but smaller
      ones are also mapped.
    Interactive version is at anita.garden/ancestrydotsna
    [citation + projection]                              7.5 pt

**The ACS release is the 2020-2024 five-year, not 2021.** Anita's draft copy
said 2021; that would have contradicted the citation directly beneath it, so
the body says 2020-2024. Canada is 2021 and that is correct.

Body text wraps to `--measure-in` (8 in), not to the title block width — the
block is 12.2 in, which is far past a readable measure.

### Frame: top trimmed, sheet 34 x 23

`--trim-top 1.0` (300 px at 300 ppi) takes the near-empty northern Canada off
the top. The point was not the empty paper — it was that carrying all that
latitude pushed every city inset a long way from where its city actually is.

Alaska's panel came down 80 px (0.267 in) to partly compensate, with its centre
moved north by half that so the cut lands on the empty ocean at the bottom
rather than being split across both edges. It cost **2 dots** out of 9,227,
which confirms that end was dead space.

`RAIL_NUDGE` applies a manual offset along a rail's own axis after the density
fit has chosen its position; the top rail is nudged 0.333 in (100 px) west.

All landmarks still pass the frame check, Edmonton included. The top rail now
covers 539 dots rather than 294 — trimming the north brings the rail closer to
populated Canada — but that is still 0.01%.

Sheet: **34 x 23 in**, map 34 x 19.13.

### Palette: "Black, no ancestry reported"

Two friends read the map as homogeneous, because the two biggest non-European
masses were muddling into their neighbours: this category against the blues,
and Mexican against "White, no ancestry reported".

Changed **h 143.1 -> 120.0 and s 0.27 -> 0.38**, l 0.452 -> 0.460, giving
`#48a148`. The hue move was what Anita asked for; **the saturation was not**,
but at 0.27 it was by far the most washed-out of the big categories (Mexican
0.48, African 0.62) and desaturation was doing as much damage as hue — a muted
grey-green reads blue when blue dots are mixed through it.

How far yellow it can go is limited by neighbours, not taste:

    42-70    Latino (Mexican 61)
    82-104   West Indian (West Indian 90, Jamaican 100)
    135-140  African

So ~105 is about the limit before colliding with Jamaican. 120 sits in the
clear corridor between West Indian and African.

### Caption at 8 pt

6 pt was too small at real size. 8 pt does not fit two columns in a 2.6 in box,
so the narrow boxes went to 3.0 in wide, and labels are now **measured and
truncated** against the actual space (`text_w`, from the font's advance widths)
rather than trusting a character count. A character count is a bad proxy here —
the columns have no slack to absorb the error.

`CAPTION_IN` is 0.72, which is the measured minimum for 8 pt: 0.10 pad +
0.095 name + 0.055 gap + 0.336 rows + 0.10 pad = 0.686. Anything above that
comes straight out of the map windows and pushes the rails onto land — an
earlier 0.80 did exactly that, taking coverage to 1.08%.

### Western rearrangement, 22 insets

    left       alaska vancouver seattle sf honolulu hawaii
    southwest  la phoenix          (over Baja and northern Mexico, out of scope)
    top        minneapolis chicago detroit toronto montreal
    right      nyc boston / philly dc / atlanta miami / pr
    bottom     dallas houston

New: Vancouver, Phoenix, Minneapolis. NYC and Boston swapped — geometry is
unchanged by that, since a vertical rail normalises both columns to the same
width. Minneapolis was worth adding: German 21.5%, Norwegian 8.3%,
Swedish 5.4%, which is the Scandinavian story nothing else on the sheet showed.

SF recentred north (dead space below San Jose); LA recentred east and south for
the Inland Empire and Orange County.

The main 500 km scale bar now starts clear of whatever occupies the
bottom-left, computed rather than fixed, since the left rail reaches the
bottom margin.

Coverage is **0.10%** (4,483 dots), up from 0.01%. All of it is Montreal
(2,452) and Minneapolis (1,502) on Canadian dots — with the north trimmed there
is no clean water up there, and a wider top rail reaches further into it.

### The caption padding bug

Atlanta and Miami looked cramped under their legends because `box_h_in` was
derived from a **truncated** cell count: `win 2.38 + caption 0.72` lands at
3.0999999, and `int(3.0999999 * 20)` is 61 cells — 3.05 in, not 3.10. The frame
was drawn 0.05 in (15 px) shorter than window + caption, so the bottom rule cut
into the caption's padding. It only showed on rows whose height happened to
land just under a cell boundary, which is why it looked arbitrary.

Fixed by deriving `box_h_in` from `win_h + CAPTION_IN` directly, and rounding
rather than truncating every inches-to-cells conversion.

### Snug bottom-left corner

The left rail is sized to fill the edge **exactly** — 18.6 in of an 18.63 in
span — so Hawaii lands flush in the corner instead of leaving a gap. Getting
there needed the heights to sum to precisely 372 cells; at 373 the rail
reported DOES NOT FIT, and at 371 it floated. Height went back into SF (the
earlier northward shift had cut real San Jose), Seattle and Alaska.

`RAIL_SNAP_AFTER` pins a rail flush against another rather than letting the
density fit place it: the southwest rail (LA, Phoenix) now starts at the left
rail's right edge plus one gap, so every spacing in that corner is the same
0.15 in. `RAIL_NUDGE["bottom"]` shifts Dallas/Houston 0.133 in east, off the
Texas barrier island.

### Caption abbreviations

A measured fit alone gave "Black (no ances…", which is worse than a real
abbreviation. `short()` now tries full -> `SHORT` -> `ABBREV` -> ellipsis, so
the fallback is "Black (n.a.)". The band legend carries full wording, and the
title block now explains the residual categories.

### Residual-category wording (verified against the code)

    Black, no ancestry reported = ACS B02009 minus groups {african, afro_carib}
    White, no ancestry reported = ACS B02008 minus {western, eastern, american}

(`BLACK_SUBTRACT_GROUPS` / `WHITE_SUBTRACT_GROUPS` in scatter_dots.py, with a
`.clip(lower=0)` on the residual.) The title-block paragraph says exactly this;
"European" was expanded to "Western European, Eastern European" to match the
legend headings a reader can actually check against.

Saturation settled at **0.34** (h 120, l 0.460) after 0.38 read too strong.

### State: ORDERED (2026-09-05)

34 x 23 in, 300 ppi, dark, 22 insets, ~$20 from Lumaprints. `poster_flat.png`
is 29.9 MB with an embedded sRGB profile.

Resolved along the way, all now done: legend band, Alaska/Hawaii/PR, inset
locator marks, scale bars, title and sources. Mexico was left as plain unlit
land — Anita judged the coastline seam obvious enough that it does not read as
missing data. Place names were deliberately **not** added: any label would
obscure dots, and she would rather have none.

### Credit added after ordering — the file on disk is no longer the ordered one

The citation line named only the Census Bureau and Statistics Canada. Both are
fine to sell from — ACS and TIGER are public domain, and the Statistics Canada
Open Licence explicitly grants the right to *sell* the Information and to make
and sell value-added products. But the **insets' coastal water is the OSM
`water-polygons-split-4326` extract, which is ODbL**, and that was uncredited.

A printed sheet is an ODbL **Produced Work**, so share-alike does not reach it
— the poster does not have to be ODbL. Attribution does reach it, and it is a
condition of the licence rather than a courtesy. The line now ends
`· Inset water © OpenStreetMap contributors, ODbL`.

Two things worth keeping straight:

- The main map's ocean and *all* the lakes are Natural Earth, which is public
  domain and needs no line. OSM only supplies the insets' coastal water (and
  only as `insets.py`'s first choice — it falls back to NE if the file is
  absent), so "Inset water" is the accurate wording, not a hedge.
- `legend.py` now **measures** the citation against the title-block width and
  shouts if it overruns. It is one unwrapped line, so it can only grow
  sideways, and the next source added to it would otherwise have run off the
  block silently. At present it is 8.6 in in a 12.2 in block.

**The sheet ordered 2026-09-05 does not carry this line**, so the file on disk
and the print on the wall now differ by one credit. Nothing else changed;
`poster_flat.png` is still 29.9 MB.

### Open, if a v2 happens

- **Inset scale bars are still 5.4 pt** while the mini-legends went to 8 pt.
  The rule was that they match, and the constant (`SCALE_PT` in compose.py) was
  not carried forward. Flagged before ordering; left alone deliberately so the
  ordered file matched what had been reviewed.
- **Bleed.** Insets sit exactly `--margin` (0.25 in) from the sheet edge, which
  is the same as a typical 0.25 in trim. If Lumaprints does trim, extend the
  *map* beyond a 34 x 23 trim box rather than moving insets inward — the left
  rail is sized to fill its edge to the cell and will not survive a bigger
  margin. The title block is already clear, at `--safe-bottom` 0.45 in.
- **Scaling up.** Anita may want 1.2x later. Everything is in inches, so this
  is one DPI constant: render at 360 dpi and print at 40.8 x 27.6 for a true
  300 ppi, rather than printing the 300 dpi file bigger and dropping to 250.
- **Top rail covers 4,453 dots (0.10%)**, essentially all Montreal and
  Minneapolis on Canadian dots. Trimming the north removed the empty band that
  rail used to sit in. Shortening those two boxes fixes it at the cost of
  smaller windows.
- The population figure in each caption counts **ancestry responses, not
  people**, and is labelled as such. Real population would need ACS B01003 by
  tract plus the Canadian equivalent.

### Workflow notes for the next poster

- Settle the **frame first**. The sheet changed five times (36x24 -> 34x24 ->
  35x24 -> 34x24 -> 34x23) and each change invalidated the main render and
  sometimes the rail placement.
- **Measure before choosing a number.** Nearly every wasted round was a guessed
  constant that had to be corrected: dist_weight 900, caption 0.80, max_fs 8.5,
  a missing title_min, a character-count label fit. Where the constraint was
  measured first — font advance widths, dot density, the Alaska crop cost — it
  worked first time.
- **Batch feedback.** A single-item change still costs a full pipeline run
  (~2-3 min). Bundled lists were far cheaper per render.

## japanrail

Source map: `riders/japanriders/`. Line thickness is 輸送密度 — passengers per
day passing through each route section — not origin–destination flow. Station
bubbles are a second, independent layer (S12 乗降客数, all operators).

Nothing needs rebuilding upstream. `data/segments.geojson` (4,729 segments,
28,098 route km) and `data/stations.geojson` (8,532 complexes) are the inputs,
and at 10.8 MB the segments file parses in under a second, so there is no
equivalent of ancestrydots' dot bake.

### What the data is concentrated in

Share of density x km, by segment midpoint:

| region              | share | segments |
|---------------------|-------|----------|
| Greater Tokyo       | 53.2% | 2,028    |
| Osaka–Kyoto–Kobe    | 15.1% | 1,229    |
| Nagoya              |  5.0% |   493    |
| Fukuoka             |  1.7% |    29    |
| Sapporo             |  0.9% |     8    |

Throughput percentiles: p50 40,251 · p90 292,486 · p99 871,868 · max 1,442,714
(東北線 神田–東京, the Keihin-Tōhoku / Yamanote quad track). Busiest station
bubble is Shinjuku at 2.20M.

So the national sheet is a shape-recognition map with two knots on it, and the
insets are load-bearing exactly as they were for ancestrydots — more so, since
half the whole dataset lives inside one 4-inch square.

### The tilt (measured)

Japan is a 1,900 km archipelago on a NE–SW axis, so north-up wastes most of a
near-square sheet. Rotating the map clockwise in projected metres lays it flat.
Bounding box of the rail network's own 387,239 vertices, Okinawa excluded:

| rotation | bbox km      | aspect | bbox area |
|----------|--------------|--------|-----------|
| 0 deg    | 1382 x 1560  | 0.89   | 2.16 Mkm2 |
| 25 deg   | 1773 x 1008  | 1.76   | 1.79      |
| 30 deg   | 1823 x  954  | 1.91   | 1.74      |
| 40 deg   | 1881 x  829  | 2.27   | 1.56      |
| 45 deg   | 1888 x  757  | 2.49   | 1.43      |
| 51 deg   | 1879 x  664  | 2.83   | 1.28      | <- minimum

51 deg is the minimum-paper tilt, and for a fixed sheet *area* it buys 32% more
scale than north-up. But that is not the real trade, because these sheets are
sized by *width*: at a fixed 36 in wide the scale barely moves (51.5 km/inch at
25 deg, 54.5 at 40 deg), and what changes is the sheet's height and the shape of
the leftover ocean.

Actual framings at 36 in wide with a 40 km margin:

| rotation | sheet          | km/inch | 1 px @300ppi |
|----------|----------------|---------|--------------|
| 25 deg   | 36 x 21.15 in  | 51.5    | 172 m        |
| 32 deg   | 36 x 18.95 in  | 53.3    | 178 m        |
| 40 deg   | 36 x 16.69 in  | 54.5    | 182 m        |

**25 deg — Anita's pick, settled.** Sheet 36 x 21.15 in, 51.5 km/inch. I had
leaned to 32 on the grounds that the Sea of Japan triangle at 25 deg was far
larger than the insets could fill; with the insets actually built at the size
she asked for, that is exactly what makes it work — Tokyo alone is 10.4 in
square, and 25 deg is the only tilt that leaves room for that *and* a title
block *and* three more cities. At 40 deg and beyond the archipelago's own curve
makes it sag across the middle of the sheet and the leftover space breaks into
corners too small to use.

Only 20.6% of the 25 deg sheet carries rail or land, so there is room for more
than is on it now.

Positive rotation is clockwise, so Kyushu is on the left and Hokkaido on the
right, north points up and to the right, the Sea of Japan runs along the top and
the Pacific along the bottom.

Okinawa is in the data (the 13 km monorail) but 1,000 km off the southwest end;
including it stretches the rotated box from 1,879 km wide to 2,491. It should
get a relocated float, the way ancestrydots handles Hawaii.

### Projection

Lambert **conformal** conic, `+lat_1=30 +lat_2=44 +lat_0=37 +lon_0=137`, then a
plain 2D rotation in projected metres. Conformal rather than the equal-area
conic ancestrydots uses, because nothing here is area-proportional — width is
throughput, bubble area is ridership — while coastline shape and the angles
between lines are what a reader navigates by. Scale error stays under about
0.5% anywhere on the sheet.

Rotating in metres rather than inside the projection definition keeps the
transform trivially invertible, so insets and locator boxes can be placed in
sheet inches without a second projection.

### Basemap

Drawn from vector sources, no MapLibre screenshot:

- **Ocean** — OSM `water-polygons-split-4326`, bbox-filtered to Japan. Fill only
  and stroked in its own fill colour; the file is split into tiles and either an
  unstroked fill or a contrasting stroke draws the tile grid across open sea.
  (The same trap ancestrydots hit twice.)
- **Coastline** — OSM `coastlines-split-4326` as *lines*, so the coast can be
  stroked without touching the water polygons at all. 82,200 ways over Japan.
- **Lakes** — HydroLAKES, >= 2 km2. Natural Earth 10m reads as visibly
  straight-edged at 1 px = 178 m.
- **Prefectures** — COD adm1, available behind `--prefectures`, off by default.

`bake.py` pulls all of these into `build/*.gpkg` once (22 s, mostly the
coastline) so a re-render never touches the 1.26 GB water shapefile.

### Foreign land

Korea sits in the top-left corner of every tilted framing and Sakhalin in the
top-right. With no rail data they render as the same land tone as Japan, which
reads as "nobody travels here" — ancestrydots' Mexico problem, but landing in
the corner where the insets want to go.

`--foreign hide` (the default, Anita's call — "black out Korea and Kurils")
fills the frame **minus Japan's land** with the water colour, opaque, so
foreign land becomes sea. `veil` does the same at 0.72 alpha, which leaves them
as ghosts; `show` leaves them alone. Over open sea the fill is invisible either
way, so nothing is lost by it.

**The Kurils.** The mask comes from the COD adm1 boundary, which is Japan's own
claim, so Hokkaido's geometry carries the Southern Kurils / Northern
Territories out to 148.9E. The map has no data there and takes no position, so
`DISPUTED_CENTROID` in render.py drops Iturup, Kunashir, Shikotan and the
Habomai group from the land mask and they are painted out along with Korea and
Sakhalin.

Selected by **centroid**, not by bounds: Kunashir's western tip (145.40E)
overlaps Yururi and Moyururi (to 145.35E), which are undisputed Japanese
islands off Nemuro and must stay. Centroid >= 145.6E, >= 43.3N separates them
cleanly.

### Width scale

`width_t()` is a port of index.html's knee curve — linear below a knee
throughput, logarithmic above it, slopes matched so there is no kink — so the
poster and the interactive map mean the same thing by a given thickness.
`--knee` is log10 of the knee, default 6 (1M), the web map's own default.

Printed widths are in inches, not pixels, so they survive a DPI change:
`--min-width` 0.005 in (1.5 px at 300 ppi, about the print floor) to
`--max-width` 0.055 in. Segments are drawn **widest first** so a branch stays
visible where it meets a trunk.

### Colour

`--color line` (default) uses the official ラインカラー, `operator` colours by JR
company with everything else grey, `mono` is one colour and lets thickness carry
it alone.

The tables are **parsed out of `riders/japanriders/index.html` by `bake.py`**
(233 line colours, 6 JR operators, `OTHER_COLOR`) rather than copied, so the
poster cannot drift from the web map — several of those entries are hand-lifted
off their official value for dark-background visibility and that work should not
be duplicated.

Note this sits against the usual rule that every colour drawn appears in the
legend: at national scale 233 line colours cannot be keyed. The working idea is
that on the national map colour is regional texture (JR East's green in the
east, JR West's blue in the west, since rural JR falls back to the company
colour) and the *insets* carry small local keys for the lines actually in them.

### Pipeline (built so far)

    posters/japanrail/frame.py     projection + rotation + framing maths
    posters/japanrail/bake.py      basemap extracts + colours -> build/
    posters/japanrail/palette.py   colour lookup, a port of colorLineFor()
    posters/japanrail/render.py    the national map
    posters/japanrail/glowfx.py    the glow behind the lines
    posters/japanrail/insets.py    the three city windows
    posters/japanrail/layout.py    box placement + the coverage check
    posters/japanrail/compose.py   layers, title block and legend
    posters/japanrail/clean.py     drop build artefacts from old tags
    posters/japanrail/typeface.py  Nunito, loaded from ../ancestrydots/

    python bake.py                              # once
    python render.py --scale 0.12               # 20 s composition check
    python render.py --rot 32                   # full 10800 x 6386, 40 s
    python render.py --crop " 139.75,35.69,6,4" # 1:1 texture check, 5 s

Full sheet is 3.3 MB — nowhere near the 100 MB cap, so there is room for 400 ppi
or for the bubbles layer later.

Note the leading space inside the quotes on `--crop`, as in ancestrydots:
argparse otherwise reads a negative number as a flag, and the habit is worth
keeping even where Japan's longitudes are all positive.

### What the first renders showed

- **Full-res texture is good.** The 1:1 Tokyo crop resolves individual private
  lines across the whole Kanto plain; only the Yamanote loop and the ~0.5 in
  core inside it saturate into one mass, which is what the Tokyo inset is for.
- **Land and water were too close.** The first theme had them six levels apart
  and Tokyo Bay was invisible; land is now `#161b22` against water `#080b10`.
- **Light mode is very legible** at whole-sheet scale — arguably more so than
  dark, and it avoids the flat near-black print risk. Undecided; needs a 1:1
  crop and ideally a proof, exactly as ancestrydots did. Unlike ancestrydots the
  palette here is saturated mid-tones rather than pastels, so it does not
  collapse on cream the way the ancestry colours did.

### Settled (Anita, first review)

1. **Tilt 25 deg.** Sheet 36 x 21.15 in, 51.5 km/inch, 1 px = 172 m at 300 ppi.
2. **Both themes.** Dark and light are developed together and both shipped —
   "people might want both". Every script takes `--theme both`.
3. **Korea, Sakhalin and the Kurils are painted out**, not veiled — `--foreign
   hide` is the default, so foreign land becomes sea.
4. **Insets are north-up**, tighter in scope, and large.
5. **Thicker lines.** See below.
6. **Glow.** See below.

### Line weight

The first pass used the web map's own knee (6, i.e. 1M), which puts almost the
whole country on the linear branch: the median segment lands at t = 0.03, which
is the floor, so rural Japan came out as one undifferentiated hairline mesh.

Now `--knee 4.3`, `--min-width 0.008`, `--max-width 0.10` in. That spreads the
curve — median t = 0.33, p90 = 0.71 — and the metro cores saturate into a solid
mass on the national sheet, which is the intent: the national view is meant to
be bold and roughly nonsense in the city centres, and the insets are where the
dense structure is actually readable.

| percentile | throughput | t     | printed width |
|------------|-----------:|------:|--------------:|
| p5         |        644 | 0.006 | 0.20 mm       |
| p25        |      7,578 | 0.073 | 0.37 mm       |
| p50        |     40,251 | 0.328 | 1.00 mm       |
| p90        |    292,486 | 0.711 | 1.86 mm       |
| max        |  1,442,714 | 1.000 | 2.54 mm       |

Judge this from `--crop`, never from a scaled preview: at `--scale 0.12` a
0.008 in line is a third of a pixel and simply vanishes, which is what made the
first pass look thinner than it was.

### Glow (built)

`glowfx.py`. The interactive map had a glow and dropped it twice over — too
expensive per frame, and drawn per segment it left a bright notch at every
joint, worst on curves, because each segment's halo ended at its own cap.

Neither problem survives a static render. The network is already one raster
when the glow runs, so blurring it blurs a continuous shape, corners included,
once per render rather than once per frame.

Two things it has to get right:

- **Premultiplied alpha.** Blurring straight RGBA drags the colour of fully
  transparent pixels into the halo and every glow greys off toward the
  background. Multiply by alpha, blur, divide back out — and divide by the
  *unclipped* alpha, or the cap darkens exactly the dense areas it exists to
  tame.
- **Two radii.** One blur gives either a tight rim or a broad haze. The tight
  pass reads as the line glowing; the wide pass as light in the air around it.

Tuned on a 1:1 Tokyo crop. A first attempt at strength 0.85 / wide 0.45 read as
a coloured haze lying over the whole Kanto plain — the wide alpha of forty
lines simply adds to 1 and the ground stops being dark. Defaults are now
`--glow 0.018 --glow-strength 0.60 --glow-wide 4.0 --glow-wide-strength 0.18
--glow-cap 0.80`, which keeps it a rim.

The radius is in **inches of printed sheet**, so an inset at 7x gets the same
halo on paper as the national map. That is what makes the two read as one
poster rather than as two maps.

Computed at quarter resolution and scaled back: a Gaussian this wide is smooth
by construction, so it is visually identical and sixteen times cheaper, which
matters at 68 megapixels. Costs about 15 s on the full sheet.

On light it reads as ink bleed rather than neon, and is worth keeping there too.

### Insets (built)

`insets.py`. North-up, because a magnified window already reads as a city map
and rotating it as well makes the reader do two things at once for no gain —
the frame says it is a different scale.

Windows are declared as a **scope in km**, not as a magnification, so "how much
country does Tokyo need" is one number to argue about; the magnification falls
out of that and the box size the layout gives it.

| inset   | scope        | box            | scale        | 1 px  |
|---------|--------------|----------------|--------------|-------|
| Tokyo   | 78 x 66 km   | 10.4 x 10.4 in | 6.9x, 7.5 km/in | 25 m |
| Osaka   | 78 x 66 km   | 9.5 x 8.0 in   | 6.2x, 8.3 km/in | 28 m |
| Nagoya  | 56 x 46 km   | 6.5 x 5.6 in   | 6.0x, 8.6 km/in | 29 m |
| Fukuoka | 44 x 32 km   | 6.5 x 4.8 in   | 7.6x, 6.8 km/in | 23 m |

Two windows were wrong first time and are worth recording:

- **Tokyo** at a 310 x 206 km crop carried most of Shizuoka. 78 km is Omiya to
  Yokohama, Tachikawa to Chiba, which is the story.
- **Osaka** at 88 km centred 0.04 deg further north spent its left third on the
  empty Tanba hills and *still* lost the Nara and Wakayama lines off the bottom.
  Osaka sits low in the Keihanshin triangle because Kyoto is at the top of it.

**Fukuoka is the weak one.** At the 72 km "north Kyushu" scope that reached
Kitakyushu it was two long red JR lines and a lot of empty ground. Tightened to
the city, where the subway and Nishitetsu are, it is better but still far
sparser than the other three. Candidate for dropping, or for swapping to
Sapporo or Sendai.

### Layout (built)

`layout.py`. Deliberately **not** an optimiser — the composition is Anita's, so
the boxes are declared and the script's job is to check them:

    title    0.40, 0.40   11.60 x 4.60      top-left
    fukuoka  0.40, 5.40    6.50 x 5.42      under the title
    osaka    7.50, 5.40    9.50 x 8.62      Sea of Japan, above Osaka
    nagoya  17.60, 0.40    6.50 x 6.22      Sea of Japan, above Nagoya
    tokyo   25.20, 9.60   10.40 x 11.02     Pacific, right of the country

Coverage is measured off the rendered layers rather than from geometry: the
lines PNG's alpha is exactly the ink a box would hide, and the base PNG's land
colour is exactly the land, with foreign land already painted out.

**Rail ink and land are counted separately.** Hiding a rail line is a real
loss; a box on empty Tohoku hillside costs almost nothing, and one summed
figure ranks those the same and pushes every inset out to sea for no reason.
Ancestrydots could use a single number because there the content *was* the
dots. Result: 2.9% of rail hidden, essentially all of it Tokyo's box clipping
the Sanriku coast, against 4.9% of land.

`--search` reports the cheapest placements inside each box's declared region,
which is how the numbers above were chosen rather than guessed.

**Osaka sits left of Nagoya because their cities do.** With the boxes the other
way round — Osaka in the wider top-middle slot — the two leader lines crossed
over the Chugoku coast, which is the same defect ancestrydots' free placement
kept producing.

Locators are drawn as **rotated quadrilaterals, not boxes**: the sheet is
turned 25 deg and the insets are north-up, so a north-up window lands on the
map tilted by exactly that much.

### Compose (built)

`compose.py` writes five layers plus a flat, at 10800 x 6345, per theme:

    poster_1_base_<theme>.png     ocean, lakes, coastline        1.5 MB
    poster_2_glow_<theme>.png     glow behind the lines          7.8 MB
    poster_3_lines_<theme>.png    the national network           1.9 MB
    poster_4_insets_<theme>.png   insets and captions            6.9 MB
    poster_5_frames_<theme>.png   borders, locators, leaders     0.3 MB
    poster_flat_<theme>.png       all of it                     10.9 MB

Well under the 100 MB cap either way. Tagged **sRGB** on write, as ancestrydots
now does — everything here is authored in sRGB and Lumaprints recommends Adobe
RGB, so an untagged file risks being read as Adobe RGB and printing
oversaturated.

Full pipeline:

    python bake.py                          # once
    python render.py --theme both --tag v8  # 88 s
    python layout.py                        # check + layout.json + plan image
    python insets.py --from-layout --theme both
    python compose.py --theme both --lang both --tag v8
    python clean.py --yes                    # drop old tags

### Gotcha: empty layers kill geopandas

A tight inset with no lakes in its bbox (Fukuoka) returns an empty GeoDataFrame,
and `.plot()` on one in a geographic CRS dies computing an aspect ratio from a
NaN latitude. `read_rotated()` returns `None` for an empty read and every caller
checks.

### Second review (Anita) — what changed

**Sheet grew to 38.6 x 21.15 in.** The Tokyo inset is not allowed to touch
Japan at all. Land reaches x = 27.38 in across the rows that box occupies, so
`--extend-right 2.6` adds paper on the right **at the same km/inch** — the map
does not shrink to make room, the sheet grows. Tokyo now hides 0.0% of rail and
0.0% of land, against 2.9% before.

`--extend-left/right/top/bottom` are all in inches and are applied after the
fit, so the scale is fixed before the sheet changes. They are written into
`frame_<tag>.json` so layout.py and compose.py cannot drift from the render.

Growing the sheet grows Tokyo one for one: every extra inch of sheet width is
an extra inch of inset. 38.6 in is the minimum that clears the land.

**Three insets, not four.** Fukuoka is dropped. Even tightened to the city it
was far sparser than the other three, and it is the *largest* of the remaining
candidates — Sapporo and Sendai would be sparser still. Its box became the
legend block.

**Two knees, and the inset one is higher.** National 5 (100k), insets 6 (1M).
This is the right way round and worth understanding: on the national sheet the
data runs from 200 to 1.4M and a low knee flatters the branch lines, making a
40,000/day country line look like a serious trunk. Inside a metro almost every
segment is already between 100k and 1.4M, so the national curve would put them
all near the top and flatten them into one weight; putting the knee at the top
of that range spends the whole width ladder on the range that actually varies.

I had 4.3 nationally after the first review. Anita reads 5 as more honest and
it is — the median segment is now a third of the width it had, and the ratio
between a main line and a country branch is much closer to the real one. Rural
lines survive on the `--min-width` floor rather than on curve shape, and at 1:1
they still read clearly; it is only the scaled previews that make them look
lost.

**Lines at 70% opacity** (`--line-alpha 0.70`), so a crossing reads as a
crossing rather than as whichever line happened to be drawn last. Widest-first
ordering already kept branches visible; this is what makes a Tokyo trunk bundle
look like several lines instead of one opaque ribbon.

**Glow halved** — strength 0.60 -> 0.30, wide 0.18 -> 0.09.

**Station bubbles are on, insets only.** Area proportional to daily 乗降客数, as
on the web map. `BUBBLE_SCALE` 1.9e-5 puts Shinjuku's 2.20M at about 0.7 mm of
radius: findable, and small enough not to bury the junction it sits on. A
3,000/day floor drops roughly half the country's stations, all of them dots too
small to print. 3,017 bubbles survive in a typical inset window. They are not on
the national map — at 51 km/inch they would be noise on top of already
saturated metros.

### Title and legend blocks (built)

`compose.py` draws both from `layout.json`.

- **Title block** (0.40, 0.40, 11.60 x 4.60): "Japan Rail Flow", the subtitle,
  what 輸送密度 actually measures (and what it is not — not trains, not O-D),
  what the station circles count, and the source and vintage lines.
- **Legend block** (0.40, 5.40, 6.90 x 5.40), in the space Fukuoka left: the
  thickness ladder, the bubble key, the 200 km scale bar and the projection
  note.

The ladder is drawn from `frame_<tag>.json`'s `width_scale` — the knee and the
min/max widths the render actually used — so the legend cannot quietly disagree
with the lines it explains. Its top rung is the real busiest segment
(1,440,000, Kanda–Tokyo) rather than a round number the map never reaches.

**The title block is Latin-only**, because Nunito is: the native terms would
set as tofu. `yuso mitsudo`, "Railway Transport Density data", "Metropolitan
Transportation Census". Those are the strings to replace when a CJK face lands.

### Gotcha: scale bar labels ran outside the frame

The inset scale bars were placed at a fixed offset from the box's right edge and
the label drawn after the bar, so "10 km" printed *outside* the inset, on the
sheet. `scale_bar()` now takes a `right_edge` and right-aligns the whole
bar-plus-label group, measuring the label with `textlength` rather than
assuming a width.

### Third review (Anita) — what changed

**Thickest lines draw last, on top.** They were drawn widest-first, which put
the quiet lines on top; that protects a rural branch where it crosses a trunk,
but it chops the Tokaido into pieces wherever anything crosses it. Thickness is
the message on this map, so the busy line wins. Matches the interactive map.

**Lines are opaque again.** 70% was tried and rejected: a line's own segments
overlap each other at every join and at every doubled-back stretch, so a partly
transparent line goes blotchy along its own length. The transparency read as a
rendering fault rather than as depth.

**Station bubbles, much bigger, and on both maps.** `BUBBLE_SCALE` went
1.9e-5 -> **9.0e-5**, putting Shinjuku's 2.20M at 0.13 in of radius (3.4 mm) —
a little over twice the widest line. At 0.7 mm they had read as noise on the
line rather than as stations. They are translucent (alpha 0.45) so a run of
them builds up along a line instead of masking it, which is what makes the
clusters at Shinjuku and Umeda appear.

The national map gets them at **a third of the inset scale** (3.0e-5). At the
full inset size the country would be a string of beads with no line left
showing; a third keeps the big cities legible as clusters. The 3,000/day floor
stands — 3,017 bubbles survive, and everything dropped was too small to print.

### Layout: text row across the top, insets top-aligned below

    title    0.40, 0.40  11.60 x 3.40     left margin
    legend  12.40, 0.40   7.50 x 3.40     immediately right of the title
    osaka    0.40, 4.20  11.00 x 8.27     left margin, top-aligned with Nagoya
    nagoya  11.90, 4.20   8.00 x 7.22     top-aligned with Osaka
    tokyo   27.75, 9.60  10.45 x 11.07    unchanged

**All three insets now hide exactly zero land and zero rail.** That constraint
is what sets the row's height: land starts at y = 12.5 (Noto and Sado) and
y = 13.0 (west Kyushu), so a row beginning at 4.20 can be at most about 8.4 in
tall including its caption. Osaka is 8.27 and clears by a sixth of an inch.

Both grew: Osaka 9.5 -> 11.0 in wide, Nagoya 6.5 -> 8.0 in (6.0x -> **7.4x**,
now the most magnified of the three).

The legend moved out from under the title to beside it, which is what freed the
whole left column for the inset row.

**Osaka's centre moved east to 135.57.** Its box is wider than the declared
scope's aspect, so `inset_frame` widens the window (78 km -> about 95 km) rather
than cropping; centred as it was, the whole of that extra width fell on empty
Hyogo. This is the third time this window has needed moving — it is the one to
re-check whenever its box changes shape.

### Legend, rebuilt tight

The first version sprawled: 1.70 in bars at 0.46 in spacing read as five
separate rules rather than as one scale, and it ran past its block into the
Nagoya inset. Now 0.62 in bars at 0.215 in spacing, following the interactive
map's own legend where the swatches are 34 px wide.

The bubble key is **two columns**, "on the main map" and "in the city insets",
because the two are drawn at different scales and a single key would be wrong
for one of the two places a reader looks. The main-map column's circles come
from the render's own `bubble_scale`, out of `frame_<tag>.json`, so they cannot
drift from the map — the same arrangement the thickness ladder already had.

### Scale bars

The 200 km bar has moved **out of the legend block and onto the map**, in the
open Pacific south of the Kii peninsula at (22.30, 20.35). Sitting under the
thickness ladder it read as one more rung of it. This is also where ancestrydots
puts its own.

Both the map bar and the inset bars are thicker (0.022 in, was 0.010) and the
label is now vertically centred on the bar rather than sitting on its baseline —
a thin rule under the text read as a long underscore, not a measurement.

### Fourth review (Anita) — what changed

**The dark theme's hues were the wrong way round.** Land was `#161b22`, a
blue-grey, against a neutral near-black sea — which reads as the sea being the
ground and the land being the water. Swapped to match the light theme's
relationship: land neutral (`#16181b`), water blue (`#091320`), land still the
lighter of the two.

This broke layout.py silently, and it is worth knowing why: `content_grid()`
was detecting land with `blue channel > 0x11`, which inverted the instant water
became the blue one. It now classifies each cell by **nearest theme colour**,
so it survives any repalette.

**Line widths up 15%** — national 0.0092 to 0.115 in, insets 0.0092 to 0.132.

**Main-map bubbles at half the inset scale**, not a third (4.5e-5 against
9.0e-5). At a third the stations barely registered at national scale.

### Type floor: 8 pt at 30 inches

Everything is authored at the sheet's design width, but the sheet may well be
ordered smaller, and the binding case is a 30 in print: at 38.6 in of design
that is a 0.777 scale, so the old 9.5 pt body landed at 7.4 pt on paper and the
7.5 pt captions at 5.8.

`compose.py` now derives a floor — `MIN_PT * sheet_width / MIN_PT_AT_IN`,
currently **10.3 pt** — and every size passes through `font()`, which clamps.
It prints the floor on every run.

That forced a second fix. The title and source copy had been hand-broken into
fixed lines, which silently overran the block the moment the type grew; it is
now written as paragraphs and **wrapped by measured advance width** to the
block. `compose.py` also reports if either text block needs more height than
layout.py reserved for it, rather than quietly running into the inset below.

### Insets: lower, larger, and allowed onto the small islands

Anita's call: covering Tsushima, Iki and the Oki islands is fine. Insisting on
zero land anywhere was what held the Osaka/Nagoya row up and kept it small.
`LAND_EXEMPT` in layout.py names those three as circles in lon/lat/km and zeroes
them out of the land grid.

That bought 1.35 in. **The real floor is the San'in coast of Honshu**, which the
box bottoms hit at y = 13.85 — measured, not guessed: the covered cells came
back as lon 130.9–133.5, lat 34.4–35.6, which is Shimane and Tottori, not an
island.

    osaka    0.40, 5.00  11.60 x 8.82   6.4x, 8.05 km/in   (was 11.00 x 8.27, 6.0x)
    nagoya  12.50, 5.00   8.60 x 7.82   7.9x, 6.51 km/in   (was  8.00 x 7.22, 7.4x)
    tokyo   27.75, 9.15  10.45 x 11.57  6.9x, 7.46 km/in   (was 10.45 x 11.07)

All three still hide zero rail; Nagoya clips a single land cell.

**Tokyo, re-aimed south.** The box was square, so the window came out 78 x 78 km
centred at 35.700 — which spent 11 km of empty Saitama above Omiya and cut
Fujisawa off the bottom by 1 km. Taking the box to 10.45 x 10.95 in makes the
window 78 x 81.7 km, and moving the centre to **35.575** puts it at
lat 35.207–35.943: Omiya (35.906) stays in with 4 km to spare, and Yokosuka
(35.28), Kurihama, Fujisawa and Ofuna all come in.

Omiya is the hard northern constraint and should be written down as one — the
Joetsu Shinkansen branches off the Tohoku there, and that junction is the thing
the north of this inset exists to show.

The box was raised to y = 9.15 rather than extending the sheet: Hokkaido's
content reaches x = 30.9 at y = 8.5, so 9.15 is about as high as it can go, and
the bottom is already flush with the margin. Extending the sheet downward would
buy more, at 46 sq in of blank Pacific per inch.

### Fifth review (Anita) — what changed

**Wider range, not just wider lines.** Ceiling up another 15% (national 0.115 ->
0.132 in, insets 0.132 -> 0.152) and the floor down 40% (0.0092 -> 0.0055 in,
both). Opening the gap is what makes a busy line read as *busy* rather than
merely thicker than its neighbour.

At the national knee of 5, on the 38.6 in sheet:

| percentile | passengers/day | printed | at a 30 in print |
|------------|---------------:|--------:|-----------------:|
| p5         |            644 | 0.15 mm |          0.11 mm |
| p25        |          7,578 | 0.21 mm |          0.16 mm |
| p50        |         40,251 | 0.50 mm |          0.39 mm |
| p75        |        143,760 | 1.37 mm |          1.06 mm |
| p90        |        292,486 | 2.01 mm |          1.56 mm |
| max        |      1,442,714 | 3.35 mm |          2.61 mm |

**0.11 mm at a 30 in print is the number to watch on the proof** (0.13 mm after the round-6 raise below). That is about
as fine as an inkjet holds on fine art paper, and it is the quietest rural
branches — the Sanriku and San'in ends of JR East and JR West — that live there.
If the proof loses them, raise `--min-width` before anything else.

**Tokyo moved 12% of its window height back north**, centre 35.575 -> 35.663.
35.575 had overcorrected. The window is now lat 35.295–36.031, so:

- Omiya (35.906) still in, with 13 km of margin — it is the hard northern
  constraint, since the Joetsu Shinkansen branches off the Tohoku there.
- Kamakura, Ofuna and Fujisawa in.
- **Zushi (35.295) is now the southern limit and Yokosuka-chuo (35.276) falls
  1.5 km outside**, so the tip of the Miura peninsula is cut. Going 8% rather
  than 12% would keep it; that is the trade if Yokosuka matters more than the
  Saitama the extra northing buys.

**Light-mode bubbles are white with a translucent rim** — fill `#ffffff` at
0.62, rim `#151a20` at 0.30, 0.005 in wide. The dark bubble on cream had read
as a hole punched in whatever line it sat on; white on white needs more body
than white on near-black, hence the higher fill alpha, and the rim is what
actually locates it.

### Gotcha: the bubble layer was theme-blind

`render.py` rendered the national bubbles **once, with `"dark"` hardcoded**, and
reused that layer under both themes. That was invisible while both themes drew
the same near-white circle and became wrong the moment the light theme got its
own treatment. Bubbles are now rendered per theme
(`bubbles_<theme>_<tag>.png`), which is what every reader of that file —
layout.py, compose.py — had to be told about too.

The insets had the same shape of bug in a milder form: `render_bubbles` was
passed the right theme but wrote to a theme-less filename, so the two themes
overwrote each other between composites. Also renamed.

`compose.py`'s legend key now mirrors `render.THEMES` for fill, rim and alpha,
so the circles in the key are drawn exactly as the map draws them.

### Sixth review (Anita) — what changed

**Widths up another 10%**, both ends this time: national 0.0061 to 0.145 in,
insets 0.0061 to 0.167. The complaint was that they still read thin in the
smaller regional areas, where nothing is near the ceiling and the whole picture
sits in the bottom half of the ladder.

At the national knee of 5, on the 38.6 in sheet:

| percentile | passengers/day | printed | at a 30 in print |
|------------|---------------:|--------:|-----------------:|
| p5         |            644 | 0.16 mm |          0.13 mm |
| p25        |          7,578 | 0.23 mm |          0.18 mm |
| p50        |         40,251 | 0.55 mm |          0.43 mm |
| p75        |        143,760 | 1.50 mm |          1.17 mm |
| p90        |        292,486 | 2.21 mm |          1.72 mm |
| max        |      1,442,714 | 3.68 mm |          2.86 mm |

The floor is now 0.13 mm at a 30 in print rather than 0.11, which is a little
more comfortable against what an inkjet holds on fine art paper. Still the
first number to revisit if a proof loses the quiet branches.

**Leader lines removed.** The faint hairline running from each locator square
across the country to its inset box read as one more railway — which, on a map
whose entire content is faint coloured hairlines, is the single thing it must
not do. The locator square on the map and the city name in the caption carry
the pairing instead, which is what ancestrydots does too.

`layout.py`'s plan image still draws them, at lower alpha: while boxes are being
placed the pairing needs to be obvious, and that image is a working tool rather
than the poster.

### Seventh review (Anita) — copy and caption

**Inset caption padding is now equal on all four sides.** Everything in the
strip is placed off *baselines* computed from `CAP_PAD_IN` (0.16 in) and the
visual line heights, rather than anchored by ascender, which had left the space
under the figures visibly adrift from the other three sides.

`visual_h(pt)` is 0.72 em — where Nunito puts both caps and ascenders, and so
what reads as the height of a line. layout.py's `CAPTION_IN` has to equal
`2*CAP_PAD + visual_h(name) + gap + visual_h(sub)`, which at 0.16 / 14 pt / a
10.3 pt floor is 0.618; the 0.62 it reserves is that number, not a guess.

**Copy, rewritten to Anita's text.** Subtitle dropped. The white paragraph is
now three short sentences instead of two long ones, and the small print is
`Sources: …` / `Projection: …` / `interactive version at
anita.garden/japanrail/` on their own lines, with the em dashes taken out —
they read as gaps at that size.

Legend headings: "Segment passengers per day" and "Station passengers per day",
down from the two clause-length versions.

**Bigger type, narrower measure.** Body 15 pt, small print 11.5 pt, both well
clear of the 10.3 pt floor. The title block is 12.6 -> **9.2 in**, which is
about 100 characters at the body size; one 12.6 in line of small type was a
very long way for an eye to travel back along. The legend moved left to 10.2 to
close the gap that opened up.

### Gotcha: a locked file killed a whole compose

Windows refuses to truncate a file another process holds open — an image viewer
left on one layer is enough — and PIL surfaces that as a bare
`OSError: [Errno 22] Invalid argument` on the save. It aborted a two-theme
compose halfway, leaving the dark layers fresh and the light ones stale, which
is a much worse state than failing outright.

`compose.py` now writes every layer to a temp file beside the target and
`os.replace()`s it into place. The rename succeeds against most viewer locks,
and where it does not, the message says which file is held and that nothing
else was lost.

**Do not probe for a lock by opening a file `'w+b'`** — that truncates it. Two
of these files were zeroed that way while diagnosing the above. They are build
artefacts and one compose run brought them back, but the check to use is
`'r+b'` (opens without truncating) or simply attempting the real write.

### Eighth review (Anita) — caption, water, housekeeping

**Inset captions are now a name and a scale, nothing else.** The city name is
alone on the left at a size **derived from the strip** — `(CAPTION_IN - 2 *
CAP_PAD_IN) / 0.72 * 72`, which at 0.62 and a 0.14 pad is 34 pt — so it fills
the space between equal paddings and stays in step if the strip height ever
moves. On the right, the magnification sits over the scale bar as "6.9x scale".

The km-per-inch figure is gone: it is the same fact the bar states, said less
usefully. `scale_bar()` now returns `(x, width)` so the magnification can be
right-aligned to the **bar's** end rather than the box's, which is what puts it
over the bar instead of over the "10 km" that follows it.

`CAP_PAD_IN` went 0.16 -> 0.14 to give the name more of the strip.

**Osaka's inset is labelled "Kansai".** It covers Kobe, Osaka, Kyoto and Nara,
and three names in a row was doing the work one does.

**Water is `#111c2a`.** `#091320` was dark and saturated enough to read as a
hole rather than as sea. Note this makes water very slightly *lighter* than
land (26.6 against 23.8 in luminance) where it used to be darker — the two are
now separated by hue more than by value, which is what the light theme does.

**Copy:** "Passenger counts are per day", not "are daily".

### clean.py

Every render is tagged and iterating leaves a tag's worth of layers behind, plus
the one-off `--crop` tests named after wherever they were cropped. `clean.py`
keeps the bake, the layout, the current tag and the composed poster, and removes
the rest; it reads the tag to keep out of compose.py's own default and the city
list out of `insets.CITIES`, so it cannot fall behind them. Dry run by default,
`--yes` to delete. First run cleared 152 files and 172 MB.

### Finished sheet — the numbers

| | |
|---|---|
| Sheet | **38.6 x 21.15 in**, aspect 1.825 |
| Pixels | **11580 x 6345** |
| Native | 300 ppi at 38.6 in |
| Scale | 51.5 km/inch, 1 px = 172 m |
| Files | `poster_flat_dark.png` 9.4 MB, `poster_flat_light.png` 10.4 MB |

Both are far under Lumaprints' 100 MB cap, and the file is over-specced for any
print smaller than its design size: at 36 in long it is 322 ppi, at 30 in it is
386. 38.6 in is the largest print that stays at a true 300.

Printed smaller, everything scales with it — which is what the 10.3 pt type
floor and the 0.16 mm line floor are pinned against; both are quoted for a
30 in print in the sections above.

**City names went back to a set 28 pt.** Deriving them from the caption strip
filled it exactly and came out at 34 pt, which was too heavy against the
poster's own 46 pt title. Both the name and the scale group are centred in the
strip rather than padded from its top.

### Locked files are now non-fatal

A held file no longer aborts the run. `save()` records the failure, leaves the
new version beside the target as `<name>.tmp.png`, and compose reports them
together at the end — so eleven layers land and one needs a rename, rather than
the run stopping at layer two with a mix of fresh and stale outputs on disk.

This fired for real on `poster_2_glow_dark.png` the same session it was written.

### Japanese edition (built)

`compose.py --lang {en,ja,both}`. Japanese output is suffixed `_ja`; English
keeps the plain names. Four sheets in all: dark/light x en/ja.

The Japanese edition is arguably the native one — every line, operator and
station name in the data is Japanese, and the English map is the translation.

**Face.** Nunito has no CJK. `typeface.py` looks for, in order: a Noto Sans JP
dropped into the project directory (SIL OFL, the licence-clean choice if these
are ever sold), then **Yu Gothic** (`YuGothR.ttc` / `YuGothB.ttc`), which ships
with Windows and is a real Japanese face with correct glyph forms, then MS
Gothic as a last resort. `.ttc` files are collections; index 0 is "Yu Gothic",
index 1 the narrower "Yu Gothic UI".

**Now rendering with Noto Sans JP**, cut by `make_fonts.py` the same way
ancestrydots cuts Nunito — and for a sharper version of the same reason. The
Google Fonts `NotoSansJP[wght].ttf` is variable and its **default instance is
Thin (wght 100)**, so neither matplotlib nor PIL, which set no variable axes,
would render anything but hairlines; at 8-11 pt on paper that is not "a bit
light", it is blank. Instanced to Regular 400 and Bold 700, 5.5 MB each.

    curl -sSL -o NotoSansJP.ttf \
      'https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf'
    python make_fonts.py

Measured against Yu Gothic on the strings compose.py actually sets (ink height
as a fraction of the em, so the comparison is of rendered glyphs and not of
point size):

| string | Noto Sans JP | Yu Gothic |
|---|---|---|
| city names | 0.930 | 0.905 |
| legend heading | 0.910 | 0.875 |
| body sentence | 0.930 | 0.925 |
| magnification | 0.940 | 0.945 |

Noto runs 0-3% taller. `EM_VISUAL["ja"]` stays at **0.88**: it was already under
Yu Gothic's real ink on three of those four strings, so it is a line-height
constant chosen to make the composition look right rather than a bound the face
has to satisfy, and moving it would shift the Japanese sheets' typography to fix
a problem nothing has reported.

**Two things had to become language-aware:**

- `EM_VISUAL` — Nunito's caps and ascenders sit at 0.72 em, but a CJK face
  fills about 0.88 of its em, so the same point size reads much bigger and the
  padding maths needs to know which it is. `JA_PT_SCALE` (0.86) also brings
  every size down, applied *before* the type floor is clamped so nothing falls
  under 8 pt at a 30 in print.
- `wrap()` — Japanese has no spaces, so the word wrapper returns a whole
  paragraph as one line. It breaks per character for `ja`, with enough kinsoku
  not to strand punctuation at a line start (`JA_NO_LINE_START` /
  `JA_NO_LINE_END`).

**Gotcha in the character wrapper, worth remembering:** the first version
handled the two kinsoku cases and then *fell through* to appending the
character, so the plain "flush and start a new line" case never ran and every
paragraph came out as one long line straight across the legend. Each branch
needs its own `continue`.

**Second gotcha, same wrapper: the Japanese copy is not purely Japanese.** The
sources paragraph carries `OpenStreetMap contributors`, `HydroLAKES`,
`Messager et al. 2016`, `gtfs-gis.jp`, and a per-character wrapper breaks
straight through the middle of them — adding the HydroLAKES credit pushed the
line boundary into "Mes / sager". `JA_LATIN_RUN` now backs the break up over a
trailing run of alphanumerics and carries the whole token down, with the
degenerate case (a token longer than the measure) still allowed to overhang.
Alphanumerics only, deliberately: a break at the hyphen in `gtfs-gis.jp` or the
slash in `anita.garden/japanrail/` is a real break opportunity and should stay
available. Worth knowing that this defect only appears when the copy changes —
the wrap was correct until a credit moved the boundary.

**And the fix needed a fix, which is the part worth remembering.** Backing up
over the Latin run lands the break wherever the run starts — and that can be
immediately after an opening bracket, stranding it at the line end. The real
case was `湖沼は HydroLAKES（` / `Messager et al. 2016）`: kinsoku had already
been handled for the character *before* the break, and the backup then
reintroduced the same defect one step later. The backup now also walks over any
trailing `JA_NO_LINE_END`. Caught by looking at a 1:1 crop, **not** by the test
that checked for mid-token breaks — that test passed on the broken version,
because it was only asking the question the first fix had raised.

**The Japanese copy is DRAFT and wants a native read before printing.** The
terms of art come from the sources themselves (輸送密度, 通過人員, 乗降客数,
大都市交通センサス, 国土数値情報) and should be right; the connecting prose is
mine. It all lives in one place, `STRINGS["ja"]` in compose.py.

City names: 東京 / 関西 / 名古屋. Magnification reads 「6.9倍」.

### City names at 20 pt

28 pt measured correct — "Kansai", which has no descender, came out at exactly
its 85 px cap height — but "Tokyo" and "Nagoya" carry a descender that takes
the line to 105 px of the strip's 186, which is what made them read large. At
20 pt those are 61 px and 75 px, or a third and two fifths of the strip.

Worth keeping the method: **measure the rendered glyphs** rather than trusting
the point size. A descender is half the apparent size of a short label.

### Locked-file retry

The file that failed to be replaced moved between runs — glow_dark once,
lines_dark and insets_light the next — which is a scanner (search indexer or
antivirus) holding whichever file was written last, not a viewer parked on one
layer. `save()` now retries the rename four times over about two seconds before
giving up, and a full four-edition compose then runs clean.

### Still open

- **Place labels.** The national map still has none.
- The **numbered-top-segments margin table** — room in the strip beyond
  Hokkaido and in the gap between Nagoya's box and Tohoku.
- **Okinawa** is still absent.
- **The Japanese copy needs a native read** before printing — see the
  Japanese-edition section. Nothing on the map itself is labelled in
  either language yet.
- The light theme's `OTHER_COLOR` grey (#9aa0ab) for private and third-sector
  lines is weak against cream. Wants a per-theme override, the way the bubbles
  now have one.
- **A proof.** Everything above is judged on a screen. The two things a
  physical print will settle are the 0.11 mm floor and whether the dark sheet's
  large flat near-black bands (NOTES' standing worry) behave on semi-gloss.

## nycriders

Source map: `riders/nycriders/`. Stripe width is riders per day through a track
segment (the O-D estimate routed through the timetable, summed over 24 h);
station bubbles are boardings — entries plus transfers — at a complex. Both come
straight out of `stats.json` (2.4 MB, parses in a fifth of a second), so there is
no bake step for the data, only for the basemap.

Single static view, not the small-multiples grid the earlier plan floated. That
plan is still worth trying later; this is the one-sheet version.

### The junction rendering had to be fixed upstream first

The interactive map fanned parallel routes by grouping segments on their
`(from_stop, to_stop)` pair, so two routes were only drawn side by side where
they shared a station pair. Track that two routes physically share but reach
from different stations collapsed onto one centreline: the E and F stayed
parallel to 5 Av/53 St and then piled up for the last block before the switch,
and B/D/F/M overlapped for the whole approach to 47-50 St and spread out again
south of it. Printing that would have printed the bug at 300 ppi.

Fixed in `riders/nycriders/index.html`, and the poster carries the same fix:

1. **Shared track is recovered from geometry, not from stop IDs.** GTFS shapes
   reuse identical vertices wherever trains share track — the E's and F's
   polylines west of 5 Av/53 St are literally the same two coordinates — so a
   graph of vertices and edges, each edge tagged with the routes on it, gives
   the real sharing. Edges with the same route set chain into a RUN. Runs are
   also cut at stations, because that is where a stripe's width changes.
   552 runs, 8,407 edges, 969 route/stop-pair features.
2. **Handoff at a run boundary.** A run's route set changes at its ends, so the
   fan re-centres and every stripe jumps sideways — small at a station, most of
   a bundle's width at a switch, which reads as a broken line. The biggest
   bundle at the node holds its offsets and everything else ramps to meet it, so
   a pair that arrives together stays together *through* the switch and only
   afterwards drifts onto its own centreline. Averaging the two offsets instead
   was tried first and is wrong: it pulls two lines together exactly where they
   are parting.

Checked against a graph-free baseline: of 336 stop-pair groups that the old
grouping fanned, only 3 lose any sharing under the new one (the R against E/F on
Queens Blvd, the R against N/Q on the Astoria line), and in those the R is
genuinely on a different alignment.

### The web map's staircase, and why the poster does not have one

`line-offset` is one number per feature, so on the web the ramp has to be
approximated by a chain of short pieces at stepped offsets. Three artefacts came
out of that, in order of discovery:

- **Round caps smear.** Each piece's cap reaches half a line-width past its own
  piece, so a chain of short pieces at drifting offsets unions into a widening
  lumpy band, and at 0.85 opacity the overlaps composite brighter as well.
  `line-cap: butt` fixes both — every piece now meets its neighbour at the same
  offset by construction, so butt ends close up seamlessly. The highlight layers
  had to follow.
- **Round caps on the constant-offset pieces cause pills.** A short middle piece
  between two long ramps is basically two half-circles poking out sideways.
  Same fix; the whole layer is butt now.
- **A cut at a polyline vertex leaves a nick.** Two butt-ended pieces meeting at
  a bend face different ways, so the outside of the turn opens a wedge. Cuts are
  now nudged toward the middle of whichever straight stretch they land in
  (`cutAwayFromVertex`), which closes them.

Ramp steps are 0.15 px of offset, up to 32 per ramp; feature count goes from 969
to about 5,600 and `renderStaticSegments` from ~5 ms to 13-19 ms, which is still
inside a scrub frame.

**None of this applies to the poster.** There is exactly one scale on a sheet, so
the offset is baked into the coordinates and varied per vertex, and the splay
comes out as a real smooth curve — no steps, no caps to reconcile, proper mitred
joins. The web map's version is the compromise; the printed one is the clean one.

### Pipeline (built)

    posters/nycriders/frame.py     projection + framing maths
    posters/nycriders/palette.py   route colours, draw order, sort overrides
    posters/nycriders/ribbons.py   shared-track graph, fan, baked offsets
    posters/nycriders/bake.py      basemap extracts -> build/
    posters/nycriders/render.py    the sheet

    python bake.py                                   # once, ~20 s
    python render.py --scale 0.12                    # composition check, 1 s
    python render.py --theme both                    # full 7200 x 10800, 37 s
    python render.py --crop " -73.988,40.700,3.2,2.4"  # 1:1 texture check, 1 s

`ribbons.py` is the interesting file. In order: WELDS, the shared-track graph,
`orient_runs`, `fan_relative` + `solve_centres` + `absolute`, `handoff`,
`smooth_nodes`, `joint_normals`, `ribbon`. The three knobs worth turning are
`--centre-reg` (how hard a bundle is pulled back onto its own track),
`--smooth-passes` / `--smooth-cap` (corner rounding), and `--width-gamma`.

Note the leading space inside the quotes on `--crop` — argparse otherwise reads
a negative longitude as a flag. Same habit as ancestrydots and japanrail.

### No tilt

A rotation sweep on the network's own 41,043 vertices: the bounding box is
23.3 x 36.4 km north-up and bottoms out at 22.7 x 35.5 km at 10 deg — 5% of the
paper, against giving up an orientation every reader already has for New York.
North-up. `--rot` still exists if that ever looks wrong.

Projection is NAD83 / New York Long Island (EPSG:32118), the zone the city's own
data is published in. Over 40 km the scale error is far under a printed hairline.

### Sheet 24 x 36 (provisional)

The network is very close to 2:3 already, so a 24 x 36 sheet frames it almost
edge to edge: height binds, 1.04 km/inch, 1 px = 3.5 m at 300 ppi. That leaves
no outer margin — but it does leave big empty areas *inside* the frame, all of
them water or Nassau County: the Sound and Westchester top right, eastern Queens
right, Jamaica Bay and the Atlantic bottom right, the Lower Bay bottom left.
Title and legend should go in there rather than in a margin band. Not yet built.

### What the first renders showed

- **1:1 texture is good, which settles the standing worry.** The open question
  was whether the parallel-line rendering cleans up at print resolution. It does
  — a 3.2 x 2.4 in crop of Downtown Brooklyn resolves every stripe in the
  Atlantic Av bundle, and the Midtown crop resolves the E/F splay at 53 St.
- **Full sheet is 4.3 MB.** Nowhere near the 100 MB cap, so there is room for
  400 ppi or a bigger sheet if the type wants it.
- **Both themes work.** Light is more legible at whole-sheet scale (same finding
  as japanrail); dark makes the route colours sing and matches the web map's
  identity. Both are built by default.
- Land/water separation is doing a lot of work here because there is no street
  grid on the sheet — the shoreline is the only thing telling a reader which
  borough they are looking at.

### Second review (Anita) — the line placement was still wonky

Six things, and four of them turned out to be two causes.

**1. The side flip (was: "the E still has 2 discontinuities here").** A real
bug, and the worst one. A stripe is offset to the right of its run's coordinate
direction, so which physical side of the corridor a route sits on depends on
which way round the run is stored. Runs were oriented by their lowest-sorted
member's stored direction-0 order — stable, but not consistent, because the
member doing the anchoring differs from run to run. Two runs that continue into
each other could therefore both point at the shared node, and the same positive
offset then put the route on OPPOSITE sides either side of it. The ribbon jumps
clean across the corridor, which reads as a break. 6 of 566 through-path joints
were flipped: both of the E's at 8 Av/53 St, plus one each on the G and the M.

`orient_runs` fixes it as a 2-colouring: two runs whose tangents at a node point
apart are a continuation, and one must arrive while the other leaves. Tangents
pointing the same way are two branches leaving a switch together and get no
constraint — without that test a fork would demand its two branches disagree.
7 constraints are left unsatisfiable (odd cycles); they are small.

**2. The out-and-back wobble** (was: "the A/C curve inward as the E leaves and
back outward as the B/D join", and most of "weirdly circuitous turns"). Same
cause: every bundle was centred on its own track, so a route joining or leaving
shifted everyone. `solve_centres` gives each run one free unknown, its bundle
centre, and asks that a route shared by two runs keep the same offset across the
node — weighted by stripe width, with a weak spring back to zero to stop a
bundle drifting off its own track. Routes are stacked in a fixed order, so a
route joining or leaving at the END of that order is a pure translation, which
the free centre absorbs exactly; 8 Av now runs dead straight through both
changes. Successive over-relaxation, 400 sweeps, about a millisecond.

Measured at reg=0.06: boundary disagreement p90 18.1 m -> 7.2 m, boundaries over
20 m 101 -> 27, mean bundle drift 6.5 m. Lower reg keeps going but drift grows
fast (at 0.002 the mean is 51 m and the max 255 m).

**3. Ribbons tying knots** (the rest of "circuitous", and "something weird near
ACE Canal St"). Not the offsets — the geometry. A stripe offset by `o` onto the
inside of a curve of radius R has radius R - o, and at R < o it turns inside
out. At about a kilometre to the inch a 60-mil stripe is 62 m wide on the
GROUND, so the four-track bundles are 100-200 m across while the Canal St
reverse curve, the South Ferry loop and the Coney Island approaches all turn
inside 100 m. 78 of 1,169 ribbons folded.

`smooth_nodes` gives the shared centreline a minimum radius by Laplacian
smoothing, with each node's displacement capped at a fraction of the widest
bundle through it — so the cap does the discriminating, and a thin branch line
keeps its geography while the Lexington trunk gets its corners opened out to
something its own width can turn through. Junctions and dead ends are pinned so
runs still meet where they used to. 80 passes at cap 0.75 takes folds to 1-3,
moving a node 13 m on average and 77 m at worst (0.07 in on the sheet). It is
also what makes an elbow read as one curve rather than a polyline of chords.

**4. Endpoints that swing apart at an angled joint.** An offset is perpendicular
to the line it offsets, so two runs meeting at an angle put their endpoints
2*offset*sin(half the turn) apart. `joint_normals` mitres the joint — both ends
use the bisector, so both land on the same point. Small effect next to the side
flip, but free.

**5. The Manhattan Bridge (built).** B/D use the south tracks from Grand St and
N/Q the north tracks from Canal St; they share no station at either end and
cross the river about 15 m apart, which the graph cannot see because it keys on
identical vertices. `WELDS` replaces the recipient's vertices with the donor's
inside a box, and the fan then treats all four as one corridor. One entry so
far. The mechanism is general if other cases turn up.

**6. Still open: the F cutting in early where the A/C leave it.** Better than it
was — the solve keeps the F's offset constant right up to the switch instead of
starting to move a run early — but "stay tangent until it turns off" is really a
statement about the ramp's shape, not its position, and the ramp is still a
smoothstep in offset rather than something curvature-matched. Worth another look
against a crop.

### The web map got all of this except the geometry fixes

`riders/nycriders/index.html` carries the orientation fix, the bundle-centre
solve and the Manhattan Bridge weld. It does NOT get `smooth_nodes` or
`joint_normals`: those bake into coordinates, and the web map offsets in screen
space through `line-offset`. Folds are much less of a problem there anyway —
offset in pixels grows as 2^(z/2) while a curve's radius in pixels grows as 2^z,
so zooming in makes them go away rather than worse.

The solve pays for itself twice over: with most boundaries now agreeing, the
handoff ramps mostly vanish, and the feature count went 5,615 -> 3,169 while
`renderStaticSegments` stayed at 12-21 ms.

### The nudge editor (built)

Automatic placement got most of the way and then stopped being worth pushing, so
there is now a hand-edit stage, the way ancestrydots hand-places its insets.

    python nudge/prepare.py            # export, ~25 s
    python nudge/serve.py              # http://127.0.0.1:8799/nudge/
    python render.py --theme both      # picks up nudges.json automatically
    python render.py --no-nudges       # ...or does not, to compare

**The design decision that made it small.** The browser does NOT rebuild the
geometry pipeline. Everything that depends on the route data and not on where a
node sits — the shared-track graph, `orient_runs`, the fan, `solve_centres` —
runs once in `prepare.py` and lands in `build/nudge_data.json` as plain numbers:
node positions in projected metres, adjacency, and per-run stripe offsets and
widths. The page re-does only the cheap half per frame — offset a polyline, ramp
its ends, mitre the joints, about 150 lines. Two implementations of the hard
part would drift and the editor would stop predicting the sheet, which is the
only thing it is for. 12,912 nodes, 564 runs, 1,193 stripes, 0.9 MB.

Exported positions are POST-smoothing, so a nudge is a correction to the FINAL
position and `render.py` applies it after `smooth_nodes` too. No mental
correction, and the smoother cannot partly undo an edit.

**The brush measures distance along the track, not through the air.** In
Manhattan unrelated lines pass within metres of each other constantly, and a
Euclidean falloff would drag the 7 Av line while you were aiming at 8 Av.
Dijkstra over the graph, smoothstep falloff, radius on a slider.

**Densifying was needed to make it usable at all.** GTFS shapes are dense round
curves and bare on the straights — one run is 646 m of track with two vertices,
both of them stations. A 200 m brush found 6 nodes to move. `densify()` now
splits any segment over 60 m, which takes that to 28. It is deterministic in the
two endpoints, so a segment shared by two routes subdivides identically in both
and the graph still sees it as shared; it runs after the welds so a welded copy
densifies like its donor. Nodes 8,385 -> 12,912; `--smooth-passes` went 80 -> 160
to keep the same physical smoothing length at the finer spacing.

**Edits are keyed by node key** — the quantised lon/lat of the original GTFS
vertex — so they survive re-running everything upstream. They do not survive the
underlying shape changing, and `apply_nudges` reports how many keys no longer
match rather than dropping them silently.

`serve.py` exists only because a static page cannot write to disk, and
downloading a file and moving it into place after every session is exactly the
friction that stops a tool getting used. It writes via a temp file and
`replace()`, so a half-written save cannot replace a good one.

Left deliberately out: no reordering of routes within a bundle, and no per-
corridor override of the bundle centre. Both are offset problems rather than
geometry problems, and neither has come up yet.

### Third review (Anita) — the editable model was wrong

Two objections, and they are the same objection. Moving a node slid all four of
B/D/F/M in parallel, so a bundle could be carried around but an individual
line's shape could not be fixed. And the F was still tearing at a junction.

The cause was that the *authored* thing was a corridor centreline plus a scalar
offset per route. A node belongs to the corridor, not to a route — hence the
parallel drag. And continuity between the pieces had to be COMPUTED, by handoff
offsets, mitred joints and tapered ramps, so every remaining tear was a case
that computation did not cover. Three rounds of chasing those was enough
evidence that it would keep producing new ones.

`chains.py` inverts it. The automatic placement still runs exactly as before,
but it is now a *generator of starting positions* rather than the final say: it
gets baked down into per-route control points, and everything after works on
those.

- **A control point belongs to one route.** Moving it moves that route and
  nothing else. Keyed `"<route>|<node key>"`.
- **Continuity is structural, not computed.** A route's line is one polyline
  through its own points, so it cannot tear. A node on a run boundary takes the
  mean of what the runs meeting there each want, so the two sides start from the
  same point instead of being talked into it afterwards. Handoffs, ramps and the
  taper constants are gone from the poster path — the splay at a junction now
  falls out of the control points either side of it having different offsets.
- **Where a route splits**, its branches share the junction control point and are
  independent everywhere else: a Y with a corner, not a tear.
- **The shape is derived, not stored.** The drawn curve is a centripetal
  Catmull-Rom through the control points, not the control polygon, so there is no
  GTFS micro-wiggle being carried around: move a point and the curve re-derives
  over its neighbours. Centripetal rather than uniform because uniform
  Catmull-Rom loops wherever two control points sit much closer together than
  their neighbours, which is what a station pair inside a long block looks like.
- Control points are every run boundary (so a span never straddles a width
  change, and every station is grabbable) plus whatever the shape needs, by RDP
  at 9 m with a 260 m maximum gap. 60 route chains, 4,736 control points.

The editor follows: drag one point, one route moves. `s` solos a route so a
dense trunk stops being a lucky dip; the spread slider walks along that route's
own chain and can never reach a neighbour on the same corridor.

**Cost of the change.** Chains are baked for one set of width parameters, so
changing `--max-width` or the hour regenerates them. Edits are keyed by route
and original GTFS vertex, and `chains.build` is told to keep any point that has
an edit, so a re-bake does not orphan them — but it does move the ground under
them slightly. Settle the width curve before doing much hand work.

### Fourth review (Anita) — widths, and adding/removing points by hand

**Width doubled.** Still linear in riders / busiest segment, still 10:1 from
thinnest to thickest, but 6->12 and 60->120 mil: the outer branches were
disappearing at whole-sheet scale. `--width-gamma` is still there if the curve
shape ever needs bending, and is still a lie about the data if used.

The parameters that decide what the sheet looks like now live in `defaults.py`,
because `render.py` and `nudge/prepare.py` both need them and they MUST agree —
the editor bakes control points from these numbers and the sheet draws from
those control points, so a stripe that is one width in the editor and another on
the sheet is an editor that lies. Duplicated argparse defaults made that a live
possibility; now there is one copy.

The cost of the doubling is real: a 120-mil stripe is 125 m wide on the ground
at a kilometre to the inch, so the four-track bundles are 300-400 m across and
the tight curves have less room than ever. That is what the hand tool is for.

**Right-click adds and deletes control points.** The trick that made this small:
a chain now carries EVERY vertex the route passes through, and `ctrl` says which
of them currently shape the curve. So adding a point is promoting a vertex that
was already there — it lands exactly on the line and carries a stable GTFS key —
and deleting one is demoting it. Both are the automatic thinning, run by hand,
which is why they survive a re-bake: `chains.build` takes `forced` and `dropped`
and applies them after its own thinning.

- Right-click a line: adds a point on whichever route is drawn ON TOP there, hit
  tested back-to-front against the actual strokes, as a click on a pixel would
  be. It promotes the nearest vertex that is not already a control point —
  skipping the taken ones matters, because vertices sit every 60 m and the
  closest one to a click is often an existing point just too far away to have
  been picked as a hit, and answering "already a point here" would read as the
  tool being broken.
- Right-click a point: deletes it. Chain ends and the junctions where a route
  splits are refused, since both would take the line apart.
- Right-*drag* still pans; a right press that moves under 4 px is a click.
- Deleting a point you added is recorded as un-adding it, not as a deletion, so
  the edit file does not accumulate no-op drops.

`nudges.json` is now `{"move": {...}, "add": [...], "drop": [...]}`. A bare
key -> [dx, dy] object is still read as moves only, which is what the first
editor wrote. Export is 1.2 MB (25,591 vertices, 4,767 of them control points).

**Editor-only styling, NOT how the sheet prints.** Lines draw at 50% opacity so
you can see what is underneath the one being dragged, which is most of the job
where four routes overlap; `LINE_ALPHA` in nudge/index.html, and the sheet draws
them solid. Control points are filled in their route's own colour rather than
white, so a point says which line it belongs to without hovering it, and the
ring carries the edit state instead — dark by default, red for moved, green for
added.

### Chains are strokes now, not segments between branch nodes

The F at 6 Av read as a southbound and an eastbound line converging head-on
rather than trains running through — a cusp where there should be a junction.

Cause: chains were split at every node where a route's own degree was not 2, so
all three legs of a branch ENDED at the shared point, each with its own free
spline end and therefore its own tangent. Three curves meeting a point from
three independent directions is a cusp, not a Y.

`_continue` now carries a chain THROUGH a branch node along the straightest
unused edge — the one a train would take if it were running through — so the two
legs that line up are one chain with tangent continuity, and only the leg that
genuinely turns off starts a new one. Plain through-nodes carry on at any angle,
as before; at a branch the turn must be under 90 degrees to be taken as the
continuation. Seeds run termini first so the long strokes get built before
anything claims their edges.

Chains 60 -> 40, and the count of nodes where two chains of one route end with
none passing through went to **0** — so this was not one bad junction, it was
every branch on the map.

### Protecting the hand edits

Code changes do not touch `nudges.json` — but it got deleted twice during this
work by `rm -f nudge/nudges.json` while clearing test edits, without anyone
checking what was in it first. Hand-placing is slow and the ways to lose it are
all cheap, so:

- **`serve.py` keeps the version it replaces** as `nudges.backup.json`, unless
  the content is unchanged (a no-op save should not burn the backup). One level
  of file-side undo, on top of the editor's own undo stack.
- **The editor warns on unload** when there are unsaved edits. There is no
  autosave; Save is the only thing that writes.
- **`nudges.json` is never deleted to "clean up".** Check its contents first;
  the test edits are always identifiable.

There is a second, quieter way to lose work that is not deletion. A move is a
DELTA from where the automatic pass put the point, so changing the widths, the
smoothing or the chain building shifts what it is relative to: the edit survives
but no longer means what it meant. The editor now records the base position with
each move, and `ribbons.edit_report` — printed by both render.py and prepare.py
— says how many edits are lost, and how many now sit on geometry that has moved
more than 4 m since they were placed. Edits saved before this exist say so
rather than being silently unchecked.

### Open

- **Title block, legend and the MTA credit line.** The credit is required, not
  optional — see the per-map status note above.
- **The ~10 ribbons that still fold.** Down from 78, and the cap on how far
  smoothing may move a node is what stops the rest; more passes does not help.
  They are the first thing to try the nudge tool on.
- **Width curve.** Currently linear in riders/max, min 6 mil and max 60 mil,
  which is the web map's own 15:1 ratio. The outer branches nearly vanish at
  whole-sheet scale. A gamma below 1 (`--width-gamma`) would lift them; whether
  that is a lie about the data is the call to make.
- **Staten Island** draws with no lines at all — the SIR has no O-D data. Either
  crop it out, or say so in the caption.
- **Which hour.** `--hour` renders any single hour; the whole day is the default.
  A 5 pm sheet would look quite different from an all-day one.
- **A proof**, as ever.
