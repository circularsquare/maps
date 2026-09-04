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

### Remaining before a full poster draft

Script-able:

1. **Main legend — small multiples.** One thumbnail map per top-level group
   (14 of them) showing only that group's dots. Lives in the 3.87 in bottom
   band. Biggest remaining piece.
2. **Alaska / Hawaii / Puerto Rico.** Alaska shrunk with the scale change
   labelled; Hawaii and PR floated. Two wrinkles: Honolulu already exists as a
   5x city inset so a true-scale Hawaii float partly duplicates it, and PR at
   true scale is 1.2 x 0.4 in holding ~52,000 dots — saturated solid, so it
   probably wants magnifying like a city inset instead.
3. **Inset locator marks** on the main map, so the reader can place the 16
   insets. Without these the insets are unmoored.
4. **Scale bar.**

Needs a decision:

5. **Mexico.** Still untreated — currently renders as solid black land with no
   dots. Grey/hatch as out-of-scope, crop tighter, or leave.
6. **Labels.** The main map has *no* place names at all right now. Needs a call
   on how many and which. Plan was hand-placed in Aseprite; a candidate layer
   can be generated to drag around.
7. **Title, source line, data vintage** (ACS year + Canada 2021 census).

Layout risk worth checking early: the 16 insets are 93.1 sq in and the legend
band is 36 x 3.87 = 139 sq in, so the insets have to float in the map's dead
space (ocean, northern Canada, Mexico) rather than in the band. The framing
proof suggested there is room, but 16 boxes at 2.5-3 in each will be tight.
If it does not fit, the options are a taller sheet (36x26, 36x28 — neither a
standard frame size) or fewer insets.

### Known issues with this framing

- **Mexico has no data.** Baja and northern Mexico are a large visible chunk of
  the bottom of the frame and will render as empty land — on a dark map, solid
  black, which reads as "nobody lives there". Needs a deliberate treatment:
  grey/hatch as out-of-scope, or crop tighter.
- **The north is very empty.** Holding the Edmonton line costs a lot of near-
  blank paper across the prairies and Canadian Shield. Defensible (Edmonton is
  1.5M) but worth a look at the proof before committing.
