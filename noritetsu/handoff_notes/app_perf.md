# App performance (2026-10-08)

Anita: "when opening the app it's kinda freezy for the first 3 or so seconds as countries
load. the app also freezes when selecting a line and graying everything else out."

## How it was measured

`tools/perf_probe.js` (headless Chrome over CDP: CPU profile, long tasks, long animation
frames). The headless profile had 100 whole-line rides in jp, gb, ch and de, so every start
loads those four countries the way a user's start does. Swiftshader makes everything 2-4x
slower than a real machine, map drawing more so: read the numbers relative to each other.
"Before" is the app as it was this morning, served beside the new one and run on the same
data in the same Chrome.

Columns: longest main-thread task / total blocking time (the part of each task over 50 ms)
/ when the last long task ends after navigation, all in ms. For actions, `sync` is the
handler's own time, `open` is click to line panel open.

| scenario | before | after |
|---|---|---|
| world view, open (118 countries' tiles) | 7490 / 8401 / 9672 | 156 / 187 / 3505 |
| world: select WCML | 169 / 142 (sync 170) | 68 / 18 (sync 69) |
| world: clear selection | 65 / 15 (sync 66) | 0 / 0 (sync 5) |
| Europe z5.5, open (28 countries) | 688 / 1597 / 3327 | 148 / 177 / 2270 |
| Europe: select WCML | 221 / 185 (sync 221) | 92 / 42 (sync 93) |
| Europe: select a short ch line | 57 / 7 (sync 57) | 0 / 0 (sync 17) |
| Europe: clear | 0 / 0 (sync 41) | 0 / 0 (sync 4) |
| Japan z5.5, open | 417 / 826 / 2763 | 153 / 226 / 2033 |
| Japan: select Tokaido (JR Central) | 91 / 41 (sync 92) | 63 / 13 (sync 64) |
| Japan: load Russia | 1395 / 1531 | 261 / 789 |
| Japan: select longest ru line | 0 / 0 (sync 39) | 60 / 10 (sync 61) |
| click WCML track at z6.5 | 215 / 247 (open 276) | 66 / 16 (open 122) |
| click WCML track at z11 | 0 / 0 (open 64) | 0 / 0 (open 48) |
| click Tokaido at z7 / z11 | 67 / 17 (open 51) / 0 / 0 (open 95) | 0 / 0 (open 53) / 54 / 4 (open 108) |

The world-view "last ends" (3.5 s) is now the tail of short tasks adding countries a few at
a time; none of them is long. Selecting the longest Russian line is ~20 ms slower than
before because the operator list (OPS) is now built when first read rather than on every
country load; in idle time after a load it is built anyway (`warmIndexes`), and the probe
selected before idle came.

## What was costing what (before)

- World view open: one 8.7 s task. `addRegionTiles` for ~120 countries: MapLibre's
  `Style._validate` serializes the whole style on every `addLayer`/`addSource`/`setFilter`
  (6.5 s), plus `applyTrackColour` re-setting all 9 properties on every country each time
  one more was added (2.8 s).
- Each country arriving: 200-480 ms tasks, mostly `buildOps` -> `unionTotals` and
  `renderHome` -> `stats()` over everything loaded, repeated per country.
- Russia arriving: one 1.5 s task, 0.7 s of it `readAlong` expanding along.json.
- Selecting the WCML: ~230 ms, of which `passThrough` 150 and `nearRows` 55 (both scanning
  every loaded section for each junction a walk reaches).
- Clicking track at low zoom: `pickByChord` ~200 ms plus the same walks. After the changes
  the walks are gone from it and the click is 66 ms; pickByChord's own loop over every loaded
  line's sections is still there (not changed).

## Selecting a line: the random slow draw (2026-10-08, second pass)

Anita, after the first pass: "selecting a line is still a little laggy in some cases ... roughly
25% of the time it takes like ~600ms to draw ... it doesnt seem to be the same lines ... feels
random."

Measured with `tools/perf_probe.js selects`: 12 lines (3 each in jp, gb, de, us, fixed seed),
selected in turn, alternating a real mouse click on the track and a list click (`showLine` with
`bring`, which may move the camera), at z6.5 / 8 / 10 / 12, each after a jump somewhere. Per run:
click -> the first frame drawn with the selection sources fully loaded after their last setData
("drawn"), the click task's own time, the worker's answer to the setData, fetches, long tasks
and GC (CDP trace). Two settings: the default (1400x900, 0.1-2.5 s between jump and click) and a
hard one (1920x1080 at DPR 1.5, click 0-0.6 s after the jump, so tiles are still arriving, as
when you pan and click straight away). Only runs that opened a line count (a map click that
landed before the track was drawn opens nothing).

Click -> drawn, ms (median / p75 / p90 / max):

| setting | before | after |
|---|---|---|
| default, 48 selects | 64 / 106 / 175 / 245 | 36 / 56 / 108 / 181 |
| hard, ~30 selects | 73 / 195 / 257 / 629 | 46 / 88 / 179 / 257 |
| hard, every click incl. clears | 80 / 213 / 364 / 1020 | 45 / 84 / 158 / 257 |

**Cause 1, most of the tail: MapLibre's one worker.** MapLibre 4 in Chrome parses every vector
tile and every GeoJSON source in ONE web worker, first come first served. The selection is a
GeoJSON source (`sel`, `selst`), so its setData, and then each of its tiles, queued behind every
track and basemap tile in flight. Right after a pan or zoom, or a list click that moved the
camera, the worker was 0.2-1 s behind: setData -> worker's answer was 174-274 ms on slow
selects and 1003 ms on one clear, against 1-10 ms when the map was settled. That is why it
looked random: it depends on what the map was loading at that moment, not on the line. Fix:
two workers, the second for the selection alone (`selectionWorker` after `new
maplibregl.Map`): `Dispatcher.getActor` round-robins over all but the last worker, and
`GeoJSONSource._updateWorkerData` moves `sel`/`selst` onto the last before their first message.
Tiles and the other sources (stations, ridden, closed) stay together on the first worker
exactly as before. After: worker answer max 19 ms (default) / 106 ms (hard, only when the main
thread itself was busy). Survives a theme switch (checked: both sources on worker 1 after
light and back, tiles all loaded). `setWorkerCount` must run before `setRTLTextPlugin`, which
starts the workers. These are MapLibre 4.7 internals: recheck on an upgrade (v5 renames them);
if they are missing the patch does nothing and the app works as before.

**Cause 2: the operator list rebuilt inside the click.** When a country had just arrived
(panning to Germany at z8 brings in its neighbours) and the idle `warmIndexes` had not run yet,
opening a line built the whole operator list, `opsNow` -> `buildOps` -> `unionTotals`, 150-200 ms,
only so `opLinks` could ask whether each operator has a page. A listed line's operators always
do (buildOps takes every key of every listed line), so `opLinks` now builds the list only for
an unlisted line. That run: 261 ms sync -> 92 ms.

Not the cause: GC (main-thread pauses 0-45 ms inside a run, no pattern with the slow ones);
the geometry fetches (3-40 ms on localhost); symbol placement.

What is left in the tail (150-260 ms, on lines with many junction ends, mostly de): the junction
walks fetching neighbours' geometry in waves, each wave a fetch, `GEO_REDRAW`'s 30 ms batching,
a render and a setData (de: 3-4 setData spaced 35-50 ms). The line itself is drawn at the first
one; the later ones add the faint rings and track past junction ends. And the idle
`warmIndexes` step `opsNow` is one ~150 ms task after each burst of countries arriving: a click
landing inside it waits for it. Neither was changed.

Crediting unchanged: `tools/line_100_probe.js` on ch, lu, jp, us gives identical output for the
old and new page in the same profile.

## The selected line before the dimming (2026-10-09, third pass)

Anita: "when selecting a line it seems like maybe we hide everything else and then change the
draw style of the selected line. could we prioritize redrawing the selected line so that feels
even more responsive?"

**What it was.** Muting (`setMuted`) is a paint swap and showed on the very next frame. The
selection is GeoJSON: setData goes to the selection worker, comes back, then each of its tiles
goes and comes back again, and before any of that the line's geometry file may have to be
fetched. So for 1-5 frames the map showed everything dimmed with nothing selected. Clearing
was the mirror image: undimmed at once, with the old highlight still drawn for 1-2 frames.

**Measured with `tools/perf_probe.js frames`**: 12 lines (3 each in jp, gb, de, us, seed 11),
each selected from nothing (list click or a real mouse click on its track, after the map
settled or 500 ms after a jump), then a switch straight to another line of the same country,
then a clear. The probe stamps each setData's features with a number and, on every map frame
(`render`), records whether the track was drawn muted (the evaluated opacity of
`track-mute-<cc>`), the newest selection data in the loaded tiles, and whether the selection
layers were hidden. "Bad" frames: muted with none of the new line drawn (for a clear: unmuted
with the old line still drawn). Two runs each, old and new page, same profile (100 rides).

| | before | after |
|---|---|---|
| selects with a bad frame | 18 of 22 (1-5 frames each) | 0 of 23 |
| select: click to line drawn, ms (median / p90 / max) | 32 / 93 / 369 | 34 / 110 / 188 |
| select: dim relative to line | dim 1-5 frames first | same frame in 20, line 1 frame first in 3 |
| switch: click to new line drawn | 27 / 44 / 114 | 24 / 43 / 50 |
| clear: undim and highlight gone | 7 of 22 left the old line 1-2 frames; gone at 17-20 ms median, max 43 | both in frame 1, median 11-14, max 23 |
| select: click to everything drawn (median) | 50-65 | 50-65 |

The line itself lands about when it did (the p90 moves with how busy the tile worker was in the
panned runs); what changed is that the dimming now lands with it instead of frames ahead.

`perf_probe.js selects` (48 runs, its own measure of click to the last setData drawn):
before 37 / 70 / 133 / 231 (median / p75 / p90 / max), after 33 / 65 / 119 / 180. No regression.
Crediting unchanged: `line_100_probe.js` on ch, lu, jp, us identical old and new.

**What changed (dist/index.html):**
- `wantMuted` / `selSet` / `selDrawn` / `selLinesDrawn`: the dimming waits until the selection's
  lines are drawn (`sourcedata` on `sel`, ignoring 'metadata', which 4.7 sends before the tiles
  reload) and goes on in that frame, never later than `MUTE_HOLD_MS` (250). One selection setData
  in flight at a time; a newer one waits and replaces any waiting (`SEL_NEXT`).
- Clearing hides the selection layers by constant opacity (`selHidden`; every `sel-*` layer has an
  explicit opacity now, so the swap reloads nothing) in the same frame as the undim; they show
  again when the next selection is drawn.
- `showLine` calls `paintSelection` before `render`; with the line's geometry in hand
  (`geomHave`, the resolved value of `geomFor`'s promise) the setData leaves in the click's own
  task, before the panel is laid out, and the worker works on it meanwhile.
- A newly selected line goes out alone first only when something would make it wait: a
  neighbour's geometry to fetch for the continuations (`fetchAfter`), or its own geometry just
  arrived and the walks past its ends are not cached (`throughCached`), in which case the walks
  wait until it is drawn (`selSettled`, cap 150 ms). Otherwise it is one setData, as before.
- `GEO_REDRAW` (panel and selection redrawn as neighbours' geometry lands) waits while a newly
  selected line's first drawing is in flight (`SEL_FLIGHT.fresh`, cap 300 ms).
- A line row's geometry is fetched on `pointerdown`, ~50-150 ms before the click (not measured:
  the probe clicks rows by script).

**Not done, and why.** A highlight from the vector tiles (a per-country layer with a filter or
feature-state on the line's ways) was not needed: the GeoJSON path now lands with the dimming.
It would also be costly: `setFilter` on a vector layer reloads every tile of the country on the
shared worker, and feature-state needs feature ids in the tiles (`promoteId` on the way id `w`
could do it without a rebuild) plus a data-driven paint property on the track layers, which
re-parses tiles. Ways are also shared by several lines and run past a line's ends.

**Left:** the de map click in the run (rbb054c430c at z10) still takes ~180-220 ms to its first
drawing and 0.5-1 s to everything, behind 100-250 ms long tasks: the walks past junction ends
(`throughEnds` / `contRows`) and `render` as each wave of neighbours' geometry arrives. Same in the
old page. Making those walks cheaper is a separate job.

## Browse draws the selection at full colour (2026-10-09)

Anita: "when in view all lines mode and we select a line, it still draws it like mostly faded if
we haven't ridden it ... this is all lines mode so we shouldn't be thinking about that and we
should draw in full color." In Browse (`MODE !== 'ridden'`): `drawLine` draws no `done` overlay
and rings in the line's own colour; `heldBack` (the dimming of `base` features, continuations
included) does nothing; the strip diagram (`drawStrip`) draws sections, dots and continuation
lanes at full colour. The tracker is unchanged. Applies to every path through drawLine and
heldBack (line, trip, operator) and survives a theme switch (checked in light after dark).

## Build-side proposal (not done here)

**Russia's along.json is out of proportion: 25.8 MB, 63,651 sections with 1,656,628 pairs
(26 a section), against de 2.4 MB / 142,599 pairs (5 a section), gb 0.8 MB, jp 0.9 MB.**
Parsing it costs 142 ms of main thread in headless Chrome every time Russia loads (lines.json
is 23 ms, foot.json 17), and the app keeps the parsed object in memory. The app no longer
expands it unless a Russian ride is credited, but the download and the parse remain. Likely
cause: tariff sections overlapping at their ends (the same thing `sameRails` works around in
the walks) making every pair along a corridor "alongside". Worth checking in along.py whether
pairs that can never reach the app's thresholds (ALONG_ENDS_SHARE 0.6, ALONG_SHARE 0.9 of
the section, summed over the sections ridden) could be dropped, or whether Russia's
twinned junction stations (HANDOFF thread on ru junctions) are the source. Measured payoff
if it came down to de's density: ~120 ms and ~22 MB less per Russia load.

No other format change is needed for what was found.

## Found, not fixed (outside this task)

**The rows past a junction end can differ between two loads of the same page** (old and new
app alike: 53-161 of de's 2,456 lines differ between two runs). With caches cleared and all
geometry in, the result is stable, so it is a timing race: `passesBy` (and `cutGraph`) read a
line's geometry with `secPts`, and when it has not arrived they treat it as absent without
marking the walk pending, and `throughEnds` caches that answer for the rest of the session
(it filters its rows with passesBy). Example: de r65cc969d80's junction eDE97186 offered the
way on via r424e737107 (0.737 km) in two loads and via r52a89b2203 (0.648 km) in a third. Fix
would be passesBy adding the line to `missing` when `secPts` is null, so the walk stays
pending until the geometry lands. It changes what rows show, so it was left for a session
on the walks.

## Routes in us/ca/au, and the mode switch (2026-10-09)

`routesOf(cc)` (the routes model, HANDOFF "Routes in the US, Canada and Australia") is worked
out once per OWN_GEN/DATA_GEN per country, on the first `listed()` of that country: us 33 ms,
ca and au a few ms, headless. It lands in the redraw after a country arrives (`dataArrived`),
beside unionTotals, which costs more. `setMode` now calls `applyTrackColour`, which sets a
colour expression on every country's track and reloads its tiles, as a theme switch does.

Same profile, old page and new served side by side: `perf_probe.js selects` click to drawn,
ms (median / p75 / p90 / max) before 47 / 74 / 130 / 267, after 45 / 72 / 167 / 285 (run to
run noise; the us lines picked differ, since the probe picks from the listed lines). `frames`:
no bad frames in either; select dim median 45 both, clear 13 / 15. One us switch after took
981 ms to everything behind two ~280 ms long tasks; the same switch (SMART to the Bellingham
Subdivision) measured alone: sync 12 / 4 ms, no long task, against 7 / 4 ms and one 56 ms task
on the old page.
