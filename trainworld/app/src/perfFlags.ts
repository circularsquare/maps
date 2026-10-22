// Measurement-only switches (T-045). Nothing sets them in play; without the parameters every
// check is false and the game behaves as before. notes/T-045.md.
//
// `?perfOff=labels,shadow,hover,clock,overlay,markers` turns parts off to A/B their CPU:
//   labels   station name DOM labels (not placed, box hidden)
//   shadow   the invisible symbol layer of station names (T-032)
//   hover    hover picking on mousemove
//   clock    the top bar's day and time (renders once, never again)
//   overlay  the overlay's WebGL draw (the clock still ticks)
//   markers  capacity marker DOM tags
//   idle     the idle-frame fix (adopted as default in T-073): frames again with no trips
//   labelPan station labels placed individually every map frame (before T-074)
//   noDouble duplicate clock-loop draws while MapLibre has a frame queued (before T-075)
//   stagger  no everyday warm-up in demand worker 0, city packs open in every worker at once
//            (before T-173)
//   early    new or changed lines get no figures until the final round (before T-324)
//   bareStart  the viewed zone opens in every worker at startup, a new game's too, and no
//            building tool opens it ahead of a first station (before T-351)
//   ahead    a click during the startup solve waits for the shown round's answer (before T-327)
//   clusters the demand workers solve the whole network as one, every population every time
//            (before T-326)
//   periods  a cluster the demand workers solve re-solves every period, even those an edit
//            did not touch (before T-124)
//   spread   periods go to the demand workers by index alone, so 2 workers put both peaks on
//            one (before T-124)
//   split    the market's zones count as one, so every network it reaches is one cluster
//            (before T-409)
//   reach    every city's demand solves on its whole cluster, not the stations it reaches
//            (before T-756)
//   chunks   the regional market solves every zone pair, none dealt in chunks (before T-780)
//   close   packs no longer needed never close (before T-301)
//   marks    the crossing, junction, ramp and portal marks are not drawn (before T-629)
//   span     demand prices every stop as served all period, no service span (before T-842)
//   zoneZoom zones load for viewing at any zoom, exact outlines from zoom 5, every outline
//            fetched redrawn on each change (before T-871)
//
// `?perfTry=...` turns on candidate fixes, measured in notes/T-045.md before anyone adopts them:
//   labelLayers  each station label and capacity tag on its own compositor layer
//                (will-change: transform), so moving it does not repaint text
//   hoverCursor  hover writes the cursor style only when it changes
//   idle         (adopted, now default; see perfOff idle)
//   noDouble     adopted as default in T-075; use perfOff=noDouble for the old behavior

const q = new URLSearchParams(location.search);
const off = new Set((q.get("perfOff") ?? "").split(",").filter(Boolean));
const tries = new Set((q.get("perfTry") ?? "").split(",").filter(Boolean));

export const perfOff = (part: "labels" | "shadow" | "hover" | "clock" | "overlay" | "markers" | "idle" | "labelPan" | "noDouble" | "stagger" | "early" | "bareStart" | "ahead" | "clusters" | "periods" | "spread" | "split" | "reach" | "chunks" | "close" | "marks" | "span" | "zoneZoom") => off.has(part);
export const perfTry = (fix: "labelLayers" | "hoverCursor" | "idle" | "noDouble") => tries.has(fix);
