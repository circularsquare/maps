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
//   spread   periods go to the demand workers by index alone, so 2 workers put three of high
//            demand's periods on two (before T-124)
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
//   openAhead a loading save's zones and markets wait for its network to be worked out (before T-889)
//   batchOpen market packs go to the demand workers as each downloads, and a worker opens each
//            `open` alone (before T-889)
//   track    zones built track crosses are not opened bare for their figures (before T-1047)
//   moving   nothing waits while the camera moves: placers, crowds, station life, live counts
//            every frame (before T-1207)
//   restart  an idle demand worker well above its memory need is never restarted (before T-1205)
//   cap      the demand pool keeps its solving workers whatever their memory (before T-1205)
//   cacheAhead  a loading save's cached demand result waits for its network, and the camera for
//            its first state (before T-1539)
//   estimate a worker's first figures wait for every one of its periods' free flow, none estimated
//            from its first period's (before T-1538)
//   splitAhead the station bubbles' origins and transfers wait for the transfer pairs asked of the
//            workers, not the transfers the results carry (before T-1530)
//   parts    a free-flow round shown before it is solved is each worker's own periods, not one
//            period solved in parts of its origins over the workers (before T-1543)
//   refine   a zone opened again at another level re-solves its whole cluster, not itself alone
//            against the others' frozen loads (before T-1609)
//   window   an edit's local solve leaves the whole market frozen (T-1735)
//   noise    a cluster whose run times and trains an hour moved within noise of what it solved on
//            is solved again, not kept (before T-1862)
//   strokeCache the track and line strokes drawn every frame, not kept in a texture between
//            camera and network changes (before T-1638; `renderer.strokeCache` switches it live)
//   strokeCull every stroke chunk drawn, also those off the view (before T-1639; `renderer.strokeCull`)
//   strokeLod the full stroke sets drawn at every zoom, not simplified copies (before T-1639; `renderer.strokeLod`)
//   strokeGather the visible strokes drawn range by range, not gathered into one buffer (before T-1644; `renderer.strokeGather`)
//   bands    lines sharing track drawn over each other, not side by side (before T-1799; `renderer.lineBands`), whatever Settings says (T-1886)
//   follow   every train drawn on its own timetable, through the trains ahead on its track (before T-1965; `renderer.follow`)
//   crossWait trains drawn through each other at flat crossings and crossing junction moves (before
//            T-1966; `renderer.guard.crossings`)
//   daySums  the bubbles of a day other than a weekday are asked of the workers' live state each
//            round, and a cached result keeps a weekday's alone (before T-1904)
//   prebuild the bubbles of an end not on show are built only when it is picked, and a newer
//            result's are waited for with no stand-in (before T-1904)
//   focus    no focus pool: during a game load the zone in view is solved rough in the main pool
//            and only until the whole solve starts (before T-2045)
//   muteThin with a line picked, the other lines drawn see-through at 55%, not thin and faded (before T-2058)
//   bandsSmooth lines side by side move sideways over T-1848's short eases, not long S-curves, and
//            order each ribbon on its own (before T-2058)
//   bandsOrder  lines side by side in long S-curves, but each ribbon ordered on its own (T-2058's A alone)
//   bandsPlace  lines side by side: a change's S-curve placed as in T-2058 stage 2, not fitted to a short
//            run nor moved to the node of a merge just ahead (T-2059, T-2060)
//
// `?perfTry=...` turns on candidate fixes, measured in notes/T-045.md before anyone adopts them:
//   labelLayers  each station label and capacity tag on its own compositor layer
//                (will-change: transform), so moving it does not repaint text
//   hoverCursor  hover writes the cursor style only when it changes
//   idle         (adopted, now default; see perfOff idle)
//   noDouble     adopted as default in T-075; use perfOff=noDouble for the old behavior
//   focusFirst   the main pool's solving workers hold their solve and opening while the focus pool
//                solves the zone in view (T-2045: Tokyo settled 5% sooner, the load's final 37% later)

const q = new URLSearchParams(location.search);
const off = new Set((q.get("perfOff") ?? "").split(",").filter(Boolean));
const tries = new Set((q.get("perfTry") ?? "").split(",").filter(Boolean));

export const perfOff = (part: "labels" | "shadow" | "hover" | "clock" | "overlay" | "markers" | "idle" | "labelPan" | "noDouble" | "stagger" | "early" | "bareStart" | "ahead" | "clusters" | "periods" | "spread" | "split" | "reach" | "chunks" | "close" | "marks" | "span" | "zoneZoom" | "openAhead" | "batchOpen" | "track" | "moving" | "restart" | "cap" | "rough" | "cacheAhead" | "estimate" | "splitAhead" | "parts" | "strokeCache" | "strokeCull" | "strokeLod" | "strokeGather" | "bands" | "follow" | "crossWait" | "stillThrottle" | "share" | "refine" | "window" | "noise" | "daySums" | "prebuild" | "focus" | "muteThin" | "bandsSmooth" | "bandsOrder" | "bandsPlace") => off.has(part);
export const perfTry = (fix: "labelLayers" | "hoverCursor" | "idle" | "noDouble" | "focusFirst") => tries.has(fix);
/** T-2058: how lines side by side are worked out (map/lineBands.ts `setBandMode`) */
export const bandTry = (): string => (perfOff("bandsSmooth") ? "" : (perfOff("bandsOrder") ? "smooth" : "smooth,order") + (perfOff("bandsPlace") ? "" : ",fit,snap"));
