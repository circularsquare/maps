// Measurement switches, for A/B runs of a check and for timing a part. Nothing sets them in play;
// without the parameters every check is false and the game runs its normal path. notes/T-045.md.
//
// `?perfOff=a,b` turns the named parts off. A name that is not listed here does nothing, but the demand
// cache's key holds the whole string, so any made-up name (`nocache`) makes the cache miss.
//
// Parts skipped, to time what is left:
//   labels   station name DOM labels (not placed, box hidden)
//   shadow   the invisible symbol layer of station names
//   hover    hover picking on mousemove
//   clock    the top bar's day and time (renders once, never again)
//   overlay  the overlay's WebGL draw (the clock still ticks)
//   markers  capacity marker DOM tags
//   marks    the crossing, junction, ramp and portal marks
//   moving   nothing waits while the camera moves: placers, crowds, station life, live counts every frame
//   stillThrottle  redraw every frame while the clock runs, not only when something visible moves
//   roads    no road tiles go to the track model, so no road crossings are counted or priced
//
// Demand solve, each against the normal path (the check named in notes/T-045.md or the task's note):
//   clusters the workers solve the whole network as one, every population every time
//   periods  a cluster the workers solve re-solves every period, even those an edit did not touch
//   split    the market's zones count as one, so every network it reaches is one cluster
//   reach    every city's demand solves on its whole cluster, not the stations it reaches
//   chunks   the regional market solves every zone pair, none sampled
//   noise    a cluster whose run times and trains an hour moved within noise is solved again, not kept
//   estimate a worker's first figures wait for every one of its periods' free flow, none estimated
//   share    every market stays with its cities' worker
//   fans     demand has no ferry fans
//   alone    stations no line stops at are in demand's network
//   firm     an edit's solve is not run to its end when a newer network arrives
//   rough    no rough solve of the zone in view while the packs open
//   selected the zone the city view shows waits its turn in the queues, and its figures wait for its cluster's final
//   viewRound  the zone in view gets no figures from each round of a cluster solved ahead
//   lengths0 the city view's trip lengths are asked of the shown round, not of round 0 ahead of it
//
// Loading and memory:
//   close    packs no longer needed never close
//   cap      the demand pool keeps its solving workers whatever their memory
//   cacheAhead  a loading save's cached demand result waits for its network, and the camera for its first state
//   cacheCard   a loaded game showing its cached demand result keeps its card and held clock until the fresh final (T-347)
//   splitAhead  the station bubbles' origins and transfers wait for the transfer pairs asked of the workers
//   zoneZoom zones load for viewing at any zoom, exact outlines from zoom 5, every outline redrawn on each change
//   focus    no focus pool: the zone in view is solved in the main pool with the rest of a load
//   focusAhead  the zone in view's cluster goes first even when the focus pool solves it, so the city view's zone waits
//   focusEnd the focus pool's workers stay until the load's final
//   cacheFocus  a loaded game showing its cached demand result ends the focus pool, so the zone in view's questions wait for the re-solve
//   focusAux the focus pool has an aux worker of its own
//   inPlace  the demand workers read the corridor table's columns through the binding's copies
//   `?focusWorkers=n` sets the focus pool's solving workers (default FOCUS_WORKERS).
//
// Drawing (each also a field on the renderer, to switch live):
//   strokeCache  track and line strokes drawn every frame, not kept in a texture (`renderer.strokeCache`)
//   strokeCull   every stroke chunk drawn, also those off the view (`renderer.strokeCull`)
//   strokeLod    the full stroke sets at every zoom, not simplified copies (`renderer.strokeLod`)
//   strokeGather the visible strokes drawn range by range, not gathered into one buffer (`renderer.strokeGather`)
//   bands    lines sharing track drawn over each other, not side by side, whatever Settings says (`renderer.lineBands`)
//   follow   every train on its own timetable, through the trains ahead (`renderer.follow`)
//   crossWait  trains drawn through each other at flat crossings and crossing junction moves (`renderer.guard.crossings`)
//   muteThin with a line picked, the other lines see-through at 55%, not thin and faded
//   bandGap  lines side by side touching, not a pixel apart
//   bandsSmooth  lines side by side move over short eases, not long S-curves, and order each ribbon on its own
//   bandsOrder   S-curves, but each ribbon ordered on its own
//   bandsPlace   a change's S-curve placed without fitting to a short run or moving to a merge just ahead
//   bandsHold    a ribbon whose tracks part for a while and rejoin is not held across the gap
//   bandsAlign   each line eases at a ribbon change where the ribbon's other lines do, not where its own track says

const q = new URLSearchParams(location.search);
const off = new Set((q.get("perfOff") ?? "").split(",").filter(Boolean));

export const perfOff = (part: "labels" | "shadow" | "hover" | "clock" | "overlay" | "markers" | "clusters" | "periods" | "split" | "reach" | "chunks" | "close" | "marks" | "fans" | "zoneZoom" | "moving" | "cap" | "rough" | "cacheAhead" | "cacheCard" | "estimate" | "splitAhead" | "strokeCache" | "strokeCull" | "strokeLod" | "strokeGather" | "bands" | "follow" | "crossWait" | "stillThrottle" | "share" | "noise" | "firm" | "alone" | "focus" | "muteThin" | "bandsSmooth" | "bandsOrder" | "bandsPlace" | "bandsHold" | "bandsAlign" | "bandGap" | "roads" | "selected" | "viewRound" | "lengths0" | "focusAhead" | "focusEnd" | "inPlace" | "focusAux" | "cacheFocus") => off.has(part);
/** how lines side by side are worked out (map/lineBands.ts `setBandMode`) */
export const bandTry = (): string => (perfOff("bandsSmooth") ? "" : (perfOff("bandsOrder") ? "smooth" : "smooth,order") + (perfOff("bandsPlace") ? "" : ",fit,snap") + (perfOff("bandsHold") ? "" : ",hold") + (perfOff("bandsAlign") ? "" : ",align"));
