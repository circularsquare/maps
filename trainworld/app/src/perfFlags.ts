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
//
// `?perfTry=...` turns on candidate fixes, measured in notes/T-045.md before anyone adopts them:
//   labelLayers  each station label and capacity tag on its own compositor layer
//                (will-change: transform), so moving it does not repaint text
//   hoverCursor  hover writes the cursor style only when it changes
//   idle         (adopted, now default; see perfOff idle)
//   noDouble     the frame loop skips its own draw when MapLibre has a frame queued (it draws
//                the overlay itself in that frame)

const q = new URLSearchParams(location.search);
const off = new Set((q.get("perfOff") ?? "").split(",").filter(Boolean));
const tries = new Set((q.get("perfTry") ?? "").split(",").filter(Boolean));

export const perfOff = (part: "labels" | "shadow" | "hover" | "clock" | "overlay" | "markers" | "idle") => off.has(part);
export const perfTry = (fix: "labelLayers" | "hoverCursor" | "idle" | "noDouble") => tries.has(fix);
