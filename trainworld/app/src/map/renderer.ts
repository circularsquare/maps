import { BOX_WIDTH_ZOOM, boxHalfPxAtBoxZoom, BRIDGE_HALF_PX, bridgeShape, connectorHalfPx, dotRing, spanningPairs, insideStation, PLATFORM_POINTS, platformPoints, SEL_GLOW_PX, SEL_OUTLINE_PX, SHAPE_GLSL, shapePoints, STATION_BOX_ZOOM, STATIONS_ZOOM, TRAINS_ZOOM, stationHaloRadius, stationRadii, stationReachPx, stationRing, stationShape, type PlatformShape, type Project, type StationShape, type StationStyle } from "./stationGeometry";
// WebGL2 network renderer: track and line strokes (one instanced quad per polyline segment, a
// casing under the colour, the theme's (T-039); blueprint solid, lighter and without casing, T-311), stations (platforms along the track, or
// dots zoomed out, T-189), the drawing tool's preview,
// and instanced trains positioned on the GPU from keyframes (shared phase tables, notes/T-040.md)
// and a time uniform. Per frame the CPU sets uniforms and issues a handful of draw calls, whatever
// the train count (notes/T-002.md).
//
// Time precision (T-016, SPEC 7): trip departures are relative to a render epoch at the start of
// the current game hour, so the float32 time uniform stays under 3600 s. The clock worker sends
// each hour's trips (and the next hour's, ahead); when the hour turns, only that buffer is
// re-uploaded. Phase tables change only on edits.

import { LEVEL_COLOURS, NET_INK, PLATFORM_BLUE, rgb, SPEED_STOPS } from "../game/palette";
import { shadeColours } from "../game/shades";
import { platformRoom, pulsesBetween, readBoardHour, QUEUE_STRIDE, type BoardHour } from "../game/boarding";
import { anchor, anchorPsi, EARTH_C, kAtY, localToMercMatrix, mul4, PSI_PER_UNIT } from "./geo";
import { GradeMarks } from "./gradeMarks";
import { F_BLUEPRINT, lineAt, type NetworkBuffers, type Strokes } from "./network";

const TEX_W = 2048;
const LINES_W = 1024;
/** The render epoch moves to the start of each game hour. */
export const EPOCH_S = 3600;
const TRACK_RGB = rgb("#77746f").map((v) => v / 255);
const BAD_RGB = rgb("#d23c3c").map((v) => v / 255);
/** Blueprint track (T-322, Anita: "light blue", "slightly wider", "like two parallel rails"): a
 * light blue middle as wide as a line, with a dark blue rail this share of a line width (at least
 * a pixel) along each edge. */
const BP_MID_RGB = rgb("#a9cdf3").map((v) => v / 255);
const BP_RAIL_RGB = rgb("#2f62b3").map((v) => v / 255);
const BP_RAIL_SHARE = 0.16;
/** T-331 (Anita on T-322: "maybe overshot just a little bit"): the whole blueprint track, rails
 * included, is this share of T-322's width; the rails stay at least a pixel. */
const BP_NARROW = 0.75;
/** Seats and crush load per 20 m car, for the gauge (map/demandLayers.ts `gauge`, T-083). */
const SEATS_PER_CAR = 44, CRUSH_PER_CAR = 160;
/** A train's width on screen by zoom, CSS px, pills and cars alike (T-199: 75% of T-107's 5/8/12,
 * about 1.4 lines wide from zoom 12); its true 3 m is wider only from about zoom 19. */
const TRAIN_PX: [number, number][] = [[10, 3], [11, 5.25], [12, 6], [15, 9]];
/** A line's width on screen by zoom, CSS px: 60% of T-084's from zoom 12 (T-191), and further out
 * thinning to 40% of T-191's 2.1 px at zoom 9 (T-205, Anita: "lower min line thickness like 40%").
 * Bare track, the casing and a selected line's extra width follow the same steps. */
const LINE_PX: [number, number][] = [[9, 0.85], [12, 3.9], [15, 6.6]];
/** The ghost of track being placed (T-1038, Anita: "a decent minimum width ... see the track in full
 * length"): never thinner than this, CSS px, with this much casing either side; a line is as wide
 * from about zoom 11. */
const GHOST_MIN_PX = 3, GHOST_CASING_PX = 1.5;
/** Double track's two tracks (T-658): their centres 4 m apart on the ground, each a line's width
 * of LANE_SHARE of that (2.4 m) once apart. Where 4 m is LANE_GROW_PX on screen a double track's
 * stroke starts growing to the pair's width, which it reaches at LANE_PART_PX, where the two tracks
 * start parting; they are fully apart at LANE_APART_PX (zoom 16.7 to 17.2 in New York). */
const LANE_M = 4, LANE_SHARE = 0.6, LANE_GROW_PX = 4, LANE_PART_PX = 7, LANE_APART_PX = 10;
/** Past LANE_APART_PX each track narrows by up to LANE_NARROW (T-713, Anita: "reduce the width of
 * the individual tracks by around 20%" so the riders waiting behind them show), reached at
 * LANE_NARROW_PX; trains keep their 3 m. */
const LANE_NARROW = 0.2, LANE_NARROW_PX = 16;

/** How far double track has opened into two tracks at `zoom`, by the size of 4 m on screen at the
 * view's anchor: the gap in CSS px, the growth and parting (0-1), and half the pair's drawn width
 * in CSS px once it has grown (0 before), for picking. */
export function laneState(zoom: number) {
  const gap = LANE_M * kAtY(0) * (512 * 2 ** zoom) / EARTH_C;
  const grow = Math.max(0, Math.min(1, (gap - LANE_GROW_PX) / (LANE_PART_PX - LANE_GROW_PX)));
  const t = Math.max(0, Math.min(1, (gap - LANE_PART_PX) / (LANE_APART_PX - LANE_PART_PX)));
  const narrow = 1 - LANE_NARROW * Math.max(0, Math.min(1, (gap - LANE_APART_PX) / (LANE_NARROW_PX - LANE_APART_PX)));
  return { gap, grow, t, narrow, share: LANE_SHARE * narrow, reach: grow > 0 ? 0.5 * (1 + LANE_SHARE * narrow) * gap : 0 };
}
/** Platform crowds (T-164, T-219): drawn from this zoom, where platforms are (T-204), as each side's
 * half of the box filling from its middle outwards in `CROWD_RGB`, red (BAD_RGB) once that side is
 * full (T-559). */
export const CROWD_ZOOM = STATION_BOX_ZOOM;
const CROWD_RGB = rgb("#77726d").map((v) => v / 255);
const PLATFORM_RGB = rgb(PLATFORM_BLUE).map((v) => v / 255);
/** The track drawn through a station box (T-236) is at most this share of the box's white, so the
 * crowd and the blue platform edges show beside it. */
const TRACK_IN_BOX = 0.3;
/** RGBA texels per selected shape in the outline's texture (SEL_FS): its frame, size and dot, then
 * its centre line (T-254). */
const SEL_STRIDE = 3 + Math.ceil(PLATFORM_POINTS / 2);
// A platform's centre line (T-254) in the shaders: PLATFORM_POINTS points packed two to a vec4,
// as flat varyings `v_q0...` or texels; generated so the point count is set in one place.
// With them the station program has 12 vec4 varyings packed, 15 unpacked: WebGL2 guarantees 15
// (some phones report no more), so no more points or varyings there without packing others (T-953).
const QV = Math.ceil(PLATFORM_POINTS / 2);
const qVars = (dir: "in" | "out") => Array.from({ length: QV }, (_, i) => `flat ${dir} vec4 v_q${i};`).join("\n");
const qPack = Array.from({ length: QV }, (_, i) => `v_q${i} = vec4(q[${2 * i}], ${2 * i + 1 < PLATFORM_POINTS ? `q[${2 * i + 1}]` : "vec2(0.0)"});`).join("\n  ");
const qZero = Array.from({ length: QV }, (_, i) => `v_q${i} = vec4(0.0);`).join(" ");
// filled element by element: PowerVR's compiler (Pixel 10) rejects an array constructor in a
// fragment shader, "no default precision defined for variable 'vec2[13]'" (T-952)
const qUnpack = (p: string) => `vec2 q[${PLATFORM_POINTS}]; ${Array.from({ length: PLATFORM_POINTS }, (_, j) => `q[${j}] = ${p}${j >> 1}.${j & 1 ? "zw" : "xy"};`).join(" ")}`;
/** At most this many selected shapes (stations and bridges) share one outline and glow (T-217). */
const SEL_MAX = Math.floor(2048 / SEL_STRIDE); // texels in a row: 2048, the narrowest texture WebGL2 guarantees
/** A selected train's ring (T-242): the train's own shape grown by this, CSS px, with a white band
 * between, so the train reads at zoom 11-13 where it is a few px. */
const SEL_RING_GAP_PX = 6;
/** Hovered (T-730): a line's or bare track's colour mixed this far toward white; a station's rim
 * likewise, `HOT_STATIONS` of a complex at most. */
const HOT_LINE_MIX = "0.3";
/** The Max speed colour as GLSL (T-843): `SPEED_STOPS` blended in order, so the stops cannot drift from the key. */
const SPEED_BLEND = (() => {
  const v3 = (hex: string) => `vec3(${rgb(hex).map((c) => (c / 255).toFixed(4)).join(", ")})`;
  const s = SPEED_STOPS;
  return [`v_color = ${v3(s[0][1])};`, ...s.slice(1).map(([kmh, hex], i) =>
    `v_color = mix(v_color, ${v3(hex)}, clamp((a_speed - ${s[i][0].toFixed(1)}) / ${(kmh - s[i][0]).toFixed(1)}, 0.0, 1.0));`)].join("\n    ");
})();
const HOT_RIM_MIX = "0.6";
const HOT_STATIONS = 16;
/** Stations as the complex rule's distances group them (game/complexView.ts `dotGroups`): zoomed out
 * each group's dots are joined by connectors, the shortest set that links them all (T-313). */
export interface StationGroups {
  members: string[][];
}

const STROKE_VS = `#version 300 es
precision highp float;
precision highp int;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform vec3 u_width;        // normal, extra when selected, each track's once apart (T-658)
uniform float u_casing;      // extra px for the casing pass, 0 for the colour pass
uniform vec3 u_casing_rgb;   // white, or dark in the dark theme (game/shades.ts)
uniform float u_cap;         // > 0: no wider than this (px), the track drawn through stations (T-236)
uniform int u_kind;          // 0 track, 1 lines, 2 preview
uniform int u_sel_line;
uniform int u_sel_edge;
uniform ivec2 u_hot;         // the hovered line and bare track edge, -1 none (T-730)
uniform int u_only_sel;
uniform int u_bp_part;       // T-695: 0 every stroke, 1 all but blueprint, 2 blueprint only
uniform vec4 u_lanes;        // T-658: 4 m in px, how far the tracks have parted (0-1), how far the
                             // stroke has grown to their width (0-1), a blueprint track's width apart
uniform int u_lane_mode;     // 0 one stroke, 1 the band under both tracks, 2 the right track, 3 the left
uniform float u_dim;        // line strokes other than the selected line's: share of full strength (T-233)
uniform int u_by_level;
uniform int u_by_speed;
uniform float u_px_per_unit;
uniform vec3 u_level_colours[7];
uniform vec3 u_track;
uniform vec3 u_bad;
uniform vec3 u_sel;
uniform vec2 u_bp;           // blueprint track (T-322): its middle's width and each rail's (px)
uniform vec3 u_bp_mid;       // its middle, light blue
uniform vec3 u_bp_rail;      // its two rails, dark blue
uniform vec3 u_bp_off_mid;   // the same switched off (T-333): cold grey rails round a nearly white
uniform vec3 u_bp_off_rail;  // middle in the light theme, dim in the dark one (game/shades.ts)
uniform vec3 u_bp_off_fade;  // what its other colours are mixed towards
uniform highp sampler2D u_line_colours;
uniform int u_full_pass;     // T-819: 1 the pass drawing a red band this much (px) wider than the
uniform float u_full_px;     // casing under the lines on track at capacity, nothing else
uniform vec3 u_full_rgb;
uniform int u_ghost;         // T-1038: 1 for the preview pass, whose placement strokes are the ghost
uniform vec3 u_ghost_rgb;
uniform vec2 u_ghost_min;    // the ghost's least width (px), and its casing's both sides together
in vec4 a_seg;
in uint a_colour;
in uint a_edge;
in float a_level;
in float a_flags;
in float a_speed;
in float a_dist;
in float a_wmul;            // width as a multiple of the line width (riders, T-084); 1 otherwise
in vec2 a_lane;             // T-658: how far apart its two tracks are at its start and end, 0-1
in float a_full;            // T-819: 1 on a line segment whose track is at capacity (traffic view)
out vec3 v_color;
out float v_alpha;
out vec4 v_cap;             // px: along from the segment's start, along to its end, across, half the width
flat out vec4 v_stripe;     // T-817: high-speed track's centre line, its colour and half width (px), 0 none
flat out float v_stripe_only; // 1: only that line is drawn (bare track between its parted tracks)
void main() {
  int flags = int(a_flags + 0.5);
  bool hsr = (flags & 256) != 0;
  v_stripe = vec4(0.0);
  v_stripe_only = 0.0;
  bool sel = (u_kind == 1 && int(a_colour) == u_sel_line) || (u_kind == 0 && int(a_edge) == u_sel_edge);
  bool blueprint = (flags & 1) != 0;
  bool placement = (flags & 32) != 0;
  // blueprint track is two dark blue rails either side of a light blue middle (T-322, Anita): the
  // rails are its casing pass, drawn for every segment before any middle, so the joints are clean;
  // the middle is a line wide, so a line planned along it fills it and the rails show either side
  bool bpTrack = blueprint && !placement && u_kind == 0;
  bool swOff = (flags & 64) != 0;
  if ((u_only_sel == 1 && !sel) || (u_casing > 0.0 && blueprint && !placement && !bpTrack) || (u_bp_part == 1 && blueprint) || (u_bp_part == 2 && !blueprint)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  // a line planned along blueprint fills the track's middle (T-331: narrower than a line now)
  bool bpLine = blueprint && !placement && u_kind == 1;
  // double track as its two tracks close in (T-658): first the stroke grows to the pair's width
  // (u_lanes.z), then the two tracks part inside it (u_lanes.y), so its outline never jumps
  // a turn-back crossover (T-658d) goes from one track to the other: drawn once, as a track
  bool turnback = (flags & 128) != 0;
  if (turnback && (u_lane_mode != 2 || u_casing > 0.0)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float lfm = turnback ? 1.0 : max(a_lane.x, a_lane.y), G = u_lanes.x, t = u_lanes.y;
  float lt = blueprint && !placement ? u_lanes.w : u_width.z;
  float core = bpTrack || bpLine ? u_bp.x + 2.0 * u_bp.y : u_width.x;
  float grow = u_lanes.z * max(0.0, lfm * (G + lt) - core * (bpTrack || bpLine ? 1.0 : a_wmul));
  float extra = sel ? u_width.y : 0.0;
  bool full = u_full_pass == 1;
  if (full && !(u_kind == 1 && a_full > 0.5)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float cas = full ? u_casing + u_full_px : u_casing;
  float w, lane_off = 0.0;
  if (u_lane_mode == 0) {
    w = (bpTrack ? u_bp.x + extra + (u_casing > 0.0 ? 2.0 * u_bp.y : 0.0)
      : bpLine ? u_bp.x + extra + cas
      : u_width.x * a_wmul + extra + cas) + grow;
  } else {
    // the gap opens from the middle while the pair's width stays: each track is half of what is
    // left; the band under both is the pair's width, or the traffic width behind them (Anita,
    // 2026-10-19)
    float gap = t * (G - lt);
    float wl = lfm > 0.0 ? 0.5 * (max(core, G + lt) - gap) : mix(core, lt, t);
    float half_ = 0.5 * (gap + wl);
    lane_off = half_;
    float band = max(core * (bpTrack || bpLine ? 1.0 : a_wmul), lfm * 2.0 * half_ + wl);
    bool casing = u_casing > 0.0;
    bool keep = u_lane_mode == 1
      ? (casing ? !bpTrack : (u_kind == 1 && !blueprint) || (hsr && u_kind == 0))
      : (casing ? bpTrack : true) && (u_lane_mode == 2 || lfm > 0.0);
    if (!keep) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    w = u_lane_mode == 1 ? band + extra + cas
      : bpTrack || bpLine ? (casing ? wl : max(0.0, wl - 2.0 * u_bp.y))
      : wl;
    if (u_cap > 0.0 && u_lane_mode > 1) w = min(w, 0.5 * u_cap);
  }
  if (u_cap > 0.0) w = min(w, u_cap);
  bool ghost = u_ghost == 1 && placement;
  if (ghost && u_lane_mode <= 1) w = max(w, u_ghost_min.x + (u_casing > 0.0 ? u_ghost_min.y : 0.0));
  // thinner than a pixel (zoomed far out, T-205): a pixel wide and fainter, rather than broken up
  float faint = clamp(w, 0.05, 1.0);
  w = max(w, 1.0);
  int id = gl_VertexID;
  float along = float(id & 1);
  float side = id < 2 ? -1.0 : 1.0;
  vec4 c0 = u_matrix * vec4(a_seg.xy, 0.0, 1.0);
  vec4 c1 = u_matrix * vec4(a_seg.zw, 0.0, 1.0);
  vec2 half_vp = u_viewport * 0.5;
  vec2 s0 = c0.xy / c0.w * half_vp;
  vec2 s1 = c1.xy / c1.w * half_vp;
  vec2 dir = s1 - s0;
  float len = length(dir);
  dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec4 c = along < 0.5 ? c0 : c1;
  vec2 off = nrm * side * w * 0.5 + dir * (along * 2.0 - 1.0) * w * 0.5;
  // trains keep right (T-319): a line's strokes run in its run 0's direction, so its run 0 has the
  // right-hand track (-nrm) and run 1 the left
  float lv = along < 0.5 ? a_lane.x : a_lane.y;
  if (turnback) lv = 2.0 * lv - 1.0;
  if (u_lane_mode > 1) off -= nrm * (u_lane_mode == 2 ? 1.0 : -1.0) * lv * lane_off;
  c.xy += off / half_vp * c.w;
  gl_Position = c;
  // round caps (T-673): the quad runs half a width past each end; the fragment shader rounds it
  float ext = (along * 2.0 - 1.0) * w * 0.5, ua = along * len + ext;
  v_cap = vec4(ua, len - ua, side * w * 0.5, w * 0.5);
  // Blueprint is solid but lighter (T-311): the colour mixed towards white, opaque, because
  // segments overlap at every joint and see-through blueprint would show a bead there. Lines
  // that do not run (faded) stay see-through.
  bool bp = blueprint && !placement;
  v_alpha = (!placement && !bp && (flags & 2) != 0 ? 0.55 : 1.0) * faint * (u_kind == 1 && !sel ? u_dim : 1.0);
  vec3 lc = u_level_colours[clamp(3 - int(round(a_level)), 0, 6)];
  bool bpMid = false;
  if (u_casing > 0.0) v_color = bpTrack ? (swOff ? u_bp_off_rail : u_bp_rail) : placement && !ghost ? u_sel : full ? u_full_rgb : u_casing_rgb;
  else if ((flags & 16) != 0) v_color = u_bad;
  else if ((u_kind == 2 && !placement) || u_by_level == 1) v_color = lc;
  else if (u_by_speed == 1) {
    ${SPEED_BLEND}
  }
  else if (placement) v_color = ghost ? u_ghost_rgb : vec3(1.0);
  else if (u_kind == 0 && bpTrack && !sel) { v_color = swOff ? u_bp_off_mid : u_bp_mid; bpMid = true; }
  else if (u_kind == 0) v_color = sel ? u_sel : u_track;
  else v_color = texelFetch(u_line_colours, ivec2(int(a_colour) % ${LINES_W}, int(a_colour) / ${LINES_W}), 0).rgb;
  // other blueprint colours (a planned line, a selected blueprint, by height or speed) lighter
  if (bp && !bpMid && u_casing <= 0.0) v_color = swOff ? mix(v_color, u_bp_off_fade, 0.8) : mix(v_color, vec3(1.0), 0.45);
  bool hot = (u_kind == 1 && int(a_colour) == u_hot.x) || (u_kind == 0 && int(a_edge) == u_hot.y);
  if (hot && u_casing <= 0.0) v_color = mix(v_color, vec3(1.0), ${HOT_LINE_MIX});
  // high-speed track (T-817): a thin centre line along the stroke, light on built track (dark in
  // the dark theme), the rail colour on blueprint; once its tracks part, the stroke's own colour
  // between them
  vec3 rail = bpTrack ? (swOff ? u_bp_off_rail : u_bp_rail) : placement && !ghost ? u_sel : u_casing_rgb;
  if (hsr && u_casing <= 0.0 && u_lane_mode < 2 && !full) {
    float hs = max(0.3, (u_lane_mode == 0 ? 0.18 * w : 0.08 * G));
    v_stripe = vec4(u_lane_mode == 0 ? rail : (bpTrack || placement ? rail : v_color), hs);
    v_stripe_only = u_lane_mode == 1 && u_kind == 0 ? 1.0 : 0.0;
  }
  // the band behind a line's two tracks, pale so they stand out on it
  if (u_lane_mode == 1 && u_casing <= 0.0) v_color = mix(v_color, u_casing_rgb, 0.6 * t);
}`;

// highp: v_cap is device px along the segment, past fp16's 65504 on a long straight zoomed in on a
// phone, where mediump is 16-bit (T-953)
const STROKE_FS = `#version 300 es
precision highp float;
in vec3 v_color;
in float v_alpha;
in vec4 v_cap;
flat in vec4 v_stripe;
flat in float v_stripe_only;
out vec4 o;
void main() {
  float past = -min(v_cap.x, v_cap.y);
  float a = past > 0.0 ? clamp(0.5 - (length(vec2(past, v_cap.z)) - v_cap.w), 0.0, 1.0) * v_alpha : v_alpha;
  vec3 c = v_color;
  if (v_stripe.w > 0.0) {
    float s = clamp(v_stripe.w + 0.5 - abs(v_cap.z), 0.0, 1.0);
    c = mix(c, v_stripe.rgb, s);
    if (v_stripe_only > 0.5) { c = v_stripe.rgb; a *= s; }
  }
  o = vec4(c * a, a);
}`;

// Stations (T-144, T-189, T-217): a platform is a band along its track at its true length, bent as
// the track bends, white with a dark rim; zoomed out a dot, one a station (T-313). The quad is laid
// out in the platform's frame at its middle (along the track there, across towards the side it
// bends to) and the fragment shader measures the distance from the band (SHAPE_GLSL). Stations
// drawn over each other make one white shape with one outline round it (T-217, Anita: "all the white
// rectangles overlapping seamlessly ... and then one continuous black outline surrounding the whole
// white shape"): every station's dark shape is drawn first (u_layer 0), then every station's white
// inside its rim (u_layer 1), so a rim inside another station is painted over; then the riders
// waiting on each platform (u_layer 2, T-219). Selected ones are drawn in a pass of their own, after
// the others (u_pass), over one outline and glow round all of them (SEL_FS). The same program draws
// the demand circles (u_halo 1).
const STATION_VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform vec2 u_radius;   // dot radius, ordinary and transfer (px)
uniform float u_line;    // a line's width (px): a station covers the lines through it
uniform float u_ring;    // its dark rim (px)
uniform float u_boxes;   // 1: platforms along the track; 0: dots (zoomed out)
uniform vec2 u_fixed;    // a box's half width (T-204, T-239): px at BOX_WIDTH_ZOOM, and local
                         // units per px there; the width on the ground, and the least on screen (T-553)
uniform highp int u_halo;
uniform highp int u_pass; // stations: 0 every one but the selected, 1 the selected only
uniform highp int u_layer; // 0 the dark shape, 1 the white inside the rim, 2 riders waiting (T-219),
                           // 3 the blue platform edges (T-236)
uniform float u_halo_radius;
uniform float u_pixel;
in vec3 a_st;
in vec2 a_axis;          // half the platform along its track at its middle, local units
in vec4 a_shape;         // the platform's middle from the node, half the turn along it (rad), half its true width (local units)
in vec2 a_sz;            // (x unused since T-239), daily riders' square-root scale
in vec2 a_halo;          // demand circle's offset, local units (a station complex's, T-146)
in vec2 a_sel;           // drawn selected (x; y unused since T-313)
in vec4 a_crowd;         // riders waiting (T-219): how full the platform's + and - side is (0-1),
                         // and whether that side is full (zw)
in vec2 a_dcrowd;        // zoomed out (T-228): the station's fullest platform side, 0-1, and 1 if
                         // it is full
uniform highp sampler2D u_stline; // RG32F, per station ${PLATFORM_POINTS} texels: its track's centre line from the node (T-254)
out vec2 v_p;            // in the platform's frame at its middle, px
flat out vec3 v_size;    // half length, half width, (0); a demand circle's radius
flat out float v_dot;    // the dot's radius, 0 for a platform
flat out float v_alpha;
flat out vec4 v_crowd;
flat out vec4 v_scr;     // a dot's frame on the screen (T-553): where its x and y axes point, as
                         // (x.x, y.x, x.y, y.y) with the screen's y up, so a pie can start at 12 o'clock
flat out float v_lane;   // 1 on double track: a platform each side of its two tracks close in (T-658)
uniform highp int u_hot_st[${HOT_STATIONS}]; // the hovered stations (T-730), u_hot_n of them
uniform highp int u_hot_n;
flat out float v_hot;
// the centre line in the frame, px (T-254), two points to a vec4
${qVars("out")}
void main() {
  v_hot = 0.0;
  for (int k = 0; k < ${HOT_STATIONS}; k++) if (k < u_hot_n && u_hot_st[k] == gl_InstanceID) v_hot = 1.0;
  v_dot = 0.0;
  v_scr = vec4(1.0, 0.0, 0.0, 1.0);
  v_crowd = a_crowd;
  int flags = int(a_st.z + 0.5);
  v_lane = (flags & 16) != 0 ? 1.0 : 0.0;
  v_alpha = (flags & 8) != 0 ? 0.3 : (flags & 2) != 0 ? 0.6 : 1.0; // 8: a blueprint switched off (T-333)
  vec2 half_vp = u_viewport * 0.5;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  if (u_halo == 1) {
    vec4 h = u_matrix * vec4(a_st.xy + a_halo, 0.0, 1.0);
    float r = u_halo_radius * a_sz.y;
    v_size = vec3(r, r, 0.0);
    v_p = corner * (r + 1.0);
    h.xy += v_p / half_vp * h.w;
    gl_Position = h;
    return;
  }
  bool boxes = u_boxes > 0.5;
  bool sel = a_sel.x > 0.5;
  bool noCrowd = !boxes || (flags & 2) != 0 || max(a_crowd.x, a_crowd.y) <= 0.0;
  // the blue platform edges (u_layer 3, T-236) are drawn on platforms only
  if ((u_pass == 1) != sel || (u_layer == 2 && noCrowd) || (u_layer == 3 && !boxes)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  // zoomed out a dot's pie is its platforms' crowd (T-228, T-553): the fullest side's share, and
  // whether it is full
  if (!boxes) v_crowd = vec4(a_dcrowd, 0.0, 0.0);
  vec2 mid = a_st.xy + a_shape.xy;
  vec4 c = u_matrix * vec4(mid, 0.0, 1.0);
  vec4 e = u_matrix * vec4(mid + a_axis, 0.0, 1.0);
  vec4 q = u_matrix * vec4(mid + vec2(-a_axis.y, a_axis.x), 0.0, 1.0);
  vec2 cs = c.xy / c.w * half_vp;
  vec2 axis = e.xy / e.w * half_vp - cs, nrm = q.xy / q.w * half_vp - cs;
  float len = length(axis);
  vec2 along = len > 1e-6 ? axis / len : vec2(1.0, 0.0);
  vec2 across = length(nrm) > 1e-6 ? normalize(nrm) : vec2(-along.y, along.x);
  v_scr = vec4(along.x, across.x, along.y, across.y);
  float axisLoc = length(a_axis);
  // a box is as wide on the ground as a plain line and its rim are at BOX_WIDTH_ZOOM less a
  // quarter, whatever lines are drawn wider (T-239), so its length and width scale together and
  // it keeps one shape from there in; further out it keeps that width on screen (T-553)
  float perLoc = axisLoc > 1e-9 ? len / axisLoc : 0.0;
  float hw = max(max(a_shape.w, u_fixed.x * u_fixed.y) * perLoc, boxes ? u_fixed.x : 0.0);
  // a dot covers a plain line through it as drawn now
  float cover = 0.3 * u_line + u_ring + u_pixel;
  float hl = max(len, hw);
  float k = len > 1e-6 && abs(a_shape.z) > 1e-3 ? a_shape.z / len : 0.0;
  // zoomed out every station is its own dot at its platform's middle (T-313)
  if (!boxes) v_dot = max((flags & 1) != 0 ? u_radius.y : u_radius.x, cover);
  float lo, hi, xl, xh;
  ${qZero}
  if (v_dot > 0.0) {
    v_size = vec3(hl, hw, 0.0);
    lo = -v_dot - 1.0;
    hi = v_dot + 1.0;
    xl = -v_dot - 1.0;
    xh = v_dot + 1.0;
  } else {
    // the track's own line along the platform (T-254, Anita: "ideally we actually fit the
    // track"), in the frame at the middle; the band is hw either side of it, the quad round it
    vec2 q[${PLATFORM_POINTS}];
    float total = 0.0;
    vec2 mn = vec2(1e20), mx = vec2(-1e20);
    for (int i = 0; i < ${PLATFORM_POINTS}; i++) {
      int t = gl_InstanceID * ${PLATFORM_POINTS} + i;
      vec2 P = texelFetch(u_stline, ivec2(t % ${TEX_W}, t / ${TEX_W}), 0).xy;
      vec4 h = u_matrix * vec4(a_st.xy + P, 0.0, 1.0);
      vec2 s = h.xy / h.w * half_vp - cs;
      q[i] = vec2(dot(s, along), dot(s, across));
      if (i > 0) total += length(q[i] - q[i - 1]);
      mn = min(mn, q[i]);
      mx = max(mx, q[i]);
    }
    ${qPack}
    hl = max(0.5 * total, hw);
    // never shorter than it is wide: the end stretches carry on
    float ext = hl - 0.5 * total;
    v_size = vec3(hl, hw, 0.0);
    lo = mn.y - hw - ext - 1.0;
    hi = mx.y + hw + ext + 1.0;
    xl = mn.x - hw - ext - 1.0;
    xh = mx.x + hw + ext + 1.0;
  }
  v_p = vec2(corner.x < 0.0 ? xl : xh, corner.y < 0.0 ? lo : hi);
  c.xy += (along * v_p.x + across * v_p.y) / half_vp * c.w;
  gl_Position = c;
}`;

// A link's bridge (T-217): a straight box between the two platforms' nearest points, as thin as
// BRIDGE_HALF_PX on the ground with the station rim along it, drawn with the stations' layers so it
// joins their white shape. Zoomed out the same program draws the connectors between dots (T-313).
const BRIDGE_VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform float u_ring;
uniform float u_bridge;  // the white's half width, local units
uniform highp int u_pass;
in vec4 a_ends;          // its two ends, local units
in vec2 a_bflag;         // drawn selected (both stations are), faded (either is a blueprint)
out vec2 v_p;
flat out vec3 v_size;
flat out float v_dot;
flat out float v_alpha;
flat out vec4 v_crowd;
flat out vec4 v_scr;
flat out float v_lane;
uniform highp int u_hot_st[${HOT_STATIONS}]; // the bridges between hovered stations (T-730)
uniform highp int u_hot_n;
flat out float v_hot;
${qVars("out")}
void main() {
  v_hot = 0.0;
  for (int k = 0; k < ${HOT_STATIONS}; k++) if (k < u_hot_n && u_hot_st[k] == gl_InstanceID) v_hot = 1.0;
  v_dot = 0.0;
  v_lane = 0.0;
  v_scr = vec4(1.0, 0.0, 0.0, 1.0);
  v_crowd = vec4(0.0);
  v_alpha = a_bflag.y > 0.5 ? 0.6 : 1.0;
  vec2 half_vp = u_viewport * 0.5;
  vec4 c0 = u_matrix * vec4(a_ends.xy, 0.0, 1.0), c1 = u_matrix * vec4(a_ends.zw, 0.0, 1.0);
  vec4 cx = u_matrix * vec4(a_ends.xy + vec2(u_bridge, 0.0), 0.0, 1.0);
  vec2 s0 = c0.xy / c0.w * half_vp, s1 = c1.xy / c1.w * half_vp;
  float len = length(s1 - s0);
  if ((u_pass == 1) != (a_bflag.x > 0.5) || len < 1e-3) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 u = (s1 - s0) / len, n = vec2(-u.y, u.x);
  float hw = length(cx.xy / cx.w * half_vp - s0) + u_ring, hl = 0.5 * len;
  v_size = vec3(hl, hw, 0.0);
  // a straight centre line (the stations' polyline frame, T-254)
  vec2 q[${PLATFORM_POINTS}];
  for (int i = 0; i < ${PLATFORM_POINTS}; i++) q[i] = vec2(-hl + 2.0 * hl * float(i) / ${(PLATFORM_POINTS - 1).toFixed(1)}, 0.0);
  ${qPack}
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  v_p = corner * vec2(hl + 1.0, hw + 1.0);
  vec4 c = u_matrix * vec4(0.5 * (a_ends.xy + a_ends.zw), 0.0, 1.0);
  c.xy += (u * v_p.x + n * v_p.y) / half_vp * c.w;
  gl_Position = c;
}`;

const STATION_FS = `#version 300 es
precision highp float;
uniform float u_ring;
uniform vec3 u_ink;
uniform vec3 u_bad;
uniform vec3 u_crowd_rgb;
uniform highp int u_halo;
uniform highp int u_layer;
uniform float u_hair;    // a CSS px in device px
uniform float u_tint;    // 1: dots show their crowds as pies (T-228, T-553)
uniform vec3 u_plat;     // the platforms' blue (T-236)
uniform float u_mask;    // 1: the white only marks the stencil (the track drawn through it, T-236)
uniform vec3 u_lanes;    // T-658: 4 m in px, how far double track's two tracks have parted (0-1), their narrowing close in (T-713)
in vec2 v_p;
flat in vec3 v_size;
flat in float v_dot;
flat in float v_alpha;
flat in vec4 v_crowd;
flat in vec4 v_scr;
flat in float v_lane;
flat in float v_hot;
${qVars("in")}
out vec4 o;
${SHAPE_GLSL}
// white between a platform's rim and its blue edge (T-236), px, from the white's half width W
float platGap(float W) { return 0.15 * W; }
// a blue platform edge's width, px: a share of the white, then 0.4 m once double track has parted
float edgeW(float W, float tl) { return mix(max(u_hair, 0.14 * W), max(u_hair, 0.1 * u_lanes.x), tl); }
void main() {
  if (u_halo == 1) {
    if (v_size.x < 0.5 || v_alpha < 0.9) discard;
    float d = length(v_p);
    float a = clamp(v_size.x - d + 0.5, 0.0, 1.0) * (d > v_size.x - u_ring ? 0.48 : 0.22);
    vec3 blue = vec3(0.118, 0.235, 0.902);
    o = vec4(blue * a, a);
    return;
  }
  // along and across the platform's own centre line (T-254); a dot's own frame
  ${qUnpack("v_q")}
  vec2 f = v_dot > 0.0 ? v_p : polyFrame(v_p, q);
  float d = v_dot > 0.0 ? length(v_p) - v_dot : boxSdf(f, v_size.xy);
  float outer = clamp(0.5 - d, 0.0, 1.0), inner = clamp(0.5 - d - u_ring, 0.0, 1.0);
  if (u_layer == 0) {
    // the whole shape dark, painted white inside the rim next; a faded (blueprint) one only its
    // rim, so its see-through white stays as light as before
    float a = (v_alpha < 0.99 ? outer - inner : outer) * v_alpha;
    o = vec4(mix(u_ink, vec3(1.0), ${HOT_RIM_MIX} * v_hot) * a, a);
  } else if (u_layer == 1) {
    // marking where the track is drawn over the station (T-236): the white's inside, no colour
    if (u_mask > 0.5) {
      if (inner < 0.5 || v_dot > 0.0) discard;
      o = vec4(0.0);
      return;
    }
    float a = inner * v_alpha;
    // a dot (zoomed out) shows its platforms' crowd as a pie (T-228, T-553): the white fills
    // clockwise from 12 o'clock with the crowd grey by the fullest side's share, red once full
    vec3 col = vec3(1.0);
    float share = clamp(v_crowd.x, 0.0, 1.0);    if (v_dot > 0.0 && u_tint > 0.5 && share > 0.0) {
      vec2 s = vec2(dot(v_p, v_scr.xy), dot(v_p, v_scr.zw));
      float turn = atan(s.x, s.y);
      if (turn < 0.0) turn += 6.2831853;
      float r = length(s);
      float cover = share > 0.995 ? 1.0 : clamp(share * 6.2831853 * r - turn * r + 0.5, 0.0, 1.0) * clamp(turn * r + 0.5, 0.0, 1.0);
      col = mix(col, v_crowd.y > 0.5 ? u_bad : u_crowd_rgb, cover);
    }
    o = vec4(col * a, a);
  } else if (u_layer == 3) {
    // the platform's two long edges as thin blue lines just inside its rim (T-236, Anita: "show
    // the actual platforms as like little blue lines"), drawn over the track and the crowd, so
    // platforms drawn over each other each show their own; they come in as the box grows wide
    // enough to hold them beside the track
    vec2 p = f;
    float W = v_size.y - u_ring;
    // close in on double track (T-658) they move in beside its two tracks, 0.4 m wide: a platform
    // each side
    float tl = u_lanes.y * v_lane;
    float gap = platGap(W), bw = edgeW(W, tl);
    float trk = (0.5 + 0.5 * ${LANE_SHARE.toFixed(1)} * u_lanes.z) * u_lanes.x;  // the pair's outer edge
    float off = mix(W - gap - 0.5 * bw, trk + u_hair + 0.5 * bw, tl);
    float a = clamp(0.5 * bw + 0.5 - abs(abs(p.y) - off), 0.0, 1.0)
      * clamp(v_size.x - u_ring - gap - abs(p.x) + 0.5, 0.0, 1.0)
      * clamp((W / u_hair - 3.0) / 2.0, 0.0, 1.0) * v_alpha;
    o = vec4(u_plat * a, a);
  } else {
    // riders waiting (T-219): each side's half of the platform fills from its middle outwards, as
    // far as its share of the room to stand there, like a train's gauge; red once full (T-559)
    // white kept between the fill and the rim (a pixel and a half, or as far as the blue platform
    // edge, T-236), so the outline stays crisp
    vec2 p = f;
    float keep = max(1.5, platGap(v_size.y - u_ring));
    inner = clamp(0.5 - keep - d - u_ring, 0.0, 1.0);
    float reach = v_size.x - u_ring - keep;
    // close in on double track (T-658) each side's riders stand on their own platform, from beside
    // its blue edge out to the rim: the + side's trains travel along the axis, keeping right, so
    // their track and platform are on the + side
    float tl = u_lanes.y * v_lane;
    float yin = tl * ((0.5 + 0.5 * ${LANE_SHARE.toFixed(1)} * u_lanes.z) * u_lanes.x + 2.0 * u_hair + edgeW(v_size.y - u_ring, tl) + 1.5);
    inner = mix(inner, clamp(0.5 - 1.5 - d - u_ring, 0.0, 1.0), tl);
    float ap = inner * clamp(p.y - yin + 0.5, 0.0, 1.0) * clamp(v_crowd.x * reach - abs(p.x) + 0.5, 0.0, 1.0);
    float am = inner * clamp(0.5 - p.y - yin, 0.0, 1.0) * clamp(v_crowd.y * reach - abs(p.x) + 0.5, 0.0, 1.0);
    o = vec4((v_crowd.z > 0.5 ? u_bad : u_crowd_rgb) * ap + (v_crowd.w > 0.5 ? u_bad : u_crowd_rgb) * am, ap + am);
  }
}`;

// The selected stations' outline and glow (T-189), round their union (T-217): one quad over all of
// them, each pixel's distance the nearest shape's, so platforms and bridges drawn as one shape get
// one outline. The shapes are the CPU twins (stationGeometry.ts) in CSS px, 3 texels each.
const SEL_VS = `#version 300 es
precision highp float;
uniform vec4 u_bbox;     // CSS px, y down: left, top, right, bottom
uniform vec2 u_css;      // the canvas, CSS px
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1), gl_VertexID < 2 ? 0.0 : 1.0);
  vec2 s = mix(u_bbox.xy, u_bbox.zw, corner);
  gl_Position = vec4(s.x / u_css.x * 2.0 - 1.0, 1.0 - s.y / u_css.y * 2.0, 0.0, 1.0);
}`;

const SEL_FS = `#version 300 es
precision highp float;
uniform highp sampler2D u_shapes; // per shape SEL_STRIDE texels: middle, along; across, half length, half width; dot radius; its centre line in its frame (T-254)
uniform int u_count;
uniform float u_dpr;
uniform float u_height;  // the canvas, device px
uniform vec3 u_ink;
uniform float u_outline; // CSS px
uniform float u_glow;
out vec4 o;
${SHAPE_GLSL}
void main() {
  vec2 p = vec2(gl_FragCoord.x, u_height - gl_FragCoord.y) / u_dpr;
  float d = 1e9;
  for (int i = 0; i < u_count; i++) {
    int k = i * ${SEL_STRIDE};
    vec4 a = texelFetch(u_shapes, ivec2(k, 0), 0);
    vec4 b = texelFetch(u_shapes, ivec2(k + 1, 0), 0);
    float dotR = texelFetch(u_shapes, ivec2(k + 2, 0), 0).x;
    vec2 r = p - a.xy;
    if (dotR > 0.0) { d = min(d, length(r) - dotR); continue; }
    // nowhere nearer than its middle's distance less its half length and width: skip the line
    if (length(r) - b.z - b.w > d) continue;
    // its own centre line (T-254)
    ${Array.from({ length: QV }, (_, i) => `vec4 t${i} = texelFetch(u_shapes, ivec2(k + ${3 + i}, 0), 0);`).join("\n    ")}
    ${qUnpack("t")}
    d = min(d, boxSdf(polyFrame(vec2(dot(r, a.zw), dot(r, b.xy)), q), b.zw));
  }
  float line = clamp(u_outline + 0.5 - d, 0.0, 1.0);
  float t = clamp((d - u_outline) / u_glow, 0.0, 1.0);
  float s = max(line, 0.35 * (1.0 - t) * (1.0 - t));
  o = vec4(u_ink * s, s);
}`;

// Shared by the draw shader and the transform-feedback probe, so the probe checks the very
// arithmetic that positions the trains on screen.
const POS_GLSL = `
uniform float u_time;        // game seconds since the render epoch
uniform highp sampler2D u_phases;  // RGBA32F: t, s, v, a
uniform highp sampler2D u_meta;    // RGBA32F per profile: first phase, count, trip s, length m
uniform highp sampler2D u_samples; // RG32F: run 0 of each line, x/y local units
uniform highp sampler2D u_lines;   // RGBA32F per line: first sample, count, length m, step m
in vec2 a_trip;              // profile, departure - epoch (s)
in float a_end;              // drawn until - epoch (s): standing at the terminal for its next departure (T-206)

ivec2 tc(int i, int w) { return ivec2(i % w, i / w); }

/** The point at distance d (m) along run 0 of a line (L from the line table). */
vec2 lineAt(vec4 L, float d) {
  float first = L.x, count = L.y, len = L.z, step = L.w;
  float dd = clamp(d, 0.0, len);
  float i0 = clamp(floor(dd / step), 0.0, max(count - 2.0, 0.0));
  float seg = max(min(step, len - i0 * step), 1e-3);
  float f = clamp((dd - i0 * step) / seg, 0.0, 1.0);
  int si = int(first + i0);
  vec2 p0 = texelFetch(u_samples, tc(si, ${TEX_W}), 0).xy;
  vec2 p1 = texelFetch(u_samples, tc(si + 1, ${TEX_W}), 0).xy;
  return mix(p0, p1, f);
}

/** The train's distance along run 0 of its line now; false if it is not on the track. */
bool trainDist(out vec4 L, out float d, out int line) {
  int prof = int(a_trip.x + 0.5);
  vec4 M = texelFetch(u_meta, tc(prof, ${TEX_W}), 0);
  float tau = u_time - a_trip.y;
  if (M.y < 0.5 || tau < 0.0 || (tau > M.z && u_time > a_end)) return false;
  int lo = int(M.x), hi = int(M.x + M.y) - 1;
  for (int k = 0; k < 18; k++) {
    if (lo >= hi) break;
    int mid = (lo + hi + 1) / 2;
    if (texelFetch(u_phases, tc(mid, ${TEX_W}), 0).x <= tau) lo = mid; else hi = mid - 1;
  }
  vec4 p = texelFetch(u_phases, tc(lo, ${TEX_W}), 0);
  float tt = max(tau - p.x, 0.0);
  if (p.w < 0.0) tt = min(tt, p.z / -p.w);
  float s = p.y + p.z * tt + 0.5 * p.w * tt * tt;
  line = prof / 6;
  int run = (prof / 3) % 2;
  d = run == 0 ? s : M.w - s;
  L = texelFetch(u_lines, tc(line, ${LINES_W}), 0);
  return true;
}`;

const TRAIN_VS = `#version 300 es
precision highp float;
precision highp int;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform float u_px_per_unit;
uniform vec2 u_min_px;      // minimum length, width in px
uniform bool u_detail;
uniform float u_dpr;
uniform highp sampler2D u_cars;
uniform highp sampler2D u_line_colours;
uniform highp sampler2D u_fill;  // RG32F per sample: how full run 0's and run 1's trains are (T-084)
uniform highp sampler2D u_loads;   // R32F: riders aboard each trip of the hour per stretch (T-164)
uniform highp sampler2D u_stretch; // R32F per sample: its stop-to-stop stretch, run 0's numbering; -1 none
uniform int u_sel_line;            // stop editing (T-233): the edited line's trains stay as they are,
uniform float u_dim;               // the others' share of full strength
uniform vec2 u_sel_trip;           // the selected train (T-242): profile, departure - epoch; x < 0 none
uniform highp int u_pass;          // 0 every train but the selected, 1 the selected's halo, 2 the selected
uniform float u_psi0;              // the anchor's isometric latitude (T-307): k = cosh(u_psi0 - y * PSI_PER_UNIT)
uniform highp sampler2D u_lane_tex; // R32F per sample: how far apart its track's two tracks are, 0-1 (T-658)
uniform vec2 u_lane;               // px from the centre line to a track's once parted, how far parted (0-1)
out float v_alpha;
in vec2 a_load;               // this trip's first load in u_loads or -1, seconds to its last stop
${POS_GLSL}
out vec2 v_uv;
flat out float v_k;
out vec2 v_px;
out vec3 v_color;
out vec3 v_empty;
out float v_fill;
out float v_front;
out vec2 v_consist;
out float v_detail;
out float v_cab;

/** How far apart the two tracks are at distance d along run 0 (laneAt in network.ts). */
float laneAt(vec4 L, float d) {
  float dd = clamp(d, 0.0, L.z);
  float i0 = clamp(floor(dd / L.w), 0.0, max(L.y - 2.0, 0.0));
  float f = clamp((dd - i0 * L.w) / max(min(L.w, L.z - i0 * L.w), 1e-3), 0.0, 1.0);
  int si = int(L.x + i0);
  return mix(texelFetch(u_lane_tex, tc(si, ${TEX_W}), 0).x, texelFetch(u_lane_tex, tc(si + (L.y > 1.5 ? 1 : 0), ${TEX_W}), 0).x, f);
}

void main() {
  vec4 L;
  float d;
  int line;
  // the selected train is drawn last, over the others, with its halo under it (T-242)
  bool chosen = u_sel_trip.x >= 0.0 && abs(a_trip.x - u_sel_trip.x) < 0.5 && abs(a_trip.y - u_sel_trip.y) < 0.01;
  if ((u_pass == 0) == chosen || !trainDist(L, d, line)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float cars = max(1.0, texelFetch(u_cars, tc(line, ${LINES_W}), 0).x);
  int car = u_detail ? gl_VertexID / 6 : 0;
  if (float(car) >= cars) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float totalM = min(cars * 20.0, L.z);
  float pitchM = totalM / cars;
  float centre = clamp(d, totalM * 0.5, max(totalM * 0.5, L.z - totalM * 0.5));
  float offsetM = u_detail ? (float(car) + 0.5) * pitchM - totalM * 0.5 : 0.0;
  float carD = u_detail ? centre + offsetM : d;
  vec2 p = lineAt(L, carD);
  // local units per metre of ground where the train is (T-307: mercator's scale, 1 / cos(lat))
  float kM = cosh(u_psi0 - p.y * ${PSI_PER_UNIT.toExponential(9)});
  // Both representations share the same width: splitting into cars must not shrink the train.
  vec2 px = max(u_min_px, vec2(cars * 20.0, 3.0) * kM * u_px_per_unit);
  if (u_detail) px.x = max(2.0 * u_dpr, pitchM * kM * u_px_per_unit - max(1.5 * u_dpr, kM * u_px_per_unit));
  // on its own track as double track parts (T-658): narrowing to its true 3 m, so two trains
  // passing never touch, and keeping right of its direction of travel (T-319)
  float lane = u_lane.y * laneAt(L, carD);
  px.y = mix(px.y, 3.0 * kM * u_px_per_unit, lane);
  // Heading along the chord between the train's two ends (as its bogies sit on the track), not
  // the sample segment under its centre, which turns in steps on curves (T-047).
  float h = max(0.5 * px.x / u_px_per_unit / kM, 1.0);
  vec2 back = lineAt(L, carD - h), front = lineAt(L, carD + h);
  vec4 c = u_matrix * vec4(p, 0.0, 1.0);
  vec4 cb = u_matrix * vec4(back, 0.0, 1.0);
  vec4 cf = u_matrix * vec4(front, 0.0, 1.0);
  vec2 half_vp = u_viewport * 0.5;
  vec2 bs = cb.xy / cb.w * half_vp, fs = cf.xy / cf.w * half_vp;
  vec2 sdir = fs - bs;
  float sl = length(sdir);
  sdir = sl > 1e-6 ? sdir / sl : vec2(1.0, 0.0);
  vec2 nrm = vec2(-sdir.y, sdir.x);
  // A train as one pill bends with its track (T-189): along the circle through its two ends and
  // its middle, as a station's platform does (SHAPE_GLSL); a car close up is short enough to stay
  // straight. k: curvature, 1/px, towards +nrm; ends: where the ends lie across the frame.
  float k = 0.0, ends = 0.0;
  if (!u_detail && sl > 1e-3) {
    float hc = 0.5 * sl, s = dot(c.xy / c.w * half_vp - 0.5 * (bs + fs), nrm);
    k = clamp(-2.0 * s / (hc * hc + s * s), -2.0 / px.x, 2.0 / px.x);
    ends = -s;
  }
  // Two independent triangles per car, so adjacent cars never share a strip across a bend.
  int tri = gl_VertexID % 6;
  int id = u_detail ? (tri == 0 ? 0 : tri == 1 ? 1 : tri == 2 ? 2 : tri == 3 ? 2 : tri == 4 ? 1 : 3) : gl_VertexID;
  vec2 corner = vec2(float(id & 1) * 2.0 - 1.0, id < 2 ? -1.0 : 1.0);
  // 1 px larger than the train all round, so the soft edge is never cut by the quad (T-047).
  vec2 halfq = px * 0.5 + 1.0;
  // The halo (T-242): room for the outline, the ring and the glow round the whole consist; bent no
  // tighter than its reach, so the bend's centre never falls inside it.
  if (u_pass == 1) {
    float reach = ${(SEL_RING_GAP_PX + SEL_GLOW_PX + 4).toFixed(1)} * u_dpr;
    halfq += reach;
    k = clamp(k, -0.8 / halfq.y, 0.8 / halfq.y);
  }
  vec2 q = vec2(corner.x * halfq.x * (1.0 + halfq.y * abs(k)), corner.y < 0.0 ? min(0.0, ends) - halfq.y : max(0.0, ends) + halfq.y);
  float shift = ((int(a_trip.x + 0.5) / 3) % 2 == 0 ? -1.0 : 1.0) * lane * u_lane.x;
  c.xy += (sdir * q.x + nrm * (q.y + shift)) / half_vp * c.w;
  gl_Position = c;
  v_uv = q;
  v_k = k;
  v_px = px;
  vec3 lc = texelFetch(u_line_colours, ivec2(line % ${LINES_W}, line / ${LINES_W}), 0).rgb;
  v_color = lc * 0.48;
  v_alpha = line == u_sel_line ? 1.0 : u_dim;
  // How full it is (T-084): the stretch it runs, from the sample ahead of it on its way.
  int run = (int(a_trip.x + 0.5) / 3) % 2;
  float i = floor(clamp(d, 0.0, L.z) / L.w) + (run == 0 ? 1.0 : 0.0);
  int si = int(L.x + clamp(i, 0.0, L.y - 1.0));
  vec2 f = texelFetch(u_fill, tc(si, ${TEX_W}), 0).xy;
  v_fill = run == 0 ? f.x : f.y;
  // Its own load from the boarding account where there is one (T-164); the fill above then only
  // says whether train loads are shown for this line. Empty once at its last stop.
  float st = texelFetch(u_stretch, tc(si, ${TEX_W}), 0).x;
  if (v_fill >= 0.0 && a_load.x >= 0.0 && st >= 0.0) {
    float aboard = u_time - a_trip.y >= a_load.y ? 0.0 : texelFetch(u_loads, tc(int(a_load.x + st + 0.5), ${TEX_W}), 0).x;
    float seats = ${SEATS_PER_CAR.toFixed(1)} * cars, crush = ${CRUSH_PER_CAR.toFixed(1)} * cars;
    v_fill = aboard <= seats ? 0.5 * aboard / seats : min(1.0, 0.5 + 0.5 * (aboard - seats) / (crush - seats));
    // packed to crush (the account never goes past it: whoever does not fit waits) draws as full
    // with the over-full red edge, so a full train stands out from one nearly full (T-207)
    if (aboard >= crush - 0.5) v_fill = 2.0;
  } else {
    // no account for this train (its line's stops just changed, or demand is not in yet): no
    // gauge, rather than the forecast's average train in its place (T-230, Anita); striped as
    // "not worked out yet" where its line shows loads (T-324), never solid like a full train
    v_fill = v_fill >= 0.0 || v_fill < -1.5 ? -2.0 : -1.0;
  }
  v_empty = mix(lc, vec3(1.0), 0.18);
  v_front = run == 0 ? 1.0 : -1.0;
  v_consist = vec2(u_detail ? totalM * kM * u_px_per_unit : px.x, offsetM * kM * u_px_per_unit);
  v_detail = u_detail ? 1.0 : 0.0;
  v_cab = u_detail && (run == 0 ? float(car) == cars - 1.0 : car == 0) ? 1.0 : 0.0;
}`;

// A capsule at city scale, or separate rounded cars close up (T-103), with a white edge.
// With riders known (T-084) the core fills from the back across the whole consist: dark up to
// how full the train is, a slight tint of the line
// colour beyond; half way means every seat taken, full means packed to crush; an over-full train
// is full and has a red edge.
const TRAIN_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
flat in float v_k;
in vec2 v_px;
in vec3 v_color;
in vec3 v_empty;
in float v_fill;
in float v_front;
in vec2 v_consist;
in float v_detail;
in float v_cab;
in float v_alpha;
uniform float u_dpr;
uniform vec3 u_bad;
uniform highp int u_pass;
uniform vec3 u_ink;
uniform vec3 u_band;
out vec4 o;
${SHAPE_GLSL}
void main() {
  // along the train's bend and across it (T-189)
  vec2 p = arc(v_uv, v_k);
  float r = v_detail > 0.5 ? min(2.0 * u_dpr, v_px.y * 0.22) : v_px.y * 0.5;
  vec2 q = abs(p) - (v_px * 0.5 - r);
  float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
  if (u_pass == 1) {
    // the selected train's halo (T-242), from the train out: an outline in the map ink as a
    // selected station has (T-189), a band in the casing's colour, a ring of the same shape in the
    // ink, and a soft glow beyond it; dark, then white, in the light theme (T-543)
    float ol = ${SEL_OUTLINE_PX.toFixed(1)} * u_dpr, gap = ${SEL_RING_GAP_PX.toFixed(1)} * u_dpr, gw = ${SEL_GLOW_PX.toFixed(1)} * u_dpr;
    float edge = clamp(ol + 0.5 - d, 0.0, 1.0);
    float ring = clamp(1.25 * u_dpr + 0.5 - abs(d - gap), 0.0, 1.0);
    float t = clamp((d - gap - 1.25 * u_dpr) / gw, 0.0, 1.0);
    float glow = d > gap ? 0.35 * (1.0 - t) * (1.0 - t) : 0.0;
    float white = d > ol && d < gap ? 0.9 : 0.0;
    float s = max(max(edge, ring), glow);
    o = vec4(u_ink, 1.0) * s + vec4(u_band, 1.0) * white * (1.0 - s);
    return;
  }
  float a = clamp(0.5 - d, 0.0, 1.0) * v_alpha;
  vec3 core = v_color;
  vec3 rim = vec3(1.0);
  if (v_fill >= 0.0) {
    float back = (p.x + v_consist.y) * v_front + v_consist.x * 0.5;
    core = mix(v_empty, v_color, clamp(min(v_fill, 1.0) * v_consist.x - back + 0.5, 0.0, 1.0));
    if (v_fill > 1.5) rim = u_bad;
  } else if (v_fill < -1.5) {
    // riders not worked out yet (T-324): diagonal stripes of the line colour and white
    float w = max(2.0, 2.0 * u_dpr);
    core = mod(p.x + p.y, 2.0 * w) < w ? v_empty : vec3(1.0);
  }
  // A front cab reads as a train once each car is large enough to see it.
  if (v_detail > 0.5 && v_px.x > 18.0 * u_dpr) {
    bool cab = v_cab > 0.5 && p.x * v_front > v_px.x * 0.30 && abs(p.y) < v_px.y * 0.30;
    if (cab) core *= 0.55;
  }
  // a full train's red edge is wider than the white one, so it reads at a glance (T-207; T-218,
  // Anita: "not super noticeable", 50% thicker than T-207's 2.6 px)
  vec3 col = mix(core, rim, clamp(d + (v_fill > 1.5 ? 3.9 : 1.7), 0.0, 1.0));
  o = vec4(col * a, a);
}`;

const PROBE_VS = `#version 300 es
precision highp float;
precision highp int;
${POS_GLSL}
out vec2 v_pos;
void main() {
  vec4 L;
  float d;
  int line;
  v_pos = trainDist(L, d, line) ? lineAt(L, d) : vec2(1e9);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;

const PROBE_FS = `#version 300 es
precision mediump float;
out vec4 o;
void main() { o = vec4(0.0); }`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string, feedback?: string[]): WebGLProgram {
  const prog = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  if (feedback) gl.transformFeedbackVaryings(prog, feedback, gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(prog));
  return prog;
}

function uniforms(gl: WebGL2RenderingContext, prog: WebGLProgram, names: string[]) {
  const u: Record<string, WebGLUniformLocation | null> = {};
  for (const n of names) u[n] = gl.getUniformLocation(prog, n);
  return u;
}

/** Piecewise-linear in zoom, like a MapLibre "interpolate" expression. */
function interp(z: number, stops: [number, number][]): number {
  if (z <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [z1, v1] = stops[i];
    if (z <= z1) {
      const [z0, v0] = stops[i - 1];
      return v0 + ((v1 - v0) * (z - z0)) / (z1 - z0);
    }
  }
  return stops[stops.length - 1][1];
}

/** Station sizes without demand: one line width where a line runs through (flags >> 2), no riders. */
export function plainStationSize(stations: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) out[i * 2] = Math.floor((stations[i * 3 + 2] + 0.5) / 4);
  return out;
}

export interface DrawOptions {
  showTrains: boolean;
  byLevel: boolean;
  bySpeed?: boolean;
  /** selected line id, -1 for none */
  selLine: number;
  /** the selected train (T-242): its profile and departure (game seconds), drawn over the others
   * with a halo; absent for none */
  selTrain?: { profile: number; dep: number } | null;
  /** selected edge id, -1 for none */
  selEdge: number;
  /** other lines' share of full strength beside the selected line's (T-233 stop editing); 1: none */
  dimOthers?: number;
  /** selected station's index in the station buffer, -1 for none */
  selStation: number;
  /** more stations drawn selected, indices in the station buffer: in complex view, the selected
   * station's whole complex (T-163) */
  selStations?: number[];
  /** riders waiting on platforms (T-164, T-219): each platform filling up as they wait */
  crowds?: boolean;
  /** the marks on flat crossings and junctions (T-673); absent: shown */
  flatMarks?: boolean;
  /** the stations the complex rule's distances group (T-313): zoomed out their dots are joined by
   * connectors; absent, only the player's links are */
  groups?: StationGroups | null;
  /** hovered (T-730): a line's number, a bare track edge's id (-1 none), stations by index */
  hotLine?: number;
  hotEdge?: number;
  hotStations?: number[];
}

interface StrokeVao {
  vao: WebGLVertexArrayObject;
  count: number;
  /** the per-segment width multipliers, rewritten by `setLayers` */
  wmulBuf: WebGLBuffer;
  /** per segment 1 where its track is at capacity, rewritten by `setFull` (T-819) */
  fullBuf: WebGLBuffer;
}

/**
 * The demand views drawn by the network renderer (T-084, map/demandLayers.ts), one value per
 * line stroke segment, train sample or station; null = drawn as without demand.
 */
export interface NetworkLayers {
  /** per line stroke segment: width multiple */
  lineWidth: Float32Array;
  /** per train sample: how full run 0's and run 1's trains are (2 each: 0-1 of the gauge, 2 over
   * full, -1 not known) */
  fill: Float32Array;
  /** per station: the widest line through it in line widths, riders 0-1 (2 each) */
  stationSize: Float32Array;
  /** per station: its demand circle's offset from the station, local units (2 each); a station
   * complex's circle sits at the complex's mean position (T-146). Absent = none moved. */
  stationHalo?: Float32Array;
}

export class NetworkRenderer {
  private gl!: WebGL2RenderingContext;
  private strokeProg!: WebGLProgram;
  private stationProg!: WebGLProgram;
  private trainProg!: WebGLProgram;
  private strokeU!: Record<string, WebGLUniformLocation | null>;
  private stationU!: Record<string, WebGLUniformLocation | null>;
  private trainU!: Record<string, WebGLUniformLocation | null>;
  private probeProg: WebGLProgram | null = null;
  private probeU: Record<string, WebGLUniformLocation | null> = {};
  /** crossings, flat junctions, ramps and portals (T-629, T-655) */
  readonly marks = new GradeMarks();
  private res: { del: () => void }[] = [];
  private previewRes: { del: () => void }[] = [];
  private track: StrokeVao | null = null;
  private lines: StrokeVao | null = null;
  /** the crossovers where lines turn back (T-658d), drawn close in only */
  private turnback: StrokeVao | null = null;
  private preview: StrokeVao | null = null;
  private highlightRes: { del: () => void }[] = [];
  /** the drag-selected blueprint (T-333): an ink band drawn under the track */
  private highlight: StrokeVao | null = null;
  private stationVao: WebGLVertexArrayObject | null = null;
  private trainVao: WebGLVertexArrayObject | null = null;
  private probeVao: WebGLVertexArrayObject | null = null;
  private tripBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: when it stops being drawn, relative to the epoch (T-206) */
  private endBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: its first load and when it reaches its last stop (T-164) */
  private loadBuf: WebGLBuffer | null = null;
  /** the hour's loads (T-164), replaced with the trips; kept apart from `res`, like the trips */
  private loadTex: WebGLTexture | null = null;
  private tex: Record<"phases" | "meta" | "samples" | "lines" | "colours" | "fill" | "cars" | "stretch" | "stationLine" | "lane", WebGLTexture | null> = { phases: null, meta: null, samples: null, lines: null, colours: null, fill: null, cars: null, stretch: null, stationLine: null, lane: null };
  /** The boarding account's hour the trip buffer holds (T-164), or null before demand has riders. */
  board: BoardHour | null = null;
  /** per platform of `board`: its station's index in the station buffer (-1 none) and its side;
   * per station which sides have platforms (1 the + side, 2 the - side) */
  private crowdOf: { station: Int32Array; side: Int8Array; sides: Uint8Array; board: BoardHour; net: NetworkBuffers } | null = null;
  private crowdBuf: WebGLBuffer | null = null;
  /** per station: how full its + and - side are (0-1), whether the last train there left anyone (T-219) */
  private crowds = new Float32Array(0);
  /** per station: riders waiting on its + and - side */
  private crowdWait = new Float32Array(0);
  /** per station, zoomed out (T-228): its dot's fullest platform side (0-1) and whether a train
   * there left riders behind; and the riders waiting at it */
  private dotCrowds = new Float32Array(0);
  private dotWait = new Float32Array(0);
  private dotCrowdBuf: WebGLBuffer | null = null;
  /** the time and hour the crowds were last worked out for */
  private crowdAt = { t: NaN, board: null as BoardHour | null };
  /** the trip buffer's index of a trip (profile, departure - epoch), for the train inspector */
  tripIndex(profile: number, depRel: number): number {
    const d = this.trips;
    for (let i = 0; i < d.length; i += 2) if (d[i] === profile && Math.abs(d[i + 1] - depRel) < 0.01) return i / 2;
    return -1;
  }
  /** ms the last crowd update took (`?debug=1`) */
  crowdMs = 0;
  /** whether the last draw drew platform crowds (on, riders known, zoomed in to CROWD_ZOOM), for
   * the crowd hover */
  crowdsShown = false;
  private maxCars = 8;
  /** Used by picking to follow the same close-up representation. */
  detailedTrains = false;
  private stationSizeBuf: WebGLBuffer | null = null;
  private stationHaloBuf: WebGLBuffer | null = null;
  private stationSizes: Float32Array = new Float32Array(0);
  private stationShapeBuf: WebGLBuffer | null = null;
  /** how the last draw sized stations, CSS px (picking, the cut-out above the demand bubbles) */
  stationStyle: StationStyle = { boxes: false, radius: stationRadii(12), linePx: 0, ringPx: stationRing(12), boxLinePx: interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx: stationRing(BOX_WIDTH_ZOOM) };
  /** zoomed out below STATIONS_ZOOM at the last draw: trains neither drawn nor picked (T-205) */
  trainsHidden = false;
  private selectedStation = -1;
  /** `DrawOptions.selStations` as a set, and as drawn into `stationSelBuf` (its key) */
  private selectedAlso = new Set<number>();
  private selectedAlsoKey = "";
  private stationSelBuf: WebGLBuffer | null = null;
  /** what `stationSelBuf` and `bridgeFlagBuf` were last written for */
  private selDrawnKey = "";
  /** the groups the connectors were last worked out for (T-313) */
  private groupSrc: { groups: StationGroups | null; net: NetworkBuffers | null } = { groups: null, net: null };
  private bridgeProg: WebGLProgram | null = null;
  private bridgeU: Record<string, WebGLUniformLocation | null> = {};
  private bridgeVao: WebGLVertexArrayObject | null = null;
  private bridgeFlagBuf: WebGLBuffer | null = null;
  /** zoomed out, the connectors between dots (T-313): the two stations of each (index pairs), their
   * dots' centres (local units), and its VAO with the selected and faded flags; rebuilt by
   * syncGroups, so kept apart from `res` */
  connPairs = new Int32Array(0);
  private connEnds = new Float32Array(0);
  private connVao: WebGLVertexArrayObject | null = null;
  private connFlagBuf: WebGLBuffer | null = null;
  private connRes: { del: () => void }[] = [];
  private selProg: WebGLProgram | null = null;
  private selU: Record<string, WebGLUniformLocation | null> = {};
  private selTex: WebGLTexture | null = null;
  private selData = new Float32Array(SEL_MAX * SEL_STRIDE * 4);
  /** Station `i` on screen as the last draw drew it (stationGeometry.ts): its platform, or zoomed
   * out its own dot (T-313). */
  drawnShapeOf(i: number, project: Project): StationShape {
    return this.stationShapeOf(i, project);
  }
  /** The bridges between linked stations (T-217) drawn selected, on screen: both their stations
   * are; zoomed out, the connectors between selected dots (T-313). */
  private selectedBridges(project: Project): StationShape[] {
    const n = this.net;
    if (!n) return [];
    const out: StationShape[] = [];
    if (!this.stationStyle.boxes) {
      const p = this.connPairs, c = this.connEnds, half = connectorHalfPx(this.stationStyle);
      for (let k = 0; k < p.length / 2; k++) {
        if (!this.drawnSelected(p[k * 2]) || !this.drawnSelected(p[k * 2 + 1])) continue;
        const b = bridgeShape(c[k * 4], c[k * 4 + 1], c[k * 4 + 2], c[k * 4 + 3], project, this.stationStyle);
        if (b) out.push({ ...b, halfWidth: half });
      }
      return out;
    }
    const idx = n.stationLinkIdx, ends = n.stationLinks;
    if (!idx || !ends) return out;
    for (let k = 0; k < idx.length / 2; k++) {
      if (!this.drawnSelected(idx[k * 2]) || !this.drawnSelected(idx[k * 2 + 1])) continue;
      const b = bridgeShape(ends[k * 4], ends[k * 4 + 1], ends[k * 4 + 2], ends[k * 4 + 3], project, this.stationStyle);
      if (b) out.push(b);
    }
    return out;
  }
  /** Station `i` on screen as a platform or its own dot (stationGeometry.ts). */
  stationShapeOf(i: number, project: Project): StationShape {
    const n = this.net!;
    const flag = Math.round(n.stations[i * 3 + 2]);
    const s = n.stationShape;
    const P2 = 2 * PLATFORM_POINTS, line = n.stationLine && n.stationLine.length >= (i + 1) * P2 ? n.stationLine.subarray(i * P2, (i + 1) * P2) : undefined;
    const sh: PlatformShape = { mx: s?.[i * 4] ?? 0, my: s?.[i * 4 + 1] ?? 0, ax: n.stationAxis?.[i * 2] ?? 0, ay: n.stationAxis?.[i * 2 + 1] ?? 0, phi: s?.[i * 4 + 2] ?? 0, pts: line };
    return stationShape(n.stations[i * 3], n.stations[i * 3 + 1], sh, s?.[i * 4 + 3] ?? 0, project, this.stationStyle, (flag & 1) !== 0);
  }
  private hotList = new Int32Array(HOT_STATIONS);
  /** The bridge program's hovered instances (T-730): the pairs in `pairs` (station indices, two
   * an instance) with both stations hovered. */
  private hotPairs(bu: Record<string, WebGLUniformLocation | null>, pairs: Int32Array, hot: number[]) {
    let k = 0;
    this.hotList.fill(-1);
    if (hot.length > 1) for (let j = 0; j * 2 < pairs.length && k < HOT_STATIONS; j++) {
      if (hot.includes(pairs[j * 2]) && hot.includes(pairs[j * 2 + 1])) this.hotList[k++] = j;
    }
    this.gl.uniform1iv(bu.u_hot_st, this.hotList);
    this.gl.uniform1i(bu.u_hot_n, k);
  }

  /** Whether station `i` is drawn selected: the selected station, or in complex view its complex. */
  private drawnSelected(i: number) {
    return i === this.selectedStation || this.selectedAlso.has(i);
  }
  /** The topmost station drawn at a point, with a small pointer tolerance; demand halos are not
   * targets. Selected stations are drawn last (T-189), so they come first; zoomed far out they are
   * the only ones drawn (T-205). Zoomed out each dot gives its own station (T-313): of dots that
   * overlap, the one whose centre is nearest. */
  stationAt(px: number, py: number, project: Project, padding = 7): number {
    const n = this.net;
    if (!n) return -1;
    const dots = !this.stationStyle.boxes;
    // T-724: on a flat camera, a station whose node is further from the pointer than its platform
    // and shape reach is skipped before its shape is worked out (same answers)
    const [ox, oy] = project(0, 0), [ex, ey] = project(1000, 0), [nx, ny] = project(0, 1000), [fx, fy] = project(1000, 1000);
    const a = (ex - ox) / 1000, c = (ey - oy) / 1000, b = (nx - ox) / 1000, e = (ny - oy) / 1000;
    const flat = Math.abs(fx - (ex + nx - ox)) + Math.abs(fy - (ey + ny - oy)) < 0.01;
    const reach = flat ? this.stationReachNow(n) : null;
    const perLoc = 1.01 * Math.max(Math.hypot(a, c), Math.hypot(b, e));
    const extra = reach ? stationReachPx(this.stationStyle, reach.halfWidth, perLoc) + padding + 2 : 0;
    for (const sel of this.stationStyle.hidden ? [true] : [true, false]) {
      let best = -1, bestD = Infinity;
      for (let i = n.stationCount - 1; i >= 0; i--) {
        if (this.drawnSelected(i) !== sel) continue;
        if (reach) {
          const x = n.stations[i * 3], y = n.stations[i * 3 + 1], r = reach.r[i] * perLoc + extra;
          const sx = ox + a * x + b * y, sy = oy + c * x + e * y;
          if (Math.abs(sx - px) > r || Math.abs(sy - py) > r) continue;
        }
        const s = this.drawnShapeOf(i, project);
        if (!insideStation(s, px, py, padding)) continue;
        if (!dots) return i;
        const d = Math.hypot(px - s.x, py - s.y);
        if (d < bestD) [best, bestD] = [i, d];
      }
      if (best >= 0) return best;
    }
    return -1;
  }
  private reachOf: { net: NetworkBuffers; r: Float32Array; halfWidth: number } | null = null;
  /** Per station, local units from its node to the farthest point of its platform's centre line,
   * and the widest true half width of any (stationAt's skip, T-724); worked out when first asked
   * for, a few ms where scanning every station's shape cost Japan's network 100 ms a move (T-1020). */
  private stationReachNow(n: NetworkBuffers): { r: Float32Array; halfWidth: number } {
    if (this.reachOf?.net === n) return this.reachOf;
    const r = new Float32Array(n.stationCount), s = n.stationShape, P2 = 2 * PLATFORM_POINTS;
    let halfWidth = 0;
    for (let i = 0; i < n.stationCount; i++) {
      const line = n.stationLine && n.stationLine.length >= (i + 1) * P2 ? n.stationLine.subarray(i * P2, (i + 1) * P2) : undefined;
      const sh: PlatformShape = { mx: s?.[i * 4] ?? 0, my: s?.[i * 4 + 1] ?? 0, ax: n.stationAxis?.[i * 2] ?? 0, ay: n.stationAxis?.[i * 2 + 1] ?? 0, phi: s?.[i * 4 + 2] ?? 0, pts: line };
      const q = platformPoints(sh);
      let m = Math.hypot(sh.mx, sh.my) + Math.hypot(sh.ax, sh.ay);
      for (let j = 0; j + 1 < q.length; j += 2) m = Math.max(m, Math.hypot(q[j], q[j + 1]));
      r[i] = m;
      halfWidth = Math.max(halfWidth, s?.[i * 4 + 3] ?? 0);
    }
    this.reachOf = { net: n, r, halfWidth };
    return this.reachOf;
  }
  /** The selected stations drawn, by index: the selected one, or its complex in complex view. */
  private selectedDrawn(): number[] {
    const n = this.net;
    if (!n) return [];
    const out = new Set<number>();
    for (const i of [this.selectedStation, ...this.selectedAlso]) if (i >= 0 && i < n.stationCount) out.add(i);
    return [...out];
  }
  /** The selected stations as drawn, with the bridges or connectors between them (T-217, T-313):
   * the outline and glow go round them, and the demand bubbles are cut away round them so they
   * show above (T-189). */
  selectedShapes(project: Project): StationShape[] {
    const out: StationShape[] = [];
    for (const i of this.selectedDrawn()) out.push(this.drawnShapeOf(i, project));
    return out.concat(this.selectedBridges(project));
  }
  /** A selected station drawn at a point, or -1 (picking: it is above the demand bubbles). */
  selectedStationAt(px: number, py: number, project: Project, padding = 2): number {
    for (const i of this.selectedDrawn()) {
      const s = this.drawnShapeOf(i, project);
      if (s && insideStation(s, px, py, padding)) return i;
    }
    return -1;
  }
  /** called when a draw drew a different selected station or complex (the demand canvas cuts the
   * selected stations out of its bubbles, T-189) */
  onSelectionDrawn: () => void = () => {};
  /** A line's width in CSS px at the last draw, 0 when lines are not drawn (picking) */
  slotPx = 0;
  /** Half a double track's drawn width in CSS px at the last draw once it has grown to its two
   * tracks' width (T-658), 0 further out (picking) */
  laneReachPx = 0;
  /** Metres a train on double track is drawn to the right of its line's centre line at the last
   * draw (T-658): 2 m once the two tracks have parted, 0 further out (picking) */
  laneShiftM = 0;
  /** per line stroke segment, its width as a multiple of `slotPx` (traffic thickness), or null
   * when all are plain: picking reaches across wide lines (T-227) */
  lineWidths: Float32Array | null = null;
  /** the network has blueprint track: it and its lines are drawn after the rest (T-695) */
  hasBlueprint = false;
  private levelColours = new Float32Array([3, 2, 1, 0, -1, -2, -3].flatMap((l) => rgb(LEVEL_COLOURS[l as 3]).map((v) => v / 255)));
  private ink = new Float32Array(rgb(NET_INK).map((v) => v / 255));
  private timerExt: any = null;
  /** the canvas has a stencil buffer: the track is drawn through station boxes (T-236) */
  private hasStencil = false;
  private pendingQuery: WebGLQuery | null = null;
  /** trips per epoch from the worker, newest version kept, with the boarding account's hour */
  private tripsBy = new Map<number, { version: number; data: Float32Array; ends?: Float32Array; nexts?: Float32Array; board?: BoardHour }>();
  /** what the trip buffer holds */
  private tripEpoch = NaN;
  private tripVersion = -1;
  tripCount = 0;
  /** the trips in the buffer, profile and departure - epoch (for picking) */
  trips: Float32Array = new Float32Array(0);
  /** per trip in `trips`, when it stops being drawn - epoch (T-206), or null */
  tripEnds: Float32Array | null = null;
  /** per trip in `trips`, the departure - epoch its train takes next, NaN leaving service (T-245) */
  tripNexts: Float32Array | null = null;
  net: NetworkBuffers | null = null;
  /** last GPU time for our draws in ms, if the timer extension exists */
  gpuMs: number | null = null;
  /** current render epoch, game seconds; NaN until the first draw */
  epoch = NaN;
  rebases = 0;
  /** called with each new epoch the clock enters (the worker sends its trips) */
  onEpoch: (epoch: number) => void = () => {};

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.strokeProg = compile(gl, STROKE_VS, STROKE_FS);
    this.stationProg = compile(gl, STATION_VS, STATION_FS);
    this.marks.init(gl);
    this.trainProg = compile(gl, TRAIN_VS, TRAIN_FS);
    this.hasStencil = (gl.getParameter(gl.STENCIL_BITS) ?? 0) > 0;
    this.strokeU = uniforms(gl, this.strokeProg, ["u_matrix", "u_viewport", "u_width", "u_casing", "u_casing_rgb", "u_cap", "u_kind", "u_sel_line", "u_sel_edge", "u_hot", "u_only_sel", "u_bp_part", "u_lanes", "u_lane_mode", "u_dim", "u_by_level", "u_by_speed", "u_px_per_unit", "u_level_colours", "u_track", "u_bad", "u_sel", "u_bp", "u_bp_mid", "u_bp_rail", "u_bp_off_mid", "u_bp_off_rail", "u_bp_off_fade", "u_line_colours", "u_full_pass", "u_full_px", "u_full_rgb", "u_ghost", "u_ghost_rgb", "u_ghost_min"]);
    this.stationU = uniforms(gl, this.stationProg, ["u_matrix", "u_viewport", "u_radius", "u_ring", "u_ink", "u_line", "u_boxes", "u_fixed", "u_pass", "u_layer", "u_halo", "u_halo_radius", "u_pixel", "u_bad", "u_crowd_rgb", "u_tint", "u_hair", "u_plat", "u_mask", "u_stline", "u_lanes", "u_hot_st", "u_hot_n"]);
    this.bridgeProg = compile(gl, BRIDGE_VS, STATION_FS);
    this.bridgeU = uniforms(gl, this.bridgeProg, ["u_matrix", "u_viewport", "u_ring", "u_bridge", "u_pass", "u_layer", "u_halo", "u_ink", "u_bad", "u_crowd_rgb", "u_mask", "u_hot_st", "u_hot_n"]);
    this.selProg = compile(gl, SEL_VS, SEL_FS);
    this.selU = uniforms(gl, this.selProg, ["u_bbox", "u_css", "u_shapes", "u_count", "u_dpr", "u_height", "u_ink", "u_outline", "u_glow"]);
    this.selTex = this.texture(this.selData, "rgba32f", SEL_MAX * SEL_STRIDE, []);
    this.trainU = uniforms(gl, this.trainProg, ["u_matrix", "u_viewport", "u_time", "u_px_per_unit", "u_min_px", "u_phases", "u_meta", "u_samples", "u_lines", "u_line_colours", "u_fill", "u_bad", "u_detail", "u_dpr", "u_cars", "u_loads", "u_stretch", "u_sel_line", "u_dim", "u_sel_trip", "u_pass", "u_ink", "u_band", "u_psi0", "u_lane_tex", "u_lane"]);
    this.timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.tripBuf = gl.createBuffer();
    this.endBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.endBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1e9]), gl.DYNAMIC_DRAW);
    this.loadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.loadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, 0]), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.loadTex = this.texture(new Float32Array(1), "r32f", TEX_W, []);
  }

  private texture(data: ArrayBufferView, kind: "r32f" | "rg32f" | "rgba32f" | "rgba8", width: number, list = this.res): WebGLTexture {
    const gl = this.gl;
    const comps = kind === "r32f" ? 1 : kind === "rg32f" ? 2 : 4;
    const texels = Math.max(1, Math.ceil((data as Float32Array).length / comps));
    const height = Math.max(1, Math.ceil(texels / width));
    const padded = kind === "rgba8" ? new Uint8Array(width * height * 4) : new Float32Array(width * height * comps);
    padded.set(data as any);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    if (kind === "r32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, padded);
    else if (kind === "rg32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, width, height, 0, gl.RG, gl.FLOAT, padded);
    else if (kind === "rgba32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, padded);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, padded);
    list.push({ del: () => gl.deleteTexture(tex) });
    return tex;
  }

  private buffer(data: ArrayBufferView, list = this.res): WebGLBuffer {
    const gl = this.gl;
    const b = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    // left bound, WebGL refuses it as a transform feedback target (the probe)
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    list.push({ del: () => gl.deleteBuffer(b) });
    return b;
  }

  private attrib(prog: WebGLProgram, name: string, buf: WebGLBuffer, size: number, type: number, divisor: number, integer = false) {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    if (integer) gl.vertexAttribIPointer(loc, size, type, 0, 0);
    else gl.vertexAttribPointer(loc, size, type, false, 0, 0);
    gl.vertexAttribDivisor(loc, divisor);
  }

  private strokes(s: Strokes, list = this.res): StrokeVao {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    list.push({ del: () => gl.deleteVertexArray(vao) });
    gl.bindVertexArray(vao);
    const p = this.strokeProg;
    const one = (a: ArrayBufferView, n: number) => (s.count ? a : new Float32Array(n));
    this.attrib(p, "a_seg", this.buffer(one(s.seg, 4), list), 4, gl.FLOAT, 1);
    this.attrib(p, "a_colour", this.buffer(s.count ? s.colour : new Uint32Array(1), list), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_edge", this.buffer(s.count ? s.edge : new Uint32Array(1), list), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_level", this.buffer(one(s.level, 1), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_flags", this.buffer(one(s.flags, 1), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_speed", this.buffer(s.count && s.speed ? s.speed : new Float32Array(Math.max(1, s.count)).fill(160), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_dist", this.buffer(one(s.dist, 1), list), 1, gl.FLOAT, 1);
    const wmulBuf = this.buffer(new Float32Array(Math.max(1, s.count)).fill(1), list);
    this.attrib(p, "a_wmul", wmulBuf, 1, gl.FLOAT, 1);
    const fullBuf = this.buffer(new Float32Array(Math.max(1, s.count)), list);
    this.attrib(p, "a_full", fullBuf, 1, gl.FLOAT, 1);
    const lane = s.count && s.lane ? s.lane : new Uint8Array(Math.max(1, s.count) * 2).fill(255);
    const loc = gl.getAttribLocation(p, "a_lane");
    if (loc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer(lane, list));
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    return { vao, count: s.count, wmulBuf, fullBuf };
  }

  /** Rewrite a dynamic buffer in place (sizes never change between `setNetwork`s). */
  private rewrite(buf: WebGLBuffer, data: Float32Array) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** The demand views (T-084) for the network now drawn, or null for the plain drawing. The
   * arrays must be sized for `this.net` (map/demandLayers.ts builds them from it). */
  setLayers(l: NetworkLayers | null) {
    const n = this.net;
    if (!n || !this.lines) return;
    this.layersSrc = l;
    const lines = this.lines;
    if (lines.count) this.rewrite(lines.wmulBuf, l ? l.lineWidth : new Float32Array(lines.count).fill(1));
    this.lineWidths = l && l.lineWidth.length === lines.count ? l.lineWidth : null;
    const samples = Math.max(1, n.samples.length / 2);
    const upload = (tex: WebGLTexture | null, data: Float32Array, kind: "r32f" | "rg32f") => {
      const gl = this.gl;
      const comps = kind === "r32f" ? 1 : 2;
      const height = Math.max(1, Math.ceil(samples / TEX_W));
      const padded = new Float32Array(TEX_W * height * comps);
      padded.set(data.subarray(0, Math.min(data.length, padded.length)));
      if (kind === "rg32f" && !l) padded.fill(-1);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, height, kind === "r32f" ? gl.RED : gl.RG, gl.FLOAT, padded);
    };
    upload(this.tex.fill, l ? l.fill : new Float32Array(samples * 2), "rg32f");
    this.stationSizes = l ? l.stationSize : plainStationSize(n.stations, n.stationCount);
    if (this.stationSizeBuf && n.stationCount) this.rewrite(this.stationSizeBuf, this.stationSizes);
    if (this.stationHaloBuf && n.stationCount) this.rewrite(this.stationHaloBuf, l?.stationHalo ?? new Float32Array(n.stationCount * 2));
  }

  /** T-819: the edges whose track is at capacity, outlined in red on the lines over them; null or
   * empty for none (any view but traffic). */
  setFull(edges: ReadonlySet<number> | null) {
    const was = this.fullEdges;
    if ((!edges || !edges.size) && (!was || !was.size)) return;
    if (edges && was && edges.size === was.size && [...edges].every((e) => was.has(e))) return;
    this.fullEdges = edges;
    this.writeFull();
  }
  private fullEdges: ReadonlySet<number> | null = null;
  private writeFull() {
    const l = this.lines, n = this.net;
    if (!l || !l.count || !n) return;
    const out = new Float32Array(l.count), f = this.fullEdges;
    if (f?.size) for (let i = 0; i < l.count; i++) if (f.has(n.lines.edge[i])) out[i] = 1;
    this.rewrite(l.fullBuf, out);
  }

  /** Replace everything drawn but the trips (an edit). */
  setNetwork(n: NetworkBuffers) {
    const gl = this.gl;
    for (const r of this.res) r.del();
    this.res = [];
    for (const r of this.serviceRes) r.del();
    this.serviceRes = [];
    this.probeVao = null;
    this.net = n;
    this.marks.setNetwork(n.marks);
    this.track = this.strokes(n.track);
    this.lines = this.strokes(n.lines);
    this.writeFull();
    this.turnback = n.turnbacks?.count ? this.strokes(n.turnbacks) : null;
    // blueprint track sorts last (network.ts)
    this.hasBlueprint = n.track.count > 0 && (n.track.flags[n.track.count - 1] & F_BLUEPRINT) !== 0;
    this.lineWidths = null;
    this.tex.phases = this.texture(n.phases.length ? n.phases : new Float32Array(4), "rgba32f", TEX_W);
    this.tex.meta = this.texture(n.meta.length ? n.meta : new Float32Array(4), "rgba32f", TEX_W);
    this.tex.samples = this.texture(n.samples.length ? n.samples : new Float32Array(2), "rg32f", TEX_W);
    this.tex.lane = this.texture(n.sampleLane?.length ? n.sampleLane : new Float32Array(Math.max(1, n.samples.length / 2)).fill(1), "r32f", TEX_W);
    this.tex.lines = this.texture(n.lineTable.length ? n.lineTable : new Float32Array(4), "rgba32f", LINES_W);
    this.tex.colours = this.texture(n.lineColours, "rgba8", LINES_W);
    this.tex.fill = this.texture(new Float32Array(Math.max(1, n.samples.length / 2) * 2).fill(-1), "rg32f", TEX_W);
    const cars = n.lineCars ?? new Float32Array(Math.max(1, n.lineTable.length / 4)).fill(8);
    this.maxCars = Math.max(1, ...cars);
    this.tex.cars = this.texture(cars, "r32f", LINES_W);
    // each train sample's stop-to-stop stretch (run 0's numbering), where a trip's loads are
    // looked up (T-164)
    const nSamples = Math.max(1, n.samples.length / 2);
    const stretch = new Float32Array(nSamples).fill(-1);
    if (n.sampleStroke && n.lines.info)
      for (let k = 0; k < nSamples; k++) {
        const i = n.sampleStroke[k];
        if (i >= 0) stretch[k] = n.lines.info[i * 4];
      }
    this.tex.stretch = this.texture(stretch, "r32f", TEX_W);
    this.crowds = new Float32Array(Math.max(1, n.stationCount) * 4);
    this.crowdWait = new Float32Array(Math.max(1, n.stationCount) * 2);
    this.dotCrowds = new Float32Array(Math.max(1, n.stationCount) * 2);
    this.dotWait = new Float32Array(Math.max(1, n.stationCount));
    this.crowdOf = null;
    this.crowdAt = { t: NaN, board: null };

    this.stationVao = gl.createVertexArray()!;
    const sv = this.stationVao;
    this.res.push({ del: () => gl.deleteVertexArray(sv) });
    gl.bindVertexArray(this.stationVao);
    this.attrib(this.stationProg, "a_st", this.buffer(n.stations.length ? n.stations : new Float32Array(3)), 3, gl.FLOAT, 1);
    this.attrib(this.stationProg, "a_axis", this.buffer(n.stationAxis?.length ? n.stationAxis : new Float32Array(Math.max(1,n.stationCount)*2)),2,gl.FLOAT,1);
    this.stationShapeBuf = this.buffer(n.stationShape?.length ? n.stationShape : new Float32Array(Math.max(1, n.stationCount) * 4));
    // each platform's track line (T-254), or the arc of its shape where the buffers have none
    const P2 = 2 * PLATFORM_POINTS;
    if (!n.stationLine || n.stationLine.length < n.stationCount * P2) {
      const line = new Float32Array(n.stationCount * P2), s = n.stationShape;
      for (let i = 0; i < n.stationCount; i++)
        line.set(platformPoints({ mx: s?.[i * 4] ?? 0, my: s?.[i * 4 + 1] ?? 0, ax: n.stationAxis?.[i * 2] ?? 0, ay: n.stationAxis?.[i * 2 + 1] ?? 0, phi: s?.[i * 4 + 2] ?? 0 }), i * P2);
      n.stationLine = line;
    }
    this.tex.stationLine = this.texture(n.stationLine.length ? n.stationLine : new Float32Array(2), "rg32f", TEX_W);
    this.attrib(this.stationProg, "a_shape", this.stationShapeBuf, 4, gl.FLOAT, 1);
    this.stationSizes=plainStationSize(n.stations,n.stationCount);
    this.stationSizeBuf = this.buffer(n.stationCount ? this.stationSizes : new Float32Array(2));
    this.attrib(this.stationProg, "a_sz", this.stationSizeBuf, 2, gl.FLOAT, 1);
    this.stationHaloBuf = this.buffer(new Float32Array(Math.max(1, n.stationCount) * 2));
    this.attrib(this.stationProg, "a_halo", this.stationHaloBuf, 2, gl.FLOAT, 1);
    this.stationSelBuf = this.buffer(new Float32Array(Math.max(1, n.stationCount) * 2));
    this.attrib(this.stationProg, "a_sel", this.stationSelBuf, 2, gl.FLOAT, 1);
    this.groupSrc = { groups: null, net: null };
    this.crowdBuf = this.buffer(this.crowds);
    this.dotCrowdBuf = this.buffer(this.dotCrowds);
    this.attrib(this.stationProg, "a_dcrowd", this.dotCrowdBuf, 2, gl.FLOAT, 1);
    this.attrib(this.stationProg, "a_crowd", this.crowdBuf, 4, gl.FLOAT, 1);
    this.selectedAlso = new Set();
    this.selectedAlsoKey = "";
    this.selDrawnKey = "";
    // the player's links as bridges (T-217)
    const links = n.stationLinkIdx?.length ?? 0;
    this.bridgeVao = gl.createVertexArray()!;
    const bv = this.bridgeVao;
    this.res.push({ del: () => gl.deleteVertexArray(bv) });
    gl.bindVertexArray(bv);
    this.attrib(this.bridgeProg!, "a_ends", this.buffer(links ? n.stationLinks! : new Float32Array(4)), 4, gl.FLOAT, 1);
    this.bridgeFlagBuf = this.buffer(new Float32Array(Math.max(1, links / 2) * 2));
    this.attrib(this.bridgeProg!, "a_bflag", this.bridgeFlagBuf, 2, gl.FLOAT, 1);
    // zoomed out the connectors between dots (T-313) are built by syncGroups

    this.trainVao = gl.createVertexArray()!;
    const tv = this.trainVao;
    this.res.push({ del: () => gl.deleteVertexArray(tv) });
    gl.bindVertexArray(this.trainVao);
    this.attrib(this.trainProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_end", this.endBuf!, 1, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_load", this.loadBuf!, 2, gl.FLOAT, 1);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Only the service changed (a train count, T-229): the run profiles' phase tables and meta are
   * replaced and every other buffer kept, rather than uploading the whole network again. */
  setService(phases: Float32Array, meta: Float32Array) {
    const n = this.net;
    if (!n) return;
    n.phases = phases;
    n.meta = meta;
    for (const r of this.serviceRes) r.del();
    this.serviceRes = [];
    // the textures `setNetwork` made stay in `res` until the next one; these are freed here
    this.tex.phases = this.texture(phases.length ? phases : new Float32Array(4), "rgba32f", TEX_W, this.serviceRes);
    this.tex.meta = this.texture(meta.length ? meta : new Float32Array(4), "rgba32f", TEX_W, this.serviceRes);
  }
  private serviceRes: { del: () => void }[] = [];

  /** The drawing tool's alignment, or null. */
  setPreview(s: Strokes | null) {
    this.previewSrc = s;
    for (const r of this.previewRes) r.del();
    this.previewRes = [];
    this.preview = s && s.count ? this.strokes(s, this.previewRes) : null;
    this.marks.setPreview(s?.marks);
  }

  /** The blueprint track picked by a drag selection (T-333), drawn as an ink band under the track, or null. */
  setHighlight(s: Strokes | null) {
    this.highlightSrc = s;
    for (const r of this.highlightRes) r.del();
    this.highlightRes = [];
    this.highlight = s && s.count ? this.strokes(s, this.highlightRes) : null;
  }

  /** What the setters were last given, for `restore`. */
  private layersSrc: NetworkLayers | null = null;
  private previewSrc: Strokes | null = null;
  private highlightSrc: Strokes | null = null;

  /** The canvas lost its context and the browser gave it a new one (T-953): every GL object went
   * with the old, so the programs are compiled again and what was drawn is uploaded again. */
  restore() {
    this.res = [];
    this.serviceRes = [];
    this.previewRes = [];
    this.highlightRes = [];
    this.connRes = [];
    this.connVao = this.connFlagBuf = null;
    this.probeProg = this.probeVao = null;
    this.pendingQuery = null;
    this.tripEpoch = NaN;
    this.tripVersion = -1;
    this.init(this.gl);
    const n = this.net, layers = this.layersSrc;
    if (n) {
      this.setNetwork(n);
      if (layers) this.setLayers(layers);
    }
    this.setPreview(this.previewSrc);
    this.setHighlight(this.highlightSrc);
  }

  /** Trips for the hour starting at `epoch` (from the worker). */
  setTrips(epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array) {
    const old = this.tripsBy.get(epoch);
    if (old && old.version > version) return;
    const hour = board ? readBoardHour(epoch, board) : undefined;
    this.tripsBy.set(epoch, { version, data, ends: ends && ends.length === data.length / 2 ? ends : undefined, nexts: nexts && nexts.length === data.length / 2 ? nexts : undefined, board: hour && hour.trips === data.length / 2 ? hour : undefined });
    if (epoch === this.tripEpoch) this.tripVersion = -1; // re-upload on the next draw
  }

  /** Make the trip buffer hold the trips of the hour `time` is in, when they are here. */
  private sync(time: number): number {
    const epoch = Math.floor(time / EPOCH_S) * EPOCH_S;
    if (epoch !== this.epoch) {
      this.epoch = epoch;
      this.rebases++;
      for (const k of this.tripsBy.keys()) if (k < epoch) this.tripsBy.delete(k);
      this.onEpoch(epoch);
    }
    const t = this.tripsBy.get(epoch);
    if (t && (this.tripEpoch !== epoch || this.tripVersion !== t.version)) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tripBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.data.length ? t.data : new Float32Array(2), gl.DYNAMIC_DRAW);
      // when each is drawn until (T-206); without them, the end of its run and turnaround
      gl.bindBuffer(gl.ARRAY_BUFFER, this.endBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.ends?.length ? t.ends : new Float32Array(Math.max(1, t.data.length / 2)).fill(-1e9), gl.DYNAMIC_DRAW);
      // each trip's place in the boarding account's loads (T-164), -1 without one
      const offs = t.board ? t.board.offs : new Float32Array(Math.max(1, t.data.length / 2) * 2).fill(-1);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.loadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, offs.length ? offs : new Float32Array([-1, 0]), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      if (t.board?.loads.length) {
        if (this.loadTex) gl.deleteTexture(this.loadTex);
        this.loadTex = this.texture(t.board.loads, "r32f", TEX_W, []);
      }
      this.board = t.board ?? null;
      this.tripEpoch = epoch;
      this.tripVersion = t.version;
      this.tripCount = t.data.length / 2;
      this.trips = t.data;
      this.tripEnds = t.ends ?? null;
      this.tripNexts = t.nexts ?? null;
    }
    // Until the new hour's trips arrive, the last hour's keep running on their own epoch.
    return time - this.tripEpoch;
  }

  /** The epoch the trip buffer is relative to. */
  get tripsEpoch() {
    return this.tripEpoch;
  }

  /**
   * Draw. `mercMatrix` maps mercator 0..1 to clip space (MapLibre's
   * `defaultProjectionData.mainMatrix`, float64); `time` is game seconds; `zoom` and `dpr` size
   * things in px. The caller has bound and cleared the target.
   */
  draw(mercMatrix: ArrayLike<number>, time: number, zoom: number, dpr: number, opt: DrawOptions) {
    const gl = this.gl;
    const n = this.net;
    if (!n) return;
    const uTime = this.sync(time);
    // the network's buffers are relative to the anchor they were built at (T-307)
    const a = n.anchor ?? anchor();
    const m = mul4(mercMatrix, localToMercMatrix(a));
    const vp = [gl.drawingBufferWidth, gl.drawingBufferHeight] as const;
    const pxPerUnit = ((512 * Math.pow(2, zoom)) / EARTH_C) * dpr;

    this.pollTimer();
    const timing = this.timerExt && !this.pendingQuery;
    if (timing) {
      this.pendingQuery = gl.createQuery();
      gl.beginQuery(this.timerExt.TIME_ELAPSED_EXT, this.pendingQuery!);
    }

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Line widths 60% of T-084's (T-191, Anita: "line thickness is a bit high overall"), 40% of
    // that at zoom 9 (T-205); bare track, a selected line's extra width and the white casing in step.
    const wLine = interp(zoom, LINE_PX) * dpr;
    const wTrack = interp(zoom, [[9, 0.4], [12, 1.8], [15, 3]]) * dpr;
    const wSel = interp(zoom, [[9, 0.5], [12, 2.1], [15, 3]]) * dpr;
    const casing = interp(zoom, [[9, 0.6], [12, 2], [15, 2.4]]) * dpr;
    gl.useProgram(this.strokeProg);
    const u = this.strokeU;
    gl.uniformMatrix4fv(u.u_matrix, false, m);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1i(u.u_sel_line, opt.selLine);
    gl.uniform1i(u.u_sel_edge, opt.selEdge);
    gl.uniform2i(u.u_hot, opt.hotLine ?? -1, opt.hotEdge ?? -1);
    gl.uniform1f(u.u_dim, opt.dimOthers ?? 1);
    gl.uniform1i(u.u_by_level, opt.byLevel ? 1 : 0);
    gl.uniform1i(u.u_by_speed, opt.bySpeed ? 1 : 0);
    gl.uniform1f(u.u_px_per_unit, pxPerUnit);
    // Lines sharing track are drawn over each other, the last on top (T-190); picking knows the
    // line width, and that no line strokes are drawn when the track is coloured by height or speed.
    this.slotPx = opt.byLevel || opt.bySpeed ? 0 : wLine / dpr;
    gl.uniform3fv(u.u_level_colours, this.levelColours);
    gl.uniform3fv(u.u_track, TRACK_RGB);
    gl.uniform3fv(u.u_bad, BAD_RGB);
    const shades = shadeColours();
    gl.uniform3fv(u.u_sel, rgb(shades.mapInk).map((v) => v / 255));
    gl.uniform3fv(u.u_casing_rgb, rgb(shades.casing).map((v) => v / 255));
    // T-819: 1.5 px of red either side, outside the casing, of a line on full track
    const fullPass = !!this.fullEdges?.size;
    gl.uniform1i(u.u_full_pass, 0);
    gl.uniform1f(u.u_full_px, 3 * dpr);
    gl.uniform3fv(u.u_full_rgb, rgb(shades.fullTrack).map((v) => v / 255));
    gl.uniform1i(u.u_ghost, 0);
    gl.uniform3fv(u.u_ghost_rgb, rgb(shades.ghost).map((v) => v / 255));
    gl.uniform2f(u.u_ghost_min, GHOST_MIN_PX * dpr, 2 * GHOST_CASING_PX * dpr);
    // blueprint track (T-322): a line-wide light blue middle between two thin dark blue rails
    const bpRail = Math.max(1, BP_NARROW * Math.max(1, BP_RAIL_SHARE * wLine / dpr)) * dpr;
    const bpMid = Math.max(0, BP_NARROW * (wLine + 2 * Math.max(1, BP_RAIL_SHARE * wLine / dpr) * dpr) - 2 * bpRail);
    gl.uniform2f(u.u_bp, bpMid, bpRail);
    gl.uniform3fv(u.u_bp_mid, BP_MID_RGB);
    gl.uniform3fv(u.u_bp_rail, BP_RAIL_RGB);
    gl.uniform3fv(u.u_bp_off_mid, rgb(shades.offMid).map((v) => v / 255));
    gl.uniform3fv(u.u_bp_off_rail, rgb(shades.offRail).map((v) => v / 255));
    gl.uniform3fv(u.u_bp_off_fade, rgb(shades.offFade).map((v) => v / 255));
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.colours);
    gl.uniform1i(u.u_line_colours, 2);
    const lanes = laneState(zoom);
    this.laneReachPx = lanes.reach;
    this.laneShiftM = LANE_M / 2 * lanes.t;
    const G = lanes.gap * dpr;
    gl.uniform4f(u.u_lanes, G, lanes.t, lanes.grow, lanes.share * G);
    const modes = lanes.t > 0 ? [1, 2, 3] : [0];
    // `cap` > 0: the track drawn through station boxes (T-236), no wider than that and no casing
    const stroke = (s: StrokeVao | null, kind: number, width: number, sel: boolean, cap = 0) => {
      if (!s || !s.count) return;
      gl.uniform1i(u.u_kind, kind);
      // each track's width once apart: a line's share of the gap, bare track as much thinner
      gl.uniform3f(u.u_width, width, wSel, lanes.share * G * width / wLine);
      gl.uniform1f(u.u_cap, cap);
      gl.bindVertexArray(s.vao);
      // [casing px, selected only, T-819's red band under the lines on full track]
      const passes: [number, number, number][] = cap > 0 ? [[0, 0, 0]] : [[casing, 0, 0], [0, 0, 0]];
      if (kind === 1 && fullPass && cap === 0) passes.unshift([casing, 0, 1]);
      if (sel) passes.push(...(cap > 0 ? [[0, 1, 0]] as [number, number, number][] : [[casing, 1, 0], [0, 1, 0]] as [number, number, number][]));
      for (const [cas, only, full] of passes) {
        gl.uniform1f(u.u_casing, cas);
        gl.uniform1i(u.u_only_sel, only);
        gl.uniform1i(u.u_full_pass, full);
        for (const mode of modes) {
          if (cap > 0 && mode === 1) continue;
          if (full && mode > 1) continue;
          gl.uniform1i(u.u_lane_mode, mode);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
        }
      }
      gl.uniform1i(u.u_full_pass, 0);
      gl.uniform1i(u.u_lane_mode, 0);
    };
    // Colour by height: the track itself, at line width, and no line strokes over it. Blueprint
    // track and the lines planned on it go over built track and its lines (T-695).
    const trackStrokes = (cap = 0) => {
      for (const part of this.hasBlueprint ? [1, 2] : [0]) {
        gl.uniform1i(u.u_bp_part, part);
        stroke(this.track, 0, opt.byLevel || opt.bySpeed ? wLine : wTrack, opt.selEdge >= 0, cap);
        if (!opt.byLevel && !opt.bySpeed) stroke(this.lines, 1, wLine, opt.selLine >= 0, cap);
      }
      gl.uniform1i(u.u_bp_part, 0);
    };
    if (this.highlight) {
      // an ink band 3 px beyond the blueprint's rails either side (kind 2 placement strokes in a
      // casing pass are the selection ink)
      gl.uniform1i(u.u_kind, 2);
      gl.uniform3f(u.u_width, bpMid + 2 * bpRail + 6 * dpr, 0, lanes.share * G + 6 * dpr);
      gl.uniform1f(u.u_cap, 0);
      gl.uniform1i(u.u_only_sel, 0);
      gl.uniform1f(u.u_casing, 0.01 * dpr);
      // round both tracks once they part
      gl.uniform1i(u.u_lane_mode, lanes.t > 0 ? 1 : 0);
      gl.bindVertexArray(this.highlight.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.highlight.count);
      gl.uniform1i(u.u_lane_mode, 0);
    }
    trackStrokes();
    // where lines turn back, over their tracks (T-658d)
    if (lanes.t > 0) stroke(this.turnback, 0, wTrack, false);
    gl.uniform1i(u.u_ghost, 1);
    stroke(this.preview, 2, wLine, false);
    gl.uniform1i(u.u_ghost, 0);
    this.marks.draw(m, vp, zoom, dpr, this.levelColours, opt.flatMarks !== false);

    if (n.stationCount) {
      // Platforms along the track from STATION_BOX_ZOOM, each one shape at every zoom (T-204), dots
      // further out (T-189), one for a whole complex (T-217), and below STATIONS_ZOOM only the
      // selected ones (T-205).
      const boxes = zoom >= STATION_BOX_ZOOM;
      const hidden = zoom < STATIONS_ZOOM;
      // a dot's rim is thinner than a box's (T-322)
      const ring = boxes ? stationRing(zoom) : dotRing(zoom);
      const boxLinePx = interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx = stationRing(BOX_WIDTH_ZOOM);
      this.stationStyle = { boxes, radius: stationRadii(zoom), linePx: wLine / dpr, ringPx: ring, boxLinePx, boxRingPx, hidden };
      this.syncGroups(n, opt.groups ?? null);
      const also = (opt.selStations ?? []).filter((i) => i >= 0 && i < n.stationCount);
      const alsoKey = also.join(",");
      const changed = opt.selStation !== this.selectedStation || alsoKey !== this.selectedAlsoKey;
      this.selectedStation = opt.selStation;
      if (alsoKey !== this.selectedAlsoKey) {
        this.selectedAlso = new Set(also);
        this.selectedAlsoKey = alsoKey;
      }
      this.syncSelection(n);
      if (changed) this.onSelectionDrawn();
      const [r, rx] = this.stationStyle.radius.map((v) => v * dpr);
      const su = this.stationU;
      gl.useProgram(this.stationProg);
      gl.uniformMatrix4fv(su.u_matrix, false, m);
      gl.uniform2f(su.u_viewport, vp[0], vp[1]);
      gl.uniform2f(su.u_radius, r, rx);
      gl.uniform1f(su.u_ring, ring * dpr);
      gl.uniform3fv(su.u_ink, this.ink);
      gl.uniform3fv(su.u_bad, BAD_RGB);
      gl.uniform3fv(su.u_crowd_rgb, CROWD_RGB);
      gl.uniform1f(su.u_line, wLine);
      gl.uniform3f(su.u_lanes, lanes.gap * dpr, lanes.t, lanes.narrow);
      gl.uniform1f(su.u_boxes, boxes ? 1 : 0);
      const hotSt = (opt.hotStations ?? []).filter((i) => i >= 0 && i < n.stationCount);
      this.hotList.fill(-1);
      hotSt.slice(0, HOT_STATIONS).forEach((i, k) => (this.hotList[k] = i));
      gl.uniform1iv(su.u_hot_st, this.hotList);
      gl.uniform1i(su.u_hot_n, Math.min(hotSt.length, HOT_STATIONS));
      // each platform's track line (T-254)
      gl.activeTexture(gl.TEXTURE11);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.stationLine);
      gl.uniform1i(su.u_stline, 11);
      gl.activeTexture(gl.TEXTURE0);
      // local units per px at BOX_WIDTH_ZOOM (the box's width, T-204)
      const unitAtBox = EARTH_C / (512 * 2 ** BOX_WIDTH_ZOOM * dpr);
      gl.uniform2f(su.u_fixed, boxHalfPxAtBoxZoom({ boxLinePx, boxRingPx }) * dpr, unitAtBox);
      gl.bindVertexArray(this.stationVao);
      // Analytical circles behind the normal, still-selectable platform boxes.
      gl.uniform1f(su.u_halo_radius, stationHaloRadius(zoom) * dpr);
      gl.uniform1f(su.u_pixel, dpr);
      gl.uniform1i(su.u_halo, 1);
      gl.uniform1i(su.u_layer, 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
      gl.uniform1i(su.u_halo, 0);
      // the player's links as bridges between platforms (T-217)
      const bridges = boxes && this.bridgeProg ? (n.stationLinkIdx?.length ?? 0) / 2 : 0;
      if (bridges) {
        const bu = this.bridgeU;
        gl.useProgram(this.bridgeProg);
        gl.uniformMatrix4fv(bu.u_matrix, false, m);
        gl.uniform2f(bu.u_viewport, vp[0], vp[1]);
        gl.uniform1f(bu.u_ring, ring * dpr);
        gl.uniform1f(bu.u_bridge, BRIDGE_HALF_PX * dpr * unitAtBox);
        gl.uniform1i(bu.u_halo, 0);
        gl.uniform3fv(bu.u_ink, this.ink);
        gl.uniform3fv(bu.u_bad, BAD_RGB);
        gl.uniform3fv(bu.u_crowd_rgb, CROWD_RGB);
        this.hotPairs(bu, n.stationLinkIdx!, hotSt);
      }
      // zoomed out, the connectors between a complex's dots and the player's links (T-313, Anita:
      // "it should look like its made of the same material"): the dots' rim, and their white
      // drawn in the same layer, so dots and connectors are one white shape with one outline
      const connectors = !boxes && this.bridgeProg && this.connVao ? this.connPairs.length / 2 : 0;
      if (connectors) {
        const bu = this.bridgeU, p = this.projector(m, vp[0] / dpr, vp[1] / dpr);
        const [x0, y0] = p(n.stations[0], n.stations[1]), [x1, y1] = p(n.stations[0] + 1, n.stations[1]);
        const pxPerLoc = Math.max(1e-9, Math.hypot(x1 - x0, y1 - y0));
        gl.useProgram(this.bridgeProg);
        gl.uniformMatrix4fv(bu.u_matrix, false, m);
        gl.uniform2f(bu.u_viewport, vp[0], vp[1]);
        gl.uniform1f(bu.u_ring, ring * dpr);
        gl.uniform1f(bu.u_bridge, (connectorHalfPx(this.stationStyle) - ring) / pxPerLoc);
        gl.uniform1i(bu.u_halo, 0);
        gl.uniform3fv(bu.u_ink, this.ink);
        gl.uniform3fv(bu.u_bad, BAD_RGB);
        gl.uniform3fv(bu.u_crowd_rgb, CROWD_RGB);
        gl.uniform1f(bu.u_mask, 0);
        this.hotPairs(bu, this.connPairs, hotSt);
      }
      const connDraw = (pass: number, layer: number) => {
        gl.useProgram(this.bridgeProg);
        gl.uniform1i(this.bridgeU.u_pass, pass);
        gl.uniform1i(this.bridgeU.u_layer, layer);
        gl.bindVertexArray(this.connVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, connectors);
      };
      // Riders waiting on the platforms (T-164, T-219), where platforms are drawn; zoomed out a pie
      // in each dot (T-228, T-553).
      this.crowdsShown = !!opt.crowds && !!this.board && !hidden;
      const any = this.crowdsShown && this.updateCrowds(n, uTime);
      const fills = any && boxes;
      gl.useProgram(this.stationProg);
      gl.uniform1f(su.u_tint, any && !boxes ? 1 : 0);
      gl.uniform1f(su.u_hair, dpr);
      gl.uniform3fv(su.u_plat, PLATFORM_RGB);
      gl.uniform1f(su.u_mask, 0);
      // The track through the boxes (T-236, Anita: "maybe the track on top of the station"): the
      // track and line strokes drawn again, no wider than TRACK_IN_BOX of the box's white and
      // without casing, where the stencil marks a station's white (1 the unselected, 2 the selected).
      const through = boxes && this.hasStencil;
      // every box has one width on the ground (T-239); any station's gives it on screen here
      const boxWhite = 2 * ((boxes ? this.stationShapeOf(0, this.projector(m, vp[0] / dpr, vp[1] / dpr)).halfWidth : 0) - ring);
      const trackCap = Math.max(1, TRACK_IN_BOX * boxWhite) * dpr;
      if (through) {
        gl.clearStencil(0);
        gl.clear(gl.STENCIL_BUFFER_BIT);
      }
      const layerDraw = (pass: number, layer: number, withBridges: boolean) => {
        gl.useProgram(this.stationProg);
        gl.uniform1i(su.u_pass, pass);
        gl.uniform1i(su.u_layer, layer);
        gl.bindVertexArray(this.stationVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
        if (bridges && withBridges) {
          gl.useProgram(this.bridgeProg);
          gl.uniform1i(this.bridgeU.u_pass, pass);
          gl.uniform1i(this.bridgeU.u_layer, layer);
          gl.bindVertexArray(this.bridgeVao);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, bridges);
        }
      };
      // every station, then the selected ones over them with their outline and glow; each pass the
      // dark shapes, then the white inside every rim, so stations drawn over each other are one white
      // shape with one outline (T-217), then the crowds, the track through them and the blue
      // platform edges (T-236)
      for (const pass of [0, 1]) {
        if (pass === 0 && hidden) continue;
        if (pass === 1) {
          const shapes = this.selectedShapes(this.projector(m, vp[0] / dpr, vp[1] / dpr));
          if (!shapes.length) break;
          this.drawGlow(shapes, dpr, vp);
        }
        // zoomed out the connectors' white goes under the dots' white, so a dot's crowd pie (T-228)
        // stays whole (T-313)
        if (connectors) connDraw(pass, 0);
        layerDraw(pass, 0, true);
        if (connectors) connDraw(pass, 1);
        layerDraw(pass, 1, true);
        if (!boxes) continue;
        if (through) {
          // mark this pass's white, no colour
          gl.enable(gl.STENCIL_TEST);
          gl.stencilFunc(gl.ALWAYS, pass + 1, 0xff);
          gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
          gl.colorMask(false, false, false, false);
          gl.useProgram(this.stationProg);
          gl.uniform1f(su.u_mask, 1);
          if (bridges) {
            gl.useProgram(this.bridgeProg);
            gl.uniform1f(this.bridgeU.u_mask, 1);
          }
          layerDraw(pass, 1, true);
          gl.useProgram(this.stationProg);
          gl.uniform1f(su.u_mask, 0);
          if (bridges) {
            gl.useProgram(this.bridgeProg);
            gl.uniform1f(this.bridgeU.u_mask, 0);
          }
          gl.colorMask(true, true, true, true);
          gl.disable(gl.STENCIL_TEST);
        }
        if (fills) layerDraw(pass, 2, false);
        if (through) {
          gl.enable(gl.STENCIL_TEST);
          gl.stencilFunc(gl.EQUAL, pass + 1, 0xff);
          gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
          gl.useProgram(this.strokeProg);
          trackStrokes(trackCap);
          gl.uniform1f(u.u_cap, 0);
          gl.disable(gl.STENCIL_TEST);
        }
        layerDraw(pass, 3, false);
      }
    }

    // No trains zoomed far out (T-205).
    this.trainsHidden = zoom < TRAINS_ZOOM;
    if (opt.showTrains && this.tripCount && !this.trainsHidden) {
      this.detailedTrains = zoom >= STATION_BOX_ZOOM;
      const tw = interp(zoom, TRAIN_PX) * dpr;
      gl.useProgram(this.trainProg);
      gl.uniformMatrix4fv(this.trainU.u_matrix, false, m);
      gl.uniform2f(this.trainU.u_viewport, vp[0], vp[1]);
      gl.uniform1f(this.trainU.u_time, uTime);
      gl.uniform1f(this.trainU.u_px_per_unit, pxPerUnit);
      gl.uniform1f(this.trainU.u_psi0, anchorPsi(a));
      // a pill no shorter than before T-199 thinned it (2.2 times T-107's width)
      gl.uniform2f(this.trainU.u_min_px, tw * 2.2 / 0.75, tw);
      this.bindTrainTextures(this.trainU);
      gl.uniform1i(this.trainU.u_line_colours, 2);
      gl.uniform1i(this.trainU.u_sel_line, opt.selLine);
      gl.uniform1f(this.trainU.u_dim, opt.dimOthers ?? 1);
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.fill);
      gl.uniform1i(this.trainU.u_fill, 6);
      gl.uniform3fv(this.trainU.u_bad, BAD_RGB);
      gl.uniform1i(this.trainU.u_detail, this.detailedTrains ? 1 : 0);
      gl.uniform1f(this.trainU.u_dpr, dpr);
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.cars);
      gl.uniform1i(this.trainU.u_cars, 7);
      gl.activeTexture(gl.TEXTURE8);
      gl.bindTexture(gl.TEXTURE_2D, this.loadTex);
      gl.uniform1i(this.trainU.u_loads, 8);
      gl.activeTexture(gl.TEXTURE9);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.stretch);
      gl.uniform1i(this.trainU.u_stretch, 9);
      gl.activeTexture(gl.TEXTURE10);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.lane);
      gl.uniform1i(this.trainU.u_lane_tex, 10);
      // on its own track close in (T-658): the shift at full parting, how far parted
      gl.uniform2f(this.trainU.u_lane, lanes.gap * dpr * 0.5, lanes.t);
      gl.bindVertexArray(this.trainVao);
      // every train but the selected; then the selected one's halo, round the whole consist as one
      // pill; then the selected train over it (T-242)
      const sel = opt.selTrain;
      gl.uniform2f(this.trainU.u_sel_trip, sel ? sel.profile : -1, sel ? sel.dep - this.tripEpoch : 0);
      const halo = shadeColours();
      gl.uniform3fv(this.trainU.u_ink, rgb(halo.mapInk).map((v) => v / 255));
      gl.uniform3fv(this.trainU.u_band, rgb(halo.casing).map((v) => v / 255));
      gl.uniform1i(this.trainU.u_pass, 0);
      const drawTrains = () => gl.drawArraysInstanced(this.detailedTrains ? gl.TRIANGLES : gl.TRIANGLE_STRIP, 0, this.detailedTrains ? 6 * this.maxCars : 4, this.tripCount);
      drawTrains();
      if (sel) {
        gl.uniform1i(this.trainU.u_pass, 1);
        gl.uniform1i(this.trainU.u_detail, 0);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.tripCount);
        gl.uniform1i(this.trainU.u_pass, 2);
        gl.uniform1i(this.trainU.u_detail, this.detailedTrains ? 1 : 0);
        drawTrains();
      }
    }

    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
    if (timing) gl.endQuery(this.timerExt.TIME_ELAPSED_EXT);
  }

  private crowdAny = false;

  /** Local units to CSS px on the canvas (y down) through the draw's matrix `m`. */
  private projector(m: ArrayLike<number>, w: number, h: number): Project {
    return (x, y) => {
      const cx = m[0] * x + m[4] * y + m[12], cy = m[1] * x + m[5] * y + m[13], cw = m[3] * x + m[7] * y + m[15];
      return [((cx / cw + 1) / 2) * w, ((1 - cy / cw) / 2) * h];
    };
  }

  /** The outline and glow round the selected shapes' union (T-189, T-217), under them. */
  private drawGlow(shapes: StationShape[], dpr: number, vp: readonly [number, number]) {
    const gl = this.gl;
    const count = Math.min(shapes.length, SEL_MAX), d = this.selData;
    const reach = SEL_OUTLINE_PX + SEL_GLOW_PX + 1;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < count; i++) {
      const s = shapes[i];
      const o = i * SEL_STRIDE * 4;
      d.fill(0, o, o + SEL_STRIDE * 4);
      d.set([s.x, s.y, s.ux, s.uy, s.nx, s.ny, s.halfLength, s.halfWidth, s.dot], o);
      if (!(s.dot > 0)) d.set(shapePoints(s), o + 12);
      // every point of a shape is within its half length plus half width of its middle
      const r = (s.dot > 0 ? s.dot : s.halfLength + s.halfWidth) + reach;
      x0 = Math.min(x0, s.x - r); y0 = Math.min(y0, s.y - r); x1 = Math.max(x1, s.x + r); y1 = Math.max(y1, s.y + r);
    }
    const w = vp[0] / dpr, h = vp[1] / dpr;
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(w, x1); y1 = Math.min(h, y1);
    if (x1 <= x0 || y1 <= y0) return;
    gl.activeTexture(gl.TEXTURE10);
    gl.bindTexture(gl.TEXTURE_2D, this.selTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, count * SEL_STRIDE, 1, gl.RGBA, gl.FLOAT, d, 0);
    const u = this.selU;
    gl.useProgram(this.selProg);
    gl.uniform1i(u.u_shapes, 10);
    gl.uniform1i(u.u_count, count);
    gl.uniform4f(u.u_bbox, x0, y0, x1, y1);
    gl.uniform2f(u.u_css, w, h);
    gl.uniform1f(u.u_dpr, dpr);
    gl.uniform1f(u.u_height, vp[1]);
    gl.uniform3fv(u.u_ink, this.ink);
    gl.uniform1f(u.u_outline, SEL_OUTLINE_PX);
    gl.uniform1f(u.u_glow, SEL_GLOW_PX);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.activeTexture(gl.TEXTURE0);
  }

  /** The connectors between dots zoomed out (T-313, Anita: "if the dots don't touch, we draw the
   * connector"): within each group of `groups` (game/complexView.ts `dotGroups`, the complex rule's
   * distances) the shortest joins linking all its dots, and each of the player's links (T-197)
   * between two groups from one linked station's dot to the other's. Each runs centre to centre;
   * where dots overlap it is hidden inside them. Rebuilt when the groups or the network change. */
  private syncGroups(n: NetworkBuffers, groups: StationGroups | null) {
    if (this.groupSrc.groups === groups && this.groupSrc.net === n) return;
    this.groupSrc = { groups, net: n };
    const S = n.stations, sh = n.stationShape;
    const mid = (i: number): [number, number] => [S[i * 3] + (sh?.[i * 4] ?? 0), S[i * 3 + 1] + (sh?.[i * 4 + 1] ?? 0)];
    const pairs: number[] = [];
    const groupOf = new Int32Array(Math.max(1, n.stationCount)).fill(-1);
    if (groups) {
      const index = new Map(n.stationIds.map((id, i) => [id, i]));
      groups.members.forEach((ids, k) => {
        const m = ids.map((id) => index.get(id)).filter((i): i is number => i !== undefined);
        for (const i of m) groupOf[i] = k;
        for (const [a, b] of spanningPairs(m.map(mid))) pairs.push(m[a], m[b]);
      });
    }
    const idx = n.stationLinkIdx;
    for (let k = 0; idx && k < idx.length / 2; k++) {
      const i = idx[k * 2], j = idx[k * 2 + 1];
      if (groupOf[i] < 0 || groupOf[i] !== groupOf[j]) pairs.push(i, j);
    }
    this.connPairs = Int32Array.from(pairs);
    const count = pairs.length / 2;
    const ce = (this.connEnds = new Float32Array(Math.max(1, count) * 4));
    for (let k = 0; k < count; k++) ce.set([...mid(pairs[k * 2]), ...mid(pairs[k * 2 + 1])], k * 4);
    // its own buffers and VAO, replaced with the connectors
    for (const r of this.connRes) r.del();
    this.connRes = [];
    this.connVao = null;
    this.connFlagBuf = null;
    if (this.gl && this.bridgeProg) {
      const gl = this.gl;
      const cv = (this.connVao = gl.createVertexArray()!);
      this.connRes.push({ del: () => gl.deleteVertexArray(cv) });
      gl.bindVertexArray(cv);
      this.attrib(this.bridgeProg, "a_ends", this.buffer(ce, this.connRes), 4, gl.FLOAT, 1);
      this.connFlagBuf = this.buffer(new Float32Array(Math.max(1, count) * 2), this.connRes);
      this.attrib(this.bridgeProg, "a_bflag", this.connFlagBuf, 2, gl.FLOAT, 1);
      gl.bindVertexArray(null);
    }
    this.selDrawnKey = "";
  }

  /** Which stations and bridges are drawn selected (`a_sel`, `a_bflag`), when that changed. */
  private syncSelection(n: NetworkBuffers) {
    const key = `${this.selectedStation}/${this.selectedAlsoKey}`;
    if (key === this.selDrawnKey) return;
    this.selDrawnKey = key;
    const flags = new Float32Array(Math.max(1, n.stationCount) * 2);
    for (let i = 0; i < n.stationCount; i++) flags[i * 2] = this.drawnSelected(i) ? 1 : 0;
    if (this.stationSelBuf) this.rewrite(this.stationSelBuf, flags);
    // a bridge or connector is selected with both its stations, and faded when either is a blueprint
    const pairFlags = (idx: ArrayLike<number>) => {
      const b = new Float32Array(Math.max(2, idx.length));
      for (let k = 0; k < idx.length / 2; k++) {
        const i = idx[k * 2], j = idx[k * 2 + 1];
        b[k * 2] = this.drawnSelected(i) && this.drawnSelected(j) ? 1 : 0;
        b[k * 2 + 1] = (Math.round(n.stations[i * 3 + 2]) | Math.round(n.stations[j * 3 + 2])) & 2 ? 1 : 0;
      }
      return b;
    };
    const idx = n.stationLinkIdx;
    if (idx?.length && this.bridgeFlagBuf) this.rewrite(this.bridgeFlagBuf, pairFlags(idx));
    if (this.connPairs.length && this.connFlagBuf) this.rewrite(this.connFlagBuf, pairFlags(this.connPairs));
  }

  /** The share of the room to stand that the platform side of queue `q` of the trip buffer holds,
   * as last drawn (0-1); null when crowds are not drawn or the queue has no platform. */
  queueShare(q: number): number | null {
    const o = this.crowdOf;
    if (!o || !this.crowdsShown || !this.net || o.net !== this.net || q < 0 || q >= o.station.length || o.station[q] < 0) return null;
    return this.crowds[o.station[q] * 4 + (o.side[q] > 0 ? 0 : 1)];
  }

  /** Zoomed out, the crowd station `i`'s dot shows as a pie (T-228, T-553): riders waiting at its platforms
   * and its fullest platform side's share of the room there; null as platforms or when crowds are not drawn. */
  dotCrowd(i: number): { waiting: number; share: number } | null {
    if (!this.crowdsShown || this.stationStyle.boxes || i < 0 || i >= this.dotWait.length) return null;
    return { waiting: this.dotWait[i], share: this.dotCrowds[i * 2] };
  }

  /** The riders waiting on each station's two sides at `t` s into the trip buffer's hour (T-164):
   * each platform's crowd is what the last train to leave it left plus who has arrived since; drawn
   * as a share of the room to stand on that side (T-219), the whole platform's when only one side
   * has trains. Worked out at most once a game second, only while drawn. False when nobody waits
   * anywhere. */
  private updateCrowds(n: NetworkBuffers, t: number): boolean {
    const b = this.board!;
    if (!this.crowdOf || this.crowdOf.board !== b || this.crowdOf.net !== n) this.crowdOf = this.placeCrowds(n, b);
    if (this.crowdAt.board === b && Math.abs(t - this.crowdAt.t) < 1) return this.crowdAny;
    const t0 = performance.now();
    const c = this.crowds, w = this.crowdWait;
    c.fill(0);
    w.fill(0);
    const { station, side, sides } = this.crowdOf;
    const Q = b.queues, E = b.events;
    let any = false;
    for (let q = 0; q < b.queueCount; q++) {
      const s = station[q];
      if (s < 0) continue;
      const base = q * QUEUE_STRIDE, first = Q[base + 8], count = Q[base + 9];
      if (count <= 0) continue;
      // the last train to leave at or before t (events are in time order)
      let lo = first, hi = first + count - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (E[mid * 2] <= t) lo = mid;
        else hi = mid - 1;
      }
      const left = E[lo * 2 + 1], waiting = left + Math.max(0, t - E[lo * 2]) * Q[base + 4] + pulsesBetween(b, q, E[lo * 2], t);
      const k = side[q] > 0 ? 0 : 1;
      w[s * 2 + k] += waiting;
      if (waiting >= 1) any = true;
    }
    for (let s = 0; s < n.stationCount; s++) {
      if (!sides[s]) continue;
      const room = platformRoom(n.stationLength?.[s] ?? 0);
      if (sides[s] === 3) {
        c[s * 4] = Math.min(1, w[s * 2] / room);
        c[s * 4 + 1] = Math.min(1, w[s * 2 + 1] / room);
        c[s * 4 + 2] = c[s * 4] >= 1 ? 1 : 0;
        c[s * 4 + 3] = c[s * 4 + 1] >= 1 ? 1 : 0;
      } else {
        // trains on one side only (a terminal): the whole platform is theirs
        const f = Math.min(1, (w[s * 2] + w[s * 2 + 1]) / (2 * room)), bad = f >= 1 ? 1 : 0;
        c.set([f, f, bad, bad], s * 4);
      }
    }
    if (this.crowdBuf) this.rewrite(this.crowdBuf, c);
    // each dot's (T-228): its station's platforms
    const dc = this.dotCrowds, dw = this.dotWait;
    dc.fill(0);
    dw.fill(0);
    for (let s = 0; s < n.stationCount; s++) {
      if (!sides[s]) continue;
      dc[s * 2] = Math.max(c[s * 4], c[s * 4 + 1]);
      if (c[s * 4 + 2] > 0 || c[s * 4 + 3] > 0) dc[s * 2 + 1] = 1; // a full side
      dw[s] = w[s * 2] + w[s * 2 + 1];
    }
    if (this.dotCrowdBuf) this.rewrite(this.dotCrowdBuf, dc);
    this.crowdAt = { t, board: b };
    this.crowdAny = any;
    this.crowdMs = performance.now() - t0;
    return any;
  }

  /** Each platform's station in the station buffer, and its side: + where its trains travel along
   * the station's drawn platform axis, - against it; and per station the sides with platforms. */
  private placeCrowds(n: NetworkBuffers, b: BoardHour) {
    const index = new Map<number, number>();
    n.stationIds.forEach((id, i) => index.set(Number(id), i));
    const station = new Int32Array(b.queueCount), side = new Int8Array(b.queueCount), sides = new Uint8Array(Math.max(1, n.stationCount));
    for (let q = 0; q < b.queueCount; q++) {
      const base = q * QUEUE_STRIDE;
      const s = index.get(b.queues[base]) ?? -1;
      station[q] = s;
      if (s < 0) continue;
      const line = b.queues[base + 1], run = b.queues[base + 2], d = b.queues[base + 3];
      const [ax, ay] = lineAt(n, line, d - 5), [bx, by] = lineAt(n, line, d + 5);
      const dot = ((bx - ax) * (n.stationAxis?.[s * 2] ?? 1) + (by - ay) * (n.stationAxis?.[s * 2 + 1] ?? 0)) * (run === 1 ? -1 : 1);
      side[q] = dot >= 0 ? 1 : -1;
      sides[s] |= side[q] > 0 ? 1 : 2;
    }
    return { station, side, sides, board: b, net: n };
  }

  private bindTrainTextures(u: Record<string, WebGLUniformLocation | null>) {
    const gl = this.gl;
    const bind = (unit: number, tex: WebGLTexture | null, loc: WebGLUniformLocation | null) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(3, this.tex.phases, u.u_phases);
    bind(4, this.tex.meta, u.u_meta);
    bind(0, this.tex.samples, u.u_samples);
    bind(1, this.tex.lines, u.u_lines);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.colours);
  }

  /**
   * GPU positions (local units, x/y pairs; 1e9 when off the track) of every trip in the buffer at
   * game time `time`, read back by transform feedback. For precision checks; changes GL state
   * freely, so only call it on a renderer with a context of its own.
   */
  probe(time: number): Float32Array {
    const gl = this.gl;
    if (!this.probeProg) {
      this.probeProg = compile(gl, PROBE_VS, PROBE_FS, ["v_pos"]);
      this.probeU = uniforms(gl, this.probeProg, ["u_time", "u_phases", "u_meta", "u_samples", "u_lines"]);
    }
    const uTime = this.sync(time);
    const n = this.tripCount;
    if (!this.probeVao) {
      this.probeVao = gl.createVertexArray()!;
      gl.bindVertexArray(this.probeVao);
      this.attrib(this.probeProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 0);
      this.attrib(this.probeProg, "a_end", this.endBuf!, 1, gl.FLOAT, 0);
      gl.bindVertexArray(null);
    }
    const out = this.buffer(new Float32Array(Math.max(1, n) * 2));
    gl.useProgram(this.probeProg);
    gl.uniform1f(this.probeU.u_time, uTime);
    this.bindTrainTextures(this.probeU);
    gl.bindVertexArray(this.probeVao);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, out);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindVertexArray(null);
    const res = new Float32Array(n * 2);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, out);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, res);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    return res;
  }

  private pollTimer() {
    const gl = this.gl;
    const q = this.pendingQuery;
    if (!q) return;
    if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) return;
    const disjoint = gl.getParameter(this.timerExt.GPU_DISJOINT_EXT);
    if (!disjoint) this.gpuMs = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
    gl.deleteQuery(q);
    this.pendingQuery = null;
  }
}
