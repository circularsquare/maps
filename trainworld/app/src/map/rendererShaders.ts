// The network renderer's shaders (map/renderer.ts): the strokes, stations and bridges, the selection's
// outline, trains and the position probe, with the constants they and the renderer share.

import { CRUSH_PER_CAR, SEATS_PER_CAR } from "../game/boarding";
import { rgb, SPEED_STOPS } from "../game/palette";
import { PSI_PER_UNIT } from "./geo";
import { PROJECT_GLSL } from "./gl";
import { PLATFORM_POINTS, SEL_GLOW_PX, SEL_OUTLINE_PX, SHAPE_GLSL } from "./stationGeometry";

export const TEX_W = 2048;
export const LINES_W = 1024;
/** Double track's tracks are each this share of the 4 m between their centres (T-658). */
export const LANE_SHARE = 0.6;
/** How much darker a line at capacity is drawn in the traffic view, towards black (T-1125). */
const FULL_DARKEN = "0.14";
/** RGBA texels per selected shape in the outline's texture (SEL_FS): its frame, size and dot, then
 * its centre line (T-254). */
export const SEL_STRIDE = 3 + Math.ceil(PLATFORM_POINTS / 2);
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
export const SEL_MAX = Math.floor(2048 / SEL_STRIDE); // texels in a row: 2048, the narrowest texture WebGL2 guarantees
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
export const HOT_STATIONS = 16;

export const STROKE_VS = `#version 300 es
precision highp float;
precision highp int;
${PROJECT_GLSL}
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
uniform int u_ghost_layer;   // T-1054: -1 the other strokes; 0 a ghost's outline, 1 its rails, 2 its middle
uniform vec3 u_ghost_rgb;    // the outline
uniform vec2 u_ghost_min;    // the ghost's least width across its rails (px), and its outline's both sides together
uniform vec3 u_hsr;          // T-1097: high-speed strokes' width as a multiple (one stroke, built), and the
                             // stroke widths (px) between which its centre line comes in
in vec4 a_seg;
in uint a_colour;
in uint a_edge;
in float a_level;
in float a_flags;
in float a_speed;
in float a_wmul;           // width as a multiple of the line width (riders, T-084); 1 otherwise
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
  // the ghost is drawn as blueprint track in three layers, outline then rails then middle (T-1054)
  bool ghost = u_ghost == 1 && placement;
  if (ghost ? u_ghost_layer < 0 : u_ghost_layer >= 0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
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
  bool bpShape = bpTrack || bpLine || ghost;
  float lt = (blueprint && !placement) || ghost ? u_lanes.w : u_width.z;
  float core = bpShape ? u_bp.x + 2.0 * u_bp.y : u_width.x;
  float grow = u_lanes.z * max(0.0, lfm * (G + lt) - core * (bpShape ? 1.0 : a_wmul));
  float extra = sel ? u_width.y : 0.0;
  bool full = u_full_pass == 1;
  if (full && !(u_kind == 1 && a_full > 0.5)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float cas = full ? u_casing + u_full_px : u_casing;
  float w, lane_off = 0.0;
  if (u_lane_mode == 0) {
    w = (ghost ? core
      : bpTrack ? u_bp.x + extra + (u_casing > 0.0 ? 2.0 * u_bp.y : 0.0)
      : bpLine ? u_bp.x + extra + cas
      : u_width.x * a_wmul * (hsr ? u_hsr.x : 1.0) + extra + cas) + grow;
  } else {
    // the gap opens from the middle while the pair's width stays: each track is half of what is
    // left; the band under both is the pair's width, or the traffic width behind them (Anita,
    // 2026-10-19)
    float gap = t * (G - lt);
    float wl = lfm > 0.0 ? 0.5 * (max(core, G + lt) - gap) : mix(core, lt, t);
    float half_ = 0.5 * (gap + wl);
    lane_off = half_;
    float band = max(core * (bpShape ? 1.0 : a_wmul), lfm * 2.0 * half_ + wl);
    bool casing = u_casing > 0.0;
    bool keep = u_lane_mode == 1
      ? !ghost && (casing ? !bpTrack : (u_kind == 1 && !blueprint) || (hsr && u_kind == 0))
      : (casing ? bpTrack : true) && (u_lane_mode == 2 || lfm > 0.0);
    if (!keep) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    w = u_lane_mode == 1 ? band + extra + cas
      : ghost ? wl
      : bpTrack || bpLine ? (casing ? wl : max(0.0, wl - 2.0 * u_bp.y))
      : wl;
    if (u_cap > 0.0 && u_lane_mode > 1) w = min(w, 0.5 * u_cap);
  }
  if (u_cap > 0.0) w = min(w, u_cap);
  if (ghost) {
    w = max(w, u_ghost_min.x);
    w = u_ghost_layer == 0 ? w + u_ghost_min.y : u_ghost_layer == 2 ? max(0.0, w - 2.0 * u_bp.y) : w;
  }
  // thinner than a pixel (zoomed far out, T-205): a pixel wide and fainter, rather than broken up
  float faint = clamp(w, 0.05, 1.0);
  w = max(w, 1.0);
  int id = gl_VertexID;
  float along = float(id & 1);
  float side = id < 2 ? -1.0 : 1.0;
  vec4 c0 = project(a_seg.xy);
  vec4 c1 = project(a_seg.zw);
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
  if (ghost && u_ghost_layer < 2) v_color = u_ghost_layer == 0 ? u_ghost_rgb : u_bp_rail;
  else if (u_casing > 0.0) v_color = bpTrack ? (swOff ? u_bp_off_rail : u_bp_rail) : placement && !ghost ? u_sel : full ? u_full_rgb : u_casing_rgb;
  else if ((flags & 16) != 0) v_color = u_bad;
  else if ((u_kind == 2 && !placement) || u_by_level == 1) v_color = lc;
  else if (u_by_speed == 1) {
    ${SPEED_BLEND}
  }
  else if (placement) v_color = ghost ? u_bp_mid : vec3(1.0);
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
  vec3 rail = bpTrack ? (swOff ? u_bp_off_rail : u_bp_rail) : ghost ? u_bp_rail : placement ? u_sel : u_casing_rgb;
  // T-1097: on a narrow built stroke no centre line, which would leave little of the colour
  float hs = u_lane_mode == 0 ? (bpShape || placement ? max(0.3, 0.18 * w) : 0.18 * w * smoothstep(u_hsr.y, u_hsr.z, w)) : max(0.3, 0.08 * G);
  if (hsr && u_casing <= 0.0 && u_lane_mode < 2 && !full && (!ghost || u_ghost_layer == 2) && hs >= 0.1) {
    v_stripe = vec4(u_lane_mode == 0 ? rail : (bpTrack || placement ? rail : v_color), hs);
    v_stripe_only = u_lane_mode == 1 && u_kind == 0 ? 1.0 : 0.0;
  }
  // T-1125: a line at capacity a little darker under its red edge
  if (u_kind == 1 && a_full > 0.5 && u_casing <= 0.0 && !full) v_color = mix(v_color, vec3(0.0), ${FULL_DARKEN});
  // the band behind a line's two tracks, pale so they stand out on it
  if (u_lane_mode == 1 && u_casing <= 0.0) v_color = mix(v_color, u_casing_rgb, 0.6 * t);
}`;

// highp: v_cap is device px along the segment, past fp16's 65504 on a long straight zoomed in on a
// phone, where mediump is 16-bit (T-953)
export const STROKE_FS = `#version 300 es
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
// drawn over each other make one white shape with one outline round it (T-217): every station's dark shape is drawn first (u_layer 0), then every station's white
// inside its rim (u_layer 1), so a rim inside another station is painted over; then the riders
// waiting on each platform (u_layer 2, T-219). Selected ones are drawn in a pass of their own, after
// the others (u_pass), over one outline and glow round all of them (SEL_FS). The same program draws
// the demand circles (u_halo 1).
export const STATION_VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
uniform vec2 u_viewport;
uniform vec2 u_radius;   // dot radius, ordinary and transfer (px)
uniform float u_line;    // a line's width (px): a station covers the lines through it
uniform float u_ring;    // its dark rim (px)
uniform float u_boxes;   // 1: platforms along the track; 0: dots (zoomed out)
uniform vec2 u_fixed;    // a box's half width (T-204, T-239): px at BOX_WIDTH_ZOOM, and local
                         // units per px there; the width on the ground, and the least on screen (T-553)
uniform highp int u_halo;
uniform highp int u_pass; // stations: 0 every one but the selected, 1 the selected only
uniform highp int u_layer; // 0 the dark shape, 1 the white inside the rim, 2 riders waiting (T-219)
uniform float u_halo_radius;
uniform float u_pixel;
in vec3 a_st;
in vec2 a_axis;          // half the platform along its track at its middle, local units
in vec4 a_shape;         // the platform's middle from the node (xy), half the turn along it (z, not read), half its true width (w), local units
in float a_sz;           // daily riders' square-root scale
in vec2 a_halo;          // demand circle's offset, local units (a station complex's, T-146)
in float a_sel;          // 1 drawn selected
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
    vec4 h = project(a_st.xy + a_halo);
    float r = u_halo_radius * a_sz;
    v_size = vec3(r, r, 0.0);
    v_p = corner * (r + 1.0);
    h.xy += v_p / half_vp * h.w;
    gl_Position = h;
    return;
  }
  bool boxes = u_boxes > 0.5;
  bool sel = a_sel > 0.5;
  bool noCrowd = !boxes || (flags & 2) != 0 || max(a_crowd.x, a_crowd.y) <= 0.0;
  if ((u_pass == 1) != sel || (u_layer == 2 && noCrowd)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  // zoomed out a dot's pie is its platforms' crowd (T-228, T-553): the fullest side's share, and
  // whether it is full
  if (!boxes) v_crowd = vec4(a_dcrowd, 0.0, 0.0);
  vec2 mid = a_st.xy + a_shape.xy;
  vec4 c = project(mid);
  vec4 e = project(mid + a_axis);
  vec4 q = project(mid + vec2(-a_axis.y, a_axis.x));
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
      vec4 h = project(a_st.xy + P);
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
export const BRIDGE_VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
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
  vec4 c0 = project(a_ends.xy), c1 = project(a_ends.zw);
  vec4 cx = project(a_ends.xy + vec2(u_bridge, 0.0));
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
  vec4 c = project(0.5 * (a_ends.xy + a_ends.zw));
  c.xy += (u * v_p.x + n * v_p.y) / half_vp * c.w;
  gl_Position = c;
}`;

export const STATION_FS = `#version 300 es
precision highp float;
uniform float u_ring;
uniform vec3 u_ink;
uniform vec3 u_bad;
uniform vec3 u_crowd_rgb;
uniform highp int u_halo;
uniform highp int u_layer;
uniform float u_tint;    // 1: dots show their crowds as pies (T-228, T-553)
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
  } else {
    // riders waiting (T-219): each side's half of the platform fills from its middle outwards, as
    // far as its share of the room to stand there, like a train's gauge; red once full (T-559);
    // right up to the rim (T-1121)
    vec2 p = f;
    float reach = v_size.x - u_ring;
    // close in on double track (T-658) each side's riders stand on their own platform, from its
    // track out to the rim: the + side's trains travel along the axis, keeping right, so their
    // track and platform are on the + side
    float tl = u_lanes.y * v_lane;
    float yin = tl * (0.5 + 0.5 * ${LANE_SHARE.toFixed(1)} * u_lanes.z) * u_lanes.x;
    float ap = inner * clamp(p.y - yin + 0.5, 0.0, 1.0) * clamp(v_crowd.x * reach - abs(p.x) + 0.5, 0.0, 1.0);
    float am = inner * clamp(0.5 - p.y - yin, 0.0, 1.0) * clamp(v_crowd.y * reach - abs(p.x) + 0.5, 0.0, 1.0);
    o = vec4((v_crowd.z > 0.5 ? u_bad : u_crowd_rgb) * ap + (v_crowd.w > 0.5 ? u_bad : u_crowd_rgb) * am, ap + am);
  }
}`;

// The selected stations' outline and glow (T-189), round their union (T-217): one quad over all of
// them, each pixel's distance the nearest shape's, so platforms and bridges drawn as one shape get
// one outline. The shapes are the CPU twins (stationGeometry.ts) in CSS px, 3 texels each.
export const SEL_VS = `#version 300 es
precision highp float;
uniform vec4 u_bbox;     // CSS px, y down: left, top, right, bottom
uniform vec2 u_css;      // the canvas, CSS px
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1), gl_VertexID < 2 ? 0.0 : 1.0);
  vec2 s = mix(u_bbox.xy, u_bbox.zw, corner);
  gl_Position = vec4(s.x / u_css.x * 2.0 - 1.0, 1.0 - s.y / u_css.y * 2.0, 0.0, 1.0);
}`;

export const SEL_FS = `#version 300 es
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

export const TRAIN_VS = `#version 300 es
precision highp float;
precision highp int;
${PROJECT_GLSL}
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
  vec4 c = project(p);
  vec4 cb = project(back);
  vec4 cf = project(front);
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
export const TRAIN_FS = `#version 300 es
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

export const PROBE_VS = `#version 300 es
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

export const PROBE_FS = `#version 300 es
precision mediump float;
out vec4 o;
void main() { o = vec4(0.0); }`;
