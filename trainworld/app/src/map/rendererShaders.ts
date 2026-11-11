// The network renderer's shaders (map/renderer.ts): the strokes, stations and bridges, the selection's
// outline, trains and the position probe, with the constants they and the renderer share.

import { BOAT_CARS, BUS_CARS, CRUSH_PER_CAR, RATED_PER_CAR, SEATS_PER_CAR } from "../game/boarding";
import { PROFIT_STOPS, rgb, SPEED_STOPS } from "../game/palette";
import { PSI_PER_UNIT } from "./geo";
import { BAND_M_RANGE, BAND_RANGE, BAND_SHARE } from "./lineBands";
import { PROJECT_GLSL } from "./gl";
import { TURNBACK_M } from "./terminalBays";
import { PASS_LEAD_M, PLATFORM_POINTS, SEL_GLOW_PX, SEL_OUTLINE_PX, SHAPE_GLSL, ST_BUILDING, ST_FLIP, ST_BUFFER, ST_PASSING, ST_ROOM, ST_TERMINAL, ST_TRAM, TERMINAL_CAP, TERMINAL_EAR } from "./stationGeometry";

/** T-2926: tram track and the lines over it as a share of a rail line's width; a tram stop's dot as a share of a station's */
const TRAM_WIDTH = 0.75, TRAM_DOT = 0.7;
/** T-3136 (Anita: "we should draw double track a bit thicker, and single a bit thinner"): rail track's
 * width as a share of a line's, single and double, so the two read apart zoomed out. Double eases back to 1
 * as its two tracks grow apart (`u_lanes.z`), and single with it, so close in nothing changes. */
const SINGLE_WIDTH = 0.71, DOUBLE_WIDTH = 1.24;
/** Zoomed far out every stroke is at least a pixel wide; double track's least is this many (device px), so
 * the two still differ where both would be at the floor. */
const DOUBLE_FLOOR_PX = 2.0;
/** T-2927 (Anita: "2 thin black lines in the middle ... to indicate the cables"): on tram strokes, each
 * wire's middle this share of the stroke's width either side of its centre while drawn as one stroke
 * (one wire along each track once they part), each this share of the width across, at least WIRE_MIN_PX */
const WIRE_OFF = 0.17, WIRE_SHARE = 0.11, WIRE_MIN_PX = 0.7;

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
/** A ramp's colour of `v` as GLSL: `stops` blended in order, so the shader cannot drift from the tags' colours. */
const rampBlend = (s: [number, string][], v: string) => {
  const v3 = (hex: string) => `vec3(${rgb(hex).map((c) => (c / 255).toFixed(4)).join(", ")})`;
  return [`v_color = ${v3(s[0][1])};`, ...s.slice(1).map(([at, hex], i) =>
    `v_color = mix(v_color, ${v3(hex)}, clamp((${v} - ${s[i][0].toFixed(4)}) / ${(at - s[i][0]).toFixed(4)}, 0.0, 1.0));`)].join("\n    ");
};
/** The Max speed colour (T-843). */
const SPEED_BLEND = rampBlend(SPEED_STOPS, "a_speed");
/** T-2420: the Profit view's colour, by the edge's stretch's fares over costs (`u_profit`). */
const PROFIT_BLEND = rampBlend(PROFIT_STOPS, "pv");
/** T-2236: a crowded stretch's red edge and darkening (the traffic view) as a share of track at
 * capacity's, by its busiest hour's load as a share of the rated load: FULL_FLOOR just past
 * CROWD_LOAD_FROM, all of it at CROWD_LOAD_STRONG, crush. */
export const FULL_FLOOR = 0.5, CROWD_LOAD_FROM = 1, CROWD_LOAD_STRONG = 1.5;
/** T-1663: the red band is this share of the line's own width on top of the zoom-scaled base, so it reads on thin and thick lines alike. */
const FULL_WIDTH_SHARE = "0.15";
const HOT_RIM_MIX = "0.6";
export const HOT_STATIONS = 16;
/** T-1812: lines a selected group can draw selected */
export const SEL_SET_MAX = 16;
/** whether a line is the selected one or in the selected group (`u_sel_line`, `u_sel_set`) */
const SEL_GLSL = `bool selectedLine(int c) {
  if (c == u_sel_line) return true;
  for (int i = 0; i < ${SEL_SET_MAX}; i++) {
    if (i >= u_sel_n) break;
    if (u_sel_set[i] == c) return true;
  }
  return false;
}`;

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
uniform int u_sel_n;         // T-1812: more lines drawn selected, a selected group's: how many, and which
uniform int u_sel_set[${SEL_SET_MAX}];
uniform int u_sel_edge;
uniform ivec2 u_hot;         // the hovered line and bare track edge, -1 none (T-730)
uniform ivec2 u_hide;        // T-2763: two edges just removed, not drawn before the state without them lands, -2 none
uniform int u_only_sel;
uniform int u_bp_part;       // T-695: 0 every stroke, 1 all but blueprint, 2 blueprint only
uniform vec4 u_lanes;        // T-658: 4 m in px, how far the tracks have parted (0-1), how far the
                             // stroke has grown to their width (0-1), a blueprint track's width apart
uniform int u_lane_mode;     // 0 one stroke, 1 the band under both tracks, 2 the right track, 3 the left
uniform float u_pass_off;    // T-2233: px from the centre line to a passing station's outer track, 0 not drawn
uniform float u_dim;        // line strokes other than the selected line's: share of full strength (T-233)
uniform vec2 u_mute;        // T-2058 prototype: those strokes' width share (0 as built) and colour mixed to the casing's
uniform int u_by_level;
uniform int u_by_speed;
uniform int u_by_profit;      // T-2420: coloured by u_profit, per edge id (TEX_W wide), u_profit_n of them
uniform highp sampler2D u_profit;
uniform int u_profit_n;
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
uniform float u_full_px;     // casing under the lines on track at capacity, nothing else; 2 (T-1284) the
uniform float u_full_out;    // casing's colour this much (px) wider again, and the red band in dashes
uniform float u_full_dash;   // of this period (px), so it reads on a red line; a crowded band's red edges (T-2184) are drawn with its colour pass
uniform vec3 u_full_rgb;
uniform int u_ghost;         // T-1038: 1 for the preview pass, whose placement strokes are the ghost
uniform int u_ghost_layer;   // T-1054: -1 the other strokes; 0 a ghost's outline, 1 its rails, 2 its middle
uniform vec3 u_ghost_rgb;    // the outline
uniform vec2 u_ghost_min;    // the ghost's least width across its rails (px), and its outline's both sides together
uniform vec3 u_hsr;          // T-1097: high-speed strokes' width as a multiple (one stroke, built), and the
                             // stroke widths (px) between which its centre line comes in (a tram's wires too)
uniform vec3 u_wire;         // T-2927: a tram's wires on built track and lines (T-3006: white)
uniform float u_band;        // T-1799: how far lines sharing track are drawn side by side (0-1)
uniform int u_band_traffic;  // 1: the Traffic view's bands, each its riders' width (a_wmul.yz)
uniform float u_band_gap;    // T-2126: px of casing between bands side by side
uniform float u_band_m;     // T-2132: px a metre of ground is drawn, for a band's metres to its ribbon's middle
in vec4 a_seg;
in uint a_colour;
in uint a_edge;
in float a_level;
in float a_flags;
in float a_speed;
in vec3 a_wmul;            // width as a multiple of the line width (riders, T-084), 1 otherwise; the
                           // Traffic view's band shift at its start and end, line widths left (T-1799)
in vec2 a_lane;             // T-658: how far apart its two tracks are at its start and end, 0-1
in vec2 a_full;             // T-819, T-1160: in the traffic view, the line segment's mark: below 0 track at capacity, above 0 its busiest hour's load over the rated load (T-2236), 0 none; and local units along its run to its start (T-2184)
in vec4 a_band;             // T-1799: its band's shift left at its start and end (share of ${BAND_RANGE} line widths), 1 - its width share at both
in vec2 a_band_m;           // T-2132: metres left from its track to its ribbon's middle at its start and end (share of ${BAND_M_RANGE})
out vec3 v_color;
out float v_alpha;
out vec4 v_cap;             // px: along from the segment's start, along to its end, across, half the width
flat out vec4 v_stripe;     // T-817: high-speed track's centre line, its colour and half width (px), 0 none
flat out float v_stripe_only; // 1: only that line is drawn (bare track between its parted tracks)
flat out vec4 v_wire;       // T-2927: a tram's wires, their colour and how far each one's middle is from the centre (px)
flat out float v_wire_hw;   // and each one's half width (px), 0 none
flat out float v_dash;      // T-1284: the red band's dash period (px), 0 for every other stroke
out float v_along;          // T-2184: px along the stroke's run, so dashes keep one length round curves
flat out vec4 v_inner;      // T-2184: a band's red edges over its neighbours: its colour's half width, the red's and the rim's px, strength; 0 none
flat out float v_aa;        // T-2183: a band's colour's half width (px), its sides smoothed so the gap between bands keeps its width; 0 none
${SEL_GLSL}
void main() {
  v_inner = vec4(0.0);
  v_aa = 0.0;
  int flags = int(a_flags + 0.5);
  bool hsr = (flags & 256) != 0;
  bool tram = (flags & 4096) != 0;
  bool striped = hsr;
  v_stripe = vec4(0.0);
  v_stripe_only = 0.0;
  v_wire = vec4(0.0);
  v_wire_hw = 0.0;
  v_dash = u_full_pass == 1 ? u_full_dash : 0.0;
  if (int(a_edge) == u_hide.x || int(a_edge) == u_hide.y) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  bool sel = (u_kind == 1 && selectedLine(int(a_colour))) || (u_kind == 0 && int(a_edge) == u_sel_edge);
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
  // a passing station's outer track (T-2233): one track, its lane how far out along its bow
  bool passing = (flags & 1024) != 0;
  if (passing && (u_pass_off <= 0.0 || u_casing > 0.0 || u_lane_mode == 1 || u_lane_mode == 3)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float lfm = turnback || (passing && u_lane_mode == 2) ? 1.0 : passing ? 0.0 : max(a_lane.x, a_lane.y), G = u_lanes.x, t = u_lanes.y;
  bool bpShape = bpTrack || bpLine || ghost;
  float lt = (blueprint && !placement) || ghost ? u_lanes.w : u_width.z * (tram ? ${TRAM_WIDTH.toFixed(2)} : 1.0);
  float tw = tram ? ${TRAM_WIDTH.toFixed(2)} : 1.0;
  float tk = tram || turnback || passing ? 1.0 : mix(mix(${SINGLE_WIDTH.toFixed(2)}, ${DOUBLE_WIDTH.toFixed(2)}, lfm), 1.0, u_lanes.z);
  float core = bpShape ? u_bp.x * tw * tk + 2.0 * u_bp.y : u_width.x * tw * tk;
  int id = gl_VertexID;
  float along = float(id & 1);
  // lines sharing track side by side (T-1799): each a band narrower than a line, shifted across;
  // every casing is drawn before any colour, so the bands get one outline round them all. In the
  // Traffic view each band is its riders' width (a_wmul.yz the shifts); through a station's box
  // (u_cap) every view takes the Line view's slots, a line wide each before the cap
  float bandOn = u_kind == 1 ? u_band : 0.0;
  bool shared = bandOn > 0.0 && max(a_band.z, a_band.w) > 0.0;
  bool tBand = u_band_traffic == 1 && u_cap <= 0.0;
  float wmul = shared && u_cap > 0.0 ? 1.0 : a_wmul.x;
  float bandW = tBand ? 1.0 : 1.0 - bandOn * mix(a_band.z, a_band.w, along);
  float grow = u_lanes.z * max(0.0, lfm * (G + lt) - core * (bpShape ? 1.0 : wmul));
  float extra = sel ? u_width.y : 0.0;
  bool muted = u_kind == 1 && u_mute.x > 0.0 && !sel;
  if (muted) bandW *= u_mute.x;
  bool full = u_full_pass == 1 || u_full_pass == 2;
  bool fullOut = u_full_pass == 2;
  // a crowded stretch on a band of shared track: its red edges drawn after every
  // line's colour (pass 3), over the bands beside it, since under the ribbon's casing they would not show
  bool waitIn = shared && tBand && a_full.x > 0.0;
  // (T-2347: drawn with the line's colour, in the line's own order, so no line's edges lie over a line above it)
  bool late = waitIn && u_kind == 1 && u_casing <= 0.0 && u_only_sel == 0 && u_full_pass == 0;
  if (late) v_dash = u_full_dash;
  if (full && (!(u_kind == 1 && a_full.x != 0.0) || waitIn)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float fs = a_full.x < 0.0 ? 1.0 : a_full.x > 0.0 ? ${FULL_FLOOR.toFixed(3)} + ${(1 - FULL_FLOOR).toFixed(3)} * clamp((a_full.x - ${CROWD_LOAD_FROM.toFixed(3)}) / ${(CROWD_LOAD_STRONG - CROWD_LOAD_FROM).toFixed(3)}, 0.0, 1.0) : 0.0;
  float cas = full ? u_casing + u_full_px + ${FULL_WIDTH_SHARE} * u_width.x * wmul + (fullOut ? u_full_out : 0.0) : u_casing;
  float w, lane_off = 0.0;
  if (u_lane_mode == 0) {
    w = (ghost ? core
      : bpTrack ? u_bp.x * tw * tk + extra + (u_casing > 0.0 ? 2.0 * u_bp.y : 0.0)
      : bpLine ? u_bp.x * tw * tk * bandW + extra + cas
      : u_width.x * wmul * tk * (hsr ? u_hsr.x : 1.0) * (tram ? ${TRAM_WIDTH.toFixed(2)} : 1.0) * bandW + extra + cas) + grow;
    // T-2126: a band of shared track a pixel narrower in colour than its slot, the casing showing between
    // (none for a line easing out of the ribbon to its full width)
    if (shared && u_casing <= 0.0 && !full) {
      w = max(1.0, w - u_band_gap * clamp(mix(a_band.z, a_band.w, along) / ${(1 - BAND_SHARE).toFixed(2)}, 0.0, 1.0));
    }
  } else {
    // the gap opens from the middle while the pair's width stays: each track is half of what is
    // left; the band under both is the pair's width, or the traffic width behind them (Anita,
    // 2026-10-19)
    float gap = t * (G - lt);
    float wl = lfm > 0.0 ? 0.5 * (max(core, G + lt) - gap) : mix(core, lt, t);
    float half_ = 0.5 * (gap + wl);
    lane_off = half_;
    float band = max(core * (bpShape ? 1.0 : wmul), lfm * 2.0 * half_ + wl);
    bool casing = u_casing > 0.0;
    bool keep = u_lane_mode == 1
      ? !ghost && (casing ? !bpTrack : (u_kind == 1 && !blueprint) || (striped && u_kind == 0))
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
  float floorPx = 1.0;
  if (u_kind == 0 && !(tram || turnback || passing)) {
    floorPx = mix(1.0, ${DOUBLE_FLOOR_PX.toFixed(1)}, lfm * (1.0 - u_lanes.z));
    // a blueprint's middle and a ghost's inner layer sit between rails of their own
    if ((bpShape && u_casing <= 0.0 && !ghost) || (ghost && u_ghost_layer == 2)) floorPx = max(1.0, floorPx - 2.0 * u_bp.y);
  }
  w = max(w, floorPx);
  if (late) {
    float hr = 0.5 * (u_full_px + ${FULL_WIDTH_SHARE} * w), ho = 0.5 * u_full_out;
    v_inner = vec4(0.5 * w, hr, ho, fs);
    w += 2.0 * (hr + ho);
  } else if (shared && u_casing <= 0.0 && !full && u_lane_mode == 0) {
    // (half a pixel more each side, for the smoothed edge)
    v_aa = 0.5 * w;
    w += 1.0;
  }
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
  if (passing) off += nrm * ((flags & 2048) != 0 ? 1.0 : -1.0) * mix(u_lane_mode == 2 ? lane_off : 0.0, u_pass_off, lv);
  else if (u_lane_mode > 1) off -= nrm * (u_lane_mode == 2 ? 1.0 : -1.0) * lv * lane_off;
  // the slot in line widths from the ribbon's middle, and the middle in metres from the track, each
  // at the zoom in view (T-2132); through a station's box (u_cap) the bands close up as they narrow to fit
  if (bandOn > 0.0) {
    float toMid = mix(a_band_m.x, a_band_m.y, along) * ${BAND_M_RANGE.toFixed(1)} * u_band_m;
    off += nrm * bandOn * (tBand ? u_width.x * mix(a_wmul.y, a_wmul.z, along) + toMid
      : (u_width.x * mix(a_band.x, a_band.y, along) * ${BAND_RANGE.toFixed(1)} + toMid) * (u_cap > 0.0 ? min(1.0, u_cap / (${BAND_SHARE.toFixed(2)} * u_width.x)) : 1.0));
  }
  c.xy += off / half_vp * c.w;
  gl_Position = c;
  // round caps (T-673): the quad runs half a width past each end; the fragment shader rounds it
  float ext = (along * 2.0 - 1.0) * w * 0.5, ua = along * len + ext;
  v_cap = vec4(ua, len - ua, side * w * 0.5, w * 0.5);
  v_along = a_full.y * len / max(length(a_seg.zw - a_seg.xy), 1e-9) + ua;
  // Blueprint is solid but lighter (T-311): the colour mixed towards white, opaque, because
  // segments overlap at every joint and see-through blueprint would show a bead there. Lines
  // that do not run (faded) stay see-through.
  bool bp = blueprint && !placement;
  v_alpha = (!placement && !bp && (flags & 2) != 0 ? 0.55 : 1.0) * faint * (u_kind == 1 && !sel ? u_dim : 1.0);
  // the height at this end (levelCode in network.ts), its colour blended between the whole levels either side
  float lq0 = floor(a_level / 256.0);
  float lh = clamp(3.0 - ((along < 0.5 ? lq0 : a_level - lq0 * 256.0) / 32.0 - 4.0), 0.0, 6.0);
  int lo = int(floor(lh));
  vec3 lc = mix(u_level_colours[lo], u_level_colours[min(lo + 1, 6)], lh - float(lo));
  bool bpMid = false;
  if (ghost && u_ghost_layer < 2) v_color = u_ghost_layer == 0 ? u_ghost_rgb : u_bp_rail;
  // T-1152: a weaker edge towards the casing's colour, not see-through, which would bead at the joints
  else if (u_casing > 0.0) v_color = bpTrack ? (swOff ? u_bp_off_rail : u_bp_rail) : placement && !ghost ? u_sel : fullOut ? u_casing_rgb : full ? mix(u_casing_rgb, u_full_rgb, fs) : u_casing_rgb;
  else if ((flags & 16) != 0) v_color = u_bad;
  else if ((u_kind == 2 && !placement) || u_by_level == 1) v_color = lc;
  else if (u_by_speed == 1) {
    ${SPEED_BLEND}
  }
  // T-3130: a ghost has no profit (its edge id, all ones, read as -1 found the texture's zero: deep red)
  else if (u_by_profit == 1 && !placement) {
    int ei = int(a_edge);
    float pv = ei < u_profit_n ? texelFetch(u_profit, ivec2(ei % ${TEX_W}, ei / ${TEX_W}), 0).x : 1e30;
    if (pv > 1e29) v_color = u_track;
    else {
      ${PROFIT_BLEND}
    }
  }
  else if (placement) v_color = ghost ? u_bp_mid : vec3(1.0);
  else if (u_kind == 0 && bpTrack && !sel) { v_color = swOff ? u_bp_off_mid : u_bp_mid; bpMid = true; }
  else if (u_kind == 0) v_color = sel ? u_sel : u_track;
  else v_color = texelFetch(u_line_colours, ivec2(int(a_colour) % ${LINES_W}, int(a_colour) / ${LINES_W}), 0).rgb;
  // other blueprint colours (a planned line, a selected blueprint, by height or speed) lighter
  if (bp && !bpMid && u_casing <= 0.0) v_color = swOff ? mix(v_color, u_bp_off_fade, 0.8) : mix(v_color, vec3(1.0), 0.45);
  bool hot = (u_kind == 1 && int(a_colour) == u_hot.x) || (u_kind == 0 && int(a_edge) == u_hot.y);
  if (hot && u_casing <= 0.0) v_color = mix(v_color, vec3(1.0), ${HOT_LINE_MIX});
  if (muted && u_casing <= 0.0) v_color = mix(v_color, u_casing_rgb, u_mute.y);
  // high-speed track (T-817): a thin centre line along the stroke, light on built track (dark in
  // the dark theme), the rail colour on blueprint; once its tracks part, the stroke's own colour
  // between them
  vec3 rail = bpTrack ? (swOff ? u_bp_off_rail : u_bp_rail) : ghost ? u_bp_rail : placement ? u_sel : u_casing_rgb;
  // T-1097: on a narrow built stroke no centre line, which would leave little of the colour
  float hs = u_lane_mode == 0 ? (bpShape || placement ? max(0.3, 0.18 * w) : 0.18 * w * smoothstep(u_hsr.y, u_hsr.z, w)) : max(0.3, 0.08 * G);
  if (striped && u_casing <= 0.0 && u_lane_mode < 2 && !full && (!ghost || u_ghost_layer == 2) && hs >= 0.1) {
    v_stripe = vec4(u_lane_mode == 0 ? rail : (bpTrack || placement ? rail : v_color), hs);
    v_stripe_only = u_lane_mode == 1 && u_kind == 0 ? 1.0 : 0.0;
  }
  // T-2927: a tram's two wires, coming in as high-speed track's centre line does; one along each track
  // once they part (blueprint's only then: between its rails there is no room for two more lines)
  if (tram && u_casing <= 0.0 && u_lane_mode != 1 && !(bpShape && u_lane_mode == 0) && !full && !late && (!ghost || u_ghost_layer == 2)) {
    float hw = max(${(WIRE_MIN_PX / 2).toFixed(2)}, ${(WIRE_SHARE / 2).toFixed(3)} * w) * (u_lane_mode > 1 ? 1.0 : smoothstep(u_hsr.y, u_hsr.z, w));
    if (hw >= 0.1) {
      v_wire = vec4(bpShape || placement ? rail : u_wire, u_lane_mode == 0 ? ${WIRE_OFF.toFixed(3)} * w : 0.0);
      v_wire_hw = hw;
    }
  }
  // T-1125: a line at capacity a little darker under its red edge, as strong as the edge (T-1152)
  if (u_kind == 1 && a_full.x != 0.0 && u_casing <= 0.0 && !full) v_color = mix(v_color, vec3(0.0), ${FULL_DARKEN} * fs);
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
flat in vec4 v_wire;
flat in float v_wire_hw;
flat in float v_dash;
in float v_along;
flat in vec4 v_inner;
flat in float v_aa;
uniform vec3 u_full_rgb;
uniform vec3 u_casing_rgb;
uniform float u_full_out;
out vec4 o;
void main() {
  float past = -min(v_cap.x, v_cap.y);
  // T-2184, T-2347: a crowded band's quad is its colour's width plus its red edges (v_inner.x the colour's half width)
  bool edge = v_inner.x > 0.0 && abs(v_cap.z) >= v_inner.x;
  float hw = v_inner.x > 0.0 ? v_inner.x : v_cap.w;
  float a = past > 0.0 ? clamp(0.5 - (length(vec2(past, v_cap.z)) - hw), 0.0, 1.0) * v_alpha : v_alpha;
  if (v_dash > 0.0 && (v_inner.x <= 0.0 || edge) && mod(v_along, v_dash) > 0.6 * v_dash) discard;
  vec3 c = v_color;
  // the red edges, as a line's own outside its casing, in a rim of the casing's colour
  if (edge) {
    // (no round caps: at a run's end, where the band's width changes, they would ring it)
    float d = abs(v_cap.z);
    if (past > 0.0) discard;
    a = v_alpha;
    c = mix(u_casing_rgb, u_full_rgb, v_inner.w * clamp(v_inner.x + v_inner.y + 0.5 - d, 0.0, 1.0));
    a *= clamp(v_inner.x + v_inner.y + v_inner.z + 0.5 - d, 0.0, 1.0);
  }
  if (v_aa > 0.0) {
    // T-2223: the cap as round as the sides, not half a pixel fatter (it pinched the gap at every
    // joint), and a sharp ramp, so a gap falling between two pixels still leaves them 70% casing
    float d = v_aa + 0.5 - (past > 0.0 ? length(vec2(past, v_cap.z)) : abs(v_cap.z));
    a = v_alpha * clamp(1.6 * d - 0.5, 0.0, 1.0);
  }
  if (v_stripe.w > 0.0) {
    float s = clamp(v_stripe.w + 0.5 - abs(v_cap.z), 0.0, 1.0);
    c = mix(c, v_stripe.rgb, s);
    if (v_stripe_only > 0.5) { c = v_stripe.rgb; a *= s; }
  }
  if (v_wire_hw > 0.0) c = mix(c, v_wire.rgb, clamp(v_wire_hw + 0.5 - abs(abs(v_cap.z) - v_wire.w), 0.0, 1.0));
  o = vec4(c * a, a);
}`;

// Stations (T-144, T-189, T-217): a platform is a band along its track at its true length, bent as
// the track bends, white with a dark rim; zoomed out a dot, one a station (T-313). The quad is laid
// out in the platform's frame at its middle (along the track there, across towards the side it
// bends to) and the fragment shader measures the distance from the band (SHAPE_GLSL). Stations
// drawn over each other make one white shape with one outline round it (T-217): every station's dark shape is drawn first (u_layer 0), then every station's white
// inside its rim (u_layer 1), so a rim inside another station is painted over; then the riders
// waiting on each platform (u_layer 2, T-219). Selected ones are drawn in a pass of their own, after
// the others (u_pass), over one outline and glow round all of them (SEL_FS).
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
uniform float u_pass_half; // a box with passing tracks is at least this wide either side, px (T-2233)
uniform highp int u_pass; // stations: 0 every one but the selected, 1 the selected only
uniform highp int u_layer; // 0 the dark shape, 1 the white inside the rim, 2 riders waiting (T-219)
uniform highp int u_room; // 1: the waiting rooms joined to the boxes instead (T-1416)
uniform highp int u_bld;  // T-2219: 1 only stations in no station building, 2 only those in one, 0 all
uniform float u_pixel;
in vec3 a_st;
in vec2 a_axis;          // half the platform along its track at its middle, local units
in vec4 a_shape;         // the platform's middle from the node (xy), half the turn along it (z, not read), half its true width (w), local units
in float a_sel;          // 1 drawn selected
in vec4 a_crowd;         // riders waiting (T-219): how full the platform's + and - side is (0-1),
                         // and whether that side is full (zw)
in vec2 a_dcrowd;        // zoomed out (T-228): the station's fullest platform side, 0-1, and 1 if
                         // it is full
uniform highp sampler2D u_stline; // RG32F, per station ${PLATFORM_POINTS} texels: its track's centre line from the node (T-254)
out vec2 v_p;            // in the platform's frame at its middle, px
flat out vec3 v_size;    // half length, half width; 1 a waiting room, 3 a terminal
flat out float v_dot;    // the dot's radius, 0 for a platform
flat out float v_alpha;
flat out vec4 v_crowd;
flat out vec4 v_scr;     // a dot's frame on the screen (T-553): where its x and y axes point, as
                         // (x.x, y.x, x.y, y.y) with the screen's y up, so a pie can start at 12 o'clock
flat out float v_lane;   // 1 on double track: a platform each side of its two tracks close in (T-658); 2 with passing tracks (T-2249)
uniform highp int u_hot_st[${HOT_STATIONS}]; // the hovered stations (T-730), u_hot_n of them
uniform highp int u_hot_n;
flat out float v_hot;
// the centre line in the frame, px (T-254), two points to a vec4
${qVars("out")}
// the point of the centre line t along it from its middle (stationGeometry.ts centreAt)
vec2 centreAt(vec2 q[${PLATFORM_POINTS}], float t) {
  float total = 0.0;
  for (int i = 0; i < ${PLATFORM_POINTS - 1}; i++) total += length(q[i + 1] - q[i]);
  float step = total / ${(PLATFORM_POINTS - 1).toFixed(1)};
  if (step < 1e-6) return q[${PLATFORM_POINTS >> 1}];
  float u = t / step + ${((PLATFORM_POINTS - 1) / 2).toFixed(1)};
  int f = int(clamp(floor(u), 0.0, ${(PLATFORM_POINTS - 2).toFixed(1)}));
  return mix(q[f], q[f + 1], u - float(f));
}
void main() {
  v_hot = 0.0;
  for (int k = 0; k < ${HOT_STATIONS}; k++) if (k < u_hot_n && u_hot_st[k] == gl_InstanceID) v_hot = 1.0;
  v_dot = 0.0;
  v_scr = vec4(1.0, 0.0, 0.0, 1.0);
  v_crowd = a_crowd;
  int flags = int(a_st.z + 0.5);
  // 2: double track with passing tracks (T-2249)
  v_lane = (flags & 16) != 0 ? ((flags & ${ST_PASSING}) != 0 ? 2.0 : 1.0) : 0.0;
  v_alpha = (flags & 8) != 0 ? 0.3 : (flags & 2) != 0 ? 0.6 : 1.0; // 8: a blueprint switched off (T-333)
  vec2 half_vp = u_viewport * 0.5;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  bool boxes = u_boxes > 0.5;
  bool sel = a_sel > 0.5;
  bool noCrowd = !boxes || (flags & 2) != 0 || max(a_crowd.x, a_crowd.y) <= 0.0;
  bool noAnnex = u_room > 0 && (!boxes || (flags & ${ST_ROOM}) == 0);
  bool inBld = (flags & ${ST_BUILDING}) != 0;
  if ((u_pass == 1) != sel || (u_layer == 2 && noCrowd) || noAnnex || (u_bld == 1 && inBld) || (u_bld == 2 && !inBld)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
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
  // wide enough for its outer tracks' trains (T-2233)
  if (boxes && (flags & ${ST_PASSING}) != 0) hw = max(hw, u_pass_half);
  // a dot covers a plain line through it as drawn now
  float cover = 0.3 * u_line + u_ring + u_pixel;
  float hl = max(len, hw);
  // zoomed out every station is its own dot at its platform's middle (T-313)
  if (!boxes) v_dot = max(((flags & 1) != 0 ? u_radius.y : u_radius.x) * ((flags & ${ST_TRAM}) != 0 ? ${TRAM_DOT.toFixed(2)} : 1.0), cover);
  float lo, hi, xl, xh;
  // the quad's offset from the platform's middle and its axes on screen (a waiting room's own)
  vec2 shift = vec2(0.0), ra = along, rn = across;
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
    hl = max(0.5 * total, hw);
    // never shorter than it is wide: the end stretches carry on
    float ext = hl - 0.5 * total;
    v_size = vec3(hl, hw, 0.0);
    lo = mn.y - hw - ext - 1.0;
    hi = mx.y + hw + ext + 1.0;
    xl = mn.x - hw - ext - 1.0;
    xh = mx.x + hw + ext + 1.0;
    // a terminal's buffer stop end is drawn thick (T-1416, T-1527, the fragment shader)
    if ((flags & ${ST_TERMINAL}) != 0 && (flags & ${ST_BUFFER}) != 0) {
      v_size.z = 3.0;
      lo -= ${TERMINAL_EAR.toFixed(2)} * u_ring;
      hi += ${TERMINAL_EAR.toFixed(2)} * u_ring;
    }
    if (u_room > 0) {
      // T-1416: the waiting room, joined to the box's side at its middle, its inner edge inside the
      // box's rim so the two are one white shape with one outline, on the side facilityPlace.ts
      // picked (T-1528). As stationGeometry.ts stationAnnexes. Its riders stand in it from its
      // middle out (T-219's fill).
      vec2 qm = q[${PLATFORM_POINTS >> 1}];
      vec2 dl = q[${(PLATFORM_POINTS >> 1) + 1}] - q[${(PLATFORM_POINTS >> 1) - 1}];
      dl = length(dl) > 1e-6 ? normalize(dl) : vec2(1.0, 0.0);
      vec2 nl = vec2(-dl.y, dl.x);
      if (nl.y * ((flags & ${ST_FLIP}) != 0 ? 1.0 : -1.0) < 0.0) nl = -nl;
      float l = clamp(0.35 * hl, 1.2 * hw, 2.5 * hw), d = hw, at = 0.0;
      float edge = min(min(dot(centreAt(q, at - l) - qm, nl), dot(centreAt(q, at) - qm, nl)), dot(centreAt(q, at + l) - qm, nl));
      vec2 rc = qm + dl * at + nl * (edge + hw - 2.0 * u_ring - u_pixel + d);
      shift = along * rc.x + across * rc.y;
      ra = along * dl.x + across * dl.y;
      rn = along * nl.x + across * nl.y;
      v_scr = vec4(ra.x, rn.x, ra.y, rn.y);
      for (int i = 0; i < ${PLATFORM_POINTS}; i++) q[i] = vec2(-l + 2.0 * l * float(i) / ${(PLATFORM_POINTS - 1).toFixed(1)}, 0.0);
      v_size = vec3(l, d, float(u_room));
      lo = -d - 1.0;
      hi = d + 1.0;
      xl = -l - 1.0;
      xh = l + 1.0;
      v_lane = 0.0;
      v_crowd = vec4(vec2(max(a_crowd.x, a_crowd.y)), vec2(max(a_crowd.z, a_crowd.w)));
    }
    ${qPack}
  }
  v_p = vec2(corner.x < 0.0 ? xl : xh, corner.y < 0.0 ? lo : hi);
  c.xy += (shift + ra * v_p.x + rn * v_p.y) / half_vp * c.w;
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
uniform highp int u_layer;
uniform float u_tint;    // 1: dots show their crowds as pies (T-228, T-553)
uniform float u_mask;    // 1: the white only marks the stencil (the track drawn through it, T-236)
uniform vec3 u_lanes;    // T-658: 4 m in px, how far double track's two tracks have parted (0-1), their narrowing close in (T-713)
uniform float u_pass_in; // T-2249: px from the centre line to a passing station's outer track's inner edge, 0 none drawn
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
// blueprint track's colours (renderer.ts BP_RAIL_RGB, BP_MID_RGB); a blueprint station is drawn at alpha 0.6
#define BP_RIM vec3(0.184, 0.384, 0.702)
#define BP_FILL vec3(0.663, 0.804, 0.953)
void main() {
  bool bp = v_alpha > 0.45 && v_alpha < 0.99;
  // along and across the platform's own centre line (T-254); a dot's own frame
  ${qUnpack("v_q")}
  vec2 f = v_dot > 0.0 ? v_p : polyFrame(v_p, q);
  float d = v_dot > 0.0 ? length(v_p) - v_dot : boxSdf(f, v_size.xy);
  float inner = clamp(0.5 - d - u_ring, 0.0, 1.0);
  // T-1416, T-1527: a terminal's buffer stop, the platform's -axis end where the track ends, is a
  // dark bar wider than the box so it shows past a train standing in it; v_size.z 1 a waiting
  // room, 3 a buffer stop
  bool terminal = v_dot <= 0.0 && v_size.z > 2.5;
  float cap = terminal ? ${TERMINAL_CAP.toFixed(2)} * u_ring : u_ring;
  if (terminal) {
    inner *= clamp(f.x + v_size.x - cap + 0.5, 0.0, 1.0);
    d = min(d, boxSdf(vec2(f.x + v_size.x - 0.5 * cap, f.y), vec2(0.5 * cap, v_size.y + ${TERMINAL_EAR.toFixed(2)} * u_ring)));
  }
  float outer = clamp(0.5 - d, 0.0, 1.0);
  if (u_layer == 0) {
    // the whole shape dark, painted white inside the rim next; a blueprint (faded to 0.6, T-2930)
    // is blueprint track's blue: a dark blue rim round a light blue fill
    float a = (v_alpha < 0.99 && !bp ? outer - inner : outer) * (bp ? 1.0 : v_alpha);
    o = vec4(mix(bp ? BP_RIM : u_ink, vec3(1.0), ${HOT_RIM_MIX} * v_hot) * a, a);
  } else if (u_layer == 1) {
    // marking where the track is drawn over the station (T-236): the white's inside, no colour
    if (u_mask > 0.5) {
      if (inner < 0.5 || v_dot > 0.0) discard;
      o = vec4(0.0);
      return;
    }
    float a = inner * (bp ? 1.0 : v_alpha);
    // a dot (zoomed out) shows its platforms' crowd as a pie (T-228, T-553): the white fills
    // clockwise from 12 o'clock with the crowd grey by the fullest side's share, red once full
    vec3 col = bp ? BP_FILL : vec3(1.0);
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
    float reach = v_size.x - (terminal && p.x < 0.0 ? cap : u_ring);
    // close in on double track (T-658) each side's riders stand on their own platform, from its
    // track out to the rim: the + side's trains travel along the axis, keeping right, so their
    // track and platform are on the + side
    float tl = u_lanes.y * min(v_lane, 1.0);
    float yin = tl * (0.5 + 0.5 * ${LANE_SHARE.toFixed(1)} * u_lanes.z) * u_lanes.x;
    // at a passing station (T-2249) on the island between the running track and the outer one,
    // where trains on both stop
    float yout = v_lane > 1.5 && u_pass_in > 0.0 ? max(u_pass_in, yin + 1.0) : 1e9;
    float ap = inner * clamp(p.y - yin + 0.5, 0.0, 1.0) * clamp(yout - p.y + 0.5, 0.0, 1.0) * clamp(v_crowd.x * reach - abs(p.x) + 0.5, 0.0, 1.0);
    float am = inner * clamp(0.5 - p.y - yin, 0.0, 1.0) * clamp(yout + p.y + 0.5, 0.0, 1.0) * clamp(v_crowd.y * reach - abs(p.x) + 0.5, 0.0, 1.0);
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
in float a_lag;              // drawn this many seconds behind its timetable (T-1965, trainGuard.ts)

ivec2 tc(int i, int w) { return ivec2(i % w, i / w); }

// set by trainDist: how far through a stand after its first stop the train is (0-1), else -1
float g_stand = -1.0;

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
  float tau = u_time - a_trip.y - a_lag;
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
  if (lo > int(M.x) && lo < int(M.x + M.y) - 1) {
    vec4 p1 = texelFetch(u_phases, tc(lo + 1, ${TEX_W}), 0);
    if (p.z == 0.0 && p.w == 0.0) g_stand = clamp((tau - p.x) / max(p1.x - p.x, 1e-3), 0.0, 1.0);
    // braking into a stand: not there yet, though the stretch ahead may already be the next one
    else if (p.w < 0.0 && p1.z == 0.0 && p1.w == 0.0) g_stand = 0.0;
  }
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
uniform int u_sel_n;               // T-1812: and a selected group's lines
uniform int u_sel_set[${SEL_SET_MAX}];
uniform float u_dim;               // the others' share of full strength
uniform vec2 u_sel_trip;           // the selected train (T-242): profile, departure - epoch; x < 0 none
uniform highp int u_pass;          // 0 every train but the selected, 1 the selected's halo, 2 the selected
uniform float u_psi0;              // the anchor's isometric latitude (T-307): k = cosh(u_psi0 - y * PSI_PER_UNIT)
uniform highp sampler2D u_lane_tex; // RGBA32F per sample: how far apart its track's two tracks are, 0-1 (T-658), its line's band's shift (T-1799, line widths left of run 0) and metres to its ribbon's middle (T-2132)
uniform vec2 u_lane;               // px from the centre line to a track's once parted, how far parted (0-1)
uniform float u_bands;             // px a band's shift of one line width is drawn, 0 none (T-1799)
uniform float u_bands_m;           // and of a metre to its ribbon's middle (T-2132)
uniform vec2 u_passing;            // T-2233: px from the centre line to a passing station's outer track, 1 drawn there
out float v_alpha;
in vec2 a_load;               // this trip's first load in u_loads or -1, seconds to its last stop
in vec4 a_stand;              // its stand at the terminal (T-1785): from, until - epoch (s), the next
                              // trip's first load in u_loads or -1, else its riders leaving or -1
in vec4 a_bay;                // where it stands at its ends (T-2980, terminalBays.ts \`assignBays\`)
${SEL_GLSL}
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
flat out float v_boat;

/** How far apart the two tracks are at distance d along run 0 (laneAt in network.ts), the
 * line's band's shift and metres there (bandAt), and m past a passing platform it stops at (T-2233). */
vec4 laneAt(vec4 L, float d) {
  float dd = clamp(d, 0.0, L.z);
  float i0 = clamp(floor(dd / L.w), 0.0, max(L.y - 2.0, 0.0));
  float f = clamp((dd - i0 * L.w) / max(min(L.w, L.z - i0 * L.w), 1e-3), 0.0, 1.0);
  int si = int(L.x + i0);
  return mix(texelFetch(u_lane_tex, tc(si, ${TEX_W}), 0), texelFetch(u_lane_tex, tc(si + (L.y > 1.5 ? 1 : 0), ${TEX_W}), 0), f);
}
/** A train's shift across, px: on its own track (T-658), or \`across\` of the way to it (-1 the other
 * track, T-2980), its band (T-1799, T-2132), and onto a passing station's outer track where it stops
 * (T-2233, network.ts \`passingTracks\`). */
float shiftAt(vec4 lb, float side, float across) {
  float base = side * across * u_lane.y * lb.x * u_lane.x + lb.y * u_bands + lb.z * u_bands_m;
  return mix(base, side * u_passing.x, u_passing.y * (1.0 - smoothstep(0.0, ${PASS_LEAD_M.toFixed(1)}, lb.w)));
}

// T-2980, set in main: its run and that run's length, m
int g_run = 0;
float g_len = 0.0;

/** A place distance of a_bay (terminalBays.ts bayP): below 0 is a through end's, \`-p - 1\` m. */
float bayP(float v) { return v < 0.0 ? -v - 1.0 : v; }
/** The nearest its middle may be to a run's end (terminalBays.ts endClear, T-3053). */
float endClear(float v, float trainM) { return v < 0.0 ? max(0.0, trainM * 0.5 + v + 1.0) : trainM * 0.5; }

/** The point \`d\` m along run 0, carried on straight past either end (into a terminal's siding). */
vec2 lineAtX(vec4 L, float d) {
  float k = min(20.0, L.z);
  if (d > L.z && k > 1e-3) { vec2 a = lineAt(L, L.z); return a + (a - lineAt(L, L.z - k)) * (d - L.z) / k; }
  if (d < 0.0 && k > 1e-3) { vec2 a = lineAt(L, 0.0); return a + (a - lineAt(L, k)) * -d / k; }
  return lineAt(L, d);
}

/** T-2980: how far across to its own track the point \`x\` m along run 0 of a train \`trainM\` long is
 * drawn (1 its own, -1 the other, between on a crossover or a siding's lead): crossing the scissors
 * into a dead end's other platform or out of it, leaving a through platform over the crossover
 * behind it, or into a terminal's siding and out to the other platform. */
float acrossAt(float x, float trainM) {
  float q = g_run == 0 ? g_len - x : x, r = g_run == 0 ? x : g_len - x;
  float sc = a_bay.x < 0.0 ? -1.0 : mod(a_bay.x, 4.0), ec = a_bay.z < 0.0 ? -1.0 : mod(a_bay.z, 4.0);
  if (r < q) {
    if (sc < 0.5 || sc > 1.5) return 1.0;
    float p = max(bayP(a_bay.y), trainM + 5.0);
    return mix(-1.0, 1.0, smoothstep(p, p + ${TURNBACK_M.toFixed(1)}, r));
  }
  if (ec > 0.5 && ec < 1.5) {
    float p = max(bayP(a_bay.w), trainM + 5.0);
    return mix(-1.0, 1.0, smoothstep(p, p + ${TURNBACK_M.toFixed(1)}, q));
  }
  return 1.0;
}

void main() {
  vec4 L;
  float d;
  int line;
  // the selected train is drawn last, over the others, with its halo under it (T-242)
  bool chosen = u_sel_trip.x >= 0.0 && abs(a_trip.x - u_sel_trip.x) < 0.5 && abs(a_trip.y - u_sel_trip.y) < 0.01;
  if ((u_pass == 0) == chosen || !trainDist(L, d, line)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float rawCars = texelFetch(u_cars, tc(line, ${LINES_W}), 0).x;
  // a tram (T-2926): its sections as a whole number plus a half; 2.4 m across, sections end to end
  bool tram = rawCars >= 1.0 && fract(rawCars) > 0.25;
  if (tram) rawCars = floor(rawCars);
  // a bus (T-1464): one car under a full car's length, which is its value here; narrower than a train.
  // A boat (T-2194): below 0, its length in cars; one hull, 8 m across
  bool boat = rawCars < 0.0;
  bool bus = !boat && rawCars < 0.99;
  float cars = boat ? 1.0 : max(1.0, rawCars);
  // T-2957: a bus is one straight body; its minimum on-screen size is 0.65 as long (Anita undid the extra 10%, 2026-11-10)
  const float BUS_LEN = 0.65;
  float carM = boat ? -rawCars * 20.0 : bus ? rawCars * 20.0 : 20.0;
  v_boat = boat ? 1.0 : 0.0;
  int car = u_detail ? gl_VertexID / 6 : 0;
  if (float(car) >= cars) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float totalM = min(cars * carM, L.z);
  float pitchM = totalM / cars;
  int prof = int(a_trip.x + 0.5);
  vec4 M = texelFetch(u_meta, tc(prof, ${TEX_W}), 0);
  g_run = (prof / 3) % 2;
  g_len = M.w;
  // T-3053: at a through end it stands with its middle at the stop, so up to half its length past the line's end
  float cLo = endClear(g_run == 0 ? a_bay.y : a_bay.w, totalM);
  float centre = clamp(d, cLo, max(cLo, L.z - endClear(g_run == 0 ? a_bay.w : a_bay.y, totalM)));
  d = centre;
  float offsetM = u_detail ? (float(car) + 0.5) * pitchM - totalM * 0.5 : 0.0;
  float carD = u_detail ? centre + offsetM : d;
  vec2 p = lineAtX(L, carD);
  // local units per metre of ground where the train is (T-307: mercator's scale, 1 / cos(lat))
  float kM = cosh(u_psi0 - p.y * ${PSI_PER_UNIT.toExponential(9)});
  // Both representations share the same width: splitting into cars must not shrink the train.
  // a bus (T-1798) is drawn 1.4 times its size from a distance and 1.2 times its real one close up
  vec2 px = max(bus ? u_min_px * vec2(0.77 * BUS_LEN, 0.77) : boat ? u_min_px * vec2(1.0, 1.3) : tram ? u_min_px * vec2(0.75, 0.885) : u_min_px, vec2(cars * carM, bus ? 2.5 : boat ? 8.0 : tram ? 2.4 : 3.0) * kM * u_px_per_unit * (bus ? 1.2 : 1.0));
  if (u_detail) px.x = max(bus ? 2.8 * BUS_LEN * u_dpr : 2.0 * u_dpr, pitchM * (bus ? 1.2 : 1.0) * kM * u_px_per_unit - (tram ? 0.0 : max(1.5 * u_dpr, kM * u_px_per_unit)));
  // on its own track as double track parts (T-658): narrowing to its true 3 m, so two trains
  // passing never touch, and keeping right of its direction of travel (T-319)
  vec4 lb = laneAt(L, carD);
  float lane = u_lane.y * lb.x;
  px.y = mix(px.y, (tram ? 2.4 : 3.0) * kM * u_px_per_unit, lane);
  // Heading along the chord between the train's two ends (as its bogies sit on the track), not
  // the sample segment under its centre, which turns in steps on curves (T-047).
  float h = max(0.5 * px.x / u_px_per_unit / kM, 1.0);
  vec2 back = lineAtX(L, carD - h), front = lineAtX(L, carD + h);
  vec4 c = project(p);
  vec4 cb = project(back);
  vec4 cf = project(front);
  vec2 half_vp = u_viewport * 0.5;
  vec2 bs = cb.xy / cb.w * half_vp, fs = cf.xy / cf.w * half_vp;
  vec2 sdir = fs - bs;
  float sl = length(sdir);
  sdir = sl > 1e-6 ? sdir / sl : vec2(1.0, 0.0);
  vec2 nrm = vec2(-sdir.y, sdir.x);
  // T-2182: its two ends each shifted by the band there, so a train easing sideways with its band
  // turns with it rather than sliding crabwise
  float side = (int(a_trip.x + 0.5) / 3) % 2 == 0 ? -1.0 : 1.0;
  vec4 lbb = laneAt(L, carD - h), lbf = laneAt(L, carD + h);
  float shB = shiftAt(lbb, side, acrossAt(carD - h, totalM)), shF = shiftAt(lbf, side, acrossAt(carD + h, totalM));
  if (shB != shF) {
    bs += nrm * shB;
    fs += nrm * shF;
    vec2 sd = fs - bs;
    float sl2 = length(sd);
    if (sl2 > 1e-6) {
      sdir = sd / sl2;
      nrm = vec2(-sdir.y, sdir.x);
      sl = sl2;
    }
    // (the ends back in the frame of the train's own centre, unshifted, for the bend below)
    float shC0 = shiftAt(lb, side, acrossAt(carD, totalM));
    bs -= nrm * shC0;
    fs -= nrm * shC0;
  }
  // A train as one pill bends with its track (T-189): along the circle through its two ends and
  // its middle, as a station's platform does (SHAPE_GLSL); a car close up is short enough to stay
  // straight. k: curvature, 1/px, towards +nrm; ends: where the ends lie across the frame.
  float k = 0.0, ends = 0.0;
  if (!u_detail && !bus && sl > 1e-3) {
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
  float shift = shiftAt(lb, side, acrossAt(carD, totalM));
  c.xy += (sdir * q.x + nrm * (q.y + shift)) / half_vp * c.w;
  gl_Position = c;
  v_uv = q;
  v_k = k;
  v_px = px;
  vec3 lc = texelFetch(u_line_colours, ivec2(line % ${LINES_W}, line / ${LINES_W}), 0).rgb;
  v_color = lc * 0.48;
  v_alpha = selectedLine(line) ? 1.0 : u_dim;
  // How full it is (T-084): the stretch it runs, from the sample ahead of it on its way.
  int run = (int(a_trip.x + 0.5) / 3) % 2;
  float i = floor(clamp(d, 0.0, L.z) / L.w) + (run == 0 ? 1.0 : 0.0);
  int si = int(L.x + clamp(i, 0.0, L.y - 1.0));
  vec2 f = texelFetch(u_fill, tc(si, ${TEX_W}), 0).xy;
  v_fill = run == 0 ? f.x : f.y;
  // Its own load from the boarding account where there is one (T-164); the fill above then only
  // says whether train loads are shown for this line. Empty once at its last stop, then filling
  // over its stand there with its next trip's riders (T-1785), as the platform empties.
  float st = texelFetch(u_stretch, tc(si, ${TEX_W}), 0).x;
  if (v_fill >= 0.0 && a_load.x >= 0.0 && st >= 0.0) {
    float aboard;
    if (u_time - a_trip.y - a_lag >= a_load.y) {
      float f = clamp((u_time - a_stand.x) / max(a_stand.y - a_stand.x, 1.0), 0.0, 1.0);
      aboard = f * (a_stand.z >= 0.0 ? texelFetch(u_loads, tc(int(a_stand.z + st + 0.5), ${TEX_W}), 0).x : max(a_stand.w, 0.0));
    } else {
      aboard = texelFetch(u_loads, tc(int(a_load.x + st + 0.5), ${TEX_W}), 0).x;
      // standing at a stop: from the load it came in with to the one it leaves with (T-1785); a
      // stand between stops has the same stretch behind it as ahead
      int sb = int(L.x + clamp(i + (run == 0 ? -2.0 : 2.0), 0.0, L.y - 1.0));
      float back = texelFetch(u_stretch, tc(sb, ${TEX_W}), 0).x;
      if (g_stand >= 0.0 && back >= 0.0 && abs(back - st) > 0.5) {
        float came = texelFetch(u_loads, tc(int(a_load.x + st + (run == 0 ? -1.0 : 1.0) + 0.5), ${TEX_W}), 0).x;
        aboard = mix(came, aboard, g_stand);
      }
    }
    // a boat holds two cars' riders, a bus a third of a car's (T-2195, T-2919, game/boarding.ts capacity)
    float held = boat ? ${BOAT_CARS.toFixed(1)} : bus ? ${BUS_CARS.toFixed(6)} : cars;
    float seats = ${SEATS_PER_CAR.toFixed(1)} * held, rated = ${RATED_PER_CAR.toFixed(1)} * held, crush = ${CRUSH_PER_CAR.toFixed(1)} * held;
    // T-2186: 1 at the rated load, then the second band to 2 at crush (game/boarding.ts gauge)
    v_fill = aboard <= seats ? 0.5 * aboard / seats : aboard <= rated ? 0.5 + 0.5 * (aboard - seats) / (rated - seats) : min(1.99, 1.0 + (aboard - rated) / (crush - rated));
    // packed to crush (the account never goes past it: whoever does not fit waits) draws with the
    // red edge, so a full train stands out from one nearly full (T-207)
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
// colour beyond; half way means every seat taken, full means the rated load. Past it a second band
// of the dark shade turned towards red fills from the back again, whole at crush, where the train
// has a red edge (T-2186).
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
flat in float v_boat;
uniform float u_dpr;
uniform vec3 u_bad;
uniform highp int u_pass;
uniform vec3 u_ink;
uniform vec3 u_band;
uniform bool u_hollow;
out vec4 o;
${SHAPE_GLSL}
void main() {
  // along the train's bend and across it (T-189)
  vec2 p = arc(v_uv, v_k);
  float r = v_detail > 0.5 || v_boat > 0.5 ? min(2.0 * u_dpr, v_px.y * 0.22) : v_px.y * 0.5;
  vec2 q = abs(p) - (v_px * 0.5 - r);
  float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
  if (v_boat > 0.5) {
    // a boat's hull (T-2194): its front third narrows to a point
    float hx = v_px.x * 0.5, hy = v_px.y * 0.5, bow = hx - min(0.66 * v_px.x, 1.6 * v_px.y);
    vec2 n = normalize(vec2(hy, hx - bow));
    d = max(d, dot(vec2(p.x * v_front - bow, abs(p.y) - hy), n));
  }
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
    if (v_fill > 1.0) core = mix(core, mix(v_color, u_bad, 0.65), clamp((v_fill - 1.0) * v_consist.x - back + 0.5, 0.0, 1.0));
    if (v_fill > 1.995) rim = u_bad;
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
  float rimW = v_fill > 1.995 ? 3.9 : 1.7;
  // the start menu's logo (T-1567): white inside a rim of the line's colour
  if (u_hollow) { core = vec3(1.0); rim = v_color / 0.48; rimW = max(1.7, 0.28 * v_px.y); }
  vec3 col = mix(core, rim, clamp(d + rimW, 0.0, 1.0));
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
