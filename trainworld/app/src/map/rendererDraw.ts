// The renderer's per-frame draw, one function a pass, moved out of renderer.ts (T-2374).

import { rgbf } from "../game/palette";
import { ANNEX_ROOM, ST_ROOM, BOX_WIDTH_ZOOM, boxHalfPxAtBoxZoom, BRIDGE_HALF_PX, connectorHalfPx, dotRing, spanningPairs, PLATFORM_HALF_M, SEL_GLOW_PX, SEL_OUTLINE_PX, shapePoints, STATION_BOX_ZOOM, STATIONS_ZOOM, TRAINS_ZOOM, stationRadii, stationRing, stationShape, type Project, type StationShape } from "./stationGeometry";
import { shadeColours } from "../game/shades";
import { MOVING_PLACE_MS } from "./moving";
import { anchor, anchorPsi, EARTH_C, kAtY, localToMercMatrix, mul4, pxPerUnitAt, unitPerPxAt, type Anchor } from "./geo";
import { clipToCss, interp, localCamera, mercToCss, setCamera, viewCentreMerc, type Camera, type LocalCamera } from "./gl";
import { StrokeCache } from "./strokeCache";
import { STENCIL_BIT } from "./buildingFloors";
import { backBaseOf, type NetworkBuffers } from "./network";
import { BAND_GAP_PX } from "./lineBands";
import { DOUBLE_WIDTH, HOT_STATIONS, SEL_MAX, SEL_STRIDE, SINGLE_WIDTH, TEX_W, TRAM_WIDTH } from "./rendererShaders";
import { hidden as vis } from "./visibility";
import { TRACK_RGB, BAD_RGB, BP_MID_RGB, BP_RAIL_RGB, BP_RAIL_SHARE, BP_NARROW, TRAIN_PX, LINE_PX, HSR_WIDTH, HSR_STRIPE_PX, setSelLines, GHOST_MIN_PX, LANE_M, BAND_FOLD_PX, MUTE, BAND_GAP, smoothstep, laneState, passingState, CONFLICT_ZOOM, CROWD_RGB, TRACK_IN_BOX, StationGroups, DrawOptions, StrokeVao, GHOST_OUTLINE_PX } from "./rendererParts";
import type { NetworkRenderer } from "./renderer";
import { buffer, attrib, pickStrokes, bindAt, rewrite } from "./rendererStrokes";

type Ctx1 = NonNullable<ReturnType<typeof frameSetup>>;
type Ctx2 = Ctx1 & ReturnType<typeof strokeUniforms>;
type Ctx3 = Ctx2 & ReturnType<typeof strokeDraws>;

function frameSetup(rnd: NetworkRenderer, cam: Camera, time: number, zoom: number, dpr: number, opt: DrawOptions, view?: readonly [number, number]) {
  const gl = rnd.gl;
  rnd.quiet = false;
  rnd.releaseStaged(time);
  const n = rnd.net;
  if (!n) return null;
  if (rnd.hiddenFor !== n || rnd.hiddenV !== vis.v) rnd.applyHidden(n);
  const uTime = rnd.sync(time);
  rnd.bubbles.hide();
  // the network's buffers are relative to the anchor they were built at (T-307)
  const a = n.anchor ?? anchor();
  const lc = localCamera(cam, a);
  rnd.cam = cam;
  const vp = view ?? ([gl.drawingBufferWidth, gl.drawingBufferHeight] as const);
  const pxPerUnit = pxPerUnitAt(zoom) * dpr;

  rnd.timer.begin();

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.STENCIL_TEST);
  gl.disable(gl.CULL_FACE);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const k = opt.lineScale ?? 1;
  const wLine = interp(zoom, LINE_PX) * dpr * k;
  const wTrack = interp(zoom, [[9, 0.4], [12, 1.8], [15, 3]]) * dpr * k;
  const wSel = interp(zoom, [[9, 0.5], [12, 2.1], [15, 3]]) * dpr * k;
  const casing = interp(zoom, [[9, 0.6], [12, 2], [15, 2.4]]) * dpr * k;
  // the stations' sizes this frame, which the track through their boxes needs (T-236)
  const boxes = zoom >= STATION_BOX_ZOOM;
  const hidden = zoom < STATIONS_ZOOM;
  // a dot's rim is thinner than a box's (T-322)
  const ring = boxes ? stationRing(zoom) : dotRing(zoom);
  const boxLinePx = interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx = stationRing(BOX_WIDTH_ZOOM);
  // passing stations close in (T-2233): their outer tracks, the trains stopping there, their boxes
  const pass = boxes && rnd.passing ? passingState(zoom, ring) : null;
  rnd.passShiftM = pass ? pass.off / pass.mPx : 0;
  // T-3332, T-3615: the track a one-track station's bar covers, rim included (its size is its box's, oneBar): the
  // line through it until double track's two tracks part, then from the centre line out over its own track;
  // single track (T-3540) either side of its line. As STROKE_VS draws them, lines and tram track narrower.
  const ls = laneState(zoom), wl = wLine / dpr, lt = ls.share * ls.gap;
  const cover = (tram: boolean) => {
    const tw = tram ? TRAM_WIDTH : 1, kS = tram ? 1 : SINGLE_WIDTH + (1 - SINGLE_WIDTH) * ls.grow, kD = tram ? 1 : DOUBLE_WIDTH + (1 - DOUBLE_WIDTH) * ls.grow;
    const core = wl * tw * kD, d = 0.5 * (core + ls.grow * Math.max(0, ls.gap + lt * tw - core)) + ring;
    return { single: 0.5 * Math.max(wl * tw * kS, lt * tw) + ring, inner: (1 - ls.t) * d, outer: d };
  };
  const cr = cover(false), ct = cover(true);
  const one = { inner: cr.inner, outer: cr.outer, innerTram: ct.inner, outerTram: ct.outer };
  const oneMid = { inner: cr.single, outer: cr.single, innerTram: ct.single, outerTram: ct.single };
  rnd.stationStyle = { boxes, radius: stationRadii(zoom), linePx: wl, ringPx: ring, boxLinePx, boxRingPx, hidden, ...(pass ? { passHalfPx: pass.half } : {}), one, oneMid };
  const project = projector(rnd, cam, a, vp[0] / dpr, vp[1] / dpr);
  // T-3481: on-screen sizes are measured at the view's middle (local units), never at a station,
  // which can be in another city behind the globe (NaN) or where the globe's blend squeezes it
  const mid = viewCentreMerc(cam);
  const midX = mid ? (mid[0] - a.merc[0]) * EARTH_C : n.stations[0] ?? 0, midY = mid ? (mid[1] - a.merc[1]) * EARTH_C : n.stations[1] ?? 0;
  // The track through the boxes (T-236, Anita: "maybe the track on top of the station"): the
  // track and line strokes drawn again, no wider than TRACK_IN_BOX of the box's white and
  // without casing, where the stencil marks a station's white (1 the unselected, 2 the selected).
  const through = n.stationCount > 0 && boxes && rnd.hasStencil;
  // every box has one width on the ground (T-239): a plain box at the view's middle gives it on screen
  const kMid = kAtY(midY, a);
  const boxWhite = 2 * ((boxes && n.stationCount ? stationShape(midX, midY, { mx: 0, my: 0, ax: kMid, ay: 0, phi: 0 }, PLATFORM_HALF_M * kMid, project, rnd.stationStyle, false).halfWidth : 0) - ring);
  const trackCap = Math.max(1, TRACK_IN_BOX * boxWhite) * dpr;
  // T-1638: everything up to the stations changes only with the camera, the network and how it is
  // styled, so it is kept in a texture (layer 0, the track through the boxes layer 1) and drawn
  // again only when one of those changes; GL uploads outside the per-frame passes count as a change
  const cache = rnd.strokeCache && !rnd.gl.isContextLost() && Math.max(vp[0], vp[1]) <= rnd.cacheMax ? (view ? (rnd.snapCache ??= new StrokeCache(gl, 2)) : (rnd.mainCache ??= new StrokeCache(gl, 2))) : null;
  const buses = !view && rnd.buses.any;
  if (!view) rnd.busesShown = buses;
  const stationsOff = opt.showStations === false;
  rnd.stationsOff = stationsOff;
  const redraw = !cache || cache.stale(vp[0], vp[1], [zoom, dpr, ...lc.m, ...lc.globe, ...lc.plane, ...lc.at, k, opt.byLevel ? 1 : 0, opt.bySpeed ? 1 : 0, opt.byProfit ? 1 : 0, opt.selLine, (opt.selLines ?? []).reduce((h, v) => (h * 31 + v + 1) | 0, 7), opt.selEdge, opt.hotLine ?? -1, opt.hotEdge ?? -1, (opt.hotBuses ?? []).reduce((h, v) => (h * 31 + v + 1) | 0, 7), (opt.hotStops ?? []).reduce((h, v) => (h * 31 + v + 1) | 0, 7), rnd.hide[0], rnd.hide[1], opt.dimOthers ?? 1, opt.flatMarks !== false ? 1 : 0, opt.rampMarks !== false ? 1 : 0, buses ? rnd.buses.version : -1, through ? 1 : 0, rnd.uploads, rnd.strokeCull ? 1 : 0, rnd.strokeLod ? 1 : 0, rnd.strokeGather ? 1 : 0, rnd.lineBands ? 1 : 0, stationsOff ? 1 : 0, vis.v], shadeColours());
  return { cam, time, zoom, dpr, opt, view, gl, n, uTime, a, lc, vp, pxPerUnit, k, wLine, wTrack, wSel, casing, boxes, hidden, ring, boxLinePx, boxRingPx, pass, project, midX, midY, through, boxWhite, trackCap, cache, buses, stationsOff, redraw };
}

function beginStrokes(rnd: NetworkRenderer, c: Ctx1) {
  const { zoom, dpr, opt, n, a, lc, vp, wLine, wSel, casing, cache, buses, stationsOff, redraw } = c;
  rnd.quiet = true;
  if (cache && redraw) cache.begin(0);
  // the complexes' car parks under everything else (T-1528)
  if (redraw && !stationsOff) rnd.floors.draw(lc, vp, dpr, zoom);
  if (redraw && !stationsOff && zoom >= STATION_BOX_ZOOM) rnd.lots.draw(lc, vp, dpr);
  // bus lines have no strokes: their own pass, under the track (T-1464, busLayer.ts)
  if (redraw && buses) {
    const sh = shadeColours();
    rnd.buses.draw(lc, a, vp, zoom, dpr, { wLine, casing, wSel, rail: BP_RAIL_RGB, railShare: BP_RAIL_SHARE, sel: opt.selLine, hot: opt.hotLine ?? -1, lit: opt.hotBuses ?? [], litStops: opt.hotStops ?? [], stationInk: rgbf(sh.stationInk), dim: opt.dimOthers ?? 1, shades: sh, fill: rgbf(sh.casing), casingRgb: rgbf(sh.casing),
      paint: opt.byLevel ? "level" : opt.bySpeed ? "speed" : opt.byProfit ? "profit" : null, phases: n.phases, meta: n.meta, trackRgb: TRACK_RGB });
  }
}

function strokeUniforms(rnd: NetworkRenderer, c: Ctx1) {
  const { zoom, dpr, opt, gl, lc, vp, k, wLine, pass } = c;
  gl.useProgram(rnd.strokeProg);
  const u = rnd.strokeU;
  setCamera(gl, u, lc);
  gl.uniform2f(u.u_viewport, vp[0], vp[1]);
  gl.uniform1i(u.u_sel_line, opt.selLine);
  setSelLines(gl, u, opt.selLines);
  gl.uniform1i(u.u_sel_edge, opt.selEdge);
  gl.uniform2i(u.u_hot, opt.hotLine ?? -1, opt.hotEdge ?? -1);
  gl.uniform2i(u.u_hide, rnd.hide[0], rnd.hide[1]);
  gl.uniform1i(u.u_level_hide, vis.levels);
  // T-2058: the other lines thinner, faded and solid
  const mute = MUTE && (opt.dimOthers ?? 1) < 1 ? MUTE : null;
  gl.uniform1f(u.u_dim, mute ? 1 : opt.dimOthers ?? 1);
  gl.uniform2f(u.u_mute, mute?.[0] ?? 0, mute?.[1] ?? 0);
  gl.uniform1i(u.u_by_level, opt.byLevel ? 1 : 0);
  gl.uniform1i(u.u_by_speed, opt.bySpeed ? 1 : 0);
  gl.uniform1i(u.u_by_profit, opt.byProfit && rnd.profitTex ? 1 : 0);
  gl.uniform1i(u.u_profit_n, rnd.profitTex ? rnd.profitN : 0);
  gl.activeTexture(gl.TEXTURE12);
  gl.bindTexture(gl.TEXTURE_2D, rnd.profitTex);
  gl.uniform1i(u.u_profit, 12);
  // Lines sharing track are drawn over each other, the last on top (T-190); picking knows the
  // line width, and that no line strokes are drawn when the track is coloured by height or speed.
  rnd.slotPx = opt.byLevel || opt.bySpeed || opt.byProfit ? 0 : wLine / dpr;
  // T-1799: lines sharing track side by side in the Line view, folding into one stroke as lines
  // thin out zoomed out and as double track opens into its two tracks close in
  const lanes = laneState(zoom);
  // and in the Traffic view (stage 2) each band its riders' width, the shifts `a_wmul.yz`
  const bandOn = !rnd.lineBands || opt.byLevel || opt.bySpeed || opt.byProfit || (rnd.trafficWidths && !rnd.trafficBands) ? 0 : smoothstep(BAND_FOLD_PX[0], BAND_FOLD_PX[1], wLine / dpr) * (1 - lanes.grow);
  rnd.bandPx = bandOn * wLine / dpr;
  rnd.bandShiftM = rnd.bandPx / (pxPerUnitAt(zoom) * kAtY(0));
  rnd.bandOn = bandOn;
  rnd.bandMPx = bandOn * pxPerUnitAt(zoom) * kAtY(0);
  gl.uniform1f(u.u_band, bandOn);
  gl.uniform1f(u.u_band_m, pxPerUnitAt(zoom) * kAtY(0) * dpr);
  gl.uniform1i(u.u_band_traffic, rnd.trafficWidths ? 1 : 0);
  gl.uniform1f(u.u_band_gap, BAND_GAP ? BAND_GAP_PX * dpr : 0);
  gl.uniform3fv(u.u_level_colours, rnd.levelColours);
  gl.uniform3fv(u.u_track, TRACK_RGB);
  gl.uniform3fv(u.u_bad, BAD_RGB);
  const shades = shadeColours();
  gl.uniform3fv(u.u_sel, rgbf(shades.mapInk));
  gl.uniform3fv(u.u_casing_rgb, rgbf(shades.casing));
  // T-819, T-1284, T-1549, T-1663: red dashes either side, outside the casing, of a line on full
  // track (this base, plus a share of the line's width in the shader), 0.5 px of casing colour round
  // them; the base scaled down with zoom, so far out they stay a hairline
  const fullPass = rnd.anyFull;
  const fullScale = interp(zoom, [[9, 0.3], [12, 0.6], [15, 1]]) * dpr * k;
  gl.uniform1i(u.u_full_pass, 0);
  gl.uniform1f(u.u_full_px, 1.75 * fullScale);
  gl.uniform1f(u.u_full_out, 1 * fullScale);
  gl.uniform1f(u.u_full_dash, 16 * dpr);
  gl.uniform3fv(u.u_full_rgb, rgbf(shades.fullTrack));
  gl.uniform1i(u.u_ghost, 0);
  gl.uniform3fv(u.u_ghost_rgb, rgbf(shades.ghost));
  gl.uniform1i(u.u_ghost_layer, -1);
  gl.uniform2f(u.u_ghost_min, GHOST_MIN_PX * dpr, 2 * GHOST_OUTLINE_PX * dpr);
  gl.uniform3f(u.u_hsr, interp(zoom, HSR_WIDTH), HSR_STRIPE_PX[0] * dpr, HSR_STRIPE_PX[1] * dpr);
  gl.uniform3fv(u.u_wire, rgbf(shades.tramWire));
  // blueprint track (T-322): a line-wide light blue middle between two thin dark blue rails
  const bpRail = Math.max(1, BP_NARROW * Math.max(1, BP_RAIL_SHARE * wLine / dpr)) * dpr;
  const bpMid = Math.max(0, BP_NARROW * (wLine + 2 * Math.max(1, BP_RAIL_SHARE * wLine / dpr) * dpr) - 2 * bpRail);
  gl.uniform2f(u.u_bp, bpMid, bpRail);
  gl.uniform3fv(u.u_bp_mid, BP_MID_RGB);
  gl.uniform3fv(u.u_bp_rail, BP_RAIL_RGB);
  gl.uniform3fv(u.u_bp_off_mid, rgbf(shades.offMid));
  gl.uniform3fv(u.u_bp_off_rail, rgbf(shades.offRail));
  gl.uniform3fv(u.u_bp_off_fade, rgbf(shades.offFade));
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, rnd.tex.colours);
  gl.uniform1i(u.u_line_colours, 2);
  rnd.laneReachPx = lanes.reach;
  rnd.laneShiftM = LANE_M / 2 * lanes.t;
  const G = lanes.gap * dpr;
  gl.uniform4f(u.u_lanes, G, lanes.t, lanes.grow, lanes.share * G);
  gl.uniform1f(u.u_pass_off, pass ? pass.off * dpr : 0);
  const modes = lanes.t > 0 ? [1, 2, 3] : [0];
  return { u, mute, lanes, bandOn, shades, fullPass, bpMid, bpRail, G, modes };
}

function strokeDraws(rnd: NetworkRenderer, c: Ctx2) {
  const { zoom, dpr, opt, gl, n, vp, wLine, wTrack, wSel, casing, pass, project, redraw, u, lanes, fullPass, G, modes } = c;
  const picked = new Map<StrokeVao, { vao: StrokeVao; ranges: number[] | null }>();
  if (redraw) {
    rnd.strokeInstances = 0;
    for (const [s, src, isLines] of [[rnd.track, n.track, false], [rnd.lines, n.lines, true]] as const) {
      const p = pickStrokes(rnd, s, src, isLines, zoom, dpr, lanes.grow, project, vp[0] / dpr, vp[1] / dpr);
      if (s && p) picked.set(s, p);
    }
  }
  // `cap` > 0: the track drawn through station boxes (T-236), no wider than that and no casing
  const stroke = (s0: StrokeVao | null, kind: number, width: number, sel: boolean, cap = 0) => {
    if (!s0 || !s0.count) return;
    const { vao: s, ranges } = picked.get(s0) ?? { vao: s0, ranges: null };
    gl.uniform1i(u.u_kind, kind);
    // each track's width once apart: a line's share of the gap, bare track as much thinner
    gl.uniform3f(u.u_width, width, wSel, lanes.share * G * width / wLine);
    gl.uniform1f(u.u_cap, cap);
    gl.bindVertexArray(s.vao);
    const drawAll = () => {
      if (!ranges) {
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
        rnd.strokeInstances += s.count;
        return;
      }
      for (let r = 0; r < ranges.length; r += 2) {
        bindAt(rnd, s, ranges[r]);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, ranges[r + 1]);
        rnd.strokeInstances += ranges[r + 1];
      }
    };
    // [casing px, selected only, T-819's red band under the lines on full track (1), and under it (2) its outline in the casing's colour, T-1284]
    const passes: [number, number, number][] = cap > 0 ? [[0, 0, 0]] : [[casing, 0, 0], [0, 0, 0]];
    // (T-2184's red edges of bands with riders waiting are drawn by the colour pass, in the line's order, T-2347)
    if (kind === 1 && fullPass && cap === 0) passes.unshift([casing, 0, 2], [casing, 0, 1]);
    if (sel) passes.push(...(cap > 0 ? [[0, 1, 0]] as [number, number, number][] : [[casing, 1, 0], [0, 1, 0]] as [number, number, number][]));
    for (const [cas, only, full] of passes) {
      gl.uniform1f(u.u_casing, cas);
      gl.uniform1i(u.u_only_sel, only);
      gl.uniform1i(u.u_full_pass, full);
      for (const mode of modes) {
        if (cap > 0 && mode === 1) continue;
        if (full && mode > 1) continue;
        gl.uniform1i(u.u_lane_mode, mode);
        drawAll();
      }
    }
    if (ranges) bindAt(rnd, s, 0);
    gl.uniform1i(u.u_full_pass, 0);
    gl.uniform1i(u.u_lane_mode, 0);
  };
  // Colour by height: the track itself, at line width, and no line strokes over it. Blueprint
  // track and the lines planned on it go over built track and its lines (T-695).
  const trackStrokes = (cap = 0) => {
    for (const part of rnd.hasBlueprint ? (vis.blueprints ? [1] : [1, 2]) : [0]) {
      gl.uniform1i(u.u_bp_part, part);
      // through a station's box blueprint track is drawn whole (rails and middle) over a blueprint box
      const c = part === 2 ? 0 : cap;
      stroke(rnd.track, 0, opt.byLevel || opt.bySpeed || opt.byProfit ? wLine : wTrack, opt.selEdge >= 0, c);
      if (!opt.byLevel && !opt.bySpeed && !opt.byProfit) stroke(rnd.lines, 1, wLine, opt.selLine >= 0 || !!opt.selLines?.length, c);
    }
    gl.uniform1i(u.u_bp_part, 0);
    // passing stations' outer tracks along their platforms (T-2233)
    if (cap > 0 && pass) stroke(rnd.turnback, 0, wTrack, false, cap);
  };
  return { stroke, trackStrokes };
}

/** T-3899: the track level view's ramp ends: the drawn width (device px) of built and blueprint, single and
 * double bare track as the stroke shader works it out (its width, grown to the pair's outer edges once double
 * track opens), and the line's thickness, 1/8 of the built double width, at least a pixel. */
function rampEndSizes(c: Ctx3) {
  const { wLine, lanes, G, bpMid, bpRail, dpr, zoom } = c;
  const kS = SINGLE_WIDTH + (1 - SINGLE_WIDTH) * lanes.grow, kD = DOUBLE_WIDTH + (1 - DOUBLE_WIDTH) * lanes.grow;
  const pair = G * (1 + lanes.share);
  const dBuilt = Math.max(wLine * kD, pair);
  // slots: variant (rail, tram, high speed) * 4 + built single, blueprint single, built double, blueprint double
  const rail = [wLine * kS, bpMid * kS + 2 * bpRail, dBuilt, Math.max(bpMid * kD + 2 * bpRail, pair)];
  // tram track is a fixed share of a line's width with no single/double narrowing; double grows to the pair's outer edges
  const tc = wLine * TRAM_WIDTH, tb = bpMid * TRAM_WIDTH + 2 * bpRail;
  const grow = (core: number, lt: number) => core + lanes.grow * Math.max(0, G + lt - core);
  const tram = [tc, tb, grow(tc, lanes.share * G * TRAM_WIDTH), grow(tb, lanes.share * G)];
  // high-speed built track is wider by u_hsr.x as one stroke; blueprint is drawn as rail
  const h = interp(zoom, HSR_WIDTH);
  const hsr = [wLine * kS * h, rail[1], wLine * kD * h + lanes.grow * Math.max(0, G * (1 + lanes.share) - wLine * kD), rail[3]];
  return { w: [...rail, ...tram, ...hsr], thick: Math.max(dpr, dBuilt / 8) };
}

function strokesPass(rnd: NetworkRenderer, c: Ctx3) {
  const { zoom, dpr, opt, gl, lc, vp, wLine, wTrack, pass, through, trackCap, cache, redraw, u, mute, lanes, bandOn, shades, bpMid, bpRail, G, modes, stroke, trackStrokes } = c;
  if (redraw) {
  if (rnd.highlight) {
    // T-1679: picked track gets a selected station's look, an ink outline (2 px) apart from the
    // track by a casing-coloured band (3 px), so it reads apart from a selected line's wider
    // stroke (kind 2 placement strokes in a casing pass take u_sel)
    gl.uniform1i(u.u_kind, 2);
    gl.uniform1f(u.u_cap, 0);
    gl.uniform1i(u.u_only_sel, 0);
    gl.uniform1f(u.u_casing, 0.01 * dpr);
    // round both tracks once they part
    gl.uniform1i(u.u_lane_mode, lanes.t > 0 ? 1 : 0);
    gl.bindVertexArray(rnd.highlight.vao);
    for (const [px, rgb] of [[10, shades.mapInk], [6, shades.casing]] as const) {
      gl.uniform3fv(u.u_sel, rgbf(rgb));
      gl.uniform3f(u.u_width, Math.max(bpMid + 2 * bpRail, wLine) + px * dpr, 0, lanes.share * G + px * dpr);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, rnd.highlight.count);
    }
    if (rnd.refused) {
      gl.bindVertexArray(rnd.refused.vao);
      for (const [px, rgb] of [[10, shades.fullTrack], [6, shades.casing]] as const) {
        gl.uniform3fv(u.u_sel, rgbf(rgb));
        gl.uniform3f(u.u_width, Math.max(bpMid + 2 * bpRail, wLine) + px * dpr, 0, lanes.share * G + px * dpr);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, rnd.refused.count);
      }
    }
    gl.uniform3fv(u.u_sel, rgbf(shades.mapInk));
    gl.uniform1i(u.u_lane_mode, 0);
  }
  trackStrokes();
  // where lines turn back, over their tracks (T-658d), and passing stations' outer tracks (T-2233)
  if (lanes.t > 0 || pass) stroke(rnd.turnback, 0, wTrack, false);
  gl.uniform1i(u.u_ghost, 1);
  // the ghost, and a placed segment's stand-in until its network is drawn (T-1407)
  for (const g of [...rnd.placed, rnd.preview]) {
    stroke(g, 2, wLine, false);
    if (!g?.count) continue;
    // the ghost: blueprint's outline, rails and middle, each over the last (T-1054)
    gl.uniform1f(u.u_casing, 0);
    gl.uniform1f(u.u_cap, 0);
    gl.uniform1i(u.u_only_sel, 0);
    gl.bindVertexArray(g.vao);
    for (let layer = 0; layer < 3; layer++) {
      gl.uniform1i(u.u_ghost_layer, layer);
      for (const mode of modes) {
        if (mode === 1) continue;
        gl.uniform1i(u.u_lane_mode, mode);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, g.count);
      }
    }
    gl.uniform1i(u.u_ghost_layer, -1);
    gl.uniform1i(u.u_lane_mode, 0);
  }
  gl.uniform1i(u.u_ghost, 0);
  rnd.marks.draw(lc, vp, zoom, dpr, rnd.levelColours, opt.flatMarks !== false, opt.rampMarks !== false, opt.byLevel ? rampEndSizes(c) : null);
  // T-2991: one-way loops' chevrons on the Line view's strokes (the Traffic view's bands move apart)
  if (!opt.byLevel && !opt.bySpeed && !opt.byProfit && !rnd.trafficWidths)
    rnd.oneWay.draw(lc, vp, zoom, dpr, { wLine, band: bandOn, mPx: pxPerUnitAt(zoom) * kAtY(0) * dpr, sel: mute ? opt.selLine : -1 });
  rnd.oneWayTrack.draw(lc, vp, zoom, dpr, { wLine: wLine * 1.8, band: 0, mPx: pxPerUnitAt(zoom) * kAtY(0) * dpr, sel: -1, rgb: rgbf(shadeColours().mapInk) });
  if (rnd.preview) rnd.oneWayGhost.draw(lc, vp, zoom, dpr, { wLine: wLine * 1.8, band: 0, mPx: pxPerUnitAt(zoom) * kAtY(0) * dpr, sel: -1, rgb: rgbf(shadeColours().mapInk) });
  }
  if (cache) {
    if (redraw) {
      cache.end(0, vp);
      if (through) {
        cache.begin(1);
        gl.useProgram(rnd.strokeProg);
        trackStrokes(trackCap);
        gl.uniform1f(u.u_cap, 0);
        cache.end(1, vp);
      }
    }
    cache.draw(0);
  }
}

function stationsPass(rnd: NetworkRenderer, c: Ctx3) {
  const { zoom, dpr, opt, gl, n, uTime, lc, vp, wLine, wTrack, boxes, hidden, ring, boxLinePx, boxRingPx, pass, project, midX, midY, through, boxWhite, trackCap, cache, buses, stationsOff, u, lanes, shades, trackStrokes } = c;
  // the station buildings' hall crowds (T-2219), over the track and under the boxes
  if (!stationsOff) rnd.floors.drawHall(lc, vp, dpr, zoom, rnd.board && !hidden ? rnd.crowds.hall : null);

  if (n.stationCount && !stationsOff) {
    // Platforms along the track from STATION_BOX_ZOOM, each one shape at every zoom (T-204), dots
    // further out (T-189), one for a whole complex (T-217), and below STATIONS_ZOOM only the
    // selected ones (T-205).
    syncGroups(rnd, n, opt.groups ?? null);
    const also = (opt.selStations ?? []).filter((i) => i >= 0 && i < n.stationCount);
    const alsoKey = also.join(",");
    const changed = opt.selStation !== rnd.selectedStation || alsoKey !== rnd.selectedAlsoKey;
    rnd.selectedStation = opt.selStation;
    if (alsoKey !== rnd.selectedAlsoKey) {
      rnd.selectedAlso = new Set(also);
      rnd.selectedAlsoKey = alsoKey;
    }
    syncSelection(rnd, n);
    if (changed) rnd.onSelectionDrawn();
    const [r, rx] = rnd.stationStyle.radius.map((v) => v * dpr);
    const su = rnd.stationU;
    gl.useProgram(rnd.stationProg);
    setCamera(gl, su, lc);
    gl.uniform2f(su.u_viewport, vp[0], vp[1]);
    gl.uniform2f(su.u_radius, r, rx);
    gl.uniform1f(su.u_ring, ring * dpr);
    gl.uniform3fv(su.u_ink, rnd.ink);
    gl.uniform3fv(su.u_bad, BAD_RGB);
    gl.uniform3fv(su.u_crowd_rgb, CROWD_RGB);
    gl.uniform1f(su.u_line, wLine);
    gl.uniform3f(su.u_lanes, lanes.gap * dpr, lanes.t, lanes.narrow);
    gl.uniform1f(su.u_boxes, boxes ? 1 : 0);
    gl.uniform1i(su.u_no_bp, vis.blueprints ? 1 : 0);
    const hotSt = (opt.hotStations ?? []).filter((i) => i >= 0 && i < n.stationCount);
    rnd.hotList.fill(-1);
    hotSt.slice(0, HOT_STATIONS).forEach((i, k) => (rnd.hotList[k] = i));
    gl.uniform1iv(su.u_hot_st, rnd.hotList);
    gl.uniform1i(su.u_hot_n, Math.min(hotSt.length, HOT_STATIONS));
    // each platform's track line (T-254)
    gl.activeTexture(gl.TEXTURE11);
    gl.bindTexture(gl.TEXTURE_2D, rnd.tex.stationLine);
    gl.uniform1i(su.u_stline, 11);
    gl.activeTexture(gl.TEXTURE0);
    // local units per px at BOX_WIDTH_ZOOM (the box's width, T-204)
    const unitAtBox = unitPerPxAt(BOX_WIDTH_ZOOM) / dpr;
    gl.uniform2f(su.u_fixed, (boxHalfPxAtBoxZoom({ boxLinePx, boxRingPx }) - (opt.boxShrinkPx ?? 0)) * dpr, unitAtBox);
    gl.uniform1f(su.u_pass_half, (pass ? pass.half - (opt.boxShrinkPx ?? 0) : 0) * dpr);
    for (const [loc, c] of [[su.u_one, rnd.stationStyle.one], [su.u_one_mid, rnd.stationStyle.oneMid]] as const)
      gl.uniform4f(loc, (c?.inner ?? 0) * dpr, (c?.outer ?? 0) * dpr, (c?.innerTram ?? 0) * dpr, (c?.outerTram ?? 0) * dpr);
    // T-2249: a passing station's riders stand between its running track and the outer track's
    // inner edge (the outer track as the track strokes draw it)
    let passIn = 0;
    if (pass) {
      const G = lanes.gap * dpr, lt = lanes.share * G * wTrack / wLine;
      const wl = lanes.t > 0 ? 0.5 * (Math.max(wTrack, G + lt) - lanes.t * (G - lt)) : wTrack;
      passIn = Math.max(0, pass.off * dpr - 0.5 * Math.max(wl, dpr));
    }
    gl.uniform1f(su.u_pass_in, passIn);
    gl.bindVertexArray(rnd.stationVao);
    gl.uniform1f(su.u_pixel, dpr);
    gl.uniform1i(su.u_layer, 0);
    // the player's links as bridges between platforms (T-217)
    const bridges = boxes && rnd.bridgeProg ? (n.stationLinkIdx?.length ?? 0) / 2 : 0;
    if (bridges) {
      useBridges(rnd, lc, vp, ring * dpr, BRIDGE_HALF_PX * dpr * unitAtBox);
      rnd.hotPairs(rnd.bridgeU, n.stationLinkIdx!, hotSt);
    }
    // zoomed out, the connectors between a complex's dots and the player's links (T-313, Anita:
    // "it should look like its made of the same material"): the dots' rim, and their white
    // drawn in the same layer, so dots and connectors are one white shape with one outline
    const connectors = !boxes && rnd.bridgeProg && rnd.connVao ? rnd.connPairs.length / 2 : 0;
    if (connectors) {
      const [x0, y0] = project(midX, midY), [x1, y1] = project(midX + 1, midY);
      const pxPerLoc = Math.max(1e-9, Math.hypot(x1 - x0, y1 - y0));
      useBridges(rnd, lc, vp, ring * dpr, (connectorHalfPx(rnd.stationStyle) - ring) / pxPerLoc);
      rnd.hotPairs(rnd.bridgeU, rnd.connPairs, hotSt);
    }
    const connDraw = (pass: number, layer: number) => {
      gl.useProgram(rnd.bridgeProg);
      gl.uniform1i(rnd.bridgeU.u_pass, pass);
      gl.uniform1i(rnd.bridgeU.u_layer, layer);
      gl.bindVertexArray(rnd.connVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, connectors);
    };
    // Riders waiting on the platforms (T-164, T-219), where platforms are drawn; zoomed out a pie
    // in each dot (T-228, T-553).
    rnd.crowdsShown = !!rnd.board && !hidden;
    let any = false;
    if (rnd.crowdsShown) {
      const now = performance.now();
      if (rnd.moving && now - rnd.crowdsAt < MOVING_PLACE_MS) any = rnd.crowdsAny;
      else {
        rnd.crowdsAt = now;
        any = rnd.crowdsAny = rnd.crowds.update(n, rnd.board!, uTime, (c, dc) => {
          if (rnd.crowdBuf) rewrite(rnd, rnd.crowdBuf, c);
          if (rnd.dotCrowdBuf) rewrite(rnd, rnd.dotCrowdBuf, dc);
        });
      }
    }
    const fills = any && boxes;
    gl.useProgram(rnd.stationProg);
    gl.uniform1f(su.u_tint, any && !boxes ? 1 : 0);
    gl.uniform1f(su.u_mask, 0);
    if (through || (boxes && rnd.hasStencil)) {
      gl.clearStencil(0);
      gl.clear(gl.STENCIL_BUFFER_BIT);
    }
    // T-2219: a station building's floor in the stencil, where its members' rims are not drawn
    const floored = boxes && rnd.hasStencil && rnd.floors.markStencil(lc);
    gl.useProgram(rnd.stationProg);
    // the annexes joined to the boxes (T-1416), in the same layers so box and annex are one
    // shape; no track runs through them
    const layerDraw = (pass: number, layer: number, withBridges: boolean, annexes = true) => {
      gl.useProgram(rnd.stationProg);
      gl.uniform1i(su.u_pass, pass);
      gl.uniform1i(su.u_layer, layer);
      gl.bindVertexArray(rnd.stationVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
      if (annexes && boxes && rnd.annexes) {
        for (const [kind, bit] of [[ANNEX_ROOM, ST_ROOM]]) {
          if (!(rnd.annexes & bit)) continue;
          gl.uniform1i(su.u_room, kind);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
        }
        gl.uniform1i(su.u_room, 0);
      }
      if (bridges && withBridges) {
        gl.useProgram(rnd.bridgeProg);
        gl.uniform1i(rnd.bridgeU.u_pass, pass);
        gl.uniform1i(rnd.bridgeU.u_layer, layer);
        gl.bindVertexArray(rnd.bridgeVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, bridges);
      }
    };
    // the dark shapes; a station building's members' only off its floor (T-2219)
    const rims = (pass: number) => {
      if (!floored) return layerDraw(pass, 0, true);
      gl.useProgram(rnd.stationProg);
      gl.uniform1i(su.u_bld, 1);
      layerDraw(pass, 0, true);
      gl.enable(gl.STENCIL_TEST);
      gl.stencilMask(0);
      gl.stencilFunc(gl.EQUAL, 0, STENCIL_BIT);
      gl.useProgram(rnd.stationProg);
      gl.uniform1i(su.u_bld, 2);
      layerDraw(pass, 0, false);
      gl.useProgram(rnd.stationProg);
      gl.uniform1i(su.u_bld, 0);
      gl.stencilMask(0xff);
      gl.disable(gl.STENCIL_TEST);
    };
    // every station, then the selected ones over them with their outline and glow; each pass the
    // dark shapes, then the white inside every rim, so stations drawn over each other are one white
    // shape with one outline (T-217), then the crowds and the track through them (T-236)
    for (const pass of [0, 1]) {
      if (pass === 0 && hidden) continue;
      if (pass === 1) {
        const shapes = rnd.selectedShapes(project);
        if (!shapes.length) break;
        // a selected building's outline goes round its floor, not its members' boxes (T-2219)
        if (floored) rnd.floors.drawSelected(new Set(rnd.selectedDrawn()), project, vp, dpr, zoom, rnd.ink);
        drawGlow(rnd, floored ? rnd.selectedShapes(project, true) : shapes, dpr, vp);
      }
      // zoomed out the connectors' white goes under the dots' white, so a dot's crowd pie (T-228)
      // stays whole (T-313)
      if (connectors) connDraw(pass, 0);
      rims(pass);
      if (connectors) connDraw(pass, 1);
      layerDraw(pass, 1, true);
      if (!boxes) continue;
      if (through) {
        // mark this pass's white, no colour
        gl.enable(gl.STENCIL_TEST);
        gl.stencilMask(~STENCIL_BIT & 0xff);
        gl.stencilFunc(gl.ALWAYS, pass + 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
        gl.colorMask(false, false, false, false);
        gl.useProgram(rnd.stationProg);
        gl.uniform1f(su.u_mask, 1);
        if (bridges) {
          gl.useProgram(rnd.bridgeProg);
          gl.uniform1f(rnd.bridgeU.u_mask, 1);
        }
        layerDraw(pass, 1, true, false);
        gl.useProgram(rnd.stationProg);
        gl.uniform1f(su.u_mask, 0);
        if (bridges) {
          gl.useProgram(rnd.bridgeProg);
          gl.uniform1f(rnd.bridgeU.u_mask, 0);
        }
        gl.colorMask(true, true, true, true);
        gl.stencilMask(0xff);
        gl.disable(gl.STENCIL_TEST);
      }
      if (fills) layerDraw(pass, 2, false);
      if (through) {
        gl.enable(gl.STENCIL_TEST);
        gl.stencilFunc(gl.EQUAL, pass + 1, ~STENCIL_BIT & 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        if (cache) cache.draw(1);
        else {
          gl.useProgram(rnd.strokeProg);
          trackStrokes(trackCap);
          gl.uniform1f(u.u_cap, 0);
        }
        gl.disable(gl.STENCIL_TEST);
      }
    }
    // riders changing trains walking between stations (T-1124), with the platform crowds
    const placed = rnd.crowds.stationOfQueue;
    if (rnd.crowdsShown && boxes && placed) {
      rnd.bubbles.sync(rnd.board, n, placed);
      rnd.bubbles.draw(lc, vp, uTime, 0.5 * boxWhite * dpr, rgbf(shades.walkers), rgbf(shades.casing), dpr);
    }
  }

  // a bus line's stops at train stations, over the stations (T-2948)
  if (buses && !stationsOff) rnd.buses.drawStationStops(lc, vp, dpr);
}

function trainsPass(rnd: NetworkRenderer, c: Ctx3) {
  const { cam, time, zoom, dpr, opt, gl, n, uTime, a, lc, vp, pxPerUnit, k, pass, project, lanes } = c;
  // No trains zoomed far out (T-205).
  rnd.trainsHidden = zoom < TRAINS_ZOOM;
  if (opt.showTrains && rnd.tripCount && !rnd.trainsHidden) {
    rnd.detailedTrains = zoom >= STATION_BOX_ZOOM && !opt.hollowTrains;
    const tw = interp(zoom, TRAIN_PX) * dpr * k;
    guardTrains(rnd, time, cam, a, project, vp[0] / dpr, vp[1] / dpr, zoom >= CONFLICT_ZOOM);
    gl.useProgram(rnd.trainProg);
    gl.uniform1i(rnd.trainU.u_hollow, opt.hollowTrains ? 1 : 0);
    gl.useProgram(rnd.trainProg);
    setCamera(gl, rnd.trainU, lc);
    gl.uniform2f(rnd.trainU.u_viewport, vp[0], vp[1]);
    gl.uniform1f(rnd.trainU.u_time, uTime);
    gl.uniform1f(rnd.trainU.u_px_per_unit, pxPerUnit);
    gl.uniform1f(rnd.trainU.u_psi0, anchorPsi(a));
    // a pill is never shorter than about 2.9 times its width
    gl.uniform2f(rnd.trainU.u_min_px, tw * 2.2 / 0.75, tw);
    bindTrainTextures(rnd, rnd.trainU);
    gl.uniform1i(rnd.trainU.u_line_colours, 2);
    gl.uniform1i(rnd.trainU.u_sel_line, opt.selLine);
    setSelLines(gl, rnd.trainU, opt.selLines);
    gl.uniform1f(rnd.trainU.u_dim, opt.dimOthers ?? 1);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, rnd.tex.fill);
    gl.uniform1i(rnd.trainU.u_fill, 6);
    gl.uniform3fv(rnd.trainU.u_bad, BAD_RGB);
    gl.uniform1i(rnd.trainU.u_detail, rnd.detailedTrains ? 1 : 0);
    gl.uniform1f(rnd.trainU.u_dpr, dpr);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, rnd.tex.cars);
    gl.uniform1i(rnd.trainU.u_cars, 7);
    gl.activeTexture(gl.TEXTURE8);
    gl.bindTexture(gl.TEXTURE_2D, rnd.loadTex);
    gl.uniform1i(rnd.trainU.u_loads, 8);
    gl.activeTexture(gl.TEXTURE9);
    gl.bindTexture(gl.TEXTURE_2D, rnd.tex.stretch);
    gl.uniform1i(rnd.trainU.u_stretch, 9);
    gl.activeTexture(gl.TEXTURE10);
    gl.bindTexture(gl.TEXTURE_2D, rnd.tex.lane);
    gl.uniform1i(rnd.trainU.u_lane_tex, 10);
    // on its own track close in (T-658): the shift at full parting, how far parted
    gl.uniform2f(rnd.trainU.u_lane, lanes.gap * dpr * 0.5, lanes.t);
    // on its line's band (T-1799)
    gl.uniform1f(rnd.trainU.u_bands, rnd.bandPx * dpr);
    gl.uniform1f(rnd.trainU.u_bands_m, rnd.bandMPx * dpr);
    gl.uniform2f(rnd.trainU.u_passing, pass ? pass.off * dpr : 0, pass ? 1 : 0);
    gl.uniform1f(rnd.trainU.u_keep, rnd.net?.keep ?? 1);
    // T-3420: cars on a hidden level, by their sample's height
    gl.uniform1i(rnd.trainU.u_level_hide, vis.levels);
    if (vis.levels && !rnd.tex.slevel) rnd.putTexture("slevel", rnd.sampleLevels(n), "r32f", TEX_W);
    gl.activeTexture(gl.TEXTURE13);
    gl.bindTexture(gl.TEXTURE_2D, rnd.tex.slevel);
    gl.uniform1i(rnd.trainU.u_slevel, 13);
    gl.bindVertexArray(rnd.trainVao);
    // every train but the selected; then the selected one's halo, round the whole consist as one
    // pill; then the selected train over it (T-242)
    const sel = opt.selTrain;
    gl.uniform2f(rnd.trainU.u_sel_trip, sel ? sel.profile : -1, sel ? sel.dep - rnd.tripEpoch : 0);
    const halo = shadeColours();
    gl.uniform3fv(rnd.trainU.u_ink, rgbf(halo.mapInk));
    gl.uniform3fv(rnd.trainU.u_band, rgbf(halo.casing));
    gl.uniform1i(rnd.trainU.u_pass, 0);
    const drawTrains = () => gl.drawArraysInstanced(rnd.detailedTrains ? gl.TRIANGLES : gl.TRIANGLE_STRIP, 0, rnd.detailedTrains ? 6 * rnd.maxCars : 4, rnd.tripCount);
    drawTrains();
    if (sel) {
      gl.uniform1i(rnd.trainU.u_pass, 1);
      gl.uniform1i(rnd.trainU.u_detail, 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, rnd.tripCount);
      gl.uniform1i(rnd.trainU.u_pass, 2);
      gl.uniform1i(rnd.trainU.u_detail, rnd.detailedTrains ? 1 : 0);
      drawTrains();
    }
  }
}

export function drawFrame(rnd: NetworkRenderer, cam: Camera, time: number, zoom: number, dpr: number, opt: DrawOptions, view?: readonly [number, number]) {
  const c1 = frameSetup(rnd, cam, time, zoom, dpr, opt, view);
  if (!c1) return;
  beginStrokes(rnd, c1);
  const c2 = { ...c1, ...strokeUniforms(rnd, c1) };
  const c3 = { ...c2, ...strokeDraws(rnd, c2) };
  strokesPass(rnd, c3);
  stationsPass(rnd, c3);
  trainsPass(rnd, c3);
  const gl = c1.gl;
  gl.bindVertexArray(null);
  gl.activeTexture(gl.TEXTURE0);
  rnd.timer.end();
  rnd.quiet = false;
}

/** T-1965: hold trains in view back behind those ahead on their track (`TrainGuard`), and upload
 * their lags when they change. `w`, `h`: the view, CSS px; `close`: zoomed in to CONFLICT_ZOOM. */
export function guardTrains(rnd: NetworkRenderer, time: number, cam: Camera, a: Anchor, project: Project, w: number, h: number, close: boolean) {
  const t0 = performance.now();
  guardTrainsOn(rnd, time, cam, a, project, w, h, close);
  rnd.guardMs = performance.now() - t0;
}

export function guardTrainsOn(rnd: NetworkRenderer, time: number, cam: Camera, a: Anchor, globeProject: Project, w: number, h: number, close: boolean) {
  let changed = rnd.lagDirty;
  rnd.lagDirty = false;
  if (rnd.follow) {
    // the flat map's projection without an array a call (`projector` makes one each)
    const m = cam.globe ? null : mul4(cam.merc, localToMercMatrix(a)), out = rnd.guardOut;
    const project = m ? (x: number, y: number) => { clipToCss(m, x, y, w, h, out); return out; } : globeProject;
    changed = rnd.guard.update({ time, epoch: rnd.tripEpoch, trips: rnd.trips, ends: rnd.tripEnds, nexts: rnd.tripNexts, bays: rnd.tripBays, project, w, h, close }) || changed;
  } else if (rnd.guard.lag.some((v) => v !== 0)) {
    rnd.guard.lag.fill(0);
    changed = true;
  }
  if (!changed || rnd.guard.lag.length !== rnd.tripCount || !rnd.tripCount) return;
  const gl = rnd.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, rnd.lagBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, rnd.guard.lag);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

/** Select the bridge program and set its uniforms for this frame: `ring` and `half` (a bridge's
 * half width in local units) as the caller works them out. */
export function useBridges(rnd: NetworkRenderer, lc: LocalCamera, vp: readonly [number, number], ring: number, half: number) {
  const gl = rnd.gl, bu = rnd.bridgeU;
  gl.useProgram(rnd.bridgeProg);
  setCamera(gl, bu, lc);
  gl.uniform2f(bu.u_viewport, vp[0], vp[1]);
  gl.uniform1f(bu.u_ring, ring);
  gl.uniform1f(bu.u_bridge, half);
  gl.uniform3fv(bu.u_ink, rnd.ink);
  gl.uniform3fv(bu.u_bad, BAD_RGB);
  gl.uniform3fv(bu.u_crowd_rgb, CROWD_RGB);
  gl.uniform1f(bu.u_mask, 0);
  gl.uniform1i(bu.u_no_bp, vis.blueprints ? 1 : 0);
}

/** Local units of anchor `a` to CSS px on the canvas (y down) through the draw's camera; NaN
 * behind the globe's horizon. */
export function projector(_rnd: NetworkRenderer, cam: Camera, a: Anchor, w: number, h: number): Project {
  if (!cam.globe) {
    // the flat map: one matrix, as the shaders' u_matrix
    const m = mul4(cam.merc, localToMercMatrix(a));
    return (x, y) => {
      const out: [number, number] = [0, 0];
      clipToCss(m, x, y, w, h, out);
      return out;
    };
  }
  return (x, y) => {
    const out: [number, number] = [0, 0];
    mercToCss(cam, a.merc[0] + x / EARTH_C, a.merc[1] + y / EARTH_C, w, h, out);
    return out;
  };
}

/** The outline and glow round the selected shapes' union (T-189, T-217), under them. */
export function drawGlow(rnd: NetworkRenderer, shapes: StationShape[], dpr: number, vp: readonly [number, number]) {
  const gl = rnd.gl;
  const count = Math.min(shapes.length, SEL_MAX), d = rnd.selData;
  const reach = SEL_OUTLINE_PX + SEL_GLOW_PX + 1;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < count; i++) {
    const s = shapes[i];
    const o = i * SEL_STRIDE * 4;
    d.fill(0, o, o + SEL_STRIDE * 4);
    d.set([s.x, s.y, s.ux, s.uy, s.nx, s.ny, s.halfLength, s.halfWidth, s.dot, ...(s.bulge ? [s.bulge.cy, s.bulge.hl, s.bulge.hw] : [])], o);
    if (!(s.dot > 0)) d.set(shapePoints(s), o + 12);
    // every point of a shape is within its half length plus half width of its middle
    const r = (s.dot > 0 ? s.dot : Math.max(s.halfLength, s.bulge?.hl ?? 0) + s.halfWidth + (s.bulge ? s.bulge.hw + Math.abs(s.bulge.cy) : 0)) + reach;
    x0 = Math.min(x0, s.x - r); y0 = Math.min(y0, s.y - r); x1 = Math.max(x1, s.x + r); y1 = Math.max(y1, s.y + r);
  }
  const w = vp[0] / dpr, h = vp[1] / dpr;
  x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(w, x1); y1 = Math.min(h, y1);
  if (x1 <= x0 || y1 <= y0) return;
  gl.activeTexture(gl.TEXTURE10);
  gl.bindTexture(gl.TEXTURE_2D, rnd.selTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, count * SEL_STRIDE, 1, gl.RGBA, gl.FLOAT, d, 0);
  const u = rnd.selU;
  gl.useProgram(rnd.selProg);
  gl.uniform1i(u.u_shapes, 10);
  gl.uniform1i(u.u_count, count);
  gl.uniform4f(u.u_bbox, x0, y0, x1, y1);
  gl.uniform2f(u.u_css, w, h);
  gl.uniform1f(u.u_dpr, dpr);
  gl.uniform1f(u.u_height, vp[1]);
  gl.uniform3fv(u.u_ink, rnd.ink);
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
export function syncGroups(rnd: NetworkRenderer, n: NetworkBuffers, groups: StationGroups | null) {
  if (rnd.groupSrc.groups === groups && rnd.groupSrc.net === n) return;
  rnd.groupSrc = { groups, net: n };
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
  rnd.connPairs = Int32Array.from(pairs);
  const count = pairs.length / 2;
  const ce = (rnd.connEnds = new Float32Array(Math.max(1, count) * 4));
  for (let k = 0; k < count; k++) ce.set([...mid(pairs[k * 2]), ...mid(pairs[k * 2 + 1])], k * 4);
  // its own buffers and VAO, replaced with the connectors
  for (const r of rnd.connRes) r.del();
  rnd.connRes = [];
  rnd.connVao = null;
  rnd.connFlagBuf = null;
  if (rnd.gl && rnd.bridgeProg) {
    const gl = rnd.gl;
    const cv = (rnd.connVao = gl.createVertexArray()!);
    rnd.connRes.push({ del: () => gl.deleteVertexArray(cv) });
    gl.bindVertexArray(cv);
    attrib(rnd, rnd.bridgeProg, "a_ends", buffer(rnd, ce, rnd.connRes), 4, gl.FLOAT, 1);
    rnd.connFlagBuf = buffer(rnd, new Float32Array(Math.max(1, count) * 2), rnd.connRes);
    attrib(rnd, rnd.bridgeProg, "a_bflag", rnd.connFlagBuf, 2, gl.FLOAT, 1);
    gl.bindVertexArray(null);
  }
  rnd.selDrawnKey = "";
}

/** Which stations and bridges are drawn selected (`a_sel`, `a_bflag`), when that changed. */
export function syncSelection(rnd: NetworkRenderer, n: NetworkBuffers) {
  const key = `${rnd.selectedStation}/${rnd.selectedAlsoKey}/${vis.v}`;
  if (key === rnd.selDrawnKey) return;
  rnd.selDrawnKey = key;
  const flags = new Float32Array(Math.max(1, n.stationCount));
  for (let i = 0; i < n.stationCount; i++) flags[i] = rnd.stationHidden(i) ? -1 : rnd.drawnSelected(i) ? 1 : 0;
  if (rnd.stationSelBuf) rewrite(rnd, rnd.stationSelBuf, flags);
  // a bridge or connector is selected with both its stations, and faded when either is a blueprint
  const pairFlags = (idx: ArrayLike<number>) => {
    const b = new Float32Array(Math.max(2, idx.length));
    for (let k = 0; k < idx.length / 2; k++) {
      const i = idx[k * 2], j = idx[k * 2 + 1];
      b[k * 2] = rnd.drawnSelected(i) && rnd.drawnSelected(j) ? 1 : 0;
      b[k * 2 + 1] = (Math.round(n.stations[i * 3 + 2]) | Math.round(n.stations[j * 3 + 2])) & 2 ? 1 : 0;
    }
    return b;
  };
  const idx = n.stationLinkIdx;
  if (idx?.length && rnd.bridgeFlagBuf) rewrite(rnd, rnd.bridgeFlagBuf, pairFlags(idx));
  if (rnd.connPairs.length && rnd.connFlagBuf) rewrite(rnd, rnd.connFlagBuf, pairFlags(rnd.connPairs));
}

export function bindTrainTextures(rnd: NetworkRenderer, u: Record<string, WebGLUniformLocation | null>) {
  const gl = rnd.gl;
  const bind = (unit: number, tex: WebGLTexture | null, loc: WebGLUniformLocation | null) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(loc, unit);
  };
  bind(3, rnd.tex.phases, u.u_phases);
  bind(4, rnd.tex.meta, u.u_meta);
  bind(0, rnd.tex.samples, u.u_samples);
  bind(1, rnd.tex.lines, u.u_lines);
  gl.uniform1i(u.u_back_base, rnd.net ? backBaseOf(rnd.net.lineTable, rnd.net.meta) : 0);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, rnd.tex.colours);
}
