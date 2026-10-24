// The Money tab's fare control (T-864, T-883, T-1112): the fare for a trip (y) against its length
// (x), both linear from 0 at the bottom left, the straight line through the player's two nodes
// (game/fareCurve.ts). Zoom changes the distance shown; the fare axis fits the curve over it. A
// node drags both ways, the second kept farther and never cheaper; a node past the plot is an
// arrow at its edge. A drag changes only the picture and the figures below it; the fare goes to
// the game once, on release, since each change starts the demand again.

import { signal } from "@preact/signals";
import { useRef, useState } from "preact/hooks";
import { fareAt, fitOrigin, MAX_KM, MIN_KM, NODE_GAP_KM, type FareNode } from "../game/fareCurve";
import { unitMetres, unitShort } from "../game/format";
import { canUndoFares, changeFares as setFares, undo } from "../game/fareUndo";
import { Icon, Stepper } from "./widgets";

/** The nodes as shown during a drag (km, US$): null when nothing is dragged. */
export const farePreview = signal<FareNode[] | null>(null);
/** The distance the graph shows, display units; null until the player zooms or drags: then it
 * fits both nodes. Kept while the game is open. */
const fareSpan = signal<number | null>(null);

const W = 300, H = 150, L = 6, R = 294, T = 10, B = 134;
const MIN_SPAN = 1;
/** Room above the curve's highest fare in view. */
const HEADROOM = 1.25;

/** The smallest 1, 2 or 5 times a power of ten that is at least `v`. */
function niceUp(v: number): number {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
  return ([1, 2, 5, 10].find((s) => s * e >= v * (1 - 1e-9)) ?? 10) * e;
}
/** The largest 1, 2 or 5 times a power of ten under `v`. */
function niceDown(v: number): number {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
  return ([5, 2, 1, 0.5].find((s) => s * e < v * (1 - 1e-9)) ?? 0.5) * e;
}
const snap = (v: number, step: number) => Math.round(v / step) * step;

const usd = (v: number) => "$" + v.toFixed(2);
const distText = (u: number) => (u < 10 ? (Math.round(u * 10) / 10).toFixed(1) : Math.round(u).toLocaleString("en-US"));
const maxSpan = (kpu: number) => MAX_KM / kpu;
const clampSpan = (s: number, kpu: number) => Math.min(maxSpan(kpu), Math.max(MIN_SPAN, s));
/** The view that holds both nodes with room to spare. */
const fitSpan = (nodes: FareNode[], kpu: number) => clampSpan(niceUp((nodes[nodes.length - 1][0] / kpu) * 1.2), kpu);
const spanOf = (nodes: FareNode[], kpu: number) => fareSpan.value ?? fitSpan(nodes, kpu);
/** The fare axis's top and tick step over a view of `span` units: the curve's fare at the right
 * edge (its highest in view, as the curve never falls) with headroom, up to a whole step. */
function fareAxis(nodes: FareNode[], span: number, kpu: number): { top: number; step: number } {
  const raw = Math.max(0.5, HEADROOM * fareAt(nodes, span * kpu));
  const step = niceUp(raw / 4);
  return { top: Math.ceil(raw / step - 1e-9) * step, step };
}

/** The undo and zoom buttons beside the graph's heading. */
export function FareButtons({ nodes }: { nodes: FareNode[] }) {
  const kpu = unitMetres() / 1000;
  const span = spanOf(nodes, kpu);
  return (
    <span class="fg-buttons">
      <button class="btn icon" aria-label="Undo the fare change" data-tip="Undo the last fare change (Ctrl+Z)" disabled={!canUndoFares.value} onClick={() => void undo()}>
        <Icon.undo />
      </button>
      <span class="group">
        <button class="btn" aria-label="Zoom out" data-tip="Zoom out" disabled={span >= maxSpan(kpu)} onClick={() => { fareSpan.value = clampSpan(niceUp(span * 1.01), kpu); }}>
          <Icon.minus />
        </button>
        <button class="btn" aria-label="Zoom in" data-tip="Zoom in" disabled={span <= MIN_SPAN} onClick={() => { fareSpan.value = clampSpan(niceDown(span), kpu); }}>
          <Icon.plus />
        </button>
      </span>
    </span>
  );
}

/** The two nodes under the graph, one row each: its distance and its fare, stepped or typed. */
export function FareNodeRows({ nodes }: { nodes: FareNode[] }) {
  const kpu = unitMetres() / 1000;
  const unit = unitShort();
  return (
    <ul class="rows">
      {nodes.slice(0, 2).map(([km, fare], i) => {
        const other = nodes[1 - i];
        const u = km / kpu;
        const lo = i === 0 ? MIN_KM / kpu : (other[0] + NODE_GAP_KM) / kpu;
        const hi = i === 0 ? (other[0] - NODE_GAP_KM) / kpu : MAX_KM / kpu;
        const fLo = i === 0 ? 0 : other[1], fHi = i === 0 ? other[1] : Infinity;
        const put = (n: FareNode) => setFares(fitOrigin(i === 0 ? [n, other] : [other, n], i));
        const setU = (v: number) => { if (lo <= hi) void put([Math.min(hi, Math.max(lo, v)) * kpu, fare]); };
        const setFare = (v: number) => void put([km, Math.min(fHi, Math.max(fLo, Math.round(v * 100) / 100))]);
        const uStep = u < 10 ? 0.5 : u < 100 ? 5 : 50;
        return (
          <li>
            <Stepper value={`${distText(u)} ${unit}`} wide label={i === 0 ? "first node's distance" : "second node's distance"} onStep={(s) => setU(snap(u + s * uStep, uStep))}
              type={{ text: distText(u), set: setU, tip: `A trip length the curve passes through, ${unit}. Click to type one.` }} />
            <span class="grow fg-costs">costs</span>
            <Stepper value={usd(fare)} wide label={i === 0 ? "first node's fare" : "second node's fare"} onStep={(s) => setFare(fare + s * 0.25)}
              type={{ text: fare.toFixed(2), set: setFare, tip: "The fare for a trip of this length. Click to type an amount." }} />
          </li>
        );
      })}
    </ul>
  );
}

export function FareGraph({ nodes }: { nodes: FareNode[] }) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ i: number; top: number; step: number; span: number } | null>(null);
  const pinch = useRef<{ ids: Map<number, { x: number; y: number }>; d0: number; span0: number } | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const shown = farePreview.value ?? nodes;
  const kpu = unitMetres() / 1000;
  const unit = unitShort();
  const span = drag.current?.span ?? spanOf(shown, kpu);
  // during a drag the scale stays put under the pointer
  const axis = fareAxis(shown, span, kpu);
  const top = drag.current?.top ?? axis.top;
  const fStep = drag.current?.step ?? axis.step;
  const fareU = (u: number) => fareAt(shown, u * kpu);
  const x = (u: number) => L + (u / span) * (R - L);
  const y = (v: number) => B - (v / top) * (B - T);
  const unitAt = (px: number) => ((px - L) / (R - L)) * span;

  // the curve is straight between its corners: 0, where it leaves $0, the nodes and the right edge
  const [[k0, f0], [k1, f1]] = [shown[0], shown[shown.length - 1]];
  const slope = k1 > k0 ? (f1 - f0) / (k1 - k0) : 0;
  const leaves = slope > 0 ? (k0 - f0 / slope) / kpu : -1;
  const corners = [0, leaves, ...shown.map((n) => n[0] / kpu), span].filter((u) => u >= 0 && u <= span).sort((a, b) => a - b);
  const curve = corners.map((u, i) => `${i ? "L" : "M"}${x(u).toFixed(1)},${y(fareU(u)).toFixed(1)}`).join("");

  const pointer = (e: PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * W) / r.width, y: ((e.clientY - r.top) * H) / r.height };
  };
  const onNode = (i: number) => (e: PointerEvent) => {
    if (e.button !== 0 || pinch.current) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { i, top, step: fStep, span };
    fareSpan.value = span;
    setHoverX(null);
    e.preventDefault();
  };
  const move = (e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = pointer(e);
    const ns = nodes.map((n) => [...n] as FareNode);
    const other = ns[1 - d.i];
    const gap = Math.max(NODE_GAP_KM, 0.02 * d.span * kpu) / kpu;
    const lo = d.i === 0 ? MIN_KM / kpu : other[0] / kpu + gap;
    const hi = Math.min(d.span, d.i === 0 ? other[0] / kpu - gap : MAX_KM / kpu);
    const u = lo > hi ? ns[d.i][0] / kpu : Math.min(hi, Math.max(lo, snap(unitAt(p.x), niceUp(d.span / 200))));
    const raw = snap(Math.max(0, Math.min(1, (B - p.y) / (B - T))) * d.top, Math.max(0.01, niceUp(d.top / 250)));
    const fare = d.i === 0 ? Math.min(raw, other[1]) : Math.max(raw, other[1]);
    ns[d.i] = [u * kpu, fare];
    farePreview.value = fitOrigin(ns, d.i);
  };
  const end = () => {
    const d = drag.current, f = farePreview.value;
    drag.current = null;
    if (!d || !f || JSON.stringify(f) === JSON.stringify(nodes)) { farePreview.value = null; return; }
    // the figures keep the dragged values until the game has taken them
    void setFares(f).then(() => { if (!drag.current) farePreview.value = null; });
  };
  const dragProps = { onPointerMove: move, onPointerUp: end, onPointerCancel: end };

  // zoom about the origin: the wheel, or two fingers apart and together
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    if (drag.current) return;
    const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
    fareSpan.value = clampSpan(span * Math.exp(dy * 0.0015), kpu);
  };
  const apart = (ids: Map<number, { x: number; y: number }>) => {
    const [a, b] = [...ids.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  const plotDown = (e: PointerEvent) => {
    if (e.pointerType !== "touch" || drag.current) return;
    const ids = pinch.current?.ids ?? new Map();
    ids.set(e.pointerId, { x: e.clientX, y: e.clientY });
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    pinch.current = { ids, d0: ids.size === 2 ? apart(ids) : 0, span0: span };
  };
  const plotMove = (e: PointerEvent) => {
    const pc = pinch.current;
    if (pc?.ids.has(e.pointerId)) {
      pc.ids.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pc.ids.size === 2 && pc.d0 > 0) fareSpan.value = clampSpan((pc.span0 * pc.d0) / Math.max(1, apart(pc.ids)), kpu);
    }
    if (!drag.current) setHoverX(Math.max(L, Math.min(R, pointer(e).x)));
  };
  const plotUp = (e: PointerEvent) => {
    const pc = pinch.current;
    if (!pc) return;
    pc.ids.delete(e.pointerId);
    if (pc.ids.size === 0) pinch.current = null;
    else pinch.current = { ids: pc.ids, d0: 0, span0: span };
  };

  const hoverU = hoverX === null ? null : unitAt(hoverX);
  const hoverTip = hoverU === null ? undefined : `${distText(hoverU)} ${unit}: ${usd(fareU(hoverU))}`;
  const xStep = niceUp(span / 5);
  const xTicks = Array.from({ length: Math.floor(span / xStep + 1e-9) }, (_, i) => (i + 1) * xStep);
  const yTicks = Array.from({ length: Math.round(top / fStep) }, (_, i) => (i + 1) * fStep).filter((v) => v <= top + 1e-9);
  const tickText = (v: number) => (xStep < 1 ? v.toFixed(xStep < 0.1 ? 2 : 1) : v.toLocaleString("en-US"));
  const fareText = (v: number) => "$" + (fStep < 1 ? v.toFixed(2) : v.toLocaleString("en-US"));
  const lastX = xTicks.length - 1;
  return (
    <svg class="fare-graph" ref={svg} viewBox={`0 0 ${W} ${H}`} onWheel={onWheel}>
      <defs><clipPath id="fare-clip"><rect x={L} y={T} width={R - L} height={B - T} /></clipPath></defs>
      {xTicks.map((t) => <line class="fg-grid" x1={x(t)} x2={x(t)} y1={T} y2={B} />)}
      {yTicks.map((v) => <line class="fg-grid" x1={L} x2={R} y1={y(v)} y2={y(v)} />)}
      <line class="fg-axis" x1={L} x2={R} y1={B} y2={B} />
      <line class="fg-axis" x1={L} x2={L} y1={T} y2={B} />
      <text class="fg-label" x={L} y={B + 12} text-anchor="middle">0</text>
      {xTicks.map((t, i) => (
        <text class="fg-label" x={x(t)} y={B + 12} text-anchor={i === lastX && x(t) > R - 30 ? "end" : "middle"}>{tickText(t) + (i === lastX ? ` ${unit}` : "")}</text>
      ))}
      {yTicks.map((v) => <text class="fg-label" x={L + 3} y={y(v) - 2}>{fareText(v)}</text>)}
      <path class="fg-curve" d={curve} clip-path="url(#fare-clip)" />
      <rect class="fg-hover" x={L} y={T} width={R - L} height={B - T} data-tip={hoverTip}
        onPointerDown={plotDown} onPointerMove={(e) => { if (drag.current) move(e); else plotMove(e); }}
        onPointerUp={(e) => { if (drag.current) end(); plotUp(e); }} onPointerCancel={(e) => { if (drag.current) end(); plotUp(e); }}
        onPointerLeave={() => setHoverX(null)} />
      {hoverU !== null && (
        <>
          <line class="fg-guide" x1={x(hoverU)} x2={x(hoverU)} y1={Math.max(T, y(fareU(hoverU)))} y2={B} />
          {y(fareU(hoverU)) >= T && <circle class="fg-dot" cx={x(hoverU)} cy={y(fareU(hoverU))} r={2.5} />}
        </>
      )}
      {shown.map(([km, f], i) => {
        const cx = x(km / kpu), cy = y(f);
        const tip = `${distText(km / kpu)} ${unit}: ${usd(f)}`;
        if (cx <= R + 0.5 && cy >= T - 0.5) {
          return (
            <g class="fg-node" onPointerDown={onNode(i)} {...dragProps}>
              <circle class="fg-hit" cx={cx} cy={cy} r={11} data-tip={`${tip}\nDrag to move`} />
              <circle class={"fg-knob" + (drag.current?.i === i ? " sel" : "")} cx={cx} cy={cy} r={4.5} />
            </g>
          );
        }
        // past the plot: an arrow at the edge pointing to it, a click zooms out to show it
        const px = Math.min(R, cx), py = Math.max(T, cy);
        const a = Math.atan2(cy - py, cx - px), c = Math.cos(a), s = Math.sin(a);
        const bx = px - 8 * c, by = py - 8 * s;
        return (
          <g class="fg-off" data-tip={`${tip}\nClick to zoom out to it`} onClick={() => { fareSpan.value = clampSpan(niceUp(Math.max(span * 1.01, (km / kpu) * 1.2)), kpu); }}>
            <circle class="fg-hit" cx={bx} cy={by} r={10} />
            <path class="fg-arrow" d={`M${px.toFixed(1)},${py.toFixed(1)}L${(bx - 4.5 * s).toFixed(1)},${(by + 4.5 * c).toFixed(1)}L${(bx + 4.5 * s).toFixed(1)},${(by - 4.5 * c).toFixed(1)}Z`} />
          </g>
        );
      })}
    </svg>
  );
}
