// The Money tab's fare control (T-864, T-883, T-1112, T-1143): the fare for a trip (y) against its
// length (x), straight stretches through the player's three nodes (game/fareCurve.ts). Zoom changes
// the distance shown and the fare shown with it, at one fixed ratio, so the line's slope on screen
// never changes; the fare axis starts at $0 unless that would put the base fare high up the plot.
// The base fare drags up and down only, the other nodes one way or the other, chosen by the drag's
// first few pixels. A node past the plot is an arrow at its edge. A drag changes only the picture
// and the figures below it; the fare goes to the game once, on release, since each change starts
// the demand again.

import { signal } from "@preact/signals";
import { useRef, useState } from "preact/hooks";
import { fareAt, MAX_KM, NODE_GAP_KM, type FareNode } from "../game/fareCurve";
import { unitMetres, unitShort } from "../game/format";
import { canUndoFares, changeFares as setFares, undo } from "../game/fareUndo";
import { Icon, Stepper } from "./widgets";

/** The nodes as shown during a drag (km, US$): null when nothing is dragged. */
export const farePreview = signal<FareNode[] | null>(null);
/** The distance the graph shows, display units; null until the player zooms or drags: then it
 * fits the nodes. Kept while the game is open. */
const fareSpan = signal<number | null>(null);

const W = 300, H = 150, L = 6, R = 294, T = 10, B = 134;
const MIN_SPAN = 1;
/** The fare the plot's height shows per km of the distance shown, US$: the fixed ratio of the two
 * scales (the default view, 200 km, shows $30). */
const FARE_PER_KM_SHOWN = 0.15;
/** Where the base fare sits once the fare axis no longer starts at $0, as a share of the height. */
const BASE_AT = 0.25;
/** Pointer travel, in the graph's units, before a drag picks its direction. */
const DRAG_PICK = 3;

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
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const usd = (v: number) => "$" + v.toFixed(2);
const distText = (u: number) => (u < 10 ? (Math.round(u * 10) / 10).toFixed(1) : Math.round(u).toLocaleString("en-US"));
const maxSpan = (kpu: number) => MAX_KM / kpu;
const clampSpan = (s: number, kpu: number) => clamp(s, MIN_SPAN, maxSpan(kpu));

/** The fares the plot shows over a distance of `span` units: from `y0` up to `y0 + h`. */
function fareView(nodes: FareNode[], span: number, kpu: number): { y0: number; h: number } {
  const h = FARE_PER_KM_SHOWN * span * kpu;
  return { y0: Math.max(0, (nodes[0]?.[1] ?? 0) - BASE_AT * h), h };
}
/** Whether every node is in the view of `span` units, with room above. */
function holds(nodes: FareNode[], span: number, kpu: number): boolean {
  const { y0, h } = fareView(nodes, span, kpu);
  return nodes.every(([k, f]) => k / kpu <= span / 1.2 + 1e-9 && f <= y0 + 0.9 * h);
}
/** The smallest nice view at least `from` units that holds the nodes. */
function fitSpan(nodes: FareNode[], kpu: number, from = MIN_SPAN): number {
  let s = clampSpan(niceUp(from), kpu);
  while (s < maxSpan(kpu) && !holds(nodes, s, kpu)) s = clampSpan(niceUp(s * 1.01), kpu);
  return s;
}
const spanOf = (nodes: FareNode[], kpu: number) => fareSpan.value ?? fitSpan(nodes, kpu);

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

/** How far node `i` may go, km and US$: past the node before it and short of the one after, by
 * `gap` km, and never cheaper than the one before or dearer than the one after. */
function limits(nodes: FareNode[], i: number, gap = NODE_GAP_KM) {
  const prev = nodes[i - 1], next = nodes[i + 1];
  return {
    kLo: prev ? prev[0] + gap : 0, kHi: next ? next[0] - gap : MAX_KM,
    fLo: prev ? prev[1] : 0, fHi: next ? next[1] : Infinity,
  };
}

/** A rate per display unit steps by this, US$ (T-1201). */
const RATE_STEP = 0.005;
const cents = (v: number) => Math.round(v * 100) / 100;
const rateText = (r: number) => "$" + r.toFixed(3);

/** The rate per km of the stretch ending at node `i` (from 1), US$. */
const rateOf = (nodes: FareNode[], i: number) => (nodes[i][1] - nodes[i - 1][1]) / (nodes[i][0] - nodes[i - 1][0]);

/** Nodes from the rows' own values: the base fare, then each stretch's end (km) and rate per km.
 * Fares are cents, so rows left alone keep their nodes' fares. */
function fromRows(base: number, stretches: [number, number][]): FareNode[] {
  const out: FareNode[] = [[0, cents(Math.max(0, base))]];
  for (const [km, rate] of stretches) {
    const [pk, pf] = out[out.length - 1];
    out.push([km, Math.max(pf, cents(pf + Math.max(0, rate) * (km - pk)))]);
  }
  return out;
}

/** The rows under the graph (T-1201): the base fare, then each stretch's distances and rate per
 * unit, stepped or typed. A row's change keeps the other rows' values, so a rate or base change
 * moves every later node's fare by the same amount; a drag on the graph sets fares instead. */
export function FareNodeRows({ nodes }: { nodes: FareNode[] }) {
  const kpu = unitMetres() / 1000;
  const unit = unitShort();
  const stretches = nodes.slice(1).map((n, j) => [n[0], rateOf(nodes, j + 1)] as [number, number]);
  const base = nodes[0][1];
  const put = (b: number, s: [number, number][]) => void setFares(fromRows(b, s));
  const setBase = (v: number) => put(v, stretches);
  return (
    <ul class="rows">
      <li>
        <span class="grow">Base fare</span>
        <Stepper value={usd(base)} wide label="base fare" onStep={(s) => setBase(cents(base + s * 0.25))}
          type={{ text: base.toFixed(2), set: setBase, tip: "The fare for the shortest trip. Click to type an amount." }} />
      </li>
      {stretches.map(([km, rate], j) => {
        const i = j + 1;
        const lim = limits(nodes, i);
        const from = nodes[i - 1][0] / kpu, u = km / kpu;
        const fromText = from === 0 ? "0" : distText(from);
        const rateU = rate * kpu;
        const last = i === nodes.length - 1;
        const setU = (v: number) => { if (lim.kLo <= lim.kHi) put(base, stretches.map((s, k) => (k === j ? [clamp(v * kpu, lim.kLo, lim.kHi), s[1]] : s))); };
        const uStep = u < 10 ? 0.5 : u < 100 ? 5 : 50;
        const withRate = (r: number) => fromRows(base, stretches.map((s, k) => (k === j ? [s[0], Math.max(0, r) / kpu] : s)));
        const setRate = (v: number) => void setFares(withRate(Math.round(v * 1000) / 1000));
        const stepRate = (s: number) => {
          const at = rateU / RATE_STEP;
          const next = withRate((s > 0 ? Math.floor(at + 1e-6) + 1 : Math.ceil(at - 1e-6) - 1) * RATE_STEP);
          // a stretch too short for the step to move its fare a cent moves it a cent
          if (next[i][1] === nodes[i][1] && (s > 0 || nodes[i][1] > nodes[i - 1][1])) {
            const dc = s > 0 ? 0.01 : -0.01;
            for (let k = i; k < next.length; k++) next[k][1] = cents(nodes[k][1] + dc);
          }
          if (JSON.stringify(next) !== JSON.stringify(nodes)) void setFares(next);
        };
        const span = last ? `past ${fromText} ${unit}` : `from ${fromText} to ${distText(u)} ${unit}`;
        return (
          <li>
            {/* the last node's distance changes no fare, since its rate holds past it: it is only dragged */}
            {last
              ? <span class="grow">Past {fromText} {unit}</span>
              : <>
                  <Stepper value={`${fromText}-${distText(u)} ${unit}`} wide label={`end of stretch ${i}`} onStep={(s) => setU(snap(u + s * uStep, uStep))}
                    type={{ text: distText(u), set: setU, tip: `Where this rate ends and the next begins, ${unit}. Click to type a distance.` }} />
                  <span class="grow" />
                </>}
            <Stepper value={`+${rateText(rateU)} per ${unit}`} wide label={`rate of stretch ${i}`} onStep={stepRate}
              type={{ text: rateU.toFixed(3), set: setRate, tip: `The fare rises ${rateText(rateU)} for each ${unit} ${span}. Click to type a rate.` }} />
          </li>
        );
      })}
    </ul>
  );
}

interface Drag {
  i: number;
  /** the one way the node moves: picked by the drag's first `DRAG_PICK` units; always "y" for the base fare */
  axis: "x" | "y" | null;
  /** where the pointer went down, and the node's distance (units) and fare then */
  px: number; py: number; u0: number; f0: number;
  /** the scale, held still during the drag */
  span: number; y0: number; h: number;
}

export function FareGraph({ nodes }: { nodes: FareNode[] }) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const pinch = useRef<{ ids: Map<number, { x: number; y: number }>; d0: number; span0: number } | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverNode, setHoverNode] = useState<number | null>(null);
  const shown = farePreview.value ?? nodes;
  const kpu = unitMetres() / 1000;
  const unit = unitShort();
  const d = drag.current;
  const span = d?.span ?? spanOf(shown, kpu);
  const view = d ?? fareView(shown, span, kpu);
  const { y0, h } = view;
  const fareU = (u: number) => fareAt(shown, u * kpu);
  const x = (u: number) => L + (u / span) * (R - L);
  const y = (v: number) => B - ((v - y0) / h) * (B - T);
  const unitAt = (px: number) => ((px - L) / (R - L)) * span;

  // the curve is straight between its corners: 0, the nodes and the right edge
  const corners = [...new Set([...shown.map((n) => n[0] / kpu), span])].filter((u) => u >= 0 && u <= span).sort((a, b) => a - b);
  const curve = corners.map((u, i) => `${i ? "L" : "M"}${x(u).toFixed(1)},${y(fareU(u)).toFixed(1)}`).join("");

  const pointer = (e: PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * W) / r.width, y: ((e.clientY - r.top) * H) / r.height };
  };
  const onNode = (i: number) => (e: PointerEvent) => {
    if (e.button !== 0 || pinch.current) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const p = pointer(e);
    drag.current = { i, axis: i === 0 ? "y" : null, px: p.x, py: p.y, u0: nodes[i][0] / kpu, f0: nodes[i][1], span, y0, h };
    fareSpan.value = span;
    setHoverX(null);
    e.preventDefault();
  };
  const move = (e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = pointer(e);
    const dx = p.x - d.px, dy = p.y - d.py;
    if (!d.axis) {
      if (Math.hypot(dx, dy) < DRAG_PICK) return;
      d.axis = Math.abs(dx) >= Math.abs(dy) ? "x" : "y";
    }
    const ns = nodes.map((n) => [...n] as FareNode);
    const lim = limits(ns, d.i, Math.max(NODE_GAP_KM, 0.02 * d.span * kpu));
    if (d.axis === "x") {
      const hi = Math.min(d.span, lim.kHi / kpu), lo = lim.kLo / kpu;
      ns[d.i][0] = (lo > hi ? d.u0 : clamp(snap(d.u0 + (dx / (R - L)) * d.span, niceUp(d.span / 200)), lo, hi)) * kpu;
    } else {
      const raw = snap(d.f0 - (dy / (B - T)) * d.h, Math.max(0.01, niceUp(d.h / 250)));
      ns[d.i][1] = clamp(raw, lim.fLo, Math.min(lim.fHi, d.y0 + d.h));
    }
    farePreview.value = ns;
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
  const fStep = niceUp(h / 4);
  const yTicks: number[] = [];
  for (let v = Math.floor(y0 / fStep + 1) * fStep; v <= y0 + h + 1e-9; v += fStep) yTicks.push(v);
  const tickText = (v: number) => (xStep < 1 ? v.toFixed(xStep < 0.1 ? 2 : 1) : v.toLocaleString("en-US"));
  const fareText = (v: number) => "$" + (fStep < 1 ? v.toFixed(2) : v.toLocaleString("en-US"));
  const lastX = xTicks.length - 1;

  // the ways a hovered or dragged node moves, as small arrows around it
  const hinted = d ? d.i : hoverNode;
  const hints = (cx: number, cy: number, i: number) => {
    const axes = d && d.i === i && d.axis ? [d.axis] : i === 0 ? ["y"] : ["x", "y"];
    const dirs = axes.flatMap((a) => (a === "x" ? [[1, 0], [-1, 0]] : [[0, 1], [0, -1]]));
    return dirs.map(([ux, uy]) => {
      const tx = cx + 13 * ux, ty = cy + 13 * uy, bx = cx + 9 * ux, by = cy + 9 * uy;
      return <path class="fg-hint" d={`M${tx.toFixed(1)},${ty.toFixed(1)}L${(bx - 3 * uy).toFixed(1)},${(by + 3 * ux).toFixed(1)}L${(bx + 3 * uy).toFixed(1)},${(by - 3 * ux).toFixed(1)}Z`} />;
    });
  };

  return (
    <svg class={"fare-graph" + (d?.axis ? ` fg-drag-${d.axis}` : "")} ref={svg} viewBox={`0 0 ${W} ${H}`} onWheel={onWheel}>
      <defs><clipPath id="fare-clip"><rect x={L} y={T} width={R - L} height={B - T} /></clipPath></defs>
      {xTicks.map((t) => <line class="fg-grid" x1={x(t)} x2={x(t)} y1={T} y2={B} />)}
      {yTicks.map((v) => <line class="fg-grid" x1={L} x2={R} y1={y(v)} y2={y(v)} />)}
      <line class="fg-axis" x1={L} x2={R} y1={B} y2={B} />
      <line class="fg-axis" x1={L} x2={L} y1={T} y2={B} />
      <text class="fg-label" x={L} y={B + 12} text-anchor="middle">0</text>
      {xTicks.map((t, i) => (
        <text class="fg-label" x={x(t)} y={B + 12} text-anchor={i === lastX && x(t) > R - 30 ? "end" : "middle"}>{tickText(t) + (i === lastX ? ` ${unit}` : "")}</text>
      ))}
      {/* a label the base fare's node would cover is left out */}
      {yTicks.filter((v) => Math.abs(y(v) - y(shown[0][1])) > 10).map((v) => <text class="fg-label" x={L + 3} y={y(v) - 2}>{fareText(v)}</text>)}
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
        const tip = i === 0 ? `Base fare: ${usd(f)}\nDrag up or down to change it` : `${distText(km / kpu)} ${unit}: ${usd(f)}\nDrag sideways for the distance, up or down for the fare`;
        if (cx <= R + 0.5 && cy >= T - 0.5) {
          return (
            <g class={"fg-node" + (i === 0 ? " ns" : "")} onPointerDown={onNode(i)} onPointerEnter={() => setHoverNode(i)} onPointerLeave={() => setHoverNode(null)} {...dragProps}>
              {hinted === i && hints(cx, cy, i)}
              <circle class="fg-hit" cx={cx} cy={cy} r={11} data-tip={d ? undefined : tip} />
              <circle class={"fg-knob" + (d?.i === i ? " sel" : "")} cx={cx} cy={cy} r={4.5} />
            </g>
          );
        }
        // past the plot: an arrow at the edge pointing to it, a click zooms out to show it
        const px = Math.min(R, cx), py = Math.max(T, cy);
        const a = Math.atan2(cy - py, cx - px), c = Math.cos(a), s = Math.sin(a);
        const bx = px - 8 * c, by = py - 8 * s;
        return (
          <g class="fg-off" data-tip={`${tip.split("\n")[0]}\nClick to zoom out to it`} onClick={() => { fareSpan.value = fitSpan([shown[0], [km, f]], kpu, span * 1.01); }}>
            <circle class="fg-hit" cx={bx} cy={by} r={10} />
            <path class="fg-arrow" d={`M${px.toFixed(1)},${py.toFixed(1)}L${(bx - 4.5 * s).toFixed(1)},${(by + 4.5 * c).toFixed(1)}L${(bx + 4.5 * s).toFixed(1)},${(by - 4.5 * c).toFixed(1)}Z`} />
          </g>
        );
      })}
    </svg>
  );
}
