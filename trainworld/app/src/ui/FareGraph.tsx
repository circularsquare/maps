// The Money tab's fare control (T-864, T-883): the fare for a trip (y, log scale from $0.10)
// against its length (x, log scale, half a km or mile to 1000), the curve through the player's
// nodes (game/fareCurve.ts). A node drags both ways, kept between its neighbours in distance and
// in fare so the curve never falls; pressing on the plot adds a node on the curve there and drags
// it; a double-click on a node removes it, down to two. A drag only changes the picture and the
// figures beside it; the fare goes to the game once, when the pointer is released, since each
// change starts the demand again.

import { signal } from "@preact/signals";
import { useRef, useState } from "preact/hooks";
import { fareAt, type FareNode } from "../game/fareCurve";
import { unitMetres, unitShort } from "../game/format";
import { setFares } from "../game/world";
import { Icon, Stepper } from "./widgets";

/** The nodes as shown during a drag (km, US$): null when nothing is dragged. */
export const farePreview = signal<FareNode[] | null>(null);
/** The node the steppers under the graph edit. */
export const fareSelected = signal(0);

const W = 300, H = 150, L = 6, R = 294, T = 10, B = 134;
const XMIN = 0.5, XMAX = 1000;
const XDEC = Math.log10(XMAX / XMIN);
const STEPS = [1, 1.5, 2, 3, 4, 5, 6, 8, 10];
/** The bottom of the fare axis; a fare at or under it is drawn on the axis. */
const YMIN = 0.1;
/** The top is at least this, US$. */
const YTOP = 100;

/** The smallest of 1, 1.5, 2, 3, 4, 5, 6, 8 and 10 times a power of ten that is at least `v`. */
function niceTop(v: number): number {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
  return (STEPS.find((s) => s * e >= v - 1e-9) ?? 10) * e;
}
/** Two significant figures. */
const sig2 = (v: number) => { const e = Math.pow(10, Math.floor(Math.log10(v)) - 1); return Math.round(v / e) * e; };
/** A dragged fare: to 5 cents under $10, 50 cents under $100, then $5. */
const fareStep = (v: number) => { const s = v < 10 ? 0.05 : v < 100 ? 0.5 : 5; return Math.round(v / s) * s; };

const xOf = (u: number) => L + (Math.log10(Math.max(u, XMIN) / XMIN) / XDEC) * (R - L);
const unitAt = (x: number) => XMIN * Math.pow(10, ((x - L) / (R - L)) * XDEC);
const usd = (v: number) => "$" + v.toFixed(2);
export const distText = (u: number) => (u < 10 ? (Math.round(u * 10) / 10).toFixed(1) : Math.round(u).toLocaleString("en-US"));

/** The range a node may move in, display units, between its neighbours with room to spare. */
export function nodeRange(nodes: FareNode[], i: number, kpu: number): [number, number] {
  const gap = 0.15 / kpu;
  const lo = i > 0 ? Math.max(nodes[i - 1][0] / kpu * 1.04, nodes[i - 1][0] / kpu + gap) : XMIN;
  const hi = i + 1 < nodes.length ? Math.min(nodes[i + 1][0] / kpu / 1.04, nodes[i + 1][0] / kpu - gap) : XMAX;
  return [lo, hi];
}

/** Exact entry for the selected node under the graph: its distance and its fare. */
export function FareNodeRows({ nodes }: { nodes: FareNode[] }) {
  const kpu = unitMetres() / 1000;
  const i = Math.min(fareSelected.value, nodes.length - 1);
  const [km, fare] = nodes[i];
  const u = km / kpu;
  const [lo, hi] = nodeRange(nodes, i, kpu);
  const fLo = i > 0 ? nodes[i - 1][1] : 0, fHi = i + 1 < nodes.length ? nodes[i + 1][1] : Infinity;
  const put = (n: FareNode) => setFares(nodes.map((x, k) => (k === i ? n : x)));
  const setU = (v: number) => { if (lo <= hi) void put([Math.min(hi, Math.max(lo, v)) * kpu, fare]); };
  const setFare = (v: number) => void put([km, Math.min(fHi, Math.max(fLo, Math.round(v * 100) / 100))]);
  const uStep = u < 10 ? 0.5 : u < 100 ? 5 : 50;
  return (
    <ul class="rows">
      <li>
        <span class="grow">Distance</span>
        {nodes.length > 2 && (
          <button class="btn" aria-label="Remove this node" data-tip="Remove this node" onClick={() => { fareSelected.value = Math.max(0, i - 1); void setFares(nodes.filter((_, k) => k !== i)); }}>
            <Icon.bin />
          </button>
        )}
        <Stepper value={`${distText(u)} ${unitShort()}`} wide label="node distance" onStep={(s) => setU(Math.round((u + s * uStep) / uStep) * uStep)}
          type={{ text: distText(u), set: setU, tip: `The trip length at this node, ${unitShort()}. Click to type one.` }} />
      </li>
      <li>
        <span class="grow">Fare</span>
        <Stepper value={usd(fare)} wide label="node fare" onStep={(s) => setFare(fare + s * 0.25)}
          type={{ text: fare.toFixed(2), set: setFare, tip: "The fare for a trip of this length. Click to type an amount." }} />
      </li>
    </ul>
  );
}

export function FareGraph({ nodes }: { nodes: FareNode[] }) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ i: number; top: number; start: FareNode[] } | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const shown = farePreview.value ?? nodes;
  const kpu = unitMetres() / 1000;
  const fareU = (u: number) => fareAt(shown, u * kpu);
  const sel = Math.min(fareSelected.value, shown.length - 1);
  // the top follows the committed curve, so a drag does not move the scale under the pointer
  const top = drag.current?.top ?? Math.max(YTOP, niceTop(1.5 * fareAt(nodes, XMAX * kpu)));
  const decades = Math.log10(top / YMIN);
  const yRaw = (v: number) => B - (Math.log10(Math.max(v, YMIN) / YMIN) / decades) * (B - T);
  const y = (v: number) => Math.max(T, yRaw(v));

  const curve = Array.from({ length: 97 }, (_, i) => {
    const u = XMIN * Math.pow(10, (i / 96) * XDEC);
    return `${i ? "L" : "M"}${xOf(u).toFixed(1)},${yRaw(fareU(u)).toFixed(1)}`;
  }).join("");

  const pointer = (e: PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * W) / r.width, y: ((e.clientY - r.top) * H) / r.height };
  };
  const startDrag = (e: PointerEvent, i: number, start: FareNode[]) => {
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { i, top, start };
    fareSelected.value = i;
    setHoverX(null);
    e.preventDefault();
  };
  const onNode = (i: number) => (e: PointerEvent) => {
    if (e.button !== 0) return;
    startDrag(e, i, nodes);
  };
  const remove = (i: number) => (e: MouseEvent) => {
    e.preventDefault();
    if (nodes.length <= 2) return;
    fareSelected.value = Math.max(0, i - 1);
    void setFares(nodes.filter((_, k) => k !== i));
  };
  const onPlot = (e: PointerEvent) => {
    if (e.button !== 0) return;
    const u = Math.min(XMAX, Math.max(XMIN, sig2(unitAt(pointer(e).x))));
    const km = u * kpu;
    const i = nodes.findIndex((n) => n[0] > km);
    const at = i < 0 ? nodes.length : i;
    const added: FareNode[] = [...nodes.slice(0, at), [km, Math.round(fareAt(nodes, km) * 100) / 100], ...nodes.slice(at)];
    const [lo, hi] = nodeRange(added, at, kpu);
    if (lo > hi) return; // no room beside its neighbours
    added[at] = [Math.min(hi, Math.max(lo, u)) * kpu, added[at][1]];
    farePreview.value = added;
    startDrag(e, at, added);
  };
  const move = (e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = pointer(e);
    const ns = d.start.map((n) => [...n] as FareNode);
    const [lo, hi] = nodeRange(ns, d.i, kpu);
    const u = lo > hi ? ns[d.i][0] / kpu : Math.min(hi, Math.max(lo, sig2(unitAt(p.x))));
    const fLo = d.i > 0 ? ns[d.i - 1][1] : 0, fHi = d.i + 1 < ns.length ? ns[d.i + 1][1] : Infinity;
    const raw = YMIN * Math.pow(d.top / YMIN, Math.max(0, Math.min(1, (B - p.y) / (B - T))));
    ns[d.i] = [u * kpu, Math.min(fHi, Math.max(fLo, fareStep(raw)))];
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

  const hoverU = hoverX === null ? null : unitAt(hoverX);
  const hoverTip = hoverU === null ? undefined : `${distText(hoverU)} ${unitShort()}: ${usd(fareU(hoverU))}\nClick to add a node`;
  const ticks = [1, 10, 100, 1000];
  const fareTicks = [1, 10, 100, 1000].filter((v) => v <= top);
  return (
    <svg class="fare-graph" ref={svg} viewBox={`0 0 ${W} ${H}`}>
      <defs><clipPath id="fare-clip"><rect x={L} y={T - 2} width={R - L} height={B - T + 2} /></clipPath></defs>
      {ticks.map((t) => <line class="fg-grid" x1={xOf(t)} x2={xOf(t)} y1={T} y2={B} />)}
      {fareTicks.map((v) => <line class="fg-grid" x1={L} x2={R} y1={y(v)} y2={y(v)} />)}
      <line class="fg-axis" x1={L} x2={R} y1={B} y2={B} />
      {ticks.map((t, i) => (
        <text class="fg-label" x={xOf(t)} y={B + 12} text-anchor={i === ticks.length - 1 ? "end" : "middle"}>{i === ticks.length - 1 ? `1,000 ${unitShort()}` : t}</text>
      ))}
      {fareTicks.map((v) => <text class="fg-label" x={L + 2} y={y(v) - 2}>{"$" + v.toLocaleString("en-US")}</text>)}
      <path class="fg-curve" d={curve} clip-path="url(#fare-clip)" />
      <rect class="fg-hover" x={L} y={T} width={R - L} height={B - T} data-tip={hoverTip} onPointerDown={onPlot} {...dragProps}
        onPointerMove={(e) => { if (drag.current) { move(e); return; } setHoverX(Math.max(L, Math.min(R, pointer(e).x))); }}
        onPointerLeave={() => setHoverX(null)} />
      {hoverU !== null && fareU(hoverU) <= top && (
        <>
          <line class="fg-guide" x1={xOf(hoverU)} x2={xOf(hoverU)} y1={y(fareU(hoverU))} y2={B} />
          <circle class="fg-dot" cx={xOf(hoverU)} cy={y(fareU(hoverU))} r={2.5} />
        </>
      )}
      {shown.map(([km, f], i) => {
        const cx = xOf(km / kpu), cy = y(f);
        return (
          <g class="fg-node" onPointerDown={onNode(i)} onDblClick={remove(i)} {...dragProps}>
            <circle class="fg-hit" cx={cx} cy={cy} r={11} data-tip={`${distText(km / kpu)} ${unitShort()}: ${usd(f)}\nDrag to move${shown.length > 2 ? ", double-click to remove" : ""}`} />
            <circle class={"fg-knob" + (i === sel ? " sel" : "")} cx={cx} cy={cy} r={4.5} />
          </g>
        );
      })}
    </svg>
  );
}
