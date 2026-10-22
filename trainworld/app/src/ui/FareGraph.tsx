// The Money tab's fare control (T-864): the fare for a ride (y, log scale from $0.10) against its
// distance (x, log scale, 1 to 1000 km or miles), the fare curve drawn, and two nodes dragged up or down: the left one
// sits at 1 distance unit and moves the base fare (the curve shifts, its slope kept), the right one at
// 100 units and moves the growth rate per distance unit (the curve pivots on its start). A drag only
// changes the picture and the figures beside it; the fare goes to the game once, when the pointer is
// released, since each change starts the demand again.

import { signal } from "@preact/signals";
import { useRef, useState } from "preact/hooks";
import { distUnit, perDistance, perKmFrom, unitShort } from "../game/format";
import { setFares } from "../game/world";

/** The fare curve as shown during a drag, in the shown unit: null when nothing is dragged. */
export const farePreview = signal<{ base: number; perKm: number } | null>(null);

const W = 300, H = 150, L = 6, R = 294, T = 10, B = 134;
const DECADES = 3;
const NODE_BASE_U = 1, NODE_GROWTH_U = 100;
const STEPS = [1, 1.5, 2, 3, 4, 5, 6, 8, 10];
/** The bottom of the fare axis; a fare at or under it is drawn on the axis. */
const YMIN = 0.1;

/** The smallest of 1, 1.5, 2, 3, 4, 5, 6, 8 and 10 times a power of ten that is at least `v`. */
function niceTop(v: number): number {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
  return (STEPS.find((s) => s * e >= v - 1e-9) ?? 10) * e;
}

const xOf = (u: number) => L + (Math.log10(Math.max(u, 1)) / DECADES) * (R - L);
const distAt = (x: number) => Math.pow(10, ((x - L) / (R - L)) * DECADES);
const usd = (v: number) => "$" + v.toFixed(2);
const dist = (u: number) => (u < 10 ? u.toFixed(1) : Math.round(u).toLocaleString("en-US"));
const unitCents = (v: number) => Math.max(0, Math.round(v * 100) / 100);

export function FareGraph({ base, perKm }: { base: number; perKm: number }) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ node: "base" | "growth"; top: number; base: number; perKm: number; perUnit: number } | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const p = farePreview.value;
  const shownBase = p ? p.base : base;
  const perUnit = p ? perDistance(p.perKm) : perDistance(perKm);
  const unit = distUnit();
  const fareAt = (u: number) => shownBase + perUnit * u;
  // the top follows the committed curve, so a drag does not move the scale under the pointer
  const top = drag.current?.top ?? Math.max(10, niceTop(2 * (base + perDistance(perKm) * NODE_GROWTH_U)));
  const decades = Math.log10(top / YMIN);
  const yRaw = (v: number) => B - (Math.log10(Math.max(v, YMIN) / YMIN) / decades) * (B - T);
  const y = (v: number) => Math.max(T, yRaw(v));

  const curve = Array.from({ length: 73 }, (_, i) => {
    const u = Math.pow(10, (i / 72) * DECADES);
    return `${i ? "L" : "M"}${xOf(u).toFixed(1)},${yRaw(fareAt(u)).toFixed(1)}`;
  }).join("");

  const fareFromPointer = (e: PointerEvent, top: number) => {
    const r = svg.current!.getBoundingClientRect();
    const vy = ((e.clientY - r.top) * H) / r.height;
    return YMIN * Math.pow(top / YMIN, Math.max(0, Math.min(1, (B - vy) / (B - T))));
  };
  const start = (node: "base" | "growth") => (e: PointerEvent) => {
    if (e.button !== 0) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { node, top, base, perKm, perUnit: perDistance(perKm) };
    setHoverX(null);
    e.preventDefault();
  };
  const move = (e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const f = fareFromPointer(e, d.top);
    if (d.node === "base") {
      const b = Math.max(0, Math.round((f - d.perUnit * NODE_BASE_U) * 20) / 20);
      farePreview.value = { base: b, perKm: d.perKm };
    } else {
      const g = unitCents((f - d.base) / NODE_GROWTH_U);
      farePreview.value = { base: d.base, perKm: unitCents(perKmFrom(g)) };
    }
  };
  const end = () => {
    const d = drag.current, f = farePreview.value;
    drag.current = null;
    if (!d || !f) { farePreview.value = null; return; }
    // the figures keep the dragged values until the game has taken them
    void setFares(f.base, f.perKm).then(() => { if (!drag.current) farePreview.value = null; });
  };
  const nodeProps = (node: "base" | "growth") => ({ onPointerDown: start(node), onPointerMove: move, onPointerUp: end, onPointerCancel: end });

  const hoverU = hoverX === null ? null : distAt(hoverX);  const hoverTip = hoverU === null ? undefined : `${dist(hoverU)} ${unitShort()}: ${usd(fareAt(hoverU))}`;
  const ax = xOf(NODE_BASE_U), bx = xOf(NODE_GROWTH_U);
  const ay = y(fareAt(NODE_BASE_U)), by = y(fareAt(NODE_GROWTH_U));
  const ticks = [1, 10, 100, 1000];
  return (
    <svg class="fare-graph" ref={svg} viewBox={`0 0 ${W} ${H}`}>
      <defs><clipPath id="fare-clip"><rect x={L} y={T - 2} width={R - L} height={B - T + 2} /></clipPath></defs>
      {ticks.map((t) => <line class="fg-grid" x1={xOf(t)} x2={xOf(t)} y1={T} y2={B} />)}
      {[1, 10, 100].filter((v) => v < top).map((v) => <line class="fg-grid" x1={L} x2={R} y1={y(v)} y2={y(v)} />)}
      <line class="fg-axis" x1={L} x2={R} y1={B} y2={B} />
      {ticks.map((t, i) => (
        <text class="fg-label" x={xOf(t)} y={B + 12} text-anchor={i === 0 ? "start" : i === ticks.length - 1 ? "end" : "middle"}>{i === ticks.length - 1 ? `1,000 ${unitShort()}` : t}</text>
      ))}
      {[1, 10, 100].filter((v) => v < top).map((v) => <text class="fg-label" x={L + 2} y={y(v) - 2}>{"$" + v}</text>)}
      <path class="fg-curve" d={curve} clip-path="url(#fare-clip)" />
      <rect class="fg-hover" x={L} y={T} width={R - L} height={B - T} data-tip={hoverTip}
        onPointerMove={(e) => { if (drag.current) return; const r = svg.current!.getBoundingClientRect(); setHoverX(Math.max(L, Math.min(R, ((e.clientX - r.left) * W) / r.width))); }}
        onPointerLeave={() => setHoverX(null)} />
      {hoverU !== null && fareAt(hoverU) <= top && (
        <>
          <line class="fg-guide" x1={xOf(hoverU)} x2={xOf(hoverU)} y1={y(fareAt(hoverU))} y2={B} />
          <circle class="fg-dot" cx={xOf(hoverU)} cy={y(fareAt(hoverU))} r={2.5} />
        </>
      )}
      <g class="fg-node" {...nodeProps("base")}>
        <circle class="fg-hit" cx={ax} cy={ay} r={11} data-tip={`Fare for boarding, ${usd(shownBase)}\nDrag up or down`} />
        <circle class="fg-knob" cx={ax} cy={ay} r={4.5} />
      </g>
      <g class="fg-node" {...nodeProps("growth")}>
        <circle class="fg-hit" cx={bx} cy={by} r={11} data-tip={`Fare added for each ${unit}, ${usd(perUnit)}\nDrag up or down`} />
        <circle class="fg-knob" cx={bx} cy={by} r={4.5} />
      </g>
    </svg>
  );
}
