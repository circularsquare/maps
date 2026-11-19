// The Money tab's fare control (T-864, T-883, T-1112, T-1143): the fare for a trip (y) against its
// length (x), straight stretches through the player's three nodes (game/fareCurve.ts). Zoom changes
// the distance shown and the fare shown with it, at one fixed ratio, so the line's slope on screen
// never changes; the fare axis starts at $0 unless that would put the base fare high up the plot.
// The base fare drags up and down only, the other nodes one way or the other, chosen by the drag's
// first few pixels. A node past the plot is an arrow at its edge. A drag changes only the picture
// and the figures below it; the fare goes to the game once, on release, since each change starts
// the demand again.

import { signal, type Signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { fareAt, MAX_KM, MAX_NODES, NODE_GAP_KM, scaleNodes, type FareNode } from "../game/fareCurve";
import { currencySpec, fromShown, price, priceTyped, toShown, typedPrice, unitMetres, unitShort } from "../game/format";
import { roundTo } from "../game/currency";
import { canUndoFares, changeFares, undo } from "../game/fareUndo";
import { playBuild } from "../map/buildSound";
import { t, tNumber, tRich } from "../i18n";
import { Icon, Stepper, TypedValue } from "./widgets";

/** T-1277: the graph shows the curve as charged where the player is looking (times that country's
 * scale); an edit goes to the game as the curve itself. */
const sendFares = (shown: FareNode[], scale: number) => changeFares(scale === 1 ? shown : scaleNodes(shown, 1 / scale));

/** The nodes as shown during a drag (km, US$): null when nothing is dragged. */
export const farePreview = signal<FareNode[] | null>(null);
/** The distance the graph shows, display units; null until the player zooms or drags: then it
 * fits the nodes. Kept while the game is open. */
const fareSpan = signal<number | null>(null);

/** The plot's frame in its viewBox units: the dock's, and T-3739's bigger one in the Advanced fares
 * panel (its own units, so labels and lines keep their size). */
const SMALL = { W: 300, H: 150, L: 6, R: 294, T: 10, B: 134 };
const BIG = { W: 560, H: 260, L: 6, R: 554, T: 10, B: 244 };
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

// T-2798: fares are US$ underneath and read in the shown currency (game/format.ts `price`)
const usd = price;
const distText = (u: number) => (u < 10 ? tNumber(Math.round(u * 10) / 10, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) : tNumber(Math.round(u)));
/** A point's distance: as `distText`, with its tenth kept past 10 when it has one (a step's riser at
 * 11.1 reads apart from 11). */
const nodeDist = (u: number) => (u >= 10 && Math.abs(u - Math.round(u)) >= 0.05 ? tNumber(Math.round(u * 10) / 10, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) : distText(u));
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
      <button class="btn icon" aria-label={t("fareGraph.undoLabel")} data-tip={t("fareGraph.undoTip")} disabled={!canUndoFares.value} onClick={() => void undo()}>
        <Icon.undo />
      </button>
      <span class="group">
        <button class="btn" aria-label={t("fareGraph.zoomOut")} data-tip={t("fareGraph.zoomOut")} disabled={span >= maxSpan(kpu)} onClick={() => { fareSpan.value = clampSpan(niceUp(span * 1.01), kpu); }}>
          <Icon.minus />
        </button>
        <button class="btn" aria-label={t("fareGraph.zoomIn")} data-tip={t("fareGraph.zoomIn")} disabled={span <= MIN_SPAN} onClick={() => { fareSpan.value = clampSpan(niceDown(span), kpu); }}>
          <Icon.plus />
        </button>
      </span>
    </span>
  );
}

/** How far node `i` may go, km and US$: past the node before it and short of the one after, by
 * `gap` km (T-3739: or by as little as it already is, so a step's riser keeps its spacing), and never
 * cheaper than the one before or dearer than the one after. */
function limits(nodes: FareNode[], i: number, gap = NODE_GAP_KM) {
  const prev = nodes[i - 1], next = nodes[i + 1], k = nodes[i][0];
  const near = (d: number) => Math.min(gap, Math.max(NODE_GAP_KM, d));
  return {
    kLo: prev ? prev[0] + near(k - prev[0]) : 0, kHi: next ? next[0] - near(next[0] - k) : MAX_KM,
    fLo: prev ? prev[1] : 0, fHi: next ? next[1] : Infinity,
  };
}

/** T-3739: a point added at `u` display units and fare `fare` (US$ shown). Within `magnet` km of a
 * point (a few pixels), it goes right beside it, after it when the pointer is above it and before it
 * when below, so a click over a point makes a step; elsewhere its distance snaps as a drag's. Its fare
 * snaps and is held between its neighbours' so fares never fall. The new nodes and its index, or null
 * where no point fits (before the base, between two points closer than two gaps, or the curve full). */
function placeAt(nodes: FareNode[], u: number, fare: number, span: number, h: number, kpu: number, magnet: number): { nodes: FareNode[]; i: number } | null {
  if (nodes.length >= MAX_NODES) return null;
  const raw = u * kpu;
  const closest = (k: number) => nodes.reduce((a, n) => (Math.abs(n[0] - k) < Math.abs(a[0] - k) ? n : a), nodes[0]);
  const beside = (n: FareNode) => (fare >= n[1] ? n[0] + NODE_GAP_KM : n[0] - NODE_GAP_KM);
  let km = snap(u, niceUp(span / 200)) * kpu;
  const by = closest(raw);
  if (Math.abs(by[0] - raw) < magnet) km = beside(by);
  else { const near = closest(km); if (Math.abs(near[0] - km) < NODE_GAP_KM - 1e-9) km = beside(near); }
  if (km < NODE_GAP_KM - 1e-9 || km > MAX_KM) return null;
  const i = nodes.findIndex((n) => n[0] > km), at = i < 0 ? nodes.length : i;
  const prev = nodes[at - 1], next = nodes[at];
  if (km < prev[0] + NODE_GAP_KM - 1e-9 || (next && km > next[0] - NODE_GAP_KM + 1e-9)) return null;
  const rate = currencySpec().rate, step = Math.max(currencySpec().grain, niceUp((h * rate) / 250));
  let f = clamp(snap(fare * rate, step) / rate, prev[1], next ? next[1] : Infinity);
  // a fare within a step of a neighbour's is its, so a flat stretch comes out flat
  if (Math.abs(f - prev[1]) * rate <= step) f = prev[1];
  else if (next && Math.abs(f - next[1]) * rate <= step) f = next[1];
  const out = nodes.map((n) => [...n] as FareNode);
  out.splice(at, 0, [km, f]);
  return { nodes: out, i: at };
}

/** US$ to the shown currency's finest step (a cent, a yen, ten won), as US$ */
const cents = (v: number) => typedPrice(toShown(v));
// T-1277: a rate scaled from the curve is whole cents to within a hundredth of a cent
// (T-2798: whole steps of the shown currency's rate grain, a cent in dollars)
const wholeCents = (r: number) => { const g = currencySpec().rateGrain, x = (r * currencySpec().rate) / g; return Math.abs(x - Math.round(x)) < 0.01; };
const rateDp = (whole: boolean) => { const g = currencySpec().rateGrain; return whole ? (g >= 1 ? 0 : 2) : g >= 10 ? 0 : g >= 1 ? 1 : 3; };
/** a rate in whole steps reads as them (cents), a typed or dragged one to a tenth of a step */
const rateTyped = (r: number) => (r * currencySpec().rate).toFixed(rateDp(wholeCents(r)));
const rateNum = (r: number) => { const dp = rateDp(wholeCents(r)); return tNumber(r * currencySpec().rate, { minimumFractionDigits: dp, maximumFractionDigits: dp }); };
const rateText = (r: number) => currencySpec().symbol + rateNum(r);

/** The rate per km of the stretch ending at node `i` (from 1), US$. */
const rateOf = (nodes: FareNode[], i: number) => (nodes[i][1] - nodes[i - 1][1]) / (nodes[i][0] - nodes[i - 1][0]);

/** T-3749: a stretch at most this long reads as a step, its row the fare's jump rather than a rate per km.
 * "Add point" and the panel's drags make steps `NODE_GAP_KM` wide; a drag's sideways snap at a closer zoom
 * (a 100 km view's is 0.5 km) can leave one up to this. */
const STEP_KM = 0.5;
/** Whether the stretch ending at node `i` is a step. Never the last: its rate carries on past it. */
const isStep = (nodes: FareNode[], i: number) => i > 0 && i < nodes.length - 1 && nodes[i][0] - nodes[i - 1][0] <= STEP_KM + 1e-6;

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

/** The rows under the graph (T-1201, T-1234): the base fare, then each stretch's end (typed) and
 * rate per unit (stepped by whole cents, or typed). A row's change keeps the other rows' values, so a
 * rate or base change moves every later node's fare by the same amount; a drag on the graph sets
 * fares instead. One grid across the rows, so the steppers line up. */
export function FareNodeRows({ nodes, scale, last, remove = false, custom }: { nodes: FareNode[]; scale: number; last?: ComponentChildren; remove?: boolean; custom?: ComponentChildren }) {
  const setFares = (ns: FareNode[]) => sendFares(ns, scale);
  // T-3739: a curve set in the Advanced fares panel shows one line here instead of its points
  if (custom) return <ul class="rows fare-rows"><li><span class="fr-what fr-custom">{custom}</span></li>{last}</ul>;
  const kpu = unitMetres() / 1000;
  const unit = unitShort();
  const stretches = nodes.slice(1).map((n, j) => [n[0], rateOf(nodes, j + 1)] as [number, number]);
  const base = nodes[0][1];
  const put = (b: number, s: [number, number][]) => void setFares(fromRows(b, s));
  const setBase = (v: number) => put(v, stretches);
  // T-3749: steps read as steps in the Advanced fares panel only; the Money tab's three rows stay rates per km.
  // Where each stretch reads as starting (by the node it ends at): one after a step from the step's own distance,
  // so "10 km +$2.50" is followed by "10 - 40 km", not "10.1 - 40 km"
  const step = (i: number) => remove && isStep(nodes, i);
  const startKm = [0, 0];
  for (let i = 2; i < nodes.length; i++) startKm.push(step(i - 1) ? startKm[i - 1] : nodes[i - 1][0]);
  const sp = currencySpec();
  const bin = (i: number) => (
    <button class="btn icon mini" aria-label={t("fareGraph.removePoint")} data-tip={t("fareGraph.removePoint")}
      onClick={() => { playBuild("pop"); void setFares(nodes.filter((_, k) => k !== i)); }}><Icon.bin /></button>
  );
  return (
    <ul class="rows fare-rows">
      <li>
        <span class="fr-what">{t("fareGraph.baseFare")}</span>
        {/* steps from the figure shown, in the shown currency (T-2798), so a step never lands on a drifted one */}
        <Stepper value={usd(base)} wide label={t("fareGraph.baseFareLabel")} onStep={(s) => setBase(fromShown(roundTo(toShown(base), currencySpec().grain) + s * currencySpec().fareStep))}
          type={{ text: priceTyped(base), set: (v) => setBase(fromShown(v)), tip: t("fareGraph.baseFareTip") }} />
        <span />
      </li>
      {stretches.map(([km, rate], j) => {
        const i = j + 1;
        const lim = limits(nodes, i);
        const from = startKm[i] / kpu, u = km / kpu;
        const fromText = from === 0 ? "0" : nodeDist(from);
        if (step(i)) {
          // T-3749: a step's row is the fare's jump there; a change keeps its width and every other row's value
          const jump = nodes[i][1] - nodes[i - 1][1];
          const setJump = (v: number) => put(base, stretches.map((s, k) => (k === j ? [s[0], Math.max(0, v) / (s[0] - nodes[i - 1][0])] : s)));
          // T-3751: its distance typed moves the whole step, its width and every rate kept
          const w = km - nodes[i - 1][0], lo = i > 1 ? limits(nodes, i - 1).kLo : 0, hi = limits(nodes, i).kHi - w;
          const setAt = (v: number) => {
            if (i === 1 || lo > hi) return false;
            const k = clamp(v * kpu, lo, hi);
            put(base, stretches.map((s, m) => (m === j - 1 ? [k, s[1]] : m === j ? [k + w, s[1]] : s)));
            return true;
          };
          // the step's foot is the end of the stretch before it, so its screen-reader label is that one's
          const at = i === 1 ? fromText : <TypedValue label={t("fareGraph.endLabel", { n: tNumber(i - 1) })}
            type={{ text: fromText, set: setAt }}>{fromText}</TypedValue>;
          return (
            <li>
              <span class="fr-what">{tRich("fareGraph.stepAt", { at, unit })}</span>
              <Stepper value={`+${usd(jump)}`} wide label={t("fareGraph.stepLabel", { at: fromText, unit })}
                onStep={(s) => (s < 0 && jump <= 0 ? false : setJump(fromShown(Math.max(0, roundTo(toShown(jump), sp.grain) + s * sp.fareStep))))}
                type={{ text: priceTyped(jump), set: (v) => setJump(fromShown(v)) }} />
              {/* its bin kept in the bins' place by an unseen "per km" */}
              <span class="fr-end"><span class="fr-unseen" aria-hidden="true">{t("fareGraph.perUnit", { unit })}</span>{bin(i)}</span>
            </li>
          );
        }
        const rateU = rate * kpu;
        const last = i === nodes.length - 1;
        const setU = (v: number) => { if (lim.kLo <= lim.kHi) put(base, stretches.map((s, k) => (k === j ? [clamp(v * kpu, lim.kLo, lim.kHi), s[1]] : s))); };
        const withRate = (r: number) => fromRows(base, stretches.map((s, k) => (k === j ? [s[0], Math.max(0, r) / kpu] : s)));
        // typed to a tenth of a step of the shown currency (a tenth of a cent in dollars), then back to US$
        const rg = sp.rateGrain;
        const setRate = (v: number) => void setFares(withRate(fromShown(roundTo(v, rg / 10))));
        // a step goes to the next whole step (a cent) of the shown currency above or below
        const stepRate = (s: number) => {
          const at = (rateU * sp.rate) / rg;
          const k = s > 0 ? Math.floor(at + 0.01) + 1 : Math.ceil(at - 0.01) - 1;
          const next = withRate(fromShown(rg < 1 ? k / Math.round(1 / rg) : k * rg));
          // a stretch too short for the step to move its fare a cent moves it a cent
          if (next[i][1] === nodes[i][1] && (s > 0 || nodes[i][1] > nodes[i - 1][1])) {
            const dc = (s > 0 ? 1 : -1) * fromShown(sp.grain);
            for (let k = i; k < next.length; k++) next[k][1] = cents(nodes[k][1] + dc);
          }
          if (JSON.stringify(next) !== JSON.stringify(nodes)) void setFares(next);
        };
        return (
          <li>
            {/* the last node's distance changes no fare, since its rate holds past it: it is only dragged */}
            {last
              ? <span class="fr-what">{t("fareGraph.past", { from: fromText, unit })}</span>
              : <span class="fr-what">{tRich("fareGraph.stretchEnd", { from: fromText, unit, end: step(i + 1)
                  // T-3751: a step follows: its row's box is this same distance, so here it is plain
                  ? <span class="v fr-plain">{nodeDist(u)}</span>
                  : <TypedValue label={t("fareGraph.endLabel", { n: tNumber(i) })}
                    type={{ text: nodeDist(u), set: setU, tip: t("fareGraph.endTip", { unit }) }}>{nodeDist(u)}</TypedValue> })}</span>}
            <Stepper value={`+${rateText(rateU)}`} wide label={t("fareGraph.rateLabel", { n: tNumber(i) })} onStep={stepRate}
              type={{ text: rateTyped(rateU), set: setRate, tip: t("fareGraph.rateTip", { last, rate: rateText(rateU), unit, from: fromText, to: nodeDist(u) }) }} />
            {remove
              ? <span class="fr-end">{t("fareGraph.perUnit", { unit })}{bin(i)}</span>
              : <span>{t("fareGraph.perUnit", { unit })}</span>}
          </li>
        );
      })}
      {last}
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

/** T-3739: the Advanced fares panel, open on the country it shows first; null closed. */
export const advancedFares = signal<string | null>(null);
/** T-3739: the Advanced fares panel's "Add point" is pressed: the next click on its graph adds one there. */
export const fareAdding = signal(false);

/** `big`: the Advanced fares panel's, with its own `preview` (its country's scale may differ from the
 * dock's) and the add-point click while `fareAdding`. */
export function FareGraph({ nodes, scale, big = false, preview = farePreview }: { nodes: FareNode[]; scale: number; big?: boolean; preview?: Signal<FareNode[] | null> }) {
  const { W, H, L, R, T, B } = big ? BIG : SMALL;
  const farePreview = preview;
  const adding = big && fareAdding.value;
  const [ghost, setGhost] = useState<{ nodes: FareNode[]; i: number } | null>(null);
  const setFares = (ns: FareNode[]) => sendFares(ns, scale);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const pinch = useRef<{ ids: Map<number, { x: number; y: number }>; d0: number; span0: number } | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  // T-1705: the viewBox is scaled to the panel, so a narrow screen sets the labels in units that draw at 12 px
  const [drawnW, setDrawnW] = useState<number | null>(null);
  useEffect(() => {
    const el = svg.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const w = el.getBoundingClientRect().width;
      el.style.setProperty("--fg-fs", `${(12 * W / w).toFixed(2)}px`);
      setDrawnW(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
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
  const pathOf = (ns: FareNode[]) => [...new Set([...ns.map((n) => n[0] / kpu), span])].filter((u) => u >= 0 && u <= span).sort((a, b) => a - b)
    .map((u, i) => `${i ? "L" : "M"}${x(u).toFixed(1)},${y(fareAt(ns, u * kpu)).toFixed(1)}`).join("");
  const curve = pathOf(shown);

  const pointer = (e: PointerEvent | MouseEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * W) / r.width, y: ((e.clientY - r.top) * H) / r.height };
  };
  // T-3739: where a click would add a point, from the pointer
  const placing = (e: PointerEvent | MouseEvent) => {
    const p = pointer(e);
    // within 8 graph units of a point sideways counts as over it
    return placeAt(nodes, unitAt(clamp(p.x, L, R)), y0 + ((B - clamp(p.y, T, B)) / (B - T)) * h, span, h, kpu, (8 / (R - L)) * span * kpu);
  };
  useEffect(() => { if (!adding) setGhost(null); }, [adding]);
  const add = (e: MouseEvent) => {
    if (!adding) return;
    const g = placing(e);
    if (!g) return;
    setGhost(null);
    fareAdding.value = false;
    playBuild("pop");
    void setFares(g.nodes);
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
    // T-3739: the panel's graph lets a point be dragged right up to its neighbour, so a drag can make a step
    const lim = limits(ns, d.i, big ? NODE_GAP_KM : Math.max(NODE_GAP_KM, 0.02 * d.span * kpu));
    if (d.axis === "x") {
      const hi = Math.min(d.span, lim.kHi / kpu), lo = lim.kLo / kpu;
      ns[d.i][0] = (lo > hi ? d.u0 : clamp(snap(d.u0 + (dx / (R - L)) * d.span, niceUp(d.span / 200)), lo, hi)) * kpu;
    } else {
      // snapped in the shown currency (T-2798): its own finest step at least, else a nice step of the plot's height
      const rate = currencySpec().rate;
      const raw = snap((d.f0 - (dy / (B - T)) * d.h) * rate, Math.max(currencySpec().grain, niceUp((d.h * rate) / 250))) / rate;
      ns[d.i][1] = clamp(raw, lim.fLo, Math.min(lim.fHi, d.y0 + d.h));
    }
    farePreview.value = ns;
  };
  const end = () => {
    const d = drag.current, f = farePreview.value;
    drag.current = null;
    if (!d || !f || JSON.stringify(f) === JSON.stringify(nodes)) { farePreview.value = null; return; }
    // T-1490: the quiet pop on release, at the pointer like the other build sounds
    playBuild("pop");
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
    if (adding && e.pointerType !== "touch") setGhost(placing(e));
  };
  const plotUp = (e: PointerEvent) => {
    const pc = pinch.current;
    if (!pc) return;
    pc.ids.delete(e.pointerId);
    if (pc.ids.size === 0) pinch.current = null;
    else pinch.current = { ids: pc.ids, d0: 0, span0: span };
  };

  const hoverU = hoverX === null ? null : unitAt(hoverX);
  const ghostAt = ghost && ghost.nodes[ghost.i];
  const hoverTip = adding
    ? t("fareGraph.addTip") + (ghostAt ? "\n" + t("fareGraph.hoverTip", { distance: nodeDist(ghostAt[0] / kpu), unit, fare: usd(ghostAt[1]) }) : "")
    : hoverU === null ? undefined : t("fareGraph.hoverTip", { distance: distText(hoverU), unit, fare: usd(fareU(hoverU)) });
  const clipId = big ? "fare-clip-big" : "fare-clip";
  // T-3768: at most five distance labels, fewer where the graph is drawn narrow, about 50 px apart at the least
  // (the last one's unit makes it the widest)
  const xLabels = drawnW ? Math.max(2, Math.min(5, Math.floor((drawnW * (R - L)) / W / 50))) : 5;
  const xStep = niceUp(span / xLabels);
  const xTicks = Array.from({ length: Math.floor(span / xStep + 1e-9) }, (_, i) => (i + 1) * xStep);
  // the fare axis in the shown currency (T-2798): ticks `yTicks` are in it, and drawn at their US$ height
  const rate = currencySpec().rate;
  const fStep = niceUp((h * rate) / 4);
  const yTicks: number[] = [];
  for (let v = Math.floor((y0 * rate) / fStep + 1) * fStep; v <= (y0 + h) * rate + 1e-9; v += fStep) yTicks.push(v);
  const tickText = (v: number) => (xStep < 1 ? tNumber(v, { minimumFractionDigits: xStep < 0.1 ? 2 : 1, maximumFractionDigits: xStep < 0.1 ? 2 : 1 }) : tNumber(v));
  const fareText = (v: number) => currencySpec().symbol + (fStep < 1 ? tNumber(v, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : tNumber(v));
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
    <svg class={"fare-graph" + (big ? " big" : "") + (adding ? " fg-adding" : "") + (d?.axis ? ` fg-drag-${d.axis}` : "")} ref={svg} viewBox={`0 0 ${W} ${H}`} onWheel={onWheel}>
      <defs><clipPath id={clipId}><rect x={L} y={T} width={R - L} height={B - T} /></clipPath></defs>
      {xTicks.map((tk) => <line class="fg-grid" x1={x(tk)} x2={x(tk)} y1={T} y2={B} />)}
      {yTicks.map((v) => <line class="fg-grid" x1={L} x2={R} y1={y(v / rate)} y2={y(v / rate)} />)}
      <line class="fg-axis" x1={L} x2={R} y1={B} y2={B} />
      <line class="fg-axis" x1={L} x2={L} y1={T} y2={B} />
      {/* centred under the axis unless half of it (about 4 px drawn) would be cut off at the graph's left edge */}
      <text class="fg-label" x={L} y={B + 12} text-anchor={drawnW !== null && (4 * W) / drawnW > L ? "start" : "middle"}>0</text>
      {xTicks.map((tk, i) => (
        <text class="fg-label" x={x(tk)} y={B + 12} text-anchor={i === lastX && x(tk) > R - 30 ? "end" : "middle"}>{i === lastX ? t("fareGraph.tickUnit", { value: tickText(tk), unit }) : tickText(tk)}</text>
      ))}
      {/* a label the base fare's node would cover is left out */}
      {yTicks.filter((v) => Math.abs(y(v / rate) - y(shown[0][1])) > 12).map((v) => <text class="fg-label" x={L + 3} y={y(v / rate) - 2}>{fareText(v)}</text>)}
      {ghost && <path class="fg-curve fg-ghost" d={pathOf(ghost.nodes)} clip-path={`url(#${clipId})`} />}
      <path class={"fg-curve" + (ghost ? " fg-under" : "")} d={curve} clip-path={`url(#${clipId})`} />
      <rect class="fg-hover" x={L} y={T} width={R - L} height={B - T} data-tip={hoverTip}
        onPointerDown={plotDown} onPointerMove={(e) => { if (drag.current) move(e); else plotMove(e); }}
        onPointerUp={(e) => { if (drag.current) end(); plotUp(e); }} onPointerCancel={(e) => { if (drag.current) end(); plotUp(e); }}
        onPointerLeave={() => { setHoverX(null); setGhost(null); }} onClick={add} />
      {ghostAt && <circle class="fg-knob fg-ghost-knob" cx={x(ghostAt[0] / kpu)} cy={y(ghostAt[1])} r={4.5} />}
      {!adding && hoverU !== null && (
        <>
          <line class="fg-guide" x1={x(hoverU)} x2={x(hoverU)} y1={Math.max(T, y(fareU(hoverU)))} y2={B} />
          {y(fareU(hoverU)) >= T && <circle class="fg-dot" cx={x(hoverU)} cy={y(fareU(hoverU))} r={2.5} />}
        </>
      )}
      {shown.map(([km, f], i) => {
        const cx = x(km / kpu), cy = y(f);
        const tip = i === 0 ? t("fareGraph.baseNodeTip", { fare: usd(f) }) : t("fareGraph.nodeTip", { distance: nodeDist(km / kpu), unit, fare: usd(f) });
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
          <g class="fg-off" data-tip={t("fareGraph.offTip", { first: tip.split("\n")[0] })} onClick={() => { fareSpan.value = fitSpan([shown[0], [km, f]], kpu, span * 1.01); }}>
            <circle class="fg-hit" cx={bx} cy={by} r={10} />
            <path class="fg-arrow" d={`M${px.toFixed(1)},${py.toFixed(1)}L${(bx - 4.5 * s).toFixed(1)},${(by + 4.5 * c).toFixed(1)}L${(bx + 4.5 * s).toFixed(1)},${(by - 4.5 * c).toFixed(1)}Z`} />
          </g>
        );
      })}
    </svg>
  );
}
