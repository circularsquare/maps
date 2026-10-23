// What is lit as hovered (T-730, T-741): the demand bubble, line, bare track edge and station
// complex under the pointer, or, while the pointer is on an entry of the resting hover's list
// (T-732), that entry's thing instead. The renderers read it as uniforms; a change only redraws.

import type { StationId } from "../game/types";

/** A disc of one disc set (the Float32Array the demand overlay was given) and its index. */
export interface DiscHit {
  discs: Float32Array;
  i: number;
}

export interface Hot {
  /** a line's number (its colour slot), -1 none */
  line: number;
  /** a bare track edge's id, -1 none */
  edge: number;
  stations: StationId[];
  disc: DiscHit | null;
}

/** T-936: CSS px added to every drawn bubble radius while a bubble is selected (the overlay's
 * `setSelecting` sets it); hit tests add it too. It is also the smallest radius then (T-1102). */
export const bubblePad = { px: 0 };
export const SELECTED_PAD_PX = 1;

const NONE: Hot = { line: -1, edge: -1, stations: [], disc: null };

let pointerNet: Hot = NONE;
let pointerDisc: DiscHit | null = null;
let list: Hot | null = null;
let current: Hot = NONE;
const listeners: (() => void)[] = [];

/** Run on every change of what is lit. */
export function onHot(f: () => void) {
  listeners.push(f);
}

export function hot(): Hot {
  return current;
}

const same = (a: Hot, b: Hot) =>
  a.line === b.line && a.edge === b.edge && a.disc?.discs === b.disc?.discs && a.disc?.i === b.disc?.i
  && a.stations.length === b.stations.length && a.stations.every((s, k) => s === b.stations[k]);

function update() {
  const next = list ?? { ...pointerNet, disc: pointerDisc };
  if (same(next, current)) return;
  current = next;
  for (const f of listeners) f();
}

/** The network piece under the pointer's track tag (trackHover.ts); null: none. */
export function setPointerNet(h: Partial<Hot> | null) {
  pointerNet = h ? { ...NONE, ...h, disc: null } : NONE;
  update();
}

/** The bubble under the pointer's demand tag (demandHover.ts). */
export function setPointerDisc(d: DiscHit | null) {
  pointerDisc = d;
  update();
}

/** The resting list's entry under the pointer (hoverRest.ts); null: the pointer's own again. */
export function setListHot(h: Partial<Hot> | null) {
  list = h ? { ...NONE, ...h } : null;
  update();
}
