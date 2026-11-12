// T-3319: each line's bullet shape. A line set one by the player keeps it (saved with the game: the
// header's `shapes`, kept by the clock worker, outside the undo history as the hand order is); any other
// line takes its mode's shape from Settings (`display.modeShapes`, per browser).

import { computed, signal } from "@preact/signals";
import { MODE_SHAPES, type LineShape } from "./lineShape";
import { display } from "./ui";
import { vehicleOf, type Vehicle } from "./vehicle";
import { world } from "./world";

/** The shapes lines were given, by line id; a line not in it has its mode's. */
export const lineShapes = signal<Readonly<Record<string, LineShape>>>({});

let sink: ((line: string, shape: LineShape | null) => void) | null = null;
/** Where a change goes (the clock worker, for the save). */
export function setLineShapeSink(f: (line: string, shape: LineShape | null) => void) { sink = f; }

/** The game's shapes as the clock worker has them: on a load, and after a new line or a copy. */
export function takeLineShapes(shapes: Record<string, LineShape>) { lineShapes.value = { ...shapes }; }

// a computed, so a bullet redraws for a change of shapes only, not for every other setting
const modeShapes = computed(() => display.value.modeShapes);
/** A mode's shape now, as Settings has it. */
export const modeShape = (v: Vehicle): LineShape => modeShapes.value?.[v] ?? MODE_SHAPES[v];

/** Line `id`'s mode, read as the world stands now (train until its figures come). */
export function lineMode(id: string): Vehicle {
  const st = world.peek()?.lineStats[id];
  return st ? vehicleOf(st) : "train";
}

/** Line `id`'s bullet shape: its own, else its mode's. Read while rendering, so a bullet redraws on a change. */
export function shapeOf(id: string): LineShape {
  // T-3333: a hidden twin wears its line's
  const p = world.peek()?.twins.get(id);
  if (p) return shapeOf(p);
  return lineShapes.value[id] ?? modeShape(lineMode(id));
}

/** Give line `id` a shape; its mode's shape, or null, gives it back to the mode, so it follows Settings again. */
export function setLineShape(id: string, shape: LineShape | null) {
  const own = shape === null || shape === modeShape(lineMode(id)) ? null : shape;
  const { [id]: _, ...rest } = lineShapes.peek();
  lineShapes.value = own ? { ...rest, [id]: own } : rest;
  sink?.(id, own);
}
