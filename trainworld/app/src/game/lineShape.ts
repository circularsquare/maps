// T-3319: the shape of a line's bullet. Pure, so the clock worker can check a save's shapes too; the
// shapes set in a game and the Settings defaults are in game/lineShapeState.ts.

import type { Vehicle } from "./vehicle";

/** What a player can pick. A saved "pill" (T-3357) is not one, so it reads as no choice. */
export type PickShape = "circle" | "square" | "diamond";
/** What can be drawn: the pill is only the ferry's default look. */
export type LineShape = PickShape | "pill";
export const LINE_SHAPES: readonly PickShape[] = ["circle", "square", "diamond"];

export const isLineShape = (v: unknown): v is PickShape => typeof v === "string" && (LINE_SHAPES as readonly string[]).includes(v);

/** Each mode's shape until the player sets another in Settings: the bullets as they were before shapes
 * could be chosen (T-1464 a bus's rounded square, T-2194 a ferry's pill). */
export const MODE_SHAPES: Readonly<Record<Vehicle, LineShape>> = { train: "circle", tram: "circle", bus: "square", boat: "pill" };
export const SHAPE_MODES: readonly Vehicle[] = ["train", "tram", "bus", "boat"];

/** The shapes in a save's header, only valid entries kept. */
export function shapesFrom(v: unknown): Map<string, LineShape> {
  const out = new Map<string, LineShape>();
  if (typeof v !== "object" || v === null || Array.isArray(v)) return out;
  for (const [id, s] of Object.entries(v)) if (/^\d+$/.test(id) && isLineShape(s)) out.set(id, s);
  return out;
}
