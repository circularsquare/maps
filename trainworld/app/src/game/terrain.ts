// T-1270: the terrain classes the track price multiplies by (sim `TERRAIN_FACTOR`, params.rs), with
// their names and the colour the track tool's tag and the blueprint breakdown give them.

export const TERRAIN_NAMES = ["Flat", "Rolling", "Hilly", "Mountain"] as const;
export const TERRAIN_FACTORS = [1, 1.25, 1.6, 2.5] as const;
export const TERRAIN_COLOURS = ["#d8d4c8", "#d9c08a", "#c99a5b", "#a8703e"] as const;

/** "Hilly ground, 1.6x the price" for a class 0-3. */
export function terrainText(c: number): string {
  const k = Math.max(0, Math.min(3, Math.round(c)));
  return `${TERRAIN_NAMES[k]} ground, ${TERRAIN_FACTORS[k]}x the price`;
}
