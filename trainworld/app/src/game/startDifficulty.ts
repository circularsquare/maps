// T-2690 (notes/T-2690.md): how hard each start is, from `start-difficulty.json`, which
// pipeline/start_difficulty.py writes: the easy and medium starts, any other being hard.

import { computed } from "@preact/signals";
import data from "./start-difficulty.json";
import { world } from "./world";
import type { CityId } from "./cities";

export type StartTier = "easy" | "medium" | "hard";

const EASY = new Set<string>(data.easy), MEDIUM = new Set<string>(data.medium);

export const startTier = (id: CityId): StartTier => (EASY.has(id) ? "easy" : MEDIUM.has(id) ? "medium" : "hard");

/** Whether the game has a line set up: running, with a train or bus at some level. Until then the
 * unopened zones' dots take their start tier's colour (map/intercityView.ts). */
export const lineSetUp = computed(() => {
  const w = world.value;
  return !!w && w.save.lines.some((l) => w.lineStats[l.id]?.status === "running" && (l.tph.high > 0 || l.tph.medium > 0 || l.tph.low > 0));
});
